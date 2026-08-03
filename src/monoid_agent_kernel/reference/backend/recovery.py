from __future__ import annotations

import asyncio
import enum
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core.agents import AgentRuntimeConfig
from monoid_agent_kernel.core._util import write_json_atomic
from monoid_agent_kernel.core.checkpoint import (
    CHECKPOINT_CODEC,
    CheckpointRecord,
    CheckpointStore,
    load_latest_checked,
)
from monoid_agent_kernel.core.durable_codec import DurableLoadResult
from monoid_agent_kernel.core.durable_metadata import (
    RUN_METADATA_CODEC,
    DurableMetadataCommitter,
    validate_recovery_metadata,
)
from monoid_agent_kernel.core.projections import status_artifact_records_close
from monoid_agent_kernel.core.result import (
    AgentRunResult,
    Suspension,
    suspension_from_checkpoint_payload,
)
from monoid_agent_kernel.core.json_ingress import loads_json_ingress
from monoid_agent_kernel.identifiers import namespaced_id
from monoid_agent_kernel.reference.backend.ports import (
    DriveOpenSessionPort,
    LeaseStorePort,
    LoopBuildPort,
    LoopPort,
    MutableRunRecordPort,
    RunRequestPort,
)
from monoid_agent_kernel.reference.backend.run_state import write_failure_status_artifact
from monoid_agent_kernel.reference.backend.runtime_config import runtime_config_from_meta

_LOGGER = logging.getLogger("monoid_agent_kernel.backend")


class ResumeOutcome(enum.Enum):
    """What one :meth:`RecoveryService.attempt_resume` concluded.

    A bare ``False`` used to fold three different refusals into one shape, and
    ``resume_run`` then blamed every one of them on a ``failure.json`` that mostly did not
    exist. The vocabulary is the decision, not the diagnosis: ``CLOSED`` — the run already
    ended (a terminal status artifact, or a terminal checkpoint); ``ALREADY_LIVE`` — a
    concurrent resume won the atomic record claim, so the run IS being resumed, just not by
    this caller; ``FAILED`` — the attempt genuinely did not resume the run (a deferred read,
    invalid durable state, or a resume exception — the give-up policy applies here and only
    here). Only ``RESUMED`` means this caller now owns a live run."""

    RESUMED = "resumed"
    CLOSED = "closed"
    ALREADY_LIVE = "already_live"
    FAILED = "failed"


@dataclass(frozen=True)
class RecoveryContext:
    run_root_provider: Callable[[], Path]
    checkpoint_store_provider: Callable[[], CheckpointStore | None]
    lease_store_provider: Callable[[], LeaseStorePort | None]
    max_recover_attempts_provider: Callable[[], int]
    worker_id_provider: Callable[[], str]
    lease_ttl_s_provider: Callable[[], float]
    is_record_tracked: Callable[[str], bool]
    record: Callable[[str], MutableRunRecordPort]
    make_request: Callable[[Mapping[str, Any], AgentRuntimeConfig], RunRequestPort]
    make_record: Callable[
        [str, RunRequestPort, Path, str, str, AgentRuntimeConfig, Mapping[str, Any]],
        MutableRunRecordPort,
    ]
    issue_llm_gateway_token: Callable[[str, RunRequestPort, AgentRuntimeConfig], str]
    issue_web_gateway_token: Callable[[str, RunRequestPort, AgentRuntimeConfig], str]
    build_loop: Callable[[str, RunRequestPort, Path, str, str], LoopBuildPort]
    register_record: Callable[[MutableRunRecordPort], bool]
    unregister_record: Callable[[MutableRunRecordPort], None]
    attach_loop: Callable[[MutableRunRecordPort, LoopBuildPort], None]
    call_soon: Callable[..., None]
    spawn: Callable[[Awaitable[Any]], object]
    drive_open_session: DriveOpenSessionPort
    record_run_result: Callable[[str, AgentRunResult], None]
    record_run_failure: Callable[[str, Exception], None]
    # ``(run_id, tenant_id)`` — the record-free metering seam for the give-up paths, which end
    # a run that was never re-registered (RunStateMutationService.meter_abandoned_run).
    meter_abandoned_run: Callable[[str, str], None]
    acquire_run_slot: Callable[[], Awaitable[None]]
    release_run_slot: Callable[[], None]


class RecoveryService:
    """Durable run recovery and stale-lease reclaim for the Reference backend."""

    def __init__(self, context: RecoveryContext) -> None:
        self._context = context

    def recover_runs(self) -> list[str]:
        recovered: list[str] = []
        run_root = self._context.run_root_provider()
        if not run_root.is_dir():
            return recovered
        for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
            run_id = run_dir.name
            if self._context.is_record_tracked(run_id):
                continue
            if (run_dir / "failure.json").exists():
                continue
            if self.attempt_resume(run_dir, run_id) is ResumeOutcome.RESUMED:
                recovered.append(run_id)
        return recovered

    def reclaim_stale_runs(self) -> list[str]:
        lease_store = self._context.lease_store_provider()
        assert lease_store is not None
        run_root = self._context.run_root_provider()
        worker_id = self._context.worker_id_provider()
        lease_ttl_s = self._context.lease_ttl_s_provider()
        reclaimed: list[str] = []
        for run_id in sorted(lease_store.candidate_run_ids()):
            if self._context.is_record_tracked(run_id):
                continue
            run_dir = run_root / run_id
            if (run_dir / "failure.json").exists():
                continue
            if not lease_store.is_stale(run_id):
                continue
            if not lease_store.try_claim(run_id, worker_id, lease_ttl_s):
                continue
            if self.attempt_resume(run_dir, run_id) is ResumeOutcome.RESUMED:
                _LOGGER.info("watchdog: reclaimed orphaned run %s", run_id)
                reclaimed.append(run_id)
            elif (
                not self._context.is_record_tracked(run_id)
                and not (run_dir / "failure.json").exists()
            ):
                lease_store.release(run_id)
        return reclaimed

    def attempt_resume(self, run_dir: Path, run_id: str) -> ResumeOutcome:
        if self._closed_by_status_artifact(run_dir):
            # A run that CLOSED limited is the one terminal outcome with no other
            # recovery-visible marker: its park checkpoint is non-terminal (a live-limited
            # park is resumable by design), ``close()`` keeps checkpoints for every
            # non-completed status, and no failure.json exists — so every recovery pass
            # re-drove it, appending another terminal run.finished and re-metering its full
            # cumulative usage per restart, forever. The durable status artifact the close
            # path already writes is the terminal marker, and consulting it on the reader
            # side covers run dirs closed before this guard existed. Not a quarantine: no
            # failure bundle, just a recognition that the run already ended.
            return ResumeOutcome.CLOSED
        try:
            checkpoint_result = load_latest_checked(self._checkpoint_store(), run_id)
        except Exception as exc:
            _LOGGER.warning("checkpoint read for run %s deferred: %s", run_id, exc)
            return ResumeOutcome.FAILED
        if not checkpoint_result.ok:
            self._record_checked_load_failure(run_dir, run_id, checkpoint_result)
            return ResumeOutcome.FAILED
        stored = checkpoint_result.value
        assert stored is not None
        if stored.checkpoint.run_id != run_id or stored.checkpoint.seq != stored.seq:
            self._record_checked_load_failure(
                run_dir,
                run_id,
                CHECKPOINT_CODEC.corrupt(
                    "checkpoint identity changed after checked load",
                    sequence=stored.seq,
                ),
            )
            return ResumeOutcome.FAILED
        if stored.checkpoint.terminal:
            # The run's own end, committed durably: nothing to resume — the checkpoint twin
            # of the status-artifact close above.
            return ResumeOutcome.CLOSED
        try:
            metadata_result = self.read_recovery_meta_checked(run_dir, run_id)
        except Exception as exc:
            _LOGGER.warning("recovery metadata read for run %s deferred: %s", run_id, exc)
            return ResumeOutcome.FAILED
        if not metadata_result.ok:
            self._record_checked_load_failure(run_dir, run_id, metadata_result)
            return ResumeOutcome.FAILED
        meta = metadata_result.value
        assert meta is not None
        try:
            validate_recovery_metadata(meta, expected_run_id=run_id)
        except (TypeError, ValueError) as exc:
            self._record_checked_load_failure(
                run_dir,
                run_id,
                RUN_METADATA_CODEC.corrupt(
                    f"backend-run recovery metadata validation failed ({exc})"
                ),
            )
            return ResumeOutcome.FAILED
        try:
            resumed = self.resume_from_checkpoint(stored, meta)
        except Exception as exc:
            attempts = self.bump_recover_attempts(run_dir)
            max_recover_attempts = self._context.max_recover_attempts_provider()
            _LOGGER.error(
                "resume of run %s failed (attempt %d/%d): %s",
                run_id,
                attempts,
                max_recover_attempts,
                exc,
            )
            if attempts >= max_recover_attempts:
                self.write_failure_bundle(
                    run_id,
                    run_dir,
                    error=f"recovery failed after {attempts} attempts: {exc}",
                    error_code="unrecoverable",
                    exc_type=type(exc).__name__,
                    overwrite=True,
                )
                write_failure_status_artifact(
                    run_dir,
                    run_id,
                    error=f"recovery failed after {attempts} attempts: {exc}",
                    error_code="unrecoverable",
                    exc_type=type(exc).__name__,
                    marker="given_up_by_recovery",
                )
                self._meter_giveup(run_dir, run_id, meta)
                _LOGGER.error("run %s marked unrecoverable", run_id)
            return ResumeOutcome.FAILED
        if resumed is False:
            # A concurrent recovery path won the atomic record claim. It owns the activation.
            return ResumeOutcome.ALREADY_LIVE
        self.clear_recover_attempts(run_dir)
        return ResumeOutcome.RESUMED

    def resume_from_checkpoint(self, stored: CheckpointRecord, meta: dict[str, Any]) -> bool:
        checkpoint = stored.checkpoint
        run_id = checkpoint.run_id
        if checkpoint.seq != stored.seq:
            raise ValueError("checkpoint manifest sequence does not match its committed record")
        validate_recovery_metadata(meta, expected_run_id=run_id)
        runtime_config = runtime_config_from_meta(meta)
        request = self._context.make_request(meta, runtime_config)
        workspace_root = request.workspace_root.resolve()
        llm_gateway_token = self._context.issue_llm_gateway_token(run_id, request, runtime_config)
        web_gateway_token = self._context.issue_web_gateway_token(run_id, request, runtime_config)
        record = self._context.make_record(
            run_id,
            request,
            workspace_root,
            llm_gateway_token,
            web_gateway_token,
            runtime_config,
            meta,
        )
        if not self._context.register_record(record):
            return False
        loop: LoopPort | None = None
        try:
            loop_build = self._context.build_loop(
                run_id, request, workspace_root, llm_gateway_token, web_gateway_token
            )
            loop = loop_build.loop
            loop.restore(checkpoint, blobs=stored.blob)
            self._context.attach_loop(record, loop_build)
            record.seen_inbox_ids = set(checkpoint.inbox_seen_ids)
            for message in checkpoint.queued_messages:
                self._context.call_soon(record.message_queue.put_nowait, message)
            restored_suspension = (
                suspension_from_checkpoint_payload(checkpoint.last_suspension)
                if checkpoint.last_suspension is not None
                else Suspension(reason="settled", status="completed")
            )
            recovered = self.run_recovered(run_id, request, loop, restored_suspension)
            try:
                self._context.spawn(recovered)
            except BaseException:
                recovered.close()
                raise
            return True
        except BaseException:
            if loop is not None:
                try:
                    loop.discard_uncommitted()
                except BaseException:
                    # Preserve the recovery/build failure. AgentLoop cleanup is best-effort and
                    # already attempts each owned activation resource before raising its first one.
                    pass
            # A failed build is not a live run. Leaving this provisional record registered makes
            # every later recovery pass skip it forever, bypassing the retry/unrecoverable policy.
            self._context.unregister_record(record)
            raise

    async def run_recovered(
        self,
        run_id: str,
        request: RunRequestPort,
        loop: LoopPort,
        suspension: Suspension,
    ) -> None:
        acquired = False
        released = False
        discarded = False
        try:
            await self._context.acquire_run_slot()
            acquired = True
            if loop.has_pending_tasks():
                # ``status="completed"``, not ``"running"``. The durable status vocabulary is
                # completed/failed/limited (core/result.py), and this synthetic park used to be
                # minted outside it — latent only because it is re-driven rather than
                # serialized, with nothing on the type saying so. The first path that
                # checkpointed it would have raised at the recovery boundary this exists to
                # serve. Behaviour is unchanged: nothing between here and the driver reads
                # ``status`` on an ``awaiting_tasks`` park, which branches on ``reason``.
                suspension = Suspension(
                    reason="awaiting_tasks", status="completed", has_external=True
                )
            record = self._context.record(run_id)
            result = await self._context.drive_open_session(
                record,
                request,
                loop,
                suspension,
                started=time.time(),
                turns=1,
            )
            released = True
            self._context.record_run_result(run_id, result)
        except Exception as exc:
            try:
                await asyncio.to_thread(loop.discard_uncommitted)
                discarded = True
            except BaseException:
                # Preserve the recovered execution failure. AgentLoop has already attempted every
                # owned activation resource before surfacing a cleanup error.
                pass
            self._context.record_run_failure(run_id, exc)
        finally:
            # Task cancellation bypasses ``except Exception`` but still owns the recovered loop.
            if not released and not discarded:
                try:
                    await asyncio.to_thread(loop.discard_uncommitted)
                except BaseException:
                    pass
            if acquired:
                self._context.release_run_slot()

    def read_recovery_meta(self, run_dir: Path, run_id: str) -> dict[str, Any] | None:
        return self.read_recovery_meta_checked(run_dir, run_id).value

    def read_recovery_meta_checked(
        self, run_dir: Path, run_id: str
    ) -> DurableLoadResult[dict[str, Any]]:
        return DurableMetadataCommitter(self._checkpoint_store()).read_recovery_metadata_checked(
            run_dir, run_id
        )

    def _record_checked_load_failure(
        self,
        run_dir: Path,
        run_id: str,
        result: DurableLoadResult[Any],
    ) -> None:
        if result.status == "missing":
            return
        sequence = f" at checkpoint seq {result.sequence}" if result.sequence is not None else ""
        self.write_failure_bundle(
            run_id,
            run_dir,
            error=f"{result.message}{sequence}",
            error_code=result.error_code or "durable_state_invalid",
            exc_type="DurableLoadError",
            overwrite=True,
        )
        write_failure_status_artifact(
            run_dir,
            run_id,
            error=f"{result.message}{sequence}",
            error_code=result.error_code or "durable_state_invalid",
            exc_type="DurableLoadError",
            marker="given_up_by_recovery",
        )
        # This quarantine ends the run for good (failure.json makes every later pass skip
        # it), so what it had spent must reach the ledger now or never.
        self._meter_giveup(run_dir, run_id, None)

    def _closed_by_status_artifact(self, run_dir: Path) -> bool:
        """Whether the run's durable status artifact records a CLOSE.

        The payload-level answer lives in ``core.projections.status_artifact_records_close``
        — ONE function for this guard and for ``list_runs``' ``recoverable`` fact, so the
        two cannot drift: it resolves current artifacts (``state`` + explicit ``terminal``)
        and legacy pre-``state`` ones, and treats a failure-quarantine statement (any
        :data:`~monoid_agent_kernel.core.projections.FAILURE_QUARANTINE_MARKERS` marker) as
        NOT a close — while the quarantine stands, every caller of ``attempt_resume``
        refuses the dir on failure.json before reaching this guard, and once an operator
        lifts it (the restore_hint's prescribed flow) this guard must not keep refusing the
        resume. Unreadable or malformed artifacts answer False — the checkpoint/metadata
        pipeline owns durable-state corruption, and a best-effort projection must not block
        a genuine recovery. A MISSING (or operator-deleted) status.json answers False the
        same way, so a closed-limited run whose artifact is gone can be resurrected once —
        it then re-closes limited and rewrites the artifact, which bounds the damage to one
        extra drive that self-heals at re-close. Falling back to deeper evidence (event-log
        replay) for that edge was considered and declined: the artifact is written from run
        start, so its absence is overwhelmingly "not a run dir we closed", and a fallible
        deep read here must not block genuine recovery."""
        try:
            payload = loads_json_ingress((run_dir / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return status_artifact_records_close(payload)

    def _meter_giveup(self, run_dir: Path, run_id: str, meta: Mapping[str, Any] | None) -> None:
        """Meter a given-up run's spend through the record-free seam, best-effort tenant.

        The resume-failure give-up already holds validated metadata; the corrupt-state
        quarantine may not, so the tenant is re-read from the recovery descriptor. A run with
        no attributable tenant cannot enter a tenant ledger — logged, not invented."""
        if meta is None:
            try:
                meta = self.read_recovery_meta(run_dir, run_id)
            except Exception:  # noqa: BLE001 - metering must never mask the recorded failure
                meta = None
        tenant_id = str((meta or {}).get("tenant_id") or "")
        if not tenant_id:
            _LOGGER.warning(
                "run %s given up with no attributable tenant; its spend is not metered", run_id
            )
            return
        try:
            self._context.meter_abandoned_run(run_id, tenant_id)
        except Exception:  # noqa: BLE001 - metering must never mask the recorded failure
            _LOGGER.exception("metering the abandoned run %s failed", run_id)

    def read_recover_attempts(self, run_dir: Path) -> int:
        try:
            payload = loads_json_ingress(
                self._recover_attempts_path(run_dir).read_text(encoding="utf-8")
            )
            return int(payload["count"])
        except (FileNotFoundError, ValueError, KeyError, OSError, TypeError):
            return 0

    def bump_recover_attempts(self, run_dir: Path) -> int:
        count = self.read_recover_attempts(run_dir) + 1
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self._recover_attempts_path(run_dir), {"count": count})
        return count

    def clear_recover_attempts(self, run_dir: Path) -> None:
        self._recover_attempts_path(run_dir).unlink(missing_ok=True)

    def write_failure_bundle(
        self,
        run_id: str,
        run_dir: Path,
        *,
        error: str,
        error_code: str,
        exc_type: str,
        overwrite: bool,
        http_status: int | None = None,
        retryable: bool = False,
        config_recoverable: bool = False,
    ) -> None:
        failure_path = run_dir / "failure.json"
        if failure_path.exists() and not overwrite:
            return
        last_good_seq = 0
        checkpoint_store = self._context.checkpoint_store_provider()
        if checkpoint_store is not None:
            try:
                stored = checkpoint_store.latest(run_id)
                last_good_seq = stored.seq if stored is not None else 0
            except Exception:  # pragma: no cover - last-good lookup must never mask the failure
                last_good_seq = 0
        bundle = {
            "schema_version": namespaced_id("failure.v1"),
            "run_id": run_id,
            "error": error,
            "error_code": error_code,
            # The core's twin of this bundle (``loop.py``) carries the provider status, and this
            # writer is the one a worker crash leaves behind -- the case where the bundle is the
            # only record there is. ``None`` for the recovery-path failures that never reached a
            # provider, which is what an absent status means there too.
            "http_status": http_status,
            # The classification twin of the status above. The defaults are the honest reading of
            # "nothing classified this": a recovery-path failure (unrecoverable, invalid durable
            # state) has no provider verdict to report, and the durable readers already treat an
            # absent flag as False. Only the caller holding the run's own exception
            # (``run_state.record_run_failure``) passes anything else.
            "retryable": retryable,
            "config_recoverable": config_recoverable,
            "type": exc_type,
            "last_good_seq": last_good_seq,
            # The actual operator flow, verified: ``recover_runs`` and ``resume_run`` both
            # SKIP/refuse a dir carrying failure.json, so the quarantine must be lifted first.
            # The previous hint said "resume via recover_runs" with the bundle still in place
            # — unfollowable as written.
            "restore_hint": (
                f"last good checkpoint is seq {last_good_seq}; delete failure.json to lift "
                "the quarantine, then recover_runs (or resume_run) restores it"
                if last_good_seq > 0
                else "no recoverable checkpoint; inspect run logs and run.json"
            ),
            "failed_at": time.time(),
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(failure_path, bundle)

    def _recover_attempts_path(self, run_dir: Path) -> Path:
        return run_dir / "recover_attempts.json"

    def _checkpoint_store(self) -> CheckpointStore:
        checkpoint_store = self._context.checkpoint_store_provider()
        assert checkpoint_store is not None
        return checkpoint_store
