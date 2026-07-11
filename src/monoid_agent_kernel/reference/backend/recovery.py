from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core.agents import AgentRuntimeConfig
from monoid_agent_kernel.core._util import write_json_atomic
from monoid_agent_kernel.core.checkpoint import CheckpointRecord, CheckpointStore, load_latest_checked
from monoid_agent_kernel.core.durable_codec import DurableLoadResult
from monoid_agent_kernel.core.durable_metadata import DurableMetadataCommitter
from monoid_agent_kernel.core.result import AgentRunResult, Suspension
from monoid_agent_kernel.identifiers import namespaced_id
from monoid_agent_kernel.reference.backend.ports import (
    DriveOpenSessionPort,
    LeaseStorePort,
    LoopBuildPort,
    LoopPort,
    MutableRunRecordPort,
    RunRequestPort,
)
from monoid_agent_kernel.reference.backend.runtime_config import runtime_config_from_meta

_LOGGER = logging.getLogger("monoid_agent_kernel.backend")


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
    register_record: Callable[[MutableRunRecordPort], None]
    attach_loop: Callable[[MutableRunRecordPort, LoopBuildPort], None]
    call_soon: Callable[..., None]
    spawn: Callable[[Awaitable[Any]], object]
    drive_open_session: DriveOpenSessionPort
    record_run_result: Callable[[str, AgentRunResult], None]
    record_run_failure: Callable[[str, Exception], None]
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
            if self.attempt_resume(run_dir, run_id):
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
            if self.attempt_resume(run_dir, run_id):
                _LOGGER.info("watchdog: reclaimed orphaned run %s", run_id)
                reclaimed.append(run_id)
            elif not (run_dir / "failure.json").exists():
                lease_store.release(run_id)
        return reclaimed

    def attempt_resume(self, run_dir: Path, run_id: str) -> bool:
        checkpoint_result = load_latest_checked(self._checkpoint_store(), run_id)
        if not checkpoint_result.ok:
            self._record_checked_load_failure(run_dir, run_id, checkpoint_result)
            return False
        stored = checkpoint_result.value
        assert stored is not None
        if stored.checkpoint.terminal:
            return False
        metadata_result = self.read_recovery_meta_checked(run_dir, run_id)
        if not metadata_result.ok:
            self._record_checked_load_failure(run_dir, run_id, metadata_result)
            return False
        meta = metadata_result.value
        assert meta is not None
        try:
            self.resume_from_checkpoint(stored, meta)
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
                _LOGGER.error("run %s marked unrecoverable", run_id)
            return False
        self.clear_recover_attempts(run_dir)
        return True

    def resume_from_checkpoint(self, stored: CheckpointRecord, meta: dict[str, Any]) -> None:
        checkpoint = stored.checkpoint
        run_id = checkpoint.run_id
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
        self._context.register_record(record)
        loop_build = self._context.build_loop(run_id, request, workspace_root, llm_gateway_token, web_gateway_token)
        loop = loop_build.loop
        loop.restore(checkpoint, blobs=stored.blob)
        self._context.attach_loop(record, loop_build)
        record.seen_inbox_ids = set(checkpoint.inbox_seen_ids)
        for message in checkpoint.queued_messages:
            self._context.call_soon(record.message_queue.put_nowait, message)
        self._context.spawn(self.run_recovered(run_id, request, loop))

    async def run_recovered(self, run_id: str, request: RunRequestPort, loop: LoopPort) -> None:
        await self._context.acquire_run_slot()
        try:
            if loop.has_pending_tasks():
                suspension = Suspension(reason="awaiting_tasks", status="running", has_external=True)
            else:
                suspension = Suspension(reason="settled", status="completed")
            record = self._context.record(run_id)
            result = await self._context.drive_open_session(
                record,
                request,
                loop,
                suspension,
                started=time.time(),
                turns=1,
            )
            self._context.record_run_result(run_id, result)
        except Exception as exc:
            self._context.record_run_failure(run_id, exc)
        finally:
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

    def read_recover_attempts(self, run_dir: Path) -> int:
        try:
            payload = json.loads(self._recover_attempts_path(run_dir).read_text(encoding="utf-8"))
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
            "type": exc_type,
            "last_good_seq": last_good_seq,
            "restore_hint": (
                f"restore checkpoint seq {last_good_seq} for run {run_id} via CheckpointStore, "
                "then resume via recover_runs"
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
