from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core.checkpoint import CheckpointStore
from monoid_agent_kernel.core.content import content_part_to_json
from monoid_agent_kernel.core.inbox import InboxMessage
from monoid_agent_kernel.core.json_ingress import normalize_json_ingress, normalize_unicode_scalars
from monoid_agent_kernel.core.lifecycle import SessionState
from monoid_agent_kernel.core.media import normalize_inline_media_dicts
from monoid_agent_kernel.errors import NativeAgentError, PermissionDenied
from monoid_agent_kernel.reference._shared.tokens import TokenError
from monoid_agent_kernel.reference.backend.ports import (
    LoopPort,
    MutableRunRecordPort,
    TokenClaimsPort,
)
from monoid_agent_kernel.reference.backend.recovery import ResumeOutcome
from monoid_agent_kernel.reference.backend.run_state import (
    record_lifecycle_payload as _record_lifecycle_payload,
)

_LOGGER = logging.getLogger("monoid_agent_kernel.backend")


# The parks at which the drive loop is quiescent (nothing stepping): a checkpoint taken here is
# a park-point artifact, so an acknowledged cancel can be committed durably at the ack. RUNNING
# is deliberately absent — mid-turn is not a park point, and a cancel that lands there is made
# durable by the pump's own terminal park when the turn hits its next boundary check.
_QUIESCENT_PARK_STATES = frozenset(
    {
        SessionState.AWAITING_INPUT,
        SessionState.AWAITING_TASKS,
        SessionState.PAUSED,
        SessionState.INTERRUPTED,
        SessionState.TURN_FAILED,
    }
)


def _normalize_inbound_message(content: str | Sequence[Any]) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return normalize_unicode_scalars(content)
    parts: list[dict[str, Any]] = []
    for item in content:
        parts.append(item if isinstance(item, dict) else content_part_to_json(item))
    if not parts:
        raise ValueError("message has no content")
    normalized = normalize_json_ingress(parts)
    if not isinstance(normalized, list):  # pragma: no cover - list topology is retained
        raise ValueError("message content must be an array")
    return normalized


@dataclass(frozen=True)
class BackendSessionContext:
    authorize_run: Callable[[str, str], None]
    verify_run_token: Callable[[str, str], TokenClaimsPort]
    verify_task_callback_token: Callable[[str, str, str], None]
    issue_task_callback_token: Callable[[str, str, str, str], str]
    record: Callable[[str], MutableRunRecordPort]
    active_record: Callable[[str], MutableRunRecordPort | None]
    run_dir_for: Callable[[str], Path]
    call_soon: Callable[..., None]
    # Run one callable on the shared drive loop and wait for it — the ordering seam for a
    # read-then-write that must not interleave with the drive's own loop iterations.
    run_on_shared_loop: Callable[[Callable[[], None]], None]
    enqueue_message_and_checkpoint: Callable[[MutableRunRecordPort, Any], None]
    persist_checkpoint_from_any_thread: Callable[[MutableRunRecordPort], None]
    checkpoint_store_provider: Callable[[], CheckpointStore | None]
    read_recovery_meta: Callable[[Path, str], dict[str, Any] | None]
    attempt_resume: Callable[[Path, str], ResumeOutcome]
    max_message_bytes_provider: Callable[[], int]
    max_message_queue_depth_provider: Callable[[], int]
    record_terminal: Callable[[MutableRunRecordPort], bool]
    live_loop: Callable[[MutableRunRecordPort], tuple[LoopPort | None, bool]]
    mark_cancel_requested: Callable[[MutableRunRecordPort], bool]
    ensure_message_enqueue_allowed: Callable[[MutableRunRecordPort], None]
    close_signal: object
    resume_signal: object


class BackendSessionService:
    """Session and control-action boundary for the RunnerBackend facade."""

    def __init__(self, context: BackendSessionContext) -> None:
        self._context = context

    def cancel_run(self, run_id: str, token: str) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        record = self._context.record(run_id)
        requested = self._context.mark_cancel_requested(record)
        if requested:
            # The whole read-then-write — quiescence check, ack checkpoint, wake signal —
            # runs as ONE callable on the shared drive loop, so it is ordered against the
            # drive's own iterations rather than reading ``record.state`` on this HTTP
            # thread and persisting later (the state could flip in between, and a same-seq
            # ``put`` from here would then replace the committed park checkpoint's content
            # with a mid-turn snapshot). The call blocks until the callable ran, so the ack
            # the caller receives is already on disk when the park case applies.
            self._context.run_on_shared_loop(lambda: self._ack_cancel_on_drive_loop(record))
        return {
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            **_record_lifecycle_payload(record),
            "cancel_requested": requested,
            "error": record.error,
            "error_code": record.error_code,
        }

    def _ack_cancel_on_drive_loop(self, record: MutableRunRecordPort) -> None:
        """Durable cancel ack + drive wakeup, run as one callable on the shared drive loop.

        The park checkpoint predates the cancel, so an ack backed only by the in-memory
        token was not durable: a crash before the terminal record restored the run
        uncancelled. When the run sits at a quiescent park, commit a fresh park checkpoint
        AFTER the token flip — ``snapshot()`` serializes ``cancellation_requested`` and the
        restore path re-applies it — and BEFORE the close signal, so the drive wakes only
        once the ack is on disk. The park test is two-sided on purpose: the record state
        says where the DRIVE parked it, and ``loop.at_quiescent_park()`` says no pump is in
        flight — the drive resumes a park and enters the pump synchronously on this same
        loop, so inside this callable the pair cannot change under us. A persist refused
        with ``run_not_open``/``run_terminal`` is a SUCCESSFUL cancel: the one path that
        raises it here is a close (idle timeout / drain) that won the race on its lifecycle
        thread, and a run that just ended is exactly what the caller asked for — swallow and
        log rather than 500 after the ack.

        Remaining honest window, by design: a cancel that lands while a turn is stepping
        (or during a close already in flight on its lifecycle worker thread — ``aclose``
        offloads, so it can interleave with this callable) stays in-memory until the pump's
        boundary check or the close promotion writes its own terminal park; a crash inside
        that window still loses the ack."""
        loop = record.loop
        if (
            loop is not None
            and not self._context.record_terminal(record)
            and record.state in _QUIESCENT_PARK_STATES
            and loop.at_quiescent_park()
        ):
            try:
                self._context.persist_checkpoint_from_any_thread(record)
            except NativeAgentError as exc:
                if exc.error_code not in {"run_not_open", "run_terminal"}:
                    raise
                _LOGGER.debug(
                    "cancel ack checkpoint skipped for %s: run already closed (%s)",
                    record.run_id,
                    exc.error_code,
                )
        record.message_queue.put_nowait(self._context.close_signal)

    def interrupt_turn(self, run_id: str, token: str) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        record = self._context.record(run_id)
        loop, terminal = self._context.live_loop(record)
        requested = not terminal and loop is not None
        if requested:
            loop.interrupt_turn()
        return {
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            **_record_lifecycle_payload(record),
            "interrupt_requested": requested,
        }

    def pause_run(self, run_id: str, token: str) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        record = self._context.record(run_id)
        loop, terminal = self._context.live_loop(record)
        requested = not terminal and loop is not None
        if requested:
            loop.pause_turn()
        return {
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            **_record_lifecycle_payload(record),
            "pause_requested": requested,
        }

    def signal_resume(self, run_id: str, token: str) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        record = self._context.record(run_id)
        if self._context.record_terminal(record):
            return {"run_id": run_id, **_record_lifecycle_payload(record), "resumed": False}
        self._context.call_soon(record.message_queue.put_nowait, self._context.resume_signal)
        return {"run_id": run_id, **_record_lifecycle_payload(record), "resumed": True}

    def revoke_capability(
        self,
        run_id: str,
        token: str,
        *,
        capability: str | None = None,
        lease_id: str | None = None,
        before: float | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        record = self._context.record(run_id)
        loop, terminal = self._context.live_loop(record)
        summary: dict[str, Any] = {}
        revoked = not terminal and loop is not None
        if revoked:
            summary = loop.revoke_capability(
                capability=capability, lease_id=lease_id, before=before, reason=reason
            )
        return {
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            **_record_lifecycle_payload(record),
            "revoked": revoked,
            **summary,
        }

    def send_message(
        self,
        run_id: str,
        token: str,
        content: str | Sequence[Any],
        *,
        message_id: str = "",
        source: str = "api",
        correlation_id: str = "",
        causation_id: str = "",
        traceparent: str = "",
        tracestate: str = "",
        message_type: str = "user_message",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        record = self._context.record(run_id)
        message_id = normalize_unicode_scalars(message_id)
        if message_id and message_id in record.seen_inbox_ids:
            return {"run_id": run_id, "status": "duplicate", "message_id": message_id}
        message = _normalize_inbound_message(content)
        checkpoint_store = self._context.checkpoint_store_provider()
        pending: dict[str, bytes] = {}
        if isinstance(message, list) and checkpoint_store is not None:
            message = normalize_inline_media_dicts(message, pending)
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("message metadata must be an object or null")
        envelope = InboxMessage(
            content=message,
            id=message_id or f"inbox_{uuid.uuid4().hex[:12]}",
            source=normalize_unicode_scalars(source),
            type=normalize_unicode_scalars(message_type),
            run_id=run_id,
            correlation_id=normalize_unicode_scalars(correlation_id),
            causation_id=normalize_unicode_scalars(causation_id),
            traceparent=normalize_unicode_scalars(traceparent),
            tracestate=normalize_unicode_scalars(tracestate),
            metadata=normalize_json_ingress(dict(metadata or {})),
        )
        envelope_payload = normalize_json_ingress(envelope.to_json())
        wire_bytes = len(json.dumps(envelope_payload, allow_nan=False).encode("utf-8"))
        max_message_bytes = self._context.max_message_bytes_provider()
        if wire_bytes > max_message_bytes:
            raise ValueError(f"message exceeds the {max_message_bytes}-byte limit")
        self._context.ensure_message_enqueue_allowed(record)
        if checkpoint_store is not None:
            for data in pending.values():
                checkpoint_store.put_blob(run_id, data)
        self._context.enqueue_message_and_checkpoint(record, envelope_payload)
        return {"run_id": run_id, "status": "queued", "message_id": envelope.id}

    def report_task_result(
        self,
        run_id: str,
        token: str,
        *,
        task_id: str,
        result: dict[str, Any],
        status: str = "answered",
    ) -> dict[str, Any]:
        loop = self.authorize_task_result(run_id, token, task_id)
        reported = loop.report_task_result(
            task_id,
            result,
            status=status,
            persist_checkpoint=False,
        )
        self._context.persist_checkpoint_from_any_thread(self._context.record(run_id))
        return reported

    def create_task(
        self,
        run_id: str,
        token: str,
        *,
        kind: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        loop = self.authorize_active_loop(run_id, token)
        record = self._context.record(run_id)
        task_id = loop.create_task(kind, request)
        callback_token = self._context.issue_task_callback_token(
            run_id, record.tenant_id, record.user_id, task_id
        )
        return {
            "task_id": task_id,
            "callback_token": callback_token,
            "callback_url": f"/v1/runs/{run_id}/tasks/{task_id}/result",
        }

    def resume_run(self, run_id: str, token: str) -> dict[str, Any]:
        claims = self._context.verify_run_token(run_id, token)
        existing = self._context.active_record(run_id)
        if existing is not None:
            if claims.tenant_id != existing.tenant_id or claims.user_id != existing.user_id:
                raise PermissionDenied("token subject mismatch")
            return {"run_id": run_id, **_record_lifecycle_payload(existing), "resumed": False}
        if any(sep in run_id for sep in ("/", "\\")) or ".." in run_id:
            raise PermissionDenied("invalid run id")
        run_dir = self._context.run_dir_for(run_id)
        meta = self._context.read_recovery_meta(run_dir, run_id)
        if meta is None:
            raise KeyError(f"unknown run: {run_id}")
        if claims.tenant_id != (meta.get("tenant_id") or "") or claims.user_id != (
            meta.get("user_id") or ""
        ):
            raise PermissionDenied("token subject mismatch")
        if (run_dir / "failure.json").exists():
            raise ValueError("run is marked unrecoverable; inspect failure.json")
        checkpoint_store = self._context.checkpoint_store_provider()
        assert checkpoint_store is not None
        stored = checkpoint_store.latest(run_id)
        if stored is None or stored.checkpoint.terminal:
            raise ValueError("run has no resumable checkpoint")
        outcome = self._context.attempt_resume(run_dir, run_id)
        if outcome is ResumeOutcome.ALREADY_LIVE:
            # The atomic record claim lost to a concurrent resume (the studio double-click
            # shape): the run IS live — the winner owns the activation — so the loser
            # answers the same already-live shape the record-exists branch above answers,
            # not an error. The subject was already verified against the durable metadata
            # the winner's record is built from. The record can be momentarily invisible
            # between the CAS and this read; the honest minimum is still "not resumed by
            # this call".
            live = self._context.active_record(run_id)
            payload = _record_lifecycle_payload(live) if live is not None else {}
            return {"run_id": run_id, **payload, "resumed": False}
        if outcome is ResumeOutcome.CLOSED:
            # The run already ended — a terminal status artifact (e.g. a close while
            # budget-limited keeps a NON-terminal park checkpoint by design, so the
            # checkpoint guard above cannot see it). Refused in the loop's own terminal
            # vocabulary instead of pointing at a failure.json that does not exist.
            raise NativeAgentError(
                "run is already closed; its durable status artifact records a terminal "
                "outcome, so there is nothing to resume",
                error_code="run_terminal",
            )
        if outcome is not ResumeOutcome.RESUMED:
            raise ValueError("resume failed; inspect run logs and failure.json (if present)")
        record = self._context.record(run_id)
        return {"run_id": run_id, **_record_lifecycle_payload(record), "resumed": True}

    def authorize_active_loop(self, run_id: str, token: str) -> LoopPort:
        self._context.authorize_run(run_id, token)
        return self.active_loop(run_id)

    def authorize_task_result(self, run_id: str, token: str, task_id: str) -> LoopPort:
        try:
            self.verify_task_callback_token(run_id, token, task_id)
        except TokenError:
            self._context.authorize_run(run_id, token)
        return self.active_loop(run_id)

    def active_loop(self, run_id: str) -> LoopPort:
        record = self._context.record(run_id)
        loop, terminal = self._context.live_loop(record)
        if terminal:
            raise ValueError("cannot drive tasks for a terminal run")
        if loop is None:
            raise ValueError("run has not started")
        return loop

    def authorize_claim_subject(self, run_id: str, claims: TokenClaimsPort) -> None:
        record = self._context.active_record(run_id)
        if record is not None and (
            claims.tenant_id != record.tenant_id or claims.user_id != record.user_id
        ):
            raise PermissionDenied("token subject mismatch")

    def verify_task_callback_token(self, run_id: str, token: str, task_id: str) -> None:
        self._context.verify_task_callback_token(run_id, token, task_id)
