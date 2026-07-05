from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core.checkpoint import CheckpointStore
from monoid_agent_kernel.core.content import content_part_to_json
from monoid_agent_kernel.core.inbox import InboxMessage
from monoid_agent_kernel.core.media import normalize_inline_media_dicts
from monoid_agent_kernel.errors import PermissionDenied
from monoid_agent_kernel.reference._shared.tokens import TokenError
from monoid_agent_kernel.reference.backend.ports import LoopPort, MutableRunRecordPort, TokenClaimsPort
from monoid_agent_kernel.reference.backend.run_state import (
    record_lifecycle_payload as _record_lifecycle_payload,
)


def _normalize_inbound_message(content: str | Sequence[Any]) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for item in content:
        parts.append(item if isinstance(item, dict) else content_part_to_json(item))
    if not parts:
        raise ValueError("message has no content")
    return parts


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
    enqueue_message_and_checkpoint: Callable[[MutableRunRecordPort, Any], None]
    persist_checkpoint_from_any_thread: Callable[[MutableRunRecordPort], None]
    checkpoint_store_provider: Callable[[], CheckpointStore | None]
    read_recovery_meta: Callable[[Path, str], dict[str, Any] | None]
    attempt_resume: Callable[[Path, str], bool]
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
            self._context.call_soon(record.message_queue.put_nowait, self._context.close_signal)
        return {
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            **_record_lifecycle_payload(record),
            "cancel_requested": requested,
            "error": record.error,
            "error_code": record.error_code,
        }

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
        if message_id and message_id in record.seen_inbox_ids:
            return {"run_id": run_id, "status": "duplicate", "message_id": message_id}
        message = _normalize_inbound_message(content)
        checkpoint_store = self._context.checkpoint_store_provider()
        if isinstance(message, list) and checkpoint_store is not None:
            pending: dict[str, bytes] = {}
            message = normalize_inline_media_dicts(message, pending)
            for data in pending.values():
                checkpoint_store.put_blob(run_id, data)
        envelope = InboxMessage(
            content=message,
            id=message_id or f"inbox_{uuid.uuid4().hex[:12]}",
            source=source,
            type=message_type,
            run_id=run_id,
            correlation_id=correlation_id,
            causation_id=causation_id,
            traceparent=traceparent,
            tracestate=tracestate,
            metadata=dict(metadata or {}),
        )
        wire_bytes = len(json.dumps(envelope.to_json()).encode("utf-8"))
        max_message_bytes = self._context.max_message_bytes_provider()
        if wire_bytes > max_message_bytes:
            raise ValueError(f"message exceeds the {max_message_bytes}-byte limit")
        self._context.ensure_message_enqueue_allowed(record)
        self._context.enqueue_message_and_checkpoint(record, envelope.to_json())
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
        if claims.tenant_id != (meta.get("tenant_id") or "") or claims.user_id != (meta.get("user_id") or ""):
            raise PermissionDenied("token subject mismatch")
        if (run_dir / "failure.json").exists():
            raise ValueError("run is marked unrecoverable; inspect failure.json")
        checkpoint_store = self._context.checkpoint_store_provider()
        assert checkpoint_store is not None
        stored = checkpoint_store.latest(run_id)
        if stored is None or stored.checkpoint.terminal:
            raise ValueError("run has no resumable checkpoint")
        if not self._context.attempt_resume(run_dir, run_id):
            raise ValueError("resume failed; inspect run logs / failure.json")
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
        if record is not None and (claims.tenant_id != record.tenant_id or claims.user_id != record.user_id):
            raise PermissionDenied("token subject mismatch")

    def verify_task_callback_token(self, run_id: str, token: str, task_id: str) -> None:
        self._context.verify_task_callback_token(run_id, token, task_id)
