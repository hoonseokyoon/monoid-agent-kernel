from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from monoid_agent_kernel.core.content import content_part_to_json
from monoid_agent_kernel.core.inbox import InboxMessage
from monoid_agent_kernel.core.media import normalize_inline_media_dicts
from monoid_agent_kernel.errors import PermissionDenied
from monoid_agent_kernel.identifiers import TASK_CALLBACK_AUDIENCE, TASK_CALLBACK_AUDIENCES
from monoid_agent_kernel.reference._shared.tokens import TokenError
from monoid_agent_kernel.reference.backend.projection import (
    _record_lifecycle_payload,
    _record_terminal,
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


class BackendSessionService:
    """Session and control-action boundary for the RunnerBackend facade."""

    def __init__(self, backend: Any, *, close_signal: object, resume_signal: object) -> None:
        self._backend = backend
        self._close_signal = close_signal
        self._resume_signal = resume_signal

    def cancel_run(self, run_id: str, token: str) -> dict[str, Any]:
        backend = self._backend
        backend._authorize_run(run_id, token)
        record = backend._record(run_id)
        with backend._lock:
            if _record_terminal(record):
                return {
                    "run_id": record.run_id,
                    "tenant_id": record.tenant_id,
                    **_record_lifecycle_payload(record),
                    "cancel_requested": False,
                    "error": record.error,
                    "error_code": record.error_code,
                }
            record.cancellation_token.cancel()
            record.error = "run cancellation requested"
            record.error_code = "cancelled"
        backend._call_soon(record.message_queue.put_nowait, self._close_signal)
        return {
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            **_record_lifecycle_payload(record),
            "cancel_requested": True,
            "error": record.error,
            "error_code": record.error_code,
        }

    def interrupt_turn(self, run_id: str, token: str) -> dict[str, Any]:
        backend = self._backend
        backend._authorize_run(run_id, token)
        record = backend._record(run_id)
        with backend._lock:
            loop = record.loop
            terminal = _record_terminal(record)
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
        backend = self._backend
        backend._authorize_run(run_id, token)
        record = backend._record(run_id)
        with backend._lock:
            loop = record.loop
            terminal = _record_terminal(record)
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
        backend = self._backend
        backend._authorize_run(run_id, token)
        record = backend._record(run_id)
        with backend._lock:
            terminal = _record_terminal(record)
        if terminal:
            return {"run_id": run_id, **_record_lifecycle_payload(record), "resumed": False}
        backend._call_soon(record.message_queue.put_nowait, self._resume_signal)
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
        backend = self._backend
        backend._authorize_run(run_id, token)
        record = backend._record(run_id)
        with backend._lock:
            loop = record.loop
            terminal = _record_terminal(record)
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
        backend = self._backend
        backend._authorize_run(run_id, token)
        record = backend._record(run_id)
        if message_id and message_id in record.seen_inbox_ids:
            return {"run_id": run_id, "status": "duplicate", "message_id": message_id}
        message = _normalize_inbound_message(content)
        if isinstance(message, list) and backend.checkpoint_store is not None:
            pending: dict[str, bytes] = {}
            message = normalize_inline_media_dicts(message, pending)
            for data in pending.values():
                backend.checkpoint_store.put_blob(run_id, data)
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
        if wire_bytes > backend.max_message_bytes:
            raise ValueError(f"message exceeds the {backend.max_message_bytes}-byte limit")
        with backend._lock:
            if _record_terminal(record):
                raise ValueError("cannot send a message to a terminal run")
            if record.message_queue.qsize() >= backend.max_message_queue_depth:
                raise ValueError("message queue is full; retry once the run drains it")
        backend._call_soon(record.message_queue.put_nowait, envelope.to_json())
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
        backend = self._backend
        loop = self.authorize_task_result(run_id, token, task_id)
        reported = loop.report_task_result(
            task_id,
            result,
            status=status,
            persist_checkpoint=False,
        )
        backend._persist_run_checkpoint_from_any_thread(backend._record(run_id))
        return reported

    def create_task(
        self,
        run_id: str,
        token: str,
        *,
        kind: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        backend = self._backend
        loop = self.authorize_active_loop(run_id, token)
        record = backend._record(run_id)
        task_id = loop.create_task(kind, request)
        callback_token = backend.token_manager.issue(
            kind="task_callback",
            audience=TASK_CALLBACK_AUDIENCE,
            run_id=run_id,
            tenant_id=record.tenant_id,
            user_id=record.user_id,
            ttl_s=backend.task_callback_token_ttl_s,
            metadata={"task_id": task_id},
        )
        return {
            "task_id": task_id,
            "callback_token": callback_token,
            "callback_url": f"/v1/runs/{run_id}/tasks/{task_id}/result",
        }

    def resume_run(self, run_id: str, token: str) -> dict[str, Any]:
        backend = self._backend
        claims = backend._verify_run_token(run_id, token)
        with backend._lock:
            existing = backend._records.get(run_id)
        if existing is not None:
            if claims.tenant_id != existing.tenant_id or claims.user_id != existing.user_id:
                raise PermissionDenied("token subject mismatch")
            return {"run_id": run_id, **_record_lifecycle_payload(existing), "resumed": False}
        if any(sep in run_id for sep in ("/", "\\")) or ".." in run_id:
            raise PermissionDenied("invalid run id")
        run_dir = backend.run_root / run_id
        meta = backend._read_recovery_meta(run_dir, run_id)
        if meta is None:
            raise KeyError(f"unknown run: {run_id}")
        if claims.tenant_id != (meta.get("tenant_id") or "") or claims.user_id != (meta.get("user_id") or ""):
            raise PermissionDenied("token subject mismatch")
        if (run_dir / "failure.json").exists():
            raise ValueError("run is marked unrecoverable; inspect failure.json")
        assert backend.checkpoint_store is not None
        stored = backend.checkpoint_store.latest(run_id)
        if stored is None or stored.checkpoint.terminal:
            raise ValueError("run has no resumable checkpoint")
        if not backend._attempt_resume(run_dir, run_id):
            raise ValueError("resume failed; inspect run logs / failure.json")
        record = backend._record(run_id)
        return {"run_id": run_id, **_record_lifecycle_payload(record), "resumed": True}

    def authorize_active_loop(self, run_id: str, token: str) -> Any:
        self._backend._authorize_run(run_id, token)
        return self.active_loop(run_id)

    def authorize_task_result(self, run_id: str, token: str, task_id: str) -> Any:
        try:
            self.verify_task_callback_token(run_id, token, task_id)
        except TokenError:
            self._backend._authorize_run(run_id, token)
        return self.active_loop(run_id)

    def active_loop(self, run_id: str) -> Any:
        backend = self._backend
        record = backend._record(run_id)
        with backend._lock:
            if _record_terminal(record):
                raise ValueError("cannot drive tasks for a terminal run")
            loop = record.loop
        if loop is None:
            raise ValueError("run has not started")
        return loop

    def authorize_claim_subject(self, run_id: str, claims: Any) -> None:
        backend = self._backend
        with backend._lock:
            record = backend._records.get(run_id)
        if record is not None and (claims.tenant_id != record.tenant_id or claims.user_id != record.user_id):
            raise PermissionDenied("token subject mismatch")

    def verify_task_callback_token(self, run_id: str, token: str, task_id: str) -> None:
        claims = self._backend.token_manager.verify(
            token, kind="task_callback", audience=TASK_CALLBACK_AUDIENCES, run_id=run_id
        )
        if str(claims.metadata.get("task_id") or "") != task_id:
            raise PermissionDenied("callback token does not match this task")
        self.authorize_claim_subject(run_id, claims)
