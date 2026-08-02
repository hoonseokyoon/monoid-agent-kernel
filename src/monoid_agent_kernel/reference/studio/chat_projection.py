"""Durable browser-facing chat projection for Agent Studio."""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from monoid_agent_kernel.core._event_log import read_committed_event_payloads
from monoid_agent_kernel.core.json_ingress import loads_json_ingress
from monoid_agent_kernel.core.model_content import (
    ModelContentSnapshot,
    active_model_content_state,
    read_model_content,
    watch_active_model_content,
)
from monoid_agent_kernel.reference.backend.content_hydration import hydrate_settled_text

CHAT_SCHEMA_V1 = "studio.chat.v1"
CHAT_SCHEMA_V2 = "studio.chat.v2"
# v2 adds the required ``event_log_error`` member to the HTTP response. Keeping the v1
# discriminator would make the expanded object invalid for strict v1 consumers.
CHAT_SCHEMA_VERSION = CHAT_SCHEMA_V2
SUPPORTED_CHAT_SCHEMA_VERSIONS = (CHAT_SCHEMA_V1, CHAT_SCHEMA_V2)
CHAT_MESSAGE_SCHEMA_VERSION = "studio.chat.message.v1"
CHAT_FILE_NAME = "studio.chat.jsonl"

_ASSISTANT_EVENT_TYPES = {"turn.settled", "turn.interrupted"}
_ERROR_EVENT_TYPES = {"turn.failed", "run.failed", "ModelAdapterError"}
_TERMINAL_PARTIAL_EVENT_TYPES = {"turn.failed", "turn.interrupted"}


def _is_chat_message(payload: object) -> bool:
    """Validate renderer-required fields while permitting additive message metadata."""
    if not isinstance(payload, dict):
        return False
    message_id = payload.get("id")
    role = payload.get("role")
    attachments = payload.get("attachments")
    created_at = payload.get("created_at")
    try:
        created_at_is_valid = (
            isinstance(created_at, (int, float))
            and not isinstance(created_at, bool)
            and math.isfinite(created_at)
        )
    except OverflowError:
        created_at_is_valid = False
    return (
        isinstance(message_id, str)
        and bool(message_id.strip())
        and isinstance(role, str)
        and role in {"user", "assistant", "error"}
        and isinstance(payload.get("content"), str)
        and isinstance(attachments, list)
        and all(
            isinstance(attachment, dict)
            and isinstance(attachment.get("name"), str)
            and isinstance(attachment.get("mime"), str)
            for attachment in attachments
        )
        and created_at_is_valid
        and ("source" not in payload or isinstance(payload.get("source"), dict))
    )


def is_supported_chat_response(payload: object) -> bool:
    """Whether ``payload`` is one exact Studio chat response shape this release reads."""
    if not isinstance(payload, dict):
        return False
    version = payload.get("schema_version")
    keys = {"schema_version", "run_id", "messages", "event_cursor"}
    if version == CHAT_SCHEMA_V2:
        keys.add("event_log_error")
        if not isinstance(payload.get("event_log_error"), str):
            return False
    elif version != CHAT_SCHEMA_V1:
        return False
    messages = payload.get("messages")
    messages_are_valid = isinstance(messages, list) and all(
        _is_chat_message(message) for message in messages
    )
    message_ids = (
        [message["id"] for message in messages if isinstance(message, dict)]
        if messages_are_valid
        else []
    )
    cursor = payload.get("event_cursor")
    return (
        set(payload) == keys
        and isinstance(payload.get("run_id"), str)
        and messages_are_valid
        and len(message_ids) == len(set(message_ids))
        and isinstance(cursor, int)
        and not isinstance(cursor, bool)
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = loads_json_ingress(line)
            except ValueError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _write_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        )


def _read_committed_events(path: Path) -> tuple[list[dict[str, Any]], str]:
    # Studio's catch-up reads events.jsonl straight off disk rather than through the backend
    # projection, so it needs hydration of its own. Without it a restored session renders empty
    # assistant bubbles while the live SSE page — which does go through the projection — shows the
    # text, and `_record_from_event` drops a message whose content is empty rather than showing
    # anything is missing.
    #
    # Through the lenient reader for the same reason `core.projections` uses it: this ran under a
    # `do_GET` with no exception handler, so one corrupt byte killed the request mid-response and
    # the session rendered as if it had no history. The reason is carried out to the caller and
    # published, because a transcript that silently stops at the damage looks like a short
    # conversation rather than a truncated one.
    read = read_committed_event_payloads(path)
    return hydrate_settled_text(read.payloads, path.parent), read.corruption


def _sorted_chat_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: tuple[int, dict[str, Any]]) -> tuple[float, int]:
        index, record = item
        try:
            created_at = float(record.get("created_at"))
        except (TypeError, ValueError):
            created_at = float("inf")
        return (created_at, index)

    return [record for _, record in sorted(enumerate(records), key=key)]


def _event_time(event: Mapping[str, Any]) -> float:
    raw = event.get("timestamp")
    if isinstance(raw, str) and raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return time.time()


def _provider_detail(data: Mapping[str, Any], message: str) -> str:
    parts: list[str] = []
    provider_code = data.get("provider_error_code")
    if provider_code and str(provider_code) not in message:
        parts.append(str(provider_code))
    http_status = data.get("http_status")
    if http_status:
        parts.append(f"HTTP {http_status}")
    return f" - {' · '.join(parts)}" if parts else ""


def _attachment_metadata(attachments: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for attachment in attachments:
        out.append(
            {
                "name": str(attachment.get("name") or "file"),
                "mime": str(attachment.get("mime") or "application/octet-stream"),
            }
        )
    return out


class ChatProjection:
    """Append-only Studio chat projection stored beside a run."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.path = run_dir / CHAT_FILE_NAME

    def read(self) -> list[dict[str, Any]]:
        return _sorted_chat_records(_read_jsonl(self.path))

    def event_cursor(self) -> int:
        return self._event_cursor_for(self.read())

    def response(
        self,
        run_id: str,
        *,
        event_log_error: str = "",
        include_model_stream_partials: bool = True,
    ) -> dict[str, Any]:
        messages = self.read()
        if not include_model_stream_partials:
            messages = [
                message
                for message in messages
                if (
                    message.get("source", {}).get("kind")
                    if isinstance(message.get("source"), dict)
                    else ""
                )
                != "model_stream_partial"
            ]
        return {
            "schema_version": CHAT_SCHEMA_VERSION,
            "run_id": run_id,
            "messages": messages,
            "event_cursor": self._event_cursor_for(messages),
            # Always present, empty when the log read cleanly, so a client tests one field rather
            # than inferring truncation from a transcript that looks complete.
            "event_log_error": event_log_error,
        }

    def append_user(
        self,
        *,
        content: str,
        attachments: Sequence[Mapping[str, Any]] = (),
        client_message_id: str = "",
        created_at: float | None = None,
        legacy: bool = False,
    ) -> dict[str, Any] | None:
        message_id = client_message_id.strip() or f"studio_user_{time.time_ns()}"
        if self._has_source("client_message_id", message_id):
            return None
        record = {
            "schema_version": CHAT_MESSAGE_SCHEMA_VERSION,
            "id": message_id,
            "role": "user",
            "content": content,
            "attachments": _attachment_metadata(attachments),
            "created_at": created_at if created_at is not None else time.time(),
            "source": {
                "kind": "client" if not legacy else "legacy",
                "client_message_id": message_id,
                "legacy": legacy,
            },
        }
        _write_jsonl(self.path, record)
        return record

    def ensure_legacy_user_from_run_meta(self) -> None:
        if any(record.get("role") == "user" for record in self.read()):
            return
        meta_path = self.run_dir / "run.json"
        try:
            meta = loads_json_ingress(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(meta, dict):
            return
        title = str(meta.get("title") or "").strip()
        if not title:
            return
        created_at = meta.get("created_at")
        self.append_user(
            content=title,
            client_message_id=f"legacy:{self.run_dir.name}:title",
            created_at=float(created_at) if isinstance(created_at, (int, float)) else None,
            legacy=True,
        )

    def project_events(
        self,
        events: Iterable[Mapping[str, Any]],
        *,
        include_model_stream_partials: bool = True,
        root_run_id: str | None = None,
    ) -> None:
        terminal_partials: (
            dict[tuple[str, str, str, str], ModelContentSnapshot] | None
        ) = None
        for event in events:
            event_type = str(event.get("type") or "")
            data = event.get("data") if isinstance(event.get("data"), dict) else {}
            if (
                include_model_stream_partials
                and event_type in _TERMINAL_PARTIAL_EVENT_TYPES
                and not (event_type == "turn.failed" and data.get("retryable"))
                and terminal_partials is None
            ):
                terminal_partials = self._terminal_model_stream_snapshots(
                    root_run_id=root_run_id
                )
            records = self._records_from_event(
                event,
                terminal_partials=terminal_partials or {},
            )
            for record in records:
                # One non-retryable failure owns two durable chat rows: its authored partial and
                # its error. Scope the legacy event-id/seq fallback by source kind so those rows
                # cannot suppress each other, while deterministic message ids remain authoritative.
                if self._has_projection_record(record):
                    continue
                _write_jsonl(self.path, record)

    def catch_up(
        self,
        run_id: str,
        *,
        include_model_stream_partials: bool = True,
    ) -> dict[str, Any]:
        self.ensure_legacy_user_from_run_meta()
        events, event_log_error = _read_committed_events(self.run_dir / "events.jsonl")
        self.project_events(
            events,
            include_model_stream_partials=include_model_stream_partials,
            root_run_id=run_id,
        )
        response = self.response(
            run_id,
            event_log_error=event_log_error,
            include_model_stream_partials=include_model_stream_partials,
        )
        superseded_turn_ids = {
            str(event.get("turn_id"))
            for event in events
            if str(event.get("type") or "") == "run.resumed"
            and event.get("run_id") == run_id
            and isinstance(event.get("turn_id"), str)
            and isinstance(event.get("data"), dict)
            and event["data"].get("reason") == "studio-retry"
        }
        if superseded_turn_ids:
            response["messages"] = [
                message
                for message in response["messages"]
                if not self._is_superseded_failed_partial(
                    message,
                    root_run_id=run_id,
                    turn_ids=superseded_turn_ids,
                )
            ]
            response["event_cursor"] = self._event_cursor_for(response["messages"])
        active = (
            self._active_model_stream_records(events, root_run_id=run_id)
            if include_model_stream_partials and not event_log_error
            else []
        )
        if active:
            existing_ids = {str(message.get("id") or "") for message in response["messages"]}
            response["messages"] = _sorted_chat_records(
                [
                    *response["messages"],
                    *(record for record in active if record["id"] not in existing_ids),
                ]
            )
        return response

    @staticmethod
    def _event_cursor_for(records: Iterable[Mapping[str, Any]]) -> int:
        cursor = -1
        for record in records:
            if record.get("role") not in {"assistant", "error"}:
                continue
            source = record.get("source") if isinstance(record.get("source"), dict) else {}
            try:
                cursor = max(cursor, int(source.get("seq")))
            except (TypeError, ValueError):
                continue
        return cursor

    def _active_model_stream_records(
        self,
        events: Iterable[Mapping[str, Any]],
        *,
        root_run_id: str,
    ) -> list[dict[str, Any]]:
        """Project durable partial prefixes for model calls with no committed terminal boundary.

        These records stay response-only. Persisting a still-growing prefix into the append-only
        chat sidecar would leave a stale duplicate once live delivery or an interruption completes.
        """

        active: dict[tuple[str, str], Mapping[str, Any]] = {}
        for event in events:
            event_type = str(event.get("type") or "")
            run_id = event.get("run_id")
            turn_id = event.get("turn_id")
            if event_type == "model.turn.started":
                if isinstance(run_id, str) and run_id and isinstance(turn_id, str) and turn_id:
                    active[(run_id, turn_id)] = event
                continue
            if event_type in {
                "model.turn.finished",
                "turn.failed",
                "turn.interrupted",
                "turn.settled",
            }:
                if isinstance(run_id, str) and isinstance(turn_id, str):
                    active.pop((run_id, turn_id), None)
                continue
            if event_type in {"run.failed", "run.finished"} and isinstance(run_id, str):
                active = {key: value for key, value in active.items() if key[0] != run_id}
        if not active:
            return []
        latest: dict[
            tuple[str, str],
            tuple[tuple[int, str, str], ModelContentSnapshot],
        ] = {}
        try:
            with watch_active_model_content(self.run_dir) as mutation_watch:
                before_read = active_model_content_state(self.run_dir)
                if not before_read.stream_ids or before_read.file_identity is None:
                    return []
                content = read_model_content(
                    self.run_dir,
                    expected_identity=before_read.file_identity,
                )
                after_read = active_model_content_state(self.run_dir)
                if (
                    after_read.file_identity is not None
                    and after_read.file_identity != before_read.file_identity
                ):
                    return []
                active_stream_ids = before_read.stream_ids & after_read.stream_ids
                if mutation_watch.changed:
                    return []
        except OSError:
            return []
        # A writer can settle or be abandoned while the sidecar is being read. Promote only ids
        # that were active on both sides of that read, giving the response a clear linearization
        # boundary instead of reviving a stream that already closed.
        if not active_stream_ids:
            return []
        for snapshot in content.snapshots:
            context = snapshot.context
            # This ChatProjection is root-scoped. A foreign/child snapshot sharing a turn id never
            # becomes the root assistant's content.
            if context.root_run_id != root_run_id or context.run_id != root_run_id:
                continue
            if context.stream_id not in active_stream_ids:
                continue
            started = active.get((context.run_id, context.turn_id))
            content = snapshot.best_output_text
            if started is None or not content:
                continue
            key = (context.run_id, context.turn_id)
            order = (context.step, context.started_at, context.stream_id)
            previous = latest.get(key)
            if previous is None or order >= previous[0]:
                latest[key] = (order, snapshot)

        records: list[dict[str, Any]] = []
        for key, (_order, snapshot) in latest.items():
            context = snapshot.context
            started = active[key]
            content = snapshot.best_output_text
            source = {
                "kind": "model_stream_active",
                "event_type": "model.turn.started",
                "event_id": str(started.get("event_id") or ""),
                "seq": started.get("seq"),
                "turn_id": context.turn_id,
                "stream_id": context.stream_id,
                "status": snapshot.status,
                "partial": True,
            }
            records.append(
                {
                    "schema_version": CHAT_MESSAGE_SCHEMA_VERSION,
                    "id": f"assistant:model-stream:{context.stream_id}:active",
                    "role": "assistant",
                    "content": content,
                    "attachments": [],
                    "created_at": _event_time(started),
                    "source": source,
                }
            )
        return records

    def _terminal_model_stream_snapshots(
        self,
        *,
        root_run_id: str | None,
    ) -> dict[tuple[str, str, str, str], ModelContentSnapshot]:
        """Select the latest displayable terminal prefix for each exact root turn."""

        latest: dict[
            tuple[str, str, str, str],
            tuple[tuple[int, str, str], ModelContentSnapshot],
        ] = {}
        for snapshot in read_model_content(self.run_dir).snapshots:
            context = snapshot.context
            if snapshot.status not in {"failed", "interrupted"}:
                continue
            if snapshot.status == "failed" and snapshot.retryable:
                continue
            if not snapshot.best_output_text:
                continue
            if root_run_id is not None and (
                context.root_run_id != root_run_id or context.run_id != root_run_id
            ):
                continue
            key = (
                context.root_run_id,
                context.run_id,
                context.turn_id,
                snapshot.status,
            )
            order = (context.step, context.started_at, context.stream_id)
            previous = latest.get(key)
            if previous is None or order >= previous[0]:
                latest[key] = (order, snapshot)
        return {key: value[1] for key, value in latest.items()}

    def _records_from_event(
        self,
        event: Mapping[str, Any],
        *,
        terminal_partials: Mapping[tuple[str, str, str, str], ModelContentSnapshot],
    ) -> tuple[dict[str, Any], ...]:
        event_type = str(event.get("type") or "")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        partial: dict[str, Any] | None = None
        if event_type in _TERMINAL_PARTIAL_EVENT_TYPES and not (
            event_type == "turn.failed" and data.get("retryable")
        ):
            root_run_id = event.get("run_id")
            turn_id = event.get("turn_id")
            if isinstance(root_run_id, str) and isinstance(turn_id, str):
                status = "failed" if event_type == "turn.failed" else "interrupted"
                snapshot = terminal_partials.get(
                    (root_run_id, root_run_id, turn_id, status)
                )
                if snapshot is not None:
                    partial = self._partial_record_from_event(event, snapshot)

        if event_type == "turn.interrupted":
            return (partial,) if partial is not None else ()
        record = self._record_from_event(event)
        if partial is None:
            return (record,) if record is not None else ()
        return (partial, record) if record is not None else (partial,)

    @staticmethod
    def _partial_record_from_event(
        event: Mapping[str, Any],
        snapshot: ModelContentSnapshot,
    ) -> dict[str, Any]:
        context = snapshot.context
        seq = event.get("seq")
        source = {
            "kind": "model_stream_partial",
            "event_type": str(event.get("type") or ""),
            "event_id": str(event.get("event_id") or ""),
            "seq": int(seq) if isinstance(seq, int) else seq,
            "root_run_id": context.root_run_id,
            "run_id": context.run_id,
            "turn_id": context.turn_id,
            "stream_id": context.stream_id,
            "status": snapshot.status,
            "partial": True,
            "retryable": snapshot.retryable,
        }
        created_at = _event_time(event)
        if context.started_at:
            try:
                created_at = datetime.fromisoformat(
                    context.started_at.replace("Z", "+00:00")
                ).timestamp()
            except ValueError:
                pass
        return {
            "schema_version": CHAT_MESSAGE_SCHEMA_VERSION,
            "id": f"assistant:model-stream:{context.stream_id}:partial",
            "role": "assistant",
            "content": snapshot.best_output_text,
            "attachments": [],
            "created_at": created_at,
            "source": source,
        }

    def _record_from_event(
        self,
        event: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        event_type = str(event.get("type") or "")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if event_type in _ASSISTANT_EVENT_TYPES:
            content = str(data.get("final_text") or "")
            if not content:
                return None
            role = "assistant"
        elif event_type in _ERROR_EVENT_TYPES:
            if event_type == "turn.failed" and data.get("retryable"):
                return None
            content = str(data.get("error") or data.get("message") or "the run failed")
            content += _provider_detail(data, content)
            role = "error"
        else:
            return None
        seq = event.get("seq")
        source: dict[str, Any] = {
            "kind": "event",
            "event_type": event_type,
            "event_id": str(event.get("event_id") or ""),
            "seq": int(seq) if isinstance(seq, int) else seq,
        }
        event_id = source["event_id"] or f"seq:{source['seq']}"
        return {
            "schema_version": CHAT_MESSAGE_SCHEMA_VERSION,
            "id": f"{role}:{event_id}",
            "role": role,
            "content": content,
            "attachments": [],
            "created_at": _event_time(event),
            "source": source,
        }

    def _has_projection_record(self, candidate: Mapping[str, Any]) -> bool:
        candidate_id = candidate.get("id")
        candidate_source = (
            candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
        )
        candidate_kind = candidate_source.get("kind")
        candidate_event_id = candidate_source.get("event_id")
        candidate_seq = candidate_source.get("seq")
        for record in self.read():
            if record.get("id") == candidate_id:
                return True
            source = record.get("source") if isinstance(record.get("source"), dict) else {}
            if source.get("kind") != candidate_kind:
                continue
            if candidate_event_id and source.get("event_id") == candidate_event_id:
                return True
            if candidate_seq is not None and source.get("seq") == candidate_seq:
                return True
        return False

    @staticmethod
    def _is_superseded_failed_partial(
        message: Mapping[str, Any],
        *,
        root_run_id: str,
        turn_ids: set[str],
    ) -> bool:
        source = message.get("source") if isinstance(message.get("source"), dict) else {}
        return (
            source.get("kind") == "model_stream_partial"
            and source.get("status") == "failed"
            and source.get("root_run_id") == root_run_id
            and source.get("run_id") == root_run_id
            and source.get("turn_id") in turn_ids
        )

    def _has_source(self, key: str, value: Any) -> bool:
        for record in self.read():
            source = record.get("source") if isinstance(record.get("source"), dict) else {}
            if source.get(key) == value:
                return True
        return False
