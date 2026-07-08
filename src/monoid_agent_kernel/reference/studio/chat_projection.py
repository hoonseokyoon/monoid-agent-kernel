"""Durable browser-facing chat projection for Agent Studio."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

CHAT_SCHEMA_VERSION = "studio.chat.v1"
CHAT_MESSAGE_SCHEMA_VERSION = "studio.chat.message.v1"
CHAT_FILE_NAME = "studio.chat.jsonl"

_ASSISTANT_EVENT_TYPES = {"turn.settled"}
_ERROR_EVENT_TYPES = {"turn.failed", "run.failed", "ModelAdapterError"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _write_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, ensure_ascii=False) + "\n")


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
        cursor = -1
        for record in self.read():
            if record.get("role") not in {"assistant", "error"}:
                continue
            source = record.get("source") if isinstance(record.get("source"), dict) else {}
            try:
                cursor = max(cursor, int(source.get("seq")))
            except (TypeError, ValueError):
                continue
        return cursor

    def response(self, run_id: str) -> dict[str, Any]:
        return {
            "schema_version": CHAT_SCHEMA_VERSION,
            "run_id": run_id,
            "messages": self.read(),
            "event_cursor": self.event_cursor(),
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
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
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

    def project_events(self, events: Iterable[Mapping[str, Any]]) -> None:
        for event in events:
            record = self._record_from_event(event)
            if record is None:
                continue
            event_id = str(record["source"].get("event_id") or "")
            seq = record["source"].get("seq")
            if event_id and self._has_source("event_id", event_id):
                continue
            if seq is not None and self._has_source("seq", seq):
                continue
            _write_jsonl(self.path, record)

    def catch_up(self, run_id: str) -> dict[str, Any]:
        self.ensure_legacy_user_from_run_meta()
        self.project_events(_read_jsonl(self.run_dir / "events.jsonl"))
        return self.response(run_id)

    def _record_from_event(self, event: Mapping[str, Any]) -> dict[str, Any] | None:
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

    def _has_source(self, key: str, value: Any) -> bool:
        for record in self.read():
            source = record.get("source") if isinstance(record.get("source"), dict) else {}
            if source.get(key) == value:
                return True
        return False
