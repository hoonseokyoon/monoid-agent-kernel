"""Helpers for run event sequence ownership and diagnostics windows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from monoid_agent_kernel.core.lifecycle import (
    TERMINAL_STATES,
    SessionState,
    lifecycle_from_status_artifact,
    session_state_from_run_status,
)

DIRECT_AUDIT_APPEND_STATUSES = frozenset({"completed", "failed", "cancelled"})

DEFAULT_DIAGNOSTIC_EVENT_DATA_KEYS = frozenset(
    {
        "attempts",
        "actor",
        "binding_id",
        "call_id",
        "capability",
        "child_run_id",
        "command",
        "command_id",
        "error",
        "error_code",
        "failure_code",
        "idempotency_key",
        "job_id",
        "reason",
        "request_id",
        "result_code",
        "run_id",
        "state",
        "status",
        "target_run_id",
        "task_id",
        "tool",
        "traceparent",
    }
)


def read_event_page(events_path: Path, *, from_seq: int, limit: int | None) -> dict[str, Any]:
    """Read a monotonic event page from an append-only run event log."""
    if from_seq < 0:
        raise ValueError("from_seq must be non-negative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    events: list[dict[str, Any]] = []
    next_seq = from_seq
    has_more = False
    if events_path.exists():
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                seq = int(event.get("seq") or 0)
                if seq < from_seq:
                    continue
                if limit is not None and len(events) >= limit:
                    has_more = True
                    break
                events.append(event)
                next_seq = seq + 1
    return {"events": events, "next_seq": next_seq, "has_more": has_more}


def diagnostic_event_summary(
    event: Mapping[str, Any],
    *,
    data_keys: frozenset[str] = DEFAULT_DIAGNOSTIC_EVENT_DATA_KEYS,
) -> dict[str, Any]:
    """Return the bounded event projection used by diagnostics APIs."""
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    return {
        "seq": event.get("seq"),
        "type": event.get("type"),
        "timestamp": event.get("timestamp"),
        "level": event.get("level"),
        "turn_id": event.get("turn_id"),
        "parent_id": event.get("parent_id"),
        "data": {key: data[key] for key in sorted(data_keys) if key in data},
    }


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


@dataclass(frozen=True)
class RunEventSequencer:
    """Decision helper for event sequence ownership around backend control events."""

    direct_append_statuses: frozenset[str] = field(default_factory=lambda: DIRECT_AUDIT_APPEND_STATUSES)

    def is_queued_before_recorder(self, state: str | SessionState) -> bool:
        """Queued runs may seed the event log before the recorder opens."""
        return session_state_from_run_status(state) is SessionState.CREATED

    def is_terminal_direct_append_status(
        self,
        state: str | SessionState,
        *,
        terminal: bool = False,
    ) -> bool:
        """Terminal run dirs may receive guarded control audit appends."""
        session_state = session_state_from_run_status(state, terminal=terminal)
        return terminal or session_state in TERMINAL_STATES

    def requires_live_sequence_owner(
        self,
        state: str | SessionState,
        *,
        terminal: bool = False,
    ) -> bool:
        """Live non-terminal records should write through the live recorder."""
        return not self.is_queued_before_recorder(state) and not self.is_terminal_direct_append_status(
            state,
            terminal=terminal,
        )

    def run_dir_allows_direct_append(self, run_dir: Path) -> bool:
        payload = _read_optional_json(run_dir / "status.json")
        if payload is None:
            return False
        state, terminal = lifecycle_from_status_artifact(payload)
        return self.is_terminal_direct_append_status(state, terminal=terminal)

    def newest_sequence(self, status: Mapping[str, Any], status_file: Mapping[str, Any] | None = None) -> int:
        """Return the newest event sequence visible across live and durable projections."""
        status_file = status_file or {}
        return max(
            int(status.get("last_event_seq") or 0),
            int(status_file.get("last_event_seq") or 0),
        )

    def diagnostics_from_seq(
        self,
        status: Mapping[str, Any],
        status_file: Mapping[str, Any] | None,
        *,
        event_limit: int,
    ) -> int:
        """Return the starting sequence for a bounded diagnostics tail window."""
        if event_limit < 1:
            raise ValueError("event_limit must be positive")
        last_event_seq = self.newest_sequence(status, status_file)
        return max(0, last_event_seq - event_limit + 1) if last_event_seq else 0

    def read_event_page(self, events_path: Path, *, from_seq: int, limit: int | None) -> dict[str, Any]:
        return read_event_page(events_path, from_seq=from_seq, limit=limit)

    def diagnostic_event_summary(self, event: Mapping[str, Any]) -> dict[str, Any]:
        return diagnostic_event_summary(event)
