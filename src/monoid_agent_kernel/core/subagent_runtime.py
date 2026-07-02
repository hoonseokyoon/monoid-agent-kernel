"""Helpers for subagent identity, lineage, and diagnostics projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from monoid_agent_kernel.core.trace_context import new_traceparent

SUBAGENT_EVENT_TYPES = frozenset({"subagent.started", "subagent.finished", "subagent.failed"})


@dataclass(frozen=True)
class SubagentRuntimeContext:
    """Stable identity envelope for one child agent run."""

    root_run_id: str
    parent_run_id: str
    child_run_id: str
    task_id: str
    definition_id: str
    depth: int
    traceparent: str
    subagent_type: str

    @classmethod
    def create(
        cls,
        *,
        parent_run_id: str,
        task_id: str,
        definition_id: str,
        parent_depth: int,
        root_run_id: str | None = None,
        traceparent: str | None = None,
    ) -> SubagentRuntimeContext:
        child_run_id = f"{parent_run_id}.sub.{task_id}"
        return cls(
            root_run_id=str(root_run_id or parent_run_id),
            parent_run_id=parent_run_id,
            child_run_id=child_run_id,
            task_id=task_id,
            definition_id=definition_id,
            depth=parent_depth + 1,
            traceparent=str(traceparent or new_traceparent()),
            subagent_type=definition_id,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "root_run_id": self.root_run_id,
            "parent_run_id": self.parent_run_id,
            "child_run_id": self.child_run_id,
            "task_id": self.task_id,
            "definition_id": self.definition_id,
            "depth": self.depth,
            "traceparent": self.traceparent,
            "subagent_type": self.subagent_type,
        }

    def child_metadata(self) -> dict[str, Any]:
        return {
            **self.to_json(),
            # Legacy aliases kept for existing readers.
            "parent_task_id": self.task_id,
            "subagent_definition_id": self.definition_id,
            "subagent_depth": self.depth,
        }

    def started_event_data(self, *, background: bool) -> dict[str, Any]:
        return {**self.to_json(), "background": background}

    def terminal_event_data(
        self,
        *,
        status: str,
        usage: Mapping[str, Any],
        error: str,
        error_code: str,
    ) -> dict[str, Any]:
        return {
            **self.to_json(),
            "status": status,
            "usage": dict(usage),
            "error": error,
            "error_code": error_code,
        }

    def result_payload(
        self,
        *,
        status: str,
        final_text: str,
        error: str,
        usage: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "type": "subagent_result",
            **self.to_json(),
            "status": status,
            "message": final_text,
            "answer": final_text,
            "final_text": final_text,
            "error": error,
            "usage": dict(usage),
        }


def validate_descendant_run_id(ancestor_run_id: str, descendant_run_id: str) -> None:
    """Raise when ``descendant_run_id`` is outside the ancestor's subagent lineage."""
    if any(sep in descendant_run_id for sep in ("/", "\\")) or ".." in descendant_run_id:
        raise ValueError("invalid descendant run id")
    if descendant_run_id != ancestor_run_id and not descendant_run_id.startswith(f"{ancestor_run_id}.sub."):
        raise ValueError("run is not a descendant of the authorized run")


def is_descendant_run_id(ancestor_run_id: str, descendant_run_id: str) -> bool:
    try:
        validate_descendant_run_id(ancestor_run_id, descendant_run_id)
    except ValueError:
        return False
    return True


def subagent_diagnostics_from_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a bounded diagnostics projection from parent subagent lifecycle events."""
    by_child: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for event in events:
        event_type = str(event.get("type") or "")
        if event_type not in SUBAGENT_EVENT_TYPES:
            continue
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        child_run_id = str(data.get("child_run_id") or "")
        if not child_run_id:
            continue
        item = by_child.get(child_run_id)
        if item is None:
            item = _base_summary(event, data)
            by_child[child_run_id] = item
            order.append(child_run_id)
        if event_type == "subagent.started":
            item["started_seq"] = event.get("seq")
            item["background"] = bool(data.get("background", False))
        else:
            item["terminal_seq"] = event.get("seq")
            item["status"] = str(data.get("status") or ("failed" if event_type == "subagent.failed" else "completed"))
            item["event_type"] = event_type
            item["usage"] = dict(data.get("usage") or {})
            error = str(data.get("error") or "")
            error_code = str(data.get("error_code") or "")
            if error:
                item["error"] = error
            if error_code:
                item["error_code"] = error_code
    items = [by_child[child_run_id] for child_run_id in order]
    return {"count": len(items), "items": items}


def _base_summary(event: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "child_run_id": str(data.get("child_run_id") or ""),
        "parent_run_id": str(data.get("parent_run_id") or ""),
        "root_run_id": str(data.get("root_run_id") or ""),
        "task_id": str(data.get("task_id") or ""),
        "definition_id": str(data.get("definition_id") or ""),
        "subagent_type": str(data.get("subagent_type") or data.get("definition_id") or ""),
        "depth": int(data.get("depth") or 0),
        "traceparent": str(data.get("traceparent") or ""),
        "status": "started",
        "event_type": str(event.get("type") or ""),
        "started_seq": event.get("seq"),
        "terminal_seq": None,
        "usage": {},
    }
