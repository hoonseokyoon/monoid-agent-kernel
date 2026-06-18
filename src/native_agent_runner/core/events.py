from __future__ import annotations

import uuid
import threading
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from native_agent_runner.core._util import utc_timestamp

EVENT_SCHEMA_VERSION = "native-agent-runner.event.v1"

AgentEventType = Literal[
    "run.started",
    "run.finished",
    "run.failed",
    "run.waiting",
    "run.resumed",
    "turn.settled",
    "checkpoint.committed",
    "agent.config.updated",
    "model.turn.started",
    "model.turn.finished",
    "model.input.degraded",
    "tool.call.started",
    "tool.call.finished",
    "tool.call.failed",
    "tool.surface.updated",
    "tool.approval.requested",
    "tool.approval.approved",
    "tool.approval.denied",
    "shell.exec.started",
    "shell.exec.finished",
    "shell.exec.failed",
    "job.started",
    "job.output.updated",
    "job.finished",
    "job.timed_out",
    "job.cancelled",
    "job.output_limited",
    "job.failed",
    "web.search.started",
    "web.search.finished",
    "web.search.failed",
    "web.fetch.started",
    "web.fetch.finished",
    "web.fetch.failed",
    "web.context.started",
    "web.context.finished",
    "web.context.failed",
    "permission.denied",
    "workspace.file.read",
    "workspace.file.changed",
    "workspace.diff.updated",
    "workspace.proposal.updated",
    "proposal.ready",
    "proposal.package.exported",
    "proposal.approved",
    "proposal.rejected",
    "proposal.applied",
    "proposal.conflict",
    "proposal.stale",
    "artifact.emitted",
    "plan.updated",
    "metrics.updated",
]

AgentEventLevel = Literal["debug", "info", "warning", "error"]


@dataclass(frozen=True)
class AgentEvent:
    schema_version: str
    event_id: str
    seq: int
    run_id: str
    timestamp: str
    type: AgentEventType
    level: AgentEventLevel = "info"
    data: dict[str, Any] = field(default_factory=dict)
    turn_id: str | None = None
    parent_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "seq": self.seq,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "parent_id": self.parent_id,
            "timestamp": self.timestamp,
            "type": self.type,
            "level": self.level,
            "data": self.data,
        }


class EventSink(Protocol):
    def emit(self, event: AgentEvent) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass
class EventBus:
    run_id: str
    sinks: tuple[EventSink, ...]
    _seq: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def emit(
        self,
        event_type: AgentEventType,
        *,
        data: dict[str, Any] | None = None,
        level: AgentEventLevel = "info",
        turn_id: str | None = None,
        parent_id: str | None = None,
    ) -> AgentEvent:
        with self._lock:
            self._seq += 1
            event = make_agent_event(
                run_id=self.run_id,
                seq=self._seq,
                event_type=event_type,
                data=data,
                level=level,
                turn_id=turn_id,
                parent_id=parent_id,
            )
            for sink in self.sinks:
                sink.emit(event)
            return event

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


def make_agent_event(
    *,
    run_id: str,
    seq: int,
    event_type: AgentEventType,
    data: dict[str, Any] | None = None,
    level: AgentEventLevel = "info",
    turn_id: str | None = None,
    parent_id: str | None = None,
) -> AgentEvent:
    return AgentEvent(
        schema_version=EVENT_SCHEMA_VERSION,
        event_id=f"evt_{uuid.uuid4().hex}",
        seq=seq,
        run_id=run_id,
        turn_id=turn_id,
        parent_id=parent_id,
        timestamp=utc_timestamp(),
        type=event_type,
        level=level,
        data=dict(data or {}),
    )
