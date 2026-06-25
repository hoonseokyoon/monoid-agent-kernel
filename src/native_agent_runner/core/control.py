"""Transport-independent control protocol (``native-agent-runner.control-command.v1``).

An external control plane (an Agent Daemon) drives a running session through ONE envelope
+ ONE seam (``ControlDispatcher.dispatch``) instead of importing engine internals or wiring
a route per operation. The envelope is plain data (``(type, run_id, args dict)``) so it rides
in-process calls, HTTP, IPC, or a queue identically — the same discipline as ``TaskReporter``.

``ControlDispatcher`` is the *contract*; ``RunnerBackend.dispatch`` (in ``reference.backend``)
is the reference implementation, which routes each command to the in-process method it already
exposes. The contract types live here and are re-exported from ``contracts``; the reference impl
stays out of the supported surface.

The v1 envelope declares all command types up front (a stable wire contract). A command whose
backing capability has not yet landed returns ``status="not_implemented"``; a command the backend
genuinely cannot satisfy for this run (e.g. ``inspect`` on a run with no live loop) returns
``unsupported``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

CONTROL_PROTOCOL_VERSION = "native-agent-runner.control-command.v1"

#: The command vocabulary. Lifecycle ops (pause/resume/cancel/interrupt/inspect/health) plus the
#: pre-existing session ops (message/config/task/status) unified under one envelope.
ControlCommandType = Literal[
    "pause",
    "resume",
    "cancel",
    "interrupt",
    "inspect",
    "health",
    "send_message",
    "runtime_config",
    "replace_runtime_config",
    "create_task",
    "report_task_result",
    "status",
    "revoke_capability",
]

ControlResultStatus = Literal["ok", "not_implemented", "unsupported", "error"]


@dataclass(frozen=True)
class ControlCommand:
    """One control command. ``args`` carries operation-specific parameters as plain JSON
    (kept dict-only so nothing engine-specific crosses the boundary)."""

    type: ControlCommandType
    run_id: str
    args: dict[str, Any] = field(default_factory=dict)
    issuer: str = ""
    reason: str = ""
    command_id: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": CONTROL_PROTOCOL_VERSION,
            "type": self.type,
            "run_id": self.run_id,
            "args": dict(self.args),
            "issuer": self.issuer,
            "reason": self.reason,
            "command_id": self.command_id,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ControlCommand:
        return cls(
            type=str(payload["type"]),  # type: ignore[arg-type]
            run_id=str(payload.get("run_id") or ""),
            args=dict(payload.get("args") or {}),
            issuer=str(payload.get("issuer") or ""),
            reason=str(payload.get("reason") or ""),
            command_id=str(payload.get("command_id") or ""),
        )


@dataclass(frozen=True)
class ControlResult:
    """The outcome of dispatching a :class:`ControlCommand`. ``data`` carries the wrapped
    operation's own result dict; ``state`` is the resulting session state when known."""

    run_id: str
    type: ControlCommandType
    status: ControlResultStatus
    state: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    error_code: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": CONTROL_PROTOCOL_VERSION,
            "run_id": self.run_id,
            "type": self.type,
            "status": self.status,
            "state": self.state,
            "data": dict(self.data),
            "error": self.error,
            "error_code": self.error_code,
        }


@runtime_checkable
class ControlDispatcher(Protocol):
    """The single seam a control plane depends on: hand it a command, get a result back.
    Transport-agnostic — an HTTP handler, an IPC frame, or an in-process caller all build a
    :class:`ControlCommand` and call ``dispatch``."""

    def dispatch(self, command: ControlCommand) -> ControlResult: ...
