from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from monoid_agent_kernel.core.tool_surface import ToolAuthorization, ToolScope


@dataclass(frozen=True)
class CallContext:
    """The in-flight tool call a service is acting for.

    Passed explicitly to service methods so services hold no mutable
    "current call" state of their own.
    """

    tool_call_id: str
    turn_id: str | None
    tool_event_id: str | None
    binding_id: str = ""
    tool_id: str = ""
    model_name: str = ""
    authorization: ToolAuthorization | None = None
    scope: ToolScope = ToolScope()
    runtime: dict[str, Any] | None = None
