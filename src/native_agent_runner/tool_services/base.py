from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CallContext:
    """The in-flight tool call a service is acting for.

    Passed explicitly to service methods so services hold no mutable
    "current call" state of their own.
    """

    tool_call_id: str
    turn_id: str | None
    tool_event_id: str | None
