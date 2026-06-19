from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from native_agent_runner.core.spec import ModelConfig
from native_agent_runner.tools.base import ToolSpec


def format_async_result_text(output: dict[str, Any]) -> str:
    """Render a background/hosted (``is_background``) observation as user-message text.
    The injector may pre-format a ``message``; otherwise a generic async-result preamble
    is used (covers shell background jobs). Shared by the loop's by-value message log and
    the OpenAI adapter's by-reference fallback so the wording stays identical."""
    message = output.get("message") if isinstance(output, dict) else None
    if message:
        return str(message)
    return (
        "An asynchronous task completed. Treat this as the result of the previously "
        f"started task:\n{json.dumps(output, ensure_ascii=False)}"
    )


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolObservation:
    call_id: str
    tool_name: str
    output: dict[str, Any]
    is_background: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "output": self.output,
            "is_background": self.is_background,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ToolObservation:
        return cls(
            call_id=str(payload.get("call_id") or ""),
            tool_name=str(payload.get("tool_name") or ""),
            output=dict(payload.get("output") or {}),
            is_background=bool(payload.get("is_background", False)),
        )


@dataclass(frozen=True)
class ModelTurn:
    response_id: str | None = None
    final_text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRequest:
    # The new user message for this turn, or None when the turn only carries tool
    # observations. Combined with ``previous_turn_handle`` this selects one of three
    # wire shapes:
    #   - no handle, instruction set        -> first turn
    #   - handle set, instruction None       -> tool continuation (observations only)
    #   - handle set, instruction set        -> user follow-up (the third shape)
    instruction: str | None
    system_prompt: str
    tools: tuple[ToolSpec, ...]
    previous_turn_handle: str | None = None
    observations: tuple[ToolObservation, ...] = ()
    model: ModelConfig | None = None
    # By-value conversation (vendor-independent): the full provider-neutral message log
    # the core owns and resends each turn. When set, an adapter sends these as the whole
    # conversation and ignores ``previous_turn_handle``; when ``None`` it falls back to the
    # by-reference handle + ``instruction``/``observations`` delta. ``system_prompt`` is
    # NOT part of ``messages`` — it is regenerated each turn and applied separately.
    messages: tuple[dict[str, Any], ...] | None = None


class ModelAdapter(Protocol):
    # Optional capability flag. The loop reads it via
    # ``getattr(adapter, "supports_multimodal", False)``; an adapter that can
    # accept non-text content parts sets it True. Defaulting off keeps existing
    # adapters valid without declaring it. Multimodal forwarding itself is not
    # yet implemented (see core/content.py) — this is the negotiation seam.
    supports_multimodal: bool = False

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        ...

