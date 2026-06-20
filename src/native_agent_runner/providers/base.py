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
    """One parsed model response — what a :class:`ModelAdapter` returns per turn.

    Either ``tool_calls`` (the model wants tools run; the engine executes them and calls
    back with observations) or ``final_text`` (the turn settles) should be set — returning
    neither fails the turn. ``response_id`` is the provider handle the engine may pass back
    as ``ModelRequest.previous_turn_handle``; ``usage`` carries token counts; ``raw`` keeps
    the unparsed provider payload for debugging.
    """

    response_id: str | None = None
    final_text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRequest:
    """One turn's input handed to :meth:`ModelAdapter.next_turn`.

    The engine builds this each step from the current instruction, system prompt, visible
    tools, and any pending tool observations. See the field comments below for the three
    wire shapes selected by ``instruction`` + ``previous_turn_handle``, and how the
    vendor-neutral ``messages`` log (by-value) overrides the by-reference handle path.
    """

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
    """The LLM seam: turn a :class:`ModelRequest` into a :class:`ModelTurn`.

    Implement this to target any backend — your own gateway, a provider SDK, or a test
    double. The single required method is :meth:`next_turn`; it must return a ``ModelTurn``
    with either ``tool_calls`` or ``final_text``. Keep provider credentials inside the
    adapter (the core never sees them). See ``examples/custom_model_adapter.py`` for a
    minimal implementation, and ``GatewayModelAdapter`` / ``FakeModelAdapter`` for shipped
    ones.

    Async: the engine runs an async core. A sync ``next_turn`` is offloaded to a thread
    automatically, so existing sync adapters keep working. To be awaited natively (no
    thread), an adapter may instead define ``async def anext_turn(request) -> ModelTurn``;
    the engine prefers it when present. A coroutine ``next_turn`` is also awaited directly.
    """

    # Optional capability flag. The loop reads it via
    # ``getattr(adapter, "supports_multimodal", False)``; an adapter that can
    # accept non-text content parts sets it True. Defaulting off keeps existing
    # adapters valid without declaring it. Multimodal forwarding itself is not
    # yet implemented (see core/content.py) — this is the negotiation seam.
    supports_multimodal: bool = False

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        ...

