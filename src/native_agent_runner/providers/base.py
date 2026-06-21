from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from native_agent_runner.core.spec import ModelConfig
from native_agent_runner.errors import ModelAdapterError
from native_agent_runner.providers._common import normalize_usage
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

    Streaming: to feed ``AgentLoop.astream`` token-by-token, an adapter may define
    ``async def astream_turn(request) -> AsyncIterator[ModelStreamChunk]`` yielding
    :class:`TextDelta` / :class:`ToolCallDelta` / :class:`TurnComplete` chunks. The engine
    prefers it only while a stream is active and folds the chunks back into a ``ModelTurn``
    (see :func:`assemble_streamed_turn`) so a streamed turn produces the same orchestration
    events and checkpoints as a non-streamed one. When absent, ``astream`` falls back to the
    one-shot path above and simply emits no token deltas.
    """

    # Optional capability flag. The loop reads it via
    # ``getattr(adapter, "supports_multimodal", False)``; an adapter that can
    # accept non-text content parts sets it True. Defaulting off keeps existing
    # adapters valid without declaring it. When True, the loop resolves by-reference
    # media in the by-value ``messages`` log to wire blocks before the call.
    supports_multimodal: bool = False
    # The wire encoding a multimodal adapter expects for resolved media. The loop reads
    # it via ``getattr(adapter, "wire_image_encoding", "base64")``. Only ``"base64"``
    # is implemented today; ``"url"`` / ``"file_id"`` are reserved for later phases.
    wire_image_encoding: str = "base64"

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        ...


# --- Streaming chunks ------------------------------------------------------------------
# The vendor-neutral units an ``astream_turn`` adapter yields. Designed to losslessly carry
# both Anthropic (content-block deltas + ``input_json_delta``) and OpenAI (Chat/Responses
# ``arguments`` fragments) streams: tool-call arguments arrive as raw string fragments that
# are NOT individually valid JSON and must be concatenated per ``index`` and parsed once at
# the end — which :func:`assemble_streamed_turn` does. Real provider→chunk mapping is P4b;
# P4a exercises these via ``FakeStreamingModelAdapter``.


@dataclass(frozen=True)
class TextDelta:
    """A fragment of assistant output text."""

    text: str

    def to_json(self) -> dict[str, Any]:
        return {"type": "text_delta", "text": self.text}


@dataclass(frozen=True)
class ToolCallDelta:
    """A fragment of one tool call, keyed by ``index`` (its slot in the response). ``id`` and
    ``name`` typically arrive once (first fragment); ``arguments_fragment`` is a raw,
    individually-invalid JSON string piece to be concatenated, not parsed, on arrival."""

    index: int
    arguments_fragment: str = ""
    id: str | None = None
    name: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "tool_call_delta",
            "index": self.index,
            "arguments_fragment": self.arguments_fragment,
            "id": self.id,
            "name": self.name,
        }


@dataclass(frozen=True)
class TurnComplete:
    """Terminal chunk carrying the provider handle and final usage for the turn."""

    response_id: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"type": "turn_complete", "response_id": self.response_id, "usage": dict(self.usage)}


ModelStreamChunk = TextDelta | ToolCallDelta | TurnComplete


def assemble_streamed_turn(chunks: list[ModelStreamChunk]) -> ModelTurn:
    """Fold a streamed chunk sequence into the same :class:`ModelTurn` a one-shot turn
    would produce: concatenate text; group tool-call argument fragments by ``index`` and
    ``json.loads`` each once at the end; take ``response_id``/``usage`` from ``TurnComplete``.
    """
    text_parts: list[str] = []
    slots: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    response_id: str | None = None
    usage: dict[str, int] = {}
    for chunk in chunks:
        if isinstance(chunk, TextDelta):
            text_parts.append(chunk.text)
        elif isinstance(chunk, ToolCallDelta):
            slot = slots.get(chunk.index)
            if slot is None:
                slot = {"id": None, "name": None, "args": ""}
                slots[chunk.index] = slot
                order.append(chunk.index)
            if chunk.id is not None:
                slot["id"] = chunk.id
            if chunk.name is not None:
                slot["name"] = chunk.name
            slot["args"] += chunk.arguments_fragment
        elif isinstance(chunk, TurnComplete):
            if chunk.response_id is not None:
                response_id = chunk.response_id
            if chunk.usage:
                usage = chunk.usage
    tool_calls: list[ToolCall] = []
    for index in order:
        slot = slots[index]
        raw = str(slot["args"]).strip()
        try:
            arguments = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise ModelAdapterError(
                f"invalid streamed tool-call arguments for {slot['name']}",
                provider_error_code="stream_bad_tool_args",
            ) from exc
        if not isinstance(arguments, dict):
            raise ModelAdapterError(
                f"streamed tool-call arguments for {slot['name']} are not an object",
                provider_error_code="stream_bad_tool_args",
            )
        tool_calls.append(ToolCall(id=str(slot["id"] or ""), name=str(slot["name"] or ""), arguments=arguments))
    return ModelTurn(
        response_id=response_id,
        final_text="".join(text_parts) if text_parts else None,
        tool_calls=tuple(tool_calls),
        usage=normalize_usage(usage) if usage else {},
    )

