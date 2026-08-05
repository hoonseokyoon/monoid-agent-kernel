from __future__ import annotations

import json
import math
from copy import copy
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from monoid_agent_kernel.core.spec import (
    ModelConfig,
    validate_generation_config,
    validate_reasoning_config,
)
from monoid_agent_kernel.core.json_ingress import (
    loads_model_json_ingress,
    normalize_json_ingress,
    normalize_unicode_scalars,
)
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.providers._common import normalize_usage
from monoid_agent_kernel.tools.base import ToolSpec, normalize_tool_spec

# Why a model turn ended, promoted from the raw provider payload onto the typed turn surface.
# ``stop`` = normal completion; ``length`` = truncated (hit max tokens); ``refusal`` = the model
# declined; ``tool_calls`` = the model wants tools run. ``None`` = the adapter did not report one
# (back-compat / a test double). The loop branches on this before validating a final response.
StopReason = Literal["stop", "length", "refusal", "tool_calls"]


def format_async_result_text(output: dict[str, Any]) -> str:
    """Render a background/hosted (``is_background``) observation as user-message text.
    The injector may pre-format a ``message``; otherwise a generic async-result preamble
    is used (covers shell background jobs). One renderer, so every route that turns a hosted
    result into a user message words it identically -- today that is the loop's by-value
    message log (``loop.py``), the only remaining caller: the OpenAI adapter's by-reference
    fallback was deleted with the shape itself, which that adapter now refuses outright."""
    message = output.get("message") if isinstance(output, dict) else None
    if message:
        return str(message)
    return (
        "An asynchronous task completed. Treat this as the result of the previously "
        f"started task:\n{json.dumps(output, ensure_ascii=False, allow_nan=False)}"
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
    # Non-text media the tool returned, by reference (``content_part_to_json`` dicts).
    # Round-tripped through the checkpoint so a resumed run still forwards the media.
    media: tuple[dict[str, Any], ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "output": self.output,
            "is_background": self.is_background,
            "media": [dict(part) for part in self.media],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ToolObservation:
        # Lenient read: accept the pre-rename ``images`` key so a checkpoint written before
        # the media rename still restores its tool-returned media.
        if not isinstance(payload, dict):
            raise ValueError("tool observation must be an object")
        media = payload.get("media")
        if media is None:
            media = payload.get("images")
        call_id = payload.get("call_id")
        tool_name = payload.get("tool_name")
        output = payload.get("output")
        if type(call_id) is not str or not call_id:
            raise ValueError("tool observation call_id must be a non-empty string")
        if type(tool_name) is not str or not tool_name:
            raise ValueError("tool observation tool_name must be a non-empty string")
        if not isinstance(output, dict):
            raise ValueError("tool observation output must be an object")
        is_background = payload.get("is_background", False)
        if type(is_background) is not bool:
            raise ValueError("tool observation is_background must be a boolean")
        if media is None:
            media = ()
        if not isinstance(media, (list, tuple)) or not all(
            isinstance(part, dict) for part in media
        ):
            raise ValueError("tool observation media must be an array of objects")
        return cls(
            call_id=call_id,
            tool_name=tool_name,
            output=dict(output),
            is_background=is_background,
            media=tuple(dict(part) for part in media),
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
    # Provider-native reasoning artifacts (e.g. OpenAI ``reasoning``/``function_call``/``message``
    # output items), captured verbatim and in their original order so the engine can round-trip
    # them on the next by-value turn. Uninterpreted by the core — never displayed, never
    # reconstructed — but not opaque: only the reasoning-type entries are provider-encrypted,
    # while a paired ``message``/``function_call`` entry duplicates the plaintext of
    # ``final_text``/``tool_calls``. Anything that logs or truncates this must treat it as model
    # content. An adapter that has no reasoning leaves this empty (the neutral seam). See the
    # OpenAI adapter and the loop's assistant-message append for capture + re-injection.
    reasoning: tuple[dict[str, Any], ...] = ()
    # Why the turn ended (promoted from ``raw``). ``None`` when the adapter does not report one.
    stop_reason: StopReason | None = None
    # Whether the adapter retried internally before producing this turn. The kernel counts one
    # adapter call per turn no matter how many attempts happened inside it, so without this an
    # audit record shows a call that failed twice and succeeded on the third try as a clean single
    # attempt. Adapters with no retry loop leave it False, which is exactly true of them.
    provider_retried: bool = False


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
    # ResponseContract delivery (W5): a standard JSON Schema the final answer should satisfy,
    # provider-neutral data. An adapter that declares ``structured_output_support = "native"``
    # translates it into its provider's constrained-decoding dialect **verbatim, never
    # transformed** — the request digest identifies exactly what was asked for. Adapters
    # without the declaration ignore it, and post-hoc validation remains the guarantee either
    # way (native delivery only reduces repairs). ``None`` = unconstrained.
    output_schema: dict[str, Any] | None = None


def _declared_support(
    adapter: Any, attribute: str, config: ModelConfig | None = None
) -> Literal["native", "none"]:
    """The fail-closed probe shared by every opt-in adapter capability declaration.

    One rule for all of them: only the exact string ``"native"`` claims the capability.
    Absence, ``None``, ``True``, and unknown future spellings all read as ``"none"``, so a
    consumer can never over-trust an adapter that did not explicitly claim it — and the two
    capabilities below cannot drift into different notions of "declared".

    Read off the *instance*. An adapter whose answer is fixed declares with a ``ClassVar``;
    one whose answer depends on configuration declares with a **callable** taking the
    effective per-call config (``request.model or self.config`` — the config the adapter will
    actually enforce with). ``config`` is passed through to it, because the claim and the
    enforcement must read the same policy: a forwarding transport probed on its *standing*
    config alone would mint proof for a call it enforces under a different per-call policy.
    An attribute that raises — on read or on call — is not a claim: the failure reads as
    ``"none"`` like any other non-declaration, because a probe that can take the call down is
    a worse contract than one that under-claims.
    """

    try:
        value = getattr(adapter, attribute, "none")
        if callable(value):
            value = value(config)
    except Exception:
        return "none"
    return "native" if value == "native" else "none"


def structured_output_support(
    adapter: Any, config: ModelConfig | None = None
) -> Literal["native", "none"]:
    """Whether ``adapter`` translates :attr:`ModelRequest.output_schema` into provider-native
    constrained decoding.

    Opt-in declaration, like ``supports_multimodal``: adapters set a
    ``structured_output_support`` class attribute, or define it as a method taking the
    effective per-call :class:`ModelConfig` when the answer depends on policy. Pass ``config``
    when probing on behalf of a specific call; ``None`` probes the adapter's standing
    configuration.
    """

    return _declared_support(adapter, "structured_output_support", config)


def generation_support(
    adapter: Any, config: ModelConfig | None = None
) -> Literal["native", "none"]:
    """Whether ``adapter`` applies :attr:`ModelConfig.generation` to the provider request.

    The twin of :func:`structured_output_support`, and for the same reason: a transport that
    *forwards* generation parameters to an adapter cannot know whether that adapter puts them
    on the wire. Only an adapter that declares this may be used to justify an
    applied-parameters proof; anything else must be reported as unproven so a fail-closed
    client refuses the turn rather than trusting parameters nobody applied.
    """

    return _declared_support(adapter, "generation_support", config)


def reasoning_support(
    adapter: Any, config: ModelConfig | None = None
) -> Literal["native", "none"]:
    """Whether ``adapter`` applies :attr:`ModelConfig.reasoning` to the provider request.

    The third member of the capability family above, same fail-closed rule, one difference
    worth naming: a conditional declaration answers off ``reasoning.on_unsupported`` — its own
    feature family's policy knob — where the generation/schema pair deliberately shares
    ``generation.on_unsupported``. A reasoning claim read off another family's knob would mint
    proof for a call whose own policy said best-effort.
    """

    return _declared_support(adapter, "reasoning_support", config)


class ModelAdapter(Protocol):
    """The LLM seam: turn a :class:`ModelRequest` into a :class:`ModelTurn`.

    Implement this to target any backend — your own gateway, a provider SDK, or a test
    double. The single required method is :meth:`next_turn`; it must return a ``ModelTurn``
    with either ``tool_calls`` or ``final_text``. Keep provider credentials inside the
    adapter (the core never sees them). See ``examples/custom_model_adapter.py`` for a
    minimal implementation, and ``GatewayModelAdapter`` / ``FakeModelAdapter`` for shipped
    ones.

    Async: the engine runs an async core. A sync ``next_turn`` is offloaded to a thread
    automatically, so existing sync adapters keep working. Native async-only adapters implement
    :class:`AsyncModelAdapter`; the engine prefers ``anext_turn`` when an adapter exposes both.
    A coroutine ``next_turn`` is also awaited directly for compatibility.

    Streaming: to feed ``AgentLoop.astream`` token-by-token, an adapter implements
    :class:`StreamingModelAdapter` and yields
    :class:`TextDelta` / :class:`ToolCallDelta` / :class:`TurnComplete` chunks. The engine
    prefers it only while a stream is active and folds the chunks back into a ``ModelTurn``
    (see :func:`assemble_streamed_turn`) so a streamed turn produces the same orchestration
    events and checkpoints as a non-streamed one. When absent, ``astream`` falls back to the
    one-shot path above and simply emits no token deltas.

    Optional capabilities: an adapter may additionally expose ``supports_multimodal`` /
    ``wire_image_encoding`` (see :class:`MultimodalModelAdapter`) and ``provider_name``
    (see :class:`ProviderNamedModelAdapter`). The engine reads each with ``getattr`` and a
    neutral default, so they are deliberately NOT members of this protocol — declaring them
    here would make them required for structural typing and reject an otherwise valid
    third-party adapter that omits them.
    """

    def next_turn(self, request: ModelRequest) -> ModelTurn: ...


class AsyncModelAdapter(Protocol):
    """Native async one-shot model adapter contract.

    An adapter may implement this contract without a synchronous ``next_turn`` method. The
    engine awaits ``anext_turn`` directly and preserves the same retry, event, and checkpoint
    behavior as the synchronous adapter path. The optional capabilities described on
    :class:`ModelAdapter` apply here too, and are likewise not members of this protocol.
    """

    async def anext_turn(self, request: ModelRequest) -> ModelTurn: ...


# --- Optional adapter capabilities -----------------------------------------------------
# Opt-in extensions, each declaring one capability the engine probes with ``getattr`` and a
# default. Implementing them is never required: a bare ``ModelAdapter`` stays valid, and the
# engine's behavior is identical whether an adapter declares the attribute or omits it. They
# exist so the attribute names and meanings are part of the checked contract rather than a
# convention, and so typed callers can narrow to "an adapter that reports this".
#
# Each member that is a *value* is declared as a read-only property, not an annotated attribute.
# That is what makes the shipped adapters — which use ``ClassVar`` — satisfy these protocols: a
# protocol member annotated ``name: str`` demands an *instance* variable and rejects a
# ``ClassVar``, while a read-only property is satisfied by a ``ClassVar``, an instance attribute,
# and a property alike. A member that answers a *question* is a method instead, because it takes
# an argument a property cannot carry — ``AddressedModelAdapter.resolve_destination`` is the one
# such member today.


class MultimodalModelAdapter(Protocol):
    """An adapter that accepts non-text content parts.

    ``supports_multimodal`` True makes the loop resolve by-reference media in the by-value
    ``messages`` log to wire blocks before the call; the loop reads it via
    ``getattr(adapter, "supports_multimodal", False)``, so omitting it means "text only".

    A multimodal adapter may also expose ``wire_image_encoding`` to name the encoding it
    expects for resolved media, read via ``getattr(adapter, "wire_image_encoding", "base64")``.
    Only ``"base64"`` is implemented today; ``"url"`` / ``"file_id"`` are reserved for later
    phases. It is deliberately not a member of this protocol: it is a parameter of the
    capability rather than the capability itself, every shipped multimodal adapter relies on
    its default, and adding it here would reject them all.
    """

    @property
    def supports_multimodal(self) -> bool: ...


class ProviderNamedModelAdapter(Protocol):
    """An adapter that identifies whose provider-native reasoning items it produces.

    The loop reads ``provider_name`` via ``getattr(adapter, "provider_name", None)`` and tags
    captured :attr:`ModelTurn.reasoning` with provider+model, so items only round-trip back to
    a matching adapter and model. Omitting it means "do not tag": reasoning is not replayed,
    which is the correct neutral behavior for an adapter with no provider-native reasoning
    artifacts.

    ``None`` carries that same sense from a *declared* member: ``str | None`` because a
    forwarding adapter's upstream is a per-deployment setting, and a deployment fronting an
    upstream with no reasoning artifacts has to be able to say so without dropping the attribute
    (``GatewayModelAdapter.provider_name`` is exactly that field). Every reader already spells
    the two the same way -- ``getattr(..., None)`` then a falsy check -- so declaring ``str``
    only made the shipped adapter fail a type it satisfies behaviorally.
    """

    @property
    def provider_name(self) -> str | None: ...


def resolved_provider_name(adapter: Any, config: ModelConfig | None) -> str | None:
    """The provider that ACTUALLY serves a call: what the adapter declares, else the config's.

    One expression, because the answer is written to three surfaces that a reader compares --
    the model-stream context's ``provider``, ``run.started``'s ``model_provider`` (and through it
    every ``gen_ai.provider.name`` the event-driven OTel sink writes), and the receipt-derived
    span's own fallback. They disagreed for one release: through a gateway the receipt named the
    upstream and the event named the transport, for the same call.

    A *forwarding* adapter declares its upstream, so preferring the declaration is what makes
    these spans describe the model that answered rather than the hop the answer arrived over.
    ``ModelConfig.provider`` remains the fallback and stays recorded verbatim on the run manifest,
    so the transport is never lost.

    Tolerant by construction: this feeds telemetry, and a third-party ``provider_name`` property
    that raises -- or whose ``str()`` does -- must not take a run down over an attribute nothing
    branches on. Mirrors the defensive probe in ``ModelCallRunner``, including its
    ``normalize_unicode_scalars``, so the receipt's own read of a declaration and this one are
    byte-identical rather than merely equal on well-behaved strings.

    Tolerance means "keep going", not "answer nothing": the declaration guard FALLS THROUGH to
    the config fallback rather than returning. It used to return ``None`` there, which made the
    one documented expression give two answers on exactly the path it exists for -- the
    model-stream context reported no provider at all while ``run.started``, the receipt and the
    OTel span beside them all reported the configured transport for the same call.
    """

    try:
        declared = getattr(adapter, "provider_name", None)
        if declared:
            return normalize_unicode_scalars(str(declared))
    except Exception:
        pass  # unreadable declaration: fall through to the config, do not answer nothing
    try:
        fallback = config.provider if config is not None else None
        return normalize_unicode_scalars(str(fallback)) if fallback else None
    except Exception:
        return None


class ConfiguredModelAdapter(Protocol):
    """An adapter that carries its own fallback :class:`ModelConfig`.

    ``ModelRequest.model`` is optional, and the shipped adapters fall back to ``self.config`` when
    it is absent — so the config the provider actually ran under is not always visible in the
    request. A caller recording what a call *was* reads ``config`` via
    ``getattr(adapter, "config", None)`` to resolve it; omitting it means "the request carries the
    whole story", which is correct for an adapter with no configuration of its own.

    Declared for the same reason as :class:`ProviderNamedModelAdapter`: the attribute was already
    being read, and a probed attribute that no protocol names is a contract nobody can check.
    """

    @property
    def config(self) -> ModelConfig: ...


class AddressedModelAdapter(Protocol):
    """An adapter that can say where a call will actually be sent.

    An adapter may route by more than its :class:`ModelConfig` -- a per-instance override, an
    environment variable, a tenant-specific host -- so the config alone does not identify the
    service that answered. A caller recording a call's identity asks for the resolved destination
    and folds it into that identity; omitting the member means "the config is the whole story",
    which is correct for an adapter that routes on config alone.

    The value is hashed, never recorded, so an internal hostname stays internal. Raising is
    permitted and treated as "unknown".
    """

    def resolve_destination(self, config: ModelConfig) -> str: ...


class StreamingModelAdapter(Protocol):
    """Optional token-streaming extension for a sync or async model adapter."""

    def astream_turn(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]: ...


# --- Streaming chunks ------------------------------------------------------------------
# The vendor-neutral units an ``astream_turn`` adapter yields. Designed to losslessly carry
# both Anthropic (content-block deltas + ``input_json_delta``) and OpenAI (Chat/Responses
# ``arguments`` fragments) streams: tool-call arguments arrive as raw string fragments that
# are NOT individually valid JSON and must be concatenated per ``index`` and parsed once at
# the end — which :func:`assemble_streamed_turn` does. Real provider→chunk mapping is P4b;
# P4a exercises these via ``FakeStreamingModelAdapter``.


# Every chunk type carries ``provider_retried``, not only ``TurnComplete``, because the terminal
# chunk is not guaranteed to arrive. Stream retries are pre-commit, so an adapter knows it retried
# the moment the stream commits -- but a run cancelled or aborted mid-stream ends without a terminal
# chunk, and evidence that rides only that one is evidence a cancelled call can never report. The
# failure receipt then denied a retry that demonstrably happened.
#
# The flag says the *stream* was retried, not that this particular fragment was; a fragment is
# simply the earliest place the fact can be put where a consumer will see it.


@dataclass(frozen=True)
class TextDelta:
    """A fragment of assistant output text."""

    text: str
    provider_retried: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "text_delta",
            "text": self.text,
            "provider_retried": self.provider_retried,
        }


@dataclass(frozen=True)
class ReasoningDelta:
    """A fragment of the model's reasoning *summary* text (display-only). Distinct from
    :class:`TextDelta` (the answer) so a consumer can render it in a separate "thinking" view.
    Purely presentational — :func:`assemble_streamed_turn` ignores it (the round-trippable
    reasoning artifacts ride :attr:`TurnComplete.reasoning`, not these deltas)."""

    text: str
    provider_retried: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "reasoning_delta",
            "text": self.text,
            "provider_retried": self.provider_retried,
        }


@dataclass(frozen=True)
class ToolCallDelta:
    """A fragment of one tool call, keyed by ``index`` (its slot in the response). ``id`` and
    ``name`` typically arrive once (first fragment); ``arguments_fragment`` is a raw,
    individually-invalid JSON string piece to be concatenated, not parsed, on arrival."""

    index: int
    arguments_fragment: str = ""
    id: str | None = None
    name: str | None = None
    provider_retried: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "tool_call_delta",
            "index": self.index,
            "arguments_fragment": self.arguments_fragment,
            "id": self.id,
            "name": self.name,
            "provider_retried": self.provider_retried,
        }


@dataclass(frozen=True)
class TurnComplete:
    """Terminal chunk carrying the provider handle, final usage, and any reasoning artifacts
    for the turn. ``reasoning`` mirrors :attr:`ModelTurn.reasoning`: the streaming path can only
    read provider reasoning items (with their ``encrypted_content``) off the final response
    object, so they ride this terminal chunk rather than the per-token deltas."""

    response_id: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    reasoning: tuple[dict[str, Any], ...] = ()
    stop_reason: StopReason | None = None
    # Mirrors :attr:`ModelTurn.provider_retried`. On the streaming path the turn is assembled by the
    # caller out of chunks, so an adapter that retried before committing its stream has no other
    # place to say so.
    provider_retried: bool = False
    # The gateway transport's applied-parameters echo (scope §5 D-a), riding the terminal frame
    # because the streaming caller has no response object to read it from. ``None`` = the wire
    # never mentioned it (an older gateway, or a transport with no echo).
    generation_applied: dict[str, Any] | None = None
    # The schema twin of ``generation_applied`` (W5 PR 4): whether the gateway forwarded
    # ``output_schema`` to an upstream that natively enforces it. A sibling key, not a member
    # of the generation echo -- changing an existing key's shape is how old clients break.
    schema_applied: bool | None = None
    # The reasoning member of the echo family (v0.21 B1): the forwarded reasoning block, in
    # the generation echo's shape because reasoning has values a client can compare. ``{}`` is
    # a real proof (``effort="default"`` forwards an empty block), so only ``None`` means the
    # wire never mentioned it.
    reasoning_applied: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        payload = {
            "type": "turn_complete",
            "response_id": self.response_id,
            "usage": dict(self.usage),
            "reasoning": [dict(item) for item in self.reasoning],
            "stop_reason": self.stop_reason,
            "provider_retried": self.provider_retried,
        }
        if self.generation_applied is not None:
            payload["generation_applied"] = dict(self.generation_applied)
        if self.schema_applied is not None:
            payload["schema_applied"] = self.schema_applied
        if self.reasoning_applied is not None:
            payload["reasoning_applied"] = dict(self.reasoning_applied)
        return payload


ModelStreamChunk = TextDelta | ReasoningDelta | ToolCallDelta | TurnComplete


def _normalize_required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return normalize_unicode_scalars(value)


def _normalize_retry_codes(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("model.retry.retry_on must be an array of non-empty strings")
    normalized: list[str] = []
    for code in value:
        text = _normalize_required_text(code, "model.retry.retry_on item")
        if not text:
            raise ValueError("model.retry.retry_on entries must be non-empty strings")
        normalized.append(text)
    return tuple(normalized)


def _normalize_optional_text(value: Any, field_name: str) -> str | None:
    normalized = normalize_json_ingress(value)
    if normalized is None:
        return None
    if not isinstance(normalized, str):
        raise ValueError(f"{field_name} must be a string or null")
    return normalized


def _require_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _copy_with_fields(value: Any, /, **changes: Any) -> Any:
    """Copy a dataclass-like extension without calling its public constructor again.

    ``dataclasses.replace`` dispatches through ``type(value).__init__``.  Public extension
    subclasses commonly expose a smaller convenience constructor, so using ``replace`` at
    an ingress boundary turned otherwise valid adapters and tools into ``TypeError``.  A
    shallow copy preserves the extension type and its private state; frozen fields can then
    be installed without mutating the caller's object.
    """

    cloned = copy(value)
    for name, replacement in changes.items():
        object.__setattr__(cloned, name, replacement)
    return cloned


def _control_number(
    value: Any,
    field_name: str,
    *,
    minimum: int,
    inclusive: bool,
) -> int | float:
    if type(value) not in (int, float):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        finite = False
    if not finite:
        raise ValueError(f"{field_name} must be a finite number")
    outside_range = value < minimum if inclusive else value <= minimum
    if outside_range:
        requirement = "non-negative" if inclusive and minimum == 0 else "greater than zero"
        raise ValueError(f"{field_name} must be {requirement}")
    return value


def _positive_control_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be an integer greater than zero")
    return value


def normalize_model_config(config: ModelConfig | None) -> ModelConfig | None:
    """Normalize model metadata and reject non-finite control values.

    Content values can become JSON ``null`` without changing their meaning. Retry delays and
    timeouts have no meaningful ``null`` behavior, so invalid direct-Python configuration is
    rejected before an adapter is invoked.
    """

    if config is None:
        return None
    # validate_reasoning_config enforces the enum, so a passing value is already inside the
    # portable ASCII domain -- same reasoning as the generation call below, same single rule
    # source as the JSON codec. Per-field text normalization here accepted any non-empty
    # string, leaving direct-Python reasoning the one construction route that failed open.
    reasoning = validate_reasoning_config(config.reasoning)
    retry = _copy_with_fields(
        config.retry,
        max_attempts=_positive_control_int(config.retry.max_attempts, "model.retry.max_attempts"),
        initial_delay_s=_control_number(
            config.retry.initial_delay_s,
            "model.retry.initial_delay_s",
            minimum=0,
            inclusive=True,
        ),
        max_delay_s=_control_number(
            config.retry.max_delay_s,
            "model.retry.max_delay_s",
            minimum=0,
            inclusive=True,
        ),
        backoff_multiplier=_control_number(
            config.retry.backoff_multiplier,
            "model.retry.backoff_multiplier",
            minimum=0,
            inclusive=False,
        ),
        jitter_s=_control_number(
            config.retry.jitter_s,
            "model.retry.jitter_s",
            minimum=0,
            inclusive=True,
        ),
        retry_on=_normalize_retry_codes(config.retry.retry_on),
    )
    # validate_generation_config enforces the enum, so a passing on_unsupported is already
    # inside the portable ASCII domain -- no per-field text normalization step is needed.
    generation = validate_generation_config(config.generation)
    return _copy_with_fields(
        config,
        provider=_normalize_required_text(config.provider, "model.provider"),
        model=_normalize_required_text(config.model, "model.model"),
        timeout_s=_control_number(
            config.timeout_s,
            "model.timeout_s",
            minimum=0,
            inclusive=False,
        ),
        gateway_url=_normalize_optional_text(config.gateway_url, "model.gateway_url"),
        reasoning=reasoning,
        retry=retry,
        generation=generation,
    )


def normalize_model_request(request: ModelRequest) -> ModelRequest:
    """Copy a call request into the portable JSON/Unicode domain."""

    observations = tuple(
        _copy_with_fields(
            observation,
            call_id=_normalize_required_text(observation.call_id, "tool observation call_id"),
            tool_name=_normalize_required_text(
                observation.tool_name,
                "tool observation tool_name",
            ),
            output=normalize_json_ingress(observation.output),
            is_background=_require_bool(
                observation.is_background,
                "tool observation is_background",
            ),
            media=tuple(normalize_json_ingress(observation.media)),
        )
        for observation in request.observations
    )
    messages = None
    if request.messages is not None:
        messages = tuple(normalize_json_ingress(request.messages))
    output_schema = request.output_schema
    if output_schema is not None:
        if not isinstance(output_schema, dict):
            raise ValueError("model request output_schema must be an object or null")
        # Strings and containers are normalized; non-finite floats are deliberately NOT
        # substituted. Everything else here is model *content*, where turning a stray ``NaN``
        # into ``null`` loses nothing -- but this is a control document the contract promises
        # to deliver **verbatim**. Substituting rewrote ``{"enum": [NaN]}`` into
        # ``{"enum": [null]}``: a different constraint, silently enforced by the provider, and
        # the strict serializer that exists to refuse the value (``allow_nan=False``, on both
        # adapters) never got to see it. Left in place, the request is refused as the
        # config-recoverable bad request it is.
        output_schema = normalize_json_ingress(output_schema, substitute_nonfinite=False)
    return _copy_with_fields(
        request,
        instruction=_normalize_optional_text(request.instruction, "model request instruction"),
        system_prompt=_normalize_required_text(
            request.system_prompt,
            "model request system_prompt",
        ),
        tools=tuple(normalize_tool_spec(spec) for spec in request.tools),
        previous_turn_handle=_normalize_optional_text(
            request.previous_turn_handle,
            "model request previous_turn_handle",
        ),
        observations=observations,
        model=normalize_model_config(request.model),
        messages=messages,
        output_schema=output_schema,
    )


def _normalize_model_turn(turn: Any) -> Any:
    """Copy one model outcome before receipts, observers, previews, or persistence see it.

    The adapter protocol names :class:`ModelTurn`, while older integrations also return a
    structurally compatible object.  Preserve that supported shape and normalize every
    attribute it exposes.
    """

    if not isinstance(turn, ModelTurn) and not any(
        hasattr(turn, field_name) for field_name in ("final_text", "tool_calls", "stop_reason")
    ):
        raise ValueError("model turn has no outcome fields")

    final_text = _normalize_optional_text(
        getattr(turn, "final_text", None),
        "model turn final_text",
    )
    stop_reason = _normalize_optional_text(
        getattr(turn, "stop_reason", None),
        "model turn stop_reason",
    )
    has_settled_outcome = bool(final_text) or stop_reason in ("refusal", "length")
    tool_calls = getattr(turn, "tool_calls", ())
    if tool_calls is None:
        tool_calls = ()
    if not isinstance(tool_calls, (list, tuple)):
        if has_settled_outcome:
            tool_calls = ()
        else:
            raise ValueError("model turn tool_calls must be an array or null")
    normalized_calls = []
    for call in tool_calls:
        if not all(hasattr(call, field_name) for field_name in ("id", "name", "arguments")):
            if has_settled_outcome:
                # Standalone runners have long preserved an odd extra entry beside a paid final
                # answer so their capture layer can record a bounded repr placeholder. It never
                # becomes an executable call because the settled answer wins in AgentLoop.
                normalized_calls.append(call)
                continue
            raise ValueError("model turn tool call has an invalid shape")
        try:
            arguments = getattr(call, "arguments", {})
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                raise ValueError("model turn tool call arguments must be an object or null")
            normalized_calls.append(
                _copy_with_fields(
                    call,
                    id=_normalize_required_text(getattr(call, "id"), "model tool call id"),
                    name=_normalize_required_text(getattr(call, "name"), "model tool call name"),
                    arguments=normalize_json_ingress(arguments),
                )
            )
        except Exception:
            if not has_settled_outcome:
                raise
    tool_calls = tuple(normalized_calls) if isinstance(tool_calls, tuple) else normalized_calls

    reasoning = getattr(turn, "reasoning", ())
    if not isinstance(reasoning, (list, tuple)):
        # DIVERGENCE, deliberate: the stream twin (``_normalize_model_stream_chunk``, on
        # ``TurnComplete``) RAISES on this same non-sequence, while the turn path coerces to ().
        # It is wire-observable — a custom adapter returning a malformed ``reasoning`` has its
        # turn silently stripped and its stream hard-failed — and it stands because the two
        # paths have different jobs. This one, per the docstring above, must keep accepting the
        # structurally-compatible legacy objects that predate the protocol, where an attribute
        # of an unexpected shape means "this adapter has no reasoning" rather than "this adapter
        # is broken"; there is no reasoning to lose. The stream path is the strict ingress: its
        # chunks are the protocol's own dataclasses, so a bad value there is a real defect, and
        # a silently-emptied terminal frame would drop artifacts the round-trip needs.
        reasoning = ()
    normalized_reasoning = normalize_json_ingress(reasoning)
    reasoning = (
        tuple(normalized_reasoning) if isinstance(reasoning, tuple) else normalized_reasoning
    )

    usage = getattr(turn, "usage", {})
    if usage is None:
        usage = {}
    if not isinstance(usage, dict):
        raise ValueError("model turn usage must be an object")
    usage = normalize_usage(usage)
    raw = getattr(turn, "raw", {})
    if not isinstance(raw, dict):
        raw = {}
    changes = {
        "response_id": _normalize_optional_text(
            getattr(turn, "response_id", None),
            "model turn response_id",
        ),
        "final_text": final_text,
        "tool_calls": tool_calls,
        "usage": usage,
        "raw": normalize_json_ingress(raw),
        "reasoning": reasoning,
        "stop_reason": stop_reason,
        "provider_retried": _require_bool(
            getattr(turn, "provider_retried", False),
            "model turn provider_retried",
        ),
    }
    try:
        return _copy_with_fields(turn, **changes)
    except Exception:
        # Some structural adapters return immutable tuple-like records.  Their extension type
        # cannot be copied safely, so converge on the protocol's concrete value instead of
        # returning un-normalized provider data.
        return ModelTurn(**changes)


def normalize_model_turn(turn: Any) -> Any:
    """Normalize provider output and classify an unusable response as a model failure."""

    try:
        return _normalize_model_turn(turn)
    except ModelAdapterError:
        raise
    except Exception as exc:
        raise ModelAdapterError("model adapter returned a non-portable response") from exc


def _normalize_model_stream_chunk(chunk: ModelStreamChunk) -> ModelStreamChunk:
    """Normalize a provider stream fragment before delivery and assembly."""

    if isinstance(chunk, TextDelta):
        return _copy_with_fields(
            chunk,
            text=_normalize_required_text(chunk.text, "text delta text"),
            provider_retried=_require_bool(
                chunk.provider_retried,
                "text delta provider_retried",
            ),
        )
    if isinstance(chunk, ReasoningDelta):
        return _copy_with_fields(
            chunk,
            text=_normalize_required_text(chunk.text, "reasoning delta text"),
            provider_retried=_require_bool(
                chunk.provider_retried,
                "reasoning delta provider_retried",
            ),
        )
    if isinstance(chunk, ToolCallDelta):
        return _copy_with_fields(
            chunk,
            index=_require_nonnegative_int(chunk.index, "tool call delta index"),
            arguments_fragment=_normalize_required_text(
                chunk.arguments_fragment,
                "tool call delta arguments_fragment",
            ),
            id=_normalize_optional_text(chunk.id, "tool call delta id"),
            name=_normalize_optional_text(chunk.name, "tool call delta name"),
            provider_retried=_require_bool(
                chunk.provider_retried,
                "tool call delta provider_retried",
            ),
        )
    if isinstance(chunk, TurnComplete):
        reasoning = chunk.reasoning
        if reasoning is None:
            reasoning = ()
        if not isinstance(reasoning, (list, tuple)):
            # DIVERGENCE, deliberate: the one-shot twin (``_normalize_model_turn``) coerces this
            # same non-sequence to () instead of raising. Strictness belongs here — a stream
            # chunk is one of this protocol's own dataclasses, not a legacy duck-typed object,
            # so a malformed value is a defect rather than "no reasoning", and emptying it
            # quietly would drop the terminal frame's artifacts on the floor. See the comment at
            # the turn-path site for why tolerance belongs there.
            raise ValueError("turn complete reasoning must be an array or null")
        normalized_reasoning = normalize_json_ingress(reasoning)
        reasoning = (
            tuple(normalized_reasoning) if isinstance(reasoning, tuple) else normalized_reasoning
        )
        usage = chunk.usage
        if usage is None:
            usage = {}
        if not isinstance(usage, dict):
            raise ValueError("turn complete usage must be an object or null")
        return _copy_with_fields(
            chunk,
            response_id=_normalize_optional_text(chunk.response_id, "turn complete response_id"),
            usage=normalize_usage(usage),
            reasoning=reasoning,
            stop_reason=_normalize_optional_text(chunk.stop_reason, "turn complete stop_reason"),
            provider_retried=_require_bool(
                chunk.provider_retried,
                "turn complete provider_retried",
            ),
        )
    raise ValueError(f"unsupported model stream fragment: {type(chunk).__name__}")


def normalize_model_stream_chunk(chunk: ModelStreamChunk) -> ModelStreamChunk:
    """Normalize one provider fragment and classify unusable output as a model failure."""

    try:
        return _normalize_model_stream_chunk(chunk)
    except ModelAdapterError:
        raise
    except Exception as exc:
        raise ModelAdapterError("model adapter returned a non-portable stream fragment") from exc


@dataclass
class _UnicodeScalarChunkBuffer:
    pending_high: str = ""
    provider_retried: bool = False
    sequence: int | None = None

    def feed(self, text: Any, *, provider_retried: bool, sequence: int) -> tuple[Any, bool]:
        if not isinstance(text, str):
            raise ValueError("model stream text fragment must be a string")
        inherited_sequence = self.sequence
        combined = self.pending_high + text
        inherited_retry = self.provider_retried
        self.pending_high = ""
        self.provider_retried = False
        self.sequence = None
        if combined and 0xD800 <= ord(combined[-1]) <= 0xDBFF:
            self.pending_high = combined[-1]
            self.provider_retried = inherited_retry or provider_retried
            self.sequence = (
                inherited_sequence if inherited_sequence is not None and not text else sequence
            )
            combined = combined[:-1]
        return normalize_unicode_scalars(combined), inherited_retry or provider_retried

    def flush(self) -> tuple[str, bool, int] | None:
        if not self.pending_high:
            return None
        retried = self.provider_retried
        sequence = self.sequence if self.sequence is not None else 0
        self.pending_high = ""
        self.provider_retried = False
        self.sequence = None
        return "\ufffd", retried, sequence


class ModelStreamIngressNormalizer:
    """Normalize chunks while preserving surrogate pairs split within one logical channel."""

    def __init__(self) -> None:
        self._text = _UnicodeScalarChunkBuffer()
        self._reasoning = _UnicodeScalarChunkBuffer()
        self._tool_arguments: dict[int, _UnicodeScalarChunkBuffer] = {}
        self._sequence = 0

    def normalize(self, chunk: ModelStreamChunk) -> list[ModelStreamChunk]:
        try:
            return self._normalize(chunk)
        except ModelAdapterError:
            raise
        except Exception as exc:
            raise ModelAdapterError(
                "model adapter returned a non-portable stream fragment"
            ) from exc

    def _normalize(self, chunk: ModelStreamChunk) -> list[ModelStreamChunk]:
        sequence = self._sequence
        self._sequence += 1
        if isinstance(chunk, TextDelta):
            value, retried = self._text.feed(
                chunk.text,
                provider_retried=_require_bool(
                    chunk.provider_retried,
                    "text delta provider_retried",
                ),
                sequence=sequence,
            )
            return [_copy_with_fields(chunk, text=value, provider_retried=retried)]
        if isinstance(chunk, ReasoningDelta):
            value, retried = self._reasoning.feed(
                chunk.text,
                provider_retried=_require_bool(
                    chunk.provider_retried,
                    "reasoning delta provider_retried",
                ),
                sequence=sequence,
            )
            return [_copy_with_fields(chunk, text=value, provider_retried=retried)]
        if isinstance(chunk, ToolCallDelta):
            index = _require_nonnegative_int(chunk.index, "tool call delta index")
            buffer = self._tool_arguments.setdefault(index, _UnicodeScalarChunkBuffer())
            value, retried = buffer.feed(
                chunk.arguments_fragment,
                provider_retried=_require_bool(
                    chunk.provider_retried,
                    "tool call delta provider_retried",
                ),
                sequence=sequence,
            )
            return [
                _copy_with_fields(
                    chunk,
                    index=index,
                    arguments_fragment=value,
                    id=_normalize_optional_text(chunk.id, "tool call delta id"),
                    name=_normalize_optional_text(chunk.name, "tool call delta name"),
                    provider_retried=retried,
                )
            ]
        if not isinstance(chunk, TurnComplete):
            raise ValueError(f"unsupported model stream fragment: {type(chunk).__name__}")
        terminal = normalize_model_stream_chunk(chunk)
        emitted = self.finish()
        emitted.append(terminal)
        return emitted

    def finish(self) -> list[ModelStreamChunk]:
        pending: list[tuple[int, ModelStreamChunk]] = []
        text = self._text.flush()
        if text is not None:
            pending.append((text[2], TextDelta(text[0], text[1])))
        reasoning = self._reasoning.flush()
        if reasoning is not None:
            pending.append((reasoning[2], ReasoningDelta(reasoning[0], reasoning[1])))
        for index, buffer in self._tool_arguments.items():
            arguments = buffer.flush()
            if arguments is not None:
                pending.append(
                    (
                        arguments[2],
                        ToolCallDelta(
                            index=index,
                            arguments_fragment=arguments[0],
                            provider_retried=arguments[1],
                        ),
                    )
                )
        pending.sort(key=lambda item: item[0])
        return [chunk for _sequence, chunk in pending]


def mark_provider_retried(error: BaseException) -> None:
    """Record on an escaping error that the adapter's retry loop had already run.

    Read back by ``ModelCallReceipt.with_error`` through ``getattr``, so an exception that refuses
    the attribute (``__slots__``) simply reports no retry rather than replacing the failure being
    reported with an AttributeError.

    Shared rather than written once per caller: the adapter stamps a failure it raises itself, and
    the runner stamps one raised *around* a stream it had already seen retry. Two copies of a rule
    about which exceptions accept an attribute is two copies that can disagree.
    """

    try:
        error.provider_retried = True  # type: ignore[attr-defined]
    except Exception:
        pass


def mark_provider_usage(error: BaseException, usage: Mapping[str, int] | None) -> None:
    """Record on an escaping error the token usage the provider already reported.

    Some failures happen *after* the provider produced — and billed for — a complete answer.
    The applied-parameters proof refusals are the clearest case: the turn parsed, its usage is
    known, and only then is the turn refused. Without this, the receipt for that call carries
    an empty usage, the loop's post-turn accounting never runs, and a paid call disappears
    from the metrics and from the cumulative token budget — a budget that under-counts is a
    bound that does not hold.

    The guarded-setattr twin of :func:`mark_provider_retried`, for the same reason: an
    exception type that refuses the attribute (``__slots__``) simply carries no usage rather
    than replacing the provider's failure with an ``AttributeError``. Read back by
    ``ModelCallReceipt.with_error`` through ``getattr``.
    """

    if not usage:
        return
    try:
        error.provider_usage = dict(usage)  # type: ignore[attr-defined]
    except Exception:
        pass


def mark_provider_error_code(error: BaseException, code: str) -> None:
    """Name the class of failure on an escaping error that was minted without one.

    A refusal raised by a field validator deep inside a reader knows its key and nothing else, so
    it mints a bare ``ModelAdapterError``; one hop out the reference gateway resolves
    ``exc.provider_error_code or GATEWAY_BAD_RESPONSE`` and blames the HOP's wire for an upstream
    payload defect. Backfill only: a refusal that DOES name a code knows something the caller
    completing it does not, and keeps it.

    The guarded-setattr sibling of :func:`mark_provider_retried` and :func:`mark_provider_usage`,
    for the same reason and in one copy for the same reason: an exception type that refuses the
    attribute (``__slots__``) simply stays unnamed rather than replacing the provider's failure
    with an ``AttributeError`` raised *inside* an except-handler. The read is inside the guard
    too -- a third-party subclass may expose the name as a property that raises.
    """

    try:
        if getattr(error, "provider_error_code", None):
            return
        error.provider_error_code = code  # type: ignore[attr-defined]
    except Exception:
        pass


def provider_usage_of(error: BaseException) -> dict[str, int]:
    """Read back what :func:`mark_provider_usage` stamped, as clean non-negative counts.

    One reader for every consumer of the stamp -- the loop's budget, the reference gateway's
    tenant meter, the error envelope that carries it across a hop. A guarded read like the
    stamp itself, and it filters rather than raises: a malformed count on a *failure* path
    must not replace the failure being reported.
    """

    try:
        usage = getattr(error, "provider_usage", None)
    except Exception:
        return {}
    if not isinstance(usage, Mapping):
        return {}
    return {
        str(key): value
        for key, value in usage.items()
        if type(value) is int and value >= 0
    }


@dataclass
class RetryProgress:
    """What an adapter has managed to report about a call that may never return one.

    Every other carrier of `provider_retried` belongs to an *outcome* -- a turn, a chunk, an
    exception the adapter raised. A call the run abandons produces none of them: a blocking
    `next_turn` keeps running on a thread nobody reads, and the failure the receipt is built from
    is the `RunCancelled`/`RunTimeout` the race raised, which the adapter never touched. A run that
    timed out *because* the provider was retrying is the case most likely to matter, and it was the
    one case that recorded a clean single attempt.

    Mutated rather than replaced, because that is what crosses a thread: the worker runs under a
    copy of the caller's context, so `ContextVar.set` there is invisible here, while a write to the
    object both sides already hold is not.
    """

    retried: bool = False


_RETRY_PROGRESS: ContextVar[RetryProgress | None] = ContextVar(
    "monoid_agent_kernel_retry_progress", default=None
)


def report_provider_retried() -> None:
    """Called by an adapter when its own retry loop is about to make another attempt.

    Optional and inert by default: an adapter that never calls it is reported as never retrying,
    which is exactly true of one with no retry loop, and a call made outside a runner does nothing.
    Report on the *decision* to retry, not on the next attempt's success -- an attempt that never
    completes is precisely the one whose evidence is otherwise lost.
    """

    progress = _RETRY_PROGRESS.get()
    if progress is not None:
        progress.retried = True


@contextmanager
def collect_retry_reports() -> Iterator[RetryProgress]:
    """Install a channel for `report_provider_retried` for the duration of one call."""

    progress = RetryProgress()
    token = _RETRY_PROGRESS.set(progress)
    try:
        yield progress
    finally:
        _RETRY_PROGRESS.reset(token)


def assemble_streamed_turn(chunks: list[ModelStreamChunk]) -> ModelTurn:
    """Fold a streamed chunk sequence into the same :class:`ModelTurn` a one-shot turn
    would produce: concatenate text; group tool-call argument fragments by ``index`` and
    decode each once at the end; take ``response_id``/``usage`` from ``TurnComplete``.
    """
    ingress = ModelStreamIngressNormalizer()
    normalized_chunks: list[ModelStreamChunk] = []
    for chunk in chunks:
        normalized_chunks.extend(ingress.normalize(chunk))
    normalized_chunks.extend(ingress.finish())

    text_parts: list[str] = []
    slots: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    response_id: str | None = None
    usage: dict[str, int] = {}
    reasoning: tuple[dict[str, Any], ...] = ()
    stop_reason: StopReason | None = None
    provider_retried = False
    for chunk in normalized_chunks:
        # Read off every chunk, not just the terminal one: a retried stream says so from its first
        # fragment onward so the fact survives a call that never reaches ``TurnComplete``.
        if chunk.provider_retried:
            provider_retried = True
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
            if chunk.reasoning:
                reasoning = chunk.reasoning
            if chunk.stop_reason is not None:
                stop_reason = chunk.stop_reason
    tool_calls: list[ToolCall] = []
    for index in order:
        slot = slots[index]
        raw = slot["args"].strip()
        # Both refusals below are refusals of a turn the provider already produced and BILLED:
        # the deltas were delivered and the terminal frame reported the cost, which the fold is
        # still holding. The one-shot twin of this act pays through the OpenAI reader's stamping
        # seam; unstamped here, a streamed turn was metered at zero at the tenant ledger and in
        # the run's token budget. ``provider_retried`` rides along for the same reason it does
        # on every other refusal: it is a fact about attempts already made. The meter skips an
        # empty mapping, so a stream that reported no cost still invents none.
        #
        # The CODE, though, deliberately does not converge with that seam's. The OpenAI reader
        # backfills ``openai_bad_response`` onto its bare refusals; these two keep
        # ``stream_bad_tool_args``, and this module never speaks a provider's name -- the fold
        # is reached by every adapter, including third-party ones whose provider it cannot know.
        # So one class of defect really does carry two codes across the sync/streamed pair, and
        # that is the intended answer rather than an unbound twin: the ingress voice attributes
        # the *shape* of the failure, the adapter voice attributes its *source*. Recorded here
        # because a silent cell in a twin census reads as an oversight to the next reviewer.
        try:
            arguments = loads_model_json_ingress(raw) if raw else {}
        except ValueError as exc:
            unparsable = ModelAdapterError(
                f"invalid streamed tool-call arguments for {slot['name']}",
                provider_error_code="stream_bad_tool_args",
                retryable=False,
                provider_retried=provider_retried,
            )
            mark_provider_usage(unparsable, usage)
            raise unparsable from exc
        if not isinstance(arguments, dict):
            wrong_type = ModelAdapterError(
                f"streamed tool-call arguments for {slot['name']} are not an object",
                provider_error_code="stream_bad_tool_args",
                retryable=False,
                provider_retried=provider_retried,
            )
            mark_provider_usage(wrong_type, usage)
            raise wrong_type
        call_id = slot["id"] if slot["id"] is not None else ""
        name = slot["name"] if slot["name"] is not None else ""
        tool_calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
    # No explicit stop_reason streamed (older gateway / a chunk source that omits it): infer the
    # common cases so the loop's branch still works — tool calls present → tool_calls, else stop.
    if stop_reason is None:
        stop_reason = "tool_calls" if tool_calls else "stop"
    try:
        # Provably a re-normalization: every chunk passed the ingress above, whose
        # TurnComplete branch already ran normalize_usage. Guarded rather than deleted —
        # removing normalization is a loosening-shaped edit, and a future path that reaches
        # this fold with garbage must refuse in the ingress's classified voice, not raw.
        normalized_usage = normalize_usage(usage) if usage else {}
    except Exception as exc:
        # Classified the way the ingress classifies, flags included. The constructor's defaults
        # are deterministic, not arbitrary -- ``retryable`` is False and ``provider_retried`` is
        # False -- so the two keywords do different work: ``retryable=False`` restates the
        # default where a reader would otherwise have to go look it up, while
        # ``provider_retried`` is the one fact the fold is holding that a default would throw
        # away, since a stream the SDK re-sent would report a clean single attempt. No usage
        # stamp -- ``usage`` is
        # itself the malformed key here, and the tolerance rule on a failure path is to record
        # nothing rather than raise a second failure over the first. Unreachable through this
        # function (the ingress above pre-normalizes every chunk), so what binds this shape is
        # a source-level pin: test_the_folds_usage_renormalization_stays_structurally_guarded
        # in tests/test_llm_gateway_stream.py.
        raise ModelAdapterError(
            "model adapter returned a non-portable stream fragment",
            retryable=False,
            provider_retried=provider_retried,
        ) from exc
    return ModelTurn(
        response_id=response_id,
        final_text="".join(text_parts) if text_parts else None,
        tool_calls=tuple(tool_calls),
        usage=normalized_usage,
        reasoning=reasoning,
        stop_reason=stop_reason,
        provider_retried=provider_retried,
    )
