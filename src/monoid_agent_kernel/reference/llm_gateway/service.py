from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from monoid_agent_kernel.reference._shared.tokens import TokenClaims, TokenError, TokenManager
from monoid_agent_kernel.core.wire_validation import (
    optional_list,
    parse_bool,
    parse_required_str,
    parse_str,
    require_list,
    require_object,
)
from monoid_agent_kernel.core.spec import GenerationConfig, ModelConfig, ReasoningConfig
from monoid_agent_kernel.core.json_ingress import normalize_json_ingress
from monoid_agent_kernel.errors import ModelAdapterError, PermissionDenied
from monoid_agent_kernel.identifiers import accepted_namespaced_ids, namespaced_id
from monoid_agent_kernel.providers._common import build_generation_payload, normalize_usage
from monoid_agent_kernel.providers.base import (
    generation_support,
    provider_usage_of,
    structured_output_support,
)
from monoid_agent_kernel.providers.base import (
    ModelAdapter,
    ModelRequest,
    ModelStreamIngressNormalizer,
    ModelStreamChunk,
    ModelTurn,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    ToolObservation,
    TurnComplete,
    assemble_streamed_turn,
    normalize_model_turn,
)
from monoid_agent_kernel.providers.openai import OpenAIModelAdapter
from monoid_agent_kernel.tools.base import ToolResult, ToolSpec

ProviderAdapterFactory = Callable[[TokenClaims, ModelConfig], ModelAdapter]


@dataclass(frozen=True)
class LlmGatewayTurnRequest:
    protocol: str
    model: str
    system_prompt: str
    tools: tuple[ToolSpec, ...]
    reasoning: ReasoningConfig
    instruction: str = ""
    previous_turn_handle: str | None = None
    observations: tuple[ToolObservation, ...] = ()
    # By-value conversation: the full message log, forwarded to the upstream provider
    # statelessly. When set, no previous_turn_handle lookup is needed.
    messages: tuple[dict[str, Any], ...] | None = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    output_schema: dict[str, Any] | None = None


@dataclass
class LlmGatewayTurnRecord:
    turn_handle: str
    provider_response_id: str | None
    run_id: str
    tenant_id: str
    user_id: str
    model: str
    created_at: float


@dataclass
class LlmGatewayUsage:
    """The tenant meter. It sums what ``normalize_usage`` emits — all of it.

    The four sub-counts below are priced differently from plain input tokens (a cache read is
    cheap, a cache write and a reasoning token are not), so a meter that folds them away cannot
    reconstruct a bill. Worse, a provider that reports a cost *only* as sub-counts metered as
    total=0: the priced call was invisible to this ledger entirely. ``total_tokens`` is still
    whatever the provider reported as the total and is not re-derived here — the sub-counts are
    reported beside it as their own columns, which is what makes such a call visible.
    """

    tenant_id: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0
    audio_tokens: int = 0

    def add(self, usage: dict[str, int]) -> None:
        normalized = normalize_usage(usage)
        self.calls += 1
        self.input_tokens += normalized["input_tokens"]
        self.output_tokens += normalized["output_tokens"]
        self.total_tokens += normalized["total_tokens"]
        # Emitted only when the adapter reported one, so each read defaults.
        self.cache_read_tokens += normalized.get("cache_read_tokens", 0)
        self.cache_creation_tokens += normalized.get("cache_creation_tokens", 0)
        self.reasoning_tokens += normalized.get("reasoning_tokens", 0)
        self.audio_tokens += normalized.get("audio_tokens", 0)

    def to_json(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "audio_tokens": self.audio_tokens,
        }


@dataclass
class LlmGatewayBackend:
    token_manager: TokenManager
    provider_adapter_factory: ProviderAdapterFactory | None = None
    _turns: dict[str, LlmGatewayTurnRecord] = field(default_factory=dict, init=False, repr=False)
    _usage: dict[str, LlmGatewayUsage] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def handle_turn(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        claims = self._authorize(token)
        payload = _normalized_turn_payload(payload)
        request = _parse_turn_request(payload)
        self._validate_request_against_claims(request, claims)
        # By-value carries the full conversation as messages → forward statelessly, no
        # handle lookup. The legacy by-reference path still translates handle → response id.
        provider_previous_response_id = (
            None
            if request.messages is not None
            else self._provider_previous_response_id(request, claims)
        )
        config = _upstream_model_config(request)
        adapter = self._build_adapter(claims, config)
        # A failure can arrive *after* the upstream produced and billed an answer -- an
        # applied-parameters refusal raised by an upstream that is itself a gateway is exactly
        # that. This handler exits on the raise, before the meter below, so those tokens left
        # the tenant's ledger entirely. Metered here, then re-raised unchanged.
        try:
            turn = normalize_model_turn(
                adapter.next_turn(
                    ModelRequest(
                        instruction=request.instruction,
                        system_prompt=request.system_prompt,
                        tools=request.tools,
                        previous_turn_handle=provider_previous_response_id,
                        observations=request.observations,
                        model=config,
                        messages=request.messages,
                        output_schema=request.output_schema,
                    )
                )
            )
        except Exception as failed:
            self._meter_failure(claims.tenant_id, failed)
            raise
        turn_handle = self._record_turn(claims, request, turn)
        self._meter(claims.tenant_id, turn.usage)
        result = {
            "protocol": namespaced_id("llm-turn-result.v1"),
            "turn_handle": turn_handle,
            "final_text": turn.final_text,
            "tool_calls": [
                {"call_id": call.id, "name": call.name, "arguments": call.arguments}
                for call in turn.tool_calls
            ],
            "usage": turn.usage,
            "stop_reason": turn.stop_reason,
            # The backend adapter's own retry evidence. Dropping it here recorded a call the
            # provider retried as a clean single attempt, since the client can only observe its
            # own HTTP attempts and this call succeeded on the first of those.
            "provider_retried": turn.provider_retried,
        }
        result.update(_applied_echoes(request, adapter, config))
        result.update(_reasoning_payload(turn))
        return result

    def handle_turn_stream(self, token: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Streaming form of :meth:`handle_turn` — yields SSE-ready frame dicts.

        Authorization, parsing and adapter construction happen eagerly (before any frame is
        produced), so a pre-stream failure raises here and the HTTP layer maps it to a normal
        error response rather than a 200 SSE error frame. Provider token deltas are forwarded
        live; the provider's ``TurnComplete`` is consumed (its response id is never exposed).
        After the drain the turn is assembled once, the opaque ``turn_handle`` is recorded (for
        the by-reference continuation path) and usage is metered, then a final gateway
        ``turn_complete`` frame is yielded.
        """
        claims = self._authorize(token)
        payload = _normalized_turn_payload(payload)
        request = _parse_turn_request(payload)
        self._validate_request_against_claims(request, claims)
        provider_previous_response_id = (
            None
            if request.messages is not None
            else self._provider_previous_response_id(request, claims)
        )
        config = _upstream_model_config(request)
        adapter = self._build_adapter(claims, config)
        model_request = ModelRequest(
            instruction=request.instruction,
            system_prompt=request.system_prompt,
            tools=request.tools,
            previous_turn_handle=provider_previous_response_id,
            observations=request.observations,
            model=config,
            messages=request.messages,
            output_schema=request.output_schema,
        )
        # Everything above can raise; only past this point are we committed to a stream body.
        return self._stream_turn(claims, request, adapter, model_request)

    def _stream_turn(
        self,
        claims: TokenClaims,
        request: LlmGatewayTurnRequest,
        adapter: ModelAdapter,
        model_request: ModelRequest,
    ) -> Iterator[dict[str, Any]]:
        collected: list[ModelStreamChunk] = []
        try:
            astream_turn = getattr(adapter, "astream_turn", None)
            if astream_turn is not None:
                ingress = ModelStreamIngressNormalizer()
                try:
                    for provider_chunk in _pump_astream(astream_turn, model_request):
                        for chunk in ingress.normalize(provider_chunk):
                            collected.append(chunk)
                            frame = _chunk_to_frame(chunk)
                            if frame is not None:
                                yield frame
                except BaseException:
                    for chunk in ingress.finish():
                        collected.append(chunk)
                        frame = _chunk_to_frame(chunk)
                        if frame is not None:
                            yield frame
                    raise
                for chunk in ingress.finish():
                    collected.append(chunk)
                    frame = _chunk_to_frame(chunk)
                    if frame is not None:
                        yield frame
            else:
                # The provider can't stream: synthesize a minimal delta sequence from the
                # one-shot turn so consumers still see text/tool frames before turn_complete.
                turn = normalize_model_turn(adapter.next_turn(model_request))
                # The synthesized chunks carry the turn's retry evidence too. They stand in for a
                # stream the provider could not produce, so anything the turn reports about the call
                # has to survive the substitution.
                if turn.final_text:
                    chunk: ModelStreamChunk = TextDelta(
                        turn.final_text, provider_retried=turn.provider_retried
                    )
                    collected.append(chunk)
                    yield _chunk_to_frame(chunk)
                for index, call in enumerate(turn.tool_calls):
                    chunk = ToolCallDelta(
                        index=index,
                        arguments_fragment=json.dumps(
                            call.arguments,
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                        id=call.id,
                        name=call.name,
                        provider_retried=turn.provider_retried,
                    )
                    collected.append(chunk)
                    yield _chunk_to_frame(chunk)
                collected.append(
                    TurnComplete(
                        response_id=turn.response_id,
                        usage=turn.usage,
                        # The reasoning artifacts ride the synthesized terminal chunk for the same
                        # reason the retry evidence does: this chunk stands in for a stream the
                        # provider could not produce, and ``assemble_streamed_turn`` reads
                        # ``reasoning`` off ``TurnComplete`` and nowhere else -- so dropping it
                        # here emptied the terminal frame on this branch alone, while the branch
                        # that forwards the provider's own ``TurnComplete`` stayed correct.
                        reasoning=turn.reasoning,
                        stop_reason=turn.stop_reason,
                        provider_retried=turn.provider_retried,
                    )
                )
            # Assemble once: the same usage drives both the meter and the outgoing frame, and the
            # assembled response id is what the opaque turn_handle maps to for continuation.
            turn = normalize_model_turn(assemble_streamed_turn(collected))
        except Exception as failed:
            # The streaming twin of handle_turn's failure meter: a refusal can arrive *after*
            # the upstream produced and billed an answer (a chained hop's proof refusal is
            # exactly that), and this generator exits on the raise before the success-path
            # meter below -- so the billed tokens left the tenant ledger entirely on this
            # transport while the sync twin metered them. One handler around both sub-branches
            # (the astream drive and the non-streaming fallback), and ``Exception`` rather than
            # ``ModelAdapterError`` for the reason ``_meter_failure`` states -- the OpenAI
            # stream's terminal refusals, the ones this transport is most likely to meet, are
            # raw types. ``BaseException`` is deliberately NOT caught: a consumer closing this
            # generator raises ``GeneratorExit`` at the yields above, which is a cancelled read
            # rather than a failed call.
            self._meter_failure(claims.tenant_id, failed)
            raise
        turn_handle = self._record_turn(claims, request, turn)
        self._meter(claims.tenant_id, turn.usage)
        frame = {
            "type": "turn_complete",
            "turn_handle": turn_handle,
            "usage": turn.usage,
            "stop_reason": turn.stop_reason,
            "provider_retried": turn.provider_retried,
        }
        # Streaming twin of handle_turn's echo -- the terminal frame is the only one a
        # streaming client can read it from, and it is built by the same function so the two
        # transports cannot answer differently.
        frame.update(_applied_echoes(request, adapter, model_request.model))
        frame.update(_reasoning_payload(turn))
        yield frame

    def tenant_usage(self, tenant_id: str) -> dict[str, Any]:
        with self._lock:
            usage = self._usage.get(tenant_id) or LlmGatewayUsage(tenant_id)
            return usage.to_json()

    def _authorize(self, token: str) -> TokenClaims:
        try:
            return self.token_manager.verify(
                token,
                kind="llm_gateway",
                audience="csp.llm-gateway",
            )
        except TokenError as exc:
            raise PermissionDenied(str(exc)) from exc

    def _validate_request_against_claims(
        self,
        request: LlmGatewayTurnRequest,
        claims: TokenClaims,
    ) -> None:
        del request, claims

    def _provider_previous_response_id(
        self,
        request: LlmGatewayTurnRequest,
        claims: TokenClaims,
    ) -> str | None:
        if request.previous_turn_handle is None:
            return None
        with self._lock:
            record = self._turns.get(request.previous_turn_handle)
        if record is None:
            raise ModelAdapterError("unknown previous_turn_handle")
        if record.run_id != claims.run_id or record.tenant_id != claims.tenant_id:
            raise PermissionDenied("previous_turn_handle does not belong to this run")
        return record.provider_response_id

    def _build_adapter(
        self,
        claims: TokenClaims,
        config: ModelConfig,
    ) -> ModelAdapter:
        """Takes the already-built per-call config (``_upstream_model_config``) rather than
        rebuilding it, so the adapter, the upstream request, and the applied-parameters proof
        share one object — "cannot disagree" by identity, not merely by value."""

        if self.provider_adapter_factory is not None:
            return self.provider_adapter_factory(claims, config)
        return OpenAIModelAdapter(config, allow_direct_provider_api=True)

    def _meter(self, tenant_id: str, usage: dict[str, int]) -> None:
        """Add one call's tokens to the tenant ledger. One writer, so the success paths and
        the billed-failure path cannot come to disagree about what gets counted."""

        if not usage:
            return
        with self._lock:
            self._usage.setdefault(tenant_id, LlmGatewayUsage(tenant_id)).add(usage)

    def _meter_failure(self, tenant_id: str, failed: BaseException) -> None:
        """Charge the tenant for what an ESCAPING failure already cost, then let it escape.

        Both transports' failure arms come through here instead of reading the stamp for
        themselves, and both catch ``Exception`` rather than ``ModelAdapterError``. The adapter
        that first sees the provider's billed body stamps refusals of more than one type:
        ``normalize_usage`` says "malformed usage" with a raw ``ValueError``, and *every* refusal
        in the OpenAI stream's terminal region is a raw ``ValueError``/``AttributeError``, because
        that path folds deltas and reads end-of-turn metadata directly rather than running the
        one-shot mapping that classifies. Gated on ``ModelAdapterError``, this meter read the
        stamp on exactly the failures that had already been classified and skipped the ones that
        had not -- so an upstream whose final payload is malformed charged the tenant nothing for
        a turn it had generated and billed, on the transport where that shape actually occurs.

        Meter and re-raise: nothing is swallowed and nothing is reclassified here, so what
        escapes is what arrived. ``provider_usage_of`` reads ``{}`` for an unbilled failure and
        :meth:`_meter` skips empty usage, which is what keeps a failure raised before the
        provider free.
        """

        self._meter(tenant_id, provider_usage_of(failed))

    def _record_turn(
        self,
        claims: TokenClaims,
        request: LlmGatewayTurnRequest,
        turn: ModelTurn,
    ) -> str:
        turn_handle = f"turn_{uuid.uuid4().hex}"
        with self._lock:
            self._turns[turn_handle] = LlmGatewayTurnRecord(
                turn_handle=turn_handle,
                provider_response_id=turn.response_id,
                run_id=claims.run_id,
                tenant_id=claims.tenant_id,
                user_id=claims.user_id,
                model=request.model,
                created_at=time.time(),
            )
        return turn_handle


# The wire spellings one tool entry may carry its argument schema under, in the order
# ``_parse_tool`` prefers them. Shared with the ingress above so the two cannot come to
# disagree about which keys hold a schema -- a key kept verbatim by one and rewritten by the
# other is the same defect wearing a second name.
_TOOL_SCHEMA_KEYS: tuple[str, ...] = ("input_schema", "parameters")


def _normalized_turn_payload(payload: dict[str, Any]) -> Any:
    """The server-side ingress rule, matching the client's exactly.

    The blanket normalize substitutes non-finite *content* values; a **schema** is config, not
    content -- the client ingress keeps one verbatim (``normalize_model_request`` and
    ``normalize_tool_spec``, both ``substitute_nonfinite=False``) so a non-finite value is
    *refused* downstream rather than silently rewritten into a different constraint. Riding
    the blanket normalize here turned the caller's ``NaN`` into ``null`` before the upstream
    adapter ever saw it -- the exact rewrite the rule exists to rule out, on the one route
    (in-process Python callers) the JSON parsers don't guard. One function, both handlers.

    This request carries schemas in two places, and the rule is about schemas, not about the
    field it was first noticed on: ``output_schema`` for the answer, and one per entry of
    ``tools`` for its arguments (under either wire spelling ``_parse_tool`` accepts). Each is
    re-normalized from the **original** payload, since the blanket copy above has already lost
    the value.
    """

    normalized = normalize_json_ingress(payload)
    if not isinstance(normalized, dict):
        return normalized
    if normalized.get("output_schema") is not None:
        normalized["output_schema"] = normalize_json_ingress(
            payload.get("output_schema"), substitute_nonfinite=False
        )
    original_tools = payload.get("tools")
    normalized_tools = normalized.get("tools")
    if isinstance(normalized_tools, list) and isinstance(original_tools, (list, tuple)):
        for normalized_tool, original_tool in zip(normalized_tools, original_tools):
            if not isinstance(normalized_tool, dict) or not isinstance(original_tool, Mapping):
                continue
            for key in _TOOL_SCHEMA_KEYS:
                if key in original_tool and normalized_tool.get(key) is not None:
                    normalized_tool[key] = normalize_json_ingress(
                        original_tool[key], substitute_nonfinite=False
                    )
    return normalized


def _upstream_model_config(request: LlmGatewayTurnRequest) -> ModelConfig:
    """The one config this turn runs under.

    Built once and shared by the adapter construction, the upstream request, and the
    applied-parameters proof, so the three cannot disagree about policy: the adapter enforces
    under ``request.model or self.config``, and the proof is only honest if it is probed under
    the same config the enforcement will read.
    """

    return ModelConfig(
        provider="openai",
        model=request.model,
        reasoning=request.reasoning,
        generation=request.generation,
    )


def _applied_echoes(
    request: LlmGatewayTurnRequest, adapter: ModelAdapter, config: ModelConfig
) -> dict[str, Any]:
    """The applied-parameters proofs for one turn — built once, emitted by both transports.

    Both echoes answer the same question, so both are derived the same way: **from what the
    upstream adapter declared it does**, never from what the request asked for. A gateway that
    copied the requested block back would produce an exact match no matter what the upstream
    did with it — an offline echo adapter, a text-only backend, or any
    ``provider_adapter_factory`` that ignores ``ModelConfig.generation`` would all read as
    "applied", and the client's ``on_unsupported="fail"`` would accept sampling parameters that
    were never sent to a model. Unproven is reported as unproven: the generation echo is simply
    absent (which a fail-closed client refuses), and the schema echo is an explicit ``False``.

    ``config`` is the per-call config the upstream call runs under (``_upstream_model_config``),
    threaded into the probes because a declaration may be policy-conditional: a *chained*
    ``GatewayModelAdapter`` claims "native" only while it is enforcing, and it enforces under
    the per-call config, not its standing one. Probing the standing config let a shared
    factory-built adapter mint proof for a call whose wire policy said ``"omit"`` — the exact
    copied-back-proof defect this function exists to rule out, one config-source hop later.

    Both stay off the response entirely when the request did not use the feature, so traffic
    that configures neither keeps its exact pre-W5 wire shape.
    """

    echoes: dict[str, Any] = {}
    requested_generation = build_generation_payload(config.generation)
    if requested_generation and generation_support(adapter, config) == "native":
        echoes["generation_applied"] = requested_generation
    if request.output_schema is not None:
        echoes["schema_applied"] = structured_output_support(adapter, config) == "native"
    return echoes


def _reasoning_payload(turn: ModelTurn) -> dict[str, Any]:
    """The turn's provider-native reasoning artifacts, for whichever transport is writing.

    The kernel captures these items and replays them verbatim on the next by-value turn, which
    is what makes a ZDR reasoning round-trip possible at all. The request half of that loop
    already crossed this hop -- ``messages`` ride by value and are forwarded untouched -- but the
    response half did not, so a run routed through the gateway captured nothing and replayed
    nothing. Relayed verbatim, because this hop has no business interpreting them.

    Not opaque, though, and the distinction matters to whoever writes the redaction policy: the
    captured subsequence is ``reasoning`` items PLUS the ``function_call``/``message`` items they
    are paired with (the provider validates that adjacency), so only the reasoning-type entries
    carry ``encrypted_content``. A ``message`` entry holds the model's plaintext answer and a
    ``function_call`` entry holds plaintext arguments -- the same content ``final_text`` and
    ``tool_calls`` carry on this very envelope. Treat the array as MODEL CONTENT when logging or
    truncating: it roughly doubles a small body, and it defeats any bound applied only to the
    fields beside it.

    Built by one function and used by both writers, exactly like :func:`_applied_echoes`, so the
    two transports cannot come to disagree about a fact neither of them authored.

    Omit-when-empty, and the conditionality is a property of the *answer* rather than of the
    request: traffic whose upstream produced no reasoning keeps its exact previous wire shape,
    and a client that never hears the key reads it as "no artifacts", which is the only thing an
    absent key can honestly mean.
    """

    if not turn.reasoning:
        return {}
    return {"reasoning": [dict(item) for item in turn.reasoning]}


def _pump_astream(
    astream_turn: Callable[[ModelRequest], Any], model_request: ModelRequest
) -> Iterator[ModelStreamChunk]:
    """Drive an async ``astream_turn`` from this sync handler thread on a private event loop.

    The async generator is created, advanced and closed on the SAME loop; cleanup runs
    ``aclose`` (so the provider's ``finally`` fires) before ``loop.close`` and also drains any
    nested async generators. ``set_event_loop`` is deliberately not called, so this private
    loop is never leaked as the thread's current loop.
    """
    loop = asyncio.new_event_loop()
    agen = astream_turn(model_request)
    try:
        while True:
            try:
                chunk = loop.run_until_complete(agen.__anext__())
            except StopAsyncIteration:
                break
            yield chunk
    finally:
        try:
            loop.run_until_complete(agen.aclose())
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()


def _chunk_to_frame(chunk: ModelStreamChunk) -> dict[str, Any] | None:
    """Translate a provider chunk into an SSE frame dict, or ``None`` to drop it.

    The provider's ``TurnComplete`` is dropped: the gateway mints its own terminal frame
    carrying the opaque ``turn_handle`` instead of the provider's response id.

    ``provider_retried`` rides every frame that carries it, not only the terminal one. A stream
    cancelled mid-flight never reaches ``turn_complete``, and that is exactly when a client needs
    to know the answer it did receive cost the provider more than one attempt.
    """
    frame: dict[str, Any] | None = None
    if isinstance(chunk, TextDelta):
        frame = {"type": "text_delta", "text": chunk.text}
    elif isinstance(chunk, ReasoningDelta):
        frame = {"type": "reasoning_delta", "text": chunk.text}
    elif isinstance(chunk, ToolCallDelta):
        frame = {
            "type": "tool_call_delta",
            "index": chunk.index,
            "arguments_fragment": chunk.arguments_fragment,
        }
        if chunk.id is not None:
            frame["id"] = chunk.id
        if chunk.name is not None:
            frame["name"] = chunk.name
    if frame is not None and getattr(chunk, "provider_retried", False):
        frame["provider_retried"] = True
    return frame


LLM_TURN_PROTOCOL_VERSION = namespaced_id("llm-turn.v1")
ACCEPTED_LLM_TURN_PROTOCOL_VERSIONS = accepted_namespaced_ids("llm-turn.v1")


def _parse_turn_request(payload: dict[str, Any]) -> LlmGatewayTurnRequest:
    payload = require_object(payload, "LLM gateway turn request")
    protocol = parse_str(payload, "protocol")
    if protocol not in ACCEPTED_LLM_TURN_PROTOCOL_VERSIONS:
        raise ValueError("unsupported LLM gateway protocol")
    previous_turn_handle = parse_str(payload, "previous_turn_handle") or None
    observations = tuple(
        _parse_observation(item) for item in optional_list(payload, "observations")
    )
    instruction = parse_str(payload, "instruction")
    messages = (
        tuple(
            require_object(item, "message")
            for item in require_list(payload["messages"], "messages")
        )
        if payload.get("messages") is not None
        else None
    )
    if messages is None and previous_turn_handle is None and not instruction.strip():
        raise ValueError("instruction is required for the first LLM turn")
    return LlmGatewayTurnRequest(
        protocol=LLM_TURN_PROTOCOL_VERSION,
        model=parse_required_str(payload, "model"),
        system_prompt=parse_required_str(payload, "system_prompt", non_empty=False),
        tools=tuple(_parse_tool(item) for item in optional_list(payload, "tools")),
        # The shared codecs are the parser (fail-closed, spec.py) rather than a second
        # per-key reader that would drift from them: an out-of-range temperature or an
        # unknown effort 400s at this boundary instead of travelling to the provider.
        reasoning=ReasoningConfig.from_json(
            require_object(payload["reasoning"], "reasoning") if "reasoning" in payload else None
        ),
        generation=GenerationConfig.from_json(
            require_object(payload["generation"], "generation")
            if "generation" in payload
            else None
        ),
        output_schema=(
            dict(require_object(payload["output_schema"], "output_schema"))
            if payload.get("output_schema") is not None
            else None
        ),
        instruction=instruction,
        previous_turn_handle=previous_turn_handle,
        observations=observations,
        messages=messages,
    )


def _parse_tool(raw: dict[str, Any]) -> ToolSpec:
    def handler(_context, _args):
        return ToolResult(ok=False, error="gateway tool proxy cannot execute tools")

    raw = require_object(raw, "tool")
    tool_id = parse_str(raw, "id") or parse_str(raw, "name")
    input_schema: dict[str, Any] = {}
    for key in _TOOL_SCHEMA_KEYS:
        if key in raw:
            input_schema = require_object(raw[key], key)
            break
    return ToolSpec(
        id=tool_id,
        provider_name=parse_str(raw, "name") or tool_id.replace(".", "_"),
        description=parse_str(raw, "description"),
        input_schema=dict(input_schema),
        capability=parse_str(raw, "capability") or "unknown",
        side_effect=parse_str(raw, "side_effect") or "read",  # type: ignore[arg-type]
        handler=handler,
    )


def _parse_observation(raw: dict[str, Any]) -> ToolObservation:
    raw = require_object(raw, "observation")
    return ToolObservation(
        call_id=parse_required_str(raw, "call_id"),
        tool_name=parse_str(raw, "tool_name"),
        output=require_object(raw["output"], "output") if "output" in raw else {},
        is_background=parse_bool(raw, "is_background", default=False),
    )
