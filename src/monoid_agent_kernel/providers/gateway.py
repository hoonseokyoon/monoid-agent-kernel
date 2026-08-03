from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from monoid_agent_kernel.core.json_ingress import (
    is_finite_json_number,
    loads_json_ingress,
    loads_model_envelope_json_ingress,
    loads_model_json_ingress,
    loads_model_stream_envelope_json_ingress,
    normalize_json_ingress,
    normalize_unicode_scalars,
)
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel._version import user_agent
from monoid_agent_kernel.env import env_name_for_error, getenv
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.identifiers import namespaced_id
from monoid_agent_kernel.providers._common import (
    build_generation_payload,
    build_reasoning_payload,
    normalize_usage,
    project_message_to_text,
)
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelStreamChunk,
    ModelTurn,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    TurnComplete,
    mark_provider_retried,
    mark_provider_usage,
    report_provider_retried,
)
from monoid_agent_kernel.tools.base import ToolSpec

DEFAULT_GATEWAY_URL_ENV = "MONOID_LLM_GATEWAY_URL"
DEFAULT_GATEWAY_TOKEN_ENV = "MONOID_LLM_GATEWAY_TOKEN"

GATEWAY_TIMEOUT = "gateway_timeout"
GATEWAY_NETWORK_ERROR = "gateway_network_error"
GATEWAY_RATE_LIMITED = "gateway_rate_limited"
GATEWAY_SERVER_ERROR = "gateway_server_error"
GATEWAY_AUTH_ERROR = "gateway_auth_error"
GATEWAY_BAD_RESPONSE = "gateway_bad_response"
GATEWAY_GENERATION_NOT_APPLIED = "gateway_generation_not_applied"
GATEWAY_SCHEMA_NOT_APPLIED = "gateway_schema_not_applied"
GATEWAY_BAD_REQUEST = "gateway_bad_request"


def _encode_request_body(payload: dict[str, Any]) -> bytes:
    """Serialize one request payload, classifying what cannot be serialized.

    ``normalize_json_ingress`` deliberately leaves arbitrary non-JSON scalars alone (the
    documented arbitrary-scalar gap), so a Python-direct caller can hand ``output_schema``,
    ``messages``, or an observation a value ``json.dumps`` refuses -- a ``set``, a function, a
    NaN under ``allow_nan=False``. Encoded here, once, for both transports: outside a
    classifier that failure escaped as a raw ``TypeError``/``ValueError`` the loop cannot
    classify at all, terminalizing the run unrecoverably for what is a config-shaped mistake.
    ``config_recoverable`` completes that sentence: the same mistake reported by a gateway
    *server* is an HTTP 400, which the loop treats as turn-recoverable -- one condition, one
    classification, whichever side of the wire noticed.

    ``RecursionError`` is caught beside them because it is the same condition wearing a
    different type: ``json.dumps`` recurses, so a container nested deeper than the interpreter
    limit raises a ``RuntimeError`` subclass rather than a ``TypeError``. Nothing upstream
    refuses it first -- ``normalize_json_ingress`` is deliberately iterative, and the 512-level
    nesting cap guards the JSON *text* parsers, not a Python-constructed value -- so this is
    the boundary, and "a request that cannot be encoded" must answer the same way however the
    encoder says so.
    """

    try:
        return json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ModelAdapterError(
            f"model request is not JSON-serializable: {exc}",
            provider_error_code=GATEWAY_BAD_REQUEST,
            retryable=False,
            config_recoverable=True,
        ) from exc


def _stamp_retry(error: BaseException, attempt: int) -> None:
    """Record on an escaping error that the adapter's retry loop had already run.

    Read back by ``ModelCallReceipt.with_error`` through ``getattr``, so an exception that refuses
    the attribute (``__slots__``) simply reports no retry rather than replacing the provider's
    failure with an AttributeError.
    """

    if attempt <= 1:
        return
    mark_provider_retried(error)


@dataclass
class GatewayModelAdapter:
    config: ModelConfig
    gateway_url: str | None = None
    token: str | None = None
    token_env: str = DEFAULT_GATEWAY_TOKEN_ENV
    token_file: Path | None = None
    # Optional token source, consulted per HTTP attempt — every retry re-resolves, on both the
    # blocking and the streamed path, not once per call. When set, it takes precedence over the
    # static token/file/env — so a backend can supply a callable that re-mints a fresh gateway token
    # near expiry, keeping a long run (one that outlives the token TTL) authenticated without a
    # restart. ``None`` = today's static behavior.
    token_provider: Callable[[], str | None] | None = None
    # Whose native reasoning artifacts this transport RELAYS -- the upstream provider behind the
    # gateway, never the hop itself. The loop reads it (ProviderNamedModelAdapter) to tag the
    # opaque items it captured off a turn, and replays a tagged block only to a matching
    # adapter and model, so the tag has to name the thing that can actually read the items back:
    # an OpenAI-encrypted reasoning item returned to anything else is an unusable request one
    # turn later. Without this field the loop dropped every artifact the gateway relayed, one
    # line after the reader reconstructed it.
    #
    # Defaults to the reference gateway's hardcoded upstream (``_upstream_model_config`` builds
    # ``provider="openai"`` and ``_build_adapter`` falls back to ``OpenAIModelAdapter``). A
    # deployment whose ``provider_adapter_factory`` routes elsewhere must set this to its real
    # upstream; ``None`` disables tagging, which is the protocol's documented "do not tag" and
    # the right answer for a gateway fronting an upstream with no reasoning artifacts.
    #
    # It also names the provider on the observability surfaces that probe an adapter for one --
    # the model-call receipt, its OTel ``gen_ai.provider.name``, and the model-stream context --
    # which is the correct attribution for all three: those spans describe the call the *model*
    # served, and "gateway" is the transport it arrived over. ``ModelConfig.provider`` still
    # carries that transport string beside it.
    provider_name: str | None = "openai"

    # Forwards resolved media blocks in the by-value ``messages`` verbatim to the gateway.
    supports_multimodal: ClassVar[bool] = True
    def structured_output_support(self, config: ModelConfig | None = None) -> str:
        """This adapter *forwards*; it does not apply. So its claim is only as good as the
        proof it insists on, and that is exactly what ``on_unsupported`` controls.

        Under ``"fail"`` a returned turn is a proven turn: the echo checks refuse anything
        else, so a gateway chained in front of this adapter inherits real proof. Under
        ``"omit"`` the same adapter deliberately accepts an unproven turn -- and a static
        ``"native"`` would then let the outer hop mint a *fresh* positive echo out of a
        declaration, reporting proof for a call where the inner hop had none. A proof that
        survives a hop that admitted it was not proving is not a proof.

        A method, not a property, because the question takes an argument a property cannot
        carry: *which call*. Enforcement runs under the effective per-call config
        (``request.model or self.config``), so the claim must be answered from the same
        config -- a claim probed off the standing config alone let a shared adapter (a
        ``provider_adapter_factory`` that ignores its config parameter) mint proof for a call
        it enforces under a wire-supplied ``"omit"``, and withhold proof from a call it
        enforces under ``"fail"``. ``None`` falls back to the standing config, which is the
        right answer for a probe made outside any call.
        """

        effective = config or self.config
        return "native" if effective.generation.on_unsupported == "fail" else "none"

    def generation_support(self, config: ModelConfig | None = None) -> str:
        """The sampling twin of :meth:`structured_output_support`, same policy, same reason --
        one knob, one answer."""

        effective = config or self.config
        return "native" if effective.generation.on_unsupported == "fail" else "none"

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        config = request.model or self.config
        url = self._resolve_gateway_url(config)
        body = _encode_request_body(self._payload(request))
        retry = config.retry
        max_attempts = max(1, retry.max_attempts)
        last_error: ModelAdapterError | None = None
        attempt = 0
        try:
            for attempt in range(1, max_attempts + 1):
                if attempt > 1:
                    # Reported before the wait, not after it. This call may never produce an outcome
                    # anyone reads: a blocking ``next_turn`` runs on a thread the run abandons when
                    # it is cancelled or times out, the receipt is then built from the boundary the
                    # race raised, and whatever this worker eventually returns or raises is
                    # discarded. The channel is the only thing that crosses that abandonment -- and
                    # the backoff wait is a window the run can end inside, since the event loop stays
                    # free while this thread sleeps.
                    report_provider_retried()
                    # Waited here rather than at each retry site below. There were five of those and
                    # the schedule had to be repeated at every one; a rule written five times is one
                    # that eventually differs in one place. ``attempt - 1`` is the attempt that just
                    # failed, which is what the schedule is indexed by.
                    _sleep_before_retry(
                        attempt - 1,
                        retry.initial_delay_s,
                        retry.max_delay_s,
                        retry.backoff_multiplier,
                        retry.jitter_s,
                    )
                http_request = Request(
                    url,
                    data=body,
                    headers=self._headers(),
                    method="POST",
                )
                try:
                    with urlopen(http_request, timeout=config.timeout_s) as response:
                        response_body = response.read()
                    try:
                        data = loads_model_envelope_json_ingress(response_body.decode("utf-8"))
                    except ValueError as exc:
                        raise ModelAdapterError(
                            "LLM gateway returned invalid JSON",
                            provider_error_code=GATEWAY_BAD_RESPONSE,
                        ) from exc
                    # ``attempt > 1`` means this call only succeeded because *this client's* retry
                    # loop ran. The kernel counts it as one adapter call, so the receipt would
                    # otherwise show a twice-failed call as a clean single attempt.
                    #
                    # Combined with what the response already carries, never assigned over it: the
                    # gateway's own backend may have retried on a request this client got right the
                    # first time, and overwriting turned that into a clean attempt. Two independent
                    # retry loops sit on this path and either one having run is the fact a receipt
                    # records.
                    turn = _parse_gateway_response(data)
                    if attempt > 1:
                        turn = replace(turn, provider_retried=True)
                    # Stamped *before* the applied-parameter checks, exactly as the streaming
                    # twin stamps the chunk before checking it: those checks raise, and the
                    # error they raise carries the retry evidence. Reading the un-stamped turn
                    # here recorded a call this client retried as a clean single attempt on
                    # every not-applied failure -- the one path where the retry evidence has no
                    # other carrier, since no turn is returned.
                    #
                    # The refusal carries the turn's usage, because this is a failure that
                    # happens *after* a complete, billed answer: the provider generated it, we
                    # simply refuse to trust that our parameters shaped it. A receipt reporting
                    # zero tokens there drops the call out of the metrics and out of the
                    # cumulative token budget. Stamped around both checks rather than inside
                    # them, since the usage belongs to the turn, not to either proof.
                    try:
                        _check_generation_applied(
                            build_generation_payload(config.generation),
                            config.generation.on_unsupported,
                            data.get("generation_applied"),
                            known_provider_retried=turn.provider_retried,
                        )
                        _check_schema_applied(
                            request.output_schema is not None,
                            config.generation.on_unsupported,
                            data.get("schema_applied"),
                            known_provider_retried=turn.provider_retried,
                        )
                    except ModelAdapterError as unproven:
                        mark_provider_usage(unproven, turn.usage)
                        raise
                    return turn
                except ModelAdapterError as exc:
                    last_error = exc
                    if not _should_retry(exc, attempt, max_attempts, retry.retry_on):
                        raise
                except HTTPError as exc:
                    last_error = _error_from_http_error(exc)
                    if not _should_retry(last_error, attempt, max_attempts, retry.retry_on):
                        raise last_error from exc
                except URLError as exc:
                    last_error = ModelAdapterError(
                        f"LLM gateway request failed: {exc.reason}",
                        provider_error_code=GATEWAY_NETWORK_ERROR,
                        retryable=True,
                    )
                    if not _should_retry(last_error, attempt, max_attempts, retry.retry_on):
                        raise last_error from exc
                except TimeoutError as exc:
                    last_error = ModelAdapterError(
                        "LLM gateway request timed out",
                        provider_error_code=GATEWAY_TIMEOUT,
                        retryable=True,
                    )
                    if not _should_retry(last_error, attempt, max_attempts, retry.retry_on):
                        raise last_error from exc
                except OSError as exc:
                    # A bare connection-level error (reset / aborted / broken pipe), e.g. raised
                    # mid-read after urlopen() returned, is transient and retryable like a
                    # URLError. URLError/TimeoutError (both OSError subclasses) are handled above,
                    # so this catches only the raw connection failures they miss.
                    last_error = ModelAdapterError(
                        f"LLM gateway connection error: {exc}",
                        provider_error_code=GATEWAY_NETWORK_ERROR,
                        retryable=True,
                    )
                    if not _should_retry(last_error, attempt, max_attempts, retry.retry_on):
                        raise last_error from exc
            if last_error is not None:
                raise last_error
            raise ModelAdapterError(
                "LLM gateway request failed", provider_error_code=GATEWAY_NETWORK_ERROR
            )
        # Marked in one place rather than at each ``raise`` inside the loop: there are five
        # of those plus the exhausted-budget one, and a scheme needing every site updated is
        # one that eventually misses a site. ``attempt`` holds whichever attempt was in flight.
        except Exception as exc:
            # Any escaping type, not just ModelAdapterError: an attempt can be retried and the
            # final one still fail on something else entirely -- a body that is not valid UTF-8
            # raises UnicodeDecodeError at the decode step -- and a failure receipt that denies the
            # retry is wrong regardless of which exception carried it.
            _stamp_retry(exc, attempt)
            raise

    async def astream_turn(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:
        """Stream a turn from the gateway's SSE endpoint, yielding ``ModelStreamChunk``.

        Opt-in: requires ``httpx`` (the ``[http-async]`` extra); the sync ``next_turn`` stays
        on stdlib ``urllib``. Retries only the initial connect/non-200 status (before the
        stream is committed) using the same ``ModelConfig.retry`` policy as ``next_turn``;
        once the 200 stream is flowing, any error is terminal (no partial-stream replay).
        """
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ModelAdapterError(
                "httpx is required for gateway streaming; install monoid-agent-kernel[http-async]",
                provider_error_code=GATEWAY_NETWORK_ERROR,
            ) from exc

        config = request.model or self.config
        url = self._resolve_gateway_url(config).rstrip("/") + "/stream"
        body = _encode_request_body(self._payload(request))
        retry = config.retry
        max_attempts = max(1, retry.max_attempts)
        last_error: ModelAdapterError | None = None
        attempt = 0
        # Bound before the client exists, because the client's own lifecycle can fail: `__aexit__`
        # raises after the loop, where a loop-local would still be unbound if no attempt reached it.
        committed = False
        try:
            # One client for the whole call, not one per attempt. Constructing it is synchronous and
            # not cheap -- measured at ~285ms warm here, and the event loop is unavailable for all of
            # it. Inside the loop that cost was paid again on every retry, and the run's cancel and
            # deadline race lives on the blocked loop, so a run told to stop kept holding the provider
            # past its own boundary -- the same defect the backoff wait had, at the next statement.
            # Hoisting also lets retries reuse the connection pool instead of opening a fresh one.
            async with httpx.AsyncClient(timeout=config.timeout_s) as client:
                for attempt in range(1, max_attempts + 1):
                    if attempt > 1:
                        # Same report the sync loop makes, for the same reason: a stream cancelled
                        # before this attempt commits leaves the chunk below undelivered too.
                        report_provider_retried()
                        # Before the attempt, not after it commits. The retry is already certain here
                        # -- the previous iteration decided it -- and this line always runs, while a
                        # commit may never happen: a run cancelled or timed out while attempt 2 was
                        # connecting produced no chunk at all, and the receipt is built from the
                        # ``RunCancelled``/``RunTimeout`` the race raises, not from anything the
                        # adapter can stamp. Both carriers missed and a retried call was recorded as a
                        # clean single attempt.
                        #
                        # An earlier fix put this at commit, calling that "the first moment the retry
                        # is certain". That was wrong: certainty arrives when ``_should_retry`` says
                        # yes, at the end of the previous iteration, and nothing between there and
                        # here can revoke it.
                        #
                        # An empty ``TextDelta`` concatenates to nothing, so the assembled turn is
                        # unchanged. A *live stream* is not: ``QueueEventSink.push_delta`` relays
                        # every chunk, so a caller of ``AgentLoop.astream`` sees one extra empty
                        # text chunk per retry. The event-emitting consumer filters on
                        # ``chunk.text`` and sees nothing new.
                        yield TextDelta(text="", provider_retried=True)
                        # Awaited, and after both reports. The blocking sleep this replaces held the
                        # event loop for the whole backoff, so nothing else in the run progressed and
                        # the run's own cancel/deadline race -- which lives on that loop -- could not
                        # fire: a run told to stop kept waiting for a provider it had given up on.
                        # Now that the wait yields, it is also a window the run can end inside, which
                        # is why the evidence above goes out first.
                        await _asleep_before_retry(
                            attempt - 1,
                            retry.initial_delay_s,
                            retry.max_delay_s,
                            retry.backoff_multiplier,
                            retry.jitter_s,
                        )
                    committed = False  # reset per attempt; see the binding above the loop
                    saw_terminal = False
                    # Resolved per attempt, like the sync loop resolves it. Hoisted out of the loop
                    # it was the one thing a retry did not refresh: ``token_provider`` re-mints near
                    # expiry (see the field), and a backoff is exactly where a token crosses that
                    # line -- the wait is up to ``max_delay_s`` long and the run may already be
                    # minutes old. A stale header then failed attempt 2 with a 401, which is
                    # ``gateway_auth_error`` and *not* retryable, so a run the sync path recovered
                    # ended terminally here. The URL and the body are hoisted because neither can
                    # change between attempts; a credential can.
                    headers = self._headers()
                    try:
                        async with client.stream(
                            "POST", url, headers=headers, content=body
                        ) as response:
                            if response.status_code != 200:
                                detail = (await response.aread()).decode("utf-8", errors="replace")
                                error = _error_from_status_body(response.status_code, detail)
                                if _should_retry(error, attempt, max_attempts, retry.retry_on):
                                    raise _StreamRetry(error)
                                raise error
                            committed = True
                            async for chunk in _aiter_sse_chunks(response):
                                # Also on each chunk, so a chunk forwarded on its own still says
                                # which stream it came from. Same ``attempt`` in the same scope as
                                # the marker above, so the two cannot disagree.
                                if attempt > 1:
                                    chunk = replace(chunk, provider_retried=True)
                                # The terminal frame is the streaming twin of the sync check in
                                # ``next_turn`` -- both transports enforce or neither does.
                                if isinstance(chunk, TurnComplete):
                                    saw_terminal = True
                                    # The streamed twin of the sync stamp: the terminal frame
                                    # is refused before it is yielded, so its usage reaches
                                    # nothing that assembles a turn -- the refusal is the only
                                    # carrier left for a call the provider already billed.
                                    try:
                                        _check_generation_applied(
                                            build_generation_payload(config.generation),
                                            config.generation.on_unsupported,
                                            chunk.generation_applied,
                                            known_provider_retried=chunk.provider_retried,
                                        )
                                        _check_schema_applied(
                                            request.output_schema is not None,
                                            config.generation.on_unsupported,
                                            chunk.schema_applied,
                                            known_provider_retried=chunk.provider_retried,
                                        )
                                    except ModelAdapterError as unproven:
                                        mark_provider_usage(unproven, chunk.usage)
                                        raise
                                yield chunk
                        if not saw_terminal:
                            # "Both transports enforce or neither does" has to include the
                            # stream that never sends the frame the checks above live on: a
                            # body that ends cleanly after its last delta is assembled into a
                            # normal turn (``assemble_streamed_turn`` synthesizes
                            # ``stop_reason="stop"``), so without this the one server the
                            # fail-closed policy exists to catch -- an older gateway that
                            # ignores the new request keys, terminal frame included -- was
                            # accepted on this transport and refused on the sync twin. Absent
                            # frame = absent echo; the shared checks already encode that case,
                            # so run them with nothing once the drain is complete. Traffic
                            # that configures neither knob keeps the pre-W5 tolerance for a
                            # frameless stream: both checks pass when nothing was requested.
                            _check_generation_applied(
                                build_generation_payload(config.generation),
                                config.generation.on_unsupported,
                                None,
                                known_provider_retried=attempt > 1,
                            )
                            _check_schema_applied(
                                request.output_schema is not None,
                                config.generation.on_unsupported,
                                None,
                                known_provider_retried=attempt > 1,
                            )
                        return
                    except _StreamRetry as retry_signal:
                        last_error = retry_signal.error
                    except httpx.HTTPError as exc:
                        if committed:
                            # The stream already started; replaying would duplicate deltas. Terminal.
                            raise ModelAdapterError(
                                f"LLM gateway stream interrupted: {exc}",
                                provider_error_code=GATEWAY_NETWORK_ERROR,
                                retryable=False,
                            ) from exc
                        error = ModelAdapterError(
                            f"LLM gateway stream connection error: {exc}",
                            provider_error_code=GATEWAY_NETWORK_ERROR,
                            retryable=True,
                        )
                        if not _should_retry(error, attempt, max_attempts, retry.retry_on):
                            raise error from exc
                        last_error = error
            if last_error is not None:
                raise last_error
            raise ModelAdapterError(
                "LLM gateway stream failed", provider_error_code=GATEWAY_NETWORK_ERROR
            )
        except httpx.HTTPError as exc:
            # The client's own lifecycle -- construction, `__aenter__`, and the `__aexit__` that
            # tears the pool down -- sits *outside* the per-attempt handler now that the client is
            # hoisted out of the retry loop. Before the hoist it was inside, so those failures were
            # classified like any other transport error; afterwards they escaped as raw `httpx`
            # exceptions and only the catch-all below saw them.
            #
            # Unclassified is not merely less descriptive. `AgentLoop._recoverable_turn_error` keys
            # off `retryable` and a 4xx `http_status`, both of which a raw `httpx` error lacks, so a
            # failure that used to end one turn -- recoverably, with the session alive and the turn
            # re-attemptable -- terminalized the whole run instead, wrote `failure.json`, and made no
            # retry at all. `httpx.CloseError` from a pool teardown is an ordinary way to reach this.
            #
            # `committed` draws the same line the in-loop handler draws: once deltas have gone out,
            # replaying would duplicate them, so a late failure is terminal rather than retryable.
            error = ModelAdapterError(
                f"LLM gateway stream interrupted: {exc}"
                if committed
                else f"LLM gateway stream connection error: {exc}",
                provider_error_code=GATEWAY_NETWORK_ERROR,
                retryable=not committed,
            )
            _stamp_retry(error, attempt)
            raise error from exc
        # Same marking as the sync path. Stream retries are all pre-commit, so an error
        # escaping after the first attempt means the stream really was retried.
        except Exception as exc:
            # Any escaping type, not just ModelAdapterError: an attempt can be retried and the
            # final one still fail on something else entirely -- a body that is not valid UTF-8
            # raises UnicodeDecodeError at the decode step -- and a failure receipt that denies the
            # retry is wrong regardless of which exception carried it.
            _stamp_retry(exc, attempt)
            raise

    def resolve_destination(self, config: ModelConfig) -> str:
        """Where a call under ``config`` would go. See ``AddressedModelAdapter``.

        Delegates to the same resolution the request itself uses, so a replay key and the actual
        request can never disagree about the destination.
        """

        return self._resolve_gateway_url(config)

    def _resolve_gateway_url(self, config: ModelConfig) -> str:
        url = (
            self.gateway_url
            or config.gateway_url
            or self.config.gateway_url
            or getenv(DEFAULT_GATEWAY_URL_ENV)
        )
        if not url:
            raise ModelAdapterError(
                f"LLM gateway URL is required via --llm-gateway-url or {env_name_for_error(DEFAULT_GATEWAY_URL_ENV)}"
            )
        return url

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": user_agent(),
        }
        token = self._resolve_gateway_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _resolve_gateway_token(self) -> str | None:
        if self.token_provider is not None:
            return self.token_provider()
        if self.token is not None:
            return self.token
        if self.token_file is not None:
            return self.token_file.read_text(encoding="utf-8").strip()
        return getenv(self.token_env)

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        config = request.model or self.config
        payload: dict[str, Any] = {
            "protocol": namespaced_id("llm-turn.v1"),
            "model": config.model,
            "system_prompt": request.system_prompt,
            "tools": [_gateway_tool_schema(tool) for tool in request.tools],
        }
        # Both blocks carry their off-default ``on_unsupported``, for one reason: the server
        # rebuilds a config object from this wire block, and a field left off is not "unset"
        # there -- it is the *default*, so a caller's "omit" silently became "fail" on the
        # server's copy. That matters as soon as a gateway's upstream is another gateway: the
        # next hop enforces the reset policy and rejects a turn the caller asked to accept
        # best-effort. Emitted only off-default, so default configs keep their exact wire shape.
        # ``build_*_payload`` deliberately does not carry policy -- those dicts are also the
        # applied-echo comparison, which is about provider knobs only.
        reasoning_payload = build_reasoning_payload(config.reasoning)
        if config.reasoning.on_unsupported != "fail":
            reasoning_payload["on_unsupported"] = config.reasoning.on_unsupported
        if config.reasoning.effort == "default":
            # ``effort`` is the one reasoning field whose omission sentinel ("default") is not
            # the codec's reconstruction default ("medium"): ``build_reasoning_payload`` leaves
            # "default" off because a *provider* payload means "no effort key", but this block
            # is a *config* the server rebuilds with ``ReasoningConfig.from_json``, where a
            # missing effort reads back as "medium" -- so a client asking for the provider
            # default silently got medium reasoning, only through a gateway. Carried
            # explicitly, like the policy above; every other effort value already rides.
            reasoning_payload["effort"] = "default"
        if reasoning_payload:
            payload["reasoning"] = reasoning_payload
        generation_payload = build_generation_payload(config.generation)
        if config.generation.on_unsupported != "fail":
            generation_payload["on_unsupported"] = config.generation.on_unsupported
        if generation_payload:
            payload["generation"] = generation_payload
        if request.output_schema is not None:
            payload["output_schema"] = request.output_schema

        if request.messages is not None:
            # By-value: the full conversation travels as messages; no continuation handle.
            # A text-only adapter projects any multimodal (list) content down to text so the
            # gateway never receives parts it cannot forward; a multimodal adapter passes the
            # resolved blocks through verbatim.
            if getattr(self, "supports_multimodal", False):
                payload["messages"] = list(request.messages)
            else:
                payload["messages"] = [project_message_to_text(m) for m in request.messages]
        elif request.previous_turn_handle:
            payload["previous_turn_handle"] = request.previous_turn_handle
            payload["observations"] = [
                {
                    "call_id": observation.call_id,
                    "tool_name": observation.tool_name,
                    "output": observation.output,
                    "is_background": observation.is_background,
                }
                for observation in request.observations
            ]
            # Third shape: a new user message delivered on top of an existing
            # continuation handle (user follow-up). observations is typically empty here.
            if request.instruction:
                payload["instruction"] = request.instruction
        else:
            payload["instruction"] = request.instruction or ""
        return payload


def _gateway_tool_schema(tool: ToolSpec) -> dict[str, Any]:
    return {
        "id": tool.id,
        "name": tool.exported_name,
        "description": tool.description,
        "input_schema": tool.input_schema,
        "capability": tool.capability,
        "side_effect": tool.side_effect,
    }


def _exact_gateway_bool(
    payload: dict[str, Any],
    key: str,
    *,
    default: bool,
    context: str,
    http_status: int | None = None,
    known_provider_retried: bool = False,
) -> bool:
    """Read a gateway control boolean without truthiness coercion.

    Gateway payloads are wire data, so a present control has one portable meaning only when it is
    an exact JSON boolean.  In particular, ``"false"`` must not become true and authorize a retry
    or fabricate evidence that the upstream provider already retried.
    """

    if key not in payload:
        return default
    value = payload[key]
    if type(value) is bool:
        return value
    raise ModelAdapterError(
        f"LLM gateway returned an invalid {context}: {key} must be a boolean",
        provider_error_code=GATEWAY_BAD_RESPONSE,
        retryable=False,
        http_status=http_status,
        provider_retried=known_provider_retried,
    )


def _exact_gateway_int(
    payload: dict[str, Any],
    key: str,
    *,
    default: int | None,
    context: str,
    minimum: int,
    maximum: int | None = None,
    allow_none: bool = False,
    http_status: int | None = None,
    known_provider_retried: bool = False,
) -> int | None:
    if key not in payload:
        return default
    value = payload[key]
    if value is None and allow_none:
        return None
    valid = type(value) is int and value >= minimum
    if maximum is not None:
        valid = valid and value <= maximum
    if valid:
        return value
    requirement = f"an integer >= {minimum}"
    if maximum is not None:
        requirement = f"an integer from {minimum} through {maximum}"
    raise ModelAdapterError(
        f"LLM gateway returned an invalid {context}: {key} must be {requirement}",
        provider_error_code=GATEWAY_BAD_RESPONSE,
        retryable=False,
        http_status=http_status,
        provider_retried=known_provider_retried,
    )


def _gateway_http_status_hint(payload: dict[str, Any]) -> int | None:
    """Return an already trustworthy status for enriching a later validation error."""

    value = payload.get("http_status")
    if type(value) is int and 100 <= value <= 599:
        return value
    return None


def _gateway_string(
    payload: dict[str, Any],
    *keys: str,
    context: str,
    required: bool = False,
    known_provider_retried: bool = False,
    http_status: int | None = None,
) -> str | None:
    """Read the first present gateway string field without identity coercion."""

    for key in keys:
        if key not in payload or payload[key] is None:
            continue
        value = payload[key]
        if type(value) is not str:
            raise ModelAdapterError(
                f"LLM gateway returned an invalid {context}: {key} must be a string",
                provider_error_code=GATEWAY_BAD_RESPONSE,
                retryable=False,
                http_status=http_status,
                provider_retried=known_provider_retried,
            )
        if value:
            return normalize_unicode_scalars(value)
    if required:
        names = " or ".join(keys)
        raise ModelAdapterError(
            f"LLM gateway returned an invalid {context}: {names} is required",
            provider_error_code=GATEWAY_BAD_RESPONSE,
            retryable=False,
            http_status=http_status,
            provider_retried=known_provider_retried,
        )
    return None


def _gateway_fragment_string(
    payload: dict[str, Any],
    key: str,
    *,
    context: str,
    http_status: int | None = None,
    known_provider_retried: bool,
) -> str | None:
    """Validate a model content fragment while deferring cross-frame Unicode repair."""

    if key not in payload or payload[key] is None:
        return None
    value = payload[key]
    if type(value) is str:
        return value
    raise ModelAdapterError(
        f"LLM gateway returned an invalid {context}: {key} must be a string",
        provider_error_code=GATEWAY_BAD_RESPONSE,
        retryable=False,
        http_status=http_status,
        provider_retried=known_provider_retried,
    )


def _gateway_usage(
    value: Any,
    *,
    context: str,
    http_status: int | None = None,
    known_provider_retried: bool = False,
) -> dict[str, int]:
    try:
        return normalize_usage(value)
    except (TypeError, ValueError) as exc:
        raise ModelAdapterError(
            f"LLM gateway returned an invalid {context}: usage must contain token counts",
            provider_error_code=GATEWAY_BAD_RESPONSE,
            retryable=False,
            http_status=http_status,
            provider_retried=known_provider_retried,
        ) from exc


def _portable_gateway_payload(
    value: Any,
    *,
    context: str,
    http_status: int | None = None,
    known_provider_retried: bool = False,
) -> Any:
    try:
        return normalize_json_ingress(value)
    except ValueError as exc:
        raise ModelAdapterError(
            f"LLM gateway returned an invalid {context}",
            provider_error_code=GATEWAY_BAD_RESPONSE,
            retryable=False,
            http_status=http_status,
            provider_retried=known_provider_retried,
        ) from exc


def _gateway_reasoning_items(
    value: Any,
    *,
    context: str,
    http_status: int | None = None,
    known_provider_retried: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Shape-check the provider-native reasoning artifacts one response carries.

    Absent or ``None`` reads as "no artifacts" -- an older gateway that never mentions the key
    and an upstream that produced none say the same thing to a client, and both leave the next
    turn with nothing to replay, which is the neutral behavior the loop already has.

    Anything present must be a list of objects, because that is the only shape the replay path
    can hand back to a provider: the items travel into the by-value ``messages`` log and out
    again to the upstream adapter verbatim, so a scalar or a half-list would be discovered by
    the *provider*, one hop and one turn later, as an unclassifiable request. Contents are not
    inspected past that -- they are opaque and provider-encrypted by construction.

    A tuple is accepted beside a list for the same reason ``tool_calls`` accepts one two dozen
    lines below: JSON only ever produces a list, but this reader also serves in-process Python
    callers, and a sequence refused by one array-valued key of a body while its neighbour
    accepts it is a difference with no rule behind it.

    Written once for both transports, like :func:`_validated_generation_echo` beside it: the
    sync response and the streamed terminal frame read the same key out of different envelopes,
    and a reader that is stricter than its twin is the shape this file keeps producing.
    """

    if value is None:
        return ()
    if isinstance(value, (list, tuple)) and all(isinstance(item, dict) for item in value):
        return tuple(dict(item) for item in value)
    raise ModelAdapterError(
        f"LLM gateway returned invalid {context} reasoning: expected an array of objects",
        provider_error_code=GATEWAY_BAD_RESPONSE,
        retryable=False,
        http_status=http_status,
        provider_retried=known_provider_retried,
    )


def _validated_generation_echo(
    applied: Any, *, provider_retried: bool = False
) -> dict[str, Any] | None:
    """Shape-check one ``generation_applied`` echo. One rule, both transports.

    Wire shape is not a policy question: a server that answers with a non-object here is
    malformed whatever ``on_unsupported`` says and whatever this call requested, so this is
    ``gateway_bad_response`` and it fires before any policy branch. Written once because the
    sync response and the streamed terminal frame read the same key out of different
    envelopes -- the streamed side used to reject a malformed echo that the sync side accepted
    under ``"omit"``.
    """

    if applied is None or isinstance(applied, dict):
        return applied
    raise ModelAdapterError(
        "LLM gateway returned an invalid generation_applied echo: expected an object",
        provider_error_code=GATEWAY_BAD_RESPONSE,
        retryable=False,
        provider_retried=provider_retried,
    )


def _validated_schema_echo(applied: Any, *, provider_retried: bool = False) -> bool | None:
    """The ``schema_applied`` twin of :func:`_validated_generation_echo`, same rule."""

    if applied is None or isinstance(applied, bool):
        return applied
    raise ModelAdapterError(
        "LLM gateway returned an invalid schema_applied echo: expected a boolean",
        provider_error_code=GATEWAY_BAD_RESPONSE,
        retryable=False,
        provider_retried=provider_retried,
    )


def _same_echoed_value(echoed: Any, requested: Any) -> bool:
    """One requested knob against its echo, without Python's cross-type numeric equality.

    ``==`` is not a proof test on a wire. Python compares ``True == 1`` and ``False == 0.0``,
    so a server answering JSON booleans proved exactly the most ordinary settings this block
    carries: ``max_output_tokens=1``, ``top_p=1``, ``temperature=0``. Every other read of this
    wire already refuses that coercion (``_exact_gateway_bool``, ``_exact_gateway_int``, and
    ``is_finite_json_number`` itself, which rejects ``bool`` and its subclasses); the proof
    comparison was the one place it did not.

    A number is proven only by a number -- but by a number of *either* JSON spelling: JSON has
    one numeric type, so a gateway that is not Python re-serializes ``1.0`` as ``1``, and
    demanding an exact Python type would invent a false refusal for every non-Python server.
    Anything else must match by type and value, which is what a non-numeric future knob needs.
    """

    if is_finite_json_number(requested):
        return is_finite_json_number(echoed) and echoed == requested
    return type(echoed) is type(requested) and echoed == requested


def _generation_echo_matches(applied: Any, requested: dict[str, Any]) -> bool:
    """Whether the echo is the block this client sent -- same keys, same values, no coercion."""

    if not isinstance(applied, dict) or set(applied) != set(requested):
        return False
    return all(_same_echoed_value(applied[key], value) for key, value in requested.items())


def _check_generation_applied(
    requested: dict[str, Any],
    on_unsupported: str,
    applied: Any,
    *,
    known_provider_retried: bool = False,
) -> None:
    """Refuse a turn whose sampling parameters cannot be proven applied (scope §5 D-a).

    ``requested`` is the exact wire block this client sent; the server echoes the block it
    forwarded upstream as ``generation_applied``. Matching it -- key for key, without numeric
    coercion (see :func:`_same_echoed_value`) -- is the proof. An absent echo is
    indistinguishable from an older server that silently discarded the block -- precisely the
    deployment ``"fail"`` exists to catch -- so under the default policy it is an error, and
    ``"omit"`` is the documented way to accept a best-effort transport.
    """

    applied = _validated_generation_echo(applied, provider_retried=known_provider_retried)
    if not requested:
        return
    if _generation_echo_matches(applied, requested):
        return
    if on_unsupported == "omit":
        return
    detail = (
        "the gateway sent no generation_applied echo (older gateway?)"
        if applied is None
        else "the generation_applied echo does not match the requested parameters"
    )
    raise ModelAdapterError(
        f"LLM gateway did not apply the requested generation parameters: {detail}; "
        'set model.generation.on_unsupported="omit" to accept best-effort transport',
        provider_error_code=GATEWAY_GENERATION_NOT_APPLIED,
        retryable=False,
        # The remedy in the message is configuration: the session survives, the user
        # switches policy or transport and resends. Without this the run terminalized on a
        # condition the same server would have reported recoverably as an HTTP 400.
        config_recoverable=True,
        provider_retried=known_provider_retried,
    )


def _check_schema_applied(
    schema_sent: bool,
    on_unsupported: str,
    applied: Any,
    *,
    known_provider_retried: bool = False,
) -> None:
    """The schema twin of :func:`_check_generation_applied`, same policy knob.

    One ``on_unsupported`` governs both proofs deliberately: "how to treat a parameter the
    transport cannot prove was applied" is one question, and two half-set knobs (fail for one,
    omit for the other) would be a new surface for exactly the kind of asymmetry W5 exists to
    close. ``applied`` is a tri-state: ``True`` proves it, ``False`` is the server saying its
    upstream cannot enforce schemas, absent (``None``) is an older server.
    """

    applied = _validated_schema_echo(applied, provider_retried=known_provider_retried)
    if not schema_sent:
        return
    if applied is True:
        return
    if on_unsupported == "omit":
        return
    detail = (
        "the gateway sent no schema_applied echo (older gateway?)"
        if applied is None
        else "the gateway's upstream does not enforce output schemas"
    )
    raise ModelAdapterError(
        f"LLM gateway did not apply the requested output schema: {detail}; "
        'set model.generation.on_unsupported="omit" to accept best-effort transport',
        provider_error_code=GATEWAY_SCHEMA_NOT_APPLIED,
        retryable=False,
        config_recoverable=True,
        provider_retried=known_provider_retried,
    )


def _parse_gateway_response(data: Any) -> ModelTurn:
    if not isinstance(data, dict):
        raise ModelAdapterError(
            "LLM gateway returned a non-object JSON response",
            provider_error_code=GATEWAY_BAD_RESPONSE,
            retryable=False,
        )
    # The best status this reader knows before any field of the payload has been validated: an
    # already-trustworthy ``http_status`` off an error envelope, and nothing at all otherwise.
    # Every validator below is handed it, so a malformed payload raises a classified failure that
    # still names the status the server reported rather than an unclassifiable one.
    status_hint = _gateway_http_status_hint(data) if "error" in data else None
    provider_retried = _exact_gateway_bool(
        data,
        "provider_retried",
        default=False,
        context="response",
        http_status=status_hint,
    )
    if "error" in data:
        error_http_status = _exact_gateway_int(
            data,
            "http_status",
            default=None,
            context="error response",
            minimum=100,
            maximum=599,
            allow_none=True,
            http_status=status_hint,
            known_provider_retried=provider_retried,
        )
        retryable = _exact_gateway_bool(
            data,
            "retryable",
            default=False,
            context="error response",
            http_status=error_http_status,
            known_provider_retried=provider_retried,
        )
        # The remedy the failure names, read back rather than inferred from the status. Absent
        # reads as False -- an older gateway that never mentions the key, and a failure that
        # really is not config-fixable, mean the same thing to a driver.
        config_recoverable = _exact_gateway_bool(
            data,
            "config_recoverable",
            default=False,
            context="error response",
            http_status=error_http_status,
            known_provider_retried=provider_retried,
        )
        envelope_error = ModelAdapterError(
            _gateway_string(
                data,
                "error",
                context="error response",
                required=True,
                known_provider_retried=provider_retried,
                http_status=error_http_status,
            )
            or "",
            provider_error_code=(
                _gateway_string(
                    data,
                    "error_code",
                    context="error response",
                    known_provider_retried=provider_retried,
                    http_status=error_http_status,
                )
                or GATEWAY_BAD_RESPONSE
            ),
            retryable=retryable,
            config_recoverable=config_recoverable,
            http_status=error_http_status,
            provider_retried=provider_retried,
        )
        # Third error reader on this wire, and the rule is the same on all three: a failure
        # that reports what it cost is recorded as having cost it.
        mark_provider_usage(envelope_error, _reported_error_usage(data))
        raise envelope_error
    _exact_gateway_bool(
        data,
        "retryable",
        default=False,
        context="response",
        known_provider_retried=provider_retried,
    )
    raw_calls = data.get("tool_calls", ())
    if raw_calls is None:
        raw_calls = ()
    if not isinstance(raw_calls, (list, tuple)):
        raise ModelAdapterError(
            "LLM gateway returned invalid tool_calls: expected an array",
            provider_error_code=GATEWAY_BAD_RESPONSE,
            provider_retried=provider_retried,
        )
    tool_calls: list[ToolCall] = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            raise ModelAdapterError(
                "LLM gateway returned an invalid tool call",
                provider_error_code=GATEWAY_BAD_RESPONSE,
                provider_retried=provider_retried,
            )
        args = raw.get("arguments")
        if args is None:
            args = {}
        if isinstance(args, str):
            try:
                args = loads_model_json_ingress(args)
            except ValueError as exc:
                raise ModelAdapterError(
                    f"invalid gateway tool call arguments for {raw.get('name')}",
                    provider_error_code=GATEWAY_BAD_RESPONSE,
                    provider_retried=provider_retried,
                ) from exc
        else:
            args = _portable_gateway_payload(
                args,
                context="tool call arguments",
                http_status=status_hint,
                known_provider_retried=provider_retried,
            )
        if not isinstance(args, dict):
            raise ModelAdapterError(
                f"invalid gateway tool call arguments for {raw.get('name')}",
                provider_error_code=GATEWAY_BAD_RESPONSE,
                provider_retried=provider_retried,
            )
        tool_calls.append(
            ToolCall(
                id=_gateway_string(
                    raw,
                    "id",
                    "call_id",
                    context="tool call",
                    required=True,
                    known_provider_retried=provider_retried,
                )
                or "",
                name=_gateway_string(
                    raw,
                    "name",
                    context="tool call",
                    required=True,
                    known_provider_retried=provider_retried,
                )
                or "",
                arguments=args,
            )
        )

    # stop_reason rides the gateway wire (added by the gateway server). Older gateways omit it;
    # infer the common cases so the loop's branch still works.
    stop_reason = _gateway_string(
        data,
        "stop_reason",
        context="response",
        known_provider_retried=provider_retried,
    )
    if stop_reason is None:
        stop_reason = "tool_calls" if tool_calls else "stop"
    return ModelTurn(
        response_id=_gateway_string(
            data,
            "response_id",
            "turn_handle",
            context="response",
            known_provider_retried=provider_retried,
        ),
        final_text=_gateway_string(
            data,
            "final_text",
            context="response",
            known_provider_retried=provider_retried,
        ),
        tool_calls=tuple(tool_calls),
        usage=_gateway_usage(
            data.get("usage"),
            context="response",
            http_status=status_hint,
            known_provider_retried=provider_retried,
        ),
        raw=_portable_gateway_payload(
            data,
            context="response",
            http_status=status_hint,
            known_provider_retried=provider_retried,
        ),
        # The opaque provider-native reasoning artifacts the upstream produced, relayed by the
        # gateway. Absent from an older gateway, which reads as "none" -- the same thing an
        # adapter with no reasoning says, and all a wire that never mentions the key can mean.
        reasoning=_gateway_reasoning_items(
            data.get("reasoning"),
            context="response",
            http_status=status_hint,
            known_provider_retried=provider_retried,
        ),
        stop_reason=stop_reason,
        # A retry the gateway's own backend made. Absent from an older gateway, which reads as
        # "did not retry" -- the same default an adapter with no retry loop carries, and the only
        # thing a wire that never mentions it can honestly mean.
        provider_retried=provider_retried,
    )


class _StreamRetry(Exception):
    """Internal signal: a pre-stream (non-200) failure that the retry loop should retry."""

    def __init__(self, error: ModelAdapterError) -> None:
        self.error = error


async def _aiter_sse_chunks(response: Any) -> AsyncIterator[ModelStreamChunk]:
    """Parse the gateway's ``text/event-stream`` body into ``ModelStreamChunk``s.

    Minimal SSE: ``data:`` lines accumulate, a blank line dispatches one JSON frame, ``:``
    comment lines (keepalives) are ignored, and a trailing frame without a terminating blank
    line is still dispatched. An ``error`` frame raises ``ModelAdapterError``.
    """
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data_lines:
                chunk = _decode_sse_chunk(data_lines)
                data_lines = []
                if chunk is not None:
                    yield chunk
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
    if data_lines:
        chunk = _decode_sse_chunk(data_lines)
        if chunk is not None:
            yield chunk


def _decode_sse_chunk(data_lines: list[str]) -> ModelStreamChunk | None:
    try:
        event = loads_model_stream_envelope_json_ingress("\n".join(data_lines))
    except ValueError as exc:
        raise ModelAdapterError(
            "LLM gateway stream returned invalid JSON",
            provider_error_code=GATEWAY_BAD_RESPONSE,
        ) from exc
    if not isinstance(event, dict):
        raise ModelAdapterError(
            "LLM gateway stream returned a non-object frame",
            provider_error_code=GATEWAY_BAD_RESPONSE,
        )
    return _chunk_from_event(event)


def _chunk_from_event(event: dict[str, Any]) -> ModelStreamChunk | None:
    # A retry the gateway's own backend made, as opposed to one this client's loop made. Read off
    # every frame that carries it, because a stream cancelled mid-flight never delivers the
    # terminal one. Absent reads as "did not retry", which is what a wire that never mentions it
    # can honestly mean; the client's own ``attempt > 1`` is combined with this, never over it.
    raw_event_type = event.get("type")
    status_hint = (
        _gateway_http_status_hint(event)
        if type(raw_event_type) is str and raw_event_type == "error"
        else None
    )
    retried = _exact_gateway_bool(
        event,
        "provider_retried",
        default=False,
        context="stream frame",
        http_status=status_hint,
    )
    event_type = _gateway_string(
        event,
        "type",
        context="stream frame",
        known_provider_retried=retried,
    )
    if event_type == "text_delta":
        return TextDelta(
            text=_gateway_fragment_string(
                event,
                "text",
                context="text delta",
                http_status=status_hint,
                known_provider_retried=retried,
            )
            or "",
            provider_retried=retried,
        )
    if event_type == "reasoning_delta":
        return ReasoningDelta(
            text=_gateway_fragment_string(
                event,
                "text",
                context="reasoning delta",
                http_status=status_hint,
                known_provider_retried=retried,
            )
            or "",
            provider_retried=retried,
        )
    if event_type == "tool_call_delta":
        return ToolCallDelta(
            index=_exact_gateway_int(
                event,
                "index",
                default=0,
                context="tool-call delta",
                minimum=0,
                http_status=status_hint,
                known_provider_retried=retried,
            ),
            arguments_fragment=(
                _gateway_fragment_string(
                    event,
                    "arguments_fragment",
                    context="tool-call delta",
                    http_status=status_hint,
                    known_provider_retried=retried,
                )
                or ""
            ),
            id=_gateway_string(
                event,
                "id",
                context="tool-call delta",
                known_provider_retried=retried,
            ),
            name=_gateway_string(
                event,
                "name",
                context="tool-call delta",
                known_provider_retried=retried,
            ),
            provider_retried=retried,
        )
    if event_type == "turn_complete":
        # Same shape rule the sync response is held to; the enforcement functions call these
        # too, so neither transport can be stricter than the other.
        try:
            applied = _validated_generation_echo(
                event.get("generation_applied"), provider_retried=retried
            )
            schema_applied = _validated_schema_echo(
                event.get("schema_applied"), provider_retried=retried
            )
        except ModelAdapterError as malformed:
            # A malformed echo on a *billed* frame still cost the tokens the same frame
            # reports. The sync twin validates inside the stamped check block, so its
            # ``gateway_bad_response`` carries ``provider_usage``; raising here at parse
            # time, before any stamp, lost the same money on one of two transports. Read
            # leniently -- a second malformation in ``usage`` must not replace the failure
            # being reported.
            mark_provider_usage(malformed, _reported_error_usage(event))
            raise
        # The gateway's opaque turn_handle is the continuation handle the core stores.
        return TurnComplete(
            generation_applied=applied,
            schema_applied=schema_applied,
            response_id=_gateway_string(
                event,
                "turn_handle",
                "response_id",
                context="turn-complete frame",
                known_provider_retried=retried,
            ),
            usage=_gateway_usage(
                event.get("usage"),
                context="turn-complete frame",
                http_status=status_hint,
                known_provider_retried=retried,
            ),
            # The terminal frame is the only frame that may carry the artifacts, and the only
            # one ``assemble_streamed_turn`` reads them off. Same validator as the sync reader:
            # a shape one transport accepts and the other refuses is the defect, not the fix.
            reasoning=_gateway_reasoning_items(
                event.get("reasoning"),
                context="turn-complete frame",
                http_status=status_hint,
                known_provider_retried=retried,
            ),
            stop_reason=_gateway_string(
                event,
                "stop_reason",
                context="turn-complete frame",
                known_provider_retried=retried,
            ),
            provider_retried=retried,
        )
    if event_type == "error":
        error_http_status = _exact_gateway_int(
            event,
            "http_status",
            default=None,
            context="stream error",
            minimum=100,
            maximum=599,
            allow_none=True,
            http_status=status_hint,
            known_provider_retried=retried,
        )
        retryable = _exact_gateway_bool(
            event,
            "retryable",
            default=False,
            context="stream error",
            http_status=error_http_status,
            known_provider_retried=retried,
        )
        # The sync twin's rule, on the transport that reports the same failure as a frame.
        config_recoverable = _exact_gateway_bool(
            event,
            "config_recoverable",
            default=False,
            context="stream error",
            http_status=error_http_status,
            known_provider_retried=retried,
        )
        stream_error = ModelAdapterError(
            _gateway_string(
                event,
                "error",
                context="stream error",
                known_provider_retried=retried,
                http_status=error_http_status,
            )
            or "LLM gateway stream error",
            provider_error_code=(
                _gateway_string(
                    event,
                    "error_code",
                    context="stream error",
                    known_provider_retried=retried,
                    http_status=error_http_status,
                )
                or GATEWAY_BAD_RESPONSE
            ),
            retryable=retryable,
            config_recoverable=config_recoverable,
            http_status=error_http_status,
            provider_retried=retried,
        )
        mark_provider_usage(stream_error, _reported_error_usage(event))
        raise stream_error
    return None  # unknown frame type: forward-compatible, ignore


def _reported_error_usage(payload: dict[str, Any]) -> dict[str, int]:
    """Tokens a *failed* gateway call reported spending, read leniently.

    The twin of the client-side stamp, for the hop: a gateway whose own upstream refused a
    billed turn carries the cost in its error envelope, and without reading it back the outer
    client reports zero for a call the provider charged for. Lenient on purpose -- a malformed
    ``usage`` on an error path must not replace the failure being reported with a different
    one, so anything unreadable simply reads as "not reported".
    """

    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {
        str(key): value
        for key, value in usage.items()
        if type(value) is int and value >= 0
    }


def _error_from_status_body(status: int, detail: str) -> ModelAdapterError:
    """Build a ModelAdapterError from a non-200 response body.

    Both transports land here: the streaming path passes the status and body it read, and
    ``_error_from_http_error`` unwraps its ``HTTPError`` into the same two values. They were separate
    near-identical copies, which is how one of them came to read a field the other did not.
    """

    error_payload: dict[str, Any] = {}
    try:
        parsed = loads_json_ingress(detail)
        if isinstance(parsed, dict):
            error_payload = parsed
        elif detail.lstrip().startswith(("{", "[")):
            return ModelAdapterError(
                f"LLM gateway returned HTTP {status}: invalid JSON error response",
                provider_error_code=GATEWAY_BAD_RESPONSE,
                retryable=False,
                http_status=status,
                provider_retried=False,
            )
    except json.JSONDecodeError:
        if detail.lstrip().startswith(("{", "[")):
            return ModelAdapterError(
                f"LLM gateway returned HTTP {status}: invalid JSON error response",
                provider_error_code=GATEWAY_BAD_RESPONSE,
                retryable=False,
                http_status=status,
                provider_retried=False,
            )
    provider_retried = _exact_gateway_bool(
        error_payload,
        "provider_retried",
        default=False,
        context="HTTP error response",
        http_status=status,
    )
    provider_error_code = _gateway_string(
        error_payload,
        "error_code",
        context="HTTP error response",
        known_provider_retried=provider_retried,
        http_status=status,
    ) or _error_code_for_http_status(status)
    retryable = _exact_gateway_bool(
        error_payload,
        "retryable",
        default=_retryable_for_http_status(status),
        context="HTTP error response",
        http_status=status,
        known_provider_retried=provider_retried,
    )
    # Third reader, same read. Unlike ``retryable`` there is nothing to derive from the status
    # line: a 4xx is a hint that the request was at fault, not a statement that configuration
    # fixes it, so an unstated key is False here rather than status-shaped.
    config_recoverable = _exact_gateway_bool(
        error_payload,
        "config_recoverable",
        default=False,
        context="HTTP error response",
        http_status=status,
        known_provider_retried=provider_retried,
    )
    message = (
        _gateway_string(
            error_payload,
            "error",
            context="HTTP error response",
            known_provider_retried=provider_retried,
            http_status=status,
        )
        or detail
        or f"HTTP {status}"
    )
    error = ModelAdapterError(
        f"LLM gateway returned HTTP {status}: {message}",
        provider_error_code=provider_error_code,
        retryable=retryable,
        config_recoverable=config_recoverable,
        http_status=status,
        # Read for the same reason as ``retryable``: it is a fact about the call the gateway is
        # reporting, and a failure is where it matters most. ``retryable`` forecasts a *future*
        # attempt; this records ones already made, upstream, by a retry loop this client cannot see.
        provider_retried=provider_retried,
    )
    mark_provider_usage(error, _reported_error_usage(error_payload))
    return error


def _error_from_http_error(exc: HTTPError) -> ModelAdapterError:
    return _error_from_status_body(int(exc.code), exc.read().decode("utf-8", errors="replace"))


def _error_code_for_http_status(status: int) -> str:
    if status == 429:
        return GATEWAY_RATE_LIMITED
    if status in {401, 403}:
        return GATEWAY_AUTH_ERROR
    if 500 <= status <= 599:
        return GATEWAY_SERVER_ERROR
    if 400 <= status <= 499:
        return GATEWAY_BAD_REQUEST
    return GATEWAY_BAD_RESPONSE


def _retryable_for_http_status(status: int) -> bool:
    return status == 429 or 500 <= status <= 599


def _should_retry(
    error: ModelAdapterError,
    attempt: int,
    max_attempts: int,
    retry_on: tuple[str, ...],
) -> bool:
    return (
        attempt < max_attempts
        and error.retryable
        and bool(error.provider_error_code)
        and error.provider_error_code in retry_on
    )


def _retry_delay(
    attempt: int,
    initial_delay_s: float,
    max_delay_s: float,
    backoff_multiplier: float,
    jitter_s: float,
) -> float:
    """How long to wait after ``attempt`` failed, before the next one.

    The schedule itself, separated from waiting on it, because the two callers wait differently and
    a backoff policy that differed between the sync and the streamed path would be a difference
    nobody chose.
    """

    delay = min(max_delay_s, initial_delay_s * (backoff_multiplier ** max(0, attempt - 1)))
    if jitter_s > 0:
        delay += random.uniform(0, jitter_s)
    return delay


def _sleep_before_retry(
    attempt: int,
    initial_delay_s: float,
    max_delay_s: float,
    backoff_multiplier: float,
    jitter_s: float,
) -> None:
    """Block the calling thread. Only correct off the event loop -- see ``_asleep_before_retry``."""

    delay = _retry_delay(attempt, initial_delay_s, max_delay_s, backoff_multiplier, jitter_s)
    if delay > 0:
        time.sleep(delay)


async def _asleep_before_retry(
    attempt: int,
    initial_delay_s: float,
    max_delay_s: float,
    backoff_multiplier: float,
    jitter_s: float,
) -> None:
    """Wait without holding the event loop.

    ``astream_turn`` used the blocking sleep, which froze the whole loop for the length of the
    backoff -- up to ``max_delay_s`` per retry, and the default policy reaches 1.1s on its second
    backoff (0.5s then 1.0s, plus up to 0.1s jitter). Nothing else in the run progressed, and the
    run's own cancellation and deadline are raced on that loop, so a run told to stop kept waiting
    for a provider it had already given up on: measured with a longer configured backoff, a 4.5s
    wait let a 100ms heartbeat tick zero times.
    """

    delay = _retry_delay(attempt, initial_delay_s, max_delay_s, backoff_multiplier, jitter_s)
    if delay > 0:
        await asyncio.sleep(delay)
