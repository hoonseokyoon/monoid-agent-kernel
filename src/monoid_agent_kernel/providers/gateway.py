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

from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel._version import user_agent
from monoid_agent_kernel.env import env_name_for_error, getenv
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.identifiers import namespaced_id
from monoid_agent_kernel.providers._common import (
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
GATEWAY_BAD_REQUEST = "gateway_bad_request"


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
    # Optional token source, consulted per request (``_headers`` already re-resolves every call).
    # When set, it takes precedence over the static token/file/env — so a backend can supply a
    # callable that re-mints a fresh gateway token near expiry, keeping a long run (one that outlives
    # the token TTL) authenticated without a restart. ``None`` = today's static behavior.
    token_provider: Callable[[], str | None] | None = None

    # Forwards resolved media blocks in the by-value ``messages`` verbatim to the gateway.
    supports_multimodal: ClassVar[bool] = True

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        config = request.model or self.config
        url = self._resolve_gateway_url(config)
        payload = self._payload(request)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
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
                        data = json.loads(response_body.decode("utf-8"))
                    except json.JSONDecodeError as exc:
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
            raise ModelAdapterError("LLM gateway request failed", provider_error_code=GATEWAY_NETWORK_ERROR)
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
        body = json.dumps(self._payload(request), ensure_ascii=False).encode("utf-8")
        headers = self._headers()
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
                    try:
                        async with client.stream("POST", url, headers=headers, content=body) as response:
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
                                yield chunk
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
            raise ModelAdapterError("LLM gateway stream failed", provider_error_code=GATEWAY_NETWORK_ERROR)
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
        url = self.gateway_url or config.gateway_url or self.config.gateway_url or getenv(DEFAULT_GATEWAY_URL_ENV)
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
        reasoning_payload = build_reasoning_payload(config.reasoning)
        if reasoning_payload:
            payload["reasoning"] = reasoning_payload

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


def _parse_gateway_response(data: dict[str, Any]) -> ModelTurn:
    if "error" in data:
        raise ModelAdapterError(
            str(data["error"]),
            provider_error_code=str(data.get("error_code") or GATEWAY_BAD_RESPONSE),
            retryable=bool(data.get("retryable", False)),
            http_status=int(data["http_status"]) if data.get("http_status") is not None else None,
            provider_retried=bool(data.get("provider_retried", False)),
        )
    raw_calls = data.get("tool_calls") or ()
    tool_calls: list[ToolCall] = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            raise ModelAdapterError(
                "LLM gateway returned an invalid tool call",
                provider_error_code=GATEWAY_BAD_RESPONSE,
            )
        args = raw.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as exc:
                raise ModelAdapterError(
                    f"invalid gateway tool call arguments for {raw.get('name')}",
                    provider_error_code=GATEWAY_BAD_RESPONSE,
                ) from exc
        if not isinstance(args, dict):
            raise ModelAdapterError(
                f"invalid gateway tool call arguments for {raw.get('name')}",
                provider_error_code=GATEWAY_BAD_RESPONSE,
            )
        tool_calls.append(
            ToolCall(
                id=str(raw.get("id") or raw.get("call_id") or ""),
                name=str(raw.get("name") or ""),
                arguments=args,
            )
        )

    # stop_reason rides the gateway wire (added by the gateway server). Older gateways omit it;
    # infer the common cases so the loop's branch still works.
    stop_reason = data.get("stop_reason")
    if stop_reason is None:
        stop_reason = "tool_calls" if tool_calls else "stop"
    return ModelTurn(
        response_id=data.get("response_id") or data.get("turn_handle"),
        final_text=data.get("final_text"),
        tool_calls=tuple(tool_calls),
        usage=normalize_usage(data.get("usage")),
        raw=data,
        stop_reason=stop_reason,
        # A retry the gateway's own backend made. Absent from an older gateway, which reads as
        # "did not retry" -- the same default an adapter with no retry loop carries, and the only
        # thing a wire that never mentions it can honestly mean.
        provider_retried=bool(data.get("provider_retried", False)),
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
                chunk = _chunk_from_event(json.loads("\n".join(data_lines)))
                data_lines = []
                if chunk is not None:
                    yield chunk
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
    if data_lines:
        chunk = _chunk_from_event(json.loads("\n".join(data_lines)))
        if chunk is not None:
            yield chunk


def _chunk_from_event(event: dict[str, Any]) -> ModelStreamChunk | None:
    event_type = event.get("type")
    # A retry the gateway's own backend made, as opposed to one this client's loop made. Read off
    # every frame that carries it, because a stream cancelled mid-flight never delivers the
    # terminal one. Absent reads as "did not retry", which is what a wire that never mentions it
    # can honestly mean; the client's own ``attempt > 1`` is combined with this, never over it.
    retried = bool(event.get("provider_retried", False))
    if event_type == "text_delta":
        return TextDelta(text=str(event.get("text") or ""), provider_retried=retried)
    if event_type == "reasoning_delta":
        return ReasoningDelta(text=str(event.get("text") or ""), provider_retried=retried)
    if event_type == "tool_call_delta":
        return ToolCallDelta(
            index=int(event.get("index") or 0),
            arguments_fragment=str(event.get("arguments_fragment") or ""),
            id=event.get("id"),
            name=event.get("name"),
            provider_retried=retried,
        )
    if event_type == "turn_complete":
        # The gateway's opaque turn_handle is the continuation handle the core stores.
        return TurnComplete(
            response_id=event.get("turn_handle") or event.get("response_id"),
            usage=normalize_usage(event.get("usage")),
            stop_reason=event.get("stop_reason"),
            provider_retried=retried,
        )
    if event_type == "error":
        raise ModelAdapterError(
            str(event.get("error") or "LLM gateway stream error"),
            provider_error_code=str(event.get("error_code") or GATEWAY_BAD_RESPONSE),
            retryable=bool(event.get("retryable", False)),
            http_status=int(event["http_status"]) if event.get("http_status") is not None else None,
            provider_retried=retried,
        )
    return None  # unknown frame type: forward-compatible, ignore


def _error_from_status_body(status: int, detail: str) -> ModelAdapterError:
    """Build a ModelAdapterError from a non-200 response body.

    Both transports land here: the streaming path passes the status and body it read, and
    ``_error_from_http_error`` unwraps its ``HTTPError`` into the same two values. They were separate
    near-identical copies, which is how one of them came to read a field the other did not.
    """

    error_payload: dict[str, Any] = {}
    try:
        parsed = json.loads(detail)
        if isinstance(parsed, dict):
            error_payload = parsed
    except json.JSONDecodeError:
        pass
    provider_error_code = str(error_payload.get("error_code") or _error_code_for_http_status(status))
    retryable = bool(error_payload.get("retryable", _retryable_for_http_status(status)))
    message = str(error_payload.get("error") or detail or f"HTTP {status}")
    return ModelAdapterError(
        f"LLM gateway returned HTTP {status}: {message}",
        provider_error_code=provider_error_code,
        retryable=retryable,
        http_status=status,
        # Read for the same reason as ``retryable``: it is a fact about the call the gateway is
        # reporting, and a failure is where it matters most. ``retryable`` forecasts a *future*
        # attempt; this records ones already made, upstream, by a retry loop this client cannot see.
        provider_retried=bool(error_payload.get("provider_retried", False)),
    )


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
