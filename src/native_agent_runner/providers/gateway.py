from __future__ import annotations

import json
import os
import random
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from native_agent_runner.core.spec import ModelConfig
from native_agent_runner.errors import ModelAdapterError
from native_agent_runner.providers._common import (
    build_reasoning_payload,
    normalize_usage,
    project_message_to_text,
)
from native_agent_runner.providers.base import (
    ModelRequest,
    ModelStreamChunk,
    ModelTurn,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    TurnComplete,
)
from native_agent_runner.tools.base import ToolSpec

DEFAULT_GATEWAY_URL_ENV = "NAR_LLM_GATEWAY_URL"
DEFAULT_GATEWAY_TOKEN_ENV = "NAR_LLM_GATEWAY_TOKEN"

GATEWAY_TIMEOUT = "gateway_timeout"
GATEWAY_NETWORK_ERROR = "gateway_network_error"
GATEWAY_RATE_LIMITED = "gateway_rate_limited"
GATEWAY_SERVER_ERROR = "gateway_server_error"
GATEWAY_AUTH_ERROR = "gateway_auth_error"
GATEWAY_BAD_RESPONSE = "gateway_bad_response"
GATEWAY_BAD_REQUEST = "gateway_bad_request"


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
        for attempt in range(1, max_attempts + 1):
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
                return _parse_gateway_response(data)
            except ModelAdapterError as exc:
                last_error = exc
                if not _should_retry(exc, attempt, max_attempts, retry.retry_on):
                    raise
                _sleep_before_retry(attempt, retry.initial_delay_s, retry.max_delay_s, retry.backoff_multiplier, retry.jitter_s)
            except HTTPError as exc:
                last_error = _error_from_http_error(exc)
                if not _should_retry(last_error, attempt, max_attempts, retry.retry_on):
                    raise last_error from exc
                _sleep_before_retry(attempt, retry.initial_delay_s, retry.max_delay_s, retry.backoff_multiplier, retry.jitter_s)
            except URLError as exc:
                last_error = ModelAdapterError(
                    f"LLM gateway request failed: {exc.reason}",
                    provider_error_code=GATEWAY_NETWORK_ERROR,
                    retryable=True,
                )
                if not _should_retry(last_error, attempt, max_attempts, retry.retry_on):
                    raise last_error from exc
                _sleep_before_retry(attempt, retry.initial_delay_s, retry.max_delay_s, retry.backoff_multiplier, retry.jitter_s)
            except TimeoutError as exc:
                last_error = ModelAdapterError(
                    "LLM gateway request timed out",
                    provider_error_code=GATEWAY_TIMEOUT,
                    retryable=True,
                )
                if not _should_retry(last_error, attempt, max_attempts, retry.retry_on):
                    raise last_error from exc
                _sleep_before_retry(attempt, retry.initial_delay_s, retry.max_delay_s, retry.backoff_multiplier, retry.jitter_s)
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
                _sleep_before_retry(attempt, retry.initial_delay_s, retry.max_delay_s, retry.backoff_multiplier, retry.jitter_s)
        if last_error is not None:
            raise last_error
        raise ModelAdapterError("LLM gateway request failed", provider_error_code=GATEWAY_NETWORK_ERROR)

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
                "httpx is required for gateway streaming; install native-agent-runner[http-async]",
                provider_error_code=GATEWAY_NETWORK_ERROR,
            ) from exc

        config = request.model or self.config
        url = self._resolve_gateway_url(config).rstrip("/") + "/stream"
        body = json.dumps(self._payload(request), ensure_ascii=False).encode("utf-8")
        headers = self._headers()
        retry = config.retry
        max_attempts = max(1, retry.max_attempts)
        last_error: ModelAdapterError | None = None
        for attempt in range(1, max_attempts + 1):
            committed = False
            try:
                async with httpx.AsyncClient(timeout=config.timeout_s) as client:
                    async with client.stream("POST", url, headers=headers, content=body) as response:
                        if response.status_code != 200:
                            detail = (await response.aread()).decode("utf-8", errors="replace")
                            error = _error_from_status_body(response.status_code, detail)
                            if _should_retry(error, attempt, max_attempts, retry.retry_on):
                                raise _StreamRetry(error)
                            raise error
                        committed = True
                        async for chunk in _aiter_sse_chunks(response):
                            yield chunk
                return
            except _StreamRetry as retry_signal:
                last_error = retry_signal.error
                _sleep_before_retry(attempt, retry.initial_delay_s, retry.max_delay_s, retry.backoff_multiplier, retry.jitter_s)
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
                _sleep_before_retry(attempt, retry.initial_delay_s, retry.max_delay_s, retry.backoff_multiplier, retry.jitter_s)
        if last_error is not None:
            raise last_error
        raise ModelAdapterError("LLM gateway stream failed", provider_error_code=GATEWAY_NETWORK_ERROR)

    def _resolve_gateway_url(self, config: ModelConfig) -> str:
        url = self.gateway_url or config.gateway_url or self.config.gateway_url or os.environ.get(DEFAULT_GATEWAY_URL_ENV)
        if not url:
            raise ModelAdapterError(
                f"LLM gateway URL is required via --llm-gateway-url or {DEFAULT_GATEWAY_URL_ENV}"
            )
        return url

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "native-agent-runner/0.2",
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
        return os.environ.get(self.token_env)

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        config = request.model or self.config
        payload: dict[str, Any] = {
            "protocol": "native-agent-runner.llm-turn.v1",
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
    if event_type == "text_delta":
        return TextDelta(text=str(event.get("text") or ""))
    if event_type == "reasoning_delta":
        return ReasoningDelta(text=str(event.get("text") or ""))
    if event_type == "tool_call_delta":
        return ToolCallDelta(
            index=int(event.get("index") or 0),
            arguments_fragment=str(event.get("arguments_fragment") or ""),
            id=event.get("id"),
            name=event.get("name"),
        )
    if event_type == "turn_complete":
        # The gateway's opaque turn_handle is the continuation handle the core stores.
        return TurnComplete(
            response_id=event.get("turn_handle") or event.get("response_id"),
            usage=normalize_usage(event.get("usage")),
            stop_reason=event.get("stop_reason"),
        )
    if event_type == "error":
        raise ModelAdapterError(
            str(event.get("error") or "LLM gateway stream error"),
            provider_error_code=str(event.get("error_code") or GATEWAY_BAD_RESPONSE),
            retryable=bool(event.get("retryable", False)),
            http_status=int(event["http_status"]) if event.get("http_status") is not None else None,
        )
    return None  # unknown frame type: forward-compatible, ignore


def _error_from_status_body(status: int, detail: str) -> ModelAdapterError:
    """Build a ModelAdapterError from a non-200 streaming response body (mirrors
    ``_error_from_http_error`` for the streaming path)."""
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
    )


def _error_from_http_error(exc: HTTPError) -> ModelAdapterError:
    detail = exc.read().decode("utf-8", errors="replace")
    error_payload: dict[str, Any] = {}
    try:
        parsed = json.loads(detail)
        if isinstance(parsed, dict):
            error_payload = parsed
    except json.JSONDecodeError:
        pass
    status = int(exc.code)
    provider_error_code = str(error_payload.get("error_code") or _error_code_for_http_status(status))
    retryable = bool(error_payload.get("retryable", _retryable_for_http_status(status)))
    message = str(error_payload.get("error") or detail or f"HTTP {status}")
    return ModelAdapterError(
        f"LLM gateway returned HTTP {status}: {message}",
        provider_error_code=provider_error_code,
        retryable=retryable,
        http_status=status,
    )


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


def _sleep_before_retry(
    attempt: int,
    initial_delay_s: float,
    max_delay_s: float,
    backoff_multiplier: float,
    jitter_s: float,
) -> None:
    delay = min(max_delay_s, initial_delay_s * (backoff_multiplier ** max(0, attempt - 1)))
    if jitter_s > 0:
        delay += random.uniform(0, jitter_s)
    if delay > 0:
        time.sleep(delay)
