from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

from monoid_agent_kernel.core.json_ingress import loads_model_json_ingress
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.env import getenv
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.providers._common import (
    build_reasoning_payload,
    normalize_usage,
)
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelStreamChunk,
    ModelTurn,
    ReasoningDelta,
    StopReason,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    TurnComplete,
    format_async_result_text,
)


@dataclass
class _ClientScope:
    """The clients an open lifecycle holds across calls, and the loop the async one belongs to.

    ``loop`` is recorded because an ``AsyncOpenAI`` client's sockets belong to the event loop that
    created them: reused from a different loop it fails, and closed from a different loop it
    cannot be closed at all. The sync client has no such affinity and needs no loop.
    """

    sync_client: Any = None
    async_client: Any = None
    loop: asyncio.AbstractEventLoop | None = None


async def _release_response_stream(stream: Any) -> None:
    """Close one call's response stream, whoever owns the client it came from.

    Duck-typed and tolerant of both spellings: the SDK's async stream closes with a coroutine, a
    stand-in may close synchronously, and one that offers no ``close`` at all has nothing to release.

    Swallowed, for the reason the kernel's own stream cleanup gives: this runs in a ``finally``, so
    whatever it raises *replaces* the call's outcome -- and a provider whose close raised would then
    destroy a turn whose tokens had already been delivered, or turn a user's interrupt into a terminal
    failure. The client close below still runs, and is the authoritative cleanup when this call owns
    the client.
    """

    if stream is None:
        return
    closer = getattr(stream, "close", None)
    if closer is None:
        return
    with contextlib.suppress(Exception):
        outcome = closer()
        if inspect.isawaitable(outcome):
            await outcome


def _loop_is_live(loop: asyncio.AbstractEventLoop | None) -> bool:
    """Whether ``loop`` can still run work.

    The one question both callers ask about a foreign loop, and they must answer it the same way:
    a live loop can be handed a close, and a live loop is also one that may have a call in flight.
    """
    return loop is not None and not loop.is_closed() and loop.is_running()


def _stream_output_index(event: Any) -> int:
    value = getattr(event, "output_index", 0)
    if type(value) is not int or value < 0:
        raise ModelAdapterError(
            "OpenAI returned an invalid stream output_index; expected a non-negative integer",
            provider_error_code="openai_bad_response",
            retryable=False,
        )
    return value


def _provider_string(value: Any, field_name: str, *, required: bool = False) -> str | None:
    if value is None or value == "":
        if required:
            raise ModelAdapterError(f"OpenAI response {field_name} is required")
        return None
    if type(value) is not str:
        raise ModelAdapterError(f"OpenAI response {field_name} must be a string")
    return value


def _first_provider_string(
    field_name: str,
    *values: Any,
    required: bool = False,
) -> str | None:
    for value in values:
        if value is None or value == "":
            continue
        return _provider_string(value, field_name, required=required)
    return _provider_string(None, field_name, required=required)


def _release_foreign_async_client(client: Any, loop: asyncio.AbstractEventLoop | None) -> None:
    """Best-effort close of an async client bound to a loop we are not running on.

    ``close()`` is a coroutine and the sockets belong to ``loop``, so it can only run there. A loop
    that is still turning can be handed the work; one that has stopped or closed cannot run
    anything again, and there is nothing this side can do about the sockets it held. That is the
    reason an async scope should be ended with ``aclose()`` on its own loop -- this path exists to
    keep a *stale* client from being reused, not to promise it was tidied up.

    Only ever called for a loop that has moved on: closing a client out from under a loop still
    using it is what ``_async_client`` refuses to set up.
    """
    if not _loop_is_live(loop):
        return
    # The outcome is deliberately not read, and nothing needs to read it. This used to attach a
    # callback that retrieved and dropped the exception, to avoid "exception never retrieved" noise
    # -- but that is `asyncio.Future` behaviour, and `run_coroutine_threadsafe` hands back a
    # `concurrent.futures.Future`, which has no `__del__` and so reports nothing when it is
    # collected unread. The asyncio task behind it is chained to that future, which does retrieve
    # its exception. Measured on a handoff whose `close()` raises: no warning, no unraisable and no
    # loop exception-handler call, read or unread. And there is no caller who could act on the
    # result of a close this function documents as best-effort.
    try:
        asyncio.run_coroutine_threadsafe(client.close(), loop)
    except RuntimeError:  # pragma: no cover - the loop stopped between the check and the handoff
        return


@dataclass
class OpenAIModelAdapter:
    """Direct OpenAI adapter for local smoke tests.

    Container and gateway-integrated runs should use GatewayModelAdapter so provider
    credentials remain inside your backend platform.

    **Client lifetime.** By default every call builds its own SDK client and closes it, which is
    correct but costs a full client construction per call -- ~0.95s warm on the machine this was
    measured on, against ~13ms for the request itself, and on the streamed path that construction
    is synchronous work sitting on the event loop. A caller that outlives a single call can open a
    scope (:meth:`open`/:meth:`close`, or ``with``/``async with``) and pay it once::

        with OpenAIModelAdapter(config, allow_direct_provider_api=True) as adapter:
            ...                      # every turn reuses one client
                                     # the scope closes it on the way out

    The scope is opt-in rather than automatic because a client held on the adapter with no defined
    end is precisely the leak this adapter used to have: nothing would close it, and its finalizer
    would run against whatever loop happened to be alive. No scope, no cached client.
    """

    config: ModelConfig
    api_key: str | None = None
    allow_direct_provider_api: bool = False

    # Set only while a scope is open; ``None`` means every call owns and closes its own client.
    _scope: _ClientScope | None = field(default=None, init=False, repr=False, compare=False)
    # Calls can arrive on several threads -- the core runs the sync path on a worker thread, and
    # subagents can have more than one in flight -- and two of them racing to fill an empty scope
    # would build two clients and keep whichever lost, leaking it. The lock is never held across
    # an await or a request.
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    # Maps resolved base64 image blocks to Responses ``input_image`` items.
    supports_multimodal: ClassVar[bool] = True
    # Identifies which provider's reasoning artifacts this adapter produces, so the loop tags
    # the captured reasoning block and replay only happens against a matching model.
    provider_name: ClassVar[str] = "openai"

    # -- lifecycle ---------------------------------------------------------------------

    def open(self) -> OpenAIModelAdapter:
        """Begin a scope in which one client is reused across calls. Idempotent; returns self."""
        with self._lock:
            if self._scope is None:
                self._scope = _ClientScope()
        return self

    def close(self) -> None:
        """End the scope and close what it held. Idempotent, and safe with no scope open.

        The sync client is closed here. An async client is handed back to its own loop if that
        loop is still running, because it cannot be closed from anywhere else; an async scope
        should end with :meth:`aclose` instead, which closes it properly.
        """
        scope = self._take_scope()
        if scope is None:
            return
        if scope.async_client is not None:
            _release_foreign_async_client(scope.async_client, scope.loop)
        if scope.sync_client is not None:
            scope.sync_client.close()

    async def aopen(self) -> OpenAIModelAdapter:
        """Async spelling of :meth:`open`, so an async caller can use one idiom throughout."""
        return self.open()

    async def aclose(self) -> None:
        """End the scope from the loop that owns the async client, closing it properly."""
        scope = self._take_scope()
        if scope is None:
            return
        if scope.async_client is not None:
            if scope.loop is asyncio.get_running_loop():
                await scope.async_client.close()
            else:
                # Opened on one loop, closed from another. Nothing here can await on that loop.
                _release_foreign_async_client(scope.async_client, scope.loop)
        if scope.sync_client is not None:
            scope.sync_client.close()

    def _take_scope(self) -> _ClientScope | None:
        """Detach the scope under the lock, so two closers cannot both release the same client."""
        with self._lock:
            scope, self._scope = self._scope, None
            return scope

    def __enter__(self) -> OpenAIModelAdapter:
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    async def __aenter__(self) -> OpenAIModelAdapter:
        return await self.aopen()

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    # -- client acquisition ------------------------------------------------------------

    def _sync_client(self, factory: Any, key: str) -> tuple[Any, bool]:
        """The sync client for one call, and whether the call owns it (and must close it).

        Owned is the unscoped default. A scope keeps its client instead, rebuilding only if
        something closed it underneath us.
        """
        with self._lock:
            scope = self._scope
            if scope is None:
                return factory(api_key=key), True
            if scope.sync_client is None or scope.sync_client.is_closed():
                scope.sync_client = factory(api_key=key)
            return scope.sync_client, False

    def _async_client(self, factory: Any, key: str) -> tuple[Any, bool]:
        """The async client for one call, and whether the call owns it.

        A scoped client is reused only when it belongs to the loop this call is running on. Bound
        to a loop that has moved on it is unusable -- its sockets live there -- so it is dropped
        and rebuilt rather than handed out to fail on first use.

        A loop that has *not* moved on is a different case, and the scope cannot serve it. One
        adapter held open across two concurrently running loops -- two runs sharing it -- meant the
        second loop's request treated the first loop's client as stale and scheduled a ``close()``
        on it, cutting off a call in flight. Reuse belongs to whichever loop the scope holds; a call
        from another live loop gets a client of its own and closes it, which is the same thing an
        unscoped call does. The scope is left untouched, so the loop that owns it keeps its reuse.
        """
        loop = asyncio.get_running_loop()
        stale: tuple[Any, asyncio.AbstractEventLoop | None] | None = None
        with self._lock:
            scope = self._scope
            if scope is None:
                return factory(api_key=key), True
            cached = scope.async_client
            # Asked of the scope's loop, not of the client it happens to be holding. The two are
            # only ever set and cleared together, but that pairing is maintained in the branches
            # below and in ``_take_scope``, and a check that has to be right depends on nothing it
            # does not test itself. Keyed off the client, a scope whose loop outlived its client
            # would be quietly taken over by whichever other loop asked next.
            if scope.loop is not None and scope.loop is not loop and _loop_is_live(scope.loop):
                return factory(api_key=key), True
            if cached is not None and (scope.loop is not loop or cached.is_closed()):
                # Only one belonging to *another* loop has to be handed back. One that is merely
                # closed has nothing left to release, and treating it as foreign would schedule a
                # pointless second close on this very loop.
                if scope.loop is not loop:
                    stale = (cached, scope.loop)
                scope.async_client = None
                scope.loop = None
                cached = None
            if cached is None:
                scope.async_client = factory(api_key=key)
                scope.loop = loop
            client = scope.async_client
        if stale is not None:
            # Outside the lock: the handoff reaches into another loop, and holding this one while
            # it happens would serialise every other call behind a loop we do not control.
            _release_foreign_async_client(*stale)
        return client, False

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        if not self.allow_direct_provider_api and getenv("MONOID_ALLOW_DIRECT_PROVIDER_API") != "1":
            raise ModelAdapterError(
                "direct provider API access is disabled; use GatewayModelAdapter for container runs"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelAdapterError("openai package is not installed") from exc

        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ModelAdapterError("OPENAI_API_KEY is required for OpenAIModelAdapter")

        payload = self._payload(request)
        config = request.model or self.config
        # The client owns an httpx connection pool and nothing else closes it, so whoever owns the
        # client owns the pool. Unscoped that is this call, and the ``finally`` releases it on
        # every exit path; inside a scope the client outlives the call and the scope closes it.
        # Left to the garbage collector instead -- as it was -- each one held its keep-alive
        # sockets open until a collection that may never come.
        #
        # The classifier wraps the block rather than sitting inside it, because the client's own
        # lifecycle -- construction and the close that tears the pool down -- is part of this
        # call's failure surface. A raw exception from either is not a ``ModelAdapterError``, and
        # that is the only type ``AgentLoop._recoverable_turn_error`` will even look at, so an
        # unclassified one terminalizes the run. Same boundary the gateway adapter's streamed path
        # draws around its own client (see its ``httpx.HTTPError`` handler); one handler, because
        # the rule is one rule.
        try:
            client, call_owned = self._sync_client(OpenAI, key)
            try:
                try:
                    response = client.responses.create(**payload, timeout=config.timeout_s)
                except TypeError:
                    response = client.responses.create(**payload)
                data = (
                    response.model_dump()
                    if hasattr(response, "model_dump")
                    else _coerce_response(response)
                )
            finally:
                if call_owned:
                    client.close()
        except ModelAdapterError:
            raise
        except Exception as exc:
            # Map provider API errors (e.g. a 400 for an unsupported reasoning effort) to a
            # classified ModelAdapterError so the gateway returns the real status (4xx, not a
            # generic 500) and the kernel can treat it as recoverable. Never echo the raw body.
            raise _model_error_from_openai(exc) from exc
        return _parse_response(data)

    async def astream_turn(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:
        """Stream a turn from the OpenAI Responses API as neutral ``ModelStreamChunk``s (text
        fragments, tool-call fragments, a terminal usage chunk). Async so the gateway's
        private-loop pump and the loop's async drive can consume it; the sync ``next_turn``
        path is unaffected. Provider errors map to a classified ``ModelAdapterError`` (no body
        leak), exactly like ``next_turn``."""
        if not self.allow_direct_provider_api and getenv("MONOID_ALLOW_DIRECT_PROVIDER_API") != "1":
            raise ModelAdapterError(
                "direct provider API access is disabled; use GatewayModelAdapter for container runs"
            )
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ModelAdapterError("openai package is not installed") from exc

        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ModelAdapterError("OPENAI_API_KEY is required for OpenAIModelAdapter")

        payload = self._payload(request)
        config = request.model or self.config
        final_data: dict[str, Any] = {}
        # An unscoped call owns its client for the same reason ``next_turn``'s does -- the pool --
        # but it has an exit path ``next_turn`` does not: a consumer that abandons the stream
        # throws ``GeneratorExit`` at one of the yields below, which no close placed after the
        # loop ever reaches. ``finally`` covers that one too, and gives an unscoped call the
        # ownership ``GatewayModelAdapter.astream_turn`` has over its httpx client. Left to the
        # collector instead, an abandoned pool kept its keep-alive sockets open and its finalizer
        # then ran ``aclose()`` against a loop that had already closed. A scoped client is not
        # closed here -- the scope owns it, and it is bound to this loop by ``_async_client``.
        #
        # The classifier wraps the block for the reason ``next_turn``'s does, and one more here:
        # the close can run while an error from the stream is already propagating, and an
        # exception raised there *replaces* it. That supersession is Python's and this does not
        # undo it -- the gateway's handler does not either. What it prevents is the replacement
        # being a raw exception, which the loop cannot classify at all.
        try:
            client, call_owned = self._async_client(AsyncOpenAI, key)
            # Bound before the request, so the cleanup below can run even when creating the stream is
            # what failed -- there is nothing to release then, and it must not raise looking.
            stream: Any = None
            try:
                try:
                    stream = await client.responses.create(
                        **payload, stream=True, timeout=config.timeout_s
                    )
                except TypeError:
                    stream = await client.responses.create(**payload, stream=True)

                async for event in stream:
                    etype = getattr(event, "type", "")
                    if etype == "response.output_text.delta":
                        text = (
                            _provider_string(getattr(event, "delta", None), "output text delta")
                            or ""
                        )
                        if text:
                            yield TextDelta(text)
                    elif etype == "response.reasoning_summary_text.delta":
                        # Display-only reasoning summary fragment (DX-13b). Only present when the
                        # request asked for a summary (reasoning.summary != "off").
                        text = (
                            _provider_string(
                                getattr(event, "delta", None), "reasoning summary delta"
                            )
                            or ""
                        )
                        if text:
                            yield ReasoningDelta(text)
                    elif etype == "response.output_item.added":
                        item = getattr(event, "item", None)
                        if item is not None and getattr(item, "type", "") == "function_call":
                            yield ToolCallDelta(
                                index=_stream_output_index(event),
                                id=_first_provider_string(
                                    "function-call id",
                                    getattr(item, "call_id", None),
                                    getattr(item, "id", None),
                                    required=True,
                                ),
                                name=_provider_string(
                                    getattr(item, "name", None),
                                    "function-call name",
                                    required=True,
                                ),
                            )
                    elif etype == "response.function_call_arguments.delta":
                        frag = (
                            _provider_string(
                                getattr(event, "delta", None),
                                "function-call arguments delta",
                            )
                            or ""
                        )
                        if frag:
                            yield ToolCallDelta(
                                index=_stream_output_index(event),
                                arguments_fragment=frag,
                            )
                    elif etype in ("response.completed", "response.incomplete"):
                        # Capture the terminal response for BOTH outcomes: ``response.incomplete``
                        # (max_output_tokens / content_filter) carries status="incomplete" +
                        # incomplete_details, which _stop_reason_from_response maps to
                        # length/refusal. Without it a truncated/refused stream would report a
                        # normal "stop".
                        response = getattr(event, "response", None)
                        if response is not None and hasattr(response, "model_dump"):
                            final_data = response.model_dump()
            finally:
                # The response is released per call, whether or not this call owns the client.
                # Leaving an `async for` does not close the iterator it drove, and closing the
                # client used to be the only cleanup here -- which happened to cover it *unscoped*,
                # because tearing the pool down took the response with it. Inside a scope the client
                # outlives the call, so every turn aborted before the stream drained left its
                # response and connection checked out until the scope ended. Measured: three aborted
                # turns, three connections still open server-side inside the scope.
                await _release_response_stream(stream)
                if call_owned:
                    await client.close()
        except ModelAdapterError:
            raise
        except Exception as exc:
            raise _model_error_from_openai(exc) from exc

        # Outside the block deliberately: the terminal chunk is built from ``final_data`` alone and
        # needs nothing from the client, and a consumer that stops at it holds a suspended
        # generator -- ``break`` does not close one -- which would pin the pool open for as long as
        # it keeps the reference.
        output_items = final_data.get("output") or []
        has_tool_calls = any(item.get("type") == "function_call" for item in output_items)
        yield TurnComplete(
            response_id=final_data.get("id"),
            usage=normalize_usage(final_data.get("usage"), legacy_aliases=True),
            # encrypted_content lives only on the final response object, so reasoning items
            # are captured here (from response.completed) rather than the per-token deltas.
            reasoning=_capture_reasoning_items(output_items),
            stop_reason=_stop_reason_from_response(final_data, tool_calls_present=has_tool_calls),
        )

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        config = request.model or self.config
        payload: dict[str, Any] = {
            "model": config.model,
            "instructions": request.system_prompt,
            "tools": [_openai_tool_schema(tool) for tool in request.tools],
            # ZDR-faithful reasoning round-trip: don't persist server-side state, and ask for
            # the encrypted reasoning so it travels by-value in the message log (re-injected by
            # ``_message_to_input_items``). The engine never relies on ``previous_response_id``.
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        reasoning_payload = build_reasoning_payload(config.reasoning)
        if reasoning_payload:
            payload["reasoning"] = reasoning_payload

        if request.messages is not None:
            # By-value: the full conversation travels as input; no server-side handle. Reasoning
            # captured on assistant turns is re-injected verbatim within the active window when it
            # matches the current model (ZDR round-trip); see ``_reasoning_replay_flags``.
            input_items: list[dict[str, Any]] = []
            replay_flags = _reasoning_replay_flags(request.messages, config.model)
            for message, replay in zip(request.messages, replay_flags):
                input_items.extend(_message_to_input_items(message, replay_reasoning=replay))
            payload["input"] = input_items
        elif request.previous_turn_handle:
            # By-reference (non-ZDR): relies on server-side storage and is unsupported for
            # reasoning round-trip — the engine uses the by-value path above in production.
            payload["previous_response_id"] = request.previous_turn_handle
            input_items = []
            # Third shape: a new user message on top of an existing continuation handle.
            if request.instruction:
                input_items.append({"role": "user", "content": request.instruction})
            input_items.extend(
                _observation_input_item(observation) for observation in request.observations
            )
            payload["input"] = input_items
        else:
            payload["input"] = [{"role": "user", "content": request.instruction or ""}]
        return payload


# Provider error codes (from the error body's ``code``/``type``) we can map to an HTTP status when
# the exception itself carries none — the streaming path raises a bare ``APIError`` with no
# ``status_code`` but a populated ``body``. retryable=False for these means "retrying won't help".
_PROVIDER_CODE_STATUS: dict[str, int] = {
    "insufficient_quota": 429,
    "rate_limit_exceeded": 429,
    "rate_limited": 429,
    "model_not_found": 404,
    "invalid_model": 404,
    "context_length_exceeded": 400,
    "invalid_request_error": 400,
}
# Of the mapped codes, the ones a retry could actually clear (a true transient rate limit). A
# quota/billing failure (``insufficient_quota``) is NOT here — retrying it is futile.
_RETRYABLE_PROVIDER_CODES = frozenset({"rate_limit_exceeded", "rate_limited"})


def _model_error_from_openai(exc: Exception) -> ModelAdapterError:
    """Classify an OpenAI SDK exception into a ModelAdapterError carrying the provider HTTP status
    and error code, so downstream (gateway HTTP mapping, kernel classification, core recoverability,
    the UI) can reason about it. Uses a synthetic, body-free message to avoid leaking prompt/PII."""
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        # The streaming path raises a bare APIError whose status lives on .response (or nowhere).
        status = getattr(getattr(exc, "response", None), "status_code", None)
    body = getattr(exc, "body", None)
    code = (body.get("code") or body.get("type")) if isinstance(body, dict) else None
    code = str(code) if code else ""

    if isinstance(status, int) and 400 <= status < 500:
        return ModelAdapterError(
            f"provider rejected the request (HTTP {status})",
            error_code="model_error",
            provider_error_code=code,
            retryable=(status == 429 and code not in {"insufficient_quota"}),
            http_status=status,
        )
    if isinstance(status, int) and 500 <= status < 600:
        return ModelAdapterError(
            f"provider server error (HTTP {status})",
            error_code="model_error",
            provider_error_code=code,
            retryable=True,
            http_status=status,
        )
    # No usable HTTP status. Recover what we can from the body code so the failure isn't masked as
    # a generic "provider call failed" (e.g. a streaming 429 insufficient_quota with no status_code).
    if code:
        return ModelAdapterError(
            f"provider error: {code}",
            error_code="model_error",
            provider_error_code=code,
            retryable=(code in _RETRYABLE_PROVIDER_CODES),
            http_status=_PROVIDER_CODE_STATUS.get(code),
        )
    return ModelAdapterError(
        f"provider call failed ({type(exc).__name__})",
        error_code="model_error",
        provider_error_code="unclassified_provider_error",
    )


def _openai_tool_schema(tool: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.exported_name,
        "description": tool.description,
        "parameters": tool.input_schema,
    }


def _observation_input_item(observation: Any) -> dict[str, Any]:
    if observation.is_background:
        # A background/hosted task result delivered as a new user message.
        return {"role": "user", "content": format_async_result_text(observation.output)}
    return {
        "type": "function_call_output",
        "call_id": observation.call_id,
        "output": json.dumps(
            observation.output,
            ensure_ascii=False,
            allow_nan=False,
        ),
    }


def _user_content_items(content: list[Any]) -> list[dict[str, Any]]:
    """Map resolved by-value user parts to OpenAI Responses content items.

    ``content`` holds text part-dicts and neutral base64 media blocks (produced by the
    loop's wire-build). A base64 image becomes an ``input_image`` data-URL; a base64 document
    becomes an ``input_file`` with a filename.
    """
    items: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            items.append({"type": "input_text", "text": str(part.get("text", ""))})
        elif part.get("type") == "image":
            source = part.get("source") or {}
            if source.get("type") == "base64":
                data_url = f"data:{source.get('media_type')};base64,{source.get('data')}"
                items.append({"type": "input_image", "image_url": data_url})
        elif part.get("type") == "document":
            source = part.get("source") or {}
            if source.get("type") == "base64":
                data_url = f"data:{source.get('media_type')};base64,{source.get('data')}"
                items.append(
                    {
                        "type": "input_file",
                        "filename": str(part.get("filename") or "document.pdf"),
                        "file_data": data_url,
                    }
                )
    return items


def _message_to_input_items(
    message: dict[str, Any], *, replay_reasoning: bool = False
) -> list[dict[str, Any]]:
    """Translate one provider-neutral by-value message into OpenAI Responses input items.
    An assistant turn with tool calls expands to an assistant text item (if any) plus a
    ``function_call`` item per call; a tool message is a ``function_call_output``.

    When ``replay_reasoning`` is set and the assistant message carries a captured ``reasoning``
    block (see ``_capture_reasoning_items``), its verbatim item subsequence is emitted instead —
    preserving the reasoning→following-item adjacency OpenAI validates — and the reconstructed
    text/function_calls are suppressed to avoid duplication. Callers gate this flag per the
    active-window model-identity rule (``_reasoning_replay_flags``)."""
    role = message.get("role")
    if role == "user":
        content = message.get("content")
        if isinstance(content, list):
            # Multimodal: the loop resolved media to neutral base64 blocks. Map each part to
            # an OpenAI Responses content item (input_text / input_image data-URL).
            return [{"role": "user", "content": _user_content_items(content)}]
        return [{"role": "user", "content": content or ""}]
    if role == "tool":
        items = [
            {
                "type": "function_call_output",
                "call_id": message.get("call_id") or "",
                "output": json.dumps(
                    message.get("content"),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            }
        ]
        # Media a tool returned cannot ride the tool/function output on OpenAI — deliver
        # it as a follow-up user message right after the tool result (the portable split).
        media = message.get("media")
        if isinstance(media, list):
            media_items = _user_content_items(media)
            if media_items:
                items.append({"role": "user", "content": media_items})
        return items
    if role == "assistant":
        reasoning = message.get("reasoning")
        if replay_reasoning and isinstance(reasoning, dict) and reasoning.get("items"):
            return [dict(item) for item in reasoning["items"]]
        items: list[dict[str, Any]] = []
        content = message.get("content") or ""
        if content:
            items.append({"role": "assistant", "content": content})
        for call in message.get("tool_calls") or []:
            items.append(
                {
                    "type": "function_call",
                    "call_id": call.get("id") or "",
                    "name": call.get("name") or "",
                    "arguments": json.dumps(
                        call.get("arguments") or {},
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                }
            )
        return items
    return []


def _reasoning_replay_flags(messages: tuple[dict[str, Any], ...], current_model: str) -> list[bool]:
    """Per-message decision of whether to replay its captured OpenAI reasoning verbatim.

    Two rules (see the DX-13a plan):
    - **Active window only**: reasoning is mandatory to round-trip only since the last ``user``
      message (the in-flight tool loop). Earlier reasoning is historical and droppable — OpenAI
      tolerates historical function_call pairs without their reasoning.
    - **All-or-nothing model identity**: ``config.model`` is re-read every step, so a hot-swap
      can land mid-loop. If any active-window reasoning block isn't ``openai`` at the current
      model, drop reasoning for the whole window so we never send a half-paired set (→ no 400).
    """
    last_user = -1
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user = index
    window_ok = True
    for message in messages[last_user + 1 :]:
        reasoning = message.get("reasoning")
        if isinstance(reasoning, dict) and reasoning.get("items"):
            if reasoning.get("provider") != "openai" or reasoning.get("model") != current_model:
                window_ok = False
                break
    return [index > last_user and window_ok for index in range(len(messages))]


def _capture_reasoning_items(output: list[Any]) -> tuple[dict[str, Any], ...]:
    """The verbatim ``reasoning``/``function_call``/``message`` output subsequence, in order.

    OpenAI pairs each reasoning item with the item that immediately follows it (a
    ``function_call`` or an assistant ``message``) and validates that adjacency on the next
    by-value request; dropping or reordering them yields a ``required following item`` 400.
    Capturing the exact subsequence verbatim — rather than reconstructing items from the parsed
    ``tool_calls``/``final_text`` — is the only construction that survives parallel/interleaved
    tool calls and reasoning→message pairings. The opaque payload (``encrypted_content`` etc.)
    is preserved; only the output-only ``status`` field is dropped, since the Responses *input*
    schema rejects it (``Unknown parameter: input[..].status``). Returns ``()`` when the turn
    carried no reasoning (non-reasoning models are untouched).
    """
    captured: list[dict[str, Any]] = []
    has_reasoning = False
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {"reasoning", "function_call", "message"}:
            captured.append({k: v for k, v in item.items() if k != "status"})
            if item_type == "reasoning":
                has_reasoning = True
    return tuple(captured) if has_reasoning else ()


def _stop_reason_from_response(data: dict[str, Any], *, tool_calls_present: bool) -> StopReason:
    """Map an OpenAI Responses-API response to the typed :data:`StopReason`. Tool calls win
    (the turn isn't final); an ``incomplete`` status is a truncation (``content_filter`` → a
    refusal); a ``refusal`` content part on an otherwise-complete response is a refusal."""
    if tool_calls_present:
        return "tool_calls"
    if data.get("status") == "incomplete":
        reason = (data.get("incomplete_details") or {}).get("reason")
        return "refusal" if reason == "content_filter" else "length"
    for item in data.get("output") or []:
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("type") == "refusal":
                    return "refusal"
    return "stop"


def _parse_response(data: dict[str, Any]) -> ModelTurn:
    output = data.get("output", [])
    if output is None:
        output = []
    if not isinstance(output, (list, tuple)):
        raise ModelAdapterError("OpenAI response output must be an array")
    tool_calls: list[ToolCall] = []
    text_parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            raise ModelAdapterError("OpenAI response output items must be objects")
        item_type = item.get("type")
        if item_type == "function_call":
            args_raw = item.get("arguments")
            if args_raw is None:
                args_raw = {}
            try:
                if isinstance(args_raw, str):
                    args = loads_model_json_ingress(args_raw)
                elif isinstance(args_raw, dict):
                    args = dict(args_raw)
                else:
                    args = None
            except ValueError as exc:
                raise ModelAdapterError(
                    f"invalid function_call arguments for {item.get('name')}"
                ) from exc
            if not isinstance(args, dict):
                raise ModelAdapterError(
                    f"invalid function_call arguments for {item.get('name')}: expected an object"
                )
            tool_calls.append(
                ToolCall(
                    id=_first_provider_string(
                        "function-call id",
                        item.get("call_id"),
                        item.get("id"),
                        required=True,
                    )
                    or "",
                    name=_provider_string(
                        item.get("name"),
                        "function-call name",
                        required=True,
                    )
                    or "",
                    arguments=args,
                )
            )
        elif item_type == "message":
            content = item.get("content", [])
            if content is None:
                content = []
            if not isinstance(content, (list, tuple)):
                raise ModelAdapterError("OpenAI message content must be an array")
            for part in content:
                if not isinstance(part, dict):
                    raise ModelAdapterError("OpenAI message content items must be objects")
                if part.get("type") in {"output_text", "text"}:
                    text_parts.append(_provider_string(part.get("text"), "output text") or "")
        elif item_type in {"output_text", "text"}:
            text_parts.append(_provider_string(item.get("text"), "output text") or "")

    usage_out = normalize_usage(data.get("usage"), legacy_aliases=True)
    return ModelTurn(
        response_id=_provider_string(data.get("id"), "response id"),
        final_text="".join(text_parts).strip() or None,
        tool_calls=tuple(tool_calls),
        usage=usage_out,
        raw=data,
        reasoning=_capture_reasoning_items(output),
        stop_reason=_stop_reason_from_response(data, tool_calls_present=bool(tool_calls)),
    )


def _coerce_response(response: object) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    raise ModelAdapterError("unsupported OpenAI response object")
