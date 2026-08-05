"""Provider-error classification for the OpenAI adapter.

The async streaming path raises a bare ``APIError`` with no ``status_code`` but a populated
``body`` carrying the real code — the classifier must recover it instead of masking everything as a
generic "provider call failed". It must never echo the body's prose (PII/prompt safety).
"""

from __future__ import annotations

import asyncio
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.loop import _recoverable_turn_error
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    TurnComplete,
)
from monoid_agent_kernel.providers.openai import (
    OpenAIModelAdapter,
    _model_error_from_openai,
    _stream_output_index,
)


class _FakeApiError(Exception):
    """Mimics an OpenAI SDK exception: optional status_code, optional body dict."""

    def __init__(self, *, status_code: object = None, body: object = None) -> None:
        super().__init__("synthetic")
        if status_code is not None:
            self.status_code = status_code
        self.body = body


_QUOTA_BODY = {
    "code": "insufficient_quota",
    "type": "insufficient_quota",
    "message": "You exceeded your current quota, please check your plan and billing details.",
}


@pytest.mark.parametrize("invalid", ["1", True, 1.9, -1])
def test_openai_stream_rejects_coercible_output_indices(invalid: object) -> None:
    with pytest.raises(ModelAdapterError) as caught:
        _stream_output_index(SimpleNamespace(output_index=invalid))

    assert caught.value.provider_error_code == "openai_bad_response"
    assert caught.value.retryable is False


def test_streaming_error_without_status_recovers_code_and_infers_status() -> None:
    # The exact shape Studio's streaming path hits: no status_code, code lives in .body.
    me = _model_error_from_openai(_FakeApiError(body=_QUOTA_BODY))
    assert me.provider_error_code == "insufficient_quota"
    assert me.http_status == 429  # inferred from the known code
    assert me.retryable is False  # a billing failure won't clear on retry
    # The short code token is fine; the body's prose must NOT leak.
    assert "You exceeded" not in str(me)
    assert "billing" not in str(me)


def test_model_not_found_without_status_infers_404() -> None:
    me = _model_error_from_openai(_FakeApiError(body={"code": "model_not_found"}))
    assert me.provider_error_code == "model_not_found"
    assert me.http_status == 404
    assert me.retryable is False


def test_4xx_with_status_is_preserved_with_body_code() -> None:
    me = _model_error_from_openai(
        _FakeApiError(status_code=429, body={"code": "rate_limit_exceeded"})
    )
    assert me.http_status == 429
    assert me.provider_error_code == "rate_limit_exceeded"
    assert me.retryable is True  # a true rate limit can clear on retry


@pytest.mark.parametrize("status", [408, 409])
def test_transient_4xx_statuses_are_retryable_not_config_shaped(status: int) -> None:
    """408/409 are timeout/conflict conditions a retry clears; classifying them non-retryable
    sent the kernel to a park that tells the operator to change configuration, when backoff
    is the remedy. The retryable flag and the config predicate must agree they are transient."""
    me = _model_error_from_openai(_FakeApiError(status_code=status))
    assert me.http_status == status
    assert me.retryable is True
    assert me.config_recoverable is False
    assert _recoverable_turn_error(me) is True


def test_unclassifiable_error_still_carries_a_nonempty_code() -> None:
    me = _model_error_from_openai(_FakeApiError())  # no status, no body
    assert me.provider_error_code == "unclassified_provider_error"
    assert me.http_status is None
    assert "_FakeApiError" in str(me)  # the exception class aids debugging, no body


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        # A 4xx a retry cannot clear: the request itself is what has to change.
        (_FakeApiError(status_code=400, body={"code": "invalid_request_error"}), True),
        (_FakeApiError(status_code=404, body={"code": "model_not_found"}), True),
        # A 4xx a retry CAN clear is transient, not config-shaped — the two are exclusive.
        (_FakeApiError(status_code=429, body={"code": "rate_limit_exceeded"}), False),
        # 408 (request timeout) and 409 (conflict) are transient server-side conditions:
        # waiting is the remedy, so neither is a configuration defect to park on.
        (_FakeApiError(status_code=408), False),
        (_FakeApiError(status_code=409, body={"code": "conflict"}), False),
        # Quota is a 429 no retry clears, and fixing it is an account/config change.
        (_FakeApiError(status_code=429, body=_QUOTA_BODY), True),
        # 5xx is the upstream's problem, not the caller's request.
        (_FakeApiError(status_code=503), False),
        # Same rule one branch down: the status is synthesized from the body code, and a
        # synthesized 4xx is as config-shaped as a reported one.
        (_FakeApiError(body={"code": "context_length_exceeded"}), True),
        (_FakeApiError(body={"code": "rate_limited"}), False),
        # No status and no code: nothing to classify, so nothing is claimed.
        (_FakeApiError(), False),
    ],
)
def test_the_classifier_flags_a_config_shaped_refusal_on_every_branch(
    exc: Exception, expected: bool
) -> None:
    """One predicate at four sites, so the flag cannot be right on one branch and absent below it.

    ``config_recoverable`` is the client-side spelling of the 4xx fact ``AgentLoop`` already
    treats as recoverable (``_recoverable_turn_error``): the turn fails, the session survives,
    and the remedy is the caller's configuration. Without it, the one adapter that can read a
    provider's own classification never states it — and one hop out, where the status is
    re-derived, the statement is all that is left.
    """

    assert _model_error_from_openai(exc).config_recoverable is expected


class _RequestWithRetryCount:
    """The SDK's final ``httpx.Request``: it carries the retry count in an outgoing header."""

    def __init__(self, count: str | None) -> None:
        self.headers = {} if count is None else {"x-stainless-retry-count": count}


def test_the_classifier_reports_the_retries_its_own_sdk_already_made() -> None:
    """The adapter owns a retry loop it does not run: the OpenAI client's.

    The SDK hands ``retries_taken`` only to the success path, so an exception carries no such
    attribute — but every request it builds stamps the count into ``x-stainless-retry-count``,
    and every ``APIError`` keeps the final request. Without reading it, a call the SDK re-sent
    twice before failing was recorded as a clean single attempt on the receipt, the failure
    record and the wire.
    """

    retried = _FakeApiError(status_code=500)
    retried.request = _RequestWithRetryCount("2")
    assert _model_error_from_openai(retried).provider_retried is True

    first_try = _FakeApiError(status_code=500)
    first_try.request = _RequestWithRetryCount("0")
    assert _model_error_from_openai(first_try).provider_retried is False

    # Every shape that cannot answer reads as "no retry", never as a claimed one: an exception
    # with no request, a request with no header, and a header the SDK did not write.
    assert _model_error_from_openai(_FakeApiError(status_code=500)).provider_retried is False
    silent = _FakeApiError(status_code=500)
    silent.request = _RequestWithRetryCount(None)
    assert _model_error_from_openai(silent).provider_retried is False
    garbled = _FakeApiError(status_code=500)
    garbled.request = _RequestWithRetryCount("many")
    assert _model_error_from_openai(garbled).provider_retried is False


@pytest.mark.parametrize(
    "exc",
    [
        _FakeApiError(status_code=400, body={"code": "invalid_request_error"}),
        _FakeApiError(status_code=503),
        _FakeApiError(body={"code": "context_length_exceeded"}),
        _FakeApiError(),
    ],
)
def test_the_retry_evidence_reaches_every_classification_branch(exc: Exception) -> None:
    """The twin binding: the retry the SDK made is a fact about the call, not about its class."""

    exc.request = _RequestWithRetryCount("1")
    assert _model_error_from_openai(exc).provider_retried is True


def test_a_requestless_transport_error_is_classified_not_replaced() -> None:
    """The retry probe's own policy — anything unreadable means "no retry" — must hold on httpx.

    ``httpx.HTTPError.request`` is a *property that raises* ``RuntimeError`` when unset, and
    ``getattr`` swallows only ``AttributeError`` — so the one probe that ran outside a ``try``
    replaced a classified mid-stream failure (a ``ReadError`` while consuming the body) with a
    raw ``RuntimeError`` the loop cannot classify at all. A real request-less httpx error, not a
    fake: the fake cannot raise from a property the way the real class does.

    The classification itself is the connection family: this is the same connection drop the
    SDK spells ``APIConnectionError`` when it lands before the response headers, so it carries
    the same code and the same remedy (another attempt), request or no request.
    """

    httpx = pytest.importorskip("httpx")
    dropped = httpx.ReadError("connection dropped while reading the body")
    me = _model_error_from_openai(dropped)
    assert me.provider_error_code == "openai_network_error"
    assert me.retryable is True
    assert me.provider_retried is False
    assert me.http_status is None


def test_a_raw_mid_stream_transport_drop_joins_the_connection_family() -> None:
    """The SDK wraps transport failures only up to the response headers — not during the body.

    openai 2.41.1 translates ``httpx`` transport errors into ``APIConnectionError`` /
    ``APITimeoutError`` around ``client.send`` (``_base_client._request``); ``_streaming.py``
    has no such translation, so a connection drop while iterating the stream raises the *raw*
    ``httpx`` exception into the classifier boundary. The identical drop one moment earlier —
    before headers — arrived wrapped and parked recoverably; mid-stream it fell to the
    unclassified tail and terminalized the run. Same condition, same verdict, both spellings:
    ``httpx.TimeoutException`` is itself a ``TransportError`` subclass, so the timeout check
    must run first or every timeout would classify as the network code.
    """

    httpx = pytest.importorskip("httpx")
    for raw, expected_code in [
        (httpx.ReadError("connection dropped mid-body"), "openai_network_error"),
        (httpx.RemoteProtocolError("peer closed connection"), "openai_network_error"),
        (httpx.ReadTimeout("no bytes before the read timeout"), "openai_timeout"),
    ]:
        me = _model_error_from_openai(raw)
        assert me.provider_error_code == expected_code, type(raw).__name__
        assert me.retryable is True, type(raw).__name__
        assert me.http_status is None, type(raw).__name__
        assert me.config_recoverable is False, type(raw).__name__
        # The flag the classification exists to reach: the loop keeps the session alive.
        assert _recoverable_turn_error(me) is True, type(raw).__name__


def test_an_exception_carrying_a_status_outranks_the_connection_family() -> None:
    """Branch order: a real response always outranks the exception's class.

    ``openai.APIStatusError`` carries ``status_code`` and must keep hitting the status
    branches above the connection branch. And the one raw ``httpx`` exception that carries a
    response — ``HTTPStatusError``, which is *not* a ``TransportError`` — classifies by that
    response's status too (the ``.response`` fallback read), never as a transport drop: a
    provider that answered 502 is not a connection that dropped.
    """

    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(502, request=request)

    sdk = _model_error_from_openai(openai.APIStatusError("boom", response=response, body=None))
    assert sdk.http_status == 502
    assert sdk.retryable is True  # the 5xx branch, not the connection branch
    assert sdk.provider_error_code not in {"openai_network_error", "openai_timeout"}

    raw = _model_error_from_openai(
        httpx.HTTPStatusError("bad gateway", request=request, response=response)
    )
    assert raw.http_status == 502
    assert raw.retryable is True
    assert raw.provider_error_code not in {"openai_network_error", "openai_timeout"}


def test_the_connection_family_parks_recoverably_like_the_gateway_twin() -> None:
    """A transient connection failure ends the turn, not the session — on BOTH adapters.

    The gateway twin classifies its transport failures (``URLError`` / ``TimeoutError`` /
    ``OSError``) retryable, so the loop parks ``turn_failed`` and the backend backoff-retries.
    The direct adapter's same condition — ``openai.APIConnectionError``, which
    ``APITimeoutError`` subclasses — fell to the unclassified tail: retryable=False,
    config_recoverable=False, terminal for the whole run. One condition, two verdicts, is the
    dispatch-shape asymmetry this suite exists to close.
    """

    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")

    network = _model_error_from_openai(openai.APIConnectionError(request=request))
    assert network.provider_error_code == "openai_network_error"
    assert network.retryable is True
    assert network.http_status is None
    assert network.config_recoverable is False
    # The flag the classification exists to reach: the loop keeps the session alive.
    assert _recoverable_turn_error(network) is True

    timeout = _model_error_from_openai(openai.APITimeoutError(request=request))
    assert timeout.provider_error_code == "openai_timeout"
    assert timeout.retryable is True
    assert timeout.http_status is None
    assert timeout.config_recoverable is False
    assert _recoverable_turn_error(timeout) is True

    # The connection branch must not leak the SDK's message prose.
    assert "Connection error" not in str(network)


def test_the_connection_branch_still_reports_the_sdk_retries() -> None:
    """The SDK retries connection failures *before* raising, and the evidence must survive.

    An ``APIConnectionError`` always carries the final ``httpx.Request``, so the retry-count
    header is readable on exactly this family — the interplay the classification change must
    keep working.
    """

    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    retried = httpx.Request(
        "POST",
        "https://api.openai.com/v1/responses",
        headers={"x-stainless-retry-count": "2"},
    )
    resent = _model_error_from_openai(openai.APIConnectionError(request=retried))
    assert resent.provider_retried is True

    first_try = httpx.Request("POST", "https://api.openai.com/v1/responses")
    fresh = _model_error_from_openai(openai.APIConnectionError(request=first_try))
    assert fresh.provider_retried is False


# --- The success half of the same fact -------------------------------------------------
#
# The SDK's retry loop runs on successes too, and a call it re-sent before *succeeding* has the
# same audit obligation as one it re-sent before failing. The evidence surfaces differ by path:
# the non-streaming call must go through ``with_raw_response`` (the parsed model keeps no
# reference to the HTTP exchange), and the stream object keeps its ``httpx.Response`` — both end
# at the same stamped ``x-stainless-retry-count`` header the exception probe reads.


_SUCCESS_DATA: dict[str, Any] = {
    "id": "resp_1",
    "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
    "usage": {},
}


class _FakeRawTurn:
    """The ``LegacyAPIResponse`` shape ``with_raw_response.create`` returns.

    ``parse()`` yields what the plain call used to return, and ``http_response`` is the final
    ``httpx.Response`` whose ``.request`` carries the same header the exception probe reads.
    The wrapper itself has NO ``.request`` — verified empirically on openai 2.41.1, and the
    reason the adapter hops to ``.http_response`` before handing it to the one parser.
    """

    def __init__(self, data: dict[str, Any], http_response: object | None = None) -> None:
        self._data = data
        if http_response is not None:
            self.http_response = http_response

    def parse(self) -> dict[str, Any]:
        return self._data


class _UnsetRequestResponse:
    """httpx spells "no request recorded" as a property that *raises* ``RuntimeError``."""

    @property
    def request(self) -> object:
        raise RuntimeError("The request instance has not been set on this response.")


def _stub_sync_openai(
    monkeypatch: pytest.MonkeyPatch, raw: object, *, close_error: Exception | None = None
) -> None:
    """Patch ``openai.OpenAI`` with a client offering both call surfaces, like the SDK.

    ``close_error`` models the exit path the wrapper's own docstring names: the call owns its
    client, the ``finally`` tears the pool down, and a teardown that raises *replaces* the
    outcome of a body that already parsed.
    """

    pytest.importorskip("openai")

    class _WithRaw:
        def create(self, **_kwargs: Any) -> object:
            return raw

    class _Responses:
        with_raw_response = _WithRaw()

        def create(self, **_kwargs: Any) -> dict[str, Any]:
            return _SUCCESS_DATA

    class _Client:
        responses = _Responses()

        def close(self) -> None:
            if close_error is not None:
                raise close_error

    monkeypatch.setattr("openai.OpenAI", lambda **_kwargs: _Client())


class _FakeAsyncStream:
    """The SDK's ``AsyncStream``: async-iterable events plus ``response`` (an ``httpx.Response``)."""

    def __init__(self, events: list[Any], response: object | None = None) -> None:
        self._events = events
        if response is not None:
            self.response = response

    def __aiter__(self) -> _FakeAsyncStream:
        self._it = iter(self._events)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


def _stub_async_openai(monkeypatch: pytest.MonkeyPatch, stream: _FakeAsyncStream) -> None:
    pytest.importorskip("openai")

    class _Responses:
        async def create(self, **_kwargs: Any) -> _FakeAsyncStream:
            return stream

    class _Client:
        responses = _Responses()

        async def close(self) -> None:
            return None

    monkeypatch.setattr("openai.AsyncOpenAI", lambda **_kwargs: _Client())


def _adapter() -> OpenAIModelAdapter:
    return OpenAIModelAdapter(
        ModelConfig(model="gpt-5.5"), api_key="test", allow_direct_provider_api=True
    )


def _stream_events() -> list[Any]:
    """One event per chunk type, so the stamp is proven on every shape the stream yields."""

    return [
        SimpleNamespace(type="response.reasoning_summary_text.delta", delta="think"),
        SimpleNamespace(type="response.output_text.delta", delta="Hi"),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item=SimpleNamespace(type="function_call", call_id="c1", id=None, name="fs_read"),
        ),
        SimpleNamespace(type="response.function_call_arguments.delta", output_index=0, delta="{}"),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                model_dump=lambda: {
                    "id": "r1",
                    "usage": {},
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "c1",
                            "name": "fs_read",
                            "arguments": "{}",
                        }
                    ],
                }
            ),
        ),
    ]


def _drain(adapter: OpenAIModelAdapter) -> list[Any]:
    async def _go() -> list[Any]:
        request = ModelRequest(instruction="hi", system_prompt="", tools=())
        return [chunk async for chunk in adapter.astream_turn(request)]

    return asyncio.run(_go())


def test_a_retried_then_successful_sync_turn_reports_the_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure probe's twin: a call the SDK re-sent and then landed must say so too.

    The parsed model the plain call returns keeps no reference to the HTTP exchange, so the
    adapter reads the raw-response wrapper, whose ``http_response.request`` carries the same
    stamped header the exception path reads — one parser, both verdicts.
    """

    _stub_sync_openai(
        monkeypatch,
        _FakeRawTurn(_SUCCESS_DATA, SimpleNamespace(request=_RequestWithRetryCount("2"))),
    )
    turn = _adapter().next_turn(ModelRequest(instruction="hi", system_prompt="", tools=()))

    assert turn.final_text == "ok"
    assert turn.provider_retried is True


def test_b_a_retried_then_successful_stream_stamps_every_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The streaming twin — and on every chunk, not only the terminal one.

    A stream abandoned mid-flight never yields ``TurnComplete``, and evidence riding only that
    chunk is evidence a cancelled call can never report (the rule ``providers/base.py`` states
    over the chunk vocabulary).
    """

    stream = _FakeAsyncStream(
        _stream_events(), response=SimpleNamespace(request=_RequestWithRetryCount("1"))
    )
    _stub_async_openai(monkeypatch, stream)
    chunks = _drain(_adapter())

    assert [type(chunk) for chunk in chunks] == [
        ReasoningDelta,
        TextDelta,
        ToolCallDelta,
        ToolCallDelta,
        TurnComplete,
    ]
    assert [chunk.provider_retried for chunk in chunks] == [True] * 5


def test_c_an_unretried_success_stays_a_clean_first_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The header the SDK stamps on a first attempt is ``0``, and 0 means no retry — both paths."""

    _stub_sync_openai(
        monkeypatch,
        _FakeRawTurn(_SUCCESS_DATA, SimpleNamespace(request=_RequestWithRetryCount("0"))),
    )
    turn = _adapter().next_turn(ModelRequest(instruction="hi", system_prompt="", tools=()))
    assert turn.provider_retried is False

    stream = _FakeAsyncStream(
        _stream_events(), response=SimpleNamespace(request=_RequestWithRetryCount("0"))
    )
    _stub_async_openai(monkeypatch, stream)
    assert [chunk.provider_retried for chunk in _drain(_adapter())] == [False] * 5


def test_d_unreadable_success_evidence_reads_as_no_retry_and_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe's policy holds on the success path: anything unreadable means "no retry".

    Three shapes that cannot answer: a raw wrapper with no ``http_response`` at all, one whose
    response's ``request`` property raises the way httpx's does when unset, and a stream with no
    ``response`` attribute. All three must produce a successful, un-flagged turn — a probe that
    raised here would destroy an answer the provider already delivered and billed.
    """

    _stub_sync_openai(monkeypatch, _FakeRawTurn(_SUCCESS_DATA))
    request = ModelRequest(instruction="hi", system_prompt="", tools=())
    turn = _adapter().next_turn(request)
    assert turn.final_text == "ok"
    assert turn.provider_retried is False

    _stub_sync_openai(monkeypatch, _FakeRawTurn(_SUCCESS_DATA, _UnsetRequestResponse()))
    turn = _adapter().next_turn(request)
    assert turn.final_text == "ok"
    assert turn.provider_retried is False

    _stub_async_openai(monkeypatch, _FakeAsyncStream(_stream_events()))
    chunks = _drain(_adapter())
    assert [chunk.provider_retried for chunk in chunks] == [False] * 5


# --- a refused body was still generated and billed --------------------------------------
#
# The SOURCE reader. Everything downstream -- the gateway's error envelope, the tenant meter,
# the outer client's receipt -- can only carry a cost this reader recorded, and it recorded
# none: ``_parse_response`` refuses ~a dozen malformed shapes on a body that carries a valid
# ``usage``, and every one of those refusals escaped empty. A model emitting non-JSON
# function-call arguments is ordinary, not exotic, so this is the common case of a paid turn
# disappearing from the ledger.


_BILLED_RESPONSE_USAGE = {"input_tokens": 120, "output_tokens": 340, "total_tokens": 460}


def _billed_response_body(**overrides: Any) -> dict[str, Any]:
    """A complete, well-formed Responses body that reports what the turn cost."""

    body: dict[str, Any] = {
        "id": "resp_1",
        "status": "completed",
        "usage": dict(_BILLED_RESPONSE_USAGE),
        "output": [
            {
                "type": "function_call",
                "call_id": "c1",
                "id": "fc_1",
                "name": "fs_read",
                "arguments": '{"path": "a.md"}',
            },
            {"type": "message", "content": [{"type": "output_text", "text": "ok"}]},
        ],
    }
    body.update(overrides)
    return body


def _call(**overrides: Any) -> dict[str, Any]:
    item = {
        "type": "function_call",
        "call_id": "c1",
        "id": "fc_1",
        "name": "fs_read",
        "arguments": "{}",
    }
    item.update(overrides)
    return {"output": [item]}


def _message(content: Any) -> dict[str, Any]:
    return {"output": [{"type": "message", "content": content}]}


# One malformed shape per raise site the body reader has, named by what it corrupts.
_BILLED_BODY_REFUSALS: dict[str, dict[str, Any]] = {
    "output-not-an-array": {"output": "nope"},
    "output-item-not-an-object": {"output": ["nope"]},
    "arguments-not-json": _call(arguments="{not json"),
    "arguments-not-an-object": _call(arguments="[1, 2]"),
    "arguments-wrong-type": _call(arguments=7),
    "call-id-missing": _call(call_id=None, id=None),
    "call-id-not-a-string": _call(call_id=7),
    "name-missing": _call(name=None),
    "name-not-a-string": _call(name=7),
    "message-content-not-an-array": _message("nope"),
    "message-content-item-not-an-object": _message(["nope"]),
    "output-text-not-a-string": _message([{"type": "output_text", "text": 7}]),
    "response-id-not-a-string": {"id": 7},
    # These two shapes used to refuse RAW (``normalize_usage``'s ``ValueError`` on the nested
    # detail block; ``_stop_reason_from_response``'s ``AttributeError`` on a non-dict
    # ``incomplete_details``) -- the classifying seam now hands every one of them out as the
    # same ``openai_bad_response``, so the parametrized pin below holds one type for the whole
    # table instead of naming exceptions to it.
    "usage-details-not-an-object": {
        "usage": {**_BILLED_RESPONSE_USAGE, "input_tokens_details": "nope"}
    },
    "malformed-incomplete-details": {
        "status": "incomplete",
        "incomplete_details": "nope",
        "output": [],
    },
}


@pytest.mark.parametrize("shape", sorted(_BILLED_BODY_REFUSALS))
def test_a_refused_billed_response_body_still_reports_the_tokens_it_burned(shape: str) -> None:
    from monoid_agent_kernel.providers.base import provider_usage_of
    from monoid_agent_kernel.providers.openai import _parse_response

    body = _billed_response_body(**_BILLED_BODY_REFUSALS[shape])
    with pytest.raises(ModelAdapterError) as refused:
        _parse_response(body)
    assert provider_usage_of(refused.value) == _BILLED_RESPONSE_USAGE, {
        "malformed_shape": shape,
        "carried_by_the_refusal": provider_usage_of(refused.value),
        "hint": "a refused turn was still generated and billed; the refusal is the only "
        "carrier left for its cost",
    }
    # ...and it names the same code whichever key was malformed. Most shapes here were minted
    # bare inside the mapping (``ModelAdapterError`` with no code at all), so one hop out the
    # gateway resolved ``exc.provider_error_code or GATEWAY_BAD_RESPONSE`` and told the client
    # the HOP's wire was bad for an UPSTREAM payload defect -- and byte-identical bodies got
    # different codes on the two transports, since the duck-typed terminal reader refused some
    # of these raw (coded) where this reader refuses them typed (uncoded).
    assert refused.value.provider_error_code == "openai_bad_response", {
        "malformed_shape": shape,
        "named_itself": refused.value.provider_error_code,
        "hint": "the classifying seam backfills the code onto refusals minted without one",
    }


def test_a_raw_malformed_billed_body_is_refused_classified() -> None:
    """The chip's sync half: the shapes that refused raw arrive as ``openai_bad_response``.

    A raw ``ValueError`` reached the loop's blanket wrapper (which dropped the usage stamp)
    in-process, and the gateway's 400/500 arms over a hop -- the 500 arm even said
    ``retryable: true`` for a payload defect, inviting a client to re-buy the same tokens.
    Classified, the refusal keeps its stamp, names its code, and states non-retryability;
    the raw cause stays chained for diagnosis.
    """

    from monoid_agent_kernel.providers.base import provider_usage_of
    from monoid_agent_kernel.providers.openai import _parse_response

    body = _billed_response_body(**_BILLED_BODY_REFUSALS["usage-details-not-an-object"])
    with pytest.raises(ModelAdapterError) as refused:
        _parse_response(body, provider_retried=True)
    assert refused.value.provider_error_code == "openai_bad_response"
    assert refused.value.retryable is False
    assert refused.value.provider_retried is True
    assert provider_usage_of(refused.value) == _BILLED_RESPONSE_USAGE
    assert isinstance(refused.value.__cause__, ValueError)

    stop_reason_shape = _billed_response_body(
        **_BILLED_BODY_REFUSALS["malformed-incomplete-details"]
    )
    with pytest.raises(ModelAdapterError) as second:
        _parse_response(stop_reason_shape)
    assert second.value.provider_error_code == "openai_bad_response"
    assert isinstance(second.value.__cause__, AttributeError)


def test_a_refusal_off_a_response_body_that_cost_nothing_stays_costless() -> None:
    """The counterweight: no reported cost, none invented -- and a malformed ``usage`` is
    itself unreadable rather than a second failure raised over the first."""

    from monoid_agent_kernel.providers.base import provider_usage_of
    from monoid_agent_kernel.providers.openai import _parse_response

    silent = _billed_response_body(**_BILLED_BODY_REFUSALS["arguments-not-json"])
    silent.pop("usage")
    with pytest.raises(ModelAdapterError) as refused:
        _parse_response(silent)
    assert provider_usage_of(refused.value) == {}

    unreadable = _billed_response_body(**_BILLED_BODY_REFUSALS["arguments-not-json"])
    unreadable["usage"] = "not-a-mapping"
    with pytest.raises(ModelAdapterError) as second:
        _parse_response(unreadable)
    assert second.value.provider_error_code == refused.value.provider_error_code
    assert provider_usage_of(second.value) == {}


def test_a_well_formed_billed_body_parses_exactly_as_it_did() -> None:
    """The stamp is a failure-path addition only: the success path is byte-identical."""

    from monoid_agent_kernel.providers.openai import _parse_response

    turn = _parse_response(_billed_response_body(), provider_retried=True)
    assert turn.response_id == "resp_1"
    assert turn.final_text == "ok"
    assert turn.usage == _BILLED_RESPONSE_USAGE
    assert [(call.id, call.name, call.arguments) for call in turn.tool_calls] == [
        ("c1", "fs_read", {"path": "a.md"})
    ]
    assert turn.stop_reason == "tool_calls"
    assert turn.provider_retried is True


# The streamed twin. The terminal chunk is built from the final response payload alone, outside
# the classifier block, and its refusals are the same act on the same billed payload.
#
# ``tool_calls_present`` short-circuits the stop-reason walk, so every probe below drops the
# function_call the well-formed body carries -- otherwise the malformed key is never reached and
# the probe proves nothing.
_BILLED_TERMINAL_REFUSALS: dict[str, dict[str, Any]] = {
    "output-not-an-array": {"output": "nope"},
    "malformed-incomplete-details": {
        "status": "incomplete",
        "incomplete_details": "nope",
        "output": [],
    },
    "message-content-not-an-array": {"output": [{"type": "message", "content": "nope"}]},
}


@pytest.mark.parametrize("shape", sorted(_BILLED_TERMINAL_REFUSALS))
def test_a_refused_billed_terminal_payload_still_reports_the_tokens_it_burned(
    shape: str,
) -> None:
    from monoid_agent_kernel.providers.base import provider_usage_of
    from monoid_agent_kernel.providers.openai import _terminal_chunk

    payload = _billed_response_body(**_BILLED_TERMINAL_REFUSALS[shape])
    with pytest.raises(ModelAdapterError) as refused:
        _terminal_chunk(payload, provider_retried=False)
    assert provider_usage_of(refused.value) == _BILLED_RESPONSE_USAGE, {
        "malformed_shape": shape,
        "hint": "the stream's end-of-turn payload is billed exactly like the one-shot body",
    }


def test_a_billed_terminal_payload_with_a_malformed_id_still_reports_its_cost() -> None:
    """The refusal the INGRESS NORMALIZER raises must carry the payload's cost too.

    ``TurnComplete`` validates nothing, so a non-string ``id`` used to leave ``_terminal_chunk``
    successfully and be refused one step later by ``normalize_model_stream_chunk`` -- outside the
    guard, so the refusal carried no usage and the receipt, the run budget and the gateway meter
    all recorded zero for a billed turn. The chunk is normalized inside the guarded region now,
    so the first validation of every field happens where the stamp is.
    """

    from monoid_agent_kernel.providers.base import (
        ModelAdapterError,
        normalize_model_stream_chunk,
        provider_usage_of,
    )
    from monoid_agent_kernel.providers.openai import _terminal_chunk

    payload = _billed_response_body(id=123)
    with pytest.raises(ModelAdapterError) as refused:
        normalize_model_stream_chunk(_terminal_chunk(payload, provider_retried=False))
    assert provider_usage_of(refused.value) == _BILLED_RESPONSE_USAGE, {
        "hint": "the ingress normalizer's rejection is the same act on the same billed payload",
    }
    # The normalizer mints its refusal without a code, so this named reachable case reached the
    # wire as the hop's own ``gateway_bad_response``; the seam backfills the same code every
    # other refusal in this region carries.
    assert refused.value.provider_error_code == "openai_bad_response"


def test_a_terminal_payload_with_an_unreadable_usage_detail_still_reports_its_counts() -> None:
    """The streamed twin of ``usage-details-not-an-object``, kept out of the table above.

    Every probe in that table drops the ``function_call`` so the stop-reason walk is reached;
    this shape refuses earlier -- ``normalize_usage`` runs while the ``TurnComplete`` arguments
    are being evaluated -- so it does not share the table's precondition and is spelled here
    instead of quietly making that comment false.
    """

    from monoid_agent_kernel.providers.base import provider_usage_of
    from monoid_agent_kernel.providers.openai import _terminal_chunk

    payload = _billed_response_body(
        usage={**_BILLED_RESPONSE_USAGE, "input_tokens_details": "nope"}
    )
    with pytest.raises(ModelAdapterError) as refused:
        _terminal_chunk(payload, provider_retried=False)
    assert refused.value.provider_error_code == "openai_bad_response"
    assert provider_usage_of(refused.value) == _BILLED_RESPONSE_USAGE


def test_a_raw_malformed_terminal_payload_is_refused_classified() -> None:
    """The chip's streamed half: the argument-evaluation refusals arrive classified too.

    ``normalize_model_stream_chunk`` runs INSIDE the guard since fa90590, but Python evaluates
    the ``TurnComplete`` arguments before that call -- so ``normalize_usage``'s ``ValueError``
    and the stop-reason walk's ``AttributeError`` still escaped raw, ahead of the normalizer.
    The guard's classifying arm is what closes them.
    """

    from monoid_agent_kernel.providers.base import provider_usage_of
    from monoid_agent_kernel.providers.openai import _terminal_chunk

    payload = _billed_response_body(
        usage={**_BILLED_RESPONSE_USAGE, "input_tokens_details": "nope"}
    )
    with pytest.raises(ModelAdapterError) as refused:
        _terminal_chunk(payload, provider_retried=True)
    assert refused.value.provider_error_code == "openai_bad_response"
    assert refused.value.retryable is False
    assert refused.value.provider_retried is True
    assert provider_usage_of(refused.value) == _BILLED_RESPONSE_USAGE
    assert isinstance(refused.value.__cause__, ValueError)

    stop_reason_shape = _billed_response_body(
        **_BILLED_TERMINAL_REFUSALS["malformed-incomplete-details"]
    )
    with pytest.raises(ModelAdapterError) as second:
        _terminal_chunk(stop_reason_shape, provider_retried=False)
    assert second.value.provider_error_code == "openai_bad_response"
    assert isinstance(second.value.__cause__, AttributeError)


def test_a_classified_refusal_carries_the_retry_the_exchange_already_spent() -> None:
    """Both seams, one rule: the arm that stamps the cost also completes the classification.

    ``provider_retried`` is a fact about attempts already MADE, and only the caller knows it --
    it is read off the HTTP exchange (``x-stainless-retry-count``), never off the body. The raw
    arms of both regions passed it into the error they minted; the classified arms re-raised
    the refusal untouched, so the ~12 shapes minted inside the mapping said "never retried"
    about an exchange the SDK had already retried. A failed audit record that denies retries in
    exactly the case where they happened is worse than no record.

    Bound on both transports in one test on purpose: this asymmetry existed *because* a rule was
    stated on one of two twin arms.
    """

    from monoid_agent_kernel.providers.base import provider_usage_of
    from monoid_agent_kernel.providers.openai import _parse_response, _terminal_chunk

    # Sync: a malformed-but-TYPED shape, refused as a bare ``ModelAdapterError`` deep in the
    # mapping -- the classified arm, not the raw one.
    body = _billed_response_body(**_BILLED_BODY_REFUSALS["output-not-an-array"])
    with pytest.raises(ModelAdapterError) as sync_side:
        _parse_response(body, provider_retried=True)
    assert sync_side.value.provider_error_code == "openai_bad_response"
    assert sync_side.value.retryable is False
    assert sync_side.value.provider_retried is True
    assert provider_usage_of(sync_side.value) == _BILLED_RESPONSE_USAGE

    # Streamed twin: the non-string ``id`` the terminal seam's docstring names as its reachable
    # case, refused by the ingress normalizer inside the guard -- classified, and codeless.
    with pytest.raises(ModelAdapterError) as stream_side:
        _terminal_chunk(_billed_response_body(id=123), provider_retried=True)
    assert stream_side.value.provider_error_code == "openai_bad_response"
    assert stream_side.value.retryable is False
    assert stream_side.value.provider_retried is True
    assert provider_usage_of(stream_side.value) == _BILLED_RESPONSE_USAGE


def test_completing_a_refusals_class_never_weakens_what_it_already_said() -> None:
    """The seam's two guards, stated directly: backfill only, upgrade only.

    A refusal that already names a code knows something the seam does not, and
    ``provider_retried=True`` is evidence that cannot be un-observed by a later caller who saw
    no retry. Both halves are one-directional or the "completion" would be a downgrade.
    """

    from monoid_agent_kernel.providers.openai import _complete_billed_refusal

    already_named = ModelAdapterError(
        "refused with its own voice",
        provider_error_code="openai_content_filter",
        provider_retried=True,
    )
    _complete_billed_refusal(already_named, {}, provider_retried=False)
    assert already_named.provider_error_code == "openai_content_filter"
    assert already_named.provider_retried is True

    bare = ModelAdapterError("refused bare")
    _complete_billed_refusal(bare, {}, provider_retried=False)
    assert bare.provider_error_code == "openai_bad_response"
    assert bare.provider_retried is False


def test_a_terminal_payload_whose_usage_is_the_malformed_key_invents_nothing() -> None:
    """The lenient read on the streamed twin: ``usage`` is the stamp's own source, so when IT
    is malformed the refusal carries nothing rather than raising a second failure."""

    from monoid_agent_kernel.providers.base import provider_usage_of
    from monoid_agent_kernel.providers.openai import _terminal_chunk

    with pytest.raises(ModelAdapterError) as refused:
        _terminal_chunk(_billed_response_body(usage="nope"), provider_retried=False)
    assert refused.value.provider_error_code == "openai_bad_response"
    assert provider_usage_of(refused.value) == {}


def test_a_well_formed_terminal_payload_builds_the_chunk_it_always_did() -> None:
    from monoid_agent_kernel.providers.openai import _terminal_chunk

    chunk = _terminal_chunk(_billed_response_body(), provider_retried=True)
    assert isinstance(chunk, TurnComplete)
    assert chunk.response_id == "resp_1"
    assert chunk.usage == _BILLED_RESPONSE_USAGE
    assert chunk.stop_reason == "tool_calls"
    assert chunk.provider_retried is True


# --- the completion seam covers the whole adapter, not two regions of it -----------------
#
# ``_parse_response`` and ``_terminal_chunk`` are stamped REGIONS inside two call paths that
# also refuse in the code between and around them: the stream's event loop mints a bare refusal
# per malformed frame, and the sync path's ``_coerce_response`` refuses a response object it
# cannot read. Both escaped through arms that re-raised untouched -- so the SAME helper
# (``_first_provider_string``) answered ``openai_bad_response`` from inside the region and an
# empty code from the loop three lines above it, and one hop out the empty ones resolved to the
# HOP's own ``gateway_bad_response``: the wrong-wire-blamed symptom the seam exists to stop.


class _OpaqueResponse:
    """Neither a pydantic model (no ``model_dump``) nor a mapping -- what ``_coerce_response``
    refuses, and what the stream's terminal reader used to discard in silence."""


class _FakeRawOpaqueTurn:
    """A raw-response wrapper whose ``parse()`` yields a response the reader cannot read."""

    def __init__(self, http_response: object) -> None:
        self.http_response = http_response

    def parse(self) -> object:
        return _OpaqueResponse()


class _FakeRawUnparsableTurn:
    """A raw-response wrapper whose ``parse()`` fails -- after an exchange the SDK retried."""

    def __init__(self, http_response: object) -> None:
        self.http_response = http_response

    def parse(self) -> object:
        raise ValueError("could not decode the response body")


def _malformed_call_frame() -> Any:
    """``response.output_item.added`` for a function_call carrying neither ``call_id`` nor ``id``.

    Refused by ``_first_provider_string(required=True)`` as a BARE ``ModelAdapterError``, minted
    in the stream's event loop -- outside both stamped regions.
    """

    return SimpleNamespace(
        type="response.output_item.added",
        output_index=0,
        item=SimpleNamespace(type="function_call", call_id=None, id=None, name="fs_read"),
    )


def test_a_stream_loop_refusal_leaves_classified_and_reports_the_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) The event loop's own refusals are the adapter's refusals, and must speak its code.

    Uncoded, this reached the reference gateway as ``exc.provider_error_code or
    GATEWAY_BAD_RESPONSE`` -- a 502 naming the hop's wire for an upstream frame defect -- while
    the byte-identical malformation one function deeper answered ``openai_bad_response``. The
    retry the exchange already spent is bound at the top of this same scope and was dropped.
    """

    _stub_async_openai(
        monkeypatch,
        _FakeAsyncStream(
            [_malformed_call_frame()],
            response=SimpleNamespace(request=_RequestWithRetryCount("2")),
        ),
    )
    with pytest.raises(ModelAdapterError) as refused:
        _drain(_adapter())

    assert refused.value.provider_error_code == "openai_bad_response"
    assert refused.value.provider_retried is True


def test_a_stream_loop_refusal_after_the_terminal_frame_still_pays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) A malformed frame that arrives AFTER ``response.completed`` is a BILLED refusal.

    The terminal payload -- with the counts the provider charged for -- is already in hand by
    then, so the refusal is the only carrier left for a cost that really was incurred.
    """

    from monoid_agent_kernel.providers.base import provider_usage_of

    events = [
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(model_dump=lambda: _billed_response_body()),
        ),
        _malformed_call_frame(),
    ]
    _stub_async_openai(
        monkeypatch,
        _FakeAsyncStream(events, response=SimpleNamespace(request=_RequestWithRetryCount("2"))),
    )
    with pytest.raises(ModelAdapterError) as refused:
        _drain(_adapter())

    assert refused.value.provider_error_code == "openai_bad_response"
    assert refused.value.provider_retried is True
    assert provider_usage_of(refused.value) == _BILLED_RESPONSE_USAGE


def test_a_sync_coerce_refusal_leaves_classified_and_reports_the_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) ``_coerce_response`` mints bare, one line below the region that would have named it."""

    _stub_sync_openai(
        monkeypatch, _FakeRawOpaqueTurn(SimpleNamespace(request=_RequestWithRetryCount("2")))
    )
    with pytest.raises(ModelAdapterError) as refused:
        _adapter().next_turn(ModelRequest(instruction="hi", system_prompt="", tools=()))

    assert refused.value.provider_error_code == "openai_bad_response"
    assert refused.value.provider_retried is True


def test_a_sync_parse_failure_reports_the_retry_the_enclosing_scope_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(d) The raw arm re-derived the retry fact from the exception and discarded the right one.

    ``_provider_retried_by_the_sdk`` reads the exchange, and a ``ValueError`` from ``parse()``
    carries no exchange at all -- so a call the SDK re-sent twice was recorded as a clean single
    attempt. The enclosing scope had already read the true value off ``raw.http_response``.
    """

    _stub_sync_openai(
        monkeypatch, _FakeRawUnparsableTurn(SimpleNamespace(request=_RequestWithRetryCount("2")))
    )
    with pytest.raises(ModelAdapterError) as refused:
        _adapter().next_turn(ModelRequest(instruction="hi", system_prompt="", tools=()))

    assert refused.value.provider_retried is True


def test_a_streamed_terminal_response_gets_the_sync_readers_coerce_twin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(e) Both halves of the twin the stream never had.

    The sync reader spells its terminal read ``model_dump() if hasattr(...) else
    _coerce_response(response)``: a plain mapping is ACCEPTED, anything else is REFUSED. The
    stream's ``if ... and hasattr(...)`` silently discarded both -- so a billed streamed turn
    whose terminal response is a mapping reported SUCCESS with zero usage (the run budget, the
    receipt and the tenant ledger all recording a free call, the captured reasoning dropped),
    and an unreadable one was not refused at all.
    """

    _stub_async_openai(
        monkeypatch,
        _FakeAsyncStream(
            [SimpleNamespace(type="response.completed", response=_billed_response_body())]
        ),
    )
    chunks = _drain(_adapter())
    assert [type(chunk) for chunk in chunks] == [TurnComplete]
    assert chunks[-1].usage == _BILLED_RESPONSE_USAGE, {
        "carried_by_the_terminal_chunk": chunks[-1].usage,
        "hint": "a mapping the sync twin accepts must not be dropped on the floor here",
    }
    assert chunks[-1].response_id == "resp_1"

    _stub_async_openai(
        monkeypatch,
        _FakeAsyncStream(
            [SimpleNamespace(type="response.completed", response=_OpaqueResponse())]
        ),
    )
    with pytest.raises(ModelAdapterError) as refused:
        _drain(_adapter())
    assert refused.value.provider_error_code == "openai_bad_response"


def test_the_new_arms_backfill_and_never_overwrite_a_refusal_that_named_itself() -> None:
    """The completion arms cover the REQUEST build too, and must leave its verdict alone.

    ``_classified_payload`` and the by-reference refusal both sit inside the widened region and
    both name themselves -- and neither is a bad *response*. Backfill-only is what keeps a
    seam that now spans the whole call from re-labelling a config-shaped refusal as an upstream
    payload defect, which would flip its remedy from "fix the request" to "the provider is
    broken" one hop out.
    """

    adapter = _adapter()
    unserializable = ModelRequest(
        instruction="hi", system_prompt="", tools=(), output_schema={"enum": {1, 2}}
    )
    with pytest.raises(ModelAdapterError) as refused:
        adapter.next_turn(unserializable)
    assert refused.value.provider_error_code == "unserializable_request"
    assert refused.value.config_recoverable is True


def test_naming_a_refusal_tolerates_a_type_that_refuses_the_attribute() -> None:
    """The seam's third mutation joins its two siblings behind a guarded helper.

    ``mark_provider_retried`` and ``mark_provider_usage`` both tolerate an exception type that
    will not take the attribute, and state why: this runs INSIDE an except-handler, so a write
    that raises replaces the provider's failure with the seam's own. The code stamp was a bare
    ``setattr`` beside them -- and the population it runs over is exactly the one the gateway's
    type-agnostic readers exist for, a third-party adapter refusing in its own type.
    """

    from monoid_agent_kernel.providers.openai import _complete_billed_refusal

    class _ReadOnlyCode(Exception):
        """A third-party refusal whose code is a computed, read-only property."""

        @property
        def provider_error_code(self) -> str:
            return ""

    refusing = _ReadOnlyCode("a third-party adapter's own refusal")
    _complete_billed_refusal(refusing, {}, provider_retried=True)  # type: ignore[arg-type]
    assert refusing.provider_error_code == ""


# --- the RAW arms of the same two trys pay what the turn already cost ---------------------
#
# The completion arm of each call path carries the cost; the raw arm beside it -- the same
# ``try``, one line down -- minted its error and threw the payload away. Both raw arms are
# reachable AFTER a billed body is in hand: the stream keeps emitting frames past
# ``response.completed`` and the connection can drop there (``_connection_error_code``'s own
# docstring names that lane), and the sync call's pool teardown runs in a ``finally`` after the
# body has parsed. A network drop is honestly retryable and stays so; what was wrong is only
# that the turn's cost went with it.


class _DroppedAsyncStream(_FakeAsyncStream):
    """A stream that keeps emitting after ``response.completed`` and then drops mid-body.

    The SDK translates transport failures into ``APIConnectionError`` only up to the response
    headers, so a drop while the BODY streams arrives as the raw ``httpx`` family instead --
    which is exactly the lane that reaches the adapter's classifier with a terminal payload
    already captured.
    """

    def __init__(
        self, events: list[Any], error: Exception, response: object | None = None
    ) -> None:
        super().__init__(events, response)
        self._error = error

    async def __anext__(self) -> Any:
        try:
            return next(self._it)
        except StopIteration:
            raise self._error from None


def _terminal_event() -> Any:
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(model_dump=lambda: _billed_response_body()),
    )


def test_a_drop_after_the_streams_terminal_response_still_reports_its_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The streamed raw arm: the counts were captured, then the connection went.

    The classified arm two lines above stamps this same ``final_data``; the raw one beside it
    minted a network error with nothing, so a turn the provider generated and billed was metered
    at zero by the receipt, the run's budget and the tenant ledger behind it.
    """

    httpx = pytest.importorskip("httpx")
    from monoid_agent_kernel.providers.base import provider_usage_of

    _stub_async_openai(
        monkeypatch,
        _DroppedAsyncStream(
            [_terminal_event()], httpx.ReadError("the connection dropped mid-body")
        ),
    )
    with pytest.raises(ModelAdapterError) as dropped:
        _drain(_adapter())

    assert provider_usage_of(dropped.value) == _BILLED_RESPONSE_USAGE
    # The classification itself is untouched: a drop is a drop, and waiting is still the remedy.
    assert dropped.value.provider_error_code == "openai_network_error"
    assert dropped.value.retryable is True


def test_a_drop_before_the_terminal_response_invents_no_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The counterweight: mid-stream usage is genuinely unknowable, so nothing is claimed."""

    httpx = pytest.importorskip("httpx")
    from monoid_agent_kernel.providers.base import provider_usage_of

    _stub_async_openai(
        monkeypatch,
        _DroppedAsyncStream(
            [SimpleNamespace(type="response.output_text.delta", delta="half an ans")],
            httpx.ReadTimeout("the connection stopped answering"),
        ),
    )
    with pytest.raises(ModelAdapterError) as dropped:
        _drain(_adapter())

    assert provider_usage_of(dropped.value) == {}
    assert dropped.value.provider_error_code == "openai_timeout"


def test_a_teardown_that_fails_after_a_billed_body_still_reports_its_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sync raw arm: ``data`` is bound, and the ``finally``'s close is what failed.

    The client's own lifecycle is inside this call's failure surface deliberately, so a pool
    teardown that raises replaces the outcome of a body that already parsed -- and the
    replacement carried none of the cost the body reported.
    """

    from monoid_agent_kernel.providers.base import provider_usage_of

    _stub_sync_openai(
        monkeypatch,
        _FakeRawTurn(_billed_response_body(), SimpleNamespace(request=_RequestWithRetryCount("0"))),
        close_error=RuntimeError("the connection pool teardown failed"),
    )
    with pytest.raises(ModelAdapterError) as failed:
        _adapter().next_turn(ModelRequest(instruction="hi", system_prompt="", tools=()))

    assert provider_usage_of(failed.value) == _BILLED_RESPONSE_USAGE


class _FakeRawMappingTurn:
    """A raw wrapper whose ``parse()`` yields a Mapping that is not a ``dict``.

    ``_coerce_response`` is ``dict``-only and refuses it; ``usage_reported_by`` accepts any
    Mapping. So the body is READABLE for its cost by the very seam that refused it -- and the
    sync completion arm was handing that seam ``None`` unconditionally.
    """

    def __init__(self, data: dict[str, Any], http_response: object) -> None:
        self._data = MappingProxyType(data)
        self.http_response = http_response

    def parse(self) -> object:
        return self._data


def test_a_coerce_refusal_off_a_readable_mapping_still_reports_its_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sync classified arm: refusing the CONTAINER is not refusing to read the cost."""

    from monoid_agent_kernel.providers.base import provider_usage_of

    _stub_sync_openai(
        monkeypatch,
        _FakeRawMappingTurn(
            _billed_response_body(), SimpleNamespace(request=_RequestWithRetryCount("2"))
        ),
    )
    with pytest.raises(ModelAdapterError) as refused:
        _adapter().next_turn(ModelRequest(instruction="hi", system_prompt="", tools=()))

    assert refused.value.provider_error_code == "openai_bad_response"
    assert refused.value.provider_retried is True
    assert provider_usage_of(refused.value) == _BILLED_RESPONSE_USAGE
