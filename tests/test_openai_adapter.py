"""Provider-error classification for the OpenAI adapter.

The async streaming path raises a bare ``APIError`` with no ``status_code`` but a populated
``body`` carrying the real code — the classifier must recover it instead of masking everything as a
generic "provider call failed". It must never echo the body's prose (PII/prompt safety).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.loop import _recoverable_turn_error
from monoid_agent_kernel.providers.openai import (
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
