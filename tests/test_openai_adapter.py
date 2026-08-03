"""Provider-error classification for the OpenAI adapter.

The async streaming path raises a bare ``APIError`` with no ``status_code`` but a populated
``body`` carrying the real code — the classifier must recover it instead of masking everything as a
generic "provider call failed". It must never echo the body's prose (PII/prompt safety).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from monoid_agent_kernel.errors import ModelAdapterError
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
