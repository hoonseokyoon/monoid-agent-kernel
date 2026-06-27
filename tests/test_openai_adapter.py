"""Provider-error classification for the OpenAI adapter.

The async streaming path raises a bare ``APIError`` with no ``status_code`` but a populated
``body`` carrying the real code — the classifier must recover it instead of masking everything as a
generic "provider call failed". It must never echo the body's prose (PII/prompt safety).
"""

from __future__ import annotations

from native_agent_runner.providers.openai import _model_error_from_openai


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
    me = _model_error_from_openai(_FakeApiError(status_code=429, body={"code": "rate_limit_exceeded"}))
    assert me.http_status == 429
    assert me.provider_error_code == "rate_limit_exceeded"
    assert me.retryable is True  # a true rate limit can clear on retry


def test_unclassifiable_error_still_carries_a_nonempty_code() -> None:
    me = _model_error_from_openai(_FakeApiError())  # no status, no body
    assert me.provider_error_code == "unclassified_provider_error"
    assert me.http_status is None
    assert "_FakeApiError" in str(me)  # the exception class aids debugging, no body
