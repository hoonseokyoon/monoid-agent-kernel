"""ModelCallReceipt: what happened on one model call, without any of what was said."""

from __future__ import annotations

import json

import pytest

from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.model_io import ModelCallReceipt
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.core.wire_validation import WireValidationError
from monoid_agent_kernel.errors import ModelAdapterError, RunCancelled

TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def test_a_receipt_carries_no_content() -> None:
    """The property that makes a receipt safe for every consumer regardless of capture policy.

    Asserted structurally rather than by inspecting one instance: any field that could hold prompt or
    response text would be a disclosure channel that bypasses `CapturePolicy` entirely.
    """
    fields = set(ModelCallReceipt().to_json())

    assert not fields & {"prompt", "messages", "final_text", "reasoning_text", "content", "response"}


def test_defaults_describe_a_successful_single_attempt_call() -> None:
    receipt = ModelCallReceipt()

    assert receipt.succeeded is True
    assert receipt.attempts == 1
    assert receipt.provider_retried is False
    assert receipt.capture_downgrades == 0
    assert dict(receipt.usage) == {}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"attempts": 0}, "attempts must be 1 or greater"),
        ({"latency_ms": -1}, "latency_ms must not be negative"),
        ({"capture_downgrades": -1}, "capture_downgrades must not be negative"),
    ],
)
def test_impossible_counts_are_rejected(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ModelCallReceipt(**kwargs)  # type: ignore[arg-type]


def test_usage_must_be_whole_token_counts() -> None:
    with pytest.raises(WireValidationError, match="mapping of str to int"):
        ModelCallReceipt(usage={"input_tokens": 1.5})  # type: ignore[dict-item]
    # ``bool`` is an ``int`` subclass, and a boolean token count is a bug, not a count of one.
    with pytest.raises(WireValidationError, match="mapping of str to int"):
        ModelCallReceipt(usage={"input_tokens": True})


def test_usage_counts_must_not_be_negative() -> None:
    """The sign check the other three counters already had. Usage is the one that gets *summed*, so
    a negative slipping through subtracts from an aggregate instead of failing visibly."""
    with pytest.raises(WireValidationError, match="'input_tokens' must not be negative"):
        ModelCallReceipt(usage={"input_tokens": -100})
    with pytest.raises(WireValidationError, match="'output_tokens' must not be negative"):
        ModelCallReceipt.from_json({"usage": {"input_tokens": 5, "output_tokens": -1}})

    # Zero is a real count -- a cache-hit call reports it -- so "reject anything not positive" is
    # not the rule and does not pass here.
    assert dict(ModelCallReceipt(usage={"input_tokens": 0, "output_tokens": 7}).usage) == {
        "input_tokens": 0,
        "output_tokens": 7,
    }


def test_usage_is_copied_away_from_the_caller() -> None:
    supplied = {"input_tokens": 5}
    receipt = ModelCallReceipt(usage=supplied)

    supplied["input_tokens"] = 999

    assert dict(receipt.usage) == {"input_tokens": 5}


def test_attempts_and_provider_retried_are_independent_facts() -> None:
    """A gateway can retry three times inside one call the kernel counts as one attempt.

    A receipt with only ``attempts`` would report that as a clean single call, which is exactly the
    thing an operator investigating latency needs to see.
    """
    internal = ModelCallReceipt(attempts=1, provider_retried=True)
    kernel_level = ModelCallReceipt(attempts=3, provider_retried=False)

    assert (internal.attempts, internal.provider_retried) == (1, True)
    assert (kernel_level.attempts, kernel_level.provider_retried) == (3, False)


def test_trace_ids_come_from_the_invocation_context() -> None:
    """One source of truth: the context is the call's own span, so the receipt does not restate it."""
    receipt = ModelCallReceipt(context=InvocationContext(traceparent=TRACEPARENT))

    assert receipt.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert receipt.span_id == "00f067aa0ba902b7"


# --- with_error -----------------------------------------------------------------------------


def test_with_error_takes_the_taxonomy_a_model_error_already_classified() -> None:
    """The providers raise a classified error, so the runner must not re-derive any of this."""
    receipt = ModelCallReceipt().with_error(
        ModelAdapterError(
            "upstream rejected the request",
            provider_error_code="rate_limit_exceeded",
            retryable=True,
            http_status=429,
        )
    )

    assert receipt.succeeded is False
    assert receipt.error_code == "model_error"
    assert receipt.provider_error_code == "rate_limit_exceeded"
    assert receipt.retryable is True
    assert receipt.http_status == 429


def test_with_error_records_an_unclassified_exception_by_type_not_message() -> None:
    """An arbitrary exception's message can carry request content, and the receipt holds none."""
    receipt = ModelCallReceipt().with_error(ValueError("prompt was: my social security number is"))

    assert receipt.error_code == "ValueError"
    assert "social security" not in json.dumps(receipt.to_json())


def test_with_error_keeps_a_kernel_error_code() -> None:
    receipt = ModelCallReceipt().with_error(RunCancelled("cancelled"))

    assert receipt.error_code == RunCancelled.error_code
    assert receipt.retryable is False


def test_with_error_preserves_everything_else() -> None:
    original = ModelCallReceipt(
        prompt_digest="sha-prompt",
        usage={"input_tokens": 5},
        latency_ms=120,
        attempts=2,
    )

    failed = original.with_error(ModelAdapterError("boom"))

    assert failed.prompt_digest == "sha-prompt"
    assert dict(failed.usage) == {"input_tokens": 5}
    assert (failed.latency_ms, failed.attempts) == (120, 2)
    # A failed call still gets its own receipt rather than mutating the one it came from.
    assert original.succeeded is True


# --- serialization --------------------------------------------------------------------------


def test_json_round_trip_preserves_every_field() -> None:
    receipt = ModelCallReceipt(
        context=InvocationContext(run_id="run_1", skill_id="lecture-note", traceparent=TRACEPARENT),
        model=ModelConfig(provider="openai", model="gpt-5.5", timeout_s=42),
        provider_name="openai",
        prompt_digest="sha-prompt",
        request_digest="sha-request",
        stop_reason="tool_calls",
        usage={"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
        latency_ms=1234,
        attempts=2,
        provider_retried=True,
        error_code="model_error",
        provider_error_code="rate_limit_exceeded",
        retryable=True,
        http_status=429,
        redaction_digest="sha-policy",
        capture_downgrades=1,
    )

    assert ModelCallReceipt.from_json(json.loads(json.dumps(receipt.to_json()))) == receipt


def test_an_unknown_stop_reason_survives_a_round_trip() -> None:
    """A receipt is an audit record: a provider that adds a fifth stop reason must be recordable
    without a kernel change, which is why the field is not the provider ``Literal``."""
    receipt = ModelCallReceipt(stop_reason="content_filter")

    assert ModelCallReceipt.from_json(receipt.to_json()).stop_reason == "content_filter"


def test_from_json_distinguishes_an_absent_http_status_from_zero() -> None:
    assert ModelCallReceipt.from_json({}).http_status is None
    assert ModelCallReceipt.from_json({"http_status": None}).http_status is None
    assert ModelCallReceipt.from_json({"http_status": 0}).http_status == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"context": []},
        {"context": False},
        {"context": 0},
        {"model": []},
        {"model": "gpt-5.5"},
        {"usage": 0},
    ],
)
def test_from_json_rejects_a_falsy_malformed_nested_payload(payload: dict[str, object]) -> None:
    """A falsy wrong type used to be read as "absent", so `{"context": []}` was accepted as an
    anonymous invocation and a corrupt audit record silently lost its run and trace attribution.

    `{"model": []}` additionally used to raise `AttributeError` out of `ModelConfig.from_json`, which
    assumes dict-or-None — a crash on untrusted input rather than a rejection.
    """
    with pytest.raises(WireValidationError):
        ModelCallReceipt.from_json(payload)


@pytest.mark.parametrize("key", ["context", "model", "usage"])
def test_from_json_still_treats_absent_and_null_nested_payloads_as_defaults(key: str) -> None:
    assert ModelCallReceipt.from_json({key: None}) == ModelCallReceipt()


@pytest.mark.parametrize(
    "model",
    [
        {"reasoning": []},
        {"reasoning": "high"},
        {"retry": []},
        {"timeout_s": "soon"},
    ],
)
def test_from_json_rejects_a_malformed_nested_model_config(model: dict[str, object]) -> None:
    """Checking the outer object is not enough.

    `ModelConfig` and its nested `ReasoningConfig` / `ModelRetryConfig` are typed `dict | None` and
    trust it, so a malformed nested object reached `.get` on a list and raised `AttributeError`. A
    consumer handling corrupt audit records through the documented validation exception would crash
    instead — so this module translates at its own boundary, which is the one that made the promise.
    """
    with pytest.raises(WireValidationError, match="model must be a valid model config"):
        ModelCallReceipt.from_json({"model": model})


def test_from_json_rejects_an_oversized_numeric_exponent() -> None:
    """`OverflowError` is an `ArithmeticError`, not a `ValueError`, so it escaped the translation.

    JSON decodes `1e999` to `inf` and `int(inf)` raises it — so a corrupt audit record could crash the
    consumer whose job is to reject corrupt audit records.
    """
    payload = json.loads('{"model": {"timeout_s": 1e999}}')

    with pytest.raises(WireValidationError, match="model must be a valid model config"):
        ModelCallReceipt.from_json(payload)


def test_from_json_still_accepts_a_well_formed_nested_model_config() -> None:
    """The containment must not have been bought by rejecting valid nested configs."""
    restored = ModelCallReceipt.from_json(
        {"model": {"provider": "openai", "model": "gpt-5.5", "reasoning": {}, "retry": {}}}
    )

    assert (restored.model.provider, restored.model.model) == ("openai", "gpt-5.5")


def test_from_json_rejects_a_non_object_and_a_mistyped_field() -> None:
    with pytest.raises(WireValidationError):
        ModelCallReceipt.from_json("not-an-object")
    with pytest.raises(WireValidationError):
        ModelCallReceipt.from_json({"attempts": "2"})
    with pytest.raises(WireValidationError):
        ModelCallReceipt.from_json({"usage": "not-an-object"})
    with pytest.raises(WireValidationError):
        ModelCallReceipt.from_json({"provider_retried": "yes"})
