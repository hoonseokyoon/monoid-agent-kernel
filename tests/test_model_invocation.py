from __future__ import annotations

import math

import pytest

from monoid_agent_kernel.core.model_invocation import (
    LOGICAL_MODEL_CALL_GENERATION,
    MODEL_INVOCATION_SCHEMA_VERSION,
    MODEL_DISPATCH_GENERATION,
    MODEL_REQUEST_DIGEST_GENERATION,
    DurableModelInvocation,
    decode_model_invocation,
    logical_model_call_id,
    model_dispatch_id,
    model_invocation_receipt,
)
from monoid_agent_kernel.core.model_io import ModelCallAttempt, ModelCallReceipt

REQUEST_DIGEST = "a" * 64


def test_durable_model_call_ids_are_deterministic_addresses_without_raw_coordinates() -> None:
    logical = logical_model_call_id("run_private", "turn_0007")

    assert logical == logical_model_call_id("run_private", "turn_0007")
    assert logical != logical_model_call_id("run_private", "turn_0008")
    assert logical.startswith("mcall_")
    assert "run_private" not in logical
    assert "turn_0007" not in logical

    first = model_dispatch_id(logical, 1)
    second = model_dispatch_id(logical, 2)
    assert first == model_dispatch_id(logical, 1)
    assert first != second
    assert first.startswith("mdispatch_")
    assert logical not in first
    assert LOGICAL_MODEL_CALL_GENERATION == "monoid.logical-model-call.v1"
    assert MODEL_DISPATCH_GENERATION == "monoid.model-dispatch.v1"


@pytest.mark.parametrize(
    ("factory", "args"),
    (
        (logical_model_call_id, ("", "step_1")),
        (logical_model_call_id, ("run_1", "private step")),
        (model_dispatch_id, ("", 1)),
        (model_dispatch_id, ("call_1", 0)),
        (model_dispatch_id, ("call_1", True)),
    ),
)
def test_durable_model_call_id_helpers_reject_unportable_coordinates(
    factory: object,
    args: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError):
        factory(*args)  # type: ignore[operator]


def test_runner_receipt_projection_keeps_only_public_safe_invocation_evidence() -> None:
    receipt = ModelCallReceipt(
        provider_name="private-provider-route",
        prompt_digest="b" * 64,
        request_digest=REQUEST_DIGEST,
        stop_reason="stop",
        usage={"input_tokens": 3, "private_counter": 99},
        latency_ms=12,
        attempts=2,
        provider_retried=True,
        error_code="model_error",
        provider_error_code="rate_limited",
        retryable=True,
        config_recoverable=True,
        http_status=429,
        destination_status="resolved",
        destination_digest="c" * 64,
        idempotency_key="idem_private",
        attempt_log=(
            ModelCallAttempt(
                index=1,
                usage={"input_tokens": 1, "private_counter": 49},
            ),
            ModelCallAttempt(
                index=2,
                usage={"input_tokens": 2, "private_counter": 50},
                stream_committed=True,
            ),
        ),
    )

    projected = model_invocation_receipt(receipt)

    assert projected == {
        "attempts": 2,
        "config_recoverable": True,
        "latency_ms": 12,
        "provider_retried": True,
        "retryable": True,
        "request_digest": REQUEST_DIGEST,
        "usage": {"input_tokens": 3},
        "stop_reason": "stop",
        "stream_committed": True,
        "provider_error_code": "rate_limited",
        "http_status": 429,
    }
    assert not {
        "provider_name",
        "prompt_digest",
        "destination_digest",
        "idempotency_key",
        "error_code",
    } & projected.keys()


def _invocation(**changes: object) -> DurableModelInvocation:
    values: dict[str, object] = {
        "run_id": "run_1",
        "logical_call_id": "call_1",
        "revision": 1,
        "dispatch_id": "dispatch_1",
        "dispatch_attempt": 1,
        "idempotency_key": "idem_1",
        "dispatch_state": "reserved",
        "request_digest": REQUEST_DIGEST,
        "digest_generation": MODEL_REQUEST_DIGEST_GENERATION,
    }
    values.update(changes)
    return DurableModelInvocation(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invocation",
    (
        _invocation(),
        _invocation(revision=2, dispatch_state="dispatch_started"),
        _invocation(
            revision=3,
            dispatch_state="settled",
            receipt={"request_digest": REQUEST_DIGEST, "attempts": 1},
            result_ref="blob:turn_1",
        ),
        _invocation(
            revision=3,
            dispatch_state="settled",
            receipt={"request_digest": REQUEST_DIGEST, "attempts": 1},
            failure_code="provider_refused",
        ),
        _invocation(revision=3, dispatch_state="unknown", failure_code="dispatch_unknown"),
    ),
)
def test_model_invocation_round_trips_every_state_shape(
    invocation: DurableModelInvocation,
) -> None:
    checked = decode_model_invocation(invocation.to_json())

    assert checked.status == "loaded"
    assert checked.value == invocation
    assert DurableModelInvocation.from_json(invocation.to_json()) == invocation


@pytest.mark.parametrize("evidence_policy", ["passive", "required", "outbox"])
def test_model_invocation_round_trips_evidence_delivery_policy(evidence_policy: str) -> None:
    invocation = _invocation(evidence_policy=evidence_policy)

    checked = decode_model_invocation(invocation.to_json())

    assert checked.status == "loaded"
    assert checked.value == invocation
    assert invocation.requires_evidence is (evidence_policy == "required")


@pytest.mark.parametrize("legacy_required", [False, True])
def test_model_invocation_reads_legacy_required_flag_as_policy(legacy_required: bool) -> None:
    payload = _invocation().to_json()
    del payload["evidence_policy"]
    payload["requires_evidence"] = legacy_required

    checked = decode_model_invocation(payload)

    assert checked.status == "loaded"
    assert checked.value is not None
    assert checked.value.requires_evidence is legacy_required
    assert checked.value.evidence_policy == ("required" if legacy_required else "passive")
    assert checked.value.to_json()["evidence_policy"] == checked.value.evidence_policy
    assert "requires_evidence" not in checked.value.to_json()


def test_model_invocation_rejects_conflicting_legacy_evidence_alias() -> None:
    payload = _invocation(evidence_policy="outbox").to_json()
    payload["requires_evidence"] = True

    assert decode_model_invocation(payload).status == "corrupt"


def test_model_invocation_reads_legacy_namespace_and_writes_canonical_namespace() -> None:
    payload = _invocation().to_json()
    payload["schema_version"] = "native-agent-runner.model-invocation.v1"
    payload["digest_generation"] = "native-agent-runner.model-request-digest.v1"

    checked = decode_model_invocation(payload)

    assert checked.status == "loaded"
    assert checked.value is not None
    assert checked.value.schema_version == "native-agent-runner.model-invocation.v1"
    assert checked.value.to_json()["schema_version"] == MODEL_INVOCATION_SCHEMA_VERSION
    assert checked.value.to_json()["digest_generation"] == MODEL_REQUEST_DIGEST_GENERATION


def test_model_invocation_checked_reader_distinguishes_corrupt_and_future() -> None:
    corrupt = _invocation().to_json()
    corrupt["revision"] = True
    future = {**_invocation().to_json(), "schema_version": "monoid.model-invocation.v99"}

    assert decode_model_invocation(corrupt).status == "corrupt"
    assert decode_model_invocation(future).status == "unsupported_version"


def test_model_invocation_rejects_equal_but_non_string_dispatch_state() -> None:
    class EqualToReserved:
        def __eq__(self, other: object) -> bool:
            return other == "reserved"

    with pytest.raises(ValueError, match="dispatch_state"):
        _invocation(dispatch_state=EqualToReserved())

    payload = _invocation().to_json()
    payload["dispatch_state"] = EqualToReserved()
    assert decode_model_invocation(payload).status == "corrupt"


@pytest.mark.parametrize(
    "field",
    ("prompt", "requestBody", "raw_response", "unknown_future_field"),
)
def test_model_invocation_checked_reader_rejects_unknown_top_level_fields(
    field: str,
) -> None:
    payload = _invocation().to_json()
    payload[field] = "private content"

    checked = decode_model_invocation(payload)

    assert checked.status == "corrupt"
    assert checked.value is None


@pytest.mark.parametrize(
    "changes",
    (
        {"run_id": ""},
        {"logical_call_id": ""},
        {"revision": 0},
        {"revision": True},
        {"dispatch_id": ""},
        {"dispatch_attempt": 0},
        {"dispatch_attempt": True},
        {"idempotency_key": ""},
        {"idempotency_key": "contains spaces"},
        {"dispatch_state": "sent"},
        {"request_digest": ""},
        {"request_digest": "A" * 64},
        {"request_digest": "private prompt"},
        {"digest_generation": ""},
        {"digest_generation": "request-v1"},
        {"evidence_policy": 1},
        {"evidence_policy": "transactional"},
        {"result_ref": 1},
        {"result_ref": "secret"},
        {"result_ref": "private result text"},
        {"failure_code": 1},
        {"failure_code": "private failure text"},
    ),
)
def test_model_invocation_rejects_invalid_identity_and_scalar_fields(
    changes: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _invocation(**changes)


@pytest.mark.parametrize("legacy_required", [1, "true"])
def test_model_invocation_reader_rejects_invalid_legacy_evidence_flag(
    legacy_required: object,
) -> None:
    payload = _invocation().to_json()
    del payload["evidence_policy"]
    payload["requires_evidence"] = legacy_required

    assert decode_model_invocation(payload).status == "corrupt"


def test_model_invocation_rejects_unserializable_counter_magnitudes() -> None:
    oversized = 10**5000

    for field in ("revision", "dispatch_attempt"):
        with pytest.raises(ValueError, match="positive integer"):
            _invocation(**{field: oversized})


def test_model_invocation_receipt_rejects_unserializable_integer_magnitudes() -> None:
    oversized = 10**5000

    for receipt in ({"attempts": oversized}, {"usage": {"input_tokens": oversized}}):
        with pytest.raises(ValueError):
            _invocation(
                dispatch_state="settled",
                receipt=receipt,
                result_ref="blob:turn",
            )


@pytest.mark.parametrize(
    "changes",
    (
        {"receipt": {"attempts": 0}},
        {"result_ref": "blob:turn"},
        {"failure_code": "failed"},
        {"dispatch_state": "dispatch_started", "receipt": {"attempts": 1}},
        {"dispatch_state": "unknown", "receipt": {"attempts": 1}},
        {"dispatch_state": "unknown", "result_ref": "blob:turn"},
        {"dispatch_state": "settled"},
        {"dispatch_state": "settled", "receipt": {"attempts": 1}},
        {
            "dispatch_state": "settled",
            "receipt": {"attempts": 1},
            "result_ref": "blob:turn",
            "failure_code": "failed",
        },
    ),
)
def test_model_invocation_rejects_illegal_state_local_shapes(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _invocation(**changes)


@pytest.mark.parametrize(
    "receipt",
    (
        {"prompt": "private"},
        {"provider_prompt": "private"},
        {"providerPrompt": "private"},
        {"SystemPrompt": "private"},
        {"provider": {"endpoint": "https://internal.invalid"}},
        {"provider": {"response_body": {"output": "private"}}},
        {"provider": {"responseBody": {"output": "private"}}},
        {"requestBody": {"input": "private"}},
        {"requestPayload": {"input": "private"}},
        {"responsePayload": {"output": "private"}},
        {"requestData": {"input": "private"}},
        {"responseData": {"output": "private"}},
        {"payload": {"output": "private"}},
        {"raw_provider_response": {"output": "private"}},
        {"rawProviderResponse": {"output": "private"}},
        {"raw_exception_message": "secret failure"},
        {"error_message": "secret failure"},
        {"request_headers": {"authorization": "secret"}},
        {"reasoning_summary": "private"},
        {"replay_payload": {"turn": "private"}},
        {"nested": [{"request_body": {"messages": []}}]},
    ),
)
def test_model_invocation_receipt_refuses_private_payload_channels(
    receipt: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="private field"):
        _invocation(
            dispatch_state="settled",
            receipt=receipt,
            result_ref="blob:turn",
        )


def test_model_invocation_receipt_allows_safe_camel_case_evidence_fields() -> None:
    invocation = _invocation(
        dispatch_state="settled",
        receipt={
            "requestDigest": REQUEST_DIGEST,
            "providerRequestId": "request_1",
            "responseId": "response_1",
            "providerResponseId": "response_1",
            "usage": {"inputTokens": 3},
        },
        result_ref="blob:turn",
    )

    assert invocation.receipt == {
        "request_digest": REQUEST_DIGEST,
        "provider_request_id": "request_1",
        "response_id": "response_1",
        "provider_response_id": "response_1",
        "usage": {"input_tokens": 3},
    }


def test_model_invocation_receipt_canonicalizes_the_full_safe_evidence_vocabulary() -> None:
    invocation = _invocation(
        dispatch_state="settled",
        receipt={
            "attempts": 2,
            "durationMs": 12.5,
            "finishReason": "stop",
            "httpStatus": 200,
            "latencyMs": 10,
            "providerErrorCode": "none",
            "providerRequestId": "provider_request_1",
            "providerResponseId": "provider_response_1",
            "providerRetried": True,
            "requestDigest": REQUEST_DIGEST,
            "requestId": "request_1",
            "responseId": "response_1",
            "retryable": False,
            "settledAt": "2026-08-21T10:00:01Z",
            "startedAt": "2026-08-21T10:00:00Z",
            "stopReason": "stop",
            "streamCommitted": True,
            "systemFingerprint": "fingerprint_1",
            "usage": {
                "audioInputTokens": 1,
                "audioOutputTokens": 2,
                "audioTokens": 3,
                "cachedInputTokens": 3,
                "cacheCreationTokens": 4,
                "cacheReadTokens": 4,
                "cacheWriteTokens": 5,
                "inputTokens": 6,
                "outputTokens": 7,
                "reasoningTokens": 8,
                "totalTokens": 9,
            },
        },
        result_ref="blob:turn",
    )

    assert set(invocation.receipt or ()) == {
        "attempts",
        "duration_ms",
        "finish_reason",
        "http_status",
        "latency_ms",
        "provider_error_code",
        "provider_request_id",
        "provider_response_id",
        "provider_retried",
        "request_digest",
        "request_id",
        "response_id",
        "retryable",
        "settled_at",
        "started_at",
        "stop_reason",
        "stream_committed",
        "system_fingerprint",
        "usage",
    }
    assert invocation.receipt is not None
    assert set(invocation.receipt["usage"]) == {
        "audio_input_tokens",
        "audio_output_tokens",
        "audio_tokens",
        "cached_input_tokens",
        "cache_creation_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
    }


def test_model_invocation_receipt_is_detached_on_input_and_output() -> None:
    receipt = {"usage": {"input_tokens": 3}, "attempts": 1}
    invocation = _invocation(
        dispatch_state="settled",
        receipt=receipt,
        result_ref="blob:turn",
    )
    receipt["usage"]["input_tokens"] = 99  # type: ignore[index]
    payload = invocation.to_json()
    payload["receipt"]["usage"]["input_tokens"] = 42  # type: ignore[index]

    assert invocation.receipt == {"usage": {"input_tokens": 3}, "attempts": 1}


@pytest.mark.parametrize("receipt_digest", ("private prompt", "A" * 64, "b" * 64))
def test_model_invocation_receipt_requires_the_same_recorded_request_digest(
    receipt_digest: str,
) -> None:
    with pytest.raises(ValueError, match="request_digest"):
        _invocation(
            dispatch_state="settled",
            receipt={"request_digest": receipt_digest, "attempts": 1},
            result_ref="blob:turn",
        )


@pytest.mark.parametrize(
    "receipt",
    (
        {"requestId": "private prompt"},
        {"providerErrorCode": "private error"},
        {"startedAt": "not-a-timestamp"},
        {"providerRetried": "false"},
        {"latencyMs": -1},
        {"latencyMs": 10**400},
        {"durationMs": math.inf},
        {"durationMs": 10**400},
        {"attempts": 0},
        {"httpStatus": 99},
        {"usage": {"inputTokens": True}},
        {"usage": {"inputTokens": -1}},
        {"usage": {"privateCounter": 1}},
        {"requestDigest": REQUEST_DIGEST, "request_digest": REQUEST_DIGEST},
        {"usage": {"inputTokens": 1, "input_tokens": 1}},
    ),
)
def test_model_invocation_receipt_rejects_values_outside_safe_evidence_shapes(
    receipt: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _invocation(
            dispatch_state="settled",
            receipt=receipt,
            result_ref="blob:turn",
        )


def test_model_invocation_receipt_rejects_nonfinite_numbers_and_cycles() -> None:
    with pytest.raises(ValueError, match="portable JSON"):
        _invocation(
            dispatch_state="settled",
            receipt={"latency": math.inf},
            result_ref="blob:turn",
        )

    cyclic: dict[str, object] = {}
    cyclic["nested"] = cyclic
    with pytest.raises(ValueError, match="portable JSON"):
        _invocation(
            dispatch_state="settled",
            receipt=cyclic,
            result_ref="blob:turn",
        )


def test_model_invocation_checked_reader_classifies_a_recursively_uncopyable_payload() -> None:
    nested: dict[str, object] = {"leaf": True}
    for _ in range(700):
        nested = {"nested": nested}
    payload = _invocation().to_json()
    payload["unknown_future_field"] = nested

    checked = decode_model_invocation(payload)

    assert checked.status == "corrupt"
    assert checked.error_code == "model_invocation_corrupt"


def test_model_invocation_constructor_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        DurableModelInvocation(  # type: ignore[misc]
            "run_1",
            "call_1",
            1,
            "dispatch_1",
            1,
            "idem_1",
            "reserved",
            REQUEST_DIGEST,
            MODEL_REQUEST_DIGEST_GENERATION,
        )
