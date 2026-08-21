from __future__ import annotations

import math

import pytest

from monoid_agent_kernel.core.model_invocation import (
    MODEL_INVOCATION_SCHEMA_VERSION,
    DurableModelInvocation,
    decode_model_invocation,
)


def _invocation(**changes: object) -> DurableModelInvocation:
    values: dict[str, object] = {
        "run_id": "run_1",
        "logical_call_id": "call_1",
        "revision": 1,
        "dispatch_id": "dispatch_1",
        "dispatch_attempt": 1,
        "idempotency_key": "idem_1",
        "dispatch_state": "reserved",
        "request_digest": "request_digest_1",
        "digest_generation": "request-v1",
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
            receipt={"request_digest": "request_digest_1", "attempts": 1},
            result_ref="blob:turn_1",
        ),
        _invocation(
            revision=3,
            dispatch_state="settled",
            receipt={"request_digest": "request_digest_1", "attempts": 1},
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


def test_model_invocation_reads_legacy_namespace_and_writes_canonical_namespace() -> None:
    payload = _invocation().to_json()
    payload["schema_version"] = "native-agent-runner.model-invocation.v1"

    checked = decode_model_invocation(payload)

    assert checked.status == "loaded"
    assert checked.value is not None
    assert checked.value.schema_version == "native-agent-runner.model-invocation.v1"
    assert checked.value.to_json()["schema_version"] == MODEL_INVOCATION_SCHEMA_VERSION


def test_model_invocation_checked_reader_distinguishes_corrupt_and_future() -> None:
    corrupt = _invocation().to_json()
    corrupt["revision"] = True
    future = {**_invocation().to_json(), "schema_version": "monoid.model-invocation.v99"}

    assert decode_model_invocation(corrupt).status == "corrupt"
    assert decode_model_invocation(future).status == "unsupported_version"


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
        {"dispatch_state": "sent"},
        {"request_digest": ""},
        {"digest_generation": ""},
        {"result_ref": 1},
        {"failure_code": 1},
    ),
)
def test_model_invocation_rejects_invalid_identity_and_scalar_fields(
    changes: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _invocation(**changes)


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
        {"provider": {"endpoint": "https://internal.invalid"}},
        {"provider": {"response_body": {"output": "private"}}},
        {"raw_provider_response": {"output": "private"}},
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
            "digest_1",
            "request-v1",
        )
