from __future__ import annotations

import json

import pytest

from monoid_agent_kernel.core.external_agent_envelope import (
    EXTERNAL_AGENT_ENVELOPE_VERSION,
    RESERVED_EXTERNAL_AGENT_METADATA_KEYS,
    ExternalAgentEnvelope,
    ExternalAgentError,
    ExternalAgentPart,
    ExternalAgentResult,
    external_agent_envelope_from_outbox_request,
    external_agent_envelope_to_inbox_message,
    merge_canonical_metadata,
    normalize_external_agent_error,
    validate_external_agent_envelope,
)
from monoid_agent_kernel.core.outbox import OutboxRequest
from monoid_agent_kernel.core.trace_context import new_traceparent, trace_id_of


def test_external_agent_envelope_round_trips_ordered_parts() -> None:
    envelope = ExternalAgentEnvelope(
        peer_id="worker",
        message_id="msg-1",
        task_id="task-1",
        correlation_id="corr-1",
        causation_id="cause-1",
        parts=(
            ExternalAgentPart(type="text", text="hello"),
            ExternalAgentPart(type="data", data={"answer": 42}),
            ExternalAgentPart(type="artifact", artifact_id="art-1", mime_type="text/plain"),
        ),
        result=ExternalAgentResult(
            state="completed",
            terminal=True,
            error=ExternalAgentError(code="none", message=""),
        ),
    )

    payload = envelope.to_json()
    assert payload["protocol"] == EXTERNAL_AGENT_ENVELOPE_VERSION

    back = validate_external_agent_envelope(payload)
    assert back.peer_id == "worker"
    assert [part.type for part in back.parts] == ["text", "data", "artifact"]
    assert back.result is not None and back.result.terminal is True


def test_external_agent_envelope_rejects_malformed_payload() -> None:
    with pytest.raises(ValueError):
        validate_external_agent_envelope(
            {
                "protocol": EXTERNAL_AGENT_ENVELOPE_VERSION,
                "peer_id": "worker",
                "message_id": "msg-1",
                "parts": [],
            }
        )


def test_external_agent_envelope_normalizes_direct_json_content() -> None:
    envelope = validate_external_agent_envelope(
        {
            "protocol": EXTERNAL_AGENT_ENVELOPE_VERSION,
            "peer_id": "worker\ud800",
            "message_id": "msg\ud800",
            "parts": [
                {"type": "text", "text": "hello\ud800"},
                {"type": "data", "data": {"score": float("nan")}},
            ],
            "result": {
                "state": "completed\ud800",
                "terminal": True,
                "metadata": {"limit": float("inf")},
                "error": {
                    "code": "peer\ud800",
                    "message": "failed\ud800",
                    "retryable": False,
                },
            },
            "created_at": 1.0,
            "metadata": {"label": "bad\ud800", "score": float("-inf")},
        }
    )

    assert envelope.peer_id == "worker\ufffd"
    assert envelope.parts[0].text == "hello\ufffd"
    assert envelope.parts[1].data == {"score": None}
    assert envelope.result is not None
    assert envelope.result.state == "completed\ufffd"
    assert envelope.result.metadata == {"limit": None}
    assert envelope.result.error is not None
    assert envelope.result.error.code == "peer\ufffd"
    assert envelope.metadata == {"label": "bad\ufffd", "score": None}

    payload = envelope.to_json()
    json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    inbox = external_agent_envelope_to_inbox_message(envelope, run_id="run\ud800")
    assert inbox.run_id == "run\ufffd"
    assert json.loads(inbox.content[1]["text"]) == {"score": None}


def test_external_agent_envelope_normalizes_direct_dataclass_and_rejects_controls() -> None:
    envelope = ExternalAgentEnvelope(
        peer_id="worker\ud800",
        message_id="msg\ud800",
        parts=(
            ExternalAgentPart(
                type="data",
                data={"score": float("nan"), "text": "bad\ud800"},
            ),
        ),
        created_at=1.0,
        metadata={"limit": float("inf")},
    )

    payload = envelope.to_json()
    assert payload["peer_id"] == "worker\ufffd"
    assert payload["parts"][0]["data"] == {"score": None, "text": "bad\ufffd"}
    assert payload["metadata"] == {"limit": None}
    inbox = external_agent_envelope_to_inbox_message(envelope, run_id="run-1")
    assert json.loads(inbox.content[0]["text"]) == {
        "score": None,
        "text": "bad\ufffd",
    }

    for invalid_created_at in (float("nan"), 10**400):
        with pytest.raises(ValueError, match="created_at must be a finite number"):
            ExternalAgentEnvelope(
                peer_id="worker",
                message_id="msg",
                parts=(ExternalAgentPart(type="text", text="hello"),),
                created_at=invalid_created_at,
            ).to_json()
    with pytest.raises(ValueError, match="retryable must be a boolean"):
        normalize_external_agent_error(
            "failed",
            retryable=1,  # type: ignore[arg-type]
        )


def test_external_agent_envelope_rejects_normalized_key_collisions() -> None:
    data = {"\ud800": 1}
    data["\ufffd"] = 2
    with pytest.raises(ValueError, match="keys collide"):
        validate_external_agent_envelope(
            {
                "protocol": EXTERNAL_AGENT_ENVELOPE_VERSION,
                "peer_id": "worker",
                "message_id": "msg",
                "parts": [{"type": "data", "data": data}],
            }
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "protocol": EXTERNAL_AGENT_ENVELOPE_VERSION,
            "peer_id": "worker",
            "message_id": "msg-1",
            "parts": [1],
        },
        {
            "protocol": EXTERNAL_AGENT_ENVELOPE_VERSION,
            "peer_id": "worker",
            "message_id": "msg-1",
            "parts": [{"type": "data", "data": 1}],
        },
        {
            "protocol": EXTERNAL_AGENT_ENVELOPE_VERSION,
            "peer_id": "worker",
            "message_id": "msg-1",
            "parts": [{"type": "text", "text": "hello"}],
            "result": 1,
        },
        {
            "protocol": EXTERNAL_AGENT_ENVELOPE_VERSION,
            "peer_id": "worker",
            "message_id": "msg-1",
            "parts": [{"type": "text", "text": "hello"}],
            "result": {"state": "completed", "metadata": []},
        },
        {
            "protocol": EXTERNAL_AGENT_ENVELOPE_VERSION,
            "peer_id": "worker",
            "message_id": "msg-1",
            "parts": [{"type": "text", "text": "hello"}],
            "metadata": 1,
        },
        {
            "protocol": EXTERNAL_AGENT_ENVELOPE_VERSION,
            "peer_id": "worker",
            "message_id": "msg-1",
            "parts": [{"type": "text", "text": "hello"}],
            "metadata": [],
        },
        {
            "protocol": EXTERNAL_AGENT_ENVELOPE_VERSION,
            "peer_id": "worker",
            "message_id": "msg-1",
            "parts": [{"type": "text", "text": "hello"}],
            "metadata": "",
        },
        {
            "protocol": EXTERNAL_AGENT_ENVELOPE_VERSION,
            "peer_id": "worker",
            "message_id": "msg-1",
            "parts": [{"type": "text", "text": "hello"}],
            "metadata": 0,
        },
        {
            "protocol": EXTERNAL_AGENT_ENVELOPE_VERSION,
            "peer_id": "worker",
            "message_id": "msg-1",
            "parts": [{"type": "data", "data": []}],
        },
    ],
)
def test_external_agent_envelope_rejects_bad_json_shapes(payload: dict) -> None:
    with pytest.raises(ValueError):
        validate_external_agent_envelope(payload)


def test_outbox_request_converts_to_external_agent_envelope() -> None:
    traceparent = new_traceparent()
    request = OutboxRequest(
        destination="worker",
        payload={"text": "please do X", "task_id": "task-1"},
        id="outbox-1",
        idempotency_key="message-1",
        correlation_id="corr-1",
        causation_id="cause-1",
        token_ref="lease-handle-1",
        traceparent=traceparent,
    )

    envelope = external_agent_envelope_from_outbox_request(request)

    assert envelope.peer_id == "worker"
    assert envelope.message_id == "message-1"
    assert envelope.parts[0].text == "please do X"
    assert envelope.capability_ref == "lease-handle-1"
    assert trace_id_of(envelope.traceparent) == trace_id_of(traceparent)


@pytest.mark.parametrize("field", ["task_id", "request_id", "reply_to_id"])
@pytest.mark.parametrize("invalid", [1, {"id": "forged"}, ["forged"]])
def test_outbox_request_rejects_non_string_external_agent_identifiers(
    field: str, invalid: object
) -> None:
    request = OutboxRequest(
        destination="worker",
        payload={"text": "please do X", field: invalid},
        id="outbox-1",
        idempotency_key="message-1",
    )

    with pytest.raises(ValueError, match=rf"{field} must be a string"):
        external_agent_envelope_from_outbox_request(request)


def test_outbox_request_converts_to_external_agent_envelope_with_sender_peer_id() -> None:
    request = OutboxRequest(
        destination="worker",
        payload={"text": "please do X"},
        id="outbox-1",
        idempotency_key="message-1",
        run_id="run-planner",
    )

    envelope = external_agent_envelope_from_outbox_request(request, peer_id="planner")

    assert envelope.peer_id == "planner"


def test_outbox_request_sender_peer_id_ignores_payload_metadata_identity() -> None:
    request = OutboxRequest(
        destination="worker",
        payload={
            "text": "please do X",
            "metadata": {"peer_id": "spoofed", "source_peer_id": "spoofed-source"},
        },
        id="outbox-1",
        idempotency_key="message-1",
        run_id="run-planner",
    )

    envelope = external_agent_envelope_from_outbox_request(request)

    assert envelope.peer_id == "run-planner"
    assert envelope.metadata["peer_id"] == "spoofed"


def test_outbox_request_ignores_non_object_metadata_for_text_message() -> None:
    request = OutboxRequest(
        destination="worker",
        payload={"text": "please do X", "metadata": "v1"},
        id="outbox-1",
        idempotency_key="message-1",
    )

    envelope = external_agent_envelope_from_outbox_request(request)

    assert envelope.parts[0].text == "please do X"
    assert envelope.metadata == {}


def test_merge_canonical_metadata_preserves_user_non_reserved_keys() -> None:
    assert RESERVED_EXTERNAL_AGENT_METADATA_KEYS == frozenset(
        {"peer_id", "task_id", "request_id", "reply_to_id", "result", "traceparent"}
    )

    merged = merge_canonical_metadata(
        {
            "custom": "kept",
            "peer_id": "spoofed",
            "task_id": "spoofed",
            "request_id": "spoofed",
            "reply_to_id": "spoofed",
            "result": {"state": "spoofed"},
            "traceparent": "spoofed",
        },
        {
            "peer_id": "planner",
            "task_id": "task-1",
            "request_id": "request-1",
            "reply_to_id": "reply-1",
            "result": {"state": "completed"},
            "traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01",
        },
    )

    assert merged["custom"] == "kept"
    assert merged["peer_id"] == "planner"
    assert merged["task_id"] == "task-1"
    assert merged["request_id"] == "request-1"
    assert merged["reply_to_id"] == "reply-1"
    assert merged["result"] == {"state": "completed"}
    assert merged["traceparent"].startswith("00-")


def test_external_agent_envelope_converts_to_inbox_message() -> None:
    envelope = ExternalAgentEnvelope(
        peer_id="planner",
        message_id="message-1",
        task_id="task-1",
        correlation_id="corr-1",
        causation_id="cause-1",
        parts=(ExternalAgentPart(type="text", text="done"),),
    )

    inbox = external_agent_envelope_to_inbox_message(envelope, run_id="run-1")

    assert inbox.id == "message-1"
    assert inbox.content == "done"
    assert inbox.source == "external-agent:planner"
    assert inbox.type == "external_agent_message"
    assert inbox.metadata["task_id"] == "task-1"


def test_external_agent_envelope_canonical_metadata_overrides_user_metadata() -> None:
    envelope = ExternalAgentEnvelope(
        peer_id="planner",
        message_id="message-1",
        task_id="task-1",
        request_id="request-1",
        reply_to_id="reply-1",
        correlation_id="corr-1",
        causation_id="cause-1",
        parts=(ExternalAgentPart(type="text", text="done"),),
        result=ExternalAgentResult(
            state="completed",
            terminal=True,
            error=ExternalAgentError(code="none", message=""),
        ),
        metadata={
            "custom": "ok",
            "peer_id": "spoofed",
            "task_id": "spoofed",
            "traceparent": "spoofed",
            "result": {"state": "spoofed"},
        },
    )

    inbox = external_agent_envelope_to_inbox_message(envelope, run_id="run-1")

    assert inbox.metadata["custom"] == "ok"
    assert inbox.metadata["peer_id"] == "planner"
    assert inbox.metadata["task_id"] == "task-1"
    assert inbox.metadata["request_id"] == "request-1"
    assert inbox.metadata["reply_to_id"] == "reply-1"
    assert inbox.metadata["result"]["state"] == "completed"
    assert inbox.metadata["traceparent"] == envelope.traceparent


def test_external_agent_data_parts_convert_to_supported_inbox_content() -> None:
    envelope = ExternalAgentEnvelope(
        peer_id="planner",
        message_id="message-1",
        parts=(
            ExternalAgentPart(type="text", text="payload follows"),
            ExternalAgentPart(type="data", data={"answer": 42}),
        ),
    )

    inbox = external_agent_envelope_to_inbox_message(envelope, run_id="run-1")

    assert inbox.content == [
        {"type": "text", "text": "payload follows"},
        {"type": "text", "text": '{"answer": 42}'},
    ]
    assert json.loads(inbox.content[1]["text"]) == {"answer": 42}


def test_external_agent_error_normalization() -> None:
    error = normalize_external_agent_error(
        RuntimeError("peer unavailable"),
        code="peer_unavailable",
        retryable=True,
    )

    assert error.to_json() == {
        "code": "peer_unavailable",
        "message": "peer unavailable",
        "retryable": True,
    }
