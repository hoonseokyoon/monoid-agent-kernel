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
