from __future__ import annotations

import json

import pytest

from monoid_agent_kernel.core.external_agent_envelope import (
    EXTERNAL_AGENT_ENVELOPE_VERSION,
    ExternalAgentEnvelope,
    ExternalAgentError,
    ExternalAgentPart,
    ExternalAgentResult,
    external_agent_envelope_from_outbox_request,
    external_agent_envelope_to_inbox_message,
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
            "metadata": 1,
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
