from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from monoid_agent_kernel.core.external_agent_envelope import (
    ExternalAgentEnvelope,
    ExternalAgentError,
    ExternalAgentPart,
    ExternalAgentResult,
    external_agent_envelope_to_inbox_message,
    validate_external_agent_envelope,
)


_json_scalar = st.none() | st.booleans() | st.integers() | st.floats(
    allow_nan=False,
    allow_infinity=False,
    width=32,
) | st.text(max_size=32)
_json_value: st.SearchStrategy[Any] = st.recursive(
    _json_scalar,
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.text(max_size=24), children, max_size=4),
    max_leaves=12,
)
_json_object = st.dictionaries(st.text(max_size=24), _json_value, max_size=4)
_wire_id = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    min_size=1,
    max_size=24,
).map(str.strip).filter(bool)
_bad_object_shape = _json_value.filter(lambda value: not isinstance(value, dict))
_bad_optional_object_shape = _bad_object_shape.filter(lambda value: value is not None)


@st.composite
def _external_agent_parts(draw: st.DrawFn) -> tuple[ExternalAgentPart, ...]:
    part_count = draw(st.integers(min_value=1, max_value=4))
    parts: list[ExternalAgentPart] = []
    for _ in range(part_count):
        parts.append(
            ExternalAgentPart(
                type=draw(_wire_id),
                text=draw(st.text(max_size=64)),
                data=draw(_json_object),
                artifact_id=draw(st.text(max_size=32)),
                mime_type=draw(st.text(max_size=32)),
            )
        )
    return tuple(parts)


@st.composite
def _external_agent_errors(draw: st.DrawFn) -> ExternalAgentError:
    return ExternalAgentError(
        code=draw(_wire_id),
        message=draw(st.text(max_size=64)),
        retryable=draw(st.booleans()),
    )


@st.composite
def _external_agent_results(draw: st.DrawFn) -> ExternalAgentResult:
    return ExternalAgentResult(
        state=draw(_wire_id),
        terminal=draw(st.booleans()),
        interrupted=draw(st.booleans()),
        error=draw(st.none() | _external_agent_errors()),
        metadata=draw(_json_object),
    )


@st.composite
def _external_agent_envelopes(draw: st.DrawFn) -> ExternalAgentEnvelope:
    return ExternalAgentEnvelope(
        peer_id=draw(_wire_id),
        parts=draw(_external_agent_parts()),
        message_id=draw(_wire_id),
        task_id=draw(_wire_id),
        request_id=draw(st.text(max_size=24)),
        reply_to_id=draw(st.text(max_size=24)),
        correlation_id=draw(_wire_id),
        causation_id=draw(st.text(max_size=24)),
        traceparent=draw(st.text(max_size=64)),
        tracestate=draw(st.text(max_size=64)),
        capability_ref=draw(st.text(max_size=24)),
        result=draw(st.none() | _external_agent_results()),
        created_at=draw(st.floats(allow_nan=False, allow_infinity=False, width=32)),
        metadata=draw(_json_object),
    )


def _valid_envelope_payload() -> dict[str, Any]:
    return ExternalAgentEnvelope(
        peer_id="worker",
        message_id="message-1",
        task_id="task-1",
        correlation_id="corr-1",
        parts=(ExternalAgentPart(type="text", text="hello"),),
    ).to_json()


@settings(max_examples=40, deadline=None)
@given(envelope=_external_agent_envelopes())
def test_external_agent_envelope_valid_payload_round_trips(envelope: ExternalAgentEnvelope) -> None:
    payload = json.loads(json.dumps(envelope.to_json()))

    parsed = validate_external_agent_envelope(payload)

    assert parsed.to_json() == payload


@settings(max_examples=25, deadline=None)
@given(metadata=_bad_object_shape)
def test_external_agent_envelope_rejects_malformed_metadata(metadata: Any) -> None:
    payload = _valid_envelope_payload()
    payload["metadata"] = metadata

    with pytest.raises(ValueError):
        validate_external_agent_envelope(payload)


@settings(max_examples=25, deadline=None)
@given(data=_bad_object_shape)
def test_external_agent_envelope_rejects_malformed_part_data(data: Any) -> None:
    payload = _valid_envelope_payload()
    payload["parts"] = [{"type": "data", "data": data}]

    with pytest.raises(ValueError):
        validate_external_agent_envelope(payload)


@settings(max_examples=25, deadline=None)
@given(result=_bad_optional_object_shape)
def test_external_agent_envelope_rejects_malformed_result_shape(result: Any) -> None:
    payload = _valid_envelope_payload()
    payload["result"] = result

    with pytest.raises(ValueError):
        validate_external_agent_envelope(payload)


@settings(max_examples=25, deadline=None)
@given(metadata=_bad_object_shape)
def test_external_agent_envelope_rejects_malformed_result_metadata(metadata: Any) -> None:
    payload = _valid_envelope_payload()
    payload["result"] = {"state": "completed", "metadata": metadata}

    with pytest.raises(ValueError):
        validate_external_agent_envelope(payload)


@settings(max_examples=40, deadline=None)
@given(
    envelope_metadata=_json_object,
    peer_id=_wire_id,
    task_id=_wire_id,
    result=st.none() | _external_agent_results(),
)
def test_external_agent_envelope_metadata_cannot_override_canonical_inbox_metadata(
    envelope_metadata: dict[str, Any],
    peer_id: str,
    task_id: str,
    result: ExternalAgentResult | None,
) -> None:
    envelope_metadata = {
        **envelope_metadata,
        "custom": "kept",
        "peer_id": "spoofed-peer",
        "task_id": "spoofed-task",
        "result": {"state": "spoofed"},
    }
    envelope = ExternalAgentEnvelope(
        peer_id=peer_id,
        message_id="message-1",
        task_id=task_id,
        correlation_id="corr-1",
        parts=(ExternalAgentPart(type="text", text="done"),),
        result=result,
        metadata=envelope_metadata,
    )

    inbox = external_agent_envelope_to_inbox_message(envelope, run_id="run-1")

    assert inbox.metadata["custom"] == "kept"
    assert inbox.metadata["peer_id"] == peer_id
    assert inbox.metadata["task_id"] == task_id
    assert inbox.metadata["result"] == (result.to_json() if result is not None else None)
