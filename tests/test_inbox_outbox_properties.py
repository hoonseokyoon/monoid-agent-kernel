from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from monoid_agent_kernel.core.inbox import InboxMessage
from monoid_agent_kernel.core.outbox import OutboxRequest


_json_scalar = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False, width=32)
    | st.text(max_size=32)
)
_json_value: st.SearchStrategy[Any] = st.recursive(
    _json_scalar,
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.text(max_size=24), children, max_size=4),
    max_leaves=12,
)
_json_object = st.dictionaries(st.text(max_size=24), _json_value, max_size=4)
_wire_id = (
    st.text(
        alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
        min_size=1,
        max_size=24,
    )
    .map(str.strip)
    .filter(bool)
)
_bad_object_shape = _json_value.filter(lambda value: not isinstance(value, dict))
_bad_bool_shape = _json_value.filter(lambda value: not isinstance(value, bool))


@st.composite
def _inbox_messages(draw: st.DrawFn) -> InboxMessage:
    content: str | list[dict[str, Any]]
    if draw(st.booleans()):
        content = draw(st.text(max_size=128))
    else:
        content = draw(st.lists(_json_object, max_size=4))
    return InboxMessage(
        content=content,
        id=draw(_wire_id),
        source=draw(st.text(max_size=24)),
        type=draw(st.text(max_size=24)),
        run_id=draw(st.text(max_size=24)),
        created_at=draw(st.floats(allow_nan=False, allow_infinity=False, width=32)),
        correlation_id=draw(st.text(max_size=24)),
        causation_id=draw(st.text(max_size=24)),
        traceparent=draw(st.text(max_size=64)),
        tracestate=draw(st.text(max_size=64)),
        metadata=draw(_json_object),
    )


@st.composite
def _outbox_requests(draw: st.DrawFn) -> OutboxRequest:
    return OutboxRequest(
        destination=draw(st.text(max_size=32)),
        payload=draw(_json_object),
        capability=draw(st.text(max_size=32)),
        token_ref=draw(st.text(max_size=32)),
        run_id=draw(st.text(max_size=32)),
        id=draw(_wire_id),
        idempotency_key=draw(_wire_id),
        correlation_id=draw(st.text(max_size=32)),
        causation_id=draw(st.text(max_size=32)),
        expect_ack=draw(st.booleans()),
        reply_to=draw(st.text(max_size=32)),
        traceparent=draw(st.text(max_size=64)),
        tracestate=draw(st.text(max_size=64)),
        created_at=draw(st.floats(allow_nan=False, allow_infinity=False, width=32)),
        status=draw(st.sampled_from(("pending", "dispatched", "failed"))),
        attempts=draw(st.integers(min_value=0, max_value=10)),
        next_attempt_at=draw(st.floats(allow_nan=False, allow_infinity=False, width=32)),
        reference=draw(st.text(max_size=32)),
        error=draw(st.text(max_size=64)),
    )


@settings(max_examples=40, deadline=None)
@given(message=_inbox_messages())
def test_inbox_message_valid_payload_round_trips(message: InboxMessage) -> None:
    payload = message.to_json()

    parsed = InboxMessage.from_json(payload)

    assert parsed.to_json() == payload


@settings(max_examples=40, deadline=None)
@given(request=_outbox_requests())
def test_outbox_request_valid_payload_round_trips(request: OutboxRequest) -> None:
    payload = request.to_json()

    parsed = OutboxRequest.from_json(payload)

    assert parsed.to_json() == payload


@settings(max_examples=25, deadline=None)
@given(metadata=_bad_object_shape)
def test_inbox_message_rejects_malformed_metadata(metadata: Any) -> None:
    payload = InboxMessage(content="hello", id="m1").to_json()
    payload["metadata"] = metadata

    with pytest.raises(ValueError):
        InboxMessage.from_json(payload)


@settings(max_examples=25, deadline=None)
@given(payload_value=_bad_object_shape)
def test_outbox_request_rejects_malformed_payload(payload_value: Any) -> None:
    payload = OutboxRequest(destination="email", id="o1").to_json()
    payload["payload"] = payload_value

    with pytest.raises(ValueError):
        OutboxRequest.from_json(payload)


@settings(max_examples=25, deadline=None)
@given(expect_ack=_bad_bool_shape)
def test_outbox_request_rejects_malformed_expect_ack(expect_ack: Any) -> None:
    payload = OutboxRequest(destination="email", id="o1").to_json()
    payload["expect_ack"] = expect_ack

    with pytest.raises(ValueError):
        OutboxRequest.from_json(payload)
