"""W3C Trace Context helpers + envelope propagation (observability metadata, never behavioral)."""

from __future__ import annotations

from monoid_agent_kernel.core.inbox import InboxMessage
from monoid_agent_kernel.core.outbox import OutboxRequest
from monoid_agent_kernel.core.trace_context import (
    child_traceparent,
    new_traceparent,
    parse_traceparent,
    trace_id_of,
)


# --- pure helpers -------------------------------------------------------------------------


def test_new_traceparent_is_well_formed() -> None:
    tp = new_traceparent()
    parsed = parse_traceparent(tp)
    assert parsed is not None
    assert parsed["version"] == "00"
    assert len(parsed["trace_id"]) == 32 and len(parsed["span_id"]) == 16
    assert parsed["flags"] == "01"  # sampled by default
    assert parse_traceparent(new_traceparent(sampled=False))["flags"] == "00"


def test_parse_rejects_malformed_and_all_zero() -> None:
    assert parse_traceparent(None) is None
    assert parse_traceparent("") is None
    assert parse_traceparent("not-a-traceparent") is None
    assert parse_traceparent("00-abc-def-01") is None  # wrong hex lengths
    assert parse_traceparent("00-" + "z" * 32 + "-" + "0" * 16 + "-01") is None  # non-hex
    # All-zero trace or span id is invalid per spec.
    assert parse_traceparent("00-" + "0" * 32 + "-" + "f" * 16 + "-01") is None
    assert parse_traceparent("00-" + "f" * 32 + "-" + "0" * 16 + "-01") is None


def test_child_keeps_trace_id_and_changes_span() -> None:
    parent = new_traceparent()
    child = child_traceparent(parent)
    p, c = parse_traceparent(parent), parse_traceparent(child)
    assert p["trace_id"] == c["trace_id"]  # same end-to-end trace
    assert p["span_id"] != c["span_id"]  # a new span
    assert p["flags"] == c["flags"]


def test_child_of_missing_mints_fresh_root() -> None:
    child = child_traceparent("")
    assert parse_traceparent(child) is not None  # always a valid traceparent
    assert trace_id_of(child) and trace_id_of("garbage") == ""


# --- envelopes round-trip the fields + tolerate their absence (back-compat) ----------------


def test_inbox_envelope_round_trips_trace_fields() -> None:
    tp = new_traceparent()
    env = InboxMessage(content="hi", traceparent=tp, tracestate="vendor=1")
    back = InboxMessage.from_json(env.to_json())
    assert back.traceparent == tp and back.tracestate == "vendor=1"
    # An old payload without the fields decodes to empty defaults.
    legacy = InboxMessage.from_json({"content": "x", "id": "m1"})
    assert legacy.traceparent == "" and legacy.tracestate == ""


def test_outbox_request_round_trips_trace_fields() -> None:
    tp = new_traceparent()
    req = OutboxRequest(destination="email", traceparent=tp, tracestate="vendor=1")
    back = OutboxRequest.from_json(req.to_json())
    assert back.traceparent == tp and back.tracestate == "vendor=1"
    legacy = OutboxRequest.from_json({"destination": "email", "id": "o1"})
    assert legacy.traceparent == "" and legacy.tracestate == ""
