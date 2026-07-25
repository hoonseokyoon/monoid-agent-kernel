"""InvocationContext: the provenance of one model call, carried without requiring a run."""

from __future__ import annotations

import json

import pytest

from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.trace_context import parse_traceparent
from monoid_agent_kernel.core.wire_validation import WireValidationError

TRACEPARENT = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def test_an_anonymous_context_is_legal() -> None:
    """A caller with no provenance to declare must still be able to make a model call.

    The whole point of the type is that the runner serves callers without an agent run, so a
    required ``run_id`` would defeat it.
    """
    context = InvocationContext()

    assert context.run_id == ""
    assert context.skill_id == ""
    assert context.attempt == 1
    assert dict(context.attributes) == {}


def test_attempt_counts_from_one() -> None:
    assert InvocationContext(attempt=1).attempt == 1
    assert InvocationContext(attempt=7).attempt == 7
    with pytest.raises(ValueError, match="attempt must be 1 or greater"):
        InvocationContext(attempt=0)


def test_attributes_must_be_strings_both_sides() -> None:
    with pytest.raises(WireValidationError, match="mapping of str to str"):
        InvocationContext(attributes={"tenant": 7})  # type: ignore[dict-item]
    with pytest.raises(WireValidationError, match="mapping of str to str"):
        InvocationContext(attributes={7: "tenant"})  # type: ignore[dict-item]


def test_attributes_are_copied_away_from_the_caller() -> None:
    """The kernel treats a context as immutable, so it cannot share a dict the caller still holds."""
    supplied = {"tenant": "acme"}
    context = InvocationContext(attributes=supplied)

    supplied["tenant"] = "other"
    supplied["leaked"] = "yes"

    assert dict(context.attributes) == {"tenant": "acme"}


def test_json_round_trip_preserves_every_field() -> None:
    context = InvocationContext(
        run_id="run_1",
        skill_id="lecture-note",
        skill_digest="sha256:abc",
        step_id="draft",
        attempt=3,
        batch_id="batch_9",
        item_id="item_4",
        case_id="case_2",
        traceparent=TRACEPARENT,
        tracestate="vendor=opaque",
        attributes={"tenant": "acme"},
    )

    # Through actual JSON text, not just the dict: the payload has to survive a checkpoint.
    restored = InvocationContext.from_json(json.loads(json.dumps(context.to_json())))

    assert restored == context


def test_from_json_defaults_a_missing_attempt_to_one() -> None:
    assert InvocationContext.from_json({"run_id": "run_1"}) == InvocationContext(run_id="run_1")


def test_from_json_rejects_a_non_object_and_a_mistyped_field() -> None:
    with pytest.raises(WireValidationError):
        InvocationContext.from_json(["not", "an", "object"])
    with pytest.raises(WireValidationError):
        InvocationContext.from_json({"attempt": "3"})
    with pytest.raises(WireValidationError):
        InvocationContext.from_json({"attributes": "not-an-object"})


@pytest.mark.parametrize("attributes", [[], 0, False, ""])
def test_from_json_rejects_falsy_attributes_of_the_wrong_type(attributes: object) -> None:
    """Same absent-vs-falsy rule as `core.model_io`, so the two halves of one receipt payload cannot
    disagree about what a malformed field means."""
    with pytest.raises(WireValidationError):
        InvocationContext.from_json({"attributes": attributes})


def test_from_json_treats_absent_and_null_attributes_as_empty() -> None:
    assert InvocationContext.from_json({}) == InvocationContext()
    assert InvocationContext.from_json({"attributes": None}) == InvocationContext()


# --- Trace context: carried, never enforced -------------------------------------------------


def test_trace_accessors_read_a_valid_traceparent() -> None:
    context = InvocationContext(traceparent=TRACEPARENT)

    assert context.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert context.span_id == "00f067aa0ba902b7"


@pytest.mark.parametrize(
    "traceparent",
    [
        "",
        "not-a-traceparent",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",  # too few parts
        "00-00000000000000000000000000000000-00f067aa0ba902b7-01",  # all-zero trace id
        "00-4bf92f3577b34da6a3ce929d0e0e4736-zzzzzzzzzzzzzzzz-01",  # non-hex span id
    ],
)
def test_an_unusable_traceparent_is_carried_not_rejected(traceparent: str) -> None:
    """Observability metadata must never be able to fail a call.

    ``core.trace_context`` is deliberately tolerant, and this type inherits that: construction
    stores whatever it is given and the derived accessors report ``""``. A caller that hands over a
    header from an untrusted peer therefore cannot turn a bad string into a failed model call.
    """
    context = InvocationContext(traceparent=traceparent)

    assert context.traceparent == traceparent
    assert context.trace_id == ""
    assert context.span_id == ""


def test_child_span_keeps_the_trace_and_advances_the_span() -> None:
    parent = InvocationContext(
        run_id="run_1",
        traceparent=TRACEPARENT,
        tracestate="vendor=opaque",
        attributes={"tenant": "acme"},
    )

    child = parent.child_span()

    assert child.trace_id == parent.trace_id
    assert child.span_id != parent.span_id
    # tracestate is an opaque vendor list, propagated verbatim rather than rewritten.
    assert child.tracestate == "vendor=opaque"
    # Everything that is not the span is untouched.
    assert child.to_json() | {"traceparent": TRACEPARENT} == parent.to_json()


@pytest.mark.parametrize("traceparent", ["", "not-a-traceparent"])
def test_child_span_mints_a_root_when_there_is_nothing_to_descend_from(traceparent: str) -> None:
    """``child_span`` doubles as "give me a trace to hang spans off"."""
    child = InvocationContext(traceparent=traceparent).child_span()

    assert parse_traceparent(child.traceparent) is not None
    assert child.trace_id != ""


def test_child_span_is_a_new_value() -> None:
    parent = InvocationContext(traceparent=TRACEPARENT)

    parent.child_span()

    assert parent.traceparent == TRACEPARENT


def test_with_attributes_merges_over_the_existing_ones() -> None:
    context = InvocationContext(attributes={"tenant": "acme", "region": "eu"})

    merged = context.with_attributes(region="us", locale="ko")

    assert dict(merged.attributes) == {"tenant": "acme", "region": "us", "locale": "ko"}
    assert dict(context.attributes) == {"tenant": "acme", "region": "eu"}


def test_with_attributes_still_rejects_a_non_string_value() -> None:
    with pytest.raises(WireValidationError, match="mapping of str to str"):
        InvocationContext().with_attributes(tenant=7)  # type: ignore[arg-type]
