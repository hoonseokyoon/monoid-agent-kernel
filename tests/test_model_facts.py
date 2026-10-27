from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from support.model_facts import facts
from monoid_agent_kernel import (
    ModelCallTermination,
    UsageEvidence,
    UsageEvidenceAttempt,
    UsageObservation,
)
from monoid_agent_kernel.core.model_facts import USAGE_COUNTERS
from monoid_agent_kernel.core.model_io import ModelCallAttempt, ModelCallReceipt
from monoid_agent_kernel.core.model_stream import ModelStreamOutcome


def test_sparse_observations_preserve_zero_and_counter_coverage() -> None:
    explicit = UsageObservation(
        reported={"input_tokens": 5, "output_tokens": 0}, source="provider", collection="drain"
    )
    legacy = UsageObservation(source="legacy_ambiguous")
    evidence = UsageEvidence(
        attempts=(
            UsageEvidenceAttempt(index=1, observation=explicit),
            UsageEvidenceAttempt(index=2, observation=legacy),
        )
    )
    assert evidence.reported == {"input_tokens": 5, "output_tokens": 0}
    assert evidence.completeness["input_tokens"] == "partial"
    assert evidence.completeness["output_tokens"] == "partial"
    assert evidence.completeness["total_tokens"] == "unknown"
    assert set(UsageEvidence().completeness.values()) == {"no_dispatch"}
    assert set(evidence.completeness) == set(USAGE_COUNTERS)
    assert UsageEvidence.from_json(evidence.to_json()) == evidence
    receipt = ModelCallReceipt(attempts=2, usage={"input_tokens": 8}, usage_evidence=evidence)
    assert ModelCallReceipt.from_json(receipt.to_json()) == receipt


@pytest.mark.parametrize(
    "reported",
    [{"input_tokens": True}, {"input_tokens": -1}, {"input_tokens": 1.0}, {"private_counter": 1}],
)
def test_observations_refuse_invalid_counts_and_private_keys(reported) -> None:
    with pytest.raises(ValueError):
        UsageObservation(reported=reported, source="provider", collection="normal")


@pytest.mark.parametrize(
    ("reported", "source", "collection"),
    [
        ({"input_tokens": 0}, None, None),
        ({"input_tokens": 1}, "legacy_ambiguous", "normal"),
        ({"input_tokens": 1}, "provider", None),
        ({}, "provider", "normal"),
        ({}, None, "drain"),
    ],
)
def test_observations_refuse_unjustified_source_or_collection(reported, source, collection) -> None:
    with pytest.raises(ValueError):
        UsageObservation(reported=reported, source=source, collection=collection)


@pytest.mark.parametrize(
    "mutation", ["aggregate", "coverage", "index", "source", "private", "count"]
)
def test_evidence_wire_refuses_contradictions(mutation: str) -> None:
    raw = facts({"input_tokens": 2})["usage_evidence"]
    if mutation == "aggregate":
        raw["reported"]["input_tokens"] = 3
    elif mutation == "coverage":
        raw["completeness"]["output_tokens"] = "complete"
    elif mutation == "index":
        raw["attempts"][0]["index"] = 2
    elif mutation == "source":
        raw["attempts"][0]["source"] = "legacy_ambiguous"
    elif mutation == "private":
        raw["attempts"][0]["prompt"] = "secret"
    else:
        raw["attempts"][0]["reported"]["input_tokens"] = True
    with pytest.raises(ValueError):
        UsageEvidence.from_json(raw)


def test_receipt_evidence_matches_counts_numeric_projection_and_itemized_observations() -> None:
    evidence = UsageEvidence.from_json(
        facts({"input_tokens": 2, "output_tokens": 0})["usage_evidence"]
    )
    observed = evidence.attempts[0].observation
    receipt = ModelCallReceipt(
        usage={"input_tokens": 2},
        usage_evidence=evidence,
        attempt_log=(ModelCallAttempt(usage={"input_tokens": 2}, usage_evidence=observed),),
    )
    assert ModelCallReceipt.from_json(receipt.to_json()) == receipt
    for changes in (
        {"attempts": 2, "attempt_log": ()},
        {"usage": {"input_tokens": 3}, "attempt_log": ()},
        {
            "attempt_log": (
                ModelCallAttempt(usage={"input_tokens": 2}, usage_evidence=UsageObservation()),
            )
        },
    ):
        with pytest.raises(ValueError):
            replace(receipt, **changes)
    with pytest.raises(ValueError):
        ModelCallAttempt(usage={"input_tokens": 1}, usage_evidence=observed)


@pytest.mark.parametrize("field", ["termination", "usage_evidence"])
def test_optional_facts_omit_absence_and_refuse_explicit_null(field: str) -> None:
    raw = ModelCallReceipt().to_json()
    assert field not in raw
    with pytest.raises(ValueError):
        ModelCallReceipt.from_json({**raw, field: None})


def test_stream_fact_outcome_agrees_with_existing_status() -> None:
    termination = ModelCallTermination(
        outcome="cancelled", requested_by="user_cancel", ended_by="provider_terminal"
    )
    outcome = ModelStreamOutcome(
        status="cancelled", termination=termination, usage_evidence=UsageEvidence()
    )
    assert outcome.termination is termination
    with pytest.raises(ValueError):
        replace(outcome, status="completed")
    with pytest.raises(ValueError):
        ModelCallTermination.from_json({**termination.to_json(), "ended_by": "provider_secret"})


def test_stream_numeric_usage_keeps_final_turn_scope_with_whole_call_evidence() -> None:
    evidence = UsageEvidence(attempts=tuple(
        UsageEvidenceAttempt(index=index, observation=UsageObservation(
            reported={"input_tokens": count}, source="provider", collection="normal",
        )) for index, count in enumerate((2, 3), 1)
    ))
    outcome = ModelStreamOutcome(status="completed", usage={"input_tokens": 3}, usage_evidence=evidence)
    assert outcome.usage == {"input_tokens": 3}
    assert outcome.usage_evidence.reported == {"input_tokens": 5}


def test_fact_input_and_output_mappings_are_detached() -> None:
    raw = facts({"input_tokens": 2})["usage_evidence"]
    expected = deepcopy(raw)
    evidence = UsageEvidence.from_json(raw)
    raw["attempts"][0]["reported"]["input_tokens"] = 40
    exported = evidence.to_json()
    exported["attempts"][0]["reported"]["input_tokens"] = 90
    exported["reported"]["input_tokens"] = 80
    assert evidence.to_json() == expected


def test_with_error_preserves_an_explicit_zero_without_replacing_the_original_failure() -> None:
    from monoid_agent_kernel.providers.base import mark_provider_usage

    receipt = ModelCallReceipt(usage_evidence=UsageEvidence.from_json(
        facts({"output_tokens": 0})["usage_evidence"],
    ))
    error = RuntimeError("later failure")
    mark_provider_usage(error, {"output_tokens": 5})
    failed = receipt.with_error(error)
    assert failed.error_code == "RuntimeError"
    assert failed.usage == {}
    assert failed.usage_evidence.reported == {"output_tokens": 0}


@pytest.mark.parametrize("reported", [{}, {"output_tokens": 0}])
def test_with_error_keeps_numeric_stamps_compatible_with_existing_observations(reported) -> None:
    from monoid_agent_kernel.providers.base import mark_provider_usage

    receipt = ModelCallReceipt(usage_evidence=UsageEvidence.from_json(facts(reported)["usage_evidence"]))
    error = RuntimeError("billed failure")
    mark_provider_usage(error, {"input_tokens": 5})
    failed = receipt.with_error(error)
    assert failed.usage == {"input_tokens": 5}
    assert failed.usage_evidence == receipt.usage_evidence
    assert failed.usage_evidence.completeness["input_tokens"] == "unknown"


def test_observer_cannot_mutate_shared_facts_or_stop_later_receipt_delivery() -> None:
    from monoid_agent_kernel.core.model_io import ModelIOSubscription, dispatch_model_call

    evidence = UsageEvidence.from_json(facts({"input_tokens": 1})["usage_evidence"])
    receipt = ModelCallReceipt(usage={"input_tokens": 1}, usage_evidence=evidence)
    received = []

    class Mutator:
        def on_model_call(self, capture):
            capture.receipt.usage_evidence.attempts[0].observation.reported["input_tokens"] = 9

    class Reader:
        def on_model_call(self, capture):
            received.append(capture)

    result = dispatch_model_call(receipt=receipt, content={}, subscriptions=(
        ModelIOSubscription(Mutator()), ModelIOSubscription(Reader()),
    ))
    assert result.usage_evidence.reported == {"input_tokens": 1}
    assert received[0].receipt.usage_evidence.reported == {"input_tokens": 1}
