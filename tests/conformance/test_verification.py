from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import replace
from typing import Any

import pytest

from monoid_agent_kernel.conformance.fixtures import load_compatibility_fixtures
from monoid_agent_kernel.conformance.provenance import (
    ConformanceEvent,
    ConformanceEvidenceBundle,
    ConformanceEvidenceReference,
    ConformanceTarget,
    build_evidence_reference,
    case_id_sha256,
    serialize_conformance_evidence,
)
from monoid_agent_kernel.conformance.report import (
    CONFORMANCE_REPORT_V2,
    ConformanceReport,
    ConformanceRuleOutcome,
    decode_conformance_report,
    observation,
    outcome_from_observations,
)
from monoid_agent_kernel.conformance.verification import (
    ConformanceVerificationResult,
    verify_conformance_report,
)

_EVIDENCE_ID = "minimal-agent.lifecycle"


def _target(implementation_id: str = "vendor.runtime") -> ConformanceTarget:
    return ConformanceTarget(
        implementation_id=implementation_id,
        implementation_version="1.2.3",
        adapter_id="vendor.monoid-adapter",
        adapter_version="2.0.0",
    )


def _bundle(**changes: Any) -> ConformanceEvidenceBundle:
    base = ConformanceEvidenceBundle(
        profile_id="minimal-agent",
        target=_target(),
        case_id_sha256=case_id_sha256("private-run-id"),
        run_id_present=True,
        submitted=True,
        states=("submitted", "running", "completed"),
        result_run_id_matches=True,
        result_status="completed",
        events=(
            ConformanceEvent(seq=1, event_type="run.started"),
            ConformanceEvent(seq=2, event_type="run.finished"),
        ),
        events_complete=True,
        next_seq=3,
    )
    return replace(base, **changes)


def _outcomes(
    bundle: ConformanceEvidenceBundle,
    *,
    profile_id: str = "minimal-agent",
    evidence_id: str = _EVIDENCE_ID,
) -> tuple[ConformanceRuleOutcome, ...]:
    sequences = tuple(event.seq for event in bundle.events)
    outcomes = (
        outcome_from_observations(
            "MIN-01-SUBMISSION",
            profile_id,
            (
                observation(
                    "run_id_present",
                    expected=True,
                    actual=bundle.run_id_present,
                ),
                observation(
                    "submission_accepted",
                    expected=True,
                    actual=bundle.submitted,
                ),
            ),
        ),
        outcome_from_observations(
            "MIN-02-LIFECYCLE",
            profile_id,
            (
                observation(
                    "terminal_state",
                    expected=True,
                    actual=bool(
                        bundle.states
                        and bundle.states[-1] in {"completed", "awaiting_input"}
                    ),
                ),
                observation(
                    "running_observed",
                    expected=True,
                    actual="running" in bundle.states,
                ),
            ),
        ),
        outcome_from_observations(
            "MIN-03-RESULT",
            profile_id,
            (
                observation(
                    "result_run_id",
                    expected=True,
                    actual=bundle.result_run_id_matches,
                ),
                observation(
                    "result_status",
                    expected=True,
                    actual=bundle.result_status == "completed",
                ),
            ),
        ),
        outcome_from_observations(
            "MIN-04-EVENT-SEQUENCE",
            profile_id,
            (
                observation(
                    "events_present",
                    expected=True,
                    actual=bool(sequences),
                ),
                observation(
                    "event_sequence_contiguous",
                    expected=tuple(range(1, len(sequences) + 1)),
                    actual=sequences,
                ),
            ),
        ),
    )
    return tuple(replace(outcome, evidence_refs=(evidence_id,)) for outcome in outcomes)


def _reference(
    bundle: ConformanceEvidenceBundle,
    *,
    evidence_id: str = _EVIDENCE_ID,
) -> ConformanceEvidenceReference:
    return build_evidence_reference(
        bundle,
        evidence_id=evidence_id,
        artifact_name="minimal-agent.evidence.json",
    )


def _report(
    bundle: ConformanceEvidenceBundle,
    *,
    target: ConformanceTarget | None = None,
    profile_id: str = "minimal-agent",
    outcomes: tuple[ConformanceRuleOutcome, ...] | None = None,
    evidence: tuple[ConformanceEvidenceReference, ...] | None = None,
) -> ConformanceReport:
    reference = _reference(bundle)
    return ConformanceReport(
        harness_id="external-adapter",
        profile_id=profile_id,
        outcomes=outcomes or _outcomes(bundle, profile_id=profile_id),
        schema_version=CONFORMANCE_REPORT_V2,
        provenance_status="available",
        target=target or bundle.target,
        evidence=evidence or (reference,),
    )


def _verify(
    bundle: ConformanceEvidenceBundle,
    report: ConformanceReport | None = None,
) -> ConformanceVerificationResult:
    return verify_conformance_report(
        report or _report(bundle),
        {_EVIDENCE_ID: serialize_conformance_evidence(bundle)},
    )


def test_verifies_passing_report_and_returns_normalized_evidence() -> None:
    bundle = _bundle()

    result = _verify(bundle)

    assert result.status == "verified"
    assert result.verified is True
    assert result.report_passed is True
    assert result.issue_codes == ()
    assert result.evidence == (bundle,)


@pytest.mark.parametrize(
    "changes",
    [
        {"submitted": False},
        {"states": ("submitted",)},
        {"result_run_id_matches": False},
        {
            "events": (
                ConformanceEvent(seq=1, event_type="run.started"),
                ConformanceEvent(seq=3, event_type="run.finished"),
            ),
            "next_seq": 4,
        },
    ],
)
def test_verified_evidence_may_support_a_reported_rule_failure(
    changes: dict[str, Any],
) -> None:
    bundle = _bundle(**changes)

    result = _verify(bundle)

    assert result.status == "verified"
    assert result.report_passed is False


@pytest.mark.parametrize("outcome_index", range(4))
def test_internally_consistent_rule_tampering_fails_semantic_verification(
    outcome_index: int,
) -> None:
    bundle = _bundle()
    report = _report(bundle)
    original = report.outcomes[outcome_index]
    observations = list(original.observations)
    first = observations[0]
    observations[0] = observation(
        first.observation_id,
        expected=first.expected,
        actual=False,
        detail=first.detail,
    )
    outcomes = list(report.outcomes)
    outcomes[outcome_index] = replace(
        outcome_from_observations(
            original.rule_id,
            original.profile_id,
            tuple(observations),
        ),
        evidence_refs=original.evidence_refs,
    )

    result = _verify(bundle, replace(report, outcomes=tuple(outcomes)))

    assert result.status == "failed"
    assert result.issue_codes == ("rule_semantics_mismatch",)
    assert result.evidence == ()


@pytest.mark.parametrize(
    ("outcome_index", "observation_index", "expected", "actual"),
    [
        (0, 0, 1, 1),
        (3, 1, (True, 2), (True, 2)),
    ],
)
def test_json_type_substitutions_fail_semantic_verification(
    outcome_index: int,
    observation_index: int,
    expected: Any,
    actual: Any,
) -> None:
    bundle = _bundle()
    report = _report(bundle)
    source = report.outcomes[outcome_index]
    observations = list(source.observations)
    original = observations[observation_index]
    observations[observation_index] = observation(
        original.observation_id,
        expected=expected,
        actual=actual,
    )
    outcomes = list(report.outcomes)
    outcomes[outcome_index] = replace(
        outcome_from_observations(
            source.rule_id,
            source.profile_id,
            tuple(observations),
        ),
        evidence_refs=source.evidence_refs,
    )

    result = _verify(bundle, replace(report, outcomes=tuple(outcomes)))

    assert result.issue_codes == ("rule_semantics_mismatch",)


@pytest.mark.parametrize("mutation", ["reordered", "duplicate", "missing"])
def test_rule_set_must_match_the_closed_minimal_agent_profile(mutation: str) -> None:
    bundle = _bundle()
    report = _report(bundle)
    if mutation == "reordered":
        outcomes = tuple(reversed(report.outcomes))
    elif mutation == "duplicate":
        outcomes = (report.outcomes[0],) * 4
    else:
        outcomes = report.outcomes[:-1]

    result = _verify(bundle, replace(report, outcomes=outcomes))

    assert result.issue_codes == ("rule_set_mismatch",)


def test_malformed_rule_structure_is_rejected_before_evidence_access() -> None:
    bundle = _bundle()
    report = _report(bundle)

    result = verify_conformance_report(
        replace(report, outcomes=tuple(reversed(report.outcomes))),
        _ExplodingEvidenceStore(),
    )

    assert result.issue_codes == ("rule_set_mismatch",)


def test_every_rule_must_reference_the_single_declared_evidence_bundle() -> None:
    bundle = _bundle()
    report = _report(bundle)
    outcomes = (replace(report.outcomes[0], evidence_refs=()), *report.outcomes[1:])

    missing_reference = _verify(bundle, replace(report, outcomes=outcomes))

    second = _reference(bundle, evidence_id="minimal-agent.other")
    extra_reference = _verify(bundle, replace(report, evidence=(*report.evidence, second)))

    assert missing_reference.issue_codes == ("rule_evidence_mismatch",)
    assert extra_reference.issue_codes == ("evidence_set_mismatch",)


def test_missing_or_tampered_evidence_returns_static_secret_safe_codes() -> None:
    bundle = _bundle()
    report = _report(bundle)

    missing = verify_conformance_report(report, {})
    secret = b"Authorization: Bearer evidence-provider-secret"
    tampered = verify_conformance_report(report, {_EVIDENCE_ID: secret})

    assert missing.issue_codes == ("evidence_missing",)
    assert tampered.issue_codes == ("evidence_integrity_failed",)
    assert secret.decode() not in repr(tampered)
    assert tampered.evidence == ()


def test_evidence_access_failures_return_a_static_secret_safe_code() -> None:
    bundle = _bundle()
    report = _report(bundle)
    secret = "Authorization: Bearer artifact-store-secret"

    class FailingStore(dict[str, bytes]):
        def __getitem__(self, key: str) -> bytes:
            raise RuntimeError(f"{secret}: {key}")

    result = verify_conformance_report(report, FailingStore())

    assert result.issue_codes == ("evidence_access_failed",)
    assert secret not in repr(result)


def test_evidence_profile_and_target_are_bound_to_the_report() -> None:
    profile_bundle = _bundle(profile_id="other-profile")
    profile_report = _report(
        profile_bundle,
        profile_id="minimal-agent",
        outcomes=_outcomes(profile_bundle, profile_id="minimal-agent"),
    )
    target_bundle = _bundle()
    target_report = _report(target_bundle, target=_target("other.runtime"))

    profile_result = _verify(profile_bundle, profile_report)
    target_result = _verify(target_bundle, target_report)

    assert profile_result.issue_codes == ("evidence_profile_mismatch",)
    assert target_result.issue_codes == ("evidence_target_mismatch",)


@pytest.mark.parametrize(
    "bundle",
    [
        _bundle(events_complete=False),
        _bundle(next_seq=99),
        _bundle(
            events=(
                ConformanceEvent(seq=2, event_type="run.finished"),
                ConformanceEvent(seq=1, event_type="run.started"),
            ),
            next_seq=2,
        ),
    ],
)
def test_incomplete_nonmonotonic_or_bad_cursor_evidence_fails_closed(
    bundle: ConformanceEvidenceBundle,
) -> None:
    result = _verify(bundle)

    assert result.issue_codes == ("evidence_lifecycle_invalid",)


def test_increasing_sequence_gap_is_valid_evidence_for_a_failed_rule() -> None:
    bundle = _bundle(
        events=(
            ConformanceEvent(seq=1, event_type="run.started"),
            ConformanceEvent(seq=3, event_type="run.finished"),
        ),
        next_seq=4,
    )

    result = _verify(bundle)

    assert result.status == "verified"
    assert result.report_passed is False


class _ExplodingEvidenceStore(Mapping[str, bytes]):
    def __getitem__(self, key: str) -> bytes:
        raise AssertionError(key)

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("evidence store must not be read")

    def __len__(self) -> int:
        raise AssertionError("evidence store must not be read")


def test_unavailable_and_unsupported_reports_do_not_read_the_evidence_store() -> None:
    fixture = next(
        item
        for item in load_compatibility_fixtures()
        if item.fixture_id == "conformance-report-v1"
    )
    migrated = decode_conformance_report(fixture.payload)
    assert migrated.value is not None
    unavailable = verify_conformance_report(migrated.value, _ExplodingEvidenceStore())

    bundle = _bundle(profile_id="other-profile")
    reference = _reference(bundle)
    other_outcome = ConformanceRuleOutcome(
        rule_id="OTHER-01",
        profile_id="other-profile",
        status="passed",
        evidence_refs=(_EVIDENCE_ID,),
    )
    other_report = ConformanceReport(
        harness_id="other-adapter",
        profile_id="other-profile",
        outcomes=(other_outcome,),
        schema_version=CONFORMANCE_REPORT_V2,
        provenance_status="available",
        target=bundle.target,
        evidence=(reference,),
    )
    unsupported = verify_conformance_report(other_report, _ExplodingEvidenceStore())

    assert unavailable.status == "unavailable"
    assert unavailable.verified is False
    assert unsupported.status == "unsupported"


def test_extra_artifact_store_entries_are_ignored() -> None:
    bundle = _bundle()
    result = verify_conformance_report(
        _report(bundle),
        {
            _EVIDENCE_ID: serialize_conformance_evidence(bundle),
            "unrelated.artifact": b"untrusted extra bytes",
        },
    )

    assert result.status == "verified"


def test_v2_result_observations_are_boolean_and_exclude_the_raw_run_id() -> None:
    raw_run_id = "private-run-id"
    bundle = _bundle(case_id_sha256=case_id_sha256(raw_run_id))
    report = _report(bundle)
    payload = json.dumps(report.to_json(), sort_keys=True)
    result = next(outcome for outcome in report.outcomes if outcome.rule_id == "MIN-03-RESULT")

    assert all(type(item.expected) is bool for item in result.observations)
    assert all(type(item.actual) is bool for item in result.observations)
    assert raw_run_id not in payload
    assert raw_run_id.encode() not in serialize_conformance_evidence(bundle)


def test_verifier_rejects_invalid_python_arguments_and_bounds() -> None:
    bundle = _bundle()
    report = _report(bundle)

    with pytest.raises(TypeError, match="ConformanceReport"):
        verify_conformance_report(object(), {})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mapping"):
        verify_conformance_report(report, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be bytes"):
        verify_conformance_report(report, {_EVIDENCE_ID: "text"})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="must be bytes"):
        verify_conformance_report(report, {_EVIDENCE_ID: None})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="max_evidence_bytes"):
        verify_conformance_report(report, {}, max_evidence_bytes=-1)


def test_public_verification_result_rejects_untyped_or_unknown_content() -> None:
    with pytest.raises(ValueError, match="issue code"):
        ConformanceVerificationResult(
            status="failed",
            report_passed=False,
            issue_codes=("secret parser output",),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="evidence must be typed"):
        ConformanceVerificationResult(
            status="verified",
            report_passed=True,
            evidence=(object(),),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="exactly one"):
        ConformanceVerificationResult(
            status="verified",
            report_passed=True,
        )


def test_public_verification_result_rejects_repr_overriding_subclasses() -> None:
    secret = "Authorization: Bearer result-repr-secret"

    class LeakyString(str):
        def __repr__(self) -> str:
            return secret

    class LeakyBundle(ConformanceEvidenceBundle):
        def __repr__(self) -> str:
            return secret

    bundle = _bundle()
    leaky_bundle = LeakyBundle(**vars(bundle))

    with pytest.raises(TypeError, match="status must be a string"):
        ConformanceVerificationResult(
            status=LeakyString("failed"),  # type: ignore[arg-type]
            report_passed=False,
            issue_codes=("evidence_missing",),
        )
    with pytest.raises(TypeError, match="issue codes must be strings"):
        ConformanceVerificationResult(
            status="failed",
            report_passed=False,
            issue_codes=(LeakyString("evidence_missing"),),
        )
    with pytest.raises(TypeError, match="issue codes must be strings"):
        ConformanceVerificationResult(
            status="failed",
            report_passed=False,
            issue_codes=([],),  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="evidence must be typed"):
        ConformanceVerificationResult(
            status="verified",
            report_passed=True,
            evidence=(leaky_bundle,),
        )
