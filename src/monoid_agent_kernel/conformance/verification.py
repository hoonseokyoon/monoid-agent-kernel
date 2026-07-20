"""Offline semantic verification for evidence-backed conformance reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from monoid_agent_kernel.conformance.profiles.minimal_agent import (
    MINIMAL_AGENT_RULE_IDS,
)
from monoid_agent_kernel.conformance.provenance import (
    MAX_CONFORMANCE_EVIDENCE_BYTES,
    ConformanceEvidenceBundle,
    verify_conformance_evidence,
)
from monoid_agent_kernel.conformance.report import (
    ConformanceReport,
    ConformanceRuleOutcome,
    observation,
    outcome_from_observations,
)

ConformanceVerificationStatus = Literal[
    "verified",
    "unavailable",
    "unsupported",
    "failed",
]
ConformanceVerificationIssueCode = Literal[
    "evidence_set_mismatch",
    "evidence_missing",
    "evidence_access_failed",
    "evidence_integrity_failed",
    "evidence_profile_mismatch",
    "evidence_target_mismatch",
    "evidence_lifecycle_invalid",
    "rule_set_mismatch",
    "rule_evidence_mismatch",
    "rule_semantics_mismatch",
]
_VERIFICATION_ISSUE_CODES = frozenset(
    {
        "evidence_set_mismatch",
        "evidence_missing",
        "evidence_access_failed",
        "evidence_integrity_failed",
        "evidence_profile_mismatch",
        "evidence_target_mismatch",
        "evidence_lifecycle_invalid",
        "rule_set_mismatch",
        "rule_evidence_mismatch",
        "rule_semantics_mismatch",
    }
)


@dataclass(frozen=True, kw_only=True)
class ConformanceVerificationResult:
    """A secret-safe result from checking a report against retained evidence."""

    status: ConformanceVerificationStatus
    report_passed: bool
    issue_codes: tuple[ConformanceVerificationIssueCode, ...] = ()
    evidence: tuple[ConformanceEvidenceBundle, ...] = ()

    def __post_init__(self) -> None:
        if type(self.status) is not str:
            raise TypeError("conformance verification status must be a string")
        if self.status not in {"verified", "unavailable", "unsupported", "failed"}:
            raise ValueError("unsupported conformance verification status")
        if type(self.report_passed) is not bool:
            raise TypeError("conformance verification report_passed must be boolean")
        issue_codes = tuple(self.issue_codes)
        evidence = tuple(self.evidence)
        if any(type(code) is not str for code in issue_codes):
            raise TypeError("conformance verification issue codes must be strings")
        if any(code not in _VERIFICATION_ISSUE_CODES for code in issue_codes):
            raise ValueError("unsupported conformance verification issue code")
        if len(set(issue_codes)) != len(issue_codes):
            raise ValueError("conformance verification issue codes must be unique")
        if any(type(item) is not ConformanceEvidenceBundle for item in evidence):
            raise TypeError("conformance verification evidence must be typed")
        if self.status == "failed" and not issue_codes:
            raise ValueError("failed conformance verification requires an issue code")
        if self.status != "failed" and issue_codes:
            raise ValueError("only failed conformance verification may contain issue codes")
        if self.status != "verified" and evidence:
            raise ValueError("only verified conformance results may retain evidence")
        if self.status == "verified" and len(evidence) != 1:
            raise ValueError(
                "verified conformance results require exactly one evidence bundle"
            )
        object.__setattr__(self, "issue_codes", issue_codes)
        object.__setattr__(self, "evidence", evidence)

    @property
    def verified(self) -> bool:
        return self.status == "verified"


def verify_conformance_report(
    report: ConformanceReport,
    evidence_by_id: Mapping[str, bytes],
    *,
    max_evidence_bytes: int = MAX_CONFORMANCE_EVIDENCE_BYTES,
) -> ConformanceVerificationResult:
    """Verify report semantics against exact retained evidence without the harness.

    The evidence mapping is keyed by the stable evidence id from the report. Extra entries are
    ignored so callers may pass a larger artifact store. Static issue codes deliberately avoid
    copying parser errors or artifact content into verification output.

    ``verified`` establishes exact-byte integrity and internal report/evidence semantic
    consistency. Harness honesty, target authenticity, freshness, and report authorship remain
    trusted assertions; they require a trusted distribution channel or external signature.
    """

    if not isinstance(report, ConformanceReport):
        raise TypeError("report must be a typed ConformanceReport")
    if not isinstance(evidence_by_id, Mapping):
        raise TypeError("evidence_by_id must be a mapping")
    if type(max_evidence_bytes) is not int or max_evidence_bytes < 0:
        raise ValueError("max_evidence_bytes must be a non-negative integer")
    if report.provenance_status == "unavailable":
        return _result(report, "unavailable")
    if report.profile_id != "minimal-agent":
        return _result(report, "unsupported")
    if report.target is None or len(report.evidence) != 1:
        return _failure(report, "evidence_set_mismatch")

    reference = report.evidence[0]
    if tuple(outcome.rule_id for outcome in report.outcomes) != MINIMAL_AGENT_RULE_IDS:
        return _failure(report, "rule_set_mismatch")
    expected_reference = (reference.evidence_id,)
    if any(outcome.evidence_refs != expected_reference for outcome in report.outcomes):
        return _failure(report, "rule_evidence_mismatch")
    try:
        data = evidence_by_id[reference.evidence_id]
    except KeyError:
        return _failure(report, "evidence_missing")
    except Exception:
        return _failure(report, "evidence_access_failed")
    if not isinstance(data, bytes):
        raise TypeError("conformance evidence values must be bytes")
    try:
        bundle = verify_conformance_evidence(
            reference,
            data,
            max_bytes=max_evidence_bytes,
        )
    except ValueError:
        return _failure(report, "evidence_integrity_failed")

    if bundle.profile_id != report.profile_id:
        return _failure(report, "evidence_profile_mismatch")
    if bundle.target != report.target:
        return _failure(report, "evidence_target_mismatch")
    if not _valid_lifecycle(bundle):
        return _failure(report, "evidence_lifecycle_invalid")

    expected_outcomes = _minimal_agent_outcomes(bundle)
    try:
        semantics_match = all(
            _semantic_signature(actual) == _semantic_signature(expected)
            for actual, expected in zip(report.outcomes, expected_outcomes, strict=True)
        )
    except (OverflowError, RecursionError, TypeError, ValueError):
        semantics_match = False
    if not semantics_match:
        return _failure(report, "rule_semantics_mismatch")
    return ConformanceVerificationResult(
        status="verified",
        report_passed=report.passed,
        evidence=(bundle,),
    )


def _valid_lifecycle(bundle: ConformanceEvidenceBundle) -> bool:
    sequences = tuple(event.seq for event in bundle.events)
    if not bundle.events_complete:
        return False
    if any(right <= left for left, right in zip(sequences, sequences[1:])):
        return False
    expected_next_seq = sequences[-1] + 1 if sequences else 0
    return bundle.next_seq == expected_next_seq


def _minimal_agent_outcomes(
    bundle: ConformanceEvidenceBundle,
) -> tuple[ConformanceRuleOutcome, ...]:
    sequences = tuple(event.seq for event in bundle.events)
    return (
        outcome_from_observations(
            MINIMAL_AGENT_RULE_IDS[0],
            bundle.profile_id,
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
            MINIMAL_AGENT_RULE_IDS[1],
            bundle.profile_id,
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
            MINIMAL_AGENT_RULE_IDS[2],
            bundle.profile_id,
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
            MINIMAL_AGENT_RULE_IDS[3],
            bundle.profile_id,
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


def _semantic_signature(outcome: ConformanceRuleOutcome) -> str:
    payload = outcome.to_json()
    payload.pop("duration_s")
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _failure(
    report: ConformanceReport,
    issue_code: ConformanceVerificationIssueCode,
) -> ConformanceVerificationResult:
    return ConformanceVerificationResult(
        status="failed",
        report_passed=report.passed,
        issue_codes=(issue_code,),
    )


def _result(
    report: ConformanceReport,
    status: Literal["unavailable", "unsupported"],
) -> ConformanceVerificationResult:
    return ConformanceVerificationResult(
        status=status,
        report_passed=report.passed,
    )


__all__ = [
    "ConformanceVerificationIssueCode",
    "ConformanceVerificationResult",
    "ConformanceVerificationStatus",
    "verify_conformance_report",
]
