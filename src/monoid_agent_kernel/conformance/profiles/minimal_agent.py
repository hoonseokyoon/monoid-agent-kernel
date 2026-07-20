"""Minimal local agent profile metadata."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from monoid_agent_kernel.conformance.harness import (
    MinimalAgentEvidenceCapture,
    MinimalAgentEvidenceHarness,
    MinimalAgentHarness,
)
from monoid_agent_kernel.conformance.provenance import (
    ConformanceEvidenceBundle,
    case_id_sha256,
)
from monoid_agent_kernel.conformance.report import (
    ConformanceRuleOutcome,
    observation,
    outcome_from_observations,
    safe_exception_summary,
)

from ._metadata import ProfileMetadata

MINIMAL_AGENT_RULE_IDS = (
    "MIN-01-SUBMISSION",
    "MIN-02-LIFECYCLE",
    "MIN-03-RESULT",
    "MIN-04-EVENT-SEQUENCE",
)

PROFILE = ProfileMetadata(
    profile_id="minimal-agent",
    title="Minimal Agent",
    summary="Local loop or chatbot-style integration with basic lifecycle and model adapter behavior.",
    rule_ids=MINIMAL_AGENT_RULE_IDS,
    harnesses=("minimal-agent",),
)


@dataclass(frozen=True, kw_only=True)
class MinimalAgentProfileExecution:
    """Profile outcomes plus optional evidence retained from the same invocation."""

    outcomes: tuple[ConformanceRuleOutcome, ...]
    evidence: tuple[ConformanceEvidenceBundle, ...] = ()

    def __post_init__(self) -> None:
        outcomes = tuple(self.outcomes)
        evidence = tuple(self.evidence)
        if any(not isinstance(item, ConformanceRuleOutcome) for item in outcomes):
            raise TypeError("minimal-agent execution outcomes must be typed")
        if any(not isinstance(item, ConformanceEvidenceBundle) for item in evidence):
            raise TypeError("minimal-agent execution evidence must be typed")
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "evidence", evidence)


def execute_minimal_agent_profile(harness: MinimalAgentHarness) -> MinimalAgentProfileExecution:
    """Execute one minimal-agent case and retain evidence when the adapter supports it."""

    started = time.perf_counter()
    try:
        capture: MinimalAgentEvidenceCapture | None = None
        if isinstance(harness, MinimalAgentEvidenceHarness):
            candidate = harness.run_minimal_lifecycle_case_with_evidence()
            if not isinstance(candidate, MinimalAgentEvidenceCapture):
                raise TypeError("enhanced minimal-agent harness returned an invalid capture")
            capture = candidate
            case = dict(candidate.case)
            _validate_evidence_matches_case(case, candidate.evidence)
        else:
            raw_case = harness.run_minimal_lifecycle_case()
            if not isinstance(raw_case, Mapping):
                raise TypeError("minimal-agent harness returned an invalid case")
            case = dict(raw_case)
    except Exception as exc:
        return MinimalAgentProfileExecution(
            outcomes=_error_outcomes(exc),
        )
    elapsed = time.perf_counter() - started
    outcomes = _outcomes_from_case(case, duration_s=elapsed)
    return MinimalAgentProfileExecution(
        outcomes=outcomes,
        evidence=((capture.evidence,) if capture is not None else ()),
    )


def run_minimal_agent_profile(harness: MinimalAgentHarness) -> tuple[ConformanceRuleOutcome, ...]:
    """Execute stable minimal-agent rules while preserving the historical return type."""

    return execute_minimal_agent_profile(harness).outcomes


def _outcomes_from_case(
    case: Mapping[str, Any],
    *,
    duration_s: float,
) -> tuple[ConformanceRuleOutcome, ...]:
    run_id = str(case.get("run_id") or "")
    states = tuple(str(state) for state in case.get("states") or ())
    result = case.get("result") if isinstance(case.get("result"), Mapping) else {}
    event_seqs = tuple(_int_values(case.get("event_seqs")))
    return (
        outcome_from_observations(
            MINIMAL_AGENT_RULE_IDS[0],
            PROFILE.profile_id,
            (
                observation("run_id_present", expected=True, actual=bool(run_id)),
                observation(
                    "submission_accepted", expected=True, actual=bool(case.get("submitted"))
                ),
            ),
            duration_s=duration_s,
        ),
        outcome_from_observations(
            MINIMAL_AGENT_RULE_IDS[1],
            PROFILE.profile_id,
            (
                observation(
                    "terminal_state",
                    expected=True,
                    actual=bool(states and states[-1] in {"completed", "awaiting_input"}),
                ),
                observation("running_observed", expected=True, actual="running" in states),
            ),
        ),
        outcome_from_observations(
            MINIMAL_AGENT_RULE_IDS[2],
            PROFILE.profile_id,
            (
                observation(
                    "result_run_id",
                    expected=True,
                    actual=str(result.get("run_id") or "") == run_id,
                ),
                observation(
                    "result_status",
                    expected=True,
                    actual=str(result.get("status") or "") == "completed",
                ),
            ),
        ),
        outcome_from_observations(
            MINIMAL_AGENT_RULE_IDS[3],
            PROFILE.profile_id,
            (
                observation("events_present", expected=True, actual=bool(event_seqs)),
                observation(
                    "event_sequence_contiguous",
                    expected=tuple(range(1, len(event_seqs) + 1)),
                    actual=event_seqs,
                ),
            ),
        ),
    )


def _error_outcomes(exc: BaseException) -> tuple[ConformanceRuleOutcome, ...]:
    return tuple(
        ConformanceRuleOutcome(
            rule_id=rule_id,
            profile_id=PROFILE.profile_id,
            status="error",
            error=safe_exception_summary(exc),
        )
        for rule_id in MINIMAL_AGENT_RULE_IDS
    )


def _validate_evidence_matches_case(
    case: Mapping[str, Any],
    evidence: ConformanceEvidenceBundle,
) -> None:
    if evidence.profile_id != PROFILE.profile_id:
        raise ValueError("minimal-agent evidence profile mismatch")
    raw_run_id = case.get("run_id")
    if not isinstance(raw_run_id, str):
        raise TypeError("minimal-agent evidence requires a string run id")
    states = case.get("states")
    if not isinstance(states, (list, tuple)) or any(
        not isinstance(state, str) for state in states
    ):
        raise TypeError("minimal-agent evidence requires string lifecycle states")
    result = case.get("result")
    if not isinstance(result, Mapping):
        raise TypeError("minimal-agent evidence requires a result mapping")
    result_run_id = result.get("run_id")
    result_status = result.get("status")
    if not isinstance(result_run_id, str) or not isinstance(result_status, str):
        raise TypeError("minimal-agent evidence requires string result identity and status")
    submitted = case.get("submitted")
    if type(submitted) is not bool:
        raise TypeError("minimal-agent evidence requires a boolean submission result")
    raw_event_seqs = case.get("event_seqs")
    if not isinstance(raw_event_seqs, (list, tuple)) or any(
        type(seq) is not int for seq in raw_event_seqs
    ):
        raise TypeError("minimal-agent evidence requires integer event sequences")
    event_seqs = tuple(raw_event_seqs)
    if any(right <= left for left, right in zip(event_seqs, event_seqs[1:])):
        raise ValueError("minimal-agent evidence sequences must increase")
    evidence_seqs = tuple(event.seq for event in evidence.events)
    if evidence.events_complete is not True:
        raise ValueError("minimal-agent evidence must be complete")
    expected_next_seq = event_seqs[-1] + 1 if event_seqs else 0
    if evidence.next_seq != expected_next_seq:
        raise ValueError("minimal-agent evidence has an inconsistent terminal cursor")
    expected = (
        case_id_sha256(raw_run_id),
        bool(raw_run_id),
        submitted,
        tuple(states),
        result_run_id == raw_run_id,
        result_status,
        event_seqs,
    )
    actual = (
        evidence.case_id_sha256,
        evidence.run_id_present,
        evidence.submitted,
        evidence.states,
        evidence.result_run_id_matches,
        evidence.result_status,
        evidence_seqs,
    )
    if actual != expected:
        raise ValueError("minimal-agent evidence does not match the trusted case")


def _int_values(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return ()
