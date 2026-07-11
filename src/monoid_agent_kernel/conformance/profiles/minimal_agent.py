"""Minimal local agent profile metadata."""

from __future__ import annotations

import time
from typing import Any

from monoid_agent_kernel.conformance.harness import MinimalAgentHarness
from monoid_agent_kernel.conformance.report import (
    ConformanceRuleOutcome,
    observation,
    outcome_from_observations,
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


def run_minimal_agent_profile(harness: MinimalAgentHarness) -> tuple[ConformanceRuleOutcome, ...]:
    """Execute stable minimal-agent rules against an external harness."""

    started = time.perf_counter()
    try:
        case = dict(harness.run_minimal_lifecycle_case())
    except Exception as exc:
        return tuple(
            ConformanceRuleOutcome(
                rule_id=rule_id,
                profile_id=PROFILE.profile_id,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
            for rule_id in MINIMAL_AGENT_RULE_IDS
        )
    elapsed = time.perf_counter() - started
    run_id = str(case.get("run_id") or "")
    states = tuple(str(state) for state in case.get("states") or ())
    result = case.get("result") if isinstance(case.get("result"), dict) else {}
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
            duration_s=elapsed,
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
                    "result_run_id", expected=run_id, actual=str(result.get("run_id") or "")
                ),
                observation(
                    "result_status", expected="completed", actual=str(result.get("status") or "")
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


def _int_values(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError):
        return ()
