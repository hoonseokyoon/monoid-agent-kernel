from __future__ import annotations

from monoid_agent_kernel.conformance import run_fenced_run_sink_contract
from support.fenced_hosting import DeterministicFencedRunHarness


def test_deterministic_fenced_sink_passes_reusable_contract() -> None:
    outcomes = run_fenced_run_sink_contract(DeterministicFencedRunHarness)

    assert [outcome.rule_id for outcome in outcomes] == [
        "FENCED-01-CHECKPOINT-CONTENT-IDENTITY",
        "FENCED-02-FENCE-PRECEDES-IDEMPOTENCY",
        "FENCED-03-EVENT-AND-TERMINAL-WINNERS",
        "FENCED-04-INVOCATION-LIFECYCLE",
        "FENCED-05-INVOCATION-REFUSES-ILLEGAL-TRANSITIONS",
        "FENCED-06-WRITER-TOKEN-RUN-BINDING",
    ]
    assert all(outcome.status == "passed" for outcome in outcomes), outcomes
