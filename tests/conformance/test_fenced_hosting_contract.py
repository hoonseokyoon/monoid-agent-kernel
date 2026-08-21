from __future__ import annotations

from collections.abc import Mapping

import pytest

from monoid_agent_kernel.conformance import run_fenced_run_sink_contract
from monoid_agent_kernel.core.checkpoint import RunCheckpoint
from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.model_invocation import DurableModelInvocation
from monoid_agent_kernel.core.outcome import TerminalOutcome
from monoid_agent_kernel.hosting import CommitResult, WriterToken
from support.fenced_hosting import (
    DeterministicFencedRunHarness,
    DeterministicFencedRunSink,
)


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


class _IdempotencyFirstSink(DeterministicFencedRunSink):
    broken_mutation = ""

    def commit_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        if (
            self.broken_mutation == "checkpoint"
            and (checkpoint.run_id, checkpoint.seq) in self._checkpoints
        ):
            return CommitResult(status="already_committed", sequence=checkpoint.seq)
        return super().commit_checkpoint(checkpoint, blobs, writer_token=writer_token)

    def append_event(
        self,
        event: AgentEvent,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        if self.broken_mutation == "event" and (event.run_id, event.seq) in self._events:
            return CommitResult(status="already_committed", sequence=event.seq)
        return super().append_event(event, writer_token=writer_token)

    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        key = (invocation.run_id, invocation.logical_call_id, invocation.revision)
        if self.broken_mutation == "invocation" and key in self._invocations:
            return CommitResult(status="already_committed", sequence=invocation.revision)
        return super().commit_invocation(invocation, blobs, writer_token=writer_token)

    def settle_terminal(
        self,
        outcome: TerminalOutcome,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        if self.broken_mutation == "terminal" and outcome.run_id in self._terminals:
            return CommitResult(status="already_committed")
        return super().settle_terminal(outcome, writer_token=writer_token)


def _idempotency_first_factory(mutation: str):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _IdempotencyFirstSink(harness._writers)
        sink.broken_mutation = mutation
        harness.sink = sink
        return harness

    return factory


@pytest.mark.parametrize("mutation", ["checkpoint", "event", "invocation", "terminal"])
def test_reusable_contract_rejects_idempotency_before_fencing(mutation: str) -> None:
    outcomes = run_fenced_run_sink_contract(_idempotency_first_factory(mutation))
    fence_rule = next(
        outcome
        for outcome in outcomes
        if outcome.rule_id == "FENCED-02-FENCE-PRECEDES-IDEMPOTENCY"
    )

    assert fence_rule.status == "failed"
    assert any(
        observation.expected == "fenced" and observation.actual == "already_committed"
        for observation in fence_rule.observations
    )


class _TerminalRetrySink(DeterministicFencedRunSink):
    def _invocation_transition_winner(
        self,
        invocation: DurableModelInvocation,
    ) -> str | None:
        call_key = (invocation.run_id, invocation.logical_call_id)
        previous_revision = self._invocation_heads.get(call_key)
        if previous_revision is not None:
            _, previous = self._invocations[
                (invocation.run_id, invocation.logical_call_id, previous_revision)
            ]
            if (
                previous.dispatch_state in {"settled", "unknown"}
                and invocation.dispatch_state == "reserved"
                and invocation.revision == previous.revision + 1
                and invocation.dispatch_attempt == previous.dispatch_attempt + 1
            ):
                return None
        return super()._invocation_transition_winner(invocation)


def _terminal_retry_factory() -> DeterministicFencedRunHarness:
    harness = DeterministicFencedRunHarness()
    harness.sink = _TerminalRetrySink(harness._writers)
    return harness


def test_reusable_contract_rejects_retry_after_terminal_invocation_states() -> None:
    outcomes = run_fenced_run_sink_contract(_terminal_retry_factory)
    refusal_rule = next(
        outcome
        for outcome in outcomes
        if outcome.rule_id == "FENCED-05-INVOCATION-REFUSES-ILLEGAL-TRANSITIONS"
    )

    assert refusal_rule.status == "failed"
    terminal_retry_observations = {
        observation.observation_id: observation.actual
        for observation in refusal_rule.observations
        if observation.observation_id.startswith("retry_after_")
    }
    assert terminal_retry_observations == {
        "retry_after_unknown": "committed",
        "retry_after_success": "committed",
        "retry_after_nonretry_failure": "committed",
    }
