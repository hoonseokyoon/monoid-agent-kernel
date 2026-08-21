from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest

from monoid_agent_kernel.conformance import run_fenced_run_sink_contract
from monoid_agent_kernel.core.checkpoint import RunCheckpoint
from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.model_invocation import DurableModelInvocation
from monoid_agent_kernel.core.outcome import TerminalOutcome
from monoid_agent_kernel.hosting import CommitResult, StorageCapabilities, WriterToken
from support.fenced_hosting import (
    DeterministicFencedRunHarness,
    DeterministicFencedRunSink,
)


def test_deterministic_fenced_sink_passes_reusable_contract() -> None:
    outcomes = run_fenced_run_sink_contract(DeterministicFencedRunHarness)

    assert [outcome.rule_id for outcome in outcomes] == [
        "FENCED-00-CAPABILITY-DECLARATION",
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


class _MissingResourceFenceBypassSink(DeterministicFencedRunSink):
    broken_mutation = ""
    _bypass_current_writer = False

    def _is_current(self, run_id: str, writer_token: WriterToken) -> bool:
        if self._bypass_current_writer:
            return writer_token.run_id == run_id
        return super()._is_current(run_id, writer_token)

    def _without_current_writer_check(self, operation):
        self._bypass_current_writer = True
        try:
            return operation()
        finally:
            self._bypass_current_writer = False

    def commit_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        key = (checkpoint.run_id, checkpoint.seq)
        if self.broken_mutation == "checkpoint" and key not in self._checkpoints:
            return self._without_current_writer_check(
                lambda: super(_MissingResourceFenceBypassSink, self).commit_checkpoint(
                    checkpoint,
                    blobs,
                    writer_token=writer_token,
                )
            )
        return super().commit_checkpoint(checkpoint, blobs, writer_token=writer_token)

    def append_event(
        self,
        event: AgentEvent,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        key = (event.run_id, event.seq)
        if self.broken_mutation == "event" and key not in self._events:
            return self._without_current_writer_check(
                lambda: super(_MissingResourceFenceBypassSink, self).append_event(
                    event,
                    writer_token=writer_token,
                )
            )
        return super().append_event(event, writer_token=writer_token)

    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        key = (invocation.run_id, invocation.logical_call_id, invocation.revision)
        if self.broken_mutation == "invocation" and key not in self._invocations:
            return self._without_current_writer_check(
                lambda: super(_MissingResourceFenceBypassSink, self).commit_invocation(
                    invocation,
                    blobs,
                    writer_token=writer_token,
                )
            )
        return super().commit_invocation(invocation, blobs, writer_token=writer_token)

    def settle_terminal(
        self,
        outcome: TerminalOutcome,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        if self.broken_mutation == "terminal" and outcome.run_id not in self._terminals:
            return self._without_current_writer_check(
                lambda: super(_MissingResourceFenceBypassSink, self).settle_terminal(
                    outcome,
                    writer_token=writer_token,
                )
            )
        return super().settle_terminal(outcome, writer_token=writer_token)


def _missing_resource_fence_bypass_factory(mutation: str):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _MissingResourceFenceBypassSink(harness._writers)
        sink.broken_mutation = mutation
        harness.sink = sink
        return harness

    return factory


@pytest.mark.parametrize("mutation", ["checkpoint", "event", "invocation", "terminal"])
def test_reusable_contract_rejects_stale_writes_to_missing_resources(mutation: str) -> None:
    outcomes = run_fenced_run_sink_contract(_missing_resource_fence_bypass_factory(mutation))
    fence_rule = next(
        outcome
        for outcome in outcomes
        if outcome.rule_id == "FENCED-02-FENCE-PRECEDES-IDEMPOTENCY"
    )

    assert fence_rule.status == "failed"
    stale_observation = next(
        observation
        for observation in fence_rule.observations
        if observation.observation_id == f"stale_new_{mutation}"
    )
    assert stale_observation.expected == "fenced"
    assert stale_observation.actual == "committed"


class _TerminalRetrySink(DeterministicFencedRunSink):
    def _invocation_transition_winner(
        self,
        invocation: DurableModelInvocation,
    ) -> str | None:
        call_key = (invocation.run_id, invocation.logical_call_id)
        previous_revision = self._invocation_heads.get(call_key)
        if previous_revision is not None:
            _, previous_record = self._invocations[
                (invocation.run_id, invocation.logical_call_id, previous_revision)
            ]
            previous = previous_record.invocation
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


class _AllocationPolicyIndependentHarness(DeterministicFencedRunHarness):
    def claim_writer(self, run_id: str, owner_id: str) -> WriterToken:
        del run_id, owner_id
        raise AssertionError("the conformance contract must install exact writer tokens")


def test_reusable_contract_does_not_call_the_host_generation_allocator() -> None:
    outcomes = run_fenced_run_sink_contract(_AllocationPolicyIndependentHarness)

    assert all(outcome.status == "passed" for outcome in outcomes), outcomes


def _missing_capability_factory() -> DeterministicFencedRunHarness:
    harness = DeterministicFencedRunHarness()
    harness.sink.capabilities = StorageCapabilities(
        concurrent_writers=True,
        compare_and_set=True,
        lease_fencing=False,
        durable_checkpoints=True,
        durable_events=True,
        durable_invocations=True,
        terminal_first_writer_wins=True,
    )
    return harness


def test_reusable_contract_rejects_missing_required_capability_declaration() -> None:
    outcomes = run_fenced_run_sink_contract(_missing_capability_factory)
    capability_rule = next(
        outcome
        for outcome in outcomes
        if outcome.rule_id == "FENCED-00-CAPABILITY-DECLARATION"
    )

    assert capability_rule.status == "failed"


class _BlobDiscardingSink(DeterministicFencedRunSink):
    broken_mutation = ""

    def commit_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        if self.broken_mutation == "checkpoint":
            blobs = {}
        return super().commit_checkpoint(checkpoint, blobs, writer_token=writer_token)

    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        if self.broken_mutation == "invocation":
            blobs = {}
        return super().commit_invocation(invocation, blobs, writer_token=writer_token)


def _blob_discarding_factory(mutation: str):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _BlobDiscardingSink(harness._writers)
        sink.broken_mutation = mutation
        harness.sink = sink
        return harness

    return factory


@pytest.mark.parametrize(
    ("mutation", "rule_id"),
    [
        ("checkpoint", "FENCED-01-CHECKPOINT-CONTENT-IDENTITY"),
        ("invocation", "FENCED-04-INVOCATION-LIFECYCLE"),
    ],
)
def test_reusable_contract_rejects_discarded_private_blobs(
    mutation: str,
    rule_id: str,
) -> None:
    outcomes = run_fenced_run_sink_contract(_blob_discarding_factory(mutation))
    blob_rule = next(outcome for outcome in outcomes if outcome.rule_id == rule_id)

    assert blob_rule.status == "failed"


class _MetadataOnlyContentIdentitySink(DeterministicFencedRunSink):
    broken_mutation = ""

    def commit_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        key = (checkpoint.run_id, checkpoint.seq)
        stored = self._checkpoints.get(key)
        if (
            self.broken_mutation == "checkpoint"
            and self._is_current(checkpoint.run_id, writer_token)
            and stored is not None
            and stored[1].checkpoint == checkpoint
        ):
            return CommitResult(status="already_committed", sequence=checkpoint.seq)
        return super().commit_checkpoint(checkpoint, blobs, writer_token=writer_token)

    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        key = (invocation.run_id, invocation.logical_call_id, invocation.revision)
        stored = self._invocations.get(key)
        if (
            self.broken_mutation == "invocation"
            and self._is_current(invocation.run_id, writer_token)
            and stored is not None
            and stored[1].invocation == invocation
        ):
            return CommitResult(status="already_committed", sequence=invocation.revision)
        return super().commit_invocation(invocation, blobs, writer_token=writer_token)


def _metadata_only_content_identity_factory(mutation: str):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _MetadataOnlyContentIdentitySink(harness._writers)
        sink.broken_mutation = mutation
        harness.sink = sink
        return harness

    return factory


@pytest.mark.parametrize(
    ("mutation", "rule_id", "observation_id"),
    [
        (
            "checkpoint",
            "FENCED-01-CHECKPOINT-CONTENT-IDENTITY",
            "blob_bytes_conflict",
        ),
        (
            "invocation",
            "FENCED-04-INVOCATION-LIFECYCLE",
            "result_blob_bytes_conflict",
        ),
    ],
)
def test_reusable_contract_includes_blobs_in_content_identity(
    mutation: str,
    rule_id: str,
    observation_id: str,
) -> None:
    outcomes = run_fenced_run_sink_contract(_metadata_only_content_identity_factory(mutation))
    identity_rule = next(outcome for outcome in outcomes if outcome.rule_id == rule_id)
    blob_observation = next(
        observation
        for observation in identity_rule.observations
        if observation.observation_id == observation_id
    )

    assert identity_rule.status == "failed"
    assert blob_observation.expected == "conflict"
    assert blob_observation.actual == "already_committed"


class _RegressingCheckpointHeadSink(DeterministicFencedRunSink):
    def commit_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        key = (checkpoint.run_id, checkpoint.seq)
        current_head = self._checkpoint_heads.get(checkpoint.run_id, -1)
        if (
            self._is_current(checkpoint.run_id, writer_token)
            and key not in self._checkpoints
            and checkpoint.seq < current_head
        ):
            self._checkpoint_heads[checkpoint.run_id] = checkpoint.seq - 1
        return super().commit_checkpoint(checkpoint, blobs, writer_token=writer_token)


def _regressing_checkpoint_factory() -> DeterministicFencedRunHarness:
    harness = DeterministicFencedRunHarness()
    harness.sink = _RegressingCheckpointHeadSink(harness._writers)
    return harness


def test_reusable_contract_rejects_checkpoint_head_regression() -> None:
    outcomes = run_fenced_run_sink_contract(_regressing_checkpoint_factory)
    checkpoint_rule = next(
        outcome
        for outcome in outcomes
        if outcome.rule_id == "FENCED-01-CHECKPOINT-CONTENT-IDENTITY"
    )
    head_observation = next(
        observation
        for observation in checkpoint_rule.observations
        if observation.observation_id == "head_after_delayed_sequence"
    )

    assert checkpoint_rule.status == "failed"
    assert head_observation.expected == 2
    assert head_observation.actual == 1


class _InvocationIdentityDriftSink(DeterministicFencedRunSink):
    ignored_field = ""

    def _invocation_transition_winner(
        self,
        invocation: DurableModelInvocation,
    ) -> str | None:
        head = self._invocation_heads.get((invocation.run_id, invocation.logical_call_id))
        if head is not None:
            _, previous_record = self._invocations[
                (invocation.run_id, invocation.logical_call_id, head)
            ]
            previous = previous_record.invocation
            if getattr(invocation, self.ignored_field) != getattr(previous, self.ignored_field):
                invocation = replace(
                    invocation,
                    **{self.ignored_field: getattr(previous, self.ignored_field)},
                )
        return super()._invocation_transition_winner(invocation)


def _invocation_identity_drift_factory(field_name: str):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _InvocationIdentityDriftSink(harness._writers)
        sink.ignored_field = field_name
        harness.sink = sink
        return harness

    return factory


@pytest.mark.parametrize(
    "field_name",
    [
        "idempotency_key",
        "request_digest",
        "digest_generation",
        "dispatch_id",
        "dispatch_attempt",
    ],
)
def test_reusable_contract_checks_each_stable_invocation_identity(field_name: str) -> None:
    outcomes = run_fenced_run_sink_contract(_invocation_identity_drift_factory(field_name))
    refusal_rule = next(
        outcome
        for outcome in outcomes
        if outcome.rule_id == "FENCED-05-INVOCATION-REFUSES-ILLEGAL-TRANSITIONS"
    )
    drift_observation = next(
        observation
        for observation in refusal_rule.observations
        if observation.observation_id == f"identity_drift_{field_name}"
    )

    assert refusal_rule.status == "failed"
    assert drift_observation.expected == "conflict"
    assert drift_observation.actual == "committed"
