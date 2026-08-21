from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from threading import Barrier, current_thread

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


_CHECKPOINT_IDENTITY_FIELDS = tuple(
    sorted(
        set(RunCheckpoint(run_id="identity-contract").to_json())
        - {"schema_version", "run_id", "seq"}
    )
)
_EVENT_IDENTITY_FIELDS = (
    "event_id",
    "timestamp",
    "type",
    "level",
    "data",
    "turn_id",
    "parent_id",
)
_TERMINAL_IDENTITY_FIELDS = (
    "kind",
    "retry_eligibility",
    "interruption_cause",
    "checkpoint_seq",
    "final_output_ref",
    "partial_output_ref",
    "last_evidence_ref",
    "error_code",
    "provider_error_code",
    "http_status",
)
_INVOCATION_IDENTITY_FIELDS = (
    "dispatch_id",
    "dispatch_attempt",
    "idempotency_key",
    "dispatch_state",
    "request_digest",
    "receipt",
    "result_ref",
    "failure_code",
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


class _RegressingInvocationHeadSink(DeterministicFencedRunSink):
    broken_revision = 0

    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        result = super().commit_invocation(invocation, blobs, writer_token=writer_token)
        if result.status == "already_committed" and invocation.revision == self.broken_revision:
            with self._lock:
                self._invocation_heads[
                    (invocation.run_id, invocation.logical_call_id)
                ] = invocation.revision
        return result


def _regressing_invocation_head_factory(revision: int):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _RegressingInvocationHeadSink(harness._writers)
        sink.broken_revision = revision
        harness.sink = sink
        return harness

    return factory


@pytest.mark.parametrize("revision", [1, 2, 3])
def test_reusable_contract_rejects_each_old_invocation_head_regression(
    revision: int,
) -> None:
    outcomes = run_fenced_run_sink_contract(_regressing_invocation_head_factory(revision))
    lifecycle_rule = next(
        outcome
        for outcome in outcomes
        if outcome.rule_id == "FENCED-04-INVOCATION-LIFECYCLE"
    )
    head_observation = next(
        observation
        for observation in lifecycle_rule.observations
        if observation.observation_id == f"old_revision_{revision}_retry_and_head"
    )

    assert lifecycle_rule.status == "failed"
    assert head_observation.expected == ("already_committed", 4)
    assert head_observation.actual == ("already_committed", revision)


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


class _CanonicalAliasTransitionDriftSink(DeterministicFencedRunSink):
    field_name = ""
    broken_direction = ""

    def _invocation_transition_winner(
        self,
        invocation: DurableModelInvocation,
    ) -> str | None:
        head = self._invocation_heads.get((invocation.run_id, invocation.logical_call_id))
        if head is not None:
            previous_digest, previous_record = self._invocations[
                (invocation.run_id, invocation.logical_call_id, head)
            ]
            previous_value = getattr(previous_record.invocation, self.field_name)
            incoming_value = getattr(invocation, self.field_name)
            previous_is_legacy = previous_value.startswith("native-agent-runner.")
            incoming_is_legacy = incoming_value.startswith("native-agent-runner.")
            direction = (
                "legacy_to_current" if previous_is_legacy else "current_to_legacy"
            )
            if (
                previous_value != incoming_value
                and previous_is_legacy != incoming_is_legacy
                and direction == self.broken_direction
            ):
                return previous_digest
        return super()._invocation_transition_winner(invocation)


def _canonical_alias_transition_drift_factory(field_name: str, direction: str):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _CanonicalAliasTransitionDriftSink(harness._writers)
        sink.field_name = field_name
        sink.broken_direction = direction
        harness.sink = sink
        return harness

    return factory


@pytest.mark.parametrize(
    ("field_name", "direction", "observation_prefix"),
    [
        ("schema_version", "current_to_legacy", "transition"),
        ("schema_version", "legacy_to_current", "recovery"),
        ("digest_generation", "current_to_legacy", "transition"),
        ("digest_generation", "legacy_to_current", "recovery"),
    ],
)
def test_reusable_contract_accepts_both_canonical_alias_transition_directions(
    field_name: str,
    direction: str,
    observation_prefix: str,
) -> None:
    outcomes = run_fenced_run_sink_contract(
        _canonical_alias_transition_drift_factory(field_name, direction)
    )
    lifecycle_rule = next(
        outcome
        for outcome in outcomes
        if outcome.rule_id == "FENCED-04-INVOCATION-LIFECYCLE"
    )
    alias_observation = next(
        observation
        for observation in lifecycle_rule.observations
        if observation.observation_id
        == f"invocation_canonical_alias_{observation_prefix}_{field_name}"
    )

    assert lifecycle_rule.status == "failed"
    assert alias_observation.expected == "committed"
    assert alias_observation.actual == "conflict"


class _ForbiddenInvocationEdgeSink(DeterministicFencedRunSink):
    allowed_edge: tuple[str, str] = ("", "")

    def _invocation_transition_winner(
        self,
        invocation: DurableModelInvocation,
    ) -> str | None:
        head = self._invocation_heads.get((invocation.run_id, invocation.logical_call_id))
        if head is not None:
            _, previous_record = self._invocations[
                (invocation.run_id, invocation.logical_call_id, head)
            ]
            if (previous_record.invocation.dispatch_state, invocation.dispatch_state) == (
                self.allowed_edge
            ):
                return None
        return super()._invocation_transition_winner(invocation)


def _forbidden_invocation_edge_factory(source: str, target: str):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _ForbiddenInvocationEdgeSink(harness._writers)
        sink.allowed_edge = (source, target)
        harness.sink = sink
        return harness

    return factory


_FORBIDDEN_INVOCATION_EDGES = (
    ("reserved", "reserved"),
    ("reserved", "settled"),
    ("reserved", "unknown"),
    ("dispatch_started", "reserved"),
    ("dispatch_started", "dispatch_started"),
    ("settled", "reserved"),
    ("settled", "dispatch_started"),
    ("settled", "settled"),
    ("settled", "unknown"),
    ("unknown", "reserved"),
    ("unknown", "dispatch_started"),
    ("unknown", "settled"),
    ("unknown", "unknown"),
)


@pytest.mark.parametrize(("source", "target"), _FORBIDDEN_INVOCATION_EDGES)
def test_reusable_contract_checks_each_forbidden_invocation_edge(
    source: str,
    target: str,
) -> None:
    outcomes = run_fenced_run_sink_contract(_forbidden_invocation_edge_factory(source, target))
    refusal_rule = next(
        outcome
        for outcome in outcomes
        if outcome.rule_id == "FENCED-05-INVOCATION-REFUSES-ILLEGAL-TRANSITIONS"
    )
    edge_observation = next(
        observation
        for observation in refusal_rule.observations
        if observation.observation_id == f"state_edge_{source}_to_{target}"
    )

    assert refusal_rule.status == "failed"
    assert edge_observation.expected == "conflict"
    assert edge_observation.actual == "committed"


class _InvalidInitialInvocationStateSink(DeterministicFencedRunSink):
    allowed_state = ""

    def _invocation_transition_winner(
        self,
        invocation: DurableModelInvocation,
    ) -> str | None:
        head = self._invocation_heads.get((invocation.run_id, invocation.logical_call_id))
        if head is None and invocation.dispatch_state == self.allowed_state:
            return None
        return super()._invocation_transition_winner(invocation)


def _invalid_initial_invocation_state_factory(state: str):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _InvalidInitialInvocationStateSink(harness._writers)
        sink.allowed_state = state
        harness.sink = sink
        return harness

    return factory


@pytest.mark.parametrize("state", ["dispatch_started", "settled", "unknown"])
def test_reusable_contract_checks_each_invalid_initial_invocation_state(state: str) -> None:
    outcomes = run_fenced_run_sink_contract(_invalid_initial_invocation_state_factory(state))
    refusal_rule = next(
        outcome
        for outcome in outcomes
        if outcome.rule_id == "FENCED-05-INVOCATION-REFUSES-ILLEGAL-TRANSITIONS"
    )
    state_observation = next(
        observation
        for observation in refusal_rule.observations
        if observation.observation_id == f"first_state_{state}"
    )

    assert refusal_rule.status == "failed"
    assert state_observation.expected == "conflict"
    assert state_observation.actual == "committed"


class _InvalidInitialInvocationCoordinateSink(DeterministicFencedRunSink):
    allowed_coordinate: tuple[int, int] = (0, 0)

    def _invocation_transition_winner(
        self,
        invocation: DurableModelInvocation,
    ) -> str | None:
        head = self._invocation_heads.get((invocation.run_id, invocation.logical_call_id))
        if (
            head is None
            and invocation.dispatch_state == "reserved"
            and (invocation.revision, invocation.dispatch_attempt) == self.allowed_coordinate
        ):
            return None
        return super()._invocation_transition_winner(invocation)


def _invalid_initial_invocation_coordinate_factory(revision: int, dispatch_attempt: int):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _InvalidInitialInvocationCoordinateSink(harness._writers)
        sink.allowed_coordinate = (revision, dispatch_attempt)
        harness.sink = sink
        return harness

    return factory


@pytest.mark.parametrize(
    ("label", "revision", "dispatch_attempt"),
    [
        ("revision_2", 2, 1),
        ("attempt_2", 1, 2),
        ("revision_2_attempt_2", 2, 2),
    ],
)
def test_reusable_contract_checks_each_invalid_initial_invocation_coordinate(
    label: str,
    revision: int,
    dispatch_attempt: int,
) -> None:
    outcomes = run_fenced_run_sink_contract(
        _invalid_initial_invocation_coordinate_factory(revision, dispatch_attempt)
    )
    refusal_rule = next(
        outcome
        for outcome in outcomes
        if outcome.rule_id == "FENCED-05-INVOCATION-REFUSES-ILLEGAL-TRANSITIONS"
    )
    coordinate_observation = next(
        observation
        for observation in refusal_rule.observations
        if observation.observation_id == f"first_coordinate_{label}"
    )

    assert refusal_rule.status == "failed"
    assert coordinate_observation.expected == "conflict"
    assert coordinate_observation.actual == "committed"


class _CorruptingLoadedRecordSink(DeterministicFencedRunSink):
    corruption = ""

    def latest_checked(self, run_id: str):
        loaded = super().latest_checked(run_id)
        if self.corruption != "checkpoint" or loaded.value is None:
            return loaded
        corrupted_checkpoint = replace(loaded.value.checkpoint, workspace_delta=[])
        return replace(
            loaded,
            value=replace(loaded.value, checkpoint=corrupted_checkpoint),
        )

    def load_invocation(self, run_id: str, logical_call_id: str):
        loaded = super().load_invocation(run_id, logical_call_id)
        if self.corruption != "invocation" or loaded.value is None:
            return loaded
        corrupted_invocation = replace(
            loaded.value.invocation,
            dispatch_id="dispatch-corrupted",
        )
        return replace(
            loaded,
            value=replace(loaded.value, invocation=corrupted_invocation),
        )


def _corrupting_loaded_record_factory(corruption: str):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _CorruptingLoadedRecordSink(harness._writers)
        sink.corruption = corruption
        harness.sink = sink
        return harness

    return factory


@pytest.mark.parametrize(
    ("corruption", "rule_id", "observation_id"),
    [
        (
            "checkpoint",
            "FENCED-01-CHECKPOINT-CONTENT-IDENTITY",
            "checkpoint_manifest_digest",
        ),
        (
            "invocation",
            "FENCED-04-INVOCATION-LIFECYCLE",
            "latest_invocation_digest",
        ),
    ],
)
def test_reusable_contract_compares_complete_reopened_records(
    corruption: str,
    rule_id: str,
    observation_id: str,
) -> None:
    outcomes = run_fenced_run_sink_contract(_corrupting_loaded_record_factory(corruption))
    durable_rule = next(outcome for outcome in outcomes if outcome.rule_id == rule_id)
    record_observation = next(
        observation
        for observation in durable_rule.observations
        if observation.observation_id == observation_id
    )

    assert durable_rule.status == "failed"
    assert record_observation.actual != record_observation.expected


class _IgnoringIdentityFieldSink(DeterministicFencedRunSink):
    record_family = ""
    ignored_field = ""

    def commit_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        stored = self._checkpoints.get((checkpoint.run_id, checkpoint.seq))
        if self.record_family == "checkpoint" and stored is not None:
            winner = stored[1].checkpoint
            checkpoint = replace(
                checkpoint,
                **{self.ignored_field: getattr(winner, self.ignored_field)},
            )
        return super().commit_checkpoint(checkpoint, blobs, writer_token=writer_token)

    def append_event(
        self,
        event: AgentEvent,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        stored = self._events.get((event.run_id, event.seq))
        if self.record_family == "event" and stored is not None:
            winner = stored[1]
            event = replace(
                event,
                **{self.ignored_field: getattr(winner, self.ignored_field)},
            )
        return super().append_event(event, writer_token=writer_token)

    def settle_terminal(
        self,
        outcome: TerminalOutcome,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        stored = self._terminals.get(outcome.run_id)
        if self.record_family == "terminal" and stored is not None:
            winner = stored[1]
            outcome = replace(
                outcome,
                **{self.ignored_field: getattr(winner, self.ignored_field)},
            )
        return super().settle_terminal(outcome, writer_token=writer_token)

    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        key = (invocation.run_id, invocation.logical_call_id, invocation.revision)
        stored = self._invocations.get(key)
        if self.record_family == "invocation" and stored is not None:
            winner = stored[1].invocation
            invocation = replace(
                invocation,
                **{self.ignored_field: getattr(winner, self.ignored_field)},
            )
        return super().commit_invocation(invocation, blobs, writer_token=writer_token)


def _ignoring_identity_field_factory(record_family: str, field_name: str):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _IgnoringIdentityFieldSink(harness._writers)
        sink.record_family = record_family
        sink.ignored_field = field_name
        harness.sink = sink
        return harness

    return factory


@pytest.mark.parametrize(
    ("record_family", "rule_id", "field_name"),
    [
        *(
            ("checkpoint", "FENCED-01-CHECKPOINT-CONTENT-IDENTITY", field_name)
            for field_name in _CHECKPOINT_IDENTITY_FIELDS
        ),
        *(
            ("event", "FENCED-03-EVENT-AND-TERMINAL-WINNERS", field_name)
            for field_name in _EVENT_IDENTITY_FIELDS
        ),
        *(
            ("terminal", "FENCED-03-EVENT-AND-TERMINAL-WINNERS", field_name)
            for field_name in _TERMINAL_IDENTITY_FIELDS
        ),
        *(
            ("invocation", "FENCED-04-INVOCATION-LIFECYCLE", field_name)
            for field_name in _INVOCATION_IDENTITY_FIELDS
        ),
    ],
)
def test_reusable_contract_checks_each_canonical_identity_field(
    record_family: str,
    rule_id: str,
    field_name: str,
) -> None:
    outcomes = run_fenced_run_sink_contract(
        _ignoring_identity_field_factory(record_family, field_name)
    )
    identity_rule = next(outcome for outcome in outcomes if outcome.rule_id == rule_id)
    field_observation = next(
        observation
        for observation in identity_rule.observations
        if observation.observation_id == f"{record_family}_identity_{field_name}"
    )

    assert identity_rule.status == "failed"
    assert field_observation.expected == "conflict"
    assert field_observation.actual == "already_committed"


class _InvalidRetryCoordinateSink(DeterministicFencedRunSink):
    allowed_coordinate: tuple[int, str] = (0, "")

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
            retryable_failure = (
                previous.dispatch_state == "settled"
                and bool(previous.failure_code)
                and previous.receipt is not None
                and previous.receipt.get("retryable") is True
            )
            if (
                retryable_failure
                and invocation.dispatch_state == "reserved"
                and (invocation.dispatch_attempt, invocation.dispatch_id)
                == self.allowed_coordinate
            ):
                return None
        return super()._invocation_transition_winner(invocation)


def _invalid_retry_coordinate_factory(attempt: int, dispatch_id: str):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _InvalidRetryCoordinateSink(harness._writers)
        sink.allowed_coordinate = (attempt, dispatch_id)
        harness.sink = sink
        return harness

    return factory


@pytest.mark.parametrize(
    ("label", "attempt", "dispatch_id"),
    [
        ("same_attempt", 1, "dispatch-2"),
        ("same_dispatch_id", 2, "dispatch-1"),
        ("same_attempt_and_dispatch_id", 1, "dispatch-1"),
        ("skipped_attempt", 3, "dispatch-3"),
    ],
)
def test_reusable_contract_rejects_each_invalid_retry_coordinate(
    label: str,
    attempt: int,
    dispatch_id: str,
) -> None:
    outcomes = run_fenced_run_sink_contract(
        _invalid_retry_coordinate_factory(attempt, dispatch_id)
    )
    refusal_rule = next(
        outcome
        for outcome in outcomes
        if outcome.rule_id == "FENCED-05-INVOCATION-REFUSES-ILLEGAL-TRANSITIONS"
    )
    coordinate_observation = next(
        observation
        for observation in refusal_rule.observations
        if observation.observation_id == f"retry_coordinate_{label}"
    )

    assert refusal_rule.status == "failed"
    assert coordinate_observation.expected == "conflict"
    assert coordinate_observation.actual == "committed"


class _VolatileReopenHarness(DeterministicFencedRunHarness):
    def reopen(self) -> DeterministicFencedRunHarness:
        return _VolatileReopenHarness()


def test_reusable_contract_rejects_process_local_only_storage() -> None:
    outcomes = run_fenced_run_sink_contract(_VolatileReopenHarness)
    durable_rules = {
        outcome.rule_id: outcome.status
        for outcome in outcomes
        if outcome.rule_id
        in {
            "FENCED-01-CHECKPOINT-CONTENT-IDENTITY",
            "FENCED-03-EVENT-AND-TERMINAL-WINNERS",
            "FENCED-04-INVOCATION-LIFECYCLE",
        }
    }

    assert durable_rules == {
        "FENCED-01-CHECKPOINT-CONTENT-IDENTITY": "failed",
        "FENCED-03-EVENT-AND-TERMINAL-WINNERS": "failed",
        "FENCED-04-INVOCATION-LIFECYCLE": "failed",
    }


class _NonAtomicRaceSink(DeterministicFencedRunSink):
    broken_mutation = ""
    _race_barrier: Barrier

    def _is_contract_racer(self) -> bool:
        return current_thread().name.startswith("fenced-contract")

    def commit_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        key = (checkpoint.run_id, checkpoint.seq)
        if (
            self.broken_mutation == "checkpoint"
            and self._is_contract_racer()
            and self._is_current(checkpoint.run_id, writer_token)
            and key not in self._checkpoints
        ):
            self._race_barrier.wait(timeout=10)
            with self._lock:
                self._checkpoints.pop(key, None)
                return self._commit_checkpoint(checkpoint, blobs, writer_token=writer_token)
        return super().commit_checkpoint(checkpoint, blobs, writer_token=writer_token)

    def append_event(
        self,
        event: AgentEvent,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        key = (event.run_id, event.seq)
        if (
            self.broken_mutation == "event"
            and self._is_contract_racer()
            and self._is_current(event.run_id, writer_token)
            and key not in self._events
        ):
            self._race_barrier.wait(timeout=10)
            with self._lock:
                self._events.pop(key, None)
                return self._append_event(event, writer_token=writer_token)
        return super().append_event(event, writer_token=writer_token)

    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        key = (invocation.run_id, invocation.logical_call_id, invocation.revision)
        head_key = (invocation.run_id, invocation.logical_call_id)
        if (
            self.broken_mutation == "invocation"
            and self._is_contract_racer()
            and self._is_current(invocation.run_id, writer_token)
            and key not in self._invocations
        ):
            self._race_barrier.wait(timeout=10)
            with self._lock:
                self._invocations.pop(key, None)
                self._invocation_heads.pop(head_key, None)
                return self._commit_invocation(invocation, blobs, writer_token=writer_token)
        return super().commit_invocation(invocation, blobs, writer_token=writer_token)

    def settle_terminal(
        self,
        outcome: TerminalOutcome,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        if (
            self.broken_mutation == "terminal"
            and self._is_contract_racer()
            and self._is_current(outcome.run_id, writer_token)
            and outcome.run_id not in self._terminals
        ):
            self._race_barrier.wait(timeout=10)
            with self._lock:
                self._terminals.pop(outcome.run_id, None)
                return self._settle_terminal(outcome, writer_token=writer_token)
        return super().settle_terminal(outcome, writer_token=writer_token)


def _non_atomic_race_factory(mutation: str):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _NonAtomicRaceSink(harness._writers)
        sink.broken_mutation = mutation
        sink._race_barrier = Barrier(2)
        harness.sink = sink
        return harness

    return factory


@pytest.mark.parametrize("mutation", ["checkpoint", "event", "invocation", "terminal"])
def test_reusable_contract_rejects_non_atomic_competing_writers(mutation: str) -> None:
    outcomes = run_fenced_run_sink_contract(_non_atomic_race_factory(mutation))
    winner_rule = next(
        outcome
        for outcome in outcomes
        if outcome.rule_id == "FENCED-03-EVENT-AND-TERMINAL-WINNERS"
    )
    race_observation = next(
        observation
        for observation in winner_rule.observations
        if observation.observation_id == f"{mutation}_race_statuses"
    )

    assert winner_rule.status == "failed"
    assert race_observation.expected == ("committed", "conflict")
    assert race_observation.actual == ("committed", "committed")


class _UnsafeWriterHandoffHarness(DeterministicFencedRunHarness):
    broken_mutation = ""

    def race_writer_handoff(
        self,
        mutation: str,
        stale_token: WriterToken,
        current_token: WriterToken,
        write,
    ) -> tuple[CommitResult, CommitResult, bool]:
        if mutation == self.broken_mutation:
            del stale_token, current_token, write
            return (
                CommitResult(status="committed"),
                CommitResult(status="already_committed"),
                True,
            )
        return super().race_writer_handoff(
            mutation,
            stale_token,
            current_token,
            write,
        )


def _unsafe_writer_handoff_factory(mutation: str):
    def factory() -> _UnsafeWriterHandoffHarness:
        harness = _UnsafeWriterHandoffHarness()
        harness.broken_mutation = mutation
        return harness

    return factory


@pytest.mark.parametrize("mutation", ["checkpoint", "event", "invocation", "terminal"])
def test_reusable_contract_rejects_write_published_after_rotation(mutation: str) -> None:
    outcomes = run_fenced_run_sink_contract(_unsafe_writer_handoff_factory(mutation))
    fence_rule = next(
        outcome
        for outcome in outcomes
        if outcome.rule_id == "FENCED-02-FENCE-PRECEDES-IDEMPOTENCY"
    )
    handoff_observation = next(
        observation
        for observation in fence_rule.observations
        if observation.observation_id == f"handoff_{mutation}_linearization"
    )

    assert fence_rule.status == "failed"
    assert handoff_observation.expected == ("fenced", "committed")
    assert handoff_observation.actual == ("committed", "already_committed")


class _PersistentHarnessFactory:
    def __init__(self) -> None:
        self.root = DeterministicFencedRunHarness()

    def __call__(self) -> DeterministicFencedRunHarness:
        return self.root.reopen()


def test_reusable_contract_namespaces_repeated_runs_on_one_backing_store() -> None:
    factory = _PersistentHarnessFactory()

    first = run_fenced_run_sink_contract(factory)
    second = run_fenced_run_sink_contract(factory)

    assert all(outcome.status == "passed" for outcome in first), first
    assert all(outcome.status == "passed" for outcome in second), second
