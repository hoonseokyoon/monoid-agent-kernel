from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from copy import copy
from dataclasses import replace
from threading import Barrier

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
    "schema_version",
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
        outcome for outcome in outcomes if outcome.rule_id == "FENCED-02-FENCE-PRECEDES-IDEMPOTENCY"
    )

    assert fence_rule.status == "failed"
    assert any(
        observation.expected == "fenced" and observation.actual == "already_committed"
        for observation in fence_rule.observations
    )


class _PartialWriterAuthoritySink(DeterministicFencedRunSink):
    compared_field = ""

    def _is_current(self, run_id: str, writer_token: WriterToken) -> bool:
        current = self.current_writers.get(run_id)
        return (
            writer_token.run_id == run_id
            and current is not None
            and getattr(writer_token, self.compared_field) == getattr(current, self.compared_field)
        )


def _partial_writer_authority_factory(compared_field: str):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _PartialWriterAuthoritySink(harness._writers)
        sink.compared_field = compared_field
        harness.sink = sink
        return harness

    return factory


@pytest.mark.parametrize(
    ("compared_field", "accepted_authority_case"),
    [
        ("owner_id", "stale_generation_current_owner"),
        ("generation", "wrong_owner_current_generation"),
    ],
)
def test_reusable_contract_checks_owner_and_generation_independently(
    compared_field: str,
    accepted_authority_case: str,
) -> None:
    outcomes = run_fenced_run_sink_contract(_partial_writer_authority_factory(compared_field))
    fence_rule = next(
        outcome for outcome in outcomes if outcome.rule_id == "FENCED-02-FENCE-PRECEDES-IDEMPOTENCY"
    )
    actual = {
        observation.observation_id: observation.actual
        for observation in fence_rule.observations
        if observation.observation_id.startswith(f"existing_{accepted_authority_case}_")
    }

    assert fence_rule.status == "failed"
    assert actual == {
        f"existing_{accepted_authority_case}_{mutation}": "already_committed"
        for mutation in ("checkpoint", "event", "invocation", "terminal")
    }


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
        outcome for outcome in outcomes if outcome.rule_id == "FENCED-02-FENCE-PRECEDES-IDEMPOTENCY"
    )

    assert fence_rule.status == "failed"
    stale_observation = next(
        observation
        for observation in fence_rule.observations
        if observation.observation_id == f"fresh_stale_owner_and_generation_{mutation}"
    )
    assert stale_observation.expected == "fenced"
    assert stale_observation.actual == "committed"


class _PartialFreshResourceAuthoritySink(_MissingResourceFenceBypassSink):
    compared_field = ""

    def _is_current(self, run_id: str, writer_token: WriterToken) -> bool:
        if not self._bypass_current_writer:
            return DeterministicFencedRunSink._is_current(self, run_id, writer_token)
        current = self.current_writers.get(run_id)
        return (
            writer_token.run_id == run_id
            and current is not None
            and getattr(writer_token, self.compared_field) == getattr(current, self.compared_field)
        )


def _partial_fresh_resource_authority_factory(
    mutation: str,
    compared_field: str,
):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _PartialFreshResourceAuthoritySink(harness._writers)
        sink.broken_mutation = mutation
        sink.compared_field = compared_field
        harness.sink = sink
        return harness

    return factory


@pytest.mark.parametrize("mutation", ["checkpoint", "event", "invocation", "terminal"])
@pytest.mark.parametrize(
    ("compared_field", "accepted_authority_case"),
    [
        ("owner_id", "stale_generation_current_owner"),
        ("generation", "wrong_owner_current_generation"),
    ],
)
def test_reusable_contract_checks_fresh_resource_authority_dimensions(
    mutation: str,
    compared_field: str,
    accepted_authority_case: str,
) -> None:
    outcomes = run_fenced_run_sink_contract(
        _partial_fresh_resource_authority_factory(mutation, compared_field)
    )
    fence_rule = next(
        outcome for outcome in outcomes if outcome.rule_id == "FENCED-02-FENCE-PRECEDES-IDEMPOTENCY"
    )
    observation = next(
        observation
        for observation in fence_rule.observations
        if observation.observation_id == f"fresh_{accepted_authority_case}_{mutation}"
    )

    assert fence_rule.status == "failed"
    assert observation.expected == "fenced"
    assert observation.actual == "committed"


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
        "retry_after_unknown_with_failure_code": "committed",
        "retry_after_success": "committed",
        "retry_after_retryable_tagged_success": "committed",
        "retry_after_nonretry_failure": "committed",
        "retry_after_retryable_failure": "committed",
    }


class _AllocationPolicyIndependentHarness(DeterministicFencedRunHarness):
    def claim_writer(self, run_id: str, owner_id: str) -> WriterToken:
        del run_id, owner_id
        raise AssertionError("the conformance contract must install exact writer tokens")


def test_reusable_contract_does_not_call_the_host_generation_allocator() -> None:
    outcomes = run_fenced_run_sink_contract(_AllocationPolicyIndependentHarness)

    assert all(outcome.status == "passed" for outcome in outcomes), outcomes


class _UnscopedResourceKeySink(DeterministicFencedRunSink):
    broken_mutation = ""
    _unscoped_winners: dict[tuple[str, object], str]

    def _unscoped_collision(
        self,
        mutation: str,
        local_key: object,
        run_id: str,
        writer_token: WriterToken,
    ) -> CommitResult | None:
        if self.broken_mutation != mutation or not self._is_current(run_id, writer_token):
            return None
        resource_key = (mutation, local_key)
        winner_run = self._unscoped_winners.setdefault(resource_key, run_id)
        if winner_run != run_id:
            return CommitResult(status="conflict")
        return None

    def commit_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        collision = self._unscoped_collision(
            "checkpoint",
            checkpoint.seq,
            checkpoint.run_id,
            writer_token,
        )
        return collision or super().commit_checkpoint(
            checkpoint,
            blobs,
            writer_token=writer_token,
        )

    def append_event(
        self,
        event: AgentEvent,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        collision = self._unscoped_collision(
            "event",
            event.seq,
            event.run_id,
            writer_token,
        )
        return collision or super().append_event(event, writer_token=writer_token)

    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        collision = self._unscoped_collision(
            "invocation",
            (invocation.logical_call_id, invocation.revision),
            invocation.run_id,
            writer_token,
        )
        return collision or super().commit_invocation(
            invocation,
            blobs,
            writer_token=writer_token,
        )

    def settle_terminal(
        self,
        outcome: TerminalOutcome,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        collision = self._unscoped_collision(
            "terminal",
            "terminal",
            outcome.run_id,
            writer_token,
        )
        return collision or super().settle_terminal(outcome, writer_token=writer_token)


def _unscoped_resource_key_factory(mutation: str):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _UnscopedResourceKeySink(harness._writers)
        sink.broken_mutation = mutation
        sink._unscoped_winners = {}
        harness.sink = sink
        return harness

    return factory


@pytest.mark.parametrize("mutation", ["checkpoint", "event", "invocation", "terminal"])
def test_reusable_contract_scopes_each_resource_key_by_run(mutation: str) -> None:
    outcomes = run_fenced_run_sink_contract(_unscoped_resource_key_factory(mutation))
    binding_rule = next(
        outcome for outcome in outcomes if outcome.rule_id == "FENCED-06-WRITER-TOKEN-RUN-BINDING"
    )
    run_b_observation = next(
        observation
        for observation in binding_rule.observations
        if observation.observation_id == f"run_b_bound_{mutation}"
    )

    assert binding_rule.status == "failed"
    assert run_b_observation.expected == "committed"
    assert run_b_observation.actual == "conflict"


class _CrossRunPayloadAliasHarness(DeterministicFencedRunHarness):
    broken_mutation = ""

    @staticmethod
    def _run_b_id(run_id: str) -> str:
        return f"{run_id.removesuffix('run-a')}run-b"

    def read_event(self, run_id: str, seq: int) -> AgentEvent | None:
        if self.broken_mutation == "event" and run_id.endswith("run-a"):
            return super().read_event(self._run_b_id(run_id), seq)
        return super().read_event(run_id, seq)

    def read_terminal(self, run_id: str) -> TerminalOutcome | None:
        if self.broken_mutation == "terminal" and run_id.endswith("run-a"):
            return super().read_terminal(self._run_b_id(run_id))
        return super().read_terminal(run_id)


def _cross_run_payload_alias_factory(mutation: str):
    def factory() -> _CrossRunPayloadAliasHarness:
        harness = _CrossRunPayloadAliasHarness()
        harness.broken_mutation = mutation
        return harness

    return factory


@pytest.mark.parametrize("mutation", ["event", "terminal"])
def test_reusable_contract_reads_each_runs_complete_payload(mutation: str) -> None:
    outcomes = run_fenced_run_sink_contract(_cross_run_payload_alias_factory(mutation))
    binding_rule = next(
        outcome for outcome in outcomes if outcome.rule_id == "FENCED-06-WRITER-TOKEN-RUN-BINDING"
    )
    payload_observation = next(
        observation
        for observation in binding_rule.observations
        if observation.observation_id == f"run_a_{mutation}_payload"
    )

    assert binding_rule.status == "failed"
    assert payload_observation.actual != payload_observation.expected


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
        outcome for outcome in outcomes if outcome.rule_id == "FENCED-00-CAPABILITY-DECLARATION"
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


class _UncheckedBlobDigestSink(DeterministicFencedRunSink):
    def _blobs_are_content_addressed(self, blobs: Mapping[str, bytes]) -> bool:
        del blobs
        return True

    def _blobs_preserve_authoritative_backing(
        self,
        run_id: str,
        blobs: Mapping[str, bytes],
    ) -> bool:
        del run_id, blobs
        return True


def _unchecked_blob_digest_factory() -> DeterministicFencedRunHarness:
    harness = DeterministicFencedRunHarness()
    harness.sink = _UncheckedBlobDigestSink(harness._writers)
    return harness


def test_reusable_contract_rejects_fresh_malformed_content_addressed_blobs() -> None:
    outcomes = run_fenced_run_sink_contract(_unchecked_blob_digest_factory)
    rules = {outcome.rule_id: outcome for outcome in outcomes}

    checkpoint_rule = rules["FENCED-01-CHECKPOINT-CONTENT-IDENTITY"]
    invocation_rule = rules["FENCED-04-INVOCATION-LIFECYCLE"]
    checkpoint_observations = {item.observation_id: item for item in checkpoint_rule.observations}
    invocation_observations = {item.observation_id: item for item in invocation_rule.observations}

    assert checkpoint_rule.status == "failed"
    assert invocation_rule.status == "failed"
    assert checkpoint_observations["malformed_fresh_blob_status"].actual == "committed"
    assert checkpoint_observations["malformed_fresh_blob_head_not_published"].actual == 2
    assert invocation_observations["malformed_fresh_blob_status"].actual == "committed"
    assert invocation_observations["malformed_fresh_blob_head_not_published"].actual == 3


class _CaseFoldingBlobDigestSink(DeterministicFencedRunSink):
    def _blobs_are_content_addressed(self, blobs: Mapping[str, bytes]) -> bool:
        return all(
            type(value) is bytes and hashlib.sha256(value).hexdigest() == key.lower()
            for key, value in blobs.items()
        )


def _case_folding_blob_digest_factory() -> DeterministicFencedRunHarness:
    harness = DeterministicFencedRunHarness()
    harness.sink = _CaseFoldingBlobDigestSink(harness._writers)
    return harness


def test_reusable_contract_rejects_uppercase_content_digest_keys() -> None:
    outcomes = run_fenced_run_sink_contract(_case_folding_blob_digest_factory)
    rules = {outcome.rule_id: outcome for outcome in outcomes}
    checkpoint_rule = rules["FENCED-01-CHECKPOINT-CONTENT-IDENTITY"]
    invocation_rule = rules["FENCED-04-INVOCATION-LIFECYCLE"]
    checkpoint_observations = {
        item.observation_id: item.actual for item in checkpoint_rule.observations
    }
    invocation_observations = {
        item.observation_id: item.actual for item in invocation_rule.observations
    }

    assert checkpoint_rule.status == "failed"
    assert invocation_rule.status == "failed"
    assert checkpoint_observations["uppercase_blob_key_status"] == "committed"
    assert checkpoint_observations["uppercase_blob_key_not_published"] == "loaded"
    assert invocation_observations["uppercase_blob_key_status"] == "committed"
    assert invocation_observations["uppercase_blob_key_not_published"] == "loaded"


def _corrupting_blob_reader(reader, target_sha256: str, corrupted: bytes):
    def read(sha256: str) -> bytes:
        if sha256 == target_sha256:
            return corrupted
        return reader(sha256)

    return read


class _OverwriteSameRunBlobOnMalformedSink(DeterministicFencedRunSink):
    def _overwrite_checkpoint_blobs(
        self,
        run_id: str,
        blobs: Mapping[str, bytes],
    ) -> None:
        for sha256, corrupted in blobs.items():
            for (record_run_id, _), (_, record) in self._checkpoints.items():
                if record_run_id != run_id:
                    continue
                try:
                    record.blob(sha256)
                except KeyError:
                    continue
                assert record._blob_reader is not None
                record._blob_reader = _corrupting_blob_reader(
                    record._blob_reader,
                    sha256,
                    corrupted,
                )

    def _overwrite_invocation_blobs(
        self,
        run_id: str,
        blobs: Mapping[str, bytes],
    ) -> None:
        for sha256, corrupted in blobs.items():
            for key, (digest, record) in list(self._invocations.items()):
                if key[0] != run_id:
                    continue
                try:
                    record.blob(sha256)
                except KeyError:
                    continue
                assert record._blob_reader is not None
                self._invocations[key] = (
                    digest,
                    replace(
                        record,
                        _blob_reader=_corrupting_blob_reader(
                            record._blob_reader,
                            sha256,
                            corrupted,
                        ),
                    ),
                )

    def commit_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        if not self._blobs_are_content_addressed(blobs):
            self._overwrite_checkpoint_blobs(checkpoint.run_id, blobs)
        return super().commit_checkpoint(checkpoint, blobs, writer_token=writer_token)

    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        if not self._blobs_are_content_addressed(blobs):
            self._overwrite_invocation_blobs(invocation.run_id, blobs)
        return super().commit_invocation(invocation, blobs, writer_token=writer_token)


def _overwrite_existing_blob_factory() -> DeterministicFencedRunHarness:
    harness = DeterministicFencedRunHarness()
    harness.sink = _OverwriteSameRunBlobOnMalformedSink(harness._writers)
    return harness


def test_reusable_contract_protects_same_run_blobs_from_malformed_writes() -> None:
    outcomes = run_fenced_run_sink_contract(_overwrite_existing_blob_factory)
    rules = {outcome.rule_id: outcome for outcome in outcomes}

    for rule_id in (
        "FENCED-01-CHECKPOINT-CONTENT-IDENTITY",
        "FENCED-04-INVOCATION-LIFECYCLE",
    ):
        rule = rules[rule_id]
        preserved = next(
            item
            for item in rule.observations
            if item.observation_id == "malformed_fresh_blob_preserves_existing_bytes"
        )
        assert rule.status == "failed"
        assert preserved.actual != preserved.expected


class _BlobValidationFirstSink(DeterministicFencedRunSink):
    def commit_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        if not self._blobs_are_content_addressed(blobs):
            return CommitResult(status="conflict", sequence=checkpoint.seq)
        return super().commit_checkpoint(checkpoint, blobs, writer_token=writer_token)

    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        if not self._blobs_are_content_addressed(blobs):
            return CommitResult(status="conflict", sequence=invocation.revision)
        return super().commit_invocation(invocation, blobs, writer_token=writer_token)


def _blob_validation_first_factory() -> DeterministicFencedRunHarness:
    harness = DeterministicFencedRunHarness()
    harness.sink = _BlobValidationFirstSink(harness._writers)
    return harness


def test_reusable_contract_checks_fencing_before_malformed_blob_validation() -> None:
    outcomes = run_fenced_run_sink_contract(_blob_validation_first_factory)
    rules = {outcome.rule_id: outcome for outcome in outcomes}
    observation_ids = {
        "FENCED-02-FENCE-PRECEDES-IDEMPOTENCY": (
            "stale_malformed_checkpoint",
            "stale_malformed_invocation",
            "malformed_stale_generation_current_owner_checkpoint",
            "malformed_stale_generation_current_owner_invocation",
            "malformed_wrong_owner_current_generation_checkpoint",
            "malformed_wrong_owner_current_generation_invocation",
        ),
        "FENCED-06-WRITER-TOKEN-RUN-BINDING": (
            "cross_run_malformed_checkpoint",
            "cross_run_malformed_invocation",
        ),
    }

    for rule_id, expected_observation_ids in observation_ids.items():
        rule = rules[rule_id]
        selected = {
            item.observation_id: item
            for item in rule.observations
            if item.observation_id in expected_observation_ids
        }
        assert rule.status == "failed"
        assert set(selected) == set(expected_observation_ids)
        assert all(item.expected == "fenced" for item in selected.values())
        assert all(item.actual == "conflict" for item in selected.values())


class _InvalidCommitEvidenceSink(DeterministicFencedRunSink):
    broken_mutation = ""
    broken_status = ""
    evidence_field = ""

    def _corrupt_evidence(self, mutation: str, result: CommitResult) -> CommitResult:
        if mutation != self.broken_mutation or result.status != self.broken_status:
            return result
        value: int | str
        if self.evidence_field == "sequence":
            value = (result.sequence or 0) + 100
        else:
            value = "f" * 64
        return replace(result, **{self.evidence_field: value})

    def commit_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        return self._corrupt_evidence(
            "checkpoint",
            super().commit_checkpoint(checkpoint, blobs, writer_token=writer_token),
        )

    def append_event(
        self,
        event: AgentEvent,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        return self._corrupt_evidence(
            "event",
            super().append_event(event, writer_token=writer_token),
        )

    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        return self._corrupt_evidence(
            "invocation",
            super().commit_invocation(invocation, blobs, writer_token=writer_token),
        )

    def settle_terminal(
        self,
        outcome: TerminalOutcome,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        return self._corrupt_evidence(
            "terminal",
            super().settle_terminal(outcome, writer_token=writer_token),
        )


def _invalid_commit_evidence_factory(
    mutation: str,
    status: str,
    evidence_field: str,
):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _InvalidCommitEvidenceSink(harness._writers)
        sink.broken_mutation = mutation
        sink.broken_status = status
        sink.evidence_field = evidence_field
        harness.sink = sink
        return harness

    return factory


@pytest.mark.parametrize(
    ("mutation", "rule_id"),
    [
        ("checkpoint", "FENCED-01-CHECKPOINT-CONTENT-IDENTITY"),
        ("event", "FENCED-03-EVENT-AND-TERMINAL-WINNERS"),
        ("invocation", "FENCED-04-INVOCATION-LIFECYCLE"),
        ("terminal", "FENCED-03-EVENT-AND-TERMINAL-WINNERS"),
    ],
)
@pytest.mark.parametrize(
    "status",
    ["committed", "already_committed", "conflict"],
)
@pytest.mark.parametrize(
    ("evidence_field", "evidence_index"),
    [("sequence", 0), ("content_digest", 1), ("winner_digest", 2)],
)
def test_reusable_contract_validates_every_populated_commit_result_field(
    mutation: str,
    rule_id: str,
    status: str,
    evidence_field: str,
    evidence_index: int,
) -> None:
    outcomes = run_fenced_run_sink_contract(
        _invalid_commit_evidence_factory(mutation, status, evidence_field)
    )
    rule = next(outcome for outcome in outcomes if outcome.rule_id == rule_id)
    evidence_observation = next(
        observation
        for observation in rule.observations
        if observation.observation_id == f"{mutation}_{status}_evidence"
    )

    assert rule.status == "failed"
    assert evidence_observation.expected == (True, True, True)
    assert evidence_observation.actual[evidence_index] is False


class _ReferentialIntegritySink(DeterministicFencedRunSink):
    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        if invocation.result_ref.startswith("blob:"):
            result_sha256 = invocation.result_ref.removeprefix("blob:")
            result_blob = blobs.get(result_sha256)
            if result_blob is None:
                result_blob = self._blobs.get((invocation.run_id, result_sha256))
            if result_blob is None or hashlib.sha256(result_blob).hexdigest() != result_sha256:
                return CommitResult(status="conflict", sequence=invocation.revision)
        return super().commit_invocation(invocation, blobs, writer_token=writer_token)


def _referential_integrity_factory() -> DeterministicFencedRunHarness:
    harness = DeterministicFencedRunHarness()
    harness.sink = _ReferentialIntegritySink(harness._writers)
    return harness


def test_reusable_contract_resolves_every_referenced_invocation_blob() -> None:
    outcomes = run_fenced_run_sink_contract(_referential_integrity_factory)

    assert all(outcome.status == "passed" for outcome in outcomes), outcomes


class _MissingReferenceAcceptingSink(DeterministicFencedRunSink):
    def _checkpoint_references_resolve(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
    ) -> bool:
        del checkpoint, blobs
        return True

    def _invocation_references_resolve(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
    ) -> bool:
        del invocation, blobs
        return True


def _missing_reference_accepting_factory() -> DeterministicFencedRunHarness:
    harness = DeterministicFencedRunHarness()
    harness.sink = _MissingReferenceAcceptingSink(harness._writers)
    return harness


def test_reusable_contract_rejects_unresolved_authoritative_blob_references() -> None:
    outcomes = run_fenced_run_sink_contract(_missing_reference_accepting_factory)
    rules = {outcome.rule_id: outcome for outcome in outcomes}

    checkpoint_rule = rules["FENCED-01-CHECKPOINT-CONTENT-IDENTITY"]
    invocation_rule = rules["FENCED-04-INVOCATION-LIFECYCLE"]
    checkpoint_status = next(
        observation.actual
        for observation in checkpoint_rule.observations
        if observation.observation_id == "missing_reference_status"
    )
    invocation_status = next(
        observation.actual
        for observation in invocation_rule.observations
        if observation.observation_id == "missing_reference_status"
    )

    assert checkpoint_rule.status == "failed"
    assert invocation_rule.status == "failed"
    assert checkpoint_status == "committed"
    assert invocation_status == "committed"


class _WorkspaceOnlyCheckpointReferenceSink(DeterministicFencedRunSink):
    def _checkpoint_blob_references(self, checkpoint: RunCheckpoint) -> set[str]:
        return {
            item["content_sha256"]
            for item in checkpoint.workspace_delta
            if isinstance(item.get("content_sha256"), str) and item["content_sha256"]
        }


def _workspace_only_checkpoint_reference_factory() -> DeterministicFencedRunHarness:
    harness = DeterministicFencedRunHarness()
    harness.sink = _WorkspaceOnlyCheckpointReferenceSink(harness._writers)
    return harness


def test_reusable_contract_checks_media_references_inside_checkpoint_messages() -> None:
    outcomes = run_fenced_run_sink_contract(_workspace_only_checkpoint_reference_factory)
    checkpoint_rule = next(
        outcome
        for outcome in outcomes
        if outcome.rule_id == "FENCED-01-CHECKPOINT-CONTENT-IDENTITY"
    )
    observations = {item.observation_id: item.actual for item in checkpoint_rule.observations}

    assert checkpoint_rule.status == "failed"
    assert observations["missing_reference_status"] == "conflict"
    assert observations["missing_media_reference_status"] == "committed"
    assert observations["missing_media_reference_not_published"] == "loaded"


class _SubmittedMapOnlyReferenceSink(DeterministicFencedRunSink):
    def _reference_is_available(
        self,
        run_id: str,
        sha256: str,
        blobs: Mapping[str, bytes],
    ) -> bool:
        del run_id
        return sha256 in blobs


def _submitted_map_only_reference_factory() -> DeterministicFencedRunHarness:
    harness = DeterministicFencedRunHarness()
    harness.sink = _SubmittedMapOnlyReferenceSink(harness._writers)
    return harness


def test_reusable_contract_accepts_references_from_same_run_authoritative_backing() -> None:
    outcomes = run_fenced_run_sink_contract(_submitted_map_only_reference_factory)
    rules = {outcome.rule_id: outcome for outcome in outcomes}
    checkpoint_rule = rules["FENCED-01-CHECKPOINT-CONTENT-IDENTITY"]
    invocation_rule = rules["FENCED-04-INVOCATION-LIFECYCLE"]
    checkpoint_observations = {
        item.observation_id: item.actual for item in checkpoint_rule.observations
    }
    invocation_observations = {
        item.observation_id: item.actual for item in invocation_rule.observations
    }

    assert checkpoint_rule.status == "failed"
    assert invocation_rule.status == "failed"
    assert checkpoint_observations["authoritative_backing_reference_statuses"] == (
        "committed",
        "conflict",
    )
    assert invocation_observations["authoritative_backing_reference_status"] == "conflict"


class _GlobalBackingReferenceSink(DeterministicFencedRunSink):
    def _reference_is_available(
        self,
        run_id: str,
        sha256: str,
        blobs: Mapping[str, bytes],
    ) -> bool:
        return super()._reference_is_available(run_id, sha256, blobs) or any(
            stored_sha256 == sha256 for _, stored_sha256 in self._blobs
        )


def _global_backing_reference_factory() -> DeterministicFencedRunHarness:
    harness = DeterministicFencedRunHarness()
    harness.sink = _GlobalBackingReferenceSink(harness._writers)
    return harness


def test_reusable_contract_rejects_references_backed_only_by_another_run() -> None:
    outcomes = run_fenced_run_sink_contract(_global_backing_reference_factory)
    binding_rule = next(
        outcome for outcome in outcomes if outcome.rule_id == "FENCED-06-WRITER-TOKEN-RUN-BINDING"
    )
    observations = {item.observation_id: item.actual for item in binding_rule.observations}

    assert binding_rule.status == "failed"
    assert observations["cross_run_blob_checkpoint_reference"] == "committed"
    assert observations["cross_run_blob_invocation_reference"] == "committed"


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


class _RejectingDelayedCheckpointSink(DeterministicFencedRunSink):
    def commit_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        current_head = self._checkpoint_heads.get(checkpoint.run_id, -1)
        if (
            self._is_current(checkpoint.run_id, writer_token)
            and (checkpoint.run_id, checkpoint.seq) not in self._checkpoints
            and checkpoint.seq < current_head
        ):
            return CommitResult(status="conflict", sequence=checkpoint.seq)
        return super().commit_checkpoint(checkpoint, blobs, writer_token=writer_token)


def _rejecting_delayed_checkpoint_factory() -> DeterministicFencedRunHarness:
    harness = DeterministicFencedRunHarness()
    harness.sink = _RejectingDelayedCheckpointSink(harness._writers)
    return harness


def test_reusable_contract_commits_fresh_delayed_checkpoint_coordinates() -> None:
    outcomes = run_fenced_run_sink_contract(_rejecting_delayed_checkpoint_factory)
    checkpoint_rule = next(
        outcome
        for outcome in outcomes
        if outcome.rule_id == "FENCED-01-CHECKPOINT-CONTENT-IDENTITY"
    )
    observations = {item.observation_id: item.actual for item in checkpoint_rule.observations}

    assert checkpoint_rule.status == "failed"
    assert observations["delayed_checkpoint"] == "conflict"
    assert observations["head_after_delayed_sequence"] == 2


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
                self._invocation_heads[(invocation.run_id, invocation.logical_call_id)] = (
                    invocation.revision
                )
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
        outcome for outcome in outcomes if outcome.rule_id == "FENCED-04-INVOCATION-LIFECYCLE"
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


class _RawInvocationAliasDigestSink(DeterministicFencedRunSink):
    field_name = ""
    broken_direction = ""

    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        key = (invocation.run_id, invocation.logical_call_id, invocation.revision)
        stored = self._invocations.get(key)
        if stored is not None and self._is_current(invocation.run_id, writer_token):
            previous_value = getattr(stored[1].invocation, self.field_name)
            incoming_value = getattr(invocation, self.field_name)
            previous_is_legacy = previous_value.startswith("native-agent-runner.")
            incoming_is_legacy = incoming_value.startswith("native-agent-runner.")
            direction = "legacy_to_current" if previous_is_legacy else "current_to_legacy"
            if (
                previous_value != incoming_value
                and previous_is_legacy != incoming_is_legacy
                and direction == self.broken_direction
            ):
                return CommitResult(status="conflict", sequence=invocation.revision)
        return super().commit_invocation(invocation, blobs, writer_token=writer_token)


def _raw_invocation_alias_digest_factory(field_name: str, direction: str):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _RawInvocationAliasDigestSink(harness._writers)
        sink.field_name = field_name
        sink.broken_direction = direction
        harness.sink = sink
        return harness

    return factory


@pytest.mark.parametrize(
    ("field_name", "direction"),
    [
        (field_name, direction)
        for field_name in ("schema_version", "digest_generation")
        for direction in ("current_to_legacy", "legacy_to_current")
    ],
)
def test_reusable_contract_normalizes_invocation_alias_retries_both_directions(
    field_name: str,
    direction: str,
) -> None:
    outcomes = run_fenced_run_sink_contract(
        _raw_invocation_alias_digest_factory(field_name, direction)
    )
    lifecycle_rule = next(
        outcome for outcome in outcomes if outcome.rule_id == "FENCED-04-INVOCATION-LIFECYCLE"
    )
    alias_observation = next(
        observation
        for observation in lifecycle_rule.observations
        if observation.observation_id == f"invocation_canonical_alias_{field_name}_{direction}"
    )

    assert lifecycle_rule.status == "failed"
    assert alias_observation.expected == "already_committed"
    assert alias_observation.actual == "conflict"


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
            direction = "legacy_to_current" if previous_is_legacy else "current_to_legacy"
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
        outcome for outcome in outcomes if outcome.rule_id == "FENCED-04-INVOCATION-LIFECYCLE"
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


class _RawCheckpointAliasDigestSink(DeterministicFencedRunSink):
    field_name = ""
    broken_direction = ""

    def _raw_alias(self, checkpoint: RunCheckpoint) -> str:
        if self.field_name == "schema_version":
            return checkpoint.schema_version
        invocation = checkpoint.last_model_invocation or {}
        nested_field = self.field_name.removeprefix("last_model_invocation_")
        return str(invocation.get(nested_field, ""))

    def commit_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        stored = self._checkpoints.get((checkpoint.run_id, checkpoint.seq))
        if stored is not None and self._is_current(checkpoint.run_id, writer_token):
            previous_value = self._raw_alias(stored[1].checkpoint)
            incoming_value = self._raw_alias(checkpoint)
            previous_is_legacy = previous_value.startswith("native-agent-runner.")
            incoming_is_legacy = incoming_value.startswith("native-agent-runner.")
            direction = "legacy_to_current" if previous_is_legacy else "current_to_legacy"
            if (
                previous_value != incoming_value
                and previous_is_legacy != incoming_is_legacy
                and direction == self.broken_direction
            ):
                return CommitResult(status="conflict", sequence=checkpoint.seq)
        return super().commit_checkpoint(checkpoint, blobs, writer_token=writer_token)


def _raw_checkpoint_alias_digest_factory(field_name: str, direction: str):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _RawCheckpointAliasDigestSink(harness._writers)
        sink.field_name = field_name
        sink.broken_direction = direction
        harness.sink = sink
        return harness

    return factory


@pytest.mark.parametrize(
    ("field_name", "direction"),
    [
        (field_name, direction)
        for field_name in (
            "schema_version",
            "last_model_invocation_schema_version",
            "last_model_invocation_digest_generation",
        )
        for direction in ("current_to_legacy", "legacy_to_current")
    ],
)
def test_reusable_contract_normalizes_every_checkpoint_alias_location(
    field_name: str,
    direction: str,
) -> None:
    outcomes = run_fenced_run_sink_contract(
        _raw_checkpoint_alias_digest_factory(field_name, direction)
    )
    checkpoint_rule = next(
        outcome
        for outcome in outcomes
        if outcome.rule_id == "FENCED-01-CHECKPOINT-CONTENT-IDENTITY"
    )
    alias_observation = next(
        observation
        for observation in checkpoint_rule.observations
        if observation.observation_id == f"checkpoint_canonical_alias_{field_name}_{direction}"
    )

    assert checkpoint_rule.status == "failed"
    assert alias_observation.expected == "already_committed"
    assert alias_observation.actual == "conflict"


class _RawTerminalAliasDigestSink(DeterministicFencedRunSink):
    broken_direction = ""

    def settle_terminal(
        self,
        outcome: TerminalOutcome,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        stored = self._terminals.get(outcome.run_id)
        if stored is not None and self._is_current(outcome.run_id, writer_token):
            previous_schema = stored[1].schema_version
            incoming_schema = outcome.schema_version
            previous_is_legacy = previous_schema.startswith("native-agent-runner.")
            incoming_is_legacy = incoming_schema.startswith("native-agent-runner.")
            direction = "legacy_to_current" if previous_is_legacy else "current_to_legacy"
            if (
                previous_schema != incoming_schema
                and previous_is_legacy != incoming_is_legacy
                and direction == self.broken_direction
            ):
                return CommitResult(status="conflict")
        return super().settle_terminal(outcome, writer_token=writer_token)


def _raw_terminal_alias_digest_factory(direction: str):
    def factory() -> DeterministicFencedRunHarness:
        harness = DeterministicFencedRunHarness()
        sink = _RawTerminalAliasDigestSink(harness._writers)
        sink.broken_direction = direction
        harness.sink = sink
        return harness

    return factory


@pytest.mark.parametrize("direction", ["current_to_legacy", "legacy_to_current"])
def test_reusable_contract_normalizes_terminal_schema_aliases(
    direction: str,
) -> None:
    outcomes = run_fenced_run_sink_contract(_raw_terminal_alias_digest_factory(direction))
    terminal_rule = next(
        outcome for outcome in outcomes if outcome.rule_id == "FENCED-03-EVENT-AND-TERMINAL-WINNERS"
    )
    alias_observation = next(
        observation
        for observation in terminal_rule.observations
        if observation.observation_id == f"terminal_canonical_alias_{direction}"
    )

    assert terminal_rule.status == "failed"
    assert alias_observation.expected == "already_committed"
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


class _CorruptingLoadedRecordHarness(DeterministicFencedRunHarness):
    corruption = ""

    def read_event(self, run_id: str, seq: int) -> AgentEvent | None:
        event = super().read_event(run_id, seq)
        if self.corruption != "event" or event is None:
            return event
        return replace(event, data={"discarded": True})

    def read_terminal(self, run_id: str) -> TerminalOutcome | None:
        terminal = super().read_terminal(run_id)
        if self.corruption != "terminal" or terminal is None:
            return terminal
        return replace(terminal, final_output_ref="blob:discarded")


def _corrupting_loaded_record_factory(corruption: str):
    def factory() -> _CorruptingLoadedRecordHarness:
        harness = _CorruptingLoadedRecordHarness()
        harness.corruption = corruption
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
        (
            "event",
            "FENCED-03-EVENT-AND-TERMINAL-WINNERS",
            "event_reopened_payload_digest",
        ),
        (
            "terminal",
            "FENCED-03-EVENT-AND-TERMINAL-WINNERS",
            "terminal_reopened_payload_digest",
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
                and (invocation.dispatch_attempt, invocation.dispatch_id) == self.allowed_coordinate
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
    outcomes = run_fenced_run_sink_contract(_invalid_retry_coordinate_factory(attempt, dispatch_id))
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


class _ReceiptFlagOnlyRetrySink(DeterministicFencedRunSink):
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
            receipt_allows_retry = (
                previous.dispatch_state == "settled"
                and previous.receipt is not None
                and previous.receipt.get("retryable") is True
            )
            legal_retry_coordinate = (
                invocation.dispatch_state == "reserved"
                and invocation.dispatch_attempt == previous.dispatch_attempt + 1
                and not self._dispatch_id_was_used(invocation)
            )
            if receipt_allows_retry and legal_retry_coordinate:
                return None
        return super()._invocation_transition_winner(invocation)


def _receipt_flag_only_retry_factory() -> DeterministicFencedRunHarness:
    harness = DeterministicFencedRunHarness()
    harness.sink = _ReceiptFlagOnlyRetrySink(harness._writers)
    return harness


def test_reusable_contract_never_retries_a_retryable_tagged_success() -> None:
    outcomes = run_fenced_run_sink_contract(_receipt_flag_only_retry_factory)
    refusal_rule = next(
        outcome
        for outcome in outcomes
        if outcome.rule_id == "FENCED-05-INVOCATION-REFUSES-ILLEGAL-TRANSITIONS"
    )
    observations = {item.observation_id: item.actual for item in refusal_rule.observations}

    assert refusal_rule.status == "failed"
    assert observations["retryable_tagged_success_history"] == (
        "committed",
        "committed",
        "committed",
    )
    assert observations["retry_after_retryable_tagged_success"] == "committed"


class _ImmediatePreviousDispatchOnlySink(DeterministicFencedRunSink):
    def _dispatch_id_was_used(self, invocation: DurableModelInvocation) -> bool:
        head = self._invocation_heads.get((invocation.run_id, invocation.logical_call_id))
        if head is None:
            return False
        _, previous_record = self._invocations[
            (invocation.run_id, invocation.logical_call_id, head)
        ]
        return previous_record.invocation.dispatch_id == invocation.dispatch_id


def _immediate_previous_dispatch_only_factory() -> DeterministicFencedRunHarness:
    harness = DeterministicFencedRunHarness()
    harness.sink = _ImmediatePreviousDispatchOnlySink(harness._writers)
    return harness


def test_reusable_contract_rejects_dispatch_id_reuse_from_any_older_attempt() -> None:
    outcomes = run_fenced_run_sink_contract(_immediate_previous_dispatch_only_factory)
    refusal_rule = next(
        outcome
        for outcome in outcomes
        if outcome.rule_id == "FENCED-05-INVOCATION-REFUSES-ILLEGAL-TRANSITIONS"
    )
    historical_observation = next(
        observation
        for observation in refusal_rule.observations
        if observation.observation_id == "retry_coordinate_historical_dispatch_id"
    )

    assert refusal_rule.status == "failed"
    assert historical_observation.expected == "conflict"
    assert historical_observation.actual == "committed"


class _RejectThirdAttemptSink(DeterministicFencedRunSink):
    def _invocation_transition_winner(
        self,
        invocation: DurableModelInvocation,
    ) -> str | None:
        if invocation.dispatch_attempt > 2:
            head = self._invocation_heads.get((invocation.run_id, invocation.logical_call_id))
            if head is not None:
                return self._invocations[(invocation.run_id, invocation.logical_call_id, head)][0]
        return super()._invocation_transition_winner(invocation)


def _reject_third_attempt_factory() -> DeterministicFencedRunHarness:
    harness = DeterministicFencedRunHarness()
    harness.sink = _RejectThirdAttemptSink(harness._writers)
    return harness


def test_reusable_contract_accepts_a_complete_valid_third_attempt() -> None:
    outcomes = run_fenced_run_sink_contract(_reject_third_attempt_factory)
    refusal_rule = next(
        outcome
        for outcome in outcomes
        if outcome.rule_id == "FENCED-05-INVOCATION-REFUSES-ILLEGAL-TRANSITIONS"
    )
    third_attempt_observation = next(
        observation
        for observation in refusal_rule.observations
        if observation.observation_id == "valid_third_attempt_lifecycle"
    )

    assert refusal_rule.status == "failed"
    assert third_attempt_observation.expected == (
        "committed",
        "committed",
        "committed",
    )
    assert third_attempt_observation.actual == ("conflict", "conflict", "conflict")


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
    _cas_gap: Callable[[], None]
    _race_active = False

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
            and self._race_active
            and self._is_current(checkpoint.run_id, writer_token)
            and key not in self._checkpoints
        ):
            self._cas_gap()
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
            and self._race_active
            and self._is_current(event.run_id, writer_token)
            and key not in self._events
        ):
            self._cas_gap()
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
        previous_head = self._invocation_heads.get(head_key)
        if (
            self.broken_mutation == "invocation"
            and self._race_active
            and self._is_current(invocation.run_id, writer_token)
            and key not in self._invocations
        ):
            self._cas_gap()
            with self._lock:
                self._invocations.pop(key, None)
                if previous_head is None:
                    self._invocation_heads.pop(head_key, None)
                else:
                    self._invocation_heads[head_key] = previous_head
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
            and self._race_active
            and self._is_current(outcome.run_id, writer_token)
            and outcome.run_id not in self._terminals
        ):
            self._cas_gap()
            with self._lock:
                self._terminals.pop(outcome.run_id, None)
                return self._settle_terminal(outcome, writer_token=writer_token)
        return super().settle_terminal(outcome, writer_token=writer_token)


class _NonAtomicRaceHarness(DeterministicFencedRunHarness):
    broken_mutation = ""

    def __post_init__(self) -> None:
        self.sink = _NonAtomicRaceSink(self._writers)
        self.sink._cas_gap = lambda: None

    def race_conflicting_writes(
        self,
        mutation: str,
        writer_token: WriterToken,
        left,
        right,
    ) -> tuple[CommitResult, CommitResult]:
        if mutation != self.broken_mutation:
            return super().race_conflicting_writes(
                mutation,
                writer_token,
                left,
                right,
            )
        barrier = Barrier(2)
        self.sink._cas_gap = lambda: barrier.wait(timeout=10)
        self.sink._race_active = True
        try:
            return super().race_conflicting_writes(
                mutation,
                writer_token,
                left,
                right,
            )
        finally:
            self.sink._race_active = False
            self.sink._cas_gap = lambda: None


def _non_atomic_race_factory(mutation: str):
    def factory() -> _NonAtomicRaceHarness:
        harness = _NonAtomicRaceHarness()
        harness.broken_mutation = mutation
        harness.sink.broken_mutation = mutation
        return harness

    return factory


@pytest.mark.parametrize("mutation", ["checkpoint", "event", "invocation", "terminal"])
def test_reusable_contract_rejects_non_atomic_competing_writers(mutation: str) -> None:
    outcomes = run_fenced_run_sink_contract(_non_atomic_race_factory(mutation))
    winner_rule = next(
        outcome for outcome in outcomes if outcome.rule_id == "FENCED-03-EVENT-AND-TERMINAL-WINNERS"
    )
    race_observation = next(
        observation
        for observation in winner_rule.observations
        if observation.observation_id == f"{mutation}_race_statuses"
    )

    assert winner_rule.status == "failed"
    assert race_observation.expected == ("committed", "conflict")
    assert race_observation.actual == ("committed", "committed")


class _LoserPayloadOverwriteHarness(DeterministicFencedRunHarness):
    broken_mutation = ""

    def race_conflicting_writes(
        self,
        mutation: str,
        writer_token: WriterToken,
        left,
        right,
    ) -> tuple[CommitResult, CommitResult]:
        results = super().race_conflicting_writes(
            mutation,
            writer_token,
            left,
            right,
        )
        if mutation != self.broken_mutation:
            return results
        left_result, right_result = results
        loser_write = right if left_result.status == "committed" else left
        loser_value = loser_write.keywords["value"]
        with self.sink._lock:
            if mutation == "checkpoint":
                key = (writer_token.run_id, 1)
                digest, record = self.sink._checkpoints[key]
                self.sink._checkpoints[key] = (
                    digest,
                    replace(record, checkpoint=loser_value),
                )
            elif mutation == "event":
                key = (writer_token.run_id, 1)
                digest, _ = self.sink._events[key]
                self.sink._events[key] = (digest, loser_value)
            elif mutation == "invocation":
                key = (writer_token.run_id, "call-1", 3)
                digest, record = self.sink._invocations[key]
                self.sink._invocations[key] = (
                    digest,
                    replace(record, invocation=loser_value),
                )
            else:
                digest, _ = self.sink._terminals[writer_token.run_id]
                self.sink._terminals[writer_token.run_id] = (digest, loser_value)
        return left_result, right_result


def _loser_payload_overwrite_factory(mutation: str):
    def factory() -> _LoserPayloadOverwriteHarness:
        harness = _LoserPayloadOverwriteHarness()
        harness.broken_mutation = mutation
        return harness

    return factory


@pytest.mark.parametrize("mutation", ["checkpoint", "event", "invocation", "terminal"])
def test_reusable_contract_reads_the_cas_winner_payload(mutation: str) -> None:
    outcomes = run_fenced_run_sink_contract(_loser_payload_overwrite_factory(mutation))
    winner_rule = next(
        outcome for outcome in outcomes if outcome.rule_id == "FENCED-03-EVENT-AND-TERMINAL-WINNERS"
    )
    payload_observation = next(
        observation
        for observation in winner_rule.observations
        if observation.observation_id == f"{mutation}_race_winner_payload"
    )

    assert winner_rule.status == "failed"
    assert payload_observation.actual != payload_observation.expected


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
        outcome for outcome in outcomes if outcome.rule_id == "FENCED-02-FENCE-PRECEDES-IDEMPOTENCY"
    )
    handoff_observation = next(
        observation
        for observation in fence_rule.observations
        if observation.observation_id == f"handoff_lease_renewal_{mutation}_linearization"
    )

    assert fence_rule.status == "failed"
    assert handoff_observation.expected == ("fenced", "committed")
    assert handoff_observation.actual == ("committed", "already_committed")


class _BlobOutsideWriterHandoffHarness(DeterministicFencedRunHarness):
    broken_mutation = ""

    def race_writer_handoff(
        self,
        mutation: str,
        stale_token: WriterToken,
        current_token: WriterToken,
        write,
    ) -> tuple[CommitResult, CommitResult, bool]:
        if mutation != self.broken_mutation:
            return super().race_writer_handoff(
                mutation,
                stale_token,
                current_token,
                write,
            )
        self.set_current_writer(current_token)
        stale_result = write(self.sink, stale_token)
        stale_blobs = write.keywords["stale_blobs"]
        with self.sink._lock:
            self.sink._publish_blobs(stale_token.run_id, stale_blobs)
        current_result = write(self.sink, current_token)
        return stale_result, current_result, True


def _blob_outside_handoff_factory(mutation: str):
    def factory() -> _BlobOutsideWriterHandoffHarness:
        harness = _BlobOutsideWriterHandoffHarness()
        harness.broken_mutation = mutation
        return harness

    return factory


@pytest.mark.parametrize("mutation", ["checkpoint", "invocation"])
def test_reusable_contract_keeps_blobs_inside_writer_handoff_fencing(
    mutation: str,
) -> None:
    outcomes = run_fenced_run_sink_contract(_blob_outside_handoff_factory(mutation))
    fence_rule = next(
        outcome for outcome in outcomes if outcome.rule_id == "FENCED-02-FENCE-PRECEDES-IDEMPOTENCY"
    )
    blob_observation = next(
        observation
        for observation in fence_rule.observations
        if observation.observation_id == f"handoff_lease_renewal_{mutation}_stale_blob_visibility"
    )

    assert fence_rule.status == "failed"
    assert blob_observation.expected == "conflict"
    assert blob_observation.actual == "committed"


class _HandoffLoserPayloadOverwriteHarness(DeterministicFencedRunHarness):
    broken_mutation = ""

    def race_writer_handoff(
        self,
        mutation: str,
        stale_token: WriterToken,
        current_token: WriterToken,
        write,
    ) -> tuple[CommitResult, CommitResult, bool]:
        results = super().race_writer_handoff(
            mutation,
            stale_token,
            current_token,
            write,
        )
        if mutation != self.broken_mutation:
            return results
        stale_result, current_result, rotation_first = results
        loser_value = (
            write.keywords["stale_value"] if rotation_first else write.keywords["current_value"]
        )
        with self.sink._lock:
            if mutation == "checkpoint":
                key = (stale_token.run_id, 1)
                digest, record = self.sink._checkpoints[key]
                self.sink._checkpoints[key] = (
                    digest,
                    replace(record, checkpoint=loser_value),
                )
            elif mutation == "invocation":
                key = (stale_token.run_id, "call-1", 3)
                digest, record = self.sink._invocations[key]
                self.sink._invocations[key] = (
                    digest,
                    replace(record, invocation=loser_value),
                )
        return stale_result, current_result, rotation_first


def _handoff_loser_payload_overwrite_factory(mutation: str):
    def factory() -> _HandoffLoserPayloadOverwriteHarness:
        harness = _HandoffLoserPayloadOverwriteHarness()
        harness.broken_mutation = mutation
        return harness

    return factory


@pytest.mark.parametrize("mutation", ["checkpoint", "invocation"])
def test_reusable_contract_reads_the_writer_handoff_winner_payload(
    mutation: str,
) -> None:
    outcomes = run_fenced_run_sink_contract(_handoff_loser_payload_overwrite_factory(mutation))
    fence_rule = next(
        outcome for outcome in outcomes if outcome.rule_id == "FENCED-02-FENCE-PRECEDES-IDEMPOTENCY"
    )
    payload_observation = next(
        observation
        for observation in fence_rule.observations
        if observation.observation_id == f"handoff_lease_renewal_{mutation}_winner_payload"
    )

    assert fence_rule.status == "failed"
    assert payload_observation.actual != payload_observation.expected


class _CloseTrackingHarness:
    def __init__(
        self,
        tracker: dict[str, int],
        inner: DeterministicFencedRunHarness | None = None,
    ) -> None:
        self._tracker = tracker
        self._inner = inner or DeterministicFencedRunHarness()
        self._closed = False
        tracker["opened"] += 1
        tracker["active"] += 1
        tracker["max_active"] = max(tracker["max_active"], tracker["active"])

    @property
    def sink(self):
        return self._inner.sink

    def set_current_writer(self, writer_token: WriterToken) -> None:
        self._inner.set_current_writer(writer_token)

    def reopen(self):
        return _CloseTrackingHarness(self._tracker, self._inner.reopen())

    def read_event(self, run_id: str, seq: int) -> AgentEvent | None:
        return self._inner.read_event(run_id, seq)

    def read_terminal(self, run_id: str) -> TerminalOutcome | None:
        return self._inner.read_terminal(run_id)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._inner.close()
        self._tracker["closed"] += 1
        self._tracker["active"] -= 1

    def race_conflicting_writes(self, mutation, writer_token, left, right):
        return self._inner.race_conflicting_writes(
            mutation,
            writer_token,
            left,
            right,
        )

    def race_writer_handoff(
        self,
        mutation,
        stale_token,
        current_token,
        write,
    ):
        return self._inner.race_writer_handoff(
            mutation,
            stale_token,
            current_token,
            write,
        )


def test_reusable_contract_closes_every_exposed_harness_facade() -> None:
    tracker = {"opened": 0, "closed": 0, "active": 0, "max_active": 0}

    def factory() -> _CloseTrackingHarness:
        return _CloseTrackingHarness(tracker)

    outcomes = run_fenced_run_sink_contract(factory)

    assert all(outcome.status == "passed" for outcome in outcomes), outcomes
    assert tracker["opened"] > 1
    assert tracker["closed"] == tracker["opened"]
    assert tracker["active"] == 0
    assert tracker["max_active"] <= 4


class _CloseSensitiveSink(DeterministicFencedRunSink):
    _open_state: dict[str, bool]

    def _guard_blob_reader(self, record):
        reader = record._blob_reader
        if reader is None:
            return record
        open_state = self._open_state

        def read(sha256: str) -> bytes:
            if not open_state["open"]:
                raise RuntimeError("blob reader used after its facade closed")
            return reader(sha256)

        return replace(record, _blob_reader=read)

    def latest_checked(self, run_id: str):
        loaded = super().latest_checked(run_id)
        if loaded.value is None:
            return loaded
        return replace(loaded, value=self._guard_blob_reader(loaded.value))

    def load_invocation(self, run_id: str, logical_call_id: str):
        loaded = super().load_invocation(run_id, logical_call_id)
        if loaded.value is None:
            return loaded
        return replace(loaded, value=self._guard_blob_reader(loaded.value))


class _CloseSensitiveHarness(DeterministicFencedRunHarness):
    def __post_init__(self) -> None:
        self._open_state = {"open": True}
        self.sink = _CloseSensitiveSink(self._writers)
        self.sink._open_state = self._open_state

    def reopen(self):
        reopened = copy(self)
        reopened._open_state = {"open": True}
        reopened.sink = copy(self.sink)
        reopened.sink._open_state = reopened._open_state
        return reopened

    def close(self) -> None:
        self._open_state["open"] = False


def test_reusable_contract_materializes_blob_observations_before_close() -> None:
    outcomes = run_fenced_run_sink_contract(_CloseSensitiveHarness)

    assert all(outcome.status == "passed" for outcome in outcomes), outcomes


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
