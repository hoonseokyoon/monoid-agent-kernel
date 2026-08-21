from __future__ import annotations

from dataclasses import replace
from typing import get_type_hints

import pytest

from monoid_agent_kernel.core.checkpoint import RunCheckpoint
from monoid_agent_kernel.core.model_invocation import (
    MODEL_REQUEST_DIGEST_GENERATION,
    DurableModelInvocation,
)
from monoid_agent_kernel.hosting import (
    CommitResult,
    FencedCheckpointStore,
    FencedRunSink,
    ModelInvocationRecord,
    StorageCapabilities,
    WriterToken,
)
from support.fenced_hosting import DeterministicFencedRunHarness


def _invocation(
    run_id: str,
    *,
    revision: int,
    state: str,
    attempt: int = 1,
    dispatch_id: str = "dispatch-1",
    retryable: bool = False,
    succeeded: bool = False,
) -> DurableModelInvocation:
    receipt = None
    result_ref = ""
    failure_code = ""
    if state == "settled":
        receipt = {"request_digest": "a" * 64, "retryable": retryable}
        if succeeded:
            result_ref = "blob:turn"
        else:
            failure_code = "provider_refused"
    return DurableModelInvocation(
        run_id=run_id,
        logical_call_id="call-1",
        revision=revision,
        dispatch_id=dispatch_id,
        dispatch_attempt=attempt,
        idempotency_key="idempotency-key",
        dispatch_state=state,  # type: ignore[arg-type]
        request_digest="a" * 64,
        digest_generation=MODEL_REQUEST_DIGEST_GENERATION,
        receipt=receipt,
        result_ref=result_ref,
        failure_code=failure_code,
    )


@pytest.mark.parametrize("owner_id", ["", "owner id", "owner?secret", "x" * 257, 7])
def test_writer_token_rejects_nonportable_owner(owner_id: object) -> None:
    with pytest.raises(ValueError, match="owner_id"):
        WriterToken(run_id="run-1", owner_id=owner_id, generation=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("run_id", ["", "run id", "run?secret", "x" * 257, 7])
def test_writer_token_rejects_nonportable_run_id(run_id: object) -> None:
    with pytest.raises(ValueError, match="run_id"):
        WriterToken(run_id=run_id, owner_id="owner-a", generation=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "generation",
    [True, 0, -1, pytest.param(10**4300, id="oversized"), 1.5, "1"],
)
def test_writer_token_requires_positive_portable_generation(generation: object) -> None:
    with pytest.raises(ValueError, match="generation"):
        WriterToken(run_id="run-1", owner_id="owner-a", generation=generation)  # type: ignore[arg-type]


def test_storage_capabilities_are_closed_exact_booleans() -> None:
    defaults = StorageCapabilities()
    assert not any(getattr(defaults, name) for name in defaults.__dataclass_fields__)

    with pytest.raises(ValueError, match="lease_fencing"):
        StorageCapabilities(lease_fencing=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("status", ["", "ok", 1, True])
def test_commit_result_rejects_unknown_status(status: object) -> None:
    with pytest.raises(ValueError, match="status"):
        CommitResult(status=status)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "sequence",
    [-1, True, pytest.param(10**4300, id="oversized"), 1.5, "1"],
)
def test_commit_result_rejects_nonportable_sequence(sequence: object) -> None:
    with pytest.raises(ValueError, match="sequence"):
        CommitResult(status="committed", sequence=sequence, content_digest="a" * 64)  # type: ignore[arg-type]


def test_commit_result_accepts_optional_bounded_evidence() -> None:
    digest = "a" * 64
    winner = "b" * 64

    assert CommitResult(status="committed", content_digest=digest).status == "committed"
    assert (
        CommitResult(status="already_committed", content_digest=digest).status
        == "already_committed"
    )
    assert (
        CommitResult(
            status="conflict", content_digest=digest, winner_digest=winner
        ).winner_digest
        == winner
    )
    assert CommitResult(status="fenced") == CommitResult(status="fenced")
    assert CommitResult(status="committed").content_digest == ""
    assert CommitResult(status="conflict", content_digest=digest).winner_digest == ""

    with pytest.raises(ValueError, match="content_digest"):
        CommitResult(status="committed", content_digest="not-a-digest")
    with pytest.raises(ValueError, match="winner_digest"):
        CommitResult(status="conflict", winner_digest="not-a-digest")


def test_model_invocation_record_binds_revision_and_private_blobs() -> None:
    invocation = _invocation("run-record", revision=1, state="reserved")
    record = ModelInvocationRecord(
        revision=1,
        invocation=invocation,
        _blob_reader=lambda digest: {"a" * 64: b"private"}[digest],
    )

    assert record.blob("a" * 64) == b"private"
    with pytest.raises(ValueError, match="revision"):
        ModelInvocationRecord(revision=2, invocation=invocation)
    with pytest.raises(KeyError):
        ModelInvocationRecord(revision=1, invocation=invocation).blob("a" * 64)


def test_fake_has_the_composite_protocol_shape_without_legacy_mutations() -> None:
    sink: FencedRunSink = DeterministicFencedRunHarness().sink
    checkpoint_store: FencedCheckpointStore = sink

    assert checkpoint_store.capabilities.lease_fencing is True
    assert callable(checkpoint_store.commit_checkpoint)
    assert callable(sink.commit_invocation)
    assert callable(sink.append_event)
    assert callable(sink.settle_terminal)
    assert not hasattr(sink, "put")
    assert not hasattr(sink, "delete")


def test_public_protocol_annotations_resolve_at_runtime() -> None:
    checkpoint_hints = get_type_hints(FencedCheckpointStore.commit_checkpoint)
    invocation_hints = get_type_hints(FencedRunSink.commit_invocation)

    assert checkpoint_hints["writer_token"] is WriterToken
    assert checkpoint_hints["return"] is CommitResult
    assert invocation_hints["writer_token"] is WriterToken
    assert invocation_hints["return"] is CommitResult


def test_local_fs_capability_annotation_resolves_at_runtime() -> None:
    from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore

    hints = get_type_hints(LocalFsCheckpointStore.capabilities.fget)  # type: ignore[arg-type]

    assert hints["return"] is StorageCapabilities


def _commit_through_settled(
    harness: DeterministicFencedRunHarness,
    run_id: str,
    *,
    retryable: bool,
    succeeded: bool = False,
) -> WriterToken:
    token = harness.claim_writer(run_id, "owner-a")
    for invocation in (
        _invocation(run_id, revision=1, state="reserved"),
        _invocation(run_id, revision=2, state="dispatch_started"),
        _invocation(
            run_id,
            revision=3,
            state="settled",
            retryable=retryable,
            succeeded=succeeded,
        ),
    ):
        assert harness.sink.commit_invocation(invocation, {}, writer_token=token).status == "committed"
    return token


def test_invocation_same_revision_is_idempotent_or_conflicting() -> None:
    harness = DeterministicFencedRunHarness()
    run_id = "run-same-revision"
    token = harness.claim_writer(run_id, "owner-a")
    reserved = _invocation(run_id, revision=1, state="reserved")

    first = harness.sink.commit_invocation(reserved, {}, writer_token=token)
    repeated = harness.sink.commit_invocation(reserved, {}, writer_token=token)
    conflict = harness.sink.commit_invocation(
        replace(reserved, dispatch_id="dispatch-other"), {}, writer_token=token
    )

    assert first.status == "committed"
    assert repeated.status == "already_committed"
    assert conflict.status == "conflict"
    assert conflict.winner_digest == first.content_digest


@pytest.mark.parametrize(
    ("retryable", "succeeded"),
    [(False, False), (False, True)],
)
def test_invocation_refuses_new_attempt_without_proven_retryable_failure(
    retryable: bool,
    succeeded: bool,
) -> None:
    harness = DeterministicFencedRunHarness()
    run_id = f"run-no-retry-{retryable}-{succeeded}"
    token = _commit_through_settled(
        harness,
        run_id,
        retryable=retryable,
        succeeded=succeeded,
    )

    result = harness.sink.commit_invocation(
        _invocation(
            run_id,
            revision=4,
            attempt=2,
            dispatch_id="dispatch-2",
            state="reserved",
        ),
        {},
        writer_token=token,
    )

    assert result.status == "conflict"
    assert harness.sink.load_invocation(run_id, "call-1").sequence == 3


def test_invocation_unknown_is_final() -> None:
    harness = DeterministicFencedRunHarness()
    run_id = "run-unknown-final"
    token = harness.claim_writer(run_id, "owner-a")
    for invocation in (
        _invocation(run_id, revision=1, state="reserved"),
        _invocation(run_id, revision=2, state="dispatch_started"),
        _invocation(run_id, revision=3, state="unknown"),
    ):
        assert harness.sink.commit_invocation(invocation, {}, writer_token=token).status == "committed"

    retry = harness.sink.commit_invocation(
        _invocation(
            run_id,
            revision=4,
            attempt=2,
            dispatch_id="dispatch-2",
            state="reserved",
        ),
        {},
        writer_token=token,
    )

    assert retry.status == "conflict"


def test_new_attempt_keeps_logical_idempotency_identity() -> None:
    harness = DeterministicFencedRunHarness()
    run_id = "run-idempotency-drift"
    token = _commit_through_settled(harness, run_id, retryable=True)
    next_attempt = _invocation(
        run_id,
        revision=4,
        attempt=2,
        dispatch_id="dispatch-2",
        state="reserved",
    )

    drift = harness.sink.commit_invocation(
        replace(next_attempt, idempotency_key="different-key"),
        {},
        writer_token=token,
    )
    accepted = harness.sink.commit_invocation(next_attempt, {}, writer_token=token)

    assert drift.status == "conflict"
    assert accepted.status == "committed"


def test_stale_invocation_retry_is_fenced_before_content_comparison() -> None:
    harness = DeterministicFencedRunHarness()
    run_id = "run-stale-invocation"
    stale = harness.claim_writer(run_id, "owner-a")
    invocation = _invocation(run_id, revision=1, state="reserved")
    first = harness.sink.commit_invocation(invocation, {}, writer_token=stale)
    current = harness.claim_writer(run_id, "owner-b")

    fenced = harness.sink.commit_invocation(invocation, {}, writer_token=stale)
    current_repeat = harness.sink.commit_invocation(invocation, {}, writer_token=current)

    assert first.status == "committed"
    assert fenced == CommitResult(status="fenced")
    assert current_repeat.status == "already_committed"


def test_writer_token_cannot_cross_run_boundary() -> None:
    harness = DeterministicFencedRunHarness()
    run_a_token = WriterToken(run_id="run-a", owner_id="owner-a", generation=7)
    run_b_token = WriterToken(run_id="run-b", owner_id="owner-a", generation=7)
    harness.set_current_writer(run_a_token)
    harness.set_current_writer(run_b_token)

    assert run_a_token.owner_id == run_b_token.owner_id
    assert run_a_token.generation == run_b_token.generation
    assert run_a_token.run_id != run_b_token.run_id
    assert (
        harness.sink.commit_checkpoint(
            RunCheckpoint(run_id="run-b", seq=1),
            {},
            writer_token=run_a_token,
        ).status
        == "fenced"
    )
