"""Deterministic in-memory host and sink used by the fenced-storage contract tests."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.core.checkpoint import (
    CHECKPOINT_CODEC,
    CheckpointRecord,
    RunCheckpoint,
)
from monoid_agent_kernel.core.durable_codec import DurableLoadResult
from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.model_invocation import (
    MODEL_INVOCATION_CODEC,
    DurableModelInvocation,
)
from monoid_agent_kernel.core.outcome import TerminalOutcome
from monoid_agent_kernel.hosting import CommitResult, StorageCapabilities, WriterToken


_FENCED_CAPABILITIES = StorageCapabilities(
    concurrent_writers=True,
    compare_and_set=True,
    lease_fencing=True,
    durable_checkpoints=True,
    durable_events=True,
    durable_invocations=True,
    terminal_first_writer_wins=True,
)


def _blob_projection(blobs: Mapping[str, bytes]) -> dict[str, str]:
    return {
        key: hashlib.sha256(value).hexdigest()
        for key, value in sorted(blobs.items())
    }


def _record_digest(payload: dict[str, Any], blobs: Mapping[str, bytes] = {}) -> str:
    return canonical_sha256({"record": payload, "blobs": _blob_projection(blobs)})


@dataclass
class DeterministicFencedRunSink:
    """A lock-free fake: callers advance ownership explicitly, so every race is reproducible."""

    current_writers: dict[str, WriterToken]
    capabilities: StorageCapabilities = _FENCED_CAPABILITIES
    _checkpoints: dict[tuple[str, int], tuple[str, CheckpointRecord]] = field(default_factory=dict)
    _checkpoint_heads: dict[str, int] = field(default_factory=dict)
    _events: dict[tuple[str, int], tuple[str, AgentEvent]] = field(default_factory=dict)
    _invocations: dict[
        tuple[str, str, int], tuple[str, DurableModelInvocation]
    ] = field(default_factory=dict)
    _invocation_heads: dict[tuple[str, str], int] = field(default_factory=dict)
    _terminals: dict[str, tuple[str, TerminalOutcome]] = field(default_factory=dict)

    def _is_current(self, run_id: str, writer_token: WriterToken) -> bool:
        return writer_token.run_id == run_id and self.current_writers.get(run_id) == writer_token

    @staticmethod
    def _stored_result(
        records: dict[Any, tuple[str, Any]],
        key: Any,
        content_digest: str,
        *,
        sequence: int | None,
    ) -> CommitResult | None:
        stored = records.get(key)
        if stored is None:
            return None
        winner_digest, _ = stored
        if winner_digest == content_digest:
            return CommitResult(
                status="already_committed",
                sequence=sequence,
                content_digest=content_digest,
            )
        return CommitResult(
            status="conflict",
            sequence=sequence,
            content_digest=content_digest,
            winner_digest=winner_digest,
        )

    def commit_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        if not self._is_current(checkpoint.run_id, writer_token):
            return CommitResult(status="fenced")
        digest = _record_digest(checkpoint.to_json(), blobs)
        key = (checkpoint.run_id, checkpoint.seq)
        existing = self._stored_result(
            self._checkpoints,
            key,
            digest,
            sequence=checkpoint.seq,
        )
        if existing is not None:
            return existing
        previous_head = self._checkpoint_heads.get(checkpoint.run_id, -1)
        if checkpoint.seq < previous_head:
            winner_digest = self._checkpoints[(checkpoint.run_id, previous_head)][0]
            return CommitResult(
                status="conflict",
                sequence=checkpoint.seq,
                content_digest=digest,
                winner_digest=winner_digest,
            )
        detached_blobs = dict(blobs)
        record = CheckpointRecord(
            seq=checkpoint.seq,
            checkpoint=checkpoint,
            _blob_reader=lambda sha256: detached_blobs[sha256],
        )
        self._checkpoints[key] = (digest, record)
        self._checkpoint_heads[checkpoint.run_id] = max(previous_head, checkpoint.seq)
        return CommitResult(
            status="committed",
            sequence=checkpoint.seq,
            content_digest=digest,
        )

    def latest_checked(self, run_id: str) -> DurableLoadResult[CheckpointRecord]:
        head = self._checkpoint_heads.get(run_id)
        if head is None:
            return CHECKPOINT_CODEC.missing().map(
                lambda checkpoint: CheckpointRecord(seq=checkpoint.seq, checkpoint=checkpoint)
            )
        record = self._checkpoints[(run_id, head)][1]
        return DurableLoadResult(
            status="loaded",
            family=CHECKPOINT_CODEC.family,
            current_schema=CHECKPOINT_CODEC.current_schema,
            value=record,
            observed_schema=record.checkpoint.schema_version,
            sequence=record.seq,
        )

    def append_event(
        self,
        event: AgentEvent,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        if not self._is_current(event.run_id, writer_token):
            return CommitResult(status="fenced")
        digest = _record_digest(event.to_json())
        key = (event.run_id, event.seq)
        existing = self._stored_result(self._events, key, digest, sequence=event.seq)
        if existing is not None:
            return existing
        self._events[key] = (digest, event)
        return CommitResult(status="committed", sequence=event.seq, content_digest=digest)

    def settle_terminal(
        self,
        outcome: TerminalOutcome,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        if not self._is_current(outcome.run_id, writer_token):
            return CommitResult(status="fenced")
        digest = _record_digest(outcome.to_json())
        existing = self._stored_result(self._terminals, outcome.run_id, digest, sequence=None)
        if existing is not None:
            return existing
        self._terminals[outcome.run_id] = (digest, outcome)
        return CommitResult(status="committed", content_digest=digest)

    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        if not self._is_current(invocation.run_id, writer_token):
            return CommitResult(status="fenced")
        digest = _record_digest(invocation.to_json(), blobs)
        key = (invocation.run_id, invocation.logical_call_id, invocation.revision)
        existing = self._stored_result(
            self._invocations,
            key,
            digest,
            sequence=invocation.revision,
        )
        if existing is not None:
            return existing
        transition_winner = self._invocation_transition_winner(invocation)
        if transition_winner is not None:
            return CommitResult(
                status="conflict",
                sequence=invocation.revision,
                content_digest=digest,
                winner_digest=transition_winner,
            )
        self._invocations[key] = (digest, invocation)
        self._invocation_heads[(invocation.run_id, invocation.logical_call_id)] = invocation.revision
        return CommitResult(
            status="committed",
            sequence=invocation.revision,
            content_digest=digest,
        )

    def _invocation_transition_winner(
        self,
        invocation: DurableModelInvocation,
    ) -> str | None:
        call_key = (invocation.run_id, invocation.logical_call_id)
        previous_revision = self._invocation_heads.get(call_key)
        if previous_revision is None:
            if (
                invocation.revision == 1
                and invocation.dispatch_attempt == 1
                and invocation.dispatch_state == "reserved"
            ):
                return None
            return ""
        previous_digest, previous = self._invocations[
            (invocation.run_id, invocation.logical_call_id, previous_revision)
        ]
        if invocation.revision != previous.revision + 1:
            return previous_digest
        if any(
            getattr(invocation, field_name) != getattr(previous, field_name)
            for field_name in (
                "logical_call_id",
                "idempotency_key",
                "request_digest",
                "digest_generation",
            )
        ):
            return previous_digest
        if previous.dispatch_state in {"settled", "unknown"}:
            retryable_failure = (
                previous.dispatch_state == "settled"
                and bool(previous.failure_code)
                and previous.receipt is not None
                and previous.receipt.get("retryable") is True
            )
            if not retryable_failure:
                return previous_digest
            if not (
                invocation.dispatch_state == "reserved"
                and invocation.dispatch_attempt == previous.dispatch_attempt + 1
                and invocation.dispatch_id != previous.dispatch_id
            ):
                return previous_digest
            return None
        if invocation.dispatch_attempt != previous.dispatch_attempt:
            return previous_digest
        if invocation.dispatch_id != previous.dispatch_id:
            return previous_digest
        allowed_next = {
            "reserved": frozenset({"dispatch_started"}),
            "dispatch_started": frozenset({"settled", "unknown"}),
        }
        if invocation.dispatch_state not in allowed_next[previous.dispatch_state]:
            return previous_digest
        return None

    def load_invocation(
        self,
        run_id: str,
        logical_call_id: str,
    ) -> DurableLoadResult[DurableModelInvocation]:
        head = self._invocation_heads.get((run_id, logical_call_id))
        if head is None:
            return MODEL_INVOCATION_CODEC.missing()
        invocation = self._invocations[(run_id, logical_call_id, head)][1]
        return DurableLoadResult(
            status="loaded",
            family=MODEL_INVOCATION_CODEC.family,
            current_schema=MODEL_INVOCATION_CODEC.current_schema,
            value=invocation,
            observed_schema=invocation.schema_version,
            sequence=invocation.revision,
        )


@dataclass
class DeterministicFencedRunHarness:
    """Host-side lease seam used by the reusable conformance contract."""

    _generations: dict[str, int] = field(default_factory=dict)
    _writers: dict[str, WriterToken] = field(default_factory=dict)
    sink: DeterministicFencedRunSink = field(init=False)

    def __post_init__(self) -> None:
        self.sink = DeterministicFencedRunSink(self._writers)

    def claim_writer(self, run_id: str, owner_id: str) -> WriterToken:
        generation = self._generations.get(run_id, 0) + 1
        self._generations[run_id] = generation
        token = WriterToken(run_id=run_id, owner_id=owner_id, generation=generation)
        self._writers[run_id] = token
        return token
