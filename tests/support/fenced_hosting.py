"""Deterministic in-memory host and sink used by the fenced-storage contract tests."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import dataclass, field
from threading import Barrier, RLock
from typing import Any, Callable, Literal

from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.core.checkpoint import (
    CHECKPOINT_CODEC,
    CheckpointRecord,
    RunCheckpoint,
    checkpoint_blob_references,
    checkpoint_payload_for_write,
)
from monoid_agent_kernel.core.durable_codec import DurableLoadResult
from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.model_invocation import (
    MODEL_INVOCATION_CODEC,
    DurableModelInvocation,
)
from monoid_agent_kernel.core.outcome import TerminalOutcome
from monoid_agent_kernel.hosting import (
    CommitResult,
    ModelInvocationRecord,
    StorageCapabilities,
    WriterToken,
)


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
    return {key: hashlib.sha256(value).hexdigest() for key, value in sorted(blobs.items())}


def _record_digest(payload: dict[str, Any], blobs: Mapping[str, bytes] = {}) -> str:
    return canonical_sha256({"record": payload, "blobs": _blob_projection(blobs)})


@dataclass
class DeterministicFencedRunSink:
    """A shared-backing fake with explicit ownership and atomic mutation critical sections."""

    current_writers: dict[str, WriterToken]
    capabilities: StorageCapabilities = _FENCED_CAPABILITIES
    _checkpoints: dict[tuple[str, int], tuple[str, CheckpointRecord]] = field(default_factory=dict)
    _checkpoint_heads: dict[str, int] = field(default_factory=dict)
    _events: dict[tuple[str, int], tuple[str, AgentEvent]] = field(default_factory=dict)
    _invocations: dict[tuple[str, str, int], tuple[str, ModelInvocationRecord]] = field(
        default_factory=dict
    )
    _invocation_heads: dict[tuple[str, str], int] = field(default_factory=dict)
    _model_evidence: dict[
        tuple[str, str, int], tuple[str, DurableModelInvocation]
    ] = field(default_factory=dict)
    _model_evidence_outbox: dict[
        tuple[str, str, int], tuple[str, DurableModelInvocation]
    ] = field(default_factory=dict)
    _terminals: dict[str, tuple[str, TerminalOutcome]] = field(default_factory=dict)
    _blobs: dict[tuple[str, str], bytes] = field(default_factory=dict)
    _checkpoint_load_faults: dict[
        str,
        Literal["corrupt", "unsupported_version"],
    ] = field(default_factory=dict)
    _invocation_load_faults: dict[
        tuple[str, str],
        Literal["corrupt", "unsupported_version"],
    ] = field(default_factory=dict)
    _lock: Any = field(default_factory=RLock, repr=False)

    def _is_current(self, run_id: str, writer_token: WriterToken) -> bool:
        return writer_token.run_id == run_id and self.current_writers.get(run_id) == writer_token

    def _blobs_are_content_addressed(self, blobs: Mapping[str, bytes]) -> bool:
        return all(
            type(value) is bytes and hashlib.sha256(value).hexdigest() == key
            for key, value in blobs.items()
        )

    def _checkpoint_blob_references(self, checkpoint: RunCheckpoint) -> set[str]:
        return checkpoint_blob_references(checkpoint)

    def _reference_is_available(
        self,
        run_id: str,
        sha256: str,
        blobs: Mapping[str, bytes],
    ) -> bool:
        return sha256 in blobs or (run_id, sha256) in self._blobs

    def _checkpoint_references_resolve(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
    ) -> bool:
        return all(
            self._reference_is_available(checkpoint.run_id, sha256, blobs)
            for sha256 in self._checkpoint_blob_references(checkpoint)
        )

    def _invocation_references_resolve(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
    ) -> bool:
        if not invocation.result_ref.startswith("blob:"):
            return True
        return self._reference_is_available(
            invocation.run_id,
            invocation.result_ref.removeprefix("blob:"),
            blobs,
        )

    def _blobs_preserve_authoritative_backing(
        self,
        run_id: str,
        blobs: Mapping[str, bytes],
    ) -> bool:
        return all(
            self._blobs.get((run_id, sha256), value) == value for sha256, value in blobs.items()
        )

    def _publish_blobs(self, run_id: str, blobs: Mapping[str, bytes]) -> None:
        for sha256, value in blobs.items():
            self._blobs.setdefault((run_id, sha256), value)

    def _blob_reader(
        self,
        run_id: str,
        blobs: Mapping[str, bytes],
    ) -> Callable[[str], bytes]:
        detached_blobs = dict(blobs)

        def read(sha256: str) -> bytes:
            if sha256 in detached_blobs:
                return detached_blobs[sha256]
            return self._blobs[(run_id, sha256)]

        return read

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
        with self._lock:
            return self._commit_checkpoint(checkpoint, blobs, writer_token=writer_token)

    def _commit_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        if not self._is_current(checkpoint.run_id, writer_token):
            return CommitResult(status="fenced")
        if not self._blobs_are_content_addressed(blobs):
            return CommitResult(status="conflict", sequence=checkpoint.seq)
        if not self._blobs_preserve_authoritative_backing(checkpoint.run_id, blobs):
            return CommitResult(status="conflict", sequence=checkpoint.seq)
        if not self._checkpoint_references_resolve(checkpoint, blobs):
            return CommitResult(status="conflict", sequence=checkpoint.seq)
        digest = _record_digest(checkpoint_payload_for_write(checkpoint), blobs)
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
        self._publish_blobs(checkpoint.run_id, blobs)
        record = CheckpointRecord(
            seq=checkpoint.seq,
            checkpoint=checkpoint,
            _blob_reader=self._blob_reader(checkpoint.run_id, blobs),
        )
        self._checkpoints[key] = (digest, record)
        self._checkpoint_heads[checkpoint.run_id] = max(previous_head, checkpoint.seq)
        return CommitResult(
            status="committed",
            sequence=checkpoint.seq,
            content_digest=digest,
        )

    def latest_checked(self, run_id: str) -> DurableLoadResult[CheckpointRecord]:
        with self._lock:
            return self._latest_checked(run_id)

    def _latest_checked(self, run_id: str) -> DurableLoadResult[CheckpointRecord]:
        head = self._checkpoint_heads.get(run_id)
        if head is None:
            return CHECKPOINT_CODEC.missing().map(
                lambda checkpoint: CheckpointRecord(seq=checkpoint.seq, checkpoint=checkpoint)
            )
        fault = self._checkpoint_load_faults.get(run_id)
        if fault == "corrupt":
            return CHECKPOINT_CODEC.corrupt(
                "injected authoritative checkpoint corruption",
                observed_schema=CHECKPOINT_CODEC.current_schema,
                sequence=head,
            ).map(lambda checkpoint: CheckpointRecord(seq=checkpoint.seq, checkpoint=checkpoint))
        if fault == "unsupported_version":
            return CHECKPOINT_CODEC.unsupported(
                f"monoid.{CHECKPOINT_CODEC.family}.v{CHECKPOINT_CODEC.current_version + 1}",
                sequence=head,
            ).map(lambda checkpoint: CheckpointRecord(seq=checkpoint.seq, checkpoint=checkpoint))
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
        with self._lock:
            return self._append_event(event, writer_token=writer_token)

    def _append_event(
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
        with self._lock:
            return self._settle_terminal(outcome, writer_token=writer_token)

    def _settle_terminal(
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
        stage_evidence: bool = False,
    ) -> CommitResult:
        with self._lock:
            if type(stage_evidence) is not bool:
                return CommitResult(status="conflict", sequence=invocation.revision)
            result = self._commit_invocation(invocation, blobs, writer_token=writer_token)
            if not stage_evidence or result.status not in {"committed", "already_committed"}:
                return result
            evidence = self._commit_model_evidence(
                invocation,
                writer_token=writer_token,
                outbox=True,
            )
            if evidence.status not in {"committed", "already_committed"}:
                raise AssertionError("atomic evidence staging rejected a committed invocation")
            return result

    def commit_model_evidence(
        self,
        invocation: DurableModelInvocation,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        with self._lock:
            return self._commit_model_evidence(
                invocation,
                writer_token=writer_token,
                outbox=False,
            )

    def _commit_model_evidence(
        self,
        invocation: DurableModelInvocation,
        *,
        writer_token: WriterToken,
        outbox: bool,
    ) -> CommitResult:
        if not self._is_current(invocation.run_id, writer_token):
            return CommitResult(status="fenced", sequence=invocation.revision)
        if invocation.dispatch_state != "settled" or invocation.receipt is None:
            return CommitResult(status="conflict", sequence=invocation.revision)
        head = self._invocation_heads.get((invocation.run_id, invocation.logical_call_id))
        if head != invocation.revision:
            return CommitResult(status="conflict", sequence=invocation.revision)
        authoritative = self._invocations.get(
            (invocation.run_id, invocation.logical_call_id, invocation.revision)
        )
        if authoritative is None or authoritative[1].invocation != invocation:
            return CommitResult(status="conflict", sequence=invocation.revision)
        records = self._model_evidence_outbox if outbox else self._model_evidence
        digest = _record_digest(invocation.to_json())
        key = (invocation.run_id, invocation.logical_call_id, invocation.revision)
        existing = self._stored_result(
            records,
            key,
            digest,
            sequence=invocation.revision,
        )
        if existing is not None:
            return existing
        records[key] = (digest, invocation)
        return CommitResult(
            status="committed",
            sequence=invocation.revision,
            content_digest=digest,
        )

    def _commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        if not self._is_current(invocation.run_id, writer_token):
            return CommitResult(status="fenced")
        if not self._blobs_are_content_addressed(blobs):
            return CommitResult(status="conflict", sequence=invocation.revision)
        if not self._blobs_preserve_authoritative_backing(invocation.run_id, blobs):
            return CommitResult(status="conflict", sequence=invocation.revision)
        if not self._invocation_references_resolve(invocation, blobs):
            return CommitResult(status="conflict", sequence=invocation.revision)
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
        self._publish_blobs(invocation.run_id, blobs)
        record = ModelInvocationRecord(
            revision=invocation.revision,
            invocation=invocation,
            _blob_reader=self._blob_reader(invocation.run_id, blobs),
        )
        self._invocations[key] = (digest, record)
        self._invocation_heads[(invocation.run_id, invocation.logical_call_id)] = (
            invocation.revision
        )
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
        previous_digest, previous_record = self._invocations[
            (invocation.run_id, invocation.logical_call_id, previous_revision)
        ]
        previous = previous_record.invocation
        if invocation.revision != previous.revision + 1:
            return previous_digest
        invocation_payload = invocation.to_json()
        previous_payload = previous.to_json()
        if any(
            invocation_payload[field_name] != previous_payload[field_name]
            for field_name in (
                "logical_call_id",
                "idempotency_key",
                "request_digest",
                "digest_generation",
                "requires_evidence",
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
                and not self._dispatch_id_was_used(invocation)
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

    def _dispatch_id_was_used(self, invocation: DurableModelInvocation) -> bool:
        return any(
            record.invocation.dispatch_id == invocation.dispatch_id
            for (run_id, logical_call_id, _), (_, record) in self._invocations.items()
            if run_id == invocation.run_id and logical_call_id == invocation.logical_call_id
        )

    def load_invocation(
        self,
        run_id: str,
        logical_call_id: str,
    ) -> DurableLoadResult[ModelInvocationRecord]:
        with self._lock:
            return self._load_invocation(run_id, logical_call_id)

    def _load_invocation(
        self,
        run_id: str,
        logical_call_id: str,
    ) -> DurableLoadResult[ModelInvocationRecord]:
        head = self._invocation_heads.get((run_id, logical_call_id))
        if head is None:
            return MODEL_INVOCATION_CODEC.missing()
        fault = self._invocation_load_faults.get((run_id, logical_call_id))
        if fault == "corrupt":
            return MODEL_INVOCATION_CODEC.corrupt(
                "injected authoritative model invocation corruption",
                observed_schema=MODEL_INVOCATION_CODEC.current_schema,
                sequence=head,
            )
        if fault == "unsupported_version":
            return MODEL_INVOCATION_CODEC.unsupported(
                (
                    f"monoid.{MODEL_INVOCATION_CODEC.family}.v"
                    f"{MODEL_INVOCATION_CODEC.current_version + 1}"
                ),
                sequence=head,
            )
        record = self._invocations[(run_id, logical_call_id, head)][1]
        return DurableLoadResult(
            status="loaded",
            family=MODEL_INVOCATION_CODEC.family,
            current_schema=MODEL_INVOCATION_CODEC.current_schema,
            value=record,
            observed_schema=record.invocation.schema_version,
            sequence=record.revision,
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
        token = WriterToken(run_id=run_id, owner_id=owner_id, generation=generation)
        self.set_current_writer(token)
        return token

    def set_current_writer(self, writer_token: WriterToken) -> None:
        """Install an exact authoritative token for deterministic contract setup."""

        with self.sink._lock:
            current = self._writers.get(writer_token.run_id)
            if current is not None and writer_token.generation <= current.generation:
                raise ValueError("writer generation must advance within a run")
            self._generations[writer_token.run_id] = writer_token.generation
            self._writers[writer_token.run_id] = writer_token

    def reopen(self) -> DeterministicFencedRunHarness:
        """Return fresh host and sink facades sharing the same simulated durable backing."""

        reopened = copy(self)
        reopened.sink = copy(self.sink)
        return reopened

    def inject_authoritative_load_fault(
        self,
        record_family: Literal["checkpoint", "invocation"],
        run_id: str,
        status: Literal["corrupt", "unsupported_version"],
        *,
        logical_call_id: str = "",
    ) -> None:
        """Install a persistent decoder fault at an existing authoritative head."""

        with self.sink._lock:
            if status not in {"corrupt", "unsupported_version"}:
                raise ValueError("load fault status is outside the conformance vocabulary")
            if record_family == "checkpoint":
                if run_id not in self.sink._checkpoint_heads:
                    raise ValueError("checkpoint load fault requires an authoritative head")
                self.sink._checkpoint_load_faults[run_id] = status
                return
            if record_family != "invocation":
                raise ValueError("load fault record family is outside the conformance vocabulary")
            if not logical_call_id:
                raise ValueError("invocation load fault requires logical_call_id")
            key = (run_id, logical_call_id)
            if key not in self.sink._invocation_heads:
                raise ValueError("invocation load fault requires an authoritative head")
            self.sink._invocation_load_faults[key] = status

    def read_event(self, run_id: str, seq: int) -> AgentEvent | None:
        """Read a complete event through the shared durable backing."""

        with self.sink._lock:
            stored = self.sink._events.get((run_id, seq))
            return stored[1] if stored is not None else None

    def read_terminal(self, run_id: str) -> TerminalOutcome | None:
        """Read the complete terminal winner through the shared durable backing."""

        with self.sink._lock:
            stored = self.sink._terminals.get(run_id)
            return stored[1] if stored is not None else None

    def close(self) -> None:
        """Release this in-memory facade; external harnesses close real client resources here."""

    def race_conflicting_writes(
        self,
        mutation: str,
        writer_token: WriterToken,
        left: Callable[[DeterministicFencedRunSink], CommitResult],
        right: Callable[[DeterministicFencedRunSink], CommitResult],
    ) -> tuple[CommitResult, CommitResult]:
        """Run the backend-specific CAS race hook through separate sink facades."""

        del mutation, writer_token
        barrier = Barrier(3)
        left_facade = self.reopen()
        right_facade = self.reopen()

        def invoke(
            operation: Callable[[DeterministicFencedRunSink], CommitResult],
            facade: DeterministicFencedRunHarness,
        ) -> CommitResult:
            barrier.wait(timeout=10)
            return operation(facade.sink)

        try:
            with ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="fenced-contract",
            ) as executor:
                left_future = executor.submit(invoke, left, left_facade)
                right_future = executor.submit(invoke, right, right_facade)
                barrier.wait(timeout=10)
                return left_future.result(timeout=10), right_future.result(timeout=10)
        finally:
            right_facade.close()
            left_facade.close()

    def race_writer_handoff(
        self,
        mutation: str,
        stale_token: WriterToken,
        current_token: WriterToken,
        write: Callable[[DeterministicFencedRunSink, WriterToken], CommitResult],
    ) -> tuple[CommitResult, CommitResult, bool]:
        """Race one stale write against rotation through the fake's shared atomic lock."""

        del mutation
        barrier = Barrier(3)
        linearization: list[str] = []
        stale_facade = self.reopen()
        current_facade: DeterministicFencedRunHarness | None = None

        def stale_write() -> CommitResult:
            barrier.wait(timeout=10)
            with self.sink._lock:
                result = write(stale_facade.sink, stale_token)
                linearization.append("write")
                return result

        def rotate() -> None:
            barrier.wait(timeout=10)
            with self.sink._lock:
                self.set_current_writer(current_token)
                linearization.append("rotation")

        try:
            with ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="fenced-handoff",
            ) as executor:
                stale_future = executor.submit(stale_write)
                rotation_future = executor.submit(rotate)
                barrier.wait(timeout=10)
                stale_result = stale_future.result(timeout=10)
                rotation_future.result(timeout=10)

            rotation_first = linearization[0] == "rotation"
            current_facade = self.reopen()
            current_result = write(current_facade.sink, current_token)
            return stale_result, current_result, rotation_first
        finally:
            if current_facade is not None:
                current_facade.close()
            stale_facade.close()
