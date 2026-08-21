"""Dependency-light contracts for hosts that own durable run fencing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Literal, Protocol, get_args

from monoid_agent_kernel.core.checkpoint import CheckpointRecord, RunCheckpoint
from monoid_agent_kernel.core.durable_codec import DurableLoadResult
from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.json_ingress import is_portable_json_integer
from monoid_agent_kernel.core.model_invocation import DurableModelInvocation
from monoid_agent_kernel.core.model_io import is_recorded_digest
from monoid_agent_kernel.core.outcome import TerminalOutcome
from monoid_agent_kernel.core.safe_evidence import is_safe_opaque_id


_CommitStatus = Literal["committed", "already_committed", "conflict", "fenced"]


@dataclass(frozen=True, kw_only=True)
class WriterToken:
    """Host-issued writer identity and monotonic lease generation.

    This value carries no credential material. The adapter compares it with authoritative lease
    state inside each mutation.
    """

    owner_id: str
    generation: int

    def __post_init__(self) -> None:
        if not is_safe_opaque_id(self.owner_id):
            raise ValueError("writer token owner_id must be a bounded opaque id")
        if not is_portable_json_integer(self.generation) or self.generation < 1:
            raise ValueError("writer token generation must be a positive portable integer")


@dataclass(frozen=True, kw_only=True)
class CommitResult:
    """Portable outcome of one fenced, content-addressed mutation."""

    status: _CommitStatus
    sequence: int | None = None
    content_digest: str = ""
    winner_digest: str = ""

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in get_args(_CommitStatus):
            raise ValueError("commit result status is outside the portable vocabulary")
        if self.sequence is not None and (
            not is_portable_json_integer(self.sequence) or self.sequence < 0
        ):
            raise ValueError("commit result sequence must be a non-negative portable integer")
        for field_name in ("content_digest", "winner_digest"):
            value = getattr(self, field_name)
            if type(value) is not str or (value and not is_recorded_digest(value)):
                raise ValueError(
                    f"commit result {field_name} must be empty or a lowercase SHA-256 digest"
                )


@dataclass(frozen=True, kw_only=True)
class StorageCapabilities:
    """Fail-closed declaration of the guarantees a storage adapter actually provides."""

    single_writer: bool = False
    concurrent_writers: bool = False
    compare_and_set: bool = False
    lease_fencing: bool = False
    durable_checkpoints: bool = False
    durable_events: bool = False
    durable_invocations: bool = False
    terminal_first_writer_wins: bool = False
    transactional_outbox: bool = False
    cross_process_notify: bool = False

    def __post_init__(self) -> None:
        for declared_field in fields(self):
            if type(getattr(self, declared_field.name)) is not bool:
                raise ValueError(f"storage capability {declared_field.name} must be a boolean")


class FencedCheckpointStore(Protocol):
    """Shared checkpoint store whose mutation rejects stale host writers."""

    @property
    def capabilities(self) -> StorageCapabilities: ...

    def commit_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult: ...

    def latest_checked(self, run_id: str) -> DurableLoadResult[CheckpointRecord]: ...


class FencedRunSink(FencedCheckpointStore, Protocol):
    """Composite authoritative journal for checkpoints, invocations, events, and terminal state."""

    def load_invocation(
        self,
        run_id: str,
        logical_call_id: str,
    ) -> DurableLoadResult[DurableModelInvocation]: ...

    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult: ...

    def append_event(
        self,
        event: AgentEvent,
        *,
        writer_token: WriterToken,
    ) -> CommitResult: ...

    def settle_terminal(
        self,
        outcome: TerminalOutcome,
        *,
        writer_token: WriterToken,
    ) -> CommitResult: ...


__all__ = [
    "WriterToken",
    "CommitResult",
    "StorageCapabilities",
    "FencedCheckpointStore",
    "FencedRunSink",
]
