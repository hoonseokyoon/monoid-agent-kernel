"""Dependency-light contracts for hosts that own durable run fencing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol, get_args

from monoid_agent_kernel.core.checkpoint import CheckpointRecord, RunCheckpoint
from monoid_agent_kernel.core.durable_codec import DurableLoadResult
from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.json_ingress import is_portable_json_integer
from monoid_agent_kernel.core.model_invocation import DurableModelInvocation
from monoid_agent_kernel.core.model_io import is_recorded_digest
from monoid_agent_kernel.core.outcome import TerminalOutcome
from monoid_agent_kernel.core.safe_evidence import is_safe_opaque_id
from monoid_agent_kernel.core._storage_capabilities import StorageCapabilities


_CommitStatus = Literal["committed", "already_committed", "conflict", "fenced"]


@dataclass(frozen=True, kw_only=True)
class WriterToken:
    """Host-issued writer identity and monotonic lease generation.

    This value carries no credential material. The adapter compares it with authoritative lease
    state inside each mutation.
    """

    run_id: str
    owner_id: str
    generation: int

    def __post_init__(self) -> None:
        if not is_safe_opaque_id(self.run_id):
            raise ValueError("writer token run_id must be a bounded opaque id")
        if not is_safe_opaque_id(self.owner_id):
            raise ValueError("writer token owner_id must be a bounded opaque id")
        if not is_portable_json_integer(self.generation) or self.generation < 1:
            raise ValueError("writer token generation must be a positive portable integer")


@dataclass(frozen=True, kw_only=True)
class CommitResult:
    """Portable outcome of one fenced, content-addressed mutation.

    Evidence fields are optional. When present, ``sequence`` is the submitted resource
    coordinate. ``content_digest`` is the canonical SHA-256 of
    ``{"record": <canonical payload>, "blobs": {<key>: sha256(<bytes>)}}``.
    ``winner_digest`` uses the same encoding for the previously committed winner and stays empty
    when no competing winner exists.
    """

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


@dataclass(frozen=True)
class ModelInvocationRecord:
    """A committed invocation revision with lazy access to its private result blobs."""

    revision: int
    invocation: DurableModelInvocation
    _blob_reader: Callable[[str], bytes] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.revision != self.invocation.revision:
            raise ValueError("model invocation record revision must match its invocation")

    def blob(self, sha256: str) -> bytes:
        """Read a private result blob by its content digest."""

        if self._blob_reader is None:
            raise KeyError(sha256)
        return self._blob_reader(sha256)


class FencedCheckpointStore(Protocol):
    """Shared checkpoint store whose mutation rejects stale host writers.

    Every mutation validates the token's run binding before owner, generation, idempotency, or
    content. Owner and generation are independent equality checks against current authority. A
    token issued for another run, a stale generation from the current owner, or a wrong owner at
    the current generation always returns ``fenced``. Every submitted blob key is the lowercase
    SHA-256 digest of its bytes. A malformed blob map returns ``conflict`` without publishing
    metadata or blob content. Every checkpoint blob reference must resolve from the submitted map
    or same-run authoritative backing before metadata and head publication. This includes
    workspace ``content_sha256`` entries, media ``blob:`` references carried by checkpoint
    messages and queued messages, and a ``blob:`` result referenced by
    ``last_model_invocation``. A loaded record exposes bytes reused from authoritative backing
    through ``blob()``.
    """

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
    """Composite authoritative journal protected by the inherited run-bound fence.

    Invocation blob maps follow the inherited content-addressed blob rule. Every ``blob:`` result
    reference must resolve from the submitted map or same-run authoritative backing before the
    invocation revision becomes authoritative. A loaded record exposes reused bytes through
    ``blob()``.
    """

    def load_invocation(
        self,
        run_id: str,
        logical_call_id: str,
    ) -> DurableLoadResult[ModelInvocationRecord]: ...

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
    "ModelInvocationRecord",
    "StorageCapabilities",
    "FencedCheckpointStore",
    "FencedRunSink",
]
