"""Provider-neutral fenced durable stream contracts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, get_args

from monoid_agent_kernel.core.json_ingress import is_portable_json_integer
from monoid_agent_kernel.core.safe_evidence import is_safe_opaque_id, is_safe_taxonomy_code
from monoid_agent_kernel.hosting.blobs import is_content_sha256
from monoid_agent_kernel.hosting.contracts import WriterToken


MAX_STREAM_CHUNK_BYTES = 4 * 1024 * 1024
MAX_STREAM_READ_CHUNKS = 100

StreamState: TypeAlias = Literal["open", "sealed"]
StreamOpenStatus: TypeAlias = Literal[
    "opened",
    "already_open",
    "sealed",
    "fenced",
    "conflict",
    "run_terminal",
]
StreamResetStatus: TypeAlias = Literal[
    "reset",
    "already_reset",
    "fenced",
    "conflict",
    "old_generation",
    "run_terminal",
]
StreamAppendStatus: TypeAlias = Literal[
    "committed",
    "already_committed",
    "fenced",
    "conflict",
    "old_generation",
    "gap",
    "sealed",
    "run_terminal",
]
StreamSealStatus: TypeAlias = Literal[
    "sealed",
    "already_sealed",
    "fenced",
    "conflict",
    "old_generation",
    "gap",
    "run_terminal",
]
StreamReadStatus: TypeAlias = Literal["ok", "reset", "gap", "not_found"]


def _portable_non_negative(value: object, field_name: str) -> None:
    if not is_portable_json_integer(value) or value < 0:  # type: ignore[operator]
        raise ValueError(f"{field_name} must be a non-negative portable integer")


def _portable_positive(value: object, field_name: str) -> None:
    if not is_portable_json_integer(value) or value < 1:  # type: ignore[operator]
        raise ValueError(f"{field_name} must be a positive portable integer")


@dataclass(frozen=True, kw_only=True)
class DurableStreamIdentity:
    """Stable private stream lane and its logical model-call lineage."""

    run_id: str
    stream_id: str
    logical_call_id: str
    channel: str

    def __post_init__(self) -> None:
        for field_name in ("run_id", "stream_id", "logical_call_id"):
            if not is_safe_opaque_id(getattr(self, field_name)):
                raise ValueError(f"durable stream {field_name} must be a bounded opaque id")
        if not is_safe_taxonomy_code(self.channel):
            raise ValueError("durable stream channel must be a public-safe taxonomy code")


@dataclass(frozen=True, kw_only=True)
class DurableStreamHead:
    """Canonical current generation, cursor, and seal for one stream lane."""

    identity: DurableStreamIdentity
    generation: int
    cursor_bytes: int
    next_chunk_sequence: int
    state: StreamState
    final_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.identity, DurableStreamIdentity):
            raise TypeError("durable stream head identity must be DurableStreamIdentity")
        _portable_positive(self.generation, "durable stream generation")
        _portable_non_negative(self.cursor_bytes, "durable stream cursor_bytes")
        _portable_positive(self.next_chunk_sequence, "durable stream next_chunk_sequence")
        if type(self.state) is not str or self.state not in get_args(StreamState):
            raise ValueError("durable stream state is outside the portable vocabulary")
        if type(self.final_sha256) is not str or (
            self.final_sha256 and not is_content_sha256(self.final_sha256)
        ):
            raise ValueError("durable stream final_sha256 must be empty or lowercase SHA-256")
        if (self.state == "sealed") != bool(self.final_sha256):
            raise ValueError("only a sealed durable stream carries final_sha256")


@dataclass(frozen=True, kw_only=True)
class DurableStreamChunk:
    """Immutable metadata for one contiguous ObjectStore-backed UTF-8 batch."""

    identity: DurableStreamIdentity
    generation: int
    sequence: int
    start_offset: int
    end_offset: int
    sha256: str
    locator: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, DurableStreamIdentity):
            raise TypeError("durable stream chunk identity must be DurableStreamIdentity")
        _portable_positive(self.generation, "durable stream chunk generation")
        _portable_positive(self.sequence, "durable stream chunk sequence")
        _portable_non_negative(self.start_offset, "durable stream chunk start_offset")
        _portable_positive(self.end_offset, "durable stream chunk end_offset")
        if self.end_offset <= self.start_offset:
            raise ValueError("durable stream chunk end_offset must follow start_offset")
        if self.end_offset - self.start_offset > MAX_STREAM_CHUNK_BYTES:
            raise ValueError("durable stream chunk exceeds MAX_STREAM_CHUNK_BYTES")
        if not is_content_sha256(self.sha256):
            raise ValueError("durable stream chunk sha256 must be lowercase SHA-256")
        if (
            type(self.locator) is not str
            or not self.locator
            or len(self.locator) > 2048
            or not self.locator.isascii()
            or not all(character.isprintable() for character in self.locator)
        ):
            raise ValueError("durable stream chunk locator must be bounded printable ASCII")

    @property
    def size_bytes(self) -> int:
        return self.end_offset - self.start_offset


@dataclass(frozen=True, kw_only=True)
class DurableStreamReadChunk:
    chunk: DurableStreamChunk
    data: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.chunk, DurableStreamChunk):
            raise TypeError("durable stream read chunk metadata must be DurableStreamChunk")
        if type(self.data) is not bytes or len(self.data) != self.chunk.size_bytes:
            raise ValueError("durable stream read bytes must match chunk offsets")
        try:
            self.data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("durable stream read bytes must be complete UTF-8") from exc
        if hashlib.sha256(self.data).hexdigest() != self.chunk.sha256:
            raise ValueError("durable stream read bytes must match the chunk SHA-256")


@dataclass(frozen=True, kw_only=True)
class DurableStreamOpenResult:
    status: StreamOpenStatus
    head: DurableStreamHead | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in get_args(StreamOpenStatus):
            raise ValueError("durable stream open status is outside the portable vocabulary")
        if self.head is not None and not isinstance(self.head, DurableStreamHead):
            raise TypeError("durable stream open head must be DurableStreamHead")
        if self.status in {"opened", "already_open", "sealed"} and self.head is None:
            raise ValueError("accepted durable stream open requires a head")


@dataclass(frozen=True, kw_only=True)
class DurableStreamResetResult:
    status: StreamResetStatus
    head: DurableStreamHead | None = None
    applied_generation: int | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in get_args(StreamResetStatus):
            raise ValueError("durable stream reset status is outside the portable vocabulary")
        if self.head is not None and not isinstance(self.head, DurableStreamHead):
            raise TypeError("durable stream reset head must be DurableStreamHead")
        if self.applied_generation is not None:
            _portable_positive(
                self.applied_generation,
                "durable stream reset applied_generation",
            )
        if self.status in {"reset", "already_reset"} and (
            self.head is None or self.applied_generation is None
        ):
            raise ValueError("accepted durable stream reset requires head and generation")


@dataclass(frozen=True, kw_only=True)
class DurableStreamAppendResult:
    status: StreamAppendStatus
    head: DurableStreamHead | None = None
    chunk: DurableStreamChunk | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in get_args(StreamAppendStatus):
            raise ValueError("durable stream append status is outside the portable vocabulary")
        if self.head is not None and not isinstance(self.head, DurableStreamHead):
            raise TypeError("durable stream append head must be DurableStreamHead")
        if self.chunk is not None and not isinstance(self.chunk, DurableStreamChunk):
            raise TypeError("durable stream append chunk must be DurableStreamChunk")
        if self.status in {"committed", "already_committed"} and (
            self.head is None or self.chunk is None
        ):
            raise ValueError("accepted durable stream append requires head and chunk")


@dataclass(frozen=True, kw_only=True)
class DurableStreamSealResult:
    status: StreamSealStatus
    head: DurableStreamHead | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in get_args(StreamSealStatus):
            raise ValueError("durable stream seal status is outside the portable vocabulary")
        if self.head is not None and not isinstance(self.head, DurableStreamHead):
            raise TypeError("durable stream seal head must be DurableStreamHead")
        if self.status in {"sealed", "already_sealed"} and (
            self.head is None or self.head.state != "sealed"
        ):
            raise ValueError("accepted durable stream seal requires a sealed head")


@dataclass(frozen=True, kw_only=True)
class DurableStreamReadResult:
    status: StreamReadStatus
    requested_generation: int
    requested_cursor: int
    head: DurableStreamHead | None = None
    chunks: tuple[DurableStreamReadChunk, ...] = ()

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in get_args(StreamReadStatus):
            raise ValueError("durable stream read status is outside the portable vocabulary")
        _portable_positive(self.requested_generation, "durable stream requested_generation")
        _portable_non_negative(self.requested_cursor, "durable stream requested_cursor")
        if self.head is not None and not isinstance(self.head, DurableStreamHead):
            raise TypeError("durable stream read head must be DurableStreamHead")
        if type(self.chunks) is not tuple or any(
            not isinstance(chunk, DurableStreamReadChunk) for chunk in self.chunks
        ):
            raise TypeError("durable stream read chunks must be DurableStreamReadChunk tuple")
        if self.status == "not_found":
            if self.head is not None or self.chunks:
                raise ValueError("not_found durable stream read has no head or chunks")
            return
        if self.head is None:
            raise ValueError("durable stream read result requires a head")
        if self.status != "ok" and self.chunks:
            raise ValueError("only an ok durable stream read carries chunks")

    @property
    def next_cursor(self) -> int:
        return self.chunks[-1].chunk.end_offset if self.chunks else self.requested_cursor


class DurableStreamStore(Protocol):
    """Fenced generation/cursor metadata with immutable private chunk bytes."""

    def open(
        self,
        identity: DurableStreamIdentity,
        *,
        writer_token: WriterToken,
    ) -> DurableStreamOpenResult: ...

    def reset(
        self,
        identity: DurableStreamIdentity,
        *,
        expected_generation: int,
        reset_id: str,
        writer_token: WriterToken,
    ) -> DurableStreamResetResult: ...

    def append(
        self,
        identity: DurableStreamIdentity,
        *,
        generation: int,
        start_offset: int,
        data: bytes,
        writer_token: WriterToken,
    ) -> DurableStreamAppendResult: ...

    def seal(
        self,
        identity: DurableStreamIdentity,
        *,
        generation: int,
        final_size_bytes: int,
        final_sha256: str,
        writer_token: WriterToken,
    ) -> DurableStreamSealResult: ...

    def read_after(
        self,
        identity: DurableStreamIdentity,
        *,
        generation: int,
        cursor: int,
        limit: int = 100,
    ) -> DurableStreamReadResult: ...


__all__ = [
    "MAX_STREAM_CHUNK_BYTES",
    "MAX_STREAM_READ_CHUNKS",
    "StreamState",
    "StreamOpenStatus",
    "StreamResetStatus",
    "StreamAppendStatus",
    "StreamSealStatus",
    "StreamReadStatus",
    "DurableStreamIdentity",
    "DurableStreamHead",
    "DurableStreamChunk",
    "DurableStreamReadChunk",
    "DurableStreamOpenResult",
    "DurableStreamResetResult",
    "DurableStreamAppendResult",
    "DurableStreamSealResult",
    "DurableStreamReadResult",
    "DurableStreamStore",
]
