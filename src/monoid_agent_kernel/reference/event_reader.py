"""Snapshot-bounded byte-offset event reads for Reference implementations."""

from __future__ import annotations

import io
import os
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, BinaryIO, Callable, cast
from weakref import WeakValueDictionary

from monoid_agent_kernel.core._event_log import (
    CommittedJsonlTail,
    EventLogBoundaryError,
    EventLogChanged,
    EventLogCorruption,
    EventLogRecord,
    inspect_open_committed_jsonl_tail,
    iter_open_committed_event_records,
    iter_open_committed_jsonl_records,
)

_SOURCE_BUFFER_SIZE = 64 * 1024
_MAX_ANCHOR_INTEGER = (1 << 63) - 1
_SHA256_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


@dataclass(frozen=True)
class _VerifiedPrefixProof:
    anchor_identity: tuple[int, int, int, str]
    source_path: str
    source_device: int
    source_inode: int
    source_modified_ns: int
    source_file_size: int
    source_committed_end: int


@dataclass(frozen=True)
class EventReadAnchor:
    """A content-verified logical sequence to physical byte-offset hint."""

    seq: int
    byte_offset: int
    next_byte_offset: int
    record_sha256: str
    _prefix_proof: _VerifiedPrefixProof = field(repr=False, compare=False)


_VERIFIED_ANCHORS: WeakValueDictionary[int, EventReadAnchor] = WeakValueDictionary()
_VERIFIED_ANCHORS_LOCK = RLock()


class EventAnchorUnavailable(ValueError):
    """A derived anchor no longer carries a live verified-prefix capability."""


AnchorSelector = Callable[[EventLogRecord], bool]


@dataclass(frozen=True)
class EventReadAnchorBatch:
    """Verified sparse candidates and the newest safe-prefix anchor from one read."""

    sparse: tuple[EventReadAnchor, ...]
    tail: EventReadAnchor | None


@dataclass(frozen=True)
class _EventReadAnchorCandidate:
    seq: int
    byte_offset: int
    next_byte_offset: int
    record_sha256: str


@dataclass(frozen=True)
class EventPageRead:
    """One page plus deterministic source-work measurements."""

    events: tuple[dict[str, Any], ...]
    next_seq: int
    has_more: bool
    start_offset: int
    snapshot_end: int
    records_examined: int
    scan_bytes: int
    source_bytes_read: int
    anchor_batch: EventReadAnchorBatch | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def to_page(self) -> dict[str, Any]:
        return {
            "events": list(self.events),
            "next_seq": self.next_seq,
            "has_more": self.has_more,
        }


def iter_verified_event_read_anchors(
    events_path: Path,
) -> Generator[EventReadAnchor, None, None]:
    """Yield anchors only after verifying a strict physical prefix from byte zero."""
    try:
        handle = events_path.open("rb")
    except FileNotFoundError:
        return

    with handle:
        source = inspect_open_committed_jsonl_tail(events_path, handle)
        source_path = _normalized_source_path(events_path)
        records = iter_open_committed_event_records(
            events_path,
            handle,
            start_offset=0,
            end_offset=source.committed_end,
        )
        previous_seq: int | None = None
        try:
            for record in records:
                if previous_seq is not None and record.seq <= previous_seq:
                    raise EventLogCorruption(
                        f"cannot anchor a non-increasing event prefix: {events_path} "
                        f"at byte {record.byte_offset}"
                    )
                previous_seq = record.seq
                yield _mint_verified_anchor(
                    record,
                    source=source,
                    source_path=source_path,
                )
        finally:
            records.close()


class _CountingRawReader(io.RawIOBase):
    """Count bytes fetched from one unbuffered source handle."""

    def __init__(self, raw: BinaryIO) -> None:
        super().__init__()
        self._raw = raw
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray | memoryview) -> int | None:
        count = self._raw.readinto(buffer)
        if count is not None:
            self.bytes_read += count
        return count

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return self._raw.seek(offset, whence)

    def tell(self) -> int:
        return self._raw.tell()

    def fileno(self) -> int:
        return self._raw.fileno()

    def close(self) -> None:
        if self.closed:
            return
        try:
            super().close()
        finally:
            self._raw.close()


def read_event_page_from_anchor(
    events_path: Path,
    *,
    from_seq: int,
    limit: int | None,
    anchor: EventReadAnchor | None = None,
    anchor_selector: AnchorSelector | None = None,
) -> EventPageRead:
    """Read one logical page from a captured committed prefix.

    The optional anchor is a disposable seek hint. One open file description supplies
    tail capture, the bounded scan, and verification, so path replacement cannot mix
    bytes from different files into one result.
    """
    _validate_request(from_seq=from_seq, limit=limit, anchor=anchor)
    try:
        raw_handle = cast(BinaryIO, events_path.open("rb", buffering=0))
    except FileNotFoundError:
        if anchor is not None:
            raise EventLogChanged(
                f"event read anchor refers to a missing log: {events_path}"
            ) from None
        return _empty_page(
            from_seq=from_seq,
            anchor_batch=(
                EventReadAnchorBatch(sparse=(), tail=None)
                if anchor_selector is not None
                else None
            ),
        )

    counted_source = _CountingRawReader(raw_handle)
    try:
        with io.BufferedReader(
            counted_source,
            buffer_size=_SOURCE_BUFFER_SIZE,
        ) as handle:
            return _read_open_event_page(
                events_path,
                handle,
                counted_source,
                from_seq=from_seq,
                limit=limit,
                anchor=anchor,
                anchor_selector=anchor_selector,
            )
    finally:
        if not counted_source.closed:
            counted_source.close()


def _read_open_event_page(
    events_path: Path,
    handle: BinaryIO,
    counted_source: _CountingRawReader,
    *,
    from_seq: int,
    limit: int | None,
    anchor: EventReadAnchor | None,
    anchor_selector: AnchorSelector | None,
) -> EventPageRead:
    before = inspect_open_committed_jsonl_tail(events_path, handle)
    if anchor is not None:
        _verify_anchor_source(events_path, anchor, before)
    start_offset = anchor.byte_offset if anchor is not None else 0
    if start_offset > before.committed_end or (
        anchor is not None and anchor.next_byte_offset > before.committed_end
    ):
        raise EventLogChanged(
            f"event read anchor is beyond the committed log: {events_path} at byte {start_offset}"
        )

    events: list[dict[str, Any]] = []
    next_seq = from_seq
    records_examined = 0
    last_examined_end = start_offset
    anchor_pending = anchor is not None
    exhausted = True
    sparse_candidates: list[_EventReadAnchorCandidate] = []
    tail_candidate: _EventReadAnchorCandidate | None = None
    index_prefix = anchor_selector is not None
    previous_index_seq = anchor.seq if anchor is not None else None
    source_path = _normalized_source_path(events_path)
    records = iter_open_committed_event_records(
        events_path,
        handle,
        start_offset=start_offset,
        end_offset=before.committed_end,
    )
    try:
        try:
            for record in records:
                records_examined += 1
                last_examined_end = record.next_byte_offset
                if anchor_pending:
                    _verify_anchor(events_path, anchor, record)
                    anchor_pending = False
                elif index_prefix:
                    if previous_index_seq is not None and record.seq <= previous_index_seq:
                        index_prefix = False
                    else:
                        candidate = _anchor_candidate(record)
                        if anchor_selector is not None and anchor_selector(record):
                            sparse_candidates.append(candidate)
                        tail_candidate = candidate
                        previous_index_seq = record.seq
                if record.seq < from_seq:
                    continue
                if limit is not None and len(events) >= limit:
                    exhausted = False
                    break
                events.append(record.payload)
                next_seq = record.seq + 1
        except EventLogBoundaryError as exc:
            if anchor is None:
                raise
            raise EventLogChanged(
                f"event read anchor is no longer a record boundary: {events_path} "
                f"at byte {start_offset}"
            ) from exc
        except EventLogChanged:
            raise
        except EventLogCorruption:
            records.close()
            _verify_open_snapshot(
                events_path,
                handle,
                before,
                reject_physical_shrink=anchor is not None,
            )
            raise
    finally:
        records.close()

    if anchor_pending:
        raise EventLogChanged(
            f"event read anchor no longer identifies a committed record: {events_path} "
            f"at byte {start_offset}"
        )
    scan_bytes = (
        before.committed_end - start_offset
        if exhausted
        else last_examined_end - start_offset
    )
    _verify_open_snapshot(
        events_path,
        handle,
        before,
        reject_physical_shrink=anchor is not None,
    )
    anchor_batch = None
    if anchor_selector is not None:
        anchor_batch = _mint_anchor_batch(
            sparse_candidates,
            tail_candidate=tail_candidate,
            source=before,
            source_path=source_path,
        )
    return EventPageRead(
        events=tuple(events),
        next_seq=next_seq,
        has_more=not exhausted,
        start_offset=start_offset,
        snapshot_end=before.committed_end,
        records_examined=records_examined,
        scan_bytes=scan_bytes,
        source_bytes_read=counted_source.bytes_read,
        anchor_batch=anchor_batch,
    )


def _empty_page(
    *,
    from_seq: int,
    anchor_batch: EventReadAnchorBatch | None = None,
) -> EventPageRead:
    return EventPageRead(
        events=(),
        next_seq=from_seq,
        has_more=False,
        start_offset=0,
        snapshot_end=0,
        records_examined=0,
        scan_bytes=0,
        source_bytes_read=0,
        anchor_batch=anchor_batch,
    )


def _validate_request(
    *,
    from_seq: int,
    limit: int | None,
    anchor: EventReadAnchor | None,
) -> None:
    if not _is_exact_int(from_seq) or from_seq < 0:
        raise ValueError("from_seq must be a non-negative integer")
    if limit is not None and (not _is_exact_int(limit) or limit < 1):
        raise ValueError("limit must be a positive integer")
    if anchor is None:
        return
    if (
        not _is_exact_int(anchor.seq)
        or anchor.seq < 1
        or anchor.seq > from_seq
    ):
        raise ValueError("anchor sequence must be positive and no later than from_seq")
    if (
        not _is_bounded_anchor_int(anchor.byte_offset)
        or not _is_bounded_anchor_int(anchor.next_byte_offset)
        or anchor.byte_offset < 0
        or anchor.next_byte_offset <= anchor.byte_offset
    ):
        raise ValueError("anchor byte range must be non-negative, bounded, and ordered")
    if (
        not isinstance(anchor.record_sha256, str)
        or len(anchor.record_sha256) != 64
        or not set(anchor.record_sha256) <= _SHA256_HEX_DIGITS
    ):
        raise ValueError("anchor record_sha256 must be a SHA-256 hex digest")
    expected_identity = (
        anchor.seq,
        anchor.byte_offset,
        anchor.next_byte_offset,
        anchor.record_sha256,
    )
    with _VERIFIED_ANCHORS_LOCK:
        registered_anchor = _VERIFIED_ANCHORS.get(id(anchor))
    if (
        registered_anchor is not anchor
        or not isinstance(anchor._prefix_proof, _VerifiedPrefixProof)
        or anchor._prefix_proof.anchor_identity != expected_identity
    ):
        raise EventAnchorUnavailable(
            "anchor must prove a contiguous strictly increasing prefix"
        )


def _mint_verified_anchor(
    record: EventLogRecord | _EventReadAnchorCandidate,
    *,
    source: CommittedJsonlTail,
    source_path: str,
) -> EventReadAnchor:
    identity = (
        record.seq,
        record.byte_offset,
        record.next_byte_offset,
        record.record_sha256,
    )
    anchor = EventReadAnchor(
        seq=record.seq,
        byte_offset=record.byte_offset,
        next_byte_offset=record.next_byte_offset,
        record_sha256=record.record_sha256,
        _prefix_proof=_VerifiedPrefixProof(
            anchor_identity=identity,
            source_path=source_path,
            source_device=source.device,
            source_inode=source.inode,
            source_modified_ns=source.modified_ns,
            source_file_size=source.file_size,
            source_committed_end=source.committed_end,
        ),
    )
    with _VERIFIED_ANCHORS_LOCK:
        _VERIFIED_ANCHORS[id(anchor)] = anchor
    return anchor


def _anchor_candidate(record: EventLogRecord) -> _EventReadAnchorCandidate:
    return _EventReadAnchorCandidate(
        seq=record.seq,
        byte_offset=record.byte_offset,
        next_byte_offset=record.next_byte_offset,
        record_sha256=record.record_sha256,
    )


def _mint_anchor_batch(
    sparse_candidates: list[_EventReadAnchorCandidate],
    *,
    tail_candidate: _EventReadAnchorCandidate | None,
    source: CommittedJsonlTail,
    source_path: str,
) -> EventReadAnchorBatch:
    sparse = tuple(
        _mint_verified_anchor(candidate, source=source, source_path=source_path)
        for candidate in sparse_candidates
    )
    if tail_candidate is None:
        tail = None
    elif sparse_candidates and tail_candidate == sparse_candidates[-1]:
        tail = sparse[-1]
    else:
        tail = _mint_verified_anchor(
            tail_candidate,
            source=source,
            source_path=source_path,
        )
    return EventReadAnchorBatch(sparse=sparse, tail=tail)


def _verify_anchor_source(
    events_path: Path,
    anchor: EventReadAnchor,
    current: CommittedJsonlTail,
) -> None:
    proof = anchor._prefix_proof
    if (
        proof.source_path != _normalized_source_path(events_path)
        or (proof.source_device, proof.source_inode) != (current.device, current.inode)
    ):
        raise EventLogChanged(f"event read anchor belongs to a different log: {events_path}")
    if current.file_size < proof.source_file_size:
        raise EventLogChanged(f"event read anchor source was truncated: {events_path}")
    if current.committed_end < proof.source_committed_end:
        raise EventLogChanged(f"event read anchor source lost committed records: {events_path}")
    if (
        current.file_size == proof.source_file_size
        and current.modified_ns != proof.source_modified_ns
    ):
        raise EventLogChanged(f"event read anchor source was rewritten: {events_path}")


def event_read_anchor_extends_source(
    previous: EventReadAnchor,
    current: EventReadAnchor,
) -> bool:
    """Return whether two minted anchors belong to one append-only source generation."""
    left = previous._prefix_proof
    right = current._prefix_proof
    if (
        left.source_path != right.source_path
        or (left.source_device, left.source_inode)
        != (right.source_device, right.source_inode)
        or right.source_file_size < left.source_file_size
        or right.source_committed_end < left.source_committed_end
    ):
        return False
    return not (
        right.source_file_size == left.source_file_size
        and right.source_modified_ns != left.source_modified_ns
    )


def _normalized_source_path(events_path: Path) -> str:
    return os.path.normcase(str(events_path.resolve()))


def _is_exact_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_bounded_anchor_int(value: object) -> bool:
    return _is_exact_int(value) and cast(int, value) <= _MAX_ANCHOR_INTEGER


def _verify_anchor(
    events_path: Path,
    anchor: EventReadAnchor | None,
    record: EventLogRecord,
) -> None:
    if anchor is None or (
        record.seq != anchor.seq
        or record.byte_offset != anchor.byte_offset
        or record.next_byte_offset != anchor.next_byte_offset
        or record.record_sha256 != anchor.record_sha256
    ):
        raise EventLogChanged(
            f"event read anchor does not match the authoritative log: {events_path} "
            f"at byte {record.byte_offset}"
        )


def _verify_open_snapshot(
    events_path: Path,
    handle: BinaryIO,
    before: CommittedJsonlTail,
    *,
    reject_physical_shrink: bool,
) -> None:
    if before.last_record_sha256:
        witnesses = iter_open_committed_jsonl_records(
            events_path,
            handle,
            start_offset=before.last_record_offset,
            end_offset=before.committed_end,
        )
        try:
            try:
                witness = next(witnesses, None)
            except EventLogBoundaryError as exc:
                raise EventLogChanged(
                    f"event log committed snapshot changed while reading: {events_path}"
                ) from exc
        finally:
            witnesses.close()
        if witness is None or (
            witness.byte_offset != before.last_record_offset
            or witness.record_sha256 != before.last_record_sha256
        ):
            raise EventLogChanged(
                f"event log committed snapshot changed while reading: {events_path}"
            )

    after = os.fstat(handle.fileno())
    if (before.device, before.inode) != (after.st_dev, after.st_ino):
        raise EventLogChanged(f"event log identity changed while reading: {events_path}")
    if after.st_size < before.committed_end:
        raise EventLogChanged(f"event log was truncated while reading: {events_path}")
    if reject_physical_shrink and after.st_size < before.file_size:
        raise EventLogChanged(f"event read anchor source shrank while reading: {events_path}")
    if after.st_size == before.file_size and after.st_mtime_ns != before.modified_ns:
        raise EventLogChanged(f"event log was rewritten while reading: {events_path}")
