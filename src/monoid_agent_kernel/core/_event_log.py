"""Binary integrity helpers for append-only run event logs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO

_REVERSE_READ_SIZE = 64 * 1024


class EventLogCorruption(ValueError):
    """Committed event-log bytes cannot be interpreted safely."""


class EventLogChanged(EventLogCorruption):
    """The event log changed while its tail was being prepared."""


@dataclass(frozen=True)
class CommittedJsonlRecord:
    """One newline-committed physical JSONL record."""

    byte_offset: int
    next_byte_offset: int
    raw_bytes: bytes
    record_sha256: str


@dataclass(frozen=True)
class EventLogRecord:
    """One verified newline-committed JSONL record and its binary location."""

    byte_offset: int
    next_byte_offset: int
    seq: int
    payload: dict[str, Any]
    raw_json: str
    record_sha256: str


@dataclass(frozen=True)
class EventLogTail:
    """Verified state at the committed tail of an event JSONL file."""

    exists: bool
    device: int
    inode: int
    file_size: int
    committed_end: int
    incomplete_size: int
    last_record_offset: int
    last_seq: int
    last_record_sha256: str
    inspected_bytes: int

    @property
    def has_incomplete_tail(self) -> bool:
        return self.incomplete_size > 0


def iter_committed_event_records(
    path: Path,
    *,
    start_offset: int = 0,
) -> Iterator[EventLogRecord]:
    """Yield newline-committed JSON object records from an exact byte boundary."""
    for record in iter_committed_jsonl_records(path, start_offset=start_offset):
        raw_record = record.raw_bytes
        if not raw_record.strip():
            continue
        payload = _decode_event_record(path, record.byte_offset, raw_record)
        raw_json = raw_record[:-1].decode("utf-8")
        if raw_json.endswith("\r"):
            raw_json = raw_json[:-1]
        yield EventLogRecord(
            byte_offset=record.byte_offset,
            next_byte_offset=record.next_byte_offset,
            seq=_event_sequence(path, record.byte_offset, payload),
            payload=payload,
            raw_json=raw_json,
            record_sha256=record.record_sha256,
        )


def iter_committed_jsonl_records(
    path: Path,
    *,
    start_offset: int = 0,
) -> Iterator[CommittedJsonlRecord]:
    """Yield raw physical records whose final byte is the JSONL commit marker."""
    if start_offset < 0:
        raise ValueError("start_offset must be non-negative")
    try:
        handle = path.open("rb")
    except FileNotFoundError:
        return

    with handle:
        if start_offset:
            handle.seek(start_offset - 1)
            if handle.read(1) != b"\n":
                raise EventLogCorruption(
                    f"event log start offset is not a record boundary: {path} at byte {start_offset}"
                )
        handle.seek(start_offset)
        while True:
            byte_offset = handle.tell()
            raw_record = handle.readline()
            if not raw_record:
                return
            if not raw_record.endswith(b"\n"):
                return
            next_byte_offset = handle.tell()
            yield CommittedJsonlRecord(
                byte_offset=byte_offset,
                next_byte_offset=next_byte_offset,
                raw_bytes=raw_record,
                record_sha256=hashlib.sha256(raw_record).hexdigest(),
            )


def validate_committed_event_sequence(path: Path) -> int:
    """Return the final sequence after verifying strict physical monotonicity."""
    previous_seq: int | None = None
    last_seq = 0
    for record in iter_committed_event_records(path):
        if previous_seq is not None and record.seq <= previous_seq:
            raise EventLogCorruption(
                f"committed event log sequence is not increasing: {path} "
                f"at byte {record.byte_offset}"
            )
        previous_seq = record.seq
        last_seq = record.seq
    return last_seq


def inspect_event_log_tail(path: Path) -> EventLogTail:
    """Inspect the final committed record with work proportional to the physical tail."""
    try:
        handle = path.open("rb")
    except FileNotFoundError:
        return _missing_tail()

    with handle:
        return _inspect_open_event_log_tail(path, handle)


def repair_event_log_tail_for_append(
    path: Path,
    *,
    advertised_last_seq: int | None = None,
    max_attempts: int = 2,
) -> EventLogTail:
    """Verify an append watermark and remove only an uncommitted suffix.

    The caller must already own the stream's queued, live, or terminal append right. This
    helper verifies physical bytes; it does not elect or replace the logical sequence owner.
    """
    if isinstance(advertised_last_seq, bool) or (
        advertised_last_seq is not None and advertised_last_seq < 0
    ):
        raise ValueError("advertised_last_seq must be non-negative")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")

    for _attempt in range(max_attempts):
        tail = inspect_event_log_tail(path)
        if advertised_last_seq is not None and advertised_last_seq > tail.last_seq:
            raise EventLogCorruption(
                f"event log is behind the acknowledged status watermark: {path}"
            )
        if not tail.exists:
            return tail

        try:
            with path.open("r+b") as handle:
                current = _inspect_open_event_log_tail(path, handle)
                if not _same_event_log_tail(tail, current):
                    continue
                if not current.has_incomplete_tail:
                    return current
                handle.truncate(tail.committed_end)
                handle.flush()
                os.fsync(handle.fileno())
        except FileNotFoundError:
            continue
        return replace(
            current,
            file_size=current.committed_end,
            incomplete_size=0,
        )
    raise EventLogChanged(f"event log changed while its tail was repaired: {path}")


def _missing_tail() -> EventLogTail:
    return EventLogTail(
        exists=False,
        device=0,
        inode=0,
        file_size=0,
        committed_end=0,
        incomplete_size=0,
        last_record_offset=0,
        last_seq=0,
        last_record_sha256="",
        inspected_bytes=0,
    )


def _inspect_open_event_log_tail(path: Path, handle: BinaryIO) -> EventLogTail:
    stat = os.fstat(handle.fileno())
    file_size = stat.st_size
    if file_size == 0:
        return EventLogTail(
            exists=True,
            device=stat.st_dev,
            inode=stat.st_ino,
            file_size=0,
            committed_end=0,
            incomplete_size=0,
            last_record_offset=0,
            last_seq=0,
            last_record_sha256="",
            inspected_bytes=0,
        )

    handle.seek(file_size - 1)
    final_byte = handle.read(1)
    inspected_bytes = 1
    if final_byte == b"\n":
        committed_end = file_size
    else:
        newline_offset, searched = _find_previous_newline(handle, file_size)
        inspected_bytes += searched
        committed_end = 0 if newline_offset is None else newline_offset + 1

    record_offset, raw_record, record_bytes = _last_nonblank_committed_record(
        handle,
        committed_end,
    )
    inspected_bytes += record_bytes
    if raw_record is None:
        last_seq = 0
        record_sha256 = ""
    else:
        payload = _decode_event_record(path, record_offset, raw_record + b"\n")
        last_seq = _event_sequence(path, record_offset, payload)
        record_sha256 = hashlib.sha256(raw_record + b"\n").hexdigest()
    return EventLogTail(
        exists=True,
        device=stat.st_dev,
        inode=stat.st_ino,
        file_size=file_size,
        committed_end=committed_end,
        incomplete_size=file_size - committed_end,
        last_record_offset=record_offset,
        last_seq=last_seq,
        last_record_sha256=record_sha256,
        inspected_bytes=inspected_bytes,
    )


def _same_event_log_tail(left: EventLogTail, right: EventLogTail) -> bool:
    return (
        left.exists == right.exists
        and left.device == right.device
        and left.inode == right.inode
        and left.file_size == right.file_size
        and left.committed_end == right.committed_end
        and left.incomplete_size == right.incomplete_size
        and left.last_record_offset == right.last_record_offset
        and left.last_seq == right.last_seq
        and left.last_record_sha256 == right.last_record_sha256
    )


def _find_previous_newline(handle: BinaryIO, end: int) -> tuple[int | None, int]:
    cursor = end
    inspected = 0
    while cursor > 0:
        size = min(_REVERSE_READ_SIZE, cursor)
        cursor -= size
        handle.seek(cursor)
        block = handle.read(size)
        inspected += len(block)
        index = block.rfind(b"\n")
        if index >= 0:
            return cursor + index, inspected
    return None, inspected


def _last_nonblank_committed_record(
    handle: BinaryIO,
    committed_end: int,
) -> tuple[int, bytes | None, int]:
    if committed_end == 0:
        return 0, None, 0

    cursor = committed_end - 1
    suffix = b""
    inspected = 0
    while cursor > 0:
        size = min(_REVERSE_READ_SIZE, cursor)
        cursor -= size
        handle.seek(cursor)
        block = handle.read(size)
        inspected += len(block)
        candidate_data = block + suffix
        while True:
            newline_index = candidate_data.rfind(b"\n")
            if newline_index < 0:
                suffix = candidate_data
                break
            record = candidate_data[newline_index + 1 :]
            record_offset = cursor + newline_index + 1
            if record.strip():
                return record_offset, record, inspected
            candidate_data = candidate_data[:newline_index]
            suffix = candidate_data

    if suffix.strip():
        return 0, suffix, inspected
    return 0, None, inspected


def _decode_event_record(path: Path, byte_offset: int, raw_record: bytes) -> dict[str, Any]:
    try:
        text = raw_record.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EventLogCorruption(
            f"committed event log record is not valid UTF-8: {path} at byte {byte_offset}"
        ) from exc
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise EventLogCorruption(
            f"committed event log record is not valid JSON: {path} at byte {byte_offset}"
        ) from exc
    if not isinstance(payload, dict):
        raise EventLogCorruption(
            f"committed event log record must be a JSON object: {path} at byte {byte_offset}"
        )
    return payload


def _event_sequence(path: Path, byte_offset: int, event: dict[str, Any]) -> int:
    seq = event.get("seq")
    if isinstance(seq, bool) or not isinstance(seq, int) or seq <= 0:
        raise EventLogCorruption(
            f"committed event log record has an invalid sequence: {path} at byte {byte_offset}"
        )
    return seq
