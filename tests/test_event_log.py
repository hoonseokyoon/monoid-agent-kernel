from __future__ import annotations

import hashlib
import json

import pytest

import monoid_agent_kernel.core._event_log as event_log
from monoid_agent_kernel.core._event_log import (
    EventLogChanged,
    EventLogCorruption,
    inspect_event_log_tail,
    iter_committed_event_records,
    repair_event_log_tail_for_append,
    validate_committed_event_sequence,
)


def _record(seq: int, *, text: str = "") -> bytes:
    return (json.dumps({"seq": seq, "data": {"text": text}}, ensure_ascii=False) + "\n").encode()


def test_committed_records_report_binary_offsets_for_utf8_lf(tmp_path) -> None:
    first = _record(1, text="한글")
    second = _record(2, text="done")
    path = tmp_path / "events.jsonl"
    path.write_bytes(first + second)

    records = list(iter_committed_event_records(path))

    assert [(item.byte_offset, item.next_byte_offset) for item in records] == [
        (0, len(first)),
        (len(first), len(first) + len(second)),
    ]
    assert [item.seq for item in records] == [1, 2]
    assert records[0].record_sha256 == hashlib.sha256(first).hexdigest()


def test_committed_records_report_binary_offsets_for_crlf(tmp_path) -> None:
    first = _record(1).replace(b"\n", b"\r\n")
    second = _record(2).replace(b"\n", b"\r\n")
    path = tmp_path / "events.jsonl"
    path.write_bytes(first + second)

    records = list(iter_committed_event_records(path, start_offset=len(first)))

    assert len(records) == 1
    assert records[0].byte_offset == len(first)
    assert records[0].next_byte_offset == len(first) + len(second)
    assert records[0].seq == 2


def test_committed_record_iterator_requires_record_boundary(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_record(1, text="private-value"))

    with pytest.raises(EventLogCorruption, match="record boundary") as caught:
        list(iter_committed_event_records(path, start_offset=3))

    assert "private-value" not in str(caught.value)


def test_committed_record_iterator_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"\n \t\r\n" + _record(3))

    assert [item.seq for item in iter_committed_event_records(path)] == [3]


@pytest.mark.parametrize("fragment", [b'{"seq":2}', b'{"seq":'])
def test_no_newline_fragment_is_uncommitted(tmp_path, fragment: bytes) -> None:
    path = tmp_path / "events.jsonl"
    first = _record(1)
    path.write_bytes(first + fragment)

    records = list(iter_committed_event_records(path))
    tail = inspect_event_log_tail(path)

    assert [item.seq for item in records] == [1]
    assert tail.last_seq == 1
    assert tail.committed_end == len(first)
    assert tail.incomplete_size == len(fragment)


@pytest.mark.parametrize(
    "committed_record, message",
    [
        (b'{"seq": "secret-marker"}\n', "invalid sequence"),
        (b'{"secret-marker"\n', "valid JSON"),
        (b"\xffsecret-marker\n", "valid UTF-8"),
        (b'["secret-marker"]\n', "JSON object"),
    ],
)
def test_committed_corruption_is_sanitized(
    tmp_path,
    committed_record: bytes,
    message: str,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(committed_record)

    with pytest.raises(EventLogCorruption, match=message) as caught:
        list(iter_committed_event_records(path))

    assert "secret-marker" not in str(caught.value)


def test_oversized_json_integer_is_sanitized_as_event_log_corruption(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"seq":' + (b"9" * 5_000) + b"}\n")

    with pytest.raises(EventLogCorruption, match="valid JSON"):
        list(iter_committed_event_records(path))


def test_tail_inspection_handles_trailing_blank_lines(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    record = _record(7)
    path.write_bytes(record + b"\n \r\n")

    tail = inspect_event_log_tail(path)

    assert tail.last_seq == 7
    assert tail.last_record_offset == 0
    assert tail.last_record_sha256 == hashlib.sha256(record).hexdigest()


def test_tail_inspection_reads_trailing_blank_bytes_once(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    record = _record(9)
    blanks = b"\n" * 100_000
    path.write_bytes(record + blanks)

    tail = inspect_event_log_tail(path)

    assert tail.last_seq == 9
    assert tail.inspected_bytes <= len(record) + len(blanks) + 1


def test_repair_truncates_only_uncommitted_suffix(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    committed = _record(1) + b"\n"
    path.write_bytes(committed + b'{"seq":2')

    tail = repair_event_log_tail_for_append(path, advertised_last_seq=1)

    assert tail.last_seq == 1
    assert tail.has_incomplete_tail is False
    assert path.read_bytes() == committed


def test_repair_refuses_ahead_watermark_without_mutation(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    original = _record(1) + b"partial"
    path.write_bytes(original)

    with pytest.raises(EventLogCorruption, match="acknowledged status watermark"):
        repair_event_log_tail_for_append(path, advertised_last_seq=2)

    assert path.read_bytes() == original


def test_repair_accepts_stale_watermark(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_record(1) + _record(2) + b"partial")

    tail = repair_event_log_tail_for_append(path, advertised_last_seq=1)

    assert tail.last_seq == 2
    assert [record.seq for record in iter_committed_event_records(path)] == [1, 2]


def test_repair_detects_same_size_tail_rewrite_before_truncation(tmp_path, monkeypatch) -> None:
    path = tmp_path / "events.jsonl"
    original = b'{"seq":1}\npartial'
    replacement = b'{"seq":9}\npartial'
    assert len(original) == len(replacement)
    path.write_bytes(original)
    inspect = event_log.inspect_event_log_tail
    mutated = False

    def inspect_then_mutate(events_path):
        nonlocal mutated
        tail = inspect(events_path)
        if not mutated:
            events_path.write_bytes(replacement)
            mutated = True
        return tail

    monkeypatch.setattr(event_log, "inspect_event_log_tail", inspect_then_mutate)

    with pytest.raises(EventLogChanged):
        repair_event_log_tail_for_append(path, advertised_last_seq=1, max_attempts=1)

    assert path.read_bytes() == replacement


@pytest.mark.parametrize("sequences", [(1, 3, 2), (1, 1)])
def test_sequence_validation_rejects_nonincreasing_committed_records(
    tmp_path,
    sequences: tuple[int, ...],
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"".join(_record(seq) for seq in sequences))

    with pytest.raises(EventLogCorruption, match="sequence is not increasing"):
        validate_committed_event_sequence(path)
