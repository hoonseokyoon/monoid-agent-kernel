from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import monoid_agent_kernel.reference.event_reader as event_reader
from monoid_agent_kernel.core._event_log import (
    EventLogChanged,
    EventLogCorruption,
    inspect_event_log_tail,
    iter_committed_event_records,
)
from monoid_agent_kernel.core.event_sequencing import read_event_page
from monoid_agent_kernel.reference.event_reader import (
    EventReadAnchor,
    read_event_page_from_anchor,
)


def _record(seq: int, *, text: str = "") -> bytes:
    return (
        json.dumps({"seq": seq, "type": "run.started", "data": {"text": text}}, ensure_ascii=False)
        + "\n"
    ).encode()


@pytest.mark.parametrize(
    ("from_seq", "limit"),
    [(0, None), (0, 1), (2, None), (2, 1), (3, 2), (9, None)],
)
def test_snapshot_reader_matches_core_page_semantics(
    tmp_path: Path,
    from_seq: int,
    limit: int | None,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(
        b"\n" + _record(1, text="한글").replace(b"\n", b"\r\n") + b" \t\n"
        + _record(3, text="three")
        + _record(7, text="seven")
        + b'{"seq":8}'
    )

    expected = read_event_page(path, from_seq=from_seq, limit=limit)
    observed = read_event_page_from_anchor(path, from_seq=from_seq, limit=limit)

    assert observed.to_page() == expected
    assert observed.start_offset == 0
    assert observed.snapshot_end == inspect_event_log_tail(path).committed_end


@pytest.mark.parametrize("contents", [None, b""])
def test_missing_and_empty_logs_match_core_page_semantics(
    tmp_path: Path,
    contents: bytes | None,
) -> None:
    path = tmp_path / "events.jsonl"
    if contents is not None:
        path.write_bytes(contents)

    observed = read_event_page_from_anchor(path, from_seq=5, limit=1)

    assert observed.to_page() == read_event_page(path, from_seq=5, limit=1)
    assert observed.source_bytes_read == 0


@pytest.mark.parametrize("sequences", [(2, 1), (1, 1)])
def test_snapshot_reader_preserves_core_nonincreasing_page_semantics(
    tmp_path: Path,
    sequences: tuple[int, ...],
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"".join(_record(seq) for seq in sequences))

    observed = read_event_page_from_anchor(path, from_seq=0, limit=None)

    assert observed.to_page() == read_event_page(path, from_seq=0, limit=None)


@pytest.mark.parametrize(
    "invalid_tail",
    [b'{"seq":\n', b'{"seq":0}\n', b"\xff\n"],
)
@pytest.mark.parametrize("anchored", [False, True])
def test_finite_page_does_not_decode_committed_tail_beyond_core_lookahead(
    tmp_path: Path,
    invalid_tail: bytes,
    anchored: bool,
) -> None:
    path = tmp_path / "events.jsonl"
    first = _record(1)
    path.write_bytes(first + _record(2) + invalid_tail)
    anchor = None
    if anchored:
        anchor = EventReadAnchor.from_record(next(iter_committed_event_records(path)))

    observed = read_event_page_from_anchor(
        path,
        from_seq=1,
        limit=1,
        anchor=anchor,
    )

    assert observed.to_page() == read_event_page(path, from_seq=1, limit=1)


def test_static_malformed_lookahead_remains_authoritative_corruption(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_record(1) + b'{"seq":\n')

    with pytest.raises(EventLogCorruption, match="valid JSON") as caught:
        read_event_page_from_anchor(path, from_seq=0, limit=1)

    assert not isinstance(caught.value, EventLogChanged)


def test_verified_anchor_skips_prefix_and_preserves_page_results(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"".join(_record(seq, text=f"value-{seq}") for seq in range(1, 101)))
    anchor_record = next(
        record for record in iter_committed_event_records(path) if record.seq == 91
    )
    anchor = EventReadAnchor.from_record(anchor_record)

    observed = read_event_page_from_anchor(path, from_seq=93, limit=3, anchor=anchor)

    assert observed.to_page() == read_event_page(path, from_seq=93, limit=3)
    assert [event["seq"] for event in observed.events] == [93, 94, 95]
    assert observed.records_examined == 6
    assert observed.scan_bytes < path.stat().st_size // 4
    assert observed.source_bytes_read < 2 * 64 * 1024


@pytest.mark.parametrize(
    "field",
    ["seq", "byte_offset", "next_byte_offset", "record_sha256"],
)
def test_anchor_content_mismatch_fails_closed(tmp_path: Path, field: str) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_record(1) + _record(2))
    record = next(iter_committed_event_records(path))
    values = {
        "seq": record.seq,
        "byte_offset": record.byte_offset,
        "next_byte_offset": record.next_byte_offset,
        "record_sha256": record.record_sha256,
    }
    values[field] = (
        2
        if field == "seq"
        else record.byte_offset + 1
        if field == "byte_offset"
        else record.next_byte_offset + 1
        if field == "next_byte_offset"
        else hashlib.sha256(b"different").hexdigest()
    )
    anchor = EventReadAnchor(**values)

    with pytest.raises(EventLogChanged):
        read_event_page_from_anchor(path, from_seq=2, limit=1, anchor=anchor)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seq", True),
        ("byte_offset", "0"),
        ("next_byte_offset", None),
        ("next_byte_offset", 1 << 63),
        ("record_sha256", b"0" * 64),
    ],
)
def test_anchor_validation_rejects_untrusted_index_types(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = tmp_path / "events.jsonl"
    record_bytes = _record(1)
    path.write_bytes(record_bytes)
    values = {
        "seq": 1,
        "byte_offset": 0,
        "next_byte_offset": len(record_bytes),
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
    }
    values[field] = value
    anchor = EventReadAnchor(**values)

    with pytest.raises(ValueError):
        read_event_page_from_anchor(path, from_seq=1, limit=1, anchor=anchor)


def test_reader_stays_within_captured_committed_snapshot(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_record(1))
    inspect = event_reader.inspect_open_committed_jsonl_tail
    calls = 0

    def append_after_snapshot(events_path: Path, handle):
        nonlocal calls
        tail = inspect(events_path, handle)
        calls += 1
        if calls == 1:
            with events_path.open("ab") as handle:
                handle.write(_record(2))
        return tail

    monkeypatch.setattr(
        event_reader,
        "inspect_open_committed_jsonl_tail",
        append_after_snapshot,
    )

    first = read_event_page_from_anchor(path, from_seq=0, limit=None)
    second = read_event_page_from_anchor(path, from_seq=0, limit=None)

    assert [event["seq"] for event in first.events] == [1]
    assert [event["seq"] for event in second.events] == [1, 2]


def test_later_malformed_append_does_not_change_captured_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_record(1))
    inspect = event_reader.inspect_open_committed_jsonl_tail
    appended = False

    def append_corruption_after_snapshot(events_path: Path, handle):
        nonlocal appended
        tail = inspect(events_path, handle)
        if not appended:
            with events_path.open("ab") as handle:
                handle.write(b'{"seq":\n')
            appended = True
        return tail

    monkeypatch.setattr(
        event_reader,
        "inspect_open_committed_jsonl_tail",
        append_corruption_after_snapshot,
    )

    captured = read_event_page_from_anchor(path, from_seq=0, limit=None)

    assert [event["seq"] for event in captured.events] == [1]
    with pytest.raises(EventLogCorruption, match="valid JSON"):
        read_event_page_from_anchor(path, from_seq=0, limit=None)


def test_reader_fails_if_captured_prefix_is_truncated(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_record(1) + _record(2))
    inspect = event_reader.inspect_open_committed_jsonl_tail
    mutated = False

    def truncate_after_snapshot(events_path: Path, handle):
        nonlocal mutated
        tail = inspect(events_path, handle)
        if not mutated:
            events_path.write_bytes(_record(1))
            mutated = True
        return tail

    monkeypatch.setattr(
        event_reader,
        "inspect_open_committed_jsonl_tail",
        truncate_after_snapshot,
    )

    with pytest.raises(EventLogChanged):
        read_event_page_from_anchor(path, from_seq=0, limit=None)


def test_reader_fails_if_same_file_is_rewritten_after_scan(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "events.jsonl"
    original = _record(1, text="original")
    replacement = _record(1, text="changed!")
    assert len(original) == len(replacement)
    path.write_bytes(original)
    iterate = event_reader.iter_open_committed_event_records
    rewritten = False

    def rewrite_after_scan(*args, **kwargs):
        nonlocal rewritten
        yield from iterate(*args, **kwargs)
        if not rewritten:
            before = path.stat()
            path.write_bytes(replacement)
            os.utime(
                path,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
            rewritten = True

    monkeypatch.setattr(
        event_reader,
        "iter_open_committed_event_records",
        rewrite_after_scan,
    )

    with pytest.raises(EventLogChanged, match="snapshot changed|rewritten"):
        read_event_page_from_anchor(path, from_seq=0, limit=None)


def test_tail_witness_boundary_mutation_is_snapshot_change(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "events.jsonl"
    first = _record(1)
    path.write_bytes(first + _record(2))
    iterate = event_reader.iter_open_committed_event_records
    mutated = False

    def mutate_boundary_after_scan(*args, **kwargs):
        nonlocal mutated
        yield from iterate(*args, **kwargs)
        if not mutated:
            with path.open("r+b") as writer:
                writer.seek(len(first) - 1)
                writer.write(b" ")
            mutated = True

    monkeypatch.setattr(
        event_reader,
        "iter_open_committed_event_records",
        mutate_boundary_after_scan,
    )

    with pytest.raises(EventLogChanged):
        read_event_page_from_anchor(path, from_seq=0, limit=None)


def test_boundary_mutation_before_scan_is_snapshot_change(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "events.jsonl"
    first = _record(1)
    path.write_bytes(first + _record(2))
    inspect = event_reader.inspect_open_committed_jsonl_tail
    mutated = False

    def mutate_boundary_after_capture(events_path: Path, handle):
        nonlocal mutated
        tail = inspect(events_path, handle)
        if not mutated:
            with events_path.open("r+b") as writer:
                writer.seek(len(first) - 1)
                writer.write(b" ")
            mutated = True
        return tail

    monkeypatch.setattr(
        event_reader,
        "inspect_open_committed_jsonl_tail",
        mutate_boundary_after_capture,
    )

    with pytest.raises(EventLogChanged):
        read_event_page_from_anchor(path, from_seq=0, limit=None)


def test_incomplete_tail_repair_preserves_captured_page(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "events.jsonl"
    committed = _record(1)
    path.write_bytes(committed + b'{"seq":2')
    inspect = event_reader.inspect_open_committed_jsonl_tail
    repaired = False

    def repair_after_snapshot(events_path: Path, handle):
        nonlocal repaired
        tail = inspect(events_path, handle)
        if not repaired:
            with events_path.open("r+b") as writer:
                writer.truncate(tail.committed_end)
            repaired = True
        return tail

    monkeypatch.setattr(
        event_reader,
        "inspect_open_committed_jsonl_tail",
        repair_after_snapshot,
    )

    observed = read_event_page_from_anchor(path, from_seq=0, limit=None)

    assert [event["seq"] for event in observed.events] == [1]
    assert path.read_bytes() == committed


def test_tail_capture_scan_and_witness_share_one_handle(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_record(1) + _record(2))
    inspect = event_reader.inspect_open_committed_jsonl_tail
    iterate = event_reader.iter_open_committed_event_records
    iterate_raw = event_reader.iter_open_committed_jsonl_records
    handles: list[object] = []

    def observe_inspect(events_path: Path, handle):
        handles.append(handle)
        return inspect(events_path, handle)

    def observe_iteration(events_path: Path, handle, **kwargs):
        handles.append(handle)
        yield from iterate(events_path, handle, **kwargs)

    def observe_raw_iteration(events_path: Path, handle, **kwargs):
        handles.append(handle)
        yield from iterate_raw(events_path, handle, **kwargs)

    monkeypatch.setattr(
        event_reader,
        "inspect_open_committed_jsonl_tail",
        observe_inspect,
    )
    monkeypatch.setattr(
        event_reader,
        "iter_open_committed_event_records",
        observe_iteration,
    )
    monkeypatch.setattr(
        event_reader,
        "iter_open_committed_jsonl_records",
        observe_raw_iteration,
    )

    observed = read_event_page_from_anchor(path, from_seq=0, limit=1)

    assert [event["seq"] for event in observed.events] == [1]
    assert len(handles) == 3
    assert all(handle is handles[0] for handle in handles)


def test_replacement_aba_cannot_mix_snapshot_bytes(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "events.jsonl"
    saved_original = tmp_path / "saved-original.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    displaced_replacement = tmp_path / "displaced-replacement.jsonl"
    path.write_bytes(_record(1, text="original"))
    replacement.write_bytes(_record(1, text="replacement"))
    iterate = event_reader.iter_open_committed_event_records
    swapped = False

    def swap_around_scan(events_path: Path, handle, **kwargs):
        nonlocal swapped
        did_swap = False
        if not swapped:
            try:
                events_path.replace(saved_original)
            except PermissionError:
                pytest.skip("platform prevents replacement while the snapshot handle is open")
            replacement.replace(events_path)
            swapped = True
            did_swap = True
        try:
            yield from iterate(events_path, handle, **kwargs)
        finally:
            if did_swap:
                events_path.replace(displaced_replacement)
                saved_original.replace(events_path)

    monkeypatch.setattr(
        event_reader,
        "iter_open_committed_event_records",
        swap_around_scan,
    )

    observed = read_event_page_from_anchor(path, from_seq=0, limit=None)

    assert [event["data"]["text"] for event in observed.events] == ["original"]
    assert path.read_bytes() == _record(1, text="original")


def test_source_bytes_are_observed_at_raw_read_boundary(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"".join(_record(seq) for seq in range(1, 101)))
    readinto = event_reader._CountingRawReader.readinto
    observed_raw_bytes = 0

    def observe_readinto(reader, buffer):
        nonlocal observed_raw_bytes
        count = readinto(reader, buffer)
        observed_raw_bytes += count or 0
        return count

    monkeypatch.setattr(event_reader._CountingRawReader, "readinto", observe_readinto)

    observed = read_event_page_from_anchor(path, from_seq=90, limit=2)

    assert observed.source_bytes_read == observed_raw_bytes
    assert observed.source_bytes_read <= 4 * 64 * 1024


def test_snapshot_handle_closes_after_success_and_decode_error(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    path.write_bytes(_record(1))
    replacement.write_bytes(_record(2))

    read_event_page_from_anchor(path, from_seq=0, limit=1)
    replacement.replace(path)
    path.write_bytes(b'{"seq":\n')

    with pytest.raises(EventLogCorruption):
        read_event_page_from_anchor(path, from_seq=0, limit=None)
    path.unlink()


def test_near_tail_source_work_is_independent_of_retained_history(tmp_path: Path) -> None:
    def read_tail(count: int) -> tuple[int, int]:
        path = tmp_path / f"events-{count}.jsonl"
        records = [_record(seq, text="fixed-width") for seq in range(1, count + 1)]
        path.write_bytes(b"".join(records))
        target_seq = count - 4
        target_offset = sum(len(record) for record in records[: target_seq - 1])
        target_raw = records[target_seq - 1]
        anchor = EventReadAnchor(
            seq=target_seq,
            byte_offset=target_offset,
            next_byte_offset=target_offset + len(target_raw),
            record_sha256=hashlib.sha256(target_raw).hexdigest(),
        )
        page = read_event_page_from_anchor(
            path,
            from_seq=target_seq,
            limit=2,
            anchor=anchor,
        )
        assert [event["seq"] for event in page.events] == [target_seq, target_seq + 1]
        expected_scan = sum(len(record) for record in records[target_seq - 1 : target_seq + 2])
        assert page.scan_bytes == expected_scan
        return page.records_examined, page.source_bytes_read

    small_work = read_tail(1_000)
    large_work = read_tail(100_000)

    assert small_work[0] == large_work[0] == 3
    assert max(small_work[1], large_work[1]) <= 4 * 64 * 1024
    assert abs(small_work[1] - large_work[1]) <= 64 * 1024
