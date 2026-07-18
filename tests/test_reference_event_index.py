from __future__ import annotations

import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

import monoid_agent_kernel.reference.event_index as event_index_module
import monoid_agent_kernel.reference.event_reader as event_reader_module
from monoid_agent_kernel.core._event_log import EventLogCorruption
from monoid_agent_kernel.core.event_sequencing import read_event_page
from monoid_agent_kernel.reference.event_index import ReferenceEventOffsetIndex


def _record(seq: int, *, text: str = "") -> bytes:
    return (
        json.dumps({"seq": seq, "type": "run.started", "data": {"text": text}}, ensure_ascii=False)
        + "\n"
    ).encode()


@pytest.mark.parametrize(
    ("from_seq", "limit"),
    [(0, None), (0, 1), (2, None), (2, 1), (4, 2), (99, None)],
)
def test_indexed_reader_matches_core_page_semantics(
    tmp_path: Path,
    from_seq: int,
    limit: int | None,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(
        b"\n"
        + _record(1, text="한글").replace(b"\n", b"\r\n")
        + b" \t\n"
        + _record(3)
        + _record(7)
        + b'{"seq":8}'
    )
    index = ReferenceEventOffsetIndex(byte_stride=128, record_stride=2)

    observed = index.read_page(path, from_seq=from_seq, limit=limit)

    assert observed.to_page() == read_event_page(path, from_seq=from_seq, limit=limit)


@pytest.mark.parametrize("contents", [None, b""])
def test_indexed_reader_matches_missing_and_empty_logs(
    tmp_path: Path,
    contents: bytes | None,
) -> None:
    path = tmp_path / "events.jsonl"
    if contents is not None:
        path.write_bytes(contents)
    index = ReferenceEventOffsetIndex()

    observed = index.read_page(path, from_seq=5, limit=1)

    assert observed.to_page() == read_event_page(path, from_seq=5, limit=1)


def test_warm_index_preserves_unbounded_python_integer_sequences(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    sequence = (1 << 63) + 1
    path.write_bytes(_record(sequence))
    index = ReferenceEventOffsetIndex()

    first = index.read_page(path, from_seq=sequence, limit=None)
    second = index.read_page(path, from_seq=sequence, limit=None)

    expected = read_event_page(path, from_seq=sequence, limit=None)
    assert first.to_page() == expected
    assert second.to_page() == expected


def test_warm_tail_work_is_independent_of_retained_history(tmp_path: Path) -> None:
    def warm_read(count: int) -> tuple[int, int, int]:
        path = tmp_path / f"events-{count}.jsonl"
        path.write_bytes(
            b"".join(_record(seq, text="fixed-width") for seq in range(1, count + 1))
        )
        index = ReferenceEventOffsetIndex(byte_stride=64 * 1024, record_stride=256)
        first = index.read_page(path, from_seq=count - 4, limit=2)
        second = index.read_page(path, from_seq=first.next_seq, limit=2)
        stats = index.stats(path)
        assert stats is not None
        assert stats.from_zero_reads == 1
        return second.records_examined, second.source_bytes_read, stats.anchor_count

    small = warm_read(1_000)
    large = warm_read(100_000)

    assert small[0] == large[0] == 3
    assert max(small[1], large[1]) <= 4 * 64 * 1024
    assert abs(small[1] - large[1]) <= 64 * 1024
    assert large[2] < 500


def test_cold_build_mints_only_sparse_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"".join(_record(seq) for seq in range(1, 10_001)))
    minted = 0
    mint_anchor = event_reader_module._mint_verified_anchor

    def count_mint(*args, **kwargs):
        nonlocal minted
        minted += 1
        return mint_anchor(*args, **kwargs)

    monkeypatch.setattr(event_reader_module, "_mint_verified_anchor", count_mint)
    index = ReferenceEventOffsetIndex(byte_stride=10**9, record_stride=64)

    index.read_page(path, from_seq=9_995, limit=2)
    stats = index.stats(path)

    assert stats is not None
    assert minted == stats.anchor_count
    assert minted < 170


def test_repeated_small_appends_do_not_turn_tail_samples_dense(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"".join(_record(seq) for seq in range(1, 6)))
    index = ReferenceEventOffsetIndex(byte_stride=10**9, record_stride=4)
    index.read_page(path, from_seq=5, limit=None)
    initial = index.stats(path)
    assert initial is not None

    for seq in range(6, 10):
        with path.open("ab") as handle:
            handle.write(_record(seq))
        index.read_page(path, from_seq=seq, limit=None)

    final = index.stats(path)
    assert final is not None
    assert final.indexed_through_seq == 9
    assert final.anchor_count <= initial.anchor_count + 1


def test_byte_stride_bounds_sparse_anchor_density(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(
        b"".join(_record(seq, text="x" * 128) for seq in range(1, 101))
    )
    byte_stride = 1_024
    index = ReferenceEventOffsetIndex(byte_stride=byte_stride, record_stride=10**9)

    index.read_page(path, from_seq=0, limit=None)
    stats = index.stats(path)

    assert stats is not None
    assert stats.anchor_count > 2
    assert stats.anchor_count <= path.stat().st_size // byte_stride + 3


def test_append_extends_warm_index_without_another_from_zero_read(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"".join(_record(seq) for seq in range(1, 11)))
    index = ReferenceEventOffsetIndex(byte_stride=128, record_stride=2)
    first = index.read_page(path, from_seq=8, limit=2)
    with path.open("ab") as handle:
        handle.write(_record(11) + _record(12) + _record(13))

    second = index.read_page(path, from_seq=first.next_seq, limit=2)
    stats = index.stats(path)

    assert second.to_page() == read_event_page(
        path,
        from_seq=first.next_seq,
        limit=2,
    )
    assert stats is not None
    assert stats.from_zero_reads == 1
    assert stats.indexed_through_seq >= 12


def test_new_index_cold_rebuilds_once_then_stays_warm(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"".join(_record(seq) for seq in range(1, 1_001)))
    first_process = ReferenceEventOffsetIndex()
    first_process.read_page(path, from_seq=995, limit=2)

    restarted = ReferenceEventOffsetIndex()
    first = restarted.read_page(path, from_seq=995, limit=2)
    restarted.read_page(path, from_seq=first.next_seq, limit=2)
    stats = restarted.stats(path)

    assert stats is not None
    assert stats.from_zero_reads == 1
    assert stats.pages == 2


def test_replacement_invalidates_and_rebuilds_from_authoritative_log(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    replacement = tmp_path / "replacement.jsonl"
    path.write_bytes(_record(1) + _record(2))
    replacement.write_bytes(_record(9) + _record(10))
    index = ReferenceEventOffsetIndex()
    index.read_page(path, from_seq=2, limit=None)
    replacement.replace(path)

    observed = index.read_page(path, from_seq=2, limit=None)
    stats = index.stats(path)

    assert observed.to_page() == read_event_page(path, from_seq=2, limit=None)
    assert stats is not None
    assert stats.invalidations == 1
    assert stats.from_zero_reads == 2


def test_same_size_rewrite_invalidates_and_rebuilds(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_record(1) + _record(2))
    index = ReferenceEventOffsetIndex()
    index.read_page(path, from_seq=2, limit=None)
    before = path.stat()
    path.write_bytes(_record(9) + _record(2))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))

    observed = index.read_page(path, from_seq=2, limit=None)
    stats = index.stats(path)

    assert observed.to_page() == read_event_page(path, from_seq=2, limit=None)
    assert stats is not None
    assert stats.invalidations == 1


def test_incomplete_tail_repair_invalidates_and_rebuilds(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    committed = _record(1) + _record(2)
    path.write_bytes(committed + b'{"seq":3')
    index = ReferenceEventOffsetIndex()
    index.read_page(path, from_seq=2, limit=None)
    with path.open("r+b") as handle:
        handle.truncate(len(committed))

    observed = index.read_page(path, from_seq=2, limit=None)
    stats = index.stats(path)

    assert observed.to_page() == read_event_page(path, from_seq=2, limit=None)
    assert stats is not None
    assert stats.invalidations == 1


def test_committed_prefix_truncation_invalidates_and_rebuilds(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    retained = _record(1) + _record(2)
    path.write_bytes(retained + _record(3) + _record(4) + _record(5))
    index = ReferenceEventOffsetIndex()
    index.read_page(path, from_seq=5, limit=None)
    with path.open("r+b") as handle:
        handle.truncate(len(retained))

    observed = index.read_page(path, from_seq=5, limit=None)
    stats = index.stats(path)

    assert observed.to_page() == read_event_page(path, from_seq=5, limit=None)
    assert stats is not None
    assert stats.invalidations == 1
    assert stats.indexed_through_seq == 2


def test_deleted_warm_log_invalidates_to_an_empty_authoritative_page(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_record(1) + _record(2))
    index = ReferenceEventOffsetIndex()
    index.read_page(path, from_seq=2, limit=None)
    path.unlink()

    observed = index.read_page(path, from_seq=2, limit=None)
    stats = index.stats(path)

    assert observed.to_page() == read_event_page(path, from_seq=2, limit=None)
    assert stats is not None
    assert stats.anchor_count == 0
    assert stats.invalidations == 1
    assert stats.from_zero_reads == 2
    assert stats.pages == 2


def test_committed_corruption_propagates_without_publishing_index(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_record(1) + b'{"seq":\n')
    index = ReferenceEventOffsetIndex()

    with pytest.raises(EventLogCorruption, match="valid JSON"):
        index.read_page(path, from_seq=0, limit=None)

    stats = index.stats(path)
    assert stats is not None
    assert stats.anchor_count == 0
    assert stats.pages == 0


def test_stale_anchor_fallback_propagates_new_committed_corruption_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_record(1) + _record(2))
    index = ReferenceEventOffsetIndex()
    index.read_page(path, from_seq=2, limit=None)
    path.write_bytes(b'{"seq":\n')
    reads = 0
    read_page = event_index_module.read_event_page_from_anchor

    def count_read(*args, **kwargs):
        nonlocal reads
        reads += 1
        return read_page(*args, **kwargs)

    monkeypatch.setattr(event_index_module, "read_event_page_from_anchor", count_read)

    with pytest.raises(EventLogCorruption, match="valid JSON"):
        index.read_page(path, from_seq=2, limit=None)

    stats = index.stats(path)
    assert reads == 2
    assert stats is not None
    assert stats.anchor_count == 0
    assert stats.invalidations == 1
    assert stats.pages == 1


@pytest.mark.parametrize("sequences", [(5, 1, 6), (1, 1, 2)])
def test_nonmonotonic_logs_preserve_core_pages_without_unsafe_extension(
    tmp_path: Path,
    sequences: tuple[int, ...],
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"".join(_record(seq) for seq in sequences))
    index = ReferenceEventOffsetIndex(byte_stride=1, record_stride=1)

    first = index.read_page(path, from_seq=0, limit=None)
    second = index.read_page(path, from_seq=sequences[0], limit=None)
    stats = index.stats(path)

    assert first.to_page() == read_event_page(path, from_seq=0, limit=None)
    assert second.to_page() == read_event_page(
        path,
        from_seq=sequences[0],
        limit=None,
    )
    assert stats is not None
    assert stats.indexed_through_seq == sequences[0]


def test_index_holds_anchor_capabilities_across_garbage_collection(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"".join(_record(seq) for seq in range(1, 101)))
    index = ReferenceEventOffsetIndex(record_stride=8)
    first = index.read_page(path, from_seq=90, limit=2)

    gc.collect()
    second = index.read_page(path, from_seq=first.next_seq, limit=2)

    assert second.to_page() == read_event_page(
        path,
        from_seq=first.next_seq,
        limit=2,
    )
    assert index.stats(path).from_zero_reads == 1  # type: ignore[union-attr]


def test_same_source_cold_reads_single_flight(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"".join(_record(seq) for seq in range(1, 1_001)))
    index = ReferenceEventOffsetIndex(record_stride=16)
    start = Barrier(3)

    def read() -> dict[str, object]:
        start.wait()
        return index.read_page(path, from_seq=995, limit=2).to_page()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(read) for _ in range(2)]
        start.wait()
        results = [future.result() for future in futures]

    assert results[0] == results[1]
    assert index.stats(path).from_zero_reads == 1  # type: ignore[union-attr]


def test_different_sources_do_not_share_io_lock(tmp_path: Path, monkeypatch) -> None:
    paths = [tmp_path / "first.jsonl", tmp_path / "second.jsonl"]
    for path in paths:
        path.write_bytes(_record(1))
    index = ReferenceEventOffsetIndex()
    entered = Barrier(2)
    read_page = event_index_module.read_event_page_from_anchor

    def synchronized_read(*args, **kwargs):
        entered.wait(timeout=5)
        return read_page(*args, **kwargs)

    monkeypatch.setattr(event_index_module, "read_event_page_from_anchor", synchronized_read)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda path: index.read_page(path, from_seq=0, limit=1).to_page(),
                paths,
            )
        )

    assert [result["events"][0]["seq"] for result in results] == [1, 1]


def test_sparse_anchor_density_is_bounded_by_record_stride(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"".join(_record(seq) for seq in range(1, 101)))
    index = ReferenceEventOffsetIndex(byte_stride=10**9, record_stride=10)

    index.read_page(path, from_seq=0, limit=None)
    stats = index.stats(path)

    assert stats is not None
    assert stats.indexed_through_seq == 100
    assert stats.anchor_count <= 12
