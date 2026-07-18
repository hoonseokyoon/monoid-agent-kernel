from __future__ import annotations

import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, Event, Lock

import pytest

import monoid_agent_kernel.reference.event_index as event_index_module
import monoid_agent_kernel.reference.event_reader as event_reader_module
from monoid_agent_kernel.core._event_log import EventLogChanged, EventLogCorruption
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


@pytest.mark.parametrize("max_sources", [True, -1, 1.5, "1"])
def test_source_capacity_rejects_invalid_values(max_sources: object) -> None:
    with pytest.raises(ValueError, match="max_sources"):
        ReferenceEventOffsetIndex(max_sources=max_sources)  # type: ignore[arg-type]


def test_source_lru_is_bounded_and_stats_does_not_change_recency(tmp_path: Path) -> None:
    paths = [tmp_path / f"events-{index}.jsonl" for index in range(3)]
    for path in paths:
        path.write_bytes(_record(1))
    index = ReferenceEventOffsetIndex(max_sources=2)

    index.read_page(paths[0], from_seq=1, limit=None)
    index.read_page(paths[1], from_seq=1, limit=None)
    assert index.stats(paths[0]) is not None
    index.read_page(paths[2], from_seq=1, limit=None)

    assert index.stats(paths[0]) is None
    assert index.stats(paths[1]) is not None
    assert index.stats(paths[2]) is not None
    first = index.cache_stats()
    assert first.sources == 2
    assert first.hits == 0
    assert first.misses == 3
    assert first.evictions == 1

    index.read_page(paths[1], from_seq=1, limit=None)
    index.read_page(paths[0], from_seq=1, limit=None)

    assert index.stats(paths[2]) is None
    final = index.cache_stats()
    assert final.sources == 2
    assert final.hits == 1
    assert final.misses == 4
    assert final.evictions == 2


def test_sequential_source_churn_returns_to_retained_capacity(tmp_path: Path) -> None:
    paths = [tmp_path / f"events-{index}.jsonl" for index in range(256)]
    index = ReferenceEventOffsetIndex(max_sources=8)

    for path in paths:
        path.write_bytes(_record(1))
        index.read_page(path, from_seq=1, limit=None)

    stats = index.cache_stats()
    assert stats.sources == 8
    assert stats.pinned_sources == 0
    assert stats.total_pins == 0
    assert stats.misses == 256
    assert stats.evictions == 248
    assert stats.bypasses == 0
    assert index.stats(paths[0]) is None
    assert index.stats(paths[-1]) is not None


def test_zero_capacity_uses_authoritative_uncached_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"".join(_record(seq) for seq in range(1, 101)))
    index = ReferenceEventOffsetIndex(max_sources=0, record_stride=8)
    read_page = event_index_module.read_event_page_from_anchor
    mint_anchor = event_reader_module._mint_verified_anchor
    anchors: list[object] = []
    selectors: list[object] = []
    minted = 0

    def observe_anchor(*args, **kwargs):
        anchors.append(kwargs.get("anchor"))
        selectors.append(kwargs.get("anchor_selector"))
        return read_page(*args, **kwargs)

    def count_mint(*args, **kwargs):
        nonlocal minted
        minted += 1
        return mint_anchor(*args, **kwargs)

    monkeypatch.setattr(event_index_module, "read_event_page_from_anchor", observe_anchor)
    monkeypatch.setattr(event_reader_module, "_mint_verified_anchor", count_mint)

    pages = [
        index.read_page(path, from_seq=95, limit=2),
        index.read_page(path, from_seq=95, limit=2),
    ]

    expected = read_event_page(path, from_seq=95, limit=2)
    assert [page.to_page() for page in pages] == [expected, expected]
    assert all(page.anchor_batch is None for page in pages)
    assert anchors == [None, None]
    assert selectors == [None, None]
    assert minted == 0
    stats = index.cache_stats()
    assert stats.sources == 0
    assert stats.pinned_sources == 0
    assert stats.total_pins == 0
    assert stats.hits == 0
    assert stats.misses == 2
    assert stats.evictions == 0
    assert stats.bypasses == 2


def test_fully_pinned_capacity_bypasses_without_cross_source_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first_path.write_bytes(_record(1))
    second_path.write_bytes(_record(1))
    index = ReferenceEventOffsetIndex(max_sources=1)
    read_page = event_index_module.read_event_page_from_anchor
    entered = {first_path: Event(), second_path: Event()}
    release = {first_path: Event(), second_path: Event()}

    def block_source(events_path: Path, *args, **kwargs):
        entered[events_path].set()
        assert release[events_path].wait(timeout=5)
        return read_page(events_path, *args, **kwargs)

    monkeypatch.setattr(event_index_module, "read_event_page_from_anchor", block_source)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(index.read_page, first_path, from_seq=0, limit=1)
        assert entered[first_path].wait(timeout=5)
        second = executor.submit(index.read_page, second_path, from_seq=0, limit=1)
        assert entered[second_path].wait(timeout=5)
        saturated = index.cache_stats()
        assert saturated.sources == 1
        assert saturated.pinned_sources == 1
        assert saturated.total_pins == 1
        assert saturated.bypasses == 1

        release[second_path].set()
        second_page = second.result()
        assert second_page.to_page() == read_event_page(
            second_path, from_seq=0, limit=1
        )
        assert second_page.anchor_batch is None
        after_bypass = index.cache_stats()
        assert after_bypass.sources == 1
        assert after_bypass.evictions == 0
        assert after_bypass.bypasses == 1
        release[first_path].set()
        assert first.result().to_page() == read_event_page(
            first_path, from_seq=0, limit=1
        )

    assert index.stats(first_path) is not None
    assert index.stats(second_path) is None


def test_waiting_same_source_keeps_one_slot_during_other_source_bypass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first_path.write_bytes(b"".join(_record(seq) for seq in range(1, 101)))
    second_path.write_bytes(_record(1))
    index = ReferenceEventOffsetIndex(max_sources=1, record_stride=8)
    read_page = event_index_module.read_event_page_from_anchor
    lease_slot = index._lease_slot
    first_entered = Event()
    second_first_leased = Event()
    second_sources_entered = Event()
    release_first = Event()
    release_second_source = Event()
    first_lease_count = 0
    second_source_count = 0
    first_lease_lock = Lock()
    second_source_lock = Lock()
    observed: list[tuple[Path, object, object]] = []

    @contextmanager
    def observe_lease(events_path: Path, *args, **kwargs):
        nonlocal first_lease_count
        with lease_slot(events_path, *args, **kwargs) as slot:
            if events_path == first_path:
                with first_lease_lock:
                    first_lease_count += 1
                    if first_lease_count == 2:
                        second_first_leased.set()
            yield slot

    def block_reads(events_path: Path, *args, **kwargs):
        nonlocal second_source_count
        observed.append(
            (events_path, kwargs.get("anchor"), kwargs.get("anchor_selector"))
        )
        if events_path == first_path and not first_entered.is_set():
            first_entered.set()
            assert release_first.wait(timeout=5)
        elif events_path == second_path:
            with second_source_lock:
                second_source_count += 1
                if second_source_count == 2:
                    second_sources_entered.set()
            assert release_second_source.wait(timeout=5)
        return read_page(events_path, *args, **kwargs)

    monkeypatch.setattr(index, "_lease_slot", observe_lease)
    monkeypatch.setattr(event_index_module, "read_event_page_from_anchor", block_reads)

    with ThreadPoolExecutor(max_workers=4) as executor:
        first = executor.submit(index.read_page, first_path, from_seq=95, limit=2)
        assert first_entered.wait(timeout=5)
        waiting = executor.submit(index.read_page, first_path, from_seq=95, limit=2)
        assert second_first_leased.wait(timeout=5)
        bypassed = [
            executor.submit(index.read_page, second_path, from_seq=0, limit=1)
            for _ in range(2)
        ]
        assert second_sources_entered.wait(timeout=5)

        saturated = index.cache_stats()
        assert saturated.sources == 1
        assert saturated.pinned_sources == 1
        assert saturated.total_pins == 2
        assert saturated.hits == 1
        assert saturated.misses == 3
        assert saturated.bypasses == 2

        release_second_source.set()
        expected_bypass = read_event_page(second_path, from_seq=0, limit=1)
        bypassed_pages = [future.result() for future in bypassed]
        assert [page.to_page() for page in bypassed_pages] == [expected_bypass, expected_bypass]
        assert all(page.anchor_batch is None for page in bypassed_pages)
        release_first.set()
        first_pages = [first.result(), waiting.result()]

    expected = read_event_page(first_path, from_seq=95, limit=2)
    assert [page.to_page() for page in first_pages] == [expected, expected]
    first_anchors = [anchor for path, anchor, _selector in observed if path == first_path]
    assert first_anchors[0] is None
    assert first_anchors[1] is not None
    second_anchors = [anchor for path, anchor, _selector in observed if path == second_path]
    assert second_anchors == [None, None]
    second_selectors = [selector for path, _anchor, selector in observed if path == second_path]
    assert second_selectors == [None, None]
    first_stats = index.stats(first_path)
    assert first_stats is not None
    assert first_stats.from_zero_reads == 1
    assert first_stats.pages == 2
    assert index.stats(second_path) is None


def test_bypass_reader_exception_leaves_no_source_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_record(1))
    index = ReferenceEventOffsetIndex(max_sources=0)
    read_page = event_index_module.read_event_page_from_anchor

    def fail_read(*args, **kwargs):
        raise RuntimeError("injected read failure")

    monkeypatch.setattr(event_index_module, "read_event_page_from_anchor", fail_read)

    with pytest.raises(RuntimeError, match="injected read failure"):
        index.read_page(path, from_seq=0, limit=1)

    failed = index.cache_stats()
    assert failed.sources == 0
    assert failed.pinned_sources == 0
    assert failed.total_pins == 0
    assert failed.evictions == 0
    assert failed.bypasses == 1

    monkeypatch.setattr(event_index_module, "read_event_page_from_anchor", read_page)
    observed = index.read_page(path, from_seq=0, limit=1)

    assert observed.to_page() == read_event_page(path, from_seq=0, limit=1)
    recovered = index.cache_stats()
    assert recovered.sources == 0
    assert recovered.misses == 2
    assert recovered.evictions == 0
    assert recovered.bypasses == 2


def test_uncached_bypass_retries_one_source_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(_record(1))
    index = ReferenceEventOffsetIndex(max_sources=0)
    read_page = event_index_module.read_event_page_from_anchor
    calls = 0

    def change_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise EventLogChanged("injected snapshot change")
        return read_page(*args, **kwargs)

    monkeypatch.setattr(event_index_module, "read_event_page_from_anchor", change_once)

    observed = index.read_page(path, from_seq=0, limit=1)

    assert observed.to_page() == read_event_page(path, from_seq=0, limit=1)
    assert observed.anchor_batch is None
    assert calls == 2
    stats = index.cache_stats()
    assert stats.misses == 1
    assert stats.bypasses == 1


def test_sparse_anchor_density_is_bounded_by_record_stride(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"".join(_record(seq) for seq in range(1, 101)))
    index = ReferenceEventOffsetIndex(byte_stride=10**9, record_stride=10)

    index.read_page(path, from_seq=0, limit=None)
    stats = index.stats(path)

    assert stats is not None
    assert stats.indexed_through_seq == 100
    assert stats.anchor_count <= 12
