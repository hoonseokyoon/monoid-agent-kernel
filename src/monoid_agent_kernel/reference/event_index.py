"""Process-local sparse byte-offset index for Reference event readers."""

from __future__ import annotations

import os
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, RLock

from monoid_agent_kernel.core._event_log import EventLogChanged, EventLogRecord
from monoid_agent_kernel.reference.event_reader import (
    EventAnchorUnavailable,
    EventPageRead,
    EventReadAnchor,
    EventReadAnchorBatch,
    event_read_anchor_extends_source,
    read_event_page_from_anchor,
)

_DEFAULT_BYTE_STRIDE = 64 * 1024
_DEFAULT_RECORD_STRIDE = 256


@dataclass(frozen=True)
class EventIndexStats:
    """Observable derived-index state for deterministic acceptance tests."""

    anchor_count: int
    indexed_through_seq: int
    indexed_through_offset: int
    from_zero_reads: int
    invalidations: int
    pages: int
    last_records_examined: int
    last_source_bytes_read: int


class _EventIndexSlot:
    def __init__(self) -> None:
        self.lock = RLock()
        self.anchors: list[EventReadAnchor] = []
        self.tail_anchor: EventReadAnchor | None = None
        self.records_since_sparse = 0
        self.from_zero_reads = 0
        self.invalidations = 0
        self.pages = 0
        self.last_records_examined = 0
        self.last_source_bytes_read = 0


class _SparseAnchorSelector:
    """Select bounded candidates without publishing state during an unverified scan."""

    def __init__(
        self,
        slot: _EventIndexSlot,
        *,
        byte_stride: int,
        record_stride: int,
        cold_sampling: bool,
    ) -> None:
        tail = slot.tail_anchor
        self._indexed_through_seq = tail.seq if tail is not None else 0
        self._last_sparse_offset = (
            slot.anchors[-1].byte_offset if slot.anchors else None
        )
        self.extension_records_since_sparse = slot.records_since_sparse
        self._byte_stride = byte_stride
        self._record_stride = record_stride
        self._cold_sampling = cold_sampling
        self._cold_last_sparse_offset: int | None = None
        self._cold_records_since_sparse = 0
        self.records_since_selected = 0

    def __call__(self, record: EventLogRecord) -> bool:
        cold_selected = self._select_cold(record) if self._cold_sampling else False
        extension_selected = False
        if record.seq > self._indexed_through_seq:
            if self._last_sparse_offset is None:
                extension_selected = True
            else:
                self.extension_records_since_sparse += 1
                extension_selected = cold_selected or (
                    record.byte_offset - self._last_sparse_offset >= self._byte_stride
                    or self.extension_records_since_sparse >= self._record_stride
                )
            if extension_selected:
                self._last_sparse_offset = record.byte_offset
                self.extension_records_since_sparse = 0

        selected = cold_selected or extension_selected
        if selected:
            self.records_since_selected = 0
        else:
            self.records_since_selected += 1
        return selected

    def _select_cold(self, record: EventLogRecord) -> bool:
        if self._cold_last_sparse_offset is None:
            selected = True
        else:
            self._cold_records_since_sparse += 1
            selected = (
                record.byte_offset - self._cold_last_sparse_offset >= self._byte_stride
                or self._cold_records_since_sparse >= self._record_stride
            )
        if selected:
            self._cold_last_sparse_offset = record.byte_offset
            self._cold_records_since_sparse = 0
        return selected


class ReferenceEventOffsetIndex:
    """Share verified sparse offsets across Reference page reads in one process."""

    def __init__(
        self,
        *,
        byte_stride: int = _DEFAULT_BYTE_STRIDE,
        record_stride: int = _DEFAULT_RECORD_STRIDE,
    ) -> None:
        if isinstance(byte_stride, bool) or not isinstance(byte_stride, int) or byte_stride < 1:
            raise ValueError("byte_stride must be a positive integer")
        if (
            isinstance(record_stride, bool)
            or not isinstance(record_stride, int)
            or record_stride < 1
        ):
            raise ValueError("record_stride must be a positive integer")
        self._byte_stride = byte_stride
        self._record_stride = record_stride
        self._slots_lock = Lock()
        self._slots: dict[str, _EventIndexSlot] = {}

    def read_page(
        self,
        events_path: Path,
        *,
        from_seq: int,
        limit: int | None,
    ) -> EventPageRead:
        """Read a page through the nearest verified offset, rebuilding once if stale."""
        _validate_request(from_seq=from_seq, limit=limit)
        slot = self._slot(events_path)
        with slot.lock:
            anchor = _select_anchor(slot, from_seq)
            selector = self._selector(slot, cold_sampling=anchor is None)
            try:
                page = read_event_page_from_anchor(
                    events_path,
                    from_seq=from_seq,
                    limit=limit,
                    anchor=anchor,
                    anchor_selector=selector,
                )
            except (EventLogChanged, EventAnchorUnavailable):
                _clear_slot(slot, count_invalidation=True)
                anchor = None
                selector = self._selector(slot, cold_sampling=True)
                page = read_event_page_from_anchor(
                    events_path,
                    from_seq=from_seq,
                    limit=limit,
                    anchor_selector=selector,
                )
            if anchor is None:
                slot.from_zero_reads += 1
            self._publish(
                slot,
                page.anchor_batch,
                used_anchor=anchor,
                selector=selector,
            )
            slot.pages += 1
            slot.last_records_examined = page.records_examined
            slot.last_source_bytes_read = page.source_bytes_read
            return page

    def invalidate(self, events_path: Path) -> None:
        """Discard one derived source generation without touching authoritative bytes."""
        key = _source_key(events_path)
        with self._slots_lock:
            slot = self._slots.get(key)
        if slot is None:
            return
        with slot.lock:
            _clear_slot(slot, count_invalidation=True)

    def stats(self, events_path: Path) -> EventIndexStats | None:
        """Return a stable snapshot of one source's process-local index metrics."""
        key = _source_key(events_path)
        with self._slots_lock:
            slot = self._slots.get(key)
        if slot is None:
            return None
        with slot.lock:
            tail = slot.tail_anchor
            tail_is_sparse = tail is not None and any(anchor is tail for anchor in slot.anchors)
            return EventIndexStats(
                anchor_count=len(slot.anchors) + int(tail is not None and not tail_is_sparse),
                indexed_through_seq=tail.seq if tail is not None else 0,
                indexed_through_offset=tail.next_byte_offset if tail is not None else 0,
                from_zero_reads=slot.from_zero_reads,
                invalidations=slot.invalidations,
                pages=slot.pages,
                last_records_examined=slot.last_records_examined,
                last_source_bytes_read=slot.last_source_bytes_read,
            )

    def _slot(self, events_path: Path) -> _EventIndexSlot:
        key = _source_key(events_path)
        with self._slots_lock:
            slot = self._slots.get(key)
            if slot is None:
                slot = _EventIndexSlot()
                self._slots[key] = slot
            return slot

    def _selector(
        self,
        slot: _EventIndexSlot,
        *,
        cold_sampling: bool,
    ) -> _SparseAnchorSelector:
        return _SparseAnchorSelector(
            slot,
            byte_stride=self._byte_stride,
            record_stride=self._record_stride,
            cold_sampling=cold_sampling,
        )

    def _publish(
        self,
        slot: _EventIndexSlot,
        batch: EventReadAnchorBatch | None,
        *,
        used_anchor: EventReadAnchor | None,
        selector: _SparseAnchorSelector,
    ) -> None:
        if batch is None:
            raise RuntimeError("indexed event read did not return anchor observations")
        witness = batch.sparse[0] if batch.sparse else batch.tail
        if witness is None:
            if used_anchor is None and slot.tail_anchor is not None:
                _clear_slot(slot, count_invalidation=True)
            return
        source_changed = slot.tail_anchor is not None and not event_read_anchor_extends_source(
            slot.tail_anchor, witness
        )
        if source_changed:
            _clear_slot(slot, count_invalidation=True)
        indexed_through_seq = slot.tail_anchor.seq if slot.tail_anchor is not None else 0
        for anchor in batch.sparse:
            if anchor.seq <= indexed_through_seq:
                continue
            slot.anchors.append(anchor)
        if batch.tail is not None and (
            slot.tail_anchor is None or batch.tail.seq > slot.tail_anchor.seq
        ):
            slot.tail_anchor = batch.tail
            slot.records_since_sparse = (
                selector.records_since_selected
                if source_changed
                else selector.extension_records_since_sparse
            )


def _select_anchor(slot: _EventIndexSlot, from_seq: int) -> EventReadAnchor | None:
    index = bisect_right(slot.anchors, from_seq, key=lambda anchor: anchor.seq) - 1
    selected = slot.anchors[index] if index >= 0 else None
    tail = slot.tail_anchor
    if tail is not None and tail.seq <= from_seq and (
        selected is None or tail.seq > selected.seq
    ):
        return tail
    return selected


def _clear_slot(slot: _EventIndexSlot, *, count_invalidation: bool) -> None:
    slot.anchors.clear()
    slot.tail_anchor = None
    slot.records_since_sparse = 0
    if count_invalidation:
        slot.invalidations += 1


def _validate_request(*, from_seq: int, limit: int | None) -> None:
    if isinstance(from_seq, bool) or not isinstance(from_seq, int) or from_seq < 0:
        raise ValueError("from_seq must be a non-negative integer")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("limit must be a positive integer")


def _source_key(events_path: Path) -> str:
    return os.path.normcase(str(events_path.resolve()))
