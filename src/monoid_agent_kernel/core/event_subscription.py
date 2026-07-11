"""Cursor-correct subscriptions over append-only run event streams."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

EventPageReader = Callable[[int, int | None], Mapping[str, Any]]
EventLifecycleReader = Callable[[], Mapping[str, Any]]
SubscriptionKind = Literal["event", "heartbeat", "end"]


class EventSequenceGap(ValueError):
    """A resumed subscription observed an event after its required next sequence."""


@dataclass
class SequenceCursor:
    """The next sequence an inclusive event-page reader must return."""

    next_seq: int = 0
    anchored: bool = False

    def __post_init__(self) -> None:
        if self.next_seq < 0:
            raise ValueError("event cursor must be non-negative")
        if self.next_seq > 0:
            self.anchored = True

    @classmethod
    def resolve(cls, *, from_seq: int = 0, last_event_id: str | None = None) -> SequenceCursor:
        """Prefer an SSE ``Last-Event-ID`` over the initial query cursor."""

        if last_event_id is None or not last_event_id.strip():
            return cls(from_seq)
        try:
            event_id = int(last_event_id)
        except ValueError as exc:
            raise ValueError("Last-Event-ID must be a non-negative sequence") from exc
        if event_id < 0:
            raise ValueError("Last-Event-ID must be a non-negative sequence")
        return cls(event_id + 1, anchored=True)

    def accept(self, sequence: int) -> bool:
        """Advance on one new event; return false for replayed events."""

        if sequence < 0:
            raise ValueError("event sequence must be non-negative")
        if sequence < self.next_seq:
            return False
        if not self.anchored and self.next_seq == 0 and sequence > 1:
            raise EventSequenceGap(f"event stream starts after sequence 1; observed {sequence}")
        if self.anchored and sequence > self.next_seq:
            raise EventSequenceGap(
                f"event stream skipped required sequence {self.next_seq}; observed {sequence}"
            )
        self.next_seq = sequence + 1
        self.anchored = True
        return True


@dataclass(frozen=True)
class EventSubscriptionFrame:
    kind: SubscriptionKind
    cursor: int
    event: Mapping[str, Any] | None = None
    lifecycle: Mapping[str, Any] | None = None
    comment: str = "keep-alive"

    @property
    def event_id(self) -> str:
        if self.kind != "event" or self.event is None:
            return ""
        return str(int(self.event["seq"]))

    def to_sse(self) -> bytes:
        """Serialize the frame using SSE ids for replay-safe reconnects."""

        if self.kind == "heartbeat":
            return f": {self.comment}\n\n".encode("utf-8")
        if self.kind == "end":
            payload = json.dumps(
                dict(self.lifecycle or {}), ensure_ascii=False, separators=(",", ":")
            )
            return f"event: end\ndata: {payload}\n\n".encode("utf-8")
        assert self.event is not None
        payload = json.dumps(dict(self.event), ensure_ascii=False, separators=(",", ":"))
        return f"id: {self.event_id}\ndata: {payload}\n\n".encode("utf-8")


class EventSubscription:
    """Reusable page polling and blocking frame iteration for one authorized stream."""

    def __init__(
        self,
        read_page: EventPageReader,
        *,
        cursor: SequenceCursor | None = None,
        read_lifecycle: EventLifecycleReader | None = None,
    ) -> None:
        self._read_page = read_page
        self._read_lifecycle = read_lifecycle
        self.cursor = cursor or SequenceCursor()

    def poll(self, *, limit: int | None = None) -> dict[str, Any]:
        """Read one page, suppressing replays and advancing the shared cursor."""

        payload = self._read_page(self.cursor.next_seq, limit)
        raw_events = payload.get("events", ())
        if not isinstance(raw_events, (list, tuple)):
            raise ValueError("event page events must be a list")
        events: list[dict[str, Any]] = []
        for raw_event in raw_events:
            if not isinstance(raw_event, Mapping):
                raise ValueError("event page item must be an object")
            event = dict(raw_event)
            try:
                sequence = int(event["seq"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("event page item requires an integer seq") from exc
            if self.cursor.accept(sequence):
                events.append(event)
        page = dict(payload)
        page.update(
            events=events,
            next_seq=self.cursor.next_seq,
            has_more=bool(payload.get("has_more")),
        )
        return page

    def frames(
        self,
        *,
        page_limit: int = 500,
        poll_interval_s: float = 0.25,
        heartbeat_interval_s: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Iterator[EventSubscriptionFrame]:
        """Yield events, heartbeat comments, then one end frame after a final drain."""

        if page_limit < 1:
            raise ValueError("page_limit must be positive")
        if poll_interval_s <= 0 or heartbeat_interval_s <= 0:
            raise ValueError("subscription intervals must be positive")
        last_output = clock()
        while True:
            page = self.poll(limit=page_limit)
            if page["events"]:
                for event in page["events"]:
                    last_output = clock()
                    yield EventSubscriptionFrame(
                        kind="event", cursor=int(event["seq"]) + 1, event=event
                    )
                continue
            lifecycle = dict(self._read_lifecycle()) if self._read_lifecycle is not None else {}
            if lifecycle.get("terminal"):
                # Status and the event log are separate projections. Re-read after observing
                # terminal state so a final event published just before status is not lost.
                final_page = self.poll(limit=page_limit)
                if final_page["events"]:
                    for event in final_page["events"]:
                        yield EventSubscriptionFrame(
                            kind="event", cursor=int(event["seq"]) + 1, event=event
                        )
                    continue
                watermark = _lifecycle_last_event_seq(lifecycle)
                if watermark > 0 and watermark >= self.cursor.next_seq:
                    raise EventSequenceGap(
                        f"terminal lifecycle advertises sequence {watermark}, "
                        f"but cursor is waiting for {self.cursor.next_seq}"
                    )
                yield EventSubscriptionFrame(
                    kind="end", cursor=self.cursor.next_seq, lifecycle=lifecycle
                )
                return
            now = clock()
            if now - last_output >= heartbeat_interval_s:
                last_output = now
                yield EventSubscriptionFrame(kind="heartbeat", cursor=self.cursor.next_seq)
            sleep(poll_interval_s)


def _lifecycle_last_event_seq(lifecycle: Mapping[str, Any]) -> int:
    status_file = lifecycle.get("status_file")
    durable = status_file if isinstance(status_file, Mapping) else {}
    return max(
        int(lifecycle.get("last_event_seq") or 0),
        int(durable.get("last_event_seq") or 0),
    )
