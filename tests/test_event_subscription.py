from __future__ import annotations

import json

import pytest

from monoid_agent_kernel.core.event_subscription import (
    EventSequenceGap,
    EventSubscription,
    SequenceCursor,
)


def _reader(events: list[dict[str, object]]):
    def read(from_seq: int, limit: int | None) -> dict[str, object]:
        page = [event for event in events if int(event["seq"]) >= from_seq]
        selected = page if limit is None else page[:limit]
        return {"events": selected, "has_more": len(page) > len(selected)}

    return read


def test_sequence_cursor_prefers_last_event_id_and_rejects_gaps() -> None:
    cursor = SequenceCursor.resolve(from_seq=1, last_event_id="4")
    assert cursor.next_seq == 5
    assert cursor.accept(4) is False
    assert cursor.accept(5) is True
    with pytest.raises(EventSequenceGap, match="required sequence 6"):
        cursor.accept(7)
    with pytest.raises(ValueError, match="Last-Event-ID"):
        SequenceCursor.resolve(last_event_id="opaque")


def test_reconnect_presents_each_event_once_with_sse_ids() -> None:
    events = [{"seq": seq, "type": "test.event"} for seq in range(1, 5)]
    first = EventSubscription(_reader(events))
    first_page = first.poll(limit=2)
    assert [event["seq"] for event in first_page["events"]] == [1, 2]
    assert first_page["next_seq"] == 3

    resumed = EventSubscription(_reader(events), cursor=SequenceCursor.resolve(last_event_id="2"))
    resumed_page = resumed.poll()
    assert [event["seq"] for event in resumed_page["events"]] == [3, 4]
    frame = next(
        EventSubscription(
            _reader(events),
            cursor=SequenceCursor.resolve(last_event_id="3"),
            read_lifecycle=lambda: {"terminal": True},
        ).frames()
    )
    assert frame.event_id == "4"
    assert frame.to_sse().startswith(b"id: 4\n")


def test_subscription_emits_heartbeat_comment_for_idle_live_stream() -> None:
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    subscription = EventSubscription(_reader([]), read_lifecycle=lambda: {"terminal": False})
    frame = next(
        subscription.frames(
            poll_interval_s=0.5,
            heartbeat_interval_s=1.0,
            clock=lambda: now[0],
            sleep=sleep,
        )
    )
    assert frame.kind == "heartbeat"
    assert frame.to_sse() == b": keep-alive\n\n"


def test_terminal_subscription_performs_final_event_drain_before_end() -> None:
    calls = 0

    def read(from_seq: int, limit: int | None) -> dict[str, object]:
        nonlocal calls
        del from_seq, limit
        calls += 1
        if calls == 2:
            return {"events": [{"seq": 1, "type": "run.finished"}]}
        return {"events": []}

    frames = list(
        EventSubscription(
            read, read_lifecycle=lambda: {"terminal": True, "state": "completed"}
        ).frames()
    )
    assert [frame.kind for frame in frames] == ["event", "end"]
    assert frames[0].event_id == "1"
    assert json.loads(frames[-1].to_sse().split(b"data: ", 1)[1]) == {
        "terminal": True,
        "state": "completed",
    }
