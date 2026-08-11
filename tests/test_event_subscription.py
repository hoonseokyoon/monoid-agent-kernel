from __future__ import annotations

import json

import pytest

from monoid_agent_kernel.core.event_subscription import (
    EventSequenceGap,
    EventSubscription,
    EventSubscriptionFrame,
    SequenceCursor,
)
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.public_view import (
    PREVIEW_BYTE_THRESHOLD,
    TRACE_PAYLOAD_BYTE_BUDGET,
    args_preview,
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


def test_each_event_frame_carries_its_own_next_sequence_cursor() -> None:
    events = [{"seq": seq, "type": "test.event"} for seq in range(1, 4)]
    frames = list(
        EventSubscription(
            _reader(events), read_lifecycle=lambda: {"terminal": True}
        ).frames()
    )

    event_frames = [frame for frame in frames if frame.kind == "event"]
    assert [frame.cursor for frame in event_frames] == [2, 3, 4]
    assert [frame.event_id for frame in event_frames] == ["1", "2", "3"]


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


def test_empty_terminal_stream_with_zero_watermark_ends_cleanly() -> None:
    frames = list(
        EventSubscription(
            _reader([]),
            read_lifecycle=lambda: {
                "terminal": True,
                "state": "failed",
                "last_event_seq": 0,
            },
        ).frames()
    )

    assert [frame.kind for frame in frames] == ["end"]


def test_a_budgeted_payload_stays_budgeted_through_this_frames_escaping() -> None:
    """The payload budget is charged in ``public_view``; this is the surface that spells it widest.

    ``to_sse`` escapes non-ASCII deliberately -- U+2028, U+2029 and U+0085 survive an unescaped
    dump and split a frame mid-string for ``str.splitlines`` readers -- so a budget counted in
    UTF-8 bytes would let a Korean payload arrive here at roughly twice its charge, and a
    two-byte script at nearly three times. Pinned from this side as well as from the builder's,
    because the two files can be edited apart: a future frame writer that stops escaping is fine,
    but a payload charge that stops covering escaping is not, and only this test fails then.

    The frame's own overhead -- ``id:``/``data:`` framing and the event envelope -- is measured
    rather than guessed, by spelling the same frame with an empty payload.
    """
    policy = PermissionPolicy()
    value = "가" * (PREVIEW_BYTE_THRESHOLD // 3)
    published = args_preview({f"arg{index:05d}": value for index in range(4000)}, policy)

    def frame_for(preview: dict[str, object]) -> EventSubscriptionFrame:
        return EventSubscriptionFrame(
            kind="event",
            cursor=2,
            event={
                "schema_version": "1.0",
                "seq": 1,
                "type": "tool.call.started",
                "ts": "2026-08-11T00:00:00Z",
                "run_id": "0" * 32,
                "data": {"args_preview": preview},
            },
        )

    overhead = len(frame_for({}).to_sse())
    payload_bytes_on_the_wire = len(frame_for(published).to_sse()) - overhead

    assert payload_bytes_on_the_wire <= TRACE_PAYLOAD_BYTE_BUDGET
    assert "truncated_keys" in published, "the fixture must actually reach the budget"
