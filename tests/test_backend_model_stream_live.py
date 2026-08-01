from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pytest

from support.runtime import runtime_config

from monoid_agent_kernel.core.model_content import (
    ModelContentStore,
    flush_active_model_content,
    read_model_content,
)
from monoid_agent_kernel.core.model_stream import (
    ModelStreamContext,
    ModelStreamDelta,
    ModelStreamObserver,
    ModelStreamOutcome,
)
from monoid_agent_kernel.errors import NativeAgentError, PermissionDenied
from monoid_agent_kernel.reference.backend.http import create_backend_server
from monoid_agent_kernel.reference.backend.model_stream import (
    MODEL_STREAM_LIVE_SCHEMA_VERSION,
    LiveModelStreamBroker,
    LiveModelStreamCursor,
    LiveModelStreamFrame,
)
from monoid_agent_kernel.reference.backend.service import BackendRunRequest


def _context(
    *,
    root_run_id: str = "root-1",
    run_id: str = "root-1",
    stream_id: str = "stream-1",
) -> ModelStreamContext:
    return ModelStreamContext(
        run_id=run_id,
        root_run_id=root_run_id,
        turn_id="turn-1",
        stream_id=stream_id,
        step=1,
        provider="provider",
        model="model",
        started_at="2026-08-01T00:00:00Z",
    )


def test_broker_multiplexes_descendant_lifecycle_and_content_by_root() -> None:
    broker = LiveModelStreamBroker(generation="test-generation")
    observer = broker.observer_factory("root-1")()
    assert isinstance(observer, ModelStreamObserver)
    context = _context(run_id="root-1.sub.research", stream_id="stream-child")
    writer = observer.open(context)
    writer.push(ModelStreamDelta(channel="reasoning", text="think"))
    writer.push(ModelStreamDelta(channel="output", text="answer"))
    writer.close(
        ModelStreamOutcome(
            status="interrupted",
            final_text="answer",
            usage={"output_tokens": 2, "ratio": float("nan")},
            error_code="turn_interrupted",
        )
    )

    # A first-time subscriber catches a call that started before its connection.
    frames = broker.subscribe("root-1").poll()

    assert [frame.kind for frame in frames] == ["opened", "delta", "delta", "closed"]
    assert [frame.sequence for frame in frames] == [1, 2, 3, 4]
    assert all(frame.root_run_id == "root-1" for frame in frames)
    assert all(frame.run_id == "root-1.sub.research" for frame in frames)
    assert frames[1].channel == "reasoning"
    assert frames[1].text == "think"
    assert (frames[1].start_offset, frames[1].end_offset) == (0, 5)
    assert frames[2].channel == "output"
    assert (frames[2].start_offset, frames[2].end_offset) == (0, 6)
    assert frames[3].status == "interrupted"
    assert frames[3].partial is True
    assert frames[3].final_text == "answer"
    assert frames[3].usage == {"output_tokens": 2, "ratio": None}
    assert frames[3].error_code == "turn_interrupted"
    payload = frames[3].to_json()
    assert payload["schema_version"] == MODEL_STREAM_LIVE_SCHEMA_VERSION
    assert payload["cursor"] == "test-generation:4"
    sse = frames[3].to_sse().decode("utf-8")
    assert sse.startswith("id: test-generation:4\nevent: model-stream\ndata: ")
    assert json.loads(sse.split("data: ", 1)[1]) == payload


def test_broker_isolates_root_rings() -> None:
    broker = LiveModelStreamBroker(generation="roots")
    first = broker.observer("root-a").open(_context(root_run_id="root-a", run_id="root-a"))
    second = broker.observer("root-b").open(
        _context(root_run_id="root-b", run_id="root-b.sub.child", stream_id="stream-b")
    )
    first.push(ModelStreamDelta(channel="output", text="a"))
    second.push(ModelStreamDelta(channel="output", text="b"))

    root_a = broker.subscribe("root-a", after_cursor="roots:0").poll()
    root_b = broker.subscribe("root-b", after_cursor="roots:0").poll()

    assert [frame.text for frame in root_a if frame.kind == "delta"] == ["a"]
    assert [frame.text for frame in root_b if frame.kind == "delta"] == ["b"]
    assert all(frame.root_run_id == "root-a" for frame in root_a)
    assert all(frame.root_run_id == "root-b" for frame in root_b)


def test_bound_observer_rejects_cross_root_and_invalid_lineage() -> None:
    broker = LiveModelStreamBroker(generation="lineage")
    observer = broker.observer("root-a")

    with pytest.raises(ValueError, match="bound root"):
        observer.open(_context(root_run_id="root-b", run_id="root-b"))
    with pytest.raises(ValueError, match="outside its root lineage"):
        broker.observer().open(_context(root_run_id="root-a", run_id="unrelated"))
    with pytest.raises(ValueError, match="invalid"):
        broker.subscribe("../root")

    assert broker.stats("root-a").latest_sequence == 0
    assert broker.stats("root-b").latest_sequence == 0


def test_ring_enforces_frame_and_byte_budgets_and_signals_cursor_gap() -> None:
    broker = LiveModelStreamBroker(generation="bounded", max_frames=3, max_bytes=900)
    writer = broker.observer("root-1").open(_context())
    for text in ("one", "two", "three", "four"):
        writer.push(ModelStreamDelta(channel="output", text=text))
    writer.close(ModelStreamOutcome(status="completed", final_text="x" * 5_000))

    stats = broker.stats("root-1")
    assert stats.frame_count <= 3
    assert stats.byte_count <= 900
    assert stats.latest_sequence == 6
    frames = broker.subscribe("root-1", after_cursor="bounded:0").poll()
    assert frames[0].kind == "reset"
    assert frames[0].reason == "cursor_gap"
    assert frames[0].latest_cursor == "bounded:6"
    assert [frame.sequence for frame in frames[1:]] == list(
        range(frames[1].sequence, stats.latest_sequence + 1)
    )
    assert frames[-1].kind == "closed"
    assert frames[-1].status == "completed"
    assert frames[-1].content_omitted is True
    assert frames[-1].final_text is None


def test_oversized_delta_creates_explicit_internal_gap() -> None:
    broker = LiveModelStreamBroker(generation="oversized", max_frames=20, max_bytes=700)
    writer = broker.observer("root-1").open(_context())
    writer.push(ModelStreamDelta(channel="output", text="x" * 5_000))

    live_frames = broker.subscribe("root-1", after_cursor="oversized:1").poll()
    assert len(live_frames) == 1
    assert live_frames[0].kind == "reset"
    assert live_frames[0].reason == "cursor_gap"
    assert live_frames[0].cursor == LiveModelStreamCursor("oversized", 2)

    writer.close(ModelStreamOutcome(status="completed", final_text="ok"))

    frames = broker.subscribe("root-1", after_cursor="oversized:0").poll()

    assert [frame.kind for frame in frames] == ["opened", "reset", "closed"]
    assert frames[1].reason == "cursor_gap"
    assert frames[1].cursor == LiveModelStreamCursor("oversized", 2)
    assert frames[2].sequence == 3
    assert broker.stats("root-1").byte_count <= 700


def test_oversized_live_gap_can_hydrate_the_exact_private_sidecar_prefix(
    tmp_path: Path,
) -> None:
    """The reset cursor cannot advance beyond a still-buffered private tail."""

    path = tmp_path / "model-content.jsonl"
    store = ModelContentStore(path, run_id="root-1", batch_interval_s=60.0)
    broker = LiveModelStreamBroker(generation="oversized", max_frames=20, max_bytes=700)
    context = _context()
    sidecar_writer = store.open(context)
    live_writer = broker.observer("root-1").open(context)
    expected = "p" + ("x" * 600_001)

    try:
        # AgentLoop delivers each provider delta to the private sidecar before the live broker.
        for delta in (
            ModelStreamDelta(channel="output", text="p"),
            ModelStreamDelta(channel="output", text="x" * 600_001),
        ):
            sidecar_writer.push(delta)
            live_writer.push(delta)

        frames = broker.subscribe("root-1", after_cursor="oversized:1").poll()
        assert [frame.kind for frame in frames] == ["delta", "reset"]
        assert frames[-1].cursor == LiveModelStreamCursor("oversized", 3)
        assert read_model_content(path).snapshots[0].output_text != expected

        assert flush_active_model_content(tmp_path) == 1
        snapshot = read_model_content(path).snapshots[0]
        assert snapshot.output_text == expected
        assert len(snapshot.output_text.encode("utf-8")) == len(expected.encode("utf-8"))
    finally:
        store.close()


def test_generation_change_and_future_cursor_return_reset_frames() -> None:
    broker = LiveModelStreamBroker(generation="new")
    broker.observer("root-1").open(_context())

    changed = broker.subscribe("root-1", after_cursor="old:9").poll(limit=1)
    ahead = broker.subscribe("root-1", after_cursor="new:99").poll(limit=1)

    assert len(changed) == 1
    assert changed[0].kind == "reset"
    assert changed[0].reason == "generation_changed"
    assert changed[0].latest_cursor == "new:1"
    assert len(ahead) == 1
    assert ahead[0].reason == "cursor_ahead"
    assert ahead[0].cursor == LiveModelStreamCursor("new", 1)


def test_explicit_cursor_replays_only_later_frames_and_limit_advances_cursor() -> None:
    broker = LiveModelStreamBroker(generation="resume")
    writer = broker.observer("root-1").open(_context())
    writer.push(ModelStreamDelta(channel="output", text="first"))
    writer.push(ModelStreamDelta(channel="output", text="second"))
    subscription = broker.subscribe("root-1", after_cursor="resume:1")

    first_page = subscription.poll(limit=1)
    second_page = subscription.poll()

    assert [frame.text for frame in first_page] == ["first"]
    assert subscription.cursor == "resume:3"
    assert [frame.text for frame in second_page] == ["second"]


def test_closing_one_waiting_subscription_never_stops_writer_or_peer() -> None:
    broker = LiveModelStreamBroker(generation="disconnect")
    observer = broker.observer("root-1")
    writer = observer.open(_context())
    disconnected = broker.subscribe("root-1", after_cursor="disconnect:1")
    peer = broker.subscribe("root-1", after_cursor="disconnect:1")
    completed = threading.Event()

    def wait_for_frame() -> None:
        assert disconnected.poll(timeout_s=2) == ()
        completed.set()

    thread = threading.Thread(target=wait_for_frame)
    thread.start()
    time.sleep(0.02)
    disconnected.close()
    assert completed.wait(1)
    writer.push(ModelStreamDelta(channel="output", text="still running"))

    peer_frames = peer.poll(timeout_s=1)
    thread.join(timeout=1)
    assert [frame.text for frame in peer_frames] == ["still running"]
    assert broker.stats("root-1").latest_sequence == 2


def test_async_poll_wakes_when_sync_observer_publishes() -> None:
    async def exercise() -> None:
        broker = LiveModelStreamBroker(generation="async")
        writer = broker.observer("root-1").open(_context())
        subscription = broker.subscribe("root-1", after_cursor="async:1")
        pending = asyncio.create_task(subscription.apoll(timeout_s=1))
        await asyncio.sleep(0.02)
        writer.push(ModelStreamDelta(channel="output", text="wake"))
        frames = await pending
        assert [frame.text for frame in frames] == ["wake"]
        subscription.close()

    asyncio.run(exercise())


def test_broker_close_wakes_waiters_and_makes_existing_and_future_writers_inert() -> None:
    broker = LiveModelStreamBroker(generation="shutdown")
    writer = broker.observer("root-1").open(_context())
    subscription = broker.subscribe("root-1", after_cursor="shutdown:1")
    completed = threading.Event()

    def wait_for_shutdown() -> None:
        assert subscription.poll(timeout_s=30) == ()
        completed.set()

    thread = threading.Thread(target=wait_for_shutdown)
    thread.start()
    time.sleep(0.02)
    broker.close()
    broker.close()

    assert completed.wait(1)
    thread.join(timeout=1)
    assert broker.closed is True
    assert subscription.closed is True
    assert broker.buffered_root_count == 0
    writer.push(ModelStreamDelta(channel="output", text="late"))
    late_writer = broker.observer("root-2").open(_context(root_run_id="root-2", run_id="root-2"))
    late_writer.push(ModelStreamDelta(channel="output", text="later"))
    assert broker.stats("root-1").latest_sequence == 0
    assert broker.stats("root-2").latest_sequence == 0
    assert broker.subscribe("root-2").closed is True


def test_root_ring_lru_bounds_long_lived_broker_memory_and_preserves_gap_watermark() -> None:
    broker = LiveModelStreamBroker(generation="lru", max_roots=2)
    for root in ("root-a", "root-b", "root-c"):
        broker.observer(root).open(_context(root_run_id=root, run_id=root, stream_id=f"{root}-s"))

    assert broker.buffered_root_count == 2
    evicted = broker.stats("root-a")
    assert evicted.frame_count == 0
    assert evicted.latest_sequence == 0
    assert broker.buffered_root_count == 2

    frames = broker.subscribe("root-a", after_cursor="lru:0").poll()
    assert len(frames) == 1
    assert frames[0].kind == "reset"
    assert frames[0].reason == "generation_changed"
    assert frames[0].cursor == LiveModelStreamCursor("lru.4", 0)
    assert frames[0].oldest_available_cursor is None


def test_backend_subscription_requires_root_token_and_enabled_broker(
    backend_factory: Any,
) -> None:
    workspace = backend_factory.workspace()
    broker = LiveModelStreamBroker(generation="service")
    backend = backend_factory.create(workspace=workspace, model_stream_broker=broker)
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant",
            user_id="user",
            workspace_root=workspace,
            instruction="finish",
            runtime_config=runtime_config("run.finish"),
        )
    )
    assert backend.wait_for_run(submission.run_id, timeout_s=10).value == "completed"

    with pytest.raises(PermissionDenied):
        backend.subscribe_model_stream(submission.run_id, "bad-token")
    subscription = backend.subscribe_model_stream(submission.run_id, submission.run_token)
    assert [frame.kind for frame in subscription.poll()] == ["opened", "closed"]
    subscription.close()

    disabled = backend_factory.create(workspace=workspace)
    disabled_submission = disabled.submit_run(
        BackendRunRequest(
            tenant_id="tenant",
            user_id="user",
            workspace_root=workspace,
            instruction="finish",
            runtime_config=runtime_config("run.finish"),
        )
    )
    assert disabled.wait_for_run(disabled_submission.run_id, timeout_s=10).value == "completed"
    with pytest.raises(NativeAgentError) as exc_info:
        disabled.subscribe_model_stream(
            disabled_submission.run_id,
            disabled_submission.run_token,
        )
    assert exc_info.value.error_code == "model_stream_unavailable"


@pytest.mark.integration
def test_backend_http_model_stream_prefers_last_event_id_and_disconnect_is_passive(
    backend_factory: Any,
) -> None:
    workspace = backend_factory.workspace()
    broker = LiveModelStreamBroker(generation="http")
    backend = backend_factory.create(workspace=workspace, model_stream_broker=broker)
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant",
            user_id="user",
            workspace_root=workspace,
            instruction="finish",
            runtime_config=runtime_config("run.finish"),
        )
    )
    assert backend.wait_for_run(submission.run_id, timeout_s=10).value == "completed"
    assert broker.stats(submission.run_id).latest_sequence == 2
    cancellation_requests: list[str] = []
    backend.request_stream_cancel = cancellation_requests.append  # type: ignore[method-assign]

    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    response = None
    try:
        request = Request(
            f"{base_url}/v1/runs/{submission.run_id}/model-stream?cursor=http:0",
            headers={
                "Authorization": f"Bearer {submission.run_token}",
                "Accept": "text/event-stream",
                "Last-Event-ID": "http:1",
            },
        )
        response = urlopen(request, timeout=5)
        block: list[str] = []
        payload: dict[str, Any] | None = None
        while payload is None:
            line = response.readline().decode("utf-8").rstrip("\r\n")
            if line:
                block.append(line)
                continue
            data_line = next((item for item in block if item.startswith("data: ")), None)
            if data_line is not None:
                payload = json.loads(data_line.removeprefix("data: "))
            block = []
        assert payload["cursor"] == "http:2"
        assert payload["kind"] == "closed"

        response.close()
        response = None
        # Publishing after the client disconnects remains valid and wakes the handler so it can
        # observe the broken pipe. No execution-control callback is reachable from this route.
        writer = broker.observer(submission.run_id).open(
            _context(
                root_run_id=submission.run_id,
                run_id=submission.run_id,
                stream_id="post-disconnect",
            )
        )
        writer.push(ModelStreamDelta(channel="output", text="still live"))
        assert broker.stats(submission.run_id).latest_sequence == 4
        assert cancellation_requests == []
    finally:
        if response is not None:
            response.close()
        broker.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
    assert not server_thread.is_alive()
    assert cancellation_requests == []


def test_delta_offsets_are_utf8_bytes_and_independent_per_channel() -> None:
    broker = LiveModelStreamBroker(generation="offsets")
    writer = broker.observer("root-1").open(_context())
    writer.push(ModelStreamDelta(channel="output", text="A😀"))
    writer.push(ModelStreamDelta(channel="reasoning", text="🧠"))
    writer.push(ModelStreamDelta(channel="output", text="é"))
    writer.push(ModelStreamDelta(channel="reasoning", text="ok"))

    deltas = [
        frame
        for frame in broker.subscribe("root-1", after_cursor="offsets:0").poll()
        if frame.kind == "delta"
    ]
    assert [
        (frame.channel, frame.text, frame.start_offset, frame.end_offset) for frame in deltas
    ] == [
        ("output", "A😀", 0, 5),
        ("reasoning", "🧠", 0, 4),
        ("output", "é", 5, 7),
        ("reasoning", "ok", 4, 6),
    ]
    assert deltas[0].to_json()["start_offset"] == 0
    assert deltas[0].to_json()["end_offset"] == 5
    assert all(frame.started_at == "2026-08-01T00:00:00Z" for frame in deltas)
    opened = broker.subscribe("root-1", after_cursor="offsets:0").poll(limit=1)[0]
    assert "start_offset" not in opened.to_json()
    assert "end_offset" not in opened.to_json()


def test_delta_frame_validates_offsets_and_non_delta_frames_reject_them() -> None:
    cursor = LiveModelStreamCursor("validation", 1)
    with pytest.raises(ValueError, match="requires UTF-8 byte offsets"):
        LiveModelStreamFrame(
            kind="delta",
            cursor=cursor,
            root_run_id="root",
            channel="output",
            text="x",
            started_at="2026-01-01T00:00:00Z",
        )
    with pytest.raises(ValueError, match="do not match"):
        LiveModelStreamFrame(
            kind="delta",
            cursor=cursor,
            root_run_id="root",
            channel="output",
            text="😀",
            start_offset=0,
            end_offset=1,
            started_at="2026-01-01T00:00:00Z",
        )
    with pytest.raises(ValueError, match="only on delta"):
        LiveModelStreamFrame(
            kind="opened",
            cursor=cursor,
            root_run_id="root",
            start_offset=0,
            end_offset=0,
        )


def test_sync_frame_generator_close_releases_subscription() -> None:
    broker = LiveModelStreamBroker(generation="sync-generator")
    subscription = broker.subscribe("root-1")
    iterator = subscription.frames(heartbeat_interval_s=0.01)

    assert next(iterator).kind == "heartbeat"
    iterator.close()

    assert subscription.closed is True


@pytest.mark.asyncio
async def test_async_frame_generator_cancellation_wakes_poll_worker() -> None:
    broker = LiveModelStreamBroker(generation="async-generator")
    subscription = broker.subscribe("root-1")
    waiting = asyncio.create_task(anext(subscription.aframes(heartbeat_interval_s=30.0)))
    await asyncio.sleep(0.01)

    started = time.monotonic()
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    assert subscription.closed is True
    assert time.monotonic() - started < 1.0


@pytest.mark.asyncio
async def test_direct_async_poll_cancellation_closes_subscription() -> None:
    broker = LiveModelStreamBroker(generation="async-poll")
    subscription = broker.subscribe("root-1")
    waiting = asyncio.create_task(subscription.apoll(timeout_s=30.0))
    await asyncio.sleep(0.01)

    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    assert subscription.closed is True


@pytest.mark.parametrize("root_run_id", ["root.sub.child", "../root", "root/child"])
def test_broker_rejects_descendant_or_unsafe_root_ids(root_run_id: str) -> None:
    broker = LiveModelStreamBroker()
    with pytest.raises(ValueError, match="root run id is invalid"):
        broker.subscribe(root_run_id)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"step": 0}, "positive integer"),
        ({"step": True}, "positive integer"),
        ({"started_at": "not-utc"}, "UTC timestamp"),
        ({"provider": 7}, "provider"),
        ({"model": []}, "model"),
    ],
)
def test_broker_rejects_context_that_cannot_form_a_wire_call_frame(
    changes: dict[str, Any], message: str
) -> None:
    values: dict[str, Any] = {
        "run_id": "root-1",
        "root_run_id": "root-1",
        "turn_id": "turn-1",
        "stream_id": "stream-1",
        "step": 1,
        "provider": "provider",
        "model": "model",
        "started_at": "2026-08-01T00:00:00Z",
    }
    values.update(changes)
    context = ModelStreamContext(**values)

    with pytest.raises(ValueError, match=message):
        LiveModelStreamBroker().observer("root-1").open(context)


@pytest.mark.parametrize(
    "value",
    ["", "missing-sequence", ":1", "generation:-1", "generation:1.0", "bad value:1"],
)
def test_cursor_parser_rejects_malformed_values(value: str) -> None:
    with pytest.raises(ValueError):
        LiveModelStreamCursor.parse(value)
