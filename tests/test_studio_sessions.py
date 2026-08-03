from __future__ import annotations

import json
from dataclasses import replace
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from support.studio_harness import (
    FakeModelAdapter,
    FakeStreamingModelAdapter,
    ModelAdapterError,
    ModelTurn,
    Path,
    StudioConfig,
    StudioServer,
    TextDelta,
    TurnComplete,
    _BlockingThenToolAdapter,
    _RaiseThenAdapter,
    _wait_event,
    _wait_settled,
    fake_tool_call,
    pytest,
    time,
)
from monoid_agent_kernel.core.external_agent_envelope import (
    ExternalAgentEnvelope,
    ExternalAgentError,
    ExternalAgentPart,
    ExternalAgentResult,
)
from monoid_agent_kernel.core.model_content import ModelContentStore, read_model_content
from monoid_agent_kernel.core.model_stream import (
    ModelStreamContext,
    ModelStreamDelta,
    ModelStreamOutcome,
    ModelStreamStatus,
)
from monoid_agent_kernel.errors import NativeAgentError

pytestmark = pytest.mark.integration


def _write_model_content_snapshot(
    run_dir: Path,
    *,
    root_run_id: str,
    run_id: str,
    stream_id: str,
    output: str,
    reasoning: str = "",
    status: ModelStreamStatus = "completed",
    retryable: bool = False,
    step: int = 1,
    turn_id: str | None = None,
    started_at: str = "2026-08-01T00:00:00Z",
) -> None:
    store = ModelContentStore(run_dir / "model-content.jsonl", run_id=run_id)
    writer = store.open(
        ModelStreamContext(
            run_id=run_id,
            root_run_id=root_run_id,
            turn_id=turn_id or f"turn_{step:04d}",
            stream_id=stream_id,
            step=step,
            provider="test",
            model="test-model",
            started_at=started_at,
        )
    )
    if reasoning:
        writer.push(ModelStreamDelta(channel="reasoning", text=reasoning))
    if output:
        writer.push(ModelStreamDelta(channel="output", text=output))
    writer.close(
        ModelStreamOutcome(
            status=status,
            final_text=output,
            usage={"output_tokens": 1},
            retryable=retryable,
        )
    )
    store.close()


def _write_open_model_content_snapshot(
    run_dir: Path,
    *,
    root_run_id: str,
    run_id: str,
    stream_id: str,
    turn_id: str,
    output: str,
) -> None:
    store = ModelContentStore(run_dir / "model-content.jsonl", run_id=run_id)
    writer = store.open(
        ModelStreamContext(
            run_id=run_id,
            root_run_id=root_run_id,
            turn_id=turn_id,
            stream_id=stream_id,
            step=1,
            provider="test",
            model="test-model",
            started_at="2026-08-01T00:00:00Z",
        )
    )
    writer.push(ModelStreamDelta(channel="output", text=output))
    # Closing the store flushes the prefix but intentionally writes no terminal sidecar record.
    store.close()


def _write_events(run_dir: Path, run_id: str, events: list[tuple[str, str | None]]) -> None:
    records = [
        {
            "seq": seq,
            "run_id": run_id,
            "turn_id": turn_id,
            "type": event_type,
            "data": {},
        }
        for seq, (event_type, turn_id) in enumerate(events, start=1)
    ]
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_studio_surfaces_turn_failed_without_terminating(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    adapter = _RaiseThenAdapter(
        [ModelAdapterError("unsupported effort", http_status=400), ModelTurn(final_text="ok now")]
    )
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: adapter,
    )
    server.start()
    try:
        run_id = server.start_chat("do the thing")["run_id"]
        failed = _wait_event(server, run_id, "turn.failed")
        assert failed is not None
        assert failed["data"]["http_status"] == 400
        assert failed["data"]["retryable"] is False
        # The session is NOT terminal — a follow-up is accepted (this is the whole point).
        assert server.run_status(run_id)["terminal"] is False
        server.continue_chat(run_id, "try again")
        assert _wait_settled(server, run_id, 1)  # the resend settles
    finally:
        server.shutdown()


def test_studio_retry_reissues_failed_by_value_request_without_duplicate_user_message(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()

    class CaptureFailThenSettle:
        def __init__(self) -> None:
            self.requests = []

        def next_turn(self, request):  # noqa: ANN001
            self.requests.append(request)
            if len(self.requests) == 1:
                raise ModelAdapterError("unsupported effort", http_status=400, retryable=False)
            return ModelTurn(response_id="r2", final_text="ok now")

    adapter = CaptureFailThenSettle()
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: adapter,
    )
    server.start()
    try:
        run_id = server.start_chat("do the thing")["run_id"]
        failed = _wait_event(server, run_id, "turn.failed")
        assert failed is not None and failed["data"]["retryable"] is False
        deadline = time.time() + 10
        while time.time() < deadline and server.run_status(run_id)["state"] != "awaiting_input":
            time.sleep(0.05)
        assert server.run_status(run_id)["state"] == "awaiting_input"

        request = Request(
            f"{server.base_url}/api/retry",
            data=json.dumps({"run_id": run_id}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            retried = json.loads(response.read().decode("utf-8"))
        assert retried["retried"] is True
        assert retried["retry_mode"] == "reissue_failed_turn"
        assert retried["retry_of_event_seq"] == failed["seq"]
        assert retried["retry_of_turn_id"] == failed["turn_id"]
        assert retried["new_attempt"] is False
        assert retried["message_snapshot_reused"] is True
        assert retried["request_snapshot_reused"] is False
        assert retried["runtime_config_source"] == "current"
        assert retried["message_snapshot"] == "existing_by_value_messages"
        retry_marker = _wait_event(server, run_id, "run.resumed")
        assert retry_marker is not None
        assert retry_marker["data"] == {"reason": "studio-retry"}
        assert retry_marker["turn_id"] == failed["turn_id"]
        assert _wait_settled(server, run_id, 1)

        assert len(adapter.requests) == 2
        first, second = adapter.requests
        assert first.messages == second.messages
        # Empty retry input is absent from the by-value messages. The ModelRequest object itself is
        # intentionally not byte-identical: instruction is empty and config is resolved afresh.
        assert second.instruction == ""
        transcript = server.chat_transcript(run_id)["messages"]
        assert [message["content"] for message in transcript if message["role"] == "user"] == [
            "do the thing"
        ]
    finally:
        server.shutdown()


def test_studio_retry_recovers_a_parked_session_before_enqueue(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs")
    )
    run_id = "run_parked"
    token = "run-token"
    server._run_tokens[run_id] = token

    class ParkedBackend:
        def __init__(self) -> None:
            self.resumed = False
            self.send_calls: list[dict[str, object]] = []

        def status(self, received_run_id: str, received_token: str) -> dict[str, object]:
            assert (received_run_id, received_token) == (run_id, token)
            return {"state": "awaiting_input", "terminal": False}

        def events(
            self, received_run_id: str, received_token: str, *, from_seq: int
        ) -> dict[str, object]:
            assert (received_run_id, received_token, from_seq) == (run_id, token, 0)
            return {
                "events": [
                    {
                        "type": "model.turn.started",
                        "seq": 6,
                        "turn_id": "turn_0001",
                        "data": {"step": 1},
                    },
                    {
                        "type": "turn.failed",
                        "seq": 7,
                        "data": {"retryable": False},
                    },
                ]
            }

        def resume_run(self, received_run_id: str, received_token: str) -> dict[str, object]:
            assert (received_run_id, received_token) == (run_id, token)
            self.resumed = True
            return {"resumed": True, "state": "awaiting_input", "terminal": False}

        def send_message(self, received_run_id: str, received_token: str, message: str, **kwargs):  # noqa: ANN001
            assert (received_run_id, received_token, message) == (run_id, token, "")
            self.send_calls.append(kwargs)
            if not self.resumed:
                raise KeyError(run_id)
            return {"status": "queued"}

    backend = ParkedBackend()
    server._backend = backend  # type: ignore[assignment]

    retried = server.retry_chat(run_id)

    assert retried["retried"] is True
    assert retried["retry_id"] == "studio_retry_7"
    assert retried["retry_of_turn_id"] == "turn_0001"
    assert backend.resumed is True
    assert backend.send_calls == [
        {
            "message_id": "studio_retry_7",
            "source": "studio-retry",
            "metadata": {
                "retry_of_event_seq": 7,
                "retry_of_turn_id": "turn_0001",
            },
        },
        {
            "message_id": "studio_retry_7",
            "source": "studio-retry",
            "metadata": {
                "retry_of_event_seq": 7,
                "retry_of_turn_id": "turn_0001",
            },
        },
    ]


def test_studio_retry_omits_unproven_malformed_turn_identity(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs")
    )
    run_id = "run_malformed_retry_identity"
    token = "run-token"
    server._run_tokens[run_id] = token

    class MalformedBackend:
        def __init__(self) -> None:
            self.send_kwargs: dict[str, object] | None = None

        def status(self, received_run_id: str, received_token: str) -> dict[str, object]:
            assert (received_run_id, received_token) == (run_id, token)
            return {"state": "awaiting_input", "terminal": False}

        def events(
            self, received_run_id: str, received_token: str, *, from_seq: int
        ) -> dict[str, object]:
            assert (received_run_id, received_token, from_seq) == (run_id, token, 0)
            return {
                "events": [
                    {
                        "type": "model.turn.started",
                        "seq": 6,
                        "turn_id": 123,
                        "data": {"step": 1},
                    },
                    {
                        "type": "turn.failed",
                        "seq": 7,
                        "turn_id": ["turn_0001"],
                        "data": {"retryable": False},
                    },
                ]
            }

        def send_message(
            self,
            received_run_id: str,
            received_token: str,
            message: str,
            **kwargs: object,
        ) -> dict[str, object]:
            assert (received_run_id, received_token, message) == (run_id, token, "")
            self.send_kwargs = kwargs
            return {"status": "queued"}

    backend = MalformedBackend()
    server._backend = backend  # type: ignore[assignment]

    retried = server.retry_chat(run_id)

    assert retried["retried"] is True
    assert "retry_of_turn_id" not in retried
    assert backend.send_kwargs is not None
    assert backend.send_kwargs["metadata"] == {"retry_of_event_seq": 7}


def test_studio_resume_route_reports_an_already_live_session(studio: StudioServer) -> None:
    run_id = studio.start_chat("hello")["run_id"]
    assert _wait_settled(studio, run_id, 1)
    request = Request(
        f"{studio.base_url}/api/resume",
        data=json.dumps({"run_id": run_id}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=5) as response:
        resumed = json.loads(response.read().decode("utf-8"))
    assert resumed == {
        "run_id": run_id,
        "state": "awaiting_input",
        "terminal": False,
        "resumed": False,
        "recovery_resumed": False,
        "turn_resumed": False,
        "resume_kind": "already_live",
    }


def test_studio_chat_emits_token_usage(studio: StudioServer) -> None:
    run_id = studio.start_chat("hello")["run_id"]
    _wait_settled(studio, run_id, 1)
    events = studio.poll_events(run_id, 0).get("events", [])
    metrics = [e for e in events if e.get("type") == "metrics.updated"]
    assert metrics, "the usage meter relies on metrics.updated events"
    assert any("total_tokens" in (e.get("data") or {}) for e in metrics)


def test_studio_cancel_terminates_run(studio: StudioServer) -> None:
    run_id = studio.start_chat("hello")["run_id"]
    _wait_settled(studio, run_id, 1)  # parks awaiting_input
    studio.cancel_chat(run_id)  # the Stop button path: cancel is run-level
    deadline = time.time() + 10
    while time.time() < deadline:
        if studio.run_status(run_id)["terminal"]:
            break
        time.sleep(0.1)
    assert studio.run_status(run_id)["terminal"] is True


def test_studio_interrupt_keeps_session_alive(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "notes.md").write_text("hello\n", encoding="utf-8")
    adapter = _BlockingThenToolAdapter()
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: adapter,
    )
    server.start()
    try:
        run_id = server.start_chat("go")["run_id"]
        assert adapter.reached_block.wait(10.0)  # a turn is now in flight (turn 2 blocking)
        result = server.interrupt_chat(run_id)  # the Stop button path
        assert result["interrupt_requested"] is True
        adapter.release.set()  # let turn 2 finish; the next boundary trips the interrupt
        # The run parks (awaiting_input) — interrupt must NOT terminalize it.
        deadline = time.time() + 10
        while time.time() < deadline:
            state = server.run_status(run_id)["state"]
            if state == "awaiting_input":
                break
            assert server.run_status(run_id)["terminal"] is False, "interrupt terminalized the run"
            time.sleep(0.05)
        assert server.run_status(run_id)["state"] == "awaiting_input"
        events = server.poll_events(run_id, 0).get("events", [])
        assert any(e.get("type") == "turn.interrupted" for e in events)
        # The session is alive: a follow-up message settles.
        server.continue_chat(run_id, "continue")
        assert len(_wait_settled(server, run_id, 1)) >= 1
    finally:
        server.shutdown()


def test_studio_pause_and_resume_routes_continue_the_frozen_turn(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "notes.md").write_text("hello\n", encoding="utf-8")
    adapter = _BlockingThenToolAdapter()
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: adapter,
    )
    server.start()
    try:
        run_id = server.start_chat("go")["run_id"]
        assert adapter.reached_block.wait(10.0)

        pause_request = Request(
            f"{server.base_url}/api/pause",
            data=json.dumps({"run_id": run_id}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(pause_request, timeout=5) as response:
            paused = json.loads(response.read().decode("utf-8"))
        assert paused["pause_requested"] is True
        assert paused["state"] == "running"
        assert paused["terminal"] is False

        adapter.release.set()
        deadline = time.time() + 10
        while time.time() < deadline and server.run_status(run_id)["state"] != "paused":
            time.sleep(0.05)
        assert server.run_status(run_id)["state"] == "paused"

        resume_request = Request(
            f"{server.base_url}/api/resume",
            data=json.dumps({"run_id": run_id}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(resume_request, timeout=5) as response:
            resumed = json.loads(response.read().decode("utf-8"))
        assert resumed["resumed"] is True
        assert resumed["turn_resumed"] is True
        assert resumed["resume_kind"] == "paused_turn"
        assert _wait_settled(server, run_id, 1)
    finally:
        adapter.release.set()
        server.shutdown()


def test_studio_streams_content_without_mirroring_tokens_into_durable_events(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    adapter = FakeStreamingModelAdapter(
        chunk_turns=[
            [
                TextDelta("Hel"),
                TextDelta("lo"),
                TurnComplete(response_id="r1", usage={"total_tokens": 3}),
            ]
        ]
    )
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: adapter,
    )
    server.start()
    try:
        run_id = server.start_chat("hi")["run_id"]
        _wait_settled(server, run_id, 1)
        assert server._backend is not None
        assert server._backend.emit_output_deltas is False
        assert server._backend.stream_model_calls is True
        assert server._backend.model_content_file is True
        assert server._backend.model_stream_broker is not None

        subscription = server.model_stream_subscription(run_id)
        frames = subscription.poll()
        subscription.close()
        assert [frame.kind for frame in frames] == ["opened", "delta", "delta", "closed"]
        assert [frame.text for frame in frames if frame.kind == "delta"] == ["Hel", "lo"]
        assert frames[-1].status == "completed"
        assert frames[-1].final_text == "Hello"

        events = server.poll_events(run_id, 0).get("events", [])
        assert not [event for event in events if str(event.get("type") or "").endswith(".delta")]
        settled = [e for e in events if e.get("type") == "turn.settled"]
        assert settled and settled[0]["data"]["final_text"] == "Hello"

        run_dir = server._run_dir_for(run_id)
        event_text = (run_dir / "events.jsonl").read_text(encoding="utf-8")
        assert "Hello" not in event_text
        assert "model.output.delta" not in event_text
        content = read_model_content(run_dir)
        assert content.snapshots[0].best_output_text == "Hello"
        assert content.snapshots[0].status == "completed"
    finally:
        server.shutdown()


def test_studio_model_stream_sse_resumes_and_disconnect_is_passive(
    studio: StudioServer,
) -> None:
    run_id = studio.start_chat("stream route")["run_id"]
    assert _wait_settled(studio, run_id, 1)
    assert studio.model_stream_enabled is True

    def read_one(*, last_event_id: str | None = None) -> tuple[str, dict]:
        headers = {"Accept": "text/event-stream"}
        if last_event_id is not None:
            headers["Last-Event-ID"] = last_event_id
        request = Request(
            f"{studio.base_url}/api/model-stream?run_id={run_id}",
            headers=headers,
        )
        with urlopen(request, timeout=5) as response:
            assert response.headers["Content-Type"].startswith("text/event-stream")
            event_id = ""
            while True:
                line = response.readline().decode("utf-8").rstrip("\r\n")
                if line.startswith("id: "):
                    event_id = line.removeprefix("id: ")
                if line.startswith("data: "):
                    return event_id, json.loads(line.removeprefix("data: "))

    first_id, first = read_one()
    resumed_id, resumed = read_one(last_event_id=first_id)

    assert first["schema_version"] == "monoid.model-stream.live.v1"
    assert first["kind"] == "opened"
    assert first["root_run_id"] == run_id
    assert resumed["sequence"] > first["sequence"]
    assert resumed_id != first_id
    # Closing either browser response released only its subscription. The multi-turn run remains
    # parked and ready for another message.
    assert studio.run_status(run_id)["terminal"] is False


def test_studio_model_stream_rejects_bad_cursor_and_unknown_run_before_sse(
    studio: StudioServer,
) -> None:
    run_id = studio.start_chat("cursor check")["run_id"]
    assert _wait_settled(studio, run_id, 1)

    for url in (
        f"{studio.base_url}/api/model-stream?run_id={run_id}&cursor=malformed",
        f"{studio.base_url}/api/model-stream?run_id={run_id}&cursor=",
        f"{studio.base_url}/api/model-stream?run_id=unknown-run",
    ):
        with pytest.raises(HTTPError) as exc_info:
            urlopen(Request(url, headers={"Accept": "text/event-stream"}), timeout=5)
        assert exc_info.value.code == 400
        assert exc_info.value.headers["Content-Type"].startswith("application/json")


def test_studio_model_content_snapshot_multiplexes_only_authorized_lineage(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    adapter = FakeStreamingModelAdapter(
        chunk_turns=[[TextDelta("한"), TurnComplete(response_id="r1")]]
    )
    server = StudioServer(
        StudioConfig(
            workspace=workspace,
            host="127.0.0.1",
            port=0,
            run_root=tmp_path / "runs",
        ),
        provider_factory=lambda _claims, _config: adapter,
    )
    server.start()
    try:
        root_run_id = server.start_chat("snapshot")["run_id"]
        assert _wait_settled(server, root_run_id, 1)
        run_root = server._backend.run_root  # type: ignore[union-attr]
        root_stream_id = read_model_content(run_root / root_run_id).snapshots[-1].context.stream_id

        child_run_id = f"{root_run_id}.sub.child"
        child_dir = run_root / child_run_id
        child_dir.mkdir()
        _write_model_content_snapshot(
            child_dir,
            root_run_id=root_run_id,
            run_id=child_run_id,
            stream_id="stream-child-old",
            output="old child history",
            step=1,
            started_at="2026-08-01T00:00:00Z",
        )
        _write_model_content_snapshot(
            child_dir,
            root_run_id=root_run_id,
            run_id=child_run_id,
            # A stream id collision across runs must preserve both root and child snapshots.
            stream_id=root_stream_id,
            output="child",
            reasoning="왜",
            status="failed",
            retryable=True,
            step=2,
            started_at="2026-08-01T00:00:01Z",
        )

        # Prefix-shaped directory with a context that disagrees with its path: never returned.
        forged_dir = run_root / f"{root_run_id}.sub.forged"
        forged_dir.mkdir()
        _write_model_content_snapshot(
            forged_dir,
            root_run_id=root_run_id,
            run_id=f"{root_run_id}.sub.somewhere-else",
            stream_id="stream-forged",
            output="must stay private",
        )
        # A fully valid foreign run is outside the requested lineage.
        foreign_dir = run_root / "foreign-root"
        foreign_dir.mkdir()
        _write_model_content_snapshot(
            foreign_dir,
            root_run_id="foreign-root",
            run_id="foreign-root",
            stream_id="stream-foreign",
            output="foreign private text",
        )
        # A descendant-shaped directory link must never escape the configured run root. This may
        # be unavailable on Windows hosts without Developer Mode; the lineage/context tests above
        # still exercise the non-link authorization checks there.
        linked_run_id = f"{root_run_id}.sub.linked"
        outside_dir = tmp_path / "outside-private-run"
        outside_dir.mkdir()
        _write_model_content_snapshot(
            outside_dir,
            root_run_id=root_run_id,
            run_id=linked_run_id,
            stream_id="stream-linked",
            output="linked escape text",
        )
        try:
            (run_root / linked_run_id).symlink_to(outside_dir, target_is_directory=True)
        except OSError:
            pass

        query = urlencode({"run_id": root_run_id})
        with urlopen(f"{server.base_url}/api/model-content?{query}", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))

        assert set(payload) == {"schema_version", "root_run_id", "streams"}
        assert payload["schema_version"] == "studio.model-content.v1"
        assert payload["root_run_id"] == root_run_id
        assert len(payload["streams"]) == 2
        assert {stream["run_id"] for stream in payload["streams"]} == {
            root_run_id,
            child_run_id,
        }
        root = next(stream for stream in payload["streams"] if stream["run_id"] == root_run_id)
        child = next(stream for stream in payload["streams"] if stream["run_id"] == child_run_id)
        assert root["output_text"] == "한"
        assert root["output_end_offset"] == 3
        assert child["reasoning_text"] == "왜"
        assert child["reasoning_end_offset"] == 3
        assert child["output_text"] == "child"
        assert child["output_end_offset"] == 5
        assert child["status"] == "failed"
        assert child["partial"] is True
        assert child["retryable"] is True
        assert root["retryable"] is False
        assert child["final_text"] == "child"
        assert child["step"] == 2
        assert root["stream_id"] == child["stream_id"] == root_stream_id
        serialized = json.dumps(payload, ensure_ascii=False)
        assert "old child history" not in serialized
        assert "must stay private" not in serialized
        assert "foreign private text" not in serialized
        assert "linked escape text" not in serialized

        for invalid_root in ("", child_run_id, "../foreign-root", "unknown-root"):
            invalid_query = urlencode({"run_id": invalid_root})
            with pytest.raises(HTTPError) as exc_info:
                urlopen(f"{server.base_url}/api/model-content?{invalid_query}", timeout=5)
            assert exc_info.value.code == 400
            assert exc_info.value.headers["Content-Type"].startswith("application/json")
    finally:
        server.shutdown()


def test_studio_model_content_rejects_unknown_root_before_reading_private_files(
    studio: StudioServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_read(_path):  # noqa: ANN001
        raise AssertionError("private content was read before root authorization")

    monkeypatch.setattr(
        "monoid_agent_kernel.reference.studio.server.read_model_content",
        unexpected_read,
    )
    with pytest.raises(NativeAgentError, match="unknown run_id"):
        studio.model_content_snapshot("unknown-root")


def test_studio_model_content_promotes_only_durably_active_abandoned_streams(
    studio: StudioServer,
) -> None:
    root_run_id = studio.start_chat("active stream recovery")["run_id"]
    assert _wait_settled(studio, root_run_id, 1)
    run_root = studio._backend.run_root  # type: ignore[union-attr]

    active_run_id = f"{root_run_id}.sub.active"
    active_dir = run_root / active_run_id
    active_dir.mkdir()
    active_store = ModelContentStore(
        active_dir / "model-content.jsonl",
        run_id=active_run_id,
        batch_interval_s=60.0,
    )
    active_writer = active_store.open(
        ModelStreamContext(
            run_id=active_run_id,
            root_run_id=root_run_id,
            turn_id="turn_active",
            stream_id="stream-active",
            step=1,
            provider="test",
            model="test-model",
            started_at="2026-08-01T00:00:00Z",
        )
    )
    active_writer.push(ModelStreamDelta(channel="output", text="still arriving"))
    _write_events(active_dir, active_run_id, [("model.turn.started", "turn_active")])

    # This is the durable shape left by a process kill: an open sidecar prefix and a committed
    # turn start, with no terminal record and no process-local writer after restart.
    crashed_run_id = f"{root_run_id}.sub.crashed"
    crashed_dir = run_root / crashed_run_id
    crashed_dir.mkdir()
    _write_open_model_content_snapshot(
        crashed_dir,
        root_run_id=root_run_id,
        run_id=crashed_run_id,
        stream_id="stream-crashed",
        turn_id="turn_crashed",
        output="stale prefix",
    )
    _write_events(
        crashed_dir,
        crashed_run_id,
        [("model.turn.started", "turn_crashed")],
    )

    corrupt_run_id = f"{root_run_id}.sub.corrupt"
    corrupt_dir = run_root / corrupt_run_id
    corrupt_dir.mkdir()
    _write_open_model_content_snapshot(
        corrupt_dir,
        root_run_id=root_run_id,
        run_id=corrupt_run_id,
        stream_id="stream-corrupt",
        turn_id="turn_corrupt",
        output="untrusted prefix",
    )
    _write_events(corrupt_dir, corrupt_run_id, [("model.turn.started", "turn_corrupt")])
    with (corrupt_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{not-json}\n")

    try:
        payload = studio.model_content_snapshot(root_run_id)
        by_run = {stream["run_id"]: stream for stream in payload["streams"]}

        assert by_run[active_run_id]["status"] == "running"
        assert by_run[active_run_id]["partial"] is True
        assert by_run[crashed_run_id]["status"] == "abandoned"
        assert by_run[crashed_run_id]["partial"] is True
        assert by_run[corrupt_run_id]["status"] == "abandoned"
        assert by_run[corrupt_run_id]["partial"] is True
    finally:
        active_store.close()


def test_studio_model_content_snapshot_flushes_the_active_private_batch(
    studio: StudioServer,
) -> None:
    root_run_id = studio.start_chat("flush reset hydration")["run_id"]
    _wait_settled(studio, root_run_id, 1)
    assert studio._backend is not None
    child_run_id = f"{root_run_id}.sub.buffered"
    child_dir = studio._backend.run_root / child_run_id
    child_dir.mkdir()
    store = ModelContentStore(
        child_dir / "model-content.jsonl",
        run_id=child_run_id,
        batch_interval_s=60.0,
    )
    turn_id = "turn_buffered"
    writer = store.open(
        ModelStreamContext(
            run_id=child_run_id,
            root_run_id=root_run_id,
            turn_id=turn_id,
            stream_id="stream-buffered",
            step=1,
            provider="test",
            model="test-model",
            started_at="2026-08-01T00:00:00Z",
        )
    )
    writer.push(ModelStreamDelta(channel="output", text="persisted"))
    writer.push(ModelStreamDelta(channel="output", text=" buffered tail"))
    _write_events(child_dir, child_run_id, [("model.turn.started", turn_id)])
    try:
        assert read_model_content(child_dir).snapshots[0].output_text == "persisted"

        payload = studio.model_content_snapshot(root_run_id)
        snapshot = next(item for item in payload["streams"] if item["run_id"] == child_run_id)

        assert snapshot["status"] == "running"
        assert snapshot["output_text"] == "persisted buffered tail"
        assert snapshot["output_end_offset"] == len("persisted buffered tail".encode("utf-8"))
    finally:
        store.close()


def test_studio_model_content_retries_an_unsettled_stream_closed_during_read(
    studio: StudioServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_run_id = studio.start_chat("close during reset hydration")["run_id"]
    assert _wait_settled(studio, root_run_id, 1)
    assert studio._backend is not None
    child_run_id = f"{root_run_id}.sub.closing"
    child_dir = studio._backend.run_root / child_run_id
    child_dir.mkdir()
    store = ModelContentStore(child_dir / "model-content.jsonl", run_id=child_run_id)
    turn_id = "turn_closing"
    writer = store.open(
        ModelStreamContext(
            run_id=child_run_id,
            root_run_id=root_run_id,
            turn_id=turn_id,
            stream_id="stream-closing",
            step=1,
            provider="test",
            model="test-model",
            started_at="2026-08-01T00:00:00Z",
        )
    )
    writer.push(ModelStreamDelta(channel="output", text="prefix before close"))
    _write_events(child_dir, child_run_id, [("model.turn.started", turn_id)])

    from monoid_agent_kernel.reference.studio import server as server_module

    original_read = server_module.read_model_content
    closed = False

    def close_after_read(path: Path, **kwargs):  # noqa: ANN003, ANN202
        nonlocal closed
        result = original_read(path, **kwargs)
        if Path(path).name == child_run_id and not closed:
            closed = True
            writer.close(
                ModelStreamOutcome(
                    status="interrupted",
                    final_text="prefix before close",
                )
            )
        return result

    monkeypatch.setattr(server_module, "read_model_content", close_after_read)
    try:
        with pytest.raises(
            NativeAgentError,
            match="model content snapshot is temporarily unavailable",
        ) as exc_info:
            studio.model_content_snapshot(root_run_id)
        assert exc_info.value.error_code == "model_content_unavailable"
    finally:
        store.close()


def test_studio_model_content_retries_an_unsettled_stream_opened_during_read(
    studio: StudioServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_run_id = studio.start_chat("open during reset hydration")["run_id"]
    assert _wait_settled(studio, root_run_id, 1)
    assert studio._backend is not None
    child_run_id = f"{root_run_id}.sub.opening"
    child_dir = studio._backend.run_root / child_run_id
    child_dir.mkdir()
    _write_model_content_snapshot(
        child_dir,
        root_run_id=root_run_id,
        run_id=child_run_id,
        stream_id="stream-history",
        output="settled history",
        step=1,
    )
    turn_id = "turn_opening"
    _write_events(child_dir, child_run_id, [("model.turn.started", turn_id)])

    from monoid_agent_kernel.reference.studio import server as server_module

    original_read = server_module.read_model_content
    opened_store: ModelContentStore | None = None

    def open_before_read(path: Path, **kwargs):  # noqa: ANN003, ANN202
        nonlocal opened_store
        if Path(path).name == child_run_id and opened_store is None:
            opened_store = ModelContentStore(
                child_dir / "model-content.jsonl",
                run_id=child_run_id,
            )
            writer = opened_store.open(
                ModelStreamContext(
                    run_id=child_run_id,
                    root_run_id=root_run_id,
                    turn_id=turn_id,
                    stream_id="stream-opening",
                    step=2,
                    provider="test",
                    model="test-model",
                    started_at="2026-08-01T00:00:01Z",
                )
            )
            writer.push(ModelStreamDelta(channel="output", text="new live prefix"))
        return original_read(path, **kwargs)

    monkeypatch.setattr(server_module, "read_model_content", open_before_read)
    try:
        with pytest.raises(
            NativeAgentError,
            match="model content snapshot is temporarily unavailable",
        ) as exc_info:
            studio.model_content_snapshot(root_run_id)
        assert exc_info.value.error_code == "model_content_unavailable"
    finally:
        if opened_store is not None:
            opened_store.close()


def test_studio_model_content_retries_a_complete_open_close_aba_during_read(
    studio: StudioServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_run_id = studio.start_chat("ABA during reset hydration")["run_id"]
    assert _wait_settled(studio, root_run_id, 1)
    assert studio._backend is not None
    child_run_id = f"{root_run_id}.sub.aba"
    child_dir = studio._backend.run_root / child_run_id
    child_dir.mkdir()
    _write_model_content_snapshot(
        child_dir,
        root_run_id=root_run_id,
        run_id=child_run_id,
        stream_id="stream-history",
        output="settled history",
        step=1,
    )
    turn_id = "turn_aba"
    _write_events(child_dir, child_run_id, [("model.turn.started", turn_id)])

    from monoid_agent_kernel.reference.studio import server as server_module

    original_read = server_module.read_model_content
    churned = False

    def churn_during_read(path: Path, **kwargs):  # noqa: ANN003, ANN202
        nonlocal churned
        if Path(path).name != child_run_id or churned:
            return original_read(path, **kwargs)
        churned = True
        store = ModelContentStore(child_dir / "model-content.jsonl", run_id=child_run_id)
        writer = store.open(
            ModelStreamContext(
                run_id=child_run_id,
                root_run_id=root_run_id,
                turn_id=turn_id,
                stream_id="stream-aba",
                step=2,
                provider="test",
                model="test-model",
                started_at="2026-08-01T00:00:01Z",
            )
        )
        writer.push(ModelStreamDelta(channel="output", text="short-lived live prefix"))
        result = original_read(path, **kwargs)
        writer.close(
            ModelStreamOutcome(
                status="completed",
                final_text="short-lived live prefix",
            )
        )
        store.close()
        return result

    monkeypatch.setattr(server_module, "read_model_content", churn_during_read)
    with pytest.raises(
        NativeAgentError,
        match="model content snapshot is temporarily unavailable",
    ) as exc_info:
        studio.model_content_snapshot(root_run_id)

    assert exc_info.value.error_code == "model_content_unavailable"
    assert churned


def test_studio_model_content_flush_failure_returns_service_unavailable(
    studio: StudioServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_run_id = studio.start_chat("failed reset hydration")["run_id"]
    assert _wait_settled(studio, root_run_id, 1)

    def fail_flush(_path: Path) -> int:
        raise OSError("simulated private sidecar failure")

    monkeypatch.setattr(
        "monoid_agent_kernel.reference.studio.server.flush_active_model_content",
        fail_flush,
    )
    query = urlencode({"run_id": root_run_id})

    with pytest.raises(HTTPError) as exc_info:
        urlopen(f"{studio.base_url}/api/model-content?{query}", timeout=5)

    assert exc_info.value.code == 503
    assert exc_info.value.headers["Content-Type"].startswith("application/json")
    payload = json.loads(exc_info.value.read().decode("utf-8"))
    assert payload == {"error": "model content snapshot is temporarily unavailable"}


def test_studio_model_content_rejects_replaced_active_sidecar_with_service_unavailable(
    studio: StudioServer,
) -> None:
    root_run_id = studio.start_chat("replaced reset hydration")["run_id"]
    assert _wait_settled(studio, root_run_id, 1)
    assert studio._backend is not None
    child_run_id = f"{root_run_id}.sub.replaced"
    child_dir = studio._backend.run_root / child_run_id
    child_dir.mkdir()
    path = child_dir / "model-content.jsonl"
    store = ModelContentStore(path, run_id=child_run_id, batch_interval_s=60.0)
    turn_id = "turn_replaced"
    writer = store.open(
        ModelStreamContext(
            run_id=child_run_id,
            root_run_id=root_run_id,
            turn_id=turn_id,
            stream_id="stream-replaced",
            step=1,
            provider="test",
            model="test-model",
            started_at="2026-08-01T00:00:00Z",
        )
    )
    writer.push(ModelStreamDelta(channel="output", text="persisted"))
    writer.push(ModelStreamDelta(channel="output", text=" buffered tail"))
    _write_events(child_dir, child_run_id, [("model.turn.started", turn_id)])
    displaced = child_dir / "displaced-model-content.jsonl"
    try:
        path.replace(displaced)
        path.write_text("replacement must remain untouched\n", encoding="utf-8")
    except OSError as exc:
        store.close()
        pytest.skip(f"open-file replacement is unavailable: {exc}")

    try:
        query = urlencode({"run_id": root_run_id})
        with pytest.raises(HTTPError) as exc_info:
            urlopen(f"{studio.base_url}/api/model-content?{query}", timeout=5)

        assert exc_info.value.code == 503
        assert path.read_text(encoding="utf-8") == "replacement must remain untouched\n"
    finally:
        store.close()


def test_studio_model_content_rejects_hardlinked_sidecar_with_service_unavailable(
    studio: StudioServer,
    tmp_path: Path,
) -> None:
    root_run_id = studio.start_chat("hardlinked reset hydration")["run_id"]
    assert _wait_settled(studio, root_run_id, 1)
    assert studio._backend is not None
    child_run_id = f"{root_run_id}.sub.hardlinked"
    child_dir = studio._backend.run_root / child_run_id
    child_dir.mkdir()
    outside_dir = tmp_path / "outside-hardlinked-sidecar"
    outside_dir.mkdir()
    _write_model_content_snapshot(
        outside_dir,
        root_run_id=root_run_id,
        run_id=child_run_id,
        stream_id="stream-hardlinked",
        output="outside hardlinked content",
    )
    try:
        (child_dir / "model-content.jsonl").hardlink_to(outside_dir / "model-content.jsonl")
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"hard links are unavailable: {exc}")
    query = urlencode({"run_id": root_run_id})

    with pytest.raises(HTTPError) as exc_info:
        urlopen(f"{studio.base_url}/api/model-content?{query}", timeout=5)

    assert exc_info.value.code == 503
    assert exc_info.value.headers["Content-Type"].startswith("application/json")
    assert "outside hardlinked content" not in exc_info.value.read().decode("utf-8")


def test_studio_model_stream_denied_egress_returns_403_before_run_lookup(
    tmp_path: Path,
) -> None:
    server = StudioServer(
        StudioConfig(
            workspace=tmp_path / "ws",
            host="127.0.0.1",
            port=0,
            run_root=tmp_path / "runs",
            stream_output_deltas=False,
        )
    )
    server.start()
    try:
        for path in ("/api/model-stream", "/api/model-content"):
            with pytest.raises(HTTPError) as exc_info:
                urlopen(
                    Request(
                        f"{server.base_url}{path}?run_id=unknown-run",
                        headers={"Accept": "text/event-stream"},
                    ),
                    timeout=5,
                )
            assert exc_info.value.code == 403
            assert exc_info.value.headers["Content-Type"].startswith("application/json")
    finally:
        server.shutdown()


def test_studio_renders_plan_updates(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = FakeModelAdapter(
        turns=[
            ModelTurn(
                tool_calls=(
                    fake_tool_call(
                        "run_update_plan",
                        {
                            "items": [
                                {"step": "Read the files", "status": "completed"},
                                {"step": "Edit the code", "status": "in_progress"},
                            ]
                        },
                        "c1",
                    ),
                )
            ),
            ModelTurn(final_text="on it"),
        ]
    )
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    server.start()
    try:
        run_id = server.start_chat("do the task")["run_id"]
        _wait_settled(server, run_id, 1)
        events = server.poll_events(run_id, 0).get("events", [])
        plans = [e for e in events if e.get("type") == "plan.updated"]
        assert plans, "the Plan panel relies on plan.updated events"
        items = plans[-1]["data"]["items"]
        assert {i["step"] for i in items} == {"Read the files", "Edit the code"}
        assert any(i["status"] == "in_progress" for i in items)
    finally:
        server.shutdown()


def test_studio_spawns_subagent_and_exposes_child_events(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "data.md").write_text("the answer is 42\n", encoding="utf-8")
    # One shared fake drives parent + child turns in sequence (the child reuses the parent's
    # adapter instance): parent delegates, child answers, parent wraps up.
    fake = FakeModelAdapter(
        turns=[
            ModelTurn(
                tool_calls=(
                    fake_tool_call(
                        "agent_spawn",
                        {"subagent_type": "researcher", "prompt": "find the answer in data.md"},
                        "c1",
                    ),
                )
            ),
            ModelTurn(final_text="The answer is 42."),
            ModelTurn(final_text="My researcher reports: 42."),
        ]
    )
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    server.start()
    try:
        run_id = server.start_chat("delegate the lookup")["run_id"]
        _wait_settled(server, run_id, 1)
        events = server.poll_events(run_id, 0).get("events", [])
        started = [e for e in events if e.get("type") == "subagent.started"]
        assert started, "the parent stream should carry subagent.started"
        child_run_id = started[0]["data"]["child_run_id"]
        assert ".sub." in child_run_id and run_id in child_run_id
        assert any(e.get("type") == "subagent.finished" for e in events)
        # The child's own work is streamable via subagent_events (reads the child's events.jsonl).
        child = server.subagent_events(child_run_id).get("events", [])
        settled = [e for e in child if e.get("type") == "turn.settled"]
        assert settled and "42" in settled[-1]["data"]["final_text"]
        # path-traversal guard
        assert server.subagent_events("../secrets")["events"] == []
    finally:
        server.shutdown()


def test_studio_sessions_lists_started_chats_newest_first(studio: StudioServer) -> None:
    r1 = studio.start_chat("first task")["run_id"]
    _wait_settled(studio, r1, 1)
    r2 = studio.start_chat("second task")["run_id"]
    _wait_settled(studio, r2, 1)
    sessions = studio.sessions()["sessions"]
    assert [s["title"] for s in sessions[:2]] == ["second task", "first task"]  # newest first
    assert {r1, r2} <= {s["run_id"] for s in sessions}
    # each entry carries a live state (active multi-turn sessions are not terminal)
    by_id = {s["run_id"]: s for s in sessions}
    assert by_id[r1]["terminal"] is False
    assert by_id[r1]["last_event_seq"] >= 1
    assert by_id[r1]["state"] == "awaiting_input"


def test_studio_profiles_scope_session_history(studio: StudioServer) -> None:
    profiles = studio.profiles()
    assert profiles["default_profile_id"] == "default"
    assert "run_update_plan tool" in profiles["system_prompt_base"]
    assert {"default", "reviewer", "builder"} <= {p["id"] for p in profiles["profiles"]}

    default_run = studio.start_chat("default task", profile_id="default")["run_id"]
    _wait_settled(studio, default_run, 1)
    reviewer_run = studio.start_chat("review task", profile_id="reviewer")["run_id"]
    _wait_settled(studio, reviewer_run, 1)

    default_sessions = studio.sessions(profile_id="default")["sessions"]
    reviewer_sessions = studio.sessions(profile_id="reviewer")["sessions"]
    all_sessions = studio.sessions()["sessions"]

    assert {s["run_id"] for s in default_sessions} == {default_run}
    assert {s["run_id"] for s in reviewer_sessions} == {reviewer_run}
    assert reviewer_sessions[0]["profile_id"] == "reviewer"
    reviewer_summary = next(
        session for session in all_sessions if session["run_id"] == reviewer_run
    )
    assert reviewer_summary["profile_id"] == "reviewer"
    assert reviewer_summary["last_event_seq"] >= 1


def test_studio_profile_preview_resolves_model_request_surface(studio: StudioServer) -> None:
    preview = studio.profile_preview(
        {
            "name": "Previewer",
            "description": "Preview test",
            "instructions": "Always mention PREVIEW_SENTINEL.",
            "capabilities": ["read", "write", "delegate"],
            "model": "gpt-preview",
            "effort": "high",
            "summary": "off",
        }
    )

    assert "PREVIEW_SENTINEL" in preview["system_prompt"]
    assert "Profile instructions:" in preview["system_prompt"]
    assert preview["request_config"]["model"] == "gpt-preview"
    assert preview["request_config"]["reasoning"] == {"effort": "high", "summary": "off"}
    tool_names = {tool["name"] for tool in preview["tools"]}
    assert {
        "run_update_plan",
        "fs_read",
        "fs_list",
        "fs_patch",
        "fs_delete",
        "agent_spawn",
    } <= tool_names
    assert "tool_surface" in preview
    assert preview["tool_surface"]["authorizations"]["fs.copy"]["decision"] == "ask"
    assert preview["tool_surface"]["authorizations"]["fs.move"]["decision"] == "ask"
    assert preview["tool_surface"]["authorizations"]["fs.delete"]["decision"] == "ask"
    read_tool = next(tool for tool in preview["tools"] if tool["name"] == "fs_read")
    assert read_tool["input_schema"]["type"] == "object"
    assert preview["schema_version"] == "studio.model-request-preview.v1"
    assert preview["snapshot_kind"] == "initial_new_chat_turn"
    assert preview["input_bound"] is False
    assert preview["unbound_fields"] == ["instruction", "messages"]
    model_request = preview["model_request"]
    assert model_request["instruction"] is None
    assert model_request["messages"] is None
    assert model_request["previous_turn_handle"] is None
    assert model_request["observations"] == []
    assert model_request["system_prompt"] == preview["system_prompt"]
    assert model_request["tools"] == preview["tools"]
    assert model_request["model"]["model"] == "gpt-preview"
    assert model_request["model"]["reasoning"]["effort"] == "high"

    bound = studio.profile_preview(
        {
            "name": "Previewer",
            "instructions": "Always mention PREVIEW_SENTINEL.",
            "capabilities": ["read"],
            "model": "gpt-preview",
            "effort": "high",
            "summary": "off",
            "preview_instruction": "Inspect the workspace.",
        }
    )
    assert bound["input_bound"] is True
    assert bound["unbound_fields"] == []
    assert bound["model_request"]["instruction"] == "Inspect the workspace."
    assert bound["model_request"]["messages"] == [
        {"role": "user", "content": "Inspect the workspace."}
    ]


def test_studio_profile_preview_substitutes_a_preserved_non_finite_tool_schema(
    studio: StudioServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The preview is a *record* of a request, so it obeys the record half of the schema rule.

    A tool's ``input_schema`` keeps its non-finite values through ingress (that is what makes the
    provider boundary refuse the call as a classified, config-recoverable bad request), so the
    value reaches every surface that embeds the schema. This endpoint serializes with
    ``allow_nan=False``: embedding the schema raw killed a request whose only purpose is to
    *look at* the tool surface, with an anonymous serialization error rather than the classified
    refusal a real call gets. Same substitution the transcript's ``_tool_spec_payload`` and the
    run manifest make.
    """

    from monoid_agent_kernel.reference.studio import server as studio_server

    real_builtin_tools = studio_server.builtin_tools

    def _builtin_tools_with_a_non_finite_schema(workspace):
        return [
            replace(tool, input_schema={**tool.input_schema, "default": float("nan")})
            if tool.id == "fs.read"
            else tool
            for tool in real_builtin_tools(workspace)
        ]

    monkeypatch.setattr(studio_server, "builtin_tools", _builtin_tools_with_a_non_finite_schema)

    request = Request(
        f"{studio.base_url}/api/profile-preview",
        data=json.dumps(
            {
                "name": "Previewer",
                "instructions": "Preview a non-finite tool schema.",
                "capabilities": ["read"],
                "model": "gpt-preview",
                "effort": "high",
                "summary": "off",
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=5) as response:
        assert response.status == 200
        preview = json.loads(response.read().decode("utf-8"))

    read_tool = next(tool for tool in preview["tools"] if tool["id"] == "fs.read")
    assert read_tool["input_schema"]["default"] is None
    # The record is portable JSON end to end, which is the property the endpoint depends on.
    assert json.loads(json.dumps(preview, allow_nan=False))["tool_count"] == preview["tool_count"]


def test_studio_profile_history_survives_restart(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    run_root = tmp_path / "runs"
    s1 = StudioServer(
        StudioConfig(
            workspace=workspace, host="127.0.0.1", port=0, provider="offline", run_root=run_root
        )
    )
    s1.start()
    try:
        rid = s1.start_chat("remember reviewer", profile_id="reviewer")["run_id"]
        _wait_settled(s1, rid, 1)
    finally:
        s1.shutdown()

    s2 = StudioServer(
        StudioConfig(
            workspace=workspace, host="127.0.0.1", port=0, provider="offline", run_root=run_root
        )
    )
    s2.start()
    try:
        reviewer_sessions = s2.sessions(profile_id="reviewer")["sessions"]
        default_sessions = s2.sessions(profile_id="default")["sessions"]
        all_sessions = s2.sessions()["sessions"]
        assert any(x["run_id"] == rid and x["profile_id"] == "reviewer" for x in reviewer_sessions)
        assert all(x["run_id"] != rid for x in default_sessions)
        restarted_summary = next(session for session in all_sessions if session["run_id"] == rid)
        assert restarted_summary["profile_id"] == "reviewer"
        assert restarted_summary["last_event_seq"] >= 1
    finally:
        s2.shutdown()


def test_run_events_carry_trace_nesting(tmp_path: Path) -> None:
    # The trace tree nests by event_id/parent_id; verify a tool call nests under its turn.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "notes.md").write_text("hi\n", encoding="utf-8")
    fake = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("fs_read", {"path": "notes.md"}, "c1"),)),
            ModelTurn(final_text="done"),
        ]
    )
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    server.start()
    try:
        run_id = server.start_chat("read it")["run_id"]
        _wait_settled(server, run_id, 1)
        events = server.poll_events(run_id, 0).get("events", [])
        ids = {e["event_id"] for e in events if e.get("event_id")}
        tool = next(e for e in events if e.get("type") == "tool.call.started")
        assert tool.get("event_id") and tool.get("parent_id") in ids  # nests under a parent event
    finally:
        server.shutdown()


def test_studio_history_survives_restart(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    run_root = tmp_path / "runs"
    s1 = StudioServer(
        StudioConfig(
            workspace=workspace, host="127.0.0.1", port=0, provider="offline", run_root=run_root
        )
    )
    s1.start()
    try:
        rid = s1.start_chat("remember me")["run_id"]
        _wait_settled(s1, rid, 1)
    finally:
        s1.shutdown()
    # A fresh studio over the same run_root == a restart (no in-memory records/tokens).
    s2 = StudioServer(
        StudioConfig(
            workspace=workspace, host="127.0.0.1", port=0, provider="offline", run_root=run_root
        )
    )
    s2.start()
    try:
        sessions = s2.sessions()["sessions"]
        assert any(x["run_id"] == rid and x["title"] == "remember me" for x in sessions)
        # the past transcript is readable even though s2 has no live record for it
        events = s2.poll_events(rid, 0)["events"]
        assert any(e.get("type") == "turn.settled" for e in events)
        transcript = s2.chat_transcript(rid)
        assert [message["role"] for message in transcript["messages"][:2]] == ["user", "assistant"]
        assert transcript["messages"][0]["content"] == "remember me"
        assert transcript["event_cursor"] >= 0
    finally:
        s2.shutdown()


def test_studio_chat_transcript_preserves_multi_turn_order(studio: StudioServer) -> None:
    run_id = studio.start_chat("first", client_message_id="client-1")["run_id"]
    _wait_settled(studio, run_id, 1)
    studio.continue_chat(run_id, "second", client_message_id="client-2")
    _wait_settled(studio, run_id, 2)

    first = studio.chat_transcript(run_id)
    second = studio.chat_transcript(run_id)

    assert len(first["messages"]) == len(second["messages"])
    assert [(m["role"], m["content"]) for m in first["messages"] if m["role"] == "user"] == [
        ("user", "first"),
        ("user", "second"),
    ]
    assert len([m for m in first["messages"] if m["role"] == "assistant"]) >= 2


def test_studio_initial_user_timestamp_precedes_fast_reply(studio: StudioServer) -> None:
    run_id = studio.start_chat("fast reply", client_message_id="client-fast")["run_id"]
    _wait_settled(studio, run_id, 1)

    messages = studio.chat_transcript(run_id)["messages"]
    user = next(message for message in messages if message["role"] == "user")
    assistant = next(message for message in messages if message["role"] == "assistant")

    assert [message["role"] for message in messages[:2]] == ["user", "assistant"]
    assert user["created_at"] <= assistant["created_at"]


def test_studio_followup_user_timestamp_precedes_fast_reply(studio: StudioServer) -> None:
    run_id = studio.start_chat("first", client_message_id="client-followup-1")["run_id"]
    _wait_settled(studio, run_id, 1)
    studio.continue_chat(run_id, "second", client_message_id="client-followup-2")
    _wait_settled(studio, run_id, 2)

    messages = studio.chat_transcript(run_id)["messages"]
    users = [message for message in messages if message["role"] == "user"]
    assistants = [message for message in messages if message["role"] == "assistant"]

    assert [message["role"] for message in messages[:4]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert users[1]["created_at"] <= assistants[1]["created_at"]


def test_studio_chat_transcript_http_route(studio: StudioServer) -> None:
    run_id = studio.start_chat("route check", client_message_id="client-route")["run_id"]
    _wait_settled(studio, run_id, 1)

    with urlopen(f"{studio.base_url}/api/chat-transcript?run_id={run_id}", timeout=5) as response:
        body = json.loads(response.read().decode("utf-8"))

    assert body["schema_version"] == "studio.chat.v2"
    assert body["run_id"] == run_id
    assert body["messages"][0]["content"] == "route check"
    assert body["event_cursor"] >= 0


def test_studio_sse_uses_event_ids_and_last_event_id_resume(studio: StudioServer) -> None:
    run_id = studio.start_chat("hello")["run_id"]
    _wait_settled(studio, run_id, 1)
    studio.cancel_chat(run_id)
    deadline = time.time() + 10
    while not studio.run_status(run_id)["terminal"] and time.time() < deadline:
        time.sleep(0.02)
    assert studio.run_status(run_id)["terminal"] is True

    def read(last_event_id: str | None = None) -> tuple[list[int], str]:
        headers = {"Accept": "text/event-stream"}
        if last_event_id is not None:
            headers["Last-Event-ID"] = last_event_id
        request = Request(
            f"{studio.base_url}/api/events?run_id={run_id}&from=0",
            headers=headers,
        )
        with urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8")
        ids = [
            int(line.removeprefix("id: ")) for line in body.splitlines() if line.startswith("id: ")
        ]
        return ids, body

    ids, body = read()
    assert ids == sorted(set(ids))
    assert "studio.stream.end" in body
    resumed_ids, _ = read(str(ids[0]))
    assert resumed_ids == ids[1:]


def test_a2a_demo_preset_wires_two_peers(studio: StudioServer) -> None:
    """The one-click A2A preset spins up two named peers wired to message each other through the
    durable outbox→inbox fabric: both are registered in the agent directory (addressable by name),
    each carries a lease-gated outbox.send binding, and each persona names its counterpart. The
    cross-agent delivery itself is covered end-to-end in test_outbox.py."""
    result = studio.start_a2a_demo("draft a release note together")
    planner_id, worker_id = result["planner"], result["worker"]
    assert planner_id and worker_id and planner_id != worker_id

    # Addressable by name; run tokens held server-side (never sent to the browser).
    assert studio._agent_directory == {"worker": worker_id, "planner": planner_id}
    assert planner_id in studio._run_tokens and worker_id in studio._run_tokens

    # Each peer carries a lease-gated outbox.send binding + a persona naming its counterpart.
    planner_cfg = studio._backend.current_runtime_config(planner_id)
    outbox = [b for b in planner_cfg.tools if b.ref.tool_id == "outbox.send"]
    assert outbox and outbox[0].runtime.get("requires_lease") is True
    assert "worker" in planner_cfg.prompt.system_prompt_base

    worker_cfg = studio._backend.current_runtime_config(worker_id)
    assert any(b.ref.tool_id == "outbox.send" for b in worker_cfg.tools)
    assert "planner" in worker_cfg.prompt.system_prompt_base


def test_studio_a2a_delivery_preserves_external_agent_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs")
    )
    captured: dict[str, object] = {}

    class BackendStub:
        def send_message(self, run_id: str, token: str, content, **kwargs):  # noqa: ANN001
            captured["run_id"] = run_id
            captured["token"] = token
            captured["content"] = content
            captured["kwargs"] = kwargs
            return {"message_id": kwargs.get("message_id")}

    server._backend = BackendStub()  # type: ignore[assignment]
    server._agent_directory["worker"] = "run-worker"
    server._run_tokens["run-worker"] = "run-token"
    envelope = ExternalAgentEnvelope(
        peer_id="planner",
        message_id="message-1",
        task_id="task-1",
        request_id="request-1",
        reply_to_id="reply-1",
        parts=(ExternalAgentPart(type="text", text="done"),),
        result=ExternalAgentResult(
            state="completed",
            terminal=True,
            error=ExternalAgentError(code="none", message=""),
        ),
        metadata={"custom": "ok", "task_id": "spoofed"},
    )

    result = server._a2a_deliver(
        "worker",
        envelope.to_json(),
        message_id="fallback",
        correlation_id="corr-1",
        causation_id="cause-1",
        traceparent="",
    )

    assert result == "a2a:run-worker:message-1"
    assert captured["run_id"] == "run-worker"
    assert captured["token"] == "run-token"
    assert captured["content"] == "done"
    kwargs = captured["kwargs"]
    assert kwargs["message_type"] == "external_agent_message"
    assert kwargs["metadata"]["custom"] == "ok"
    assert kwargs["metadata"]["task_id"] == "task-1"
    assert kwargs["metadata"]["request_id"] == "request-1"
    assert kwargs["metadata"]["reply_to_id"] == "reply-1"
    assert kwargs["metadata"]["result"]["state"] == "completed"
