from __future__ import annotations

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

pytestmark = pytest.mark.integration


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
        assert server.run_status(run_id)["status"] not in {"completed", "failed", "limited"}
        server.continue_chat(run_id, "try again")
        assert _wait_settled(server, run_id, 1)  # the resend settles
    finally:
        server.shutdown()


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
        if studio.run_status(run_id)["status"] in {"completed", "failed", "limited"}:
            break
        time.sleep(0.1)
    assert studio.run_status(run_id)["status"] in {"completed", "failed", "limited"}


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
            status = server.run_status(run_id)["status"]
            if status == "awaiting_input":
                break
            assert status not in {"completed", "failed", "limited"}, "interrupt terminalized the run"
            time.sleep(0.05)
        assert server.run_status(run_id)["status"] == "awaiting_input"
        events = server.poll_events(run_id, 0).get("events", [])
        assert any(e.get("type") == "turn.interrupted" for e in events)
        # The session is alive: a follow-up message settles.
        server.continue_chat(run_id, "continue")
        assert len(_wait_settled(server, run_id, 1)) >= 1
    finally:
        server.shutdown()


def test_studio_streams_output_deltas(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    adapter = FakeStreamingModelAdapter(
        chunk_turns=[[TextDelta("Hel"), TextDelta("lo"), TurnComplete(response_id="r1", usage={"total_tokens": 3})]]
    )
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: adapter,
    )
    server.start()
    try:
        run_id = server.start_chat("hi")["run_id"]
        _wait_settled(server, run_id, 1)
        events = server.poll_events(run_id, 0).get("events", [])
        deltas = [e for e in events if e.get("type") == "model.output.delta"]
        assert [d["data"]["text"] for d in deltas] == ["Hel", "lo"]
        settled = [e for e in events if e.get("type") == "turn.settled"]
        assert settled and settled[0]["data"]["final_text"] == "Hello"
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
                        {"items": [
                            {"step": "Read the files", "status": "completed"},
                            {"step": "Edit the code", "status": "in_progress"},
                        ]},
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
    # each entry carries a live status (active multi-turn sessions are not terminal)
    by_id = {s["run_id"]: s for s in sessions}
    assert by_id[r1]["status"] not in {"completed", "failed", "limited"}


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
    s1 = StudioServer(StudioConfig(workspace=workspace, host="127.0.0.1", port=0, provider="offline", run_root=run_root))
    s1.start()
    try:
        rid = s1.start_chat("remember me")["run_id"]
        _wait_settled(s1, rid, 1)
    finally:
        s1.shutdown()
    # A fresh studio over the same run_root == a restart (no in-memory records/tokens).
    s2 = StudioServer(StudioConfig(workspace=workspace, host="127.0.0.1", port=0, provider="offline", run_root=run_root))
    s2.start()
    try:
        sessions = s2.sessions()["sessions"]
        assert any(x["run_id"] == rid and x["title"] == "remember me" for x in sessions)
        # the past transcript is readable even though s2 has no live record for it
        events = s2.poll_events(rid, 0)["events"]
        assert any(e.get("type") == "turn.settled" for e in events)
    finally:
        s2.shutdown()


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
