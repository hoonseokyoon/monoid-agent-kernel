from __future__ import annotations

from support.studio_harness import (
    FakeModelAdapter,
    ModelTurn,
    Path,
    StudioConfig,
    StudioServer,
    _python_command,
    _settled,
    _shell_studio,
    _wait_event,
    _wait_settled,
    describe_event,
    fake_tool_call,
    pytest,
)

pytestmark = pytest.mark.integration


def test_hitl_gate_parks_the_run_then_resumes_on_answer(tmp_path: Path) -> None:
    # The agent calls hitl.request; the run parks awaiting a human decision; answering it
    # resumes the run. This is the danger-op approval gate.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = FakeModelAdapter(
        turns=[
            ModelTurn(
                tool_calls=(
                    fake_tool_call(
                        "hitl_request",
                        {"prompt": "Delete all files in the workspace?", "choices": ["Approve", "Deny"]},
                        "c1",
                    ),
                )
            ),
            ModelTurn(final_text="Thanks — proceeding per your choice."),
        ]
    )
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    server.start()
    try:
        run_id = server.start_chat("clean up the workspace")["run_id"]
        started = _wait_event(server, run_id, "task.started")
        assert started is not None
        assert started["data"]["kind"] == "hitl"
        assert "Delete all files" in started["data"]["prompt"]
        # Parked: the turn has not settled while it waits for the human.
        assert not _settled(server, run_id)
        # Answer the gate -> the run resumes and settles (a final assistant turn appears).
        task_id = started["data"]["task_id"]
        server.answer_hitl(run_id, task_id, "Approve")
        settled = _wait_settled(server, run_id, 1)
        assert settled and settled[0]["data"]["final_text"]
    finally:
        server.shutdown()


def test_shell_foreground_command_runs(tmp_path: Path) -> None:
    server = _shell_studio(
        tmp_path,
        [
            ModelTurn(tool_calls=(fake_tool_call("shell_exec", {"command": _python_command("print('hi')")}, "c1"),)),
            ModelTurn(final_text="ran it"),
        ],
    )
    try:
        run_id = server.start_chat("run a command")["run_id"]
        _wait_settled(server, run_id, 1)
        events = server.poll_events(run_id, 0).get("events", [])
        finished = [e for e in events if e.get("type") == "shell.exec.finished"]
        assert finished and finished[0]["data"]["exit_code"] == 0
        # The shell call narrates into the activity feed.
        started = [e for e in events if e.get("type") == "tool.call.started" and e["data"].get("tool") == "shell_exec"]
        assert started and (describe_event(started[0]) or "").startswith("Running")
    finally:
        server.shutdown()


def test_shell_destructive_command_is_denied(tmp_path: Path) -> None:
    server = _shell_studio(
        tmp_path,
        [
            ModelTurn(tool_calls=(fake_tool_call("shell_exec", {"command": "rm -rf ."}, "c1"),)),
            ModelTurn(final_text="couldn't do that"),
        ],
    )
    try:
        run_id = server.start_chat("delete everything")["run_id"]
        _wait_settled(server, run_id, 1)
        events = server.poll_events(run_id, 0).get("events", [])
        # The destructive command is blocked by the binding's deny scope (before it can run).
        denied = {"tool_scope_denied", "shell_binding_denied"}
        failed = [e for e in events if (e.get("data") or {}).get("error_code") in denied]
        assert failed, "destructive command should be denied by the binding gate"
    finally:
        server.shutdown()


def test_shell_background_job_is_listed(tmp_path: Path) -> None:
    server = _shell_studio(
        tmp_path,
        [
            ModelTurn(
                tool_calls=(
                    fake_tool_call("shell_exec", {"command": _python_command("print('bg')"), "background": True}, "c1"),
                )
            ),
            ModelTurn(final_text="started a background job"),
        ],
    )
    try:
        run_id = server.start_chat("run something in the background")["run_id"]
        _wait_settled(server, run_id, 1)
        jobs = server.jobs(run_id).get("jobs", [])
        assert jobs, "the background job should be listed"
        assert jobs[0].get("job_id")
    finally:
        server.shutdown()


def test_shutdown_drains_background_shell_job_and_server_threads(tmp_path: Path) -> None:
    server = _shell_studio(
        tmp_path,
        [
            ModelTurn(
                tool_calls=(
                    fake_tool_call(
                        "shell_exec",
                        {"command": _python_command("import time; time.sleep(30)"), "background": True},
                        "c1",
                    ),
                )
            ),
            ModelTurn(final_text="started a long background job"),
        ],
    )
    server.start()
    run_id = server.start_chat("run a long job")["run_id"]
    _wait_settled(server, run_id, 1)
    assert server.jobs(run_id).get("jobs")

    server.shutdown()

    assert server.run_status(run_id)["terminal"] is True
    assert server._ui_thread is None
    assert server._gateway_thread is None
    assert server._web_gateway_thread is None
    assert server._mcp_thread is None


def test_web_search_runs_through_the_gateway(tmp_path: Path) -> None:
    fake = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("web_search", {"query": "monoid agent kernel"}, "c1"),)),
            ModelTurn(final_text="searched the web"),
        ]
    )
    server = StudioServer(
        StudioConfig(workspace=tmp_path / "ws", host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
    server.start()
    try:
        run_id = server.start_chat("look it up online")["run_id"]
        _wait_settled(server, run_id, 1)
        events = server.poll_events(run_id, 0).get("events", [])
        finished = [e for e in events if e.get("type") == "web.search.finished"]
        assert finished, "web search should run through the bundled gateway"
        assert not (finished[0].get("data") or {}).get("error")
        started = [
            e for e in events if e.get("type") == "tool.call.started" and e["data"].get("tool") == "web_search"
        ]
        assert started and "Searching the web for" in (describe_event(started[0]) or "")
    finally:
        server.shutdown()
