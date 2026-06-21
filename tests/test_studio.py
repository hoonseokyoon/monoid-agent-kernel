"""R0 coverage for the Agent Studio reference app.

Studio is a reference example, but it is the pressure test for "build an app from the surface
alone", so it gets the same regression coverage the other reference services have. These tests
drive the Studio server through its Python API (no browser / no Chromium window) against the
offline echo model, so they are deterministic and key-less.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from native_agent_runner.providers.base import ModelRequest, ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call
from native_agent_runner.reference.llm_gateway.providers import EchoModelAdapter
from native_agent_runner.reference.studio.activity import describe_event
from native_agent_runner.reference.studio.server import (
    StudioConfig,
    StudioServer,
    _agent_runtime_config,
)


def _settled(server: StudioServer, run_id: str) -> list[dict]:
    events = server.poll_events(run_id, 0).get("events", [])
    return [e for e in events if e.get("type") == "turn.settled"]


def _wait_settled(server: StudioServer, run_id: str, n: int, timeout: float = 10.0) -> list[dict]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        settled = _settled(server, run_id)
        if len(settled) >= n:
            return settled
        time.sleep(0.1)
    return _settled(server, run_id)


@pytest.fixture
def studio(tmp_path: Path):
    server = StudioServer(
        StudioConfig(
            workspace=tmp_path / "ws",
            host="127.0.0.1",
            port=0,
            provider="offline",
            run_root=tmp_path / "runs",
        )
    )
    server.start()
    try:
        yield server
    finally:
        server.shutdown()


def test_echo_adapter_replies_with_latest_user_text() -> None:
    adapter = EchoModelAdapter()
    request = ModelRequest(
        instruction="hello there",
        system_prompt="",
        tools=(),
        messages=({"role": "user", "content": "hello there"},),
    )
    turn = adapter.next_turn(request)
    assert turn.final_text
    assert "hello there" in turn.final_text
    assert turn.tool_calls == ()
    assert turn.usage["total_tokens"] > 0


def test_offline_chat_produces_assistant_reply(studio: StudioServer) -> None:
    result = studio.start_chat("summarize the workspace")
    run_id = result["run_id"]
    settled = _wait_settled(studio, run_id, 1)
    assert len(settled) == 1
    assert settled[0]["data"]["final_text"]


def test_multi_turn_session_yields_a_reply_per_message(studio: StudioServer) -> None:
    run_id = studio.start_chat("first")["run_id"]
    assert len(_wait_settled(studio, run_id, 1)) == 1
    studio.continue_chat(run_id, "second")
    assert len(_wait_settled(studio, run_id, 2)) == 2
    studio.continue_chat(run_id, "third")
    assert len(_wait_settled(studio, run_id, 3)) == 3
    # The session stays open for the next message rather than going terminal.
    assert studio.run_status(run_id)["status"] not in {"completed", "failed", "limited"}


def test_run_tokens_are_not_exposed_to_callers(studio: StudioServer) -> None:
    # The BFF holds run tokens server-side; start_chat returns only the run id + status.
    result = studio.start_chat("hello")
    assert set(result) == {"run_id", "status"}
    assert "run_token" not in result


# --- R1: read tools, file tree, activity feed -------------------------------------------


def test_runtime_config_binds_read_and_write() -> None:
    config = _agent_runtime_config()
    refs = {binding.ref.tool_id for binding in config.tools}
    assert {"fs.read", "fs.write"} <= refs
    # The model-facing name is the dotted id sanitized to underscores.
    read = next(b for b in config.tools if b.ref.tool_id == "fs.read")
    assert read.model_name == "fs_read"


def test_describe_event_maps_tool_activity_to_human_text() -> None:
    started = {
        "type": "tool.call.started",
        "data": {"tool": "fs_read", "args_preview": {"path": "notes.md"}, "paths": ["notes.md"]},
    }
    assert describe_event(started) == "Reading notes.md"
    # A successful finish is implied by the next step — not shown.
    assert describe_event({"type": "tool.call.finished", "data": {"tool": "fs_read", "ok": True}}) is None
    # A failure surfaces with its error.
    failed = describe_event(
        {"type": "tool.call.finished", "data": {"tool": "fs_read", "ok": False, "error": "boom"}}
    )
    assert failed is not None and "boom" in failed
    # Chat / lifecycle events do not appear in the activity feed.
    assert describe_event({"type": "turn.settled", "data": {"final_text": "hi"}}) is None
    assert describe_event({"type": "run.started", "data": {}}) is None


def test_list_files_returns_workspace_tree(studio: StudioServer) -> None:
    (studio.workspace / "notes.md").write_text("hello\n", encoding="utf-8")
    (studio.workspace / "sub").mkdir(exist_ok=True)
    (studio.workspace / "sub" / "inner.txt").write_text("x\n", encoding="utf-8")
    paths = {entry["path"] for entry in studio.list_files()}
    assert "notes.md" in paths
    assert "sub/inner.txt" in paths


def test_agent_reads_a_file_and_emits_activity(tmp_path: Path) -> None:
    # End-to-end with a tool-calling fake model injected via the provider seam: the agent calls
    # fs.read, the read flows through as events, and describe_event renders an activity line.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "notes.md").write_text("the secret is 42\n", encoding="utf-8")

    fake = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("fs_read", {"path": "notes.md"}, "c1"),)),
            ModelTurn(final_text="I read notes.md for you."),
        ]
    )
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    server.start()
    try:
        run_id = server.start_chat("read notes.md")["run_id"]
        _wait_settled(server, run_id, 1)
        events = server.poll_events(run_id, 0).get("events", [])
        started = [e for e in events if e.get("type") == "tool.call.started"]
        assert any(e["data"].get("tool") == "fs_read" for e in started)
        assert any("notes.md" in (describe_event(e) or "") for e in started)
        # The model's final text settles the turn.
        settled = [e for e in events if e.get("type") == "turn.settled"]
        assert settled and "notes.md" in settled[0]["data"]["final_text"]
    finally:
        server.shutdown()


# --- R2: write, proposal/diff, approve & apply ------------------------------------------


def _wait_proposal(server: StudioServer, run_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        proposal = server.proposal(run_id)
        if proposal.get("ready") and proposal.get("diff"):
            return proposal
        time.sleep(0.1)
    return server.proposal(run_id)


def test_agent_write_is_staged_then_applied(tmp_path: Path) -> None:
    # The propose->apply loop: the agent writes a file (staged in the overlay, not on disk),
    # Studio surfaces it as a diff, and apply materializes it into the workspace.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    target = workspace / "OUT.md"

    fake = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("fs_write", {"path": "OUT.md", "content": "hello\n"}, "c1"),)),
            ModelTurn(final_text="Wrote OUT.md."),
        ]
    )
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    server.start()
    try:
        run_id = server.start_chat("create OUT.md")["run_id"]
        _wait_settled(server, run_id, 1)

        # Staged, not yet on disk.
        assert not target.exists()
        proposal = _wait_proposal(server, run_id)
        assert proposal["ready"]
        assert "OUT.md" in proposal["diff"]
        assert "hello" in proposal["diff"]

        # Approve & apply -> the file lands in the workspace.
        result = server.apply(run_id)
        assert result["status"] != "conflict"
        assert "OUT.md" in str(result.get("applied_paths"))
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "hello\n"
    finally:
        server.shutdown()
