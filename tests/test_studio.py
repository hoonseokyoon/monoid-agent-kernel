"""R0 coverage for the Agent Studio reference app.

Studio is a reference example, but it is the pressure test for "build an app from the surface
alone", so it gets the same regression coverage the other reference services have. These tests
drive the Studio server through its Python API (no browser / no Chromium window) against the
offline echo model, so they are deterministic and key-less.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from native_agent_runner.errors import ModelAdapterError, NativeAgentError
from native_agent_runner.providers.base import ModelRequest, ModelTurn, TextDelta, TurnComplete
from native_agent_runner.providers.fake import (
    FakeModelAdapter,
    FakeStreamingModelAdapter,
    fake_tool_call,
)
from native_agent_runner.reference.llm_gateway.providers import EchoModelAdapter
from native_agent_runner.reference.studio.activity import describe_event
from native_agent_runner.reference.studio.server import (
    _ALL_CAPABILITIES,
    StudioConfig,
    StudioServer,
    _agent_runtime_config,
    _runtime_config_for,
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


def test_runtime_config_binds_read_write_hitl_shell_and_web() -> None:
    config = _agent_runtime_config()
    refs = {binding.ref.tool_id for binding in config.tools}
    assert {"fs.read", "fs.write", "hitl.request", "shell.exec", "web.search", "web.context"} <= refs
    # The plan tool is always bound (observability) and the prompt nudges its use.
    assert "run.update_plan" in refs
    assert config.prompt.system_prompt_base and "run_update_plan" in config.prompt.system_prompt_base
    # The shell binding refuses obviously destructive commands and auto-approves the rest.
    shell = next(b for b in config.tools if b.ref.tool_id == "shell.exec")
    assert any(p.startswith("rm") for p in shell.scope.command_deny_prefixes)
    assert shell.runtime["shell"]["approval_mode"] == "auto-approve"
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


def test_partial_approval_applies_only_selected_paths(tmp_path: Path) -> None:
    # R9: the per-file approval gate. The agent stages two files; Studio approves only one, so
    # apply writes that file and reports the other as skipped (never touching disk for it).
    workspace = tmp_path / "ws"
    workspace.mkdir()

    fake = FakeModelAdapter(
        turns=[
            ModelTurn(
                tool_calls=(
                    fake_tool_call("fs_write", {"path": "KEEP.md", "content": "keep\n"}, "c1"),
                    fake_tool_call("fs_write", {"path": "DROP.md", "content": "drop\n"}, "c2"),
                )
            ),
            ModelTurn(final_text="Wrote two files."),
        ]
    )
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    server.start()
    try:
        run_id = server.start_chat("stage two files")["run_id"]
        _wait_settled(server, run_id, 1)
        _wait_proposal(server, run_id)

        result = server.apply(run_id, approved_paths=("KEEP.md",))
        assert result["status"] != "conflict"
        assert "KEEP.md" in str(result.get("applied_paths"))
        assert "DROP.md" in str(result.get("skipped_paths"))
        assert (workspace / "KEEP.md").exists()
        assert not (workspace / "DROP.md").exists()
    finally:
        server.shutdown()


def test_export_package_returns_digest_receipt_fetched_as_bytes(tmp_path: Path) -> None:
    # R9 (fundamental): export returns a RECEIPT (digest) — never a server path — and the bytes are
    # fetched back by digest through the data-returning seam (works co-located or remote).
    import io
    import tarfile

    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("fs_write", {"path": "OUT.md", "content": "hi\n"}, "c1"),)),
            ModelTurn(final_text="done"),
        ]
    )
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    server.start()
    try:
        run_id = server.start_chat("write OUT.md")["run_id"]
        _wait_settled(server, run_id, 1)
        _wait_proposal(server, run_id)

        receipt = server.export_package(run_id)
        # The receipt is a handle, not a path — no run_dir path leaks across the boundary.
        assert "package_path" not in receipt
        assert len(receipt["digest"]) == 64
        assert receipt["size_bytes"] > 0

        data, name = server.read_artifact(run_id, receipt["digest"])
        # The fetched bytes match the digest (content-addressed self-verification).
        import hashlib

        assert hashlib.sha256(data).hexdigest() == receipt["digest"]
        assert name.endswith(".tar")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r") as archive:
            names = archive.getnames()
        assert "proposal.package.json" in names
        assert any(n == "proposal.json" for n in names)

        # A malformed digest is rejected (ValueError → 400); an unknown well-formed one is
        # not-found (KeyError → 404).
        with pytest.raises(ValueError):
            server.read_artifact(run_id, "not-a-digest")
        with pytest.raises(KeyError):
            server.read_artifact(run_id, "f" * 64)
    finally:
        server.shutdown()


def test_continue_chat_resumes_a_parked_session_after_restart(tmp_path: Path) -> None:
    # The studio "continue an old chat" path: a multi-turn session parked awaiting input is
    # evicted from memory (simulating a process restart), then continue_chat transparently
    # resumes it from the checkpoint and delivers the follow-up.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", final_text="first"),
            ModelTurn(response_id="r2", final_text="second"),
        ]
    )
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    server.start()
    try:
        run_id = server.start_chat("hello")["run_id"]
        _wait_settled(server, run_id, 1)

        def _await_status(target: str, timeout: float = 10.0) -> bool:
            deadline = time.time() + timeout
            while time.time() < deadline:
                if server.run_status(run_id).get("status") == target:
                    return True
                time.sleep(0.05)
            return False

        assert _await_status("awaiting_input")

        # Simulate a restart: drop the in-memory record. A bare send_message would now KeyError;
        # continue_chat must resume from the durable checkpoint first.
        backend = server._backend
        assert backend.checkpoint_store.latest(run_id) is not None
        with backend._lock:
            backend._records.pop(run_id)

        result = server.continue_chat(run_id, "again")
        assert result["status"] == "queued"

        # The resumed session threads the follow-up as a real second model turn, with the
        # conversation rebuilt from the checkpoint (user "hello" → assistant "first" → user "again").
        def _again_threaded() -> bool:
            return any(
                msg.get("role") == "user" and msg.get("content") == "again"
                for req in fake.requests
                for msg in (req.messages or [])
            )

        deadline = time.time() + 10.0
        while time.time() < deadline and not _again_threaded():
            time.sleep(0.05)
        assert _again_threaded()
        # And the prior turn's assistant reply survived the restart (proves checkpoint restore, not
        # a fresh conversation).
        assert any(
            msg.get("role") == "assistant" and msg.get("content") == "first"
            for req in fake.requests
            for msg in (req.messages or [])
        )
    finally:
        server.shutdown()


def test_start_chat_attaches_image_and_forwards_resolved_block(tmp_path: Path) -> None:
    # R13: an attached image is persisted under the workspace and forwarded to a multimodal
    # adapter as a resolved base64 block (the loop resolves the by-reference source_ref).
    import base64

    from native_agent_runner.providers.fake import FakeMultimodalModelAdapter

    png_1x1 = base64.b64encode(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk"
            "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
    ).decode("ascii")

    workspace = tmp_path / "ws"
    workspace.mkdir()
    adapter = FakeMultimodalModelAdapter(turns=[ModelTurn(final_text="I see a 1x1 image.")])
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: adapter,
    )
    server.start()
    try:
        result = server.start_chat(
            "what is this?",
            [{"name": "pic.png", "mime": "image/png", "data_b64": png_1x1}],
        )
        run_id = result["run_id"]
        _wait_settled(server, run_id, 1)

        # Inline ingress: the studio writes NO attachment file into the workspace — the bytes ride
        # a data: URI and the core normalizes them to a content-addressed blob.
        assert not (workspace / ".studio-attachments").exists()

        # The adapter received a resolved base64 image block on the user turn.
        def _image_forwarded() -> bool:
            return any(
                isinstance(msg.get("content"), list)
                and any(isinstance(b, dict) and b.get("type") == "image" for b in msg["content"])
                for req in adapter.requests
                for msg in req.messages
            )

        assert _image_forwarded()
    finally:
        server.shutdown()


# --- R3: human-in-the-loop approval gate ------------------------------------------------


def _wait_event(server: StudioServer, run_id: str, etype: str, timeout: float = 10.0) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for event in server.poll_events(run_id, 0).get("events", []):
            if event.get("type") == etype:
                return event
        time.sleep(0.1)
    return None


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


# --- R4: shell + background jobs --------------------------------------------------------


def _python_command(code: str) -> str:
    return 'python -c "' + code.replace('"', '\\"') + '"'


def _shell_studio(tmp_path: Path, turns: list) -> StudioServer:
    fake = FakeModelAdapter(turns=turns)
    server = StudioServer(
        StudioConfig(workspace=tmp_path / "ws", host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
    server.start()
    return server


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


# --- R5: web tools (through the bundled WebGateway) -------------------------------------


def test_web_search_runs_through_the_gateway(tmp_path: Path) -> None:
    fake = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("web_search", {"query": "native agent runner"}, "c1"),)),
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


# --- R6: settings window + live Agent-spec editing --------------------------------------


def test_runtime_config_for_subset() -> None:
    # run.update_plan is always bound (observability); capability toggles add the rest.
    refs = {b.ref.tool_id for b in _runtime_config_for(["read"]).tools}
    assert refs == {"fs.read", "run.update_plan"}
    assert {b.ref.tool_id for b in _runtime_config_for([]).tools} == {"run.update_plan"}


def test_settings_lists_capabilities_and_provider(tmp_path: Path) -> None:
    server = StudioServer(StudioConfig(workspace=tmp_path / "ws", provider="offline"))  # not started
    s = server.settings()
    assert s["provider"] == "offline" and s["offline"] is True
    assert set(s["capabilities"]) == set(_ALL_CAPABILITIES)
    assert {a["key"] for a in s["available"]} == set(_ALL_CAPABILITIES)


def test_read_file_returns_workspace_content(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "notes.md").write_bytes(b"# hi\nbody")  # bytes: avoid Windows newline translation
    server = StudioServer(StudioConfig(workspace=ws, provider="offline"))
    r = server.read_file("notes.md")
    assert r["binary"] is False and r["truncated"] is False
    assert r["content"] == "# hi\nbody"


def test_read_file_rejects_traversal(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
    server = StudioServer(StudioConfig(workspace=ws, provider="offline"))
    with pytest.raises(NativeAgentError):
        server.read_file("../secret.txt")


def test_read_file_flags_binary(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "blob.bin").write_bytes(b"PK\x03\x04\x00\x00binary")
    server = StudioServer(StudioConfig(workspace=ws, provider="offline"))
    r = server.read_file("blob.bin")
    assert r["binary"] is True and r["content"] == ""


def test_settings_change_applies_to_new_chats(studio: StudioServer) -> None:
    studio.update_settings(capabilities=["read"])
    run_id = studio.start_chat("hi")["run_id"]
    config = studio._backend.current_runtime_config(run_id)  # type: ignore[union-attr]
    assert {b.ref.tool_id for b in config.tools} == {"fs.read", "run.update_plan"}


def test_settings_model_and_effort_apply_to_new_chats(studio: StudioServer) -> None:
    studio.update_settings(model="gpt-x-test", effort="high")
    run_id = studio.start_chat("hi")["run_id"]
    config = studio._backend.current_runtime_config(run_id)  # type: ignore[union-attr]
    assert config.model is not None
    assert config.model.model == "gpt-x-test"
    assert config.model.reasoning.effort == "high"


def test_settings_reasoning_summary_visibility(studio: StudioServer) -> None:
    # DX-13b: the "Thinking" (reasoning summary) toggle is a real setting that flows into the
    # runtime config so the model returns a displayable summary.
    s = studio.settings()
    assert s["summary"] == "auto" and "off" in s["summaries"]
    studio.update_settings(summary="off")
    run_id = studio.start_chat("hi")["run_id"]
    config = studio._backend.current_runtime_config(run_id)  # type: ignore[union-attr]
    assert config.model is not None
    assert config.model.reasoning.summary == "off"


def test_otel_toggle_attaches_per_run_sink_factory(studio: StudioServer, monkeypatch: pytest.MonkeyPatch) -> None:
    # Tier-3: toggling OTel attaches a per-run OtelEventSink FACTORY (not a shared instance) on
    # the backend; toggling off detaches it. Provider setup is stubbed (real SDK = Jaeger live test).
    import native_agent_runner.reference.studio.server as srv
    from native_agent_runner.observability.otel import OtelEventSink

    monkeypatch.setattr(srv, "_ensure_otel_provider", lambda endpoint: None)
    assert studio.settings()["otel"] is False
    studio.update_settings(otel=True)
    assert studio.settings()["otel"] is True
    assert studio._backend.extra_event_sink_factories == (OtelEventSink,)  # type: ignore[union-attr]
    studio.update_settings(otel=False)
    assert studio._backend.extra_event_sink_factories == ()  # type: ignore[union-attr]


def test_settings_hot_swaps_active_session(studio: StudioServer) -> None:
    run_id = studio.start_chat("hello")["run_id"]
    _wait_settled(studio, run_id, 1)  # turn settles -> session active (awaiting input), not terminal
    before = studio._backend.current_runtime_config(run_id)  # type: ignore[union-attr]
    assert "fs.write" in {b.ref.tool_id for b in before.tools}
    result = studio.update_settings(capabilities=["read"])
    assert result["applied_runs"] == 1
    after = studio._backend.current_runtime_config(run_id)  # type: ignore[union-attr]
    assert {b.ref.tool_id for b in after.tools} == {"fs.read", "run.update_plan"}
    assert after.config_version > before.config_version


# --- recoverable turn errors (studio surfaces turn.failed; session stays alive) ---------


class _RaiseThenAdapter:
    """A gateway provider that raises scripted exceptions, otherwise returns turns."""

    def __init__(self, script: list) -> None:
        self.script = list(script)

    def next_turn(self, request):  # noqa: ANN001
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


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


# --- Tier 1: usage events + cancel (Stop) ----------------------------------------------


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


# --- DX-9: turn-level interrupt (Stop keeps the session alive) --------------------------


class _BlockingThenToolAdapter:
    """Turn 1 calls a tool; turn 2 blocks until released (giving the test a window to call
    interrupt_chat while a turn is in flight), then calls a tool so the next step boundary
    trips the interrupt. Turn 3 (after resume) settles."""

    def __init__(self) -> None:
        self.calls = 0
        self.reached_block = threading.Event()
        self.release = threading.Event()

    def next_turn(self, request):  # noqa: ANN001
        self.calls += 1
        if self.calls == 1:
            return ModelTurn(response_id="r1", tool_calls=(fake_tool_call("fs_read", {"path": "notes.md"}, "c1"),))
        if self.calls == 2:
            self.reached_block.set()
            self.release.wait(5.0)
            return ModelTurn(response_id="r2", tool_calls=(fake_tool_call("fs_read", {"path": "notes.md"}, "c2"),))
        return ModelTurn(response_id="r3", final_text="resumed ok")


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


# --- DX-8: live token streaming (model.output.delta over the event stream) --------------


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


# --- Tier-2: Plan panel (run.update_plan -> plan.updated) -------------------------------


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


# --- Tier-2: subagents (agent.spawn) + streaming child work ----------------------------


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


# --- R8: session history --------------------------------------------------------------


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
