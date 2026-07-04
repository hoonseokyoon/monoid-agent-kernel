from __future__ import annotations

from support.studio_harness import (
    EchoModelAdapter,
    FakeModelAdapter,
    ModelRequest,
    ModelTurn,
    Path,
    StudioConfig,
    StudioServer,
    _agent_runtime_config,
    _wait_settled,
    describe_event,
    fake_tool_call,
    pytest,
)

pytestmark = pytest.mark.integration


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


def test_vendor_route_serves_katex_offline(studio: StudioServer) -> None:
    import urllib.error
    import urllib.request

    base = studio.base_url
    # The locally-vendored KaTeX CSS is served (no CDN) with a css content-type and still
    # references the woff2 fonts that ship alongside it.
    with urllib.request.urlopen(f"{base}/vendor/katex/katex.min.css") as resp:
        assert resp.status == 200
        assert "text/css" in resp.headers["Content-Type"]
        css = resp.read()
    assert b"@font-face" in css and b".woff2" in css

    # A woff2 font asset resolves too, with a font content-type.
    with urllib.request.urlopen(f"{base}/vendor/katex/fonts/KaTeX_Main-Regular.woff2") as resp:
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "font/woff2"

    # Unknown vendor paths 404 rather than leaking a stack trace or other files.
    try:
        urllib.request.urlopen(f"{base}/vendor/katex/nope.js")
    except urllib.error.HTTPError as exc:
        assert exc.code == 404
    else:
        raise AssertionError("missing vendor asset should 404")


def test_index_serves_onboarding_panel(studio: StudioServer) -> None:
    # The served UI includes the first-run onboarding empty-state + the sendPrompt hook its
    # suggested-prompt buttons call.
    import urllib.request

    with urllib.request.urlopen(f"{studio.base_url}/") as resp:
        html = resp.read().decode("utf-8")
    assert "#onboarding" in html  # the empty-state styles ship in the page
    assert "function showOnboarding" in html  # the panel is built on a fresh chat
    assert "function sendPrompt" in html  # suggested-prompt buttons call it
    # A failed run surfaces the provider error detail (shared by the run.failed + turn.failed paths).
    assert "function providerDetail" in html
    assert "provider_error_code" in html
    for hook in (
        'data-testid="studio-shell"',
        'data-testid="left-config-panel"',
        'data-testid="profile-switcher"',
        'data-testid="profile-add"',
        'data-testid="profile-list"',
        'data-testid="profile-editor-popup"',
        'class="profile-editor-panel profile-preview-panel"',
        'id="prompt-preview-system"',
        'id="prompt-preview-tools"',
        'id="prompt-preview-tool-count"',
        'id="prompt-preview-settings"',
        'id="prompt-preview-notes"',
        'data-testid="chat-log"',
        'data-testid="composer"',
        'data-testid="right-panel-tabs"',
        'data-testid="settings-config-popup"',
        'data-testid="capability-toggles"',
    ):
        assert hook in html
    # Saving a profile activates that profile and clears the current run so the next message does
    # not continue a session created under another profile.
    assert "activeProfileId = body.profile.id;" in html
    assert "runId = null;" in html
    assert "resetChatView();" in html
    assert "function refreshPromptPreview" in html
    assert 'fetch("/api/profile-preview"' in html


def test_settings_page_serves_static_test_hooks(studio: StudioServer) -> None:
    import urllib.request

    with urllib.request.urlopen(f"{studio.base_url}/settings") as resp:
        html = resp.read().decode("utf-8")
    assert 'data-testid="settings-popup"' in html
    assert 'data-testid="capability-toggles"' in html
    assert 'data-testid="capability-toggle-' in html


def test_offline_chat_produces_assistant_reply(studio: StudioServer) -> None:
    result = studio.start_chat("summarize the workspace")
    run_id = result["run_id"]
    settled = _wait_settled(studio, run_id, 1)
    assert len(settled) == 1
    assert settled[0]["data"]["final_text"]


def test_studio_profiles_can_be_saved_and_reloaded(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    run_root = tmp_path / "runs"
    server = StudioServer(StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=run_root))

    saved = server.save_profile(
        {
            "name": "Release Reviewer",
            "description": "Checks release risk.",
            "instructions": "Always call out release blockers.",
            "capabilities": ["read", "web"],
            "model": "gpt-profile",
            "effort": "high",
            "summary": "off",
        }
    )["profile"]

    assert saved["id"] == "release-reviewer"
    assert saved["capabilities"] == ["read", "web"]
    assert saved["built_in"] is False

    reloaded = StudioServer(StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=run_root))
    profiles = {profile["id"]: profile for profile in reloaded.profiles()["profiles"]}
    assert profiles["release-reviewer"]["instructions"] == "Always call out release blockers."
    assert profiles["release-reviewer"]["model"] == "gpt-profile"


def test_studio_start_chat_uses_selected_profile_runtime_config(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = FakeModelAdapter(turns=[ModelTurn(final_text="profile ok")])
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    server.start()
    try:
        profile = server.save_profile(
            {
                "name": "Read Only Reviewer",
                "description": "Focused read-only review.",
                "instructions": "Mention the profile sentinel.",
                "capabilities": ["read"],
                "model": "gpt-profile",
                "effort": "high",
                "summary": "off",
            }
        )["profile"]
        run_id = server.start_chat("use the profile", profile_id=profile["id"])["run_id"]
        _wait_settled(server, run_id, 1)
        request = fake.requests[0]
        tool_ids = {tool.id for tool in request.tools}
        assert "Mention the profile sentinel." in request.system_prompt
        assert request.model is not None
        assert request.model.model == "gpt-profile"
        assert request.model.reasoning.effort == "high"
        assert request.model.reasoning.summary == "off"
        assert {"run.update_plan", "fs.read"} <= tool_ids
        assert "fs.write" not in tool_ids
        assert "shell.exec" not in tool_ids
    finally:
        server.shutdown()


def test_studio_settings_hot_swap_preserves_run_profiles(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    server = StudioServer(StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"))
    server.start()
    try:
        profile = server.save_profile(
            {
                "name": "Read Only Reviewer",
                "description": "Focused read-only review.",
                "instructions": "PROFILE_SENTINEL",
                "capabilities": ["read"],
                "model": "gpt-profile",
                "effort": "high",
                "summary": "off",
            }
        )["profile"]
        server._run_tokens["profile-run"] = "profile-token"
        server._remember_run_profile("profile-run", profile["id"])
        server._run_tokens["default-run"] = "default-token"
        server._remember_run_profile("default-run", "default")

        current_by_run = {
            "profile-run": server._build_config(profile["id"]),
            "default-run": server._build_config("default"),
        }
        replaced: dict[str, object] = {}

        def current_runtime_config(run_id: str):
            return current_by_run[run_id]

        def replace_runtime_config(run_id: str, _token: str, **kwargs):
            replaced[run_id] = kwargs["config"]

        assert server._backend is not None
        server._backend.current_runtime_config = current_runtime_config  # type: ignore[method-assign]
        server._backend.replace_runtime_config = replace_runtime_config  # type: ignore[method-assign]

        all_caps = [item["key"] for item in server.settings()["available"]]
        server.update_settings(capabilities=all_caps, model="gpt-global", effort="low", summary="auto")

        profile_config = replaced["profile-run"]
        default_config = replaced["default-run"]
        assert profile_config.model is not None
        assert profile_config.model.model == "gpt-profile"
        assert "PROFILE_SENTINEL" in profile_config.prompt.system_prompt_base
        assert {binding.ref.tool_id for binding in profile_config.tools} == {"run.update_plan", "fs.read"}
        assert default_config.model is not None
        assert default_config.model.model == "gpt-global"
        assert "fs.write" in {binding.ref.tool_id for binding in default_config.tools}
    finally:
        server.shutdown()


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


def test_subagent_events_uses_root_ancestor_token_for_nested_child(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs")
    )

    class FakeBackend:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str, int]] = []

        def descendant_events(
            self,
            parent_run_id: str,
            token: str,
            child_run_id: str,
            *,
            from_seq: int = 0,
        ) -> dict:
            self.calls.append((parent_run_id, token, child_run_id, from_seq))
            return {"events": [{"type": "child"}]}

    backend = FakeBackend()
    server._backend = backend  # type: ignore[assignment]
    server._run_tokens["run_parent"] = "root-token"

    result = server.subagent_events("run_parent.sub.task_1.sub.task_2", from_seq=7)

    assert result["events"] == [{"type": "child"}]
    assert backend.calls == [("run_parent", "root-token", "run_parent.sub.task_1.sub.task_2", 7)]
    assert server.subagent_events("run_parent")["events"] == []
    assert server.subagent_events("../secret.sub.task")["events"] == []


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
