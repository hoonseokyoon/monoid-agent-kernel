from __future__ import annotations

from support.studio_harness import (
    FakeModelAdapter,
    ModelTurn,
    Path,
    StudioConfig,
    StudioServer,
    _mcp_studio,
    _wait_settled,
    describe_event,
    fake_tool_call,
    pytest,
)

pytestmark = pytest.mark.integration


def test_skills_capability_offered_and_enabled_by_default(studio: StudioServer) -> None:
    settings = studio.settings()
    assert "skills" in settings["capabilities"]  # enabled by default (provider attached)
    assert any(item["key"] == "skills" for item in settings["available"])
    # The built config binds the skill tool.
    refs = {b.ref.tool_id for b in studio._build_config().tools}
    assert "skill" in refs


def test_bundled_review_skill_can_read_its_checklist(studio: StudioServer) -> None:
    assert studio._skill_provider is not None
    definition = studio._skill_provider.subagent_definitions()["skill:code-review-checklist"]
    assert "skill.read_file" in definition.tools


def test_skill_catalog_injected_into_system_prompt(tmp_path: Path) -> None:
    # The L1 catalog (skill name + description) is a context-provider static segment, so it must
    # reach the model's system prompt — proving the SkillProvider is wired as a context provider.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = FakeModelAdapter(turns=[ModelTurn(final_text="ok")])
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    server.start()
    try:
        run_id = server.start_chat("hi")["run_id"]
        _wait_settled(server, run_id, 1)
        assert fake.requests, "the model was never called"
        assert "commit-message" in fake.requests[0].system_prompt
    finally:
        server.shutdown()


def test_skill_activation_flows_through_studio(tmp_path: Path) -> None:
    # A `skill` tool call issued by the model executes (provider attached as a tool provider) and
    # surfaces the typed skill.activated signal in the studio event stream.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("skill", {"name": "commit-message"}, "c1"),)),
            ModelTurn(final_text="loaded the commit-message skill"),
        ]
    )
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    server.start()
    try:
        run_id = server.start_chat("write a commit message")["run_id"]
        _wait_settled(server, run_id, 1)
        events = server.poll_events(run_id, 0).get("events", [])
        started = [e for e in events if e.get("type") == "tool.call.started"]
        assert any(e["data"].get("tool") == "skill" for e in started)
        assert any(e.get("type") == "skill.activated" for e in events)
    finally:
        server.shutdown()


def test_skills_capability_hot_swaps_off_and_on(studio: StudioServer) -> None:
    # Toggling "skills" off then on removes/restores the skill tool binding; the config stays
    # valid across the hot-swap (the provider-tool validation extension covers replace_runtime_config).
    base = [c for c in studio.settings()["capabilities"] if c != "skills"]
    studio.update_settings(capabilities=base)  # skills OFF
    assert "skills" not in studio.settings()["capabilities"]
    assert "skill" not in {b.ref.tool_id for b in studio._build_config().tools}

    studio.update_settings(capabilities=base + ["skills"])  # skills ON
    assert "skills" in studio.settings()["capabilities"]
    assert "skill" in {b.ref.tool_id for b in studio._build_config().tools}


def test_no_skills_directory_is_a_clean_no_op(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs", skills_directory=None)
    )
    server.start()
    try:
        assert server._skill_provider is None
        settings = server.settings()
        assert "skills" not in settings["capabilities"]
        assert all(item["key"] != "skills" for item in settings["available"])
        assert "skill" not in {b.ref.tool_id for b in server._build_config().tools}
    finally:
        server.shutdown()


def test_mcp_boot_discovers_tools_before_serving(tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    server = _mcp_studio(tmp_path)
    server.start()
    try:
        assert server._mcp_provider is not None
        settings = server.settings()
        assert "mcp" in settings["capabilities"]
        assert any(item["key"] == "mcp" for item in settings["available"])
        # Discovery ran at boot (not lazily): the config already binds the discovered tools.
        refs = {b.ref.tool_id for b in server._build_config().tools}
        assert "mcp.studio.echo" in refs and "mcp.studio.uppercase" in refs
    finally:
        server.shutdown()


def test_builtin_profiles_include_provider_capabilities_when_saved(tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    server = _mcp_studio(tmp_path)
    server.start()
    try:
        profiles = {profile["id"]: profile for profile in server.profiles()["profiles"]}
        default_profile = dict(profiles["default"])
        assert {"skills", "mcp"} <= set(default_profile["capabilities"])

        default_profile["description"] = "Edited default profile."
        saved = server.save_profile(default_profile)["profile"]
        assert {"skills", "mcp"} <= set(saved["capabilities"])
        assert {"skill", "mcp.studio.echo"} <= {binding.ref.tool_id for binding in server._build_config("default").tools}
    finally:
        server.shutdown()


def test_saved_profile_preserves_unavailable_provider_capabilities(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    with_mcp = _mcp_studio(tmp_path)
    with_mcp.start()
    try:
        profile = with_mcp.save_profile(
            {
                "id": "researcher",
                "name": "Researcher",
                "capabilities": ["read", "mcp"],
            }
        )["profile"]
        assert "mcp" in profile["capabilities"]
    finally:
        with_mcp.shutdown()

    without_mcp = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs", mcp=False)
    )
    without_mcp.start()
    try:
        profile = {item["id"]: item for item in without_mcp.profiles()["profiles"]}["researcher"]
        assert "mcp" in profile["capabilities"]
        saved = without_mcp.save_profile(
            {
                "id": "researcher",
                "name": "Researcher edited",
                "capabilities": ["read"],
            }
        )["profile"]
        assert "mcp" in saved["capabilities"]
    finally:
        without_mcp.shutdown()

    with_mcp_again = _mcp_studio(tmp_path)
    with_mcp_again.start()
    try:
        config = with_mcp_again._build_config("researcher")
        assert "mcp.studio.echo" in {binding.ref.tool_id for binding in config.tools}
    finally:
        with_mcp_again.shutdown()


def test_mcp_tool_call_flows_end_to_end(tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    fake = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("mcp_studio_echo", {"text": "hi"}, "c1"),)),
            ModelTurn(final_text="called the MCP echo tool"),
        ]
    )
    server = _mcp_studio(tmp_path, provider_factory=lambda _claims, _config: fake)
    server.start()
    try:
        run_id = server.start_chat("use the echo tool")["run_id"]
        _wait_settled(server, run_id, 1)
        events = server.poll_events(run_id, 0).get("events", [])
        started = [e for e in events if e.get("type") == "tool.call.started"]
        assert any(e["data"].get("tool") == "mcp_studio_echo" for e in started)
        # The tool ran against the real fake MCP server (loopback), then the turn settled.
        assert [e for e in events if e.get("type") == "turn.settled"]
    finally:
        server.shutdown()


def test_mcp_catalog_context_reaches_model_when_helpers_bound(tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    fake = FakeModelAdapter(turns=[ModelTurn(final_text="ok")])
    server = _mcp_studio(tmp_path, provider_factory=lambda _claims, _config: fake)
    server.start()
    try:
        run_id = server.start_chat("what MCP context is available?")["run_id"]
        _wait_settled(server, run_id, 1)
        assert fake.requests, "the model was never called"
        assert "fake://studio/guide" in fake.requests[0].system_prompt
        assert "summarize" in fake.requests[0].system_prompt
    finally:
        server.shutdown()


def test_mcp_discovery_failure_degrades_to_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # If the MCP server can't be reached at boot, studio must come up Skills-only — never crash.
    import monoid_agent_kernel.reference.studio.server as server_mod

    def _never_ready(_base_url: str, *, timeout_s: float = 10.0) -> None:
        raise TimeoutError("simulated: MCP server never became ready")

    monkeypatch.setattr(server_mod, "wait_http_ready", _never_ready)
    (tmp_path / "ws").mkdir()
    server = _mcp_studio(tmp_path)
    server.start()  # must not raise
    try:
        assert server._mcp_provider is None
        assert "mcp" not in server.settings()["capabilities"]
        # The app still works (a chat settles) without MCP.
        run_id = server.start_chat("hello")["run_id"]
        assert _wait_settled(server, run_id, 1)
    finally:
        server.shutdown()


def test_mcp_shutdown_closes_provider_and_server(tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    server = _mcp_studio(tmp_path)
    server.start()
    thread = server._mcp_thread
    assert thread is not None and thread.is_alive()
    server.shutdown()
    thread.join(timeout=5)
    assert not thread.is_alive()  # the fake MCP server thread was stopped


def test_mcp_capability_hot_swaps_off_and_on(tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    server = _mcp_studio(tmp_path)
    server.start()
    try:
        base = [c for c in server.settings()["capabilities"] if c != "mcp"]
        server.update_settings(capabilities=base)  # mcp OFF
        assert "mcp.studio.echo" not in {b.ref.tool_id for b in server._build_config().tools}
        server.update_settings(capabilities=base + ["mcp"])  # mcp ON
        assert "mcp.studio.echo" in {b.ref.tool_id for b in server._build_config().tools}
    finally:
        server.shutdown()


def test_capabilities_catalog_lists_skills_and_mcp_tools(tmp_path: Path) -> None:
    (tmp_path / "ws").mkdir()
    server = _mcp_studio(tmp_path)  # skills (bundled) + mcp both attached
    server.start()
    try:
        catalog = server.capabilities_catalog()
        assert any(s["name"] == "commit-message" for s in catalog["skills"])
        assert {t["id"] for t in catalog["mcp_tools"]} >= {"mcp.studio.echo", "mcp.studio.uppercase"}
        assert {r["uri"] for r in catalog["mcp_resources"]} == {"fake://studio/guide"}
        assert {p["name"] for p in catalog["mcp_prompts"]} == {"summarize"}
    finally:
        server.shutdown()


def test_describe_event_surfaces_skill_and_mcp_tool() -> None:
    skill = {"type": "tool.call.started", "data": {"tool": "skill", "args_preview": {"name": "commit-message"}}}
    assert describe_event(skill) == "Using skill: commit-message"
    mcp = {"type": "tool.call.started", "data": {"tool": "mcp_studio_echo", "args_preview": {"text": "hi"}}}
    assert describe_event(mcp) == "Calling MCP tool: mcp_studio_echo"


def test_skill_catalog_disappears_on_capability_hot_swap(tmp_path: Path) -> None:
    # DX-17: the L1 skill catalog is a config-gated per-turn segment, so toggling "skills" off
    # mid-run removes it from the NEXT turn's system prompt (not just the skill tool).
    workspace = tmp_path / "ws"
    workspace.mkdir()
    fake = FakeModelAdapter(turns=[ModelTurn(final_text="one"), ModelTurn(final_text="two")])
    server = StudioServer(
        StudioConfig(workspace=workspace, host="127.0.0.1", port=0, run_root=tmp_path / "runs"),
        provider_factory=lambda _claims, _config: fake,
    )
    server.start()
    try:
        run_id = server.start_chat("first")["run_id"]
        _wait_settled(server, run_id, 1)
        assert "commit-message" in fake.requests[0].system_prompt  # skills on → catalog present

        base = [c for c in server.settings()["capabilities"] if c != "skills"]
        server.update_settings(capabilities=base)  # hot-swap skills OFF
        server.continue_chat(run_id, "second")
        _wait_settled(server, run_id, 2)
        assert "commit-message" not in fake.requests[1].system_prompt  # catalog gone next turn
    finally:
        server.shutdown()
