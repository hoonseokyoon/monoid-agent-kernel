from __future__ import annotations

from support.studio_harness import (
    NativeAgentError,
    Path,
    StudioConfig,
    StudioServer,
    _ALL_CAPABILITIES,
    _runtime_config_for,
    _wait_settled,
    pytest,
)

pytestmark = pytest.mark.integration


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
    assert "memory" not in s["capabilities"]
    assert {a["key"] for a in s["available"]} == set(_ALL_CAPABILITIES) | {"memory"}


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


def test_memory_capability_is_available_but_disabled_by_default(studio: StudioServer) -> None:
    settings = studio.settings()
    assert "memory" not in settings["capabilities"]
    assert "memory" in {item["key"] for item in settings["available"]}

    profile = next(item for item in studio.profiles()["profiles"] if item["id"] == "default")
    assert "memory" not in profile["capabilities"]

    studio.update_settings(capabilities=["memory"])
    run_id = studio.start_chat("hi")["run_id"]
    config = studio._backend.current_runtime_config(run_id)  # type: ignore[union-attr]
    refs = {binding.ref.tool_id for binding in config.tools}
    assert {
        "memory.view",
        "memory.search",
        "memory.create",
        "memory.str_replace",
        "memory.insert",
        "memory.delete",
        "memory.rename",
    } <= refs


def test_settings_reasoning_summary_visibility(studio: StudioServer) -> None:
    # DX-13b: Studio exposes the reasoning summary as a checkbox: auto when checked, off when
    # unchecked. Detailed summary support stays provider/model-specific and is not a Studio mode.
    s = studio.settings()
    assert s["summary"] == "auto"
    assert s["summaries"] == ["off", "auto"]
    studio.update_settings(summary="off")
    run_id = studio.start_chat("hi")["run_id"]
    config = studio._backend.current_runtime_config(run_id)  # type: ignore[union-attr]
    assert config.model is not None
    assert config.model.reasoning.summary == "off"


def test_otel_toggle_attaches_per_run_sink_factory(studio: StudioServer, monkeypatch: pytest.MonkeyPatch) -> None:
    # Tier-3: toggling OTel attaches a per-run OtelEventSink FACTORY (not a shared instance) on
    # the backend; toggling off detaches it. Provider setup is stubbed (real SDK = Jaeger live test).
    import monoid_agent_kernel.reference.studio.server as srv
    from monoid_agent_kernel.observability.otel import OtelEventSink

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
