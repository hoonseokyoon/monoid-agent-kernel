from __future__ import annotations

import os
from pathlib import Path

import pytest

from support.backend_harness import (
    BackendRunRequest,
    FakeModelAdapter,
    ModelTurn,
    RunnerBackend,
    _token_manager,
    fake_tool_call,
    runtime_config,
)
from support.runtime import runtime_provider, tool_binding

from monoid_agent_kernel.core.agents import AgentRuntimeConfig, ToolBinding
from monoid_agent_kernel.core.context import TurnContext
from monoid_agent_kernel.core.lifecycle import SessionState
from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.core.tool_surface import ToolQuota
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.memory import (
    MEMORY_CREATE_TOOL_ID,
    MEMORY_SEARCH_TOOL_ID,
    MEMORY_STR_REPLACE_TOOL_ID,
    MEMORY_TOOL_IDS,
    MEMORY_VIEW_TOOL_ID,
    LocalFilesystemMemoryProvider,
    LocalFilesystemMemoryStore,
    MemoryProvider,
    MemoryToolError,
)


def test_local_memory_store_view_create_edit_insert_delete_and_rename(tmp_path: Path) -> None:
    store = LocalFilesystemMemoryStore(tmp_path / "memory")

    empty = store.view("/memories")
    assert empty["entries"][0]["path"] == "/memories"

    created = store.create("/memories/notes.md", "alpha\nbeta\n")
    assert created["status"] == "created"
    assert (tmp_path / "memory" / "notes.md").read_text(encoding="utf-8") == "alpha\nbeta\n"

    viewed = store.view("/memories/notes.md", (2, -1))
    assert viewed["lines"] == {"start": 2, "end": 2, "total": 2}
    assert "     2\tbeta" in viewed["content"]

    with pytest.raises(MemoryToolError) as bad_range:
        store.view("/memories/notes.md", (10, 20))
    assert bad_range.value.code == "memory_invalid_view_range"
    assert bad_range.value.retryable is True

    replaced = store.str_replace("/memories/notes.md", "beta", "gamma")
    assert replaced["status"] == "edited"
    assert "gamma" in (tmp_path / "memory" / "notes.md").read_text(encoding="utf-8")

    inserted = store.insert("/memories/notes.md", 1, "inserted\n")
    assert inserted["status"] == "edited"
    assert (tmp_path / "memory" / "notes.md").read_text(encoding="utf-8") == "alpha\ninserted\ngamma\n"

    renamed = store.rename("/memories/notes.md", "/memories/archive/notes.md")
    assert renamed["status"] == "renamed"
    assert (tmp_path / "memory" / "archive" / "notes.md").exists()

    deleted = store.delete("/memories/archive")
    assert deleted["status"] == "deleted"
    assert not (tmp_path / "memory" / "archive").exists()


def test_local_memory_store_rejects_recoverable_edit_errors(tmp_path: Path) -> None:
    store = LocalFilesystemMemoryStore(tmp_path / "memory")
    store.create("/memories/notes.md", "alpha\nbeta\nbeta\n")

    with pytest.raises(MemoryToolError) as not_found:
        store.str_replace("/memories/notes.md", "missing", "x")
    assert not_found.value.code == "memory_old_str_not_found"
    assert not_found.value.retryable is True

    with pytest.raises(MemoryToolError) as ambiguous:
        store.str_replace("/memories/notes.md", "beta", "x")
    assert ambiguous.value.code == "memory_ambiguous_replace"
    assert ambiguous.value.retryable is True

    with pytest.raises(MemoryToolError) as bad_line:
        store.insert("/memories/notes.md", 9, "x")
    assert bad_line.value.code == "memory_invalid_insert_line"

    with pytest.raises(MemoryToolError) as root_delete:
        store.delete("/memories")
    assert root_delete.value.code == "memory_root_operation_rejected"

    store.create("/memories/target.md", "target")
    with pytest.raises(MemoryToolError) as collision:
        store.rename("/memories/notes.md", "/memories/target.md")
    assert collision.value.code == "memory_destination_exists"

    with pytest.raises(MemoryToolError) as file_parent_create:
        store.create("/memories/notes.md/child.md", "child")
    assert file_parent_create.value.code == "memory_parent_not_directory"

    with pytest.raises(MemoryToolError) as file_parent_rename:
        store.rename("/memories/target.md", "/memories/notes.md/child.md")
    assert file_parent_rename.value.code == "memory_parent_not_directory"
    assert "target" in store.view("/memories/target.md")["content"]


def test_local_memory_store_rejects_traversal_binary_and_symlink_escape(tmp_path: Path) -> None:
    store = LocalFilesystemMemoryStore(tmp_path / "memory")
    with pytest.raises(MemoryToolError) as traversal:
        store.view("/memories/%2e%2e/secret.txt")
    assert traversal.value.code == "memory_path_traversal"

    (tmp_path / "memory" / "blob.bin").write_bytes(b"\x00binary")
    with pytest.raises(MemoryToolError) as binary:
        store.view("/memories/blob.bin")
    assert binary.value.code == "memory_unsupported_media"

    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, tmp_path / "memory" / "escape", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available in this environment")
    with pytest.raises(MemoryToolError) as escape:
        store.view("/memories/escape")
    assert escape.value.code == "memory_symlink_unsupported"

    target = tmp_path / "memory" / "target.md"
    target.write_text("safe\n", encoding="utf-8")
    link = tmp_path / "memory" / "target-link.md"
    os.symlink(target, link)
    with pytest.raises(MemoryToolError) as symlink:
        store.delete("/memories/target-link.md")
    assert symlink.value.code == "memory_symlink_unsupported"
    assert target.exists()
    assert link.is_symlink()


def test_local_memory_store_supports_multiple_mounts(tmp_path: Path) -> None:
    project = tmp_path / "project-memory"
    user = tmp_path / "user-memory"
    store = LocalFilesystemMemoryStore(
        mounts={
            "/memories/project": project,
            "/memories/user": user,
        }
    )

    store.create("/memories/project/progress.md", "done\n")
    store.create("/memories/user/preferences.md", "short answers\n")

    assert (project / "progress.md").read_text(encoding="utf-8") == "done\n"
    assert (user / "preferences.md").read_text(encoding="utf-8") == "short answers\n"
    root = store.view("/memories")
    assert {entry["path"] for entry in root["entries"]} >= {"/memories/project", "/memories/user"}


def test_local_memory_store_views_nested_mount_virtual_parents(tmp_path: Path) -> None:
    project = tmp_path / "team-project-memory"
    store = LocalFilesystemMemoryStore(mounts={"/memories/team/project": project})
    store.create("/memories/team/project/progress.md", "nested\n")

    root = store.view("/memories")
    assert {entry["path"] for entry in root["entries"]} == {"/memories", "/memories/team"}

    team = store.view("/memories/team")
    assert {entry["path"] for entry in team["entries"]} == {"/memories/team", "/memories/team/project"}

    viewed = store.view("/memories/team/project/progress.md")
    assert "     1\tnested" in viewed["content"]


def test_local_memory_store_rejects_destructive_mount_root_operations(tmp_path: Path) -> None:
    project = tmp_path / "project-memory"
    user = tmp_path / "user-memory"
    store = LocalFilesystemMemoryStore(
        mounts={
            "/memories/project": project,
            "/memories/user": user,
        }
    )
    store.create("/memories/project/progress.md", "done\n")
    store.create("/memories/user/preferences.md", "short answers\n")

    with pytest.raises(MemoryToolError) as mount_delete:
        store.delete("/memories/project")
    assert mount_delete.value.code == "memory_root_operation_rejected"
    assert (project / "progress.md").exists()

    with pytest.raises(MemoryToolError) as mount_rename:
        store.rename("/memories/user", "/memories/project/user")
    assert mount_rename.value.code == "memory_root_operation_rejected"
    assert (user / "preferences.md").exists()


def test_local_memory_store_searches_namespace_with_limit_and_filters(tmp_path: Path) -> None:
    store = LocalFilesystemMemoryStore(tmp_path / "memory")
    store.create("/memories/project/progress.md", "Alpha shipped\nbeta later\n")
    store.create("/memories/project/notes.txt", "alpha note\n")
    store.create("/memories/user/preferences.md", "alpha preference\n")
    store.create("/memories/project/skip.py", "alpha code\n")

    project = store.search("alpha", namespace="project", limit=10)
    assert project["operation"] == "search"
    assert project["namespace"] == "/memories/project"
    assert {match["path"] for match in project["matches"]} == {
        "/memories/project/progress.md",
        "/memories/project/notes.txt",
        "/memories/project/skip.py",
    }
    assert project["matches"][0]["line"] == 1

    filtered = store.search("alpha", namespace="/memories/project", filters={"file_glob": "*.md"})
    assert [match["path"] for match in filtered["matches"]] == ["/memories/project/progress.md"]

    limited = store.search("alpha", limit=1)
    assert limited["count"] == 1

    root_memory = tmp_path / "root-memory"
    nested_memory = tmp_path / "nested-memory"
    overlapping = LocalFilesystemMemoryStore(
        mounts={
            "/memories": root_memory,
            "/memories/project": nested_memory,
        }
    )
    overlapping.create("/memories/root.md", "alpha root\n")
    overlapping.create("/memories/project/progress.md", "alpha nested\n")
    (root_memory / "project").mkdir()
    (root_memory / "project" / "shadow.md").write_text("alpha shadow\n", encoding="utf-8")

    root_view = overlapping.view("/memories")
    assert {entry["path"] for entry in root_view["entries"]} == {
        "/memories",
        "/memories/root.md",
        "/memories/project",
    }
    project_view = overlapping.view("/memories/project")
    assert {entry["path"] for entry in project_view["entries"]} == {
        "/memories/project",
        "/memories/project/progress.md",
    }

    aggregate = overlapping.search("alpha")
    assert {match["path"] for match in aggregate["matches"]} == {
        "/memories/root.md",
        "/memories/project/progress.md",
    }


def test_memory_provider_tools_bindings_and_context_gate(tmp_path: Path) -> None:
    provider = LocalFilesystemMemoryProvider(tmp_path / "memory")
    specs = {spec.id: spec for spec in provider.get_tools(None)}

    assert set(specs) == set(MEMORY_TOOL_IDS)
    assert specs[MEMORY_SEARCH_TOOL_ID].capability == "memory.read"
    assert specs[MEMORY_SEARCH_TOOL_ID].side_effect == "read"
    assert specs[MEMORY_VIEW_TOOL_ID].capability == "memory.read"
    assert specs[MEMORY_VIEW_TOOL_ID].side_effect == "read"
    for tool_id in MEMORY_TOOL_IDS:
        if tool_id not in {MEMORY_SEARCH_TOOL_ID, MEMORY_VIEW_TOOL_ID}:
            assert specs[tool_id].capability == "memory.write"
            assert specs[tool_id].side_effect == "write"

    bindings = {binding.ref.tool_id: binding for binding in provider.tool_bindings()}
    assert bindings[MEMORY_SEARCH_TOOL_ID].authorization == "allow"
    assert bindings[MEMORY_VIEW_TOOL_ID].authorization == "allow"
    assert bindings[MEMORY_CREATE_TOOL_ID].authorization == "ask"

    provider.store.create("/memories/MEMORY.md", "## Index\n- progress.md\n")
    bounded_index = provider.store.startup_index(max_lines=10, max_bytes=len("## Index\n"))
    assert bounded_index is not None
    assert "## Index" in bounded_index
    assert "progress.md" not in bounded_index
    provider.store.str_replace("/memories/MEMORY.md", "## Index\n- progress.md\n", "## Index\n한글\n")
    multibyte_cut_index = provider.store.startup_index(max_lines=10, max_bytes=len("## Index\n한".encode("utf-8")) - 1)
    assert multibyte_cut_index is not None
    assert "## Index" in multibyte_cut_index
    assert "한" not in multibyte_cut_index
    turn_without_memory = TurnContext(1, 9, 20, None, (), 0, frozenset({"fs.read"}))
    assert provider.dynamic_segment(turn_without_memory) is None
    turn_with_write_only_memory = TurnContext(1, 9, 20, None, (), 0, frozenset({MEMORY_CREATE_TOOL_ID}))
    assert provider.dynamic_segment(turn_with_write_only_memory) is None
    turn_with_memory = TurnContext(1, 9, 20, None, (), 0, frozenset({MEMORY_VIEW_TOOL_ID}))
    segment = provider.dynamic_segment(turn_with_memory)
    assert segment is not None
    assert "Persistent memory is available under /memories" in segment
    assert "## Index" in segment

    turn_with_search_only = TurnContext(1, 9, 20, None, (), 0, frozenset({MEMORY_SEARCH_TOOL_ID}))
    search_segment = provider.dynamic_segment(turn_with_search_only)
    assert search_segment is not None
    assert "Persistent memory is available under /memories" in search_segment
    assert "## Index" not in search_segment

    approval_gated_provider = LocalFilesystemMemoryProvider(
        tmp_path / "ask-memory",
        read_authorization="ask",
    )
    approval_gated_provider.store.create("/memories/MEMORY.md", "## Ask Index\n- gated.md\n")
    gated_segment = approval_gated_provider.dynamic_segment(turn_with_memory)
    assert gated_segment is not None
    assert "Persistent memory is available under /memories" in gated_segment
    assert "## Ask Index" not in gated_segment


def _first_prompt_with_memory_bindings(
    tmp_path: Path,
    provider: LocalFilesystemMemoryProvider,
    bindings: tuple[ToolBinding, ...],
) -> str:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("run_finish", {"summary": "done"}, "c1"),),
            )
        ]
    )
    config = AgentRuntimeConfig(
        definition_id="memory-context-agent",
        tools=(*bindings, ToolBinding.for_tool("run.finish")),
    )
    AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
        tool_providers=(provider,),
        context_providers=(provider,),
    ).run_once("finish")
    return adapter.requests[0].system_prompt


def test_memory_context_requires_authorized_read_surface(tmp_path: Path) -> None:
    provider = LocalFilesystemMemoryProvider(tmp_path / "memory")
    provider.store.create("/memories/MEMORY.md", "## Secret Index\n- hidden.md\n")

    prompts = [
        _first_prompt_with_memory_bindings(
            tmp_path,
            provider,
            (ToolBinding.for_tool(MEMORY_CREATE_TOOL_ID),),
        ),
        _first_prompt_with_memory_bindings(
            tmp_path,
            provider,
            (ToolBinding.for_tool(MEMORY_VIEW_TOOL_ID, authorization="deny"),),
        ),
        _first_prompt_with_memory_bindings(
            tmp_path,
            provider,
            (ToolBinding.for_tool(MEMORY_SEARCH_TOOL_ID, exposure="hidden"),),
        ),
        _first_prompt_with_memory_bindings(
            tmp_path,
            provider,
            (ToolBinding.for_tool(MEMORY_VIEW_TOOL_ID, quota=ToolQuota(max_calls_per_run=0)),),
        ),
        _first_prompt_with_memory_bindings(
            tmp_path,
            provider,
            (ToolBinding.for_tool(MEMORY_VIEW_TOOL_ID, exposure="searchable"),),
        ),
    ]

    for prompt in prompts:
        assert "Persistent memory is available under /memories" not in prompt
        assert "## Secret Index" not in prompt

    ask_prompt = _first_prompt_with_memory_bindings(
        tmp_path,
        provider,
        (ToolBinding.for_tool(MEMORY_VIEW_TOOL_ID, authorization="ask"),),
    )
    assert "Persistent memory is available under /memories" in ask_prompt
    assert "## Secret Index" not in ask_prompt

    approval_gated_provider = LocalFilesystemMemoryProvider(
        tmp_path / "approval-gated-memory",
        read_authorization="ask",
    )
    approval_gated_provider.store.create("/memories/MEMORY.md", "## Allowed Override Index\n")
    allow_prompt = _first_prompt_with_memory_bindings(
        tmp_path,
        approval_gated_provider,
        (ToolBinding.for_tool(MEMORY_VIEW_TOOL_ID, authorization="allow"),),
    )
    assert "## Allowed Override Index" in allow_prompt


def test_memory_tool_results_return_structured_failures(tmp_path: Path) -> None:
    provider = LocalFilesystemMemoryProvider(tmp_path / "memory")
    tools = {spec.id: spec for spec in provider.get_tools(None)}

    result = tools[MEMORY_STR_REPLACE_TOOL_ID].handler(
        object(),
        {"path": "/memories/missing.md", "old_str": "x", "new_str": "y"},
    )

    assert result.ok is False
    assert result.error_code == "memory_path_not_found"
    assert result.retryable is True


def test_memory_search_tool_returns_matches(tmp_path: Path) -> None:
    provider = LocalFilesystemMemoryProvider(tmp_path / "memory")
    provider.store.create("/memories/project/progress.md", "ship memory.search\n")
    tools = {spec.id: spec for spec in provider.get_tools(None)}

    result = tools[MEMORY_SEARCH_TOOL_ID].handler(
        object(),
        {"query": "memory.search", "namespace": "project", "limit": 5},
    )

    assert result.ok is True
    assert result.content["matches"] == [
        {
            "path": "/memories/project/progress.md",
            "line": 1,
            "text": "ship memory.search",
            "snippet": "     1\tship memory.search",
        }
    ]


def test_memory_persists_across_loop_instances(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = MemoryProvider(
        LocalFilesystemMemoryStore(tmp_path / "memory"),
        write_authorization="allow",
    )
    memory_bindings = provider.tool_bindings()
    config = AgentRuntimeConfig(
        definition_id="memory-agent",
        tools=memory_bindings + (ToolBinding.for_tool("run.finish"),),
    )

    first = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call(
                        "memory_create",
                        {"path": "/memories/progress.md", "file_text": "done\n"},
                        "c1",
                    ),
                ),
            ),
            ModelTurn(
                response_id="r2",
                tool_calls=(fake_tool_call("run_finish", {"summary": "created"}, "c2"),),
            ),
        ]
    )
    AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs1"),
        model_adapter=first,
        runtime_config_provider=runtime_provider(config),
        tool_providers=(provider,),
        context_providers=(provider,),
    ).run_once("remember progress")

    second = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r3",
                tool_calls=(fake_tool_call("memory_view", {"path": "/memories/progress.md"}, "c3"),),
            ),
            ModelTurn(
                response_id="r4",
                tool_calls=(fake_tool_call("run_finish", {"summary": "read"}, "c4"),),
            ),
        ]
    )
    AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs2"),
        model_adapter=second,
        runtime_config_provider=runtime_provider(config),
        tool_providers=(provider,),
        context_providers=(provider,),
    ).run_once("read progress")

    assert "     1\tdone" in second.requests[1].messages[-1]["content"]["result"]["content"]


def test_backend_validation_accepts_memory_provider_tools(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = LocalFilesystemMemoryProvider(tmp_path / "memory")
    config = runtime_config(
        bindings=(provider.tool_bindings()[0], tool_binding("run.finish")),
    )
    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=lambda _spec, _token: FakeModelAdapter(turns=[ModelTurn(final_text="done")]),
        tool_providers=(provider,),
        context_providers=(provider,),
    )

    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="done",
            runtime_config=config,
        )
    )

    assert backend.wait_for_run(submission.run_id, timeout_s=5) is SessionState.COMPLETED
