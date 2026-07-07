from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from support.runtime import runtime_config, runtime_provider, tool_binding

from monoid_agent_kernel.core.agents import compile_bound_tool_catalog
from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.core.tool_surface import DefaultToolSurfaceResolver, ToolQuota
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.tools.base import ToolContext, ToolRegistry, ToolResult, ToolSpec


def _simple_tool(tool_id: str, *, capability: str = "test", side_effect: str = "read") -> ToolSpec:
    def handler(_context: ToolContext, _args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content={"tool": tool_id})

    return ToolSpec(
        id=tool_id,
        description=f"{tool_id} description",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        capability=capability,
        side_effect=side_effect,  # type: ignore[arg-type]
        handler=handler,
    )


def test_resolver_uses_bound_catalog_and_binding_authorization() -> None:
    registry = ToolRegistry()
    registry.register_many((_simple_tool("alpha"), _simple_tool("beta"), _simple_tool("gamma")))
    config = runtime_config(
        bindings=(
            tool_binding("alpha", binding_id="alpha_read", model_name="alpha_read", authorization="ask"),
            tool_binding("beta", binding_id="beta_hidden", model_name="beta_hidden", exposure="hidden"),
            tool_binding("gamma", binding_id="gamma_search", model_name="gamma_search", exposure="searchable"),
        )
    )
    catalog = compile_bound_tool_catalog(config, registry)

    snapshot = DefaultToolSurfaceResolver().resolve(bound_catalog=catalog, turn=type("Turn", (), {"turn_id": "t1"})())

    assert [tool.id for tool in snapshot.immediate_tools] == ["alpha_read"]
    assert snapshot.authorization_for("alpha_read").decision == "ask"  # type: ignore[union-attr]
    assert "beta_hidden" in snapshot.hidden_tool_ids
    assert snapshot.search_entries[0].binding_id == "gamma_search"


def test_resolver_hides_exhausted_quota_and_public_json_has_metadata() -> None:
    registry = ToolRegistry()
    registry.register_many((_simple_tool("fs.read", capability="fs.read"),))
    config = runtime_config(
        bindings=(tool_binding("fs.read", quota=ToolQuota(max_calls_per_run=1), guidance="Read files."),)
    )
    catalog = compile_bound_tool_catalog(config, registry)

    snapshot = DefaultToolSurfaceResolver().resolve(
        bound_catalog=catalog,
        turn=type("Turn", (), {"turn_id": "t1"})(),
        call_counts={"fs.read": 1},
    )
    public = snapshot.to_public_json()

    assert "fs.read" in snapshot.hidden_tool_ids
    assert snapshot.authorization_for("fs.read").reason == "quota_exhausted"  # type: ignore[union-attr]
    assert public["hidden_binding_ids"] == ["fs.read"]
    assert public["authorizations"]["fs.read"]["decision"] == "deny"
    assert public["surface_warnings"]


def test_search_entries_include_grouping_metadata_and_defaults() -> None:
    registry = ToolRegistry()
    registry.register_many(
        (
            _simple_tool("web.search", capability="web.search"),
            _simple_tool("fs.read", capability="fs.read"),
        )
    )
    config = runtime_config(
        bindings=(
            tool_binding(
                "web.search",
                binding_id="docs_search",
                exposure="searchable",
                metadata={
                    "tool_search": {
                        "namespace": "docs",
                        "groups": ["reference"],
                        "tags": ["python", "api"],
                    }
                },
            ),
            tool_binding("fs.read", exposure="searchable"),
        )
    )
    catalog = compile_bound_tool_catalog(config, registry)

    snapshot = DefaultToolSurfaceResolver().resolve(bound_catalog=catalog, turn=type("Turn", (), {"turn_id": "t1"})())

    entries = {entry.binding_id: entry for entry in snapshot.search_entries}
    docs = entries["docs_search"]
    assert docs.namespace == "docs"
    assert docs.groups == ("reference",)
    assert docs.tags[:2] == ("python", "api")
    assert docs.binding_id == "docs_search"  # still the only load key
    defaulted = entries["fs.read"]
    assert defaulted.namespace == "fs"
    assert defaulted.groups == ("fs",)
    assert {"read", "fs.read"}.issubset(set(defaulted.tags))


def test_tool_surface_searchable_binding_loads_next_turn(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("hello\n", encoding="utf-8")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("tool_search", {"query": "read"}, "search1"),)),
            ModelTurn(response_id="r2", tool_calls=(fake_tool_call("fs_read", {"path": "notes.md"}, "read1"),)),
            ModelTurn(final_text="done"),
        ]
    )
    config = runtime_config(
        bindings=(
            tool_binding("fs.read", exposure="searchable"),
            tool_binding("run.finish"),
        )
    )

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
    ).run_once("Read notes.")

    assert result.status == "completed"
    assert "fs.read" not in {tool.id for tool in adapter.requests[0].tools}
    assert "tool.search" in {tool.id for tool in adapter.requests[0].tools}
    assert "fs.read" in {tool.id for tool in adapter.requests[1].tools}
    assert "fs_read" in {tool.exported_name for tool in adapter.requests[1].tools}
    transcript = _transcript(result.run_dir)
    snapshots = [item for item in transcript if item["kind"] == "tool_surface_snapshot"]
    assert snapshots[0]["search_entries"][0]["binding_id"] == "fs.read"


def test_tool_search_filters_by_group_and_tag_before_next_turn_load(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call(
                        "tool_search",
                        {"query": "", "groups": ["files"], "tag": "text"},
                        "search1",
                    ),
                ),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    config = runtime_config(
        bindings=(
            tool_binding(
                "fs.read",
                exposure="searchable",
                metadata={"tool_search": {"namespace": "workspace", "group": "files", "tags": ["text"]}},
            ),
            tool_binding(
                "fs.write",
                exposure="searchable",
                metadata={"tool_search": {"namespace": "workspace", "group": "files", "tags": ["write"]}},
            ),
            tool_binding(
                "web.search",
                exposure="searchable",
                metadata={"tool_search": {"namespace": "docs", "group": "reference", "tags": ["text"]}},
            ),
            tool_binding("run.finish"),
        )
    )

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
    ).run_once("Find a text file tool.")

    assert result.status == "completed"
    observations = [obs.output for req in adapter.requests for obs in req.observations]
    dumped = json.dumps(observations)
    assert '"binding_id": "fs.read"' in dumped
    assert '"binding_id": "fs.write"' not in dumped
    assert '"binding_id": "web.search"' not in dumped
    assert "fs.read" in {tool.id for tool in adapter.requests[1].tools}
    assert "fs.write" not in {tool.id for tool in adapter.requests[1].tools}
    assert "web.search" not in {tool.id for tool in adapter.requests[1].tools}


def test_search_result_is_not_callable_in_same_turn(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("hello\n", encoding="utf-8")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call("tool_search", {"query": "read"}, "search1"),
                    fake_tool_call("fs_read", {"path": "notes.md"}, "read1"),
                ),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    config = runtime_config(bindings=(tool_binding("fs.read", exposure="searchable"), tool_binding("run.finish")))

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
    ).run_once("Read notes.")

    assert result.status == "completed"
    assert "tool_not_in_surface" in result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")


def test_quota_and_hidden_bindings_are_enforced(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("hello\n", encoding="utf-8")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call("fs_read", {"path": "notes.md"}, "read1"),
                    fake_tool_call("fs_read", {"path": "notes.md"}, "read2"),
                    fake_tool_call("fs_write", {"path": "x.md", "content": "x", "create_dirs": False}, "write1"),
                ),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    config = runtime_config(
        bindings=(
            tool_binding("fs.read", quota=ToolQuota(max_calls_per_run=1)),
            tool_binding("fs.write", exposure="hidden"),
            tool_binding("run.finish"),
        )
    )

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
    ).run_once("Read notes.")

    transcript = result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")
    assert "tool_quota_exceeded" in transcript
    assert "tool_not_in_surface" in transcript
    assert not workspace.joinpath("x.md").exists()


def _transcript(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
