from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from conftest import runtime_config, runtime_provider, tool_binding

from native_agent_runner.core.agents import compile_bound_tool_catalog
from native_agent_runner.core.spec import AgentRunSpec
from native_agent_runner.core.tool_surface import DefaultToolSurfaceResolver, ToolQuota
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call
from native_agent_runner.tools.base import ToolContext, ToolRegistry, ToolResult, ToolSpec


def _simple_tool(tool_id: str) -> ToolSpec:
    def handler(_context: ToolContext, _args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content={"tool": tool_id})

    return ToolSpec(
        id=tool_id,
        description=f"{tool_id} description",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        capability="test",
        side_effect="read",
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
        spec=AgentRunSpec(instruction="Read notes.", workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
    ).run()

    assert result.status == "completed"
    assert "fs.read" not in {tool.id for tool in adapter.requests[0].tools}
    assert "tool.search" in {tool.id for tool in adapter.requests[0].tools}
    assert "fs.read" in {tool.id for tool in adapter.requests[1].tools}
    assert "fs_read" in {tool.exported_name for tool in adapter.requests[1].tools}
    transcript = _transcript(result.run_dir)
    snapshots = [item for item in transcript if item["kind"] == "tool_surface_snapshot"]
    assert snapshots[0]["search_entries"][0]["binding_id"] == "fs.read"


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
        spec=AgentRunSpec(instruction="Read notes.", workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
    ).run()

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
        spec=AgentRunSpec(instruction="Read notes.", workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
    ).run()

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
