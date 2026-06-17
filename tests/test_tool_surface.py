from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from native_agent_runner.cli import main
from native_agent_runner.core.spec import AgentRunSpec
from native_agent_runner.core.tool_surface import (
    DefaultToolSurfaceResolver,
    ToolExposureRule,
    ToolQuota,
    ToolSurfacePolicy,
    ToolSurfaceSnapshot,
)
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call
from native_agent_runner.tools.base import ToolContext, ToolRegistry, ToolResult, ToolSpec
from native_agent_runner.tools.policy import ToolPolicy


def _events(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in run_dir.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _transcript(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _simple_tool(tool_id: str, *, capability: str = "fs.read") -> ToolSpec:
    def handler(_context: ToolContext, _args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content={"tool": tool_id})

    return ToolSpec(
        id=tool_id,
        description=f"{tool_id} description",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        capability=capability,
        side_effect="read",
        handler=handler,
    )


class _DynamicProvider:
    def __init__(self, tools: tuple[ToolSpec, ...]) -> None:
        self.tools = tools

    def get_tools_for_turn(self, _context: ToolContext, _turn: Any) -> tuple[ToolSpec, ...]:
        return self.tools


class _ChangingResolver:
    name = "changing"

    def resolve(self, **kwargs: Any) -> ToolSurfaceSnapshot:
        run_spec = kwargs["run_spec"]
        turn = kwargs["turn"]
        policy = (
            ToolSurfacePolicy(rules=(ToolExposureRule(tool="fs.read", exposure="searchable"),))
            if turn.step == 1
            else ToolSurfacePolicy(rules=(ToolExposureRule(tool="fs.read", exposure="hidden"),))
        )
        kwargs["run_spec"] = replace(run_spec, tool_surface_policy=policy)
        return DefaultToolSurfaceResolver().resolve(**kwargs)


def test_tool_surface_resolver_maps_legacy_policy_to_authorization(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register_many(
        [
            _simple_tool("alpha"),
            _simple_tool("beta"),
            _simple_tool("gamma", capability="missing.capability"),
        ]
    )
    spec = AgentRunSpec(
        instruction="x",
        workspace_root=tmp_path,
        run_root=tmp_path / "runs",
        tool_policy=ToolPolicy(ask_tools=("alpha",), denied_tools=("beta",)),
    )
    capabilities = frozenset({"fs.read"})
    legacy = registry.policy_view(spec.tool_policy, capabilities)

    snapshot = DefaultToolSurfaceResolver().resolve(
        registry=registry,
        run_spec=spec,
        turn=object(),
        legacy_tool_policy=legacy,
        capabilities=capabilities,
    )

    assert [tool.id for tool in snapshot.immediate_tools] == ["alpha"]
    assert snapshot.authorization_for("alpha").decision == "ask"  # type: ignore[union-attr]
    assert snapshot.authorization_for("beta").decision == "deny"  # type: ignore[union-attr]
    assert snapshot.authorization_for("gamma").reason == "missing_capability"  # type: ignore[union-attr]
    assert set(snapshot.hidden_tool_ids) == {"beta", "gamma"}


def test_tool_surface_searchable_tool_loads_next_turn(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("hello\n", encoding="utf-8")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("tool_search", {"query": "read"}, "search1"),),
            ),
            ModelTurn(
                response_id="r2",
                tool_calls=(fake_tool_call("fs_read", {"path": "notes.md"}, "read1"),),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    spec = AgentRunSpec(
        instruction="Read notes.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        tool_surface_policy=ToolSurfacePolicy(
            rules=(ToolExposureRule(tool="fs.read", exposure="searchable"),),
        ),
    )

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    assert result.status == "completed"
    first_tools = {tool.id for tool in adapter.requests[0].tools}
    second_tools = {tool.id for tool in adapter.requests[1].tools}
    assert "fs.read" not in first_tools
    assert "tool.search" in first_tools
    assert "fs.read" in second_tools
    transcript = _transcript(result.run_dir)
    snapshots = [item for item in transcript if item["kind"] == "tool_surface_snapshot"]
    assert snapshots[0]["search_entries"][0]["load_hint"] == "available_next_turn"


def test_tool_search_result_is_not_callable_in_same_turn(tmp_path: Path) -> None:
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
    spec = AgentRunSpec(
        instruction="Read notes.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        tool_surface_policy=ToolSurfacePolicy(
            rules=(ToolExposureRule(tool="fs.read", exposure="searchable"),),
        ),
    )

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    assert result.status == "completed"
    transcript = result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")
    assert "tool_not_in_surface" in transcript


def test_pending_tool_load_is_rechecked_against_latest_policy(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("hello\n", encoding="utf-8")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("tool_search", {"query": "read"}, "search1"),),
            ),
            ModelTurn(
                response_id="r2",
                tool_calls=(fake_tool_call("fs_read", {"path": "notes.md"}, "read1"),),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    spec = AgentRunSpec(
        instruction="Read notes.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
    )

    result = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        tool_surface_resolver=_ChangingResolver(),  # type: ignore[arg-type]
    ).run()

    assert result.status == "completed"
    assert "fs.read" not in {tool.id for tool in adapter.requests[1].tools}
    transcript = result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")
    assert "Some requested tool loads are unavailable under the current policy." in transcript
    assert "tool_policy_denied" in transcript


def test_tool_surface_quota_is_enforced_per_run(tmp_path: Path) -> None:
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
                ),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    spec = AgentRunSpec(
        instruction="Read notes.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        tool_surface_policy=ToolSurfacePolicy(
            rules=(
                ToolExposureRule(
                    tool="fs.read",
                    quota=ToolQuota(max_calls_per_run=1),
                ),
            ),
        ),
    )

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    assert result.status == "completed"
    transcript = result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")
    assert "tool_quota_exceeded" in transcript


def test_tool_surface_hidden_tool_is_denied(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("hello\n", encoding="utf-8")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_read", {"path": "notes.md"}, "read1"),),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    spec = AgentRunSpec(
        instruction="Read notes.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        tool_surface_policy=ToolSurfacePolicy(
            rules=(ToolExposureRule(tool="fs.read", exposure="hidden"),),
        ),
    )

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    assert result.status == "completed"
    transcript = result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")
    assert "tool_policy_denied" in transcript
    denied = [event for event in _events(result.run_dir) if event["type"] == "permission.denied"]
    assert denied[0]["data"]["policy_decision"] == "deny"


def test_dynamic_tool_provider_adds_tools_for_turn(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    dynamic_tool = _simple_tool("dynamic.echo", capability="run.control")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("dynamic_echo", {}, "dyn1"),),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    spec = AgentRunSpec(
        instruction="Use dynamic.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
    )

    result = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        dynamic_tool_providers=(_DynamicProvider((dynamic_tool,)),),  # type: ignore[arg-type]
    ).run()

    assert result.status == "completed"
    assert "dynamic.echo" in {tool.id for tool in adapter.requests[0].tools}


def test_dynamic_tool_provider_duplicate_id_fails_run(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    spec = AgentRunSpec(
        instruction="Use dynamic.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
    )

    result = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        dynamic_tool_providers=(_DynamicProvider((_simple_tool("fs.read"),)),),  # type: ignore[arg-type]
    ).run()

    assert result.status == "failed"
    assert "duplicate tool id: fs.read" in result.error


def test_cli_loads_tool_surface_policy_file(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy_file = tmp_path / "tool-surface.json"
    policy_file.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "tool": "fs.read",
                        "exposure": "searchable",
                        "guidance": {"summary": "Use for workspace reads."},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    monkeypatch.setattr("native_agent_runner.cli._model_adapter", lambda *_a, **_k: adapter)

    result = CliRunner().invoke(
        main,
        [
            "run",
            "--workspace",
            str(workspace),
            "--instruction",
            "Finish.",
            "--run-root",
            str(tmp_path / "runs"),
            "--tool-surface-policy-file",
            str(policy_file),
        ],
    )

    assert result.exit_code == 0, result.output
    run_id = next(line for line in result.output.splitlines() if line.startswith("run_id: ")).removeprefix("run_id: ")
    manifest = json.loads((tmp_path / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tool_surface"]["policy_version"] == "tool-surface.v1"
    assert "fs.read" not in {tool["id"] for tool in manifest["tool_specs"]}
