from __future__ import annotations

import json
from pathlib import Path

from native_agent_runner.core.agents import (
    AgentDefinition,
    AgentRuntimeConfig,
    PromptSpec,
    RegistryToolRef,
    ToolBinding,
    ToolSearchConfig,
    generated_tool_bindings,
    validate_runtime_config,
)
from native_agent_runner.core.schemas import validate_run_dir
from native_agent_runner.core.tool_surface import ToolGuidance
from native_agent_runner.core.spec import AgentRunSpec
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call
from native_agent_runner.tools.builtin import builtin_tools
from native_agent_runner.tools.base import ToolRegistry
from native_agent_runner.workspace.local import LocalWorkspaceBackend


class _RuntimeProvider:
    def __init__(self, configs: tuple[AgentRuntimeConfig, ...]) -> None:
        self.configs = configs
        self.calls = 0

    def current_config(self, _run_id: str) -> AgentRuntimeConfig:
        self.calls += 1
        index = min(self.calls - 1, len(self.configs) - 1)
        return self.configs[index]


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("hello\n", encoding="utf-8")
    return workspace


def _config(*bindings: ToolBinding, version: int = 1) -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        definition_id="test-agent",
        config_version=version,
        prompt=PromptSpec(runtime_segments=("runtime guidance",)),
        tools=bindings,
    )


def _binding(
    tool_id: str,
    *,
    exposure: str = "immediate",
    guidance: str = "",
) -> ToolBinding:
    return ToolBinding(
        binding_id=tool_id,
        model_name=tool_id.replace(".", "_"),
        ref=RegistryToolRef(tool_id),
        exposure=exposure,  # type: ignore[arg-type]
        guidance=ToolGuidance(summary=guidance),
        title=tool_id,
    )


def test_agent_definition_runtime_config_and_binding_round_trip() -> None:
    definition = AgentDefinition(
        id="coding",
        version="2026-06-17",
        description="Coding agent",
        prompt=PromptSpec(persona_segments=("Be direct.",)),
        tools=(_binding("fs.read", guidance="Read workspace files."),),
        metadata={"owner": "platform"},
    )
    config = AgentRuntimeConfig.from_definition(definition)

    restored_definition = AgentDefinition.from_json(json.loads(json.dumps(definition.to_json())))
    restored_config = AgentRuntimeConfig.from_json(json.loads(json.dumps(config.to_json())))

    assert restored_definition == definition
    assert restored_config == config
    assert restored_config.config_hash == config.config_hash


def test_generated_builtin_bindings_are_registry_refs(tmp_path: Path) -> None:
    workspace = LocalWorkspaceBackend(_workspace(tmp_path))
    bindings = generated_tool_bindings(builtin_tools(workspace))

    assert "fs.read" in {binding.ref.tool_id for binding in bindings}
    assert "tool.search" not in {binding.ref.tool_id for binding in bindings}
    assert all(binding.ref.kind == "registry" for binding in bindings)


def test_runtime_config_guidance_updates_model_request(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    config = _config(
        _binding("fs.read", guidance="Use the updated read guidance."),
        _binding("run.finish"),
    )

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=_RuntimeProvider((config,)),
    ).run_once("Finish.")

    assert result.status == "completed"
    read_tool = next(tool for tool in adapter.requests[0].tools if tool.id == "fs.read")
    assert "Use the updated read guidance." in read_tool.description
    assert "runtime guidance" in adapter.requests[0].system_prompt


def test_runtime_config_changes_apply_on_next_turn(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_read", {"path": "notes.md"}, "read1"),),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    first = _config(_binding("fs.read"), _binding("run.finish"), version=1)
    second = _config(_binding("run.finish"), version=2)

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=_RuntimeProvider((first, first, second)),
    ).run_once("Read.")

    assert result.status == "completed"
    assert "fs.read" in {tool.id for tool in adapter.requests[0].tools}
    assert "fs.read" not in {tool.id for tool in adapter.requests[1].tools}
    events = result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8")
    assert "agent.config.updated" in events
    transcript = result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")
    assert "agent_runtime_config_snapshot" in transcript
    assert validate_run_dir(result.run_dir) == []


def test_unknown_runtime_tool_ref_fails_run(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    config = _config(_binding("missing.tool"))

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=_RuntimeProvider((config,)),
    ).run_once("Finish.")

    assert result.status == "failed"
    assert result.error_code == "agent_config_invalid"
    assert "missing.tool" in result.error


def test_agent_loop_requires_runtime_config_provider(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])

    try:
        AgentLoop(
            AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
            adapter,
        )
    except TypeError as exc:
        assert "runtime_config_provider" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("AgentLoop accepted a missing runtime_config_provider")


def test_tool_search_binding_identity_conflicts_are_rejected(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register_many(builtin_tools(LocalWorkspaceBackend(_workspace(tmp_path))))

    binding_id_conflict = _config(
        ToolBinding(
            binding_id="tool.search",
            model_name="read_file",
            ref=RegistryToolRef("fs.read"),
        )
    )
    model_name_conflict = _config(
        ToolBinding(
            binding_id="read_file",
            model_name="tool_search",
            ref=RegistryToolRef("fs.read"),
        )
    )
    cross_call_name_conflict = AgentRuntimeConfig(
        definition_id="test-agent",
        tools=(
            ToolBinding(
                binding_id="read_file",
                model_name="read_file",
                ref=RegistryToolRef("fs.read"),
            ),
        ),
        tool_search=ToolSearchConfig(enabled=True, binding_id="search_tools", model_name="read_file"),
    )

    for config, expected in (
        (binding_id_conflict, "duplicate tool binding_id: tool.search"),
        (model_name_conflict, "duplicate tool model_name: tool_search"),
        (cross_call_name_conflict, "duplicate tool call name: read_file"),
    ):
        try:
            validate_runtime_config(config, registry)
        except Exception as exc:
            assert expected in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"accepted invalid config: {expected}")
