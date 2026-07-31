from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from monoid_agent_kernel.core.agents import (
    AgentDefinition,
    AgentRuntimeConfig,
    PromptSpec,
    RegistryToolRef,
    SubagentDefinition,
    ToolBinding,
    ToolSearchConfig,
    generated_tool_bindings,
    validate_runtime_config,
)
from monoid_agent_kernel.core.schemas import validate_run_dir
from monoid_agent_kernel.core.tool_surface import ToolGuidance
from monoid_agent_kernel.core.spec import AgentRunSpec, ModelConfig, ModelRetryConfig
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.tools.builtin import builtin_tools
from monoid_agent_kernel.tools.base import ToolRegistry, ToolResult, ToolSpec
from monoid_agent_kernel.workspace.local import LocalWorkspaceBackend


class _RuntimeProvider:
    def __init__(self, configs: tuple[AgentRuntimeConfig, ...]) -> None:
        self.configs = configs
        self.calls = 0

    def current_config(self, _run_id: str) -> AgentRuntimeConfig:
        self.calls += 1
        index = min(self.calls - 1, len(self.configs) - 1)
        return self.configs[index]


def test_subagent_tool_filters_normalize_authoritative_patterns() -> None:
    definition = SubagentDefinition(
        tools=("tool\ud800",),
        disallowed_tools=("blocked\ud800",),
    )

    assert definition.tools == ("tool\ufffd",)
    assert definition.disallowed_tools == ("blocked\ufffd",)

    with pytest.raises(ValueError, match="tools must be a tuple of strings"):
        SubagentDefinition(tools=["tool"])  # type: ignore[arg-type]


class _ToolProvider:
    def __init__(self, *specs: ToolSpec) -> None:
        self.specs = specs

    def get_tools(self, _context):
        return self.specs


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


@pytest.mark.parametrize(
    ("overrides", "expected_error"),
    (
        ({"config_version": "999"}, "config_version"),
        ({"config_version": True}, "config_version"),
        ({"tool_search": {"enabled": "false"}}, "tool_search.enabled"),
        ({"tool_search": {"top_k": "999"}}, "tool_search.top_k"),
        ({"output_validators": False}, "output_validators"),
        ({"metadata": False}, "runtime config metadata"),
        (
            {"output_validators": [{"id": "format", "enabled": "false"}]},
            "output validator enabled",
        ),
        (
            {
                "tools": [
                    {
                        "binding_id": "read",
                        "ref": "fs.read",
                        "quota": {"max_calls_per_run": "999"},
                    }
                ]
            },
            "tool quota max_calls_per_run",
        ),
        (
            {
                "tools": [
                    {
                        "binding_id": "read",
                        "ref": "fs.read",
                        "requires_approval": "false",
                    }
                ]
            },
            "requires_approval",
        ),
        (
            {
                "tools": [
                    {
                        "binding_id": "read",
                        "ref": "fs.read",
                        "runtime": {"requires_lease": float("nan")},
                    }
                ]
            },
            "requires_lease",
        ),
        (
            {"tools": [{"binding_id": "read", "ref": "fs.read", "authorization": False}]},
            "authorization must be a string",
        ),
        (
            {"tools": [{"binding_id": "read", "ref": "fs.read", "exposure": False}]},
            "exposure must be a string",
        ),
        (
            {"tools": [{"binding_id": "read", "ref": "fs.read", "runtime": False}]},
            "tool binding runtime",
        ),
        (
            {
                "tools": [
                    {
                        "binding_id": "read",
                        "ref": "fs.read",
                        "scope": {"command_allow_prefixes": [1]},
                    }
                ]
            },
            "array of strings",
        ),
    ),
)
def test_runtime_config_json_rejects_coercible_operational_controls(
    overrides: dict[str, object],
    expected_error: str,
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        AgentRuntimeConfig.from_json({"definition_id": "test-agent", **overrides})


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


def test_runtime_config_ingress_normalizes_identity_content_and_model_before_use(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)

    def handler(_context, _arguments):
        return ToolResult(ok=True)

    raw_tool_id = "custom.\ud800"
    spec = ToolSpec(
        id=raw_tool_id,
        description="Custom \ud800 tool",
        input_schema={"type": "object", "default": float("nan")},
        capability="custom.\ud800",
        side_effect="read",
        handler=handler,
    )
    binding = ToolBinding(
        binding_id="binding.\ud800",
        ref=RegistryToolRef(raw_tool_id),
        model_name="call_\ud800",
        guidance=ToolGuidance(
            summary="Guidance \ud800",
            examples=({"value": float("nan")},),
            annotations={"value": float("inf")},
        ),
        title="Title \ud800",
        runtime={"label": "Runtime \ud800", "value": float("nan")},
        metadata={"label": "Metadata \ud800", "value": float("inf")},
    )
    config = AgentRuntimeConfig(
        definition_id="agent.\ud800",
        model=ModelConfig(provider="fake", model="model-\ud800", gateway_url="local://\ud800"),
        prompt=PromptSpec(
            system_prompt_base="System \ud800",
            persona_segments=("Persona \ud800",),
            runtime_segments=("Runtime prompt \ud800",),
        ),
        tools=(binding,),
        tool_search=ToolSearchConfig(
            enabled=False,
            binding_id="search.\ud800",
            model_name="search_\ud800",
        ),
        metadata={"label": "Config \ud800", "value": float("nan")},
    )
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=_RuntimeProvider((config,)),
        tool_providers=(_ToolProvider(spec),),
    )
    registry = ToolRegistry()
    registry.register(spec)

    normalized = loop._current_runtime_config(registry)
    normalized_binding = normalized.tools[0]

    assert normalized.definition_id == "agent.\ufffd"
    assert normalized.prompt.system_prompt_base == "System \ufffd"
    assert normalized.prompt.persona_segments == ("Persona \ufffd",)
    assert normalized.model is not None
    assert normalized.model.model == "model-\ufffd"
    assert normalized.model.gateway_url == "local://\ufffd"
    assert normalized_binding.ref.tool_id == "custom.\ufffd"
    assert normalized_binding.binding_id == "binding.\ufffd"
    assert normalized_binding.model_name == "call_\ufffd"
    assert normalized_binding.runtime == {"label": "Runtime \ufffd", "value": None}
    assert normalized_binding.guidance.examples == ({"value": None},)
    assert normalized_binding.guidance.annotations == {"value": None}
    assert normalized.metadata == {"label": "Config \ufffd", "value": None}
    assert math.isnan(config.metadata["value"])

    result = loop.run_once("Finish.")

    assert result.status == "completed"
    assert len(adapter.requests) == 1
    request = adapter.requests[0]
    assert "System \ufffd" in request.system_prompt
    assert "Persona \ufffd" in request.system_prompt
    assert request.model is not None and request.model.model == "model-\ufffd"
    assert request.tools[0].id == "binding.\ufffd"
    assert request.tools[0].provider_name == "call_\ufffd"
    assert request.tools[0].input_schema["default"] is None
    assert request.tools[0].annotations["value"] is None


@pytest.mark.parametrize(
    ("invalid_control", "expected_error"),
    (
        ("authorization", "authorization"),
        ("tool_search_enabled", "tool_search.enabled"),
        ("config_version", "config_version"),
        ("tool_search_top_k", "tool_search.top_k"),
        ("requires_approval", "requires_approval"),
        ("requires_lease", "requires_lease"),
    ),
)
def test_invalid_runtime_config_controls_fail_before_model_dispatch(
    tmp_path: Path,
    invalid_control: str,
    expected_error: str,
) -> None:
    workspace = _workspace(tmp_path)

    def handler(_context, _arguments):
        return ToolResult(ok=True)

    spec = ToolSpec(
        id="custom.control",
        description="Control test",
        input_schema={"type": "object"},
        capability="custom.control",
        side_effect="read",
        handler=handler,
    )
    binding_kwargs = {}
    search_kwargs = {}
    config_version = 1
    if invalid_control == "authorization":
        binding_kwargs["authorization"] = None
    elif invalid_control == "tool_search_enabled":
        search_kwargs["enabled"] = float("nan")
    elif invalid_control == "config_version":
        config_version = float("nan")  # type: ignore[assignment]
    elif invalid_control == "tool_search_top_k":
        search_kwargs["top_k"] = float("nan")
    elif invalid_control == "requires_approval":
        binding_kwargs["requires_approval"] = float("nan")
    else:
        binding_kwargs["runtime"] = {"requires_lease": float("nan")}
    binding = ToolBinding(
        binding_id="custom.control",
        ref=RegistryToolRef("custom.control"),
        **binding_kwargs,  # type: ignore[arg-type]
    )
    config = AgentRuntimeConfig(
        definition_id="control-test",
        config_version=config_version,
        tools=(binding,),
        tool_search=ToolSearchConfig(**search_kwargs),  # type: ignore[arg-type]
    )
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="should not be called")])

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / f"runs-{invalid_control}"),
        model_adapter=adapter,
        runtime_config_provider=config,
        tool_providers=(_ToolProvider(spec),),
    ).run_once("Finish.")

    assert result.status == "failed"
    assert result.error_code == "agent_config_invalid"
    assert expected_error in result.error
    assert adapter.requests == []


@pytest.mark.parametrize(
    ("model", "expected_error"),
    (
        (ModelConfig(provider="fake", timeout_s=0), "model.timeout_s"),
        (ModelConfig(provider="fake", timeout_s=-1), "model.timeout_s"),
        (ModelConfig(provider="fake", timeout_s=True), "model.timeout_s"),
        (
            ModelConfig(
                provider="fake",
                retry=ModelRetryConfig(initial_delay_s=-1),
            ),
            "model.retry.initial_delay_s",
        ),
        (
            ModelConfig(
                provider="fake",
                retry=ModelRetryConfig(max_delay_s=-1),
            ),
            "model.retry.max_delay_s",
        ),
        (
            ModelConfig(
                provider="fake",
                retry=ModelRetryConfig(backoff_multiplier=0),
            ),
            "model.retry.backoff_multiplier",
        ),
        (
            ModelConfig(
                provider="fake",
                retry=ModelRetryConfig(jitter_s=-1),
            ),
            "model.retry.jitter_s",
        ),
    ),
)
def test_invalid_model_timing_controls_fail_before_model_dispatch(
    tmp_path: Path,
    model: ModelConfig,
    expected_error: str,
) -> None:
    workspace = _workspace(tmp_path)
    config = AgentRuntimeConfig(definition_id="model-control-test", model=model)
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="should not be called")])

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=config,
    ).run_once("Finish.")

    assert result.status == "failed"
    assert result.error_code == "agent_config_invalid"
    assert expected_error in result.error
    assert adapter.requests == []


def test_model_retry_timing_controls_accept_documented_zero_delays(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    config = AgentRuntimeConfig(
        definition_id="model-zero-delay-test",
        model=ModelConfig(
            provider="fake",
            timeout_s=1,
            retry=ModelRetryConfig(
                initial_delay_s=0,
                max_delay_s=0,
                backoff_multiplier=0.5,
                jitter_s=0,
            ),
        ),
    )
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=FakeModelAdapter(turns=[ModelTurn(final_text="done")]),
        runtime_config_provider=config,
    )

    normalized = loop._current_runtime_config(ToolRegistry())

    assert normalized.model is not None
    assert normalized.model.retry.initial_delay_s == 0
    assert normalized.model.retry.max_delay_s == 0
    assert normalized.model.retry.backoff_multiplier == 0.5
    assert normalized.model.retry.jitter_s == 0


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


def test_unknown_tool_message_lists_tools_and_suggests(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register_many(builtin_tools(LocalWorkspaceBackend(_workspace(tmp_path))))
    config = _config(_binding("fs.writ"))  # typo of fs.write

    try:
        validate_runtime_config(config, registry)
    except Exception as exc:
        message = str(exc)
        assert "fs.writ" in message
        assert "Did you mean 'fs.write'?" in message
        assert "Available tools:" in message and "fs.read" in message
    else:  # pragma: no cover
        raise AssertionError("accepted unknown tool ref")


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
        tool_search=ToolSearchConfig(
            enabled=True, binding_id="search_tools", model_name="read_file"
        ),
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


def test_tool_binding_for_tool_derives_binding_id_and_ref() -> None:
    b = ToolBinding.for_tool("fs.read")
    assert b.binding_id == "fs.read"
    assert b.ref == RegistryToolRef("fs.read")
    assert b.model_name == "fs_read"  # derived: dots -> underscores


def test_tool_binding_for_tool_forwards_overrides() -> None:
    b = ToolBinding.for_tool(
        "fs.read", binding_id="reader", model_name="read_file", exposure="searchable"
    )
    assert (b.binding_id, b.model_name, b.exposure) == ("reader", "read_file", "searchable")


def test_tool_binding_accepts_bare_string_ref() -> None:
    b = ToolBinding(binding_id="x", ref="fs.read")
    assert b.ref == RegistryToolRef("fs.read")
    # identical to the explicit form
    assert b == ToolBinding(binding_id="x", ref=RegistryToolRef("fs.read"))


def test_tool_binding_for_tool_round_trips() -> None:
    b = ToolBinding.for_tool("fs.read")
    restored = ToolBinding.from_json(json.loads(json.dumps(b.to_json())))
    assert restored == b


def test_validate_returns_empty_for_valid_config(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register_many(builtin_tools(LocalWorkspaceBackend(_workspace(tmp_path))))
    config = AgentRuntimeConfig(
        definition_id="t",
        tools=(ToolBinding.for_tool("fs.read"), ToolBinding.for_tool("fs.write")),
    )
    assert AgentLoop.validate(config, registry=registry) == []


def test_validate_collects_all_issues_not_just_first(tmp_path: Path) -> None:
    # An unknown tool AND a duplicate binding_id — both must be reported in one call.
    config = AgentRuntimeConfig(
        definition_id="t",
        tools=(
            ToolBinding.for_tool("fs.read"),
            ToolBinding.for_tool("fs.read"),  # duplicate binding_id
            ToolBinding.for_tool("does.not.exist"),  # unknown tool id
        ),
    )
    issues = AgentLoop.validate(config)  # builtins-only registry
    assert any("duplicate tool binding_id: fs.read" in m for m in issues)
    assert any("does.not.exist" in m for m in issues)
    assert len(issues) >= 2  # collected, not first-and-raise


def test_validate_collects_empty_model_name_instead_of_raising() -> None:
    # A directly-constructed binding with a whitespace model_name makes _resolved_model_name raise;
    # validate() must collect it, not throw (the whole point of the collect-all preflight).
    config = AgentRuntimeConfig(
        definition_id="t",
        tools=(ToolBinding(binding_id="rd", ref=RegistryToolRef("fs.read"), model_name="   "),),
    )
    issues = AgentLoop.validate(config)
    assert any("empty model name" in m for m in issues), issues


def test_validate_accepts_agent_spawn_delegation_binding() -> None:
    # agent.spawn is a conditional (subagent) tool; validate() must still accept a config that
    # binds it (e.g. Studio's delegate capability) rather than report it as unknown.
    config = AgentRuntimeConfig(
        definition_id="t",
        tools=(ToolBinding.for_tool("fs.read"), ToolBinding.for_tool("agent.spawn")),
    )
    assert AgentLoop.validate(config) == []


def test_validate_collects_tool_registration_collision() -> None:
    # A custom tool that shadows a builtin id must be collected by validate(), not raised — the
    # registration happens inside the preflight, which advertises returning a list.
    from monoid_agent_kernel import tool

    @tool(id="fs.read", side_effect="read")  # collides with the builtin fs.read
    def clash(text: str) -> dict:
        return {"x": text}

    config = AgentRuntimeConfig(definition_id="t", tools=(ToolBinding.for_tool("fs.write"),))
    issues = AgentLoop.validate(config, tools=[clash])
    assert any("duplicate tool id: fs.read" in m for m in issues), issues


def test_contracts_core_curated_namespace() -> None:
    import types

    import monoid_agent_kernel as nar
    from monoid_agent_kernel.contracts import core

    assert core.AgentLoop is AgentLoop
    # The curated namespace must NOT shadow the monoid_agent_kernel.core package at the root.
    assert isinstance(nar.core, types.ModuleType) and hasattr(nar.core, "agents")
    assert "core" not in nar.contracts.__all__
    names = {n for n in vars(core) if not n.startswith("_")}
    assert names == {
        "AgentLoop",
        "AgentRunSpec",
        "AgentRuntimeConfig",
        "ModelAdapter",
        "ToolSpec",
        "tool",
        "EventSink",
        "Workspace",
        "PermissionPolicy",
    }
