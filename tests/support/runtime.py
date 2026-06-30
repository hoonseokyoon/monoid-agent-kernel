from __future__ import annotations

from native_agent_runner.core.agents import (
    AgentRuntimeConfig,
    RegistryToolRef,
    StaticRuntimeConfigProvider,
    ToolBinding,
)
from native_agent_runner.core.spec import ModelConfig
from native_agent_runner.core.tool_surface import (
    ToolAuthorizationDecision,
    ToolExposure,
    ToolGuidance,
    ToolQuota,
    ToolScope,
)


def tool_binding(
    tool_id: str,
    *,
    binding_id: str | None = None,
    model_name: str | None = None,
    exposure: ToolExposure = "immediate",
    authorization: ToolAuthorizationDecision = "allow",
    guidance: str = "",
    scope: ToolScope | None = None,
    quota: ToolQuota | None = None,
    runtime: dict | None = None,
) -> ToolBinding:
    resolved_binding_id = binding_id or tool_id
    return ToolBinding(
        binding_id=resolved_binding_id,
        model_name=model_name or resolved_binding_id.replace(".", "_"),
        ref=RegistryToolRef(tool_id),
        exposure=exposure,
        authorization=authorization,
        guidance=ToolGuidance(summary=guidance),
        scope=scope or ToolScope(),
        quota=quota or ToolQuota(),
        runtime=runtime or {},
    )


def runtime_config(
    *tool_ids: str,
    definition_id: str = "test-agent",
    version: int = 1,
    model: ModelConfig | None = None,
    bindings: tuple[ToolBinding, ...] | None = None,
) -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        definition_id=definition_id,
        config_version=version,
        model=model,
        tools=bindings if bindings is not None else tuple(tool_binding(tool_id) for tool_id in tool_ids),
    )


def runtime_provider(config: AgentRuntimeConfig) -> StaticRuntimeConfigProvider:
    return StaticRuntimeConfigProvider(config)

