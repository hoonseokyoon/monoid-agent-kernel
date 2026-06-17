from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from native_agent_runner.core._util import utc_timestamp
from native_agent_runner.core.spec import AgentRunSpec
from native_agent_runner.permissions import PermissionPolicy
from native_agent_runner.tools.base import ToolSpec

MANIFEST_SCHEMA_VERSION = "native-agent-runner.manifest.v1"


@dataclass(frozen=True)
class RunManifest:
    schema_version: str
    run_id: str
    created_at: str
    mode: str
    workspace_backend: str
    workspace_root: str
    workspace_base_path: str
    model_provider: str
    model: str
    reasoning_effort: str
    limits: dict[str, Any]
    capabilities: list[str]
    permission_policy: dict[str, Any]
    tool_policy: dict[str, Any]
    tool_surface: dict[str, Any]
    shell_policy: dict[str, Any]
    web_policy: dict[str, Any]
    tool_specs: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    workspace_index_path: str = "workspace.index.json"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def build_run_manifest(
    spec: AgentRunSpec,
    *,
    tool_specs: list[ToolSpec],
    permission_policy: PermissionPolicy,
    tool_policy: dict[str, Any],
    tool_surface: dict[str, Any] | None = None,
    workspace_index_path: str = "workspace.index.json",
    workspace_base_path: str = "workspace.base.json",
) -> RunManifest:
    return RunManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        run_id=spec.run_id,
        created_at=utc_timestamp(),
        mode=spec.mode,
        workspace_backend=spec.workspace_backend,
        workspace_root=str(spec.workspace_root),
        workspace_base_path=workspace_base_path,
        model_provider=spec.model.provider,
        model=spec.model.model,
        reasoning_effort=spec.model.reasoning.effort,
        limits={
            "max_steps": spec.limits.max_steps,
            "max_tool_calls": spec.limits.max_tool_calls,
            "max_bytes_read": spec.limits.max_bytes_read,
            "max_duration_s": spec.limits.max_duration_s,
        },
        capabilities=sorted(spec.effective_capabilities()),
        permission_policy=permission_policy.to_json(),
        tool_policy=tool_policy,
        tool_surface=dict(tool_surface or {}),
        shell_policy=spec.shell_policy.to_manifest(),
        web_policy=spec.web_policy.to_manifest(),
        tool_specs=[_tool_spec_payload(tool) for tool in tool_specs],
        metadata=dict(spec.metadata),
        workspace_index_path=workspace_index_path,
    )


def _tool_spec_payload(tool: ToolSpec) -> dict[str, Any]:
    return {
        "id": tool.id,
        "exported_name": tool.exported_name,
        "description": tool.description,
        "input_schema": tool.input_schema,
        "capability": tool.capability,
        "side_effect": tool.side_effect,
        "path_args": list(tool.path_args),
        "guidance": dict(tool.guidance),
        "examples": [dict(item) for item in tool.examples],
        "annotations": dict(tool.annotations),
    }
