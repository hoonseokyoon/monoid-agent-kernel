from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from support.runtime import runtime_config
from monoid_agent_kernel.core.agents import SubagentDefinition
from monoid_agent_kernel.core.spec import AgentRunSpec, RunLimits
from monoid_agent_kernel.errors import NativeAgentError
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.providers.fake import FakeModelAdapter, ModelTurn, fake_tool_call
from monoid_agent_kernel.tool_loader import FunctionToolProvider
from monoid_agent_kernel.tools.base import ToolResult, ToolSpec
from monoid_agent_kernel.tools.host_bindings import (
    HostToolBinding,
    HostToolKey,
    ToolWorkspace,
    require_host_tool_binding,
)

pytestmark = pytest.mark.serial
FILES = HostToolKey[ToolWorkspace]("child.files", accepts=lambda v: isinstance(v, ToolWorkspace))


class Resource:
    def __init__(self, activation: Any) -> None:
        self.activation = activation
        self.inner = activation.workspace_resource()
        self.leases: list[Any] = []
        self.closed = 0

    def bind_call(self, lease: Any) -> ToolWorkspace:
        self.leases.append(lease)
        return self.inner.bind_call(lease)

    def close(self) -> None:
        self.closed += 1
        self.inner.close()


def _call(name: str = "test_action", call_id: str = "action", **args: Any) -> ModelTurn:
    return ModelTurn(tool_calls=(fake_tool_call(name, args, call_id),))


def _spawn(call_id: str = "spawn", **args: Any) -> ModelTurn:
    return _call("agent_spawn", call_id, subagent_type="worker", prompt=call_id, **args)


def _loop(
    root: Path,
    turns: list[ModelTurn],
    handler: Any,
    *,
    bindings: tuple = (),
    context: str = "fresh",
    spec: AgentRunSpec | None = None,
    **kwargs: Any,
) -> AgentLoop:
    workspace = root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    action = ToolSpec("test.action", "action", {"type": "object"}, "", "write", handler)
    return AgentLoop.from_config(
        spec
        or AgentRunSpec(
            workspace_root=workspace,
            run_root=root / "runs",
            limits=RunLimits(max_subagent_depth=3),
        ),
        FakeModelAdapter(turns=turns),
        runtime_config("agent.spawn", "test.action", "fs.write"),
        subagent_definitions={"worker": SubagentDefinition(context=context)},
        tool_providers=(FunctionToolProvider(lambda _ctx: (action,)),),
        host_tool_bindings=bindings,
        **kwargs,
    )


def _binding(resources: list[Resource], *, inherit: bool = True) -> HostToolBinding:
    def factory(activation: Any) -> Resource:
        resource = Resource(activation)
        resources.append(resource)
        return resource

    if inherit:
        return HostToolBinding(FILES, open=factory, inherit_to_children=True)
    return HostToolBinding(FILES, open=factory)


@pytest.mark.parametrize("context", ["fresh", "fork"])
@pytest.mark.parametrize("inherit", [False, True])
def test_child_bindings_default_off_and_explicit_opt_in(
    tmp_path: Path, context: str, inherit: bool
) -> None:
    resources, outcomes = [], []

    def handler(ctx: Any, _args: Any) -> ToolResult:
        try:
            view = require_host_tool_binding(ctx, FILES)
            outcomes.append(view.exists("report.txt"))
        except NativeAgentError as exc:
            outcomes.append(exc.error_code)
        return ToolResult(ok=True)

    loop = _loop(
        tmp_path,
        [_spawn(), _call(), ModelTurn(final_text="child"), ModelTurn(final_text="parent")],
        handler,
        bindings=(_binding(resources, inherit=inherit),),
        context=context,
    )
    assert loop.run_once("go").status == "completed"
    assert outcomes == ([False] if inherit else ["host_tool_binding_missing"])
    assert len(resources) == (2 if inherit else 1)
    assert all(resource.closed == 1 for resource in resources)
    if inherit:
        parent, child = (r.activation for r in resources)
        assert child.parent_run_id == parent.run_id == child.root_run_id
        assert child.run_id != parent.run_id
        assert resources[0] is not resources[1]


@pytest.mark.parametrize("context", ["fresh", "fork"])
@pytest.mark.parametrize("policy_source", ["spec", "loop", "override"])
def test_child_inherits_effective_operator_policy(
    tmp_path: Path, context: str, policy_source: str
) -> None:
    denied = PermissionPolicy(deny_patterns=("reports/secret.txt",))
    spec_policy = denied if policy_source == "spec" else PermissionPolicy()
    if policy_source == "override":
        spec_policy = PermissionPolicy(deny_patterns=("reports/spec-only.txt",))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "reports").mkdir()
    spec = AgentRunSpec(
        workspace_root=workspace, run_root=tmp_path / "runs", permission_policy=spec_policy
    )
    loop = _loop(
        tmp_path,
        [
            _call("fs_write", "parent-write", path="reports/secret.txt", content="parent"),
            _spawn(),
            _call("fs_write", "child-write", path="reports/secret.txt", content="child"),
            ModelTurn(final_text="child"),
            ModelTurn(final_text="parent"),
        ],
        lambda _ctx, _args: ToolResult(ok=True),
        context=context,
        spec=spec,
        **({"permission_policy": denied} if policy_source != "spec" else {}),
    )
    assert loop.run_once("go").status == "completed"
    observations = {
        obs.call_id: obs.output
        for request in loop.model_adapter.requests
        for obs in request.observations
    }
    for call_id in ("parent-write", "child-write"):
        assert observations[call_id].get("error", {}).get("code") == "permission_denied"
