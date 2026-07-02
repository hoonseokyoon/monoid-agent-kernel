from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from conformance.capability_harness import ReferenceCapabilityHarness
from support.runtime import tool_binding

from monoid_agent_kernel.conformance.profiles.multi_agent import (
    assert_multi_agent_backend_boundary_profile,
    assert_multi_agent_backend_capability_boundary_profile,
    assert_multi_agent_shared_revocation_profile,
)
from monoid_agent_kernel.core.agents import AgentRuntimeConfig, PromptSpec, SubagentDefinition
from monoid_agent_kernel.core.capability import AutoGrantBroker
from monoid_agent_kernel.core.control import ControlCommand
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn
from monoid_agent_kernel.providers.fake import fake_tool_call
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.backend.service import BackendRunRequest, RunnerBackend
from monoid_agent_kernel.tools.base import ToolContext, ToolResult, ToolSpec

PARENT_MARK = "[[ROLE=PARENT]]"
CHILD_MARK = "[[ROLE=CHILD]]"


class _ReferenceMultiAgentAdapter:
    def __init__(self, *, scenario: str, backend: RunnerBackend, run_id: str) -> None:
        self.scenario = scenario
        self.backend = backend
        self.run_id = run_id
        self.revoked = False
        self.requests: list[ModelRequest] = []
        self.turns = self._turns_for(scenario)

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        if self.scenario == "subagent-capability-revoked" and not self.revoked:
            record = self.backend._record(self.run_id)
            assert record.loop is not None
            record.loop.revoke_capability(capability="mcp.demo.gated")
            self.revoked = True
        if not self.turns:
            return ModelTurn(final_text="idle")
        return self.turns.pop(0)

    @staticmethod
    def _turns_for(scenario: str) -> list[ModelTurn]:
        if scenario == "subagent-foreground":
            return [
                ModelTurn(
                    tool_calls=(
                        fake_tool_call(
                            "agent_spawn",
                            {"subagent_type": "researcher", "prompt": "summarize notes"},
                            "spawn_1",
                        ),
                    )
                ),
                ModelTurn(final_text="child found answer", usage={"total_tokens": 10}),
                ModelTurn(final_text="parent done"),
            ]
        if scenario == "subagent-capability-revoked":
            return [
                ModelTurn(
                    tool_calls=(
                        fake_tool_call(
                            "agent_spawn",
                            {"subagent_type": "researcher", "prompt": "use the gated tool"},
                            "spawn_1",
                        ),
                    )
                ),
                ModelTurn(tool_calls=(fake_tool_call("mcp_demo_gated", {}, "gated_1"),)),
                ModelTurn(final_text="child observed capability_revoked", usage={"total_tokens": 10}),
                ModelTurn(final_text="parent done"),
            ]
        raise AssertionError(f"unsupported multi-agent scenario: {scenario}")


class _GatedToolProvider:
    def __init__(self) -> None:
        self.calls = 0

    def get_tools(self, context: ToolContext | None = None) -> list[ToolSpec]:
        del context

        def handler(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            del ctx, args
            self.calls += 1
            return ToolResult(ok=True, content={"ran": True})

        return [
            ToolSpec(
                id="mcp.demo.gated",
                description="demo gated tool",
                input_schema={"type": "object", "properties": {}, "additionalProperties": True},
                capability="mcp.demo.gated",
                side_effect="read",
                handler=handler,
            )
        ]


class _ReferenceMultiAgentBackendHarness:
    def __init__(self, tmp_path: Path) -> None:
        self.workspace = tmp_path / "workspace"
        self.workspace.mkdir()
        self.workspace.joinpath("notes.md").write_text("notes\n", encoding="utf-8")
        self.gated_provider = _GatedToolProvider()

        def factory(spec: Any, llm_gateway_token: str) -> _ReferenceMultiAgentAdapter:
            del llm_gateway_token
            scenario = str(spec.metadata.get("scenario") or "")
            return _ReferenceMultiAgentAdapter(scenario=scenario, backend=self.backend, run_id=spec.run_id)

        self.backend = RunnerBackend(
            run_root=tmp_path / "runs",
            token_manager=TokenManager.from_secret("x" * 32),
            allowed_workspace_roots=(self.workspace,),
            llm_gateway_url="http://llm-gateway.internal/v1/turns",
            model_adapter_factory=factory,
            subagent_definitions={"researcher": _child_def()},
            tool_providers=(self.gated_provider,),
            capability_broker_factory=lambda req: AutoGrantBroker(),
        )
        self.backend.idle_timeout_s = 10.0

    @property
    def harness_id(self) -> str:
        return "reference-multi-agent"

    @property
    def supported_profiles(self) -> tuple[str, ...]:
        return ("multi-agent",)

    def submit_run(self, request: dict[str, Any]) -> dict[str, Any]:
        scenario = str(request["scenario"])
        submission = self.backend.submit_run(
            BackendRunRequest(
                tenant_id="tenant_a",
                user_id="user_a",
                workspace_root=self.workspace,
                instruction=f"{scenario} run",
                runtime_config=_parent_config(include_gated=scenario == "subagent-capability-revoked"),
                metadata={"scenario": scenario},
            )
        )
        assert self.backend.wait_for_run(submission.run_id, timeout_s=20) == "completed"
        return {"run_id": submission.run_id, "token": submission.run_token}

    def status(self, run_id: str, token: str) -> dict[str, Any]:
        return self.backend.status(run_id, token)

    def events(self, run_id: str, token: str, *, from_seq: int = 0, limit: int | None = None) -> dict[str, Any]:
        return self.backend.events(run_id, token, from_seq=from_seq, limit=limit)

    def descendant_events(
        self,
        run_id: str,
        token: str,
        descendant_run_id: str,
        *,
        from_seq: int = 0,
        limit: int | None = None,
    ) -> dict[str, Any]:
        return self.backend.descendant_events(
            run_id,
            token,
            descendant_run_id,
            from_seq=from_seq,
            limit=limit,
        )

    def diagnostics(self, run_id: str, token: str, *, event_limit: int = 50) -> dict[str, Any]:
        return self.backend.diagnostics(run_id, token, event_limit=event_limit)

    def result(self, run_id: str, token: str) -> dict[str, Any]:
        return self.backend.result(run_id, token)

    def runtime_config(self, run_id: str, token: str) -> dict[str, Any]:
        return self.backend.runtime_config(run_id, token)

    def replace_runtime_config(
        self,
        run_id: str,
        token: str,
        config: dict[str, Any],
        *,
        expected_version: int,
        issuer: str,
        reason: str,
    ) -> dict[str, Any]:
        return self.backend.replace_runtime_config(
            run_id,
            token,
            expected_version=expected_version,
            issuer=issuer,
            reason=reason,
            config=AgentRuntimeConfig.from_json(dict(config)),
        )

    def resume_run(self, run_id: str, token: str) -> dict[str, Any]:
        return self.backend.resume_run(run_id, token)

    def recover_runs(self) -> tuple[str, ...]:
        return tuple(self.backend.recover_runs())

    def restart(self, *, local_state: str = "same") -> _ReferenceMultiAgentBackendHarness:
        del local_state
        raise NotImplementedError("multi-agent profile does not use restart")

    def task_result(self, run_id: str, token: str, task_id: str) -> dict[str, Any]:
        self.backend.status(run_id, token)
        task_path = self.backend._record(run_id).run_dir / "artifacts" / "tasks" / task_id / "task.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        return {"result": task["result"]}

    def dispatch(self, command: dict[str, Any]) -> dict[str, Any]:
        return self.backend.dispatch(ControlCommand.from_json(dict(command))).to_json()


def _parent_config(*, include_gated: bool) -> AgentRuntimeConfig:
    tools = [tool_binding("agent.spawn"), tool_binding("run.finish")]
    if include_gated:
        tools.append(tool_binding("mcp.demo.gated", runtime={"requires_lease": True}))
    return AgentRuntimeConfig(
        definition_id="parent",
        prompt=PromptSpec(persona_segments=(PARENT_MARK,)),
        tools=tuple(tools),
    )


def _child_def() -> SubagentDefinition:
    return SubagentDefinition(
        description="Researcher",
        prompt=PromptSpec(persona_segments=(CHILD_MARK,)),
        tools=("mcp.demo.gated",),
    )


def test_reference_capability_vault_satisfies_multi_agent_revocation_profile() -> None:
    assert_multi_agent_shared_revocation_profile(ReferenceCapabilityHarness())


def test_reference_backend_satisfies_multi_agent_boundary_profile(tmp_path: Path) -> None:
    assert_multi_agent_backend_boundary_profile(_ReferenceMultiAgentBackendHarness(tmp_path))


def test_reference_backend_satisfies_multi_agent_capability_boundary_profile(tmp_path: Path) -> None:
    harness = _ReferenceMultiAgentBackendHarness(tmp_path)
    assert_multi_agent_backend_capability_boundary_profile(harness)
    assert harness.gated_provider.calls == 0
