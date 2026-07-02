"""Reference conformance harnesses for the bundled implementation."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core.agents import AgentRuntimeConfig, PromptSpec, SubagentDefinition, ToolBinding
from monoid_agent_kernel.core.capability import (
    AutoGrantBroker,
    CapabilityLease,
    CapabilityRequest,
    CapabilityVault,
)
from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore
from monoid_agent_kernel.core.control import ControlCommand
from monoid_agent_kernel.core.lease_admission import sanitize_denied_capability_result
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.backend.service import BackendRunRequest, RunnerBackend
from monoid_agent_kernel.reference.studio.server import StudioConfig, StudioServer
from monoid_agent_kernel.reference.web_gateway.service import FakeWebProvider, WebGatewayBackend
from monoid_agent_kernel.tools.base import ToolContext, ToolResult, ToolSpec

PARENT_MARK = "[[ROLE=PARENT]]"
CHILD_MARK = "[[ROLE=CHILD]]"

__all__ = [
    "ReferenceBackendHarness",
    "ReferenceCapabilityHarness",
    "ReferenceConformanceFactory",
    "ReferenceGatewayHarness",
]


@dataclass
class ReferenceConformanceFactory:
    """Factory for fresh Reference harness instances used by conformance profiles."""

    root: Path
    _counter: int = 0

    def _next_root(self, label: str) -> Path:
        self._counter += 1
        path = self.root / f"{label}-{self._counter}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def new_backend(self) -> ReferenceBackendHarness:
        return ReferenceBackendHarness(self._next_root("backend"))

    def new_capability(self) -> ReferenceCapabilityHarness:
        return ReferenceCapabilityHarness()

    def new_gateway(self, *, provider: FakeWebProvider | None = None) -> ReferenceGatewayHarness:
        return ReferenceGatewayHarness(provider=provider or FakeWebProvider())

    def run_studio_smoke(self) -> dict[str, Any]:
        root = self._next_root("studio")
        server = StudioServer(
            StudioConfig(
                workspace=root / "workspace",
                host="127.0.0.1",
                port=0,
                provider="offline",
                run_root=root / "runs",
                env_file=None,
            )
        )
        server.start()
        try:
            settings = server.settings()
            assert settings["provider"] == "offline"
            assert settings["offline"] is True
            submitted = server.start_chat("hello from reference-full smoke")
            run_id = str(submitted["run_id"])
            events = _wait_for_event(server, run_id, "turn.settled")
            assert _eventually(lambda: server.run_status(run_id)["status"] == "awaiting_input")
            status = server.run_status(run_id)
            sessions = server.sessions()
            assert status["status"] == "awaiting_input"
            assert any(session["run_id"] == run_id for session in sessions["sessions"])
            return {"run_id": run_id, "event_count": len(events)}
        finally:
            server.shutdown()


@dataclass
class ReferenceCapabilityHarness:
    vault: CapabilityVault = field(default_factory=CapabilityVault)
    requests: dict[str, CapabilityRequest] = field(default_factory=dict)

    @property
    def harness_id(self) -> str:
        return "reference-capability-vault"

    @property
    def supported_profiles(self) -> tuple[str, ...]:
        return ("capability-security", "multi-agent", "reference-full")

    def request_capability(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = CapabilityRequest(
            capability=str(payload.get("capability") or ""),
            scope=dict(payload.get("scope") or {}),
        )
        self.requests[request.request_id] = request
        return request.to_json()

    def grant_capability(self, request_id: str, lease: dict[str, Any]) -> dict[str, Any]:
        admitted = self.vault.admit(self.requests[request_id], CapabilityLease.from_json(dict(lease)))
        return admitted.to_json()

    def deny_capability(self, request_id: str, result: dict[str, Any]) -> dict[str, Any]:
        del request_id
        return sanitize_denied_capability_result(result, reason="denied by profile")

    def revoke_capability(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.vault.revoke(
            capability=(str(payload["capability"]) if payload.get("capability") else None),
            lease_id=(str(payload["lease_id"]) if payload.get("lease_id") else None),
            before=(float(payload["before"]) if payload.get("before") is not None else None),
        )

    def token_for(self, capability: str, *, now: float) -> str | None:
        return self.vault.token_for(capability, now=now)

    def valid_lease(self, capability: str, scope: dict[str, Any], *, now: float) -> dict[str, Any] | None:
        lease = self.vault.get_valid(capability, dict(scope), now=now)
        return lease.to_json() if lease is not None else None

    def export_revocations(self) -> dict[str, Any]:
        return self.vault.export_revocations()

    def import_revocations(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.vault.import_revocations(
            lease_ids=list(payload.get("revoked_lease_ids") or ()),
            capabilities=list(payload.get("revoked_capabilities") or ()),
            before=float(payload.get("revoked_before") or 0.0),
            all_revoked=bool(payload.get("revoked_all", False)),
        )
        return self.vault.export_revocations()

    def fork_child(self) -> ReferenceCapabilityHarness:
        return ReferenceCapabilityHarness(vault=self.vault.fork_for_child())


@dataclass
class ReferenceGatewayHarness:
    provider: FakeWebProvider = field(default_factory=FakeWebProvider)
    manager: TokenManager = field(default_factory=lambda: TokenManager.from_secret("w" * 32))
    _counter: int = 0

    def __post_init__(self) -> None:
        self.gateway = WebGatewayBackend(token_manager=self.manager, provider=self.provider)

    @property
    def harness_id(self) -> str:
        return "reference-web-gateway"

    @property
    def supported_profiles(self) -> tuple[str, ...]:
        return ("provider-gateway", "reference-full")

    def call_gateway(
        self,
        capability: str,
        payload: dict[str, Any],
        *,
        signed_capability: str | None = None,
        signed_scope: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        token = self._token(
            capability=signed_capability or capability,
            scope=signed_scope or {},
        )
        if capability == "web.search":
            return self.gateway.handle_search(token, dict(payload))
        if capability == "web.fetch":
            return self.gateway.handle_fetch(token, dict(payload))
        if capability == "web.context":
            return self.gateway.handle_context(token, dict(payload))
        raise AssertionError(f"unsupported gateway capability: {capability}")

    def _token(self, *, capability: str, scope: dict[str, Any]) -> str:
        self._counter += 1
        return self.manager.issue(
            kind="web_gateway",
            audience="csp.web-gateway",
            run_id=f"run_{self._counter}",
            tenant_id="tenant_a",
            user_id="user_a",
            ttl_s=600,
            metadata={"capability": capability, "scope": scope},
        )


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
        return [ModelTurn(response_id=f"r_{uuid.uuid4().hex[:8]}", final_text="first")]


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


class ReferenceBackendHarness:
    def __init__(
        self,
        root: Path,
        *,
        workspace: Path | None = None,
        run_root: Path | None = None,
        checkpoint_store: LocalFsCheckpointStore | None = None,
        token_manager: TokenManager | None = None,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.workspace = workspace or root / "workspace"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.workspace.joinpath("notes.md").write_text("notes\n", encoding="utf-8")
        self.run_root = run_root or root / "runs"
        self.checkpoint_store = checkpoint_store or LocalFsCheckpointStore(root / "shared-checkpoints")
        self.token_manager = token_manager or TokenManager.from_secret("x" * 32)
        self.gated_provider = _GatedToolProvider()

        def factory(spec: Any, llm_gateway_token: str) -> _ReferenceMultiAgentAdapter | FakeModelAdapter:
            del llm_gateway_token
            scenario = str(spec.metadata.get("scenario") or "")
            if scenario in {"subagent-foreground", "subagent-capability-revoked"}:
                return _ReferenceMultiAgentAdapter(scenario=scenario, backend=self.backend, run_id=spec.run_id)
            return FakeModelAdapter(turns=[ModelTurn(response_id=f"r_{uuid.uuid4().hex[:8]}", final_text="first")])

        self.backend = RunnerBackend(
            run_root=self.run_root,
            token_manager=self.token_manager,
            allowed_workspace_roots=(self.workspace,),
            llm_gateway_url="http://llm-gateway.internal/v1/turns",
            model_adapter_factory=factory,
            checkpoint_store=self.checkpoint_store,
            subagent_definitions={"researcher": _child_def()},
            tool_providers=(self.gated_provider,),
            capability_broker_factory=lambda req: AutoGrantBroker(),
        )
        self.backend.idle_timeout_s = 30.0
        self.backend.max_recover_attempts = 10_000

    @property
    def harness_id(self) -> str:
        return "reference-backend"

    @property
    def supported_profiles(self) -> tuple[str, ...]:
        return ("control-plane", "durable-runner", "multi-agent", "reference-full")

    def submit_run(self, request: dict[str, Any]) -> dict[str, Any]:
        scenario = str(request["scenario"])
        multi_turn = scenario in {"multi-turn", "recoverable-multi-turn", "parked-hitl"}
        submission = self.backend.submit_run(
            BackendRunRequest(
                tenant_id="tenant_a",
                user_id="user_a",
                workspace_root=self.workspace,
                instruction=f"{scenario} run",
                runtime_config=_runtime_config_for_scenario(scenario),
                multi_turn=multi_turn,
                metadata={"scenario": scenario},
            )
        )
        if scenario in {"completed", "subagent-foreground", "subagent-capability-revoked"}:
            assert self.backend.wait_for_run(submission.run_id, timeout_s=20) == "completed"
        elif multi_turn:
            assert _eventually(lambda: self.backend._record(submission.run_id).status == "awaiting_input")
            if scenario == "recoverable-multi-turn":
                assert _eventually(lambda: self.checkpoint_store.latest(submission.run_id) is not None)
        else:
            raise AssertionError(f"unsupported backend scenario: {scenario}")
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

    def restart(self, *, local_state: str = "same") -> ReferenceBackendHarness:
        if local_state == "same":
            run_root = self.run_root
        elif local_state == "empty":
            run_root = self.root / f"empty-runs-{uuid.uuid4().hex[:8]}"
        else:
            raise AssertionError(f"unsupported restart local_state: {local_state}")
        return ReferenceBackendHarness(
            self.root,
            workspace=self.workspace,
            run_root=run_root,
            checkpoint_store=self.checkpoint_store,
            token_manager=self.token_manager,
        )

    def task_result(self, run_id: str, token: str, task_id: str) -> dict[str, Any]:
        self.backend.status(run_id, token)
        task_path = self.backend._record(run_id).run_dir / "artifacts" / "tasks" / task_id / "task.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        return {"result": task["result"]}

    def dispatch(self, command: dict[str, Any]) -> dict[str, Any]:
        return self.backend.dispatch(ControlCommand.from_json(dict(command))).to_json()


def _runtime_config_for_scenario(scenario: str) -> AgentRuntimeConfig:
    if scenario == "subagent-capability-revoked":
        return AgentRuntimeConfig(
            definition_id="parent",
            prompt=PromptSpec(persona_segments=(PARENT_MARK,)),
            tools=(
                ToolBinding.for_tool("agent.spawn"),
                ToolBinding.for_tool("run.finish"),
                ToolBinding.for_tool("mcp.demo.gated", runtime={"requires_lease": True}),
            ),
        )
    if scenario == "subagent-foreground":
        return AgentRuntimeConfig(
            definition_id="parent",
            prompt=PromptSpec(persona_segments=(PARENT_MARK,)),
            tools=(ToolBinding.for_tool("agent.spawn"), ToolBinding.for_tool("run.finish")),
        )
    return AgentRuntimeConfig(
        definition_id="test-agent",
        tools=(
            ToolBinding.for_tool("fs.read"),
            ToolBinding.for_tool("fs.write"),
            ToolBinding.for_tool("run.finish"),
        ),
    )


def _child_def() -> SubagentDefinition:
    return SubagentDefinition(
        description="Researcher",
        prompt=PromptSpec(persona_segments=(CHILD_MARK,)),
        tools=("mcp.demo.gated",),
    )


def _eventually(fn: Any, *, timeout_s: float = 10.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if fn():
            return True
        time.sleep(0.05)
    return bool(fn())


def _wait_for_event(server: StudioServer, run_id: str, event_type: str) -> list[dict[str, Any]]:
    deadline = time.time() + 10.0
    events: list[dict[str, Any]] = []
    while time.time() < deadline:
        events = list(server.poll_events(run_id, 0).get("events", []))
        if any(event.get("type") == event_type for event in events):
            return events
        time.sleep(0.1)
    return events
