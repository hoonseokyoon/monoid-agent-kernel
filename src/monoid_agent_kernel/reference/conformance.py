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
from monoid_agent_kernel.core.external_agent_envelope import (
    external_agent_envelope_to_inbox_message,
    validate_external_agent_envelope,
)
from monoid_agent_kernel.core.lease_admission import sanitize_denied_capability_result
from monoid_agent_kernel.core.tool_surface import ToolQuota
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.backend.service import BackendRunRequest, RunnerBackend
from monoid_agent_kernel.reference.outbox import (
    InboxRoutingOutboxSender,
    OutboxToolProvider,
    RecordingOutboxSender,
)
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

    def new_side_effect(self) -> ReferenceBackendHarness:
        return ReferenceBackendHarness(self._next_root("side-effect"))

    def new_message_fabric(self) -> ReferenceBackendHarness:
        return ReferenceBackendHarness(self._next_root("message-fabric"))

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


class _ApprovalToolProvider:
    def __init__(self) -> None:
        self.calls = 0

    def get_tools(self, context: ToolContext | None = None) -> list[ToolSpec]:
        del context

        def handler(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            del ctx
            self.calls += 1
            return ToolResult(ok=True, content={"approved_call": True, "value": args.get("value")})

        return [
            ToolSpec(
                id="demo.approval",
                description="demo approval-gated tool",
                input_schema={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "additionalProperties": True,
                },
                capability="",
                side_effect="write",
                handler=handler,
            )
        ]


class _SideEffectDemoProvider:
    def get_tools(self, context: ToolContext | None = None) -> list[ToolSpec]:
        del context

        def unsafe_handler(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            del ctx, args
            return ToolResult(ok=True, content={"unsafe_call": True})

        def idempotent_handler(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            del ctx
            key = str(args.get("idempotency_key") or "")
            return ToolResult(ok=True, content={"idempotency_key": key})

        return [
            ToolSpec(
                id="demo.external_unsafe",
                description="demo external side-effect tool without durable delivery",
                input_schema={"type": "object", "properties": {}, "additionalProperties": True},
                capability="",
                side_effect="write",
                handler=unsafe_handler,
            ),
            ToolSpec(
                id="demo.external_idempotent",
                description="demo external side-effect tool with caller-provided idempotency",
                input_schema={
                    "type": "object",
                    "properties": {"idempotency_key": {"type": "string"}},
                    "additionalProperties": True,
                },
                capability="",
                side_effect="write",
                handler=idempotent_handler,
            ),
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
        self.approval_provider = _ApprovalToolProvider()
        self.side_effect_provider = _SideEffectDemoProvider()
        self.outbox_sender = RecordingOutboxSender()
        self.message_fabric_directory: dict[str, str] = {}
        self.message_fabric_tokens: dict[str, str] = {}

        def factory(spec: Any, llm_gateway_token: str) -> _ReferenceMultiAgentAdapter | FakeModelAdapter:
            del llm_gateway_token
            scenario = str(spec.metadata.get("scenario") or "")
            if scenario in {"subagent-foreground", "subagent-capability-revoked"}:
                return _ReferenceMultiAgentAdapter(scenario=scenario, backend=self.backend, run_id=spec.run_id)
            return FakeModelAdapter(turns=_turns_for_scenario(scenario))

        self.backend = RunnerBackend(
            run_root=self.run_root,
            token_manager=self.token_manager,
            allowed_workspace_roots=(self.workspace,),
            llm_gateway_url="http://llm-gateway.internal/v1/turns",
            model_adapter_factory=factory,
            checkpoint_store=self.checkpoint_store,
            subagent_definitions={"researcher": _child_def()},
            tool_providers=(
                self.gated_provider,
                self.approval_provider,
                self.side_effect_provider,
                OutboxToolProvider(),
            ),
            capability_broker_factory=lambda req: AutoGrantBroker(),
            outbox_sender_factory=self._outbox_sender_for,
        )
        self.backend.idle_timeout_s = 30.0
        self.backend.max_recover_attempts = 10_000

    def _outbox_sender_for(self, request: BackendRunRequest) -> Any:
        scenario = str(request.metadata.get("scenario") or "")
        if scenario == "tool-side-effect-pending-recovery":
            return None
        if scenario.startswith("tool-side-effect-"):
            return self.outbox_sender
        if scenario.startswith("message-fabric-"):
            return InboxRoutingOutboxSender(
                deliver=self._deliver_message_fabric,
                source_peer_id=str(request.metadata.get("message_fabric_peer_id") or ""),
            )
        return None

    @property
    def harness_id(self) -> str:
        return "reference-backend"

    @property
    def supported_profiles(self) -> tuple[str, ...]:
        return (
            "control-plane",
            "durable-runner",
            "multi-agent",
            "tool-agent",
            "side-effect-tool-agent",
            "message-fabric",
            "reference-full",
        )

    def submit_run(self, request: dict[str, Any]) -> dict[str, Any]:
        scenario = str(request["scenario"])
        if scenario == "message-fabric-two-peer":
            return self._submit_message_fabric_two_peer()
        if scenario == "message-fabric-duplicate-restart":
            return self._submit_message_fabric_duplicate_restart()
        multi_turn = scenario in {
            "multi-turn",
            "recoverable-multi-turn",
            "parked-hitl",
            "tool-ask-approved",
            "tool-ask-denied",
            "tool-ask-stale-denied",
            "tool-side-effect-pending-recovery",
            "message-fabric-receiver",
            "message-fabric-worker",
        }
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
        if scenario in {
            "completed",
            "subagent-foreground",
            "subagent-capability-revoked",
            "tool-quota-denied",
            "tool-side-effect-outbox-dispatched",
            "tool-side-effect-strict-rejected",
            "tool-side-effect-idempotent-inline",
            "message-fabric-peer-unavailable",
        }:
            assert self.backend.wait_for_run(submission.run_id, timeout_s=20) == "completed"
        elif multi_turn:
            assert _eventually(lambda: self.backend._record(submission.run_id).status == "awaiting_input")
            if scenario == "recoverable-multi-turn":
                assert _eventually(lambda: self.checkpoint_store.latest(submission.run_id) is not None)
            elif scenario == "tool-side-effect-pending-recovery":
                assert _eventually(lambda: self.checkpoint_store.latest(submission.run_id) is not None)
            elif scenario in {"tool-ask-approved", "tool-ask-denied", "tool-ask-stale-denied"}:
                task = self._pending_task(submission.run_id, "tool_approval")
                if scenario == "tool-ask-stale-denied":
                    current = self.runtime_config(submission.run_id, submission.run_token)
                    stale_config = AgentRuntimeConfig.from_json(dict(current["config"]))
                    stale_config = AgentRuntimeConfig(
                        definition_id=stale_config.definition_id,
                        prompt=stale_config.prompt,
                        tools=(ToolBinding.for_tool("run.finish"),),
                        config_version=stale_config.config_version + 1,
                    )
                    self.replace_runtime_config(
                        submission.run_id,
                        submission.run_token,
                        stale_config.to_json(),
                        expected_version=int(current["config_version"]),
                        issuer="profile",
                        reason="remove approval tool",
                    )
                result = (
                    {"answer": "Deny", "approved": False, "reason": "profile denied"}
                    if scenario == "tool-ask-denied"
                    else {"answer": "Approve", "approved": True, "reason": "profile approved"}
                )
                self.report_task_result(
                    submission.run_id,
                    submission.run_token,
                    str(task["task_id"]),
                    result,
                )
                assert _eventually(
                    lambda: _has_event(
                        self.backend.events(submission.run_id, submission.run_token, limit=200)["events"],
                        "tool.approval.approved"
                        if scenario in {"tool-ask-approved", "tool-ask-stale-denied"}
                        else "tool.approval.denied",
                    ),
                    timeout_s=20.0,
                )
        else:
            raise AssertionError(f"unsupported backend scenario: {scenario}")
        return {"run_id": submission.run_id, "token": submission.run_token}

    def _submit_backend_scenario(
        self,
        scenario: str,
        *,
        instruction: str | None = None,
        multi_turn: bool = False,
    ) -> dict[str, str]:
        metadata: dict[str, Any] = {"scenario": scenario}
        if scenario.startswith("message-fabric-"):
            peer_id = scenario.removeprefix("message-fabric-")
            metadata["message_fabric_peer_id"] = "planner" if peer_id == "peer-unavailable" else peer_id
        submission = self.backend.submit_run(
            BackendRunRequest(
                tenant_id="tenant_a",
                user_id="user_a",
                workspace_root=self.workspace,
                instruction=instruction or f"{scenario} run",
                runtime_config=_runtime_config_for_scenario(scenario),
                multi_turn=multi_turn,
                metadata=metadata,
            )
        )
        return {"run_id": submission.run_id, "token": submission.run_token}

    def _submit_message_fabric_two_peer(self) -> dict[str, Any]:
        worker = self._submit_backend_scenario(
            "message-fabric-worker",
            instruction="stand by for planner",
            multi_turn=True,
        )
        self.message_fabric_directory["worker"] = worker["run_id"]
        self.message_fabric_tokens[worker["run_id"]] = worker["token"]
        assert _eventually(lambda: self.backend._record(worker["run_id"]).status == "awaiting_input")

        planner = self._submit_backend_scenario(
            "message-fabric-planner",
            instruction="collaborate with worker",
            multi_turn=True,
        )
        self.message_fabric_directory["planner"] = planner["run_id"]
        self.message_fabric_tokens[planner["run_id"]] = planner["token"]
        assert _eventually(
            lambda: _has_event(
                self.events(planner["run_id"], planner["token"], limit=200)["events"],
                "outbox.dispatched",
            ),
            timeout_s=20.0,
        )
        assert _eventually(
            lambda: _has_event(
                self.events(worker["run_id"], worker["token"], limit=200)["events"],
                "outbox.dispatched",
            ),
            timeout_s=20.0,
        )
        return {
            "run_id": planner["run_id"],
            "token": planner["token"],
            "peer_run_id": worker["run_id"],
            "peer_token": worker["token"],
        }

    def _submit_message_fabric_duplicate_restart(self) -> dict[str, Any]:
        receiver = self._submit_backend_scenario(
            "message-fabric-receiver",
            instruction="receive external agent messages",
            multi_turn=True,
        )
        assert _eventually(lambda: self.backend._record(receiver["run_id"]).status == "awaiting_input")
        envelope = _external_agent_envelope("mf-duplicate-1", peer_id="planner", text="hello worker")
        first = self.deliver_external_agent_message(receiver["run_id"], receiver["token"], envelope)
        assert first["status"] == "queued"
        assert _eventually(
            lambda: "mf-duplicate-1"
            in self.message_fabric_state(receiver["run_id"], receiver["token"])["seen_inbox_ids"],
            timeout_s=20.0,
        )
        assert _eventually(
            lambda: (
                self.checkpoint_store.latest(receiver["run_id"]) is not None
                and "mf-duplicate-1"
                in self.checkpoint_store.latest(receiver["run_id"]).checkpoint.inbox_seen_ids  # type: ignore[union-attr]
            ),
            timeout_s=20.0,
        )
        restarted = self.restart(local_state="same")
        restarted.resume_run(receiver["run_id"], receiver["token"])
        duplicate = restarted.deliver_external_agent_message(
            receiver["run_id"],
            receiver["token"],
            envelope,
        )
        state = restarted.message_fabric_state(receiver["run_id"], receiver["token"])
        _cancel_backend_run(restarted, receiver["run_id"], receiver["token"])
        return {
            "run_id": receiver["run_id"],
            "token": receiver["token"],
            "message_id": "mf-duplicate-1",
            "first_status": first["status"],
            "duplicate_status_after_restart": duplicate["status"],
            "seen_inbox_ids_after_restart": state["seen_inbox_ids"],
        }

    def _pending_task(self, run_id: str, kind: str) -> dict[str, Any]:
        record = self.backend._record(run_id)
        deadline = time.time() + 10.0
        while time.time() < deadline:
            for task in record.loop._session.res.context.job_manager.list_jobs():  # type: ignore[union-attr]
                if task.get("kind") == kind and task.get("status") == "running":
                    return task
            time.sleep(0.05)
        raise AssertionError(f"missing pending task kind: {kind}")

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

    def side_effects(self, run_id: str, token: str) -> dict[str, Any]:
        payloads: list[dict[str, Any]] = []
        try:
            self.backend.status(run_id, token)
            record = self.backend._record(run_id)
        except KeyError:
            record = None
        if record is not None and record.loop is not None:
            payloads = record.loop._outbox.export()
        if not payloads:
            latest = self.checkpoint_store.latest(run_id)
            if latest is not None:
                payloads = list(latest.checkpoint.outbox_requests)
        return {"requests": [_side_effect_request_summary(payload) for payload in payloads]}

    def run_outbox_dispatched_case(self) -> dict[str, Any]:
        submitted = self.submit_run({"scenario": "tool-side-effect-outbox-dispatched"})
        events = list(self.events(str(submitted["run_id"]), str(submitted["token"]))["events"])
        return {
            "requested": _has_event(events, "outbox.requested", destination="email"),
            "dispatched": _has_event(events, "outbox.dispatched", destination="email"),
        }

    def run_pending_recovery_case(self) -> dict[str, Any]:
        submitted = self.submit_run({"scenario": "tool-side-effect-pending-recovery"})
        pending_requests = list(self.side_effects(str(submitted["run_id"]), str(submitted["token"]))["requests"])
        if len(pending_requests) != 1:
            raise AssertionError("expected one pending side-effect request")
        restarted = self.restart(local_state="same")
        recovered_requests = list(
            restarted.side_effects(str(submitted["run_id"]), str(submitted["token"]))["requests"]
        )
        if len(recovered_requests) != 1:
            raise AssertionError("expected one recovered side-effect request")
        _cancel_backend_run(self, str(submitted["run_id"]), str(submitted["token"]))
        return {
            "request_id": pending_requests[0]["request_id"],
            "initial_status": pending_requests[0]["status"],
            "recovered_request_id": recovered_requests[0]["request_id"],
            "recovered_status": recovered_requests[0]["status"],
        }

    def run_strict_rejected_case(self) -> dict[str, Any]:
        submitted = self.submit_run({"scenario": "tool-side-effect-strict-rejected"})
        events = list(self.events(str(submitted["run_id"]), str(submitted["token"]))["events"])
        return {
            "denied": _has_event(
                events,
                "permission.denied",
                call_id="unsafe_1",
                error_code="tool_side_effect_policy_denied",
            ),
            "handler_finished": _has_event(events, "tool.call.finished", call_id="unsafe_1"),
        }

    def run_idempotent_inline_case(self) -> dict[str, Any]:
        submitted = self.submit_run({"scenario": "tool-side-effect-idempotent-inline"})
        events = list(self.events(str(submitted["run_id"]), str(submitted["token"]))["events"])
        return {
            "missing_denied": _has_event(
                events,
                "permission.denied",
                call_id="idempotent_missing",
                error_code="tool_side_effect_policy_denied",
            ),
            "valid_finished": _has_event(
                events,
                "tool.call.finished",
                call_id="idempotent_ok",
                tool="demo_external_idempotent",
            ),
        }

    def deliver_external_agent_message(
        self,
        run_id: str,
        token: str,
        envelope: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            parsed = validate_external_agent_envelope(dict(envelope))
        except ValueError as exc:
            return {
                "run_id": run_id,
                "status": "rejected",
                "error_code": "external_agent_envelope_invalid",
                "error": str(exc),
            }
        message = external_agent_envelope_to_inbox_message(parsed, run_id=run_id)
        return self.backend.send_message(
            run_id,
            token,
            message.content,
            message_id=message.id,
            source=message.source,
            correlation_id=message.correlation_id,
            causation_id=message.causation_id,
            traceparent=message.traceparent,
            tracestate=message.tracestate,
        )

    def message_fabric_state(self, run_id: str, token: str) -> dict[str, Any]:
        payloads: list[dict[str, Any]] = []
        record = None
        try:
            self.backend.status(run_id, token)
            record = self.backend._record(run_id)
        except KeyError:
            pass
        if record is not None and record.loop is not None:
            payloads = record.loop._outbox.export()
        if not payloads:
            latest = self.checkpoint_store.latest(run_id)
            if latest is not None:
                payloads = list(latest.checkpoint.outbox_requests)
        seen_ids = sorted(record.seen_inbox_ids) if record is not None else []
        if not seen_ids:
            latest = self.checkpoint_store.latest(run_id)
            if latest is not None:
                seen_ids = sorted(latest.checkpoint.inbox_seen_ids)
        return {
            "run_id": run_id,
            "seen_inbox_ids": seen_ids,
            "requests": [_side_effect_request_summary(payload) for payload in payloads],
        }

    def run_two_peer_exchange_case(self) -> dict[str, Any]:
        submitted = self.submit_run({"scenario": "message-fabric-two-peer"})
        planner_events = list(
            self.events(str(submitted["run_id"]), str(submitted["token"]), limit=200)["events"]
        )
        worker_events = list(
            self.events(str(submitted["peer_run_id"]), str(submitted["peer_token"]), limit=200)["events"]
        )
        return {
            "planner_dispatched": _has_event(planner_events, "outbox.dispatched", destination="worker"),
            "worker_replied": _has_event(worker_events, "outbox.dispatched", destination="planner"),
            "planner_trace_preserved": _event_with_trace(
                planner_events,
                "outbox.dispatched",
                destination="worker",
            ),
            "worker_trace_preserved": _event_with_trace(
                worker_events,
                "outbox.dispatched",
                destination="planner",
            ),
        }

    def run_malformed_envelope_case(self) -> dict[str, Any]:
        receiver = self.submit_run({"scenario": "message-fabric-receiver"})
        malformed = self.deliver_external_agent_message(
            str(receiver["run_id"]),
            str(receiver["token"]),
            {"protocol": "monoid.external-agent-envelope.v1", "peer_id": "planner"},
        )
        _cancel_backend_run(self, str(receiver["run_id"]), str(receiver["token"]))
        return malformed

    def run_duplicate_after_restart_case(self) -> dict[str, Any]:
        return self.submit_run({"scenario": "message-fabric-duplicate-restart"})

    def run_peer_unavailable_case(self) -> dict[str, Any]:
        submitted = self.submit_run({"scenario": "message-fabric-peer-unavailable"})
        state = self.message_fabric_state(str(submitted["run_id"]), str(submitted["token"]))
        pending = [request for request in state["requests"] if request["destination"] == "missing-worker"]
        if len(pending) != 1:
            raise AssertionError("expected one pending message-fabric request")
        return {
            "pending": True,
            "status": pending[0]["status"],
            "attempts": pending[0]["attempts"],
        }

    def _deliver_message_fabric(
        self,
        destination: str,
        envelope: dict[str, Any],
        *,
        message_id: str,
        correlation_id: str,
        causation_id: str,
        traceparent: str,
    ) -> str:
        del message_id, correlation_id, causation_id, traceparent
        run_id = self.message_fabric_directory.get(destination)
        if not run_id:
            raise LookupError(f"no agent {destination!r}")
        token = self.message_fabric_tokens[run_id]
        result = self.deliver_external_agent_message(run_id, token, envelope)
        return f"external-agent:{run_id}:{result.get('message_id', '')}"

    def dispatch(self, command: dict[str, Any]) -> dict[str, Any]:
        return self.backend.dispatch(ControlCommand.from_json(dict(command))).to_json()

    def report_task_result(
        self,
        run_id: str,
        token: str,
        task_id: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        return self.backend.report_task_result(run_id, token, task_id=task_id, result=result)


def _runtime_config_for_scenario(scenario: str) -> AgentRuntimeConfig:
    if scenario in {"tool-side-effect-outbox-dispatched", "tool-side-effect-pending-recovery"}:
        return AgentRuntimeConfig(
            definition_id="tool-agent",
            tools=(
                ToolBinding.for_tool(
                    "outbox.send",
                    runtime={
                        "requires_lease": True,
                        "external_side_effect": True,
                        "side_effect_delivery": "outbox",
                    },
                ),
                ToolBinding.for_tool("run.finish"),
            ),
            metadata={"tool_side_effect_policy": {"mode": "strict"}},
        )
    if scenario in {
        "message-fabric-planner",
        "message-fabric-worker",
        "message-fabric-peer-unavailable",
    }:
        return AgentRuntimeConfig(
            definition_id="message-fabric-agent",
            tools=(
                ToolBinding.for_tool("outbox.send", runtime={"requires_lease": True}),
                ToolBinding.for_tool("run.finish"),
            ),
        )
    if scenario == "message-fabric-receiver":
        return AgentRuntimeConfig(
            definition_id="message-fabric-agent",
            tools=(ToolBinding.for_tool("run.finish"),),
        )
    if scenario == "tool-side-effect-strict-rejected":
        return AgentRuntimeConfig(
            definition_id="tool-agent",
            tools=(
                ToolBinding.for_tool(
                    "demo.external_unsafe",
                    runtime={"external_side_effect": True},
                ),
                ToolBinding.for_tool("run.finish"),
            ),
            metadata={"tool_side_effect_policy": {"mode": "strict"}},
        )
    if scenario == "tool-side-effect-idempotent-inline":
        return AgentRuntimeConfig(
            definition_id="tool-agent",
            tools=(
                ToolBinding.for_tool(
                    "demo.external_idempotent",
                    runtime={
                        "external_side_effect": True,
                        "side_effect_delivery": "idempotent",
                        "idempotency_key_arg": "idempotency_key",
                    },
                ),
                ToolBinding.for_tool("run.finish"),
            ),
            metadata={"tool_side_effect_policy": {"mode": "strict"}},
        )
    if scenario in {"tool-ask-approved", "tool-ask-denied", "tool-ask-stale-denied"}:
        return AgentRuntimeConfig(
            definition_id="tool-agent",
            tools=(
                ToolBinding.for_tool("demo.approval", authorization="ask"),
                ToolBinding.for_tool("run.finish"),
            ),
        )
    if scenario == "tool-quota-denied":
        return AgentRuntimeConfig(
            definition_id="tool-agent",
            tools=(
                ToolBinding.for_tool("demo.approval", quota=ToolQuota(max_calls_per_run=0)),
                ToolBinding.for_tool("run.finish"),
            ),
        )
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


def _turns_for_scenario(scenario: str) -> list[ModelTurn]:
    if scenario in {"tool-side-effect-outbox-dispatched", "tool-side-effect-pending-recovery"}:
        return [
            ModelTurn(
                tool_calls=(
                    fake_tool_call(
                        "outbox_send",
                        {
                            "destination": "email",
                            "payload": {"to": "x@example.test"},
                            "idempotency_key": f"{scenario}:email:1",
                        },
                        "outbox_1",
                    ),
                )
            ),
            ModelTurn(final_text=f"{scenario} completed"),
        ]
    if scenario == "message-fabric-planner":
        return [
            ModelTurn(
                tool_calls=(
                    fake_tool_call(
                        "outbox_send",
                        {
                            "destination": "worker",
                            "payload": {
                                "text": "please do X",
                                "task_id": "message-fabric-task-1",
                            },
                            "idempotency_key": "planner-to-worker-1",
                        },
                        "planner_send_1",
                    ),
                )
            ),
            ModelTurn(final_text="planner sent request"),
        ]
    if scenario == "message-fabric-worker":
        return [
            ModelTurn(final_text="worker standing by"),
            ModelTurn(
                tool_calls=(
                    fake_tool_call(
                        "outbox_send",
                        {
                            "destination": "planner",
                            "payload": {
                                "text": "done: ok",
                                "task_id": "message-fabric-task-1",
                            },
                            "idempotency_key": "worker-to-planner-1",
                        },
                        "worker_send_1",
                    ),
                )
            ),
            ModelTurn(final_text="worker replied"),
        ]
    if scenario == "message-fabric-peer-unavailable":
        return [
            ModelTurn(
                tool_calls=(
                    fake_tool_call(
                        "outbox_send",
                        {
                            "destination": "missing-worker",
                            "payload": {"text": "are you there?"},
                            "idempotency_key": "missing-worker-1",
                        },
                        "missing_send_1",
                    ),
                )
            ),
            ModelTurn(final_text="queued unavailable peer"),
        ]
    if scenario == "message-fabric-receiver":
        return [
            ModelTurn(final_text="receiver standing by"),
            ModelTurn(final_text="receiver processed message"),
        ]
    if scenario == "tool-side-effect-strict-rejected":
        return [
            ModelTurn(tool_calls=(fake_tool_call("demo_external_unsafe", {}, "unsafe_1"),)),
            ModelTurn(final_text="strict rejection completed"),
        ]
    if scenario == "tool-side-effect-idempotent-inline":
        return [
            ModelTurn(
                tool_calls=(
                    fake_tool_call("demo_external_idempotent", {}, "idempotent_missing"),
                    fake_tool_call(
                        "demo_external_idempotent",
                        {"idempotency_key": "idem-1"},
                        "idempotent_ok",
                    ),
                )
            ),
            ModelTurn(final_text="idempotent inline completed"),
        ]
    if scenario in {"tool-ask-approved", "tool-ask-denied", "tool-ask-stale-denied"}:
        return [
            ModelTurn(
                tool_calls=(fake_tool_call("demo_approval", {"value": scenario}, "approval_1"),)
            ),
            ModelTurn(response_id="ignored_until_approval", final_text="waiting for approval"),
            ModelTurn(final_text=f"{scenario} completed"),
        ]
    if scenario == "tool-quota-denied":
        return [
            ModelTurn(tool_calls=(fake_tool_call("demo_approval", {"value": "quota"}, "approval_1"),)),
            ModelTurn(final_text="quota denied completed"),
        ]
    return [ModelTurn(response_id=f"r_{uuid.uuid4().hex[:8]}", final_text="first")]


def _child_def() -> SubagentDefinition:
    return SubagentDefinition(
        description="Researcher",
        prompt=PromptSpec(persona_segments=(CHILD_MARK,)),
        tools=("mcp.demo.gated",),
    )


def _side_effect_request_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": str(payload.get("id") or ""),
        "destination": str(payload.get("destination") or ""),
        "status": str(payload.get("status") or ""),
        "idempotency_key": str(payload.get("idempotency_key") or ""),
        "attempts": int(payload.get("attempts") or 0),
        "token_ref_present": bool(payload.get("token_ref")),
    }


def _external_agent_envelope(message_id: str, *, peer_id: str, text: str) -> dict[str, Any]:
    from monoid_agent_kernel.core.external_agent_envelope import (
        EXTERNAL_AGENT_ENVELOPE_VERSION,
    )

    return {
        "protocol": EXTERNAL_AGENT_ENVELOPE_VERSION,
        "peer_id": peer_id,
        "message_id": message_id,
        "task_id": "message-fabric-task-1",
        "correlation_id": "message-fabric-correlation-1",
        "causation_id": "message-fabric-cause-1",
        "parts": [{"type": "text", "text": text}],
    }


def _cancel_backend_run(harness: ReferenceBackendHarness, run_id: str, token: str) -> None:
    try:
        harness.dispatch(
            {
                "type": "cancel",
                "run_id": run_id,
                "args": {"token": token},
                "issuer": "reference-message-fabric",
            }
        )
    except Exception:
        pass


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


def _has_event(events: list[dict[str, Any]], event_type: str, **data: str) -> bool:
    for event in events:
        if event.get("type") != event_type:
            continue
        event_data = event.get("data") or {}
        if all(event_data.get(key) == value for key, value in data.items()):
            return True
    return False


def _event_with_trace(events: list[dict[str, Any]], event_type: str, **data: str) -> bool:
    for event in events:
        if event.get("type") != event_type:
            continue
        event_data = event.get("data") or {}
        if not all(event_data.get(key) == value for key, value in data.items()):
            continue
        return bool(event_data.get("traceparent"))
    return False
