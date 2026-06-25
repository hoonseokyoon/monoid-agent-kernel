"""Capability contract: scope math, the vault's fail-closed admit, and the AutoGrantBroker."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from conftest import runtime_config, runtime_provider, tool_binding

from native_agent_runner.core.capability import (
    AutoGrantBroker,
    CapabilityDenial,
    CapabilityGrant,
    CapabilityLease,
    CapabilityRequest,
    CapabilityVault,
    scope_within,
)
from native_agent_runner.core.spec import AgentRunSpec
from native_agent_runner.core.tool_surface import ToolScope
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call
from native_agent_runner.reference._shared.tokens import TokenManager
from native_agent_runner.reference.backend.service import BackendRunRequest, RunnerBackend
from native_agent_runner.reference.capability import (
    DenyAllBroker,
    GatewayCapabilityBroker,
    HumanEscalationBroker,
)
from native_agent_runner.tools.base import ToolContext, ToolResult, ToolSpec


def test_scope_within_list_subset_and_scalar_equality() -> None:
    assert scope_within({"allowed_domains": ["a.edu"]}, {"allowed_domains": ["a.edu", "b.edu"]})
    assert not scope_within({"allowed_domains": ["c.edu"]}, {"allowed_domains": ["a.edu"]})
    assert scope_within({"region": "us"}, {"region": "us"})
    assert not scope_within({"region": "eu"}, {"region": "us"})
    # A key absent from the outer scope means "unconstrained" there -> inner is within.
    assert scope_within({"allowed_domains": ["x"]}, {})


def test_autogrant_broker_grants_requested_scope() -> None:
    broker = AutoGrantBroker(ttl_seconds=900, now=lambda: 1000.0)
    lease = broker.request(
        CapabilityRequest(
            capability="web.search", scope={"allowed_domains": ["a.edu"]}, ttl_seconds=300
        )
    )
    assert isinstance(lease, CapabilityLease)
    assert lease.capability == "web.search"
    # The request's ttl wins; the broker's ttl_seconds is only the fallback when unset.
    assert lease.expires_at == 1300.0
    assert lease.token_ref.startswith("auto:")
    assert lease.is_valid(now=1299.0) and not lease.is_valid(now=1301.0)


def test_vault_caches_valid_lease_and_expires() -> None:
    vault = CapabilityVault()
    request = CapabilityRequest(capability="web.search", scope={"allowed_domains": ["a.edu"]})
    lease = CapabilityLease(
        capability="web.search", token_ref="t", expires_at=2000.0, scope={"allowed_domains": ["a.edu"]}
    )
    vault.admit(request, lease)
    assert vault.get_valid("web.search", {"allowed_domains": ["a.edu"]}, now=1999.0) is lease
    # Expired -> miss.
    assert vault.get_valid("web.search", {"allowed_domains": ["a.edu"]}, now=2001.0) is None
    # A need broader than the cached lease -> miss (must re-request).
    assert vault.get_valid("web.search", {"allowed_domains": ["a.edu", "b.edu"]}, now=1999.0) is None


def test_vault_admit_rejects_scope_widening() -> None:
    vault = CapabilityVault()
    request = CapabilityRequest(capability="web.search", scope={"allowed_domains": ["a.edu"]})
    widened = CapabilityLease(
        capability="web.search",
        token_ref="t",
        expires_at=2000.0,
        scope={"allowed_domains": ["a.edu", "evil.com"]},  # broader than requested
    )
    with pytest.raises(ValueError):
        vault.admit(request, widened)


def test_request_and_lease_round_trip_json() -> None:
    req = CapabilityRequest(capability="email.send", scope={"to": ["x@example.edu"]}, reason="reply")
    assert req.to_json()["protocol"] == "native-agent-runner.capability-request.v1"
    lease = CapabilityLease(capability="email.send", token_ref="secret-ref://l", expires_at=1.0)
    assert lease.to_json()["protocol"] == "native-agent-runner.capability-lease.v1"
    assert CapabilityDenial(capability="email.send", reason="nope").to_json()["reason"] == "nope"


# --- loop integration: implicit (binding-declared) gating ---------------------------------


class _CapToolProvider:
    """A custom tool whose binding will declare a capability requirement."""

    def __init__(self) -> None:
        self.calls = 0
        self.seen_tokens: list[str | None] = []

    def get_tools(self, context: ToolContext | None = None) -> list[ToolSpec]:
        provider = self

        def handler(ctx: ToolContext, args: dict) -> ToolResult:
            provider.calls += 1
            # The handler obtains the access handle the gate acquired (A-1: token delivery).
            provider.seen_tokens.append(ctx.capability_token("web.search"))
            return ToolResult(ok=True, content={"ran": True})

        return [
            ToolSpec(
                id="ext.fetch",
                description="external fetch needing web.search capability",
                input_schema={"type": "object", "properties": {}, "additionalProperties": True},
                capability="web.search",
                side_effect="read",
                handler=handler,
            )
        ]


class _CountingBroker:
    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.requests = 0

    def request(self, req: CapabilityRequest) -> CapabilityGrant:
        self.requests += 1
        return self.inner.request(req)  # type: ignore[attr-defined]


def _cap_loop(tmp_path: Path, provider: _CapToolProvider, broker: object, turns: list[ModelTurn]) -> AgentLoop:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    binding = tool_binding(
        "ext.fetch", runtime={"requires_lease": True}, scope=ToolScope(allowed_domains=("a.edu",))
    )
    return AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=FakeModelAdapter(turns=turns),
        runtime_config_provider=runtime_provider(runtime_config(bindings=(binding,))),
        tool_providers=(provider,),
        capability_broker=broker,  # type: ignore[arg-type]
    )


def _events(run_dir: Path) -> list[dict]:
    lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


_FETCH = ModelTurn(response_id="r1", tool_calls=(fake_tool_call("ext_fetch", {}, "c1"),))
_FETCH2 = ModelTurn(response_id="r2", tool_calls=(fake_tool_call("ext_fetch", {}, "c2"),))
_DONE = ModelTurn(response_id="rN", final_text="done")


def test_loop_grants_lease_then_runs_tool(tmp_path: Path) -> None:
    provider = _CapToolProvider()
    loop = _cap_loop(tmp_path, provider, AutoGrantBroker(), [_FETCH, _DONE])
    result = loop.run_once("go")

    assert result.status == "completed"
    assert provider.calls == 1  # the tool ran AFTER the grant
    assert provider.seen_tokens == ["auto:web.search"]  # the handle reached the handler
    events = _events(result.run_dir)
    assert any(
        e["type"] == "capability.granted" and e["data"]["capability"] == "web.search" for e in events
    )


def test_loop_denied_capability_blocks_tool(tmp_path: Path) -> None:
    provider = _CapToolProvider()
    loop = _cap_loop(tmp_path, provider, DenyAllBroker(), [_FETCH, _DONE])
    result = loop.run_once("go")

    # The run still completes; only the gated TOOL call failed (the model got an error obs).
    assert result.status == "completed"
    assert provider.calls == 0  # the tool never executed
    events = _events(result.run_dir)
    assert any(e["type"] == "capability.denied" and e["data"]["capability"] == "web.search" for e in events)


def test_loop_caches_lease_across_calls(tmp_path: Path) -> None:
    provider = _CapToolProvider()
    broker = _CountingBroker(AutoGrantBroker())
    loop = _cap_loop(tmp_path, provider, broker, [_FETCH, _FETCH2, _DONE])
    result = loop.run_once("go")

    assert result.status == "completed"
    assert provider.calls == 2  # tool ran twice
    assert broker.requests == 1  # but the lease was brokered once, then cached


def test_gateway_broker_mints_verifiable_token() -> None:
    token_manager = TokenManager.from_secret("x" * 32)
    broker = GatewayCapabilityBroker(token_manager=token_manager, tenant_id="t", user_id="u")
    lease = broker.request(
        CapabilityRequest(
            capability="web.search", scope={"allowed_domains": ["a.edu"]}, run_id="run_1", ttl_seconds=300
        )
    )
    assert isinstance(lease, CapabilityLease)
    # The lease's token_ref IS a gateway token (absorption): a capability gateway verifies it.
    claims = token_manager.verify(
        lease.token_ref, kind="capability", audience="csp.capability-gateway", run_id="run_1"
    )
    assert claims.metadata["capability"] == "web.search"


# --- B: human-escalation (async approval) -------------------------------------------------


def test_loop_escalates_capability_then_resumes_after_grant(tmp_path: Path) -> None:
    provider = _CapToolProvider()
    binding = tool_binding(
        "ext.fetch", runtime={"requires_lease": True}, scope=ToolScope(allowed_domains=("a.edu",))
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=FakeModelAdapter(
            turns=[
                _FETCH,  # 1) model calls the gated tool -> escalation -> parks
                ModelTurn(response_id="rw", final_text="waiting"),  # 2) settles so the run parks
                _FETCH2,  # 3) after the grant, the model retries the tool
                ModelTurn(response_id="rd", final_text="done"),  # 4) settles
            ]
        ),
        runtime_config_provider=runtime_provider(runtime_config(bindings=(binding,))),
        tool_providers=(provider,),
        capability_broker=HumanEscalationBroker(),
    )
    loop.open()

    suspension = loop.run_until_suspended("go")
    assert suspension.reason == "awaiting_tasks"  # parked awaiting the capability grant
    assert provider.calls == 0  # gated: the tool did NOT run
    task_id = suspension.awaiting_task_ids[0]

    # The Daemon/human approves, reporting a lease for the capability.
    loop.report_task_result(
        task_id,
        {
            "granted": True,
            "lease": {
                "capability": "web.search",
                "token_ref": "approved:web.search",
                "expires_at": time.time() + 600,
                "scope": {"allowed_domains": ["a.edu"]},
            },
        },
    )

    resumed = loop.run_until_suspended(None)
    assert resumed.reason == "settled"
    assert provider.calls == 1  # retried and ran once the lease was admitted
    assert provider.seen_tokens == ["approved:web.search"]  # the approved handle reached the tool
    loop.close()


def test_human_escalation_broker_returns_pending() -> None:
    from native_agent_runner.core.capability import CapabilityPending

    broker = HumanEscalationBroker()
    grant = broker.request(CapabilityRequest(capability="web.search", scope={"allowed_domains": ["a.edu"]}))
    assert isinstance(grant, CapabilityPending)
    assert "web.search" in grant.prompt


# --- backend injection (A-2): RunnerBackend provisions a per-run broker -------------------


def _cap_backend(tmp_path: Path, provider: _CapToolProvider, broker_factory: object) -> tuple[RunnerBackend, Path]:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("n\n", encoding="utf-8")

    def factory(spec: object, llm_gateway_token: str) -> FakeModelAdapter:
        del spec, llm_gateway_token
        return FakeModelAdapter(turns=[_FETCH, _DONE])

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=TokenManager.from_secret("x" * 32),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
        tool_providers=(provider,),
        capability_broker_factory=broker_factory,  # type: ignore[arg-type]
    )
    return backend, workspace


def _run_cap_backend(backend: RunnerBackend, workspace: Path) -> str:
    binding = tool_binding(
        "ext.fetch", runtime={"requires_lease": True}, scope=ToolScope(allowed_domains=("a.edu",))
    )
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="go",
            runtime_config=runtime_config(bindings=(binding,)),
        )
    )
    return backend.wait_for_run(submission.run_id, timeout_s=20)


def test_backend_grants_capability_via_factory(tmp_path: Path) -> None:
    provider = _CapToolProvider()
    backend, workspace = _cap_backend(tmp_path, provider, lambda req: AutoGrantBroker())
    assert _run_cap_backend(backend, workspace) == "completed"
    assert provider.calls == 1
    assert provider.seen_tokens == ["auto:web.search"]  # broker reached the tool through the backend


def test_backend_denies_capability_via_factory(tmp_path: Path) -> None:
    provider = _CapToolProvider()
    backend, workspace = _cap_backend(tmp_path, provider, lambda req: DenyAllBroker())
    # The run completes; the gated tool call was blocked (the model got an error obs).
    assert _run_cap_backend(backend, workspace) in {"completed", "limited", "failed"}
    assert provider.calls == 0  # the tool never executed


def test_backend_no_factory_leaves_gating_off(tmp_path: Path) -> None:
    provider = _CapToolProvider()
    backend, workspace = _cap_backend(tmp_path, provider, None)
    assert _run_cap_backend(backend, workspace) == "completed"
    # No broker -> requires_lease is a no-op; the tool runs and sees no token.
    assert provider.calls == 1
    assert provider.seen_tokens == [None]
