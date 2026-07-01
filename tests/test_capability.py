"""Capability contract: scope math, the vault's fail-closed admit, and the AutoGrantBroker."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from support.runtime import runtime_config, runtime_provider, tool_binding

from monoid_agent_kernel.core.capability import (
    AutoGrantBroker,
    CapabilityDenial,
    CapabilityGrant,
    CapabilityLease,
    CapabilityRequest,
    CapabilityVault,
    scope_within,
)
from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.core.tool_surface import ToolScope
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.backend.service import BackendRunRequest, RunnerBackend
from monoid_agent_kernel.reference.capability import (
    DenyAllBroker,
    GatewayCapabilityBroker,
    HumanEscalationBroker,
)
from monoid_agent_kernel.tools.base import ToolContext, ToolResult, ToolSpec


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
    assert req.to_json()["protocol"] == "monoid.capability-request.v1"
    lease = CapabilityLease(capability="email.send", token_ref="secret-ref://l", expires_at=1.0)
    assert lease.to_json()["protocol"] == "monoid.capability-lease.v1"
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
        self.last_request: CapabilityRequest | None = None

    def request(self, req: CapabilityRequest) -> CapabilityGrant:
        self.requests += 1
        self.last_request = req
        return self.inner.request(req)  # type: ignore[attr-defined]


def _cap_loop(
    tmp_path: Path,
    provider: _CapToolProvider,
    broker: object | None,
    turns: list[ModelTurn],
    *,
    rotate_skew: float = 0.0,
    requires_lease: bool | str = True,
) -> AgentLoop:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    binding = tool_binding(
        "ext.fetch",
        runtime={"requires_lease": requires_lease},
        scope=ToolScope(allowed_domains=("a.edu",)),
    )
    return AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=FakeModelAdapter(turns=turns),
        runtime_config_provider=runtime_provider(runtime_config(bindings=(binding,))),
        tool_providers=(provider,),
        capability_broker=broker,  # type: ignore[arg-type]
        capability_rotate_skew_seconds=rotate_skew,
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


def test_loop_requires_lease_without_broker_fails_closed(tmp_path: Path) -> None:
    provider = _CapToolProvider()
    loop = _cap_loop(tmp_path, provider, None, [_FETCH, _DONE])
    result = loop.run_once("go")

    # The run can continue after the model sees the tool error, but the protected handler never ran.
    assert result.status == "completed"
    assert provider.calls == 0
    events = _events(result.run_dir)
    assert any(
        e["type"] == "capability.denied"
        and e["data"]["capability"] == "web.search"
        and e["data"]["reason"] == "capability broker required"
        for e in events
    )
    assert any(
        e["type"] == "tool.call.failed"
        and e["data"]["error_code"] == "capability_broker_required"
        for e in events
    )


def test_loop_requires_lease_optional_without_broker_keeps_dev_bypass(tmp_path: Path) -> None:
    provider = _CapToolProvider()
    loop = _cap_loop(tmp_path, provider, None, [_FETCH, _DONE], requires_lease="optional")
    result = loop.run_once("go")

    assert result.status == "completed"
    assert provider.calls == 1
    assert provider.seen_tokens == [None]


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
    # A generic (non-gateway-mapped) capability mints a "capability"-kind token. (web.* is mapped to
    # the web gateway — see test_gateway_broker_mints_web_gateway_token_for_web_capabilities.)
    lease = broker.request(
        CapabilityRequest(
            capability="email.send", scope={"to": ["x@example.edu"]}, run_id="run_1", ttl_seconds=300
        )
    )
    assert isinstance(lease, CapabilityLease)
    # The lease's token_ref IS a gateway token (absorption): a capability gateway verifies it.
    claims = token_manager.verify(
        lease.token_ref, kind="capability", audience="csp.capability-gateway", run_id="run_1"
    )
    assert claims.metadata["capability"] == "email.send"


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
        capability_auto_redispatch=False,  # exercise the model-retry path explicitly
    )
    loop.open()

    suspension = loop.run_until_suspended("go")
    assert suspension.reason == "awaiting_tasks"  # parked awaiting the capability grant
    assert provider.calls == 0  # gated: the tool did NOT run
    task_id = suspension.awaiting_task_ids[0]

    loop.report_task_result(task_id, _grant_lease())  # Daemon/human approves with a lease

    resumed = loop.run_until_suspended(None)
    assert resumed.reason == "settled"
    assert provider.calls == 1  # the MODEL retried and the tool ran once the lease was admitted
    assert provider.seen_tokens == ["approved:web.search"]  # the approved handle reached the tool
    loop.close()


def _redispatch_loop(tmp_path: Path, provider: _CapToolProvider, turns: list[ModelTurn]) -> AgentLoop:
    binding = tool_binding(
        "ext.fetch", runtime={"requires_lease": True}, scope=ToolScope(allowed_domains=("a.edu",))
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=FakeModelAdapter(turns=turns),
        runtime_config_provider=runtime_provider(runtime_config(bindings=(binding,))),
        tool_providers=(provider,),
        capability_broker=HumanEscalationBroker(),  # capability_auto_redispatch defaults True
    )


def _capability_task_count(loop: AgentLoop) -> int:
    jobs = loop._session.res.context.job_manager.jobs  # type: ignore[union-attr]
    return sum(1 for job in jobs.values() if job.kind == "capability")


def test_auto_redispatch_runs_gated_tool_without_model_retry(tmp_path: Path) -> None:
    provider = _CapToolProvider()
    # No retry turn — the model never re-issues the call; the loop auto-runs it after the grant.
    loop = _redispatch_loop(
        tmp_path,
        provider,
        turns=[_FETCH, ModelTurn(response_id="rw", final_text="waiting"), ModelTurn(response_id="rd", final_text="done")],
    )
    loop.open()
    parked = loop.run_until_suspended("go")
    assert parked.reason == "awaiting_tasks"
    loop.report_task_result(parked.awaiting_task_ids[0], _grant_lease())

    resumed = loop.run_until_suspended(None)
    assert resumed.reason == "settled"
    assert provider.calls == 1  # auto-executed exactly once; the model did NOT retry
    assert provider.seen_tokens == ["approved:web.search"]
    assert _capability_task_count(loop) == 1  # no re-escalation (a single capability task)
    loop.close()


def test_denied_capability_skips_replay(tmp_path: Path) -> None:
    provider = _CapToolProvider()
    loop = _redispatch_loop(
        tmp_path,
        provider,
        turns=[_FETCH, ModelTurn(response_id="rw", final_text="waiting"), ModelTurn(response_id="rd", final_text="done")],
    )
    loop.open()
    parked = loop.run_until_suspended("go")
    assert parked.reason == "awaiting_tasks"
    # The approver denies (no lease) -> no auto-redispatch, the tool never runs.
    loop.report_task_result(parked.awaiting_task_ids[0], {"granted": False, "reason": "policy"})

    resumed = loop.run_until_suspended(None)
    assert resumed.reason == "settled"
    assert provider.calls == 0
    loop.close()


def test_human_escalation_broker_returns_pending() -> None:
    from monoid_agent_kernel.core.capability import CapabilityPending

    broker = HumanEscalationBroker()
    grant = broker.request(CapabilityRequest(capability="web.search", scope={"allowed_domains": ["a.edu"]}))
    assert isinstance(grant, CapabilityPending)
    assert "web.search" in grant.prompt


# --- ④ durable-lease checkpoint -----------------------------------------------------------


def test_vault_export_durable_and_install_roundtrip() -> None:
    vault = CapabilityVault()
    request = CapabilityRequest(capability="web.search", scope={"allowed_domains": ["a.edu"]})
    # An ephemeral (sync) lease is NOT exported; a durable (approved) one is.
    vault.admit(request, CapabilityLease(capability="web.search", token_ref="t", expires_at=9e9, scope={"allowed_domains": ["a.edu"]}))
    assert vault.export_durable() == []
    vault.admit(
        CapabilityRequest(capability="email.send", scope={}),
        CapabilityLease(capability="email.send", token_ref="secret-ref://l", expires_at=9e9, durable=True),
    )
    exported = vault.export_durable()
    assert [e["capability"] for e in exported] == ["email.send"]

    # install() rehydrates without a scope re-check.
    fresh = CapabilityVault()
    fresh.install(CapabilityLease.from_json(exported[0]))
    assert fresh.token_for("email.send", now=0.0) == "secret-ref://l"


def _escalation_loop(tmp_path: Path, provider: _CapToolProvider, turns: list[ModelTurn], run_id: str | None = None) -> AgentLoop:
    binding = tool_binding(
        "ext.fetch", runtime={"requires_lease": True}, scope=ToolScope(allowed_domains=("a.edu",))
    )
    workspace = tmp_path / "ws"
    if not workspace.exists():
        workspace.mkdir()
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs", run_id=run_id) if run_id else AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")
    return AgentLoop(
        spec=spec,
        model_adapter=FakeModelAdapter(turns=turns),
        runtime_config_provider=runtime_provider(runtime_config(bindings=(binding,))),
        tool_providers=(provider,),
        capability_broker=HumanEscalationBroker(),
        capability_auto_redispatch=False,  # exercise the model-retry path for ④
    )


def _grant_lease() -> dict:
    return {
        "granted": True,
        "lease": {
            "capability": "web.search",
            "token_ref": "approved:web.search",
            "expires_at": time.time() + 600,
            "scope": {"allowed_domains": ["a.edu"]},
        },
    }


def test_approved_lease_survives_restart_no_reprompt(tmp_path: Path) -> None:
    # Escalate -> approve -> the lease is admitted durable. A fresh loop restored from the
    # checkpoint already holds the lease, so a later gated call runs WITHOUT re-escalating.
    provider1 = _CapToolProvider()
    loop1 = _escalation_loop(
        tmp_path,
        provider1,
        turns=[_FETCH, ModelTurn(response_id="rw", final_text="waiting"), _FETCH2, ModelTurn(response_id="rd", final_text="done")],
    )
    loop1.open()
    parked = loop1.run_until_suspended("go")
    assert parked.reason == "awaiting_tasks"
    loop1.report_task_result(parked.awaiting_task_ids[0], _grant_lease())
    settled = loop1.run_until_suspended(None)
    assert settled.reason == "settled"
    assert provider1.calls == 1  # model retried + the tool ran

    cp = loop1.snapshot()
    assert cp is not None
    assert [lease["capability"] for lease in cp.capability_leases] == ["web.search"]  # durable, persisted
    blobs = loop1.collect_checkpoint_blobs()
    run_id = loop1.spec.run_id

    # Fresh "process": restore into a new loop whose model calls the gated tool again.
    provider2 = _CapToolProvider()
    loop2 = _escalation_loop(
        tmp_path,
        provider2,
        turns=[ModelTurn(response_id="r1b", tool_calls=(fake_tool_call("ext_fetch", {}, "cb"),)), ModelTurn(response_id="r2b", final_text="done2")],
        run_id=run_id,
    )
    loop2.restore(cp, blobs=blobs)
    resumed = loop2.run_until_suspended("again")
    assert resumed.reason == "settled"  # NOT awaiting_tasks — no re-prompt
    assert provider2.calls == 1  # the gated tool ran on the restored lease
    assert provider2.seen_tokens == ["approved:web.search"]
    loop2.close()


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


def test_backend_no_factory_fails_requires_lease_closed(tmp_path: Path) -> None:
    provider = _CapToolProvider()
    backend, workspace = _cap_backend(tmp_path, provider, None)
    assert _run_cap_backend(backend, workspace) == "completed"
    # No broker -> required lease fails closed; the model gets an error observation, and the
    # protected handler never runs.
    assert provider.calls == 0
    run_id = next(iter(backend._records))
    events = _events(backend._record(run_id).run_dir)
    assert any(
        e["type"] == "tool.call.failed"
        and e["data"]["error_code"] == "capability_broker_required"
        for e in events
    )


# --- revocation: the vault's fail-closed read path + the operator kill switch --------------


def test_vault_revoke_per_capability_blocks_reads() -> None:
    vault = CapabilityVault()
    request = CapabilityRequest(capability="web.search", scope={"allowed_domains": ["a.edu"]})
    vault.admit(request, CapabilityLease(capability="web.search", token_ref="t", expires_at=9e9, scope={"allowed_domains": ["a.edu"]}))
    assert vault.token_for("web.search", now=0.0) == "t"
    vault.revoke(capability="web.search")
    assert vault.is_capability_revoked("web.search")
    # Fail-closed: both reads now miss even though the lease has not expired.
    assert vault.token_for("web.search", now=0.0) is None
    assert vault.get_valid("web.search", {"allowed_domains": ["a.edu"]}, now=0.0) is None


def test_vault_revoke_per_lease_id_and_watermark() -> None:
    vault = CapabilityVault()
    early = CapabilityLease(capability="cap.a", token_ref="ta", expires_at=9e9, issued_at=100.0)
    late = CapabilityLease(capability="cap.b", token_ref="tb", expires_at=9e9, issued_at=200.0)
    vault.admit(CapabilityRequest(capability="cap.a"), early)
    vault.admit(CapabilityRequest(capability="cap.b"), late)
    # A watermark kills the cohort issued before T (early), leaving the later one usable.
    vault.revoke(before=150.0)
    assert vault.token_for("cap.a", now=0.0) is None
    assert vault.token_for("cap.b", now=0.0) == "tb"
    # A per-lease_id revoke kills exactly that grant (and does NOT block re-brokering at the gate).
    vault.revoke(lease_id=late.lease_id)
    assert vault.token_for("cap.b", now=0.0) is None
    assert not vault.is_capability_revoked("cap.b")


def test_vault_export_import_revocations_roundtrip() -> None:
    vault = CapabilityVault()
    vault.revoke(capability="web.search", lease_id="lease_x", before=42.0)
    exported = vault.export_revocations()
    assert exported == {
        "revoked_lease_ids": ["lease_x"],
        "revoked_capabilities": ["web.search"],
        "revoked_before": 42.0,
    }
    fresh = CapabilityVault()
    fresh.import_revocations(**{
        "lease_ids": exported["revoked_lease_ids"],
        "capabilities": exported["revoked_capabilities"],
        "before": exported["revoked_before"],
    })
    assert fresh.is_capability_revoked("web.search")
    assert "lease_x" in fresh._revoked_lease_ids
    assert fresh._revoked_before == 42.0


def test_loop_revoke_blocks_next_call_without_rebrokering(tmp_path: Path) -> None:
    provider = _CapToolProvider()
    broker = _CountingBroker(AutoGrantBroker())
    loop = _cap_loop(
        tmp_path,
        provider,
        broker,
        turns=[_FETCH, ModelTurn(response_id="rw", final_text="waiting"), _FETCH2, _DONE],
    )
    loop.open()
    # First turn: the tool runs on a freshly-brokered lease.
    first = loop.run_until_suspended("go")
    assert first.reason == "settled"
    assert provider.calls == 1
    assert broker.requests == 1

    loop.revoke_capability(capability="web.search")  # operator kill switch

    # Second turn: the gated call is refused at the gate — and crucially NOT re-brokered, so even
    # this permissive AutoGrantBroker cannot resurrect it.
    second = loop.run_until_suspended("again")
    assert second.reason == "settled"
    assert provider.calls == 1  # the tool never ran again
    assert broker.requests == 1  # no re-broker after revocation
    run_dir = loop._session.res.recorder.run_dir  # type: ignore[union-attr]
    events = _events(run_dir)
    assert any(e["type"] == "capability.revoked" and e["data"]["capability"] == "web.search" for e in events)
    loop.close()


def test_revocation_survives_restart(tmp_path: Path) -> None:
    # Approve a durable lease, revoke it, then restore a fresh loop from the checkpoint: the
    # revocation must persist so the gated call stays blocked (the kill switch is not forgotten).
    provider1 = _CapToolProvider()
    loop1 = _escalation_loop(
        tmp_path,
        provider1,
        turns=[_FETCH, ModelTurn(response_id="rw", final_text="waiting"), _FETCH2, ModelTurn(response_id="rd", final_text="done")],
    )
    loop1.open()
    parked = loop1.run_until_suspended("go")
    assert parked.reason == "awaiting_tasks"
    loop1.report_task_result(parked.awaiting_task_ids[0], _grant_lease())
    settled = loop1.run_until_suspended(None)
    assert settled.reason == "settled"
    assert provider1.calls == 1

    loop1.revoke_capability(capability="web.search")
    cp = loop1.snapshot()
    assert cp is not None
    assert cp.revoked_capabilities == ["web.search"]
    blobs = loop1.collect_checkpoint_blobs()
    run_id = loop1.spec.run_id

    provider2 = _CapToolProvider()
    loop2 = _escalation_loop(
        tmp_path,
        provider2,
        turns=[ModelTurn(response_id="r1b", tool_calls=(fake_tool_call("ext_fetch", {}, "cb"),)), ModelTurn(response_id="r2b", final_text="done2")],
        run_id=run_id,
    )
    loop2.restore(cp, blobs=blobs)
    resumed = loop2.run_until_suspended("again")
    assert resumed.reason == "settled"
    assert provider2.calls == 0  # revoked across the restart -> the gated tool stays blocked
    loop2.close()


# --- rotation: refresh a near-expiry lease under a stable contract, bounded by a ceiling ----


def test_lease_can_rotate_respects_skew_and_ceiling() -> None:
    lease = CapabilityLease(capability="web.search", token_ref="t", expires_at=1000.0)
    assert not lease.can_rotate(now=100.0, skew=50.0)  # far from expiry -> no
    assert lease.can_rotate(now=970.0, skew=50.0)  # within skew of expiry -> yes
    assert not lease.can_rotate(now=1001.0, skew=50.0)  # already expired -> no
    capped = CapabilityLease(capability="web.search", token_ref="t", expires_at=1000.0, max_expires_at=980.0)
    assert capped.can_rotate(now=970.0, skew=50.0)  # within skew, before the ceiling -> yes
    assert not capped.can_rotate(now=985.0, skew=50.0)  # past the absolute ceiling -> no


def test_lease_max_expires_at_round_trips() -> None:
    lease = CapabilityLease(capability="c", token_ref="t", expires_at=10.0, max_expires_at=20.0, durable=True)
    assert CapabilityLease.from_json(lease.to_json()).max_expires_at == 20.0
    # Absent ceiling stays None across the round-trip (the ephemeral-grant default).
    plain = CapabilityLease(capability="c", token_ref="t", expires_at=1.0)
    assert CapabilityLease.from_json(plain.to_json()).max_expires_at is None


def test_loop_rotates_near_expiry_lease_on_use(tmp_path: Path) -> None:
    provider = _CapToolProvider()
    broker = _CountingBroker(AutoGrantBroker())  # ttl 600; a large skew forces rotation each use
    loop = _cap_loop(tmp_path, provider, broker, [_FETCH, _FETCH2, _DONE], rotate_skew=700.0)
    result = loop.run_once("go")

    assert result.status == "completed"
    assert provider.calls == 2  # tool ran twice
    # 1 initial broker + 1 rotation on the second (cached-but-near-expiry) call.
    assert broker.requests == 2
    events = _events(result.run_dir)
    rotated = [e for e in events if e["type"] == "capability.rotated"]
    assert len(rotated) == 1
    assert rotated[0]["data"]["capability"] == "web.search"
    assert rotated[0]["data"]["old_lease_id"] != rotated[0]["data"]["new_lease_id"]


def test_loop_default_skew_does_not_rotate(tmp_path: Path) -> None:
    provider = _CapToolProvider()
    broker = _CountingBroker(AutoGrantBroker())
    # Default rotate_skew=0.0: the cached lease is reused as-is, never re-brokered.
    loop = _cap_loop(tmp_path, provider, broker, [_FETCH, _FETCH2, _DONE])
    result = loop.run_once("go")
    assert result.status == "completed"
    assert provider.calls == 2
    assert broker.requests == 1  # cached, no rotation
    assert not any(e["type"] == "capability.rotated" for e in _events(result.run_dir))


# --- web capability routing (Phase A): web tools pull a lease token via the gate ----------


class _RecordingWebClient:
    """A duck-typed WebGatewayClient that records the per-call credential it was handed."""

    def __init__(self) -> None:
        self.tokens: list[str | None] = []

    def search(self, payload: dict, *, token: str | None = None) -> dict:
        self.tokens.append(token)
        return {"results": [], "result_count": 0}


def _web_loop(
    tmp_path: Path,
    client: object,
    broker: object | None,
    *,
    requires_lease: bool,
    turns: list[ModelTurn] | None = None,
) -> AgentLoop:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    runtime = {"requires_lease": True} if requires_lease else {}
    binding = tool_binding("web.search", runtime=runtime, scope=ToolScope(allowed_domains=("a.edu",)))
    turns = turns or [
        ModelTurn(response_id="r1", tool_calls=(fake_tool_call("web_search", {"query": "hi"}, "c1"),)),
        ModelTurn(response_id="rN", final_text="done"),
    ]
    return AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=FakeModelAdapter(turns=turns),
        runtime_config_provider=runtime_provider(runtime_config(bindings=(binding,))),
        web_gateway_client=client,  # type: ignore[arg-type]
        capability_broker=broker,  # type: ignore[arg-type]
    )


def test_web_tool_uses_lease_token_when_required(tmp_path: Path) -> None:
    client = _RecordingWebClient()
    loop = _web_loop(tmp_path, client, AutoGrantBroker(), requires_lease=True)
    result = loop.run_once("go")
    assert result.status == "completed"
    # The gate brokered a web.search lease; its handle (not the static credential) reached the client.
    assert client.tokens == ["auto:web.search"]


def test_web_capability_request_scope_includes_signed_gateway_constraints(tmp_path: Path) -> None:
    client = _RecordingWebClient()
    broker = _CountingBroker(AutoGrantBroker())
    loop = _web_loop(tmp_path, client, broker, requires_lease=True)

    result = loop.run_once("go")

    assert result.status == "completed"
    assert broker.last_request is not None
    assert broker.last_request.scope["binding_id"] == "web.search"
    assert broker.last_request.scope["max_calls"] == 20
    assert broker.last_request.scope["max_results"] == 10
    assert broker.last_request.scope["allowed_domains"] == ["a.edu"]


def test_web_tool_falls_back_to_static_credential_when_not_gated(tmp_path: Path) -> None:
    client = _RecordingWebClient()
    # No requires_lease, no broker -> no lease; the per-call override is None and the client uses
    # its own static credential (back-compat).
    loop = _web_loop(tmp_path, client, None, requires_lease=False)
    result = loop.run_once("go")
    assert result.status == "completed"
    assert client.tokens == [None]


def test_gateway_broker_mints_web_gateway_token_for_web_capabilities() -> None:
    manager = TokenManager.from_secret("x" * 32)
    broker = GatewayCapabilityBroker(token_manager=manager, tenant_id="t", user_id="u")
    web = broker.request(
        CapabilityRequest(
            capability="web.search",
            scope={"allowed_domains": ["a.edu"], "max_calls": 2},
            run_id="run_1",
            binding_id="search_docs",
        )
    )
    assert isinstance(web, CapabilityLease)
    # The web lease's token_ref IS a web-gateway token the existing web gateway already accepts.
    claims = manager.verify(web.token_ref, kind="web_gateway", audience="csp.web-gateway", run_id="run_1")
    assert claims.metadata["capability"] == "web.search"
    assert claims.metadata["scope"] == {
        "allowed_domains": ["a.edu"],
        "binding_id": "search_docs",
        "max_calls": 2,
    }
    # A non-web capability still mints the generic capability-kind token.
    other = broker.request(CapabilityRequest(capability="email.send", run_id="run_1"))
    assert isinstance(other, CapabilityLease)
    manager.verify(other.token_ref, kind="capability", audience="csp.capability-gateway", run_id="run_1")


def test_web_access_can_be_revoked_mid_run(tmp_path: Path) -> None:
    # The headline payoff: routing web through the capability gate means an operator can kill web
    # access on a live run (without cancelling it) — web inherits revocation for free.
    client = _RecordingWebClient()
    loop = _web_loop(
        tmp_path,
        client,
        AutoGrantBroker(),
        requires_lease=True,
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("web_search", {"query": "a"}, "c1"),)),
            ModelTurn(response_id="rw", final_text="first"),
            ModelTurn(response_id="r2", tool_calls=(fake_tool_call("web_search", {"query": "b"}, "c2"),)),
            ModelTurn(response_id="rd", final_text="second"),
        ],
    )
    loop.open()
    first = loop.run_until_suspended("go")
    assert first.reason == "settled"
    assert client.tokens == ["auto:web.search"]  # web ran on the lease

    loop.revoke_capability(capability="web.search")  # operator kills web access

    second = loop.run_until_suspended("again")
    assert second.reason == "settled"
    assert client.tokens == ["auto:web.search"]  # the 2nd web call was refused at the gate (no new call)
    loop.close()
