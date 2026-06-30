"""Control protocol: RunnerBackend.dispatch routing + the POST /v1/runs/{id}/control route."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from support.http import http_json, serving
from support.runtime import runtime_config, tool_binding
from support.waiting import eventually

from monoid_agent_kernel.core.capability import AutoGrantBroker
from monoid_agent_kernel.core.control import ControlCommand
from monoid_agent_kernel.core.lifecycle import SessionState
from monoid_agent_kernel.core.tool_surface import ToolScope
from monoid_agent_kernel.errors import PermissionDenied
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.backend.http import create_backend_server
from monoid_agent_kernel.reference.backend.service import BackendRunRequest, RunnerBackend
from monoid_agent_kernel.tools.base import ToolContext, ToolResult, ToolSpec


def _token_manager() -> TokenManager:
    return TokenManager.from_secret("x" * 32)


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("notes\n", encoding="utf-8")
    return workspace


def _config() -> Any:
    return runtime_config("fs.read", "fs.write", "run.finish")


def _backend(tmp_path: Path, workspace: Path, turns: list[ModelTurn]) -> RunnerBackend:
    def factory(spec: Any, llm_gateway_token: str) -> FakeModelAdapter:
        del spec, llm_gateway_token
        return FakeModelAdapter(turns=list(turns))

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
    )
    backend.idle_timeout_s = 10.0
    return backend


def _parked_multi_turn_run(backend: RunnerBackend, workspace: Path) -> tuple[str, str]:
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hello",
            runtime_config=_config(),
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert eventually(lambda: backend._record(run_id).status == "awaiting_input")
    return run_id, token


def _dispatch(backend: RunnerBackend, run_id: str, token: str, ctype: str, **args: Any) -> Any:
    return backend.dispatch(ControlCommand(type=ctype, run_id=run_id, args={"token": token, **args}))  # type: ignore[arg-type]


def test_dispatch_inspect_and_health_report_live_state(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(tmp_path, workspace, [ModelTurn(response_id="r1", final_text="first")])
    run_id, token = _parked_multi_turn_run(backend, workspace)

    inspect = _dispatch(backend, run_id, token, "inspect")
    assert inspect.status == "ok"
    assert inspect.state == "awaiting_input"
    assert inspect.data["state"] == "awaiting_input"
    assert inspect.data["run_id"] == run_id
    assert inspect.data["terminal"] is False

    health = _dispatch(backend, run_id, token, "health")
    assert health.status == "ok"
    assert health.state == "awaiting_input"
    assert health.data["alive"] is True
    assert health.data["can_accept_input"] is True

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_dispatch_routes_existing_ops_and_unknown(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(tmp_path, workspace, [ModelTurn(response_id="r1", final_text="first")])
    run_id, token = _parked_multi_turn_run(backend, workspace)

    assert _dispatch(backend, run_id, token, "status").status == "ok"
    assert _dispatch(backend, run_id, token, "runtime_config").status == "ok"

    # Pause/resume acks (the deep freeze/continue is covered at the loop level).
    pause = _dispatch(backend, run_id, token, "pause")
    assert pause.status == "ok"
    assert pause.data["pause_requested"] is True
    resume = _dispatch(backend, run_id, token, "resume")
    assert resume.status == "ok"
    assert resume.data["resumed"] is True

    # Unknown command type stays forward-compatible: unsupported, not a crash.
    unknown = _dispatch(backend, run_id, token, "frobnicate")
    assert unknown.status == "unsupported"
    assert unknown.error_code == "unknown_control_command"

    cancel = _dispatch(backend, run_id, token, "cancel")
    assert cancel.status == "ok"
    assert backend.wait_for_run(run_id, timeout_s=20) in {"completed", "failed", "limited"}


def test_dispatch_inspect_on_terminal_run_is_error(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(tmp_path, workspace, [ModelTurn(response_id="r1", final_text="done")])
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hello",
            runtime_config=_config(),
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert backend.wait_for_run(run_id, timeout_s=20) == "completed"

    # inspect/health need a live loop; on a terminal run they report a controlled error.
    result = _dispatch(backend, run_id, token, "inspect")
    assert result.status == "error"
    # status still works on a terminal run (it reads the record).
    assert _dispatch(backend, run_id, token, "status").status == "ok"


def test_dispatch_bad_token_raises_permission_denied(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(tmp_path, workspace, [ModelTurn(response_id="r1", final_text="first")])
    run_id, token = _parked_multi_turn_run(backend, workspace)
    with pytest.raises(PermissionDenied):
        backend.dispatch(ControlCommand(type="inspect", run_id=run_id, args={"token": "bad"}))
    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_http_control_route_dispatches_inspect(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(tmp_path, workspace, [ModelTurn(response_id="r1", final_text="first")])
    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    with serving(server) as base_url:
        created = http_json(
            f"{base_url}/v1/runs",
            {
                "tenant_id": "tenant_a",
                "user_id": "user_a",
                "workspace_root": str(workspace),
                "instruction": "hello",
                "runtime_config": _config().to_json(),
                "multi_turn": True,
            },
            token="admin",
        )
        run_id, run_token = created["run_id"], created["run_token"]
        assert eventually(lambda: backend._record(run_id).status == "awaiting_input")

        result = http_json(
            f"{base_url}/v1/runs/{run_id}/control",
            {"type": "inspect"},
            token=run_token,
        )
        assert result["status"] == "ok"
        assert result["state"] == "awaiting_input"
        assert result["protocol"] == "monoid.control-command.v1"

        backend.cancel_run(run_id, run_token)
        backend.wait_for_run(run_id, timeout_s=20)


def test_capability_task_kind_creates_and_resolves(tmp_path: Path) -> None:
    # Step 5: a scoped-capability request rides the hosted-task seam. The Daemon creates a
    # capability park and resolves it via report_task_result (both reachable through dispatch).
    workspace = _workspace(tmp_path)
    backend = _backend(tmp_path, workspace, [ModelTurn(response_id="r1", final_text="first")])
    run_id, token = _parked_multi_turn_run(backend, workspace)

    created = backend.create_task(
        run_id,
        token,
        kind="capability",
        request={"capability": "web.search", "scope": {"allowed_domains": ["example.edu"]}},
    )
    assert "task_id" in created and "callback_token" in created

    resolved = backend.report_task_result(
        run_id,
        token,
        task_id=created["task_id"],
        result={"granted": True, "token_ref": "secret-ref://lease-1"},
    )
    assert resolved.get("delivered") is True

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


class _GateToolProvider:
    """Yields one tool whose handler blocks until released — lets a test hold a run mid-turn
    so it can request a pause deterministically."""

    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        self._entered = entered
        self._release = release

    def get_tools(self, context: ToolContext | None = None) -> list[ToolSpec]:
        def handler(ctx: ToolContext, args: dict) -> ToolResult:
            self._entered.set()
            self._release.wait(timeout=10)
            return ToolResult(ok=True, content={"gated": True})

        return [
            ToolSpec(
                id="test.gate",
                description="block until released",
                input_schema={"type": "object", "properties": {}, "additionalProperties": True},
                capability="test.gate",
                side_effect="read",
                handler=handler,
            )
        ]


class _CapCountingProvider:
    """A capability-gated tool that counts executions — for the revoke end-to-end test."""

    def __init__(self) -> None:
        self.calls = 0

    def get_tools(self, context: ToolContext | None = None) -> list[ToolSpec]:
        provider = self

        def handler(ctx: ToolContext, args: dict) -> ToolResult:
            provider.calls += 1
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


def test_dispatch_revoke_capability_blocks_subsequent_call(tmp_path: Path) -> None:
    # End-to-end operator kill switch: a gated tool runs on a granted lease, the Daemon dispatches
    # revoke_capability, and the next gated call is refused — through the Control protocol.
    workspace = _workspace(tmp_path)
    provider = _CapCountingProvider()
    turns = [
        ModelTurn(response_id="r1", tool_calls=(fake_tool_call("ext_fetch", {}, "c1"),)),
        ModelTurn(response_id="r2", final_text="first"),
        ModelTurn(response_id="r3", tool_calls=(fake_tool_call("ext_fetch", {}, "c2"),)),
        ModelTurn(response_id="r4", final_text="second"),
    ]

    def factory(spec: Any, llm_gateway_token: str) -> FakeModelAdapter:
        del spec, llm_gateway_token
        return FakeModelAdapter(turns=list(turns))

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
        tool_providers=(provider,),
        capability_broker_factory=lambda req: AutoGrantBroker(),
    )
    backend.idle_timeout_s = 10.0
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
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert eventually(lambda: backend._record(run_id).status == "awaiting_input")
    assert provider.calls == 1  # the tool ran on the granted lease

    revoke = _dispatch(backend, run_id, token, "revoke_capability", capability="web.search")
    assert revoke.status == "ok"
    assert revoke.data["revoked"] is True
    assert revoke.data["capabilities"] == ["web.search"]

    # A follow-up message re-issues the gated call; revocation refuses it (no re-broker).
    backend.send_message(run_id, token, content="again")
    assert eventually(lambda: backend._record(run_id).status == "awaiting_input")
    assert provider.calls == 1  # still 1 — the gated tool stayed blocked after revocation

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_driver_pauses_mid_turn_then_resumes_to_settle(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    entered, release = threading.Event(), threading.Event()
    turns = [
        ModelTurn(response_id="r1", tool_calls=(fake_tool_call("test_gate", {}, "c1"),)),
        ModelTurn(response_id="r2", final_text="done"),
    ]

    def factory(spec: Any, llm_gateway_token: str) -> FakeModelAdapter:
        del spec, llm_gateway_token
        return FakeModelAdapter(turns=list(turns))

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
        tool_providers=(_GateToolProvider(entered, release),),
    )
    backend.idle_timeout_s = 10.0
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="go",
            runtime_config=runtime_config("test.gate"),
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token

    # The gate tool is executing -> the run is mid-turn. Request a pause, then release it.
    assert entered.wait(timeout=10)
    assert backend.pause_run(run_id, token)["pause_requested"] is True
    release.set()

    # The loop hits the next step boundary, raises TurnPaused; the driver parks the run PAUSED.
    assert eventually(lambda: backend._record(run_id).session_state == SessionState.PAUSED)
    inspect = _dispatch(backend, run_id, token, "inspect")
    assert inspect.state == "paused"

    # Resume re-pumps the SAME turn (the gate observation is re-sent) to settle.
    assert _dispatch(backend, run_id, token, "resume").data["resumed"] is True
    assert eventually(lambda: backend._record(run_id).status == "awaiting_input")

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)
