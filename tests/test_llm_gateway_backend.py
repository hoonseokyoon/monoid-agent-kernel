from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from conftest import http_json, runtime_config, wait_http_ready

from native_agent_runner.reference.backend.http import create_backend_server
from native_agent_runner.reference.backend.service import BackendRunRequest, RunnerBackend
from native_agent_runner.reference._shared.tokens import TokenManager
from native_agent_runner.errors import ModelAdapterError, PermissionDenied
from native_agent_runner.reference.llm_gateway.http import create_llm_gateway_server
from native_agent_runner.reference.llm_gateway.service import LlmGatewayBackend
from native_agent_runner.providers.base import ModelTurn, ToolCall
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call


def _token_manager() -> TokenManager:
    return TokenManager.from_secret("y" * 32)


def _llm_token(manager: TokenManager, *, run_id: str = "run_1", tenant_id: str = "tenant_a") -> str:
    return manager.issue(
        kind="llm_gateway",
        audience="csp.llm-gateway",
        run_id=run_id,
        tenant_id=tenant_id,
        user_id="user_a",
        ttl_s=600,
        metadata={"agent_config_hash": "test"},
    )


def _payload(*, previous_turn_handle: str | None = None) -> dict:
    payload = {
        "protocol": "native-agent-runner.llm-turn.v1",
        "model": "gpt-5.5",
        "system_prompt": "sys",
        "reasoning": {"effort": "low"},
        "tools": [
            {
                "id": "fs.read",
                "name": "fs_read",
                "description": "Read file.",
                "input_schema": {"type": "object"},
                "capability": "fs.read",
                "side_effect": "read",
            }
        ],
    }
    if previous_turn_handle:
        payload["previous_turn_handle"] = previous_turn_handle
        payload["observations"] = [{"call_id": "call_1", "tool_name": "fs_read", "output": {"ok": True}}]
    else:
        payload["instruction"] = "Read notes."
    return payload


def test_llm_gateway_validates_token_and_returns_opaque_turn_handle() -> None:
    manager = _token_manager()
    seen_previous_ids: list[str | None] = []

    def factory(_claims, _config):
        index = len(seen_previous_ids)

        class Adapter:
            def next_turn(self, request):
                seen_previous_ids.append(request.previous_turn_handle)
                if index == 0:
                    return ModelTurn(
                        response_id="provider_response_secret_1",
                        tool_calls=(ToolCall("call_1", "fs_read", {"path": "notes.md"}),),
                        usage={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
                    )
                return ModelTurn(
                    response_id="provider_response_secret_2",
                    final_text="done",
                    usage={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
                )

        return Adapter()

    gateway = LlmGatewayBackend(token_manager=manager, provider_adapter_factory=factory)
    token = _llm_token(manager)

    first = gateway.handle_turn(token, _payload())
    assert first["turn_handle"].startswith("turn_")
    assert "provider_response_secret_1" not in json.dumps(first)
    assert first["tool_calls"][0]["name"] == "fs_read"

    second = gateway.handle_turn(token, _payload(previous_turn_handle=first["turn_handle"]))
    assert second["final_text"] == "done"
    assert seen_previous_ids == [None, "provider_response_secret_1"]
    assert gateway.tenant_usage("tenant_a")["total_tokens"] == 14

    other_model = _payload()
    other_model["model"] = "other-model"
    assert gateway.handle_turn(token, other_model)["turn_handle"].startswith("turn_")


def test_llm_gateway_rejects_cross_run_turn_handle() -> None:
    manager = _token_manager()
    gateway = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, _config: FakeModelAdapter(
            turns=[ModelTurn(response_id="provider_1", final_text="done")]
        ),
    )
    first = gateway.handle_turn(_llm_token(manager, run_id="run_a"), _payload())

    with pytest.raises(PermissionDenied):
        gateway.handle_turn(
            _llm_token(manager, run_id="run_b"),
            _payload(previous_turn_handle=first["turn_handle"]),
        )


def test_llm_gateway_http_endpoint_and_usage(tmp_path: Path) -> None:
    manager = _token_manager()
    gateway = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, _config: FakeModelAdapter(
            turns=[ModelTurn(response_id="provider_1", final_text="done", usage={"total_tokens": 9})]
        ),
    )
    server = create_llm_gateway_server(gateway, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        _wait_http_ready(base_url)
        with pytest.raises(HTTPError) as exc_info:
            _json_post(f"{base_url}/internal/llm/turns", _payload())
        assert exc_info.value.code == 401

        result = _json_post(
            f"{base_url}/internal/llm/turns",
            _payload(),
            token=_llm_token(manager),
        )
        assert result["final_text"] == "done"
        usage = _json_get(f"{base_url}/internal/llm/tenants/tenant_a/usage", token="admin")
        assert usage["calls"] == 1
        assert usage["total_tokens"] == 9
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_llm_gateway_http_normalizes_model_adapter_error() -> None:
    manager = _token_manager()

    class FailingAdapter:
        def next_turn(self, _request):
            raise ModelAdapterError(
                "provider overloaded",
                provider_error_code="gateway_server_error",
                retryable=True,
            )

    gateway = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, _config: FailingAdapter(),
    )
    server = create_llm_gateway_server(gateway, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        _wait_http_ready(base_url)
        with pytest.raises(HTTPError) as exc_info:
            _json_post(
                f"{base_url}/internal/llm/turns",
                _payload(),
                token=_llm_token(manager),
            )
        body = json.loads(exc_info.value.read().decode("utf-8"))
        assert exc_info.value.code == 503
        assert body["error_code"] == "gateway_server_error"
        assert body["retryable"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_runner_backend_can_use_http_llm_gateway_end_to_end(tmp_path: Path) -> None:
    manager = _token_manager()
    gateway = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, _config: FakeModelAdapter(
            turns=[ModelTurn(response_id="provider_1", final_text="gateway done", usage={"total_tokens": 11})]
        ),
    )
    server = create_llm_gateway_server(gateway, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    gateway_url = f"http://127.0.0.1:{server.server_address[1]}"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("notes\n", encoding="utf-8")
    try:
        _wait_http_ready(gateway_url)
        runner_backend = RunnerBackend(
            run_root=tmp_path / "runs",
            token_manager=manager,
            allowed_workspace_roots=(workspace,),
            llm_gateway_url=f"{gateway_url}/internal/llm/turns",
        )
        submission = runner_backend.submit_run(
            BackendRunRequest(
                tenant_id="tenant_a",
                user_id="user_a",
                workspace_root=workspace,
                instruction="Finish through gateway.",
                runtime_config=runtime_config("run.finish"),
            )
        )
        assert runner_backend.wait_for_run(submission.run_id, timeout_s=20) == "completed"
        result = runner_backend.result(submission.run_id, submission.run_token)
        assert result["final_text"] == "gateway done"
        assert runner_backend.tenant_usage("tenant_a")["total_tokens"] == 11
        assert gateway.tenant_usage("tenant_a")["total_tokens"] == 11
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_fake_full_stack_contract_propose_proposal_usage_and_auth(tmp_path: Path) -> None:
    manager = _token_manager()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("alpha notes\n", encoding="utf-8")
    sentinel_key = "sk-test-hidden-provider-key"
    adapters: dict[str, FakeModelAdapter] = {}

    def factory(claims, _config):
        if claims.run_id not in adapters:
            adapters[claims.run_id] = FakeModelAdapter(
                turns=[
                    ModelTurn(
                        response_id="provider_1",
                        tool_calls=(
                            fake_tool_call("fs_read", {"path": "notes.md"}, "call_read"),
                            fake_tool_call(
                                "fs_write",
                                {
                                    "path": "SUMMARY.md",
                                    "content": "Summary from fake gateway\n",
                                    "create_dirs": False,
                                },
                                "call_write",
                            ),
                        ),
                        usage={"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
                    ),
                    ModelTurn(
                        response_id="provider_2",
                        tool_calls=(
                            fake_tool_call(
                                "run_finish",
                                {"summary": "Created SUMMARY.md", "outputs": ["SUMMARY.md"]},
                                "call_finish",
                            ),
                        ),
                        usage={"input_tokens": 2, "output_tokens": 5, "total_tokens": 7},
                    ),
                ]
            )
        return adapters[claims.run_id]

    gateway = LlmGatewayBackend(token_manager=manager, provider_adapter_factory=factory)
    gateway_server = create_llm_gateway_server(gateway, host="127.0.0.1", port=0, admin_token="gateway-admin")
    gateway_thread = threading.Thread(target=gateway_server.serve_forever, daemon=True)
    gateway_thread.start()
    gateway_url = f"http://127.0.0.1:{gateway_server.server_address[1]}"
    runner_backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url=f"{gateway_url}/internal/llm/turns",
    )
    runner_server = create_backend_server(runner_backend, host="127.0.0.1", port=0, admin_token="runner-admin")
    runner_thread = threading.Thread(target=runner_server.serve_forever, daemon=True)
    runner_thread.start()
    runner_url = f"http://127.0.0.1:{runner_server.server_address[1]}"
    try:
        _wait_http_ready(gateway_url)
        _wait_http_ready(runner_url)
        created = _json_post(
            f"{runner_url}/v1/runs",
            {
                "tenant_id": "tenant_a",
                "user_id": "user_a",
                "workspace_root": str(workspace),
                "instruction": "Read notes.md and propose SUMMARY.md.",
                "mode": "propose",
                "runtime_config": runtime_config("fs.read", "fs.write", "run.finish").to_json(),
            },
            token="runner-admin",
        )
        run_id = created["run_id"]
        run_token = created["run_token"]
        assert runner_backend.wait_for_run(run_id, timeout_s=5) == "completed"

        with pytest.raises(HTTPError) as exc_info:
            urlopen(Request(f"{runner_url}/v1/runs/{run_id}/result", method="GET"), timeout=5)
        assert exc_info.value.code == 401

        assert not workspace.joinpath("SUMMARY.md").exists()
        proposal = _json_get(f"{runner_url}/v1/runs/{run_id}/proposal", token=run_token)
        assert proposal["ready"] is True
        assert proposal["proposal_hash"]
        assert proposal["diff_sha256"]
        assert proposal["files"][0]["path"] == "SUMMARY.md"
        proposed_file = _json_get(f"{runner_url}/v1/runs/{run_id}/proposal/files/SUMMARY.md", token=run_token)
        assert proposed_file["content"] == "Summary from fake gateway\n"
        result = _json_get(f"{runner_url}/v1/runs/{run_id}/result", token=run_token)
        assert result["ready"] is True
        assert result["metrics"]["total_tokens"] == 14
        assert result["proposal"]["proposal_hash"] == proposal["proposal_hash"]
        events = _json_get(f"{runner_url}/v1/runs/{run_id}/events", token=run_token)["events"]
        assert events[0]["type"] == "run.started"
        assert events[-1]["type"] == "run.finished"
        assert any(event["type"] == "workspace.proposal.updated" for event in events)
        assert runner_backend.tenant_usage("tenant_a")["total_tokens"] == 14
        assert gateway.tenant_usage("tenant_a")["total_tokens"] == 14
        run_text = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in Path(result["run_dir"]).rglob("*")
            if path.is_file()
        )
        assert sentinel_key not in run_text
        assert "OPENAI_API_KEY" not in run_text
    finally:
        runner_server.shutdown()
        runner_server.server_close()
        runner_thread.join(timeout=5)
        gateway_server.shutdown()
        gateway_server.server_close()
        gateway_thread.join(timeout=5)


def _json_post(url: str, payload: dict, *, token: str | None = None) -> dict:
    return http_json(url, payload, token=token)


def _json_get(url: str, *, token: str) -> dict:
    return http_json(url, token=token, method="GET")


def _wait_http_ready(base_url: str, *, timeout_s: float = 15.0) -> None:
    wait_http_ready(base_url, timeout_s=timeout_s)
