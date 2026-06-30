from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from support.http import http_get_json as _json_get

from native_agent_runner.core.agents import AgentRuntimeConfig, RegistryToolRef, ToolBinding
from native_agent_runner.core.tool_surface import ToolGuidance
from native_agent_runner.providers.base import ModelRequest, ModelTurn
from native_agent_runner.providers.fake import fake_tool_call
from native_agent_runner.reference._shared.tokens import TokenManager
from native_agent_runner.reference.backend.http import create_backend_server
from native_agent_runner.reference.backend.service import BackendRunRequest, RunnerBackend


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("hello\n", encoding="utf-8")
    return workspace


def _token_manager() -> TokenManager:
    return TokenManager.from_secret("test-secret-" * 4)


def _binding(tool_id: str, *, guidance: str = "") -> ToolBinding:
    return ToolBinding(
        binding_id=tool_id,
        model_name=tool_id.replace(".", "_"),
        ref=RegistryToolRef(tool_id),
        guidance=ToolGuidance(summary=guidance),
        title=tool_id,
    )


def _config(version: int, *bindings: ToolBinding) -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        definition_id="backend-agent",
        config_version=version,
        tools=bindings,
    )


class _BlockingAdapter:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.first_call_started = threading.Event()
        self.allow_first_return = threading.Event()

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        if len(self.requests) == 1:
            self.first_call_started.set()
            assert self.allow_first_return.wait(timeout=5)
            return ModelTurn(
                response_id="turn_1",
                tool_calls=(fake_tool_call("fs_read", {"path": "notes.md"}, "read1"),),
            )
        return ModelTurn(final_text="done")


def test_backend_runtime_config_endpoint_updates_next_turn(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    adapter = _BlockingAdapter()
    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=lambda _spec, _token: adapter,
    )
    initial = _config(1, _binding("fs.read", guidance="initial read"), _binding("run.finish"))
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Read and finish.",
            runtime_config=initial,
        )
    )
    assert adapter.first_call_started.wait(timeout=5)

    current = backend.runtime_config(submission.run_id, submission.run_token)
    assert current["config_version"] == 1
    replacement = _config(2, _binding("fs.read", guidance="replacement read"), _binding("run.finish"))
    updated = backend.replace_runtime_config(
        submission.run_id,
        submission.run_token,
        expected_version=1,
        issuer="test",
        reason="replace guidance",
        config=replacement,
    )
    assert updated["config_version"] == 2
    assert updated["config_hash"] == replacement.config_hash
    adapter.allow_first_return.set()

    assert backend.wait_for_run(submission.run_id, timeout_s=5) == "completed"
    second_read = next(tool for tool in adapter.requests[1].tools if tool.id == "fs.read")
    assert "replacement read" in second_read.description


def test_backend_http_runtime_config_get_post_and_version_mismatch(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    adapter = _BlockingAdapter()
    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=lambda _spec, _token: adapter,
    )
    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        created = _json_post(
            f"{base_url}/v1/runs",
            {
                "tenant_id": "tenant_a",
                "user_id": "user_a",
                "workspace_root": str(workspace),
                "instruction": "Read and finish.",
                "runtime_config": _config(
                    1,
                    _binding("fs.read", guidance="http initial"),
                    _binding("run.finish"),
                ).to_json(),
            },
            token="admin",
        )
        run_id = created["run_id"]
        run_token = created["run_token"]
        assert adapter.first_call_started.wait(timeout=5)

        current = _json_get(f"{base_url}/v1/runs/{run_id}/runtime-config", token=run_token)
        assert current["ready"] is True
        assert current["config_version"] == 1

        mismatch = _json_post_raw(
            f"{base_url}/v1/runs/{run_id}/runtime-config",
            {
                "expected_version": 0,
                "issuer": "test",
                "reason": "bad version",
                "config": _config(2, _binding("run.finish")).to_json(),
            },
            token=run_token,
        )
        assert mismatch["status"] == 400

        invalid_tool = _json_post_raw(
            f"{base_url}/v1/runs/{run_id}/runtime-config",
            {
                "expected_version": 1,
                "issuer": "test",
                "reason": "bad tool",
                "config": _config(2, _binding("missing.tool")).to_json(),
            },
            token=run_token,
        )
        assert invalid_tool["status"] == 400
        assert "unknown registry tool" in invalid_tool["body"]["error"]

        updated = _json_post(
            f"{base_url}/v1/runs/{run_id}/runtime-config",
            {
                "expected_version": 1,
                "issuer": "test",
                "reason": "replace",
                "config": _config(
                    2,
                    _binding("fs.read", guidance="http replacement"),
                    _binding("run.finish"),
                ).to_json(),
            },
            token=run_token,
        )
        assert updated["config_version"] == 2
        adapter.allow_first_return.set()
        assert backend.wait_for_run(run_id, timeout_s=5) == "completed"
        second_read = next(tool for tool in adapter.requests[1].tools if tool.id == "fs.read")
        assert "http replacement" in second_read.description
    finally:
        adapter.allow_first_return.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _json_post(url: str, payload: dict, *, token: str) -> dict:
    result = _json_post_raw(url, payload, token=token)
    assert result["status"] < 400, result
    return result["body"]


def _json_post_raw(url: str, payload: dict, *, token: str) -> dict:
    # Captures the HTTP response (including 4xx error bodies) rather than raising, while
    # retrying transient connection-level errors under load (never an HTTPError, which is a
    # real response).
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urlopen(Request(url, data=body, headers=headers, method="POST"), timeout=5) as response:
                return {"status": response.status, "body": json.loads(response.read().decode("utf-8"))}
        except HTTPError as exc:
            return {"status": exc.code, "body": json.loads(exc.read().decode("utf-8"))}
        except (URLError, ConnectionError, OSError) as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    raise last_error if last_error is not None else RuntimeError("request failed without an error")
