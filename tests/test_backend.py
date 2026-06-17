from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from conftest import runtime_config, tool_binding

from native_agent_runner.core.tool_surface import ToolScope
from native_agent_runner.errors import PermissionDenied
from native_agent_runner.permissions import PermissionPolicy
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call
from native_agent_runner.reference._shared.tokens import TokenError, TokenManager
from native_agent_runner.reference.backend.http import create_backend_server
from native_agent_runner.reference.backend.service import BackendRunRequest, RunnerBackend


def _token_manager() -> TokenManager:
    return TokenManager.from_secret("x" * 32)


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("notes\n", encoding="utf-8")
    return workspace


def _python_command(code: str) -> str:
    return f'python -c "{code.replace(chr(34), chr(92) + chr(34))}"'


def _default_config():
    return runtime_config("fs.read", "fs.write", "run.finish")


def _backend(tmp_path: Path, workspace: Path, captured_gateway_tokens: list[str]) -> RunnerBackend:
    token_manager = _token_manager()

    def factory(spec, llm_gateway_token):
        captured_gateway_tokens.append(llm_gateway_token)
        claims = token_manager.verify(
            llm_gateway_token,
            kind="llm_gateway",
            audience="csp.llm-gateway",
            run_id=spec.run_id,
        )
        assert claims.tenant_id == "tenant_a"
        return FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="turn_1",
                    final_text="done",
                    usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                )
            ]
        )

    return RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
    )


def test_token_manager_binds_kind_audience_run_and_expiry() -> None:
    manager = _token_manager()
    token = manager.issue(
        kind="run_access",
        audience="native-agent-runner.backend",
        run_id="run_1",
        tenant_id="tenant_a",
        user_id="user_a",
        ttl_s=60,
    )

    claims = manager.verify(token, kind="run_access", audience="native-agent-runner.backend", run_id="run_1")
    assert claims.tenant_id == "tenant_a"
    with pytest.raises(TokenError):
        manager.verify(token, kind="llm_gateway", audience="csp.llm-gateway")
    with pytest.raises(TokenError):
        manager.verify(token, kind="run_access", audience="native-agent-runner.backend", run_id="other")


def test_backend_requires_runtime_config() -> None:
    workspace = Path(".").resolve()
    backend = RunnerBackend(
        run_root=workspace / "runs-test-unused",
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=lambda _spec, _token: FakeModelAdapter(turns=[ModelTurn(final_text="done")]),
    )

    with pytest.raises(ValueError, match="agent_definition or runtime_config is required"):
        backend.submit_run(
            BackendRunRequest(
                tenant_id="tenant_a",
                user_id="user_a",
                workspace_root=workspace,
                instruction="Run.",
            )
        )


def test_backend_submits_run_issues_tokens_and_returns_usage(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    captured_gateway_tokens: list[str] = []
    backend = _backend(tmp_path, workspace, captured_gateway_tokens)

    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Summarize notes.",
            mode="propose",
            runtime_config=_default_config(),
        )
    )

    assert backend.wait_for_run(submission.run_id, timeout_s=5) == "completed"
    status = backend.status(submission.run_id, submission.run_token)
    assert status["status"] == "completed"
    result = backend.result(submission.run_id, submission.run_token)
    assert result["final_text"] == "done"
    assert result["metrics"]["total_tokens"] == 10
    usage = backend.tenant_usage("tenant_a")
    assert usage["runs"] == 1
    assert usage["total_tokens"] == 10
    run_files = "\n".join(path.read_text(encoding="utf-8") for path in submission.run_dir.glob("*.json*") if path.is_file())
    assert captured_gateway_tokens
    assert captured_gateway_tokens[0] not in run_files


def test_backend_permission_policy_reaches_manifest_and_execution(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    workspace.joinpath(".env").write_text("x", encoding="utf-8")

    def factory(_spec, _llm_gateway_token):
        return FakeModelAdapter(
            turns=[
                ModelTurn(response_id="turn_1", tool_calls=(fake_tool_call("fs_read", {"path": ".env"}, "call_env"),)),
                ModelTurn(final_text="done"),
            ]
        )

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
    )
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Read env.",
            runtime_config=runtime_config("fs.read", "run.finish"),
            permission_policy=PermissionPolicy(deny_patterns=(".env",), redact_patterns=(".env",)),
        )
    )

    assert backend.wait_for_run(submission.run_id, timeout_s=5) == "completed"
    manifest = json.loads(submission.run_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert manifest["permission_policy"] == {"deny_patterns": [".env"], "redact_patterns": [".env"]}
    events = backend.events(submission.run_id, submission.run_token)["events"]
    assert any(event["type"] == "tool.call.failed" and event["data"]["call_id"] == "call_env" for event in events)


def test_backend_web_binding_requires_gateway_url(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=lambda _spec, _token: FakeModelAdapter(turns=[ModelTurn(final_text="done")]),
    )

    with pytest.raises(ValueError, match="web_gateway_url"):
        backend.submit_run(
            BackendRunRequest(
                tenant_id="tenant_a",
                user_id="user_a",
                workspace_root=workspace,
                instruction="Use web.",
                runtime_config=runtime_config("web.search", "run.finish"),
            )
        )


def test_backend_shell_binding_auto_approves_without_provider_env_leak(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    workspace = _workspace(tmp_path)

    def factory(_spec, _llm_gateway_token):
        return FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="turn_1",
                    tool_calls=(
                        fake_tool_call(
                            "shell_exec",
                            {
                                "command": _python_command(
                                    "import os; from pathlib import Path; "
                                    "Path('BACKEND.md').write_text(str(os.getenv('OPENAI_API_KEY')), encoding='utf-8')"
                                )
                            },
                            "call_shell",
                        ),
                    ),
                ),
                ModelTurn(final_text="done"),
            ]
        )

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
    )
    config = runtime_config(
        bindings=(
            tool_binding(
                "shell.exec",
                runtime={"shell": {"approval_mode": "auto-approve"}},
                scope=ToolScope(env_allowlist=()),
            ),
            tool_binding("run.finish"),
        )
    )
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Use shell.",
            runtime_config=config,
        )
    )

    assert backend.wait_for_run(submission.run_id, timeout_s=5) == "completed"
    assert submission.run_dir.joinpath("proposal", "files", "BACKEND.md").read_text(encoding="utf-8") == "None"
    run_text = "\n".join(path.read_text(encoding="utf-8") for path in submission.run_dir.rglob("*.json*"))
    assert "provider-secret" not in run_text


def test_backend_rejects_bad_run_token_and_workspace_escape(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(tmp_path, workspace, [])
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Summarize notes.",
            runtime_config=_default_config(),
        )
    )
    backend.wait_for_run(submission.run_id, timeout_s=5)

    with pytest.raises(PermissionDenied):
        backend.status(submission.run_id, "bad-token")

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(PermissionDenied):
        backend.submit_run(
            BackendRunRequest(
                tenant_id="tenant_a",
                user_id="user_a",
                workspace_root=outside,
                instruction="No.",
                runtime_config=_default_config(),
            )
        )


def test_backend_http_create_status_result_events_and_usage(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(tmp_path, workspace, [])
    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        _wait_http_ready(base_url)
        with pytest.raises(HTTPError) as exc_info:
            _json_request(
                f"{base_url}/v1/runs",
                {"tenant_id": "tenant_a", "user_id": "user_a", "workspace_root": str(workspace), "instruction": "Run."},
            )
        assert exc_info.value.code == 401

        created = _json_request(
            f"{base_url}/v1/runs",
            {
                "tenant_id": "tenant_a",
                "user_id": "user_a",
                "workspace_root": str(workspace),
                "instruction": "Run.",
                "runtime_config": _default_config().to_json(),
            },
            token="admin",
        )
        run_id = created["run_id"]
        run_token = created["run_token"]
        assert backend.wait_for_run(run_id, timeout_s=5) == "completed"
        status = _json_get(f"{base_url}/v1/runs/{run_id}/status", token=run_token)
        assert status["status"] == "completed"
        result = _json_get(f"{base_url}/v1/runs/{run_id}/result", token=run_token)
        assert result["final_text"] == "done"
        events = _json_get(f"{base_url}/v1/runs/{run_id}/events?from_seq=1", token=run_token)
        assert events["events"][0]["seq"] == 1
        usage = _json_get(f"{base_url}/v1/tenants/tenant_a/usage", token="admin")
        assert usage["total_tokens"] == 10
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_backend_http_cancel_marks_run_limited_with_code(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    class SlowAdapter:
        def next_turn(self, _request):
            time.sleep(0.2)
            return ModelTurn(response_id="turn_1", final_text="too late")

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=lambda _spec, _token: SlowAdapter(),
    )
    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        _wait_http_ready(base_url)
        created = _json_request(
            f"{base_url}/v1/runs",
            {
                "tenant_id": "tenant_a",
                "user_id": "user_a",
                "workspace_root": str(workspace),
                "instruction": "Run slowly.",
                "runtime_config": _default_config().to_json(),
            },
            token="admin",
        )
        run_id = created["run_id"]
        run_token = created["run_token"]
        cancelled = _json_request(f"{base_url}/v1/runs/{run_id}/cancel", {}, token=run_token)
        assert cancelled["cancel_requested"] is True
        assert backend.wait_for_run(run_id, timeout_s=5) == "limited"
        status = _json_get(f"{base_url}/v1/runs/{run_id}/status", token=run_token)
        assert status["error_code"] == "cancelled"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _json_request(url: str, payload: dict, *, token: str | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, data=body, headers=headers, method="POST")
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _json_get(url: str, *, token: str) -> dict:
    request = Request(url, headers={"Authorization": f"Bearer {token}"}, method="GET")
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_http_ready(base_url: str, *, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            _json_get(f"{base_url}/healthz", token="unused")
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.02)
    raise TimeoutError(f"server did not become ready: {last_error}")
