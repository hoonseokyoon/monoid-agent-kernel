from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from native_agent_runner.reference.backend.http import create_backend_server
from native_agent_runner.reference.backend.service import BackendRunRequest, RunnerBackend
from native_agent_runner.reference._shared.tokens import TokenError, TokenManager
from native_agent_runner.errors import PermissionDenied
from native_agent_runner.permissions import PermissionPolicy
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call
from native_agent_runner.shell import ShellPolicy
from native_agent_runner.tools.policy import ToolPolicy
from native_agent_runner.web import WebPolicy


def _token_manager() -> TokenManager:
    return TokenManager.from_secret("x" * 32)


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("notes\n", encoding="utf-8")
    return workspace


def _python_command(code: str) -> str:
    escaped = code.replace('"', '\\"')
    return f'python -c "{escaped}"'


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

    claims = manager.verify(
        token,
        kind="run_access",
        audience="native-agent-runner.backend",
        run_id="run_1",
    )
    assert claims.tenant_id == "tenant_a"
    with pytest.raises(TokenError):
        manager.verify(token, kind="llm_gateway", audience="csp.llm-gateway")
    with pytest.raises(TokenError):
        manager.verify(token, kind="run_access", audience="native-agent-runner.backend", run_id="other")


def test_backend_submits_run_issues_tokens_and_returns_status_result_usage(tmp_path: Path) -> None:
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
        )
    )
    assert submission.run_token
    assert submission.status == "queued"

    final_status = backend.wait_for_run(submission.run_id, timeout_s=5)
    assert final_status == "completed"
    status = backend.status(submission.run_id, submission.run_token)
    assert status["status"] == "completed"
    assert status["tenant_id"] == "tenant_a"
    assert status["last_event_type"] == "run.finished"

    result = backend.result(submission.run_id, submission.run_token)
    assert result["ready"] is True
    assert result["final_text"] == "done"
    assert result["error_code"] == ""
    assert result["manifest_path"].endswith("manifest.json")
    assert result["metrics"]["total_tokens"] == 10
    events = backend.events(submission.run_id, submission.run_token)
    assert events["events"][0]["type"] == "run.started"
    usage = backend.tenant_usage("tenant_a")
    assert usage == {
        "tenant_id": "tenant_a",
        "runs": 1,
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
        "web_search_calls": 0,
        "web_fetch_calls": 0,
        "web_context_calls": 0,
        "web_failed_calls": 0,
        "web_result_count": 0,
        "web_bytes_returned": 0,
        "web_context_source_count": 0,
        "web_context_bytes_returned": 0,
    }

    run_files = "\n".join(
        path.read_text(encoding="utf-8")
        for path in submission.run_dir.glob("*.json*")
        if path.is_file()
    )
    assert captured_gateway_tokens
    assert captured_gateway_tokens[0] not in run_files


def test_backend_request_workspace_backend_is_written_to_manifest(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(tmp_path, workspace, [])

    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Summarize notes.",
            mode="propose",
            workspace_backend="staging",
        )
    )

    assert backend.wait_for_run(submission.run_id, timeout_s=5) == "completed"
    manifest = json.loads(submission.run_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert manifest["workspace_backend"] == "staging"
    assert manifest["workspace_base_path"] == "workspace.base.json"


def test_backend_request_tool_policy_reaches_agent_manifest(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    token_manager = _token_manager()
    adapters: dict[str, FakeModelAdapter] = {}

    def factory(spec, _llm_gateway_token):
        adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
        adapters[spec.run_id] = adapter
        return adapter

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
    )

    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Finish.",
            tool_policy=ToolPolicy(allowed_tools=("run.finish",)),
        )
    )
    assert backend.wait_for_run(submission.run_id, timeout_s=5) == "completed"

    assert {tool.id for tool in adapters[submission.run_id].requests[0].tools} == {"run.finish"}
    manifest = json.loads(submission.run_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert manifest["tool_policy"]["allowed_tools"] == ["run.finish"]
    assert {tool["id"] for tool in manifest["tool_specs"]} == {"run.finish"}


def test_backend_request_permission_policy_reaches_agent_manifest_and_execution(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    workspace.joinpath(".env").write_text("x", encoding="utf-8")
    token_manager = _token_manager()

    def factory(_spec, _llm_gateway_token):
        return FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="turn_1",
                    tool_calls=(fake_tool_call("fs_read", {"path": ".env"}, "call_env"),),
                ),
                ModelTurn(final_text="done"),
            ]
        )

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=token_manager,
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
            permission_policy=PermissionPolicy(deny_patterns=(".env",), redact_patterns=(".env",)),
        )
    )
    assert backend.wait_for_run(submission.run_id, timeout_s=5) == "completed"

    manifest = json.loads(submission.run_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert manifest["permission_policy"] == {
        "deny_patterns": [".env"],
        "redact_patterns": [".env"],
    }
    events = backend.events(submission.run_id, submission.run_token)["events"]
    assert any(
        event["type"] == "tool.call.failed"
        and event["data"]["call_id"] == "call_env"
        and event["data"]["error_code"] == "permission_denied"
        for event in events
    )


def test_backend_request_web_policy_reaches_manifest_and_requires_gateway_url(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    token_manager = _token_manager()

    backend_without_gateway = RunnerBackend(
        run_root=tmp_path / "no-web-runs",
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=lambda _spec, _token: FakeModelAdapter(turns=[ModelTurn(final_text="done")]),
    )
    with pytest.raises(ValueError, match="web_gateway_url"):
        backend_without_gateway.submit_run(
            BackendRunRequest(
                tenant_id="tenant_a",
                user_id="user_a",
                workspace_root=workspace,
                instruction="Use web.",
                web_policy=WebPolicy(enabled=True),
            )
        )

    adapters: dict[str, FakeModelAdapter] = {}

    def factory(spec, _llm_gateway_token):
        adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
        adapters[spec.run_id] = adapter
        return adapter

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        web_gateway_url="http://web-gateway.internal",
        model_adapter_factory=factory,
    )

    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Use web.",
            web_policy=WebPolicy(enabled=True, context_enabled=True, allowed_domains=("docs.example.test",)),
        )
    )
    assert backend.wait_for_run(submission.run_id, timeout_s=5) == "completed"
    assert {"web.search", "web.fetch", "web.context"}.issubset(
        {tool.id for tool in adapters[submission.run_id].requests[0].tools}
    )
    manifest = json.loads(submission.run_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert manifest["web_policy"]["enabled"] is True
    assert manifest["web_policy"]["context_enabled"] is True
    assert manifest["web_policy"]["allowed_domains"] == ["docs.example.test"]


def test_backend_shell_policy_auto_approves_and_does_not_leak_provider_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    workspace = _workspace(tmp_path)
    token_manager = _token_manager()
    adapters: dict[str, FakeModelAdapter] = {}

    def factory(spec, _llm_gateway_token):
        adapter = FakeModelAdapter(
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
        adapters[spec.run_id] = adapter
        return adapter

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
    )

    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Use shell.",
            shell_policy=ShellPolicy(enabled=True),
        )
    )
    assert backend.wait_for_run(submission.run_id, timeout_s=5) == "completed"

    assert "shell.exec" in {tool.id for tool in adapters[submission.run_id].requests[0].tools}
    assert not workspace.joinpath("BACKEND.md").exists()
    manifest = json.loads(submission.run_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert manifest["shell_policy"]["enabled"] is True
    proposal_file = submission.run_dir.joinpath("proposal", "files", "BACKEND.md")
    assert proposal_file.read_text(encoding="utf-8") == "None"
    run_text = "\n".join(path.read_text(encoding="utf-8") for path in submission.run_dir.rglob("*.json*"))
    assert "provider-secret" not in run_text
    events = backend.events(submission.run_id, submission.run_token)["events"]
    assert any(event["type"] == "tool.approval.approved" for event in events)
    assert any(event["type"] == "shell.exec.finished" for event in events)


def test_backend_exposes_background_job_artifacts(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    token_manager = _token_manager()
    adapters: dict[str, FakeModelAdapter] = {}

    def factory(spec, _llm_gateway_token):
        adapter = FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="turn_1",
                    tool_calls=(
                        fake_tool_call(
                            "shell_exec",
                            {
                                "command": _python_command(
                                    "import time; from pathlib import Path; "
                                    "time.sleep(0.2); print('job stdout'); "
                                    "Path('JOB.md').write_text('done', encoding='utf-8')"
                                ),
                                "background": True,
                            },
                            "call_shell",
                        ),
                    ),
                ),
                ModelTurn(response_id="turn_2"),
                ModelTurn(final_text="done"),
            ]
        )
        adapters[spec.run_id] = adapter
        return adapter

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
    )
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Run a background job.",
            shell_policy=ShellPolicy(enabled=True),
        )
    )

    assert backend.wait_for_run(submission.run_id, timeout_s=5) == "completed"
    jobs = backend.jobs(submission.run_id, submission.run_token)["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["status"] == "exited"
    job_id = jobs[0]["job_id"]
    status = backend.job_status(submission.run_id, submission.run_token, job_id)
    assert status["job"]["job_id"] == job_id
    logs = backend.job_logs(submission.run_id, submission.run_token, job_id, stream="stdout")
    assert "job stdout" in logs["content"]
    cancel = backend.cancel_job(submission.run_id, submission.run_token, job_id)
    assert cancel["cancel_requested"] is True
    events = backend.events(submission.run_id, submission.run_token)["events"]
    assert any(event["type"] == "run.resumed" for event in events)
    assert any(
        obs.output.get("type") == "background_job_result"
        for obs in adapters[submission.run_id].requests[-1].observations
    )


def test_backend_rejects_bad_run_token_and_workspace_escape(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    captured_gateway_tokens: list[str] = []
    backend = _backend(tmp_path, workspace, captured_gateway_tokens)
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Summarize notes.",
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
            )
        )


def test_backend_http_create_status_result_events_and_usage(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    captured_gateway_tokens: list[str] = []
    backend = _backend(tmp_path, workspace, captured_gateway_tokens)
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
                "permission_policy": {"deny_patterns": [".env"], "redact_patterns": ["*.key"]},
                "tool_policy": {"allowed_tools": ["run.finish"]},
                "shell_policy": {"enabled": True, "approval_mode": "backend"},
            },
            token="admin",
        )
        run_id = created["run_id"]
        run_token = created["run_token"]
        backend.wait_for_run(run_id, timeout_s=5)

        status = _json_get(f"{base_url}/v1/runs/{run_id}/status", token=run_token)
        assert status["status"] == "completed"
        result = _json_get(f"{base_url}/v1/runs/{run_id}/result", token=run_token)
        assert result["final_text"] == "done"
        assert result["error_code"] == ""
        manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
        assert manifest["permission_policy"] == {
            "deny_patterns": [".env"],
            "redact_patterns": ["*.key"],
        }
        assert manifest["tool_policy"]["allowed_tools"] == ["run.finish"]
        assert manifest["shell_policy"]["enabled"] is True
        events = _json_get(f"{base_url}/v1/runs/{run_id}/events?from_seq=1", token=run_token)
        assert events["events"][0]["seq"] == 1
        usage = _json_get(f"{base_url}/v1/tenants/tenant_a/usage", token="admin")
        assert usage["total_tokens"] == 10
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_backend_http_exposes_proposal_snapshot_and_file(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    token_manager = _token_manager()

    def factory(_spec, _llm_gateway_token):
        return FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="turn_1",
                    tool_calls=(
                        fake_tool_call("fs_read", {"path": "notes.md"}, "call_read"),
                        fake_tool_call(
                            "fs_write",
                            {
                                "path": "SUMMARY.md",
                                "content": "Observed proposal\n",
                                "create_dirs": False,
                            },
                            "call_write",
                        ),
                    ),
                ),
                ModelTurn(
                    response_id="turn_2",
                    tool_calls=(
                        fake_tool_call(
                            "run_finish",
                            {"summary": "Created SUMMARY.md", "outputs": ["SUMMARY.md"]},
                            "call_finish",
                        ),
                    ),
                ),
            ]
        )

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
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
                "instruction": "Create a proposal.",
                "mode": "propose",
            },
            token="admin",
        )
        run_id = created["run_id"]
        run_token = created["run_token"]
        backend.wait_for_run(run_id, timeout_s=5)

        assert not workspace.joinpath("SUMMARY.md").exists()
        proposal = _json_get(f"{base_url}/v1/runs/{run_id}/proposal", token=run_token)
        assert proposal["ready"] is True
        assert proposal["proposal_hash"]
        assert proposal["diff_sha256"]
        assert proposal["files"][0]["path"] == "SUMMARY.md"
        proposed_file = _json_get(
            f"{base_url}/v1/runs/{run_id}/proposal/files/SUMMARY.md",
            token=run_token,
        )
        assert proposed_file["encoding"] == "utf-8"
        assert proposed_file["content"] == "Observed proposal\n"
        result = _json_get(f"{base_url}/v1/runs/{run_id}/result", token=run_token)
        assert result["proposal"]["files"][0]["path"] == "SUMMARY.md"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_backend_http_cancel_marks_run_limited_with_code(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    token_manager = _token_manager()

    class SlowAdapter:
        def next_turn(self, _request):
            time.sleep(0.2)
            return ModelTurn(response_id="turn_1", final_text="too late")

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=token_manager,
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
            },
            token="admin",
        )
        run_id = created["run_id"]
        run_token = created["run_token"]
        cancelled = _json_request(f"{base_url}/v1/runs/{run_id}/cancel", {}, token=run_token)
        assert cancelled["cancel_requested"] is True
        assert cancelled["error_code"] == "cancelled"
        assert backend.wait_for_run(run_id, timeout_s=5) == "limited"
        status = _json_get(f"{base_url}/v1/runs/{run_id}/status", token=run_token)
        assert status["status"] == "limited"
        assert status["error_code"] == "cancelled"
        result = _json_get(f"{base_url}/v1/runs/{run_id}/result", token=run_token)
        assert result["status"] == "limited"
        assert result["error_code"] == "cancelled"
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
        except Exception as exc:  # pragma: no cover - only exercised under startup races.
            last_error = exc
            time.sleep(0.02)
    raise TimeoutError(f"server did not become ready: {last_error}")
