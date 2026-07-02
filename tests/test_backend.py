from __future__ import annotations

import base64

from support.backend_harness import (
    BackendRunRequest,
    FakeModelAdapter,
    ModelTurn,
    Path,
    PermissionDenied,
    PermissionPolicy,
    RunnerBackend,
    TokenError,
    TokenManager,
    ToolScope,
    _BlockingAdapter,
    _backend,
    _default_config,
    _hitl_backend,
    _python_command,
    _running_hitl_tasks,
    _token_manager,
    _workspace,
    eventually,
    fake_tool_call,
    json,
    pytest,
    runtime_config,
    threading,
    time,
    tool_binding,
)
from monoid_agent_kernel.tools.base import ToolContext, ToolResult, ToolSpec

pytestmark = pytest.mark.integration


def test_backend_report_task_result_completes_parked_hitl_run(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    adapters: list = []
    backend = _hitl_backend(
        tmp_path,
        workspace,
        adapters,
        turns=[ModelTurn(response_id="r1", tool_calls=(fake_tool_call("hitl_request", {"prompt": "Pick a name"}, "c1"),))],
    )
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Name the project, ask me if unsure.",
            runtime_config=runtime_config("hitl.request"),
        )
    )
    run_id, token = submission.run_id, submission.run_token

    def _drain() -> None:
        for _ in range(1000):
            if backend._record(run_id).status in {"completed", "failed", "limited"}:
                return
            for task in _running_hitl_tasks(backend, run_id):
                try:
                    backend.report_task_result(run_id, token, task_id=task.job_id, result={"answer": "Ada"})
                except Exception:
                    pass
            time.sleep(0.01)

    responder = threading.Thread(target=_drain)
    responder.start()
    status = backend.wait_for_run(run_id, timeout_s=20)
    responder.join(timeout=5)

    assert status == "completed"
    hitl_obs = [
        obs
        for adapter in adapters
        for request in adapter.requests
        for obs in request.observations
        if obs.tool_name == "human_input"
    ]
    assert hitl_obs, "the human answer never reached the model through the backend"
    assert hitl_obs[0].output["answer"] == "Ada"


def test_backend_task_result_checkpoint_preserves_queued_messages(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    adapters: list = []
    backend = _hitl_backend(
        tmp_path,
        workspace,
        adapters,
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("hitl_request", {"prompt": "Pick a name"}, "c1"),),
            ),
            ModelTurn(response_id="r2", final_text="answered"),
        ],
    )
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Name the project, ask me if unsure.",
            runtime_config=runtime_config("hitl.request"),
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert eventually(lambda: backend._record(run_id).status == "awaiting_input")
    assert eventually(lambda: bool(_running_hitl_tasks(backend, run_id)))
    task = _running_hitl_tasks(backend, run_id)[0]

    queued = backend.send_message(run_id, token, "queued while approval is pending", message_id="queued-1")
    assert queued["status"] == "queued"
    assert eventually(lambda: backend._record(run_id).message_queue.qsize() == 1)

    backend.report_task_result(run_id, token, task_id=task.job_id, result={"answer": "Ada"})
    stored = backend.checkpoint_store.latest(run_id)

    assert stored is not None
    assert any(
        isinstance(message, dict) and message.get("id") == "queued-1"
        for message in stored.checkpoint.queued_messages
    )

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_backend_tool_approval_replay_checkpoint_preserves_queued_messages(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    observed: dict[str, object] = {}
    holder: dict[str, object] = {}

    class ApprovalProvider:
        calls = 0

        def get_tools(self, context: ToolContext | None = None) -> list[ToolSpec]:
            del context

            def handler(ctx: ToolContext, args: dict) -> ToolResult:
                del ctx, args
                self.calls += 1
                backend = holder["backend"]
                run_id = str(holder["run_id"])
                latest = backend.checkpoint_store.latest(run_id)
                assert latest is not None
                observed["queued_ids"] = [
                    message.get("id")
                    for message in latest.checkpoint.queued_messages
                    if isinstance(message, dict)
                ]
                observed["pending_replays"] = list(latest.checkpoint.pending_tool_approval_replays)
                return ToolResult(ok=True, content={"value": "ok"})

            return [
                ToolSpec(
                    id="demo.approval",
                    description="approval demo",
                    input_schema={"type": "object", "additionalProperties": True},
                    capability="",
                    side_effect="write",
                    handler=handler,
                )
            ]

    provider = ApprovalProvider()
    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=lambda _spec, _token: FakeModelAdapter(
            turns=[
                ModelTurn(tool_calls=(fake_tool_call("demo_approval", {"value": "ok"}, "call_1"),)),
                ModelTurn(final_text="park"),
                ModelTurn(final_text="done"),
            ]
        ),
        tool_providers=(provider,),
    )
    holder["backend"] = backend
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="use approval",
            runtime_config=runtime_config(
                bindings=(
                    tool_binding("demo.approval", authorization="ask"),
                    tool_binding("run.finish"),
                )
            ),
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token
    holder["run_id"] = run_id

    def pending_approval_tasks() -> list:
        record = backend._record(run_id)
        if record.loop is None or record.loop._session is None:
            return []
        return [
            task
            for task in record.loop._session.res.context.job_manager.list_jobs()
            if task.get("kind") == "tool_approval" and task.get("status") == "running"
        ]

    assert eventually(lambda: bool(pending_approval_tasks()))
    task_id = pending_approval_tasks()[0]["task_id"]
    queued = backend.send_message(run_id, token, "queued while approval replays", message_id="queued-approval")
    assert queued["status"] == "queued"
    backend.report_task_result(run_id, token, task_id=task_id, result={"approved": True})

    assert eventually(lambda: provider.calls == 1 and "queued_ids" in observed)
    assert "queued-approval" in observed["queued_ids"]
    assert observed["pending_replays"] == []

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_backend_recovered_tool_approval_replay_checkpoint_preserves_queued_messages(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    token_manager = _token_manager()
    observed: dict[str, object] = {}
    holder: dict[str, object] = {}

    class ApprovalProvider:
        calls = 0

        def get_tools(self, context: ToolContext | None = None) -> list[ToolSpec]:
            del context

            def handler(ctx: ToolContext, args: dict) -> ToolResult:
                del ctx, args
                self.calls += 1
                backend = holder["backend"]
                run_id = str(holder["run_id"])
                latest = backend.checkpoint_store.latest(run_id)
                assert latest is not None
                observed["queued_ids"] = [
                    message.get("id")
                    for message in latest.checkpoint.queued_messages
                    if isinstance(message, dict)
                ]
                observed["pending_replays"] = list(latest.checkpoint.pending_tool_approval_replays)
                return ToolResult(ok=True, content={"value": "ok"})

            return [
                ToolSpec(
                    id="demo.approval",
                    description="approval demo",
                    input_schema={"type": "object", "additionalProperties": True},
                    capability="",
                    side_effect="write",
                    handler=handler,
                )
            ]

    provider = ApprovalProvider()

    backend1 = RunnerBackend(
        run_root=run_root,
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=lambda _spec, _token: FakeModelAdapter(
            turns=[ModelTurn(tool_calls=(fake_tool_call("demo_approval", {"value": "ok"}, "call_1"),))]
        ),
        tool_providers=(provider,),
    )
    submission = backend1.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="use approval",
            runtime_config=runtime_config(
                bindings=(
                    tool_binding("demo.approval", authorization="ask"),
                    tool_binding("run.finish"),
                )
            ),
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token
    holder["run_id"] = run_id

    def pending_approval_tasks(backend: RunnerBackend) -> list:
        record = backend._record(run_id)
        if record.loop is None or record.loop._session is None:
            return []
        return [
            task
            for task in record.loop._session.res.context.job_manager.list_jobs()
            if task.get("kind") == "tool_approval" and task.get("status") == "running"
        ]

    assert eventually(lambda: bool(pending_approval_tasks(backend1)))
    assert eventually(lambda: backend1.checkpoint_store.latest(run_id) is not None)

    backend2 = RunnerBackend(
        run_root=run_root,
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=lambda _spec, _token: FakeModelAdapter(turns=[ModelTurn(final_text="done")]),
        tool_providers=(provider,),
    )
    backend2.max_recover_attempts = 10_000
    assert eventually(lambda: run_id in backend2.recover_runs() or run_id in backend2._records)
    assert eventually(lambda: bool(pending_approval_tasks(backend2)))
    holder["backend"] = backend2

    queued = backend2.send_message(run_id, token, "queued while recovered approval replays", message_id="queued-recovered")
    assert queued["status"] == "queued"
    task_id = pending_approval_tasks(backend2)[0]["task_id"]
    backend2.report_task_result(run_id, token, task_id=task_id, result={"approved": True})

    assert eventually(lambda: provider.calls == 1 and "queued_ids" in observed)
    assert "queued-recovered" in observed["queued_ids"]
    assert observed["pending_replays"] == []

    backend2.cancel_run(run_id, token)
    backend2.wait_for_run(run_id, timeout_s=20)
    backend1.shutdown(drain=True)


def test_backend_create_task_injects_into_running_run(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    adapters: list = []
    backend = _hitl_backend(
        tmp_path,
        workspace,
        adapters,
        turns=[ModelTurn(response_id="r1", tool_calls=(fake_tool_call("hitl_request", {"prompt": "Pick a name"}, "c1"),))],
    )
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Name it, ask me.",
            runtime_config=runtime_config("hitl.request"),
        )
    )
    run_id, token = submission.run_id, submission.run_token
    created: dict = {}

    def _drain() -> None:
        for _ in range(1000):
            if backend._record(run_id).status in {"completed", "failed", "limited"}:
                return
            running = _running_hitl_tasks(backend, run_id)
            # Once the model's task is parked, inject a backend-initiated task too.
            if running and "task_id" not in created:
                try:
                    created.update(backend.create_task(run_id, token, kind="hitl", request={"prompt": "backend asks"}))
                except Exception:
                    pass
            for task in _running_hitl_tasks(backend, run_id):
                try:
                    backend.report_task_result(run_id, token, task_id=task.job_id, result={"answer": "X"})
                except Exception:
                    pass
            time.sleep(0.01)

    responder = threading.Thread(target=_drain)
    responder.start()
    status = backend.wait_for_run(run_id, timeout_s=20)
    responder.join(timeout=5)

    assert status == "completed"
    assert created.get("task_id"), "backend-initiated create_task did not return a task id"
    hitl_obs = [
        obs
        for adapter in adapters
        for request in adapter.requests
        for obs in request.observations
        if obs.tool_name == "human_input"
    ]
    # Both the model-initiated and backend-initiated tasks were delivered.
    assert len(hitl_obs) >= 2


def test_backend_multi_turn_session_threads_two_user_messages(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    adapters: list = []
    backend = _hitl_backend(tmp_path, workspace, adapters, turns=[ModelTurn(response_id="r1", final_text="first")])
    backend.idle_timeout_s = 10.0
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hello",
            runtime_config=_default_config(),
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token

    # First turn settles -> session parks awaiting the next user message.
    assert eventually(lambda: backend._record(run_id).status == "awaiting_input")

    backend.send_message(run_id, token, content="again")
    # The follow-up message reaches the model as a second user turn.
    assert eventually(lambda: len([r for a in adapters for r in a.requests if r.instruction]) >= 2)

    backend.cancel_run(run_id, token)  # stop the open session
    status = backend.wait_for_run(run_id, timeout_s=20)
    assert status in {"completed", "limited", "failed"}

    instructions = [r.instruction for a in adapters for r in a.requests if r.instruction]
    assert "hello" in instructions
    assert "again" in instructions


def test_backend_single_turn_run_closes_after_first_settle(tmp_path: Path) -> None:
    # Without multi_turn the run closes after the first settle (no awaiting_input hang).
    workspace = _workspace(tmp_path)
    adapters: list = []
    backend = _hitl_backend(tmp_path, workspace, adapters, turns=[ModelTurn(response_id="r1", final_text="done")])
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hello",
            runtime_config=_default_config(),
        )
    )
    status = backend.wait_for_run(submission.run_id, timeout_s=20)
    assert status == "completed"


def test_backend_proposal_diff_returns_unified_diff(tmp_path: Path) -> None:
    # DX-4: the diff is available via a token-scoped API (not only via result() at run end,
    # and without reading run artifacts off disk).
    workspace = _workspace(tmp_path)
    adapters: list = []
    backend = _hitl_backend(
        tmp_path,
        workspace,
        adapters,
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("fs_write", {"path": "NEW.md", "content": "hi\n"}, "c1"),)),
            ModelTurn(response_id="r2", final_text="wrote NEW.md"),
        ],
    )
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="write NEW.md",
            runtime_config=_default_config(),
        )
    )
    backend.wait_for_run(submission.run_id, timeout_s=20)
    out = backend.proposal_diff(submission.run_id, submission.run_token)
    assert out["ready"] is True
    assert "NEW.md" in out["diff"]
    assert "hi" in out["diff"]


def test_backend_drain_ends_parked_multi_turn_sessions(tmp_path: Path) -> None:
    # DX-2: drain() cooperatively ends owned runs in one call, so a parked multi-turn session
    # reaches a terminal state (no dangling coroutine on the shared loop).
    workspace = _workspace(tmp_path)
    adapters: list = []
    backend = _hitl_backend(tmp_path, workspace, adapters, turns=[ModelTurn(response_id="r1", final_text="first")])
    backend.idle_timeout_s = 30.0
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hello",
            runtime_config=_default_config(),
            multi_turn=True,
        )
    )
    run_id = submission.run_id

    assert eventually(lambda: backend._record(run_id).status == "awaiting_input")
    pending = backend.drain(timeout_s=20)
    assert pending == []
    assert backend._record(run_id).status in {"completed", "failed", "limited"}


def test_token_manager_binds_kind_audience_run_and_expiry() -> None:
    manager = _token_manager()
    token = manager.issue(
        kind="run_access",
        audience="monoid.backend",
        run_id="run_1",
        tenant_id="tenant_a",
        user_id="user_a",
        ttl_s=60,
    )

    claims = manager.verify(token, kind="run_access", audience="monoid.backend", run_id="run_1")
    assert claims.tenant_id == "tenant_a"
    with pytest.raises(TokenError):
        manager.verify(token, kind="llm_gateway", audience="csp.llm-gateway")
    with pytest.raises(TokenError):
        manager.verify(token, kind="run_access", audience="monoid.backend", run_id="other")


def test_token_manager_emits_monoid_header_type() -> None:
    manager = _token_manager()
    token = manager.issue(
        kind="run_access",
        audience="monoid.backend",
        run_id="run_1",
        tenant_id="tenant_a",
        user_id="user_a",
        ttl_s=60,
    )
    header_raw = token.split(".", 1)[0]
    padded = header_raw + "=" * (-len(header_raw) % 4)
    header = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))

    assert header["typ"] == "MAK"


def test_token_manager_accepts_legacy_issuer_and_audience_for_migration() -> None:
    legacy_manager = TokenManager.from_secret("x" * 32)
    legacy_manager = TokenManager(secret=legacy_manager.secret, issuer="native-agent-runner")
    token = legacy_manager.issue(
        kind="run_access",
        audience="native-agent-runner.backend",
        run_id="run_1",
        tenant_id="tenant_a",
        user_id="user_a",
        ttl_s=60,
    )

    current_manager = TokenManager(secret=legacy_manager.secret)
    claims = current_manager.verify(
        token,
        kind="run_access",
        audience=("monoid.backend", "native-agent-runner.backend"),
        run_id="run_1",
    )

    assert claims.audience == "native-agent-runner.backend"
    with pytest.raises(TokenError):
        current_manager.verify(token, kind="run_access", audience="monoid.backend", run_id="run_1")


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


def test_backend_send_message_rejects_oversized_content(tmp_path: Path) -> None:
    # An over-large follow-up message is rejected (the size check precedes the terminal and
    # queue checks), bounding per-message memory.
    workspace = _workspace(tmp_path)
    backend = _backend(tmp_path, workspace, [])
    backend.max_message_bytes = 50
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Summarize.",
            runtime_config=_default_config(),
        )
    )
    backend.wait_for_run(submission.run_id, timeout_s=5)
    with pytest.raises(ValueError, match="exceeds"):
        backend.send_message(submission.run_id, submission.run_token, "x" * 200)


def test_backend_bounds_concurrent_runs(tmp_path: Path) -> None:
    # max_concurrent_runs caps active runs: a second submission while the slot is held stays
    # ``queued`` (it never reaches its adapter) until the first run releases the slot.
    workspace = _workspace(tmp_path)
    started = threading.Event()
    release = threading.Event()
    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=lambda _spec, _token: _BlockingAdapter(started, release),
        max_concurrent_runs=1,
    )

    def _submit(label: str):
        return backend.submit_run(
            BackendRunRequest(
                tenant_id="tenant_a",
                user_id="user_a",
                workspace_root=workspace,
                instruction=label,
                runtime_config=_default_config(),
            )
        )

    first = _submit("A")
    assert started.wait(5)  # A entered its adapter and holds the only slot
    second = _submit("B")
    # B is blocked on the concurrency semaphore before its adapter; it cannot progress.
    assert backend._record(second.run_id).status == "queued"

    release.set()
    assert backend.wait_for_run(first.run_id, timeout_s=10) == "completed"
    assert backend.wait_for_run(second.run_id, timeout_s=10) == "completed"
