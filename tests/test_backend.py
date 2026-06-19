from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from conftest import http_json, runtime_config, tool_binding, wait_http_ready

from native_agent_runner.core._util import write_json_atomic
from native_agent_runner.core.checkpoint import RunCheckpoint
from native_agent_runner.core.tool_surface import ToolScope
from native_agent_runner.errors import PermissionDenied
from native_agent_runner.permissions import PermissionPolicy
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call
from native_agent_runner.reference._shared.tokens import TokenError, TokenManager
from native_agent_runner.reference.backend.http import create_backend_server
from native_agent_runner.reference.backend.service import (
    _RUN_META_SCHEMA_VERSION,
    BackendRunRequest,
    RunnerBackend,
)


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


def _hitl_backend(tmp_path: Path, workspace: Path, adapters: list, turns: list) -> RunnerBackend:
    token_manager = _token_manager()

    def factory(spec, llm_gateway_token):
        del spec, llm_gateway_token
        adapter = FakeModelAdapter(turns=list(turns))
        adapters.append(adapter)
        return adapter

    return RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
    )


def _running_hitl_tasks(backend: RunnerBackend, run_id: str) -> list:
    record = backend._record(run_id)
    loop = record.loop
    if loop is None or loop._session is None:
        return []
    manager = loop._session.res.context.job_manager
    return [task for task in list(manager.jobs.values()) if task.kind == "hitl" and task.status == "running"]


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

    def _wait(predicate, tries: int = 1000) -> bool:
        for _ in range(tries):
            if predicate():
                return True
            time.sleep(0.01)
        return False

    # First turn settles -> session parks awaiting the next user message.
    assert _wait(lambda: backend._record(run_id).status == "awaiting_input")

    backend.send_message(run_id, token, content="again")
    # The follow-up message reaches the model as a second user turn.
    assert _wait(lambda: len([r for a in adapters for r in a.requests if r.instruction]) >= 2)

    backend.cancel_run(run_id, token)  # stop the open session
    status = backend.wait_for_run(run_id, timeout_s=20)
    assert status in {"completed", "limited", "failed"}

    instructions = [r.instruction for a in adapters for r in a.requests if r.instruction]
    assert "hello" in instructions
    assert "again" in instructions


def _recoverable_backend(run_root: Path, token_manager: TokenManager, workspace: Path, adapters: list, turns: list) -> RunnerBackend:
    def factory(spec, llm_gateway_token):
        del spec, llm_gateway_token
        adapter = FakeModelAdapter(turns=list(turns))
        adapters.append(adapter)
        return adapter

    return RunnerBackend(
        run_root=run_root,
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
    )


def test_backend_recovers_parked_hitl_run_from_checkpoint(tmp_path: Path) -> None:
    # A run parked on a hosted task is durably checkpointed; a *fresh backend* (new
    # process, empty _records) over the same run_root resumes it from checkpoint.json.
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    token_manager = _token_manager()

    def _wait(predicate, tries: int = 1000) -> bool:
        for _ in range(tries):
            if predicate():
                return True
            time.sleep(0.01)
        return False

    # Process 1: open the run, park on a human-input request, write the checkpoint.
    crashed: list = []
    backend1 = _recoverable_backend(
        run_root,
        token_manager,
        workspace,
        crashed,
        turns=[ModelTurn(response_id="r1", tool_calls=(fake_tool_call("hitl_request", {"prompt": "Pick"}, "c1"),))],
    )
    submission = backend1.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Name it, ask me.",
            runtime_config=runtime_config("hitl.request"),
        )
    )
    run_id, token = submission.run_id, submission.run_token
    run_dir = run_root / run_id
    assert _wait(lambda: backend1.checkpoint_store.latest(run_id) is not None)
    assert (run_dir / "run.json").exists()

    # Process 2: a brand-new backend recovers the parked run from disk. Its adapter
    # settles the resumed turn (the conversation continues by handle from r1). backend1's
    # worker is defunct (parked, never answered); we leave it and stop it at the end.
    resumed: list = []
    backend2 = _recoverable_backend(
        run_root,
        token_manager,
        workspace,
        resumed,
        turns=[ModelTurn(response_id="r2", final_text="named it")],
    )
    # recover_runs is idempotent; retry it because process 1's worker is still alive
    # in-process (a real crash would have freed run_dir), so reopening its files can
    # transiently race. The high attempt cap keeps a transient miss from marking the run
    # unrecoverable before it succeeds.
    backend2.max_recover_attempts = 10_000
    assert _wait(lambda: run_id in backend2.recover_runs() or run_id in backend2._records)

    # Deliver the human answer to the recovered run -> it resumes and completes.
    def _drain() -> None:
        for _ in range(1000):
            if backend2._record(run_id).status in {"completed", "failed", "limited"}:
                return
            for task in _running_hitl_tasks(backend2, run_id):
                try:
                    backend2.report_task_result(run_id, token, task_id=task.job_id, result={"answer": "Ada"})
                except Exception:
                    pass
            time.sleep(0.01)

    responder = threading.Thread(target=_drain)
    responder.start()
    status = backend2.wait_for_run(run_id, timeout_s=20)
    responder.join(timeout=5)

    assert status == "completed"
    hitl_obs = [
        obs
        for adapter in resumed
        for request in adapter.requests
        for obs in request.observations
        if obs.tool_name == "human_input"
    ]
    assert hitl_obs and hitl_obs[0].output["answer"] == "Ada"
    # The resumed turn continued from the pre-crash handle, not a replayed transcript.
    assert resumed[0].requests[0].previous_turn_handle == "r1"
    backend1.cancel_run(run_id, token)  # cleanup: stop the defunct first-process worker


def test_recover_runs_skips_terminal_and_metaless_checkpoints(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    backend = _recoverable_backend(run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")])

    # A terminal checkpoint is a finished run -> never resumed.
    backend.checkpoint_store.put(RunCheckpoint(run_id="run_terminal", seq=1, terminal=True))

    # A non-terminal checkpoint with no run.json descriptor cannot be rebuilt -> skipped.
    backend.checkpoint_store.put(RunCheckpoint(run_id="run_orphan", seq=1, terminal=False))

    assert backend.recover_runs() == []


def test_backend_worker_failure_writes_failure_bundle(tmp_path: Path) -> None:
    # A worker-level crash (here the model-adapter factory raises before the loop is even
    # built) must still leave a durable failure.json. Without it, a restart's recover_runs
    # would treat the run as merely parked and resume it into a crash loop.
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"

    def factory(spec, llm_gateway_token):
        del spec, llm_gateway_token
        raise RuntimeError("adapter boom")

    backend = RunnerBackend(
        run_root=run_root,
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
            instruction="do it",
            runtime_config=_default_config(),
        )
    )
    run_id = submission.run_id
    assert backend.wait_for_run(run_id, timeout_s=10) == "failed"

    failure_path = run_root / run_id / "failure.json"
    assert failure_path.exists()
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["schema_version"] == "native-agent-runner.failure.v1"
    assert failure["type"] == "RuntimeError"
    assert "last_good_seq" in failure


def test_recover_runs_marks_unrecoverable_after_max_attempts(tmp_path: Path, monkeypatch) -> None:
    # A checkpoint that repeatedly fails to resume is poison: after max_recover_attempts it
    # is marked unrecoverable (durable failure.json) and skipped forever — no crash loop.
    # The orphan state is built directly so the attempt accounting is deterministic.
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    backend = _recoverable_backend(run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")])
    backend.max_recover_attempts = 2

    run_id = "run_poison"
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True)
    backend.checkpoint_store.put(RunCheckpoint(run_id=run_id, seq=1, terminal=False))
    write_json_atomic(run_dir / "run.json", {"schema_version": _RUN_META_SCHEMA_VERSION, "run_id": run_id})

    def _boom(stored, meta):
        del stored, meta
        raise RuntimeError("resume boom")

    monkeypatch.setattr(backend, "_resume_from_checkpoint", _boom)

    assert backend.recover_runs() == []  # attempt 1
    assert not (run_dir / "failure.json").exists()
    assert json.loads((run_dir / "recover_attempts.json").read_text(encoding="utf-8"))["count"] == 1

    assert backend.recover_runs() == []  # attempt 2 -> hits the cap
    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["error_code"] == "unrecoverable"

    # Now permanently skipped: failure.json is the terminal mark.
    assert backend.recover_runs() == []


def _stale_lease_payload(run_id: str) -> dict:
    # A lease whose heartbeat is far in the past -> the owning worker is presumed crashed.
    return {"run_id": run_id, "worker_id": "dead", "pid": 1, "heartbeat_at": time.time() - 1000.0, "lease_ttl_s": 30.0}


def test_watchdog_reclaims_stale_lease_run(tmp_path: Path, monkeypatch) -> None:
    # The watchdog tick selects an orphaned run (stale lease + resumable checkpoint),
    # CAS-claims its lease for this backend, and invokes resume. The on-disk orphan state is
    # built directly (no live in-process worker to race), and the resume — already covered
    # end-to-end by test_backend_recovers_parked_hitl_run_from_checkpoint — is stubbed so the
    # assertion is deterministic under load.
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    backend = _recoverable_backend(run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")])

    run_id = "run_orphan"
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True)
    backend.checkpoint_store.put(RunCheckpoint(run_id=run_id, seq=1, terminal=False))
    write_json_atomic(run_dir / "run.json", {"schema_version": _RUN_META_SCHEMA_VERSION, "run_id": run_id})
    write_json_atomic(run_dir / "lease.json", _stale_lease_payload(run_id))

    resumed: list = []
    monkeypatch.setattr(
        backend,
        "_resume_from_checkpoint",
        lambda stored, meta: resumed.append(stored.checkpoint.run_id),
    )

    assert backend._reclaim_stale_runs() == [run_id]
    assert resumed == [run_id]  # resume was invoked for the orphan
    lease = json.loads((run_dir / "lease.json").read_text(encoding="utf-8"))
    assert lease["worker_id"] == backend._worker_id  # CAS claim flipped ownership before resume


def test_watchdog_skips_run_with_fresh_lease(tmp_path: Path) -> None:
    # A run whose lease is fresh (a live peer owns it) must NOT be reclaimed.
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    run_dir = run_root / "run_live"
    run_dir.mkdir(parents=True)
    write_json_atomic(
        run_dir / "lease.json",
        {"run_id": "run_live", "worker_id": "peer", "pid": 2, "heartbeat_at": time.time(), "lease_ttl_s": 30.0},
    )
    backend = _recoverable_backend(run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")])
    assert backend._reclaim_stale_runs() == []
    assert "run_live" not in backend._records

    # start/stop lifecycle is a clean no-op smoke (no orphans to reclaim).
    backend.watchdog_interval_s = 0.01
    backend.start_watchdog()
    backend.stop_watchdog()


def test_watchdog_concurrent_claim_has_single_winner(tmp_path: Path) -> None:
    # Two backends racing to reclaim the same stale-lease run must produce exactly one
    # winner (lease CAS under a cross-process lock).
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    run_dir = run_root / "run_x"
    run_dir.mkdir(parents=True)
    write_json_atomic(run_dir / "lease.json", _stale_lease_payload("run_x"))

    b1 = _recoverable_backend(run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")])
    b2 = _recoverable_backend(run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")])
    results: list = []
    barrier = threading.Barrier(2)

    def claim(backend) -> None:
        barrier.wait()
        results.append(backend._claim_lease(run_dir, "run_x"))

    threads = [threading.Thread(target=claim, args=(b,)) for b in (b1, b2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count(True) == 1


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


class _BlockingAdapter:
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self._started = started
        self._release = release

    def next_turn(self, _request):
        self._started.set()
        self._release.wait(timeout=10)
        return ModelTurn(response_id="r1", final_text="done", usage={"total_tokens": 1})


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


def test_backend_http_rejects_oversized_request(tmp_path: Path) -> None:
    # A request whose declared Content-Length exceeds the limit is rejected with 413 before
    # any body bytes are read (DoS / OOM guard). The body is tiny; only the header is spoofed.
    workspace = _workspace(tmp_path)
    backend = _backend(tmp_path, workspace, [])
    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        _wait_http_ready(base_url)
        request = Request(
            f"{base_url}/v1/runs",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer admin",
                "Content-Length": "20000000",
            },
            method="POST",
        )
        with pytest.raises(HTTPError) as exc_info:
            urlopen(request, timeout=5)
        assert exc_info.value.code == 413
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


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
    started = threading.Event()
    release = threading.Event()

    class SlowAdapter:
        # Signals when the turn is in-flight and blocks until released, so the cancel below
        # is guaranteed to land mid-run (no reliance on a fixed sleep racing the HTTP RTT).
        def next_turn(self, _request):
            started.set()
            release.wait(timeout=10)
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
        assert started.wait(5)  # the run is now actively in its turn
        cancelled = _json_request(f"{base_url}/v1/runs/{run_id}/cancel", {}, token=run_token)
        assert cancelled["cancel_requested"] is True
        release.set()  # let the turn return; the loop then observes the cancel
        assert backend.wait_for_run(run_id, timeout_s=10) == "limited"
        status = _json_get(f"{base_url}/v1/runs/{run_id}/status", token=run_token)
        assert status["error_code"] == "cancelled"
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _start_server(backend: RunnerBackend):
    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    _wait_http_ready(base_url)
    return server, thread, base_url


def _poll(predicate, *, tries: int = 400) -> bool:
    for _ in range(tries):
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_backend_http_multi_turn_messages_and_task_endpoints(tmp_path: Path) -> None:
    # One server/worker exercising the full multi-turn HTTP surface: follow-up
    # messages, task creation with a scoped callback token, and result delivery.
    # (Detailed worker/injection behavior is covered by the in-process tests above.)
    workspace = _workspace(tmp_path)
    adapters: list = []
    backend = _hitl_backend(tmp_path, workspace, adapters, turns=[ModelTurn(response_id="r1", final_text="first")])
    backend.idle_timeout_s = 15.0
    server, thread, base_url = _start_server(backend)
    try:
        created = _json_request(
            f"{base_url}/v1/runs",
            {
                "tenant_id": "tenant_a",
                "user_id": "user_a",
                "workspace_root": str(workspace),
                "instruction": "hello",
                "runtime_config": _default_config().to_json(),
                "multi_turn": True,
            },
            token="admin",
        )
        run_id, run_token = created["run_id"], created["run_token"]

        # First turn settles -> the session parks awaiting the next user message.
        assert _poll(lambda: backend._record(run_id).status == "awaiting_input")

        # A follow-up message is threaded as a second user turn.
        queued = _json_request(f"{base_url}/v1/runs/{run_id}/messages", {"content": "again"}, token=run_token)
        assert queued["status"] == "queued"
        assert _poll(lambda: len([r for a in adapters for r in a.requests if r.instruction]) >= 2)
        instructions = [r.instruction for a in adapters for r in a.requests if r.instruction]
        assert "hello" in instructions and "again" in instructions

        # Create an automation task -> scoped callback token + URL.
        assert _poll(lambda: backend._record(run_id).status == "awaiting_input")
        task = _json_request(
            f"{base_url}/v1/runs/{run_id}/tasks",
            {"kind": "automation", "request": {"description": "call external system"}},
            token=run_token,
        )
        task_id = task["task_id"]
        callback_token = task["callback_token"]
        assert task["callback_url"] == f"/v1/runs/{run_id}/tasks/{task_id}/result"

        # A bogus token is rejected; the scoped callback token completes the task.
        with pytest.raises(HTTPError) as exc_info:
            _json_request(
                f"{base_url}/v1/runs/{run_id}/tasks/{task_id}/result",
                {"result": {"answer": "x"}},
                token="not-a-real-token",
            )
        assert exc_info.value.code == 401

        done = _json_request(
            f"{base_url}{task['callback_url']}",
            {"result": {"answer": "external done"}},
            token=callback_token,
        )
        assert done.get("delivered") is True

        _json_request(f"{base_url}/v1/runs/{run_id}/cancel", {}, token=run_token)
        assert backend.wait_for_run(run_id, timeout_s=20) in {"completed", "limited", "failed"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _json_request(url: str, payload: dict, *, token: str | None = None) -> dict:
    return http_json(url, payload, token=token)


def _json_get(url: str, *, token: str) -> dict:
    return http_json(url, token=token, method="GET")


def _wait_http_ready(base_url: str, *, timeout_s: float = 15.0) -> None:
    wait_http_ready(base_url, timeout_s=timeout_s)
