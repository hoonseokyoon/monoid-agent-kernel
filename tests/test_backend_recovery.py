from __future__ import annotations

from support.backend_harness import (
    BackendRunRequest,
    FakeModelAdapter,
    ModelTurn,
    Path,
    PermissionDenied,
    PermissionPolicy,
    RunCheckpoint,
    RunnerBackend,
    SqliteCheckpointStore,
    SqliteLeaseStore,
    _RUN_META_SCHEMA_VERSION,
    _ScriptedTurnAdapter,
    _default_config,
    _recoverable_backend,
    _running_hitl_tasks,
    _scripted_backend,
    _stale_lease_payload,
    _submit_multi_turn,
    _token_manager,
    _workspace,
    eventually,
    wait_for_durable_status,
    fake_tool_call,
    json,
    pytest,
    runtime_config,
    threading,
    time,
    tool_binding,
    write_json_atomic,
)
import dataclasses

from monoid_agent_kernel.errors import NativeAgentError
from monoid_agent_kernel.reference.backend.recovery import ResumeOutcome
from monoid_agent_kernel.reference.backend.service import _read_run_meta
from monoid_agent_kernel.core._event_log import inspect_event_log_tail
from monoid_agent_kernel.core.lifecycle import SessionState
from monoid_agent_kernel.core.projections import project_run_status

pytestmark = pytest.mark.integration


def _recovery_metadata(run_id: str, workspace: Path) -> dict:
    config = _default_config()
    return {
        "schema_version": _RUN_META_SCHEMA_VERSION,
        "run_id": run_id,
        "tenant_id": "tenant_a",
        "user_id": "user_a",
        "workspace_root": str(workspace),
        "runtime_config": config.to_json(),
        "runtime_config_hash": config.config_hash,
    }


def test_read_run_meta_accepts_legacy_schema_version(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    write_json_atomic(
        run_dir / "run.json",
        {"schema_version": "native-agent-runner.backend-run.v1", "run_id": "run_1"},
    )

    assert _read_run_meta(run_dir) == {
        "schema_version": "native-agent-runner.backend-run.v1",
        "run_id": "run_1",
    }


def test_recovery_request_reads_pre_v020_permission_policy_literal_bangs(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    meta = {
        **_recovery_metadata("run_legacy", workspace),
        "permission_policy": {
            "deny_patterns": ["!odd"],
            "redact_patterns": ["./!private"],
        },
    }

    request = RunnerBackend._recovery_request(  # type: ignore[arg-type]
        None, meta, _default_config()
    )

    assert request.permission_policy.deny_patterns == ("!odd",)
    assert request.permission_policy.redact_patterns == ("./!private",)
    assert request.permission_policy.to_json() == {
        "deny_patterns": ["!odd"],
        "redact_patterns": ["./!private"],
        "path_pattern_encoding": "monoid.literal-bang.v1",
    }


def test_backend_accepts_checkpoint_store_without_metadata_methods(tmp_path: Path) -> None:
    class LegacyCheckpointStore:
        def put(self, checkpoint: RunCheckpoint, blobs: dict[str, bytes] | None = None) -> None:
            del checkpoint, blobs

        def latest(self, run_id: str):
            del run_id
            return None

        def delete(self, run_id: str) -> None:
            del run_id

        def put_blob(self, run_id: str, data: bytes) -> str:
            del run_id, data
            return "0" * 64

        def get_blob(self, run_id: str, sha256: str) -> bytes:
            del run_id
            raise KeyError(sha256)

    workspace = _workspace(tmp_path)

    def factory(spec, llm_gateway_token):
        del spec, llm_gateway_token
        return FakeModelAdapter([ModelTurn(response_id="r1", final_text="done")])

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
        checkpoint_store=LegacyCheckpointStore(),  # type: ignore[arg-type]
    )

    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="run",
            runtime_config=_default_config(),
        )
    )

    assert backend.wait_for_run(submission.run_id, timeout_s=10) == "completed"


@pytest.mark.slow
def test_backend_recovers_parked_hitl_run_from_checkpoint(tmp_path: Path) -> None:
    # A run parked on a hosted task is durably checkpointed; a *fresh backend* (new
    # process, empty _records) over the same run_root resumes it from checkpoint.json.
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    token_manager = _token_manager()

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
    assert eventually(lambda: backend1.checkpoint_store.latest(run_id) is not None)
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
    assert eventually(lambda: run_id in backend2.recover_runs() or run_id in backend2._records)

    # Deliver the human answer to the recovered run -> it resumes and completes.
    def _drain() -> None:
        for _ in range(1000):
            if backend2._record(run_id).terminal:
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

    assert status is SessionState.COMPLETED
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
    backend1.shutdown(drain=True)  # cleanup: stop the defunct first-process worker


def test_resume_run_single_run_then_continue_after_restart(tmp_path: Path) -> None:
    # The token-scoped, single-run analog of recover_runs: a parked multi-turn session is resumed
    # by run id from a *fresh backend*, then a follow-up send_message threads a new user turn.
    # This is the studio "continue an old chat after a restart" path.
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    token_manager = _token_manager()

    # Process 1: open a multi-turn session; the first turn settles and parks awaiting input.
    crashed: list = []
    backend1 = _recoverable_backend(
        run_root, token_manager, workspace, crashed,
        turns=[ModelTurn(response_id="r1", final_text="first")],
    )
    backend1.idle_timeout_s = 30.0
    submission = backend1.submit_run(
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
    assert eventually(lambda: backend1.checkpoint_store.latest(run_id) is not None)
    assert eventually(lambda: backend1._record(run_id).state is SessionState.AWAITING_INPUT)
    wait_for_durable_status(run_root, run_id, where=lambda s: s["state"] == "awaiting_input")

    # Process 2: a fresh backend (empty _records). send_message would KeyError; resume_run
    # materializes the record from the checkpoint, then the follow-up threads a second turn.
    resumed: list = []
    backend2 = _recoverable_backend(
        run_root, token_manager, workspace, resumed,
        turns=[ModelTurn(response_id="r2", final_text="second")],
    )
    backend2.idle_timeout_s = 30.0
    backend2.max_recover_attempts = 10_000

    with pytest.raises(KeyError):
        backend2.send_message(run_id, token, "before resume")

    info = backend2.resume_run(run_id, token)
    assert info["resumed"] is True
    assert run_id in backend2._records
    # Idempotent: a second resume on the now-live run is a no-op.
    assert backend2.resume_run(run_id, token)["resumed"] is False

    assert backend2.send_message(run_id, token, "again")["status"] == "queued"
    assert eventually(lambda: len([r for a in resumed for r in a.requests if r.instruction]) >= 1)

    backend2.cancel_run(run_id, token)
    assert backend2.wait_for_run(run_id, timeout_s=20) in {
        SessionState.COMPLETED,
        SessionState.LIMITED,
        SessionState.FAILED,
        SessionState.CANCELLED,
    }
    instructions = [r.instruction for a in resumed for r in a.requests if r.instruction]
    assert "again" in instructions
    backend1.cancel_run(run_id, token)  # stop the defunct first-process worker


def test_resume_run_restores_paused_boundary_after_restart(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    token_manager = _token_manager()
    reached_model = threading.Event()
    release_model = threading.Event()

    class _PauseBoundaryAdapter:
        def __init__(self) -> None:
            self.requests: list = []

        def next_turn(self, request):  # noqa: ANN001
            self.requests.append(request)
            reached_model.set()
            release_model.wait(timeout=10)
            return ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_read", {"path": "notes.md"}, "read1"),),
            )

    paused_adapter = _PauseBoundaryAdapter()
    backend1 = RunnerBackend(
        run_root=run_root,
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=lambda _spec, _token: paused_adapter,
    )
    backend1.idle_timeout_s = 30.0
    submission = backend1.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="pause before the second model step",
            runtime_config=_default_config(),
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert reached_model.wait(10.0)
    assert backend1.pause_run(run_id, token)["pause_requested"] is True
    release_model.set()
    assert eventually(lambda: backend1._record(run_id).state is SessionState.PAUSED)
    wait_for_durable_status(run_root, run_id, where=lambda s: s["state"] == "paused")
    stored = backend1.checkpoint_store.latest(run_id)
    assert stored is not None
    assert stored.checkpoint.last_suspension is not None
    assert stored.checkpoint.last_suspension["reason"] == "paused"

    resumed: list = []
    backend2 = _recoverable_backend(
        run_root,
        token_manager,
        workspace,
        resumed,
        turns=[ModelTurn(response_id="r2", final_text="resumed after restart")],
    )
    backend2.idle_timeout_s = 30.0
    backend2.max_recover_attempts = 10_000

    assert backend2.resume_run(run_id, token)["resumed"] is True
    assert eventually(lambda: backend2._record(run_id).state is SessionState.PAUSED)
    assert backend2.signal_resume(run_id, token)["resumed"] is True
    assert eventually(lambda: bool(resumed and resumed[0].requests))
    assert eventually(lambda: backend2._record(run_id).state is SessionState.AWAITING_INPUT)
    assert resumed[0].requests[0].previous_turn_handle == "r1"

    backend2.cancel_run(run_id, token)
    backend2.wait_for_run(run_id, timeout_s=20)
    backend1.cancel_run(run_id, token)
    backend1.wait_for_run(run_id, timeout_s=20)


def test_resume_run_uses_latest_runtime_config_after_hotswap(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    token_manager = _token_manager()

    backend1 = _recoverable_backend(
        run_root,
        token_manager,
        workspace,
        [],
        turns=[ModelTurn(response_id="r1", final_text="first")],
    )
    backend1.idle_timeout_s = 30.0
    initial = runtime_config(
        bindings=(tool_binding("fs.read", guidance="initial read"), tool_binding("run.finish")),
    )
    submission = backend1.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hello",
            runtime_config=initial,
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert eventually(lambda: backend1.checkpoint_store.latest(run_id) is not None)
    assert eventually(lambda: backend1._record(run_id).state is SessionState.AWAITING_INPUT)
    wait_for_durable_status(run_root, run_id, where=lambda s: s["state"] == "awaiting_input")

    replacement = runtime_config(
        version=2,
        bindings=(tool_binding("fs.read", guidance="replacement read"), tool_binding("run.finish")),
    )
    updated = backend1.replace_runtime_config(
        run_id,
        token,
        expected_version=1,
        issuer="operator",
        reason="replace before restart",
        config=replacement,
    )
    assert updated["config_hash"] == replacement.config_hash

    resumed: list = []
    backend2 = _recoverable_backend(
        run_root,
        token_manager,
        workspace,
        resumed,
        turns=[ModelTurn(response_id="r2", final_text="second")],
    )
    backend2.idle_timeout_s = 30.0
    backend2.max_recover_attempts = 10_000

    assert backend2.resume_run(run_id, token)["resumed"] is True
    assert backend2.runtime_config(run_id, token)["config_hash"] == replacement.config_hash
    backend2.send_message(run_id, token, "again")
    assert eventually(lambda: any(adapter.requests for adapter in resumed))

    read_tool = next(tool for tool in resumed[0].requests[0].tools if tool.id == "fs.read")
    assert "replacement read" in read_tool.description

    backend2.cancel_run(run_id, token)
    backend2.wait_for_run(run_id, timeout_s=20)
    backend1.cancel_run(run_id, token)


def test_recover_runs_skips_terminal_and_metaless_checkpoints(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    backend = _recoverable_backend(run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")])

    # A terminal checkpoint is a finished run -> never resumed.
    backend.checkpoint_store.put(RunCheckpoint(run_id="run_terminal", seq=1, terminal=True))

    # A non-terminal checkpoint with no run.json descriptor cannot be rebuilt -> skipped.
    backend.checkpoint_store.put(RunCheckpoint(run_id="run_orphan", seq=1, terminal=False))

    assert backend.recover_runs() == []


def _closed_limited_run(run_root: Path, workspace: Path, adapters: list) -> tuple:
    """Drive one real backend run to a closed LIMITED terminal (max_steps=1), then stop the
    backend — the run dir this leaves behind is the exact shape finding 1 resurrects: a
    non-terminal park checkpoint, no failure.json, and a terminal status artifact."""
    backend = _recoverable_backend(
        run_root,
        _token_manager(),
        workspace,
        adapters,
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_list", {"path": "."}, "c1"),),
                usage={"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
            ),
            ModelTurn(final_text="done"),
        ],
    )
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="spend the budget",
            runtime_config=runtime_config("fs.list"),
            max_steps=1,
        )
    )
    run_id = submission.run_id
    assert backend.wait_for_run(run_id, timeout_s=20) is SessionState.LIMITED
    # ``wait_for_run`` answers off the record, which the drive marks LIMITED at the park
    # promotion — BEFORE ``aclose()`` appends ``run.finished`` to events.jsonl. Reading the
    # event log immediately after it lost that race once on a coverage-slowed CI box, so wait
    # for the close's durable statement too before handing the dir to the caller.
    deadline = time.monotonic() + 20
    while _finished_count(run_root, run_id) < 1:
        assert time.monotonic() < deadline, "run.finished never reached events.jsonl"
        time.sleep(0.05)
    stored = backend.checkpoint_store.latest(run_id)
    assert stored is not None and stored.checkpoint.terminal is False  # the resurrection bait
    assert not (run_root / run_id / "failure.json").exists()
    return backend, run_id


def _finished_count(run_root: Path, run_id: str) -> int:
    events = (run_root / run_id / "events.jsonl").read_text(encoding="utf-8")
    return sum(1 for line in events.splitlines() if '"run.finished"' in line)


def test_recover_runs_does_not_resurrect_a_closed_limited_run(tmp_path: Path) -> None:
    """A run that closed LIMITED satisfied none of recovery's filters (non-terminal park
    checkpoint, no failure.json, checkpoints kept), so EVERY recovery pass re-drove it:
    another terminal run.finished per restart, and the full cumulative usage re-metered
    into each fresh tenant ledger, forever. The durable status artifact is the terminal
    marker recovery must consult."""
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    backend1, run_id = _closed_limited_run(run_root, workspace, [])
    assert _finished_count(run_root, run_id) == 1
    backend1.shutdown()

    for _ in range(2):  # every pass, not just the first
        backend2 = _recoverable_backend(
            run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")]
        )
        assert backend2.recover_runs() == []
        assert run_id not in backend2._records
        assert backend2.tenant_usage("tenant_a")["runs"] == 0
        backend2.shutdown()
    assert _finished_count(run_root, run_id) == 1
    # The skip is a recognition of a closed run, not a quarantine.
    assert not (run_root / run_id / "failure.json").exists()


def test_recovery_keys_off_the_durable_status_artifact(tmp_path: Path) -> None:
    """Both directions of the guard, on one genuinely resumable run dir.

    A pre-v0.21 closed-limited dir carries a bare legacy ``status: "limited"`` (no ``state``,
    no ``terminal``) beside its non-terminal park checkpoint — it must not resurrect either.
    The same dir with a NON-terminal status artifact is a run that crashed at the limited park
    before close, and that one must still recover."""
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    backend1, run_id = _closed_limited_run(run_root, workspace, [])
    backend1.shutdown()
    status_path = run_root / run_id / "status.json"

    # Legacy closed dir: bare status="limited" resolves terminal-limited -> never resumed.
    write_json_atomic(status_path, {"run_id": run_id, "status": "limited"})
    backend2 = _recoverable_backend(
        run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")]
    )
    backend2.max_recover_attempts = 10_000
    assert backend2.recover_runs() == []
    assert not (run_root / run_id / "failure.json").exists()

    # Crashed at the park before close: a non-terminal artifact must NOT block recovery.
    write_json_atomic(status_path, {"run_id": run_id, "state": "running", "terminal": False})
    assert eventually(lambda: run_id in backend2.recover_runs() or run_id in backend2._records)
    assert backend2.wait_for_run(run_id, timeout_s=20) is SessionState.LIMITED


def test_recover_runs_records_unsupported_checkpoint_instead_of_skipping(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    backend = _recoverable_backend(
        run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")]
    )
    run_id = "run_future_checkpoint"
    backend.checkpoint_store.put(RunCheckpoint(run_id=run_id, seq=7, terminal=False))
    manifest = run_root / run_id / "checkpoints" / "7" / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = "monoid.checkpoint.v99"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert backend.recover_runs() == []

    failure = json.loads((run_root / run_id / "failure.json").read_text(encoding="utf-8"))
    assert failure["error_code"] == "checkpoint_unsupported_version"
    assert "checkpoint seq 7" in failure["error"]


def test_recover_runs_rejects_string_terminal_as_corrupt_checkpoint(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    backend = _recoverable_backend(
        run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")]
    )
    run_id = "run_string_terminal"
    backend.checkpoint_store.put(RunCheckpoint(run_id=run_id, seq=1, terminal=False))
    manifest = run_root / run_id / "checkpoints" / "1" / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["terminal"] = "no"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert backend.recover_runs() == []

    failure = json.loads((run_root / run_id / "failure.json").read_text(encoding="utf-8"))
    assert failure["error_code"] == "checkpoint_corrupt"


def test_recover_runs_records_corrupt_metadata_instead_of_treating_it_as_missing(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    backend = _recoverable_backend(
        run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")]
    )
    run_id = "run_corrupt_metadata"
    backend.checkpoint_store.put(RunCheckpoint(run_id=run_id, seq=1, terminal=False))
    run_dir = run_root / run_id
    (run_dir / "run.json").write_text("{", encoding="utf-8")

    assert backend.recover_runs() == []

    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["error_code"] == "backend_run_corrupt"


def test_transient_checkpoint_read_failure_does_not_quarantine_run(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    backend = _recoverable_backend(
        tmp_path / "runs", _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")]
    )
    run_id = "run_checkpoint_store_unavailable"
    run_dir = tmp_path / "runs" / run_id
    calls = 0

    def unavailable(store, requested_run_id):
        nonlocal calls
        del store
        assert requested_run_id == run_id
        calls += 1
        raise OSError("checkpoint store unavailable")

    monkeypatch.setattr(
        "monoid_agent_kernel.reference.backend.recovery.load_latest_checked", unavailable
    )

    # Pin moved with the round-3 typed-refusal change: a transient deferral is a genuine
    # non-resume (FAILED), never a close and never a quarantine.
    assert backend._recovery.attempt_resume(run_dir, run_id) is ResumeOutcome.FAILED
    assert backend._recovery.attempt_resume(run_dir, run_id) is ResumeOutcome.FAILED
    assert calls == 2
    assert not (run_dir / "failure.json").exists()


def test_transient_metadata_read_failure_does_not_quarantine_run(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    backend = _recoverable_backend(
        run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")]
    )
    run_id = "run_metadata_store_unavailable"
    run_dir = run_root / run_id
    backend.checkpoint_store.put(RunCheckpoint(run_id=run_id, seq=1, terminal=False))
    calls = 0

    def unavailable(requested_run_dir, requested_run_id):
        nonlocal calls
        assert requested_run_dir == run_dir
        assert requested_run_id == run_id
        calls += 1
        raise OSError("metadata store unavailable")

    monkeypatch.setattr(backend._recovery, "read_recovery_meta_checked", unavailable)

    # Same typed pin as the checkpoint twin above.
    assert backend._recovery.attempt_resume(run_dir, run_id) is ResumeOutcome.FAILED
    assert backend._recovery.attempt_resume(run_dir, run_id) is ResumeOutcome.FAILED
    assert calls == 2
    assert not (run_dir / "failure.json").exists()


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
    assert backend.wait_for_run(run_id, timeout_s=10) is SessionState.FAILED

    failure_path = run_root / run_id / "failure.json"
    assert failure_path.exists()
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["schema_version"] == "monoid.failure.v1"
    assert failure["type"] == "RuntimeError"
    assert "last_good_seq" in failure
    diagnostics = backend.diagnostics(run_id, submission.run_token)
    assert diagnostics["status"]["state"] == "failed"
    assert diagnostics["status"]["terminal"] is True
    assert diagnostics["failure"]["type"] == "RuntimeError"
    assert diagnostics["recovery"]["failure_marked"] is True
    assert diagnostics["events"]["items"] == []
    with pytest.raises(PermissionDenied):
        backend.diagnostics(run_id, "bad-token")


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
    write_json_atomic(run_dir / "run.json", _recovery_metadata(run_id, workspace))

    def _boom(stored, meta):
        del stored, meta
        raise RuntimeError("resume boom")

    monkeypatch.setattr(backend._recovery, "resume_from_checkpoint", _boom)

    assert backend.recover_runs() == []  # attempt 1
    assert not (run_dir / "failure.json").exists()
    assert json.loads((run_dir / "recover_attempts.json").read_text(encoding="utf-8"))["count"] == 1

    assert backend.recover_runs() == []  # attempt 2 -> hits the cap
    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["error_code"] == "unrecoverable"

    # Now permanently skipped: failure.json is the terminal mark.
    assert backend.recover_runs() == []


def test_a_given_up_run_reads_terminal_failed_on_every_status_surface(
    tmp_path: Path, monkeypatch
) -> None:
    """The give-up path wrote failure.json + metered, but no terminal status artifact — so
    ``status()``, ``list_runs`` and the offline projection all answered a healthy
    ``state=awaiting_input, terminal=False, error=""`` for a permanently dead run, while
    ``resume_run`` simultaneously refused it as "marked unrecoverable". Both give-up sites
    now write the terminal artifact beside the bundle, and the offline projection honors it
    over the stale (necessarily park-ending) event log."""

    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    backend1 = _recoverable_backend(
        run_root,
        _token_manager(),
        workspace,
        [],
        turns=[ModelTurn(response_id="r1", final_text="answer")],
    )
    backend1.idle_timeout_s = 300.0
    submission = _submit_multi_turn(backend1, workspace)
    run_id, token = submission.run_id, submission.run_token
    assert eventually(lambda: backend1._record(run_id).state is SessionState.AWAITING_INPUT)
    wait_for_durable_status(run_root, run_id, where=lambda s: s["state"] == "awaiting_input")
    backend1.stop_watchdog()  # "crash": the park checkpoint + park-shaped status.json remain

    backend2 = _recoverable_backend(
        run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")]
    )
    backend2.max_recover_attempts = 1

    def _boom(stored, meta):
        del stored, meta
        raise RuntimeError("boom")

    monkeypatch.setattr(backend2._recovery, "resume_from_checkpoint", _boom)
    assert backend2.recover_runs() == []
    run_dir = run_root / run_id
    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["error_code"] == "unrecoverable"
    # The corrected hint names the actual operator flow: the quarantine must be lifted
    # first, because recover_runs/resume_run refuse a dir carrying failure.json.
    assert "delete failure.json" in failure["restore_hint"]

    # A fresh operator process, all three readers, one answer.
    backend3 = _recoverable_backend(
        run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")]
    )
    status = backend3.status(run_id, token)
    assert (status["state"], status["terminal"]) == ("failed", True)
    assert status["error_code"] == "unrecoverable"
    assert "recovery failed" in status["error"]
    row = next(
        entry
        for entry in backend3.list_runs("tenant_a")["runs"]
        if entry["run_id"] == run_id
    )
    assert (row["state"], row["terminal"], row["recoverable"]) == ("failed", True, False)
    projection = project_run_status(run_dir)
    assert (projection["state"], projection["terminal"]) == ("failed", True)
    assert projection["error_code"] == "unrecoverable"
    # ...and resume_run still refuses, now consistently with what status() says.
    with pytest.raises(ValueError, match="unrecoverable"):
        backend3.resume_run(run_id, token)


def test_corrupt_state_giveup_also_writes_the_terminal_status_artifact(tmp_path: Path) -> None:
    """The second give-up site (``_record_checked_load_failure``) binds the same rule."""

    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    backend = _recoverable_backend(
        run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")]
    )
    run_id = "run_string_terminal_artifact"
    backend.checkpoint_store.put(RunCheckpoint(run_id=run_id, seq=1, terminal=False))
    manifest = run_root / run_id / "checkpoints" / "1" / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["terminal"] = "no"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert backend.recover_runs() == []

    artifact = json.loads((run_root / run_id / "status.json").read_text(encoding="utf-8"))
    assert (artifact["state"], artifact["terminal"]) == ("failed", True)
    assert artifact["error_code"] == "checkpoint_corrupt"
    assert artifact["given_up_by_recovery"] is True
    # The terminal statement mirrors the sink's run.failed vocabulary: the four facts are
    # assigned (honest empties — no provider verdict exists) and provider_retried is absent.
    assert artifact["retryable"] is False
    assert artifact["config_recoverable"] is False
    assert artifact["http_status"] is None
    assert artifact["provider_error_code"] == ""
    assert "provider_retried" not in artifact


def test_the_restore_hint_flow_actually_recovers_the_run(tmp_path: Path, monkeypatch) -> None:
    """The hint's prescribed flow — delete failure.json, then recover_runs — must work with
    the terminal give-up artifact in place: ``_closed_by_status_artifact`` reads the
    ``given_up_by_recovery`` marker and does not mistake the give-up statement for a close."""

    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    backend = _recoverable_backend(
        run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")]
    )
    backend.max_recover_attempts = 1
    run_id = "run_quarantine_lifted"
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True)
    backend.checkpoint_store.put(RunCheckpoint(run_id=run_id, seq=1, terminal=False))
    write_json_atomic(run_dir / "run.json", _recovery_metadata(run_id, workspace))

    original = type(backend._recovery).resume_from_checkpoint

    def _boom(stored, meta):
        del stored, meta
        raise RuntimeError("resume boom")

    monkeypatch.setattr(backend._recovery, "resume_from_checkpoint", _boom)
    assert backend.recover_runs() == []
    assert (run_dir / "failure.json").exists()
    assert json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["terminal"] is True

    # The operator flow the hint prescribes.
    (run_dir / "failure.json").unlink()
    monkeypatch.setattr(
        backend._recovery, "resume_from_checkpoint", original.__get__(backend._recovery)
    )
    assert backend.recover_runs() == [run_id]
    assert backend.wait_for_run(run_id, timeout_s=20) in {
        SessionState.COMPLETED,
        SessionState.AWAITING_INPUT,
    }


def test_a_recovered_run_whose_drive_fails_reads_dead_after_restart(tmp_path: Path) -> None:
    """The THIRD failure.json writer (``record_run_failure``, reached here through
    ``run_recovered``'s except) binds the same rule as the two give-up sites: the terminal
    statement reaches ``status.json``, so a fresh process's ``status()``/``list_runs`` answer
    dead — not the old park — while ``recover_runs`` skips the quarantined dir."""

    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    backend1 = _recoverable_backend(
        run_root,
        _token_manager(),
        workspace,
        [],
        turns=[ModelTurn(response_id="r1", final_text="answer")],
    )
    backend1.idle_timeout_s = 300.0
    submission = _submit_multi_turn(backend1, workspace)
    run_id, token = submission.run_id, submission.run_token
    assert eventually(lambda: backend1._record(run_id).state is SessionState.AWAITING_INPUT)
    wait_for_durable_status(run_root, run_id, where=lambda s: s["state"] == "awaiting_input")
    backend1.stop_watchdog()  # "crash": the park checkpoint + park-shaped status.json remain

    backend2 = _recoverable_backend(
        run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")]
    )

    async def _drive_boom(record, request, loop, suspension, *, started, turns):  # noqa: ANN001
        del record, request, loop, suspension, started, turns
        raise RuntimeError("drive died after restore")

    backend2._recovery._context = dataclasses.replace(
        backend2._recovery._context, drive_open_session=_drive_boom
    )
    assert backend2.recover_runs() == [run_id]
    run_dir = run_root / run_id
    assert eventually(lambda: (run_dir / "failure.json").exists())
    assert eventually(
        lambda: json.loads((run_dir / "status.json").read_text(encoding="utf-8")).get("terminal")
        is True
    ), "the failure never reached the durable status artifact"
    artifact = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert artifact["state"] == "failed"
    assert artifact["error_code"] == "internal_error"  # the failure's own code
    assert artifact["recorded_by_run_failure"] is True

    # A fresh operator process, all readers, one answer.
    backend3 = _recoverable_backend(
        run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")]
    )
    status = backend3.status(run_id, token)
    assert (status["state"], status["terminal"]) == ("failed", True)
    row = next(
        entry for entry in backend3.list_runs("tenant_a")["runs"] if entry["run_id"] == run_id
    )
    assert (row["state"], row["terminal"], row["recoverable"]) == ("failed", True, False)
    assert backend3.recover_runs() == []


def test_a_parked_artifact_beside_a_failure_bundle_reads_dead(tmp_path: Path) -> None:
    """The reader-side backstop for pre-fix dirs: a failure bundle outranks a NON-terminal
    parked artifact (the run is dead), while lifting the quarantine — the restore_hint's
    prescribed flow — makes the park readable again."""

    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    backend = _recoverable_backend(
        run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")]
    )
    run_id = "run_prefix_quarantine"
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True)
    write_json_atomic(run_dir / "run.json", _recovery_metadata(run_id, workspace))
    write_json_atomic(
        run_dir / "status.json",
        {
            "run_id": run_id,
            "state": "awaiting_input",
            "terminal": False,
            "last_event_seq": 3,
            "last_event_type": "run.awaiting_input",
            "updated_at": "2026-08-03T00:00:00Z",
        },
    )
    write_json_atomic(
        run_dir / "failure.json",
        {
            "schema_version": "monoid.failure.v1",
            "run_id": run_id,
            "error": "boom",
            "error_code": "internal_error",
            "type": "RuntimeError",
            "last_good_seq": 0,
        },
    )
    token = backend.token_manager.issue(
        kind="run_access",
        audience="monoid.backend",
        run_id=run_id,
        tenant_id="tenant_a",
        user_id="user_a",
        ttl_s=600,
    )

    status = backend.status(run_id, token)
    assert (status["state"], status["terminal"]) == ("failed", True)
    row = next(
        entry for entry in backend.list_runs("tenant_a")["runs"] if entry["run_id"] == run_id
    )
    assert (row["state"], row["terminal"]) == ("failed", True)

    # Delete failure.json -> the park is readable again, so the hinted resume can proceed.
    (run_dir / "failure.json").unlink()
    status = backend.status(run_id, token)
    assert (status["state"], status["terminal"]) == ("awaiting_input", False)


def test_the_giveup_artifact_is_schema_valid_with_no_prior_status_json(tmp_path: Path) -> None:
    """A quarantine over a run that never wrote status.json (bootstrap-shaped dirs) must still
    mint a STATUS_SCHEMA-valid artifact: the required watermark keys are seeded (``0`` /
    ``""`` = "no committed event known to this writer"), so ``monoid validate`` accepts the
    very file the fix writes."""

    from monoid_agent_kernel.core.schemas import STATUS_SCHEMA, _validate_json_file

    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    backend = _recoverable_backend(
        run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")]
    )
    run_id = "run_schema_valid_giveup"
    backend.checkpoint_store.put(RunCheckpoint(run_id=run_id, seq=1, terminal=False))
    manifest = run_root / run_id / "checkpoints" / "1" / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["terminal"] = "no"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert backend.recover_runs() == []

    status_path = run_root / run_id / "status.json"
    artifact = json.loads(status_path.read_text(encoding="utf-8"))
    assert (artifact["state"], artifact["terminal"]) == ("failed", True)
    assert artifact["last_event_seq"] == 0
    assert artifact["last_event_type"] == ""
    issues: list = []
    _validate_json_file(status_path, STATUS_SCHEMA, issues)
    assert issues == [], issues


def test_a_closed_limited_run_is_not_advertised_recoverable(tmp_path: Path) -> None:
    """A run that CLOSED limited (terminal status artifact, non-terminal park checkpoint) was
    advertised ``recoverable: true`` beside ``terminal: true`` in ``list_runs``, and
    ``resume_run`` then 400'd pointing at a failure.json that does not exist. The projection
    now consults the same close-recording artifact fact recovery consults, and the refusal is
    typed: ``run_terminal``, with no attempt bumped and no bundle minted."""

    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    backend1 = _recoverable_backend(
        run_root,
        _token_manager(),
        workspace,
        [],
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_read", {"path": "notes.md"}, "c1"),),
            ),
            ModelTurn(
                response_id="r2",
                tool_calls=(fake_tool_call("fs_read", {"path": "notes.md"}, "c2"),),
            ),
            ModelTurn(response_id="r3", final_text="never reached"),
        ],
    )
    submission = backend1.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="steps",
            runtime_config=runtime_config("fs.read"),
            multi_turn=False,
            max_steps=1,
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert eventually(lambda: backend1._record(run_id).terminal)
    wait_for_durable_status(run_root, run_id, where=lambda s: s["terminal"])
    stored = backend1.checkpoint_store.latest(run_id)
    assert stored is not None and stored.checkpoint.terminal is False  # the closed-limited shape
    backend1.stop_watchdog()

    # The restart view: dead runs must not be advertised resumable.
    backend2 = _recoverable_backend(
        run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")]
    )
    row = next(
        entry for entry in backend2.list_runs("tenant_a")["runs"] if entry["run_id"] == run_id
    )
    assert row["terminal"] is True
    assert row["recoverable"] is False
    with pytest.raises(NativeAgentError) as excinfo:
        backend2.resume_run(run_id, token)
    assert excinfo.value.error_code == "run_terminal"
    # A refusal is not an attempt: nothing bumped, nothing quarantined.
    assert not (run_root / run_id / "recover_attempts.json").exists()
    assert not (run_root / run_id / "failure.json").exists()
    assert backend2.recover_runs() == []


def test_a_concurrent_resume_loser_answers_the_already_live_shape(tmp_path: Path) -> None:
    """The register-record CAS loser used to answer the same misleading 400 (studio
    double-click shape). Losing the claim race means the run IS being resumed — the loser
    answers the already-live success shape (``resumed: false``), like the record-exists
    branch. Deterministic: ``register_record`` is patched to register (the winner) and then
    report the loss."""

    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    backend1 = _recoverable_backend(
        run_root,
        _token_manager(),
        workspace,
        [],
        turns=[ModelTurn(response_id="r1", final_text="parked answer")],
    )
    backend1.idle_timeout_s = 300.0
    submission = _submit_multi_turn(backend1, workspace)
    run_id, token = submission.run_id, submission.run_token
    assert eventually(lambda: backend1._record(run_id).state is SessionState.AWAITING_INPUT)
    wait_for_durable_status(run_root, run_id, where=lambda s: s["state"] == "awaiting_input")
    backend1.stop_watchdog()

    backend2 = _recoverable_backend(
        run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="resumed")]
    )
    original_register = backend2._recovery._context.register_record
    lost = {"count": 0}

    def _lose_once(record):  # noqa: ANN001
        if lost["count"] == 0:
            lost["count"] += 1
            assert original_register(record) is True  # the "winner" holds the live record
            return False  # ...and this caller lost the claim race
        return original_register(record)

    backend2._recovery._context = dataclasses.replace(
        backend2._recovery._context, register_record=_lose_once
    )

    out = backend2.resume_run(run_id, token)
    assert out["run_id"] == run_id
    assert out["resumed"] is False
    assert out["terminal"] is False
    # The loss is not a failure: nothing bumped, nothing quarantined.
    assert not (run_root / run_id / "recover_attempts.json").exists()
    assert not (run_root / run_id / "failure.json").exists()
    # The injected "winner" record has no driver behind it; drop it so cleanup does not
    # wait on a run nothing is driving.
    with backend2._lock:
        backend2._records.pop(run_id, None)


def test_recover_runs_rejects_runtime_config_hash_mismatch(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    backend = _recoverable_backend(run_root, _token_manager(), workspace, [], turns=[ModelTurn(final_text="x")])
    backend.max_recover_attempts = 1

    run_id = "run_bad_config_hash"
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True)
    backend.checkpoint_store.put(RunCheckpoint(run_id=run_id, seq=1, terminal=False))
    write_json_atomic(
        run_dir / "run.json",
        {
            **_recovery_metadata(run_id, workspace),
            "runtime_config_hash": "not-the-config-hash",
        },
    )

    assert backend.recover_runs() == []
    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["error_code"] == "backend_run_corrupt"
    assert "runtime config hash mismatch" in failure["error"]


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
    write_json_atomic(run_dir / "run.json", _recovery_metadata(run_id, workspace))
    write_json_atomic(run_dir / "lease.json", _stale_lease_payload(run_id))

    resumed: list = []
    monkeypatch.setattr(
        backend._recovery,
        "resume_from_checkpoint",
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
    errors: list = []
    barrier = threading.Barrier(2, timeout=30.0)

    def claim(backend) -> None:
        # Bound the rendezvous and capture any failure: under the full suite's background-thread
        # contention a worker can stall before it reaches the barrier; an unbounded wait/join
        # there would wedge the whole run forever (only the faulthandler watchdog could break it).
        # A bounded barrier + surfaced error fails this test loudly instead of hanging the suite.
        try:
            barrier.wait()
            results.append(backend.lease_store.try_claim("run_x", backend._worker_id, backend.lease_ttl_s))
        except BaseException as exc:  # noqa: BLE001 - surface to the main thread, don't swallow
            errors.append(exc)

    threads = [threading.Thread(target=claim, args=(b,)) for b in (b1, b2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60.0)
    assert not any(thread.is_alive() for thread in threads), "claim worker did not finish in time"
    assert not errors, f"claim worker raised: {errors}"

    assert results.count(True) == 1


def test_multinode_reclaim_over_shared_sqlite(tmp_path: Path, monkeypatch) -> None:
    # Two backends share ONLY a SQLite db (separate run_roots, no shared files). Backend A
    # "crashes" leaving an orphan run in the shared db (checkpoint + stale lease); backend B,
    # which never hosted it, discovers and reclaims it across the instance boundary. This is
    # what a per-host lease.json cannot do. Resume internals (run.json, restore) are covered
    # elsewhere, so the resume is stubbed — the point here is cross-instance discovery + CAS.
    workspace = _workspace(tmp_path)
    db = tmp_path / "shared.db"
    shared_checkpoints = SqliteCheckpointStore(db)

    run_id = "run_orphan"
    shared_checkpoints.put(RunCheckpoint(run_id=run_id, seq=1, terminal=False))
    SqliteLeaseStore(db).heartbeat(run_id, "worker_a", ttl_s=0.0)  # A crashed -> lease is stale
    time.sleep(0.02)

    backend_b = RunnerBackend(
        run_root=tmp_path / "b_runs",  # B's own run_root — it never saw run_orphan's files
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=lambda spec, token: FakeModelAdapter(turns=[ModelTurn(final_text="x")]),
        checkpoint_store=shared_checkpoints,
        lease_store=SqliteLeaseStore(db),
    )
    resumed: list = []
    monkeypatch.setattr(
        backend_b._recovery,
        "attempt_resume",
        # Pin moved with the typed-refusal change: reclaim counts only RESUMED.
        lambda run_dir, rid: (resumed.append(rid) or ResumeOutcome.RESUMED),
    )

    reclaimed = backend_b._reclaim_stale_runs()

    assert reclaimed == [run_id]  # B found A's orphan through the shared db
    assert resumed == [run_id]  # and invoked resume across the instance boundary
    assert backend_b.lease_store.owner(run_id) == backend_b._worker_id  # CAS flipped ownership to B


def test_multinode_reclaim_resumes_from_shared_metadata_without_local_run_json(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    db = tmp_path / "shared.db"
    shared_checkpoints = SqliteCheckpointStore(db)
    shared_leases = SqliteLeaseStore(db)
    token_manager = _token_manager()
    run_id = "run_shared_meta"
    config = _default_config()
    meta = {
        "schema_version": _RUN_META_SCHEMA_VERSION,
        "run_id": run_id,
        "tenant_id": "tenant_a",
        "user_id": "user_a",
        "workspace_root": str(workspace),
        "mode": "propose",
        "workspace_backend": "overlay",
        "multi_turn": True,
        "created_at": time.time(),
        "title": "shared metadata resume",
        "limits": {
            "max_steps": 30,
            "max_tool_calls": 100,
            "max_bytes_read": 1_000_000,
            "max_duration_s": 900,
        },
        "permission_policy": PermissionPolicy().to_json(),
        "runtime_config": config.to_json(),
        "runtime_config_version": config.config_version,
        "runtime_config_hash": config.config_hash,
        "runtime_config_issuer": "test",
        "runtime_config_reason": "shared metadata fixture",
        "runtime_config_committed_at": time.time(),
    }
    shared_checkpoints.put_run_metadata(run_id, meta)
    shared_checkpoints.put(
        RunCheckpoint(
            run_id=run_id,
            seq=1,
            status="completed",
            previous_turn_handle="r1",
            terminal=False,
        )
    )
    shared_leases.heartbeat(run_id, "worker_a", ttl_s=0.0)
    time.sleep(0.02)

    local_run_root = tmp_path / "b_runs"
    adapters: list[FakeModelAdapter] = []

    def factory(spec, llm_gateway_token):
        del spec, llm_gateway_token
        adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r2", final_text="recovered")])
        adapters.append(adapter)
        return adapter

    backend_b = RunnerBackend(
        run_root=local_run_root,
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
        checkpoint_store=shared_checkpoints,
        lease_store=shared_leases,
    )

    assert not (local_run_root / run_id / "run.json").exists()
    reclaimed = backend_b._reclaim_stale_runs()

    assert reclaimed == [run_id]
    assert run_id in backend_b._records
    assert (local_run_root / run_id / "run.json").exists()
    assert _read_run_meta(local_run_root / run_id)["runtime_config_hash"] == config.config_hash
    assert backend_b.lease_store.owner(run_id) == backend_b._worker_id
    backend_b.shutdown(drain=True)


def test_shared_run_metadata_requires_supported_schema_before_materializing(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    db = tmp_path / "shared.db"
    shared_checkpoints = SqliteCheckpointStore(db)
    run_id = "run_future_meta"
    shared_checkpoints.put_run_metadata(
        run_id,
        {
            "schema_version": "future.backend-run.v99",
            "run_id": run_id,
            "tenant_id": "tenant_a",
            "user_id": "user_a",
            "workspace_root": str(workspace),
        },
    )
    backend = RunnerBackend(
        run_root=tmp_path / "local_runs",
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        checkpoint_store=shared_checkpoints,
    )
    run_dir = backend.run_root / run_id

    assert backend._read_recovery_meta(run_dir, run_id) is None
    assert not (run_dir / "run.json").exists()


def test_sqlite_lease_concurrent_claim_across_instances(tmp_path: Path) -> None:
    # The cross-instance guarantee: two SqliteLeaseStore instances on the same db (standing
    # in for two hosts) race to claim the same absent/stale run; the transactional CAS lets
    # exactly one win.
    db = tmp_path / "shared.db"
    # Initialize the db file (schema + WAL-mode switch, which needs an EXCLUSIVE lock) ONCE up
    # front. Doing it concurrently inside both workers raced the WAL init against the CAS write,
    # and under the full suite's background-thread contention one worker could sit out the entire
    # 30s busy_timeout ("database is locked"), miss the barrier, and previously wedge the suite.
    # Pre-creating each instance keeps the raced section to exactly the try_claim CAS — the thing
    # under test — with no setup-time lock contention.
    stores = [SqliteLeaseStore(db) for _ in range(2)]
    results: list[bool] = []
    errors: list[BaseException] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=30.0)

    def claim(worker_id: str, store: SqliteLeaseStore) -> None:
        # Bound the rendezvous and capture any failure: a worker that still stalls fails this test
        # loudly instead of hanging the suite forever (an unbounded barrier/join would wedge the
        # whole run, breakable only by the faulthandler watchdog).
        try:
            barrier.wait()
            won = store.try_claim("run_x", worker_id, ttl_s=30.0)
            with results_lock:
                results.append(won)
        except BaseException as exc:  # noqa: BLE001 - surface to the main thread, don't swallow
            with results_lock:
                errors.append(exc)

    threads = [threading.Thread(target=claim, args=(f"w{i}", stores[i])) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60.0)
    assert not any(thread.is_alive() for thread in threads), "claim worker did not finish in time"
    assert not errors, f"claim worker raised: {errors}"

    assert results.count(True) == 1


def test_backend_list_runs_and_historical_reads_survive_restart(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    adapters: list = []
    backend1 = _scripted_backend(
        tmp_path, workspace, adapters, [ModelTurn(response_id="r1", final_text="hello world")]
    )
    submission = _submit_multi_turn(backend1, workspace)  # instruction "hi"
    run_id = submission.run_id
    try:
        assert eventually(lambda: backend1._record(run_id).state is SessionState.AWAITING_INPUT, timeout_s=20)
    finally:
        backend1.cancel_run(run_id, submission.run_token)
        backend1.wait_for_run(run_id, timeout_s=20)

    # JSONL commits before the best-effort status projection. Simulate a kill in that window and
    # require restart listing/status to use the authoritative committed tail.
    run_dir = tmp_path / "runs" / run_id
    wait_for_durable_status(tmp_path / "runs", run_id, where=lambda s: s["terminal"])
    committed_seq = inspect_event_log_tail(run_dir / "events.jsonl").last_seq
    assert committed_seq >= 2
    status_payload = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    status_payload["last_event_seq"] = committed_seq - 1
    write_json_atomic(run_dir / "status.json", status_payload)

    # "restart": a brand-new backend over the same run_root, with NO in-memory records.
    backend2 = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=lambda *_a, **_k: _ScriptedTurnAdapter([]),
    )
    listing = backend2.list_runs("tenant_a")["runs"]
    entry = next(r for r in listing if r["run_id"] == run_id)
    assert entry["title"] == "hi"
    assert entry["last_event_seq"] == committed_seq
    token = entry["read_token"]
    # historical event read with no live record
    events = backend2.events(run_id, token)["events"]
    assert any(e.get("type") == "turn.settled" for e in events)
    historical_status = backend2.status(run_id, token)
    # The run above was ended by cancel_run while parked; since the close boundary promotes an
    # acknowledged cancel, the historical record says so (it used to read "completed").
    assert historical_status["state"] == "cancelled"
    assert historical_status["terminal"] is True
    assert historical_status["last_event_seq"] == entry["last_event_seq"]
    assert "status" not in historical_status
    # tenant scoping
    assert backend2.list_runs("nobody")["runs"] == []
    # auth: a bad token, and a path-traversal run id, are rejected
    with pytest.raises(PermissionDenied):
        backend2.events(run_id, "not-a-token")
    traversal = backend2.token_manager.issue(
        kind="run_access", audience="monoid.backend",
        run_id="../escape", tenant_id="tenant_a", user_id="user_a", ttl_s=60,
    )
    with pytest.raises(PermissionDenied):
        backend2.events("../escape", traversal)


def test_backend_list_runs_materializes_shared_recovery_metadata(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    run_id = "run_shared_meta"
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True)
    runtime = _default_config()
    shared_checkpoints = SqliteCheckpointStore(tmp_path / "checkpoints.db")
    shared_checkpoints.put_run_metadata(
        run_id,
        {
            "schema_version": _RUN_META_SCHEMA_VERSION,
            "run_id": run_id,
            "tenant_id": "tenant_a",
            "user_id": "user_a",
            "title": "shared title",
            "created_at": 123.0,
            "workspace_root": str(workspace),
            "runtime_config": runtime.to_json(),
            "runtime_config_hash": runtime.config_hash,
        },
    )
    bad_run_id = "run_bad_meta"
    (run_root / bad_run_id).mkdir()
    shared_checkpoints.put_run_metadata(
        bad_run_id,
        {"schema_version": "unsupported.v0", "run_id": bad_run_id, "tenant_id": "tenant_a"},
    )
    backend = RunnerBackend(
        run_root=run_root,
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=lambda *_a, **_k: _ScriptedTurnAdapter([]),
        checkpoint_store=shared_checkpoints,
    )

    listing = backend.list_runs("tenant_a")["runs"]

    assert listing[0]["run_id"] == run_id
    assert listing[0]["title"] == "shared title"
    assert all(entry["run_id"] != bad_run_id for entry in listing)
    assert (run_dir / "run.json").exists()
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["title"] == "shared title"


def test_backend_history_without_status_artifact_is_not_reported_completed(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    run_id = "run_created_only"
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True)
    runtime = _default_config()
    write_json_atomic(
        run_dir / "run.json",
        {
            "schema_version": _RUN_META_SCHEMA_VERSION,
            "run_id": run_id,
            "tenant_id": "tenant_a",
            "user_id": "user_a",
            "title": "created only",
            "created_at": 123.0,
            "workspace_root": str(workspace),
            "runtime_config": runtime.to_json(),
            "runtime_config_hash": runtime.config_hash,
        },
    )
    backend = RunnerBackend(
        run_root=run_root,
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=lambda *_a, **_k: _ScriptedTurnAdapter([]),
    )

    listing = backend.list_runs("tenant_a")["runs"]
    entry = next(item for item in listing if item["run_id"] == run_id)

    assert entry["state"] == "created"
    assert entry["terminal"] is False
    assert entry["last_event_seq"] == 0
    status = backend.status(run_id, entry["read_token"])
    assert status["state"] == "created"
    assert status["terminal"] is False
