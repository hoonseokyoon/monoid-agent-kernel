from __future__ import annotations

from support.backend_harness import (
    BackendRunRequest,
    FakeModelAdapter,
    ModelTurn,
    Path,
    PermissionDenied,
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
    fake_tool_call,
    json,
    pytest,
    runtime_config,
    threading,
    time,
    tool_binding,
    write_json_atomic,
)
from monoid_agent_kernel.reference.backend.service import _read_run_meta

pytestmark = pytest.mark.integration


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
    assert eventually(lambda: backend1._record(run_id).status == "awaiting_input")

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
    assert backend2.wait_for_run(run_id, timeout_s=20) in {"completed", "limited", "failed"}
    instructions = [r.instruction for a in resumed for r in a.requests if r.instruction]
    assert "again" in instructions
    backend1.cancel_run(run_id, token)  # stop the defunct first-process worker


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
    assert eventually(lambda: backend1._record(run_id).status == "awaiting_input")

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
    assert failure["schema_version"] == "monoid.failure.v1"
    assert failure["type"] == "RuntimeError"
    assert "last_good_seq" in failure
    diagnostics = backend.diagnostics(run_id, submission.run_token)
    assert diagnostics["status"]["status"] == "failed"
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
            "schema_version": _RUN_META_SCHEMA_VERSION,
            "run_id": run_id,
            "runtime_config": _default_config().to_json(),
            "runtime_config_hash": "not-the-config-hash",
        },
    )

    assert backend.recover_runs() == []
    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["error_code"] == "unrecoverable"
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
        backend_b, "_attempt_resume", lambda run_dir, rid: (resumed.append(rid) or True)
    )

    reclaimed = backend_b._reclaim_stale_runs()

    assert reclaimed == [run_id]  # B found A's orphan through the shared db
    assert resumed == [run_id]  # and invoked resume across the instance boundary
    assert backend_b.lease_store.owner(run_id) == backend_b._worker_id  # CAS flipped ownership to B


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
        assert eventually(lambda: backend1._record(run_id).status == "awaiting_input", timeout_s=20)
    finally:
        backend1.cancel_run(run_id, submission.run_token)
        backend1.wait_for_run(run_id, timeout_s=20)

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
    token = entry["read_token"]
    # historical event read with no live record
    events = backend2.events(run_id, token)["events"]
    assert any(e.get("type") == "turn.settled" for e in events)
    assert "status" in backend2.status(run_id, token)
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
