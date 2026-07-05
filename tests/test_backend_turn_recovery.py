from __future__ import annotations

from support.backend_harness import (
    ModelAdapterError,
    ModelTurn,
    Path,
    PermissionDenied,
    RunnerBackend,
    _InterruptingTurnAdapter,
    _calls,
    _scripted_backend,
    _submit_multi_turn,
    _token_manager,
    _workspace,
    eventually,
    json,
    pytest,
)

pytestmark = pytest.mark.integration


def test_backend_auto_retries_transient_turn_failure(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    adapters: list = []
    backend = _scripted_backend(
        tmp_path,
        workspace,
        adapters,
        [ModelAdapterError("rate limited", http_status=503, retryable=True),
         ModelTurn(response_id="r2", final_text="recovered")],
    )
    submission = _submit_multi_turn(backend, workspace)
    try:
        # the transient failure is auto-retried; the run settles + parks awaiting input
        assert eventually(lambda: backend._record(submission.run_id).state.value == "awaiting_input", timeout_s=20)
        assert backend._record(submission.run_id).state.value != "failed"
        assert _calls(adapters) == 2  # initial attempt + one retry
    finally:
        backend.cancel_run(submission.run_id, submission.run_token)
        backend.wait_for_run(submission.run_id, timeout_s=20)


def test_backend_parks_on_nonretryable_turn_failure_then_resumes(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    adapters: list = []
    backend = _scripted_backend(
        tmp_path,
        workspace,
        adapters,
        [ModelAdapterError("bad effort", http_status=400, retryable=False),
         ModelTurn(response_id="r2", final_text="fixed")],
    )
    backend.idle_timeout_s = 30.0
    submission = _submit_multi_turn(backend, workspace)
    run_id, token = submission.run_id, submission.run_token
    try:
        # config 4xx is NOT auto-retried — it parks for the user (status awaiting_input, not failed)
        assert eventually(lambda: backend._record(run_id).state.value == "awaiting_input", timeout_s=20)
        assert _calls(adapters) == 1
        # send_message succeeds (run is NOT terminal) and the resend settles
        assert backend.send_message(run_id, token, "try again")["status"] == "queued"
        assert eventually(lambda: _calls(adapters) >= 2, timeout_s=20)
    finally:
        backend.cancel_run(run_id, token)
        backend.wait_for_run(run_id, timeout_s=20)


def test_backend_gives_up_after_max_consecutive_turn_failures(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    adapters: list = []
    backend = _scripted_backend(
        tmp_path,
        workspace,
        adapters,
        [ModelAdapterError("transient", http_status=503, retryable=True) for _ in range(10)],
    )
    backend.max_consecutive_turn_failures = 2
    submission = _submit_multi_turn(backend, workspace)
    status = backend.wait_for_run(submission.run_id, timeout_s=20)
    assert status == "failed"
    assert _calls(adapters) == 2  # initial attempt + one retry, then give up at the cap


def test_backend_consecutive_failure_counter_resets_on_settle(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    adapters: list = []
    backend = _scripted_backend(
        tmp_path,
        workspace,
        adapters,
        [
            ModelAdapterError("t1", http_status=503, retryable=True),
            ModelTurn(response_id="a", final_text="a"),
            ModelAdapterError("t2", http_status=503, retryable=True),
            ModelTurn(response_id="b", final_text="b"),
        ],
    )
    backend.max_consecutive_turn_failures = 2
    backend.idle_timeout_s = 30.0
    submission = _submit_multi_turn(backend, workspace)
    run_id, token = submission.run_id, submission.run_token
    try:
        assert eventually(lambda: backend._record(run_id).state.value == "awaiting_input", timeout_s=20)  # retried + settled
        assert _calls(adapters) == 2
        backend.send_message(run_id, token, "again")  # drives the 2nd fail+retry+settle
        assert eventually(lambda: _calls(adapters) >= 4, timeout_s=20)
        # streak reset between settles -> cap of 2 never tripped
        assert backend._record(run_id).state.value != "failed"
    finally:
        backend.cancel_run(run_id, token)
        backend.wait_for_run(run_id, timeout_s=20)


def test_backend_interrupt_parks_turn_then_resumes(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "x.md").write_text("hi\n", encoding="utf-8")
    captured: dict = {}

    def factory(spec, llm_gateway_token):  # noqa: ANN001
        del spec, llm_gateway_token
        adapter = _InterruptingTurnAdapter()
        captured["adapter"] = adapter
        return adapter

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=_token_manager(),
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
    )
    backend.idle_timeout_s = 30.0
    submission = _submit_multi_turn(backend, workspace)
    run_id, token = submission.run_id, submission.run_token
    try:
        # Hand the loop to the adapter so it can interrupt itself mid-turn (deterministic).
        assert eventually(lambda: backend._record(run_id).loop is not None, timeout_s=20)
        captured["adapter"].loop_box.append(backend._record(run_id).loop)
        # The interrupt parks the multi-turn session (awaiting_input) — it is NOT terminal.
        assert eventually(lambda: backend._record(run_id).state.value == "awaiting_input", timeout_s=20)
        assert backend._record(run_id).terminal is False
        # The session is alive: a follow-up message resumes and settles.
        backend.send_message(run_id, token, "continue")
        assert eventually(lambda: captured["adapter"].calls >= 3, timeout_s=20)
        assert eventually(lambda: backend._record(run_id).state.value == "awaiting_input", timeout_s=20)
    finally:
        backend.cancel_run(run_id, token)
        backend.wait_for_run(run_id, timeout_s=20)


def test_backend_descendant_events_reads_child_and_checks_lineage(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    adapters: list = []
    backend = _scripted_backend(tmp_path, workspace, adapters, [ModelTurn(response_id="r1", final_text="ok")])
    submission = _submit_multi_turn(backend, workspace)
    run_id, token = submission.run_id, submission.run_token
    try:
        assert eventually(lambda: backend._record(run_id).state.value == "awaiting_input", timeout_s=20)
        # Simulate an isolated child subagent run's events.jsonl under the same run_root.
        child_id = f"{run_id}.sub.task_abc"
        child_dir = backend.run_root / child_id
        child_dir.mkdir(parents=True)
        (child_dir / "events.jsonl").write_text(
            json.dumps({"seq": 0, "type": "model.output.delta", "data": {"text": "hi"}}) + "\n"
            + json.dumps({"seq": 1, "type": "turn.settled", "data": {"final_text": "hi"}}) + "\n",
            encoding="utf-8",
        )
        out = backend.descendant_events(run_id, token, child_id)
        assert [e["type"] for e in out["events"]] == ["model.output.delta", "turn.settled"]
        # from_seq filters
        tail = backend.descendant_events(run_id, token, child_id, from_seq=1)
        assert [e["seq"] for e in tail["events"]] == [1]
        page = backend.descendant_events(run_id, token, child_id, from_seq=0, limit=1)
        assert [e["seq"] for e in page["events"]] == [0]
        assert page["next_seq"] == 1
        assert page["has_more"] is True
        next_page = backend.descendant_events(run_id, token, child_id, from_seq=page["next_seq"], limit=1)
        assert [e["seq"] for e in next_page["events"]] == [1]
        assert next_page["next_seq"] == 2
        # a non-descendant id is rejected even with a valid token
        with pytest.raises(PermissionDenied):
            backend.descendant_events(run_id, token, "some.other.run")
        # path traversal is rejected
        with pytest.raises(PermissionDenied):
            backend.descendant_events(run_id, token, f"{run_id}.sub.../escape")
        # a bad token is rejected
        with pytest.raises(Exception):  # noqa: B017 - TokenError family
            backend.descendant_events(run_id, "bad-token", child_id)
    finally:
        backend.cancel_run(run_id, token)
        backend.wait_for_run(run_id, timeout_s=20)
