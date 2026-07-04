from __future__ import annotations

from support.backend_harness import (
    HTTPError,
    ModelTurn,
    Path,
    Request,
    RunnerBackend,
    URLError,
    _backend,
    _default_config,
    _hitl_backend,
    _json_get,
    _json_request,
    _start_server,
    _token_manager,
    _wait_http_ready,
    _workspace,
    create_backend_server,
    eventually,
    pytest,
    threading,
    urlopen,
)
from monoid_agent_kernel.core.trace_context import new_traceparent, trace_id_of
from monoid_agent_kernel.recorder import append_event_to_run

pytestmark = pytest.mark.integration


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
        # Invariant: the oversized request is rejected before the body is read — NOT processed.
        # Over a real socket the server refuses the spoofed Content-Length and closes; the
        # client therefore sees EITHER a clean 413 OR a connection reset (the close racing the
        # unconsumed body, common on Windows). Both prove "rejected"; a 2xx would be the bug.
        try:
            urlopen(request, timeout=5)
        except HTTPError as exc:
            assert exc.code == 413
        except (URLError, OSError):
            pass  # reject surfaced as a connection reset — still rejected, not processed
        else:
            pytest.fail("oversized request was not rejected")
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
        assert backend.wait_for_run(run_id, timeout_s=5).value == "completed"
        status = _json_get(f"{base_url}/v1/runs/{run_id}/status", token=run_token)
        assert status["state"] == "completed"
        assert status["terminal"] is True
        assert "status" not in status
        result = _json_get(f"{base_url}/v1/runs/{run_id}/result", token=run_token)
        assert result["state"] == "completed"
        assert result["terminal"] is True
        assert result["final_text"] == "done"
        events = _json_get(f"{base_url}/v1/runs/{run_id}/events?from_seq=1", token=run_token)
        assert events["events"][0]["seq"] == 1
        page1 = _json_get(f"{base_url}/v1/runs/{run_id}/events?from_seq=1&limit=2", token=run_token)
        assert [event["seq"] for event in page1["events"]] == [1, 2]
        assert page1["next_seq"] == 3
        assert page1["has_more"] is True
        page2 = _json_get(
            f"{base_url}/v1/runs/{run_id}/events?from_seq={page1['next_seq']}&limit=2",
            token=run_token,
        )
        assert page2["events"][0]["seq"] == 3
        traceparent = new_traceparent()
        trace_event = append_event_to_run(
            backend._record(run_id).run_dir,
            "outbox.requested",
            data={
                "request_id": "trace_fixture",
                "destination": "diagnostics",
                "capability": "test.trace",
                "traceparent": traceparent,
            },
        )
        diagnostics = _json_get(f"{base_url}/v1/runs/{run_id}/diagnostics?event_limit=1", token=run_token)
        assert diagnostics["status"]["state"] == "completed"
        assert diagnostics["status"]["terminal"] is True
        assert [event["seq"] for event in diagnostics["events"]["items"]] == [trace_event.seq]
        assert diagnostics["events"]["next_seq"] >= diagnostics["events"]["from_seq"]
        assert diagnostics["failure"] is None
        assert diagnostics["recovery"]["attempts"] == 0
        assert trace_id_of(traceparent) in diagnostics["trace_ids"]
        usage = _json_get(f"{base_url}/v1/tenants/tenant_a/usage", token="admin")
        assert usage["total_tokens"] == 10
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "path,payload,token",
    [
        (
            "/v1/runs",
            {
                "tenant_id": "tenant_a",
                "user_id": "user_a",
                "workspace_root": "__workspace__",
                "instruction": "Run.",
                "runtime_config": "__runtime_config__",
                "multi_turn": "false",
            },
            "admin",
        ),
        (
            "/v1/runs",
            {
                "tenant_id": "tenant_a",
                "user_id": "user_a",
                "workspace_root": "__workspace__",
                "instruction": "Run.",
                "runtime_config": "__runtime_config__",
                "metadata": [],
            },
            "admin",
        ),
        ("/v1/runs/run_1/control", {"type": "status", "args": []}, "bad-run-token"),
        ("/v1/runs/run_1/tasks", {"kind": "automation", "request": []}, "bad-run-token"),
        ("/v1/runs/run_1/tasks/task_1/result", {"result": []}, "bad-run-token"),
        ("/v1/runs/run_1/proposal/apply", {"target": ".", "dry_run": "false"}, "bad-run-token"),
    ],
)
def test_backend_http_rejects_present_wrong_type_payload_fields(
    tmp_path: Path,
    path: str,
    payload: dict,
    token: str,
) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(tmp_path, workspace, [])
    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    request_payload = {
        key: str(workspace)
        if value == "__workspace__"
        else _default_config().to_json()
        if value == "__runtime_config__"
        else value
        for key, value in payload.items()
    }
    try:
        _wait_http_ready(base_url)
        with pytest.raises(HTTPError) as exc_info:
            _json_request(f"{base_url}{path}", request_payload, token=token)
        assert exc_info.value.code == 400
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
        assert backend.wait_for_run(run_id, timeout_s=10).value == "cancelled"
        status = _json_get(f"{base_url}/v1/runs/{run_id}/status", token=run_token)
        assert status["state"] == "cancelled"
        assert status["terminal"] is True
        assert status["error_code"] == "cancelled"
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


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
        assert eventually(lambda: backend._record(run_id).state.value == "awaiting_input", timeout_s=20)

        # A follow-up message is threaded as a second user turn.
        queued = _json_request(f"{base_url}/v1/runs/{run_id}/messages", {"content": "again"}, token=run_token)
        assert queued["status"] == "queued"
        assert eventually(
            lambda: len([r for a in adapters for r in a.requests if r.instruction]) >= 2,
            timeout_s=20,
        )
        instructions = [r.instruction for a in adapters for r in a.requests if r.instruction]
        assert "hello" in instructions and "again" in instructions

        # Create an automation task -> scoped callback token + URL.
        assert eventually(lambda: backend._record(run_id).state.value == "awaiting_input", timeout_s=20)
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
        backend.wait_for_run(run_id, timeout_s=20)
        assert backend._record(run_id).terminal is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
