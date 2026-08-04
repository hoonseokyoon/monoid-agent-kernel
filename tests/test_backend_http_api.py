from __future__ import annotations

import json
from typing import Any

from support.backend_harness import (
    BackendRunRequest,
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
from support.http import LINE_SEPARATORS, sse_data_frames_by_line

from monoid_agent_kernel.core.trace_context import new_traceparent, trace_id_of
from monoid_agent_kernel.recorder import append_event_to_run
from monoid_agent_kernel.reference.backend.http import make_backend_handler

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_backend_http_rejects_nonfinite_json_constants(tmp_path: Path, value: float) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(tmp_path, workspace, [])
    server, thread, base_url = _start_server(backend)
    try:
        with pytest.raises(HTTPError) as exc_info:
            _json_request(
                f"{base_url}/v1/runs",
                {
                    "tenant_id": "tenant_a",
                    "user_id": "user_a",
                    "workspace_root": str(workspace),
                    "instruction": "Run.",
                    "metadata": {"value": value},
                },
                token="admin",
            )
        assert exc_info.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_backend_http_rejects_finite_syntax_that_overflows_the_runtime(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(tmp_path, workspace, [])
    server, thread, base_url = _start_server(backend)
    try:
        body = (
            "{"
            '"tenant_id":"tenant_a",'
            '"user_id":"user_a",'
            f'"workspace_root":{json.dumps(str(workspace))},'
            '"instruction":"Run.",'
            '"max_duration_s":1e9999'
            "}"
        ).encode("utf-8")
        request = Request(
            f"{base_url}/v1/runs",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer admin",
            },
            method="POST",
        )

        with pytest.raises(HTTPError) as exc_info:
            urlopen(request, timeout=5)

        assert exc_info.value.code == 400
        assert b"invalid JSON request body" in exc_info.value.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


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
                {
                    "tenant_id": "tenant_a",
                    "user_id": "user_a",
                    "workspace_root": str(workspace),
                    "instruction": "Run.",
                },
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
        diagnostics = _json_get(
            f"{base_url}/v1/runs/{run_id}/diagnostics?event_limit=1", token=run_token
        )
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


def test_backend_event_sse_resumes_from_last_event_id_without_duplicates(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(tmp_path, workspace, [])
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Run.",
            runtime_config=_default_config(),
        )
    )
    assert backend.wait_for_run(submission.run_id, timeout_s=5).value == "completed"
    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    def read(last_event_id: str | None = None) -> tuple[list[int], str]:
        headers = {
            "Authorization": f"Bearer {submission.run_token}",
            "Accept": "text/event-stream",
        }
        if last_event_id is not None:
            headers["Last-Event-ID"] = last_event_id
        request = Request(
            f"{base_url}/v1/runs/{submission.run_id}/events?from_seq=1",
            headers=headers,
        )
        with urlopen(request, timeout=10) as response:
            assert response.headers["Content-Type"].startswith("text/event-stream")
            body = response.read().decode("utf-8")
        ids = [
            int(line.removeprefix("id: ")) for line in body.splitlines() if line.startswith("id: ")
        ]
        return ids, body

    try:
        _wait_http_ready(base_url)
        ids, body = read()
        assert ids == sorted(set(ids))
        assert "event: end" in body
        resumed_ids, _ = read(str(ids[0]))
        assert resumed_ids == ids[1:]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_event_sse_frames_survive_a_line_splitting_reader(tmp_path: Path) -> None:
    """The events route is the second writer of the same shape, on a different frame class.

    ``EventSubscriptionFrame.to_sse`` dumped with ``ensure_ascii=False``, so U+2028, U+2029 and
    U+0085 in any event string reached the wire as themselves and a ``str.splitlines`` reader --
    httpx's ``aiter_lines`` -- broke the frame there. Event data is not content-free: tool
    arguments and results, error messages and model text all ride it, so a separator arrives here
    the moment a model or a file supplies one. The split takes the frame's ``id:`` line with it,
    which is what a reconnect resumes ``Last-Event-ID`` from, so a truncated frame also costs the
    reader its place in the stream. Injected rather than modelled, so the carrier is the frame
    writer and not whichever event happens to carry text under today's content policy.
    """

    pytest.importorskip("httpx")
    workspace = _workspace(tmp_path)
    backend = _backend(tmp_path, workspace, [])
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Run.",
            runtime_config=_default_config(),
        )
    )
    assert backend.wait_for_run(submission.run_id, timeout_s=5).value == "completed"
    carried = f"before{LINE_SEPARATORS}after"
    injected = append_event_to_run(
        backend._record(submission.run_id).run_dir,
        "outbox.requested",
        data={"request_id": "separator_fixture", "destination": carried},
    )
    server, thread, base_url = _start_server(backend)
    try:
        frames = sse_data_frames_by_line(
            f"{base_url}/v1/runs/{submission.run_id}/events?from_seq=1",
            token=submission.run_token,
            headers={"Accept": "text/event-stream"},
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    delivered = [frame for frame in frames if frame.get("seq") == injected.seq]
    assert delivered, {
        "seqs": [frame.get("seq") for frame in frames],
        "hint": "the injected event never arrived whole",
    }
    assert delivered[0]["data"]["destination"] == carried
    # The terminal ``event: end`` frame rides the same writer and is what tells a client the
    # stream is over rather than cut.
    assert frames[-1].get("terminal") is True


def test_backend_event_sse_establishes_immediately_when_live_stream_is_caught_up(
    backend_factory: Any,
) -> None:
    workspace = backend_factory.workspace()
    backend = backend_factory.create(workspace=workspace)
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant",
            user_id="user",
            workspace_root=workspace,
            instruction="wait",
            runtime_config=_default_config(),
            multi_turn=True,
        )
    )
    assert eventually(
        lambda: backend._record(submission.run_id).state.value == "awaiting_input",
        timeout_s=10,
    )
    last_seq = backend.events(submission.run_id, submission.run_token)["events"][-1]["seq"]
    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        _wait_http_ready(base_url)
        request = Request(
            f"{base_url}/v1/runs/{submission.run_id}/events?from_seq=0",
            headers={
                "Authorization": f"Bearer {submission.run_token}",
                "Accept": "text/event-stream",
                "Last-Event-ID": str(last_seq),
            },
        )
        with urlopen(request, timeout=2) as response:
            assert response.headers["Content-Type"].startswith("text/event-stream")
            assert response.readline() == b": connected\n"
        backend.cancel_run(submission.run_id, submission.run_token)
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
    backend = _hitl_backend(
        tmp_path, workspace, adapters, turns=[ModelTurn(response_id="r1", final_text="first")]
    )
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
        assert eventually(
            lambda: backend._record(run_id).state.value == "awaiting_input", timeout_s=20
        )

        # A follow-up message is threaded as a second user turn.
        queued = _json_request(
            f"{base_url}/v1/runs/{run_id}/messages", {"content": "again"}, token=run_token
        )
        assert queued["status"] == "queued"
        assert eventually(
            lambda: len([r for a in adapters for r in a.requests if r.instruction]) >= 2,
            timeout_s=20,
        )
        instructions = [r.instruction for a in adapters for r in a.requests if r.instruction]
        assert "hello" in instructions and "again" in instructions

        # Create an automation task -> scoped callback token + URL.
        assert eventually(
            lambda: backend._record(run_id).state.value == "awaiting_input", timeout_s=20
        )
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


# --- the request body an error response leaves in the socket -------------------------------
#
# ``_require_admin()`` is the first statement of both run routes, so the request most likely to
# be refused is the one whose body is guaranteed unread. This handler is HTTP/1.0 and closes
# after every response; closing a socket that still holds unread data makes the platform send an
# RST rather than a FIN, and the 401 already written is discarded with it -- the caller sees a
# dropped connection and cannot tell a rejected token from a broken network. Whether the RST wins
# is a race, so the behavioral test for it (``test_backend_stream_rejects_non_admin``) failed
# roughly one run in seven rather than every time. These pin the logic instead of the race.


class _FakeSocket:
    def __init__(self) -> None:
        self.timeouts: list[float | None] = []
        self._timeout: float | None = 30.0

    def gettimeout(self) -> float | None:
        return self._timeout

    def settimeout(self, value: float | None) -> None:
        self._timeout = value
        self.timeouts.append(value)


class _RecordingReader:
    def __init__(self, payload: bytes, *, fail: Exception | None = None) -> None:
        self._payload = payload
        self._fail = fail
        self.reads: list[int] = []

    def read(self, size: int) -> bytes:
        self.reads.append(size)
        if self._fail is not None:
            raise self._fail
        return self._payload[:size]


def _drain_handler(headers: dict[str, str], reader: Any) -> Any:
    """A handler instance with only the attributes the drain touches — no socket, no server."""

    handler_cls = make_backend_handler(object(), admin_token="admin")
    handler = object.__new__(handler_cls)
    handler.headers = headers
    handler.rfile = reader
    handler.connection = _FakeSocket()
    handler.close_connection = False
    handler._body_consumed = False
    return handler


def test_an_error_response_drains_the_body_its_route_never_read() -> None:
    reader = _RecordingReader(b"x" * 40)
    handler = _drain_handler({"Content-Length": "40"}, reader)

    handler._discard_unread_request_body()

    assert reader.reads == [40], "the refused request's body must leave the socket"
    assert handler.close_connection is False
    # The wait is bounded well below the 30s request timeout and then restored, so a client that
    # declares a body and dribbles it cannot hold a handler thread for the long one.
    assert handler.connection.timeouts == [0.5, 30.0]


def test_a_body_already_read_by_the_route_is_not_read_twice() -> None:
    """The success path consumes the body itself; a later error must not read past it into
    whatever the socket holds next."""

    reader = _RecordingReader(b"x" * 40)
    handler = _drain_handler({"Content-Length": "40"}, reader)
    handler._body_consumed = True

    handler._discard_unread_request_body()

    assert reader.reads == []
    assert handler.close_connection is False


@pytest.mark.parametrize(
    "headers,why",
    [
        ({"Content-Length": "9999999"}, "a body past the drain cap is closed on, not read"),
        ({"Content-Length": "not-a-number"}, "an unparseable length cannot be drained"),
    ],
)
def test_a_body_the_error_path_will_not_drain_closes_the_connection(
    headers: dict[str, str], why: str
) -> None:
    reader = _RecordingReader(b"")
    handler = _drain_handler(headers, reader)

    handler._discard_unread_request_body()

    assert reader.reads == [], why
    assert handler.close_connection is True, why


def test_a_body_that_never_arrives_closes_rather_than_holding_the_thread() -> None:
    """The drain's own timeout is the bound. It is caught here rather than escaping into
    ``_write_error``, which is already writing a response and cannot raise."""

    reader = _RecordingReader(b"", fail=TimeoutError("timed out"))
    handler = _drain_handler({"Content-Length": "40"}, reader)

    handler._discard_unread_request_body()

    assert handler.close_connection is True
    assert handler.connection.timeouts == [0.5, 30.0], "the request timeout is restored"


def test_a_request_with_no_body_neither_reads_nor_closes() -> None:
    reader = _RecordingReader(b"")
    handler = _drain_handler({}, reader)

    handler._discard_unread_request_body()

    assert reader.reads == []
    assert handler.close_connection is False
