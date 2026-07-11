from __future__ import annotations

import json
import logging
import queue
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from monoid_agent_kernel.core.agents import AgentDefinition, AgentRuntimeConfig
from monoid_agent_kernel.core.control import ControlCommand
from monoid_agent_kernel.core.wire_validation import (
    optional_list,
    parse_bool,
    parse_int,
    parse_str,
    require_object,
)
from monoid_agent_kernel.reference._shared.http_util import (
    HardenedThreadingHTTPServer,
    HttpRequestTooLarge,
    log_http_request,
    read_json_limited,
    redact_internal_error,
)
from monoid_agent_kernel.reference.backend.service import BackendRunRequest, RunnerBackend
from monoid_agent_kernel.errors import NativeAgentError, PermissionDenied
from monoid_agent_kernel.permissions import PermissionPolicy

_LOGGER = logging.getLogger("monoid_agent_kernel.backend.http")

# Bridge sentinel: the pump coroutine pushes this once (in finally) to mark end-of-stream.
_STREAM_SENTINEL = object()
# Frames buffered before a slow/disconnected client is deemed too slow and the run cancelled.
_STREAM_HIGH_WATER = 2000


def _drain_to_sentinel(q: queue.Queue, *, deadline_s: float = 15.0) -> None:
    """Discard queued frames until the sentinel (bounded), so a disconnect can't pin the
    handler thread forever waiting on the pump to finalize."""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            item = q.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is _STREAM_SENTINEL:
            return


def make_backend_handler(backend: RunnerBackend, *, admin_token: str | None) -> type[BaseHTTPRequestHandler]:
    class BackendHttpHandler(BaseHTTPRequestHandler):
        server_version = "MonoidBackend/0.2"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/healthz":
                    self._write_json({"ok": True})
                    return
                parts = [part for part in parsed.path.split("/") if part]
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "status":
                    run_id = parts[2]
                    self._write_json(backend.status(run_id, self._bearer_token()))
                    return
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "result":
                    run_id = parts[2]
                    self._write_json(backend.result(run_id, self._bearer_token()))
                    return
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "events":
                    run_id = parts[2]
                    query = parse_qs(parsed.query)
                    from_seq = int((query.get("from_seq") or ["0"])[0])
                    limit_raw = (query.get("limit") or [None])[0]
                    if "text/event-stream" in (self.headers.get("Accept") or ""):
                        token = self._bearer_token()
                        # Authenticate and resolve the run before committing a 200 SSE response.
                        backend.status(run_id, token)
                        self._stream_event_subscription(
                            backend.subscribe_events(
                                run_id,
                                token,
                                from_seq=from_seq,
                                last_event_id=self.headers.get("Last-Event-ID"),
                            )
                        )
                        return
                    self._write_json(
                        backend.events(
                            run_id,
                            self._bearer_token(),
                            from_seq=from_seq,
                            limit=None if limit_raw is None else int(limit_raw),
                        )
                    )
                    return
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "diagnostics":
                    run_id = parts[2]
                    query = parse_qs(parsed.query)
                    limit_raw = (query.get("event_limit") or ["50"])[0]
                    self._write_json(
                        backend.diagnostics(
                            run_id,
                            self._bearer_token(),
                            event_limit=int(limit_raw),
                        )
                    )
                    return
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "runtime-config":
                    run_id = parts[2]
                    self._write_json(backend.runtime_config(run_id, self._bearer_token()))
                    return
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "jobs":
                    run_id = parts[2]
                    self._write_json(backend.jobs(run_id, self._bearer_token()))
                    return
                if len(parts) == 5 and parts[:2] == ["v1", "runs"] and parts[3] == "jobs":
                    run_id = parts[2]
                    self._write_json(backend.job_status(run_id, self._bearer_token(), parts[4]))
                    return
                if len(parts) == 6 and parts[:2] == ["v1", "runs"] and parts[3] == "jobs" and parts[5] == "logs":
                    run_id = parts[2]
                    query = parse_qs(parsed.query)
                    tail_raw = (query.get("tail_bytes") or [None])[0]
                    offset_raw = (query.get("offset") or [None])[0]
                    self._write_json(
                        backend.job_logs(
                            run_id,
                            self._bearer_token(),
                            parts[4],
                            stream=(query.get("stream") or ["stdout"])[0],
                            tail_bytes=None if tail_raw is None else int(tail_raw),
                            offset=None if offset_raw is None else int(offset_raw),
                        )
                    )
                    return
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "proposal":
                    run_id = parts[2]
                    self._write_json(backend.proposal(run_id, self._bearer_token()))
                    return
                if len(parts) == 5 and parts[:2] == ["v1", "runs"] and parts[3:5] == ["proposal", "diff"]:
                    run_id = parts[2]
                    self._write_json(backend.proposal_diff(run_id, self._bearer_token()))
                    return
                if len(parts) >= 6 and parts[:2] == ["v1", "runs"] and parts[3:5] == ["proposal", "files"]:
                    run_id = parts[2]
                    proposal_path = unquote("/".join(parts[5:]))
                    self._write_json(backend.proposal_file(run_id, self._bearer_token(), proposal_path))
                    return
                if len(parts) == 4 and parts[:2] == ["v1", "tenants"] and parts[3] == "usage":
                    self._require_admin()
                    self._write_json(backend.tenant_usage(parts[2]))
                    return
                self._write_error(HTTPStatus.NOT_FOUND, "not found")
            except Exception as exc:
                self._write_exception(exc)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/v1/runs":
                    self._require_admin()
                    request = self._parse_run_request(self._read_json())
                    self._write_json(backend.submit_run(request).to_json(), status=HTTPStatus.ACCEPTED)
                    return
                if parsed.path == "/v1/runs/stream":
                    self._require_admin()
                    request = self._parse_run_request(self._read_json())
                    self._stream_run_sse(request)
                    return
                parts = [part for part in parsed.path.split("/") if part]
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "cancel":
                    run_id = parts[2]
                    self._write_json(backend.cancel_run(run_id, self._bearer_token()))
                    return
                if len(parts) == 6 and parts[:2] == ["v1", "runs"] and parts[3] == "jobs" and parts[5] == "cancel":
                    run_id = parts[2]
                    self._write_json(backend.cancel_job(run_id, self._bearer_token(), parts[4]))
                    return
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "runtime-config":
                    run_id = parts[2]
                    payload = self._read_json()
                    self._write_json(
                        backend.replace_runtime_config(
                            run_id,
                            self._bearer_token(),
                            expected_version=parse_int(payload, "expected_version", default=0),
                            issuer=parse_str(payload, "issuer"),
                            reason=parse_str(payload, "reason"),
                            config=AgentRuntimeConfig.from_json(payload["config"]),
                        )
                    )
                    return
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "messages":
                    run_id = parts[2]
                    payload = self._read_json()
                    # Optional inbox envelope fields: a client-supplied message_id makes the send
                    # idempotent (a retry with the same id is processed once).
                    self._write_json(
                        backend.send_message(
                            run_id,
                            self._bearer_token(),
                            content=str(payload.get("content") or ""),
                            message_id=str(payload.get("message_id") or ""),
                            source=str(payload.get("source") or "http"),
                            correlation_id=str(payload.get("correlation_id") or ""),
                        )
                    )
                    return
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "control":
                    run_id = parts[2]
                    payload = self._read_json()
                    # The bearer token authorizes the run; carry it into args so dispatch (and the
                    # method it wraps) can authorize, without putting it on the wire envelope.
                    args = require_object(payload["args"], "args") if "args" in payload else {}
                    args["token"] = self._bearer_token()
                    command = ControlCommand.from_json(
                        {
                            **payload,
                            "run_id": run_id,
                            "args": args,
                        }
                    )
                    self._write_json(backend.dispatch(command).to_json())
                    return
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] and parts[3] == "tasks":
                    run_id = parts[2]
                    payload = self._read_json()
                    self._write_json(
                        backend.create_task(
                            run_id,
                            self._bearer_token(),
                            kind=parse_str(payload, "kind"),
                            request=require_object(payload["request"], "request") if "request" in payload else {},
                        )
                    )
                    return
                if (
                    len(parts) == 6
                    and parts[:2] == ["v1", "runs"]
                    and parts[3] == "tasks"
                    and parts[5] == "result"
                ):
                    run_id = parts[2]
                    task_id = parts[4]
                    payload = self._read_json()
                    self._write_json(
                        backend.report_task_result(
                            run_id,
                            self._bearer_token(),
                            task_id=task_id,
                            result=require_object(payload["result"], "result") if "result" in payload else {},
                            status=parse_str(payload, "status", default="answered"),
                        )
                    )
                    return
                if len(parts) == 5 and parts[:2] == ["v1", "runs"] and parts[3] == "proposal":
                    run_id = parts[2]
                    action = parts[4]
                    payload = self._read_json()
                    if action == "export":
                        self._write_json(backend.export_proposal_package(run_id, self._bearer_token()))
                        return
                    if action == "approve":
                        self._write_json(
                            backend.approve_proposal(
                                run_id,
                                self._bearer_token(),
                                approver_id=str(payload["approver_id"]),
                                approved_paths=tuple(
                                    parse_str({"path": path}, "path")
                                    for path in optional_list(payload, "approved_paths")
                                ),
                                note=parse_str(payload, "note"),
                            )
                        )
                        return
                    if action == "reject":
                        self._write_json(
                            backend.reject_proposal(
                                run_id,
                                self._bearer_token(),
                                approver_id=str(payload["approver_id"]),
                                reason=str(payload["reason"]),
                            )
                        )
                        return
                    if action == "apply":
                        approval_path = payload.get("approval_path")
                        self._write_json(
                            backend.apply_proposal(
                                run_id,
                                self._bearer_token(),
                                target=Path(str(payload["target"])),
                                approval_path=Path(str(approval_path)) if approval_path else None,
                                dry_run=parse_bool(payload, "dry_run", default=False),
                            )
                        )
                        return
                self._write_error(HTTPStatus.NOT_FOUND, "not found")
            except Exception as exc:
                self._write_exception(exc)

        def log_request(self, code: Any = "-", size: Any = "-") -> None:  # noqa: ARG002
            log_http_request(_LOGGER, self, code)

        def log_message(self, _format: str, *_args: Any) -> None:
            return None

        def _read_json(self) -> dict[str, Any]:
            return read_json_limited(self)

        def _parse_run_request(self, payload: dict[str, Any]) -> BackendRunRequest:
            max_duration_raw = payload.get("max_duration_s", 900)
            return BackendRunRequest(
                tenant_id=str(payload["tenant_id"]),
                user_id=str(payload["user_id"]),
                workspace_root=Path(str(payload["workspace_root"])),
                instruction=str(payload["instruction"]),
                mode=str(payload.get("mode") or "propose"),  # type: ignore[arg-type]
                workspace_backend=str(payload.get("workspace_backend") or "overlay"),  # type: ignore[arg-type]
                max_steps=int(payload.get("max_steps") or 30),
                max_tool_calls=int(payload.get("max_tool_calls") or 100),
                max_bytes_read=int(payload.get("max_bytes_read") or 1_000_000),
                max_duration_s=None if max_duration_raw is None else int(max_duration_raw),
                permission_policy=PermissionPolicy.from_json(payload.get("permission_policy")),
                agent_definition=(
                    AgentDefinition.from_json(payload["agent_definition"])
                    if payload.get("agent_definition") is not None
                    else None
                ),
                runtime_config=(
                    AgentRuntimeConfig.from_json(payload["runtime_config"])
                    if payload.get("runtime_config") is not None
                    else None
                ),
                multi_turn=parse_bool(payload, "multi_turn", default=False),
                metadata=require_object(payload["metadata"], "metadata") if "metadata" in payload else {},
            )

        def _stream_run_sse(self, request: BackendRunRequest) -> None:
            """Drive a stream-run over SSE: bridge the async ``astream_run`` (shared loop) to
            this sync handler thread via a thread-safe queue, writing one ``data:`` frame each.

            The pump only ever ``put_nowait``s (a blocking put would freeze the shared loop and
            every run on it); a slow/disconnected client is bounded by a high-water-mark that
            cooperatively cancels the run. SSE headers are sent only once the first frame is in
            hand, so a pre-stream failure (validation/tokens, before any frame) surfaces as a
            normal HTTP error rather than a 200 empty stream."""
            q: queue.Queue = queue.Queue()
            state: dict[str, Any] = {"run_id": None}

            async def _pump() -> None:
                try:
                    async for frame in backend.astream_run(request):
                        if frame.get("kind") == "meta":
                            state["run_id"] = frame.get("run_id")
                        q.put_nowait(frame)
                        if q.qsize() > _STREAM_HIGH_WATER and state["run_id"]:
                            backend.request_stream_cancel(str(state["run_id"]))
                finally:
                    q.put_nowait(_STREAM_SENTINEL)

            future = backend.spawn_coroutine(_pump())
            first = q.get()
            if first is _STREAM_SENTINEL:
                # No frame was produced -> a pre-stream failure; re-raise it as a normal error
                # response (SSE headers were never sent). future is done once the sentinel lands.
                exc = future.exception()
                raise exc if exc is not None else NativeAgentError(
                    "stream produced no output", error_code="internal_error"
                )
            # Commit to the SSE body. This route is long-lived, so clear the 30s socket timeout.
            self.connection.settimeout(None)
            self.send_response(int(HTTPStatus.OK))
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            frame = first
            while True:
                try:
                    self.wfile.write(
                        b"data: "
                        + json.dumps(frame, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                        + b"\n\n"
                    )
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    # Client disconnected: cancel cooperatively, drain to the sentinel (bounded),
                    # then cancel the pump future as a last resort.
                    if state["run_id"]:
                        backend.request_stream_cancel(str(state["run_id"]))
                    _drain_to_sentinel(q)
                    future.cancel()
                    return
                frame = q.get()
                if frame is _STREAM_SENTINEL:
                    return

        def _stream_event_subscription(self, subscription: Any) -> None:
            """Write a reusable event subscription as replay-safe SSE frames."""

            frames = iter(subscription.frames())
            self.close_connection = True
            self.send_response(int(HTTPStatus.OK))
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                # Establish the stream immediately even when the subscriber is caught up.
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                for frame in frames:
                    self.wfile.write(frame.to_sse())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def _bearer_token(self) -> str:
            header = self.headers.get("Authorization") or ""
            prefix = "Bearer "
            if not header.startswith(prefix):
                raise PermissionDenied("missing bearer token")
            return header[len(prefix) :].strip()

        def _require_admin(self) -> None:
            if admin_token is None:
                raise PermissionDenied("admin token is not configured")
            if self._bearer_token() != admin_token:
                raise PermissionDenied("invalid admin token")

        def _write_exception(self, exc: Exception) -> None:
            if isinstance(exc, PermissionDenied):
                self._write_error(HTTPStatus.UNAUTHORIZED, str(exc))
            elif isinstance(exc, HttpRequestTooLarge):
                self._write_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, str(exc))
            elif isinstance(exc, KeyError):
                self._write_error(HTTPStatus.NOT_FOUND, str(exc))
            elif isinstance(exc, (ValueError, NativeAgentError)):
                self._write_error(HTTPStatus.BAD_REQUEST, str(exc))
            else:
                # Unexpected: redact (no stack trace / internals to the client) and log full
                # detail server-side under a correlation id.
                self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, redact_internal_error(_LOGGER, self, exc))

        def _write_error(self, status: HTTPStatus, message: str) -> None:
            self._write_json({"error": message}, status=status)

        def _write_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return BackendHttpHandler


def create_backend_server(
    backend: RunnerBackend,
    *,
    host: str,
    port: int,
    admin_token: str,
) -> HardenedThreadingHTTPServer:
    return HardenedThreadingHTTPServer((host, port), make_backend_handler(backend, admin_token=admin_token))
