from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from native_agent_runner.reference.backend.service import BackendRunRequest, RunnerBackend
from native_agent_runner.errors import NativeAgentError, PermissionDenied
from native_agent_runner.permissions import PermissionPolicy
from native_agent_runner.shell import ShellPolicy
from native_agent_runner.tools.policy import ToolPolicy
from native_agent_runner.web import WebPolicy


def make_backend_handler(backend: RunnerBackend, *, admin_token: str | None) -> type[BaseHTTPRequestHandler]:
    class BackendHttpHandler(BaseHTTPRequestHandler):
        server_version = "NativeAgentRunnerBackend/0.2"

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
                    self._write_json(backend.events(run_id, self._bearer_token(), from_seq=from_seq))
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
                    payload = self._read_json()
                    max_duration_raw = payload.get("max_duration_s", 900)
                    request = BackendRunRequest(
                        tenant_id=str(payload["tenant_id"]),
                        user_id=str(payload["user_id"]),
                        workspace_root=Path(str(payload["workspace_root"])),
                        instruction=str(payload["instruction"]),
                        mode=str(payload.get("mode") or "propose"),  # type: ignore[arg-type]
                        workspace_backend=str(payload.get("workspace_backend") or "overlay"),  # type: ignore[arg-type]
                        model=str(payload.get("model") or "gpt-5.5"),
                        reasoning_effort=str(payload.get("reasoning_effort") or "medium"),
                        reasoning_summary=str(payload.get("reasoning_summary") or "off"),
                        max_steps=int(payload.get("max_steps") or 30),
                        max_tool_calls=int(payload.get("max_tool_calls") or 100),
                        max_bytes_read=int(payload.get("max_bytes_read") or 1_000_000),
                        max_duration_s=None if max_duration_raw is None else int(max_duration_raw),
                        permission_policy=PermissionPolicy.from_json(payload.get("permission_policy")),
                        tool_policy=ToolPolicy.from_json(payload.get("tool_policy")),
                        shell_policy=ShellPolicy.from_json(payload.get("shell_policy")),
                        web_policy=WebPolicy.from_json(payload.get("web_policy")),
                        metadata=dict(payload.get("metadata") or {}),
                    )
                    self._write_json(backend.submit_run(request).to_json(), status=HTTPStatus.ACCEPTED)
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
                                approved_paths=tuple(str(path) for path in payload.get("approved_paths") or ()),
                                note=str(payload.get("note") or ""),
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
                                dry_run=bool(payload.get("dry_run", False)),
                            )
                        )
                        return
                self._write_error(HTTPStatus.NOT_FOUND, "not found")
            except Exception as exc:
                self._write_exception(exc)

        def log_message(self, _format: str, *_args: Any) -> None:
            return None

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0:
                return {}
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON request body") from exc
            if not isinstance(payload, dict):
                raise ValueError("JSON request body must be an object")
            return payload

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
            elif isinstance(exc, KeyError):
                self._write_error(HTTPStatus.NOT_FOUND, str(exc))
            elif isinstance(exc, (ValueError, NativeAgentError)):
                self._write_error(HTTPStatus.BAD_REQUEST, str(exc))
            else:
                self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

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
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_backend_handler(backend, admin_token=admin_token))
