from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from native_agent_runner.errors import NativeAgentError, PermissionDenied
from native_agent_runner.web import WebGatewayError
from native_agent_runner.reference.web_gateway.service import WebGatewayBackend


def make_web_gateway_handler(
    gateway: WebGatewayBackend,
    *,
    admin_token: str | None,
) -> type[BaseHTTPRequestHandler]:
    class WebGatewayHttpHandler(BaseHTTPRequestHandler):
        server_version = "NativeAgentRunnerWebGateway/0.9"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/healthz":
                    self._write_json({"ok": True})
                    return
                parts = [part for part in parsed.path.split("/") if part]
                if len(parts) == 5 and parts[:3] == ["internal", "web", "tenants"] and parts[4] == "usage":
                    self._require_admin()
                    self._write_json(gateway.tenant_usage(parts[3]))
                    return
                self._write_error(HTTPStatus.NOT_FOUND, "not found")
            except Exception as exc:
                self._write_exception(exc)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/internal/web/search":
                    self._write_json(gateway.handle_search(self._bearer_token(), self._read_json()))
                    return
                if parsed.path == "/internal/web/fetch":
                    self._write_json(gateway.handle_fetch(self._bearer_token(), self._read_json()))
                    return
                if parsed.path == "/internal/web/context":
                    self._write_json(gateway.handle_context(self._bearer_token(), self._read_json()))
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
                self._write_error(HTTPStatus.UNAUTHORIZED, str(exc), error_code="web_auth_error")
            elif isinstance(exc, WebGatewayError):
                self._write_error(
                    _status_for_web_error(exc),
                    str(exc),
                    error_code=exc.error_code,
                )
            elif isinstance(exc, ValueError):
                self._write_error(HTTPStatus.BAD_REQUEST, str(exc), error_code="web_bad_request")
            elif isinstance(exc, NativeAgentError):
                self._write_error(
                    HTTPStatus.BAD_REQUEST,
                    str(exc),
                    error_code=getattr(exc, "error_code", "web_bad_request"),
                )
            else:
                self._write_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc), error_code="web_server_error")

        def _write_error(
            self,
            status: HTTPStatus,
            message: str,
            *,
            error_code: str = "web_gateway_error",
        ) -> None:
            self._write_json(
                {
                    "error": message,
                    "error_code": error_code,
                    "http_status": int(status),
                },
                status=status,
            )

        def _write_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return WebGatewayHttpHandler


def _status_for_web_error(exc: WebGatewayError) -> HTTPStatus:
    if exc.error_code == "web_policy_denied":
        return HTTPStatus.FORBIDDEN
    if exc.error_code == "web_not_found":
        return HTTPStatus.NOT_FOUND
    if exc.error_code.endswith("_limit_exceeded"):
        return HTTPStatus.TOO_MANY_REQUESTS
    return HTTPStatus.BAD_REQUEST


def create_web_gateway_server(
    gateway: WebGatewayBackend,
    *,
    host: str,
    port: int,
    admin_token: str,
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_web_gateway_handler(gateway, admin_token=admin_token))
