"""HTTP shell for the fake MCP server: JSON-RPC 2.0 over a single ``POST /mcp`` endpoint plus
``GET /healthz`` (the readiness probe an in-process embedder polls before wiring a client).

Mirrors ``web_gateway/http.py``: the handler owns the transport (envelope, session id, auth,
error mapping) and delegates logic to ``FakeMcpServer``.
"""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.parse import urlparse

from monoid_agent_kernel.errors import PermissionDenied
from monoid_agent_kernel.reference._shared.http_util import (
    HardenedThreadingHTTPServer,
    HttpRequestTooLarge,
    log_http_request,
    read_json_limited,
    redact_internal_error,
)
from monoid_agent_kernel.reference.mcp_gateway.service import FakeMcpError, FakeMcpServer

_LOGGER = logging.getLogger("monoid_agent_kernel.mcp_gateway.http")

# JSON-RPC error code for an unknown method (mirrors the spec / test stub).
_METHOD_NOT_FOUND = -32601


def make_mcp_handler(
    server: FakeMcpServer,
    *,
    admin_token: str | None,
) -> type[BaseHTTPRequestHandler]:
    class McpGatewayHttpHandler(BaseHTTPRequestHandler):
        server_version = "MonoidMcpGateway/0.1"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/healthz":
                    self._write_json({"ok": True})
                    return
                self._write_error(HTTPStatus.NOT_FOUND, "not found")
            except Exception as exc:  # noqa: BLE001 - mapped to a client-safe response
                self._write_exception(exc)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path != "/mcp":
                    self._write_error(HTTPStatus.NOT_FOUND, "not found")
                    return
                self._require_auth()
                message = read_json_limited(self)
                self._dispatch(message)
            except Exception as exc:  # noqa: BLE001 - mapped to a client-safe response
                self._write_exception(exc)

        # -- JSON-RPC dispatch ----------------------------------------------------------
        def _dispatch(self, message: dict[str, Any]) -> None:
            method = message.get("method")
            rid = message.get("id")
            if method == "notifications/initialized":
                self._write_empty(HTTPStatus.ACCEPTED)  # a notification has no response body
                return
            if method == "initialize":
                self._write_result(rid, server.initialize(), session_id=server.session_id)
                return
            if method == "tools/list":
                self._write_result(rid, server.list_tools())
                return
            if method == "tools/call":
                params = message.get("params") or {}
                try:
                    result = server.call_tool(params.get("name"), params.get("arguments"))
                except FakeMcpError as exc:
                    self._write_jsonrpc_error(rid, exc.code, exc.message)
                    return
                self._write_result(rid, result)
                return
            if method == "resources/list":
                self._write_result(rid, server.list_resources())
                return
            if method == "resources/read":
                params = message.get("params") or {}
                try:
                    result = server.read_resource(str(params.get("uri") or ""))
                except FakeMcpError as exc:
                    self._write_jsonrpc_error(rid, exc.code, exc.message)
                    return
                self._write_result(rid, result)
                return
            if method == "prompts/list":
                self._write_result(rid, server.list_prompts())
                return
            if method == "prompts/get":
                params = message.get("params") or {}
                try:
                    result = server.get_prompt(str(params.get("name") or ""), params.get("arguments"))
                except FakeMcpError as exc:
                    self._write_jsonrpc_error(rid, exc.code, exc.message)
                    return
                self._write_result(rid, result)
                return
            self._write_jsonrpc_error(rid, _METHOD_NOT_FOUND, f"method not found: {method}")

        # -- auth + logging -------------------------------------------------------------
        def _require_auth(self) -> None:
            if admin_token is None:
                return
            header = self.headers.get("Authorization") or ""
            prefix = "Bearer "
            token = header[len(prefix):].strip() if header.startswith(prefix) else ""
            if token != admin_token:
                raise PermissionDenied("invalid or missing MCP gateway token")

        def log_request(self, code: Any = "-", size: Any = "-") -> None:  # noqa: ARG002
            log_http_request(_LOGGER, self, code)

        def log_message(self, _format: str, *_args: Any) -> None:
            return None

        # -- response writers -----------------------------------------------------------
        def _write_result(self, rid: Any, result: dict[str, Any], *, session_id: str | None = None) -> None:
            self._write_json({"jsonrpc": "2.0", "id": rid, "result": result}, session_id=session_id)

        def _write_jsonrpc_error(self, rid: Any, code: int, message: str) -> None:
            # A JSON-RPC error rides a 200 HTTP response (the error lives in the body); the
            # client raises McpError off the "error" field.
            self._write_json({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}})

        def _write_exception(self, exc: Exception) -> None:
            if isinstance(exc, PermissionDenied):
                self._write_error(HTTPStatus.UNAUTHORIZED, str(exc), error_code="mcp_auth_error")
            elif isinstance(exc, HttpRequestTooLarge):
                self._write_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, str(exc), error_code="mcp_bad_request")
            elif isinstance(exc, ValueError):
                self._write_error(HTTPStatus.BAD_REQUEST, str(exc), error_code="mcp_bad_request")
            else:
                self._write_error(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    redact_internal_error(_LOGGER, self, exc),
                    error_code="mcp_server_error",
                )

        def _write_error(self, status: HTTPStatus, message: str, *, error_code: str = "mcp_gateway_error") -> None:
            self._write_json(
                {"error": message, "error_code": error_code, "http_status": int(status)},
                status=status,
            )

        def _write_empty(self, status: HTTPStatus) -> None:
            self.send_response(int(status))
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _write_json(
            self,
            payload: dict[str, Any],
            *,
            status: HTTPStatus = HTTPStatus.OK,
            session_id: str | None = None,
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            if session_id:
                self.send_header("Mcp-Session-Id", session_id)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return McpGatewayHttpHandler


def create_mcp_server(
    server: FakeMcpServer,
    *,
    host: str,
    port: int,
    admin_token: str | None = None,
) -> HardenedThreadingHTTPServer:
    return HardenedThreadingHTTPServer((host, port), make_mcp_handler(server, admin_token=admin_token))
