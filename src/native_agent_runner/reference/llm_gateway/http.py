from __future__ import annotations

import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.parse import urlparse

from native_agent_runner.errors import ModelAdapterError, NativeAgentError, PermissionDenied
from native_agent_runner.reference._shared.http_util import (
    HardenedThreadingHTTPServer,
    HttpRequestTooLarge,
    log_http_request,
    read_json_limited,
    redact_internal_error,
)
from native_agent_runner.reference.llm_gateway.service import LlmGatewayBackend
from native_agent_runner.providers.gateway import (
    GATEWAY_AUTH_ERROR,
    GATEWAY_BAD_REQUEST,
    GATEWAY_BAD_RESPONSE,
    GATEWAY_SERVER_ERROR,
)

_LOGGER = logging.getLogger("native_agent_runner.llm_gateway.http")


def make_llm_gateway_handler(
    gateway: LlmGatewayBackend,
    *,
    admin_token: str | None,
) -> type[BaseHTTPRequestHandler]:
    class LlmGatewayHttpHandler(BaseHTTPRequestHandler):
        server_version = "NativeAgentRunnerLlmGateway/0.2"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/healthz":
                    self._write_json({"ok": True})
                    return
                parts = [part for part in parsed.path.split("/") if part]
                if len(parts) == 5 and parts[:3] == ["internal", "llm", "tenants"] and parts[4] == "usage":
                    self._require_admin()
                    self._write_json(gateway.tenant_usage(parts[3]))
                    return
                self._write_error(HTTPStatus.NOT_FOUND, "not found")
            except Exception as exc:
                self._write_exception(exc)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/internal/llm/turns":
                    self._write_json(gateway.handle_turn(self._bearer_token(), self._read_json()))
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
                self._write_error(
                    HTTPStatus.UNAUTHORIZED,
                    str(exc),
                    error_code=GATEWAY_AUTH_ERROR,
                    retryable=False,
                )
            elif isinstance(exc, ModelAdapterError):
                status = _model_error_status(exc)
                self._write_error(
                    status,
                    str(exc),
                    error_code=exc.provider_error_code or GATEWAY_BAD_RESPONSE,
                    retryable=exc.retryable,
                )
            elif isinstance(exc, HttpRequestTooLarge):
                self._write_error(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    str(exc),
                    error_code=GATEWAY_BAD_REQUEST,
                    retryable=False,
                )
            elif isinstance(exc, ValueError):
                self._write_error(
                    HTTPStatus.BAD_REQUEST,
                    str(exc),
                    error_code=GATEWAY_BAD_REQUEST,
                    retryable=False,
                )
            elif isinstance(exc, NativeAgentError):
                self._write_error(
                    HTTPStatus.BAD_REQUEST,
                    str(exc),
                    error_code=getattr(exc, "error_code", GATEWAY_BAD_REQUEST),
                    retryable=False,
                )
            else:
                self._write_error(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    redact_internal_error(_LOGGER, self, exc),
                    error_code=GATEWAY_SERVER_ERROR,
                    retryable=True,
                )

        def _write_error(
            self,
            status: HTTPStatus,
            message: str,
            *,
            error_code: str = GATEWAY_BAD_RESPONSE,
            retryable: bool = False,
        ) -> None:
            self._write_json(
                {
                    "error": message,
                    "error_code": error_code,
                    "retryable": retryable,
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

    return LlmGatewayHttpHandler


def _model_error_status(exc: ModelAdapterError) -> HTTPStatus:
    if exc.http_status is not None and 400 <= exc.http_status <= 599:
        try:
            return HTTPStatus(exc.http_status)
        except ValueError:
            pass
    return HTTPStatus.SERVICE_UNAVAILABLE if exc.retryable else HTTPStatus.BAD_GATEWAY


def create_llm_gateway_server(
    gateway: LlmGatewayBackend,
    *,
    host: str,
    port: int,
    admin_token: str,
) -> HardenedThreadingHTTPServer:
    return HardenedThreadingHTTPServer((host, port), make_llm_gateway_handler(gateway, admin_token=admin_token))
