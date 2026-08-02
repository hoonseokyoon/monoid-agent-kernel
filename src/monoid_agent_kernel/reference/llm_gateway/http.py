from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.parse import urlparse

from monoid_agent_kernel.errors import ModelAdapterError, NativeAgentError, PermissionDenied
from monoid_agent_kernel.reference._shared.http_util import (
    HardenedThreadingHTTPServer,
    HttpRequestTooLarge,
    log_http_request,
    read_json_limited,
    redact_internal_error,
)
from monoid_agent_kernel.providers.base import provider_usage_of
from monoid_agent_kernel.reference.llm_gateway.service import LlmGatewayBackend
from monoid_agent_kernel.providers.gateway import (
    GATEWAY_AUTH_ERROR,
    GATEWAY_BAD_REQUEST,
    GATEWAY_BAD_RESPONSE,
    GATEWAY_SERVER_ERROR,
)

_LOGGER = logging.getLogger("monoid_agent_kernel.llm_gateway.http")


def make_llm_gateway_handler(
    gateway: LlmGatewayBackend,
    *,
    admin_token: str | None,
) -> type[BaseHTTPRequestHandler]:
    class LlmGatewayHttpHandler(BaseHTTPRequestHandler):
        server_version = "MonoidLlmGateway/0.2"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/healthz":
                    self._write_json({"ok": True})
                    return
                parts = [part for part in parsed.path.split("/") if part]
                if (
                    len(parts) == 5
                    and parts[:3] == ["internal", "llm", "tenants"]
                    and parts[4] == "usage"
                ):
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
                if parsed.path == "/internal/llm/turns/stream":
                    # Auth/parse/build run eagerly inside handle_turn_stream and may raise
                    # here, before any SSE byte — those map to a normal error response below.
                    frames = gateway.handle_turn_stream(self._bearer_token(), self._read_json())
                    self._write_sse(frames)
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
                    provider_retried=exc.provider_retried,
                    usage=provider_usage_of(exc),
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
            provider_retried: bool = False,
            usage: Mapping[str, int] | None = None,
        ) -> None:
            self._write_json(
                _error_body(
                    status,
                    message,
                    error_code=error_code,
                    retryable=retryable,
                    provider_retried=provider_retried,
                    usage=usage,
                ),
                status=status,
            )

        def _write_json(
            self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _write_sse(self, frames: Iterable[dict[str, Any]]) -> None:
            # A streaming turn is expected to be long-lived, so clear this route's 30s socket
            # timeout (which would otherwise kill the connection on a >30s gap between tokens).
            # The provider call's own timeout still bounds a wedged upstream. Other routes keep
            # the default. No Content-Length: the body is unbounded and ends on connection close.
            self.connection.settimeout(None)
            self.send_response(int(HTTPStatus.OK))
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for frame in frames:
                    self._write_sse_frame(frame)
            except Exception as exc:
                # We are already committed to a 200 SSE body, so a mid-stream failure surfaces
                # as a terminal error frame (the client raises ModelAdapterError from it).
                self._write_sse_frame(_stream_error_frame(self, exc))

        def _write_sse_frame(self, frame: dict[str, Any]) -> None:
            # Single-line JSON (no indent), flushed per frame so the stream is live.
            self.wfile.write(
                b"data: "
                + json.dumps(frame, ensure_ascii=False, allow_nan=False).encode("utf-8")
                + b"\n\n"
            )
            self.wfile.flush()

    return LlmGatewayHttpHandler


def _error_body(
    status: HTTPStatus,
    message: str,
    *,
    error_code: str = GATEWAY_BAD_RESPONSE,
    retryable: bool = False,
    provider_retried: bool = False,
    usage: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """The fields every gateway error carries, whatever transport reports it.

    One definition rather than one per writer. The non-200 body and the SSE error frame are read
    back by the same client code but were written separately, so every field added to one had to be
    remembered for the other -- ``provider_retried`` was added to both in the same commit only
    because they were reviewed together. One definition removes the chance to forget.

    ``usage`` is what the failing call already cost. Some failures happen *after* a complete,
    billed answer -- an applied-parameters refusal raised by an upstream that is itself a
    gateway is exactly that -- and without carrying it the hop turned a paid call into a
    zero-token one for every client behind it: the outer receipt, the run's token budget, and
    the metrics all under-counted. Omitted when empty, which is what an error raised before
    reaching a provider means.

    ``provider_retried`` is a retry the gateway's *backend* made before failing. The client can only
    see its own attempts, so without it a call the provider retried and then failed was recorded as
    a clean single attempt -- and a failure is where that record matters most. It defaults False,
    which is what an error the gateway raised on its own, before reaching a provider, means.
    """

    body: dict[str, Any] = {
        "error": message,
        "error_code": error_code,
        "retryable": retryable,
        "http_status": int(status),
        "provider_retried": provider_retried,
    }
    # Present only when the failing call actually burned tokens, so an error the gateway
    # raised on its own keeps its exact previous wire shape.
    if usage:
        body["usage"] = dict(usage)
    return body


def _stream_error_frame(handler: BaseHTTPRequestHandler, exc: Exception) -> dict[str, Any]:
    """Mid-stream error as an SSE frame, carrying ``_error_body``'s fields so the client maps it
    back to a ModelAdapterError identically to a non-200 response."""
    if isinstance(exc, ModelAdapterError):
        return {
            "type": "error",
            **_error_body(
                _model_error_status(exc),
                str(exc),
                error_code=exc.provider_error_code or GATEWAY_BAD_RESPONSE,
                retryable=exc.retryable,
                provider_retried=exc.provider_retried,
                usage=provider_usage_of(exc),
            ),
        }
    return {
        "type": "error",
        **_error_body(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            redact_internal_error(_LOGGER, handler, exc),
            error_code=GATEWAY_SERVER_ERROR,
            retryable=True,
        ),
    }


def _model_error_status(exc: ModelAdapterError) -> HTTPStatus:
    if exc.http_status is not None and 400 <= exc.http_status <= 599:
        try:
            return HTTPStatus(exc.http_status)
        except ValueError:
            pass
    if getattr(exc, "config_recoverable", False):
        # A config-fixable refusal (an applied-parameters proof failure from this gateway's
        # own upstream) must not be laundered into a 502 across the hop: the outer client's
        # classifier reads 5xx as terminal, so a condition the direct client survives
        # recoverably killed the run one hop out. 4xx is the same statement in HTTP.
        return HTTPStatus.UNPROCESSABLE_ENTITY
    return HTTPStatus.SERVICE_UNAVAILABLE if exc.retryable else HTTPStatus.BAD_GATEWAY


def create_llm_gateway_server(
    gateway: LlmGatewayBackend,
    *,
    host: str,
    port: int,
    admin_token: str,
) -> HardenedThreadingHTTPServer:
    return HardenedThreadingHTTPServer(
        (host, port), make_llm_gateway_handler(gateway, admin_token=admin_token)
    )
