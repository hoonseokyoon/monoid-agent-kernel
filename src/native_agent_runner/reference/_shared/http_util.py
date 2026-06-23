"""Shared hardening helpers for the reference HTTP gateways.

The backend, llm-gateway and web-gateway HTTP layers are thin ``BaseHTTPRequestHandler``
shells with identical request parsing and serving. These helpers centralize the
production-hardening concerns — bounded request size, per-connection timeouts,
internal-error redaction (no stack traces to clients), and structured request logging —
so all three layers harden in one place.
"""

from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.request import Request, urlopen

# Reject a declared request body larger than this outright (DoS / OOM guard). 10 MB
# comfortably covers a by-value conversation turn while bounding a single request's cost.
MAX_REQUEST_BYTES = 10_000_000
# Per-connection socket timeout so a slow or stalled client cannot hold a worker thread
# (and thus a thread-pool slot) open indefinitely.
REQUEST_TIMEOUT_S = 30.0


class HttpRequestTooLarge(Exception):
    """The request body's declared Content-Length exceeds ``MAX_REQUEST_BYTES``."""


def read_json_limited(handler: BaseHTTPRequestHandler, *, max_bytes: int = MAX_REQUEST_BYTES) -> dict[str, Any]:
    """Read a JSON object body, rejecting an over-large declared Content-Length before any
    bytes are read. Returns ``{}`` for an empty body. Raises ``HttpRequestTooLarge`` (-> 413)
    or ``ValueError`` (-> 400) on a malformed body."""
    length = int(handler.headers.get("Content-Length") or "0")
    if length > max_bytes:
        # The declared body is rejected WITHOUT reading it (the OOM guard). That leaves the
        # client's already-sent bytes unconsumed, so the connection cannot be safely reused —
        # close it after the 413 rather than attempting keep-alive (an unconsumed body would
        # also otherwise be misread as the next request, and the close races a TCP reset).
        handler.close_connection = True
        raise HttpRequestTooLarge(f"request body exceeds the {max_bytes}-byte limit")
    if length <= 0:
        return {}
    try:
        payload = json.loads(handler.rfile.read(length).decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON request body") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON request body must be an object")
    return payload


def redact_internal_error(logger: Any, handler: BaseHTTPRequestHandler, exc: Exception) -> str:
    """Log an unexpected (5xx) exception in full server-side under a fresh correlation id and
    return a generic client-facing message carrying only that id — never the exception text,
    stack trace, or internal paths."""
    correlation_id = uuid.uuid4().hex
    logger.error(
        "unhandled error [%s] %s %s: %r",
        correlation_id,
        getattr(handler, "command", "?"),
        getattr(handler, "path", "?"),
        exc,
        exc_info=exc,
    )
    return f"internal server error (ref {correlation_id})"


def wait_http_ready(base_url: str, *, timeout_s: float = 15.0) -> None:
    """Poll ``<base_url>/healthz`` until the server answers, or raise ``TimeoutError``. The
    runtime counterpart of the test harness's poll — an embedder that boots an auxiliary HTTP
    server in-process (e.g. studio's fake MCP gateway) must wait for it to serve before wiring a
    client that discovers against it."""
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(Request(f"{base_url}/healthz"), timeout=2) as response:
                response.read()
            return
        except Exception as exc:  # noqa: BLE001 - any failure means not-yet-ready
            last_error = exc
            time.sleep(0.02)
    raise TimeoutError(f"server did not become ready: {last_error}")


def log_http_request(logger: Any, handler: BaseHTTPRequestHandler, code: Any) -> None:
    """Structured access log for one request (method, path, status)."""
    logger.info("http %s %s -> %s", getattr(handler, "command", "?"), getattr(handler, "path", "?"), code)


class HardenedThreadingHTTPServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` hardened for clean shutdown under load.

    A per-connection socket timeout stops a slow client from pinning a request thread open.
    Request threads are non-daemon with ``block_on_close``, so ``server_close()`` joins any
    in-flight handler instead of abandoning it — abandoned daemon handlers racing a closing
    listen socket are what surface as ``ConnectionAborted`` / "I/O on closed file" errors.
    The socket timeout bounds that join so it can never hang."""

    daemon_threads = False
    block_on_close = True
    request_timeout_s: float = REQUEST_TIMEOUT_S

    def finish_request(self, request: Any, client_address: Any) -> None:
        try:
            request.settimeout(self.request_timeout_s)
        except OSError:  # pragma: no cover - platform without settable timeout
            pass
        super().finish_request(request, client_address)
