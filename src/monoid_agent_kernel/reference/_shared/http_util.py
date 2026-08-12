"""Shared hardening helpers for the reference HTTP gateways.

The backend, llm-gateway and web-gateway HTTP layers are thin ``BaseHTTPRequestHandler``
shells with identical request parsing and serving. These helpers centralize the
production-hardening concerns — bounded request size, per-connection timeouts,
internal-error redaction (no stack traces to clients), and structured request logging —
so all three layers harden in one place.
"""

from __future__ import annotations

import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.request import Request, urlopen

from monoid_agent_kernel.core.json_ingress import loads_json_ingress

# Reject a declared request body larger than this outright (DoS / OOM guard). 10 MB
# comfortably covers a by-value conversation turn while bounding a single request's cost.
MAX_REQUEST_BYTES = 10_000_000
# Per-connection socket timeout so a slow or stalled client cannot hold a worker thread
# (and thus a thread-pool slot) open indefinitely.
REQUEST_TIMEOUT_S = 30.0
# How much of an unread body a REJECTED request will consume so its response survives the close
# (see :func:`drain_request_body`). Deliberately far below ``MAX_REQUEST_BYTES``: this runs on
# the path where the request was refused -- often before authentication -- so it must not become
# a way to make the server read megabytes on behalf of a caller it has already turned away. 64 KiB
# covers any ordinary request that gets rejected early; a body larger than this keeps exactly
# today's behavior, which is that the client may see a reset instead of the status.
MAX_DRAIN_BYTES = 64 * 1024
# And the wait is capped far below ``REQUEST_TIMEOUT_S`` for the same reason: a client that has
# already sent its body is drained instantly, while one that declares a body and dribbles it holds
# a handler thread for half a second rather than thirty -- on a request that was already refused.
DRAIN_TIMEOUT_S = 0.5
# Records WHICH request's body has left the socket, so a drain never double-reads (a second read
# would block for the whole timeout, turning a lost error response into a hang). The value is the
# request's own headers object rather than a boolean: ``handle_one_request`` re-parses headers per
# request, so the marker invalidates itself on a keep-alive connection with nothing to reset. A
# boolean would need every handler to clear it per request -- four more sites that can drift, and
# a stale one silently skips the drain for every request after the first on that connection.
_BODY_READ_ATTR = "_monoid_request_body_read_for"


class HttpRequestTooLarge(Exception):
    """The request body's declared Content-Length exceeds ``MAX_REQUEST_BYTES``."""


def mark_request_body_read(handler: BaseHTTPRequestHandler) -> None:
    """Record that THIS request's body is no longer sitting unread in the socket."""

    setattr(handler, _BODY_READ_ATTR, handler.headers)


def drain_request_body(
    handler: BaseHTTPRequestHandler, *, max_bytes: int = MAX_DRAIN_BYTES
) -> None:
    """Consume an unread request body so the response about to be written actually arrives.

    A handler that answers before reading the body leaves the client's already-sent bytes in the
    kernel receive buffer. Closing a socket in that state sends an RST rather than a FIN on
    Windows, and the RST discards whatever the client has not yet pulled out of ITS buffer -- so a
    response that was written and flushed successfully never reaches the reader, which sees
    ``ConnectionAbortedError`` instead. Every early rejection has this shape: an unknown path, a
    missing or bad token, an over-large body.

    What that costs is not a lost message but a RECLASSIFIED one. ``gateway_auth_error`` says
    ``retryable: false``; a transport abort says "transient" to every retry policy in this repo, so
    a credential failure gets retried as though it might succeed next time.

    Why it is intermittent without a deliberate probe: the handler's header read is buffered
    (``rfile``, 8 KiB), so a body that shares the segment its headers arrived in is swallowed by
    that read and the socket is left clean. ``http.client`` writes headers and body separately, so
    they often land separately -- and a hand-written probe using one ``sendall`` usually does not,
    which is how this reads as "cannot reproduce" when measured the obvious way.

    Bounded in BOTH directions, because this runs for a caller the server has already refused:
    ``max_bytes`` caps the size and :data:`DRAIN_TIMEOUT_S` caps the wait, well under the
    connection's own 30-second timeout. Exceeding either bound closes the connection undrained,
    which is what ``read_json_limited``'s 413 branch already does and is no worse than the
    behavior this replaces -- a partial drain buys nothing, since any byte left unread resets the
    connection just as surely as all of them.

    Idempotent, and safe to put on a shared write path that also serves responses issued *after*
    the body was parsed: the first read of a request marks it and every later call is a no-op.

    Generalized from ``BackendHttpHandler._discard_unread_request_body``, which had this fix --
    and this analysis -- for one of the four reference servers. The other three wrote their
    rejections into the same reset for as long as it existed. One function, four callers.

    Cannot raise, and that is a requirement rather than defensiveness: this runs as the first
    statement of every ``_write_error``, so an exception escaping here would replace the error
    response with no response at all -- strictly worse than the reset it exists to prevent, and on
    the same path. A handler without ``headers`` (nothing was parsed) or without a ``connection``
    to bound the read against is therefore skipped rather than read from: an unbounded read on the
    error path is the one thing worse than not draining.
    """

    headers = getattr(handler, "headers", None)
    connection = getattr(handler, "connection", None)
    if headers is None or connection is None:
        return
    if getattr(handler, _BODY_READ_ATTR, None) is headers:
        return
    # Marked before the read, not after: whatever happens below (a short body, a timeout, a
    # client that vanished), this request's body is never worth reading a second time.
    mark_request_body_read(handler)
    try:
        length = int(headers.get("Content-Length") or "0")
    except (TypeError, ValueError):
        handler.close_connection = True  # nothing can be known about where the body ends
        return
    if length <= 0:
        return
    if length > max_bytes:
        handler.close_connection = True
        return
    previous_timeout = None
    try:
        previous_timeout = connection.gettimeout()
        connection.settimeout(DRAIN_TIMEOUT_S)
        handler.rfile.read(length)
    except OSError:
        # Includes the drain timeout: a body that has not arrived is not worth waiting for, and
        # the close is exactly what would have happened without the drain.
        handler.close_connection = True
    finally:
        if previous_timeout is not None:
            try:
                connection.settimeout(previous_timeout)
            except OSError:  # pragma: no cover - socket already torn down
                pass


def read_json_limited(
    handler: BaseHTTPRequestHandler, *, max_bytes: int = MAX_REQUEST_BYTES
) -> dict[str, Any]:
    """Read a JSON object body, rejecting an over-large declared Content-Length before any
    bytes are read. Returns ``{}`` for an empty body. Raises ``HttpRequestTooLarge`` (-> 413)
    or ``ValueError`` (-> 400) on a malformed body.

    Every path that leaves the socket with nothing left to read marks the request through
    :func:`mark_request_body_read`, so the drain on the error-writing path knows not to read
    again. The over-large path deliberately does NOT mark: it refused without reading, so the body
    is still there and the bounded drain is exactly what should run for it."""
    try:
        length = int(handler.headers.get("Content-Length") or "0")
    except (TypeError, ValueError) as exc:
        handler.close_connection = True
        raise ValueError("invalid Content-Length") from exc
    if length < 0:
        handler.close_connection = True
        raise ValueError("Content-Length must be non-negative")
    if length > max_bytes:
        # The declared body is rejected WITHOUT reading it (the OOM guard). That leaves the
        # client's already-sent bytes unconsumed, so the connection cannot be safely reused —
        # close it after the 413 rather than attempting keep-alive (an unconsumed body would
        # also otherwise be misread as the next request, and the close races a TCP reset).
        # The reset half of that sentence is what ``drain_request_body`` now answers: the 413 is
        # written through ``_write_error``, which drains up to ``MAX_DRAIN_BYTES`` first, so a
        # body that overshoots the limit by a little still gets its status back. One that
        # overshoots by a lot does not, and that is the bound doing its job -- which is also
        # exactly why the close below STAYS: past the drain cap the body is still unread, and a
        # reused connection would read the remainder of it as the next request line.
        handler.close_connection = True
        raise HttpRequestTooLarge(f"request body exceeds the {max_bytes}-byte limit")
    if length == 0:
        mark_request_body_read(handler)
        return {}
    # Marked around the read itself, so a body that was consumed and then failed to decode is
    # still known to be gone: the 400 that follows goes through the same draining writer.
    try:
        raw = handler.rfile.read(length)
    finally:
        mark_request_body_read(handler)
    try:
        payload = loads_json_ingress(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
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
    logger.info(
        "http %s %s -> %s", getattr(handler, "command", "?"), getattr(handler, "path", "?"), code
    )


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
