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
    drain_request_body,
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
            # What the failing call already cost, read once and carried by EVERY arm rather than
            # by the classified one alone. The stamp does not belong to a type: the adapter that
            # sees the provider's billed body first refuses in raw ``ValueError``/``AttributeError``
            # as readily as in ``ModelAdapterError`` (``normalize_usage``, and the whole terminal
            # region of the OpenAI stream), and those land on the ``ValueError`` and the generic
            # arms below -- which wrote no ``usage`` at all, so the client one hop out recorded
            # zero for a turn the upstream generated and billed. ``_error_body`` omits the key
            # when it is empty, so the arms that never carry a cost keep their exact wire shape.
            usage = provider_usage_of(exc)
            # Its sibling fact, read the same way and in the same place, because it belongs to
            # a type no more than the cost does. ``mark_provider_retried`` stamps an arbitrary
            # ``BaseException`` -- the gateway client's own ``_stamp_retry`` documents exactly
            # that -- so an upstream adapter that retried and then refused in a raw
            # ``ValueError``/``AttributeError`` is carrying the flag on every arm below, and
            # only the ``ModelAdapterError`` arm passed it. The receipt one hop out then
            # recorded a clean single attempt for a call that had demonstrably retried, on the
            # one carrier a failed call leaves behind -- while the cost of those same attempts
            # rode the same body. Read leniently, like the stamp's own reader: a type that
            # refuses the attribute reports no retry rather than crashing this writer.
            retried = bool(getattr(exc, "provider_retried", False))
            if isinstance(exc, PermissionDenied):
                self._write_error(
                    HTTPStatus.UNAUTHORIZED,
                    str(exc),
                    error_code=GATEWAY_AUTH_ERROR,
                    retryable=False,
                    provider_retried=retried,
                    usage=usage,
                )
            elif isinstance(exc, ModelAdapterError):
                status = _model_error_status(exc)
                self._write_error(
                    status,
                    str(exc),
                    error_code=exc.provider_error_code or GATEWAY_BAD_RESPONSE,
                    retryable=exc.retryable,
                    config_recoverable=exc.config_recoverable,
                    provider_retried=retried,
                    usage=usage,
                )
            elif isinstance(exc, HttpRequestTooLarge):
                self._write_error(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    str(exc),
                    error_code=GATEWAY_BAD_REQUEST,
                    retryable=False,
                    provider_retried=retried,
                    usage=usage,
                )
            elif isinstance(exc, ValueError):
                self._write_error(
                    HTTPStatus.BAD_REQUEST,
                    str(exc),
                    error_code=GATEWAY_BAD_REQUEST,
                    retryable=False,
                    provider_retried=retried,
                    usage=usage,
                )
            elif isinstance(exc, NativeAgentError):
                self._write_error(
                    HTTPStatus.BAD_REQUEST,
                    str(exc),
                    error_code=getattr(exc, "error_code", GATEWAY_BAD_REQUEST),
                    retryable=False,
                    provider_retried=retried,
                    usage=usage,
                )
            else:
                self._write_error(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    redact_internal_error(_LOGGER, self, exc),
                    error_code=GATEWAY_SERVER_ERROR,
                    retryable=True,
                    provider_retried=retried,
                    usage=usage,
                )

        def _write_error(
            self,
            status: HTTPStatus,
            message: str,
            *,
            error_code: str = GATEWAY_BAD_RESPONSE,
            retryable: bool = False,
            config_recoverable: bool = False,
            provider_retried: bool = False,
            usage: Mapping[str, int] | None = None,
        ) -> None:
            # Before the status, not after: the bytes have to leave the receive buffer before the
            # close, and the close follows this write immediately. This wire is the one where the
            # loss is worst -- every field below (``retryable``, ``config_recoverable``, the
            # provider code, the billed ``usage``) is a classification the client acts on, and a
            # reset replaces all of it with "network error", which reads as retryable.
            drain_request_body(self)
            self._write_json(
                _error_body(
                    status,
                    message,
                    error_code=error_code,
                    retryable=retryable,
                    config_recoverable=config_recoverable,
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
            #
            # ``ensure_ascii=True`` here and NOT in ``_write_json`` above, because "single-line"
            # is the whole framing on this route and the two ends disagree about what a line is.
            # U+2028 LINE SEPARATOR, U+2029 PARAGRAPH SEPARATOR and U+0085 NEXT LINE survive an
            # ``ensure_ascii=False`` dump as themselves, and the line-splitting readers clients
            # use -- httpx's ``aiter_lines``, which this repo's own GatewayModelAdapter reads
            # with -- break on all three. The client then parses a JSON object that stops
            # mid-string and reports a bad response for a turn this server produced, framed and
            # already metered. Model text can contain any of them (``final_text``, and the
            # relayed ``reasoning`` array, whose plaintext entries need never appear in the
            # answer at all), so escaping them is the frame writer's job, not the model's. The
            # length-delimited body has no such ambiguity and keeps its smaller encoding.
            self.wfile.write(
                b"data: "
                + json.dumps(frame, ensure_ascii=True, allow_nan=False).encode("utf-8")
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
    config_recoverable: bool = False,
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

    ``config_recoverable`` says the remedy is the caller's *configuration* rather than another
    attempt. It is the one classification the kernel mints that had no transport: the 4xx
    ``_model_error_status`` picks below is a hint a client has to interpret, and every non-status
    carrier of the same fact -- an applied-parameters proof refusal raised with no HTTP status at
    all -- had nothing left to say one hop out. Written unconditionally, like ``retryable`` and
    ``provider_retried``: a reader must not have to distinguish "not config-fixable" from "an
    older gateway that never mentions it", and both spellings mean the same False anyway.
    """

    body: dict[str, Any] = {
        "error": message,
        "error_code": error_code,
        "retryable": retryable,
        "config_recoverable": config_recoverable,
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
    # Both facts read once above the branch, exactly as ``_write_exception`` reads them and for
    # the same reason: neither belongs to a type. See that writer for why the retry flag is
    # readable on a raw refusal, and read leniently there too.
    usage = provider_usage_of(exc)
    retried = bool(getattr(exc, "provider_retried", False))
    if isinstance(exc, ModelAdapterError):
        return {
            "type": "error",
            **_error_body(
                _model_error_status(exc),
                str(exc),
                error_code=exc.provider_error_code or GATEWAY_BAD_RESPONSE,
                retryable=exc.retryable,
                config_recoverable=exc.config_recoverable,
                provider_retried=retried,
                usage=usage,
            ),
        }
    return {
        "type": "error",
        # The unclassified arm carries the cost too, exactly like its twin above and like
        # ``_write_exception``'s. A stream that folds provider deltas and then refuses its own
        # end-of-turn payload fails with a RAW ``ValueError``/``AttributeError`` -- the one shape
        # this arm exists for -- and that is a refusal of a turn the upstream already generated
        # and billed. Without the key, the only carrier a streaming client has says the call was
        # free, and ``retryable=True`` below then invites it to buy the same tokens again. The
        # attempts that cost it ride the same body, for the same reason.
        **_error_body(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            redact_internal_error(_LOGGER, handler, exc),
            error_code=GATEWAY_SERVER_ERROR,
            retryable=True,
            provider_retried=retried,
            usage=usage,
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
