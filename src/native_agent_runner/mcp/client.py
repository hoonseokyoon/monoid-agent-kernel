"""A minimal, hand-rolled MCP (Model Context Protocol) client over Streamable HTTP.

Scope is deliberately small — the ``tools/list`` + ``tools/call`` slice with a static bearer
token — which the MCP spec (rev 2025-06-18) explicitly lets a server answer with a plain
``application/json`` body, so no SSE parser is needed on the happy path. This avoids the
official ``mcp`` SDK's heavy, async-only dependency tree (starlette/uvicorn/pydantic/pyjwt)
for a JSON-RPC handshake that is ~150 lines over the ``httpx`` we already ship as an extra.

stdio transport, OAuth flows, resources/prompts/sampling, and ``list_changed`` notifications
are out of scope (add the official SDK behind the ``[mcp]`` extra later if needed).
"""

from __future__ import annotations

import itertools
import json
import threading
from typing import Any

PROTOCOL_VERSION = "2025-06-18"


class McpError(Exception):
    """An MCP protocol-level failure (JSON-RPC error, transport error, or missing dependency).

    Note: a *tool* execution error (``CallToolResult.isError``) is NOT this — that is a normal
    result the caller maps to a failed tool observation so the model can self-correct.
    """

    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class McpHttpClient:
    """Synchronous MCP client for Streamable HTTP. Safe to share across threads: ``httpx.Client``
    is thread-safe and the lock guards only (re)initialization."""

    def __init__(
        self,
        url: str,
        token: str | None = None,
        *,
        client_info: tuple[str, str] = ("native-agent-runner", "0.11"),
        timeout_s: float = 30.0,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise McpError("httpx is required for MCP; install native-agent-runner[mcp]") from exc
        self._httpx = httpx
        headers = {
            # The client MUST list both content types it can accept (spec: transports).
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._url = url
        self._http = httpx.Client(headers=headers, timeout=timeout_s)
        self._client_name, self._client_version = client_info
        self._ids = itertools.count(1)
        self._session_id: str | None = None
        self._initialized = False
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------------------

    def initialize(self) -> dict[str, Any]:
        """Handshake: ``initialize`` then the ``notifications/initialized`` notification.
        Captures the server-assigned ``Mcp-Session-Id`` for subsequent requests. Idempotent."""
        with self._lock:
            if self._initialized:
                return {}
            result, response = self._post(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},  # a tools-only client advertises no capabilities
                    "clientInfo": {"name": self._client_name, "version": self._client_version},
                },
            )
            session_id = response.headers.get("Mcp-Session-Id") or response.headers.get("mcp-session-id")
            if session_id:
                self._session_id = session_id
            self._post("notifications/initialized", None, notify=True)
            self._initialized = True
            return result

    def close(self) -> None:
        self._http.close()

    # -- operations --------------------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        """Return every tool the server exposes, following ``nextCursor`` pagination (spec rev
        2025-06-18). A large server splits ``tools/list`` across pages; reading only the first
        would silently drop tools the caller declared bindings for."""
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        seen: set[str] = set()
        restarted = False
        for _ in range(1000):  # bound against a server that never stops handing out cursors
            params = {"cursor": cursor} if cursor is not None else None
            # A cursor is opaque state the *current* session issued; a server that scopes cursors
            # to sessions will reject one minted by an expired session. So for cursor-bearing
            # pages we disable _post's transparent replay (which would resend the stale cursor)
            # and instead restart pagination from the top once after reconnecting.
            try:
                result, _ = self._post("tools/list", params, _allow_reconnect=cursor is None)
            except McpError as exc:
                if exc.code == 404 and cursor is not None and not restarted:
                    restarted = True
                    self.initialize()  # re-handshake (the prior _post already dropped the session)
                    tools.clear()
                    seen.clear()
                    cursor = None
                    continue
                raise
            page = result.get("tools")
            if isinstance(page, list):
                tools.extend(page)
            cursor = result.get("nextCursor")
            if not cursor or not isinstance(cursor, str) or cursor in seen:
                break  # done, or a malformed/repeating cursor — stop rather than loop forever
            seen.add(cursor)
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        result, _ = self._post("tools/call", {"name": name, "arguments": arguments or {}})
        return result

    # -- transport ---------------------------------------------------------------------

    def _post(
        self, method: str, params: Any = None, *, notify: bool = False, _allow_reconnect: bool = True
    ) -> tuple[dict[str, Any], Any]:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if not notify:
            message["id"] = next(self._ids)
        if params is not None:
            message["params"] = params
        headers: dict[str, str] = {}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        if method != "initialize":
            # Required on every post-init HTTP request (spec rev 2025-06-18).
            headers["MCP-Protocol-Version"] = PROTOCOL_VERSION
        try:
            response = self._http.post(self._url, json=message, headers=headers)
        except self._httpx.HTTPError as exc:
            raise McpError(f"MCP request failed: {exc}") from exc
        if response.status_code == 404 and self._session_id is not None:
            # The session expired. Drop the stale session and, for a normal operation, reconnect
            # once and retry transparently — a benign expiry shouldn't surface as a hard error.
            self._session_id = None
            self._initialized = False
            handshake = method in ("initialize", "notifications/initialized")
            if _allow_reconnect and not handshake:
                self.initialize()  # re-handshakes under _lock; idempotent
                return self._post(method, params, notify=notify, _allow_reconnect=False)
            raise McpError("MCP session expired (HTTP 404); reconnect required", code=404)
        if response.status_code >= 400:
            raise McpError(f"MCP returned HTTP {response.status_code}: {response.text[:200]}")
        if notify:
            return {}, response
        return self._parse_result(response, message["id"]), response

    def _parse_result(self, response: Any, request_id: int) -> dict[str, Any]:
        content_type = response.headers.get("content-type", "")
        if content_type.startswith("text/event-stream"):
            data = _read_sse_until(response, request_id)
        else:
            try:
                data = response.json()
            except json.JSONDecodeError as exc:
                raise McpError("MCP returned invalid JSON") from exc
        if isinstance(data, dict) and "error" in data:
            error = data["error"] or {}
            raise McpError(str(error.get("message") or "MCP error"), code=error.get("code"))
        result = data.get("result") if isinstance(data, dict) else None
        return result if isinstance(result, dict) else {}


def _read_sse_until(response: Any, request_id: int) -> dict[str, Any]:
    """Defensive fallback: read SSE ``data:`` frames until the JSON-RPC object whose ``id``
    matches arrives. Most servers answer tools/list+tools/call with plain JSON, so this is
    rarely hit; it handles a server that chooses to stream the single response."""
    data_lines: list[str] = []
    for raw in response.iter_lines():
        line = raw if isinstance(raw, str) else raw.decode("utf-8")
        if line == "":
            if data_lines:
                message = json.loads("\n".join(data_lines))
                data_lines = []
                if isinstance(message, dict) and message.get("id") == request_id:
                    return message
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip(" "))
    if data_lines:
        message = json.loads("\n".join(data_lines))
        if isinstance(message, dict):
            return message
    raise McpError("MCP stream ended without a matching response")
