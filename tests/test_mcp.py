"""MCP client: the McpToolProvider lists a server's tools and proxies calls. Exercised against
a local fake MCP server speaking JSON-RPC over HTTP — no real server, no `mcp` SDK, no key.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

from support.http import serving
from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core.spec import AgentRunSpec, RunLimits
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.core.context import TurnContext
from monoid_agent_kernel.mcp import McpToolProvider
from monoid_agent_kernel.mcp.client import McpHttpClient
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.reference._shared.http_util import HardenedThreadingHTTPServer

_TOOLS = [
    {
        "name": "echo",
        "description": "Echo the input text.",
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    },
    {"name": "boom", "description": "Always fails.", "inputSchema": {"type": "object"}, "annotations": {"readOnlyHint": True}},
]
_RESOURCES = [
    {
        "uri": "fake://one",
        "name": "one",
        "description": "First fake resource.",
        "mimeType": "text/plain",
    },
    {
        "uri": "fake://two",
        "name": "two",
        "description": "Second fake resource.",
        "mimeType": "text/plain",
    },
]
_PROMPTS = [
    {
        "name": "brief",
        "description": "Brief prompt.",
        "arguments": [{"name": "topic", "required": True}],
    },
    {"name": "plan", "description": "Plan prompt."},
]


def _mcp_handler() -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:
            return None

        def do_GET(self) -> None:  # noqa: N802 - health check for the serving() harness
            self._send(200, b'{"ok":true}', content_type="application/json")

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            message = json.loads(self.rfile.read(length) or b"{}")
            method = message.get("method")
            rid = message.get("id")
            if method == "notifications/initialized":
                self._send(202, b"")
                return
            if method == "initialize":
                self._result(rid, {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}, "serverInfo": {"name": "fake", "version": "1"}}, session_id="sess-1")
                return
            if method == "tools/list":
                self._result(rid, {"tools": _TOOLS})
                return
            if method == "tools/call":
                params = message.get("params") or {}
                name, args = params.get("name"), params.get("arguments") or {}
                if name == "echo":
                    self._result(rid, {"content": [{"type": "text", "text": f"echo: {args.get('text')}"}], "structuredContent": {"echoed": args.get("text")}})
                elif name == "boom":
                    self._result(rid, {"content": [{"type": "text", "text": "kaboom"}], "isError": True})
                else:
                    self._error(rid, -32602, f"Unknown tool: {name}")
                return
            self._error(rid, -32601, "method not found")

        def _result(self, rid: Any, result: dict[str, Any], *, session_id: str | None = None) -> None:
            body = json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}).encode("utf-8")
            self._send(200, body, content_type="application/json", session_id=session_id)

        def _error(self, rid: Any, code: int, msg: str) -> None:
            body = json.dumps({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}).encode("utf-8")
            self._send(200, body, content_type="application/json")

        def _send(self, status: int, body: bytes, *, content_type: str | None = None, session_id: str | None = None) -> None:
            self.send_response(status)
            if content_type:
                self.send_header("Content-Type", content_type)
            if session_id:
                self.send_header("Mcp-Session-Id", session_id)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _server() -> HardenedThreadingHTTPServer:
    return HardenedThreadingHTTPServer(("127.0.0.1", 0), _mcp_handler())


def test_mcp_provider_lists_and_proxies_tools() -> None:
    with serving(_server()) as base_url:
        with McpToolProvider(f"{base_url}/mcp", server="t") as mcp:
            specs = {s.id: s for s in mcp.get_tools()}
            assert set(specs) == {"mcp.t.echo", "mcp.t.boom"}
            echo = specs["mcp.t.echo"]
            # Prefixed exported name (avoids registry collisions); input schema passed through.
            assert echo.exported_name == "mcp_t_echo"
            assert echo.input_schema["required"] == ["text"]
            assert echo.side_effect == "run"
            assert specs["mcp.t.boom"].side_effect == "read"  # readOnlyHint

            ok = echo.handler(None, {"text": "hi"})
            assert ok.ok and ok.content["text"] == "echo: hi" and ok.content["structured"] == {"echoed": "hi"}

            failed = specs["mcp.t.boom"].handler(None, {})
            assert not failed.ok and failed.error_code == "mcp_tool_error" and "kaboom" in failed.error


def _paginated_handler() -> type[BaseHTTPRequestHandler]:
    """A server that splits tools/list across two pages keyed by ``cursor``/``nextCursor``."""
    page1 = [_TOOLS[0]]  # echo
    page2 = [_TOOLS[1]]  # boom

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:
            return None

        def do_GET(self) -> None:  # noqa: N802
            self._send(200, b'{"ok":true}')

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            message = json.loads(self.rfile.read(length) or b"{}")
            method, rid = message.get("method"), message.get("id")
            if method == "notifications/initialized":
                self._send(202, b"")
            elif method == "initialize":
                self._result(rid, {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}}, session_id="sess-1")
            elif method == "tools/list":
                cursor = (message.get("params") or {}).get("cursor")
                if cursor is None:
                    self._result(rid, {"tools": page1, "nextCursor": "c2"})
                elif cursor == "c2":
                    self._result(rid, {"tools": page2})  # last page: no nextCursor
                else:
                    self._error(rid, -32602, f"bad cursor: {cursor}")
            else:
                self._error(rid, -32601, "method not found")

        def _result(self, rid: Any, result: dict[str, Any], *, session_id: str | None = None) -> None:
            body = json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}).encode("utf-8")
            self._send(200, body, session_id=session_id)

        def _error(self, rid: Any, code: int, msg: str) -> None:
            body = json.dumps({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}).encode("utf-8")
            self._send(200, body)

        def _send(self, status: int, body: bytes, *, session_id: str | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if session_id:
                self.send_header("Mcp-Session-Id", session_id)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def test_mcp_list_tools_follows_pagination() -> None:
    # A server that paginates tools/list must not be truncated to page one.
    server = HardenedThreadingHTTPServer(("127.0.0.1", 0), _paginated_handler())
    with serving(server) as base_url:
        with McpToolProvider(f"{base_url}/mcp", server="t") as mcp:
            ids = {s.id for s in mcp.get_tools()}
            assert ids == {"mcp.t.echo", "mcp.t.boom"}  # both pages surfaced


def _paginated_resource_prompt_handler() -> type[BaseHTTPRequestHandler]:
    """A server that splits resources/list and prompts/list across two pages."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:
            return None

        def do_GET(self) -> None:  # noqa: N802
            self._send(200, b'{"ok":true}')

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            message = json.loads(self.rfile.read(length) or b"{}")
            method, rid = message.get("method"), message.get("id")
            if method == "notifications/initialized":
                self._send(202, b"")
            elif method == "initialize":
                self._result(rid, {"protocolVersion": "2025-06-18", "capabilities": {}}, session_id="sess-1")
            elif method == "resources/list":
                cursor = (message.get("params") or {}).get("cursor")
                if cursor is None:
                    self._result(rid, {"resources": [_RESOURCES[0]], "nextCursor": "r2"})
                else:
                    self._result(rid, {"resources": [_RESOURCES[1]]})
            elif method == "prompts/list":
                cursor = (message.get("params") or {}).get("cursor")
                if cursor is None:
                    self._result(rid, {"prompts": [_PROMPTS[0]], "nextCursor": "p2"})
                else:
                    self._result(rid, {"prompts": [_PROMPTS[1]]})
            else:
                self._error(rid, -32601, "method not found")

        def _result(self, rid: Any, result: dict[str, Any], *, session_id: str | None = None) -> None:
            body = json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}).encode("utf-8")
            self._send(200, body, session_id=session_id)

        def _error(self, rid: Any, code: int, msg: str) -> None:
            body = json.dumps({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}).encode("utf-8")
            self._send(200, body)

        def _send(self, status: int, body: bytes, *, session_id: str | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if session_id:
                self.send_header("Mcp-Session-Id", session_id)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def test_mcp_client_lists_resources_and_prompts_with_pagination() -> None:
    server = HardenedThreadingHTTPServer(("127.0.0.1", 0), _paginated_resource_prompt_handler())
    with serving(server) as base_url:
        client = McpHttpClient(f"{base_url}/mcp")
        try:
            client.initialize()
            assert [r["uri"] for r in client.list_resources()] == ["fake://one", "fake://two"]
            assert [p["name"] for p in client.list_prompts()] == ["brief", "plan"]
        finally:
            client.close()


def _resource_prompt_handler(state: dict[str, int]) -> type[BaseHTTPRequestHandler]:
    """A server with mutable resource/prompt catalogs for provider invalidation tests."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:
            return None

        def do_GET(self) -> None:  # noqa: N802
            self._send(200, b'{"ok":true}')

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            message = json.loads(self.rfile.read(length) or b"{}")
            method, rid = message.get("method"), message.get("id")
            if method == "notifications/initialized":
                self._send(202, b"")
            elif method == "initialize":
                self._result(rid, {"protocolVersion": "2025-06-18", "capabilities": {}}, session_id="sess-1")
            elif method == "tools/list":
                self._result(rid, {"tools": []})
            elif method == "resources/list":
                state["resources"] += 1
                self._result(rid, {"resources": [_RESOURCES[min(state["resource_version"], 1)]]})
            elif method == "resources/read":
                uri = (message.get("params") or {}).get("uri")
                self._result(rid, {"contents": [{"uri": uri, "mimeType": "text/plain", "text": f"read {uri}"}]})
            elif method == "prompts/list":
                state["prompts"] += 1
                self._result(rid, {"prompts": [_PROMPTS[min(state["prompt_version"], 1)]]})
            elif method == "prompts/get":
                params = message.get("params") or {}
                self._result(
                    rid,
                    {
                        "description": "Rendered prompt.",
                        "messages": [
                            {
                                "role": "user",
                                "content": {"type": "text", "text": f"prompt {params.get('name')}"},
                            }
                        ],
                    },
                )
            else:
                self._error(rid, -32601, "method not found")

        def _result(self, rid: Any, result: dict[str, Any], *, session_id: str | None = None) -> None:
            body = json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}).encode("utf-8")
            self._send(200, body, session_id=session_id)

        def _error(self, rid: Any, code: int, msg: str) -> None:
            body = json.dumps({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}).encode("utf-8")
            self._send(200, body)

        def _send(self, status: int, body: bytes, *, session_id: str | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if session_id:
                self.send_header("Mcp-Session-Id", session_id)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def test_mcp_provider_exposes_resources_prompts_context_and_invalidation() -> None:
    state = {"resources": 0, "prompts": 0, "resource_version": 0, "prompt_version": 0}
    server = HardenedThreadingHTTPServer(("127.0.0.1", 0), _resource_prompt_handler(state))
    with serving(server) as base_url:
        with McpToolProvider(f"{base_url}/mcp", server="t") as mcp:
            specs = {s.id: s for s in mcp.get_tools()}
            assert set(specs) == {"mcp.t.resource.read", "mcp.t.prompt.get"}

            read = specs["mcp.t.resource.read"].handler(None, {"uri": "fake://one"})
            assert read.ok and read.content["contents"][0]["text"] == "read fake://one"
            prompt = specs["mcp.t.prompt.get"].handler(None, {"name": "brief", "arguments": {"topic": "x"}})
            assert prompt.ok and prompt.content["messages"][0]["content"]["text"] == "prompt brief"

            empty_turn = TurnContext(1, 1, 1, None, (), 0, bound_tools=frozenset())
            assert mcp.dynamic_segment(empty_turn) is None
            bound_turn = TurnContext(
                1,
                1,
                1,
                None,
                (),
                0,
                bound_tools=frozenset({"mcp.t.resource.read", "mcp.t.prompt.get"}),
            )
            segment = mcp.dynamic_segment(bound_turn)
            assert segment is not None and "fake://one" in segment and "brief" in segment

            state["resource_version"] = 1
            assert mcp.catalog()["resources"][0]["uri"] == "fake://one"  # cached
            assert mcp.handle_list_changed("notifications/resources/list_changed") is True
            assert mcp.catalog()["resources"][0]["uri"] == "fake://two"

            state["prompt_version"] = 1
            assert mcp.catalog()["prompts"][0]["name"] == "brief"  # cached
            mcp.invalidate_prompts()
            assert mcp.catalog()["prompts"][0]["name"] == "plan"
            assert mcp.handle_list_changed("notifications/unknown/list_changed") is False


def test_mcp_resource_prompt_helpers_respect_provider_tool_filters() -> None:
    state = {"resources": 0, "prompts": 0, "resource_version": 0, "prompt_version": 0}
    server = HardenedThreadingHTTPServer(("127.0.0.1", 0), _resource_prompt_handler(state))
    with serving(server) as base_url:
        with McpToolProvider(f"{base_url}/mcp", server="t", allowed_tools=("echo",)) as mcp:
            assert [spec.id for spec in mcp.get_tools()] == []
            assert mcp.tool_bindings() == ()
            assert mcp.catalog()["resources"] == []
            assert mcp.catalog()["prompts"] == []
            assert state["resources"] == 0
            assert state["prompts"] == 0

        with McpToolProvider(
            f"{base_url}/mcp",
            server="t",
            allowed_tools=("resource.read", "mcp.t.prompt.get"),
        ) as mcp:
            assert {spec.id for spec in mcp.get_tools()} == {
                "mcp.t.resource.read",
                "mcp.t.prompt.get",
            }

        with McpToolProvider(
            f"{base_url}/mcp",
            server="t",
            blocked_tools=("resources/read", "mcp_t_prompt_get"),
        ) as mcp:
            assert [spec.id for spec in mcp.get_tools()] == []
            assert mcp.tool_bindings() == ()


def _expiring_handler(state: dict[str, int]) -> type[BaseHTTPRequestHandler]:
    """A server whose session expires once: the first tools/call after init returns 404, forcing
    the client to re-initialize and retry. ``state`` records init/call counts for assertions."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:
            return None

        def do_GET(self) -> None:  # noqa: N802
            self._send(200, b'{"ok":true}')

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            message = json.loads(self.rfile.read(length) or b"{}")
            method, rid = message.get("method"), message.get("id")
            if method == "notifications/initialized":
                self._send(202, b"")
            elif method == "initialize":
                state["inits"] += 1
                self._result(rid, {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}}, session_id=f"sess-{state['inits']}")
            elif method == "tools/list":
                self._result(rid, {"tools": _TOOLS})
            elif method == "tools/call":
                state["calls"] += 1
                if state["calls"] == 1:  # pretend the session expired on the first call
                    self._send(404, b'{"error":"session expired"}')
                    return
                args = (message.get("params") or {}).get("arguments") or {}
                self._result(rid, {"content": [{"type": "text", "text": f"echo: {args.get('text')}"}]})
            else:
                self._error(rid, -32601, "method not found")

        def _result(self, rid: Any, result: dict[str, Any], *, session_id: str | None = None) -> None:
            body = json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}).encode("utf-8")
            self._send(200, body, session_id=session_id)

        def _error(self, rid: Any, code: int, msg: str) -> None:
            body = json.dumps({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}).encode("utf-8")
            self._send(200, body)

        def _send(self, status: int, body: bytes, *, session_id: str | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if session_id:
                self.send_header("Mcp-Session-Id", session_id)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def test_mcp_call_tool_reconnects_after_session_expiry() -> None:
    # A benign 404 session expiry should reconnect + retry transparently, not raise.
    state = {"inits": 0, "calls": 0}
    server = HardenedThreadingHTTPServer(("127.0.0.1", 0), _expiring_handler(state))
    with serving(server) as base_url:
        with McpToolProvider(f"{base_url}/mcp", server="t") as mcp:
            specs = {s.id: s for s in mcp.get_tools()}
            result = specs["mcp.t.echo"].handler(None, {"text": "hi"})

    assert result.ok and result.content["text"] == "echo: hi"
    assert state["inits"] == 2  # re-initialized once after the expiry
    assert state["calls"] == 2  # first call 404'd, retried after reconnect


def _expiring_paginated_handler(state: dict[str, int]) -> type[BaseHTTPRequestHandler]:
    """A paginating server that expires the session the first time a *cursor-bearing* page is
    requested. The cursor ``c2`` is session-scoped: replaying it under a fresh session must NOT
    happen — the client should restart pagination from page one after reconnecting."""
    page1 = [_TOOLS[0]]  # echo
    page2 = [_TOOLS[1]]  # boom

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a: Any) -> None:
            return None

        def do_GET(self) -> None:  # noqa: N802
            self._send(200, b'{"ok":true}')

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            message = json.loads(self.rfile.read(length) or b"{}")
            method, rid = message.get("method"), message.get("id")
            if method == "notifications/initialized":
                self._send(202, b"")
            elif method == "initialize":
                state["inits"] += 1
                self._result(rid, {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}}, session_id=f"sess-{state['inits']}")
            elif method == "tools/list":
                cursor = (message.get("params") or {}).get("cursor")
                if cursor is None:
                    self._result(rid, {"tools": page1, "nextCursor": "c2"})
                elif cursor == "c2" and not state["expired"]:
                    state["expired"] = 1  # session dies exactly when the cursor page is fetched
                    self._send(404, b'{"error":"session expired"}')
                elif cursor == "c2":
                    self._result(rid, {"tools": page2})  # served cleanly under the fresh session
                else:
                    self._error(rid, -32602, f"bad cursor: {cursor}")
            else:
                self._error(rid, -32601, "method not found")

        def _result(self, rid: Any, result: dict[str, Any], *, session_id: str | None = None) -> None:
            body = json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}).encode("utf-8")
            self._send(200, body, session_id=session_id)

        def _error(self, rid: Any, code: int, msg: str) -> None:
            body = json.dumps({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}).encode("utf-8")
            self._send(200, body)

        def _send(self, status: int, body: bytes, *, session_id: str | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if session_id:
                self.send_header("Mcp-Session-Id", session_id)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def test_mcp_list_tools_restarts_pagination_after_session_expiry() -> None:
    # Expiring mid-pagination must reconnect and restart from page one (not replay the stale,
    # session-scoped cursor) — and surface every tool exactly once, no duplicates.
    state = {"inits": 0, "expired": 0}
    server = HardenedThreadingHTTPServer(("127.0.0.1", 0), _expiring_paginated_handler(state))
    with serving(server) as base_url:
        with McpToolProvider(f"{base_url}/mcp", server="t") as mcp:
            ids = [s.id for s in mcp.get_tools()]

    assert sorted(ids) == ["mcp.t.boom", "mcp.t.echo"]  # both pages, no dupes despite the restart
    assert state["inits"] == 2  # re-initialized once after the expiry


def test_mcp_tool_bindings_and_filter() -> None:
    with serving(_server()) as base_url:
        with McpToolProvider(f"{base_url}/mcp", server="t", blocked_tools=("boom",)) as mcp:
            ids = {s.id for s in mcp.get_tools()}
            assert ids == {"mcp.t.echo"}  # boom filtered out
            bindings = mcp.tool_bindings()
            assert [b.ref.tool_id for b in bindings] == ["mcp.t.echo"]
            assert bindings[0].authorization == "allow"


def test_agentloop_calls_mcp_tool(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("mcp_t_echo", {"text": "hello"}, "c1"),)),
            ModelTurn(response_id="r2", final_text="done"),
        ]
    )
    with serving(_server()) as base_url:
        with McpToolProvider(f"{base_url}/mcp", server="t", blocked_tools=("boom",)) as mcp:
            loop = AgentLoop(
                spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs", limits=RunLimits(max_steps=4)),
                model_adapter=adapter,
                runtime_config_provider=runtime_provider(runtime_config(bindings=mcp.tool_bindings())),
                tool_providers=(mcp,),
            )
            result = loop.run_once("use the echo tool")

    assert result.status == "completed"
    assert result.final_text == "done"
    # The MCP tool result reached the model as an observation.
    observations = [obs for request in adapter.requests for obs in request.observations]
    assert any(obs.output.get("result", {}).get("text") == "echo: hello" for obs in observations)
