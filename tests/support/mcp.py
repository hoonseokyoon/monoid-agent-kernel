from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class ScriptedMcpHandler(BaseHTTPRequestHandler):
    """Small JSON-RPC handler for tests that need a fake MCP endpoint."""

    responses: list[dict[str, Any]] = []

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self.send_response(200)
            self.end_headers()
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        response = self.responses.pop(0) if self.responses else {"result": {}}
        response.setdefault("jsonrpc", "2.0")
        response.setdefault("id", body.get("id"))
        data = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return None


def scripted_mcp_server(*responses: dict[str, Any]) -> ThreadingHTTPServer:
    handler = type("ConfiguredScriptedMcpHandler", (ScriptedMcpHandler,), {"responses": list(responses)})
    return ThreadingHTTPServer(("127.0.0.1", 0), handler)

