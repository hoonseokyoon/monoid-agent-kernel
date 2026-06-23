"""The offline reference MCP server (reference/mcp_gateway) exercised by the *production* MCP
client (core mcp/), proving wire compatibility with no key/egress/external process.
"""

from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from conftest import serving

from native_agent_runner.mcp import McpToolProvider
from native_agent_runner.reference._shared.http_util import wait_http_ready
from native_agent_runner.reference.mcp_gateway import FakeMcpServer, create_mcp_server


def _server(admin_token: str | None = None):
    return create_mcp_server(FakeMcpServer(), host="127.0.0.1", port=0, admin_token=admin_token)


def test_mcp_gateway_discovers_and_round_trips_a_call() -> None:
    with serving(_server()) as base_url:
        with McpToolProvider(f"{base_url}/mcp", server="studio") as mcp:
            specs = {s.id: s for s in mcp.get_tools()}
            assert set(specs) == {"mcp.studio.echo", "mcp.studio.uppercase"}
            # readOnlyHint annotation maps to side_effect="read"; the unmarked tool is "run".
            assert specs["mcp.studio.echo"].side_effect == "run"
            assert specs["mcp.studio.uppercase"].side_effect == "read"
            assert specs["mcp.studio.echo"].input_schema["required"] == ["text"]

            ok = specs["mcp.studio.echo"].handler(None, {"text": "hi"})
            assert ok.ok and ok.content["text"] == "echo: hi"

            up = specs["mcp.studio.uppercase"].handler(None, {"text": "hi"})
            assert up.ok and up.content["text"] == "HI"


def test_mcp_gateway_healthz_and_readiness_poll() -> None:
    with serving(_server()) as base_url:
        wait_http_ready(base_url, timeout_s=5)  # must not raise
        with urlopen(Request(f"{base_url}/healthz"), timeout=2) as response:
            assert json.loads(response.read())["ok"] is True


def test_mcp_gateway_unknown_method_is_jsonrpc_error() -> None:
    with serving(_server()) as base_url:
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "nope"}).encode("utf-8")
        request = Request(f"{base_url}/mcp", data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=2) as response:
            payload = json.loads(response.read())
        assert payload["error"]["code"] == -32601


def test_mcp_gateway_rejects_bad_token_when_admin_configured() -> None:
    with serving(_server(admin_token="secret")) as base_url:
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode("utf-8")
        request = Request(
            f"{base_url}/mcp",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer wrong"},
        )
        with pytest.raises(HTTPError) as excinfo:
            urlopen(request, timeout=2)
        assert excinfo.value.code == 401

        # The matching token discovers normally.
        with McpToolProvider(f"{base_url}/mcp", server="studio", token="secret") as mcp:
            assert {s.id for s in mcp.get_tools()} == {"mcp.studio.echo", "mcp.studio.uppercase"}
