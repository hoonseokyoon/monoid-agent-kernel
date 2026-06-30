from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

from native_agent_runner.reference._shared.http_util import HardenedThreadingHTTPServer

import pytest

from support.http import (
    http_get_json as _json_get,
    http_json,
    wait_http_ready as _wait_http_ready,
)
from support.runtime import runtime_config, runtime_provider, tool_binding

from native_agent_runner.core.schemas import validate_run_dir
from native_agent_runner.core.spec import AgentRunSpec
from native_agent_runner.core.tool_surface import ToolScope
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call
from native_agent_runner.reference._shared.tokens import TokenError, TokenManager
from native_agent_runner.reference.web_gateway.http import create_web_gateway_server
from native_agent_runner.reference.web_gateway.providers import (
    BraveLlmContextProvider,
    BraveSearchProvider,
    CompositeWebProvider,
    HttpFetchProvider,
    SearchFetchContextProvider,
)
from native_agent_runner.reference.web_gateway.service import WebGatewayBackend
from native_agent_runner.web import WebGatewayClient

pytestmark = pytest.mark.integration


def _token_manager() -> TokenManager:
    return TokenManager.from_secret("w" * 32)


def _web_token(manager: TokenManager, *, run_id: str = "run_1", tenant_id: str = "tenant_a") -> str:
    return manager.issue(
        kind="web_gateway",
        audience="csp.web-gateway",
        run_id=run_id,
        tenant_id=tenant_id,
        user_id="user_a",
        ttl_s=600,
        metadata={"agent_config_hash": "test"},
    )


def test_web_gateway_enforces_binding_constraints_usage_and_domains() -> None:
    manager = _token_manager()
    gateway = WebGatewayBackend(token_manager=manager)
    token = _web_token(manager)
    common = {
        "binding_id": "search_docs",
        "max_calls": 1,
        "allowed_domains": ["docs.example.test"],
        "blocked_domains": ["blog.example.test"],
    }

    search = gateway.handle_search(token, {"query": "binding", "max_results": 5, **common})

    assert search["result_count"] >= 1
    assert {result["domain"] for result in search["results"]} == {"docs.example.test"}
    assert gateway.tenant_usage("tenant_a")["search_calls"] == 1
    with pytest.raises(Exception, match="limit exceeded"):
        gateway.handle_search(token, {"query": "binding", **common})
    with pytest.raises(Exception, match="not allowed"):
        gateway.handle_fetch(
            token,
            {
                "binding_id": "fetch_docs",
                "url": "https://blog.example.test/agent-observability",
                "allowed_domains": ["docs.example.test"],
            },
        )
    with pytest.raises(TokenError):
        manager.verify(token, kind="llm_gateway", audience="csp.llm-gateway")


def test_web_gateway_client_retries_transient_connection_error(monkeypatch) -> None:
    # A bare connection-level error (OSError, neither HTTPError nor a real response) is
    # transient and must be retried, not surfaced as a failed web call.
    calls = 0

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return None

        def read(self):
            return b'{"result_count": 0, "results": []}'

    def fake_urlopen(_request, timeout):
        del timeout
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionResetError("connection reset by peer")
        return _Resp()

    monkeypatch.setattr("native_agent_runner.web.urlopen", fake_urlopen)
    monkeypatch.setattr("native_agent_runner.web.time.sleep", lambda _d: None)
    client = WebGatewayClient("http://gateway.local", token="t")

    result = client.search({"binding_id": "b", "query": "q"})

    assert calls == 2
    assert result["result_count"] == 0


def test_web_gateway_http_client_and_usage() -> None:
    manager = _token_manager()
    gateway = WebGatewayBackend(token_manager=manager)
    server = create_web_gateway_server(gateway, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        _wait_http_ready(base_url)
        with pytest.raises(HTTPError) as exc_info:
            _json_post(f"{base_url}/internal/web/search", {"query": "web"})
        assert exc_info.value.code == 401
        client = WebGatewayClient(base_url, token=_web_token(manager))
        search = client.search({"binding_id": "search_docs", "query": "binding", "max_results": 1})
        assert search["result_count"] == 1
        fetched = client.fetch({"binding_id": "fetch_docs", "url": search["results"][0]["url"], "max_bytes": 80})
        assert "content" in fetched
        usage = _json_get(f"{base_url}/internal/web/tenants/tenant_a/usage", token="admin")
        assert usage["search_calls"] == 1
        assert usage["fetch_calls"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_agent_loop_web_bindings_events_metrics_and_private_transcript(tmp_path: Path) -> None:
    manager = _token_manager()
    gateway = WebGatewayBackend(token_manager=manager)
    server = create_web_gateway_server(gateway, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret_query = "binding tenant-secret-query-123"
    try:
        _wait_http_ready(base_url)
        adapter = FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="r1",
                    tool_calls=(fake_tool_call("web_search", {"query": secret_query, "max_results": 1}, "search_1"),),
                ),
                ModelTurn(
                    response_id="r2",
                    tool_calls=(
                        fake_tool_call(
                            "web_fetch",
                            {"url": "https://docs.example.test/native-agent-runner/web", "max_bytes": 120},
                            "fetch_1",
                        ),
                    ),
                ),
                ModelTurn(response_id="r3", tool_calls=(fake_tool_call("run_finish", {"summary": "done"}, "finish_1"),)),
            ]
        )
        config = runtime_config(
            bindings=(
                tool_binding("web.search", scope=ToolScope(allowed_domains=("docs.example.test",)), runtime={"web": {"max_calls": 2}}),
                tool_binding("web.fetch", scope=ToolScope(allowed_domains=("docs.example.test",)), runtime={"web": {"max_calls": 2}}),
                tool_binding("run.finish"),
            )
        )

        spec = AgentRunSpec(
            workspace_root=workspace,
            run_root=tmp_path / "runs",
            run_id="web_loop",
        )
        result = AgentLoop(
            spec=spec,
            model_adapter=adapter,
            runtime_config_provider=runtime_provider(config),
            web_gateway_client=WebGatewayClient(base_url, token=_web_token(manager, run_id=spec.run_id)),
        ).run_once("Use web.")

        assert result.status == "completed"
        metrics = json.loads(result.run_dir.joinpath("metrics.json").read_text(encoding="utf-8"))
        assert metrics["web_search_calls"] >= 1
        events_text = result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8")
        assert "web.search.started" in events_text
        assert "tenant-secret-query-123" not in events_text
        transcript_text = result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")
        assert "tenant-secret-query-123" in transcript_text
        assert validate_run_dir(result.run_dir) == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_web_providers_contract() -> None:
    upstream = _FakeUpstreamServer()
    upstream.start()
    try:
        search_provider = BraveSearchProvider(
            api_key="brave-test-key",
            endpoint=f"{upstream.base_url}/brave/search",
            timeout_s=5,
        )
        fetch_provider = HttpFetchProvider(timeout_s=5, max_raw_bytes=20_000)
        provider = CompositeWebProvider(
            search_provider=search_provider,
            fetch_provider=fetch_provider,
            context_provider=SearchFetchContextProvider(search_provider=search_provider, fetch_provider=fetch_provider),
        )
        gateway = WebGatewayBackend(token_manager=_token_manager(), provider=provider)
        token = _web_token(gateway.token_manager)
        search = gateway.handle_search(
            token,
            {
                "binding_id": "search",
                "query": "native agent",
                "max_results": 2,
                "allowed_domains": ["127.0.0.1"],
            },
        )
        assert search["result_count"] == 1
        fetched = gateway.handle_fetch(token, {"binding_id": "fetch", "url": search["results"][0]["url"], "format": "text"})
        assert "Brave search result body" in fetched["content"]
        context = gateway.handle_context(
            token,
            {"binding_id": "context", "query": "native agent", "max_tokens": 1024, "max_urls": 1, "allowed_domains": ["127.0.0.1"]},
        )
        assert "Brave search result body" in context["context"]
    finally:
        upstream.stop()


def test_brave_llm_context_provider_contract() -> None:
    upstream = _FakeUpstreamServer()
    upstream.start()
    try:
        provider = BraveLlmContextProvider(
            api_key="brave-context-key",
            endpoint=f"{upstream.base_url}/brave/context",
            timeout_s=5,
        )
        context = provider.context(
            "native agent context",
            max_tokens=2048,
            max_urls=2,
            max_snippets=3,
            locale="en",
            freshness="pw",
            allowed_domains=(),
            blocked_domains=(),
        )
        assert context["source"] == "brave-llm-context"
        assert "LLM-ready Brave context" in context["context"]
    finally:
        upstream.stop()


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("BRAVE_SEARCH_API_KEY"), reason="BRAVE_SEARCH_API_KEY is required")
def test_brave_search_provider_live_smoke() -> None:
    assert BraveSearchProvider.from_env(timeout_s=10).search("native agent runner web tools", max_results=2)


def _json_post(url: str, payload: dict, *, token: str | None = None) -> dict:
    return http_json(url, payload, token=token)


class _FakeUpstreamServer:
    def __init__(self) -> None:
        self.last_brave_headers: dict[str, str] = {}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/healthz":
                    self._write_json({"ok": True})
                    return
                if parsed.path == "/brave/search":
                    outer.last_brave_headers = dict(self.headers.items())
                    query = parse_qs(parsed.query).get("q", [""])[0]
                    self._write_json(
                        {
                            "web": {
                                "results": [
                                    {
                                        "title": "Native Agent Docs",
                                        "url": f"http://127.0.0.1:{outer.port}/docs/native-agent",
                                        "description": f"Result for {query}",
                                    }
                                ]
                            }
                        }
                    )
                    return
                if parsed.path == "/docs/native-agent":
                    self._write_html(
                        "<html><head><title>Native Agent Docs</title></head>"
                        "<body><main><p>Brave search result body for the runner.</p></main></body></html>"
                    )
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self) -> None:  # noqa: N802
                if urlparse(self.path).path == "/brave/context":
                    self._write_json(
                        {
                            "context": "LLM-ready Brave context for native agent runner.",
                            "sources": [{"title": "Native Agent Docs", "url": "https://docs.example.test/native-agent-runner/web"}],
                            "chunks": [
                                {
                                    "title": "Native Agent Docs",
                                    "url": "https://docs.example.test/native-agent-runner/web",
                                    "text": "LLM-ready Brave context for native agent runner.",
                                }
                            ],
                        }
                    )
                    return
                self.send_response(404)
                self.end_headers()

            def log_message(self, _format: str, *_args: Any) -> None:
                return None

            def _write_json(self, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _write_html(self, html: str) -> None:
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = HardenedThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever)

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.thread.start()
        _wait_http_ready(self.base_url)

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
