from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

import pytest

from native_agent_runner.reference._shared.tokens import TokenError, TokenManager
from native_agent_runner.reference.backend.service import BackendRunRequest, RunnerBackend
from native_agent_runner.core.schemas import validate_run_dir
from native_agent_runner.core.spec import AgentRunSpec
from native_agent_runner.reference.llm_gateway.http import create_llm_gateway_server
from native_agent_runner.reference.llm_gateway.service import LlmGatewayBackend
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call
from native_agent_runner.web import WebGatewayClient, WebPolicy
from native_agent_runner.reference.web_gateway.http import create_web_gateway_server
from native_agent_runner.reference.web_gateway.providers import (
    BraveLlmContextProvider,
    BraveSearchProvider,
    CompositeWebProvider,
    HttpFetchProvider,
    SearchFetchContextProvider,
)
from native_agent_runner.reference.web_gateway.service import WebGatewayBackend


def _token_manager() -> TokenManager:
    return TokenManager.from_secret("w" * 32)


def _web_token(
    manager: TokenManager,
    *,
    run_id: str = "run_1",
    tenant_id: str = "tenant_a",
    policy: WebPolicy | None = None,
) -> str:
    return manager.issue(
        kind="web_gateway",
        audience="csp.web-gateway",
        run_id=run_id,
        tenant_id=tenant_id,
        user_id="user_a",
        ttl_s=600,
        metadata={"web_policy": (policy or WebPolicy(enabled=True)).to_json()},
    )


def test_web_policy_json_merge_domain_and_limit_clamping() -> None:
    policy = WebPolicy.from_json(
        {
            "enabled": True,
            "allowed_domains": ["docs.example.test"],
            "blocked_domains": ["blocked.example.test"],
            "max_search_calls": 2,
            "max_fetch_calls": 3,
            "context_enabled": True,
            "max_context_calls": 4,
            "default_max_results": 5,
            "max_results": 3,
            "default_max_context_tokens": 9000,
            "max_context_tokens": 8000,
            "default_max_context_urls": 10,
            "max_context_urls": 4,
            "default_max_context_snippets": 99,
            "max_context_snippets": 10,
            "default_max_response_bytes": 100,
            "max_response_bytes": 50,
        }
    )

    merged = policy.merged(
        blocked_domains=("*.private.test",),
        max_results=2,
        max_context_tokens=6000,
        max_context_urls=3,
        max_context_snippets=5,
    )

    assert merged.enabled is True
    assert merged.context_enabled is True
    assert merged.allowed_domains == ("docs.example.test",)
    assert merged.blocked_domains == ("blocked.example.test", "*.private.test")
    assert merged.effective_max_results(99) == 2
    assert merged.effective_max_context_tokens(99_999) == 6000
    assert merged.effective_max_context_urls(99) == 3
    assert merged.effective_max_context_snippets(99) == 5
    assert merged.effective_max_response_bytes(99) == 50


def test_web_gateway_token_policy_usage_and_domain_controls() -> None:
    manager = _token_manager()
    gateway = WebGatewayBackend(token_manager=manager)
    policy = WebPolicy(
        enabled=True,
        allowed_domains=("docs.example.test",),
        blocked_domains=("blog.example.test",),
        max_search_calls=1,
        max_fetch_calls=2,
        context_enabled=True,
        max_context_calls=1,
        max_results=5,
    )
    token = _web_token(manager, policy=policy)

    search = gateway.handle_search(token, {"query": "web policy", "max_results": 5})
    assert search["result_count"] == 2
    assert {result["domain"] for result in search["results"]} == {"docs.example.test"}
    fetched = gateway.handle_fetch(token, {"url": search["results"][0]["url"], "max_bytes": 40})
    assert fetched["domain"] == "docs.example.test"
    assert fetched["content_bytes"] <= 40
    assert gateway.tenant_usage("tenant_a")["search_calls"] == 1
    assert gateway.tenant_usage("tenant_a")["fetch_calls"] == 1
    context = gateway.handle_context(token, {"query": "web policy", "max_tokens": 1024, "max_urls": 2})
    assert context["source_count"] == 2
    assert "WebPolicy controls" in context["context"]
    assert gateway.tenant_usage("tenant_a")["context_calls"] == 1
    assert gateway.tenant_usage("tenant_a")["context_source_count"] == 2

    with pytest.raises(Exception, match="limit exceeded"):
        gateway.handle_search(token, {"query": "web"})
    with pytest.raises(Exception, match="limit exceeded"):
        gateway.handle_context(token, {"query": "web policy"})
    with pytest.raises(Exception, match="not allowed"):
        gateway.handle_fetch(token, {"url": "https://blog.example.test/agent-observability"})
    with pytest.raises(TokenError):
        manager.verify(token, kind="llm_gateway", audience="csp.llm-gateway")


def test_web_gateway_http_client_and_usage(tmp_path: Path) -> None:
    del tmp_path
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
        search = client.search({"query": "web policy", "max_results": 1})
        assert search["result_count"] == 1
        fetched = client.fetch({"url": search["results"][0]["url"], "max_bytes": 80})
        assert "content" in fetched
        usage = _json_get(f"{base_url}/internal/web/tenants/tenant_a/usage", token="admin")
        assert usage["search_calls"] == 1
        assert usage["fetch_calls"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_brave_search_and_http_fetch_provider_contract(tmp_path: Path) -> None:
    del tmp_path
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
        token = _web_token(
            gateway.token_manager,
            policy=WebPolicy(enabled=True, context_enabled=True, allowed_domains=("127.0.0.1",), max_results=3),
        )

        search = gateway.handle_search(token, {"query": "native agent", "max_results": 2})
        assert search["result_count"] == 1
        assert search["results"][0]["source"] == "brave"
        assert search["results"][0]["url"] == f"{upstream.base_url}/docs/native-agent"
        assert upstream.last_brave_headers["X-Subscription-Token"] == "brave-test-key"

        fetched = gateway.handle_fetch(token, {"url": search["results"][0]["url"], "format": "text"})
        assert fetched["source"] == "http"
        assert fetched["title"] == "Native Agent Docs"
        assert "Brave search result body" in fetched["content"]
        context = gateway.handle_context(token, {"query": "native agent", "max_tokens": 1024, "max_urls": 1})
        assert context["source"] == "search-fetch-context"
        assert context["source_count"] == 1
        assert "Brave search result body" in context["context"]
        assert gateway.tenant_usage("tenant_a")["search_calls"] == 1
        assert gateway.tenant_usage("tenant_a")["fetch_calls"] == 1
        assert gateway.tenant_usage("tenant_a")["context_calls"] == 1
    finally:
        upstream.stop()


def test_brave_llm_context_provider_contract(tmp_path: Path) -> None:
    del tmp_path
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
        assert context["sources"][0]["domain"] == "docs.example.test"
        assert upstream.last_brave_headers["X-Subscription-Token"] == "brave-context-key"
    finally:
        upstream.stop()


@pytest.mark.skipif(
    not os.environ.get("BRAVE_SEARCH_API_KEY"),
    reason="BRAVE_SEARCH_API_KEY is required for live Brave Search smoke",
)
def test_brave_search_provider_live_smoke() -> None:
    provider = BraveSearchProvider.from_env(timeout_s=10)

    results = provider.search("native agent runner web tools", max_results=2)

    assert results
    assert all(result["url"].startswith(("http://", "https://")) for result in results)
    assert all(result["source"] == "brave" for result in results)


@pytest.mark.skipif(
    not os.environ.get("BRAVE_SEARCH_API_KEY"),
    reason="BRAVE_SEARCH_API_KEY is required for live Brave LLM Context smoke",
)
def test_brave_llm_context_provider_live_smoke() -> None:
    provider = BraveLlmContextProvider.from_env(timeout_s=15)

    context = provider.context(
        "native agent runner web tools",
        max_tokens=2048,
        max_urls=2,
        max_snippets=5,
        locale="en",
        freshness=None,
        allowed_domains=(),
        blocked_domains=(),
    )

    assert context["source"] == "brave-llm-context"
    assert context["context"]


def test_agent_loop_web_search_fetch_events_metrics_and_private_transcript(tmp_path: Path) -> None:
    manager = _token_manager()
    policy = WebPolicy(enabled=True, max_search_calls=2, max_fetch_calls=2)
    gateway = WebGatewayBackend(token_manager=manager)
    server = create_web_gateway_server(gateway, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret_query = "web policy tenant-secret-query-123"
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
                            {"url": "https://docs.example.test/native-agent-runner/policy", "max_bytes": 120},
                            "fetch_1",
                        ),
                    ),
                ),
                ModelTurn(
                    response_id="r3",
                    tool_calls=(fake_tool_call("run_finish", {"summary": "web done"}, "finish_1"),),
                ),
            ]
        )
        spec = AgentRunSpec(
            instruction="Use web.",
            workspace_root=workspace,
            run_root=tmp_path / "runs",
            web_policy=policy,
        )

        result = AgentLoop(
            spec=spec,
            model_adapter=adapter,
            web_gateway_client=WebGatewayClient(base_url, token=_web_token(manager, run_id=spec.run_id, policy=policy)),
        ).run()

        assert result.status == "completed"
        assert {"web.search", "web.fetch"}.issubset({tool.id for tool in adapter.requests[0].tools})
        metrics = json.loads(result.run_dir.joinpath("metrics.json").read_text(encoding="utf-8"))
        assert metrics["web_search_calls"] == 1
        assert metrics["web_fetch_calls"] == 1
        assert metrics["web_result_count"] == 1
        events_text = result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8")
        assert "web.search.started" in events_text
        assert "web.fetch.finished" in events_text
        assert "tenant-secret-query-123" not in events_text
        transcript_text = result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")
        assert "tenant-secret-query-123" in transcript_text
        assert "WebPolicy controls" in transcript_text
        manifest = json.loads(result.run_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
        assert manifest["web_policy"]["enabled"] is True
        assert validate_run_dir(result.run_dir) == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_agent_loop_web_context_events_metrics_and_private_transcript(tmp_path: Path) -> None:
    manager = _token_manager()
    policy = WebPolicy(enabled=True, context_enabled=True, max_context_calls=2, max_context_tokens=2048)
    gateway = WebGatewayBackend(token_manager=manager)
    server = create_web_gateway_server(gateway, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret_query = "native agent context tenant-secret-query-456"
    try:
        _wait_http_ready(base_url)
        adapter = FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="r1",
                    tool_calls=(
                        fake_tool_call(
                            "web_context",
                            {"query": secret_query, "max_tokens": 2048, "max_urls": 2, "max_snippets": 2},
                            "context_1",
                        ),
                    ),
                ),
                ModelTurn(
                    response_id="r2",
                    tool_calls=(fake_tool_call("run_finish", {"summary": "context done"}, "finish_1"),),
                ),
            ]
        )
        spec = AgentRunSpec(
            instruction="Use web context.",
            workspace_root=workspace,
            run_root=tmp_path / "runs",
            web_policy=policy,
        )

        result = AgentLoop(
            spec=spec,
            model_adapter=adapter,
            web_gateway_client=WebGatewayClient(base_url, token=_web_token(manager, run_id=spec.run_id, policy=policy)),
        ).run()

        assert result.status == "completed"
        assert "web.context" in {tool.id for tool in adapter.requests[0].tools}
        metrics = json.loads(result.run_dir.joinpath("metrics.json").read_text(encoding="utf-8"))
        assert metrics["web_context_calls"] == 1
        assert metrics["web_context_source_count"] >= 1
        assert metrics["web_context_bytes_returned"] > 0
        events_text = result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8")
        assert "web.context.started" in events_text
        assert "web.context.finished" in events_text
        assert "tenant-secret-query-456" not in events_text
        assert "Native Agent Runner exposes web.search" not in events_text
        transcript_text = result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")
        assert "tenant-secret-query-456" in transcript_text
        assert "Native Agent Runner exposes web.search" in transcript_text
        manifest = json.loads(result.run_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
        assert manifest["web_policy"]["context_enabled"] is True
        assert validate_run_dir(result.run_dir) == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_agent_loop_web_disabled_hides_tool_and_stale_call_reports_capability_disabled(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("web_search", {"query": "web"}, "search_1"),),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    spec = AgentRunSpec(instruction="Try web.", workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    assert result.status == "completed"
    assert "web.search" not in {tool.id for tool in adapter.requests[0].tools}
    transcript = result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")
    assert "capability_disabled" in transcript


def test_agent_loop_web_context_disabled_hides_tool_and_stale_call_reports_capability_disabled(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("web_context", {"query": "web"}, "context_1"),),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    spec = AgentRunSpec(
        instruction="Try web context.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        web_policy=WebPolicy(enabled=True, context_enabled=False),
    )

    result = AgentLoop(spec=spec, model_adapter=adapter).run()

    assert result.status == "completed"
    assert "web.context" not in {tool.id for tool in adapter.requests[0].tools}
    transcript = result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")
    assert "capability_disabled" in transcript


def test_full_stack_fake_llm_runner_backend_and_web_gateway_contract(tmp_path: Path) -> None:
    manager = _token_manager()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("Use web docs.\n", encoding="utf-8")
    web_policy = WebPolicy(
        enabled=True,
        context_enabled=True,
        allowed_domains=("docs.example.test",),
        max_search_calls=2,
        max_fetch_calls=2,
        max_context_calls=2,
    )
    adapters: dict[str, FakeModelAdapter] = {}

    def llm_factory(claims, _config):
        if claims.run_id not in adapters:
            adapters[claims.run_id] = FakeModelAdapter(
                turns=[
                    ModelTurn(
                        response_id="provider_1",
                        tool_calls=(
                            fake_tool_call("web_search", {"query": "native agent runner web", "max_results": 1}, "web_s"),
                            fake_tool_call(
                                "web_context",
                                {"query": "native agent runner web", "max_tokens": 2048, "max_urls": 2},
                                "web_c",
                            ),
                        ),
                        usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                    ),
                    ModelTurn(
                        response_id="provider_2",
                        tool_calls=(
                            fake_tool_call(
                                "web_fetch",
                                {"url": "https://docs.example.test/native-agent-runner/web", "max_bytes": 200},
                                "web_f",
                            ),
                            fake_tool_call(
                                "fs_write",
                                {
                                    "path": "WEB_SUMMARY.md",
                                    "content": "Summary from WebGateway docs\n",
                                    "create_dirs": False,
                                },
                                "write_1",
                            ),
                        ),
                        usage={"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
                    ),
                    ModelTurn(
                        response_id="provider_3",
                        tool_calls=(fake_tool_call("run_finish", {"summary": "web summary ready"}, "finish_1"),),
                        usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                    ),
                ]
            )
        return adapters[claims.run_id]

    web_gateway = WebGatewayBackend(token_manager=manager)
    web_server = create_web_gateway_server(web_gateway, host="127.0.0.1", port=0, admin_token="web-admin")
    web_thread = threading.Thread(target=web_server.serve_forever, daemon=True)
    web_thread.start()
    web_url = f"http://127.0.0.1:{web_server.server_address[1]}"
    llm_gateway = LlmGatewayBackend(token_manager=manager, provider_adapter_factory=llm_factory)
    llm_server = create_llm_gateway_server(llm_gateway, host="127.0.0.1", port=0, admin_token="llm-admin")
    llm_thread = threading.Thread(target=llm_server.serve_forever, daemon=True)
    llm_thread.start()
    llm_url = f"http://127.0.0.1:{llm_server.server_address[1]}"
    try:
        _wait_http_ready(web_url)
        _wait_http_ready(llm_url)
        runner_backend = RunnerBackend(
            run_root=tmp_path / "runs",
            token_manager=manager,
            allowed_workspace_roots=(workspace,),
            llm_gateway_url=f"{llm_url}/internal/llm/turns",
            web_gateway_url=web_url,
        )
        submission = runner_backend.submit_run(
            BackendRunRequest(
                tenant_id="tenant_a",
                user_id="user_a",
                workspace_root=workspace,
                instruction="Use web docs and write a summary.",
                mode="propose",
                web_policy=web_policy,
            )
        )
        assert runner_backend.wait_for_run(submission.run_id, timeout_s=5) == "completed"
        assert not workspace.joinpath("WEB_SUMMARY.md").exists()
        result = runner_backend.result(submission.run_id, submission.run_token)
        assert result["metrics"]["web_search_calls"] == 1
        assert result["metrics"]["web_fetch_calls"] == 1
        assert result["metrics"]["web_context_calls"] == 1
        assert runner_backend.tenant_usage("tenant_a")["web_search_calls"] == 1
        assert runner_backend.tenant_usage("tenant_a")["web_context_calls"] == 1
        assert web_gateway.tenant_usage("tenant_a")["search_calls"] == 1
        assert web_gateway.tenant_usage("tenant_a")["fetch_calls"] == 1
        assert web_gateway.tenant_usage("tenant_a")["context_calls"] == 1
        proposal_file = runner_backend.proposal_file(submission.run_id, submission.run_token, "WEB_SUMMARY.md")
        assert proposal_file["content"] == "Summary from WebGateway docs\n"
        run_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in submission.run_dir.rglob("*") if path.is_file())
        assert "web_gateway" not in run_text
    finally:
        llm_server.shutdown()
        llm_server.server_close()
        llm_thread.join(timeout=5)
        web_server.shutdown()
        web_server.server_close()
        web_thread.join(timeout=5)


def _json_post(url: str, payload: dict, *, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _json_get(url: str, *, token: str) -> dict:
    request = Request(url, headers={"Authorization": f"Bearer {token}"}, method="GET")
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_http_ready(base_url: str, *, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            _json_get(f"{base_url}/healthz", token="unused")
            return
        except Exception as exc:  # pragma: no cover - only exercised under startup races.
            last_error = exc
            time.sleep(0.02)
    raise TimeoutError(f"server did not become ready: {last_error}")


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
                        "<body><main><h1>Native Agent</h1>"
                        "<p>Brave search result body for the runner.</p>"
                        "<script>hidden()</script></main></body></html>"
                    )
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/brave/context":
                    outer.last_brave_headers = dict(self.headers.items())
                    length = int(self.headers.get("Content-Length") or "0")
                    payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                    self._write_json(
                        {
                            "query": payload.get("q"),
                            "context": "LLM-ready Brave context for native agent runner.",
                            "sources": [
                                {
                                    "title": "Native Agent Docs",
                                    "url": "https://docs.example.test/native-agent-runner/web",
                                }
                            ],
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

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

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
