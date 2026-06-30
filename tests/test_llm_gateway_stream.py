"""P4b-①: LLM-gateway token streaming.

The gateway side (endpoint -> handle_turn_stream -> sync pump -> SSE framing) is verified
with a stdlib urlopen streaming read, so those tests need neither httpx nor an API key. The
adapter side (GatewayModelAdapter.astream_turn) needs httpx and is skipped if it is absent.
Async tests use asyncio.run from sync functions (no pytest-asyncio), matching the suite.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from support.http import serving
from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core.spec import AgentRunSpec, ModelConfig, RunLimits
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelTurn,
    TextDelta,
    ToolCallDelta,
    TurnComplete,
    assemble_streamed_turn,
)
from monoid_agent_kernel.providers.fake import FakeModelAdapter, FakeStreamingModelAdapter
from monoid_agent_kernel.providers.gateway import GatewayModelAdapter
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.llm_gateway.http import create_llm_gateway_server
from monoid_agent_kernel.reference.llm_gateway.service import LlmGatewayBackend


def _token_manager() -> TokenManager:
    return TokenManager.from_secret("y" * 32)


def _llm_token(manager: TokenManager, *, run_id: str = "run_1", tenant_id: str = "tenant_a") -> str:
    return manager.issue(
        kind="llm_gateway",
        audience="csp.llm-gateway",
        run_id=run_id,
        tenant_id=tenant_id,
        user_id="user_a",
        ttl_s=600,
        metadata={"agent_config_hash": "test"},
    )


def _turn_payload() -> dict[str, Any]:
    return {
        "protocol": "native-agent-runner.llm-turn.v1",
        "model": "gpt-5.5",
        "system_prompt": "sys",
        "reasoning": {"effort": "low"},
        "tools": [],
        "instruction": "go",
    }


def _server_for(provider_factory) -> Any:
    manager = _token_manager()
    gateway = LlmGatewayBackend(token_manager=manager, provider_adapter_factory=provider_factory)
    server = create_llm_gateway_server(gateway, host="127.0.0.1", port=0, admin_token="admin")
    return server, manager


def _post_sse(base_url: str, token: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """POST to the streaming endpoint and parse the SSE frames (stdlib only, no httpx)."""
    request = Request(
        f"{base_url}/internal/llm/turns/stream",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        assert response.headers.get("Content-Type", "").startswith("text/event-stream")
        raw = response.read().decode("utf-8")
    frames: list[dict[str, Any]] = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if block.startswith("data:"):
            frames.append(json.loads(block[len("data:") :].strip()))
    return frames


async def _collect(agen) -> list[Any]:
    return [chunk async for chunk in agen]


def _adapter(base_url: str, token: str) -> GatewayModelAdapter:
    return GatewayModelAdapter(ModelConfig(), gateway_url=f"{base_url}/internal/llm/turns", token=token)


# --- Gateway side (no httpx, no key) ---------------------------------------------------


def test_gateway_streams_sse_frames() -> None:
    chunks = [
        TextDelta("Hel"),
        TextDelta("lo"),
        ToolCallDelta(index=0, arguments_fragment='{"path":"A', id="c1", name="fs_write"),
        ToolCallDelta(index=0, arguments_fragment='.md"}'),
        TurnComplete(response_id="provider_secret", usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}),
    ]
    server, manager = _server_for(lambda *_: FakeStreamingModelAdapter(chunk_turns=[chunks]))
    with serving(server) as base_url:
        frames = _post_sse(base_url, _llm_token(manager), _turn_payload())

    assert [f["type"] for f in frames] == [
        "text_delta",
        "text_delta",
        "tool_call_delta",
        "tool_call_delta",
        "turn_complete",
    ]
    assert "".join(f["text"] for f in frames if f["type"] == "text_delta") == "Hello"
    # The provider's response id is never exposed; only the opaque turn_handle, last frame only.
    assert "provider_secret" not in json.dumps(frames)
    assert frames[-1]["turn_handle"].startswith("turn_")
    assert all("turn_handle" not in f for f in frames[:-1])
    assert frames[-1]["usage"]["total_tokens"] == 5


def test_gateway_stream_falls_back_for_nonstreaming_provider() -> None:
    server, manager = _server_for(
        lambda *_: FakeModelAdapter(turns=[ModelTurn(response_id="prov", final_text="done", usage={"total_tokens": 4})])
    )
    with serving(server) as base_url:
        frames = _post_sse(base_url, _llm_token(manager), _turn_payload())

    assert frames[-1]["type"] == "turn_complete"
    assert frames[-1]["turn_handle"].startswith("turn_")
    assert "".join(f["text"] for f in frames if f["type"] == "text_delta") == "done"


def test_gateway_stream_rejects_bad_token_before_streaming() -> None:
    # A pre-stream auth failure is a normal non-200 JSON error, not a 200 SSE error frame.
    server, _ = _server_for(lambda *_: FakeStreamingModelAdapter())
    with serving(server) as base_url:
        with pytest.raises(HTTPError) as excinfo:
            _post_sse(base_url, "not-a-valid-token", _turn_payload())
    assert excinfo.value.code in (401, 403)


# --- Adapter side (needs httpx) --------------------------------------------------------


def test_gateway_adapter_astream_turn_round_trips() -> None:
    pytest.importorskip("httpx")
    chunks = [
        TextDelta("hi"),
        ToolCallDelta(index=0, arguments_fragment='{"x":1}', id="c1", name="fs_write"),
        TurnComplete(response_id="provider_secret", usage={"total_tokens": 7}),
    ]
    server, manager = _server_for(lambda *_: FakeStreamingModelAdapter(chunk_turns=[chunks]))
    with serving(server) as base_url:
        adapter = _adapter(base_url, _llm_token(manager))
        request = ModelRequest(instruction="go", system_prompt="sys", tools=())
        collected = asyncio.run(_collect(adapter.astream_turn(request)))

    assert any(isinstance(c, TextDelta) for c in collected)
    completes = [c for c in collected if isinstance(c, TurnComplete)]
    assert completes and completes[0].response_id.startswith("turn_")  # gateway handle, not provider id
    turn = assemble_streamed_turn(collected)
    assert turn.final_text == "hi"
    assert turn.tool_calls[0].arguments == {"x": 1}
    assert turn.response_id.startswith("turn_")


def test_agentloop_astream_over_gateway_streams_real_tokens(tmp_path: Path) -> None:
    pytest.importorskip("httpx")
    server, manager = _server_for(
        lambda *_: FakeStreamingModelAdapter(chunk_turns=[[TextDelta("done"), TurnComplete(response_id="prov")]])
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    with serving(server) as base_url:
        adapter = _adapter(base_url, _llm_token(manager))
        loop = AgentLoop(
            spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs", limits=RunLimits()),
            model_adapter=adapter,
            runtime_config_provider=runtime_provider(runtime_config("run.finish")),
        )

        async def go() -> tuple[list[Any], Any]:
            await loop.aopen()
            items: list[Any] = []
            async with loop.astream("go") as stream:
                async for item in stream:
                    items.append(item)
                result = stream.result
            await loop.aclose()
            return items, result

        items, result = asyncio.run(go())

    assert result.final_text == "done"
    # Real token deltas flowed over HTTP (not just orchestration events).
    assert any(isinstance(item, TextDelta) for item in items)


def test_gateway_adapter_raises_on_mid_stream_error_frame() -> None:
    pytest.importorskip("httpx")

    class BoomAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:  # pragma: no cover
            raise AssertionError("astream_turn should be used")

        async def astream_turn(self, request: ModelRequest):
            yield TextDelta("partial")
            raise ModelAdapterError(
                "provider blew up",
                provider_error_code="gateway_server_error",
                retryable=True,
                http_status=503,
            )

    server, manager = _server_for(lambda *_: BoomAdapter())
    with serving(server) as base_url:
        adapter = _adapter(base_url, _llm_token(manager))
        request = ModelRequest(instruction="go", system_prompt="sys", tools=())
        with pytest.raises(ModelAdapterError) as excinfo:
            asyncio.run(_collect(adapter.astream_turn(request)))
    assert excinfo.value.provider_error_code == "gateway_server_error"
