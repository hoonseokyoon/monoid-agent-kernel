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

from monoid_agent_kernel.core.spec import AgentRunSpec, ModelConfig, ModelRetryConfig, RunLimits
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
        "protocol": "monoid.llm-turn.v1",
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


def _retried_stream_adapter(monkeypatch, *, body: list[str]) -> Any:
    """A gateway whose first attempt fails pre-commit and whose second commits ``body``.

    Returns the adapter and a mutable ``attempts`` counter; set ``attempts["n"] = 1`` before a
    second call to get a stream that commits first time.
    """
    httpx = pytest.importorskip("httpx")
    attempts = {"n": 0}

    class _Response:
        status_code = 200

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def aiter_lines(self):
            for line in body:
                yield line

    class _Client:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        def stream(self, *_args: Any, **_kwargs: Any) -> Any:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise httpx.HTTPError("connection reset before the stream committed")
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    # Stubbed at the schedule, not at either sleeper: one patch then covers the blocking wait and
    # the awaited one, so a test cannot start sleeping for real because a path switched sleepers.
    monkeypatch.setattr("monoid_agent_kernel.providers.gateway._retry_delay", lambda *_a: 0.0)
    adapter = GatewayModelAdapter(
        ModelConfig(
            gateway_url="http://gateway.local/internal/llm/turns",
            retry=ModelRetryConfig(max_attempts=3, initial_delay_s=0, jitter_s=0),
        ),
        token="run-token",
    )
    return adapter, attempts


def _stream(adapter: GatewayModelAdapter) -> list[Any]:
    request = ModelRequest(instruction="go", system_prompt="s", tools=())
    return asyncio.run(_collect(adapter.astream_turn(request)))


def test_gateway_marks_a_retried_stream_from_its_first_chunk(monkeypatch) -> None:
    """Retry evidence has to reach the wire before the terminal chunk does.

    Stream retries are all pre-commit, so the adapter knows it retried the moment the stream
    commits. Attaching that fact to ``TurnComplete`` alone lost it whenever a call was cancelled or
    aborted mid-stream: the terminal chunk never arrived, and the failure receipt then reported no
    retry for a call the gateway had demonstrably retried.
    """
    adapter, attempts = _retried_stream_adapter(
        monkeypatch,
        body=[
            'data: {"type":"text_delta","text":"hi"}',
            "",
            'data: {"type":"turn_complete","response_id":"turn_1"}',
            "",
        ],
    )
    chunks = _stream(adapter)

    assert attempts["n"] == 2
    assert all(chunk.provider_retried for chunk in chunks)
    # The marker leads and contributes no text, so the assembled turn is what it always was.
    assert isinstance(chunks[0], TextDelta) and chunks[0].text == ""
    assert assemble_streamed_turn(chunks).final_text == "hi"

    # Counterweight: a stream that commits first time marks nothing and gains no marker chunk.
    attempts["n"] = 1
    clean = _stream(adapter)
    assert [chunk.provider_retried for chunk in clean] == [False, False]


def test_gateway_marks_a_retried_stream_that_carries_no_chunks(monkeypatch) -> None:
    """The body is not a carrier anything can count on.

    Marking the chunks was still content-dependent: a 200 stream can yield no recognized chunk at
    all -- an empty body, or, as here, only frames this version forwards past -- and then there was
    nothing to stamp and the turn reported the opposite of what happened. Emitting the evidence at
    commit ties it to the fact rather than to the payload.
    """
    adapter, attempts = _retried_stream_adapter(
        monkeypatch, body=['data: {"type":"some_future_frame","v":1}', ""]
    )
    chunks = _stream(adapter)

    assert attempts["n"] == 2
    assert assemble_streamed_turn(chunks).provider_retried is True

    # Counterweight: the same chunkless stream without a retry claims none.
    attempts["n"] = 1
    assert assemble_streamed_turn(_stream(adapter)).provider_retried is False


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


def test_the_streamed_backoff_waits_without_holding_the_event_loop(monkeypatch: Any) -> None:
    """The streamed retry path must not use the blocking wait.

    It did, and the blocking sleep is called from inside an async generator: the whole event loop
    stopped for the length of the backoff -- up to ``max_delay_s`` per retry, 4.5s at the default
    policy. Nothing else in the run progressed, and the run's own cancel/deadline race lives on that
    loop, so a run told to stop kept waiting for a provider it had already given up on.

    Asserted on which wait the path took rather than on elapsed time, so it cannot go flaky on a
    loaded machine while still pinning the property exactly.
    """
    from monoid_agent_kernel.providers import gateway

    blocking: list[Any] = []
    awaited: list[Any] = []

    async def _fake_await(*args: Any) -> None:
        awaited.append(args)

    monkeypatch.setattr(gateway, "_sleep_before_retry", lambda *args: blocking.append(args))
    monkeypatch.setattr(gateway, "_asleep_before_retry", _fake_await)

    adapter, _attempts = _retried_stream_adapter(
        monkeypatch,
        body=[
            'data: {"type":"text_delta","text":"hi"}',
            "",
            'data: {"type":"turn_complete","response_id":"turn_1"}',
            "",
        ],
    )
    _stream(adapter)

    assert awaited, "the streamed retry must wait on the event loop, not block it"
    assert blocking == [], "the streamed retry must not use the blocking wait"


def test_the_async_backoff_lets_other_tasks_run() -> None:
    """And the awaited wait really does yield -- the counterweight to the test above.

    Checking only *which* function the path calls would pass even if that function blocked.
    """
    from monoid_agent_kernel.providers import gateway

    async def run() -> list[int]:
        ticks: list[int] = []

        async def tick() -> None:
            for _ in range(5):
                await asyncio.sleep(0)
                ticks.append(1)

        task = asyncio.create_task(tick())
        await gateway._asleep_before_retry(1, 0.05, 1.0, 1.0, 0.0)
        task.cancel()
        return ticks

    assert len(asyncio.run(run())) == 5


def test_both_waits_follow_one_schedule() -> None:
    """The sync and streamed paths must not drift apart on backoff policy."""
    from monoid_agent_kernel.providers import gateway

    args = (2, 0.5, 4.0, 2.0, 0.0)
    assert gateway._retry_delay(*args) == 1.0
    # Capped, and the cap applies to both because both read the same schedule.
    assert gateway._retry_delay(9, 0.5, 4.0, 2.0, 0.0) == 4.0
