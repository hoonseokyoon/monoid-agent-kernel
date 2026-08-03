"""P4b-①: LLM-gateway token streaming.

The gateway side (endpoint -> handle_turn_stream -> sync pump -> SSE framing) is verified
with a stdlib urlopen streaming read, so those tests need neither httpx nor an API key. The
adapter side (GatewayModelAdapter.astream_turn) needs httpx and is skipped if it is absent.
Async tests use asyncio.run from sync functions (no pytest-asyncio), matching the suite.
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

from support.http import serving
from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core.spec import AgentRunSpec, ModelConfig, ModelRetryConfig, RunLimits
from monoid_agent_kernel.errors import ModelAdapterError
import monoid_agent_kernel.providers.gateway as gateway_module
from monoid_agent_kernel.providers.gateway import GATEWAY_NETWORK_ERROR
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


# --- the terminal payload a refused stream was already billed for -------------------------
#
# The OpenAI adapter stamps what a refused end-of-turn payload cost (``_terminal_chunk``), and
# every refusal in that region is a RAW ``ValueError``/``AttributeError``: the stream never runs
# the one-shot mapping that raises ``ModelAdapterError``. Both consumers on THIS route read the
# stamp off the escaping exception -- the tenant ledger and the SSE error frame -- and both were
# gated on ``ModelAdapterError``, so a stream whose final payload is malformed charged the tenant
# nothing, told the client nothing, and came back ``retryable=True``: an invitation to buy the
# same tokens again.

_BILLED_TERMINAL_USAGE = {"input_tokens": 120, "output_tokens": 340, "total_tokens": 460}


def _malformed_billed_terminal_payload() -> dict[str, Any]:
    """A ``response.completed`` body that reports what the turn cost and is then refused."""

    return {
        "id": "resp_1",
        "status": "completed",
        "usage": dict(_BILLED_TERMINAL_USAGE),
        # The terminal reader walks ``output`` for tool calls and for the stop reason, so a
        # string there refuses with a raw ``AttributeError`` -- on a turn already generated.
        "output": "not-an-array",
    }


class _BilledTerminalRefusal:
    """An upstream whose stream delivers tokens and then refuses its own final payload."""

    def next_turn(self, request: ModelRequest) -> ModelTurn:  # pragma: no cover - stream only
        raise AssertionError("this upstream is only driven through astream_turn")

    async def astream_turn(self, request: ModelRequest):
        del request
        from monoid_agent_kernel.providers.openai import _terminal_chunk

        yield TextDelta("partial")
        yield _terminal_chunk(_malformed_billed_terminal_payload(), provider_retried=False)


def _billed_refusal_server() -> tuple[Any, TokenManager, LlmGatewayBackend]:
    """The shipped gateway in front of that upstream, with a handle on its tenant ledger."""

    manager = _token_manager()
    gateway = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda *_: _BilledTerminalRefusal(),
    )
    server = create_llm_gateway_server(gateway, host="127.0.0.1", port=0, admin_token="admin")
    return server, manager, gateway


def test_a_stream_refused_on_its_terminal_payload_still_charges_the_tenant() -> None:
    server, manager, gateway = _billed_refusal_server()
    with serving(server) as base_url:
        frames = _post_sse(base_url, _llm_token(manager), _turn_payload())

    errors = [frame for frame in frames if frame["type"] == "error"]
    assert len(errors) == 1, frames
    assert errors[0].get("usage") == _BILLED_TERMINAL_USAGE, {
        "error_frame": errors[0],
        "hint": "the refusal carries what the stream already burned; the frame is the only "
        "carrier a streaming client has left",
    }
    assert gateway.tenant_usage("tenant_a")["total_tokens"] == 460, {
        "tenant_ledger": gateway.tenant_usage("tenant_a"),
        "hint": "the generator exits on the raise, before the success-path meter",
    }


def test_the_client_behind_that_stream_reports_what_the_refusal_cost() -> None:
    """End to end over real HTTP: the frame is read back onto the client's own exception."""

    pytest.importorskip("httpx")
    from monoid_agent_kernel.providers.base import provider_usage_of

    server, manager, gateway = _billed_refusal_server()
    with serving(server) as base_url:
        adapter = _adapter(base_url, _llm_token(manager))
        request = ModelRequest(instruction="go", system_prompt="sys", tools=())
        with pytest.raises(ModelAdapterError) as refused:
            asyncio.run(_collect(adapter.astream_turn(request)))

    assert provider_usage_of(refused.value) == _BILLED_TERMINAL_USAGE
    assert gateway.tenant_usage("tenant_a")["total_tokens"] == 460


# --- the separators an SSE frame must not put on the wire raw -----------------------------
#
# SSE is a LINE protocol, and "line" is not the same word on both ends of it. The frame writer
# serialized with ``ensure_ascii=False``, which leaves U+2028, U+2029 and U+0085 in the body as
# themselves; httpx -- what this repo's own streaming client reads with -- splits ``aiter_lines``
# on all three. The client's parser then sees a JSON object that stops mid-string and reports
# ``gateway_bad_response`` for a turn the server has already produced, already framed and already
# metered. ``final_text`` could always carry one; the relayed ``reasoning`` array made it
# reachable from content that never appears in the answer at all, which is why it is pinned on
# both carriers. Real HTTP on both tests: a hand-fed line list cannot fail this way.

_LINE_SEPARATORS = "\u2028\u2029\u0085"


def _streamed_turn_over_http(chunks: list[Any]) -> Any:
    """Drive the shipped client against the shipped server and assemble what it received."""

    server, manager = _server_for(lambda *_: FakeStreamingModelAdapter(chunk_turns=[chunks]))
    with serving(server) as base_url:
        adapter = _adapter(base_url, _llm_token(manager))
        request = ModelRequest(instruction="go", system_prompt="sys", tools=())
        return assemble_streamed_turn(asyncio.run(_collect(adapter.astream_turn(request))))


def test_a_relayed_reasoning_entry_survives_the_unicode_line_separators() -> None:
    pytest.importorskip("httpx")
    plaintext = f"weighed{_LINE_SEPARATORS}the options"
    reasoning = (
        {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque"},
        {"type": "message", "content": [{"type": "output_text", "text": plaintext}]},
    )
    turn = _streamed_turn_over_http(
        [
            TextDelta("ok"),
            TurnComplete(response_id="prov", usage={"total_tokens": 5}, reasoning=reasoning),
        ]
    )

    assert turn.final_text == "ok"
    assert tuple(turn.reasoning) == reasoning, {
        "relayed": turn.reasoning,
        "hint": "the terminal frame was split mid-JSON on a separator the client calls a line",
    }


def test_final_text_survives_a_unicode_line_separator_on_the_stream() -> None:
    """The carrier that predates the reasoning array, on the delta frames rather than the
    terminal one -- separate frames, same writer, and only one of them was ever exercised."""

    pytest.importorskip("httpx")
    answer = "before\u2028after"
    turn = _streamed_turn_over_http(
        [TextDelta(answer), TurnComplete(response_id="prov", usage={"total_tokens": 5})]
    )
    assert turn.final_text == answer


def test_the_length_delimited_transport_carries_them_as_it_always_did() -> None:
    """The counterweight: the non-streaming body is framed by ``Content-Length``, not by lines.

    Nothing in it can be split by a separator, so it keeps ``ensure_ascii=False`` and the
    smaller body that goes with it -- the escape is a property of the LINE protocol, not of the
    gateway's JSON.
    """

    answer = f"before{_LINE_SEPARATORS}after"
    server, manager = _server_for(
        lambda *_: FakeModelAdapter(
            turns=[ModelTurn(response_id="prov", final_text=answer, usage={"total_tokens": 4})]
        )
    )
    with serving(server) as base_url:
        adapter = _adapter(base_url, _llm_token(manager))
        turn = adapter.next_turn(ModelRequest(instruction="go", system_prompt="sys", tools=()))
    assert turn.final_text == answer


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
    # Which schedule step it asked for, not merely that it waited. The args were already being
    # recorded and never inspected, so shifting the streamed loop's index one step up the curve --
    # real extra seconds per retry -- passed the whole suite.
    assert [args[0] for args in awaited] == [1], (
        "the backoff is indexed by the attempt that just failed"
    )


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


def test_the_streamed_retry_is_reported_before_the_wait_not_after_it(monkeypatch: Any) -> None:
    """The streamed twin of `test_the_retry_is_reported_before_the_wait_not_after_it`.

    That test is named for "the loops" and binds one. The streamed case is the *stronger* one: its
    wait actually yields, so the event loop stays free and the run's own cancel/deadline race can
    fire inside the window. A report issued after the wait is therefore a report that genuinely may
    never happen -- on the blocking path the loop is frozen and the race cannot fire at all.

    Both reports go out before the wait: the channel one and the empty `TextDelta` that carries the
    same fact on the wire, since a stream cancelled mid-backoff never commits a chunk to stamp.
    """
    from monoid_agent_kernel.providers import gateway

    order: list[str] = []

    async def _fake_await(*_args: Any) -> None:
        order.append("wait")

    monkeypatch.setattr(gateway, "_asleep_before_retry", _fake_await)
    monkeypatch.setattr(gateway, "report_provider_retried", lambda: order.append("report"))

    adapter, _attempts = _retried_stream_adapter(
        monkeypatch,
        body=[
            'data: {"type":"text_delta","text":"hi"}',
            "",
            'data: {"type":"turn_complete","response_id":"turn_1"}',
            "",
        ],
    )
    chunks = _stream(adapter)

    assert order == ["report", "wait"], (
        "the streamed loop must report the retry before waiting, not after"
    )
    marker = [c for c in chunks if getattr(c, "provider_retried", False)]
    assert marker, "the wire half of the evidence must also precede the wait"


def _client_stub(httpx: Any, *, fail_at: str) -> Any:
    """An `AsyncClient` whose *own lifecycle* fails, with the stream itself perfectly healthy."""

    class _Response:
        status_code = 200

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def aiter_lines(self):  # noqa: ANN202
            yield 'data: {"type":"text_delta","text":"hi"}'
            yield ""
            yield 'data: {"type":"turn_complete","response_id":"turn_1"}'
            yield ""

    class _Client:
        def __init__(self, **_kwargs: Any) -> None:
            if fail_at == "construct":
                raise httpx.ConnectError("pool construction failed")

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            if fail_at == "close":
                raise httpx.CloseError("pool teardown failed")

        def stream(self, *_args: Any, **_kwargs: Any) -> Any:
            return _Response()

    return _Client


@pytest.mark.parametrize(
    ("fail_at", "retryable", "fragment"),
    [("close", False, "interrupted"), ("construct", True, "connection error")],
)
def test_a_failure_in_the_clients_own_lifecycle_is_still_classified(
    monkeypatch: Any, fail_at: str, retryable: bool, fragment: str
) -> None:
    """Hoisting the client out of the retry loop moved its lifecycle outside the handler.

    Construction, `__aenter__` and the `__aexit__` that tears the pool down used to sit inside the
    per-attempt `try`, so an `httpx` failure from any of them was classified like any other transport
    error. After the hoist only `client.stream(...)` was covered, and those escaped raw.

    Unclassified is not just less descriptive. `AgentLoop._recoverable_turn_error` keys off
    `retryable` and a 4xx `http_status`, neither of which a raw `httpx` error carries, so a failure
    that had ended one turn -- recoverably, session alive, turn re-attemptable -- terminalized the
    whole run and wrote `failure.json` instead. `httpx.CloseError` from a pool teardown is an
    ordinary way in.

    `committed` draws the same line the in-loop handler draws: once deltas have gone out, replaying
    would duplicate them, so a late failure is terminal rather than retryable.
    """
    httpx = pytest.importorskip("httpx")
    monkeypatch.setattr(httpx, "AsyncClient", _client_stub(httpx, fail_at=fail_at))
    monkeypatch.setattr(gateway_module, "_retry_delay", lambda *_a: 0.0)

    adapter = GatewayModelAdapter(
        config=ModelConfig(
            gateway_url="http://gateway.invalid",
            retry=ModelRetryConfig(max_attempts=2),
        ),
        token="t",
    )

    with pytest.raises(ModelAdapterError) as caught:
        _stream(adapter)

    assert caught.value.provider_error_code == GATEWAY_NETWORK_ERROR
    assert caught.value.retryable is retryable
    assert fragment in str(caught.value)


def _token_minting_adapter() -> tuple[GatewayModelAdapter, list[str]]:
    """An adapter whose token is re-minted on every resolution, as a backend's source is.

    `reference.backend.loop_factory._GatewayTokenSource` re-issues once the current token is within
    `refresh_skew_s` of expiry, so a long run sees a *different* token part-way through. Minting on
    every call is the same behaviour with the clock removed.
    """

    minted: list[str] = []

    def provider() -> str:
        minted.append(f"tok-{len(minted) + 1}")
        return minted[-1]

    adapter = GatewayModelAdapter(
        config=ModelConfig(
            gateway_url="http://gateway.invalid",
            retry=ModelRetryConfig(max_attempts=3, initial_delay_s=0, jitter_s=0),
        ),
        token_provider=provider,
    )
    return adapter, minted


def test_both_transports_re_resolve_the_token_on_every_attempt(monkeypatch: Any) -> None:
    """A retry has to carry a fresh credential, and the streamed loop carried the stale one.

    The streamed path resolved its headers once, above the retry loop, next to the URL and the body
    -- and those two genuinely cannot change between attempts. A credential can: the backoff is up
    to `max_delay_s` long, the run may already be minutes old, and a token source that re-mints near
    expiry crosses that line exactly during a wait. Attempt 2 then replayed the expired token and
    came back 401, which is `gateway_auth_error` and not retryable, so the whole call ended
    terminally -- where the blocking loop, which rebuilds its request per attempt, recovered.

    Both halves in one test on purpose: this is a rule *about* the two loops agreeing, and the
    existing `token_provider` test states it by calling `_headers()` twice directly, which no request
    path has to obey. Driving one loop here would leave the claim exactly as unbound as it was.
    """
    httpx = pytest.importorskip("httpx")
    # Stubbed at the schedule, so the one patch covers the blocking wait and the awaited one.
    monkeypatch.setattr(gateway_module, "_retry_delay", lambda *_a: 0.0)
    request = ModelRequest(instruction="go", system_prompt="s", tools=())

    def sent_by_the_blocking_loop() -> list[str]:
        adapter, _minted = _token_minting_adapter()
        sent: list[str] = []
        attempts = {"n": 0}

        class _Body(io.BytesIO):
            def __enter__(self) -> Any:
                return self

            def __exit__(self, *_args: Any) -> None:
                return None

        def _urlopen(http_request: Any, timeout: float | None = None) -> Any:
            sent.append(http_request.headers["Authorization"])
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise URLError("reset before the response")
            return _Body(json.dumps({"final_text": "hi", "turn_handle": "t1"}).encode("utf-8"))

        monkeypatch.setattr(gateway_module, "urlopen", _urlopen)
        adapter.next_turn(request)
        return sent

    def sent_by_the_streaming_loop() -> list[str]:
        adapter, _minted = _token_minting_adapter()
        sent: list[str] = []
        attempts = {"n": 0}

        class _Response:
            status_code = 200

            async def __aenter__(self) -> Any:
                return self

            async def __aexit__(self, *_args: Any) -> None:
                return None

            async def aiter_lines(self):  # noqa: ANN202
                yield 'data: {"type":"turn_complete","response_id":"turn_1"}'
                yield ""

        class _Client:
            def __init__(self, **_kwargs: Any) -> None:
                return None

            async def __aenter__(self) -> Any:
                return self

            async def __aexit__(self, *_args: Any) -> None:
                return None

            def stream(self, *_args: Any, **kwargs: Any) -> Any:
                sent.append(kwargs["headers"]["Authorization"])
                attempts["n"] += 1
                if attempts["n"] == 1:
                    raise httpx.HTTPError("reset before the stream committed")
                return _Response()

        monkeypatch.setattr(httpx, "AsyncClient", _Client)
        _stream(adapter)
        return sent

    fresh_on_the_retry = ["Bearer tok-1", "Bearer tok-2"]
    assert sent_by_the_blocking_loop() == fresh_on_the_retry
    assert sent_by_the_streaming_loop() == fresh_on_the_retry, (
        "the streamed loop resolved the token once per call and replayed it on the retry"
    )


def test_both_transports_honour_retry_on_for_a_transport_failure(monkeypatch: Any) -> None:
    """`retry_on` is a policy, and a transport failure is not exempt from it on either loop.

    Both loops classify a connection-level failure as retryable and then ask `_should_retry`, which
    also requires the code to be listed in `retry_on`. That second question was unobserved on both
    sides: a mutant that drops the gate and always loops still makes `max_attempts` attempts under
    the default policy, so nothing separated it -- the difference only shows when the policy excludes
    the code, which is precisely when a caller is relying on it.

    `retry_on=("gateway_timeout",)` keeps timeouts retryable and takes `gateway_network_error` out,
    so one attempt is the whole call.
    """
    httpx = pytest.importorskip("httpx")
    monkeypatch.setattr(gateway_module, "_retry_delay", lambda *_a: 0.0)
    config = ModelConfig(
        gateway_url="http://gateway.invalid",
        retry=ModelRetryConfig(
            max_attempts=3, initial_delay_s=0, jitter_s=0, retry_on=("gateway_timeout",)
        ),
    )
    adapter = GatewayModelAdapter(config=config, token="t")
    request = ModelRequest(instruction="go", system_prompt="s", tools=())

    def blocking_attempts() -> int:
        attempts = {"n": 0}

        def _urlopen(_http_request: Any, timeout: float | None = None) -> Any:
            attempts["n"] += 1
            raise URLError("connection refused")

        monkeypatch.setattr(gateway_module, "urlopen", _urlopen)
        with pytest.raises(ModelAdapterError) as caught:
            adapter.next_turn(request)
        assert caught.value.provider_error_code == GATEWAY_NETWORK_ERROR
        return attempts["n"]

    def streaming_attempts() -> int:
        attempts = {"n": 0}

        class _Client:
            def __init__(self, **_kwargs: Any) -> None:
                return None

            async def __aenter__(self) -> Any:
                return self

            async def __aexit__(self, *_args: Any) -> None:
                return None

            def stream(self, *_args: Any, **_kwargs: Any) -> Any:
                attempts["n"] += 1
                raise httpx.HTTPError("connection refused")

        monkeypatch.setattr(httpx, "AsyncClient", _Client)
        with pytest.raises(ModelAdapterError) as caught:
            _stream(adapter)
        assert caught.value.provider_error_code == GATEWAY_NETWORK_ERROR
        return attempts["n"]

    assert blocking_attempts() == 1, "the blocking loop retried a code the policy excludes"
    assert streaming_attempts() == 1, "the streamed loop retried a code the policy excludes"


def _status_client(httpx: Any, status: int, detail: str) -> tuple[Any, dict[str, int]]:
    """An `AsyncClient` whose stream always answers `status`, counting the attempts."""

    attempts = {"n": 0}

    class _Response:
        def __init__(self) -> None:
            self.status_code = status

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def aread(self) -> bytes:
            return detail.encode("utf-8")

        async def aiter_lines(self):  # noqa: ANN202 - unreachable: status is never 200 here
            raise AssertionError("a non-200 response must not be streamed")

    class _Client:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        def stream(self, *_args: Any, **_kwargs: Any) -> Any:
            attempts["n"] += 1
            return _Response()

    return _Client, attempts


@pytest.mark.parametrize(
    ("status", "expected_attempts", "code", "retried"),
    [(401, 1, "gateway_auth_error", False), (503, 3, "gateway_server_error", True)],
)
def test_the_streamed_loop_retries_on_the_gate_not_on_the_attempt_budget(
    monkeypatch: Any, status: int, expected_attempts: int, code: str, retried: bool
) -> None:
    """A non-200 is retried only if `_should_retry` says so, and a 401 never does.

    The streamed loop's decision was unbound: a mutant that dropped the gate and kept only
    `attempt < max_attempts` hammered a 401 three times and the whole suite stayed green. Two
    independent reasons say no there -- `retryable` is false for a 4xx, and `gateway_auth_error` is
    not in `retry_on` -- and neither was observed on this path. The 503 case holds the other side of
    the gate open, so a mutant that simply never retries is caught too.

    The escaping error's retry stamp rides along: three attempts exhausting the budget raise from
    inside the loop, and the receipt that failure becomes must not describe a thrice-tried call as a
    clean single attempt. The 401 is the counterweight -- one attempt, nothing to report.
    """
    httpx = pytest.importorskip("httpx")
    client, attempts = _status_client(httpx, status, json.dumps({"error": f"HTTP {status}"}))
    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setattr(gateway_module, "_retry_delay", lambda *_a: 0.0)

    adapter = GatewayModelAdapter(
        config=ModelConfig(
            gateway_url="http://gateway.invalid",
            retry=ModelRetryConfig(max_attempts=3, initial_delay_s=0, jitter_s=0),
        ),
        token="t",
    )

    with pytest.raises(ModelAdapterError) as caught:
        _stream(adapter)

    assert attempts["n"] == expected_attempts
    assert caught.value.provider_error_code == code
    assert caught.value.http_status == status
    assert getattr(caught.value, "provider_retried", False) is retried


def test_a_client_lifecycle_failure_after_a_retry_still_reports_the_retry(monkeypatch: Any) -> None:
    """The retry stamp has to survive the *other* boundary too.

    `test_a_failure_in_the_clients_own_lifecycle_is_still_classified` proves the client-lifecycle
    handler classifies what escapes it, on a call that never retried. Its `_stamp_retry` was
    unobserved: a receipt built from a pool teardown that failed *after* a retry reported a clean
    single attempt, and this is the one boundary where no chunk and no turn survives to carry the
    fact instead.
    """
    httpx = pytest.importorskip("httpx")
    monkeypatch.setattr(gateway_module, "_retry_delay", lambda *_a: 0.0)
    attempts = {"n": 0}

    class _Response:
        status_code = 200

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def aiter_lines(self):  # noqa: ANN202
            yield 'data: {"type":"turn_complete","response_id":"turn_1"}'
            yield ""

    class _Client:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            raise httpx.CloseError("pool teardown failed")

        def stream(self, *_args: Any, **_kwargs: Any) -> Any:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise httpx.HTTPError("reset before the stream committed")
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    adapter = GatewayModelAdapter(
        config=ModelConfig(
            gateway_url="http://gateway.invalid",
            retry=ModelRetryConfig(max_attempts=3, initial_delay_s=0, jitter_s=0),
        ),
        token="t",
    )

    with pytest.raises(ModelAdapterError) as caught:
        _stream(adapter)

    assert attempts["n"] == 2, "the teardown must fail on a call that had already retried"
    assert caught.value.provider_error_code == GATEWAY_NETWORK_ERROR
    assert getattr(caught.value, "provider_retried", False) is True


def test_neither_loop_assigns_its_own_retry_verdict_over_the_wires(monkeypatch: Any) -> None:
    """Two retry loops sit on this path and a receipt records that *either* one ran.

    The gateway's backend can retry a request this client got right the first time, so a client that
    wrote its own verdict over the wire's turned that into a clean single attempt. Both loops combine
    instead -- and both were unbound: the test that carries this rule in its name calls
    `_parse_gateway_response` directly, which is the parser, not the loop that decides what to do
    with what it returns.

    First attempt succeeds in both halves, so `attempt > 1` is false and only the wire's fact is
    left to carry.
    """
    httpx = pytest.importorskip("httpx")
    adapter = GatewayModelAdapter(
        config=ModelConfig(gateway_url="http://gateway.invalid"), token="t"
    )

    class _Body(io.BytesIO):
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

    def _urlopen(_request: Any, timeout: float | None = None) -> Any:
        payload = {"final_text": "hi", "turn_handle": "t1", "provider_retried": True}
        return _Body(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(gateway_module, "urlopen", _urlopen)

    class _Response:
        status_code = 200

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def aiter_lines(self):  # noqa: ANN202
            yield 'data: {"type":"text_delta","text":"hi","provider_retried":true}'
            yield ""
            yield 'data: {"type":"turn_complete","turn_handle":"t1","provider_retried":true}'
            yield ""

    class _Client:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        def stream(self, *_args: Any, **_kwargs: Any) -> Any:
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    request = ModelRequest(instruction="go", system_prompt="s", tools=())

    blocking = adapter.next_turn(request)
    streamed = _stream(adapter)

    assert blocking.provider_retried is True, "the blocking loop overwrote the gateway's own retry"
    assert [chunk.provider_retried for chunk in streamed] == [True, True], (
        "the streamed loop overwrote the gateway's own retry"
    )
    assert assemble_streamed_turn(streamed).provider_retried is True
