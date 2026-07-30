from __future__ import annotations

import asyncio
import contextlib
import json
import os
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from support.http import serving
from support.runtime import runtime_config, tool_binding

from monoid_agent_kernel.cli import main
from monoid_agent_kernel.core.spec import ModelConfig, ReasoningConfig
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelTurn,
    ReasoningDelta,
    TextDelta,
    TurnComplete,
    assemble_streamed_turn,
)
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.providers.openai import (
    OpenAIModelAdapter,
    _capture_reasoning_items,
    _parse_response,
    _reasoning_replay_flags,
)


def _openai_responses_available() -> bool:
    try:
        from openai import OpenAI
    except ImportError:
        return False
    return hasattr(OpenAI(api_key="test"), "responses")


def _write_config(path: Path, *tool_ids: str, model: ModelConfig | None = None) -> Path:
    path.write_text(
        json.dumps(runtime_config(*tool_ids, model=model).to_json()),
        encoding="utf-8",
    )
    return path


def test_cli_requires_runtime_config(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = CliRunner().invoke(main, ["run", "--workspace", str(workspace), "--instruction", "Finish."])

    assert result.exit_code != 0
    assert "--runtime-config-file or --agent-definition-file is required" in result.output


def test_cli_run_accepts_runtime_config_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    monkeypatch.setattr("monoid_agent_kernel.cli._model_adapter", lambda *_args, **_kwargs: adapter)
    config_file = _write_config(tmp_path / "runtime.json", "fs.read", "run.finish")

    result = CliRunner().invoke(
        main,
        [
            "run",
            "--workspace",
            str(workspace),
            "--instruction",
            "Finish.",
            "--run-root",
            str(tmp_path / "runs"),
            "--runtime-config-file",
            str(config_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert {tool.id for tool in adapter.requests[0].tools} == {"fs.read", "run.finish"}
    run_id = next(line for line in result.output.splitlines() if line.startswith("run_id: ")).removeprefix("run_id: ")
    manifest = json.loads((tmp_path / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["agent_config"]["definition_id"] == "test-agent"


def test_cli_auto_grant_capabilities_gates_tool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.md").write_text("hi\n", encoding="utf-8")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("fs_read", {"path": "notes.md"}, "c1"),)),
            ModelTurn(response_id="r2", final_text="done"),
        ]
    )
    monkeypatch.setattr("monoid_agent_kernel.cli._model_adapter", lambda *_a, **_k: adapter)
    binding = tool_binding("fs.read", runtime={"requires_lease": True})
    config_file = tmp_path / "runtime.json"
    config_file.write_text(
        json.dumps(runtime_config(bindings=(binding, tool_binding("run.finish"))).to_json()),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        [
            "run", "--workspace", str(workspace), "--instruction", "go",
            "--run-root", str(tmp_path / "runs"), "--runtime-config-file", str(config_file),
            "--auto-grant-capabilities",
        ],
    )

    assert result.exit_code == 0, result.output
    run_id = next(line for line in result.output.splitlines() if line.startswith("run_id: ")).removeprefix("run_id: ")
    events = (tmp_path / "runs" / run_id / "events.jsonl").read_text(encoding="utf-8")
    assert "capability.granted" in events  # the broker was wired and gated the requires_lease tool


def test_cli_capability_flags_are_mutually_exclusive(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        "monoid_agent_kernel.cli._model_adapter",
        lambda *_a, **_k: FakeModelAdapter(turns=[ModelTurn(final_text="done")]),
    )
    config_file = _write_config(tmp_path / "runtime.json", "run.finish")
    result = CliRunner().invoke(
        main,
        [
            "run", "--workspace", str(workspace), "--instruction", "go",
            "--run-root", str(tmp_path / "runs"), "--runtime-config-file", str(config_file),
            "--auto-grant-capabilities", "--capability-broker", "x.py:make",
        ],
    )
    assert result.exit_code != 0
    assert "not both" in result.output


def test_cli_spec_file_pairs_with_runtime_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(
        json.dumps({"workspace_root": str(workspace), "run_root": str(tmp_path / "runs")}),
        encoding="utf-8",
    )
    config_file = _write_config(tmp_path / "runtime.json", "run.finish")
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    monkeypatch.setattr("monoid_agent_kernel.cli._model_adapter", lambda *_args, **_kwargs: adapter)

    result = CliRunner().invoke(
        main,
        ["run", "--spec", str(spec_file), "--instruction", "Finish.", "--runtime-config-file", str(config_file)],
    )

    assert result.exit_code == 0, result.output
    assert {tool.id for tool in adapter.requests[0].tools} == {"run.finish"}


def test_cli_permission_policy_flags_remain_run_boundary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    monkeypatch.setattr("monoid_agent_kernel.cli._model_adapter", lambda *_args, **_kwargs: adapter)
    config_file = _write_config(tmp_path / "runtime.json", "run.finish")

    result = CliRunner().invoke(
        main,
        [
            "run",
            "--workspace",
            str(workspace),
            "--instruction",
            "Finish.",
            "--run-root",
            str(tmp_path / "runs"),
            "--runtime-config-file",
            str(config_file),
            "--deny-path",
            ".env",
            "--redact-path",
            "*.key",
        ],
    )

    assert result.exit_code == 0, result.output
    run_id = next(line for line in result.output.splitlines() if line.startswith("run_id: ")).removeprefix("run_id: ")
    manifest = json.loads((tmp_path / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["permission_policy"] == {
        "deny_patterns": [".env"],
        "redact_patterns": ["*.key"],
    }


@pytest.mark.parametrize("option", ["--deny-path", "--redact-path"])
def test_cli_rejects_negated_permission_patterns(option: str, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_file = _write_config(tmp_path / "runtime.json", "run.finish")

    result = CliRunner().invoke(
        main,
        [
            "run",
            "--workspace",
            str(workspace),
            "--instruction",
            "Finish.",
            "--runtime-config-file",
            str(config_file),
            option,
            "!secrets/**",
        ],
    )

    assert result.exit_code != 0
    assert "negated path patterns are not supported" in result.output


def test_cli_requires_web_gateway_for_web_bindings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    monkeypatch.setattr("monoid_agent_kernel.cli._model_adapter", lambda *_args, **_kwargs: adapter)
    config_file = _write_config(tmp_path / "runtime.json", "web.search", "run.finish")

    result = CliRunner().invoke(
        main,
        [
            "run",
            "--workspace",
            str(workspace),
            "--instruction",
            "Finish.",
            "--runtime-config-file",
            str(config_file),
        ],
    )

    assert result.exit_code != 0
    assert "runtime config binds web tools; --web-gateway-url is required" in result.output


def test_openai_payload_uses_turn_model_config() -> None:
    adapter = OpenAIModelAdapter(ModelConfig(model="fallback"))
    request = ModelRequest(
        instruction="hello",
        system_prompt="sys",
        tools=(),
        model=ModelConfig(model="gpt-5.5", reasoning=ReasoningConfig(effort="high", summary="detailed")),
    )

    payload = adapter._payload(request)

    assert payload["model"] == "gpt-5.5"
    assert payload["reasoning"] == {"effort": "high", "summary": "detailed"}


# --- DX-13a: faithful OpenAI reasoning round-trip (ZDR) --------------------------------------

_RS_A = {"type": "reasoning", "id": "rs_a", "summary": [], "encrypted_content": "enc_a"}
_RS_B = {"type": "reasoning", "id": "rs_b", "summary": [], "encrypted_content": "enc_b"}
_FC_A = {"type": "function_call", "call_id": "c_a", "name": "fs_read", "arguments": "{}"}
_FC_B = {"type": "function_call", "call_id": "c_b", "name": "text_search", "arguments": "{}"}


def _assistant_with_reasoning(model: str, items: list[dict], tool_calls: list[dict]) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": tool_calls,
        "reasoning": {"provider": "openai", "model": model, "items": items},
    }


def test_openai_payload_sets_zdr_store_and_include() -> None:
    # ZDR round-trip: never persist server-side, and ask for encrypted reasoning so it can
    # travel by-value. (decision #1)
    adapter = OpenAIModelAdapter(ModelConfig(model="gpt-5.5"))
    payload = adapter._payload(ModelRequest(instruction="hi", system_prompt="", tools=()))

    assert payload["store"] is False
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert "previous_response_id" not in payload


def test_openai_parse_captures_reasoning_subsequence_verbatim() -> None:
    # The reasoning/function_call/message subsequence is captured in order, verbatim; tool_calls
    # are still parsed independently.
    msg = {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}
    data = {"id": "resp1", "output": [_RS_A, _FC_A, _RS_B, _FC_B, msg], "usage": {}}

    turn = _parse_response(data)

    assert turn.reasoning == (_RS_A, _FC_A, _RS_B, _FC_B, msg)
    assert tuple(c.id for c in turn.tool_calls) == ("c_a", "c_b")
    assert turn.final_text == "ok"


class _Ev:
    def __init__(self, type: str, **kw) -> None:  # noqa: A002, ANN003
        self.type = type
        for k, v in kw.items():
            setattr(self, k, v)


class _StreamResp:
    def model_dump(self) -> dict:
        return {"id": "r1", "usage": {}, "output": []}


class _AsyncStream:
    def __init__(self, events: list) -> None:
        self._events = events

    def __aiter__(self):  # noqa: ANN204
        self._it = iter(self._events)
        return self

    async def __anext__(self):  # noqa: ANN204
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


class _StubClient:
    """Stands in for an SDK client: ``responses``, ``close()`` and ``is_closed()``.

    The close half is part of the contract, not decoration -- it is how the adapter releases the
    connection pool, and it is what the scope calls at its end. Two of these stubs used to model
    only ``responses``, and the tests passed while every pool the adapter opened was leaked.
    ``close()`` is sync here and awaited in :class:`_StubAsyncClient`, mirroring the SDK's split.
    """

    def __init__(self, responses: Any, close_error: Exception | None = None) -> None:
        self.responses = responses
        self.closed = False
        self._close_error = close_error

    def is_closed(self) -> bool:
        return self.closed

    def _mark_closed(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error

    def close(self) -> None:
        self._mark_closed()


class _StubAsyncClient(_StubClient):
    async def close(self) -> None:  # type: ignore[override]
        self._mark_closed()


def _stub_openai(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    responses: Any,
    *,
    close_error: Exception | None = None,
) -> list[_StubClient]:
    """Patch ``openai.OpenAI``/``openai.AsyncOpenAI`` to hand out a stub client.

    Returns the list the adapter's constructions land in. ``close_error`` makes the teardown
    itself fail, which is a failure surface the adapter owns and has to classify.
    """
    stub = _StubAsyncClient if attribute == "AsyncOpenAI" else _StubClient
    built: list[_StubClient] = []

    def _build(**_kwargs: Any) -> _StubClient:
        built.append(stub(responses, close_error))
        return built[-1]

    monkeypatch.setattr(f"openai.{attribute}", _build)
    return built


def _async_stream_responses(events: list) -> Any:
    class _Responses:
        async def create(self, **kwargs):  # noqa: ANN003, ANN202
            return _AsyncStream(events)

    return _Responses()


def test_openai_astream_yields_reasoning_delta_then_text(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")  # exercises the real SDK stream; skip on a minimal install
    # DX-13b: a reasoning-summary stream event maps to a ReasoningDelta, distinct from the
    # answer's TextDelta, ahead of the terminal TurnComplete.
    events = [
        _Ev("response.reasoning_summary_text.delta", delta="think"),
        _Ev("response.output_text.delta", delta="Hi"),
        _Ev("response.completed", response=_StreamResp()),
    ]

    _stub_openai(monkeypatch, "AsyncOpenAI", _async_stream_responses(events))
    adapter = OpenAIModelAdapter(ModelConfig(model="gpt-5.5"), api_key="test", allow_direct_provider_api=True)
    request = ModelRequest(instruction="hi", system_prompt="", tools=())

    async def _drain() -> list:
        return [chunk async for chunk in adapter.astream_turn(request)]

    chunks = asyncio.run(_drain())

    assert isinstance(chunks[0], ReasoningDelta) and chunks[0].text == "think"
    assert isinstance(chunks[1], TextDelta) and chunks[1].text == "Hi"
    assert isinstance(chunks[-1], TurnComplete)


def test_openai_astream_captures_incomplete_stop_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    # A stream that ends with response.incomplete (max_output_tokens) must surface stop_reason
    # "length", not a normal "stop" — else a validator would re-prompt a truncated partial answer.
    class _IncompleteResp:
        def model_dump(self) -> dict:
            return {
                "id": "r1",
                "usage": {},
                "output": [],
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            }

    events = [
        _Ev("response.output_text.delta", delta="partial"),
        _Ev("response.incomplete", response=_IncompleteResp()),
    ]

    _stub_openai(monkeypatch, "AsyncOpenAI", _async_stream_responses(events))
    adapter = OpenAIModelAdapter(ModelConfig(model="gpt-5.5"), api_key="test", allow_direct_provider_api=True)
    request = ModelRequest(instruction="hi", system_prompt="", tools=())

    async def _drain() -> list:
        return [chunk async for chunk in adapter.astream_turn(request)]

    chunks = asyncio.run(_drain())
    assert isinstance(chunks[-1], TurnComplete)
    assert chunks[-1].stop_reason == "length"


def test_openai_capture_strips_output_only_status() -> None:
    # The Responses *input* schema rejects the output-only `status` field
    # (Unknown parameter: input[..].status), so it must be dropped on capture.
    out = [
        {"type": "reasoning", "id": "rs_x", "encrypted_content": "e", "status": "completed"},
        {"type": "function_call", "call_id": "c1", "name": "fs_read", "arguments": "{}", "status": "completed"},
    ]
    captured = _capture_reasoning_items(out)
    assert all("status" not in item for item in captured)
    assert captured[0] == {"type": "reasoning", "id": "rs_x", "encrypted_content": "e"}


def test_openai_capture_empty_without_reasoning() -> None:
    # A non-reasoning turn (no reasoning item) captures nothing — the neutral seam.
    assert _capture_reasoning_items([_FC_A, {"type": "message", "content": []}]) == ()


def test_openai_stream_carries_reasoning_off_final_chunk() -> None:
    turn = assemble_streamed_turn(
        [TextDelta("hi"), TurnComplete(response_id="r1", reasoning=(_RS_A, _FC_A))]
    )
    assert turn.reasoning == (_RS_A, _FC_A)


def test_openai_reasoning_roundtrips_verbatim_in_active_window() -> None:
    # messages = [user, assistant(reasoning+tool_call), tool]. The reasoning item is re-injected
    # immediately followed by its function_call, verbatim — and the reconstructed function_call
    # is suppressed (no duplicate).
    messages = (
        {"role": "user", "content": "go"},
        _assistant_with_reasoning("gpt-5.5", [_RS_A, _FC_A], [{"id": "c_a", "name": "fs_read", "arguments": {}}]),
        {"role": "tool", "call_id": "c_a", "content": {"ok": True}},
    )
    adapter = OpenAIModelAdapter(ModelConfig(model="fallback"))
    payload = adapter._payload(
        ModelRequest(instruction=None, system_prompt="", tools=(), model=ModelConfig(model="gpt-5.5"), messages=messages)
    )

    items = payload["input"]
    fc_items = [it for it in items if it.get("type") == "function_call"]
    assert fc_items == [_FC_A]  # exactly the verbatim one, no reconstruction
    reasoning_idx = items.index(_RS_A)
    assert items[reasoning_idx + 1] == _FC_A  # adjacency preserved
    # function_call_output for the same call is still present.
    assert any(it.get("type") == "function_call_output" and it.get("call_id") == "c_a" for it in items)


def test_openai_reasoning_parallel_interleave_preserved() -> None:
    items = [_RS_A, _FC_A, _RS_B, _FC_B]
    messages = (
        {"role": "user", "content": "go"},
        _assistant_with_reasoning(
            "gpt-5.5",
            items,
            [{"id": "c_a", "name": "fs_read", "arguments": {}}, {"id": "c_b", "name": "text_search", "arguments": {}}],
        ),
        {"role": "tool", "call_id": "c_a", "content": {"ok": True}},
        {"role": "tool", "call_id": "c_b", "content": {"ok": True}},
    )
    adapter = OpenAIModelAdapter(ModelConfig(model="gpt-5.5"))
    payload = adapter._payload(ModelRequest(instruction=None, system_prompt="", tools=(), messages=messages))

    emitted = [it for it in payload["input"] if it.get("type") in {"reasoning", "function_call"}]
    assert emitted == items  # exact interleaved order


def test_openai_reasoning_dropped_on_model_mismatch() -> None:
    # A hot-swap to a different model invalidates the captured reasoning → drop it (and fall back
    # to a reconstructed function_call), never send a half-paired set.
    messages = (
        {"role": "user", "content": "go"},
        _assistant_with_reasoning("gpt-5.5", [_RS_A, _FC_A], [{"id": "c_a", "name": "fs_read", "arguments": {}}]),
        {"role": "tool", "call_id": "c_a", "content": {"ok": True}},
    )
    adapter = OpenAIModelAdapter(ModelConfig(model="gpt-4o"))  # different model
    payload = adapter._payload(ModelRequest(instruction=None, system_prompt="", tools=(), messages=messages))

    assert not any(it.get("type") == "reasoning" for it in payload["input"])
    fc_items = [it for it in payload["input"] if it.get("type") == "function_call"]
    assert fc_items == [{"type": "function_call", "call_id": "c_a", "name": "fs_read", "arguments": "{}"}]


def test_openai_reasoning_historical_dropped_only_active_window_replayed() -> None:
    # Two user turns; only the reasoning since the last user message (asstB) is replayed.
    messages = (
        {"role": "user", "content": "u1"},
        _assistant_with_reasoning("gpt-5.5", [_RS_A, _FC_A], [{"id": "c_a", "name": "fs_read", "arguments": {}}]),
        {"role": "tool", "call_id": "c_a", "content": {"ok": True}},
        {"role": "user", "content": "u2"},
        _assistant_with_reasoning("gpt-5.5", [_RS_B, _FC_B], [{"id": "c_b", "name": "text_search", "arguments": {}}]),
        {"role": "tool", "call_id": "c_b", "content": {"ok": True}},
    )
    # Active window = everything after the last user message (index 3): asstB + its tool result.
    # The tool message's flag is irrelevant (only the assistant branch reads it).
    flags = _reasoning_replay_flags(messages, "gpt-5.5")
    assert flags == [False, False, False, False, True, True]

    adapter = OpenAIModelAdapter(ModelConfig(model="gpt-5.5"))
    payload = adapter._payload(ModelRequest(instruction=None, system_prompt="", tools=(), messages=messages))
    reasoning_items = [it for it in payload["input"] if it.get("type") == "reasoning"]
    assert reasoning_items == [_RS_B]  # rs_a is historical, dropped


def test_openai_reasoning_all_or_nothing_on_mixed_active_window() -> None:
    # If ANY active-window block mismatches, drop reasoning for the WHOLE window.
    messages = (
        {"role": "user", "content": "go"},
        _assistant_with_reasoning("gpt-5.5", [_RS_A, _FC_A], [{"id": "c_a", "name": "fs_read", "arguments": {}}]),
        {"role": "tool", "call_id": "c_a", "content": {"ok": True}},
        _assistant_with_reasoning("gpt-4o", [_RS_B, _FC_B], [{"id": "c_b", "name": "text_search", "arguments": {}}]),
        {"role": "tool", "call_id": "c_b", "content": {"ok": True}},
    )
    assert _reasoning_replay_flags(messages, "gpt-5.5") == [False, False, False, False, False]


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or not _openai_responses_available(),
    reason="OPENAI_API_KEY or OpenAI Responses SDK support not available",
)
def test_openai_smoke_payload_only() -> None:
    adapter = OpenAIModelAdapter(ModelConfig(), allow_direct_provider_api=True)
    request = ModelRequest(instruction="Say ok.", system_prompt="sys", tools=())

    payload = adapter._payload(request)

    assert payload["input"] == [{"role": "user", "content": "Say ok."}]


def test_openai_adapter_maps_provider_400_to_model_adapter_error(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")  # uses real SDK error types; skip on a minimal install
    # A provider 400 (e.g. unsupported reasoning effort) must surface as a classified, non-
    # retryable ModelAdapterError carrying http_status — NOT a raw SDK error (which the gateway
    # would mistranslate to a retryable 500). The message must not echo the request body.
    class _FakeBadRequest(Exception):
        def __init__(self) -> None:
            super().__init__("Unsupported value: 'minimal' is not supported with 'gpt-5.5'.")
            self.status_code = 400
            self.body = {"code": "unsupported_value"}

    class _FakeResponses:
        def create(self, **kwargs):  # noqa: ANN003
            raise _FakeBadRequest()

    built = _stub_openai(monkeypatch, "OpenAI", _FakeResponses())
    adapter = OpenAIModelAdapter(ModelConfig(), api_key="test", allow_direct_provider_api=True)
    request = ModelRequest(instruction="my secret prompt", system_prompt="", tools=())
    with pytest.raises(ModelAdapterError) as excinfo:
        adapter.next_turn(request)
    err = excinfo.value
    assert err.http_status == 400
    assert err.retryable is False
    assert err.provider_error_code == "unsupported_value"
    assert "secret prompt" not in str(err)  # no prompt/body leak
    # A rejected call still owns its client: the throwing path releases the pool too.
    assert [client.closed for client in built] == [True]


# --- Client ownership: the adapter must not leak its HTTP connection pool --------------
#
# Driven against a local stand-in with the real SDK over a real socket, because the property is
# what the SDK's httpx connection pool does -- a hand-written stub cannot show it. Mirrors the
# gateway streaming tests' shape (``serving`` + ``asyncio.run``, no pytest-asyncio).


_STANDIN_RESPONSE: dict[str, Any] = {
    "id": "resp_1",
    "object": "response",
    "created_at": 0,
    "model": "gpt-5.5",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Hi", "annotations": []}],
        }
    ],
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
    "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
}

# A text delta ahead of the terminal event on purpose: it gives the abandonment test a yield
# *inside* the stream loop to park on, which is the position the leak depended on.
_STANDIN_SSE = "".join(
    f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
    for event in (
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hi",
            "logprobs": [],
        },
        {"type": "response.completed", "sequence_number": 2, "response": _STANDIN_RESPONSE},
    )
).encode("utf-8")

_STANDIN_ERROR = json.dumps(
    {"error": {"message": "unsupported value", "type": "invalid_request_error", "code": "unsupported_value"}}
).encode("utf-8")

# Three, not one: one call proves a client is closed, several prove nothing accumulates across
# calls -- the reported symptom was open connections growing with the call count.
_STANDIN_CALLS = 3


class _ResponsesStandIn(BaseHTTPRequestHandler):
    """Enough of the Responses API to drive the real SDK.

    ``HTTP/1.1`` with a ``Content-Length`` deliberately: the connection is then keep-alive, so a
    pool that is never closed holds a socket open here and the server's count sees it. Serves the
    streamed and non-streamed shapes off the request's own ``stream`` flag.
    """

    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self.server.track(+1)

    def finish(self) -> None:
        self.server.track(-1)
        super().finish()

    def do_GET(self) -> None:  # /healthz, which ``serving`` polls before yielding
        self._respond(200, b"ok", "text/plain")

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.server.status != 200:
            self._respond(self.server.status, _STANDIN_ERROR, "application/json")
        elif json.loads(body).get("stream"):
            self._respond(200, _STANDIN_SSE, "text/event-stream")
        else:
            self._respond(200, json.dumps(_STANDIN_RESPONSE).encode("utf-8"), "application/json")

    def _respond(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:
        return None  # keep pytest output clean


class _StandInServer(ThreadingHTTPServer):
    """Counts connections that are still open.

    The client-side ``is_closed()`` assertions say the adapter released the client; this says the
    sockets actually went away, observed from outside the code under test.
    """

    def __init__(self, status: int = 200) -> None:
        super().__init__(("127.0.0.1", 0), _ResponsesStandIn)
        self.status = status
        self.live = 0
        self._live_lock = threading.Lock()

    def track(self, delta: int) -> None:
        with self._live_lock:
            self.live += delta


@contextlib.contextmanager
def _responses_stand_in(monkeypatch: pytest.MonkeyPatch, *, status: int = 200) -> Iterator[_StandInServer]:
    """Serve the stand-in and point the SDK at it via ``OPENAI_BASE_URL``."""
    server = _StandInServer(status)
    with serving(server) as base_url:
        monkeypatch.setenv("OPENAI_BASE_URL", f"{base_url}/v1")
        yield server


def _recording_client(monkeypatch: pytest.MonkeyPatch, attribute: str) -> list[Any]:
    """Keep every real SDK client the adapter builds, so a test can ask whether it was closed.

    The adapter builds one per call and never exposes it, and subclassing leaves the SDK's own
    construction and pool behaviour intact.
    """
    import openai

    built: list[Any] = []
    real = getattr(openai, attribute)

    class _Recorded(real):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            built.append(self)

    monkeypatch.setattr(openai, attribute, _Recorded)
    return built


def _standin_adapter() -> OpenAIModelAdapter:
    return OpenAIModelAdapter(ModelConfig(model="gpt-5.5"), api_key="test", allow_direct_provider_api=True)


def _standin_request() -> ModelRequest:
    return ModelRequest(instruction="hi", system_prompt="sys", tools=())


def _wait_for_no_open_connections(server: _StandInServer, *, timeout_s: float = 10.0) -> int:
    """Poll until the server sees no open connection, and report what is left if it never does.

    Closing the client closes the socket, so a call that owns its client lands here in
    milliseconds; polling rather than sleeping a fixed interval keeps that from being a timing
    assumption on a loaded machine.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.live == 0:
            return 0
        time.sleep(0.02)
    return server.live


def test_openai_astream_closes_its_client_when_the_stream_is_drained(monkeypatch: pytest.MonkeyPatch) -> None:
    """A client built per call has to be released per call.

    Nothing closed these, so each drained call left an httpx pool holding its keep-alive socket
    open until a garbage collection that may never come -- server-side connections climbed with
    the call count, and the finaliser that eventually ran tried to ``aclose()`` on a loop that had
    already closed. The gateway adapter never had this: its client lives in an ``async with``.
    """
    pytest.importorskip("openai")
    built = _recording_client(monkeypatch, "AsyncOpenAI")
    with _responses_stand_in(monkeypatch) as server:
        adapter, request = _standin_adapter(), _standin_request()

        async def drain() -> list[list[Any]]:
            return [[chunk async for chunk in adapter.astream_turn(request)] for _ in range(_STANDIN_CALLS)]

        calls = asyncio.run(drain())
        still_open = _wait_for_no_open_connections(server)

    assert len(built) == _STANDIN_CALLS  # the adapter builds one client per call
    assert [client.is_closed() for client in built] == [True] * _STANDIN_CALLS
    assert still_open == 0, f"{still_open} connection(s) still open server-side"
    # And the turn survives the ownership: closing the client must not cost the terminal chunk.
    turn = assemble_streamed_turn(calls[-1])
    assert turn.final_text == "Hi"
    assert turn.usage["total_tokens"] == 5


def test_openai_astream_closes_its_client_when_the_consumer_abandons_mid_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exit path a close written after the stream loop never reaches.

    A consumer that stops early -- a cancelled run, a deadline, a caller that takes what it needs
    -- closes the generator, and that throws ``GeneratorExit`` at the yield it is parked on. Only a
    close bound to the generator's own exit runs then, which is why this is the case that decides
    between ``async with`` and a close at the end of the body.
    """
    pytest.importorskip("openai")
    built = _recording_client(monkeypatch, "AsyncOpenAI")
    with _responses_stand_in(monkeypatch) as server:
        adapter, request = _standin_adapter(), _standin_request()

        async def abandon() -> Any:
            stream = adapter.astream_turn(request)
            first = await anext(stream)
            await stream.aclose()
            return first

        first = asyncio.run(abandon())
        still_open = _wait_for_no_open_connections(server)

    # Parked inside the stream loop, not past it -- otherwise this asserts the drained case again.
    assert isinstance(first, TextDelta) and first.text == "Hi"
    assert [client.is_closed() for client in built] == [True]
    assert still_open == 0, f"{still_open} connection(s) still open server-side"


def test_openai_astream_releases_its_client_before_the_terminal_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A consumer that stops at the last chunk stops *inside* a suspended generator.

    `break` does not close an async generator, so a caller that reads the `TurnComplete` and stops
    -- the ordinary shape, since that chunk is the end of the turn -- holds it parked at that yield
    for as long as it keeps the reference. If the yield sat inside the region that owns the client,
    the pool would stay open for exactly that long. The terminal chunk is built from the captured
    response alone and needs nothing from the client, so it is yielded after the release.

    The observation has to be made **on the live loop**. `asyncio.run` calls
    `shutdown_asyncgens` on the way out, which closes every suspended generator and runs its
    cleanup, so checking after the run reports "released" whichever side of the release the yield is
    on. A real run's loop outlives the turn, which is what this reproduces.
    """
    pytest.importorskip("openai")
    built = _recording_client(monkeypatch, "AsyncOpenAI")
    with _responses_stand_in(monkeypatch) as server:
        adapter, request = _standin_adapter(), _standin_request()

        async def stop_at_the_end() -> tuple[Any, bool]:
            stream = adapter.astream_turn(request)
            last: Any = None
            async for chunk in stream:
                last = chunk
                if isinstance(chunk, TurnComplete):
                    break
            # `stream` is still referenced and still parked at that yield.
            return last, built[0].is_closed()

        last, released_while_parked = asyncio.run(stop_at_the_end())
        still_open = _wait_for_no_open_connections(server)

    assert isinstance(last, TurnComplete), "the consumer must stop at the terminal chunk"
    assert released_while_parked, (
        "the pool was still open while the consumer held the generator parked at the terminal "
        "chunk: the yield is inside the region that owns the client"
    )
    assert still_open == 0, f"{still_open} connection(s) still open server-side"


def test_openai_astream_closes_its_client_when_the_provider_rejects_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The third exit path: a call that raises still owns the pool it opened."""
    pytest.importorskip("openai")
    built = _recording_client(monkeypatch, "AsyncOpenAI")
    with _responses_stand_in(monkeypatch, status=400) as server:
        adapter, request = _standin_adapter(), _standin_request()

        async def drain() -> list[Any]:
            return [chunk async for chunk in adapter.astream_turn(request)]

        with pytest.raises(ModelAdapterError) as excinfo:
            asyncio.run(drain())
        still_open = _wait_for_no_open_connections(server)

    assert excinfo.value.http_status == 400  # still classified, not swallowed by the new scope
    assert [client.is_closed() for client in built] == [True]
    assert still_open == 0, f"{still_open} connection(s) still open server-side"


def test_openai_next_turn_closes_its_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sync path has no ``GeneratorExit`` to answer for, but the same unclosed pool."""
    pytest.importorskip("openai")
    built = _recording_client(monkeypatch, "OpenAI")
    with _responses_stand_in(monkeypatch) as server:
        turns = [_standin_adapter().next_turn(_standin_request()) for _ in range(_STANDIN_CALLS)]
        still_open = _wait_for_no_open_connections(server)

    assert [turn.final_text for turn in turns] == ["Hi"] * _STANDIN_CALLS
    assert [client.is_closed() for client in built] == [True] * _STANDIN_CALLS
    assert still_open == 0, f"{still_open} connection(s) still open server-side"


def test_openai_astream_classifies_a_failure_from_the_clients_own_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owning the client puts its teardown on this call's failure surface.

    Closing a pool can fail -- ``httpx.CloseError`` is the realistic instance -- and an exception
    raised in ``__aexit__`` *replaces* whatever was propagating through it. A classifier left
    inside the block would therefore let a raw error out of a seam whose whole contract is
    ``ModelAdapterError``: ``AgentLoop._recoverable_turn_error`` looks at no other type, so an
    unclassified one terminalizes the run instead of ending a single turn. The gateway adapter
    answers for the same surface on its own streamed path.
    """
    pytest.importorskip("openai")
    events = [_Ev("response.output_text.delta", delta="Hi"), _Ev("response.completed", response=_StreamResp())]
    built = _stub_openai(
        monkeypatch,
        "AsyncOpenAI",
        _async_stream_responses(events),
        close_error=RuntimeError("pool teardown failed"),
    )
    adapter = OpenAIModelAdapter(ModelConfig(model="gpt-5.5"), api_key="test", allow_direct_provider_api=True)
    request = ModelRequest(instruction="hi", system_prompt="", tools=())

    async def _drain() -> list:
        return [chunk async for chunk in adapter.astream_turn(request)]

    with pytest.raises(ModelAdapterError) as excinfo:
        asyncio.run(_drain())

    assert excinfo.value.provider_error_code == "unclassified_provider_error"
    assert "RuntimeError" in str(excinfo.value)  # the type aids debugging; no body is echoed
    assert [client.closed for client in built] == [True]  # the teardown was reached, then failed


def test_openai_next_turn_classifies_a_failure_from_the_clients_own_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sync twin: ``__exit__`` sits inside the classified region too."""
    pytest.importorskip("openai")

    class _Responses:
        def create(self, **_kwargs: Any) -> Any:
            return _StreamResp()

    built = _stub_openai(
        monkeypatch, "OpenAI", _Responses(), close_error=RuntimeError("pool teardown failed")
    )
    adapter = OpenAIModelAdapter(ModelConfig(), api_key="test", allow_direct_provider_api=True)

    with pytest.raises(ModelAdapterError) as excinfo:
        adapter.next_turn(ModelRequest(instruction="hi", system_prompt="", tools=()))

    assert excinfo.value.provider_error_code == "unclassified_provider_error"
    assert [client.closed for client in built] == [True]


# --- Client reuse inside an explicit scope ---------------------------------------------
#
# The counterweight to the `*_closes_its_client_*` tests above: those pin what an unscoped call
# does (own its client, close it), these pin what a scope changes (one client, closed once at the
# end) and what it must not change.


def test_openai_astream_reuses_one_client_across_calls_inside_a_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scope is what makes reuse safe, and reuse is the whole point of the scope.

    Unscoped, every call pays a full client construction -- measured at ~0.95s warm against ~13ms
    for the request itself, and on this path that cost is synchronous work sitting on the event
    loop, so the run's own cancel/deadline race cannot fire while it runs. A caller that outlives
    a call can pay it once instead.
    """
    pytest.importorskip("openai")
    built = _recording_client(monkeypatch, "AsyncOpenAI")
    with _responses_stand_in(monkeypatch) as server:
        adapter, request = _standin_adapter(), _standin_request()

        async def go() -> list[bool]:
            async with adapter:
                for _ in range(_STANDIN_CALLS):
                    async for _chunk in adapter.astream_turn(request):
                        pass
                return [client.is_closed() for client in built]

        during = asyncio.run(go())
        still_open = _wait_for_no_open_connections(server)

    assert len(built) == 1, "the scope must build one client, not one per call"
    assert during == [False], "and hold it open between calls -- otherwise there is no reuse"
    assert built[0].is_closed(), "the scope closes it on the way out"
    assert still_open == 0, f"{still_open} connection(s) still open server-side"


def test_openai_next_turn_reuses_one_client_across_calls_inside_a_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sync twin. No loop affinity on this client, so the scope is a plain ``with``."""
    pytest.importorskip("openai")
    built = _recording_client(monkeypatch, "OpenAI")
    with _responses_stand_in(monkeypatch) as server:
        adapter = _standin_adapter()
        with adapter:
            turns = [adapter.next_turn(_standin_request()) for _ in range(_STANDIN_CALLS)]
            during = [client.is_closed() for client in built]
        still_open = _wait_for_no_open_connections(server)

    assert [turn.final_text for turn in turns] == ["Hi"] * _STANDIN_CALLS
    assert len(built) == 1
    assert during == [False]
    assert built[0].is_closed()
    assert still_open == 0, f"{still_open} connection(s) still open server-side"


def test_openai_scope_rebuilds_a_client_whose_loop_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scope can outlive the loop its client is bound to. The client cannot.

    An ``AsyncOpenAI`` client's sockets belong to the loop that created them, so one carried over
    from a finished loop is unusable rather than merely stale -- handing it out would fail on
    first use. It has to be dropped and rebuilt, and quietly: a caller that spans loops has done
    nothing wrong, only something the client cannot follow.
    """
    pytest.importorskip("openai")
    built = _recording_client(monkeypatch, "AsyncOpenAI")
    with _responses_stand_in(monkeypatch):
        adapter, request = _standin_adapter(), _standin_request()
        adapter.open()

        async def calls(close: bool = False) -> list[Any]:
            # Two calls per loop, so the count below separates per-loop from per-call: caching
            # off, this is four clients; shared across loops, one; correct, two.
            chunks: list[Any] = []
            for _ in range(2):
                chunks = [chunk async for chunk in adapter.astream_turn(request)]
            if close:
                await adapter.aclose()
            return chunks

        asyncio.run(calls())  # loop A builds the scope's client and reuses it
        first = built[0]
        second = asyncio.run(calls(close=True))  # loop B cannot use A's, and ends the scope on B

    assert len(built) == 2, "one client per loop -- not shared across, and not one per call"
    assert built[1] is not first
    assert assemble_streamed_turn(second).final_text == "Hi", "the rebuilt client really works"
    assert built[1].is_closed(), "the scope closed the client belonging to the loop it ended on"
    # Deliberately no server-side connection assertion: ``first`` belongs to a loop that no longer
    # exists, and ``close()`` is a coroutine that can only run there, so nothing can close it.
    # That is the limitation ending an async scope with ``aclose`` on its own loop avoids; this
    # path only promises not to *reuse* the dead client.


def test_openai_astream_releases_a_stream_it_abandons_inside_a_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-call response has to be released per call, even when the client is not.

    Leaving an `async for` does not close the iterator it drove, and the call's `finally` only closed
    the *client* -- and only when the call owned it. Unscoped that was enough by accident: tearing the
    pool down took the response with it. Inside a scope the client outlives the call, so every turn
    aborted before the stream drained -- cancelled, deadlined, or stopped by `should_abort`, all
    ordinary -- left its response and connection checked out until the whole scope ended. Enough of
    them exhaust the pool and later calls stall waiting for a connection that will never come back.

    Measured *inside* the scope, which is the whole point: after it, the client is closed and both
    behaviours look identical.
    """
    pytest.importorskip("openai")
    built = _recording_client(monkeypatch, "AsyncOpenAI")
    with _responses_stand_in(monkeypatch) as server:
        adapter, request = _standin_adapter(), _standin_request()

        async def abandon_repeatedly() -> int:
            async with adapter:
                for _ in range(_STANDIN_CALLS):
                    stream = adapter.astream_turn(request)
                    await anext(stream)  # parked mid-response
                    await stream.aclose()
                # Still inside the scope: the client is alive by design, the responses must not be.
                for _ in range(250):
                    if server.live == 0:
                        break
                    await asyncio.sleep(0.02)
                return server.live

        live_inside_the_scope = asyncio.run(abandon_repeatedly())
        after = _wait_for_no_open_connections(server)

    assert len(built) == 1, "the scope must still reuse one client"
    assert live_inside_the_scope == 0, (
        f"{live_inside_the_scope} response connection(s) still checked out inside the scope after "
        f"{_STANDIN_CALLS} aborted turns"
    )
    assert after == 0, f"{after} connection(s) still open server-side"


def test_openai_scope_rebuilds_a_client_closed_underneath_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scope holds a client it does not exclusively control, so it checks before handing it out.

    `test_openai_scope_rebuilds_a_client_whose_loop_is_gone` covers the other half of the same
    condition -- a client bound to a loop that has moved on. This is the half where the client is
    still on the right loop and simply closed: whoever holds the adapter can close the client
    directly, and `close()`/`aclose()` themselves close it while another thread may still be inside a
    call. Handing a closed client out fails on first use, and it fails for every remaining call in
    the scope, because nothing would ever replace it.
    """
    pytest.importorskip("openai")
    built = _recording_client(monkeypatch, "AsyncOpenAI")
    with _responses_stand_in(monkeypatch) as server:
        adapter, request = _standin_adapter(), _standin_request()

        async def go() -> list[Any]:
            async with adapter:
                async for _chunk in adapter.astream_turn(request):
                    pass
                await built[0].close()  # closed underneath the scope, on the scope's own loop
                return [chunk async for chunk in adapter.astream_turn(request)]

        chunks = asyncio.run(go())
        still_open = _wait_for_no_open_connections(server)

    assert len(built) == 2, "a closed client was handed out again instead of being rebuilt"
    assert built[1] is not built[0]
    assert assemble_streamed_turn(chunks).final_text == "Hi", "the rebuilt client really works"
    assert still_open == 0, f"{still_open} connection(s) still open server-side"


def test_openai_sync_scope_rebuilds_a_client_closed_underneath_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sync twin. Same guard, separate accessor -- and only one of the two was ever read."""
    pytest.importorskip("openai")
    built = _recording_client(monkeypatch, "OpenAI")
    with _responses_stand_in(monkeypatch) as server:
        adapter = _standin_adapter()
        with adapter:
            adapter.next_turn(_standin_request())
            built[0].close()
            turn = adapter.next_turn(_standin_request())
        still_open = _wait_for_no_open_connections(server)

    assert len(built) == 2, "a closed client was handed out again instead of being rebuilt"
    assert built[1] is not built[0]
    assert turn.final_text == "Hi", "the rebuilt client really works"
    assert still_open == 0, f"{still_open} connection(s) still open server-side"


def test_openai_close_ends_the_scope_so_later_calls_own_their_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ending the scope has to *detach* it, not just close what it held.

    `_take_scope` is the single place that does, for both `close()` and `aclose()`, and both halves
    of `close()`'s contract rest on it. A scope merely read and left in place stays live: the next
    call finds it, rebuilds into it because the old client is closed, and gets `call_owned=False` --
    so nothing ever closes that client. The leak this whole scope mechanism exists to fix comes
    straight back, on the ordinary path of a caller that keeps using the adapter afterwards. And
    `close()` stops being idempotent, since a second call finds the same scope again.
    """
    pytest.importorskip("openai")
    built = _recording_client(monkeypatch, "OpenAI")
    with _responses_stand_in(monkeypatch) as server:
        adapter = _standin_adapter()
        with adapter:
            adapter.next_turn(_standin_request())
        adapter.close()  # idempotent: the scope is already gone
        turn = adapter.next_turn(_standin_request())
        still_open = _wait_for_no_open_connections(server)

    assert turn.final_text == "Hi"
    assert len(built) == 2, "the call after the scope must build its own client"
    assert [client.is_closed() for client in built] == [True, True], (
        "a call made after the scope ended must own and close its client; one that rebuilt into a "
        "scope still holding on has nobody to close it"
    )
    assert still_open == 0, f"{still_open} connection(s) still open server-side"


def test_openai_only_a_client_from_another_loop_is_handed_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Foreign and closed are different conditions, and only the first needs a handoff.

    A cached client is dropped when it belongs to another loop *or* when it is simply closed. Only
    the foreign one has anything left to release, and it costs a cross-loop handoff to do it.
    Treating a merely-closed client as foreign schedules a second `close()` onto the loop already
    running -- work on a client that has nothing left to give up.
    """
    pytest.importorskip("openai")
    import monoid_agent_kernel.providers.openai as openai_adapter

    handed_back: list[Any] = []
    monkeypatch.setattr(
        openai_adapter,
        "_release_foreign_async_client",
        lambda client, loop: handed_back.append(loop),
    )
    _recording_client(monkeypatch, "AsyncOpenAI")
    with _responses_stand_in(monkeypatch):
        adapter, request = _standin_adapter(), _standin_request()
        adapter.open()

        async def call_then_close_the_client() -> None:
            async for _chunk in adapter.astream_turn(request):
                pass
            await adapter._scope.async_client.close()
            async for _chunk in adapter.astream_turn(request):  # rebuilds, same loop
                pass

        asyncio.run(call_then_close_the_client())
        closed_on_this_loop = list(handed_back)
        asyncio.run(call_then_close_the_client())  # a second loop: the first loop's client is stale
        across_loops = list(handed_back)
        # Measured before this: ``close()`` hands its client back unconditionally, which would
        # otherwise land in the counts above and hide what the accessor decided.
        adapter.close()

    assert closed_on_this_loop == [], (
        "a closed client on the running loop was handed back for a close it does not need"
    )
    # Counterweight: the foreign case does get handed back, so this is not asserting "never".
    assert len(across_loops) == 1, "a client bound to a loop that has moved on must be handed back"


def test_openai_scope_does_not_close_a_client_another_live_loop_is_using() -> None:
    """Reuse belongs to the loop the scope holds; another *live* loop gets its own client.

    One adapter held open across two concurrently running loops -- two runs sharing it -- had the
    second loop's request treat the first loop's client as stale, purely because a different loop
    asked, and then schedule `close()` on it. That loop was still running, so the handoff succeeded
    and cut off a call in flight.

    "Moved on" and "still running" are different conditions and only the first is safe to release.
    A call from another live loop now gets a client it owns and closes, which is what an unscoped
    call does anyway, and the scope is left alone so its own loop keeps reusing.

    Driven at the accessor rather than through `astream_turn`: the rule is about two loops racing for
    one scope, and two real stand-in servers with real SDK clients would add network timing to a test
    whose whole point is the handoff decision. The two loops here are genuinely concurrent -- loop A
    is parked in an executor wait, so its loop is running when loop B asks.
    """
    closed: list[str] = []
    built: list[Any] = []

    class _FakeAsyncClient:
        def __init__(self) -> None:
            self.name = f"client-{len(built) + 1}"
            self._closed = False

        def is_closed(self) -> bool:
            return self._closed

        async def close(self) -> None:
            self._closed = True
            closed.append(self.name)

    def factory(**_kwargs: Any) -> Any:
        built.append(_FakeAsyncClient())
        return built[-1]

    adapter = OpenAIModelAdapter(
        ModelConfig(model="gpt-5.5"), api_key="k", allow_direct_provider_api=True
    )
    adapter.open()
    handed: dict[str, Any] = {}
    a_has_client = threading.Event()
    b_is_done = threading.Event()
    failures: list[BaseException] = []

    def loop_a() -> None:
        async def go() -> None:
            client, owned = adapter._async_client(factory, "k")
            handed["a"] = (client.name, owned)
            a_has_client.set()
            # Parked here, so this loop is *running* while loop B asks for a client.
            await asyncio.get_running_loop().run_in_executor(None, b_is_done.wait, 10)
            handed["a_client_closed"] = client.is_closed()

        try:
            asyncio.run(go())
        except BaseException as exc:  # noqa: BLE001 - reported on the main thread instead
            failures.append(exc)

    def loop_b() -> None:
        try:
            assert a_has_client.wait(10), "loop A never got a client"

            async def go() -> None:
                client, owned = adapter._async_client(factory, "k")
                handed["b"] = (client.name, owned)
                await asyncio.sleep(0.2)  # long enough for a scheduled cross-loop close to land

            asyncio.run(go())
        except BaseException as exc:  # noqa: BLE001
            failures.append(exc)
        finally:
            b_is_done.set()

    threads = [threading.Thread(target=loop_a), threading.Thread(target=loop_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)
    adapter.close()

    assert not failures, f"a loop thread failed: {failures}"
    assert all(not thread.is_alive() for thread in threads), "a loop thread did not finish"
    assert handed["a"] == ("client-1", False), "the scope's own loop must keep reusing"
    assert handed["b"] == ("client-2", True), (
        "a call from another live loop must get a client of its own, and own it"
    )
    assert handed["a_client_closed"] is False, (
        "loop A's client was closed while loop A was still using it"
    )
    assert closed == [], f"nothing should have been released here, closed: {closed}"


def test_openai_scope_outlives_a_call_that_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing call must not close a client it does not own.

    Unscoped, a call that raises releases its own pool on the way out -- that is the point of the
    ``finally``. Scoped, the same ``finally`` has to keep its hands off: one provider rejection
    closing the shared client would either end the reuse silently or, worse, leave later turns
    holding a closed client.
    """
    pytest.importorskip("openai")
    built = _recording_client(monkeypatch, "OpenAI")
    with _responses_stand_in(monkeypatch) as server:
        adapter = _standin_adapter()
        with adapter:
            server.status = 400
            with pytest.raises(ModelAdapterError) as excinfo:
                adapter.next_turn(_standin_request())
            survived = not built[0].is_closed()
            server.status = 200
            turn = adapter.next_turn(_standin_request())
        still_open = _wait_for_no_open_connections(server)

    assert excinfo.value.http_status == 400  # still classified, scope or no scope
    assert survived, "the failed call closed a client the scope owns"
    assert turn.final_text == "Hi", "and the scope still works afterwards"
    assert len(built) == 1, "the same client was reused, not rebuilt after the failure"
    assert built[0].is_closed()
    assert still_open == 0, f"{still_open} connection(s) still open server-side"


@pytest.mark.parametrize("context_manager", [False, True], ids=["open-close-only", "also-a-cm"])
def test_cli_run_scopes_an_adapter_that_can_hold_its_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, context_manager: bool
) -> None:
    """The run's one adapter is opened around every turn and closed after.

    Nothing else in the CLI can do this: the adapter is built once per run and all the turns
    happen inside ``run_once``, so without a scope here the direct-OpenAI adapter builds and
    throws away a client per turn. Adapters with no client to hold -- the fakes, the gateway one
    -- have no ``open`` and are left alone, which every other CLI test above covers.

    Both shapes, because the CLI probes ``open`` and once used ``__enter__``. This stub used to
    implement all four members, so it could not tell the two apart -- and the ``open``/``close``
    adapter the probe invites (the lifecycle pair ``AgentLoop`` and ``LoopSession`` use, and all
    ``OpenAIModelAdapter``'s own context manager delegates to) died with ``TypeError`` before the
    first turn, outside the CLI's error handler, so the run ended in a raw traceback.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    events: list[str] = []

    class _ScopedAdapter:
        def __init__(self) -> None:
            self._inner = FakeModelAdapter(turns=[ModelTurn(final_text="done")])

        def open(self) -> _ScopedAdapter:
            events.append("open")
            return self

        def close(self) -> None:
            events.append("close")

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            events.append("turn")
            return self._inner.next_turn(request)

    class _ScopedContextManagerAdapter(_ScopedAdapter):
        def __enter__(self) -> _ScopedAdapter:
            return self.open()

        def __exit__(self, *_exc: object) -> None:
            self.close()

    build = _ScopedContextManagerAdapter if context_manager else _ScopedAdapter
    monkeypatch.setattr("monoid_agent_kernel.cli._model_adapter", lambda *_a, **_k: build())
    config_file = _write_config(tmp_path / "runtime.json", "run.finish")

    result = CliRunner().invoke(
        main,
        [
            "run", "--workspace", str(workspace), "--instruction", "Finish.",
            "--run-root", str(tmp_path / "runs"), "--runtime-config-file", str(config_file),
        ],
    )

    assert result.exit_code == 0, result.output
    # Order, not just membership: a close before the turn would defeat the reuse, and an open
    # after it would never have covered the turn at all.
    assert events == ["open", "turn", "close"]


@pytest.mark.parametrize("failing", ["open", "close"])
def test_cli_run_reports_an_adapter_lifecycle_failure_as_a_cli_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failing: str
) -> None:
    """Both lifecycle calls sit below the handler that normalizes every other startup failure.

    A pool that fails to construct, or to tear down, ended `monoid run` in a bare traceback. The
    close case carries a second failure: the run's outcome was echoed *after* the scope unwound, and
    an exception from a cleanup callback replaces whatever is leaving the block — so a teardown that
    raised swallowed the status of a run that had completed.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class _LifecycleFailure:
        def __init__(self) -> None:
            self._inner = FakeModelAdapter(turns=[ModelTurn(final_text="done")])

        def open(self) -> _LifecycleFailure:
            if failing == "open":
                raise RuntimeError("pool construction failed")
            return self

        def close(self) -> None:
            if failing == "close":
                raise RuntimeError("pool teardown failed")

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            return self._inner.next_turn(request)

    monkeypatch.setattr(
        "monoid_agent_kernel.cli._model_adapter", lambda *_a, **_k: _LifecycleFailure()
    )
    config_file = _write_config(tmp_path / "runtime.json", "run.finish")

    result = CliRunner().invoke(
        main,
        [
            "run", "--workspace", str(workspace), "--instruction", "Finish.",
            "--run-root", str(tmp_path / "runs"), "--runtime-config-file", str(config_file),
        ],
    )

    assert result.exit_code != 0
    assert not isinstance(result.exception, RuntimeError), (
        "the lifecycle failure escaped as a traceback instead of a reported CLI error"
    )
    assert f"{failing}() failed" in result.output, result.output
    if failing == "close":
        assert "status: completed" in result.output, (
            f"a failing teardown swallowed the outcome of a completed run: {result.output}"
        )


def test_cli_run_keeps_the_real_failure_when_teardown_also_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cleanup failure is a footnote to a real failure, never a replacement for it.

    Raising from an `ExitStack` callback *replaces* the exception leaving the block, so making a
    failing `close()` report itself as a CLI error -- which it must, when nothing else is wrong --
    turned it into a mask. Measured: a run failed by a dead provider reported only
    `close() failed`, with the provider error nowhere in the output. `result.error` was raised after
    the scope, so it was never even reached.

    Both are reported now: the run's failure as the command's error, the teardown as a warning beside
    it. The counterweight -- a clean run whose teardown fails, where the teardown *is* the error -- is
    `test_cli_run_reports_an_adapter_lifecycle_failure_as_a_cli_error[close]`.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class _FailingRunAndTeardown:
        def open(self) -> _FailingRunAndTeardown:
            return self

        def close(self) -> None:
            raise RuntimeError("pool teardown failed")

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            raise ModelAdapterError("the provider is down")

    monkeypatch.setattr(
        "monoid_agent_kernel.cli._model_adapter", lambda *_a, **_k: _FailingRunAndTeardown()
    )
    config_file = _write_config(tmp_path / "runtime.json", "run.finish")

    result = CliRunner().invoke(
        main,
        [
            "run", "--workspace", str(workspace), "--instruction", "Finish.",
            "--run-root", str(tmp_path / "runs"), "--runtime-config-file", str(config_file),
        ],
    )

    assert result.exit_code != 0
    assert "the provider is down" in result.output, (
        f"the teardown masked the failure that actually needs diagnosing: {result.output}"
    )
    assert "close() failed" in result.output, (
        f"the teardown failure was swallowed instead of demoted: {result.output}"
    )


def test_cli_run_refuses_an_adapter_offering_open_without_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Half a lifecycle pair is reported before anything is allocated, not after.

    Registering the bound `close` after calling `open()` looked early enough and was not: `open()`
    had already taken whatever it takes -- a connection pool, for the adapter this scope exists for
    -- and the `AttributeError` from the missing `close` then escaped past the CLI's own handler,
    with nothing left able to release it. `open()` must not have run at all.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    allocated: list[str] = []

    class _OpenWithoutClose:
        def __init__(self) -> None:
            self._inner = FakeModelAdapter(turns=[ModelTurn(final_text="done")])

        def open(self) -> _OpenWithoutClose:
            allocated.append("client")
            return self

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            return self._inner.next_turn(request)

    monkeypatch.setattr(
        "monoid_agent_kernel.cli._model_adapter", lambda *_a, **_k: _OpenWithoutClose()
    )
    config_file = _write_config(tmp_path / "runtime.json", "run.finish")

    result = CliRunner().invoke(
        main,
        [
            "run", "--workspace", str(workspace), "--instruction", "Finish.",
            "--run-root", str(tmp_path / "runs"), "--runtime-config-file", str(config_file),
        ],
    )

    assert result.exit_code != 0
    assert allocated == [], "open() ran before the missing close() was noticed"
    assert "close()" in result.output, f"the reason must reach the user: {result.output}"
    assert not isinstance(result.exception, AttributeError), (
        "the failure must be a reported CLI error, not a raw attribute lookup escaping the handler"
    )
