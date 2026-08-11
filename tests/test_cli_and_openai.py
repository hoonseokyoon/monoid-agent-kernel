from __future__ import annotations

import asyncio
import contextlib
import json
import os
import threading
import time
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from support.http import serving
from support.runtime import runtime_config, tool_binding

from monoid_agent_kernel.cli import main
from monoid_agent_kernel.core.spec import ModelConfig, ModelRetryConfig, ReasoningConfig
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelTurn,
    ReasoningDelta,
    TextDelta,
    TurnComplete,
    assemble_streamed_turn,
)
from monoid_agent_kernel.providers._common import (
    prune_dead_reasoning,
    reasoning_replay_window_start,
)
from monoid_agent_kernel.providers.fake import (
    FakeModelAdapter,
    FakeStreamingModelAdapter,
    fake_tool_call,
)
import monoid_agent_kernel.providers.openai as openai_module
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


def test_openai_refuses_the_by_reference_shape_under_zdr() -> None:
    """``store=False`` above and ``previous_response_id`` are contradictory in one adapter: no
    response is ever persisted, so a handle naming one can never resolve. The shape was emitted
    anyway and failed as an opaque provider 404 at call time -- on the *original* call, not
    merely on a validation repair. It is refused at the adapter boundary instead, classified
    the same way every other config-shaped refusal here is, and the message names the supported
    route."""

    adapter = OpenAIModelAdapter(ModelConfig(model="gpt-5.5"))
    request = ModelRequest(
        instruction=None,
        system_prompt="sys",
        tools=(),
        previous_turn_handle="resp_1",
    )

    with pytest.raises(ModelAdapterError) as refused:
        adapter._payload(request)

    assert refused.value.provider_error_code == "unsupported_request_shape"
    assert refused.value.retryable is False
    assert refused.value.config_recoverable is True
    assert "messages" in str(refused.value)


def test_the_by_reference_refusal_fires_on_both_openai_entry_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``next_turn`` and ``astream_turn`` are the adapter's only two entry points and both
    build their body through ``_classified_payload`` -> ``_payload``; the refusal must reach
    the caller unchanged through each (``ModelAdapterError`` is outside the
    ``TypeError``/``ValueError``/``RecursionError`` family that classifier converts)."""

    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    adapter = OpenAIModelAdapter(ModelConfig(model="gpt-5.5"), allow_direct_provider_api=True)
    request = ModelRequest(
        instruction="follow up",
        system_prompt="sys",
        tools=(),
        previous_turn_handle="resp_1",
    )

    with pytest.raises(ModelAdapterError) as blocking:
        adapter.next_turn(request)
    assert blocking.value.provider_error_code == "unsupported_request_shape"
    assert blocking.value.config_recoverable is True

    async def _drive() -> None:
        async for _chunk in adapter.astream_turn(request):
            pass

    with pytest.raises(ModelAdapterError) as streamed:
        asyncio.run(_drive())
    assert streamed.value.provider_error_code == "unsupported_request_shape"
    assert streamed.value.config_recoverable is True


def test_a_stale_handle_beside_by_value_messages_is_not_refused() -> None:
    """The refusal is bound to the *shape*, not to the field: ``messages`` overrides the handle
    path (documented, and both adapters select on ``messages is not None``), and the loop does
    hand a by-value request a leftover handle. Refusing on the field alone would have killed
    the ordinary production path."""

    adapter = OpenAIModelAdapter(ModelConfig(model="gpt-5.5"))
    payload = adapter._payload(
        ModelRequest(
            instruction=None,
            system_prompt="sys",
            tools=(),
            previous_turn_handle="stale-handle",
            messages=({"role": "user", "content": "hi"},),
        )
    )

    assert payload["input"] == [{"role": "user", "content": "hi"}]
    assert "previous_response_id" not in payload


def test_the_gateway_maps_the_by_reference_refusal_to_a_bad_request() -> None:
    """Blast radius: the reference gateway's own by-reference continuation maps its opaque
    turn_handle to a stored provider response id and passes it upstream, so with the default
    OpenAI upstream that continuation now inherits this refusal. That is the coherent outcome
    -- a classified 422 the outer client survives, instead of the opaque provider 404 it used
    to become. Gateway by-reference support itself is untouched: an upstream that *does* keep
    responses still continues by handle."""

    from monoid_agent_kernel.reference.llm_gateway.http import _model_error_status

    adapter = OpenAIModelAdapter(ModelConfig(model="gpt-5.5"))
    with pytest.raises(ModelAdapterError) as refused:
        adapter._payload(
            ModelRequest(
                instruction=None,
                system_prompt="sys",
                tools=(),
                previous_turn_handle="provider_response_1",
            )
        )

    assert _model_error_status(refused.value) == HTTPStatus.UNPROCESSABLE_ENTITY


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


# --- the active window as ONE rule, and the prune the kernel builds on it ---------------------


@pytest.mark.parametrize(
    ("roles", "expected_start"),
    (
        ((), 0),
        # No user message at all: the whole log is the window (what the flag rule always said).
        (("assistant", "tool"), 0),
        (("user", "assistant", "tool"), 1),
        (("user", "assistant", "tool", "user", "assistant", "tool"), 4),
        # A trailing user message opens an empty window -- nothing after it yet.
        (("user", "assistant", "user"), 3),
    ),
)
def test_the_replay_window_start_is_the_rule_the_flags_are_built_from(
    roles: tuple[str, ...], expected_start: int
) -> None:
    """One definition of "active window", read by both halves that depend on it.

    The adapter decides what to REPLAY from it and the kernel decides what to SEND into it; two
    copies of the same index arithmetic is exactly the twin-drift this repo keeps paying for, so
    the rule lives in one function and this pins the flags to it rather than to a hand-copy.
    """

    messages = tuple({"role": role, "content": ""} for role in roles)
    assert reasoning_replay_window_start(messages) == expected_start
    assert _reasoning_replay_flags(messages, "gpt-5.5") == [
        index >= expected_start for index in range(len(messages))
    ]


def _two_window_conversation(historical_model: str = "gpt-5.5") -> tuple[dict, ...]:
    """Two user turns: an assistant block outside the window, and one inside it."""
    return (
        {"role": "user", "content": "u1"},
        _assistant_with_reasoning(
            historical_model, [_RS_A, _FC_A], [{"id": "c_a", "name": "fs_read", "arguments": {}}]
        ),
        {"role": "tool", "call_id": "c_a", "content": {"ok": True}},
        {"role": "user", "content": "u2"},
        _assistant_with_reasoning(
            "gpt-5.5", [_RS_B, _FC_B], [{"id": "c_b", "name": "text_search", "arguments": {}}]
        ),
        {"role": "tool", "call_id": "c_b", "content": {"ok": True}},
    )


def test_pruning_a_dead_reasoning_block_drops_it_only_outside_the_window() -> None:
    messages = _two_window_conversation()
    pruned = prune_dead_reasoning(messages)

    assert "reasoning" not in pruned[1], "the historical block is unreachable — drop it"
    assert pruned[4]["reasoning"] == messages[4]["reasoning"], "the live one must survive"
    # Nothing else about the log may change: only that one key leaves.
    assert pruned[1] == {k: v for k, v in messages[1].items() if k != "reasoning"}
    assert [pruned[i] for i in (0, 2, 3, 5)] == [messages[i] for i in (0, 2, 3, 5)]
    # The caller's log is untouched — the prune builds the wire copy, it does not mutate.
    assert messages[1]["reasoning"]["items"] == [_RS_A, _FC_A]


def test_the_openai_input_is_byte_identical_once_a_block_leaves_the_window() -> None:
    """The safety proof for the prune: what the provider SEES does not change.

    Outside the window ``_message_to_input_items`` takes the reconstruction branch whether or
    not the key is there (the replay flag is False, and the flag is all it reads), so removing
    it can only remove bytes from the request — never items from the input.
    """

    messages = _two_window_conversation()
    adapter = OpenAIModelAdapter(ModelConfig(model="gpt-5.5"))

    def payload_for(log: tuple[dict, ...]) -> dict:
        return adapter._payload(
            ModelRequest(instruction=None, system_prompt="", tools=(), messages=log)
        )

    assert payload_for(prune_dead_reasoning(messages)) == payload_for(messages)
    # And the prune really did remove something the un-pruned request was still paying for.
    assert "enc_a" in json.dumps(messages)
    assert "enc_a" not in json.dumps(prune_dead_reasoning(messages))
    assert "enc_b" in json.dumps(prune_dead_reasoning(messages))


def test_a_dead_block_cannot_poison_the_window_before_or_after_the_prune() -> None:
    """The one way this could have gone wrong: the all-or-nothing rule reads only the window.

    A historical block tagged with a *different* model is already ignored, so pruning it must
    not flip the live window's decision either way. If the model-identity scan ever widened to
    the whole log, the prune would silently start changing what is replayed — this fails first.
    """

    messages = _two_window_conversation(historical_model="gpt-4o")
    pruned = prune_dead_reasoning(messages)

    assert _reasoning_replay_flags(pruned, "gpt-5.5") == _reasoning_replay_flags(messages, "gpt-5.5")
    adapter = OpenAIModelAdapter(ModelConfig(model="gpt-5.5"))
    replayed = [
        item
        for item in adapter._payload(
            ModelRequest(instruction=None, system_prompt="", tools=(), messages=pruned)
        )["input"]
        if item.get("type") == "reasoning"
    ]
    assert replayed == [_RS_B]


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

    class _FakeRawCalls:
        def create(self, **kwargs):  # noqa: ANN003
            raise _FakeBadRequest()

    class _FakeResponses:
        # The surface ``next_turn`` drives: the raw-response wrapper (it keeps the final request
        # the success-path retry probe reads), which raises exactly as the plain call would.
        with_raw_response = _FakeRawCalls()

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


def test_the_kernel_key_is_never_presented_to_the_openai_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CONTRACTS: only the gateway transport presents `Idempotency-Key`; "the OpenAI adapter
    does not read the field, so nothing is sent there." Stated since W7-3 and checked by
    nothing until now -- the documented-rule-nobody-enforces shape three W7-2 review rounds
    kept finding. Scoped to the KERNEL's token: the SDK may run idempotency machinery of its
    own and this pin says nothing about it, only that the value the runner minted reaches no
    argument the adapter hands the SDK. Three pins close the claim together: this capture
    (the dispatch kwargs), the exact `with_options` equality below (`{"max_retries": 0}` and
    nothing else), and the AST census that every SDK call sits on `_call_client`."""
    pytest.importorskip("openai")  # the dispatch path imports the SDK; skip on a minimal install

    sentinel = "idem_" + "cafe" * 8
    seen: list[dict[str, Any]] = []

    class _RefusalAfterCapture(Exception):
        # The 400 shape the classifier already maps (its own test above): the pin needs the
        # dispatch kwargs, not a parseable success body, so the cheapest settled outcome ends
        # the call right after the capture.
        def __init__(self) -> None:
            super().__init__("refused after capture")
            self.status_code = 400
            self.body = {"code": "unsupported_value"}

    class _CapturingRawCalls:
        def create(self, **kwargs):  # noqa: ANN003, ANN202
            seen.append(kwargs)
            raise _RefusalAfterCapture()

    class _CapturingResponses:
        with_raw_response = _CapturingRawCalls()

    _stub_openai(monkeypatch, "OpenAI", _CapturingResponses())
    adapter = OpenAIModelAdapter(ModelConfig(), api_key="test", allow_direct_provider_api=True)
    request = ModelRequest(
        instruction="hi", system_prompt="", tools=(), idempotency_key=sentinel
    )

    with pytest.raises(ModelAdapterError):
        adapter.next_turn(request)

    assert request.idempotency_key == sentinel  # the field was there to leak
    assert seen, "the dispatch never reached the SDK stub"
    assert sentinel not in repr(seen), "the kernel's key reached an SDK argument"


def test_the_openai_adapter_never_reads_the_idempotency_field() -> None:
    """The mechanism behind the pin above, checkable without the SDK installed: "the OpenAI
    adapter does not read the field" (CONTRACTS). An adapter that never names the field
    cannot present it -- the payload is hand-listed, not serialized off the request -- so
    this source census is the half of the claim that runs on a minimal install, where the
    capture pin above skips. A future edit that starts reading the field fails here first
    and has to argue with the contract sentence it contradicts."""

    import inspect

    import monoid_agent_kernel.providers.openai as openai_module

    source = inspect.getsource(openai_module)
    assert "idempotency" not in source.lower(), (
        "the OpenAI adapter names the idempotency field; CONTRACTS says only the gateway "
        "transport presents it"
    )


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
        if self.server.take_failure():
            # A transient 500 the SDK's own retry loop absorbs. ``retry-after-ms`` keeps the
            # SDK's backoff out of the test clock (its schedule honours the header).
            self._respond(
                500,
                _STANDIN_ERROR,
                "application/json",
                extra_headers={"retry-after-ms": "1"},
            )
        elif self.server.status != 200:
            self._respond(self.server.status, _STANDIN_ERROR, "application/json")
        elif json.loads(body).get("stream"):
            self._respond(200, _STANDIN_SSE, "text/event-stream")
        else:
            self._respond(200, json.dumps(_STANDIN_RESPONSE).encode("utf-8"), "application/json")

    def _respond(
        self,
        status: int,
        body: bytes,
        content_type: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:
        return None  # keep pytest output clean


class _StandInServer(ThreadingHTTPServer):
    """Counts connections that are still open.

    The client-side ``is_closed()`` assertions say the adapter released the client; this says the
    sockets actually went away, observed from outside the code under test.
    """

    def __init__(self, status: int = 200, fail_first: int = 0) -> None:
        super().__init__(("127.0.0.1", 0), _ResponsesStandIn)
        self.status = status
        self.fail_first = fail_first
        self.live = 0
        self._live_lock = threading.Lock()

    def track(self, delta: int) -> None:
        with self._live_lock:
            self.live += delta

    def take_failure(self) -> bool:
        """Consume one budgeted transient failure, if any remain."""
        with self._live_lock:
            if self.fail_first <= 0:
                return False
            self.fail_first -= 1
            return True


@contextlib.contextmanager
def _responses_stand_in(
    monkeypatch: pytest.MonkeyPatch, *, status: int = 200, fail_first: int = 0
) -> Iterator[_StandInServer]:
    """Serve the stand-in and point the SDK at it via ``OPENAI_BASE_URL``."""
    server = _StandInServer(status, fail_first)
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


def test_openai_next_turn_reports_the_sdk_retry_behind_a_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empirical, against the real SDK: a 500 its retry loop absorbs still marks the success.

    The stand-in refuses the first POST and answers the second, which is invisible to the
    adapter's own control flow -- the SDK hands back a parsed model either way. The evidence
    is the ``x-stainless-retry-count`` header the SDK stamps on its final request, read off the
    raw-response wrapper; without it this call was written to the transcript, the receipt and
    the gateway wire as a clean first attempt. The follow-up call on the now-healthy server
    proves the flag is per-call evidence, not adapter state.
    """
    pytest.importorskip("openai")
    with _responses_stand_in(monkeypatch, fail_first=1):
        retried = _standin_adapter().next_turn(_standin_request())
        clean = _standin_adapter().next_turn(_standin_request())

    assert retried.final_text == "Hi"
    assert retried.provider_retried is True
    assert clean.final_text == "Hi"
    assert clean.provider_retried is False


def test_openai_astream_reports_the_sdk_retry_on_every_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The streaming twin, and on every chunk: an abandoned stream never yields the terminal
    one, so evidence riding only ``TurnComplete`` is evidence a cancelled call cannot report."""
    pytest.importorskip("openai")
    with _responses_stand_in(monkeypatch, fail_first=1):
        adapter, request = _standin_adapter(), _standin_request()

        async def drain_twice() -> tuple[list[Any], list[Any]]:
            first = [chunk async for chunk in adapter.astream_turn(request)]
            second = [chunk async for chunk in adapter.astream_turn(request)]
            return first, second

        retried, clean = asyncio.run(drain_twice())

    assert [type(chunk) for chunk in retried] == [TextDelta, TurnComplete]
    assert [chunk.provider_retried for chunk in retried] == [True, True]
    assert [type(chunk) for chunk in clean] == [TextDelta, TurnComplete]
    assert [chunk.provider_retried for chunk in clean] == [False, False]


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

    class _RawTurn:
        # The ``LegacyAPIResponse`` shape: ``parse()`` yields the model, and no ``request``
        # attribute at all -- which the retry probe must read as "no retry", not crash on.
        def parse(self) -> Any:
            return _StreamResp()

    class _RawCalls:
        def create(self, **_kwargs: Any) -> Any:
            return _RawTurn()

    class _Responses:
        with_raw_response = _RawCalls()

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


def test_cli_run_recording_flags_produce_the_sidecars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--model-calls-file`` / ``--model-payload-file`` are the CLI's halves of the switches
    the backend carries as fields. Without them this CLI shipped ``monoid gc`` and ``monoid
    validate`` -- consumer verbs -- while no ``monoid run`` invocation could produce the
    artifacts they consume. The witness is the digest join, as in the backend twin: the
    ledger line's 64-hex key must name the corpus request record beside it. And the flags
    stay opt-in: the corpus is content-classified, so the second run pins that omitting them
    writes neither file."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        "monoid_agent_kernel.cli._model_adapter",
        lambda *_a, **_k: FakeModelAdapter(turns=[ModelTurn(final_text="done")]),
    )
    config_file = _write_config(tmp_path / "runtime.json", "run.finish")
    run_root = tmp_path / "runs"

    result = CliRunner().invoke(
        main,
        [
            "run", "--workspace", str(workspace), "--instruction", "Finish.",
            "--run-root", str(run_root), "--runtime-config-file", str(config_file),
            "--run-id", "cli-recording",
            "--model-calls-file", "--model-payload-file",
        ],
    )

    assert result.exit_code == 0, result.output
    run_dir = run_root / "cli-recording"
    ledger_lines = [
        json.loads(line)
        for line in (run_dir / "model_calls.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert ledger_lines, "the ledger flag reached the run"
    digest = ledger_lines[0]["request_digest"]
    assert len(digest) == 64
    request_records = [
        record
        for record in (
            json.loads(line)
            for line in (run_dir / "model_payloads.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        if record.get("kind") == "model_request"
    ]
    assert [record["request_digest"] for record in request_records] == [digest]

    quiet = CliRunner().invoke(
        main,
        [
            "run", "--workspace", str(workspace), "--instruction", "Finish.",
            "--run-root", str(run_root), "--runtime-config-file", str(config_file),
            "--run-id", "cli-quiet",
        ],
    )

    assert quiet.exit_code == 0, quiet.output
    assert not (run_root / "cli-quiet" / "model_calls.jsonl").exists()
    assert not (run_root / "cli-quiet" / "model_payloads.jsonl").exists()
    assert not (run_root / "cli-quiet" / "model-content.jsonl").exists()

    # The third sidecar, whose flag landed with these two so that `monoid validate`'s
    # model-content arm stops being a consumer with no producer. Driven by a *streaming* adapter,
    # because this flag is the one with a side effect: it selects the streaming dispatch, and the
    # non-streaming fake above produces a file with `stream_opened`/`stream_closed` and not one
    # `stream_segment` -- so an existence check on that adapter pins the wiring and none of the
    # behaviour the flag's own help advertises.
    monkeypatch.setattr(
        "monoid_agent_kernel.cli._model_adapter",
        lambda *_a, **_k: FakeStreamingModelAdapter(
            chunk_turns=[[TextDelta("streamed "), TextDelta("answer"), TurnComplete()]]
        ),
    )
    content = CliRunner().invoke(
        main,
        [
            "run", "--workspace", str(workspace), "--instruction", "Finish.",
            "--run-root", str(run_root), "--runtime-config-file", str(config_file),
            "--run-id", "cli-content", "--model-content-file",
        ],
    )

    assert content.exit_code == 0, content.output
    kinds = [
        json.loads(line)["kind"]
        for line in (run_root / "cli-content" / "model-content.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert "stream_segment" in kinds, kinds


@pytest.mark.parametrize("requested", [True, False], ids=["asked", "omitted"])
def test_backend_serve_carries_the_recording_flags_to_the_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, requested: bool
) -> None:
    """A deployment is served, not run one-shot, so the deployment shape needs the flags too.

    `monoid run` and the `RunnerBackend` field are two of the three surfaces the precedent this
    wiring follows shipped together -- `--llm-gateway-provider` landed on `monoid run`, on
    `monoid backend serve`, and as the field, because a switch reachable from two of three leaves
    the served deployment with `monoid gc` and `monoid validate` and no way to produce what they
    consume. Both parities are pinned: absent flags must leave the deployment recording nothing.
    """
    built: list[Any] = []

    def capture(runner_backend, **_kwargs):
        built.append(runner_backend)
        raise KeyboardInterrupt  # stop before the socket; serve_forever is not under test

    monkeypatch.setattr("monoid_agent_kernel.cli.create_backend_server", capture)
    monkeypatch.setenv("MONOID_BACKEND_ADMIN_TOKEN", "admin-token")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    argv = [
        "backend", "serve",
        "--run-root", str(tmp_path / "runs"),
        "--workspace-root", str(workspace),
        "--llm-gateway-url", "http://llm-gateway.internal/v1/turns",
        "--ephemeral-token-secret",
    ]
    if requested:
        argv += ["--model-calls-file", "--model-payload-file", "--model-content-file"]

    result = CliRunner().invoke(main, argv)

    assert built, result.output
    backend = built[0]
    try:
        # All three private sidecars, because a deployment reachable for two of them is the
        # asymmetry this branch exists to close, and `monoid validate` re-checks all three.
        assert backend.model_calls_file is requested
        assert backend.model_payload_file is requested
        assert backend.model_content_file is requested
    finally:
        backend.shutdown()


@pytest.mark.parametrize(
    "command",
    [["backend", "serve"], ["llm-gateway", "serve"], ["web-gateway", "serve"]],
    ids=["backend", "llm-gateway", "web-gateway"],
)
def test_every_serve_command_reports_a_bind_failure_as_a_cli_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: list[str]
) -> None:
    """Three commands, one rule. The first version of this fix was bound on `backend serve` alone
    while its two siblings, eighty and two hundred lines below in the same file, kept the bare
    traceback the commit message said had been removed -- and that message's own words were "the
    CLI error *every other* startup failure gets".

    Port 99999 rather than a genuinely bound socket, because that is the shape that escaped:
    `click`'s `int` accepts it and the socket layer answers with `OverflowError`, not `OSError`,
    so an `except OSError` catches nothing at all here.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in (
        "MONOID_BACKEND_ADMIN_TOKEN",
        "MONOID_LLM_GATEWAY_ADMIN_TOKEN",
        "MONOID_WEB_GATEWAY_ADMIN_TOKEN",
    ):
        monkeypatch.setenv(name, "admin-token")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    argv = [*command, "--port", "99999", "--ephemeral-token-secret"]
    if command[0] == "backend":
        argv += [
            "--run-root", str(tmp_path / "runs"),
            "--workspace-root", str(workspace),
            "--llm-gateway-url", "http://llm-gateway.internal/v1/turns",
        ]

    result = CliRunner().invoke(main, argv)

    assert result.exit_code != 0
    assert not isinstance(result.exception, (OSError, OverflowError)), (
        f"the bind failure escaped as a traceback: {result.exception!r}"
    )
    assert "could not listen on" in result.output, result.output


def test_backend_serve_releases_the_backend_when_the_socket_cannot_be_taken(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bound port is the everyday failure of this command, and it happens after the backend is
    built. With the release keyed to `serve_forever`, that path returned a constructed backend to
    nobody and reported a bare `OSError` traceback instead of the CLI error every other startup
    failure gets."""
    from monoid_agent_kernel.reference.backend.service import RunnerBackend

    released: list[str] = []

    class _Tracking(RunnerBackend):  # type: ignore[misc]
        def shutdown(self, *args: object, **kwargs: object) -> object:
            released.append("shutdown")
            return super().shutdown(*args, **kwargs)

    monkeypatch.setattr("monoid_agent_kernel.cli.RunnerBackend", _Tracking)
    monkeypatch.setattr(
        "monoid_agent_kernel.cli.create_backend_server",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError(48, "address already in use")),
    )
    monkeypatch.setenv("MONOID_BACKEND_ADMIN_TOKEN", "admin-token")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = CliRunner().invoke(
        main,
        [
            "backend", "serve",
            "--run-root", str(tmp_path / "runs"),
            "--workspace-root", str(workspace),
            "--llm-gateway-url", "http://llm-gateway.internal/v1/turns",
            "--ephemeral-token-secret",
        ],
    )

    assert result.exit_code != 0
    assert released == ["shutdown"], "the constructed backend was never released"
    assert not isinstance(result.exception, OSError), (
        "the bind failure must be a reported CLI error, not a traceback"
    )
    assert "could not listen on" in result.output, result.output


def test_the_kernel_layer_reaches_every_sdk_call_through_one_helper() -> None:
    """The census: every `.responses` access in the adapter sits on a `_call_client(...)` result.

    The OpenAI SDK's own retry loop is not governed by `ModelRetryConfig` -- the adapter only
    reads its evidence -- so the layer contract can only reach it through client options, and
    `_call_client` is the single place that happens. Derived from the source rather than
    listed, so a new SDK call site joins the census by existing instead of by being
    remembered.
    """

    import ast as ast_module

    source = Path(openai_module.__file__).read_text(encoding="utf-8")
    accesses = [
        node
        for node in ast_module.walk(ast_module.parse(source))
        if isinstance(node, ast_module.Attribute) and node.attr == "responses"
    ]
    assert accesses, "the census matched no SDK call route; the adapter moved -- re-derive it"
    bypassing = [
        node.lineno
        for node in accesses
        if not (
            isinstance(node.value, ast_module.Call)
            and isinstance(node.value.func, ast_module.Name)
            and node.value.func.id == "_call_client"
        )
    ]
    assert not bypassing, f"SDK call routes bypassing _call_client at lines {bypassing}"


def test_the_kernel_layer_disables_the_sdks_own_retries() -> None:
    """`layer="kernel"` reaches the SDK as `max_retries=0`; the default leaves it untouched.

    Identity for the default layer matters as much as the option for the kernel one: the
    client may be scope-cached, and an options copy taken on every call under the default
    would be a behavior change nobody asked for.
    """

    class _Recording:
        def __init__(self) -> None:
            self.options: dict[str, Any] | None = None

        def with_options(self, **kwargs: Any) -> _Recording:
            self.options = dict(kwargs)
            return self

    kernel_client = _Recording()
    kernel_config = ModelConfig(retry=ModelRetryConfig(layer="kernel"))
    assert openai_module._call_client(kernel_client, kernel_config) is kernel_client
    assert kernel_client.options == {"max_retries": 0}

    default_client = _Recording()
    assert openai_module._call_client(default_client, ModelConfig()) is default_client
    assert default_client.options is None
