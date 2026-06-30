from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

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


def test_openai_astream_yields_reasoning_delta_then_text(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")  # exercises the real SDK stream; skip on a minimal install
    # DX-13b: a reasoning-summary stream event maps to a ReasoningDelta, distinct from the
    # answer's TextDelta, ahead of the terminal TurnComplete.
    events = [
        _Ev("response.reasoning_summary_text.delta", delta="think"),
        _Ev("response.output_text.delta", delta="Hi"),
        _Ev("response.completed", response=_StreamResp()),
    ]

    class _Responses:
        async def create(self, **kwargs):  # noqa: ANN003, ANN202
            return _AsyncStream(events)

    class _AsyncClient:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.responses = _Responses()

    monkeypatch.setattr("openai.AsyncOpenAI", _AsyncClient)
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

    class _Responses:
        async def create(self, **kwargs):  # noqa: ANN003, ANN202
            return _AsyncStream(events)

    class _AsyncClient:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.responses = _Responses()

    monkeypatch.setattr("openai.AsyncOpenAI", _AsyncClient)
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

    class _FakeClient:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.responses = _FakeResponses()

    monkeypatch.setattr("openai.OpenAI", _FakeClient)
    adapter = OpenAIModelAdapter(ModelConfig(), api_key="test", allow_direct_provider_api=True)
    request = ModelRequest(instruction="my secret prompt", system_prompt="", tools=())
    with pytest.raises(ModelAdapterError) as excinfo:
        adapter.next_turn(request)
    err = excinfo.value
    assert err.http_status == 400
    assert err.retryable is False
    assert err.provider_error_code == "unsupported_value"
    assert "secret prompt" not in str(err)  # no prompt/body leak
