from __future__ import annotations

import asyncio
import io
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

import pytest
from click.testing import CliRunner

from support.runtime import runtime_config

from monoid_agent_kernel.cli import main
from monoid_agent_kernel.core.spec import ModelConfig, ModelRetryConfig, ReasoningConfig
from monoid_agent_kernel.errors import ModelAdapterError, RunTimeout
from monoid_agent_kernel.providers.base import ModelRequest, ToolObservation, assemble_streamed_turn
from monoid_agent_kernel.providers.gateway import GatewayModelAdapter, _parse_gateway_response
from monoid_agent_kernel.providers.openai import OpenAIModelAdapter
from monoid_agent_kernel.tools.base import ToolResult, ToolSpec


def _tool() -> ToolSpec:
    def handler(_context, _args):
        return ToolResult(ok=True)

    return ToolSpec(
        id="fs.read",
        description="Read a file.",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        capability="fs.read",
        side_effect="read",
        handler=handler,
        path_args=("path",),
    )


def test_gateway_payload_is_provider_keyless_and_uses_opaque_turn_handle(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text("run-token", encoding="utf-8")
    adapter = GatewayModelAdapter(
        ModelConfig(
            model="gpt-5.5",
            reasoning=ReasoningConfig(effort="low", summary="auto"),
            gateway_url="https://llm-gateway.internal/v1/turns",
        ),
        token_file=token_file,
    )
    request = ModelRequest(
        instruction="Inspect files.",
        system_prompt="sys",
        tools=(_tool(),),
        previous_turn_handle=None,
    )

    payload = adapter._payload(request)
    headers = adapter._headers()

    assert payload["protocol"] == "monoid.llm-turn.v1"
    assert payload["model"] == "gpt-5.5"
    assert payload["reasoning"] == {"effort": "low", "summary": "auto"}
    assert payload["tools"][0]["name"] == "fs_read"
    assert payload["instruction"] == "Inspect files."
    assert "api_key" not in str(payload).lower()
    assert headers["Authorization"] == "Bearer run-token"

    # Tool continuation: handle + observations, no new user message.
    followup = adapter._payload(
        ModelRequest(
            instruction=None,
            system_prompt="sys",
            tools=(_tool(),),
            previous_turn_handle="opaque-turn-handle",
            observations=(ToolObservation("call_1", "fs_read", {"ok": True}),),
        )
    )
    assert followup["previous_turn_handle"] == "opaque-turn-handle"
    assert "previous_response_id" not in followup
    assert "instruction" not in followup
    assert followup["observations"][0]["call_id"] == "call_1"

    # Third shape: a new user message on top of an existing continuation handle.
    user_followup = adapter._payload(
        ModelRequest(
            instruction="Now also summarize.",
            system_prompt="sys",
            tools=(_tool(),),
            previous_turn_handle="opaque-turn-handle",
        )
    )
    assert user_followup["previous_turn_handle"] == "opaque-turn-handle"
    assert user_followup["instruction"] == "Now also summarize."
    assert user_followup["observations"] == []


def test_gateway_adapter_prefers_token_provider_and_reresolves() -> None:
    # The refresh seam: a token_provider takes precedence over the static token and is consulted on
    # every request (so a backend that re-mints near expiry keeps a long run authenticated).
    calls = {"n": 0}

    def provider() -> str:
        calls["n"] += 1
        return f"tok-{calls['n']}"

    adapter = GatewayModelAdapter(
        ModelConfig(gateway_url="https://llm-gateway.internal/v1/turns"),
        token="static",
        token_provider=provider,
    )
    assert adapter._headers()["Authorization"] == "Bearer tok-1"
    assert adapter._headers()["Authorization"] == "Bearer tok-2"  # re-resolved each request

    # No provider -> the static token is used unchanged (back-compat).
    plain = GatewayModelAdapter(
        ModelConfig(gateway_url="https://llm-gateway.internal/v1/turns"), token="static"
    )
    assert plain._headers()["Authorization"] == "Bearer static"


def test_gateway_adapter_prefers_monoid_env_and_accepts_legacy_alias(monkeypatch) -> None:
    monkeypatch.setenv("MONOID_LLM_GATEWAY_URL", "https://monoid-gateway.internal/v1/turns")
    monkeypatch.setenv("MONOID_LLM_GATEWAY_TOKEN", "monoid-token")
    monkeypatch.setenv("NAR_LLM_GATEWAY_URL", "https://legacy-gateway.internal/v1/turns")
    monkeypatch.setenv("NAR_LLM_GATEWAY_TOKEN", "legacy-token")

    adapter = GatewayModelAdapter(ModelConfig())

    assert adapter._resolve_gateway_url(ModelConfig()) == "https://monoid-gateway.internal/v1/turns"
    assert adapter._headers()["Authorization"] == "Bearer monoid-token"

    monkeypatch.delenv("MONOID_LLM_GATEWAY_URL")
    monkeypatch.delenv("MONOID_LLM_GATEWAY_TOKEN")

    assert adapter._resolve_gateway_url(ModelConfig()) == "https://legacy-gateway.internal/v1/turns"
    assert adapter._headers()["Authorization"] == "Bearer legacy-token"


def test_gateway_token_source_remints_near_expiry(monkeypatch) -> None:
    from monoid_agent_kernel.reference._shared.tokens import TokenManager
    from monoid_agent_kernel.reference.backend import service as svc

    clock = {"t": 1000.0}
    monkeypatch.setattr(svc.time, "time", lambda: clock["t"])
    manager = TokenManager.from_secret("x" * 32)
    source = svc._GatewayTokenSource(
        token_manager=manager,
        kind="llm_gateway",
        audience="csp.llm-gateway",
        run_id="run_1",
        tenant_id="t",
        user_id="u",
        ttl_s=100,
        refresh_skew_s=20,
    )
    first = source()
    # Refresh boundary = expires_at(1100) - skew(20) = 1080. Before it -> the cached token.
    clock["t"] = 1079.0
    assert source() == first
    # Past the boundary -> a fresh token (new jti), still a valid llm_gateway token for this run.
    clock["t"] = 1081.0
    second = source()
    assert second != first
    claims = manager.verify(second, kind="llm_gateway", audience="csp.llm-gateway", run_id="run_1")
    assert claims.run_id == "run_1"


def test_gateway_response_parser_returns_model_turn() -> None:
    turn = _parse_gateway_response(
        {
            "turn_handle": "turn_1",
            "final_text": None,
            "tool_calls": [
                {"call_id": "call_1", "name": "fs_read", "arguments": '{"path":"a.md"}'}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
    )

    assert turn.response_id == "turn_1"
    assert turn.tool_calls[0].id == "call_1"
    assert turn.tool_calls[0].arguments == {"path": "a.md"}
    assert turn.usage["total_tokens"] == 15


@pytest.mark.parametrize("payload", [[], None, "response"])
def test_gateway_response_parser_rejects_non_object_envelopes(payload: Any) -> None:
    with pytest.raises(ModelAdapterError) as caught:
        _parse_gateway_response(payload)

    assert caught.value.provider_error_code == "gateway_bad_response"
    assert caught.value.retryable is False


def test_gateway_one_shot_normalizes_nonfinite_model_arguments(monkeypatch: Any) -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return (
                b'{"tool_calls":[{"call_id":"call_1","name":"score",'
                b'"arguments":{"value":NaN,"overflow":1e9999}}]}'
            )

    monkeypatch.setattr(
        "monoid_agent_kernel.providers.gateway.urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    adapter = GatewayModelAdapter(
        ModelConfig(gateway_url="http://gateway.local/internal/llm/turns"),
        token="run-token",
    )

    turn = adapter.next_turn(ModelRequest("score", "sys", (), None))

    assert turn.tool_calls[0].arguments == {"value": None, "overflow": None}
    assert turn.raw["tool_calls"][0]["arguments"] == {"value": None, "overflow": None}


@pytest.mark.parametrize(
    "response_body",
    (
        b'{"final_text":"partial","stop_reason":NaN}',
        b'{"final_text":"partial","response_id":Infinity}',
        b'{"final_text":"partial","usage":NaN}',
    ),
)
def test_gateway_one_shot_rejects_nonfinite_envelope_controls(
    monkeypatch: Any,
    response_body: bytes,
) -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return response_body

    monkeypatch.setattr(
        "monoid_agent_kernel.providers.gateway.urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    adapter = GatewayModelAdapter(
        ModelConfig(gateway_url="http://gateway.local/internal/llm/turns"),
        token="run-token",
    )

    with pytest.raises(ModelAdapterError) as caught:
        adapter.next_turn(ModelRequest("finish", "sys", (), None))

    assert caught.value.provider_error_code == "gateway_bad_response"
    assert caught.value.retryable is False


@pytest.mark.parametrize(
    "frame",
    (
        '{"type":NaN}',
        '{"type":"turn_complete","stop_reason":NaN}',
        '{"type":"text_delta","text":NaN}',
    ),
)
def test_gateway_sse_rejects_nonfinite_envelope_controls(frame: str) -> None:
    from monoid_agent_kernel.providers.gateway import _decode_sse_chunk

    with pytest.raises(ModelAdapterError) as caught:
        _decode_sse_chunk([frame])

    assert caught.value.provider_error_code == "gateway_bad_response"
    assert caught.value.retryable is False


def test_gateway_sse_preserves_surrogate_pairs_split_across_text_frames() -> None:
    from monoid_agent_kernel.providers.gateway import _aiter_sse_chunks

    frames = [
        {"type": "text_delta", "text": "\ud83d"},
        {"type": "text_delta", "text": "\ude00"},
        {"type": "turn_complete"},
    ]

    class Response:
        async def aiter_lines(self):
            for frame in frames:
                yield f"data: {json.dumps(frame)}"
                yield ""

    async def consume() -> list[Any]:
        return [chunk async for chunk in _aiter_sse_chunks(Response())]

    turn = assemble_streamed_turn(asyncio.run(consume()))

    assert turn.final_text == "😀"


def test_gateway_sse_preserves_surrogate_pairs_split_across_tool_arguments() -> None:
    from monoid_agent_kernel.providers.gateway import _aiter_sse_chunks

    frames = [
        {
            "type": "tool_call_delta",
            "index": 0,
            "id": "call_1",
            "name": "score",
            "arguments_fragment": '{"emoji":"\ud83d',
        },
        {
            "type": "tool_call_delta",
            "index": 0,
            "arguments_fragment": '\ude00"}',
        },
        {"type": "turn_complete"},
    ]

    class Response:
        async def aiter_lines(self):
            for frame in frames:
                yield f"data: {json.dumps(frame)}"
                yield ""

    async def consume() -> list[Any]:
        return [chunk async for chunk in _aiter_sse_chunks(Response())]

    turn = assemble_streamed_turn(asyncio.run(consume()))

    assert turn.tool_calls[0].arguments == {"emoji": "😀"}


@pytest.mark.parametrize(
    "payload",
    (
        {"provider_retried": True, "final_text": 42},
        {"provider_retried": True, "final_text": "partial", "usage": {"input_tokens": 1.5}},
        {"provider_retried": True, "error": 42},
    ),
)
def test_gateway_response_validation_preserves_upstream_retry_evidence(
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ModelAdapterError) as caught:
        _parse_gateway_response(payload)

    assert caught.value.provider_error_code == "gateway_bad_response"
    assert caught.value.provider_retried is True


@pytest.mark.parametrize(
    "frame",
    (
        {"type": "text_delta", "provider_retried": True, "text": 42},
        {
            "type": "turn_complete",
            "provider_retried": True,
            "usage": {"output_tokens": 1.5},
        },
    ),
)
def test_gateway_stream_validation_preserves_upstream_retry_evidence(
    frame: dict[str, Any],
) -> None:
    from monoid_agent_kernel.providers.gateway import _chunk_from_event

    with pytest.raises(ModelAdapterError) as caught:
        _chunk_from_event(frame)

    assert caught.value.provider_error_code == "gateway_bad_response"
    assert caught.value.provider_retried is True


@pytest.mark.parametrize(
    ("payload", "field_name"),
    [
        ({"final_text": "ok", "retryable": "false"}, "retryable"),
        ({"final_text": "ok", "provider_retried": "false"}, "provider_retried"),
    ],
)
def test_gateway_one_shot_rejects_truthy_non_boolean_controls(
    monkeypatch: Any,
    payload: dict[str, Any],
    field_name: str,
) -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(
        "monoid_agent_kernel.providers.gateway.urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    adapter = GatewayModelAdapter(
        ModelConfig(gateway_url="http://gateway.local/internal/llm/turns"),
        token="run-token",
    )

    with pytest.raises(ModelAdapterError) as caught:
        adapter.next_turn(ModelRequest("finish", "sys", (), None))

    assert caught.value.provider_error_code == "gateway_bad_response"
    assert caught.value.retryable is False
    assert caught.value.provider_retried is False
    assert field_name in str(caught.value)


def test_gateway_one_shot_maps_non_utf8_json_to_bad_response(monkeypatch: Any) -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self) -> bytes:
            return b"\xff\xfe not utf-8"

    monkeypatch.setattr(
        "monoid_agent_kernel.providers.gateway.urlopen",
        lambda *_args, **_kwargs: Response(),
    )
    adapter = GatewayModelAdapter(
        ModelConfig(gateway_url="http://gateway.local/internal/llm/turns"),
        token="run-token",
    )

    with pytest.raises(ModelAdapterError) as caught:
        adapter.next_turn(ModelRequest("finish", "sys", (), None))

    assert caught.value.provider_error_code == "gateway_bad_response"
    assert caught.value.retryable is False
    assert isinstance(caught.value.__cause__, UnicodeDecodeError)


@pytest.mark.parametrize(
    "frame",
    [
        {"type": "text_delta", "text": "hi", "provider_retried": "false"},
        {"type": "error", "error": "busy", "retryable": "false"},
        {"type": "error", "error": "busy", "config_recoverable": "false"},
    ],
)
def test_gateway_sse_rejects_truthy_non_boolean_controls(frame: dict[str, Any]) -> None:
    from monoid_agent_kernel.providers.gateway import _aiter_sse_chunks

    class Response:
        async def aiter_lines(self):
            yield f"data: {json.dumps(frame)}"
            yield ""

    async def consume() -> list[Any]:
        return [chunk async for chunk in _aiter_sse_chunks(Response())]

    with pytest.raises(ModelAdapterError) as caught:
        asyncio.run(consume())

    assert caught.value.provider_error_code == "gateway_bad_response"
    assert caught.value.retryable is False
    assert caught.value.provider_retried is False


@pytest.mark.parametrize("invalid", ["1", True, 1.9, -1])
def test_gateway_sse_rejects_coercible_tool_call_indices(invalid: object) -> None:
    from monoid_agent_kernel.providers.gateway import _chunk_from_event

    with pytest.raises(ModelAdapterError) as caught:
        _chunk_from_event(
            {
                "type": "tool_call_delta",
                "index": invalid,
                "arguments_fragment": "{}",
            }
        )

    assert caught.value.provider_error_code == "gateway_bad_response"
    assert caught.value.retryable is False
    assert "index" in str(caught.value)


@pytest.mark.parametrize(
    "payload",
    [
        {"error": "bad", "http_status": "503"},
        {"type": "error", "error": "bad", "http_status": 1.9},
    ],
)
def test_gateway_rejects_coercible_wire_http_status(payload: dict[str, Any]) -> None:
    from monoid_agent_kernel.providers.gateway import _chunk_from_event, _parse_gateway_response

    parser = _chunk_from_event if "type" in payload else _parse_gateway_response
    with pytest.raises(ModelAdapterError) as caught:
        parser(payload)

    assert caught.value.provider_error_code == "gateway_bad_response"
    assert caught.value.retryable is False
    assert "http_status" in str(caught.value)


@pytest.mark.parametrize("field_name", ["retryable", "provider_retried", "config_recoverable"])
def test_gateway_non_200_rejects_truthy_non_boolean_controls(field_name: str) -> None:
    from monoid_agent_kernel.providers.gateway import _error_from_status_body

    body = json.dumps({"error": "busy", field_name: "false"})
    with pytest.raises(ModelAdapterError) as caught:
        _error_from_status_body(503, body)

    assert caught.value.provider_error_code == "gateway_bad_response"
    assert caught.value.http_status == 503
    assert caught.value.retryable is False
    assert caught.value.provider_retried is False
    assert field_name in str(caught.value)


@pytest.mark.parametrize(
    "payload",
    [
        {"provider_retried": True, "error": 42},
        {"provider_retried": True, "error": "busy", "retryable": "false"},
    ],
)
def test_gateway_non_200_validation_preserves_upstream_retry_evidence(
    payload: dict[str, Any],
) -> None:
    from monoid_agent_kernel.providers.gateway import _error_from_status_body

    with pytest.raises(ModelAdapterError) as caught:
        _error_from_status_body(503, json.dumps(payload))

    assert caught.value.provider_error_code == "gateway_bad_response"
    assert caught.value.retryable is False
    assert caught.value.provider_retried is True


@pytest.mark.parametrize(
    ("invalid_fields", "expected_retried"),
    [
        ({"error": 42}, True),
        ({"error": "busy", "retryable": "false"}, True),
        ({"error": "busy", "provider_retried": "false"}, False),
        # The third boolean control on this wire goes through the same exact-boolean reader, so
        # a coerced ``"false"`` must refuse on all three readers rather than authorize a config
        # fix nobody stated.
        ({"error": "busy", "config_recoverable": "false"}, True),
    ],
)
def test_gateway_error_validation_preserves_valid_status_and_retry_evidence(
    invalid_fields: dict[str, Any],
    expected_retried: bool,
) -> None:
    from monoid_agent_kernel.providers.gateway import _chunk_from_event, _error_from_status_body

    payload = {"provider_retried": True, "http_status": 400, **invalid_fields}
    parsers = (
        lambda: _parse_gateway_response(payload),
        lambda: _chunk_from_event({"type": "error", **payload}),
        lambda: _error_from_status_body(400, json.dumps(payload)),
    )

    for parse in parsers:
        with pytest.raises(ModelAdapterError) as caught:
            parse()
        assert caught.value.provider_error_code == "gateway_bad_response"
        assert caught.value.http_status == 400
        assert caught.value.retryable is False
        assert caught.value.provider_retried is expected_retried


def test_gateway_non_200_rejects_json_array_error_envelopes() -> None:
    from monoid_agent_kernel.providers.gateway import _error_from_status_body

    error = _error_from_status_body(503, '[{"error":"busy"}]')

    assert error.provider_error_code == "gateway_bad_response"
    assert error.retryable is False
    assert error.provider_retried is False


def test_gateway_retries_retryable_http_error_then_succeeds(monkeypatch) -> None:
    calls = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"turn_handle":"turn_ok","final_text":"done","usage":{"total_tokens":1}}'

    def fake_urlopen(request, timeout):
        del timeout
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPError(
                request.full_url,
                429,
                "Too Many Requests",
                {},
                io.BytesIO(
                    b'{"error":"rate limited","error_code":"gateway_rate_limited","retryable":true}'
                ),
            )
        return Response()

    monkeypatch.setattr("monoid_agent_kernel.providers.gateway.urlopen", fake_urlopen)
    monkeypatch.setattr("monoid_agent_kernel.providers.gateway.time.sleep", lambda _delay: None)
    adapter = GatewayModelAdapter(
        ModelConfig(
            gateway_url="http://gateway.local/internal/llm/turns",
            retry=ModelRetryConfig(max_attempts=2, initial_delay_s=0, jitter_s=0),
        ),
        token="run-token",
    )

    turn = adapter.next_turn(ModelRequest("finish", "sys", (), None))

    assert calls == 2
    assert turn.final_text == "done"


def test_gateway_retries_transient_connection_error_then_succeeds(monkeypatch) -> None:
    # A bare connection-level error (here ConnectionResetError, an OSError that is neither
    # URLError nor TimeoutError) is transient and must be retried, not surfaced as a failed run.
    calls = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"turn_handle":"turn_ok","final_text":"done","usage":{"total_tokens":1}}'

    def fake_urlopen(request, timeout):
        del request, timeout
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionResetError("connection reset by peer")
        return Response()

    monkeypatch.setattr("monoid_agent_kernel.providers.gateway.urlopen", fake_urlopen)
    monkeypatch.setattr("monoid_agent_kernel.providers.gateway.time.sleep", lambda _delay: None)
    adapter = GatewayModelAdapter(
        ModelConfig(
            gateway_url="http://gateway.local/internal/llm/turns",
            retry=ModelRetryConfig(max_attempts=3, initial_delay_s=0, jitter_s=0),
        ),
        token="run-token",
    )

    turn = adapter.next_turn(ModelRequest("finish", "sys", (), None))

    assert calls == 2
    assert turn.final_text == "done"


def test_gateway_does_not_retry_auth_error(monkeypatch) -> None:
    calls = 0

    def fake_urlopen(request, timeout):
        del timeout
        nonlocal calls
        calls += 1
        raise HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            {},
            io.BytesIO(
                b'{"error":"bad token","error_code":"gateway_auth_error","retryable":false}'
            ),
        )

    monkeypatch.setattr("monoid_agent_kernel.providers.gateway.urlopen", fake_urlopen)
    adapter = GatewayModelAdapter(
        ModelConfig(
            gateway_url="http://gateway.local/internal/llm/turns",
            retry=ModelRetryConfig(max_attempts=3, initial_delay_s=0, jitter_s=0),
        ),
        token="bad-token",
    )

    try:
        adapter.next_turn(ModelRequest("finish", "sys", (), None))
    except ModelAdapterError as exc:
        assert exc.provider_error_code == "gateway_auth_error"
        assert exc.retryable is False
        assert exc.http_status == 401
    else:
        raise AssertionError("GatewayModelAdapter should fail on auth error")
    assert calls == 1


def test_openai_adapter_requires_explicit_direct_provider_allow(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    adapter = OpenAIModelAdapter(ModelConfig(provider="openai"))
    request = ModelRequest("hello", "sys", (), None)

    try:
        adapter.next_turn(request)
    except ModelAdapterError as exc:
        assert "direct provider API access is disabled" in str(exc)
    else:
        raise AssertionError("OpenAIModelAdapter should require explicit direct provider allow")


def test_cli_openai_provider_requires_explicit_direct_allow(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_file = tmp_path / "runtime.json"
    config_file.write_text(
        json.dumps(runtime_config("run.finish", model=ModelConfig(provider="openai")).to_json()),
        encoding="utf-8",
    )
    runner = CliRunner()

    result = runner.invoke(
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
    assert "OpenAI runtime configs require --allow-direct-provider-api" in result.output


def test_adapters_send_full_messages_by_value(tmp_path: Path) -> None:
    # When ModelRequest.messages is set, both adapters send the whole conversation and
    # drop the by-reference handle (vendor-independent continuation).
    messages = (
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [{"id": "c1", "name": "fs_read", "arguments": {"path": "a"}}],
        },
        {"role": "tool", "call_id": "c1", "content": {"ok": True}},
    )
    request = ModelRequest(
        instruction=None,
        system_prompt="sys",
        tools=(_tool(),),
        previous_turn_handle="stale-handle",
        messages=messages,
    )

    token_file = tmp_path / "token"
    token_file.write_text("run-token", encoding="utf-8")
    gateway = GatewayModelAdapter(
        ModelConfig(model="gpt-5.5", gateway_url="https://llm-gateway.internal/v1/turns"),
        token_file=token_file,
    )
    gw_payload = gateway._payload(request)
    assert gw_payload["messages"] == [dict(m) for m in messages]
    assert "previous_turn_handle" not in gw_payload
    assert "instruction" not in gw_payload

    openai = OpenAIModelAdapter(ModelConfig(model="gpt-5.5"), allow_direct_provider_api=True)
    oa_payload = openai._payload(request)
    items = oa_payload["input"]
    assert {"role": "user", "content": "hi"} in items
    assert any(it.get("type") == "function_call" and it.get("call_id") == "c1" for it in items)
    assert any(
        it.get("type") == "function_call_output" and it.get("call_id") == "c1" for it in items
    )
    assert "previous_response_id" not in oa_payload


def test_gateway_reports_the_retry_on_a_successful_turn(monkeypatch) -> None:
    """`attempts` and `provider_retried` are different facts and only the adapter knows the second.

    The kernel counts one adapter call per turn however many attempts happened inside it, so
    without this a call that failed once and succeeded on the retry is recorded as a clean single
    attempt.
    """
    calls = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"turn_handle":"turn_ok","final_text":"done","usage":{"total_tokens":1}}'

    def fake_urlopen(request, timeout):
        del request, timeout
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionResetError("connection reset by peer")
        return Response()

    monkeypatch.setattr("monoid_agent_kernel.providers.gateway.urlopen", fake_urlopen)
    monkeypatch.setattr("monoid_agent_kernel.providers.gateway.time.sleep", lambda _delay: None)
    adapter = GatewayModelAdapter(
        ModelConfig(
            gateway_url="http://gateway.local/internal/llm/turns",
            retry=ModelRetryConfig(max_attempts=3, initial_delay_s=0, jitter_s=0),
        ),
        token="run-token",
    )

    turn = adapter.next_turn(ModelRequest(instruction="hi", system_prompt="s", tools=()))

    assert turn.provider_retried is True

    # Counterweight: a call that succeeds first time reports no retry.
    calls = 1
    assert (
        adapter.next_turn(
            ModelRequest(instruction="hi", system_prompt="s", tools=())
        ).provider_retried
        is False
    )


def test_gateway_keeps_retry_evidence_when_the_final_failure_is_not_an_adapter_error(
    monkeypatch,
) -> None:
    """A retried attempt can still end in something that is not a `ModelAdapterError`.

    A body that is not valid UTF-8 raises `UnicodeDecodeError` at the decode step, which used to
    escape unstamped — so the failure receipt denied a retry that had demonstrably happened. The
    marker has to survive whichever exception type carries the failure out.
    """
    calls = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b"\xff\xfe not utf-8"

    def fake_urlopen(request, timeout):
        del request, timeout
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionResetError("connection reset by peer")
        return Response()

    monkeypatch.setattr("monoid_agent_kernel.providers.gateway.urlopen", fake_urlopen)
    monkeypatch.setattr("monoid_agent_kernel.providers.gateway.time.sleep", lambda _delay: None)
    adapter = GatewayModelAdapter(
        ModelConfig(
            gateway_url="http://gateway.local/internal/llm/turns",
            retry=ModelRetryConfig(max_attempts=3, initial_delay_s=0, jitter_s=0),
        ),
        token="run-token",
    )

    with pytest.raises(ModelAdapterError) as caught:
        adapter.next_turn(ModelRequest(instruction="hi", system_prompt="s", tools=()))

    assert caught.value.provider_error_code == "gateway_bad_response"
    assert caught.value.provider_retried is True


def test_the_shipped_adapter_reports_its_retry_through_the_channel(monkeypatch: Any) -> None:
    """Binds `GatewayModelAdapter` to the seam, not a hand-written fake to itself.

    Every other channel test uses an adapter that calls `report_provider_retried` in its own body,
    so deleting the call from the shipped adapter changed nothing that was checked. Mutation testing
    found exactly that: the line the design rests on had no test holding it.

    Driven through the abandonment the channel exists for -- the worker is still inside its second
    attempt when the run's deadline expires, so nothing it returns is ever read.
    """
    from monoid_agent_kernel.model_call import ModelCallRunner
    from monoid_agent_kernel.providers import gateway as gateway_module

    attempts = {"n": 0}

    def _urlopen(*_args: Any, **_kwargs: Any) -> Any:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise URLError("connection reset")
        time.sleep(3)  # the retried attempt never returns

    monkeypatch.setattr(gateway_module, "urlopen", _urlopen)
    monkeypatch.setattr(gateway_module, "_retry_delay", lambda *_a: 0.0)
    adapter = GatewayModelAdapter(
        config=ModelConfig(
            gateway_url="http://gateway.invalid", retry=ModelRetryConfig(max_attempts=3)
        ),
        token="t",
    )

    async def run() -> None:
        await ModelCallRunner(adapter=adapter, cancel_grace_s=0.05).acall(
            ModelRequest(system_prompt="s", instruction="hi", tools=()),
            deadline=time.time() + 0.3,
        )

    with pytest.raises(RunTimeout) as caught:
        asyncio.run(run())
    assert attempts["n"] == 2, "the fixture must reach a retried attempt"
    assert getattr(caught.value, "provider_retried", False) is True


def test_a_shipped_adapter_that_never_retried_reports_nothing(monkeypatch: Any) -> None:
    """The counterweight: abandonment alone must not be read as a retry."""
    from monoid_agent_kernel.model_call import ModelCallRunner
    from monoid_agent_kernel.providers import gateway as gateway_module

    attempts = {"n": 0}

    def _urlopen(*_args: Any, **_kwargs: Any) -> Any:
        attempts["n"] += 1
        time.sleep(3)  # the very first attempt wedges

    monkeypatch.setattr(gateway_module, "urlopen", _urlopen)
    adapter = GatewayModelAdapter(
        config=ModelConfig(gateway_url="http://gateway.invalid"), token="t"
    )

    async def run() -> None:
        await ModelCallRunner(adapter=adapter, cancel_grace_s=0.05).acall(
            ModelRequest(system_prompt="s", instruction="hi", tools=()),
            deadline=time.time() + 0.3,
        )

    with pytest.raises(RunTimeout) as caught:
        asyncio.run(run())
    assert attempts["n"] == 1
    assert getattr(caught.value, "provider_retried", False) is False


def test_a_backend_retry_survives_a_failed_gateway_call() -> None:
    """The failure half of the wire, which is where this record matters most.

    The success half was wired first and the failure half was left, so a gateway whose backend
    retried and *then* failed still reported a clean single attempt. It shows only when this
    client's own retry loop does not run -- a 400/401/quota, the ordinary failure -- because when
    the client retries too, its own stamp masks the loss.
    """
    from monoid_agent_kernel.providers.gateway import (
        _chunk_from_event,
        _error_from_status_body,
        _parse_gateway_response,
    )
    from monoid_agent_kernel.reference.llm_gateway.http import _stream_error_frame

    retried_body = {
        "error": "refused",
        "error_code": "gateway_bad_request",
        "provider_retried": True,
    }
    with pytest.raises(ModelAdapterError) as one_shot:
        _parse_gateway_response(retried_body)
    assert one_shot.value.provider_retried is True

    with pytest.raises(ModelAdapterError) as frame:
        _chunk_from_event({"type": "error", **retried_body})
    assert frame.value.provider_retried is True

    assert _error_from_status_body(400, json.dumps(retried_body)).provider_retried is True

    emitted = ModelAdapterError("refused", provider_error_code="gateway_bad_request")
    emitted.provider_retried = True
    assert _stream_error_frame(None, emitted)["provider_retried"] is True

    # Counterweight: a wire that says nothing must not be read as claiming a retry.
    silent = json.dumps({"error": "refused", "error_code": "gateway_bad_request"})
    assert _error_from_status_body(400, silent).provider_retried is False
    assert _stream_error_frame(None, ModelAdapterError("refused"))["provider_retried"] is False


def test_duplicate_gateway_error_controls_do_not_override_fail_closed_defaults() -> None:
    from monoid_agent_kernel.providers.gateway import _error_from_status_body

    duplicate_retryable = '{"error":"refused","retryable":false,"retryable":true}'
    duplicate_retried = '{"error":"refused","provider_retried":false,"provider_retried":true}'

    retryable_error = _error_from_status_body(503, duplicate_retryable)
    retried_error = _error_from_status_body(503, duplicate_retried)

    assert retryable_error.retryable is False
    assert retryable_error.provider_error_code == "gateway_bad_response"
    assert retried_error.provider_retried is False


def _http_error(status: int, body: str) -> HTTPError:
    return HTTPError("http://gateway.invalid", status, "err", {}, io.BytesIO(body.encode("utf-8")))


def test_every_error_constructor_reads_the_backend_retry() -> None:
    """All five, not four. `_error_from_http_error` is the sync `next_turn` HTTP path — the
    ordinary 400/401/quota failure — and it was the one the round-trip test never called. Today only
    its delegation to `_error_from_status_body` holds the field; nothing stopped a copy coming back.
    """
    from monoid_agent_kernel.providers.gateway import _error_from_http_error

    body = json.dumps(
        {"error": "refused", "error_code": "gateway_bad_request", "provider_retried": True}
    )
    assert _error_from_http_error(_http_error(400, body)).provider_retried is True
    clean = json.dumps({"error": "refused", "error_code": "gateway_bad_request"})
    assert _error_from_http_error(_http_error(400, clean)).provider_retried is False


def test_a_non_200_body_carries_the_backend_retry_end_to_end() -> None:
    """The server half of the same fact. `_stream_error_frame` was covered; `_write_error` was not."""
    from monoid_agent_kernel.providers.gateway import _error_from_status_body
    from monoid_agent_kernel.reference.llm_gateway.http import _error_body

    body = _error_body(400, "refused", error_code="gateway_bad_request", provider_retried=True)
    assert _error_from_status_body(400, json.dumps(body)).provider_retried is True
    plain = _error_body(400, "refused", error_code="gateway_bad_request")
    assert _error_from_status_body(400, json.dumps(plain)).provider_retried is False


def test_every_gateway_validator_puts_the_status_it_was_given_on_the_error_it_raises() -> None:
    """Four of the six validators could not name the status their caller already knew.

    ``_exact_gateway_bool`` and ``_gateway_string`` forward it and the other four did not, so the
    *same* malformed error envelope produced a classified failure carrying HTTP 400 or one
    carrying nothing at all, decided by which field of it was malformed. Each is driven directly:
    accepting the parameter and dropping it on the floor would satisfy a signature census and
    fix nothing.

    Ten now, not six. Every validator that joined the wire afterwards inherited the same
    obligation -- ``_gateway_reasoning_items`` with X-3's reasoning hop, the two echo
    validators the value-validator census turned up unregistered, and B1's
    ``_validated_reasoning_echo``, which joined the censused way. The list the conformance
    suite pins against is derived from "does it raise", so an eleventh arrives here as a
    failing census rather than as a quiet omission.
    """
    from monoid_agent_kernel.providers.gateway import (
        _exact_gateway_bool,
        _exact_gateway_int,
        _gateway_fragment_string,
        _gateway_reasoning_items,
        _gateway_string,
        _gateway_usage,
        _portable_gateway_payload,
        _validated_generation_echo,
        _validated_reasoning_echo,
        _validated_schema_echo,
    )

    raisers = {
        "_exact_gateway_bool": lambda: _exact_gateway_bool(
            {"retryable": "false"}, "retryable", default=False, context="c", http_status=400
        ),
        "_gateway_string": lambda: _gateway_string(
            {"error": 42}, "error", context="c", http_status=400
        ),
        "_exact_gateway_int": lambda: _exact_gateway_int(
            {"index": "1"}, "index", default=0, context="c", minimum=0, http_status=400
        ),
        "_gateway_fragment_string": lambda: _gateway_fragment_string(
            {"text": 42},
            "text",
            context="c",
            http_status=400,
            known_provider_retried=False,
        ),
        "_gateway_usage": lambda: _gateway_usage(
            {"input_tokens": "many"}, context="c", http_status=400
        ),
        # A non-string object key is what portable JSON cannot carry at all (a non-finite number
        # is substituted, not refused).
        "_portable_gateway_payload": lambda: _portable_gateway_payload(
            {1: "one"}, context="c", http_status=400
        ),
        # An array of objects is the only shape the replay path can hand back to a provider.
        "_gateway_reasoning_items": lambda: _gateway_reasoning_items(
            ["not an object"], context="c", http_status=400
        ),
        "_validated_generation_echo": lambda: _validated_generation_echo(
            "not an object", http_status=400
        ),
        "_validated_schema_echo": lambda: _validated_schema_echo("not a bool", http_status=400),
        "_validated_reasoning_echo": lambda: _validated_reasoning_echo(
            "not an object", http_status=400
        ),
    }

    for name, raiser in raisers.items():
        with pytest.raises(ModelAdapterError) as caught:
            raiser()
        assert caught.value.http_status == 400, name
        assert caught.value.provider_error_code == "gateway_bad_response", name

    # And an unstated status stays unstated rather than being invented from the failure class.
    with pytest.raises(ModelAdapterError) as unstated:
        _gateway_usage({"input_tokens": "many"}, context="c")
    assert unstated.value.http_status is None


def test_every_error_constructor_reads_the_config_recoverability() -> None:
    """The `provider_retried` twin above, for the classification that had no wire slot at all.

    `config_recoverable` says "the remedy is configuration, not another attempt", and it used to
    die at the hop: no server writer emitted it and all three client readers rebuilt `False`, so
    a config-fixable refusal arrived one hop out as an ordinary terminal failure and only the 4xx
    `_model_error_status` picks hinted at it. Every reader is driven, because a fact bound on one
    of them and not its siblings is the shape this wire keeps producing.
    """
    from monoid_agent_kernel.providers.gateway import (
        _chunk_from_event,
        _error_from_http_error,
        _error_from_status_body,
    )

    body = {
        "error": "upstream refused an unproven turn",
        "error_code": "gateway_generation_not_applied",
        "retryable": False,
        "http_status": 422,
        "config_recoverable": True,
    }

    with pytest.raises(ModelAdapterError) as sync_read:
        _parse_gateway_response(dict(body))
    assert sync_read.value.config_recoverable is True

    with pytest.raises(ModelAdapterError) as stream_read:
        _chunk_from_event({"type": "error", **body})
    assert stream_read.value.config_recoverable is True

    assert _error_from_status_body(422, json.dumps(body)).config_recoverable is True
    assert _error_from_http_error(_http_error(422, json.dumps(body))).config_recoverable is True

    # An older gateway that never mentions the key still reads as "not config-fixable" — the
    # compatibility contract of the added field, stated on every reader.
    silent = {key: value for key, value in body.items() if key != "config_recoverable"}
    with pytest.raises(ModelAdapterError) as silent_sync:
        _parse_gateway_response(dict(silent))
    assert silent_sync.value.config_recoverable is False
    with pytest.raises(ModelAdapterError) as silent_stream:
        _chunk_from_event({"type": "error", **silent})
    assert silent_stream.value.config_recoverable is False
    assert _error_from_status_body(422, json.dumps(silent)).config_recoverable is False


def test_both_server_writers_put_the_config_recoverability_on_the_wire() -> None:
    """The writer half: one body definition, two writers, and the same key on both.

    The non-200 body and the SSE terminal frame are separate call sites around `_error_body`,
    which is exactly where a field goes missing — `provider_retried` reached both only because
    they were reviewed together.
    """
    from monoid_agent_kernel.providers.gateway import _error_from_status_body
    from monoid_agent_kernel.reference.llm_gateway.http import _error_body, _stream_error_frame

    refused = ModelAdapterError(
        "upstream refused an unproven turn",
        provider_error_code="gateway_generation_not_applied",
        retryable=False,
        config_recoverable=True,
    )
    body = _error_body(
        422,
        str(refused),
        error_code=refused.provider_error_code,
        retryable=refused.retryable,
        config_recoverable=refused.config_recoverable,
    )
    assert body["config_recoverable"] is True
    assert _error_from_status_body(422, json.dumps(body)).config_recoverable is True

    frame = _stream_error_frame(None, refused)
    assert frame["config_recoverable"] is True
    assert {key: value for key, value in frame.items() if key != "type"} == body

    # The default direction: a failure the gateway raised on its own is not config-fixable, and
    # the key is written rather than omitted, so a reader never has to guess which it was.
    plain = _stream_error_frame(None, ModelAdapterError("refused"))
    assert plain["config_recoverable"] is False


def test_a_first_attempt_failure_does_not_claim_a_retry() -> None:
    """The false-positive direction. Every other test asks whether a real retry is recorded; this
    asks whether an imaginary one is, which a boundary slip on `_stamp_retry` would produce for
    every single-attempt failure in the system.
    """
    adapter = GatewayModelAdapter(
        config=ModelConfig(gateway_url="http://gateway.invalid"), token="t"
    )
    with pytest.raises(ModelAdapterError) as caught:
        _drive_urlopen_error(adapter, _http_error(401, json.dumps({"error": "nope"})))
    assert getattr(caught.value, "provider_retried", False) is False


def _drive_urlopen_error(adapter: GatewayModelAdapter, error: Exception) -> None:
    import monoid_agent_kernel.providers.gateway as gateway_module

    original = gateway_module.urlopen
    try:
        gateway_module.urlopen = lambda *_a, **_k: (_ for _ in ()).throw(error)
        adapter.next_turn(ModelRequest(system_prompt="s", instruction="hi", tools=()))
    finally:
        gateway_module.urlopen = original


def test_the_retry_loops_ask_for_the_schedule_of_the_attempt_that_failed(
    monkeypatch: Any,
) -> None:
    """What the loops *pass* the schedule, which no test bound.

    `_retry_delay` is tested in isolation, so shifting the loops' argument by one moved every
    backoff a step up the curve — real extra seconds per retry — with nothing failing. Recorded
    rather than slept through, so this stays fast.
    """
    import monoid_agent_kernel.providers.gateway as gateway_module

    asked: list[int] = []
    monkeypatch.setattr(
        gateway_module, "_retry_delay", lambda attempt, *_a: asked.append(attempt) or 0.0
    )
    attempts = {"n": 0}

    def _urlopen(*_args: Any, **_kwargs: Any) -> Any:
        attempts["n"] += 1
        raise URLError("reset")

    monkeypatch.setattr(gateway_module, "urlopen", _urlopen)
    adapter = GatewayModelAdapter(
        config=ModelConfig(
            gateway_url="http://gateway.invalid", retry=ModelRetryConfig(max_attempts=3)
        ),
        token="t",
    )
    with pytest.raises(ModelAdapterError):
        adapter.next_turn(ModelRequest(system_prompt="s", instruction="hi", tools=()))

    assert attempts["n"] == 3
    # Indexed by the attempt that just failed: 1 then 2, never 2 then 3.
    assert asked == [1, 2]


def test_the_retry_is_reported_before_the_wait_not_after_it(monkeypatch: Any) -> None:
    """Ordering across the backoff, which every other fixture hides by stubbing the wait to zero.

    The wait is a window the run can end inside — the worker sleeps on its thread while the event
    loop stays free to time out and abandon it — so a report issued after it is a report that may
    never happen.
    """
    import monoid_agent_kernel.providers.gateway as gateway_module

    order: list[str] = []
    monkeypatch.setattr(
        gateway_module,
        "_sleep_before_retry",
        lambda *_a: order.append("wait"),
    )
    monkeypatch.setattr(gateway_module, "report_provider_retried", lambda: order.append("report"))

    def _urlopen(*_args: Any, **_kwargs: Any) -> Any:
        raise URLError("reset")

    monkeypatch.setattr(gateway_module, "urlopen", _urlopen)
    adapter = GatewayModelAdapter(
        config=ModelConfig(
            gateway_url="http://gateway.invalid", retry=ModelRetryConfig(max_attempts=3)
        ),
        token="t",
    )
    with pytest.raises(ModelAdapterError):
        adapter.next_turn(ModelRequest(system_prompt="s", instruction="hi", tools=()))

    assert order == ["report", "wait", "report", "wait"]


def test_the_kernel_layer_turns_the_adapters_own_loop_off(monkeypatch: Any) -> None:
    """Exactly one layer may multiply attempts, and `layer` names it.

    Under `layer="kernel"` the runner owns the retry loop, so this adapter must make exactly
    one HTTP attempt no matter what `max_attempts` says -- the schedule fields govern
    whichever layer loops, not this one. The default-layer half is the control: the same
    refused call IS loop-eligible (retryable, its code in `retry_on`), so the kernel half
    passing cannot mean the error was never retryable to begin with.
    """

    calls: list[int] = []

    def _refused(*_args: Any, **_kwargs: Any) -> Any:
        calls.append(1)
        raise URLError("unreachable")

    monkeypatch.setattr("monoid_agent_kernel.providers.gateway.urlopen", _refused)
    monkeypatch.setattr("monoid_agent_kernel.providers.gateway._retry_delay", lambda *_a: 0.0)

    kernel = GatewayModelAdapter(
        ModelConfig(
            gateway_url="http://gateway.local/internal/llm/turns",
            retry=ModelRetryConfig(max_attempts=3, layer="kernel"),
        ),
        token="run-token",
    )
    with pytest.raises(ModelAdapterError) as caught:
        kernel.next_turn(ModelRequest("go", "sys", (), None))
    assert caught.value.retryable is True
    assert len(calls) == 1

    calls.clear()
    adapter_layer = GatewayModelAdapter(
        ModelConfig(
            gateway_url="http://gateway.local/internal/llm/turns",
            retry=ModelRetryConfig(max_attempts=3),
        ),
        token="run-token",
    )
    with pytest.raises(ModelAdapterError):
        adapter_layer.next_turn(ModelRequest("go", "sys", (), None))
    assert len(calls) == 3
