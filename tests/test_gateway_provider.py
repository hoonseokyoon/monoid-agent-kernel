from __future__ import annotations

import io
from pathlib import Path
from urllib.error import HTTPError

from click.testing import CliRunner

from native_agent_runner.cli import main
from native_agent_runner.core.spec import ModelConfig, ModelRetryConfig, ReasoningConfig
from native_agent_runner.errors import ModelAdapterError
from native_agent_runner.providers.base import ModelRequest, ToolObservation
from native_agent_runner.providers.gateway import GatewayModelAdapter, _parse_gateway_response
from native_agent_runner.providers.openai import OpenAIModelAdapter
from native_agent_runner.tools.base import ToolResult, ToolSpec


def _tool() -> ToolSpec:
    def handler(_context, _args):
        return ToolResult(ok=True)

    return ToolSpec(
        id="fs.read",
        description="Read a file.",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        capability="filesystem.read",
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
        previous_response_id=None,
    )

    payload = adapter._payload(request)
    headers = adapter._headers()

    assert payload["protocol"] == "native-agent-runner.llm-turn.v1"
    assert payload["model"] == "gpt-5.5"
    assert payload["reasoning"] == {"effort": "low", "summary": "auto"}
    assert payload["tools"][0]["name"] == "fs_read"
    assert payload["instruction"] == "Inspect files."
    assert "api_key" not in str(payload).lower()
    assert headers["Authorization"] == "Bearer run-token"

    followup = adapter._payload(
        ModelRequest(
            instruction="ignored on followup",
            system_prompt="sys",
            tools=(_tool(),),
            previous_response_id="opaque-turn-handle",
            observations=(ToolObservation("call_1", "fs_read", {"ok": True}),),
        )
    )
    assert followup["previous_turn_handle"] == "opaque-turn-handle"
    assert "previous_response_id" not in followup
    assert "instruction" not in followup


def test_gateway_response_parser_returns_model_turn() -> None:
    turn = _parse_gateway_response(
        {
            "turn_handle": "turn_1",
            "final_text": None,
            "tool_calls": [{"call_id": "call_1", "name": "fs_read", "arguments": "{\"path\":\"a.md\"}"}],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
    )

    assert turn.response_id == "turn_1"
    assert turn.tool_calls[0].id == "call_1"
    assert turn.tool_calls[0].arguments == {"path": "a.md"}
    assert turn.usage["total_tokens"] == 15


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

    monkeypatch.setattr("native_agent_runner.providers.gateway.urlopen", fake_urlopen)
    monkeypatch.setattr("native_agent_runner.providers.gateway.time.sleep", lambda _delay: None)
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
            io.BytesIO(b'{"error":"bad token","error_code":"gateway_auth_error","retryable":false}'),
        )

    monkeypatch.setattr("native_agent_runner.providers.gateway.urlopen", fake_urlopen)
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
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "run",
            "--workspace",
            str(workspace),
            "--instruction",
            "Finish.",
            "--model-provider",
            "openai",
        ],
    )

    assert result.exit_code != 0
    assert "--model-provider openai requires --allow-direct-provider-api" in result.output
