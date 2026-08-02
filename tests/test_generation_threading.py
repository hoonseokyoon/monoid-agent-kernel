"""W5 PR 2: generation parameters thread client → wire → server → provider config, and the
gateway transport proves application through the ``generation_applied`` echo (scope §5 D-a).

Mutation gate (implementation plan §PR 2): mutating ``build_generation_payload`` must fail
tests on all three surfaces below — the OpenAI request body, the gateway client wire/enforce
path, and the reference gateway server echo. If one survives, the binding is broken.
"""

from __future__ import annotations

import json

import pytest

from monoid_agent_kernel.core.spec import GenerationConfig, ModelConfig, ReasoningConfig
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.providers._common import build_generation_payload
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn, TurnComplete
from monoid_agent_kernel.providers.gateway import (
    GATEWAY_BAD_RESPONSE,
    GATEWAY_GENERATION_NOT_APPLIED,
    GatewayModelAdapter,
    _check_generation_applied,
    _chunk_from_event,
)
from monoid_agent_kernel.providers.openai import OpenAIModelAdapter
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.llm_gateway.service import LlmGatewayBackend

_SET = GenerationConfig(temperature=0.2, top_p=0.9, max_output_tokens=256)
_SET_WIRE = {"temperature": 0.2, "top_p": 0.9, "max_output_tokens": 256}


def _request(config: ModelConfig) -> ModelRequest:
    return ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=config)


# --- the shared payload builder --------------------------------------------------------


def test_build_generation_payload_emits_only_set_keys() -> None:
    assert build_generation_payload(GenerationConfig()) == {}
    assert build_generation_payload(GenerationConfig(top_p=0.5)) == {"top_p": 0.5}
    assert build_generation_payload(_SET) == _SET_WIRE


def test_build_generation_payload_never_carries_policy() -> None:
    """on_unsupported is the caller's policy, not a provider knob -- if it leaked here it
    would reach the OpenAI request body verbatim."""

    payload = build_generation_payload(GenerationConfig(temperature=1, on_unsupported="omit"))
    assert payload == {"temperature": 1}


# --- surface 1: OpenAI request body ----------------------------------------------------


def test_openai_payload_carries_sampling_controls() -> None:
    config = ModelConfig(provider="openai", generation=_SET)
    payload = OpenAIModelAdapter(config)._payload(_request(config))
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.9
    assert payload["max_output_tokens"] == 256


def test_openai_payload_is_unchanged_without_generation() -> None:
    config = ModelConfig(provider="openai")
    payload = OpenAIModelAdapter(config)._payload(_request(config))
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "max_output_tokens" not in payload


# --- surface 2: gateway client wire + enforcement --------------------------------------


def test_gateway_payload_carries_generation_block_only_when_set() -> None:
    configured = ModelConfig(generation=_SET)
    payload = GatewayModelAdapter(config=configured)._payload(_request(configured))
    assert payload["generation"] == _SET_WIRE

    default = ModelConfig()
    assert "generation" not in GatewayModelAdapter(config=default)._payload(_request(default))


def test_gateway_payload_seals_the_reasoning_on_unsupported_drop() -> None:
    config = ModelConfig(reasoning=ReasoningConfig(effort="low", on_unsupported="omit"))
    payload = GatewayModelAdapter(config=config)._payload(_request(config))
    assert payload["reasoning"]["on_unsupported"] == "omit"

    default = ModelConfig(reasoning=ReasoningConfig(effort="low"))
    default_payload = GatewayModelAdapter(config=default)._payload(_request(default))
    assert "on_unsupported" not in default_payload["reasoning"]


def test_check_generation_applied_matrix() -> None:
    # No generation requested: transport owes no proof, any echo state passes.
    _check_generation_applied({}, "fail", None)
    _check_generation_applied({}, "fail", {"temperature": 9})
    # Exact echo is the proof.
    _check_generation_applied(_SET_WIRE, "fail", dict(_SET_WIRE))
    # Absent echo (older gateway) under "omit" is accepted best-effort.
    _check_generation_applied(_SET_WIRE, "omit", None)

    with pytest.raises(ModelAdapterError) as missing:
        _check_generation_applied(_SET_WIRE, "fail", None)
    assert missing.value.provider_error_code == GATEWAY_GENERATION_NOT_APPLIED
    assert missing.value.retryable is False

    with pytest.raises(ModelAdapterError):
        _check_generation_applied(_SET_WIRE, "fail", {"temperature": 0.2})


def test_turn_complete_frame_carries_and_validates_the_echo() -> None:
    frame = {
        "type": "turn_complete",
        "turn_handle": "turn_1",
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "stop_reason": "stop",
        "generation_applied": dict(_SET_WIRE),
    }
    chunk = _chunk_from_event(frame)
    assert isinstance(chunk, TurnComplete)
    assert chunk.generation_applied == _SET_WIRE

    without = _chunk_from_event({**frame, "generation_applied": None})
    assert isinstance(without, TurnComplete)
    assert without.generation_applied is None

    with pytest.raises(ModelAdapterError) as bad:
        _chunk_from_event({**frame, "generation_applied": [1, 2]})
    assert bad.value.provider_error_code == GATEWAY_BAD_RESPONSE


class _FakeHttpResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeHttpResponse:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _served_turn(extra: dict | None = None) -> bytes:
    body = {
        "protocol": "monoid.llm-turn-result.v1",
        "turn_handle": "turn_1",
        "final_text": "ok",
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "stop_reason": "stop",
    }
    body.update(extra or {})
    return json.dumps(body).encode("utf-8")


def test_next_turn_rejects_a_server_that_never_echoes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The old-server simulation: a pre-W5 gateway answers 200 and silently discards the
    generation block. Under the default policy that is an error, not a silent misapplication."""

    import monoid_agent_kernel.providers.gateway as gateway_module

    monkeypatch.setattr(
        gateway_module, "urlopen", lambda *_a, **_k: _FakeHttpResponse(_served_turn())
    )
    config = ModelConfig(generation=_SET, gateway_url="http://gateway.test")
    adapter = GatewayModelAdapter(config=config)

    with pytest.raises(ModelAdapterError) as rejected:
        adapter.next_turn(_request(config))
    assert rejected.value.provider_error_code == GATEWAY_GENERATION_NOT_APPLIED

    omit = ModelConfig(
        generation=GenerationConfig(temperature=0.2, top_p=0.9, max_output_tokens=256, on_unsupported="omit"),
        gateway_url="http://gateway.test",
    )
    turn = GatewayModelAdapter(config=omit).next_turn(_request(omit))
    assert turn.final_text == "ok"


def test_next_turn_accepts_a_matching_echo(monkeypatch: pytest.MonkeyPatch) -> None:
    import monoid_agent_kernel.providers.gateway as gateway_module

    monkeypatch.setattr(
        gateway_module,
        "urlopen",
        lambda *_a, **_k: _FakeHttpResponse(_served_turn({"generation_applied": dict(_SET_WIRE)})),
    )
    config = ModelConfig(generation=_SET, gateway_url="http://gateway.test")
    turn = GatewayModelAdapter(config=config).next_turn(_request(config))
    assert turn.final_text == "ok"


# --- surface 3: the reference gateway server --------------------------------------------


def _token_manager() -> TokenManager:
    return TokenManager.from_secret("y" * 32)


def _llm_token(manager: TokenManager) -> str:
    return manager.issue(
        kind="llm_gateway",
        audience="csp.llm-gateway",
        run_id="run_1",
        tenant_id="tenant_a",
        user_id="user_a",
        ttl_s=600,
        metadata={"agent_config_hash": "test"},
    )


def _turn_payload(**extra: object) -> dict:
    payload: dict = {
        "protocol": "monoid.llm-turn.v1",
        "model": "gpt-5.5",
        "system_prompt": "sys",
        "instruction": "Read notes.",
    }
    payload.update(extra)
    return payload


def _recording_backend() -> tuple[LlmGatewayBackend, TokenManager, list[ModelConfig]]:
    manager = _token_manager()
    captured: list[ModelConfig] = []

    def factory(_claims, config):
        captured.append(config)

        class Adapter:
            def next_turn(self, request):
                return ModelTurn(
                    response_id="provider_1",
                    final_text="done",
                    usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    stop_reason="stop",
                )

        return Adapter()

    backend = LlmGatewayBackend(token_manager=manager, provider_adapter_factory=factory)
    return backend, manager, captured


def test_gateway_service_threads_generation_into_the_upstream_config() -> None:
    backend, manager, captured = _recording_backend()
    backend.handle_turn(
        _llm_token(manager),
        _turn_payload(
            generation=dict(_SET_WIRE),
            reasoning={"effort": "low", "on_unsupported": "omit"},
        ),
    )
    assert captured[0].generation == _SET
    assert captured[0].reasoning.on_unsupported == "omit"


def test_gateway_service_rejects_invalid_generation_at_the_boundary() -> None:
    backend, manager, _ = _recording_backend()
    with pytest.raises(ValueError, match="model.generation.temperature"):
        backend.handle_turn(_llm_token(manager), _turn_payload(generation={"temperature": 5}))


def test_gateway_service_echoes_what_it_applied() -> None:
    backend, manager, _ = _recording_backend()
    result = backend.handle_turn(
        _llm_token(manager), _turn_payload(generation=dict(_SET_WIRE))
    )
    assert result["generation_applied"] == _SET_WIRE


def test_gateway_service_omits_the_echo_without_generation() -> None:
    """Wire stability for generation-free traffic: pre-W5 clients keep seeing the exact
    response shape they were built against."""

    backend, manager, _ = _recording_backend()
    result = backend.handle_turn(_llm_token(manager), _turn_payload())
    assert "generation_applied" not in result


def test_gateway_service_stream_terminal_frame_echoes_too() -> None:
    backend, manager, _ = _recording_backend()
    frames = list(
        backend.handle_turn_stream(
            _llm_token(manager), _turn_payload(generation=dict(_SET_WIRE))
        )
    )
    terminal = frames[-1]
    assert terminal["type"] == "turn_complete"
    assert terminal["generation_applied"] == _SET_WIRE

    plain = list(backend.handle_turn_stream(_llm_token(manager), _turn_payload()))
    assert "generation_applied" not in plain[-1]
