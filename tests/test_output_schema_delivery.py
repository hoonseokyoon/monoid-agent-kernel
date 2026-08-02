"""W5 PR 4: output-schema delivery on the standalone path (ResponseContract, family C).

The schema travels verbatim -- request field → OpenAI ``text.format`` / gateway wire →
upstream config -- and the gateway proves enforcement through the ``schema_applied`` echo,
under the same ``on_unsupported`` policy knob as the sampling parameters. Post-hoc validation
(family B) remains the guarantee on every adapter; these tests pin that the delivery layer
never becomes the guarantee (``parsed`` is a convenience, fallback keeps working).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from monoid_agent_kernel.core.spec import GenerationConfig, ModelConfig
from monoid_agent_kernel.core.output_validator import FinalOutputView, ValidationOutcome
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.model_call import ModelCallRunner, _digest, _request_payload
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelTurn,
    TurnComplete,
    normalize_model_request,
    structured_output_support,
)
from monoid_agent_kernel.providers.fake import FakeModelAdapter
from monoid_agent_kernel.providers.gateway import (
    GATEWAY_BAD_RESPONSE,
    GATEWAY_SCHEMA_NOT_APPLIED,
    GatewayModelAdapter,
    _check_schema_applied,
    _chunk_from_event,
)
from monoid_agent_kernel.providers.openai import OpenAIModelAdapter
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.llm_gateway.service import LlmGatewayBackend
from monoid_agent_kernel.validated_call import ValidatedCallRunner

_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _request(config: ModelConfig | None = None, **kw: object) -> ModelRequest:
    return ModelRequest(
        instruction="answer", system_prompt="sys", tools=(), model=config, **kw
    )


# --- request field + normalization + digest --------------------------------------------


def test_normalize_model_request_threads_and_rejects_non_object_schema() -> None:
    normalized = normalize_model_request(_request(output_schema=dict(_SCHEMA)))
    assert normalized.output_schema == _SCHEMA

    with pytest.raises(ValueError, match="output_schema must be an object or null"):
        normalize_model_request(_request(output_schema=["not", "an", "object"]))  # type: ignore[arg-type]


def test_schema_free_requests_keep_their_digest_and_schema_changes_it() -> None:
    free = _request_payload(_request(), ModelConfig(), provider="fake", destination="")
    assert "output_schema" not in free

    constrained = _request_payload(
        _request(output_schema=dict(_SCHEMA)), ModelConfig(), provider="fake", destination=""
    )
    assert constrained["output_schema"] == _SCHEMA
    assert _digest(constrained) != _digest(free)


# --- surface 1: OpenAI text.format ------------------------------------------------------


def test_openai_payload_delivers_the_schema_verbatim() -> None:
    config = ModelConfig(provider="openai")
    payload = OpenAIModelAdapter(config)._payload(_request(config, output_schema=dict(_SCHEMA)))
    assert payload["text"] == {
        "format": {
            "type": "json_schema",
            "name": "response",
            "strict": True,
            "schema": _SCHEMA,
        }
    }

    without = OpenAIModelAdapter(config)._payload(_request(config))
    assert "text" not in without


def test_support_probe_is_opt_in_and_fail_closed() -> None:
    assert structured_output_support(OpenAIModelAdapter(ModelConfig())) == "native"
    assert structured_output_support(GatewayModelAdapter(config=ModelConfig())) == "native"
    assert structured_output_support(FakeModelAdapter()) == "none"

    class Vague:
        structured_output_support = "probably"

    assert structured_output_support(Vague()) == "none"


# --- surface 2: gateway client wire + enforcement ---------------------------------------


def test_gateway_payload_carries_the_schema_only_when_set() -> None:
    config = ModelConfig()
    adapter = GatewayModelAdapter(config=config)
    assert (
        adapter._payload(_request(config, output_schema=dict(_SCHEMA)))["output_schema"]
        == _SCHEMA
    )
    assert "output_schema" not in adapter._payload(_request(config))


def test_check_schema_applied_matrix() -> None:
    _check_schema_applied(False, "fail", None)  # nothing sent: no proof owed
    _check_schema_applied(True, "fail", True)  # proven
    _check_schema_applied(True, "omit", None)  # best-effort accepted
    _check_schema_applied(True, "omit", False)

    with pytest.raises(ModelAdapterError) as missing:
        _check_schema_applied(True, "fail", None)
    assert missing.value.provider_error_code == GATEWAY_SCHEMA_NOT_APPLIED
    assert missing.value.retryable is False

    with pytest.raises(ModelAdapterError) as refused:
        _check_schema_applied(True, "fail", False)
    assert refused.value.provider_error_code == GATEWAY_SCHEMA_NOT_APPLIED

    with pytest.raises(ModelAdapterError) as malformed:
        _check_schema_applied(True, "fail", "yes")
    assert malformed.value.provider_error_code == GATEWAY_BAD_RESPONSE


@pytest.mark.parametrize("policy", ("fail", "omit"))
@pytest.mark.parametrize("schema_sent", (True, False))
def test_a_malformed_schema_echo_is_a_bad_response_on_both_transports(
    policy: str, schema_sent: bool
) -> None:
    """The generation echo's twin rule: a non-boolean ``schema_applied`` is malformed whatever
    the policy says and whether or not this call sent a schema. The streamed frame parser
    always rejected it; the sync check skipped the shape check entirely when no schema went
    out, so the same bytes were accepted on one transport and refused on the other."""

    with pytest.raises(ModelAdapterError) as sync_side:
        _check_schema_applied(schema_sent, policy, "yes")
    assert sync_side.value.provider_error_code == GATEWAY_BAD_RESPONSE

    with pytest.raises(ModelAdapterError) as stream_side:
        _chunk_from_event(
            {"type": "turn_complete", "turn_handle": "turn_1", "schema_applied": "yes"}
        )
    assert stream_side.value.provider_error_code == GATEWAY_BAD_RESPONSE


def test_turn_complete_frame_carries_and_validates_schema_applied() -> None:
    frame = {
        "type": "turn_complete",
        "turn_handle": "turn_1",
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "stop_reason": "stop",
        "schema_applied": True,
    }
    chunk = _chunk_from_event(frame)
    assert isinstance(chunk, TurnComplete)
    assert chunk.schema_applied is True
    assert chunk.to_json()["schema_applied"] is True

    silent = _chunk_from_event({k: v for k, v in frame.items() if k != "schema_applied"})
    assert isinstance(silent, TurnComplete)
    assert silent.schema_applied is None
    assert "schema_applied" not in silent.to_json()

    with pytest.raises(ModelAdapterError):
        _chunk_from_event({**frame, "schema_applied": "yes"})


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
        "final_text": '{"answer": "ok"}',
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "stop_reason": "stop",
    }
    body.update(extra or {})
    return json.dumps(body).encode("utf-8")


def test_next_turn_enforces_the_schema_echo(monkeypatch: pytest.MonkeyPatch) -> None:
    import monoid_agent_kernel.providers.gateway as gateway_module

    monkeypatch.setattr(
        gateway_module, "urlopen", lambda *_a, **_k: _FakeHttpResponse(_served_turn())
    )
    config = ModelConfig(gateway_url="http://gateway.test")
    adapter = GatewayModelAdapter(config=config)

    with pytest.raises(ModelAdapterError) as rejected:
        adapter.next_turn(_request(config, output_schema=dict(_SCHEMA)))
    assert rejected.value.provider_error_code == GATEWAY_SCHEMA_NOT_APPLIED

    omit_config = ModelConfig(
        gateway_url="http://gateway.test",
        generation=GenerationConfig(on_unsupported="omit"),
    )
    turn = GatewayModelAdapter(config=omit_config).next_turn(
        _request(omit_config, output_schema=dict(_SCHEMA))
    )
    assert turn.final_text == '{"answer": "ok"}'

    monkeypatch.setattr(
        gateway_module,
        "urlopen",
        lambda *_a, **_k: _FakeHttpResponse(_served_turn({"schema_applied": True})),
    )
    proven = GatewayModelAdapter(config=config).next_turn(
        _request(config, output_schema=dict(_SCHEMA))
    )
    assert proven.final_text == '{"answer": "ok"}'


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
        "instruction": "Answer.",
    }
    payload.update(extra)
    return payload


class _NativeUpstream:
    structured_output_support = "native"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        return ModelTurn(
            response_id="provider_1",
            final_text='{"answer": "done"}',
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            stop_reason="stop",
        )


class _PlainUpstream(_NativeUpstream):
    structured_output_support = "none"


def _backend(upstream: _NativeUpstream) -> tuple[LlmGatewayBackend, TokenManager]:
    manager = _token_manager()
    backend = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, _config: upstream,
    )
    return backend, manager


def test_gateway_service_threads_the_schema_to_the_upstream_request() -> None:
    upstream = _NativeUpstream()
    backend, manager = _backend(upstream)
    backend.handle_turn(_llm_token(manager), _turn_payload(output_schema=dict(_SCHEMA)))
    assert upstream.requests[0].output_schema == _SCHEMA


def test_gateway_service_echoes_schema_applied_per_upstream_support() -> None:
    native_backend, manager = _backend(_NativeUpstream())
    result = native_backend.handle_turn(
        _llm_token(manager), _turn_payload(output_schema=dict(_SCHEMA))
    )
    assert result["schema_applied"] is True

    plain_backend, plain_manager = _backend(_PlainUpstream())
    honest = plain_backend.handle_turn(
        _llm_token(plain_manager), _turn_payload(output_schema=dict(_SCHEMA))
    )
    assert honest["schema_applied"] is False

    schema_free = native_backend.handle_turn(_llm_token(manager), _turn_payload())
    assert "schema_applied" not in schema_free


def test_gateway_service_stream_terminal_frame_echoes_schema_applied() -> None:
    backend, manager = _backend(_NativeUpstream())
    frames = list(
        backend.handle_turn_stream(
            _llm_token(manager), _turn_payload(output_schema=dict(_SCHEMA))
        )
    )
    assert frames[-1]["type"] == "turn_complete"
    assert frames[-1]["schema_applied"] is True

    plain = list(backend.handle_turn_stream(_llm_token(manager), _turn_payload()))
    assert "schema_applied" not in plain[-1]


# --- the standalone executor: parsed + fallback -----------------------------------------


class _SeesParsed:
    id = "sees-parsed"
    schema = None

    def __init__(self) -> None:
        self.parsed_seen: list[object] = []

    def validate(self, view: FinalOutputView) -> ValidationOutcome:
        self.parsed_seen.append(view.parsed)
        return ValidationOutcome(ok=True, value=view.parsed)


def test_validated_call_populates_parsed_when_a_schema_was_requested() -> None:
    validator = _SeesParsed()
    runner = ValidatedCallRunner(
        runner=ModelCallRunner(
            adapter=FakeModelAdapter(
                turns=[ModelTurn(final_text='{"answer": "hi"}', stop_reason="stop")]
            )
        ),
        validators=(validator,),
    )
    result = asyncio.run(runner.acall(_request(output_schema=dict(_SCHEMA))))
    assert result.status == "ok"
    assert validator.parsed_seen == [{"answer": "hi"}]
    assert result.value == {"answer": "hi"}


@pytest.mark.parametrize(
    ("request_kw", "final_text"),
    [
        ({}, '{"answer": "hi"}'),  # no schema requested -> parsed stays None
        ({"output_schema": _SCHEMA}, "plain prose"),  # not JSON -> parsed stays None
        # Python's json.loads accepts these non-standard constants. A validator reading
        # `parsed` -- a schema validator will call NaN a number -- would then accept an answer
        # that is not JSON at all, so they must leave `parsed` exactly where prose leaves it.
        ({"output_schema": _SCHEMA}, "NaN"),
        ({"output_schema": _SCHEMA}, "Infinity"),
        ({"output_schema": _SCHEMA}, '{"answer": NaN}'),
        ({"output_schema": _SCHEMA}, '{"answer": -Infinity}'),
        # The neighbouring strictness the same ingress brings, for the same reason.
        ({"output_schema": _SCHEMA}, '{"answer": "a", "answer": "b"}'),
        ({"output_schema": _SCHEMA}, '{"answer": 1e400}'),
    ],
)
def test_parsed_stays_none_without_schema_or_without_json(
    request_kw: dict, final_text: str
) -> None:
    validator = _SeesParsed()
    runner = ValidatedCallRunner(
        runner=ModelCallRunner(
            adapter=FakeModelAdapter(turns=[ModelTurn(final_text=final_text, stop_reason="stop")])
        ),
        validators=(validator,),
    )
    result = asyncio.run(runner.acall(_request(**dict(request_kw))))
    assert result.status == "ok"
    assert validator.parsed_seen == [None]


def test_repair_preserves_the_schema_while_stripping_tools() -> None:
    """Family-B fallback pin: on an adapter with no native support the schema still rides
    every call (including repairs), tools never do, and validation drives the outcome."""

    class _WantsJson:
        id = "wants-json"
        schema = None

        def validate(self, view: FinalOutputView) -> ValidationOutcome:
            if view.parsed is None:
                return ValidationOutcome(ok=False, feedback="must be JSON")
            return ValidationOutcome(ok=True, value=view.parsed)

    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(final_text="prose", stop_reason="stop"),
            ModelTurn(final_text='{"answer": "fixed"}', stop_reason="stop"),
        ]
    )
    runner = ValidatedCallRunner(
        runner=ModelCallRunner(adapter=adapter), validators=(_WantsJson(),)
    )
    result = asyncio.run(runner.acall(_request(output_schema=dict(_SCHEMA))))
    assert result.status == "ok"
    assert result.value == {"answer": "fixed"}
    repair = adapter.requests[1]
    assert repair.output_schema == _SCHEMA
    assert repair.tools == ()
