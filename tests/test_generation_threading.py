"""W5 PR 2: generation parameters thread client → wire → server → provider config, and the
gateway transport proves application through the ``generation_applied`` echo (scope §5 D-a).

Mutation gate (implementation plan §PR 2): mutating ``build_generation_payload`` must fail
tests on all three surfaces below — the OpenAI request body, the gateway client wire/enforce
path, and the reference gateway server echo. If one survives, the binding is broken.
"""

from __future__ import annotations

import json
from dataclasses import replace
from http import HTTPStatus
from urllib.error import URLError

import pytest

from monoid_agent_kernel.core.spec import GenerationConfig, ModelConfig, ReasoningConfig
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.providers._common import build_generation_payload
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelTurn,
    TurnComplete,
    generation_support,
    provider_usage_of,
    structured_output_support,
)
from monoid_agent_kernel.providers.fake import FakeModelAdapter
from monoid_agent_kernel.providers.gateway import (
    GATEWAY_BAD_RESPONSE,
    GATEWAY_GENERATION_NOT_APPLIED,
    GatewayModelAdapter,
    _check_generation_applied,
    _chunk_from_event,
)
from monoid_agent_kernel.providers.openai import OpenAIModelAdapter
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.llm_gateway.service import (
    LlmGatewayBackend,
    LlmGatewayTurnRequest,
    _applied_echoes,
    _upstream_model_config,
)

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


def test_gateway_wire_carries_the_default_effort_sentinel() -> None:
    """``effort="default"`` is the one reasoning field whose omission sentinel differs from
    the codec's reconstruction default ("medium"): left off the wire, the server's rebuilt
    config silently asked its upstream for medium reasoning on a call that asked for the
    provider default. The client payload therefore carries it explicitly — and only it; every
    other effort keeps its exact pre-W5 wire shape."""

    config = ModelConfig(reasoning=ReasoningConfig(effort="default"))
    payload = GatewayModelAdapter(config=config)._payload(_request(config))
    assert payload["reasoning"] == {"effort": "default"}
    # The server-side codec reads the block back as what the client meant.
    assert ReasoningConfig.from_json(payload["reasoning"]).effort == "default"

    default_config = ModelConfig()
    default_payload = GatewayModelAdapter(config=default_config)._payload(
        _request(default_config)
    )
    assert default_payload["reasoning"] == {"effort": "medium"}


def test_gateway_payload_seals_the_generation_on_unsupported_drop() -> None:
    """The reasoning fix's twin. The server rebuilds a GenerationConfig from this block, so a
    field left off is not "unset" there -- it is the default, and a caller's "omit" came back
    as "fail". It matters as soon as a gateway's upstream is another gateway: the next hop
    enforces the reset policy and rejects a turn the caller asked to accept best-effort. The
    same knob gates the schema echo, so schema callers are reset too."""

    config = ModelConfig(
        generation=GenerationConfig(temperature=0.2, on_unsupported="omit"),
    )
    payload = GatewayModelAdapter(config=config)._payload(_request(config))
    assert payload["generation"] == {"temperature": 0.2, "on_unsupported": "omit"}

    default = ModelConfig(generation=GenerationConfig(temperature=0.2))
    default_payload = GatewayModelAdapter(config=default)._payload(_request(default))
    assert "on_unsupported" not in default_payload["generation"]

    # Policy alone is still an explicit configuration -- it must survive even with no sampling
    # values, because that is exactly the shape an output_schema-only caller sends.
    policy_only = ModelConfig(generation=GenerationConfig(on_unsupported="omit"))
    policy_payload = GatewayModelAdapter(config=policy_only)._payload(_request(policy_only))
    assert policy_payload["generation"] == {"on_unsupported": "omit"}


def test_gateway_server_reconstructs_the_caller_policy_from_the_wire() -> None:
    """End of the hop: what the client seals must be what the server rebuilds, or the fix is
    only half a wire."""

    backend, manager, captured = _recording_backend()
    config = ModelConfig(generation=GenerationConfig(temperature=0.2, on_unsupported="omit"))
    wire = GatewayModelAdapter(config=config)._payload(_request(config))

    backend.handle_turn(_llm_token(manager), _turn_payload(generation=wire["generation"]))
    assert captured[0].generation == GenerationConfig(temperature=0.2, on_unsupported="omit")

    policy_only = ModelConfig(generation=GenerationConfig(on_unsupported="omit"))
    policy_wire = GatewayModelAdapter(config=policy_only)._payload(_request(policy_only))
    backend.handle_turn(
        _llm_token(manager), _turn_payload(generation=policy_wire["generation"])
    )
    assert captured[1].generation.on_unsupported == "omit"


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


def _recording_backend(
    *, upstream_applies: bool = True
) -> tuple[LlmGatewayBackend, TokenManager, list[ModelConfig]]:
    """A backend whose upstream adapter is a stub.

    ``upstream_applies`` mirrors the real question the echo answers: does the adapter behind
    the gateway actually put the sampling controls on its provider request? Only an adapter
    that declares so may be used to justify the proof.
    """

    manager = _token_manager()
    captured: list[ModelConfig] = []

    def factory(_claims, config):
        captured.append(config)

        class Adapter:
            if upstream_applies:
                generation_support = "native"

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


def test_generation_support_probe_is_opt_in_and_fail_closed() -> None:
    assert generation_support(OpenAIModelAdapter(ModelConfig())) == "native"
    assert generation_support(GatewayModelAdapter(config=ModelConfig())) == "native"
    assert generation_support(FakeModelAdapter()) == "none"

    class Vague:
        generation_support = True

    assert generation_support(Vague()) == "none"

    class Hostile:
        @property
        def generation_support(self) -> str:
            raise RuntimeError("boom")

    # A declaration that raises is not a declaration; it must not take the call down either.
    assert generation_support(Hostile()) == "none"


def test_a_forwarding_adapter_claims_only_while_it_is_enforcing() -> None:
    """The gateway adapter forwards, it does not apply -- so its claim is worth exactly the
    proof it insists on. Under "omit" it deliberately accepts an unproven turn, and a static
    claim would let the *next* hop mint a fresh positive echo out of it."""

    proving = GatewayModelAdapter(config=ModelConfig())
    assert generation_support(proving) == "native"
    assert structured_output_support(proving) == "native"

    best_effort = GatewayModelAdapter(
        config=ModelConfig(generation=GenerationConfig(on_unsupported="omit"))
    )
    assert generation_support(best_effort) == "none"
    assert structured_output_support(best_effort) == "none"

    # OpenAI applies the parameters itself, so its claim is unconditional.
    assert (
        generation_support(
            OpenAIModelAdapter(
                ModelConfig(generation=GenerationConfig(on_unsupported="omit"))
            )
        )
        == "native"
    )


def test_a_chained_gateway_does_not_mint_proof_the_inner_hop_never_had() -> None:
    request = LlmGatewayTurnRequest(
        protocol="monoid.llm-turn.v1",
        model="gpt-5.5",
        system_prompt="sys",
        tools=(),
        reasoning=ReasoningConfig(),
        generation=GenerationConfig(temperature=0.2, on_unsupported="omit"),
        output_schema={"type": "object"},
    )
    upstream = GatewayModelAdapter(
        config=ModelConfig(generation=GenerationConfig(temperature=0.2, on_unsupported="omit"))
    )
    echoes = _applied_echoes(request, upstream, _upstream_model_config(request))
    assert "generation_applied" not in echoes
    assert echoes["schema_applied"] is False

    proving_request = replace(
        request,
        generation=GenerationConfig(temperature=0.2),
        output_schema={"type": "object"},
    )
    proving_upstream = GatewayModelAdapter(
        config=ModelConfig(generation=GenerationConfig(temperature=0.2))
    )
    proven = _applied_echoes(
        proving_request, proving_upstream, _upstream_model_config(proving_request)
    )
    assert proven["generation_applied"] == {"temperature": 0.2}
    assert proven["schema_applied"] is True


def test_gateway_service_never_asserts_application_from_the_request() -> None:
    """An upstream that ignores ``ModelConfig.generation`` -- the offline echo adapter, or any
    ``provider_adapter_factory`` backend -- must not produce a proof. Echoing the requested
    block back would match exactly on the client and let ``on_unsupported="fail"`` accept
    sampling parameters no model ever saw."""

    backend, manager, _ = _recording_backend(upstream_applies=False)
    payload = _turn_payload(generation=dict(_SET_WIRE))

    assert "generation_applied" not in backend.handle_turn(_llm_token(manager), payload)
    frames = list(backend.handle_turn_stream(_llm_token(manager), payload))
    assert "generation_applied" not in frames[-1]

    # ...and the client refuses that turn under the default policy, on both transports.
    with pytest.raises(ModelAdapterError) as rejected:
        _check_generation_applied(_SET_WIRE, "fail", None)
    assert rejected.value.provider_error_code == GATEWAY_GENERATION_NOT_APPLIED


@pytest.mark.parametrize("policy", ("fail", "omit"))
@pytest.mark.parametrize("requested", ({}, dict(_SET_WIRE)))
def test_a_malformed_echo_is_a_bad_response_on_both_transports(
    policy: str, requested: dict
) -> None:
    """Wire shape is not a policy question. The streamed frame parser always rejected a
    non-object echo; the sync check used to accept one under "omit" (and misreport it as
    "not applied" under "fail"), so the two transports disagreed about the same bytes."""

    with pytest.raises(ModelAdapterError) as sync_side:
        _check_generation_applied(requested, policy, "not-an-object")
    assert sync_side.value.provider_error_code == GATEWAY_BAD_RESPONSE

    with pytest.raises(ModelAdapterError) as stream_side:
        _chunk_from_event(
            {
                "type": "turn_complete",
                "turn_handle": "turn_1",
                "generation_applied": "not-an-object",
            }
        )
    assert stream_side.value.provider_error_code == GATEWAY_BAD_RESPONSE


def test_not_applied_error_carries_the_clients_own_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The receipt's only carrier for a retried-then-rejected call is the error itself: no
    turn is returned. The streaming twin stamps the chunk before checking it, so the sync path
    must stamp the turn before checking it too."""

    import monoid_agent_kernel.providers.gateway as gateway_module

    attempts = {"n": 0}

    def _urlopen(*_a: object, **_k: object) -> _FakeHttpResponse:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise URLError("connection reset")
        return _FakeHttpResponse(_served_turn())

    monkeypatch.setattr(gateway_module, "urlopen", _urlopen)
    monkeypatch.setattr(gateway_module, "_sleep_before_retry", lambda *_a, **_k: None)
    config = ModelConfig(generation=_SET, gateway_url="http://gateway.test")

    with pytest.raises(ModelAdapterError) as rejected:
        GatewayModelAdapter(config=config).next_turn(_request(config))
    assert attempts["n"] == 2
    assert rejected.value.provider_error_code == GATEWAY_GENERATION_NOT_APPLIED
    assert rejected.value.provider_retried is True


def test_not_applied_errors_are_config_recoverable_on_both_checks() -> None:
    """The error's own message instructs the user to set on_unsupported="omit" and resend —
    config the user can fix, which is exactly the class the loop's classifier keeps the
    session alive for. Without the flag, a mid-run failover to a non-echoing gateway replica
    terminalized the whole run while the same condition reported by a server as HTTP 400
    ended only the turn."""

    from monoid_agent_kernel.providers.gateway import _check_schema_applied
    from monoid_agent_kernel.reference.llm_gateway.http import _model_error_status

    with pytest.raises(ModelAdapterError) as generation:
        _check_generation_applied(_SET_WIRE, "fail", None)
    assert generation.value.config_recoverable is True
    assert generation.value.retryable is False

    with pytest.raises(ModelAdapterError) as schema:
        _check_schema_applied(True, "fail", None)
    assert schema.value.config_recoverable is True

    # Across a reference-gateway hop the property must not be laundered into a 502: the HTTP
    # mapping turns it into a 4xx so the outer client's classifier reads the same answer.
    assert 400 <= int(_model_error_status(generation.value)) < 500


# --- streaming enforcement without a terminal frame -------------------------------------


def _sse_adapter(
    monkeypatch: pytest.MonkeyPatch, config: ModelConfig, lines: list[str]
) -> GatewayModelAdapter:
    """A gateway adapter whose streamed body is exactly ``lines``.

    One fake server for every streaming case here, so the frameless shape and the
    terminal-frame shapes cannot drift into two differently-behaving doubles.
    """

    httpx = pytest.importorskip("httpx")

    class _Response:
        status_code = 200

        async def __aenter__(self) -> object:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def aiter_lines(self):
            for line in lines:
                yield line

    class _Client:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def __aenter__(self) -> object:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def stream(self, *_args: object, **_kwargs: object) -> object:
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return GatewayModelAdapter(config=config)


def _terminal_frameless_adapter(
    monkeypatch: pytest.MonkeyPatch, config: ModelConfig
) -> GatewayModelAdapter:
    """A gateway whose SSE body ends cleanly after one delta — no ``turn_complete`` frame.

    This is the older/foreign-server shape ``assemble_streamed_turn`` tolerates by
    synthesizing ``stop_reason="stop"``, so nothing downstream of the adapter can tell the
    frame was missing. If the applied-parameter checks run only on the frame, this stream is
    accepted with the parameters unproven while the sync twin refuses the same server.
    """

    return _sse_adapter(
        monkeypatch, config, ['data: {"type":"text_delta","text":"unproven answer"}', ""]
    )


def _drain(adapter: GatewayModelAdapter, request: ModelRequest) -> list:
    import asyncio

    async def _collect() -> list:
        return [chunk async for chunk in adapter.astream_turn(request)]

    return asyncio.run(_collect())


def test_a_stream_without_a_terminal_frame_is_an_unproven_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both transports enforce or neither does — including when the frame the streaming
    check lives on never arrives. Absent frame = absent echo, the same older-gateway case
    the shared checks already refuse under "fail"."""

    config = ModelConfig(generation=_SET, gateway_url="http://gateway.test")
    adapter = _terminal_frameless_adapter(monkeypatch, config)
    with pytest.raises(ModelAdapterError) as rejected:
        _drain(adapter, _request(config))
    assert rejected.value.provider_error_code == GATEWAY_GENERATION_NOT_APPLIED


def test_a_stream_without_a_terminal_frame_refuses_an_unproven_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from monoid_agent_kernel.providers.gateway import GATEWAY_SCHEMA_NOT_APPLIED

    config = ModelConfig(gateway_url="http://gateway.test")
    adapter = _terminal_frameless_adapter(monkeypatch, config)
    request = replace(_request(config), output_schema={"type": "object"})
    with pytest.raises(ModelAdapterError) as rejected:
        _drain(adapter, request)
    assert rejected.value.provider_error_code == GATEWAY_SCHEMA_NOT_APPLIED


def test_a_stream_without_a_terminal_frame_is_accepted_under_omit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ModelConfig(
        generation=GenerationConfig(temperature=0.2, on_unsupported="omit"),
        gateway_url="http://gateway.test",
    )
    adapter = _terminal_frameless_adapter(monkeypatch, config)
    chunks = _drain(adapter, _request(config))
    assert any(getattr(chunk, "text", "") == "unproven answer" for chunk in chunks)


def test_a_stream_without_a_terminal_frame_still_streams_plain_traffic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No generation, no schema: the transport owes no proof, so the pre-W5 tolerance for a
    frameless stream (older gateway) must survive the enforcement fix."""

    config = ModelConfig(gateway_url="http://gateway.test")
    adapter = _terminal_frameless_adapter(monkeypatch, config)
    chunks = _drain(adapter, _request(config))
    assert any(getattr(chunk, "text", "") == "unproven answer" for chunk in chunks)


# --- the proof question is per call, not per adapter ------------------------------------


def test_forwarding_adapter_claim_follows_the_effective_config() -> None:
    """The claim and the enforcement must read the same policy. The adapter enforces under
    ``request.model or self.config``; a claim probed off the standing config alone lets a
    shared adapter (a ``provider_adapter_factory`` that ignores its config parameter) mint
    proof for a call it will not enforce — or withhold proof from a call it will."""

    standing_fail = GatewayModelAdapter(config=ModelConfig())
    per_call_omit = ModelConfig(generation=GenerationConfig(temperature=0.2, on_unsupported="omit"))
    assert generation_support(standing_fail, per_call_omit) == "none"
    assert structured_output_support(standing_fail, per_call_omit) == "none"

    standing_omit = GatewayModelAdapter(
        config=ModelConfig(generation=GenerationConfig(on_unsupported="omit"))
    )
    per_call_fail = ModelConfig(generation=GenerationConfig(temperature=0.2))
    assert generation_support(standing_omit, per_call_fail) == "native"
    assert structured_output_support(standing_omit, per_call_fail) == "native"

    # No per-call config: the standing config is the effective config (a client-side probe).
    assert generation_support(standing_fail) == "native"
    assert generation_support(standing_omit) == "none"

    # A declaration that raises when *called* is not a claim either.
    class HostileCallable:
        def generation_support(self, _config: object = None) -> str:
            raise RuntimeError("boom")

    assert generation_support(HostileCallable(), per_call_fail) == "none"


def test_gateway_service_probes_the_config_the_call_runs_under() -> None:
    """A shared/preconfigured chained adapter must not answer the capability question from
    its standing config: the call enforces under the wire request's policy. Standing "fail" +
    wire "omit" minted proof nothing enforced; standing "omit" + wire "fail" withheld proof
    the inner hop actually insisted on."""

    manager = _token_manager()

    def _shared(standing: ModelConfig) -> GatewayModelAdapter:
        class _StubbedChainedGateway(GatewayModelAdapter):
            def next_turn(self, request: ModelRequest) -> ModelTurn:
                return ModelTurn(
                    response_id="inner_1",
                    final_text="ok",
                    usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                    stop_reason="stop",
                )

        return _StubbedChainedGateway(config=standing)

    proving_shared = _shared(ModelConfig())  # standing "fail"
    backend = LlmGatewayBackend(
        token_manager=manager, provider_adapter_factory=lambda _claims, _config: proving_shared
    )
    wire_omit = _turn_payload(
        generation={"temperature": 0.2, "on_unsupported": "omit"},
        output_schema={"type": "object"},
    )
    result = backend.handle_turn(_llm_token(manager), wire_omit)
    assert "generation_applied" not in result
    assert result["schema_applied"] is False

    best_effort_shared = _shared(
        ModelConfig(generation=GenerationConfig(on_unsupported="omit"))
    )
    backend = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, _config: best_effort_shared,
    )
    wire_fail = _turn_payload(
        generation={"temperature": 0.2}, output_schema={"type": "object"}
    )
    result = backend.handle_turn(_llm_token(manager), wire_fail)
    assert result["generation_applied"] == {"temperature": 0.2}
    assert result["schema_applied"] is True


# --- a proof is not Python equality -----------------------------------------------------


_BOOLEAN_SPOOFS = [
    (GenerationConfig(max_output_tokens=1), {"max_output_tokens": True}),
    (GenerationConfig(top_p=1), {"top_p": True}),
    (GenerationConfig(temperature=0), {"temperature": False}),
    (GenerationConfig(temperature=0.0), {"temperature": False}),
]


@pytest.mark.parametrize("generation,echo", _BOOLEAN_SPOOFS)
def test_a_boolean_echo_never_proves_a_numeric_parameter(
    generation: GenerationConfig, echo: dict
) -> None:
    """``True == 1`` and ``False == 0`` in Python, so comparing the echo dict with ``==``
    let a gateway answering JSON booleans prove the most ordinary settings on this wire --
    ``temperature=0``, ``top_p=1``, ``max_output_tokens=1``. Every other read of this wire
    already refuses that coercion (``_exact_gateway_bool`` / ``_exact_gateway_int``); the
    proof comparison was the one place it slipped through. A number is proven by a number."""

    requested = build_generation_payload(generation)
    with pytest.raises(ModelAdapterError) as rejected:
        _check_generation_applied(requested, "fail", echo)
    assert rejected.value.provider_error_code == GATEWAY_GENERATION_NOT_APPLIED
    assert rejected.value.config_recoverable is True
    # "omit" still accepts best-effort transport — the policy half is untouched.
    _check_generation_applied(requested, "omit", echo)


def test_an_equal_number_of_the_other_json_type_still_proves() -> None:
    """The defect is boolean coercion, not int-vs-float: a gateway that is not Python
    re-serializes ``1.0`` as ``1`` (JSON has one number type), and refusing that would be a
    new false refusal invented by the fix."""

    _check_generation_applied({"top_p": 1.0}, "fail", {"top_p": 1})
    _check_generation_applied({"max_output_tokens": 256}, "fail", {"max_output_tokens": 256.0})


def test_a_boolean_echo_is_refused_on_the_sync_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import monoid_agent_kernel.providers.gateway as gateway_module

    monkeypatch.setattr(
        gateway_module,
        "urlopen",
        lambda *_a, **_k: _FakeHttpResponse(
            _served_turn({"generation_applied": {"max_output_tokens": True}})
        ),
    )
    config = ModelConfig(
        generation=GenerationConfig(max_output_tokens=1), gateway_url="http://gateway.test"
    )
    with pytest.raises(ModelAdapterError) as rejected:
        GatewayModelAdapter(config=config).next_turn(_request(config))
    assert rejected.value.provider_error_code == GATEWAY_GENERATION_NOT_APPLIED


def test_a_boolean_echo_is_refused_on_the_streamed_terminal_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The twin transport: the same spoof riding ``turn_complete``. Either both refuse it or
    the client's answer depends on which transport it happened to use."""

    config = ModelConfig(
        generation=GenerationConfig(max_output_tokens=1), gateway_url="http://gateway.test"
    )
    adapter = _sse_adapter(
        monkeypatch,
        config,
        [
            'data: {"type":"text_delta","text":"unproven answer"}',
            "",
            'data: {"type":"turn_complete","turn_handle":"t1",'
            '"generation_applied":{"max_output_tokens":true}}',
            "",
        ],
    )
    with pytest.raises(ModelAdapterError) as rejected:
        _drain(adapter, _request(config))
    assert rejected.value.provider_error_code == GATEWAY_GENERATION_NOT_APPLIED


# --- a refused turn was still paid for ----------------------------------------------------


_BILLED = {"input_tokens": 120, "output_tokens": 340, "total_tokens": 460}


def _receipt_for(adapter: GatewayModelAdapter, request: ModelRequest, *, stream: bool = False):
    """Drive one call through ``ModelCallRunner`` and return the receipt it published.

    ``stream=True`` passes a consumer, which is what selects the streamed dispatch shape.
    """

    import asyncio

    from monoid_agent_kernel.core.model_io import ModelIOSubscription
    from monoid_agent_kernel.model_call import ModelCallRunner

    class _Observer:
        def __init__(self) -> None:
            self.receipts: list = []

        def on_model_call(self, capture) -> None:  # noqa: ANN001
            self.receipts.append(capture.receipt)

    observer = _Observer()
    runner = ModelCallRunner(
        adapter=adapter, subscriptions=(ModelIOSubscription(observer=observer),)
    )
    with pytest.raises(ModelAdapterError) as refused:
        asyncio.run(
            runner.acall(
                request, delta_consumer=(lambda _chunk: None) if stream else None
            )
        )
    assert refused.value.provider_error_code == GATEWAY_GENERATION_NOT_APPLIED
    assert len(observer.receipts) == 1
    return observer.receipts[0]


def test_a_refused_turn_still_reports_the_tokens_it_burned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The proof refusal happens *after* a complete, billed answer: the gateway generated the
    text and reported its usage, and only then does the client refuse to trust that its
    parameters shaped it. The failed receipt reported zero tokens, so the call vanished from
    the metrics and from the cumulative token budget — a budget that under-counts is a bound
    that does not hold."""

    import monoid_agent_kernel.providers.gateway as gateway_module

    monkeypatch.setattr(
        gateway_module,
        "urlopen",
        lambda *_a, **_k: _FakeHttpResponse(_served_turn({"usage": dict(_BILLED)})),
    )
    config = ModelConfig(generation=_SET, gateway_url="http://gateway.test")
    receipt = _receipt_for(GatewayModelAdapter(config=config), _request(config))

    assert dict(receipt.usage) == _BILLED
    assert receipt.provider_error_code == GATEWAY_GENERATION_NOT_APPLIED


def test_the_streamed_refusal_reports_its_tokens_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """The twin transport. Here the terminal frame is refused *before* it is yielded, so its
    usage reaches nothing that assembles a turn — the refusal is the only carrier left."""

    config = ModelConfig(generation=_SET, gateway_url="http://gateway.test")
    adapter = _sse_adapter(
        monkeypatch,
        config,
        [
            'data: {"type":"text_delta","text":"an answer that was generated and billed"}',
            "",
            'data: {"type":"turn_complete","turn_handle":"t1","usage":'
            '{"input_tokens":120,"output_tokens":340,"total_tokens":460}}',
            "",
        ],
    )
    with pytest.raises(ModelAdapterError) as refused:
        _drain(adapter, _request(config))
    assert refused.value.provider_error_code == GATEWAY_GENERATION_NOT_APPLIED
    assert getattr(refused.value, "provider_usage", None) == _BILLED

    # ...and it reaches the receipt, which is where the accounting actually reads it. Asserting
    # only on the exception would pin the stamp without pinning that anything consumes it.
    receipt = _receipt_for(adapter, _request(config), stream=True)
    assert dict(receipt.usage) == _BILLED


def test_a_refusal_with_no_usage_reported_stays_empty() -> None:
    """The stamp carries what the provider said, and invents nothing when it said nothing."""

    from monoid_agent_kernel.providers.base import mark_provider_usage

    error = ModelAdapterError("refused")
    mark_provider_usage(error, {})
    assert getattr(error, "provider_usage", None) is None
    mark_provider_usage(error, None)
    assert getattr(error, "provider_usage", None) is None


def test_billed_usage_survives_the_gateway_hop(monkeypatch: pytest.MonkeyPatch) -> None:
    """The chained-hop twin of the direct-call fix.

    When a reference gateway's upstream is another ``GatewayModelAdapter`` and the inner hop
    returns a billed turn without the required echo, the inner adapter stamps the usage on the
    refusal — and the hop then dropped it: the error envelope carried only message, code,
    retryability, status and retry evidence, so the outer client rebuilt an exception with no
    usage and reported zero tokens for a call the provider billed. The gateway's own tenant
    meter missed it too, because ``handle_turn`` raises before ``_usage.add``."""

    import monoid_agent_kernel.providers.gateway as gateway_module
    from monoid_agent_kernel.reference.llm_gateway.http import _error_body, _model_error_status
    from monoid_agent_kernel.providers.gateway import _error_from_status_body

    monkeypatch.setattr(
        gateway_module,
        "urlopen",
        lambda *_a, **_k: _FakeHttpResponse(_served_turn({"usage": dict(_BILLED)})),
    )
    manager = _token_manager()
    inner = ModelConfig(generation=_SET, gateway_url="http://inner.test")

    def factory(_claims, _config):
        return GatewayModelAdapter(config=inner)

    backend = LlmGatewayBackend(token_manager=manager, provider_adapter_factory=factory)
    token = _llm_token(manager)

    with pytest.raises(ModelAdapterError) as refused:
        backend.handle_turn(token, _turn_payload(generation=dict(_SET_WIRE)))
    assert refused.value.provider_error_code == GATEWAY_GENERATION_NOT_APPLIED
    assert getattr(refused.value, "provider_usage", None) == _BILLED

    # 1. The hop's error envelope carries what the call cost.
    body = _error_body(
        _model_error_status(refused.value),
        str(refused.value),
        error_code=refused.value.provider_error_code,
        retryable=refused.value.retryable,
        provider_retried=refused.value.provider_retried,
        usage=getattr(refused.value, "provider_usage", None),
    )
    assert body["usage"] == _BILLED
    assert body["http_status"] == 422

    # 2. The outer client reads it back onto the exception it reconstructs.
    rebuilt = _error_from_status_body(422, json.dumps(body))
    assert getattr(rebuilt, "provider_usage", None) == _BILLED

    # 3. And the gateway metered the call locally rather than losing it to the raise.
    assert backend.tenant_usage("tenant_a")["total_tokens"] == 460


def test_a_billed_failure_is_metered_on_the_stream_transport_too() -> None:
    """The sync twin (`handle_turn`) meters a billed refusal before re-raising; `_stream_turn`
    exited on the raise before its success-path meter, so the same refused call left the
    tenant ledger at zero on this transport — on both its sub-branches (the astream drive
    and the non-streaming fallback)."""

    from monoid_agent_kernel.providers.base import mark_provider_usage

    _BILLED_LOCAL = {"input_tokens": 120, "output_tokens": 340, "total_tokens": 460}

    class _BilledRefusal:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            error = ModelAdapterError(
                "upstream refused an unproven billed turn",
                provider_error_code=GATEWAY_GENERATION_NOT_APPLIED,
                retryable=False,
                config_recoverable=True,
            )
            mark_provider_usage(error, dict(_BILLED_LOCAL))
            raise error

    class _BilledStreamRefusal(_BilledRefusal):
        async def astream_turn(self, request: ModelRequest):
            self.next_turn(request)
            yield  # pragma: no cover — next_turn always raises

    for upstream in (_BilledRefusal(), _BilledStreamRefusal()):
        manager = _token_manager()
        backend = LlmGatewayBackend(
            token_manager=manager,
            provider_adapter_factory=lambda _claims, _config, _u=upstream: _u,
        )
        with pytest.raises(ModelAdapterError):
            list(
                backend.handle_turn_stream(
                    _llm_token(manager), _turn_payload(generation=dict(_SET_WIRE))
                )
            )
        assert backend.tenant_usage("tenant_a")["total_tokens"] == 460, type(upstream).__name__


def test_a_malformed_echo_on_a_billed_frame_keeps_the_cost() -> None:
    """The sync transport validates the echo inside the stamped check block, so a malformed
    echo on a billed 200 response raises `gateway_bad_response` WITH `provider_usage`; the
    stream-frame parser raised at parse time, before any stamp could run — the same money,
    gone on one of two transports."""

    from monoid_agent_kernel.providers.gateway import _chunk_from_event

    with pytest.raises(ModelAdapterError) as bad:
        _chunk_from_event(
            {
                "type": "turn_complete",
                "turn_handle": "t1",
                "usage": dict(_BILLED),
                "generation_applied": [1, 2],
            }
        )
    assert bad.value.provider_error_code == GATEWAY_BAD_RESPONSE
    assert getattr(bad.value, "provider_usage", None) == _BILLED


def test_a_gateway_error_that_cost_nothing_keeps_its_wire_shape() -> None:
    """The counterweight to the hop fix: an error the gateway raised on its own reached no
    provider, so its envelope must be byte-identical to what it was before ``usage`` existed
    — and the client must not invent tokens for it."""

    from monoid_agent_kernel.reference.llm_gateway.http import _error_body
    from monoid_agent_kernel.providers.gateway import _error_from_status_body

    body = _error_body(
        HTTPStatus.BAD_REQUEST, "bad request", error_code=GATEWAY_BAD_RESPONSE, retryable=False
    )
    assert "usage" not in body
    assert getattr(_error_from_status_body(400, json.dumps(body)), "provider_usage", None) is None


def test_a_streamed_error_frame_carries_the_cost_too() -> None:
    """The SSE twin of the hop: same ``_error_body``, so the frame gains the field for free —
    but the *client's* stream-frame reader is a separate function, and a rule bound on one of
    two readers is the shape this branch keeps tripping over."""

    from monoid_agent_kernel.providers.gateway import _chunk_from_event

    with pytest.raises(ModelAdapterError) as raised:
        _chunk_from_event(
            {
                "type": "error",
                "error": "upstream refused an unproven turn",
                "error_code": GATEWAY_GENERATION_NOT_APPLIED,
                "retryable": False,
                "http_status": 422,
                "usage": dict(_BILLED),
            }
        )
    assert getattr(raised.value, "provider_usage", None) == _BILLED


def test_a_two_hundred_error_envelope_carries_the_cost_too() -> None:
    """The third reader on this wire: a 200 response whose body is an error object."""

    from monoid_agent_kernel.providers.gateway import _parse_gateway_response

    with pytest.raises(ModelAdapterError) as raised:
        _parse_gateway_response(
            {
                "error": "upstream refused an unproven turn",
                "error_code": GATEWAY_GENERATION_NOT_APPLIED,
                "http_status": 422,
                "usage": dict(_BILLED),
            }
        )
    assert getattr(raised.value, "provider_usage", None) == _BILLED


# --- and every OTHER refusal off a billed success envelope --------------------------------
#
# The rule the malformed-echo test above states -- "a malformed value on a *billed* frame still
# cost the tokens the same frame reports" -- was bound to the echo pair and to nothing else. A
# 200 body or a terminal frame whose ``stop_reason``/``final_text``/``reasoning``/... is
# malformed is refused by the same readers on the same billed payload, and every one of those
# refusals escaped with an empty ``provider_usage``: the run's budget lost the call.


def _billed_success_body(**overrides: object) -> dict:
    """A complete, well-formed 200 success body that reports what the turn cost."""

    body: dict = {
        "protocol": "monoid.llm-turn-result.v1",
        "turn_handle": "turn_1",
        "final_text": "answered",
        "tool_calls": [],
        "usage": dict(_BILLED),
        "stop_reason": "stop",
        "provider_retried": False,
    }
    body.update(overrides)
    return body


def _billed_terminal_frame(**overrides: object) -> dict:
    """The streamed twin of :func:`_billed_success_body`."""

    frame: dict = {
        "type": "turn_complete",
        "turn_handle": "turn_1",
        "usage": dict(_BILLED),
        "stop_reason": "stop",
        "provider_retried": False,
    }
    frame.update(overrides)
    return frame


# One malformed value per key each reader validates on its success envelope, keyed by the key
# it corrupts. The behavioral census in tests/test_carriage_conformance.py diffs the same table
# against the pinned read sets, so a NEW wire key cannot repeat this miss silently.
_SYNC_BODY_REFUSALS: dict[str, dict] = {
    "reasoning": {"reasoning": "not-an-array"},
    "stop_reason": {"stop_reason": 7},
    "final_text": {"final_text": 7},
    "response_id": {"response_id": 7},
    "turn_handle": {"turn_handle": 7},
    "tool_calls": {"tool_calls": {"not": "an array"}},
    "retryable": {"retryable": "yes"},
    "provider_retried": {"provider_retried": "yes"},
}
_TERMINAL_FRAME_REFUSALS: dict[str, dict] = {
    "reasoning": {"reasoning": "not-an-array"},
    "stop_reason": {"stop_reason": 7},
    "turn_handle": {"turn_handle": 7},
    "generation_applied": {"generation_applied": [1, 2]},
    "provider_retried": {"provider_retried": "yes"},
}


@pytest.mark.parametrize("key", sorted(_SYNC_BODY_REFUSALS))
def test_a_refused_success_body_still_reports_the_tokens_it_burned(key: str) -> None:
    from monoid_agent_kernel.providers.gateway import _parse_gateway_response

    with pytest.raises(ModelAdapterError) as refused:
        _parse_gateway_response(_billed_success_body(**_SYNC_BODY_REFUSALS[key]))
    assert refused.value.provider_error_code == GATEWAY_BAD_RESPONSE
    assert provider_usage_of(refused.value) == _BILLED


@pytest.mark.parametrize("key", sorted(_TERMINAL_FRAME_REFUSALS))
def test_a_refused_terminal_frame_still_reports_the_tokens_it_burned(key: str) -> None:
    with pytest.raises(ModelAdapterError) as refused:
        _chunk_from_event(_billed_terminal_frame(**_TERMINAL_FRAME_REFUSALS[key]))
    assert refused.value.provider_error_code == GATEWAY_BAD_RESPONSE
    assert provider_usage_of(refused.value) == _BILLED


@pytest.mark.parametrize(
    "usage, stamped",
    [
        ("not-a-mapping", {}),
        ({"input_tokens": 120, "output_tokens": "many", "total_tokens": 460}, {}),
    ],
    ids=["unreadable", "partly-readable"],
)
def test_a_refusal_off_a_malformed_usage_invents_no_tokens(usage: object, stamped: dict) -> None:
    """The counterweight, and the reason the stamp reads leniently rather than validating.

    ``usage`` is the *source* of the stamp, so when it is the malformed key there is nothing
    trustworthy to stamp: a second malformation must not replace the failure being reported,
    and a count that cannot be read must not be invented. Whatever counts survive the filter
    are the ones the payload actually stated.
    """

    from monoid_agent_kernel.providers.gateway import _parse_gateway_response

    expected = dict(stamped)
    if isinstance(usage, dict):
        expected = {
            name: value
            for name, value in usage.items()
            if type(value) is int and value >= 0
        }

    with pytest.raises(ModelAdapterError) as sync:
        _parse_gateway_response(_billed_success_body(usage=usage))
    assert sync.value.provider_error_code == GATEWAY_BAD_RESPONSE
    assert provider_usage_of(sync.value) == expected

    with pytest.raises(ModelAdapterError) as streamed:
        _chunk_from_event(_billed_terminal_frame(usage=usage))
    assert streamed.value.provider_error_code == GATEWAY_BAD_RESPONSE
    assert provider_usage_of(streamed.value) == expected


def test_a_refusal_off_an_envelope_that_cost_nothing_stays_costless() -> None:
    """The other counterweight: a malformed body that never reported a cost invents none."""

    from monoid_agent_kernel.providers.gateway import _parse_gateway_response

    body = _billed_success_body(stop_reason=7)
    body.pop("usage")
    with pytest.raises(ModelAdapterError) as refused:
        _parse_gateway_response(body)
    assert provider_usage_of(refused.value) == {}

    frame = _billed_terminal_frame(stop_reason=7)
    frame.pop("usage")
    with pytest.raises(ModelAdapterError) as framed:
        _chunk_from_event(frame)
    assert provider_usage_of(framed.value) == {}
