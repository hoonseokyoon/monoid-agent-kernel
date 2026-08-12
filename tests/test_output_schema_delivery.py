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
from monoid_agent_kernel.model_call import (
    _REQUEST_DIGEST_GENERATION,
    ModelCallRunner,
    _digest,
    _request_payload,
)
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
    # Read through the generation wrapper deliberately: asserting `"output_schema" not in free`
    # against the whole payload passes vacuously once the terms sit one level down, which is a
    # pin that has stopped checking what it names.
    free = _request_payload(_request(), ModelConfig(), provider="fake")
    assert "output_schema" not in free[_REQUEST_DIGEST_GENERATION]

    constrained = _request_payload(
        _request(output_schema=dict(_SCHEMA)), ModelConfig(), provider="fake"
    )
    assert constrained[_REQUEST_DIGEST_GENERATION]["output_schema"] == _SCHEMA
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


def test_the_server_never_rewrites_the_schema_it_forwards() -> None:
    """'Never rewrites' includes the server's own ingress. The blanket request normalize
    substitutes non-finite content values, and `output_schema` riding it was silently
    rewritten (`NaN` → `None`) — enforcing a *different* constraint than the caller wrote,
    the exact rewrite the client-side rule (normalize_model_request,
    substitute_nonfinite=False) exists to refuse. Live path: in-process Python callers —
    HTTP bodies already reject non-finite constants at the JSON parser."""

    import math

    upstream = _NativeUpstream()
    backend, manager = _backend(upstream)
    hostile = {"type": "number", "maximum": float("nan")}

    backend.handle_turn(_llm_token(manager), _turn_payload(output_schema=dict(hostile)))
    forwarded = upstream.requests[0].output_schema
    assert forwarded is not None and math.isnan(forwarded["maximum"])

    list(
        backend.handle_turn_stream(
            _llm_token(manager), _turn_payload(output_schema=dict(hostile))
        )
    )
    forwarded_stream = upstream.requests[1].output_schema
    assert forwarded_stream is not None and math.isnan(forwarded_stream["maximum"])


# --- the standalone executor: parsed + fallback -----------------------------------------


class _SeesParsed:
    id = "sees-parsed"
    schema = None

    def __init__(self) -> None:
        self.parsed_seen: list[object] = []
        self.parsed_ok_seen: list[bool] = []

    def validate(self, view: FinalOutputView) -> ValidationOutcome:
        self.parsed_seen.append(view.parsed)
        self.parsed_ok_seen.append(view.parsed_ok)
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
    assert validator.parsed_ok_seen == [False]


def test_a_valid_root_null_is_distinguishable_from_a_failed_parse() -> None:
    """``parsed is None`` cannot answer "was there a parse?": a schema permitting a root
    ``null`` yields a valid parsed value of ``None``, identical to "not JSON" and to "no
    schema". A validator rejecting on ``parsed is None`` would fail a conforming answer and
    spend its repair budget on it, so ``parsed_ok`` is the authority."""

    validator = _SeesParsed()
    runner = ValidatedCallRunner(
        runner=ModelCallRunner(
            adapter=FakeModelAdapter(turns=[ModelTurn(final_text="null", stop_reason="stop")])
        ),
        validators=(validator,),
    )
    result = asyncio.run(runner.acall(_request(output_schema={"type": ["object", "null"]})))

    assert result.status == "ok"
    assert validator.parsed_seen == [None]
    assert validator.parsed_ok_seen == [True]


def test_parsed_ok_defaults_false_on_a_bare_view() -> None:
    """The loop builds its view without a parse, so the flag must default to "no parse" --
    a validator shared across both surfaces reads the same answer there."""

    assert FinalOutputView(final_text="hi").parsed_ok is False
    assert FinalOutputView(final_text="hi").parsed is None


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


def test_a_schema_rejection_400_names_the_offending_param() -> None:
    """OpenAI strict mode rejects ordinary JSON Schemas (missing additionalProperties:false,
    unlisted required keys) with a 400 whose body names the offending ``param``. The
    synthetic body-free message deliberately drops the provider's text (PII policy), but the
    ``param`` field is a provider-authored field path, not user content — without it the
    caller sees only "provider rejected the request" and cannot tell the schema was the
    problem."""

    from monoid_agent_kernel.providers.openai import _model_error_from_openai

    class _Fake(Exception):
        status_code = 400
        body = {"type": "invalid_request_error", "param": "text.format.schema"}

    error = _model_error_from_openai(_Fake("boom"))
    assert error.http_status == 400
    assert error.provider_error_code == "invalid_request_error"
    assert "text.format.schema" in str(error)


# --- an unserializable request is a classified error, not a raw TypeError ---------------


def test_an_unserializable_request_is_a_classified_bad_request() -> None:
    """``normalize_json_ingress`` deliberately passes arbitrary scalars through (the
    documented arbitrary-scalar gap), so the serialization boundary is where the failure
    lands -- and ``json.dumps`` sat outside the adapter's classifier, escaping as a raw
    ``TypeError`` the loop cannot classify at all. One encoder, both transports, covering
    ``output_schema``, ``messages``, and observations uniformly."""

    from monoid_agent_kernel.providers.gateway import GATEWAY_BAD_REQUEST

    config = ModelConfig(gateway_url="http://gateway.test")
    adapter = GatewayModelAdapter(config=config)

    with pytest.raises(ModelAdapterError) as schema_case:
        adapter.next_turn(_request(config, output_schema={"a": {1, 2}}))  # type: ignore[dict-item]
    assert schema_case.value.provider_error_code == GATEWAY_BAD_REQUEST
    assert schema_case.value.retryable is False
    # The classifier's own rationale is "a config-shaped mistake" — the same mistake reported
    # by a gateway *server* is an HTTP 400, which the loop treats as turn-recoverable, so the
    # client-side detection must carry the same classification.
    assert schema_case.value.config_recoverable is True

    with pytest.raises(ModelAdapterError) as messages_case:
        adapter.next_turn(
            ModelRequest(
                instruction=None,
                system_prompt="sys",
                tools=(),
                model=config,
                messages=({"role": "user", "content": print},),  # type: ignore[dict-item]
            )
        )
    assert messages_case.value.provider_error_code == GATEWAY_BAD_REQUEST

    # NaN rides a different exception (ValueError, from allow_nan=False) -- same rule.
    with pytest.raises(ModelAdapterError) as nan_case:
        adapter.next_turn(_request(config, output_schema={"a": float("nan")}))
    assert nan_case.value.provider_error_code == GATEWAY_BAD_REQUEST


def test_an_unserializable_request_is_classified_on_the_stream_too() -> None:
    pytest.importorskip("httpx")
    from monoid_agent_kernel.providers.gateway import GATEWAY_BAD_REQUEST

    config = ModelConfig(gateway_url="http://gateway.test")
    adapter = GatewayModelAdapter(config=config)

    async def _drive() -> None:
        async for _chunk in adapter.astream_turn(
            _request(config, output_schema={"a": {1, 2}})  # type: ignore[dict-item]
        ):
            pass

    with pytest.raises(ModelAdapterError) as rejected:
        asyncio.run(_drive())
    assert rejected.value.provider_error_code == GATEWAY_BAD_REQUEST


def test_openai_payload_build_failures_are_classified_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OpenAI twin: ``_payload`` ran outside the classifier ``try`` whose own comment
    says unclassified exceptions terminalize the run -- an unserializable tool result
    escaped as a raw ``TypeError`` before the classifier could see anything.

    Carried by the by-value ``messages`` log, which is where a tool result travels on this
    adapter: the by-reference shape it used to ride is refused outright under ZDR."""

    pytest.importorskip("openai")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = ModelConfig(provider="openai")
    adapter = OpenAIModelAdapter(config, allow_direct_provider_api=True)
    request = ModelRequest(
        instruction=None,
        system_prompt="sys",
        tools=(),
        model=config,
        messages=(
            {"role": "user", "content": "go"},
            {"role": "tool", "call_id": "c1", "content": {"bytes": {1, 2}}},
        ),
    )
    with pytest.raises(ModelAdapterError) as rejected:
        adapter.next_turn(request)
    # The twin names the same defect the gateway path names: a bad request, recoverable by
    # fixing the value — not an anonymous "provider call failed (TypeError)".
    assert rejected.value.provider_error_code == "unserializable_request"
    assert rejected.value.config_recoverable is True


# --- the schema is delivered verbatim, and refused at the serialization boundary ----------


def test_a_non_finite_schema_value_survives_ingress_for_the_serializer() -> None:
    """``output_schema`` is a control document promised verbatim, so ingress must not
    *rewrite* it. ``normalize_json_ingress`` substitutes non-finite floats with ``None`` --
    correct for model content, wrong for a schema: ``{"enum": [NaN]}`` became
    ``{"enum": [null]}``, a different constraint the provider silently enforced, and the
    strict serializer that exists to refuse the value never saw it."""

    import math

    config = ModelConfig(gateway_url="http://gateway.test")
    normalized = normalize_model_request(_request(config, output_schema={"enum": [float("nan")]}))
    assert math.isnan(normalized.output_schema["enum"][0])

    # Strings and containers are still normalized -- only the substitution is dropped.
    text = normalize_model_request(
        _request(config, output_schema={"title": "a\ud800b", "any_of": ({"type": "string"},)})
    )
    assert text.output_schema["title"] == "a\ufffdb"
    assert text.output_schema["any_of"] == [{"type": "string"}]

    # ... and the boundary that was stepped over now refuses the value, classified.
    from monoid_agent_kernel.providers.gateway import GATEWAY_BAD_REQUEST

    with pytest.raises(ModelAdapterError) as rejected:
        GatewayModelAdapter(config=config).next_turn(normalized)
    assert rejected.value.provider_error_code == GATEWAY_BAD_REQUEST
    assert rejected.value.config_recoverable is True


def test_openai_preflights_the_whole_payload_not_only_what_it_serializes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_payload`` embeds ``output_schema`` without serializing it, so the classifier saw
    nothing: a set inside the schema failed later inside ``client.responses.create`` and the
    outer handler named it an anonymous ``unclassified_provider_error`` with no
    ``config_recoverable`` -- terminalizing the run for what the gateway twin reports as a
    recoverable bad request. ``NaN`` was worse: it serialized to the JSON-invalid literal
    ``NaN`` and went out to the provider."""

    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = ModelConfig(provider="openai")
    adapter = OpenAIModelAdapter(config, allow_direct_provider_api=True)

    for schema in ({"a": {1, 2}}, {"a": float("nan")}, {"a": float("inf")}):
        with pytest.raises(ModelAdapterError) as rejected:
            adapter.next_turn(_request(config, output_schema=schema))  # type: ignore[arg-type]
        assert rejected.value.provider_error_code == "unserializable_request"
        assert rejected.value.config_recoverable is True
        assert rejected.value.retryable is False


def test_the_openai_stream_preflights_the_same_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The streamed twin of the preflight -- one helper, both call paths, or the rule is
    bound on one transport only."""

    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = ModelConfig(provider="openai")
    adapter = OpenAIModelAdapter(config, allow_direct_provider_api=True)

    async def _drive() -> None:
        async for _chunk in adapter.astream_turn(
            _request(config, output_schema={"a": {1, 2}})  # type: ignore[arg-type]
        ):
            pass

    with pytest.raises(ModelAdapterError) as rejected:
        asyncio.run(_drive())
    assert rejected.value.provider_error_code == "unserializable_request"
    assert rejected.value.config_recoverable is True


def test_a_well_formed_schema_still_reaches_the_provider_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preflight must refuse only what cannot be sent: an ordinary schema still builds."""

    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = ModelConfig(provider="openai")
    adapter = OpenAIModelAdapter(config, allow_direct_provider_api=True)
    payload = adapter._classified_payload(_request(config, output_schema=_SCHEMA))
    assert payload["text"]["format"]["schema"] == _SCHEMA


def _deeply_nested_schema() -> dict:
    """An acyclic schema deeper than the interpreter's recursion limit.

    Nothing upstream refuses it: ``normalize_json_ingress`` is deliberately iterative, and the
    512-level nesting cap belongs to the JSON *text* parsers, not to a Python-constructed
    value. ``json.dumps`` is recursive, so this is where it lands.
    """

    import sys

    node: dict = {"type": "object"}
    for _ in range(sys.getrecursionlimit() * 6):
        node = {"type": "object", "properties": {"x": node}}
    return node


def test_a_too_deep_request_is_classified_on_both_adapters() -> None:
    """``json.dumps`` answers a too-deep container with ``RecursionError`` — a ``RuntimeError``
    subclass, not the ``TypeError``/``ValueError`` family the classifiers caught — so it
    escaped raw from both encoders. ``AgentLoop._recoverable_turn_error`` only inspects a
    ``ModelAdapterError``, so a nesting mistake terminalized the whole run on both shipped
    adapters instead of failing the turn recoverably like every other unsendable request."""

    from monoid_agent_kernel.providers.gateway import GATEWAY_BAD_REQUEST

    schema = _deeply_nested_schema()
    config = ModelConfig(gateway_url="http://gateway.test")
    request = normalize_model_request(_request(config, output_schema=schema))

    with pytest.raises(ModelAdapterError) as gateway_sync:
        GatewayModelAdapter(config=config).next_turn(request)
    assert gateway_sync.value.provider_error_code == GATEWAY_BAD_REQUEST
    assert gateway_sync.value.config_recoverable is True

    pytest.importorskip("httpx")

    async def _drive() -> None:
        async for _chunk in GatewayModelAdapter(config=config).astream_turn(request):
            pass

    with pytest.raises(ModelAdapterError) as gateway_stream:
        asyncio.run(_drive())
    assert gateway_stream.value.provider_error_code == GATEWAY_BAD_REQUEST


def test_a_too_deep_request_is_classified_on_the_openai_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = ModelConfig(provider="openai")
    adapter = OpenAIModelAdapter(config, allow_direct_provider_api=True)

    with pytest.raises(ModelAdapterError) as rejected:
        adapter.next_turn(_request(config, output_schema=_deeply_nested_schema()))
    assert rejected.value.provider_error_code == "unserializable_request"
    assert rejected.value.config_recoverable is True
