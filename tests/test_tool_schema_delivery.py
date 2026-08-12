"""A tool's ``input_schema`` is a control document, not model content.

The twin of the W5 ``output_schema`` rule (``test_output_schema_delivery.py``), on the other
schema this request carries. Ingress normalizes strings and containers but never *substitutes*
a non-finite value: a schema is either delivered exactly as its author wrote it, or refused at
the strict serializer as the config-recoverable bad request it is. Substituting rewrote
``{"enum": [NaN]}`` into ``{"enum": [null]}`` -- a different constraint, which the registry
then validated calls against and the provider then enforced -- and the serializer that exists
to refuse such a value (``allow_nan=False``, on both adapters) never saw it.

Every carrier of the rule is pinned here: the one client-side normalizer
(``normalize_tool_spec``, which the registry, the loop and ``normalize_model_request`` all
share), the reference gateway's own server-side ingress, both providers' serialization
boundaries on both transports, and the replay key.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.model_call import _digest, _prompt_payload, _request_payload
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelTurn,
    normalize_model_request,
)
from monoid_agent_kernel.providers.gateway import GATEWAY_BAD_REQUEST, GatewayModelAdapter
from monoid_agent_kernel.providers.openai import OpenAIModelAdapter
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.llm_gateway.service import LlmGatewayBackend
from monoid_agent_kernel.tools.base import ToolRegistry, ToolSpec, normalize_tool_spec

_NAN_SCHEMA = {
    "type": "object",
    "properties": {"score": {"enum": [float("nan")]}},
}


def _handler(_context: object, _arguments: object) -> None:  # pragma: no cover - never called
    return None


def _spec(schema: dict) -> ToolSpec:
    return ToolSpec(
        id="score.read",
        provider_name="score_read",
        description="score a thing",
        input_schema=schema,
        capability="score",
        side_effect="read",
        handler=_handler,
    )


def _request(config: ModelConfig | None, schema: dict) -> ModelRequest:
    return ModelRequest(
        instruction="go",
        system_prompt="sys",
        tools=(_spec(schema),),
        model=config,
    )


# --- carrier 1: the one client-side normalizer -------------------------------------------


def test_a_non_finite_tool_schema_value_survives_ingress_for_the_serializer() -> None:
    """``normalize_tool_spec`` is the single client-side carrier -- the registry, the loop's
    per-turn spec copy and ``normalize_model_request`` all go through it."""

    normalized = normalize_tool_spec(_spec({"enum": [float("nan")]}))
    assert math.isnan(normalized.input_schema["enum"][0])

    # Strings and containers are still normalized -- only the substitution is dropped.
    text = normalize_tool_spec(
        _spec({"title": "a\ud800b", "any_of": ({"type": "string"},)})
    )
    assert text.input_schema["title"] == "a�b"
    assert text.input_schema["any_of"] == [{"type": "string"}]


def test_the_registry_validates_calls_against_the_authors_constraint() -> None:
    """The registry builds its ``Draft202012Validator`` from the *normalized* spec, so a
    rewritten schema is the one that judges every tool call for the rest of the run."""

    registry = ToolRegistry()
    registry.register(_spec({"enum": [float("nan")]}))
    assert math.isnan(registry.resolve("score.read").input_schema["enum"][0])

    forwarded = normalize_model_request(_request(None, {"enum": [float("nan")]}))
    assert math.isnan(forwarded.tools[0].input_schema["enum"][0])


def test_provider_built_tool_schemas_ride_the_same_normalizer() -> None:
    """The MCP provider hands the registry an ``inputSchema`` it copied off the server
    verbatim, and the ``@tool`` decorator derives one from type hints -- pydantic emits a
    ``NaN`` default for ``float = float('nan')``. Neither has a normalizer of its own: both
    reach the registry, so binding the rule once above binds it for them."""

    from monoid_agent_kernel.tools.decorator import tool

    mcp_shaped = ToolSpec(  # exactly the shape mcp/provider.py yields
        id="mcp.notes.search",
        description="search",
        input_schema=dict({"type": "object", "properties": {"k": {"maximum": float("inf")}}}),
        capability="mcp.notes",
        side_effect="read",
        handler=_handler,
        provider_name="mcp_notes_search",
    )
    registry = ToolRegistry()
    registry.register(mcp_shaped)
    registered = registry.resolve("mcp.notes.search")
    assert math.isinf(registered.input_schema["properties"]["k"]["maximum"])

    @tool(capability="demo")
    def demo(score: float = float("nan")) -> dict:
        """demo"""
        return {}

    assert math.isnan(normalize_tool_spec(demo).input_schema["properties"]["score"]["default"])


# --- the other side of the split: a record substitutes what it cannot carry ----------------


def test_the_transcript_record_substitutes_what_portable_json_cannot_carry() -> None:
    """A *record* is not a request. The run's transcript writer encodes with
    ``allow_nan=False``, so a preserved schema reaching it raw killed the run there with an
    anonymous ``internal_error`` — one boundary *before* the classified provider refusal, and
    for a durability reason rather than a config one. The transcript projection substitutes,
    exactly as ``RunManifest.to_json`` and the event writer already do; the request does not."""

    import json

    from monoid_agent_kernel.core.tool_surface import ToolSurfaceSnapshot

    snapshot = ToolSurfaceSnapshot(
        turn_id="turn_1",
        immediate_tools=(normalize_tool_spec(_spec(dict(_NAN_SCHEMA))),),
        searchable_tools=(),
        search_entries=(),
        hidden_tool_ids=(),
        authorizations={},
    )
    recorded = snapshot.to_transcript_json()

    assert recorded["immediate_tools"][0]["input_schema"]["properties"]["score"]["enum"] == [None]
    json.dumps(recorded, allow_nan=False)  # what the transcript writer does


# --- carrier 2: both providers, both transports -------------------------------------------


def test_a_non_finite_tool_schema_is_refused_at_the_gateway_boundary() -> None:
    config = ModelConfig(gateway_url="http://gateway.test")
    request = normalize_model_request(_request(config, dict(_NAN_SCHEMA)))

    with pytest.raises(ModelAdapterError) as rejected:
        GatewayModelAdapter(config=config).next_turn(request)
    assert rejected.value.provider_error_code == GATEWAY_BAD_REQUEST
    assert rejected.value.retryable is False
    assert rejected.value.config_recoverable is True

    pytest.importorskip("httpx")

    async def _drive() -> None:
        async for _chunk in GatewayModelAdapter(config=config).astream_turn(request):
            pass

    with pytest.raises(ModelAdapterError) as streamed:
        asyncio.run(_drive())
    assert streamed.value.provider_error_code == GATEWAY_BAD_REQUEST
    assert streamed.value.config_recoverable is True


def test_a_non_finite_tool_schema_is_refused_at_the_openai_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = ModelConfig(provider="openai")
    adapter = OpenAIModelAdapter(config, allow_direct_provider_api=True)
    request = normalize_model_request(_request(config, dict(_NAN_SCHEMA)))

    with pytest.raises(ModelAdapterError) as rejected:
        adapter.next_turn(request)
    assert rejected.value.provider_error_code == "unserializable_request"
    assert rejected.value.config_recoverable is True

    async def _drive() -> None:
        async for _chunk in adapter.astream_turn(request):
            pass

    with pytest.raises(ModelAdapterError) as streamed:
        asyncio.run(_drive())
    assert streamed.value.provider_error_code == "unserializable_request"
    assert streamed.value.config_recoverable is True


# --- carrier 3: the reference gateway's own ingress ---------------------------------------


class _CapturingUpstream:
    structured_output_support = "native"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        return ModelTurn(
            response_id="provider_1",
            final_text="ok",
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            stop_reason="stop",
        )


def _backend() -> tuple[LlmGatewayBackend, str, _CapturingUpstream]:
    upstream = _CapturingUpstream()
    manager = TokenManager.from_secret("y" * 32)
    backend = LlmGatewayBackend(
        token_manager=manager,
        provider_adapter_factory=lambda _claims, _config: upstream,
    )
    token = manager.issue(
        kind="llm_gateway",
        audience="csp.llm-gateway",
        run_id="run_1",
        tenant_id="tenant_a",
        user_id="user_a",
        ttl_s=600,
        metadata={"agent_config_hash": "test"},
    )
    return backend, token, upstream


def _turn_payload(tool: dict) -> dict:
    return {
        "protocol": "monoid.llm-turn.v1",
        "model": "gpt-5.5",
        "system_prompt": "sys",
        "instruction": "Answer.",
        "tools": [tool],
    }


@pytest.mark.parametrize("schema_key", ("input_schema", "parameters"))
def test_the_server_never_rewrites_the_tool_schemas_it_forwards(schema_key: str) -> None:
    """The server-side twin of the ingress rule, on both handlers and both wire spellings of
    the schema key. The blanket request normalize substitutes non-finite *content* values, and
    a tool schema riding it was rewritten before the upstream adapter ever saw it -- the exact
    rewrite the client-side rule exists to rule out, on the one route (in-process Python
    callers) the JSON parsers do not guard."""

    backend, token, upstream = _backend()
    tool = {
        "id": "score.read",
        "name": "score_read",
        "description": "score a thing",
        schema_key: {"type": "object", "properties": {"score": {"enum": [float("nan")]}}},
        "capability": "score",
        "side_effect": "read",
    }

    backend.handle_turn(token, _turn_payload(dict(tool)))
    forwarded = upstream.requests[0].tools[0].input_schema
    assert math.isnan(forwarded["properties"]["score"]["enum"][0])

    list(backend.handle_turn_stream(token, _turn_payload(dict(tool))))
    streamed = upstream.requests[1].tools[0].input_schema
    assert math.isnan(streamed["properties"]["score"]["enum"][0])


def test_the_server_still_normalizes_strings_and_containers_in_a_tool_schema() -> None:
    backend, token, upstream = _backend()
    tool = {
        "id": "score.read",
        "name": "score_read",
        "description": "score a thing",
        "input_schema": {"title": "a\ud800b", "any_of": [{"type": "string"}]},
        "capability": "score",
        "side_effect": "read",
    }

    backend.handle_turn(token, _turn_payload(tool))
    forwarded = upstream.requests[0].tools[0].input_schema
    assert forwarded["title"] == "a�b"
    assert forwarded["any_of"] == [{"type": "string"}]


# --- the replay key ------------------------------------------------------------------------


def test_a_non_finite_tool_schema_issues_no_replay_key() -> None:
    """``_digest`` encodes with ``allow_nan=False``, so a request no canonical JSON can carry
    gets *no key* rather than a fabricated one -- the documented empty-digest answer. The
    prompt digest is unaffected: tool definitions are deliberately outside it."""

    request = normalize_model_request(_request(None, dict(_NAN_SCHEMA)))
    payload = _request_payload(request, ModelConfig(), provider="fake")

    assert _digest(payload) == ""
    assert _digest(_prompt_payload(request)) != ""
