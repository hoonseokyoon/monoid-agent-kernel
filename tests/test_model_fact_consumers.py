"""Call facts through shipped transport, live, and diagnostic consumers."""
from __future__ import annotations

import asyncio
import io
import json
from urllib.error import HTTPError, URLError

import pytest

from support.http import serving
from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.model_facts import (
    ModelCallTermination, UsageEvidence, UsageEvidenceAttempt, UsageObservation,
)
from monoid_agent_kernel.core.model_stream import ModelStreamContext, ModelStreamOutcome
from monoid_agent_kernel.core.spec import GenerationConfig, ModelConfig, ModelRetryConfig, ReasoningConfig
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.model_call import ModelCallRunner
from monoid_agent_kernel.providers._usage import mark_usage_observation, usage_observation_of
from monoid_agent_kernel.providers.base import (
    ModelRequest, ModelTurn, TextDelta, TurnComplete, mark_provider_usage, provider_usage_of,
)
from monoid_agent_kernel.providers.gateway import (
    GatewayModelAdapter, _chunk_from_event, _error_from_status_body, _parse_gateway_response,
)
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.backend.model_stream import LiveModelStreamBroker
from monoid_agent_kernel.reference.llm_gateway.http import create_llm_gateway_server
from monoid_agent_kernel.reference.llm_gateway.service import LlmGatewayBackend


REQUEST = ModelRequest(instruction="go", system_prompt="sys", tools=())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("case", [
    "bad_observation", "bad_control", "invalid_json", "nonobject",
    "proof_generation", "proof_schema", "proof_reasoning", "remote_error", "remote_http",
    "remote_bad_observation", "network",
])
def test_gateway_runner_records_the_owner_of_local_refusals(monkeypatch, stream, case):
    payload = {
        "type": "turn_complete", "final_text": "answer", "usage": {"input_tokens": 7},
        "usage_evidence": {"reported": {"input_tokens": 7}, "source": "provider", "collection": "normal"},
    }
    config_args = {"retry": ModelRetryConfig(max_attempts=1)}
    request = REQUEST
    if case == "bad_observation":
        payload["usage_evidence"]["collection"] = "bogus"
    elif case == "bad_control":
        payload["provider_retried"] = "false"
    elif case.startswith("remote_"):
        # The same classified error code can originate remotely. Ownership decides read-end.
        payload.update(type="error", error="remote refusal", error_code="gateway_bad_response", retryable=False)
    elif case == "proof_generation":
        config_args["generation"] = GenerationConfig(max_output_tokens=17, on_unsupported="fail")
    elif case == "proof_schema":
        request = ModelRequest(instruction="go", system_prompt="sys", tools=(), output_schema={"type": "object"})
    elif case == "proof_reasoning":
        config_args["reasoning"] = ReasoningConfig(effort="low", on_unsupported="fail")
    if case == "remote_bad_observation":
        payload["usage_evidence"]["collection"] = "bogus"
    body = "{" if case == "invalid_json" else "[]" if case == "nonobject" else json.dumps(payload)

    def sync_response(*args, **kwargs):
        if case == "network":
            raise URLError("connection failed")
        if case == "remote_http":
            raise HTTPError("http://gateway.invalid", 422, "refused", {}, io.BytesIO(body.encode()))
        return io.BytesIO(body.encode())

    monkeypatch.setattr("monoid_agent_kernel.providers.gateway.urlopen", sync_response)
    if stream:
        httpx = pytest.importorskip("httpx")
        client_type = httpx.AsyncClient

        def stream_response(http_request):
            if case == "network":
                raise httpx.ReadError("connection failed", request=http_request)
            if case == "remote_http":
                return httpx.Response(422, content=body)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=f"data: {body}\n\n")

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client_type(
            transport=httpx.MockTransport(stream_response), **kwargs,
        ))

    received = []
    runner = ModelCallRunner(
        GatewayModelAdapter(ModelConfig(**config_args), gateway_url="http://gateway.invalid", token="fake"),
        settled_sink=received.append,
    )
    with pytest.raises(ModelAdapterError) as caught:
        asyncio.run(runner.acall(request, delta_consumer=(lambda _: None) if stream else None))
    receipt = received[0].receipt
    expected_usage = 0 if case in {"invalid_json", "nonobject", "network"} else 7
    assert provider_usage_of(caught.value).get("input_tokens", 0) == expected_usage
    assert receipt.usage.get("input_tokens", 0) == expected_usage
    expected_code = (
        f"gateway_{case.removeprefix('proof_')}_not_applied" if case.startswith("proof_")
        else "gateway_network_error" if case == "network" else "gateway_bad_response"
    )
    assert caught.value.provider_error_code == expected_code
    assert receipt.termination.ended_by == (
        "provider_error" if case.startswith("remote_") or case == "network" else "local_failure"
    )


def _server(adapter):
    manager = TokenManager.from_secret("f" * 32)
    backend = LlmGatewayBackend(token_manager=manager, provider_adapter_factory=lambda *_: adapter)
    server = create_llm_gateway_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    token = manager.issue(
        kind="llm_gateway", audience="csp.llm-gateway", run_id="facts", tenant_id="tenant",
        user_id="user", ttl_s=600, metadata={"agent_config_hash": "facts"},
    )
    return server, backend, token


@pytest.mark.parametrize("mode", ["sync", "fallback", "stream"])
@pytest.mark.parametrize("source", ["provider", "adapter"])
@pytest.mark.parametrize("collection", ["normal", "drain"])
def test_gateway_round_trip_preserves_observation_at_actual_runner(mode, source, collection):
    pytest.importorskip("httpx")
    observed = UsageObservation(reported={"input_tokens": 0}, source=source, collection=collection)

    class Adapter:
        def next_turn(self, request):
            return ModelTurn(final_text="answer", usage={"input_tokens": 0}, usage_evidence=observed)

    class Streaming(Adapter):
        async def astream_turn(self, request):
            yield TextDelta("answer")
            yield TurnComplete(usage={"input_tokens": 0}, usage_evidence=observed)

    server, backend, token = _server(Streaming() if mode == "stream" else Adapter())
    received = []
    with serving(server) as url:
        runner = ModelCallRunner(
            GatewayModelAdapter(ModelConfig(), gateway_url=f"{url}/internal/llm/turns", token=token),
            settled_sink=received.append,
        )
        turn, _ = asyncio.run(runner.acall(
            REQUEST, context=InvocationContext(run_id="facts"),
            delta_consumer=(lambda _: None) if mode != "sync" else None,
        ))
    assert turn.final_text == "answer"
    assert turn.usage_evidence == observed
    evidence = received[0].receipt.usage_evidence
    assert evidence.attempts[0].observation == observed
    assert evidence.completeness["input_tokens"] == "complete"
    assert evidence.completeness["total_tokens"] == "unknown"
    assert backend.tenant_usage("tenant")["input_tokens"] == 0


@pytest.mark.parametrize("reader", ["body", "sse", "status"])
@pytest.mark.parametrize("malformed", [False, True])
def test_gateway_error_readers_preserve_valid_observation_and_original_classification(reader, malformed):
    observed = UsageObservation(reported={"input_tokens": 0}, source="adapter", collection="drain")
    payload = {
        "type": "error", "error": "refused", "error_code": "upstream_refused", "retryable": False,
        "http_status": 422, "usage": {"input_tokens": 0},
        "usage_evidence": {"bad": True} if malformed else observed.to_json(),
    }
    if reader == "status":
        error = _error_from_status_body(422, json.dumps(payload))
    else:
        with pytest.raises(ModelAdapterError) as caught:
            (_parse_gateway_response if reader == "body" else _chunk_from_event)(payload)
        error = caught.value
    assert error.provider_error_code == "upstream_refused" and error.retryable is False
    assert provider_usage_of(error)["input_tokens"] == 0
    assert usage_observation_of(error) == (None if malformed else observed)


@pytest.mark.parametrize("stream", [False, True])
def test_gateway_real_http_failure_carries_observation(stream):
    pytest.importorskip("httpx")
    observed = UsageObservation(reported={"output_tokens": 3}, source="provider", collection="drain")

    class Adapter:
        def next_turn(self, request):
            failure = ModelAdapterError("billed refusal", provider_error_code="upstream_refused")
            mark_provider_usage(failure, {"output_tokens": 3})
            mark_usage_observation(failure, observed)
            raise failure

    server, backend, token = _server(Adapter())
    with serving(server) as url:
        adapter = GatewayModelAdapter(ModelConfig(), gateway_url=f"{url}/internal/llm/turns", token=token)
        with pytest.raises(ModelAdapterError) as caught:
            if stream:
                async def consume():
                    return [chunk async for chunk in adapter.astream_turn(REQUEST)]
                asyncio.run(consume())
            else:
                adapter.next_turn(REQUEST)
    assert usage_observation_of(caught.value) == observed
    assert provider_usage_of(caught.value)["output_tokens"] == 3
    assert backend.tenant_usage("tenant")["output_tokens"] == 3


@pytest.mark.parametrize("stream", [False, True])
def test_gateway_applied_proof_refusal_preserves_observation_without_a_runner(stream):
    pytest.importorskip("httpx")
    observed = UsageObservation(reported={"input_tokens": 0}, source="adapter", collection="drain")

    class Adapter:
        def next_turn(self, request):
            return ModelTurn(final_text="billed", usage={"input_tokens": 0}, usage_evidence=observed)

    server, _, token = _server(Adapter())
    with serving(server) as url:
        adapter = GatewayModelAdapter(ModelConfig(
            generation=GenerationConfig(max_output_tokens=17, on_unsupported="fail"),
        ), gateway_url=f"{url}/internal/llm/turns", token=token)
        with pytest.raises(ModelAdapterError) as caught:
            if stream:
                async def consume():
                    return [chunk async for chunk in adapter.astream_turn(REQUEST)]
                asyncio.run(consume())
            else:
                adapter.next_turn(REQUEST)
    assert caught.value.config_recoverable
    assert usage_observation_of(caught.value) == observed


def test_chained_gateway_failure_keeps_observation_and_each_hop_meters_once():
    observed = UsageObservation(reported={"output_tokens": 3}, source="adapter", collection="drain")

    class Adapter:
        def next_turn(self, request):
            error = ModelAdapterError("billed refusal", provider_error_code="upstream_refused")
            mark_provider_usage(error, {"output_tokens": 3})
            mark_usage_observation(error, observed)
            raise error

    inner, inner_backend, inner_token = _server(Adapter())
    with serving(inner) as inner_url:
        upstream = GatewayModelAdapter(ModelConfig(), gateway_url=f"{inner_url}/internal/llm/turns", token=inner_token)
        outer, outer_backend, token = _server(upstream)
        with serving(outer) as url:
            adapter = GatewayModelAdapter(ModelConfig(), gateway_url=f"{url}/internal/llm/turns", token=token)
            with pytest.raises(ModelAdapterError) as caught:
                adapter.next_turn(REQUEST)
    assert usage_observation_of(caught.value) == observed
    assert inner_backend.tenant_usage("tenant")["output_tokens"] == 3
    assert outer_backend.tenant_usage("tenant")["output_tokens"] == 3


@pytest.mark.parametrize("last", [0, 5])
def test_gateway_withheld_terminal_then_error_uses_last_snapshot_once(last):
    pytest.importorskip("httpx")
    observed = UsageObservation(reported={"output_tokens": last}, source="adapter", collection="normal")

    class Adapter:
        async def astream_turn(self, request):
            yield TextDelta("partial")
            yield TurnComplete(usage={"output_tokens": 9}, usage_evidence=UsageObservation(
                reported={"output_tokens": 9}, source="adapter", collection="normal",
            ))
            yield TurnComplete(usage={"output_tokens": last}, usage_evidence=observed)
            raise ConnectionError("after terminal")

    server, backend, token = _server(Adapter())
    with serving(server) as url:
        adapter = GatewayModelAdapter(ModelConfig(), gateway_url=f"{url}/internal/llm/turns", token=token)
        async def consume():
            return [chunk async for chunk in adapter.astream_turn(REQUEST)]
        with pytest.raises(ModelAdapterError) as caught:
            asyncio.run(consume())
    assert usage_observation_of(caught.value) == observed
    assert provider_usage_of(caught.value)["output_tokens"] == last
    assert backend.tenant_usage("tenant")["output_tokens"] == last


def test_relayed_repeated_terminals_keep_collection_through_runner_assembly():
    observed = UsageObservation(reported={"output_tokens": 0}, source="adapter", collection="drain")

    class Adapter:
        async def astream_turn(self, request):
            for count in (9, 0):
                yield _chunk_from_event({
                    "type": "turn_complete", "usage": {"output_tokens": count},
                    "usage_evidence": UsageObservation(
                        reported={"output_tokens": count}, source="adapter", collection="drain",
                    ).to_json(),
                })

    received = []
    turn, _ = asyncio.run(ModelCallRunner(Adapter(), settled_sink=received.append).acall(
        REQUEST, context=InvocationContext(run_id="facts"), delta_consumer=lambda _: None,
    ))
    assert turn.usage_evidence == observed
    assert received[0].receipt.usage_evidence.attempts[0].observation == observed


def _live_close(*, budget=100000, text="answer", attempts=1):
    context = ModelStreamContext(
        run_id="facts", root_run_id="facts", turn_id="turn", stream_id="stream", step=1,
        provider="provider", model="model", started_at="2026-10-27T00:00:00Z",
    )
    evidence = UsageEvidence(attempts=tuple(UsageEvidenceAttempt(
        index=index, observation=UsageObservation(
            reported={"input_tokens": 0, "output_tokens": 3}, source="adapter", collection="normal",
        ),
    ) for index in range(1, attempts + 1)))
    outcome = ModelStreamOutcome(
        status="completed", final_text=text, usage={"input_tokens": 0, "output_tokens": 3},
        termination=ModelCallTermination(outcome="completed", ended_by="provider_terminal"),
        usage_evidence=evidence,
    )
    broker = LiveModelStreamBroker(generation="facts", max_bytes=budget)
    broker.observer("facts").open(context).close(outcome)
    frames = broker.subscribe("facts").poll()
    return frames[-1], outcome


def test_live_broker_projects_facts_independent_of_final_turn_usage():
    frame, outcome = _live_close(attempts=2)
    assert frame.to_json()["termination"] == outcome.termination.to_json()
    assert frame.to_json()["usage_evidence"] == outcome.usage_evidence.to_json()
    assert frame.usage["output_tokens"] == 3
    assert frame.to_json()["usage_evidence"]["reported"]["output_tokens"] == 6


def test_live_compact_marker_retains_facts_when_they_fit():
    frame, outcome = _live_close(budget=2000, text="answer" * 1000)
    assert frame.content_omitted and frame.final_text is None
    assert frame.to_json()["usage_evidence"] == outcome.usage_evidence.to_json()


def test_live_optional_facts_do_not_displace_fitting_legacy_content():
    frame, _ = _live_close(budget=2000, attempts=100)
    assert frame.kind == "closed" and frame.final_text == "answer" and not frame.content_omitted
    assert "usage_evidence" not in frame.to_json()


def test_live_oversized_facts_still_leave_the_existing_compact_marker():
    frame, _ = _live_close(budget=2000, attempts=100, text="answer" * 1000)
    assert frame.kind == "closed" and frame.content_omitted and frame.final_text is None
    assert "usage_evidence" not in frame.to_json()


def test_otel_actual_retry_reports_bounded_facts_on_existing_spans():
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from monoid_agent_kernel.observability.otel import OtelEventSink

    class Adapter:
        calls = 0

        def next_turn(self, request):
            self.calls += 1
            if self.calls == 1:
                error = ModelAdapterError("retry", retryable=True)
                mark_provider_usage(error, {"input_tokens": 2})
                mark_usage_observation(error, UsageObservation(
                    reported={"input_tokens": 2}, source="provider", collection="normal",
                ))
                raise error
            return ModelTurn(final_text="ok", usage={"input_tokens": 0}, usage_evidence=UsageObservation(
                reported={"input_tokens": 0}, source="adapter", collection="normal",
            ))

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    sink = OtelEventSink(tracer_provider=provider, span_mode="model_call")
    request = ModelRequest(instruction="go", system_prompt="sys", tools=(), model=ModelConfig(
        retry=ModelRetryConfig(layer="kernel", max_attempts=2, initial_delay_s=0, jitter_s=0),
    ))
    asyncio.run(ModelCallRunner(Adapter(), subscriptions=(sink.model_io_subscription(),)).acall(
        request, context=InvocationContext(run_id="facts"),
    ))
    sink.close()
    spans = exporter.get_finished_spans()
    parent = next(span for span in spans if span.name.startswith("chat"))
    assert parent.attributes["monoid.model.termination.outcome"] == "completed"
    assert parent.attributes["gen_ai.usage.input_tokens"] == 2
    assert json.loads(parent.attributes["monoid.model.usage.reported"]) == {"input_tokens": 2}
    coverage = json.loads(parent.attributes["monoid.model.usage.completeness"])
    assert coverage["input_tokens"] == "complete" and coverage["total_tokens"] == "unknown"
    children = sorted((s for s in spans if s.name.startswith("model.attempt")), key=lambda s: s.name)
    assert len(children) == 2
    assert [s.attributes["monoid.model.attempt.usage.source"] for s in children] == ["provider", "adapter"]
    assert all(not any(key.startswith("gen_ai.") for key in s.attributes) for s in children)
