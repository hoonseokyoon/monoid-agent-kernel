"""OTel: the OtelEventSink maps the event tree to GenAI spans. Needs opentelemetry-sdk
(InMemorySpanExporter); skipped if absent. The sink takes an explicit TracerProvider so the
test never touches the process-global provider.
"""

from __future__ import annotations

# Imports below the importorskip guard are intentionally not at top of file.
# ruff: noqa: E402

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core.events import make_agent_event
from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.model_io import (
    CapturePolicy,
    ModelCallAttempt,
    ModelCallCapture,
    ModelCallReceipt,
    RedactionPolicy,
)
from monoid_agent_kernel.core.spec import AgentRunSpec, ModelConfig, RunLimits
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.observability.otel import OtelEventSink
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call


def _spans_and_run(tmp_path: Path, adapter: FakeModelAdapter, *tool_ids: str):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    loop = AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace, run_root=tmp_path / "runs", limits=RunLimits(max_steps=4)
        ),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config(*tool_ids)),
        event_sinks=(OtelEventSink(tracer_provider=provider),),
    )
    result = asyncio.run(loop.arun_once("go"))
    return exporter.get_finished_spans(), result


def _by_name(spans):
    return {s.name: s for s in spans}


def _provider_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _run_with_preset(
    tmp_path: Path,
    adapter: FakeModelAdapter,
    *,
    policy: CapturePolicy,
    instruction: str,
    invocation_context: InvocationContext | None = None,
):
    provider, exporter = _provider_and_exporter()
    preset = OtelEventSink(
        tracer_provider=provider,
        parent_context=invocation_context,
        capture_policy=policy,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    loop = AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace,
            run_root=tmp_path / "runs",
            limits=RunLimits(max_steps=2),
        ),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
        invocation_context=invocation_context,
        event_sinks=(preset,),
        model_io_subscriptions=(preset.model_io_subscription(),),
    )
    result = asyncio.run(loop.arun_once(instruction))
    return exporter.get_finished_spans(), result


def test_otel_sink_builds_genai_span_tree(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_write", {"path": "A.md", "content": "x"}, "c1"),),
                usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            ),
            ModelTurn(
                response_id="r2",
                final_text="done",
                usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            ),
        ]
    )
    spans, result = _spans_and_run(tmp_path, adapter, "fs.write", "run.finish")
    assert result.status == "completed"

    names = [s.name for s in spans]
    # One invoke_agent root, two chat spans, one execute_tool span.
    assert names.count("invoke_agent") == 1
    assert names.count("chat gpt-5.5") == 2
    assert names.count("execute_tool fs_write") == 1

    root = next(s for s in spans if s.name == "invoke_agent")
    chat = next(s for s in spans if s.name == "chat gpt-5.5")
    tool = next(s for s in spans if s.name == "execute_tool fs_write")

    # chat and execute_tool are SIBLINGS under invoke_agent (not nested under each other).
    assert chat.parent.span_id == root.context.span_id
    assert tool.parent.span_id == root.context.span_id

    # GenAI attributes.
    assert root.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert chat.attributes["gen_ai.operation.name"] == "chat"
    assert chat.attributes["gen_ai.provider.name"] == "gateway"
    assert chat.attributes["gen_ai.request.model"] == "gpt-5.5"
    assert tool.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool.attributes["gen_ai.tool.name"] == "fs_write"
    assert tool.attributes["gen_ai.tool.call.id"] == "c1"
    assert tool.attributes["turn_id"]

    # Token usage rolled onto the chat span(s); finish reasons reflect tool-call vs final.
    chats = [s for s in spans if s.name == "chat gpt-5.5"]
    assert any(s.attributes.get("gen_ai.usage.input_tokens") == 7 for s in chats)
    assert any(
        tuple(s.attributes.get("gen_ai.response.finish_reasons") or ()) == ("tool_calls",)
        for s in chats
    )
    assert any(
        tuple(s.attributes.get("gen_ai.response.finish_reasons") or ()) == ("stop",) for s in chats
    )


def test_the_event_only_sink_attributes_the_chat_span_to_the_answering_provider(
    tmp_path: Path,
) -> None:
    """One preset, two configurations, and they used to disagree about one attribute.

    ``OtelEventSink`` writes ``gen_ai.provider.name`` from two places: the receipt-derived model
    -call spans (``receipt.provider_name or receipt.model.provider``) and this event-driven chat
    span, which read ``run.started``'s ``model_provider`` -- filled from ``ModelConfig.provider``.
    Through the gateway that made the receipt say the upstream ("openai") and the event say the
    transport ("gateway") for the *same call*, and the docs' own quickstart (``OtelEventSink()``
    with no model-I/O subscription) exercises only the disagreeing half.

    ``run.started`` now reports the provider that actually serves the run. Its only reader in the
    repo is this sink; the transport is still recorded, on ``manifest.json`` beside it, which is
    the artifact that documents the run's *configuration*.
    """

    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="done")])
    # Tagged like the gateway adapter relaying an OpenAI upstream: transport "gateway", answers
    # attributed to "openai".
    adapter.provider_name = "openai"

    spans, result = _spans_and_run(tmp_path, adapter, "run.finish")

    assert result.status == "completed"
    chat = next(s for s in spans if s.name.startswith("chat"))
    assert chat.attributes["gen_ai.provider.name"] == "openai"

    run_dir = result.run_dir
    started = next(
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["type"] == "run.started"
    )
    assert started["data"]["model_provider"] == "openai"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_provider"] == "gateway", "the transport must stay legible on the record"


def test_the_event_only_sink_falls_back_to_the_configured_provider(tmp_path: Path) -> None:
    """An adapter that declares nothing leaves the config's string in place -- the neutral case,
    and the reason the pin above is about agreement rather than about renaming a field."""

    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="done")])

    spans, result = _spans_and_run(tmp_path, adapter, "run.finish")

    assert result.status == "completed"
    chat = next(s for s in spans if s.name.startswith("chat"))
    assert chat.attributes["gen_ai.provider.name"] == "gateway"


def test_otel_sink_marks_failed_tool_span(tmp_path: Path) -> None:
    from opentelemetry.trace.status import StatusCode

    adapter = FakeModelAdapter(
        turns=[
            # Unknown tool -> the tool call fails -> tool.call.failed.
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("does_not_exist", {}, "c1"),)),
            ModelTurn(response_id="r2", final_text="done"),
        ]
    )
    spans, _ = _spans_and_run(tmp_path, adapter, "fs.write", "run.finish")

    tool_spans = [s for s in spans if s.name.startswith("execute_tool")]
    assert tool_spans
    failed = tool_spans[0]
    assert failed.status.status_code == StatusCode.ERROR
    assert failed.attributes.get("error.type")


def test_otel_records_output_validation_failure_on_run_span(tmp_path: Path) -> None:
    from monoid_agent_kernel.core.output_validator import ValidationOutcome

    class _RequireDone:
        id = "otel.requires_done"
        schema = None

        def validate(self, view) -> ValidationOutcome:
            if "DONE" in view.final_text:
                return ValidationOutcome(ok=True, value=None)
            return ValidationOutcome(ok=False, feedback="must contain DONE")

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", final_text="nope", stop_reason="stop"),
            ModelTurn(response_id="r2", final_text="DONE now", stop_reason="stop"),
        ]
    )
    loop = AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace, run_root=tmp_path / "runs", limits=RunLimits(max_steps=4)
        ),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
        output_validators=(_RequireDone(),),
        event_sinks=(OtelEventSink(tracer_provider=provider),),
    )
    result = asyncio.run(loop.arun_once("go"))
    assert result.status == "completed"

    root = next(s for s in exporter.get_finished_spans() if s.name == "invoke_agent")
    assert "output.validation.failed" in [e.name for e in root.events]


def test_otel_records_output_validator_exhausted_on_run_span(tmp_path: Path) -> None:
    from monoid_agent_kernel.core.output_validator import ValidationOutcome

    class _AlwaysFail:
        id = "otel.always_fail"
        schema = None

        def validate(self, view) -> ValidationOutcome:
            return ValidationOutcome(ok=False, feedback="nope")

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    adapter = FakeModelAdapter(
        turns=[ModelTurn(response_id="r1", final_text="x", stop_reason="stop")]
    )
    loop = AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace,
            run_root=tmp_path / "runs",
            limits=RunLimits(max_steps=4, max_output_retries=0),
        ),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
        output_validators=(_AlwaysFail(),),
        event_sinks=(OtelEventSink(tracer_provider=provider),),
    )
    result = asyncio.run(loop.arun_once("go"))
    assert result.status == "limited"

    root = next(s for s in exporter.get_finished_spans() if s.name == "invoke_agent")
    assert "output.validator.exhausted" in [e.name for e in root.events]


def test_otel_preset_preserves_upstream_parent_and_defaults_to_no_capture(tmp_path: Path) -> None:
    trace_id = "1" * 32
    parent_span_id = "2" * 16
    invocation = InvocationContext(
        traceparent=f"00-{trace_id}-{parent_span_id}-01",
        tracestate="vendor=value",
    )
    spans, result = _run_with_preset(
        tmp_path,
        FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="SECRET output")]),
        policy=CapturePolicy(mode="none"),
        instruction="SECRET input",
        invocation_context=invocation,
    )
    assert result.status == "completed"

    root = next(span for span in spans if span.name == "invoke_agent")
    chat = next(span for span in spans if span.name.startswith("chat"))
    assert f"{root.context.trace_id:032x}" == trace_id
    assert root.parent is not None
    assert f"{root.parent.span_id:016x}" == parent_span_id
    assert chat.parent.span_id == root.context.span_id
    assert chat.attributes["monoid.model.capture.mode"] == "none"
    exported = repr([(span.attributes, span.events) for span in spans])
    assert "SECRET" not in exported
    assert "monoid.model.capture.digests" not in exported
    assert "monoid.model.capture.lengths" not in exported
    assert "monoid.model.capture.content" not in exported


def test_otel_preset_digest_enriches_existing_chat_without_duplicate(tmp_path: Path) -> None:
    spans, result = _run_with_preset(
        tmp_path,
        FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="private answer")]),
        policy=CapturePolicy(mode="digest"),
        instruction="private question",
    )
    assert result.status == "completed"
    chats = [span for span in spans if span.name.startswith("chat")]
    assert len(chats) == 1
    chat = chats[0]
    assert chat.attributes["monoid.model.capture.mode"] == "digest"
    assert "instruction" in chat.attributes["monoid.model.capture.digests"]
    assert "monoid.model.capture.content" not in chat.attributes
    assert "private question" not in repr(chat.attributes)
    assert "private answer" not in repr(chat.attributes)


def test_otel_preset_redaction_failure_downgrades_without_disclosure(tmp_path: Path) -> None:
    class _FailingRedactor:
        def redact(self, value, *, policy: RedactionPolicy):
            del value, policy
            raise RuntimeError("redactor unavailable")

    spans, result = _run_with_preset(
        tmp_path,
        FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="TOP SECRET")]),
        policy=CapturePolicy(mode="redacted", redactor=_FailingRedactor()),
        instruction="TOP SECRET",
    )
    assert result.status == "completed"
    chat = next(span for span in spans if span.name.startswith("chat"))
    assert chat.attributes["monoid.model.capture.mode"] == "digest"
    assert chat.attributes["monoid.model.capture.downgraded_from"] == "redacted"
    assert "monoid.model.capture.content" not in chat.attributes
    assert "TOP SECRET" not in repr(chat.attributes)


def test_otel_preset_full_capture_is_explicit(tmp_path: Path) -> None:
    spans, result = _run_with_preset(
        tmp_path,
        FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="VISIBLE output")]),
        policy=CapturePolicy(mode="full"),
        instruction="VISIBLE input",
    )
    assert result.status == "completed"
    chat = next(span for span in spans if span.name.startswith("chat"))
    content = chat.attributes["monoid.model.capture.content"]
    assert "VISIBLE input" in content
    assert "VISIBLE output" in content


def test_otel_model_call_mode_emits_one_span_under_receipt_parent() -> None:
    provider, exporter = _provider_and_exporter()
    trace_id = "3" * 32
    parent_span_id = "4" * 16
    context = InvocationContext(
        traceparent=f"00-{trace_id}-{parent_span_id}-01", step_id="skill-step"
    )
    preset = OtelEventSink(
        tracer_provider=provider,
        span_mode="model_call",
        capture_policy=CapturePolicy(mode="none"),
    )
    preset.on_model_call(
        ModelCallCapture(
            receipt=ModelCallReceipt(
                context=context,
                model=ModelConfig(model="standalone-model"),
                provider_name="gateway",
                stop_reason="stop",
                usage={"input_tokens": 3, "output_tokens": 2},
                latency_ms=5,
            )
        )
    )
    preset.close()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["chat standalone-model"]
    span = spans[0]
    assert f"{span.context.trace_id:032x}" == trace_id
    assert span.parent is not None
    assert f"{span.parent.span_id:016x}" == parent_span_id
    assert span.attributes["gen_ai.usage.input_tokens"] == 3
    assert span.attributes["monoid.model.capture.mode"] == "none"


def test_otel_agent_mode_lazily_opens_root_for_restored_activation() -> None:
    provider, exporter = _provider_and_exporter()
    preset = OtelEventSink(
        tracer_provider=provider,
        capture_policy=CapturePolicy(mode="digest"),
    )
    started = make_agent_event(
        run_id="restored-run",
        seq=9,
        event_type="model.turn.started",
        turn_id="turn_0009",
    )
    preset.emit(started)
    preset.on_model_call(
        ModelCallCapture(
            receipt=ModelCallReceipt(
                context=InvocationContext(run_id="restored-run", step_id="turn_0009"),
                model=ModelConfig(model="restored-model"),
                provider_name="gateway",
                stop_reason="stop",
            ),
            mode="digest",
            digests={"instruction": "a" * 64},
        )
    )
    preset.emit(
        make_agent_event(
            run_id="restored-run",
            seq=10,
            event_type="model.turn.finished",
            turn_id="turn_0009",
            parent_id=started.event_id,
            data={"has_final": True},
        )
    )
    preset.close()
    preset.close()

    spans = exporter.get_finished_spans()
    root = next(span for span in spans if span.name == "invoke_agent")
    chat = next(span for span in spans if span.name == "chat restored-model")
    assert root.attributes["run_id"] == "restored-run"
    assert chat.parent.span_id == root.context.span_id
    assert len([span for span in spans if span.name.startswith("chat")]) == 1


def test_otel_rejects_unknown_span_mode() -> None:
    with pytest.raises(ValueError, match="span_mode"):
        OtelEventSink(span_mode="both")  # type: ignore[arg-type]


def test_otel_defaults_to_none_capture_policy() -> None:
    provider, _exporter = _provider_and_exporter()
    preset = OtelEventSink(tracer_provider=provider)
    assert preset.capture_policy.mode == "none"
    assert preset.model_io_subscription().policy.mode == "none"
    preset.close()


def test_otel_ignores_malformed_parent_and_never_copies_error_prose_to_status() -> None:
    from opentelemetry.trace.status import StatusCode

    provider, exporter = _provider_and_exporter()
    preset = OtelEventSink(
        tracer_provider=provider,
        parent_context=InvocationContext(traceparent="malformed"),
    )
    preset.emit(
        make_agent_event(
            run_id="failed-run",
            seq=1,
            event_type="run.started",
            data={"model": "gpt-5.5", "model_provider": "gateway"},
        )
    )
    preset.emit(
        make_agent_event(
            run_id="failed-run",
            seq=2,
            event_type="run.failed",
            data={"error": "SECRET provider body", "error_code": "model_error"},
        )
    )
    preset.close()

    root = next(span for span in exporter.get_finished_spans() if span.name == "invoke_agent")
    assert root.parent is None
    assert root.status.status_code == StatusCode.ERROR
    assert not root.status.description
    assert "SECRET provider body" not in repr(root)


def test_otel_close_contains_processor_failure_and_still_attempts_every_span() -> None:
    class _RaisingEndProcessor(SpanProcessor):
        def on_start(self, span, parent_context=None) -> None:
            del span, parent_context

        def on_end(self, span) -> None:
            del span
            raise RuntimeError("exporter unavailable")

        def shutdown(self) -> None:
            return None

        def force_flush(self, timeout_millis: int = 30_000) -> bool:
            del timeout_millis
            return True

    provider, exporter = _provider_and_exporter()
    provider.add_span_processor(_RaisingEndProcessor())
    preset = OtelEventSink(tracer_provider=provider)
    preset.emit(
        make_agent_event(
            run_id="run-close",
            seq=1,
            event_type="run.started",
            data={"model": "gpt-5.5"},
        )
    )
    preset.emit(
        make_agent_event(
            run_id="run-close",
            seq=2,
            event_type="model.turn.started",
            turn_id="turn_0001",
        )
    )

    preset.close()
    preset.close()
    assert sorted(span.name for span in exporter.get_finished_spans()) == [
        "chat gpt-5.5",
        "invoke_agent",
    ]


def test_otel_standalone_serialization_failure_still_ends_fixed_duration_span() -> None:
    provider, exporter = _provider_and_exporter()
    preset = OtelEventSink(tracer_provider=provider, span_mode="model_call")
    preset.on_model_call(
        ModelCallCapture(
            receipt=ModelCallReceipt(latency_ms=5),
            mode="full",
            content={"not_portable_json": float("nan")},
        )
    )
    preset.close()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].end_time - spans[0].start_time == 5_000_000


def test_otel_agent_abort_is_interruption_not_error() -> None:
    from opentelemetry.trace.status import StatusCode

    provider, exporter = _provider_and_exporter()
    preset = OtelEventSink(
        tracer_provider=provider,
        capture_policy=CapturePolicy(mode="none"),
    )
    preset.emit(make_agent_event(run_id="run-abort", seq=1, event_type="run.started"))
    started = make_agent_event(
        run_id="run-abort",
        seq=2,
        event_type="model.turn.started",
        turn_id="turn_0001",
    )
    preset.emit(started)
    preset.on_model_call(
        ModelCallCapture(
            receipt=ModelCallReceipt(
                context=InvocationContext(run_id="run-abort", step_id="turn_0001"),
                error_code="model_call_aborted",
            )
        )
    )
    preset.emit(make_agent_event(run_id="run-abort", seq=3, event_type="turn.interrupted"))
    preset.close()

    chat = next(span for span in exporter.get_finished_spans() if span.name.startswith("chat"))
    assert chat.status.status_code == StatusCode.UNSET
    assert "error.type" not in chat.attributes


def test_otel_agent_non_abort_failure_keeps_receipt_taxonomy() -> None:
    from opentelemetry.trace.status import StatusCode

    provider, exporter = _provider_and_exporter()
    preset = OtelEventSink(
        tracer_provider=provider,
        capture_policy=CapturePolicy(mode="none"),
    )
    preset.emit(make_agent_event(run_id="run-failed-call", seq=1, event_type="run.started"))
    started = make_agent_event(
        run_id="run-failed-call",
        seq=2,
        event_type="model.turn.started",
        turn_id="turn_0001",
    )
    preset.emit(started)
    preset.on_model_call(
        ModelCallCapture(
            receipt=ModelCallReceipt(
                context=InvocationContext(run_id="run-failed-call", step_id="turn_0001"),
                error_code="model_error",
                provider_error_code="quota",
                retryable=True,
                http_status=429,
            )
        )
    )
    preset.emit(
        make_agent_event(
            run_id="run-failed-call",
            seq=3,
            event_type="run.failed",
            data={"error_code": "model_error"},
        )
    )
    preset.close()

    chat = next(span for span in exporter.get_finished_spans() if span.name.startswith("chat"))
    assert chat.status.status_code == StatusCode.ERROR
    assert chat.attributes["error.type"] == "quota"
    assert chat.attributes["monoid.model.retryable"] is True
    assert chat.attributes["monoid.model.http_status"] == 429


def test_otel_receipt_stop_reason_wins_over_event_fallback(tmp_path: Path) -> None:
    spans, result = _run_with_preset(
        tmp_path,
        FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="r1",
                    final_text="truncated",
                    stop_reason="length",
                )
            ]
        ),
        policy=CapturePolicy(mode="none"),
        instruction="go",
    )
    assert result.status == "limited"
    chat = next(span for span in spans if span.name.startswith("chat"))
    assert tuple(chat.attributes["gen_ai.response.finish_reasons"]) == ("length",)


def test_otel_terminal_only_restored_activation_still_emits_root() -> None:
    from opentelemetry.trace.status import StatusCode

    provider, exporter = _provider_and_exporter()
    preset = OtelEventSink(tracer_provider=provider)
    preset.emit(
        make_agent_event(
            run_id="restored-terminal",
            seq=12,
            event_type="run.failed",
            data={"error_code": "cancelled"},
        )
    )
    # Some terminal failure paths emit the same boundary again during finalization. The preset is
    # per activation and must not turn that duplicate into a second root span.
    preset.emit(
        make_agent_event(
            run_id="restored-terminal",
            seq=13,
            event_type="run.failed",
            data={"error_code": "cancelled"},
        )
    )
    preset.close()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["invoke_agent"]
    assert spans[0].attributes["run_id"] == "restored-terminal"
    assert spans[0].status.status_code == StatusCode.ERROR


def test_otel_successful_redaction_records_policy_and_masks_content(tmp_path: Path) -> None:
    policy = CapturePolicy(
        mode="redacted",
        redaction=RedactionPolicy(literals=("SECRET",)),
    )
    spans, result = _run_with_preset(
        tmp_path,
        FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="SECRET output")]),
        policy=policy,
        instruction="SECRET input",
    )
    assert result.status == "completed"
    chat = next(span for span in spans if span.name.startswith("chat"))
    assert chat.attributes["monoid.model.capture.mode"] == "redacted"
    assert (
        chat.attributes["monoid.model.capture.redaction_digest"]
        == policy.effective_redaction.digest
    )
    assert "SECRET" not in chat.attributes["monoid.model.capture.content"]
    assert "[redacted]" in chat.attributes["monoid.model.capture.content"]
    assert "monoid.model.capture.digests" in chat.attributes
    assert "monoid.model.capture.lengths" in chat.attributes


# --- W7-2: per-attempt children synthesized from the settled attempt log ----------------------


def _attempt_receipt() -> ModelCallReceipt:
    """Two dispatches: a billed retryable failure, then the answering attempt after a 40ms wait."""

    return ModelCallReceipt(
        context=InvocationContext(run_id="run-attempts", step_id="turn_0001"),
        model=ModelConfig(model="retry-model"),
        provider_name="gateway",
        stop_reason="stop",
        usage={"input_tokens": 3, "output_tokens": 7},
        latency_ms=60,
        attempts=2,
        attempt_log=(
            ModelCallAttempt(
                index=1,
                elapsed_ms=5,
                backoff_ms=0,
                error_code="model_error",
                provider_error_code="overloaded",
                retryable=True,
                http_status=529,
                usage={"output_tokens": 2},
            ),
            ModelCallAttempt(
                index=2,
                elapsed_ms=3,
                backoff_ms=40,
                usage={"input_tokens": 3, "output_tokens": 5},
                stream_committed=True,
            ),
        ),
    )


def _mode_preset(span_mode: str, provider) -> OtelEventSink:
    return OtelEventSink(
        tracer_provider=provider,
        span_mode=span_mode,  # type: ignore[arg-type]
        capture_policy=CapturePolicy(mode="none"),
    )


def _deliver(preset: OtelEventSink, receipt: ModelCallReceipt, span_mode: str) -> None:
    """One settled call, through whichever wiring the mode uses."""

    if span_mode == "agent":
        preset.emit(make_agent_event(run_id="run-attempts", seq=1, event_type="run.started"))
        preset.emit(
            make_agent_event(
                run_id="run-attempts",
                seq=2,
                event_type="model.turn.started",
                turn_id="turn_0001",
            )
        )
    preset.on_model_call(ModelCallCapture(receipt=receipt))


def _attempt_children(spans):
    return sorted(
        (span for span in spans if span.name.startswith("model.attempt")),
        key=lambda span: span.attributes["monoid.model.attempt.index"],
    )


@pytest.mark.parametrize("span_mode", ["agent", "model_call"])
def test_otel_synthesizes_attempt_children_under_the_chat_span(span_mode: str) -> None:
    """One INTERNAL child per logged dispatch, in BOTH wirings -- the mode census. Widths are
    the entries' own `elapsed_ms`, the gap is the recorded backoff, the failed dispatch
    carries the error and the answering one does not."""

    from opentelemetry.trace import SpanKind
    from opentelemetry.trace.status import StatusCode

    provider, exporter = _provider_and_exporter()
    preset = _mode_preset(span_mode, provider)
    _deliver(preset, _attempt_receipt(), span_mode)
    preset.close()

    spans = exporter.get_finished_spans()
    chat = next(span for span in spans if span.name.startswith("chat"))
    first, second = _attempt_children(spans)

    assert [first.name, second.name] == ["model.attempt 1", "model.attempt 2"]
    for child in (first, second):
        assert child.kind is SpanKind.INTERNAL
        assert child.parent is not None
        assert child.parent.span_id == chat.context.span_id
    assert first.end_time - first.start_time == 5_000_000
    assert second.end_time - second.start_time == 3_000_000
    assert second.start_time - first.end_time == 40_000_000
    assert first.status.status_code == StatusCode.ERROR
    assert first.attributes["error.type"] == "overloaded"
    assert first.attributes["monoid.model.attempt.error_code"] == "model_error"
    assert first.attributes["monoid.model.attempt.http_status"] == 529
    assert first.attributes["monoid.model.attempt.retryable"] is True
    assert second.status.status_code == StatusCode.UNSET
    assert "error.type" not in second.attributes
    assert second.attributes["monoid.model.attempt.backoff_ms"] == 40
    assert second.attributes["monoid.model.attempt.stream_committed"] is True
    assert "output_tokens" in second.attributes["monoid.model.attempt.usage"]
    assert chat.attributes["monoid.model.attempts"] == 2


@pytest.mark.parametrize("span_mode", ["agent", "model_call"])
def test_otel_attempt_children_carry_only_their_own_namespace(span_mode: str) -> None:
    """The attribute rule as a class, not a key list: everything on a child is
    `monoid.model.attempt.*` or `error.type`. `gen_ai.*` stays on the parent -- a GenAI-aware
    backend aggregating usage or operation counts over those spans would double-count the
    call otherwise -- and capture content never propagates down."""

    provider, exporter = _provider_and_exporter()
    preset = _mode_preset(span_mode, provider)
    _deliver(preset, _attempt_receipt(), span_mode)
    preset.close()

    children = _attempt_children(exporter.get_finished_spans())
    assert children
    for child in children:
        for key in child.attributes:
            assert key == "error.type" or key.startswith("monoid.model.attempt."), key


@pytest.mark.parametrize("span_mode", ["agent", "model_call"])
def test_otel_single_dispatch_and_legacy_logs_synthesize_no_children(span_mode: str) -> None:
    """The threshold's other arm, both ways it happens: one dispatch (the chat span IS that
    attempt, a child would restate it at double the span volume) and a legacy receipt whose
    log predates the field (`attempts` says 3, the log says nothing to draw)."""

    single = ModelCallReceipt(
        context=InvocationContext(run_id="run-attempts", step_id="turn_0001"),
        model=ModelConfig(model="retry-model"),
        stop_reason="stop",
        usage={"output_tokens": 7},
        attempts=1,
        attempt_log=(ModelCallAttempt(index=1, usage={"output_tokens": 7}),),
    )
    legacy = ModelCallReceipt(
        context=InvocationContext(run_id="run-attempts", step_id="turn_0001"),
        model=ModelConfig(model="retry-model"),
        stop_reason="stop",
        attempts=3,
    )
    for receipt in (single, legacy):
        provider, exporter = _provider_and_exporter()
        preset = _mode_preset(span_mode, provider)
        _deliver(preset, receipt, span_mode)
        preset.close()
        names = [span.name for span in exporter.get_finished_spans()]
        assert not [name for name in names if name.startswith("model.attempt")], names


def test_otel_an_aborted_final_attempt_is_not_an_error() -> None:
    """The parent's abort rule, held per entry: the dispatch the abort ended reads UNSET, and
    the billed failure before it stays the error."""

    from opentelemetry.trace.status import StatusCode

    receipt = ModelCallReceipt(
        context=InvocationContext(run_id="run-attempts", step_id="turn_0001"),
        model=ModelConfig(model="retry-model"),
        usage={"output_tokens": 2},
        latency_ms=60,
        attempts=2,
        error_code="model_call_aborted",
        attempt_log=(
            ModelCallAttempt(
                index=1,
                elapsed_ms=5,
                backoff_ms=0,
                error_code="model_error",
                provider_error_code="overloaded",
                retryable=True,
                usage={"output_tokens": 2},
            ),
            ModelCallAttempt(
                index=2, elapsed_ms=3, backoff_ms=40, error_code="model_call_aborted"
            ),
        ),
    )
    provider, exporter = _provider_and_exporter()
    preset = _mode_preset("agent", provider)
    _deliver(preset, receipt, "agent")
    preset.emit(make_agent_event(run_id="run-attempts", seq=3, event_type="turn.interrupted"))
    preset.close()

    first, second = _attempt_children(exporter.get_finished_spans())
    assert first.status.status_code == StatusCode.ERROR
    assert second.status.status_code == StatusCode.UNSET
    assert "error.type" not in second.attributes


def test_otel_duplicate_delivery_does_not_duplicate_children() -> None:
    """`on_model_call` is public and may be called directly; the agent-mode enrich is
    idempotent on attributes, and the synthesized children must not multiply with it."""

    provider, exporter = _provider_and_exporter()
    preset = _mode_preset("agent", provider)
    receipt = _attempt_receipt()
    _deliver(preset, receipt, "agent")
    preset.on_model_call(ModelCallCapture(receipt=receipt))
    preset.close()

    assert len(_attempt_children(exporter.get_finished_spans())) == 2


def test_otel_receipt_without_an_open_chat_span_synthesizes_no_orphans() -> None:
    """No matching chat span (the existing enrich no-op) means no children either -- attempt
    spans annotate a call the trace already shows, they do not invent one."""

    provider, exporter = _provider_and_exporter()
    preset = _mode_preset("agent", provider)
    preset.on_model_call(ModelCallCapture(receipt=_attempt_receipt()))
    preset.close()

    assert exporter.get_finished_spans() == ()


def test_otel_legacy_backoff_packs_children_edge_to_edge() -> None:
    """Entries parsed from pre-`backoff_ms` lines carry None: durations and order stay exact,
    the unknown gaps collapse to zero, and no backoff attribute is invented."""

    receipt = ModelCallReceipt(
        context=InvocationContext(run_id="run-attempts", step_id="turn_0001"),
        model=ModelConfig(model="retry-model"),
        stop_reason="stop",
        usage={"output_tokens": 7},
        latency_ms=60,
        attempts=2,
        attempt_log=(
            ModelCallAttempt(
                index=1, elapsed_ms=5, error_code="model_error", retryable=True
            ),
            ModelCallAttempt(index=2, elapsed_ms=3, usage={"output_tokens": 7}),
        ),
    )
    provider, exporter = _provider_and_exporter()
    preset = _mode_preset("agent", provider)
    _deliver(preset, receipt, "agent")
    preset.close()

    first, second = _attempt_children(exporter.get_finished_spans())
    assert second.start_time - first.end_time == 0
    assert "monoid.model.attempt.backoff_ms" not in first.attributes
    assert "monoid.model.attempt.backoff_ms" not in second.attributes


def test_otel_full_capture_children_stay_content_free() -> None:
    """Content is the parent's opt-in; the children are metadata by construction, whatever the
    policy says."""

    provider, exporter = _provider_and_exporter()
    preset = OtelEventSink(
        tracer_provider=provider,
        span_mode="model_call",
        capture_policy=CapturePolicy(mode="full"),
    )
    preset.on_model_call(
        ModelCallCapture(
            receipt=_attempt_receipt(),
            mode="full",
            content={"instruction": "SECRET input", "output_text": "SECRET output"},
        )
    )
    preset.close()

    spans = exporter.get_finished_spans()
    chat = next(span for span in spans if span.name.startswith("chat"))
    assert "SECRET" in chat.attributes["monoid.model.capture.content"]
    for child in _attempt_children(spans):
        assert "SECRET" not in repr(child.attributes)


def test_otel_agent_run_with_kernel_retry_exports_attempt_children(tmp_path: Path) -> None:
    """End to end through the loop: both facets registered, the kernel absorbs one retryable
    failure, and the exported trace shows the chat span with its two dispatch children."""

    from monoid_agent_kernel.core.spec import ModelRetryConfig
    from monoid_agent_kernel.errors import ModelAdapterError
    from opentelemetry.trace.status import StatusCode

    class _FlakyOnce:
        def __init__(self) -> None:
            self.calls = 0

        def next_turn(self, request):  # noqa: ANN001
            del request
            self.calls += 1
            if self.calls == 1:
                raise ModelAdapterError("transient", retryable=True)
            return ModelTurn(response_id="r1", final_text="done", stop_reason="stop")

    provider, exporter = _provider_and_exporter()
    preset = OtelEventSink(tracer_provider=provider, capture_policy=CapturePolicy(mode="none"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    loop = AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace, run_root=tmp_path / "runs", limits=RunLimits(max_steps=2)
        ),
        model_adapter=_FlakyOnce(),
        runtime_config_provider=runtime_provider(
            runtime_config(
                "run.finish",
                model=ModelConfig(
                    retry=ModelRetryConfig(
                        layer="kernel", max_attempts=2, initial_delay_s=0.0, jitter_s=0.0
                    )
                ),
            )
        ),
        event_sinks=(preset,),
        model_io_subscriptions=(preset.model_io_subscription(),),
    )
    result = asyncio.run(loop.arun_once("go"))
    assert result.status == "completed"

    spans = exporter.get_finished_spans()
    chat = next(span for span in spans if span.name.startswith("chat"))
    children = _attempt_children(spans)
    assert chat.attributes["monoid.model.attempts"] == 2
    assert [child.name for child in children] == ["model.attempt 1", "model.attempt 2"]
    assert all(child.parent.span_id == chat.context.span_id for child in children)
    assert children[0].status.status_code == StatusCode.ERROR
    assert children[0].attributes["monoid.model.attempt.retryable"] is True
    assert children[1].status.status_code == StatusCode.UNSET
