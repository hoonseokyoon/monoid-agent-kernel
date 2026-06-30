"""OTel: the OtelEventSink maps the event tree to GenAI spans. Needs opentelemetry-sdk
(InMemorySpanExporter); skipped if absent. The sink takes an explicit TracerProvider so the
test never touches the process-global provider.
"""

from __future__ import annotations

# Imports below the importorskip guard are intentionally not at top of file.
# ruff: noqa: E402

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from support.runtime import runtime_config, runtime_provider

from native_agent_runner.core.spec import AgentRunSpec, RunLimits
from native_agent_runner.loop import AgentLoop
from native_agent_runner.observability.otel import OtelEventSink
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call


def _spans_and_run(tmp_path: Path, adapter: FakeModelAdapter, *tool_ids: str):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs", limits=RunLimits(max_steps=4)),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config(*tool_ids)),
        event_sinks=(OtelEventSink(tracer_provider=provider),),
    )
    result = asyncio.run(loop.arun_once("go"))
    return exporter.get_finished_spans(), result


def _by_name(spans):
    return {s.name: s for s in spans}


def test_otel_sink_builds_genai_span_tree(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_write", {"path": "A.md", "content": "x"}, "c1"),),
                usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            ),
            ModelTurn(response_id="r2", final_text="done", usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}),
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
    assert any(tuple(s.attributes.get("gen_ai.response.finish_reasons") or ()) == ("tool_calls",) for s in chats)
    assert any(tuple(s.attributes.get("gen_ai.response.finish_reasons") or ()) == ("stop",) for s in chats)


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
    from native_agent_runner.core.output_validator import ValidationOutcome

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
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs", limits=RunLimits(max_steps=4)),
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
    from native_agent_runner.core.output_validator import ValidationOutcome

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
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="x", stop_reason="stop")])
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
