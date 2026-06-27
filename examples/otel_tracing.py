"""Trace a run as OpenTelemetry spans with OtelEventSink.

Runs offline with a scripted FakeModelAdapter (no gateway, no API key) and prints the GenAI
span tree to the console via a local ConsoleSpanExporter, so you can see the shape without a
collector:

    invoke_agent
    ├── chat {model}          (one per model turn)
    └── execute_tool {tool}   (one per tool call)

Needs the SDK + exporter for *output*; install with: pip install 'native-agent-runner[otel-export]'.
In a real app you'd install a global TracerProvider with an OTLP exporter instead and just pass
``event_sinks=(OtelEventSink(),)`` — here we build a local provider and hand it to the sink so the
example never touches global process state.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from native_agent_runner import (
    AgentLoop,
    AgentRunSpec,
    AgentRuntimeConfig,
    FakeModelAdapter,
    OtelEventSink,
    RegistryToolRef,
    ToolBinding,
    tool_ids,
)
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import fake_tool_call


def _console_tracer_provider():
    """A local TracerProvider that prints spans to stdout — no collector, no global state."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "otel-tracing-example"}))
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    return provider


def main() -> None:
    provider = _console_tracer_provider()
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "workspace"
        workspace.mkdir()
        (workspace / "notes.md").write_text("alpha beta gamma\n", encoding="utf-8")

        spec = AgentRunSpec(workspace_root=workspace, run_root=Path(tmp) / "runs", mode="apply")
        config = AgentRuntimeConfig(
            definition_id="otel-tracing",
            tools=(
                ToolBinding(binding_id="fs.write", ref=RegistryToolRef(tool_ids.FS_WRITE)),
                ToolBinding(binding_id="run.finish", ref=RegistryToolRef(tool_ids.RUN_FINISH)),
            ),
        )
        adapter = FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="t1",
                    tool_calls=(
                        fake_tool_call(
                            "fs_write", {"path": "SUMMARY.md", "content": "# Summary\n3 words.\n"}, "c1"
                        ),
                    ),
                ),
                ModelTurn(response_id="t2", final_text="Wrote SUMMARY.md."),
            ]
        )

        # Pass the local provider to the sink; an app with a global OTLP provider just uses
        # OtelEventSink() with no argument.
        result = AgentLoop.from_config(
            spec, adapter, config, event_sinks=(OtelEventSink(tracer_provider=provider),)
        ).run_once("Summarize notes.md")

    provider.force_flush()  # ensure the spans above are flushed to the console
    print("status     :", result.status)
    print("final_text :", result.final_text)


if __name__ == "__main__":
    main()
