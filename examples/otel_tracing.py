"""Trace a run as OpenTelemetry spans with OtelEventSink.

Runs offline with a scripted FakeModelAdapter (no gateway, no API key) and prints the GenAI
span tree to the console via a local ConsoleSpanExporter, so you can see the shape without a
collector:

    invoke_agent
    ├── chat {model}          (one per model turn)
    └── execute_tool {tool}   (one per tool call)

Needs the SDK + exporter for *output*; install with: pip install 'monoid-agent-kernel[otel-export]'.
In a real app you'd install a global TracerProvider with an OTLP exporter instead and just pass
``event_sinks=(OtelEventSink(),)`` — here we build a local provider and hand it to the sink so the
example never touches global process state.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make the example runnable from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from monoid_agent_kernel import (  # noqa: E402
    AgentLoop,
    AgentRunSpec,
    AgentRuntimeConfig,
    RegistryToolRef,
    ToolBinding,
)
from monoid_agent_kernel.observability.otel import OtelEventSink  # noqa: E402
from monoid_agent_kernel.providers.base import ModelTurn  # noqa: E402
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call  # noqa: E402
from monoid_agent_kernel.tools import tool_ids  # noqa: E402


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
