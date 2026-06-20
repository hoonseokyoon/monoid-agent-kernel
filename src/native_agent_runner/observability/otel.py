"""Map the runner's event tree to OpenTelemetry GenAI spans (opt-in ``[otel]`` extra).

``OtelEventSink`` is an :class:`~native_agent_runner.core.events.EventSink` — the same seam
``JsonlEventSink`` uses — that turns the ``run -> model.turn -> tool.call`` event tree into a
GenAI-semantic-convention span tree:

    invoke_agent                 (run.started -> run.finished/run.failed)
    ├── chat {model}             (model.turn.started -> model.turn.finished)
    └── execute_tool {tool}      (tool.call.started -> tool.call.finished/failed)

``chat`` and ``execute_tool`` are SIBLINGS under ``invoke_agent`` (not nested) — which both
matches the GenAI convention and is forced by the event order (``model.turn.finished`` fires
before the tools run, so the chat span measures only the inference). The turn↔tool link is
preserved via a ``turn_id`` attribute rather than nesting.

Zero-dep by design: depends only on ``opentelemetry-api`` (a no-op when no SDK/exporter is
configured), imported lazily so the package imports without the extra. Inject it via
``AgentLoop(..., event_sinks=(OtelEventSink(),))``.

NOTE: the GenAI semantic conventions are status "Development" and may change. Attributes here
target the ~v1.42 shape (``gen_ai.provider.name``, not the deprecated ``gen_ai.system``).
Content (prompts/responses) is intentionally NOT captured — metadata only.
"""

from __future__ import annotations

from typing import Any

from native_agent_runner.core.events import AgentEvent


class OtelEventSink:
    """An ``EventSink`` that emits OpenTelemetry GenAI spans from the run's event stream."""

    def __init__(self, *, tracer_name: str = "native_agent_runner", tracer_provider: Any = None) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.trace import SpanKind
            from opentelemetry.trace.status import Status, StatusCode
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "OtelEventSink requires opentelemetry; install native-agent-runner[otel]"
            ) from exc
        self._trace = trace
        self._SpanKind = SpanKind
        self._Status = Status
        self._StatusCode = StatusCode
        # tracer_provider=None uses the globally-configured provider (a no-op until the app
        # installs an SDK + exporter); an explicit provider is handy for tests and embedding.
        self._tracer = trace.get_tracer(tracer_name, tracer_provider=tracer_provider)
        self._run_span: Any = None
        self._model: str | None = None
        self._provider: str | None = None
        # event_id -> live span, for the started/finished pairs (chat, execute_tool). A finish
        # event's parent_id equals its start event's event_id, so close by popping parent_id.
        self._spans: dict[str, Any] = {}

    def emit(self, event: AgentEvent) -> None:
        kind = event.type
        if kind == "run.started":
            self._model = event.data.get("model")
            self._provider = event.data.get("model_provider")
            self._run_span = self._tracer.start_span(
                "invoke_agent",
                kind=self._SpanKind.INTERNAL,
                attributes=_clean({"gen_ai.operation.name": "invoke_agent", "run_id": event.run_id}),
            )
        elif kind in ("run.finished", "run.failed"):
            self._finish_run(event)
        elif kind == "model.turn.started":
            self._open_child(
                event,
                name=("chat " + self._model).strip() if self._model else "chat",
                kind=self._SpanKind.CLIENT,
                attrs={
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": self._provider,
                    "gen_ai.request.model": self._model,
                    "turn_id": event.turn_id,
                },
            )
        elif kind == "model.turn.finished":
            self._close_child(event, finish=_chat_finish_attrs(event.data))
        elif kind == "tool.call.started":
            tool = event.data.get("tool")
            self._open_child(
                event,
                name="execute_tool " + tool if tool else "execute_tool",
                kind=self._SpanKind.INTERNAL,
                attrs={
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": tool,
                    "gen_ai.tool.call.id": event.data.get("call_id"),
                    "turn_id": event.turn_id,
                },
            )
        elif kind in ("tool.call.finished", "tool.call.failed"):
            self._close_child(event, error=(kind == "tool.call.failed"))

    def close(self) -> None:
        # Leak guard: end any spans still open (abnormal termination, missing finish events).
        for span in self._spans.values():
            span.end()
        self._spans.clear()
        if self._run_span is not None:
            self._run_span.end()
            self._run_span = None

    def _open_child(self, event: AgentEvent, *, name: str, kind: Any, attrs: dict[str, Any]) -> None:
        # Parent is the run span (siblings), reconstructed explicitly so async/thread hops never
        # matter — the ambient current-span is never read.
        context = self._trace.set_span_in_context(self._run_span) if self._run_span is not None else None
        self._spans[event.event_id] = self._tracer.start_span(
            name, context=context, kind=kind, attributes=_clean(attrs)
        )

    def _close_child(self, event: AgentEvent, *, finish: dict[str, Any] | None = None, error: bool = False) -> None:
        span = self._spans.pop(event.parent_id or "", None)
        if span is None:
            return
        if span.is_recording():
            for key, value in (finish or {}).items():
                if value is not None:
                    span.set_attribute(key, value)
            if error:
                span.set_attribute("error.type", event.data.get("error_code") or "error")
                span.set_status(self._Status(self._StatusCode.ERROR, event.data.get("error") or ""))
        span.end()

    def _finish_run(self, event: AgentEvent) -> None:
        span = self._run_span
        if span is None:
            return
        if span.is_recording():
            failed = event.type == "run.failed" or event.data.get("status") == "failed"
            if failed:
                span.set_attribute("error.type", event.data.get("error_code") or "error")
                span.set_status(self._Status(self._StatusCode.ERROR, event.data.get("error") or ""))
        # Close any dangling child spans before the parent.
        for child in self._spans.values():
            child.end()
        self._spans.clear()
        span.end()
        self._run_span = None


def _clean(attrs: dict[str, Any]) -> dict[str, Any]:
    """Drop None-valued attributes (OTel rejects them) and keep the rest."""
    return {key: value for key, value in attrs.items() if value is not None}


def _chat_finish_attrs(data: dict[str, Any]) -> dict[str, Any]:
    """GenAI attributes set when a chat (model turn) span ends: token usage, response id, and
    a coarse finish reason derived from whether the turn produced tool calls or final text."""
    usage = data.get("usage") or {}
    attrs: dict[str, Any] = {}
    if usage.get("input_tokens") is not None:
        attrs["gen_ai.usage.input_tokens"] = int(usage["input_tokens"])
    if usage.get("output_tokens") is not None:
        attrs["gen_ai.usage.output_tokens"] = int(usage["output_tokens"])
    if data.get("response_id"):
        attrs["gen_ai.response.id"] = data["response_id"]
    if data.get("tool_calls"):
        attrs["gen_ai.response.finish_reasons"] = ("tool_calls",)
    elif data.get("has_final"):
        attrs["gen_ai.response.finish_reasons"] = ("stop",)
    return attrs
