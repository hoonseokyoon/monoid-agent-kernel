"""Map the kernel's event tree to OpenTelemetry GenAI spans (opt-in ``[otel]`` extra).

``OtelEventSink`` is an :class:`~monoid_agent_kernel.core.events.EventSink` — the same seam
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

from monoid_agent_kernel.core.events import AgentEvent


class OtelEventSink:
    """An ``EventSink`` that emits OpenTelemetry GenAI spans from the run's event stream."""

    def __init__(self, *, tracer_name: str = "monoid_agent_kernel", tracer_provider: Any = None) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.trace import SpanKind
            from opentelemetry.trace.status import Status, StatusCode
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "OtelEventSink requires opentelemetry; install monoid-agent-kernel[otel]"
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
        elif kind == "subagent.started":
            sub = event.data.get("subagent_type")
            self._open_child(
                event,
                name="execute_subagent " + sub if sub else "execute_subagent",
                kind=self._SpanKind.INTERNAL,
                attrs={
                    "gen_ai.operation.name": "execute_subagent",
                    "subagent.type": sub,
                    "subagent.run_id": event.data.get("child_run_id"),
                    "subagent.background": event.data.get("background"),
                    "turn_id": event.turn_id,
                },
                # Nest under the spawn tool span when it is still open (foreground); a
                # background spawn's tool span has already closed, so fall back to run.
                parent_event_id=event.parent_id,
            )
        elif kind in ("subagent.finished", "subagent.failed"):
            self._close_child(
                event,
                finish=_subagent_finish_attrs(event.data),
                error=(kind == "subagent.failed" or event.data.get("status") == "failed"),
            )
        elif kind == "skill.activated":
            # A point-in-time event (no started/finished pair): enrich the still-open
            # ``execute_tool`` span of the skill tool call (its event_id is this event's
            # parent_id) rather than opening an orphan span.
            self._enrich(
                event.parent_id,
                {
                    "skill.name": event.data.get("name"),
                    "skill.resource_count": event.data.get("resource_count"),
                },
            )
        elif kind == "output.validation.failed":
            # Output validation runs at settle, AFTER model.turn.finished closes the turn span, so
            # the failure is recorded as an event on the (still-open) run span rather than the turn.
            self._run_span_event(
                "output.validation.failed",
                {
                    "output.validation.attempt": event.data.get("attempt"),
                    "output.validation.reason": event.data.get("reason"),
                },
            )
        elif kind == "output.validator.error":
            self._run_span_event(
                "output.validator.error",
                {"output.validator.id": event.data.get("validator_id")},
            )
        elif kind == "output.validator.exhausted":
            self._run_span_event(
                "output.validator.exhausted",
                {"output.validation.retries": event.data.get("retries")},
            )

    def close(self) -> None:
        # Leak guard: end any spans still open (abnormal termination, missing finish events).
        for span in self._spans.values():
            span.end()
        self._spans.clear()
        if self._run_span is not None:
            self._run_span.end()
            self._run_span = None

    def _open_child(
        self,
        event: AgentEvent,
        *,
        name: str,
        kind: Any,
        attrs: dict[str, Any],
        parent_event_id: str | None = None,
    ) -> None:
        # Default parent is the run span (siblings), reconstructed explicitly so async/thread
        # hops never matter — the ambient current-span is never read. ``parent_event_id`` nests
        # under another still-open child span (e.g. a subagent under its spawn tool span).
        anchor = self._spans.get(parent_event_id or "") or self._run_span
        context = self._trace.set_span_in_context(anchor) if anchor is not None else None
        self._spans[event.event_id] = self._tracer.start_span(
            name, context=context, kind=kind, attributes=_clean(attrs)
        )

    def _run_span_event(self, name: str, attrs: dict[str, Any]) -> None:
        """Add a timestamped event (with attributes) to the still-open run span. Used for
        run-level point-in-time signals — like output validation — that fire after the relevant
        child span has already closed. No-op if the run span isn't recording."""
        span = self._run_span
        if span is not None and span.is_recording():
            span.add_event(name, attributes=_clean(attrs))

    def _enrich(self, span_event_id: str | None, attrs: dict[str, Any]) -> None:
        """Set attributes on a still-open child span (keyed by the event_id that opened it).
        No-op if that span is not open. Used for point-in-time events that annotate an
        existing span rather than opening their own."""
        span = self._spans.get(span_event_id or "")
        if span is None or not span.is_recording():
            return
        for key, value in attrs.items():
            if value is not None:
                span.set_attribute(key, value)

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


def _subagent_finish_attrs(data: dict[str, Any]) -> dict[str, Any]:
    """GenAI attributes set when an execute_subagent span ends: the child's token usage
    (so a parent trace shows delegated cost) and its terminal status."""
    usage = data.get("usage") or {}
    attrs: dict[str, Any] = {}
    if usage.get("input_tokens") is not None:
        attrs["gen_ai.usage.input_tokens"] = int(usage["input_tokens"])
    if usage.get("output_tokens") is not None:
        attrs["gen_ai.usage.output_tokens"] = int(usage["output_tokens"])
    if data.get("status"):
        attrs["subagent.status"] = data["status"]
    return attrs
