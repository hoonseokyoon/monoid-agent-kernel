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
Content is metadata-only by default. Register the sink's model-I/O subscription to opt into
policy-gated digests, redaction, or full capture on the same chat span.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, Literal

from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.model_io import (
    CapturePolicy,
    ModelCallCapture,
    ModelIOSubscription,
)

if TYPE_CHECKING:
    from opentelemetry.context import Context as OtelContext
else:
    OtelContext = Any

OtelSpanMode = Literal["agent", "model_call"]
_OTEL_SPAN_MODES = frozenset(("agent", "model_call"))


class OtelEventSink:
    """OpenTelemetry preset for agent runs and standalone model calls.

    ``agent`` mode preserves the event-derived ``invoke_agent`` tree. Its model-I/O observer facet
    enriches the already-open chat span, so enabling capture never duplicates spans. ``model_call``
    mode is for a standalone :class:`~monoid_agent_kernel.model_call.ModelCallRunner`; each settled
    capture owns one chat span and the event facet is inactive.

    Model content remains off unless the caller both supplies an explicit ``capture_policy`` and
    registers :meth:`model_io_subscription`. The default is deliberately ``none`` rather than
    :class:`CapturePolicy`'s general ``full`` default, preserving the sink's historical
    metadata-only contract.
    """

    def __init__(
        self,
        *,
        tracer_name: str = "monoid_agent_kernel",
        tracer_provider: Any = None,
        parent_context: InvocationContext | OtelContext | None = None,
        span_mode: OtelSpanMode = "agent",
        capture_policy: CapturePolicy | None = None,
    ) -> None:
        if span_mode not in _OTEL_SPAN_MODES:
            raise ValueError("OTel span_mode must be 'agent' or 'model_call'")
        try:
            from opentelemetry import trace
            from opentelemetry.context import Context
            from opentelemetry.trace import SpanKind
            from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
            from opentelemetry.trace.status import Status, StatusCode
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "OtelEventSink requires opentelemetry; install monoid-agent-kernel[otel]"
            ) from exc
        self._trace = trace
        self._SpanKind = SpanKind
        self._Status = Status
        self._StatusCode = StatusCode
        self._Context = Context
        self._propagator = TraceContextTextMapPropagator()
        # tracer_provider=None uses the globally-configured provider (a no-op until the app
        # installs an SDK + exporter); an explicit provider is handy for tests and embedding.
        self._tracer = trace.get_tracer(tracer_name, tracer_provider=tracer_provider)
        self._span_mode = span_mode
        self._capture_policy = capture_policy or CapturePolicy(mode="none")
        self._parent_context = self._resolve_parent_context(parent_context)
        self._run_span: Any = None
        self._run_span_started = False
        self._run_id = ""
        self._model: str | None = None
        self._provider: str | None = None
        # event_id -> live span, for the started/finished pairs (chat, execute_tool). A finish
        # event's parent_id equals its start event's event_id, so close by popping parent_id.
        self._spans: dict[str, Any] = {}
        self._model_span_ids: dict[str, str] = {}
        self._authoritative_finish_span_ids: set[str] = set()
        # Chat spans whose attempt children have been synthesized, keyed like the span map.
        # ``on_model_call`` is public and may be called twice for one call; the enrich is
        # idempotent on attributes, and this is what keeps the children from multiplying.
        self._attempt_synthesized_span_ids: set[str] = set()
        self._pending_span_ends: dict[int, tuple[Any, int | None]] = {}
        self._closed = False

    @property
    def span_mode(self) -> OtelSpanMode:
        return self._span_mode

    @property
    def capture_policy(self) -> CapturePolicy:
        return self._capture_policy

    def model_io_subscription(self) -> ModelIOSubscription:
        """Return the policy-gated observer facet paired with this sink's span state."""

        return ModelIOSubscription(observer=self, policy=self._capture_policy)

    def emit(self, event: AgentEvent) -> None:
        """Consume one public event without allowing telemetry failures to fail the run."""

        if self._closed or self._span_mode != "agent":
            return
        try:
            self._emit(event)
        except Exception:
            # An exporter or third-party OTel implementation is diagnostic infrastructure. The
            # EventBus does not isolate sink failures, so the preset must uphold the same
            # failure-containment rule as the model-I/O observer pipeline itself.
            return

    def _emit(self, event: AgentEvent) -> None:
        kind = event.type
        if kind == "run.started":
            self._model = event.data.get("model")
            self._provider = event.data.get("model_provider")
            self._ensure_run_span(event)
        elif kind in ("run.finished", "run.failed"):
            # A restored activation may terminate before emitting any child event, and restore does
            # not replay ``run.started``. Preserve that activation as a (possibly zero-duration)
            # run span rather than dropping it from the trace.
            self._ensure_run_span(event)
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
            if event.turn_id:
                self._model_span_ids[event.turn_id] = event.event_id
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
        elif kind == "turn.failed":
            self._close_latest_model_span(
                error=True,
                error_type=event.data.get("provider_error_code")
                or event.data.get("error_code")
                or "error",
            )
        elif kind == "turn.interrupted":
            self._close_latest_model_span()

    def on_model_call(self, capture: ModelCallCapture) -> None:
        """Enrich the active agent chat span or emit one standalone model-call span."""

        if self._closed:
            return
        try:
            if self._span_mode == "model_call":
                self._emit_model_call_span(capture)
            else:
                self._enrich_agent_model_span(capture)
        except Exception:
            # ``dispatch_model_call`` also isolates observers, but this method is public and may be
            # called directly. Observability must remain unable to alter the model-call outcome.
            return

    def close(self) -> None:
        self._closed = True
        # Retry ends that a prior ownership path could not complete. A processor may raise after
        # the SDK has already marked the span ended; ``_end_span`` detects that and does not retry.
        pending = tuple(self._pending_span_ends.values())
        self._pending_span_ends.clear()
        for span, end_time in pending:
            self._end_span(span, end_time=end_time)
        # Leak guard: make a best effort for every span even when one exporter is broken. Nothing
        # from telemetry teardown may escape EventBus.close and change run lifecycle.
        for span in tuple(self._spans.values()):
            self._end_span(span)
        self._spans.clear()
        self._model_span_ids.clear()
        self._authoritative_finish_span_ids.clear()
        self._attempt_synthesized_span_ids.clear()
        if self._run_span is not None:
            self._end_span(self._run_span)
            self._run_span = None

    def _resolve_parent_context(
        self, parent_context: InvocationContext | OtelContext | None
    ) -> Any:
        if not isinstance(parent_context, InvocationContext):
            return parent_context
        try:
            carrier = {"traceparent": parent_context.traceparent}
            if parent_context.tracestate:
                carrier["tracestate"] = parent_context.tracestate
            extracted = self._propagator.extract(carrier=carrier, context=self._Context())
            if self._trace.get_current_span(extracted).get_span_context().is_valid:
                return extracted
        except Exception:
            pass
        return None

    def _start_span(self, name: str, *, context: Any = None, **kwargs: Any) -> Any:
        # Omitting ``context`` preserves the OTel API's ambient-parent behavior. Passing None is
        # equivalent in current releases but omission keeps the compatibility contract explicit.
        if context is None:
            return self._tracer.start_span(name, **kwargs)
        return self._tracer.start_span(name, context=context, **kwargs)

    def _ensure_run_span(self, event: AgentEvent) -> Any:
        if self._run_span is not None:
            return self._run_span
        if self._run_span_started:
            return None
        self._run_id = event.run_id
        span = self._start_span(
            "invoke_agent",
            context=self._parent_context,
            kind=self._SpanKind.INTERNAL,
            attributes=_clean({"gen_ai.operation.name": "invoke_agent", "run_id": event.run_id}),
        )
        self._run_span = span
        self._run_span_started = True
        return self._run_span

    def _open_child(
        self,
        event: AgentEvent,
        *,
        name: str,
        kind: Any,
        attrs: dict[str, Any],
        parent_event_id: str | None = None,
    ) -> None:
        self._ensure_run_span(event)
        # Default parent is the run span (siblings), reconstructed explicitly so async/thread
        # hops never matter — the ambient current-span is never read. ``parent_event_id`` nests
        # under another still-open child span (e.g. a subagent under its spawn tool span).
        anchor = self._spans.get(parent_event_id or "") or self._run_span
        context = self._trace.set_span_in_context(anchor) if anchor is not None else None
        self._spans[event.event_id] = self._start_span(
            name, context=context, kind=kind, attributes=_clean(attrs)
        )

    def _enrich_agent_model_span(self, capture: ModelCallCapture) -> None:
        receipt = capture.receipt
        step_id = receipt.context.step_id
        candidates = (step_id, step_id.rsplit("/", 1)[-1]) if step_id else ()
        event_id = next(
            (
                self._model_span_ids[candidate]
                for candidate in candidates
                if candidate in self._model_span_ids
            ),
            None,
        )
        span = self._spans.get(event_id or "")
        if span is None:
            return
        model = receipt.model.model
        if model:
            span.update_name(f"chat {model}")
        # An aborted model call is followed by ``turn.interrupted`` and is not an error. Every other
        # failed receipt is authoritative even when the run skips ``turn.failed`` and goes directly
        # to ``run.failed``; lifecycle events still own when the open span ends.
        self._apply_capture(
            span,
            capture,
            mark_error=(receipt.error_code != "model_call_aborted"),
        )
        if receipt.stop_reason:
            self._authoritative_finish_span_ids.add(event_id or "")
        # Marked before emitting rather than after: a processor that raises mid-synthesis is
        # contained by ``on_model_call``, and a redelivery must not append a second set of
        # children next to a partial first.
        if event_id and event_id not in self._attempt_synthesized_span_ids:
            self._attempt_synthesized_span_ids.add(event_id)
            self._emit_attempt_spans(span, receipt, time.time_ns())

    def _emit_model_call_span(self, capture: ModelCallCapture) -> None:
        receipt = capture.receipt
        context = self._parent_context
        if context is None:
            context = self._resolve_parent_context(receipt.context)
        model = receipt.model.model
        end_time = time.time_ns()
        start_time = max(0, end_time - receipt.latency_ms * 1_000_000)
        span = self._start_span(
            f"chat {model}" if model else "chat",
            context=context,
            kind=self._SpanKind.CLIENT,
            attributes=_clean(
                {
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": receipt.provider_name or receipt.model.provider,
                    "gen_ai.request.model": model,
                }
            ),
            start_time=start_time,
        )
        try:
            self._apply_capture(
                span,
                capture,
                mark_error=(capture.receipt.error_code != "model_call_aborted"),
            )
            self._emit_attempt_spans(span, receipt, end_time)
        finally:
            # Fix the span to the receipt's settled instant. Capture serialization/exporter work is
            # observer overhead, not model latency, and must not stretch this span.
            self._end_span(span, end_time=end_time)

    def _apply_capture(
        self, span: Any, capture: ModelCallCapture, *, mark_error: bool = True
    ) -> None:
        if not span.is_recording():
            return
        receipt = capture.receipt
        attrs: dict[str, Any] = {
            "gen_ai.provider.name": receipt.provider_name or receipt.model.provider,
            "gen_ai.request.model": receipt.model.model,
            "monoid.model.capture.mode": capture.mode,
            "monoid.model.latency_ms": receipt.latency_ms,
            "monoid.model.attempts": receipt.attempts,
            "monoid.model.provider_retried": receipt.provider_retried,
            "monoid.invocation.run_id": receipt.context.run_id,
            "monoid.invocation.skill_id": receipt.context.skill_id,
            "monoid.invocation.step_id": receipt.context.step_id,
            "monoid.invocation.attempt": receipt.context.attempt,
        }
        if capture.downgraded_from:
            attrs["monoid.model.capture.downgraded_from"] = capture.downgraded_from
        if receipt.redaction_digest:
            attrs["monoid.model.capture.redaction_digest"] = receipt.redaction_digest
        if receipt.capture_downgrades:
            attrs["monoid.model.capture.downgrades"] = receipt.capture_downgrades
        if receipt.stop_reason:
            attrs["gen_ai.response.finish_reasons"] = (receipt.stop_reason,)
        usage = receipt.usage
        if usage.get("input_tokens") is not None:
            attrs["gen_ai.usage.input_tokens"] = int(usage["input_tokens"])
        if usage.get("output_tokens") is not None:
            attrs["gen_ai.usage.output_tokens"] = int(usage["output_tokens"])
        if capture.digests:
            attrs["monoid.model.capture.digests"] = _json_attribute(capture.digests)
        if capture.lengths:
            attrs["monoid.model.capture.lengths"] = _json_attribute(capture.lengths)
        if capture.content is not None:
            attrs["monoid.model.capture.content"] = _json_attribute(capture.content)
        if receipt.http_status is not None:
            attrs["monoid.model.http_status"] = receipt.http_status
        if not receipt.succeeded:
            attrs["monoid.model.retryable"] = receipt.retryable
            if mark_error:
                attrs["error.type"] = receipt.provider_error_code or receipt.error_code or "error"
                span.set_status(self._Status(self._StatusCode.ERROR))
        for key, value in _clean(attrs).items():
            span.set_attribute(key, value)

    def _emit_attempt_spans(self, parent_span: Any, receipt: Any, anchor_ns: int) -> None:
        """Synthesize one INTERNAL child per logged dispatch, placed backward from settle.

        Only when the kernel dispatched more than once: a single-attempt call's chat span IS
        that attempt, and a child would restate it at double the span volume of every
        subscribed call. Children carry ``monoid.model.attempt.*`` and never ``gen_ai.*`` —
        a GenAI-aware backend aggregating usage or operation counts over those spans would
        double-count the parent otherwise — and never capture content: the log is metadata by
        construction, the entry's own rule.

        Placement walks backward from the anchor — the capture-processing instant, the same
        instant the standalone span pins as its end — so each entry spans its ``elapsed_ms``
        preceded by its recorded ``backoff_ms`` gap. An entry parsed from a line that predates
        the field (``backoff_ms is None``) packs edge to edge instead: durations and order stay
        exact, the unknown gaps collapse. Wall-clock skew against the monotonic durations is
        the standalone span's stated limitation, unchanged here. The failed-dispatch error rule
        is the parent's, held per entry: ``model_call_aborted`` is an interruption, not an
        error.
        """

        log = getattr(receipt, "attempt_log", ()) or ()
        if len(log) < 2 or not parent_span.is_recording():
            return
        context = self._trace.set_span_in_context(parent_span)
        cursor = anchor_ns
        for entry in reversed(log):
            end_time = cursor
            start_time = max(0, end_time - entry.elapsed_ms * 1_000_000)
            attrs: dict[str, Any] = {
                "monoid.model.attempt.index": entry.index,
                "monoid.model.attempt.elapsed_ms": entry.elapsed_ms,
                "monoid.model.attempt.backoff_ms": entry.backoff_ms,
                "monoid.model.attempt.retryable": entry.retryable,
                "monoid.model.attempt.config_recoverable": entry.config_recoverable,
                "monoid.model.attempt.http_status": entry.http_status,
                "monoid.model.attempt.provider_retried": entry.provider_retried,
                "monoid.model.attempt.stream_committed": entry.stream_committed,
            }
            if entry.error_code:
                attrs["monoid.model.attempt.error_code"] = entry.error_code
            if entry.provider_error_code:
                attrs["monoid.model.attempt.provider_error_code"] = entry.provider_error_code
            if entry.usage:
                attrs["monoid.model.attempt.usage"] = _json_attribute(dict(entry.usage))
            span = self._start_span(
                f"model.attempt {entry.index}",
                context=context,
                kind=self._SpanKind.INTERNAL,
                attributes=_clean(attrs),
                start_time=start_time,
            )
            if entry.error_code not in ("", "model_call_aborted") and span.is_recording():
                span.set_attribute("error.type", entry.provider_error_code or entry.error_code)
                span.set_status(self._Status(self._StatusCode.ERROR))
            self._end_span(span, end_time=end_time)
            cursor = max(0, start_time - (entry.backoff_ms or 0) * 1_000_000)

    def _end_model_span(
        self, event_id: str, *, error: bool = False, error_type: str = "error"
    ) -> None:
        span = self._spans.pop(event_id, None)
        if span is None:
            return
        if error and span.is_recording():
            span.set_attribute("error.type", error_type)
            span.set_status(self._Status(self._StatusCode.ERROR))
        self._end_span(span)
        for turn_id, mapped_event_id in tuple(self._model_span_ids.items()):
            if mapped_event_id == event_id:
                self._model_span_ids.pop(turn_id, None)
        self._authoritative_finish_span_ids.discard(event_id)
        self._attempt_synthesized_span_ids.discard(event_id)

    def _close_latest_model_span(self, *, error: bool = False, error_type: str = "error") -> None:
        if not self._model_span_ids:
            return
        event_id = next(reversed(self._model_span_ids.values()))
        self._end_model_span(event_id, error=error, error_type=error_type)

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

    def _close_child(
        self, event: AgentEvent, *, finish: dict[str, Any] | None = None, error: bool = False
    ) -> None:
        span = self._spans.pop(event.parent_id or "", None)
        if span is None:
            return
        for turn_id, event_id in tuple(self._model_span_ids.items()):
            if event_id == event.parent_id:
                self._model_span_ids.pop(turn_id, None)
        finish_attrs = dict(finish or {})
        if event.parent_id in self._authoritative_finish_span_ids:
            finish_attrs.pop("gen_ai.response.finish_reasons", None)
        self._authoritative_finish_span_ids.discard(event.parent_id or "")
        self._attempt_synthesized_span_ids.discard(event.parent_id or "")
        if span.is_recording():
            for key, value in finish_attrs.items():
                if value is not None:
                    span.set_attribute(key, value)
            if error:
                span.set_attribute("error.type", event.data.get("error_code") or "error")
                span.set_status(self._Status(self._StatusCode.ERROR))
        self._end_span(span)

    def _finish_run(self, event: AgentEvent) -> None:
        span = self._run_span
        if span is None:
            return
        if span.is_recording():
            failed = event.type == "run.failed" or event.data.get("status") == "failed"
            if failed:
                span.set_attribute("error.type", event.data.get("error_code") or "error")
                span.set_status(self._Status(self._StatusCode.ERROR))
        # Close any dangling child spans before the parent.
        for child in tuple(self._spans.values()):
            self._end_span(child)
        self._spans.clear()
        self._model_span_ids.clear()
        self._authoritative_finish_span_ids.clear()
        self._attempt_synthesized_span_ids.clear()
        self._end_span(span)
        self._run_span = None

    def _end_span(self, span: Any, *, end_time: int | None = None) -> bool:
        """End one span without exporting failures into run control flow.

        A raising processor can fail before or after the SDK marks the span ended. Only a span that
        still reports itself recording is retained for the next ownership close to retry.
        """

        try:
            if end_time is None:
                span.end()
            else:
                span.end(end_time=end_time)
        except Exception:
            try:
                if not span.is_recording():
                    self._pending_span_ends.pop(id(span), None)
                    return True
            except Exception:
                pass
            self._pending_span_ends[id(span)] = (span, end_time)
            return False
        self._pending_span_ends.pop(id(span), None)
        return True


def _clean(attrs: dict[str, Any]) -> dict[str, Any]:
    """Drop None-valued attributes (OTel rejects them) and keep the rest."""
    return {key: value for key, value in attrs.items() if value is not None}


def _json_attribute(value: Any) -> str:
    """Serialize structured capture data into the scalar shape OTel attributes accept."""

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


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
