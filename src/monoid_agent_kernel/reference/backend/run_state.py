from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core.checkpoint import CheckpointStore
from monoid_agent_kernel.core.event_sequencing import RunEventSequencer
from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.json_ingress import loads_json_ingress
from monoid_agent_kernel.core.lifecycle import (
    TERMINAL_STATES,
    SessionState,
    session_state_from_run_status,
    session_state_value,
)
from monoid_agent_kernel.core.result import AgentRunResult
from monoid_agent_kernel.public_view import public_error_message
from monoid_agent_kernel.reference.backend.ports import (
    LoopPort,
    MutableRunRecordPort,
    RunRecordPort,
)


def _provider_http_status(exc: Exception) -> int | None:
    """The provider HTTP status an exception carries, or ``None`` if it carries none.

    ``ModelAdapterError`` is the only exception here that has one, and this arm answers for every
    exception a run can die of -- so the read is by name rather than by isinstance, and anything
    that is not an ``int`` is no status. ``bool`` is an ``int`` subclass and is not a status.
    """

    status = getattr(exc, "http_status", None)
    return status if type(status) is int else None


def _error_flag(exc: Exception, name: str) -> bool:
    """A boolean classification an exception carries, or ``False`` if it carries none.

    The guarded twin of :func:`_provider_http_status`, and guarded for the same reason: this arm
    answers for every exception a run can die of, not only ``ModelAdapterError``. Read by name,
    and anything that is not a ``bool`` is no classification at all — a truthy string must not
    become a claim that the failure is retryable.
    """

    value = getattr(exc, name, None)
    return value if type(value) is bool else False


def _nonnegative_metric(metrics: Mapping[str, Any], key: str) -> int:
    value = metrics.get(key, 0)
    if type(value) is not int or value < 0:
        raise ValueError(f"run metric {key} must be a non-negative integer")
    return value


def set_record_state(
    record: RunRecordPort,
    state: SessionState | str,
    *,
    terminal: bool | None = None,
) -> None:
    session_state = session_state_from_run_status(state)
    record.state = session_state
    record.terminal = bool(terminal) if terminal is not None else session_state in TERMINAL_STATES


def record_terminal(record: RunRecordPort) -> bool:
    return record.terminal or record.state in TERMINAL_STATES


def record_lifecycle_payload(record: RunRecordPort) -> dict[str, Any]:
    return {
        "state": session_state_value(record.state),
        "terminal": record_terminal(record),
    }


@dataclass
class TenantUsage:
    """The backend tenant ledger — the twin of the gateway's ``LlmGatewayUsage``.

    The four priced sub-counts are summed here for the same reason they are summed there: they
    are billed differently from plain input tokens, and a ledger that folds them away
    under-reports a cache-heavy or reasoning-heavy run. Two meters, one rule; fixing one of them
    and leaving the other is exactly the shape the carriage census exists to refuse.
    """

    tenant_id: str
    runs: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0
    audio_tokens: int = 0
    web_search_calls: int = 0
    web_fetch_calls: int = 0
    web_context_calls: int = 0
    web_failed_calls: int = 0
    web_result_count: int = 0
    web_bytes_returned: int = 0
    web_context_source_count: int = 0
    web_context_bytes_returned: int = 0

    def add_metrics(self, metrics: dict[str, Any], *, count_run: bool = True) -> None:
        """Fold one run's metrics in. ``count_run=False`` for a second fold of the SAME run —
        the failure path meters what a run spent before it died, and a later recovery that
        completes must add its remainder without the ledger counting the run twice."""
        if count_run:
            self.runs += 1
        self.input_tokens += _nonnegative_metric(metrics, "input_tokens")
        self.output_tokens += _nonnegative_metric(metrics, "output_tokens")
        self.total_tokens += _nonnegative_metric(metrics, "total_tokens")
        self.cache_read_tokens += _nonnegative_metric(metrics, "cache_read_tokens")
        self.cache_creation_tokens += _nonnegative_metric(metrics, "cache_creation_tokens")
        self.reasoning_tokens += _nonnegative_metric(metrics, "reasoning_tokens")
        self.audio_tokens += _nonnegative_metric(metrics, "audio_tokens")
        self.web_search_calls += _nonnegative_metric(metrics, "web_search_calls")
        self.web_fetch_calls += _nonnegative_metric(metrics, "web_fetch_calls")
        self.web_context_calls += _nonnegative_metric(metrics, "web_context_calls")
        self.web_failed_calls += _nonnegative_metric(metrics, "web_failed_calls")
        self.web_result_count += _nonnegative_metric(metrics, "web_result_count")
        self.web_bytes_returned += _nonnegative_metric(metrics, "web_bytes_returned")
        self.web_context_source_count += _nonnegative_metric(metrics, "web_context_source_count")
        self.web_context_bytes_returned += _nonnegative_metric(
            metrics, "web_context_bytes_returned"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "runs": self.runs,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "audio_tokens": self.audio_tokens,
            "web_search_calls": self.web_search_calls,
            "web_fetch_calls": self.web_fetch_calls,
            "web_context_calls": self.web_context_calls,
            "web_failed_calls": self.web_failed_calls,
            "web_result_count": self.web_result_count,
            "web_bytes_returned": self.web_bytes_returned,
            "web_context_source_count": self.web_context_source_count,
            "web_context_bytes_returned": self.web_context_bytes_returned,
        }


class BackendRunStateSink:
    def __init__(self, emit_event: Callable[[str, AgentEvent], None], run_id: str) -> None:
        self._emit_event = emit_event
        self._run_id = run_id

    def emit(self, event: AgentEvent) -> None:
        self._emit_event(self._run_id, event)

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class RunStateMutationContext:
    with_record_lock: Callable[[Callable[[], Any]], Any]
    active_record: Callable[[str], MutableRunRecordPort | None]
    record: Callable[[str], MutableRunRecordPort]
    run_root_provider: Callable[[], Path]
    now: Callable[[], float]
    write_failure_bundle: Callable[..., None]
    append_event: Callable[..., Any]
    # How the failure path finds what the run had already spent. A run that dies of a driver
    # exception never produces an ``AgentRunResult``, so its cumulative usage lives only in the
    # last committed checkpoint (and, failing that, in the status projection on disk).
    checkpoint_store_provider: Callable[[], CheckpointStore | None] | None = None
    event_sequencer: RunEventSequencer = field(default_factory=RunEventSequencer)
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("monoid_agent_kernel.backend")
    )


class RunStateMutationService:
    """Live run state, backend event append, terminal result/failure, and usage ledger."""

    def __init__(self, context: RunStateMutationContext) -> None:
        self._context = context
        self._usage: dict[str, TenantUsage] = {}
        # Per-run high-water mark of what has already been metered, keyed by run id. Both
        # terminal paths meter cumulative totals — the failure path from the last checkpoint,
        # the result path from ``AgentRunResult.metrics`` — and a run that is metered on
        # failure, recovered, and then completes would otherwise be billed twice for every
        # token it spent before the crash. Process-local, exactly like ``_usage`` itself.
        self._metered: dict[str, dict[str, int]] = {}

    def tenant_usage(self, tenant_id: str) -> dict[str, Any]:
        def _read() -> dict[str, Any]:
            usage = self._usage.get(tenant_id) or TenantUsage(tenant_id)
            return usage.to_json()

        return self._context.with_record_lock(_read)

    def record_event(self, run_id: str, event: AgentEvent) -> None:
        def _mutate() -> None:
            record = self._context.active_record(run_id)
            if record is None:
                return
            record.last_event_seq = event.seq
            record.last_event_type = event.type
            if event.type == "run.started":
                set_record_state(record, SessionState.RUNNING, terminal=False)
                record.started_at = self._context.now()
            elif event.type == "run.awaiting_input":
                if not record_terminal(record):
                    set_record_state(record, SessionState.AWAITING_INPUT, terminal=False)
            elif event.type == "run.waiting":
                # The other park on this same stream, reachable here through the SSE path. This
                # consumer handled ``run.awaiting_input`` and not ``run.waiting`` while the
                # offline projection had exactly the mirror-image hole; both readers now see
                # both parks.
                if not record_terminal(record):
                    set_record_state(record, SessionState.AWAITING_TASKS, terminal=False)
            elif event.type == "turn.failed":
                # Classification only — the state is NOT touched. ``session_drive`` owns this
                # record's lifecycle and drives TURN_FAILED itself; writing a state here would
                # race it. What was missing is the classification: a run parked in TURN_FAILED
                # served error="" over HTTP while the event carried the whole taxonomy.
                record.error = str(event.data.get("error") or "")
                record.error_code = str(event.data.get("error_code") or "")
            elif event.type in {"run.resumed", "model.turn.started"}:
                if record.state in {SessionState.AWAITING_INPUT, SessionState.AWAITING_TASKS}:
                    set_record_state(record, SessionState.RUNNING, terminal=False)
            elif event.type == "run.finished":
                # Terminal readiness is owned by record_run_result(), which flips lifecycle and
                # stores the result under the same lock.
                record.finished_at = self._context.now()
                record.error = str(event.data.get("error") or "")
                record.error_code = str(event.data.get("error_code") or "")
            elif event.type == "run.failed":
                set_record_state(record, SessionState.FAILED, terminal=True)
                record.error = str(event.data.get("error") or "")
                record.error_code = str(event.data.get("error_code") or "")

        self._context.with_record_lock(_mutate)

    def emit_backend_event(
        self,
        run_id: str,
        event_type: str,
        data: Mapping[str, Any],
        *,
        level: str = "info",
    ) -> None:
        if any(sep in run_id for sep in ("/", "\\")) or ".." in run_id:
            return

        def _snapshot() -> tuple[MutableRunRecordPort | None, LoopPort | None, Path, bool, bool]:
            record = self._context.active_record(run_id)
            loop = record.loop if record is not None else None
            run_dir = (
                record.run_dir if record is not None else self._context.run_root_provider() / run_id
            )
            queued_direct = (
                self._context.event_sequencer.is_queued_before_recorder(record.state)
                if record is not None
                else False
            )
            requires_live_owner = (
                self._context.event_sequencer.requires_live_sequence_owner(
                    record.state,
                    terminal=record.terminal,
                )
                if record is not None
                else False
            )
            return record, loop, run_dir, queued_direct, requires_live_owner

        record, loop, run_dir, direct_append_allowed, requires_live_owner = (
            self._context.with_record_lock(_snapshot)
        )
        if record is not None:
            if loop is not None and loop.emit_external_event(
                event_type, data=dict(data), level=level
            ):
                return
            if not direct_append_allowed and requires_live_owner:
                return
        if not run_dir.exists():
            return
        if (
            not direct_append_allowed
            and not self._context.event_sequencer.run_dir_allows_direct_append(run_dir)
        ):
            return
        try:
            self._context.append_event(run_dir, event_type, data=dict(data), level=level)
        except OSError:
            self._context.logger.debug("backend event write skipped", exc_info=True)

    def record_run_result(self, run_id: str, result: AgentRunResult) -> None:
        def _mutate() -> None:
            record = self._context.record(run_id)
            record.result = result
            set_record_state(
                record,
                session_state_from_run_status(
                    result.status, error_code=result.error_code, terminal=True
                ),
                terminal=True,
            )
            # Through the kernel's own error filter before it becomes an HTTP projection.
            # `AgentRunResult.error` is deliberately raw -- the embedding application is inside
            # the trust boundary and needs the whole message to debug -- but `record.error` is
            # served by `status`, `result` and `diagnostics`, and a gateway 400 embeds the entire
            # provider response body in it, which is not the run's own data to hand back.
            record.error = public_error_message(result.error)
            record.error_code = result.error_code
            record.finished_at = self._context.now()
            self._meter_run(record.tenant_id, run_id, result.metrics)

        self._context.with_record_lock(_mutate)

    def _meter_run(self, tenant_id: str, run_id: str, metrics: Mapping[str, Any]) -> None:
        """Fold a run's CUMULATIVE metrics into the tenant ledger, counting each token once.

        The single seam both terminal paths go through. ``metrics`` is a running total, not a
        delta, so what is added is the part above this run's high-water mark; the run itself is
        counted on the first metering only. Values this ledger cannot read as counts pass
        through untouched, so ``add_metrics``'s own validation still answers for them.
        """

        first_metering = run_id not in self._metered
        mark = self._metered.setdefault(run_id, {})
        unmetered: dict[str, Any] = {}
        for key, value in metrics.items():
            if type(value) is not int or value < 0:
                unmetered[key] = value
                continue
            unmetered[key] = max(0, value - mark.get(key, 0))
            mark[key] = max(mark.get(key, 0), value)
        self._usage.setdefault(tenant_id, TenantUsage(tenant_id)).add_metrics(
            unmetered, count_run=first_metering
        )

    def _spent_before_failure(self, run_id: str) -> dict[str, Any]:
        """What a run had already spent when it died, from the most durable source available.

        A run that dies of a driver exception produces no ``AgentRunResult``, so its usage lives
        in the last committed checkpoint. The status projection on disk is the fallback (it holds
        the last ``metrics.updated`` payload), and an empty mapping is the honest answer when
        neither exists — the run is still counted, which is more than the ledger used to say.
        """

        provider = self._context.checkpoint_store_provider
        store = provider() if provider is not None else None
        if store is not None:
            try:
                stored = store.latest(run_id)
            except Exception:  # pragma: no cover - a lookup failure must not mask the failure
                stored = None
            if stored is not None:
                return dict(stored.checkpoint.total_usage)
        status_path = self._context.run_root_provider() / run_id / "status.json"
        try:
            payload = loads_json_ingress(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        metrics = payload.get("metrics") if isinstance(payload, dict) else None
        return dict(metrics) if isinstance(metrics, dict) else {}

    def record_run_failure(self, run_id: str, exc: Exception) -> None:
        self._context.write_failure_bundle(
            run_id,
            self._context.run_root_provider() / run_id,
            # Filtered, like `record.error` below. `diagnostics()` returns the whole `failure.json`,
            # so writing it raw put the message back on the same response the filter two lines
            # down was added to clean. The kernel's own writer (`loop.py`) already filters here.
            error=public_error_message(str(exc)),
            error_code=getattr(exc, "error_code", "internal_error"),
            exc_type=type(exc).__name__,
            overwrite=False,
            # The same fact the core's bundle carries, off the same exception. Read defensively
            # (this arm answers for every exception type, not only ModelAdapterError) and never
            # coerced: anything that is not an int is no status at all.
            http_status=_provider_http_status(exc),
            # ...and the classification beside it, so this bundle answers the operator's next
            # question ("resend after a config fix, or will it fail the same way?") with what the
            # exception actually said rather than with the writer's default.
            retryable=_error_flag(exc, "retryable"),
            config_recoverable=_error_flag(exc, "config_recoverable"),
        )
        # Read outside the record lock: this reaches the checkpoint store and the run directory,
        # exactly like the bundle write above it.
        spent = self._spent_before_failure(run_id)

        def _mutate() -> None:
            record = self._context.record(run_id)
            set_record_state(record, SessionState.FAILED, terminal=True)
            record.error = public_error_message(str(exc))
            record.error_code = getattr(exc, "error_code", "internal_error")
            record.finished_at = self._context.now()
            # A run that dies of a driver exception after N billed turns used to leave the
            # ledger reporting zero for every one of them -- not even the run count -- while
            # ``record_run_result`` beside it fed the same ledger from the same seam.
            self._meter_run(record.tenant_id, run_id, spent)

        self._context.with_record_lock(_mutate)
