from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core._util import read_text_resilient, utc_timestamp, write_json_atomic
from monoid_agent_kernel.core.checkpoint import CheckpointStore
from monoid_agent_kernel.core.event_sequencing import RunEventSequencer
from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.json_ingress import loads_json_ingress, portable_type_name
from monoid_agent_kernel.core.interruption import parse_interruption_cause
from monoid_agent_kernel.core.lifecycle import (
    TERMINAL_STATES,
    SessionState,
    session_state_from_run_status,
    session_state_value,
)
from monoid_agent_kernel.core.outcome import InterruptionCause
from monoid_agent_kernel.core.projections import FAILURE_QUARANTINE_MARKERS
from monoid_agent_kernel.core.result import AgentRunResult
from monoid_agent_kernel.public_view import public_error_message
from monoid_agent_kernel.reference.backend.ports import (
    LoopPort,
    MutableRunRecordPort,
    RunRecordPort,
)

_LOGGER = logging.getLogger("monoid_agent_kernel.backend")


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


def _error_text(exc: Exception, name: str) -> str:
    """The string twin of :func:`_error_flag`, guarded for the same reason: this arm answers
    for every exception a run can die of, and only a real string is a provider code."""

    value = getattr(exc, name, None)
    return value if type(value) is str else ""


def write_failure_status_artifact(
    run_dir: Path,
    run_id: str,
    *,
    error: str,
    error_code: str,
    exc_type: str,
    marker: str,
    provider_error_code: str = "",
    http_status: int | None = None,
    retryable: bool = False,
    config_recoverable: bool = False,
) -> None:
    """Write the terminal status artifact beside a failure quarantine — the ONE shared writer.

    A failure quarantine ends a run without a live recorder: ``failure.json`` makes every
    later recovery pass skip the dir, but ``status.json`` used to keep saying whatever the
    run last parked as, so ``status()``, ``list_runs`` and the offline projection all
    answered ``state=awaiting_input, terminal=False`` for a permanently dead run. All three
    quarantine lanes — recovery's two give-up sites and the backend's
    ``record_run_failure`` — make the same statement through this writer (the pairing is
    bound by the writer census in ``tests/test_carriage_conformance.py``): ``state="failed"``
    + ``terminal=true`` with the error pair, ``error_type``, and the four classification
    facts (the recovery lanes have no provider verdict, so theirs carry the honest empty
    defaults; ``record_run_failure`` passes what its exception carried) — and
    ``provider_retried`` dropped, exactly as the sink's ``run.failed`` branch drops it.

    ``marker`` names the lane (one of :data:`FAILURE_QUARANTINE_MARKERS`) so the two readers
    of the quarantine bit — the closed-run guard and the offline replay override — can tell
    this statement from a genuine close: while the bundle stands every resume path refuses
    the dir on failure.json anyway, and once an operator lifts the quarantine (the
    restore-hint flow) the closed-run guard must not keep refusing the resume.

    Merged over the existing payload (read with the resilient reader, so the atomic-replace
    race cannot drop it) — run identity and metrics survive. STATUS_SCHEMA's required
    watermark pair is seeded (``last_event_seq: 0`` / ``last_event_type: ""`` — "no committed
    event known to this writer") when the base payload lacks it, so the artifact minted over
    a run that never wrote status.json still validates; if the prior payload is genuinely
    unreadable the seeded minimum is what remains. Best-effort like every status.json write —
    a failed write must not mask the recorded failure."""

    if marker not in FAILURE_QUARANTINE_MARKERS:
        raise ValueError(f"unknown failure-quarantine marker: {marker!r}")
    status_path = run_dir / "status.json"
    try:
        payload = loads_json_ingress(read_text_resilient(status_path))
    except (OSError, ValueError):
        payload = None
    state: dict[str, Any] = dict(payload) if isinstance(payload, dict) else {}
    state.setdefault("run_id", run_id)
    state.setdefault("last_event_seq", 0)
    state.setdefault("last_event_type", "")
    state.update(
        {
            "state": session_state_value(SessionState.FAILED),
            "terminal": True,
            "error": error,
            "error_code": error_code,
            "error_type": exc_type,
            "provider_error_code": provider_error_code,
            "http_status": http_status,
            "retryable": retryable,
            "config_recoverable": config_recoverable,
            # The artifact's own freshness key, in the sink's own format (STATUS_SCHEMA
            # declares it a string); the failure instant lives in failure.json's
            # ``failed_at``.
            "updated_at": utc_timestamp(),
            marker: True,
        }
    )
    state.pop("provider_retried", None)
    state.pop("interruption_cause", None)
    try:
        write_json_atomic(status_path, state)
    except OSError:
        _LOGGER.debug("failure status artifact write skipped", exc_info=True)


def _event_flag(data: Mapping[str, Any], name: str) -> bool:
    """The event-data twin of :func:`_error_flag`, guarded for the same reason: this reader
    answers for whatever a durable log carries, and a truthy string must not become a claim."""

    value = data.get(name)
    return value if type(value) is bool else False


def _event_http_status(data: Mapping[str, Any]) -> int | None:
    """The event-data twin of :func:`_provider_http_status` (``bool`` is not a status)."""

    value = data.get("http_status")
    return value if type(value) is int else None


def _event_interruption_cause(data: Mapping[str, Any]) -> InterruptionCause | None:
    return parse_interruption_cause(data.get("interruption_cause"))


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
                # race it. What was missing is the classification — ALL of it: this branch
                # used to keep error/error_code and drop the five facts beside them, so a run
                # parked in TURN_FAILED served half the taxonomy the event carried. Guarded
                # reads, same rule as the exception twins above.
                record.error = str(event.data.get("error") or "")
                record.error_code = str(event.data.get("error_code") or "")
                record.provider_error_code = str(event.data.get("provider_error_code") or "")
                record.http_status = _event_http_status(event.data)
                record.retryable = _event_flag(event.data, "retryable")
                record.config_recoverable = _event_flag(event.data, "config_recoverable")
                record.provider_retried = _event_flag(event.data, "provider_retried")
            elif event.type == "turn.settled":
                record.interruption_cause = _event_interruption_cause(event.data)
            elif event.type == "turn.interrupted":
                record.interruption_cause = _event_interruption_cause(event.data)
            elif event.type in {"run.resumed", "model.turn.started"}:
                if record.state in {
                    SessionState.AWAITING_INPUT,
                    SessionState.AWAITING_TASKS,
                    # The third non-terminal park this stream can wake: without it a resumed
                    # pause read as paused through the whole resumed turn on this record.
                    SessionState.PAUSED,
                }:
                    set_record_state(record, SessionState.RUNNING, terminal=False)
                if event.type == "model.turn.started" and not record_terminal(record):
                    # The unpark clear the two status readers already bind: a model turn
                    # starting supersedes the parked failure, so the record must not serve a
                    # dead turn's error beside state="running" for the whole retried turn.
                    record.error = ""
                    record.error_code = ""
                    record.provider_error_code = ""
                    record.http_status = None
                    record.retryable = False
                    record.config_recoverable = False
                    record.provider_retried = False
                    record.interruption_cause = None
            elif event.type == "run.finished":
                # Terminal readiness is owned by record_run_result(), which flips lifecycle and
                # stores the result under the same lock.
                record.finished_at = self._context.now()
                record.error = str(event.data.get("error") or "")
                record.error_code = str(event.data.get("error_code") or "")
                record.interruption_cause = _event_interruption_cause(event.data)
            elif event.type == "run.failed":
                set_record_state(record, SessionState.FAILED, terminal=True)
                record.error = str(event.data.get("error") or "")
                record.error_code = str(event.data.get("error_code") or "")
                # The event's own classification, same guarded reads as the turn.failed twin.
                # This branch copied the error pair only, justified by the driver's park
                # promotion — but a FRESH terminal (a non-recoverable failure on the stream
                # lane, or any first-turn failure) never parks, so nothing promoted and
                # ``record_run_result``'s FAILED heal kept the record's defaults while
                # status.json carried the truth. ``provider_retried`` is not on run.failed:
                # the terminal vocabulary drops the per-call fact, so the record does too.
                record.provider_error_code = str(event.data.get("provider_error_code") or "")
                record.http_status = _event_http_status(event.data)
                record.retryable = _event_flag(event.data, "retryable")
                record.config_recoverable = _event_flag(event.data, "config_recoverable")
                record.provider_retried = False
                record.interruption_cause = None

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
            terminal_state = session_state_from_run_status(
                result.status, error_code=result.error_code, terminal=True
            )
            set_record_state(record, terminal_state, terminal=True)
            # Through the kernel's own error filter before it becomes an HTTP projection.
            # `AgentRunResult.error` is deliberately raw -- the embedding application is inside
            # the trust boundary and needs the whole message to debug -- but `record.error` is
            # served by `status`, `result` and `diagnostics`, and a gateway 400 embeds the entire
            # provider response body in it, which is not the run's own data to hand back.
            record.error = public_error_message(result.error)
            record.error_code = result.error_code
            record.interruption_cause = result.interruption_cause
            # The record-side terminal heal — the third consumer of the rule the status sink
            # and the offline projection already bind at their terminal branches. Terminals
            # minted at the CLOSE boundary (a pending-cancel promotion, the unrecovered
            # turn-failure promotion, an unsettled-close promotion) never pass the driver's
            # top-of-loop park promotion, so without this seam the live record kept the dead
            # turn's classification beside a cancelled/limited/completed terminal while
            # status.json healed it — and the two branches of the same status() endpoint
            # disagreed across a restart. Bound here, on the single seam every terminal
            # result funnels through, rather than per close path. A FAILED terminal keeps
            # the four facts (they say what the run died of) and drops only
            # ``provider_retried`` — the per-call fact the terminal vocabulary drops
            # everywhere (``run.failed``'s sink branch pops it for the same reason).
            if terminal_state is SessionState.FAILED:
                record.provider_retried = False
            else:
                record.retryable = False
                record.http_status = None
                record.config_recoverable = False
                record.provider_error_code = ""
                record.provider_retried = False
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

    def _spent_before_failure(self, run_id: str) -> dict[str, int]:
        """What a run had already spent when it died, per key, from BOTH durable sources.

        A run that dies of a driver exception produces no ``AgentRunResult``, so its usage lives
        in the last committed checkpoint — but checkpoints commit at parks while the status
        projection updates per billed event, so a run that parks, bills more turns, and then
        dies mid-turn has a status.json strictly ahead of its checkpoint. Each key rides at the
        larger of the two readings (both are cumulative totals of the same run, so max — not
        sum — is "billed once per token"), and the high-water seam in :meth:`_meter_run` keeps
        the delta semantics intact.

        Guarded per key: a value this ledger cannot read as a count (a corrupt or hand-edited
        status.json — ``{"input_tokens": 12.5}`` was observed) is dropped rather than passed
        through, because a raise here escapes the failure paths AFTER ``runs`` was counted and
        eats the streaming client's terminal frame. ``record_run_result``'s strictness for
        kernel-written result metrics is untouched. An empty mapping is the honest answer when
        neither source is readable — the run is still counted.
        """

        readings: dict[str, int] = {}

        def _fold(metrics: Any) -> None:
            if not isinstance(metrics, Mapping):
                return
            for key, value in metrics.items():
                if type(value) is int and value >= 0:
                    readings[key] = max(readings.get(key, 0), value)

        provider = self._context.checkpoint_store_provider
        store = provider() if provider is not None else None
        if store is not None:
            try:
                stored = store.latest(run_id)
            except Exception:  # pragma: no cover - a lookup failure must not mask the failure
                stored = None
            if stored is not None:
                _fold(stored.checkpoint.total_usage)
        status_path = self._context.run_root_provider() / run_id / "status.json"
        try:
            payload = loads_json_ingress(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
        if isinstance(payload, dict):
            _fold(payload.get("metrics"))
        return readings

    def meter_abandoned_run(self, run_id: str, tenant_id: str) -> None:
        """Meter what a run had spent when recovery gave it up for good — no live record.

        The give-up paths (resume failed ``max_recover_attempts`` times; corrupt durable state)
        end a run without ever re-registering it, so :meth:`record_run_failure`'s
        record-mutating half has nothing to mutate — but its metering half still owes the
        ledger: a run that crashed after N billed turns and can never be resumed was otherwise
        never counted and its checkpointed spend never reached any tenant. Same spend source,
        same high-water seam, same run count as the failure path."""

        spent = self._spent_before_failure(run_id)
        self._context.with_record_lock(lambda: self._meter_run(tenant_id, run_id, spent))

    def record_run_failure(self, run_id: str, exc: Exception) -> None:
        run_dir = self._context.run_root_provider() / run_id
        self._context.write_failure_bundle(
            run_id,
            run_dir,
            # Filtered, like `record.error` below. `diagnostics()` returns the whole `failure.json`,
            # so writing it raw put the message back on the same response the filter two lines
            # down was added to clean. The kernel's own writer (`loop.py`) already filters here.
            error=public_error_message(str(exc)),
            error_code=getattr(exc, "error_code", "internal_error"),
            exc_type=portable_type_name(exc),
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
        # The terminal statement beside the bundle, through the one shared writer — this was
        # the third failure.json writer and the only one that left status.json parked, so a
        # restart served ``awaiting_input, terminal=false`` forever for a run whose dir every
        # recovery pass skips on failure.json. The lane's own marker and the failure's own
        # error_code (not the recovery lanes' "unrecoverable"), and the classification the
        # exception itself carried — the same guarded reads the bundle above uses.
        write_failure_status_artifact(
            run_dir,
            run_id,
            error=public_error_message(str(exc)),
            error_code=getattr(exc, "error_code", "internal_error"),
            exc_type=portable_type_name(exc),
            marker="recorded_by_run_failure",
            provider_error_code=_error_text(exc, "provider_error_code"),
            http_status=_provider_http_status(exc),
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
            # The record states what the artifact above states: the EXCEPTION's own
            # classification, through the same guarded reads. The run died of this driver
            # exception, not of its last park — keeping the park's four facts here left the
            # live ``status()``/``result()`` (which prefer the active record) disagreeing
            # with the just-written status.json until the record was released. Only the
            # per-call ``provider_retried`` is dropped — the terminal vocabulary drops it
            # everywhere.
            record.retryable = _error_flag(exc, "retryable")
            record.config_recoverable = _error_flag(exc, "config_recoverable")
            record.http_status = _provider_http_status(exc)
            record.provider_error_code = _error_text(exc, "provider_error_code")
            record.provider_retried = False
            record.interruption_cause = None
            record.finished_at = self._context.now()
            # A run that dies of a driver exception after N billed turns used to leave the
            # ledger reporting zero for every one of them -- not even the run count -- while
            # ``record_run_result`` beside it fed the same ledger from the same seam.
            self._meter_run(record.tenant_id, run_id, spent)

        self._context.with_record_lock(_mutate)
