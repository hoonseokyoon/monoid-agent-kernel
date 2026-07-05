from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core.event_sequencing import RunEventSequencer
from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.lifecycle import (
    TERMINAL_STATES,
    SessionState,
    session_state_from_run_status,
    session_state_value,
)
from monoid_agent_kernel.core.result import AgentRunResult
from monoid_agent_kernel.reference.backend.ports import LoopPort, MutableRunRecordPort, RunRecordPort


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
    tenant_id: str
    runs: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    web_search_calls: int = 0
    web_fetch_calls: int = 0
    web_context_calls: int = 0
    web_failed_calls: int = 0
    web_result_count: int = 0
    web_bytes_returned: int = 0
    web_context_source_count: int = 0
    web_context_bytes_returned: int = 0

    def add_metrics(self, metrics: dict[str, Any]) -> None:
        self.runs += 1
        self.input_tokens += int(metrics.get("input_tokens") or 0)
        self.output_tokens += int(metrics.get("output_tokens") or 0)
        self.total_tokens += int(metrics.get("total_tokens") or 0)
        self.web_search_calls += int(metrics.get("web_search_calls") or 0)
        self.web_fetch_calls += int(metrics.get("web_fetch_calls") or 0)
        self.web_context_calls += int(metrics.get("web_context_calls") or 0)
        self.web_failed_calls += int(metrics.get("web_failed_calls") or 0)
        self.web_result_count += int(metrics.get("web_result_count") or 0)
        self.web_bytes_returned += int(metrics.get("web_bytes_returned") or 0)
        self.web_context_source_count += int(metrics.get("web_context_source_count") or 0)
        self.web_context_bytes_returned += int(metrics.get("web_context_bytes_returned") or 0)

    def to_json(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "runs": self.runs,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
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
    event_sequencer: RunEventSequencer = field(default_factory=RunEventSequencer)
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("monoid_agent_kernel.backend")
    )


class RunStateMutationService:
    """Live run state, backend event append, terminal result/failure, and usage ledger."""

    def __init__(self, context: RunStateMutationContext) -> None:
        self._context = context
        self._usage: dict[str, TenantUsage] = {}

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
            run_dir = record.run_dir if record is not None else self._context.run_root_provider() / run_id
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
            if loop is not None and loop.emit_external_event(event_type, data=dict(data), level=level):
                return
            if not direct_append_allowed and requires_live_owner:
                return
        if not run_dir.exists():
            return
        if not direct_append_allowed and not self._context.event_sequencer.run_dir_allows_direct_append(run_dir):
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
                session_state_from_run_status(result.status, error_code=result.error_code, terminal=True),
                terminal=True,
            )
            record.error = result.error
            record.error_code = result.error_code
            record.finished_at = self._context.now()
            self._usage.setdefault(record.tenant_id, TenantUsage(record.tenant_id)).add_metrics(
                result.metrics
            )

        self._context.with_record_lock(_mutate)

    def record_run_failure(self, run_id: str, exc: Exception) -> None:
        self._context.write_failure_bundle(
            run_id,
            self._context.run_root_provider() / run_id,
            error=str(exc),
            error_code=getattr(exc, "error_code", "internal_error"),
            exc_type=type(exc).__name__,
            overwrite=False,
        )

        def _mutate() -> None:
            record = self._context.record(run_id)
            set_record_state(record, SessionState.FAILED, terminal=True)
            record.error = str(exc)
            record.error_code = getattr(exc, "error_code", "internal_error")
            record.finished_at = self._context.now()

        self._context.with_record_lock(_mutate)
