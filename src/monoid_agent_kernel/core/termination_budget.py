"""Immutable observations of process-local termination wait owners."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from monoid_agent_kernel.core import _deadline as deadline_clock

_CallPhase = Literal[
    "preparing", "waiting", "streaming", "draining", "backoff", "closing", "settling"
]
_GraceSource = Literal["wait_cancel", "stream_close", "drain_detach"]
_ExitPhase = Literal[
    "idle", "cooperative_wait", "wait_interrupted", "hard_cancel_requested", "settled"
]


@dataclass(frozen=True)
class GraceBudget:
    """One cleanup owner's selected budget and current remaining seconds.

    A finite budget with ``remaining_s=None`` is reserved and has not started.
    ``budget_s=None, remaining_s=None`` represents an unbounded cleanup wait.
    """

    source: _GraceSource
    budget_s: float | None
    remaining_s: float | None


@dataclass(frozen=True)
class CallTerminationBudget:
    """One logical model call still owned by a runner, through settlement.

    IDs are process-local observations. ``attempt_index`` counts actual kernel
    dispatches; it is None before dispatch, including provider-free recovery.
    """

    call_id: str
    phase: _CallPhase
    logical_call_id: str | None = None
    attempt_index: int | None = None
    deadline_remaining_s: float | None = None
    drain_budget_s: float | None = None
    drain_remaining_s: float | None = None
    grace_windows: tuple[GraceBudget, ...] = ()


@dataclass(frozen=True)
class StreamExitBudget:
    """RunStream's wait before requesting hard cancellation of its driver.

    A requested hard cancel keeps remaining=0 through settlement. Other settled
    or interrupted waits have remaining=None. Driver cleanup can outlast this wait.
    """

    phase: _ExitPhase = "idle"
    wait_budget_s: float | None = None
    hard_cancel_remaining_s: float | None = None


@dataclass(frozen=True)
class TerminationBudgetSnapshot:
    """A read-only snapshot measured once on this process's monotonic clock.

    ``calls`` covers this owner's model calls. Tools, child runs and host resource
    cleanup have independent lifetimes. This value is not a checkpoint format.
    """

    observed_at_monotonic_s: float
    run_deadline_remaining_s: float | None = None
    calls: tuple[CallTerminationBudget, ...] = ()
    stream_exit: StreamExitBudget | None = None


def _remaining(cutoff: float | None, now: float, bound: float | None = None) -> float | None:
    if cutoff is None:
        return None
    value = max(0.0, cutoff - now)
    if bound is not None:
        value = min(value, bound)
    return value if math.isfinite(value) else None


def _observed_grace(value: object) -> tuple[bool, float | None]:
    # Observation never invokes a user's numeric conversion or comparison. The
    # actual wait retains its own normalization, error timing and fallback.
    if type(value) not in (bool, int, float):
        return False, None
    try:
        effective = max(0.0, float(value))
    except OverflowError:
        return False, None
    return True, effective if math.isfinite(effective) else None


@dataclass
class _GraceWindow:
    source: _GraceSource
    budget: float | None
    cutoff: float | None = None


@dataclass(eq=False)
class _CallBudget:
    _registry: _CallBudgetRegistry
    call_id: str
    deadline: float | None
    logical_call_id: str | None
    phase: _CallPhase = "preparing"
    attempt_index: int | None = None
    drain_budget: float | None = None
    drain_cutoff: float | None = None
    _drain_owner_open: bool = False
    windows: dict[object, _GraceWindow] = field(default_factory=dict)

    def _active(self) -> bool:
        return self._registry._calls.get(self.call_id) is self

    def retire(self) -> None:
        with self._registry._lock:
            self._registry._calls.pop(self.call_id, None)

    def set_phase(self, phase: _CallPhase) -> None:
        with self._registry._lock:
            if self._active():
                self.phase = phase
                if phase == "preparing":
                    self.drain_budget = None
                    self.drain_cutoff = None
                    self._drain_owner_open = False

    def set_attempt(self, index: int) -> None:
        with self._registry._lock:
            if self._active():
                self.attempt_index = index

    def set_stream_budget(self, budget_s: float) -> None:
        with self._registry._lock:
            if self._active():
                self.drain_budget = budget_s
                self._drain_owner_open = budget_s > 0

    def begin_drain(self, abort_at: float, budget_s: float) -> None:
        with self._registry._lock:
            if self._active() and self._drain_owner_open and self.drain_cutoff is None:
                cutoff = abort_at + budget_s
                self.drain_budget = budget_s
                self.drain_cutoff = min(cutoff, self.deadline) if self.deadline is not None else cutoff
                if self.phase not in ("closing", "settling"):
                    self.phase = "draining"

    def end_drain(self) -> None:
        with self._registry._lock:
            if self._active():
                self.drain_cutoff = None
                # A canceller may deliver on_begin after this wait released.
                self._drain_owner_open = False

    @contextmanager
    def grace(
        self, source: _GraceSource, already_read_value: object
    ) -> Iterator[Callable[[float], None]]:
        """Reserve one owner's grace; its callback marks the actual wait start."""
        key = object()
        observed, budget = _observed_grace(already_read_value)
        with self._registry._lock:
            if self._active() and observed:
                self.windows[key] = _GraceWindow(source, budget)
        started = False

        def start(actual_timeout: float) -> None:
            nonlocal started
            observed, budget = _observed_grace(actual_timeout)
            with self._registry._lock:
                if started or not self._active():
                    return
                started = True
                if observed:
                    now = deadline_clock.monotonic()
                    self.windows[key] = _GraceWindow(
                        source, budget, None if budget is None else now + budget,
                    )
                else:
                    self.windows.pop(key, None)

        try:
            yield start
        finally:
            with self._registry._lock:
                if self._active():
                    self.windows.pop(key, None)


class _CallBudgetRegistry:
    """Plain owned state; callbacks and provider tasks stay with their existing owners."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: dict[str, _CallBudget] = {}

    def open(self, deadline: float | None, logical_call_id: str) -> _CallBudget:
        budget = _CallBudget(
            self, uuid4().hex, deadline,
            logical_call_id if type(logical_call_id) is str and logical_call_id else None,
        )
        with self._lock:
            self._calls[budget.call_id] = budget
        return budget

    def snapshot(
        self, *, now: float | None = None, run_deadline: float | None = None
    ) -> TerminationBudgetSnapshot:
        with self._lock:
            if now is None:
                now = deadline_clock.monotonic()
            return TerminationBudgetSnapshot(
                observed_at_monotonic_s=now,
                run_deadline_remaining_s=_remaining(run_deadline, now),
                calls=tuple(
                    CallTerminationBudget(
                        call_id=call.call_id,
                        phase=call.phase,
                        logical_call_id=call.logical_call_id,
                        attempt_index=call.attempt_index,
                        deadline_remaining_s=_remaining(call.deadline, now),
                        drain_budget_s=call.drain_budget,
                        drain_remaining_s=_remaining(call.drain_cutoff, now, call.drain_budget),
                        grace_windows=tuple(
                            GraceBudget(window.source, window.budget, _remaining(
                                window.cutoff, now, window.budget,
                            )) for window in call.windows.values()
                        ),
                    ) for call in self._calls.values()
                ),
            )
