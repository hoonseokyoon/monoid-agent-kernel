"""Live streaming surface for the async core.

``AgentLoop.astream`` is the streaming analog of ``asubmit``: instead of returning
only the settled turn, it surfaces engine events *as they happen* — model-turn and
tool lifecycle events (already emitted to the :class:`EventBus`) plus token-level
model-output deltas.

The mechanism is a tap, not a rewrite: a dormant :class:`QueueEventSink` lives on the
run's EventBus for the whole run; ``astream`` activates it for the stream's duration so
orchestration events flow into an ``asyncio.Queue`` the consumer drains. Token deltas
bypass the durable sinks entirely (no per-token ``status.json``/``events.jsonl`` writes)
and are pushed straight onto the same queue via :meth:`QueueEventSink.push_delta`.

The public handle is :class:`RunStream`, an async context manager + async iterator in
the shape of Pydantic AI's ``run_stream``::

    await loop.aopen()
    async with loop.astream("go") as stream:
        async for item in stream:          # item: AgentEvent | ModelStreamChunk
            ...
    result = stream.result                 # AgentTurnResult, or None when it parked
    await loop.aclose()
"""

from __future__ import annotations

import asyncio
import math
import threading
from dataclasses import replace
from typing import Any, Awaitable, Callable

from monoid_agent_kernel.core import _deadline as deadline_clock
from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.result import AgentTurnResult, Suspension
from monoid_agent_kernel.core.termination_budget import (
    StreamExitBudget,
    TerminationBudgetSnapshot,
    _remaining,
)
from monoid_agent_kernel.errors import NativeAgentError


class _Sentinel:
    """Single end-of-stream marker pushed onto the queue when the driver settles."""


_STREAM_END = _Sentinel()

_DEFAULT_CANCEL_GRACE_S = 8.0
"""How long an early exit waits for the cancelled drive before hard-cancelling it.

The loop adds a stopped-stream drain on top of this when one is on (`AgentLoop.astream`), read
when the wait starts."""


class QueueEventSink:
    """An :class:`~monoid_agent_kernel.core.events.EventSink` that forwards events onto
    an ``asyncio.Queue`` for a live consumer.

    Installed (dormant) on the run's EventBus at bootstrap and shared across the run, so
    multiple sequential ``astream`` calls reuse it. ``activate`` binds it to a stream's
    queue + loop; ``deactivate`` makes it inert again (late cross-thread emits are dropped,
    mirroring the EventBus ``_closed`` guard). Both ``emit`` (called by the EventBus, possibly
    from a ``to_thread`` worker) and ``push_delta`` (called by the model-call accumulator on
    the loop thread) marshal onto the captured loop with ``call_soon_threadsafe`` so the
    queue receives items in call order.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def activate(self, queue: asyncio.Queue[Any], loop: asyncio.AbstractEventLoop) -> None:
        self._queue = queue
        self._loop = loop
        self._active = True

    def deactivate(self) -> None:
        # Flip the flag only; leave queue/loop bound so a racing worker-thread emit either
        # sees active=True (pushes harmlessly) or active=False (drops) — never a None deref.
        self._active = False

    def _put(self, item: Any) -> None:
        if not self._active:
            return
        loop, queue = self._loop, self._queue
        if loop is None or queue is None:
            return
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def emit(self, event: AgentEvent) -> None:
        self._put(event)

    def push_delta(self, chunk: Any) -> None:
        self._put(chunk)

    def close(self) -> None:
        # The run's EventBus.close() calls this; the sink outlives no resources of its own.
        self.deactivate()


class RunStream:
    """Async context manager + iterator over one streamed ``astream`` turn.

    Yields ``AgentEvent`` (orchestration) interleaved with ``ModelStreamChunk`` (token
    deltas); discriminate with ``isinstance``. After the stream drains, ``result`` holds the
    settled :class:`AgentTurnResult`, or ``suspension`` holds any non-settled park such as an
    external task or user interruption. Single-consumer: a second ``async for`` raises.
    """

    def __init__(
        self,
        *,
        sink: QueueEventSink,
        drive_factory: Callable[[], Awaitable[Any]],
        request_cancel: Callable[[], None],
        cancel_grace_s: float = _DEFAULT_CANCEL_GRACE_S,
        current_extra_grace_s: Callable[[], float] | None = None,
        _termination_budget: Callable[[float], TerminationBudgetSnapshot] | None = None,
        _exit_extra_grace_s: Callable[[TerminationBudgetSnapshot], float] | None = None,
    ) -> None:
        self._sink = sink
        self._drive_factory = drive_factory
        self._request_cancel = request_cancel
        self._cancel_grace_s = cancel_grace_s
        # Read when an early exit starts its wait, not here: what the exit has to outlast (a
        # stopped stream's drain) is decided by knobs the embedder may still move after
        # ``astream()`` returns, and by the call that is in flight when the exit happens.
        self._current_extra_grace_s = current_extra_grace_s
        self._owner_budget = _termination_budget
        self._exit_extra_grace_s = _exit_extra_grace_s
        self._exit_lock = threading.Lock()
        self._exit_phase = "idle"
        self._exit_frozen = False
        self._exit_wait_value: Any = None
        self._exit_budget_s: float | None = None
        self._exit_cutoff: float | None = None
        self._hard_cancel_requested = False
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[Any] | None = None
        self._outcome: Any = None
        self._error: BaseException | None = None
        self._sentinel_pushed = False
        self._iter_started = False
        self._iter_done = False

    async def __aenter__(self) -> RunStream:
        self._loop = asyncio.get_running_loop()
        self._sink.activate(self._queue, self._loop)
        self._task = asyncio.create_task(self._run())
        self._task.add_done_callback(lambda _task: self._mark_exit_settled())
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        task = self._task
        if task is not None and not task.done():
            # Early break or an exception in the consumer body: stop cooperatively so the
            # in-flight turn reaches a boundary and emits its terminal events, with a
            # bounded hard-cancel fallback for an adapter wedged in a thread.
            self._request_cancel()
            try:
                timeout = self._start_exit_wait()
                _done, pending = await asyncio.wait({task}, timeout=timeout)
            except BaseException:
                # asyncio.wait removes its timer when this waiter exits. No
                # background owner remains to enforce a countdown after that.
                with self._exit_lock:
                    if not self._hard_cancel_requested and self._exit_phase != "settled":
                        self._exit_phase = "wait_interrupted"
                raise
            if pending and task.cancel():
                with self._exit_lock:
                    self._hard_cancel_requested = True
                    if self._exit_phase != "settled":
                        self._exit_phase = "hard_cancel_requested"
        if task is not None:
            try:
                await task
            except BaseException:  # noqa: BLE001 — outcome/error already captured in _run
                pass
        self._sink.deactivate()
        # Surface a genuine driver bug only when the block itself exited cleanly; a
        # cancellation we induced is expected and swallowed.
        if (
            exc_type is None
            and self._error is not None
            and not isinstance(self._error, asyncio.CancelledError)
        ):
            raise self._error
        return False

    def termination_budget(self) -> TerminationBudgetSnapshot:
        """Read current model waits and this stream's hard-cancel request budget.

        The query starts no timer and reads no configuration getter. A remaining
        value of zero after hard cancellation can coexist with pending cleanup.
        """
        now = deadline_clock.monotonic()
        snapshot = (
            TerminationBudgetSnapshot(now) if self._owner_budget is None
            else self._owner_budget(now)
        )
        with self._exit_lock:
            remaining = (
                0.0 if self._hard_cancel_requested else
                _remaining(self._exit_cutoff, now, self._exit_budget_s)
                if self._exit_phase == "cooperative_wait" else None
            )
            stream_exit = StreamExitBudget(
                phase=self._exit_phase,
                wait_budget_s=self._exit_budget_s,
                hard_cancel_remaining_s=remaining,
            )
        return replace(snapshot, stream_exit=stream_exit)

    def _start_exit_wait(self) -> Any:
        # Only the wait owner freezes a budget. Read callbacks outside the lock;
        # private sizing queries remain independent of this lifecycle transition.
        with self._exit_lock:
            frozen = self._exit_frozen
        if not frozen:
            now = deadline_clock.monotonic()
            value = self._exit_wait_s(_now=now)
            if self._owner_budget is None:
                # Legacy callbacks finish before their selected wait begins.
                # Managed allowances already measure time from the owner snapshot.
                now = deadline_clock.monotonic()
            budget = None
            if type(value) in (bool, int, float):
                try:
                    numeric = float(value)
                    if math.isfinite(numeric):
                        budget = max(0.0, numeric)
                except OverflowError:
                    pass
            with self._exit_lock:
                if not self._exit_frozen:
                    self._exit_frozen = True
                    self._exit_wait_value = value
                    self._exit_budget_s = budget
                    self._exit_cutoff = None if budget is None else now + budget
        with self._exit_lock:
            if not self._hard_cancel_requested and self._exit_phase != "settled":
                self._exit_phase = "cooperative_wait"
            cutoff, value = self._exit_cutoff, self._exit_wait_value
        # A direct re-entry after waiter cancellation consumes the original
        # cutoff. Legacy nonfinite/invalid wait values retain their old behavior.
        return value if cutoff is None else max(0.0, cutoff - deadline_clock.monotonic())

    def _exit_wait_s(self, *, _now: float | None = None) -> float:
        """How long an early exit waits for the cancelled drive, sized as the wait starts.

        The fixed grace plus whatever the owner says the exit may have started on top of it. An
        owner callback that raises adds nothing: the exit must still close the stream.
        """

        extra_s = 0.0
        if self._owner_budget is not None:
            try:
                snapshot = self._owner_budget(
                    deadline_clock.monotonic() if _now is None else _now
                )
                if self._exit_extra_grace_s is not None:
                    extra_s = self._exit_extra_grace_s(snapshot)
            except Exception:  # noqa: BLE001 - keep the base fallback if owner sizing fails
                extra_s = 0.0
        elif self._current_extra_grace_s is not None:
            try:
                extra_s = self._current_extra_grace_s()
            except Exception:  # noqa: BLE001 - the exit falls back to the fixed grace
                extra_s = 0.0
        return self._cancel_grace_s + extra_s if extra_s else self._cancel_grace_s

    def _mark_exit_settled(self) -> None:
        with self._exit_lock:
            self._exit_phase = "settled"

    async def _run(self) -> None:
        try:
            self._outcome = await self._drive_factory()
        except BaseException as exc:  # noqa: BLE001 — relayed to the consumer via result/__aexit__
            self._error = exc
        finally:
            self._mark_exit_settled()
            self._sink.deactivate()
            self._push_sentinel()

    def _push_sentinel(self) -> None:
        if self._sentinel_pushed:
            return
        self._sentinel_pushed = True
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._queue.put_nowait, _STREAM_END)

    def __aiter__(self) -> RunStream:
        if self._iter_started:
            raise NativeAgentError(
                "a RunStream supports a single consumer; iterate it once",
                error_code="stream_already_consumed",
            )
        self._iter_started = True
        return self

    async def __anext__(self) -> Any:
        if self._iter_done:
            raise StopAsyncIteration
        item = await self._queue.get()
        if item is _STREAM_END:
            self._iter_done = True
            raise StopAsyncIteration
        return item

    def _require_settled(self) -> None:
        if self._task is None or not self._task.done():
            raise NativeAgentError(
                "stream result is unavailable until the stream is fully consumed",
                error_code="stream_not_drained",
            )
        if self._error is not None and not isinstance(self._error, asyncio.CancelledError):
            raise self._error

    @property
    def result(self) -> AgentTurnResult | None:
        """The settled turn, or ``None`` when the run parked (see :attr:`suspension`).

        Raises if read before the stream is fully drained.
        """
        self._require_settled()
        if isinstance(self._outcome, Suspension):
            return None
        return self._outcome

    @property
    def suspension(self) -> Suspension | None:
        """The non-settled park returned by the loop, else ``None``."""
        self._require_settled()
        return self._outcome if isinstance(self._outcome, Suspension) else None
