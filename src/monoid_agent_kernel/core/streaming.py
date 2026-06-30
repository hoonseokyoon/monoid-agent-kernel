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
    result = stream.result                 # AgentTurnResult, after the stream drains
    await loop.aclose()
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.result import AgentTurnResult, Suspension
from monoid_agent_kernel.errors import NativeAgentError


class _Sentinel:
    """Single end-of-stream marker pushed onto the queue when the driver settles."""


_STREAM_END = _Sentinel()


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
    settled :class:`AgentTurnResult`, or ``suspension`` holds the park when the run stopped on
    an external hosted task. Single-consumer: a second ``async for`` raises.
    """

    def __init__(
        self,
        *,
        sink: QueueEventSink,
        drive_factory: Callable[[], Awaitable[Any]],
        request_cancel: Callable[[], None],
        cancel_grace_s: float = 8.0,
    ) -> None:
        self._sink = sink
        self._drive_factory = drive_factory
        self._request_cancel = request_cancel
        self._cancel_grace_s = cancel_grace_s
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
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        task = self._task
        if task is not None and not task.done():
            # Early break or an exception in the consumer body: stop cooperatively so the
            # in-flight turn reaches a boundary and emits its terminal events, with a
            # bounded hard-cancel fallback for an adapter wedged in a thread.
            self._request_cancel()
            _done, pending = await asyncio.wait({task}, timeout=self._cancel_grace_s)
            if pending:
                task.cancel()
        if task is not None:
            try:
                await task
            except BaseException:  # noqa: BLE001 — outcome/error already captured in _run
                pass
        self._sink.deactivate()
        # Surface a genuine driver bug only when the block itself exited cleanly; a
        # cancellation we induced is expected and swallowed.
        if exc_type is None and self._error is not None and not isinstance(self._error, asyncio.CancelledError):
            raise self._error
        return False

    async def _run(self) -> None:
        try:
            self._outcome = await self._drive_factory()
        except BaseException as exc:  # noqa: BLE001 — relayed to the consumer via result/__aexit__
            self._error = exc
        finally:
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
        """The settled turn, or ``None`` when the run parked on an external task (see
        :attr:`suspension`). Raises if read before the stream is fully drained."""
        self._require_settled()
        if isinstance(self._outcome, Suspension):
            return None
        return self._outcome

    @property
    def suspension(self) -> Suspension | None:
        """The park when the run stopped awaiting an external hosted task, else ``None``."""
        self._require_settled()
        return self._outcome if isinstance(self._outcome, Suspension) else None
