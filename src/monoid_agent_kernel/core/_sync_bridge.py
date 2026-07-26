"""Running a blocking call on a daemon thread so an async run can abandon it.

Core-internal only. The supported public surface is exported from
``monoid_agent_kernel.contracts`` and the package root.

Both halves of the kernel's synchronous surface need this: a sync ``next_turn`` on a model adapter
and a sync tool handler. It lives here rather than in ``loop`` because ``model_call`` needs it too
and ``loop`` imports ``model_call``, so leaving it in ``loop`` would close a cycle.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from collections.abc import Callable
from contextvars import copy_context
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.errors import RunCancelled, RunTimeout

_LOGGER = logging.getLogger("monoid_agent_kernel.core.sync_bridge")

_T = TypeVar("_T")


class CalleeCancelled(Exception):
    """The callee's own cancellation, told apart from cancellation delivered to the awaiter.

    ``await_abandonable_call`` raises this only for a ``CancelledError`` read off the callee's
    result. A ``CancelledError`` propagating out of that function is therefore always the awaiting
    task's own -- the host cancelling the run -- and has to keep propagating.

    The distinction needs a separate type because both arrive as the same exception class and a
    caller cannot tell them apart afterwards. Catching plain ``CancelledError`` around the call
    turned a cancelled *run* into one failed tool call and let the run continue issuing model and
    tool work the host had already stopped.
    """


def consume_task_outcome(task: asyncio.Future[Any]) -> None:
    """Retrieve a detached task outcome so late cleanup cannot emit an unhandled warning."""

    try:
        task.result()
    except BaseException:
        pass


@dataclass(frozen=True)
class AbandonableSyncCall(Generic[_T]):
    """A blocking call in flight on a daemon thread, as the two handles an awaiter needs.

    ``result`` is the cancellable waiter: cancelling it releases the awaiter, which is the whole
    point of the daemon thread. It says nothing about the worker, which cannot be interrupted and
    keeps running. ``settled`` completes only when the worker actually delivers its outcome, and is
    never cancelled -- so a caller that wants to grant the worker a bounded grace period must wait
    on ``settled``, not on ``result``. ``warn_if_unsettled`` reports the abandonment once that
    grace has expired; it is the awaiter's call to make, because only the awaiter knows the grace.
    """

    result: asyncio.Future[_T]
    settled: asyncio.Future[None]
    warn_if_unsettled: Callable[[], None]


def start_abandonable_sync_call(
    call: Callable[[], _T],
    *,
    thread_name: str,
) -> AbandonableSyncCall[_T]:
    """Run a blocking call on a daemon thread, exposed as futures on the running loop.

    Used for both halves of the sync surface -- a sync ``next_turn`` and a sync tool handler --
    so each observes run cancellation and the run deadline like its async counterpart.

    ``asyncio.to_thread`` is unusable for a call the run may have to abandon: it borrows the
    event loop's *default* executor, and ``asyncio.run`` joins every worker in that executor
    before returning. A callee that never returns would hang the caller at loop shutdown even
    though the run itself already produced its ``run_timeout`` or ``cancelled`` result -- the
    deadline would be enforced internally but unobservable from the documented async entry point.
    A daemon thread is joined by nobody, neither loop shutdown nor the interpreter's exit hooks,
    so cancelling the returned ``result`` future really does release the caller.

    The abandoned worker still runs to completion; its late outcome is dropped, because ``result``
    is already cancelled by then, and delivery is skipped outright once the loop has closed. The
    returned ``settled`` future is what an awaiter waits on to grant the worker a bounded grace
    period: cancelling ``result`` completes it instantly -- there is no coroutine to throw
    ``CancelledError`` into -- so waiting on ``result`` after cancelling it would grant no grace at
    all.

    Known limitation, in two parts. Nothing can reclaim the thread of a call that never returns, and
    the run no longer waits for it, so an implementation that wedges *permanently* accumulates one
    thread per abandoned call across runs; each abandonment is logged as a warning so that growth is
    visible rather than silent. And a thread per call gives up the bound a shared executor provided:
    ``asyncio.to_thread`` queued behind the default executor's ``max_workers``, while this starts a
    thread immediately. Within one run sync calls are sequential, so the exposure is a process
    driving many runs at once, where a burst can reach the process thread limit and fail calls that
    would otherwise succeed.

    Both bounds want admission control -- a decision about how much concurrent work a *process*
    admits, informed by run and tenant policy -- so neither belongs in this helper, which only knows
    about one call. A dedicated pool here would not settle it either: it would trade an unbounded
    thread count for a queue whose depth and eviction are the same policy question, and a wedged
    call would hold a pool slot instead of a thread. Tracked for a later release; hosts running many
    concurrent sessions with synchronous adapters or tools should bound admission themselves.
    """

    loop = asyncio.get_running_loop()
    future: asyncio.Future[_T] = loop.create_future()
    settled: asyncio.Future[None] = loop.create_future()
    outcome: list[tuple[bool, Any]] = []
    # Match ``asyncio.to_thread``, which runs its target in a copy of the caller's context. Without
    # this the worker would start with an empty context, so a sync adapter or handler reading
    # credentials, tenant identity, or tracing state from a ``ContextVar`` would see defaults. The
    # copy is also what keeps an abandoned worker's tool-call authorization alive: it is taken after
    # the call's ``CallContext`` is set, and the caller's later reset cannot reach into it.
    caller_context = copy_context()

    def discard_late_awaitable(*, on_live_loop: bool) -> None:
        """Dispose an awaitable that arrived after the run gave up on the call.

        A sync tool handler may return one, and the normal path accepts any awaitable, so the late
        path has to handle the same shapes. Nothing downstream will await this one: a coroutine is
        closed so its cleanup runs and it cannot surface as an unawaited-coroutine warning, and a
        future or task is cancelled and consumed so it stops running and cannot surface as a
        never-retrieved exception. Any other awaitable has no generic disposal and is left alone.

        ``on_live_loop`` is False when the run's loop has already closed. Cancelling a future there
        is unsafe -- it schedules callbacks on the dead loop -- and a still-pending future can no
        longer run, so there is nothing to stop and no outcome to read. An *already settled* one is
        different: reading its outcome touches no loop, and an unretrieved exception is exactly what
        warns at collection, so that case is consumed rather than skipped.
        """

        succeeded, payload = outcome[0]
        if not succeeded:
            return
        if inspect.iscoroutine(payload):
            payload.close()
        elif isinstance(payload, asyncio.Future):
            if on_live_loop:
                payload.cancel()
                payload.add_done_callback(consume_task_outcome)
            elif payload.done():
                consume_task_outcome(payload)

    def deliver() -> None:
        if not settled.done():
            settled.set_result(None)
        if future.done():
            discard_late_awaitable(on_live_loop=True)
            return
        succeeded, payload = outcome[0]
        if succeeded:
            future.set_result(payload)
        elif isinstance(payload, StopIteration):
            # ``Future.set_exception`` refuses ``StopIteration`` by contract and raises TypeError.
            # That TypeError would surface here, inside a ``call_soon_threadsafe`` callback, where
            # nothing awaits it -- so the future stayed pending forever while ``settled`` was
            # already resolved, and the run hung with no warning and no deadline able to end it.
            # A callee raising it is ordinary: ``next(...)`` on an exhausted iterator does.
            future.set_exception(
                RuntimeError("synchronous call raised StopIteration").with_traceback(
                    payload.__traceback__
                )
            )
        else:
            future.set_exception(payload)

    def warn_if_unsettled() -> None:
        if settled.done():
            return
        _LOGGER.warning(
            "abandoned a synchronous call still running on %r: the run stopped waiting for it, but "
            "nothing can reclaim its thread until it returns on its own. An implementation that "
            "never returns leaks one thread per abandoned call; enforce a timeout at its I/O edge.",
            thread_name,
        )

    def worker() -> None:
        try:
            outcome.append((True, caller_context.run(call)))
        except BaseException as exc:  # surfaced to the awaiter, never swallowed here
            outcome.append((False, exc))
        try:
            loop.call_soon_threadsafe(deliver)
        except RuntimeError:
            # The run was abandoned and its loop has since closed, so ``deliver`` will never run.
            # Nothing will await a late awaitable either, so discard it here instead.
            discard_late_awaitable(on_live_loop=False)

    threading.Thread(target=worker, name=thread_name, daemon=True).start()
    return AbandonableSyncCall(result=future, settled=settled, warn_if_unsettled=warn_if_unsettled)


async def await_abandonable_call(
    pending: Any,
    *,
    deadline: float | None,
    token: CancellationToken | None,
    grace_s: float,
    check_boundary: Callable[[float | None], None],
) -> Any:
    """Await a call while propagating run cancellation and the run deadline.

    The one race behind both halves of the kernel's async surface -- a model call and a tool
    handler. It was written twice and the copies had already drifted: one tested `task in done`
    against the set `asyncio.wait` returned, the other `task.done()`. Equivalent in practice, since
    nothing awaits between the two statements, but the pair is exactly the shape that stops being
    equivalent the day someone adds an await.

    What the two callers genuinely differ on is passed in. `check_boundary` is the caller's notion
    of which boundaries are terminal *while this kind of call is in flight*: a model call answers to
    cancellation and the deadline only, a tool handler also answers to interrupt and pause.
    `grace_s` is how long an abandoned worker gets to settle.

    A synchronous callee cannot be interrupted: its thread keeps running to completion. Cancelling
    its future therefore *abandons* the call -- the run stops waiting within `grace_s` and the
    detached outcome is consumed so late cleanup cannot warn, but the callee's socket and CPU work
    continue until it returns on its own. Abandoning is only real because the thread is a daemon
    nobody joins.

    `asyncio.CancelledError` from the callee surfaces as `CalleeCancelled`: the two callers report
    it differently and neither meaning belongs here, but neither can report it at all if it cannot
    be told from the cancellation the host delivered to the awaiting task. Anything raising plain
    `CancelledError` out of here is that second kind.
    """

    sync_call = pending if isinstance(pending, AbandonableSyncCall) else None
    task = sync_call.result if sync_call is not None else asyncio.ensure_future(pending)
    loop = asyncio.get_running_loop()
    cancelled: asyncio.Future[None] = loop.create_future()
    outcome_consumed = False

    def signal_cancelled() -> None:
        def resolve() -> None:
            if not cancelled.done():
                cancelled.set_result(None)

        loop.call_soon_threadsafe(resolve)

    # Everything from here on is inside the ``try``, because the call is already running: the task
    # exists above, and a blocking callee's thread was started before this function was entered.
    # Registration or the timeout arithmetic failing used to skip the ``finally`` entirely, so the
    # call was neither cancelled, detached, nor consumed -- it ran to completion behind a run that
    # had already reported a failure.
    def remove_callback() -> None:
        return None

    try:
        if token is not None:
            remove_callback = token.add_cancel_callback(signal_cancelled)
        timeout = None if deadline is None else max(0.0, deadline - time.time())
        await asyncio.wait({task, cancelled}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        # Checked before the result is read, so a boundary that lands in the same tick as a
        # completed call still wins. A run told to stop must not report work it decided not to do.
        check_boundary(deadline)
        if task.done():
            outcome_consumed = True
            try:
                return task.result()
            except asyncio.CancelledError as exc:
                raise CalleeCancelled from exc
        if cancelled.done():
            raise RunCancelled("run cancelled")
        raise RunTimeout("run exceeded max duration")
    finally:
        remove_callback()
        if not cancelled.done():
            cancelled.cancel()
        if not task.done():
            await detach_unfinished_call(task, sync_call, grace_s=grace_s)
        elif not outcome_consumed:
            # The callee finished -- possibly by raising -- in the same loop turn that made a run
            # boundary observable, so ``check_boundary`` raised before anything read the outcome.
            # Nothing downstream will read it now either, and an unretrieved exception surfaces as a
            # "Future exception was never retrieved" warning at collection.
            consume_task_outcome(task)


async def detach_unfinished_call(
    task: asyncio.Future[Any],
    sync_call: AbandonableSyncCall[Any] | None,
    *,
    grace_s: float,
) -> None:
    """Release the awaiter's hold on a call that outlived a run boundary.

    Cancelling ``task`` is what frees the awaiter. For a native async call that also *delivers*
    the cancellation, so the grace interval is spent letting the callee's cleanup run. A sync
    call has no cancellation to deliver and its waiter is a plain future, so cancelling it
    completes it immediately: waiting on it would grant no grace at all. The grace is spent
    waiting on the worker's own completion instead, which is what the interval is for -- a
    worker that finishes inside it lands its writes before the run finalizes rather than racing
    it, and is never reported as abandoned.

    Both halves report. The warning used to be gated on there being a sync call, so an async callee
    whose cleanup outran the grace was detached in silence -- and it has the same unbounded shape:
    the task and whatever it holds stay alive with nobody to reclaim them, one per abandonment, on a
    loop that may run for days. For a streamed model call that is an open provider connection pool.
    Visible growth rather than silent growth is the property this module claims for itself, and it
    was only ever true of the half that happened to be written first.
    """

    task.cancel()
    watched: asyncio.Future[Any] = task if sync_call is None else sync_call.settled
    done, _pending = await asyncio.wait({watched}, timeout=max(0.0, grace_s))
    if watched not in done:
        if sync_call is not None:
            sync_call.warn_if_unsettled()
        else:
            _LOGGER.warning(
                "abandoned an asynchronous call whose cleanup outran the %.3gs grace interval: the "
                "run stopped waiting for it, and nothing will reclaim the task or what it holds "
                "until its cleanup returns on its own. An implementation that never returns leaks "
                "one task per abandoned call; enforce a timeout at its I/O edge.",
                grace_s,
            )
    if task.done():
        consume_task_outcome(task)
    else:
        task.add_done_callback(consume_task_outcome)
