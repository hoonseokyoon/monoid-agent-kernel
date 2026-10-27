"""Observe actual model-call owners through the public termination snapshot."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from monoid_agent_kernel import CallTerminationBudget, GraceBudget, TerminationBudgetSnapshot
from monoid_agent_kernel.core import _deadline as deadline_clock
from monoid_agent_kernel.core._deadline import _MonotonicDeadline
from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.spec import ModelConfig, ModelRetryConfig
from monoid_agent_kernel.errors import ModelDispatchRefused, RunCancelled
from monoid_agent_kernel.model_call import ModelCallRunner
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn, TextDelta, TurnComplete

pytestmark = pytest.mark.unit
REQUEST = ModelRequest(instruction="answer", system_prompt="system", tools=())


async def _until(predicate):
    async def poll():
        while not (value := predicate()):
            await asyncio.sleep(0)
        return value
    return await asyncio.wait_for(poll(), 3)


async def _finish(task):
    if not task.done():
        task.cancel()
    with contextlib.suppress(BaseException):
        await asyncio.wait_for(task, 3)


@pytest.mark.parametrize("shape", ["native", "sync", "sync_awaitable", "stream"])
def test_concurrent_same_ids_are_immutable_single_sample_owned_calls(monkeypatch, shape):
    now, samples = [100.0], []

    def clock():
        samples.append(now[0])
        return now[0]

    monkeypatch.setattr(deadline_clock, "monotonic", clock)
    getters, calls = [], []

    async def run():
        loop = asyncio.get_running_loop()
        entered, release = asyncio.Event(), asyncio.Event()
        sync_release = threading.Event()

        def began(request):
            calls.append(request)
            if len(calls) == 2:
                loop.call_soon_threadsafe(entered.set)

        async def answer(request):
            began(request)
            await release.wait()
            return ModelTurn(final_text="answer")

        def sync_answer(request):
            began(request)
            assert sync_release.wait(3)
            return ModelTurn(final_text="answer")

        async def stream(request):
            began(request)
            yield TextDelta("answer")
            await release.wait()
            yield TurnComplete()

        adapter = {
            "native": SimpleNamespace(anext_turn=answer),
            "sync": SimpleNamespace(next_turn=sync_answer),
            "sync_awaitable": SimpleNamespace(next_turn=lambda request: answer(request)),
            "stream": SimpleNamespace(astream_turn=stream),
        }[shape]

        def current_adapter():
            getters.append("adapter")
            return adapter

        def current_grace():
            getters.append("grace")
            return 0.6

        runner = ModelCallRunner(
            adapter=adapter, current_adapter=current_adapter,
            current_cancel_grace_s=current_grace, abort_drain_s=0,
        )
        tasks = [asyncio.create_task(runner.acall(
            REQUEST, logical_call_id="same-logical-id", _deadline=_MonotonicDeadline(105),
            delta_consumer=(lambda chunk: None) if shape == "stream" else None,
        )) for _ in range(2)]
        try:
            await asyncio.wait_for(entered.wait(), 3)
            sampled_getters = list(getters)
            samples.clear()
            snapshot = await asyncio.to_thread(runner.termination_budget)
            assert samples == [100]
            assert getters == sampled_getters
            assert isinstance(snapshot, TerminationBudgetSnapshot)
            assert snapshot.observed_at_monotonic_s == 100
            assert snapshot.run_deadline_remaining_s is None and snapshot.stream_exit is None
            assert isinstance(snapshot.calls, tuple) and len(snapshot.calls) == 2
            assert len({call.call_id for call in snapshot.calls}) == 2
            for call in snapshot.calls:
                assert isinstance(call, CallTerminationBudget)
                assert call.logical_call_id == "same-logical-id" and call.attempt_index == 1
                assert call.phase == ("streaming" if shape == "stream" else "waiting")
                assert call.deadline_remaining_s == 5
                assert call.drain_budget_s == 0 and call.drain_remaining_s is None
                assert isinstance(call.grace_windows, tuple)
                assert len(call.grace_windows) == 1
                grace = call.grace_windows[0]
                assert isinstance(grace, GraceBudget)
                assert (grace.source, grace.budget_s, grace.remaining_s) == ("wait_cancel", 0.6, None)
                with pytest.raises(FrozenInstanceError):
                    grace.budget_s = 99
                with pytest.raises(FrozenInstanceError):
                    call.phase = "settling"
            with pytest.raises(FrozenInstanceError):
                snapshot.calls = ()
            now[0] += 1
            assert all(call.deadline_remaining_s == 4 for call in runner.termination_budget().calls)
            assert all(call.deadline_remaining_s == 5 for call in snapshot.calls)
            assert getters == sampled_getters
            release.set()
            sync_release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), 3)
            assert runner.termination_budget().calls == ()
        finally:
            release.set()
            sync_release.set()
            for task in tasks:
                await _finish(task)

    asyncio.run(run())


def test_preparing_backoff_and_receipt_settlement_keep_one_call_identity():
    snapshots, dispatches = [], []

    async def run():
        backoff, resume = asyncio.Event(), asyncio.Event()

        class Runner(ModelCallRunner):
            async def _abackoff(self, delay_s, deadline):
                snapshots.append(self.termination_budget())
                backoff.set()
                await resume.wait()

        async def answer(request):
            dispatches.append(request)
            if len(dispatches) == 1:
                raise ModelDispatchRefused("busy", error_code="provider_busy", retryable=True)
            return ModelTurn(final_text="answer")

        adapter = SimpleNamespace(anext_turn=answer)

        def current_adapter():
            snapshots.append(runner.termination_budget())
            return adapter

        def settled(call):
            snapshots.append(runner.termination_budget())
            raise RuntimeError("diagnostic sink failed after observing settlement")

        runner = Runner(adapter=adapter, current_adapter=current_adapter, settled_sink=settled)
        request = replace(REQUEST, model=ModelConfig(retry=ModelRetryConfig(
            layer="kernel", max_attempts=2, initial_delay_s=0.1, max_delay_s=0.1, jitter_s=0,
        )))
        task = asyncio.create_task(runner.acall(request))
        try:
            await asyncio.wait_for(backoff.wait(), 3)
            assert runner.termination_budget().calls[0].phase == "backoff"
            resume.set()
            turn, receipt = await asyncio.wait_for(task, 3)
            assert turn.final_text == "answer" and receipt.attempts == 2
            assert runner.termination_budget().calls == ()
        finally:
            resume.set()
            await _finish(task)

    asyncio.run(run())
    observed = [snapshot.calls[0] for snapshot in snapshots]
    assert [(call.phase, call.attempt_index) for call in observed] == [
        ("preparing", None), ("backoff", 1), ("settling", 2),
    ]
    assert len({call.call_id for call in observed}) == 1


def test_preparing_getter_failure_retires_its_call():
    seen = []

    def getter():
        seen.append(runner.termination_budget().calls[0])
        raise RuntimeError("adapter unavailable")

    runner = ModelCallRunner(adapter=object(), current_adapter=getter)
    with pytest.raises(RuntimeError, match="adapter unavailable"):
        asyncio.run(runner.acall(REQUEST))
    assert seen[0].phase == "preparing" and seen[0].attempt_index is None
    assert runner.termination_budget().calls == ()


def test_grace_reservation_starts_at_cleanup_and_keeps_its_sampled_value(monkeypatch):
    now, live, reads = [100.0], [2.0], []
    monkeypatch.setattr(deadline_clock, "monotonic", lambda: now[0])
    token = CancellationToken()

    async def run():
        entered, closing, release = (asyncio.Event() for _ in range(3))

        async def answer(request):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                closing.set()
                await release.wait()
                return ModelTurn(final_text="late")

        def grace():
            reads.append(live[0])
            return live[0]

        runner = ModelCallRunner(adapter=SimpleNamespace(anext_turn=answer),
                                 current_cancellation_token=lambda: token, current_cancel_grace_s=grace)
        task = asyncio.create_task(runner.acall(REQUEST))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            reserved = runner.termination_budget().calls[0].grace_windows[0]
            assert (reserved.budget_s, reserved.remaining_s) == (2, None)
            live[0], now[0] = 9, 101
            token.cancel()
            await asyncio.wait_for(closing.wait(), 3)
            now[0] += 0.25
            started = runner.termination_budget().calls[0].grace_windows[0]
            assert (started.budget_s, started.remaining_s) == (2, 1.75)
            assert reads == [2] and reserved.remaining_s is None
            release.set()
            with pytest.raises(RunCancelled):
                await asyncio.wait_for(task, 3)
            assert runner.termination_budget().calls == ()
        finally:
            release.set()
            await _finish(task)

    asyncio.run(run())


class _UncoercibleGrace:
    def __float__(self):
        raise AssertionError("snapshot coerced a host value")

    def __lt__(self, other):
        raise AssertionError("snapshot compared a host value")


@pytest.mark.parametrize(("value", "expected"), [
    (-2.0, 0.0), (float("nan"), 0.0), (float("inf"), None), (_UncoercibleGrace(), "omitted"),
])
def test_snapshot_observes_safe_grace_values_without_evaluating_custom_values(value, expected):
    reads = []

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def answer(request):
            entered.set()
            await release.wait()
            return ModelTurn(final_text="answer")

        def grace():
            reads.append(value)
            return value

        runner = ModelCallRunner(adapter=SimpleNamespace(anext_turn=answer), current_cancel_grace_s=grace)
        task = asyncio.create_task(runner.acall(REQUEST))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            windows = runner.termination_budget().calls[0].grace_windows
            if expected == "omitted":
                assert windows == ()
            else:
                assert len(windows) == 1
                assert windows[0].budget_s == expected and windows[0].remaining_s is None
            assert len(reads) == 1
            release.set()
            assert (await asyncio.wait_for(task, 3))[0].final_text == "answer"
        finally:
            release.set()
            await _finish(task)

    asyncio.run(run())


def test_stream_close_and_outer_drain_keep_separate_live_windows(monkeypatch):
    now, live = [100.0], [1.0]
    monkeypatch.setattr(deadline_clock, "monotonic", lambda: now[0])
    token = CancellationToken()

    async def run():
        entered, eof, closing, release = (asyncio.Event() for _ in range(4))

        class Stream:
            first = True

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.first:
                    self.first = False
                    return TextDelta("first")
                entered.set()
                await eof.wait()
                raise StopAsyncIteration

            async def aclose(self):
                closing.set()
                await release.wait()

        runner = ModelCallRunner(
            adapter=SimpleNamespace(astream_turn=lambda request: Stream()),
            current_cancellation_token=lambda: token, abort_drain_s=5,
            current_cancel_grace_s=lambda: live[0],
        )
        task = asyncio.create_task(runner.acall(REQUEST, delta_consumer=lambda chunk: None))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            live[0] = 2
            token.cancel()

            def draining_wait_started():
                call = runner.termination_budget().calls[0]
                return call.phase == "draining" and any(
                    window.source == "wait_cancel" and window.budget_s == 2
                    for window in call.grace_windows
                )

            await _until(draining_wait_started)
            now[0], live[0] = 102, 3
            eof.set()
            await asyncio.wait_for(closing.wait(), 3)
            call = runner.termination_budget().calls[0]
            assert call.phase == "closing" and call.drain_remaining_s == 3
            windows = {window.source: window for window in call.grace_windows}
            assert windows["wait_cancel"].budget_s == 2
            assert windows["wait_cancel"].remaining_s is None
            assert (windows["stream_close"].budget_s, windows["stream_close"].remaining_s) == (3, 3)
            token.cancel()
            assert runner.termination_budget().calls[0].phase == "closing"
            release.set()
            with pytest.raises(RunCancelled):
                await asyncio.wait_for(task, 3)
            assert runner.termination_budget().calls == ()
        finally:
            eof.set()
            release.set()
            await _finish(task)

    asyncio.run(run())


def test_failed_grace_getter_uses_fallback_and_late_worker_cannot_revive_the_call(monkeypatch):
    monkeypatch.setattr(deadline_clock, "monotonic", lambda: 100.0)
    reads = []

    async def run():
        loop = asyncio.get_running_loop()
        release, finished = threading.Event(), asyncio.Event()

        def answer(request):
            try:
                assert release.wait(3)
                return ModelTurn(final_text="late")
            finally:
                loop.call_soon_threadsafe(finished.set)

        def grace():
            reads.append("grace")
            raise RuntimeError("grace unavailable")

        runner = ModelCallRunner(adapter=SimpleNamespace(next_turn=answer),
                                 current_cancel_grace_s=grace, cancel_grace_s=0.2)
        task = asyncio.create_task(runner.acall(REQUEST))
        try:
            def cleanup_started():
                snapshot = runner.termination_budget()
                if snapshot.calls and any(window.remaining_s is not None
                                          for window in snapshot.calls[0].grace_windows):
                    return snapshot
                return None

            snapshot = await _until(cleanup_started)
            window = snapshot.calls[0].grace_windows[0]
            assert (window.budget_s, window.remaining_s) == (0.2, pytest.approx(0.2))
            assert reads == ["grace"]
            with pytest.raises(RuntimeError, match="grace unavailable"):
                await asyncio.wait_for(task, 3)
            assert runner.termination_budget().calls == ()
            release.set()
            await asyncio.wait_for(finished.wait(), 3)
            await asyncio.sleep(0)
            assert runner.termination_budget().calls == ()
        finally:
            release.set()
            await _finish(task)
            await asyncio.wait_for(finished.wait(), 3)

    asyncio.run(run())
