"""The public stream snapshot follows the real exit wait and driver lifetime."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import FrozenInstanceError

import pytest

from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel import StreamExitBudget
from monoid_agent_kernel.core import _deadline as deadline_clock
from monoid_agent_kernel.core.spec import AgentRunSpec, RunLimits
from monoid_agent_kernel.core.streaming import QueueEventSink, RunStream
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn, TextDelta, TurnComplete

pytestmark = pytest.mark.unit


async def _until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0)
    await asyncio.wait_for(poll(), 3)


async def _finish(task):
    if task is not None:
        if not task.done():
            task.cancel()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(task, 3)


@pytest.mark.parametrize("cancel_during_cleanup", [False, True])
def test_interrupted_exit_reuses_first_cutoff_and_keeps_hard_cancel_until_cleanup_settles(
    monkeypatch, cancel_during_cleanup,
):
    now, extra, reads, cancels = [100.0], [0.2], [], []
    monkeypatch.setattr(deadline_clock, "monotonic", lambda: now[0])

    async def run():
        entered, cleanup, repeated, release = (asyncio.Event() for _ in range(4))

        async def drive():
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup.set()
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        repeated.set()

        def allowance():
            reads.append(extra[0])
            return extra[0]

        stream = RunStream(sink=QueueEventSink(), drive_factory=drive,
                           request_cancel=lambda: cancels.append("stop"),
                           cancel_grace_s=0.8, current_extra_grace_s=allowance)
        idle = stream.termination_budget()
        assert idle.calls == () and idle.run_deadline_remaining_s is None
        assert isinstance(idle.stream_exit, StreamExitBudget)
        assert (idle.stream_exit.phase, idle.stream_exit.wait_budget_s,
                idle.stream_exit.hard_cancel_remaining_s) == ("idle", None, None)
        assert reads == [] and cancels == []
        await stream.__aenter__()
        first = second = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            first = asyncio.create_task(stream.__aexit__(None, None, None))
            await _until(lambda: stream.termination_budget().stream_exit.phase == "cooperative_wait")
            active = stream.termination_budget().stream_exit
            assert active.wait_budget_s == 1
            with pytest.raises(FrozenInstanceError):
                active.phase = "settled"
            extra[0], now[0] = 99, 100.25
            assert stream.termination_budget().stream_exit.hard_cancel_remaining_s == 0.75
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            stopped = stream.termination_budget().stream_exit
            assert stopped.phase == "wait_interrupted" and stopped.hard_cancel_remaining_s is None
            assert stopped.wait_budget_s == 1
            now[0] = 102
            assert not cleanup.is_set()  # Querying an expired interrupted wait starts no timer.
            assert stream.termination_budget().stream_exit.phase == "wait_interrupted"
            second = asyncio.create_task(stream.__aexit__(None, None, None))
            await asyncio.wait_for(cleanup.wait(), 3)
            hard = stream.termination_budget().stream_exit
            assert hard.phase == "hard_cancel_requested" and hard.hard_cancel_remaining_s == 0
            assert not second.done()
            assert reads == [0.2]
            if cancel_during_cleanup:
                second.cancel()
                await asyncio.wait_for(repeated.wait(), 3)
                hard = stream.termination_budget().stream_exit
                assert hard.phase == "hard_cancel_requested" and hard.hard_cancel_remaining_s == 0
                assert not second.done()
            release.set()
            await asyncio.wait_for(second, 3)
            settled = stream.termination_budget().stream_exit
            assert settled.phase == "settled" and settled.hard_cancel_remaining_s == 0
            assert settled.wait_budget_s == 1
        finally:
            release.set()
            await _finish(first)
            await _finish(second)
            await asyncio.wait_for(stream.__aexit__(None, None, None), 3)

    asyncio.run(run())


def test_driver_natural_completion_marks_settled_without_creating_an_exit_timer():
    async def run():
        async def drive():
            return None

        def unexpected_stop():
            raise AssertionError("a completed driver needs no stop request")

        stream = RunStream(sink=QueueEventSink(), drive_factory=drive, request_cancel=unexpected_stop)
        await stream.__aenter__()
        await _until(lambda: stream.termination_budget().stream_exit.phase == "settled")
        snapshot = stream.termination_budget().stream_exit
        assert snapshot.wait_budget_s is None and snapshot.hard_cancel_remaining_s is None
        await asyncio.wait_for(stream.__aexit__(None, None, None), 3)

    asyncio.run(run())


@pytest.mark.parametrize("callback_raises", [False, True])
def test_legacy_callback_runtime_precedes_the_first_exit_wait(monkeypatch, callback_raises):
    now, reads, completed = [100.0], [], []
    monkeypatch.setattr(deadline_clock, "monotonic", lambda: now[0])

    async def run():
        entered, stop = asyncio.Event(), asyncio.Event()

        async def drive():
            entered.set()
            await stop.wait()
            # A real timer distinguishes a cooperative wait from immediate cancellation.
            await asyncio.sleep(0.01)
            completed.append("cooperative")

        def allowance():
            reads.append(now[0])
            now[0] += 2  # Deterministic elapsed callback time; asyncio's clock stays real.
            if callback_raises:
                raise RuntimeError("legacy callback failed after doing work")
            return 0

        stream = RunStream(sink=QueueEventSink(), drive_factory=drive, request_cancel=stop.set,
                           cancel_grace_s=0.2, current_extra_grace_s=allowance)
        await stream.__aenter__()
        try:
            await asyncio.wait_for(entered.wait(), 3)
            await asyncio.wait_for(stream.__aexit__(None, None, None), 3)
            assert completed == ["cooperative"]
            snapshot = stream.termination_budget().stream_exit
            assert snapshot.phase == "settled" and snapshot.wait_budget_s == 0.2
            assert snapshot.hard_cancel_remaining_s is None
            assert reads == [100.0]
        finally:
            stop.set()
            await asyncio.wait_for(stream.__aexit__(None, None, None), 3)

    asyncio.run(run())


@pytest.mark.parametrize(("extra", "expected"), [
    ("raise", 0.2), (-1.0, 0.0), (float("nan"), None), (float("inf"), None),
    ("malformed", "error"),
])
def test_raw_legacy_callback_preserves_wait_values_and_is_not_evaluated_by_queries(extra, expected):
    reads = []

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def drive():
            entered.set()
            await release.wait()

        def allowance():
            reads.append(extra)
            if extra == "raise":
                raise RuntimeError("legacy callback failed")
            return object() if extra == "malformed" else extra

        stream = RunStream(sink=QueueEventSink(), drive_factory=drive, request_cancel=release.set,
                           cancel_grace_s=0.2, current_extra_grace_s=allowance)
        stream.termination_budget()
        assert reads == []
        await stream.__aenter__()
        try:
            await asyncio.wait_for(entered.wait(), 3)
            if expected == "error":
                with pytest.raises(TypeError):
                    await asyncio.wait_for(stream.__aexit__(None, None, None), 3)
                await _until(lambda: stream.termination_budget().stream_exit.phase == "settled")
            else:
                await asyncio.wait_for(stream.__aexit__(None, None, None), 3)
            snapshot = stream.termination_budget()
            assert snapshot.stream_exit.phase == "settled"
            assert snapshot.stream_exit.wait_budget_s == (None if expected == "error" else expected)
            assert snapshot.stream_exit.hard_cancel_remaining_s is None
            assert len(reads) == 1
        finally:
            release.set()
            await asyncio.wait_for(stream.__aexit__(None, None, None), 3)

    asyncio.run(run())


@pytest.mark.parametrize(("kind", "maximum", "expected"), [
    ("partial_drain", 60, 2.9), ("deadline_cap", 2, 2.4),
    ("one_shot", 60, 1.7), ("parallel", 60, 1.7), ("unbounded_grace", 60, 0.5),
])
def test_managed_exit_uses_remaining_parallel_allowance_and_freezes_it(
    tmp_path, monkeypatch, kind, maximum, expected,
):
    now, samples = [100.0], []

    def clock():
        samples.append(now[0])
        return now[0]

    monkeypatch.setattr(deadline_clock, "monotonic", clock)
    monkeypatch.setitem(RunStream.__init__.__kwdefaults__, "cancel_grace_s", 0.5)
    streaming = kind in ("partial_drain", "deadline_cap")

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        dispatches = []

        class Adapter:
            async def anext_turn(self, request):
                dispatches.append(request)
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    await release.wait()
                return ModelTurn(final_text="answer")

        class StreamingAdapter:
            async def astream_turn(self, request):
                dispatches.append(request)
                yield TextDelta("answer")
                entered.set()
                await release.wait()
                yield TurnComplete(usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3})

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        loop = AgentLoop(
            spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs",
                              limits=RunLimits(max_duration_s=maximum)),
            model_adapter=StreamingAdapter() if streaming else Adapter(), stream_model_calls=True,
            runtime_config_provider=runtime_provider(runtime_config("fs.write")),
        )
        assert loop.termination_budget().calls == ()
        assert loop.termination_budget().run_deadline_remaining_s is None
        loop.async_model_abort_drain_s = 5
        loop.async_model_cancel_grace_s = (float("inf") if kind == "unbounded_grace"
                                           else 0.4 if streaming else 1.2)
        stream = exiting = extra_call = None
        await loop.aopen()
        try:
            assert loop.termination_budget().run_deadline_remaining_s == maximum
            stream = loop.astream("go")
            assert stream._exit_wait_s() == 0.5  # No active call grants only the base allowance.
            await stream.__aenter__()
            await asyncio.wait_for(entered.wait(), 3)
            if kind == "parallel":
                extra_call = asyncio.create_task(loop._session.res.model_runner.acall(
                    ModelRequest(instruction="another", system_prompt="system", tools=()),
                ))
                await _until(lambda: len(dispatches) == 2)
                assert len(stream.termination_budget().calls) == 2
            if streaming:
                now[0] = 100.5 if kind == "deadline_cap" else 100
                loop.cancellation_token.cancel()
                await _until(lambda: stream.termination_budget().calls[0].phase == "draining")
                if kind == "partial_drain":
                    now[0] = 103
            assert stream.termination_budget().stream_exit.phase == "idle"
            exiting = asyncio.create_task(stream.__aexit__(None, None, None))
            await _until(lambda: stream.termination_budget().stream_exit.phase == "cooperative_wait")
            samples.clear()
            snapshot = stream.termination_budget()
            assert samples == [now[0]]
            assert snapshot.observed_at_monotonic_s == now[0]
            assert snapshot.stream_exit.wait_budget_s == pytest.approx(expected)
            assert snapshot.stream_exit.hard_cancel_remaining_s == pytest.approx(expected)
            loop.async_model_abort_drain_s = loop.async_model_cancel_grace_s = 99
            now[0] += 0.025
            frozen = stream.termination_budget().stream_exit
            assert frozen.wait_budget_s == pytest.approx(expected)
            assert frozen.hard_cancel_remaining_s == pytest.approx(expected - 0.025)
            release.set()
            await asyncio.wait_for(exiting, 3)
            await _finish(extra_call)
            completed = stream.termination_budget()
            assert completed.calls == ()
            assert completed.stream_exit.phase == "settled"
            assert completed.stream_exit.hard_cancel_remaining_s is None
            await loop.aclose()
            assert loop.termination_budget().calls == ()
            assert loop.termination_budget().run_deadline_remaining_s is None
        finally:
            release.set()
            await _finish(extra_call)
            if stream is not None:
                await _finish(exiting)
                await asyncio.wait_for(stream.__aexit__(None, None, None), 3)
            if loop._session is not None:
                await loop.aclose()

    asyncio.run(run())
