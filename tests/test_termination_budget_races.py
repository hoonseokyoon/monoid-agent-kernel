"""Cross-thread stop publication can finish after its drain wait owner."""

import asyncio
import contextlib
import threading
from types import SimpleNamespace

import pytest

from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.termination_budget import _CallBudget
from monoid_agent_kernel.errors import RunCancelled
from monoid_agent_kernel.model_call import ModelCallRunner
from monoid_agent_kernel.providers.base import ModelRequest, TextDelta, TurnComplete

pytestmark = pytest.mark.unit


def test_late_stop_callback_cannot_reopen_a_finished_drain_during_settlement(monkeypatch):
    callback_entered, release_callback = threading.Event(), threading.Event()
    token = CancellationToken()
    original = _CallBudget.begin_drain

    def delayed_begin(self, abort_at, budget_s):
        callback_entered.set()
        assert release_callback.wait(3), "test did not release stop callback"
        original(self, abort_at, budget_s)

    monkeypatch.setattr(_CallBudget, "begin_drain", delayed_begin)
    snapshots = []

    async def run():
        entered, release_stream = asyncio.Event(), asyncio.Event()

        async def stream(request):
            yield TextDelta("answer")
            entered.set()
            await release_stream.wait()
            yield TurnComplete()

        canceller = threading.Thread(target=token.cancel, daemon=True)

        def settled(call):
            # The stream already observed the accepted stop and settled, while the
            # canceller is still between accepting that stop and publishing its signal.
            before = runner.termination_budget().calls[0]
            assert before.phase == "settling"
            assert before.drain_remaining_s is None
            release_callback.set()
            canceller.join(3)
            assert not canceller.is_alive()
            snapshots.append(runner.termination_budget())

        runner = ModelCallRunner(
            adapter=SimpleNamespace(astream_turn=stream),
            current_cancellation_token=lambda: token, abort_drain_s=5,
            settled_sink=settled,
        )
        task = asyncio.create_task(runner.acall(
            ModelRequest(instruction="answer", system_prompt="", tools=()),
            delta_consumer=lambda chunk: None,
        ))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            canceller.start()
            assert await asyncio.to_thread(callback_entered.wait, 3)
            release_stream.set()
            with pytest.raises(RunCancelled):
                await asyncio.wait_for(task, 3)
        finally:
            release_callback.set()
            release_stream.set()
            if not task.done():
                task.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(task, 3)
            if canceller.ident is not None:
                canceller.join(3)
        assert runner.termination_budget().calls == ()

    asyncio.run(run())
    assert len(snapshots) == 1
    assert snapshots[0].calls[0].phase == "settling"
    assert snapshots[0].calls[0].drain_remaining_s is None
