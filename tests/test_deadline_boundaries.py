"""Exact deadline boundaries without changing asyncio's independent timer."""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from types import SimpleNamespace

import pytest

from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core import _deadline as deadline_clock
from monoid_agent_kernel.core._deadline import _MonotonicDeadline
from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.checkpoint import RunCheckpoint
from monoid_agent_kernel.core.spec import AgentRunSpec, RunLimits
from monoid_agent_kernel.errors import RunCancelled, RunTimeout
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.model_call import ModelCallRunner
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn, TextDelta, TurnComplete
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call

pytestmark = pytest.mark.unit

REQUEST = ModelRequest(instruction="answer", system_prompt="system", tools=())


@pytest.mark.parametrize("origin", [-1_000_000.0, 123.0, 1_000_000_000.0])
def test_epoch_conversion_uses_the_current_process_monotonic_origin(monkeypatch, origin) -> None:
    monkeypatch.setattr(deadline_clock, "monotonic", lambda: origin)
    before = time.time()
    epoch_deadline = before + 60
    converted = _MonotonicDeadline.from_epoch(epoch_deadline)
    after = time.time()
    assert origin + (epoch_deadline - after) <= converted.cutoff_s
    assert converted.cutoff_s <= origin + (epoch_deadline - before)


@pytest.mark.parametrize("now", [
    math.nextafter(100.0, -math.inf), 100.0, math.nextafter(100.0, math.inf),
])
def test_actual_dispatch_obeys_the_exact_monotonic_cutoff(monkeypatch, now) -> None:
    monkeypatch.setattr(deadline_clock, "monotonic", lambda: now)
    calls = []

    async def answer(request):
        calls.append(request)
        return ModelTurn(final_text="answer")

    settled = []
    runner = ModelCallRunner(adapter=SimpleNamespace(anext_turn=answer), settled_sink=settled.append)

    async def run():
        call = runner.acall(REQUEST, _deadline=_MonotonicDeadline(100.0))
        if now < 100.0:
            turn, _receipt = await asyncio.wait_for(call, 3)
            assert turn.final_text == "answer"
        else:
            with pytest.raises(RunTimeout):
                await asyncio.wait_for(call, 3)

    asyncio.run(run())
    assert len(calls) == (1 if now < 100.0 else 0)
    assert len(settled) == 1
    assert settled[0].receipt.attempts == (1 if now < 100.0 else 0)


@pytest.mark.parametrize("origin", [-1_000_000.0, 123.0])
def test_equal_drain_and_run_cutoffs_keep_deadline_classification(monkeypatch, origin) -> None:
    now = [origin]
    monkeypatch.setattr(deadline_clock, "monotonic", lambda: now[0])
    token = CancellationToken()
    settled, seen = [], []
    budget = 0.25

    async def run():
        entered, release, closed = (asyncio.Event() for _ in range(3))

        async def stream(request):
            try:
                yield TextDelta("first")
                entered.set()
                await release.wait()
                yield TurnComplete()
            finally:
                closed.set()

        runner = ModelCallRunner(
            adapter=SimpleNamespace(astream_turn=stream),
            current_cancellation_token=lambda: token, abort_drain_s=budget,
            cancel_grace_s=0.2, settled_sink=settled.append,
        )
        task = asyncio.create_task(runner.acall(
            REQUEST, delta_consumer=seen.append,
            _deadline=_MonotonicDeadline(origin + budget),
        ))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            token.cancel()  # begin() fixes abort_at=origin before returning.
            now[0] = origin + budget
            with pytest.raises(RunCancelled):
                await asyncio.wait_for(task, 3)
            await asyncio.wait_for(closed.wait(), 3)
        finally:
            release.set()
            if not task.done():
                task.cancel()
            with contextlib.suppress(BaseException):
                await task

    asyncio.run(run())
    assert [chunk.text for chunk in seen] == ["first"]
    assert len(settled) == 1
    receipt = settled[0].receipt
    assert receipt.termination.requested_by == "user_cancel"
    assert receipt.termination.ended_by == "deadline"
    assert receipt.attempts == 1
    assert receipt.usage == {}


@pytest.mark.parametrize("restored_origin", [-1_000_000.0, 1_000_000_000.0])
def test_parked_checkpoint_restores_budget_and_elapsed_at_a_new_clock_origin(
    tmp_path, monkeypatch, restored_origin,
) -> None:
    now = [100.0]
    monkeypatch.setattr(deadline_clock, "monotonic", lambda: now[0])
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = AgentRunSpec(
        workspace_root=workspace, run_root=tmp_path / "runs",
        limits=RunLimits(max_duration_s=60),
    )
    provider = runtime_provider(runtime_config("hitl.request"))
    source = AgentLoop(
        spec=spec, runtime_config_provider=provider,
        model_adapter=FakeModelAdapter(turns=[ModelTurn(
            response_id="first", tool_calls=(
                fake_tool_call("hitl_request", {"prompt": "Pick"}, "human-call"),
            ),
        )]),
    )
    restored_adapter = FakeModelAdapter(turns=[ModelTurn(response_id="second", final_text="answer")])
    restored = AgentLoop(spec=spec, model_adapter=restored_adapter, runtime_config_provider=provider)
    try:
        source.open()
        now[0] += 7
        parked = source.run_until_suspended("ask the human")
        assert parked.reason == "awaiting_tasks"
        checkpoint = source.snapshot()
        assert checkpoint is not None
        assert checkpoint.remaining_duration_s == 53
        blobs = source.collect_checkpoint_blobs()
        saved = checkpoint.to_json()
        source.release_parked()

        # A new process may place its monotonic zero anywhere. Persisted state carries
        # only the remaining duration; the inactive interval consumes no run budget.
        now[0] = restored_origin
        recovered = RunCheckpoint.from_json(saved)
        assert recovered is not None
        restored.restore(recovered, blobs=blobs)
        snapshot = restored.snapshot()
        assert snapshot is not None
        assert snapshot.remaining_duration_s == 53

        restored.report_task_result(parked.awaiting_task_ids[0], {"answer": "reported"})
        now[0] += 3
        resumed = restored.run_until_suspended(None)
        assert resumed.reason == "settled"
        assert resumed.turn is not None
        assert resumed.turn.final_text == "answer"
        assert resumed.turn.metrics["duration_s"] == 10
        snapshot = restored.snapshot()
        assert snapshot is not None
        assert snapshot.remaining_duration_s == 50
        result = restored.close()
        assert result.metrics["duration_s"] == 10
        assert len(restored_adapter.requests) == 1
    finally:
        for loop in (source, restored):
            if loop._session is not None:
                loop.discard_uncommitted()
