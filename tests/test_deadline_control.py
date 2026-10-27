"""Public deadline behavior with a moving wall clock and a live asyncio clock."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from support.fenced_hosting import DeterministicFencedRunHarness

from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.model_io import CapturePolicy, ModelIOSubscription
from monoid_agent_kernel.core.output_validator import ValidationOutcome
from monoid_agent_kernel.core.spec import ModelConfig, ModelRetryConfig
from monoid_agent_kernel.errors import ModelAdapterError, ModelDispatchRefused, RunCancelled, RunTimeout
from monoid_agent_kernel.hosting.model_calls import FencedModelCallLifecycle
from monoid_agent_kernel.model_call import ModelCallRunner
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelTurn,
    TextDelta,
    TurnComplete,
    mark_provider_usage,
)
from monoid_agent_kernel.providers.fake import FakeModelAdapter
from monoid_agent_kernel.validated_call import ValidatedCallRunner

pytestmark = pytest.mark.unit

REQUEST = ModelRequest(instruction="answer", system_prompt="system", tools=())
BILL = {"input_tokens": 0, "output_tokens": 3, "total_tokens": 3}
REPAIR_BILL = {"input_tokens": 0, "output_tokens": 7, "total_tokens": 7}


@pytest.fixture
def wall_shift(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Only the epoch view moves; asyncio, thread waits, and watchdogs keep real time."""
    real_time = time.time
    shift = [0.0]
    monkeypatch.setattr(time, "time", lambda: real_time() + shift[0])
    return shift


async def _release_task(task: asyncio.Task[Any]) -> None:
    if not task.done():
        task.cancel()
    with contextlib.suppress(BaseException):
        await task


@pytest.mark.parametrize("shape", ["native", "sync", "sync_awaitable"])
@pytest.mark.parametrize("shift", [0.0, 3600.0])
def test_public_epoch_call_survives_a_wall_jump_after_dispatch(shape, shift, wall_shift) -> None:
    settled = []
    calls = []

    async def run():
        loop = asyncio.get_running_loop()
        entered, release, worker_done = (asyncio.Event() for _ in range(3))
        sync_release = threading.Event()

        async def answer(request):
            calls.append(request)
            entered.set()
            await release.wait()
            return ModelTurn(final_text="answer", usage=BILL)

        def sync_answer(request):
            calls.append(request)
            loop.call_soon_threadsafe(entered.set)
            try:
                assert sync_release.wait(5), "test did not release the provider worker"
                return ModelTurn(final_text="answer", usage=BILL)
            finally:
                loop.call_soon_threadsafe(worker_done.set)

        adapter = (
            SimpleNamespace(anext_turn=answer) if shape == "native" else
            SimpleNamespace(next_turn=sync_answer) if shape == "sync" else
            SimpleNamespace(next_turn=lambda request: answer(request))
        )
        runner = ModelCallRunner(adapter=adapter, settled_sink=settled.append)
        task = asyncio.create_task(runner.acall(REQUEST, deadline=time.time() + 60))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            wall_shift[0] = shift
            release.set()
            sync_release.set()
            return await asyncio.wait_for(task, 3)
        finally:
            release.set()
            sync_release.set()
            await _release_task(task)
            if shape == "sync" and calls:
                await asyncio.wait_for(worker_done.wait(), 3)

    turn, receipt = asyncio.run(run())
    assert turn.final_text == "answer"
    assert len(calls) == 1
    assert receipt.attempts == 1
    assert receipt.usage == BILL
    assert receipt.termination.outcome == "completed"
    assert [call.receipt for call in settled] == [receipt]


def _retry_request(max_attempts: int) -> ModelRequest:
    return replace(REQUEST, model=ModelConfig(retry=ModelRetryConfig(
        layer="kernel", max_attempts=max_attempts, initial_delay_s=1, max_delay_s=1, jitter_s=0,
    )))


@pytest.mark.parametrize("recovered", [False, True])
@pytest.mark.parametrize("shift", [0.0, -3600.0])
def test_retry_that_cannot_fit_keeps_the_provider_error_and_single_bill(
    recovered, shift, wall_shift, monkeypatch,
) -> None:
    """The old wall-clock fit check permits a second paid dispatch after a rollback."""
    settled = []
    calls = []
    refused = ModelDispatchRefused("busy", error_code="provider_busy", retryable=True)
    mark_provider_usage(refused, BILL)

    async def answer(request):
        calls.append(request)
        if len(calls) == 1:
            if not recovered:
                wall_shift[0] = shift
            raise refused
        return ModelTurn(final_text="extra paid dispatch", usage=REPAIR_BILL)

    harness = DeterministicFencedRunHarness()
    token = harness.claim_writer("deadline-retry", "worker")
    hook = FencedModelCallLifecycle(harness.sink, token) if recovered else None
    runner = ModelCallRunner(
        adapter=SimpleNamespace(anext_turn=answer), lifecycle_hook=hook, settled_sink=settled.append,
    )
    logical_id = "logical-deadline-retry" if recovered else ""

    async def run():
        historical = None
        if recovered:
            with pytest.raises(ModelDispatchRefused):
                await runner.acall(_retry_request(1), logical_call_id=logical_id)
            historical = harness.sink.load_invocation(token.run_id, logical_id).value.invocation
            original_recover = hook.recover

            def recover(query):
                result = original_recover(query)
                wall_shift[0] = shift
                return result

            monkeypatch.setattr(hook, "recover", recover)
            settled.clear()
        with pytest.raises(ModelAdapterError) as caught:
            await asyncio.wait_for(runner.acall(
                _retry_request(2), deadline=time.time() + 0.5, logical_call_id=logical_id,
            ), 3)
        assert caught.value.error_code == "provider_busy"
        assert caught.value.retryable is True
        if recovered:
            current = harness.sink.load_invocation(token.run_id, logical_id).value.invocation
            assert current.to_json() == historical.to_json()

    asyncio.run(run())
    assert len(calls) == 1
    assert len(settled) == 1
    assert settled[0].receipt.attempts == 1
    assert settled[0].receipt.usage == BILL


@pytest.mark.parametrize(("shift", "exhaust"), [
    (0.0, False), (3600.0, False), (0.0, True), (-3600.0, True),
])
def test_validated_repairs_reuse_the_first_call_budget(shift, exhaust, wall_shift) -> None:
    adapter = FakeModelAdapter(turns=[
        ModelTurn(final_text="reject", usage=BILL),
        ModelTurn(final_text="accept", usage=REPAIR_BILL),
    ])
    settled = []

    async def run():
        loop = asyncio.get_running_loop()
        validating = asyncio.Event()
        release = threading.Event()

        class Validator:
            id = "accept"

            def validate(self, view):
                if view.final_text == "accept":
                    return ValidationOutcome(ok=True, value="accepted")
                wall_shift[0] = shift
                loop.call_soon_threadsafe(validating.set)
                assert release.wait(5), "test did not release validation"
                return ValidationOutcome(ok=False, feedback="answer accept")

        runner = ValidatedCallRunner(
            ModelCallRunner(adapter=adapter, settled_sink=settled.append),
            validators=(Validator(),), max_repair_calls=1,
        )
        task = asyncio.create_task(runner.acall(
            REQUEST, deadline=time.time() + (0.15 if exhaust else 60),
        ))
        timer = None
        try:
            await asyncio.wait_for(validating.wait(), 3)
            if exhaust:
                # Schedule from validator entry, after the call's overall clock already started.
                timer = loop.call_later(0.25, release.set)
                with pytest.raises(RunTimeout) as caught:
                    await asyncio.wait_for(task, 3)
                assert [receipt.usage for receipt in caught.value.receipts] == [BILL]
            else:
                release.set()
                result = await asyncio.wait_for(task, 3)
                assert result.status == "ok"
                assert result.value == "accepted"
                assert result.repair_calls_used == 1
                assert [receipt.usage for receipt in result.receipts] == [BILL, REPAIR_BILL]
        finally:
            if timer is not None:
                timer.cancel()
            release.set()
            await _release_task(task)

    asyncio.run(run())
    assert len(adapter.requests) == (1 if exhaust else 2)
    assert sum(call.receipt.attempts for call in settled) == (1 if exhaust else 2)


@pytest.mark.parametrize(("ending", "shift"), [
    ("provider_terminal", 0.0), ("provider_terminal", 3600.0),
    ("drain_timeout", 0.0), ("drain_timeout", -3600.0),
    ("deadline", 0.0), ("deadline", -3600.0),
])
def test_accepted_stop_keeps_its_original_drain_window(ending, shift, wall_shift) -> None:
    token = CancellationToken()
    settled, seen = [], []

    async def run():
        entered, release, closed = (asyncio.Event() for _ in range(3))

        async def stream(request):
            try:
                yield TextDelta("first")
                entered.set()
                await release.wait()
                yield TurnComplete(usage=BILL)
            finally:
                closed.set()

        runner = ModelCallRunner(
            adapter=SimpleNamespace(astream_turn=stream),
            current_cancellation_token=lambda: token,
            abort_drain_s=0.15 if ending == "drain_timeout" else 5,
            cancel_grace_s=0.2, settled_sink=settled.append,
        )
        task = asyncio.create_task(runner.acall(
            REQUEST, delta_consumer=seen.append,
            deadline=time.time() + 0.25 if ending == "deadline" else None,
        ))
        timer = None
        try:
            await asyncio.wait_for(entered.wait(), 3)
            # begin() accepts the stop synchronously. The jump occurs after that first sample.
            token.cancel()
            wall_shift[0] = shift
            timer = asyncio.get_running_loop().call_later(
                0.03 if ending == "provider_terminal" else 0.4, release.set,
            )
            with pytest.raises(RunCancelled):
                await asyncio.wait_for(task, 3)
            await asyncio.wait_for(closed.wait(), 3)
        finally:
            if timer is not None:
                timer.cancel()
            release.set()
            await _release_task(task)

    asyncio.run(run())
    assert [chunk.text for chunk in seen] == ["first"]
    assert len(settled) == 1
    receipt = settled[0].receipt
    assert receipt.termination.requested_by == "user_cancel"
    assert receipt.termination.ended_by == ending
    assert receipt.usage == (BILL if ending == "provider_terminal" else {})
    assert receipt.attempts == 1


@pytest.mark.parametrize("validated", [False, True])
@pytest.mark.parametrize("deadline", [
    pytest.param(float("nan"), id="nan"),
    pytest.param(True, id="true"), pytest.param(False, id="false"),
    pytest.param("123", id="text"), pytest.param(object(), id="object"),
    pytest.param(10**400, id="overflow"),
])
def test_invalid_public_deadline_precedes_getters_invocation_and_receipts(validated, deadline) -> None:
    effects, settled, captures, attempt_events = [], [], [], []
    harness = DeterministicFencedRunHarness()
    token = harness.claim_writer("invalid-deadline", "worker")

    class Hook(FencedModelCallLifecycle):
        def recover(self, query):
            effects.append("recover")
            return super().recover(query)

    def answer(request):
        effects.append("provider")
        return ModelTurn(final_text="unexpected")

    adapter = SimpleNamespace(next_turn=answer)

    def current_adapter():
        effects.append("getter")
        return adapter

    hook = Hook(harness.sink, token)
    runner = ModelCallRunner(
        adapter=adapter, current_adapter=current_adapter,
        # The validated API has no durable logical_call_id. Its row verifies the
        # getter, attempt marker and sinks; the direct row verifies durable ingress too.
        lifecycle_hook=None if validated else hook,
        subscriptions=(ModelIOSubscription(
            observer=SimpleNamespace(on_model_call=captures.append), policy=CapturePolicy(mode="none"),
        ),), settled_sink=settled.append,
    )

    async def run():
        call = (
            ValidatedCallRunner(runner).acall(
                REQUEST, deadline=deadline,
                delta_consumer=lambda *event: attempt_events.append(event),
            ) if validated else runner.acall(
                REQUEST, deadline=deadline, logical_call_id="invalid-deadline-call",
                before_dispatch=lambda: effects.append("before_dispatch"),
            )
        )
        with pytest.raises(ValueError):
            await asyncio.wait_for(call, 3)

    asyncio.run(run())
    assert effects == []
    assert attempt_events == []
    assert captures == []
    assert settled == []
    assert hook.last_invocation is None
    assert harness.sink.load_invocation(token.run_id, "invalid-deadline-call").status == "missing"


@pytest.mark.parametrize("validated", [False, True])
@pytest.mark.parametrize(("value", "expired"), [
    (None, False), (float("inf"), False), ("future_float", False), ("future_int", False),
    ("past", True), (0, True), (float("-inf"), True),
])
def test_public_epoch_deadline_controls_keep_their_meaning(validated, value, expired) -> None:
    deadline = (
        time.time() + 60 if value == "future_float" else
        int(time.time()) + 60 if value == "future_int" else
        time.time() - 60 if value == "past" else value
    )
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="answer", usage=BILL)])
    settled = []
    runner = ModelCallRunner(adapter=adapter, settled_sink=settled.append)

    async def run():
        call = (
            ValidatedCallRunner(runner).acall(REQUEST, deadline=deadline)
            if validated else runner.acall(REQUEST, deadline=deadline)
        )
        if expired:
            with pytest.raises(RunTimeout):
                await asyncio.wait_for(call, 3)
        else:
            result = await asyncio.wait_for(call, 3)
            turn = result.turn if validated else result[0]
            assert turn.final_text == "answer"

    asyncio.run(run())
    assert len(adapter.requests) == (0 if expired else 1)
    assert len(settled) == 1
    assert settled[0].receipt.attempts == (0 if expired else 1)
    assert settled[0].receipt.usage == ({} if expired else BILL)
