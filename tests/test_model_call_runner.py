"""``ModelCallRunner`` -- adapter dispatch, the cancel/deadline race, and the capture receipt."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.model_io import (
    CapturePolicy,
    ModelCallCapture,
    ModelIOSubscription,
)
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.errors import ModelCallAborted, RunCancelled, RunTimeout
from monoid_agent_kernel.model_call import ModelCallRunner
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelTurn,
    TextDelta,
    ToolCallDelta,
    TurnComplete,
)
from monoid_agent_kernel.tools.base import ToolSpec

REQUEST = ModelRequest(instruction="hi", system_prompt="sys", tools=())


class SyncAdapter:
    def next_turn(self, request: ModelRequest) -> ModelTurn:
        del request
        return ModelTurn(response_id="r", final_text="answer")


class CoroutineAdapter:
    async def next_turn(self, request: ModelRequest) -> ModelTurn:
        del request
        return ModelTurn(response_id="r", final_text="answer")


class AsyncAdapter:
    async def anext_turn(self, request: ModelRequest) -> ModelTurn:
        del request
        return ModelTurn(response_id="r", final_text="answer")

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        del request
        return ModelTurn(final_text="one-shot fallback")


class StreamingAdapter:
    def __init__(self, *, chunks: list[Any] | None = None) -> None:
        self.chunks = chunks or [
            TextDelta("ans"),
            TextDelta("wer"),
            TurnComplete(response_id="r", usage={"input_tokens": 3}, stop_reason="stop"),
        ]
        self.closed = False

    async def astream_turn(self, request: ModelRequest):  # noqa: ANN201
        del request
        try:
            for chunk in self.chunks:
                yield chunk
        finally:
            self.closed = True

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        del request
        return ModelTurn(final_text="one-shot fallback")


class RecordingObserver:
    def __init__(self) -> None:
        self.captures: list[ModelCallCapture] = []

    def on_model_call(self, capture: ModelCallCapture) -> None:
        self.captures.append(capture)


# --- dispatch -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adapter", [SyncAdapter(), CoroutineAdapter(), AsyncAdapter(), StreamingAdapter()]
)
def test_every_adapter_shape_reaches_the_same_turn(adapter: Any) -> None:
    """The point of the runner: an adapter's async-ness is not observable in the result.

    ``StreamingAdapter`` is here with no ``delta_consumer``, so it lands on the one-shot path -- the
    shapes agree across the dispatch fork, not merely within one branch of it.
    """

    async def run() -> ModelTurn:
        turn, _receipt = await ModelCallRunner(adapter=adapter).acall(REQUEST)
        return turn

    turn = asyncio.run(run())
    assert turn.final_text in {"answer", "one-shot fallback"}
    assert turn.final_text is not None


def test_anext_turn_is_preferred_over_next_turn() -> None:
    """An adapter exposing both is async-native; falling back would block the event loop."""

    async def run() -> ModelTurn:
        turn, _receipt = await ModelCallRunner(adapter=AsyncAdapter()).acall(REQUEST)
        return turn

    assert asyncio.run(run()).final_text == "answer"


def test_streaming_is_selected_by_the_call_arguments_not_by_adapter_capability() -> None:
    """Path selection is a function of the arguments alone.

    An adapter that *can* stream is still driven one-shot when the caller wants no deltas. This is
    what lets the same runner serve a live stream and a plain call without consulting any state
    outside the call.
    """
    adapter = StreamingAdapter()

    async def one_shot() -> ModelTurn:
        turn, _receipt = await ModelCallRunner(adapter=adapter).acall(REQUEST)
        return turn

    async def streamed() -> tuple[ModelTurn, list[Any]]:
        seen: list[Any] = []
        turn, _receipt = await ModelCallRunner(adapter=adapter).acall(
            REQUEST, delta_consumer=seen.append
        )
        return turn, seen

    assert asyncio.run(one_shot()).final_text == "one-shot fallback"
    turn, seen = asyncio.run(streamed())
    assert turn.final_text == "answer"
    assert len(seen) == 3


def test_every_chunk_reaches_the_consumer_including_non_text() -> None:
    """The runner relays chunks; deciding which ones matter is the consumer's job.

    A runner that filtered would have to know whether it was serving a live stream or an
    event-emitting run, which is exactly the coupling the extraction removes.
    """
    adapter = StreamingAdapter(
        chunks=[
            TextDelta("a"),
            ToolCallDelta(index=0, id="c1", name="t", arguments_fragment="{}"),
            TurnComplete(response_id="r"),
        ]
    )
    seen: list[Any] = []

    async def run() -> None:
        await ModelCallRunner(adapter=adapter).acall(REQUEST, delta_consumer=seen.append)

    asyncio.run(run())
    assert [type(chunk).__name__ for chunk in seen] == [
        "TextDelta",
        "ToolCallDelta",
        "TurnComplete",
    ]


# --- cooperative abort --------------------------------------------------------------------------


def test_a_chunk_is_delivered_before_should_abort_is_polled() -> None:
    """A stop arriving while a chunk is in flight does not retract that chunk.

    The order is observable: it decides whether the text a user already saw stays on screen. It
    stops the chunk *after* the one that was in flight, never the one already handed over.
    """
    adapter = StreamingAdapter(chunks=[TextDelta("one"), TextDelta("two"), TextDelta("three")])
    seen: list[Any] = []

    async def run() -> None:
        await ModelCallRunner(adapter=adapter).acall(
            REQUEST, delta_consumer=seen.append, should_abort=lambda: len(seen) >= 2
        )

    with pytest.raises(ModelCallAborted):
        asyncio.run(run())
    assert [chunk.text for chunk in seen] == ["one", "two"]
    assert adapter.closed is True, "the provider's generator must be closed on abort"


def test_should_abort_is_not_polled_on_the_one_shot_path() -> None:
    """A one-shot call cannot be stopped part-way, so polling would only invite a false stop."""
    polled = []

    async def run() -> ModelTurn:
        turn, _receipt = await ModelCallRunner(adapter=SyncAdapter()).acall(
            REQUEST, should_abort=lambda: polled.append(1) or True
        )
        return turn

    assert asyncio.run(run()).final_text == "answer"
    assert polled == []


def test_a_stream_without_should_abort_runs_to_completion() -> None:
    """The counterweight to the abort tests: no predicate means no stopping."""
    adapter = StreamingAdapter(chunks=[TextDelta("a"), TextDelta("b"), TurnComplete()])
    seen: list[Any] = []

    async def run() -> ModelTurn:
        turn, _receipt = await ModelCallRunner(adapter=adapter).acall(
            REQUEST, delta_consumer=seen.append
        )
        return turn

    assert asyncio.run(run()).final_text == "ab"
    assert len(seen) == 3


def test_a_stream_source_without_aclose_is_driven_anyway() -> None:
    """``astream_turn`` may return a hand-rolled async iterator, not only an async generator.

    Only async *generators* are guaranteed an ``aclose``; the protocol in ``providers/base.py`` asks
    for an ``AsyncIterator``. Closing is best-effort for that reason, and an adapter that cannot be
    closed must still be drivable rather than crashing on cleanup.
    """

    class BareIterator:
        def __init__(self) -> None:
            self.remaining = [TextDelta("ok"), TurnComplete(response_id="r")]

        def __aiter__(self) -> BareIterator:
            return self

        async def __anext__(self) -> Any:
            if not self.remaining:
                raise StopAsyncIteration
            return self.remaining.pop(0)

    class BareStreamAdapter:
        def astream_turn(self, request: ModelRequest) -> Any:
            del request
            return BareIterator()

    seen: list[Any] = []

    async def run() -> ModelTurn:
        turn, _receipt = await ModelCallRunner(adapter=BareStreamAdapter()).acall(
            REQUEST, delta_consumer=seen.append
        )
        return turn

    assert asyncio.run(run()).final_text == "ok"
    assert len(seen) == 2


# --- cancellation and deadline ------------------------------------------------------------------


class SlowAdapter:
    async def anext_turn(self, request: ModelRequest) -> ModelTurn:
        del request
        await asyncio.sleep(5)
        return ModelTurn(final_text="too late")


def test_the_deadline_bounds_a_slow_adapter() -> None:
    async def run() -> None:
        await ModelCallRunner(adapter=SlowAdapter()).acall(REQUEST, deadline=time.time() + 0.05)

    started = time.monotonic()
    with pytest.raises(RunTimeout):
        asyncio.run(run())
    assert time.monotonic() - started < 2.0


def test_cancellation_releases_a_slow_adapter() -> None:
    async def run() -> None:
        token = CancellationToken()
        asyncio.get_running_loop().call_later(0.05, token.cancel)
        await ModelCallRunner(
            adapter=SlowAdapter(), current_cancellation_token=lambda: token
        ).acall(REQUEST)

    started = time.monotonic()
    with pytest.raises(RunCancelled):
        asyncio.run(run())
    assert time.monotonic() - started < 2.0


def test_the_token_is_read_per_call_not_captured_at_construction() -> None:
    """``AgentLoop.astream`` installs a token on a run already in progress.

    A runner holding the token it saw at construction would watch one nobody cancels, and
    cancellation would be silently lost on exactly the path that streams to a user.
    """
    holder: dict[str, CancellationToken | None] = {"token": None}
    runner = ModelCallRunner(
        adapter=SlowAdapter(), current_cancellation_token=lambda: holder["token"]
    )

    async def run() -> None:
        token = CancellationToken()
        holder["token"] = token  # installed *after* the runner was built
        asyncio.get_running_loop().call_later(0.05, token.cancel)
        await runner.acall(REQUEST)

    with pytest.raises(RunCancelled):
        asyncio.run(run())


def test_a_run_told_to_stop_does_not_report_a_turn_it_happened_to_finish() -> None:
    """Cancellation is checked before the completed result is read, not after.

    Without that ordering a call settling in the same tick as the cancel would return a turn the
    caller already gave up on, and the run would record work it had decided not to do.
    """

    class InstantAdapter:
        async def anext_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(final_text="finished anyway")

    token = CancellationToken()
    token.cancel()

    async def run() -> None:
        await ModelCallRunner(
            adapter=InstantAdapter(), current_cancellation_token=lambda: token
        ).acall(REQUEST)

    with pytest.raises(RunCancelled):
        asyncio.run(run())


def test_a_blocking_adapter_is_abandoned_rather_than_awaited() -> None:
    """A sync ``next_turn`` cannot be interrupted, so the deadline abandons its thread."""
    class WedgedAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            time.sleep(3)
            return ModelTurn(final_text="eventually")

    async def run() -> None:
        await ModelCallRunner(adapter=WedgedAdapter(), cancel_grace_s=0.05).acall(
            REQUEST, deadline=time.time() + 0.05
        )

    started = time.monotonic()
    with pytest.raises(RunTimeout):
        asyncio.run(run())
    assert time.monotonic() - started < 2.0, "the run must not wait for the wedged worker"


# --- receipts -----------------------------------------------------------------------------------


def test_the_prompt_digest_ignores_what_surrounds_the_prompt() -> None:
    """``prompt_digest`` answers "did the model see the same conversation twice".

    Adding a tool to the surface or changing a generation setting must not perturb it, or the
    question it answers becomes unanswerable across ordinary configuration drift.
    """
    spec = ToolSpec(
        id="t.one",
        description="d",
        input_schema={"type": "object"},
        capability="read",
        side_effect="read",
        handler=lambda **kwargs: None,
    )
    plain = ModelRequest(instruction="hi", system_prompt="sys", tools=())
    with_tool = ModelRequest(instruction="hi", system_prompt="sys", tools=(spec,))
    with_model = ModelRequest(
        instruction="hi", system_prompt="sys", tools=(), model=ModelConfig(model="other")
    )

    async def digests(request: ModelRequest) -> tuple[str, str]:
        _turn, receipt = await ModelCallRunner(adapter=SyncAdapter()).acall(request)
        return receipt.prompt_digest, receipt.request_digest

    plain_prompt, plain_request = asyncio.run(digests(plain))
    tool_prompt, tool_request = asyncio.run(digests(with_tool))
    model_prompt, model_request = asyncio.run(digests(with_model))

    assert tool_prompt == plain_prompt
    assert model_prompt == plain_prompt
    # ...but the replay key does distinguish them, or replay would reuse the wrong call.
    assert tool_request != plain_request
    assert model_request != plain_request


def test_a_different_prompt_changes_both_digests() -> None:
    """The counterweight: a digest that never changes would pass the test above."""

    async def digests(instruction: str) -> tuple[str, str]:
        request = ModelRequest(instruction=instruction, system_prompt="sys", tools=())
        _turn, receipt = await ModelCallRunner(adapter=SyncAdapter()).acall(request)
        return receipt.prompt_digest, receipt.request_digest

    first = asyncio.run(digests("hi"))
    second = asyncio.run(digests("bye"))
    assert first[0] != second[0]
    assert first[1] != second[1]


def test_the_receipt_carries_the_invocation_context() -> None:
    context = InvocationContext(run_id="run-1", step_id="step-2", attempt=3)

    async def run() -> Any:
        _turn, receipt = await ModelCallRunner(adapter=SyncAdapter()).acall(
            REQUEST, context=context
        )
        return receipt

    receipt = asyncio.run(run())
    assert receipt.context.run_id == "run-1"
    assert receipt.context.step_id == "step-2"
    assert receipt.context.attempt == 3


def test_a_failed_call_still_produces_a_receipt_for_its_observers() -> None:
    """A failed call is precisely the one an audit trail needs; it must not be the one it loses."""

    class BrokenAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            raise RuntimeError("provider exploded")

    observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=BrokenAdapter(),
            subscriptions=(ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),),
        ).acall(REQUEST)

    with pytest.raises(RuntimeError):
        asyncio.run(run())
    assert len(observer.captures) == 1
    receipt = observer.captures[0].receipt
    assert receipt.succeeded is False
    assert receipt.error_code == "RuntimeError"


def test_observers_see_the_settled_call() -> None:
    observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=SyncAdapter(),
            subscriptions=(ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="full")),),
        ).acall(REQUEST)

    asyncio.run(run())
    assert len(observer.captures) == 1
    capture = observer.captures[0]
    assert capture.mode == "full"
    assert capture.content is not None
    assert capture.content["output_text"] == "answer"
    assert capture.content["system_prompt"] == "sys"
    assert capture.receipt.succeeded is True


def test_a_broken_observer_does_not_fail_a_call_the_provider_was_paid_for() -> None:
    class ExplodingObserver:
        def on_model_call(self, capture: ModelCallCapture) -> None:
            del capture
            raise RuntimeError("exporter is down")

    healthy = RecordingObserver()

    async def run() -> ModelTurn:
        turn, _receipt = await ModelCallRunner(
            adapter=SyncAdapter(),
            subscriptions=(
                ModelIOSubscription(observer=ExplodingObserver(), policy=CapturePolicy(mode="full")),
                ModelIOSubscription(observer=healthy, policy=CapturePolicy(mode="full")),
            ),
        ).acall(REQUEST)
        return turn

    assert asyncio.run(run()).final_text == "answer"
    assert len(healthy.captures) == 1


def test_a_malformed_usage_count_is_dropped_rather_than_failing_the_call() -> None:
    """The receipt refuses a negative count. Refusing it must not undo a call already billed.

    Counterweight in the same assertion: the well-formed counters still land, so "drop everything"
    does not pass.
    """

    class OddUsageAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(
                final_text="answer",
                usage={"input_tokens": 5, "output_tokens": -3, "cached": True},  # type: ignore[dict-item]
            )

    async def run() -> Any:
        _turn, receipt = await ModelCallRunner(adapter=OddUsageAdapter()).acall(REQUEST)
        return receipt

    receipt = asyncio.run(run())
    assert dict(receipt.usage) == {"input_tokens": 5}


def test_the_receipt_records_the_stop_reason_and_latency() -> None:
    async def run() -> Any:
        adapter = StreamingAdapter()
        _turn, receipt = await ModelCallRunner(adapter=adapter).acall(
            REQUEST, delta_consumer=lambda chunk: None
        )
        return receipt

    receipt = asyncio.run(run())
    assert receipt.stop_reason == "stop"
    assert receipt.latency_ms >= 0
    assert dict(receipt.usage)["input_tokens"] == 3


def test_no_subscriptions_means_no_capture_work() -> None:
    """Capture is opt-in. A runner wired to nothing still returns a usable receipt."""

    async def run() -> Any:
        _turn, receipt = await ModelCallRunner(adapter=SyncAdapter()).acall(REQUEST)
        return receipt

    receipt = asyncio.run(run())
    assert receipt.capture_downgrades == 0
    assert receipt.prompt_digest != ""
