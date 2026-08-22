"""``ModelCallRunner`` -- adapter dispatch, the cancel/deadline race, and the capture receipt."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import threading
import time
from urllib.error import URLError
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from monoid_agent_kernel.core._sync_bridge import CalleeCancelled
from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.model_io import (
    CapturePolicy,
    ModelCallCapture,
    ModelIOSubscription,
)
from monoid_agent_kernel.core.model_invocation import DurableModelInvocation
from monoid_agent_kernel.core.outcome import InterruptionCause
from monoid_agent_kernel.core.spec import ModelConfig, ModelRetryConfig
from monoid_agent_kernel.model_call import ShouldAbort
from monoid_agent_kernel.core.streaming import QueueEventSink
from monoid_agent_kernel.errors import (
    DurableModelCallError,
    ModelAdapterError,
    ModelDispatchRefused,
    ModelCallAborted,
    RunCancelled,
    RunTimeout,
)
from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.model_call import (
    ModelCallRunner,
    SettledModelCall,
    _DigestResult,
    _digest,
    _encoded_digest,
    _prompt_payload,
    _request_payload,
)
from monoid_agent_kernel.model_lifecycle import (
    ModelDispatchReservation,
    ModelDispatchSettlement,
    UnknownModelDispatch,
    dispatch_evidence,
)
from support.fenced_hosting import DeterministicFencedRunHarness
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelTurn,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolObservation,
    TurnComplete,
    assemble_streamed_turn,
    collect_retry_reports,
    mark_provider_usage,
    normalize_model_request,
    provider_usage_of,
    report_provider_retried,
)
from monoid_agent_kernel.providers.gateway import GatewayModelAdapter
from monoid_agent_kernel.tools.base import ToolSpec

REQUEST = ModelRequest(instruction="hi", system_prompt="sys", tools=())


def _raises(self: Any, *args: Any, **kwargs: Any) -> Any:
    del self, args, kwargs
    raise RuntimeError("container hook exploded")


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


class _AsyncCallableNextTurn:
    """``next_turn`` as an object with an async ``__call__`` -- a wrapper or middleware layer."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        del request
        self.calls += 1
        return ModelTurn(response_id="r", final_text="answer")


class CallableObjectAdapter:
    def __init__(self) -> None:
        self.next_turn = _AsyncCallableNextTurn()


class AwaitableReturningAdapter:
    """A synchronous ``next_turn`` that hands back an awaitable -- it delegates to an async client."""

    def __init__(self) -> None:
        self.calls = 0

    def next_turn(self, request: ModelRequest) -> Any:
        del request
        self.calls += 1

        async def answer() -> ModelTurn:
            return ModelTurn(response_id="r", final_text="answer")

        return answer()


class RecordingObserver:
    def __init__(self) -> None:
        self.captures: list[ModelCallCapture] = []

    def on_model_call(self, capture: ModelCallCapture) -> None:
        self.captures.append(capture)


# --- dispatch -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adapter",
    [
        SyncAdapter(),
        CoroutineAdapter(),
        AsyncAdapter(),
        StreamingAdapter(),
        CallableObjectAdapter(),
        AwaitableReturningAdapter(),
    ],
)
def test_every_adapter_shape_reaches_the_same_turn(adapter: Any) -> None:
    """The point of the runner: an adapter's async-ness is not observable in the result.

    ``StreamingAdapter`` is here with no ``delta_consumer``, so it lands on the one-shot path -- the
    shapes agree across the dispatch fork, not merely within one branch of it.

    The last two shapes are the ones a predicate over *functions* misses. `iscoroutinefunction` says
    no to an object whose `__call__` is async, and cannot say anything about a synchronous callable
    that returns an awaitable -- both were run on the sync worker and the awaitable they produced was
    handed back as the turn. `isinstance` is the assertion that catches it: nothing downstream reads
    a coroutine as a failure, since every receipt field read is defensive, so the call was recorded
    as a success for a provider that had never been invoked.
    """

    async def run() -> ModelTurn:
        turn, _receipt = await ModelCallRunner(adapter=adapter).acall(REQUEST)
        return turn

    turn = asyncio.run(run())
    assert isinstance(turn, ModelTurn), f"the dispatch returned a {type(turn).__name__}, not a turn"
    assert turn.final_text in {"answer", "one-shot fallback"}
    assert turn.final_text is not None


def test_one_shot_normalizes_request_and_turn_before_adapter_receipt_and_observer() -> None:
    class HostileAdapter:
        def __init__(self) -> None:
            self.request: ModelRequest | None = None

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            self.request = request
            return ModelTurn(
                response_id="r\ud800",
                final_text="answer\udc00",
                tool_calls=(
                    ToolCall(
                        id="call\ud800",
                        name="tool\udc00",
                        arguments={"text": "\ud800", "number": float("nan")},
                    ),
                ),
                usage={"bad": float("inf")},  # type: ignore[dict-item]
                raw={"value": -float("inf")},
                reasoning=({"summary": "\ud83d\ude00", "score": float("nan")},),
            )

    original = ModelRequest(
        instruction="prompt\ud800",
        system_prompt="system\udc00",
        tools=(),
        messages=({"role": "user", "content": "\ud800", "number": float("nan")},),
    )
    adapter = HostileAdapter()
    observer = RecordingObserver()

    async def run() -> tuple[ModelTurn, Any]:
        return await ModelCallRunner(
            adapter=adapter,
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="full")),
            ),
        ).acall(original)

    turn, receipt = asyncio.run(run())

    assert adapter.request is not None
    assert adapter.request.instruction == "prompt�"
    assert adapter.request.system_prompt == "system�"
    assert adapter.request.messages == ({"role": "user", "content": "�", "number": None},)
    assert original.instruction == "prompt\ud800"
    assert receipt.prompt_digest == _digest(_prompt_payload(normalize_model_request(original)))
    assert turn.final_text == "answer�"
    assert turn.tool_calls[0].arguments == {"text": "�", "number": None}
    assert turn.raw == {"value": None}
    assert turn.reasoning == ({"summary": "😀", "score": None},)
    rendered_capture = json.dumps(observer.captures[0].content, allow_nan=False, ensure_ascii=False)
    rendered_capture.encode("utf-8")


def test_stream_normalizes_each_visible_chunk_and_the_assembled_turn() -> None:
    adapter = StreamingAdapter(
        chunks=[
            TextDelta("answer\ud800"),
            ReasoningDelta("reason\udc00"),
            ToolCallDelta(
                index=0,
                id="call\ud800",
                name="tool\udc00",
                arguments_fragment='{"value": NaN, "text": "\\ud800"}',
            ),
            TurnComplete(
                response_id="r\ud800",
                usage={"input_tokens": 3},
                reasoning=({"summary": "\ud800", "score": float("nan")},),
            ),
        ]
    )
    seen: list[Any] = []

    async def run() -> ModelTurn:
        turn, _receipt = await ModelCallRunner(adapter=adapter).acall(
            REQUEST, delta_consumer=seen.append
        )
        return turn

    turn = asyncio.run(run())

    text = "".join(chunk.text for chunk in seen if isinstance(chunk, TextDelta))
    reasoning = "".join(chunk.text for chunk in seen if isinstance(chunk, ReasoningDelta))
    tool_chunks = [chunk for chunk in seen if isinstance(chunk, ToolCallDelta)]
    terminal = [chunk for chunk in seen if isinstance(chunk, TurnComplete)]
    assert text == "answer�"
    assert reasoning == "reason�"
    assert tool_chunks == [
        ToolCallDelta(
            index=0,
            id="call�",
            name="tool�",
            arguments_fragment='{"value": NaN, "text": "\\ud800"}',
        )
    ]
    assert terminal[0].usage == {"input_tokens": 3, "output_tokens": 0, "total_tokens": 0}
    assert terminal[0].reasoning == ({"summary": "�", "score": None},)
    assert turn.response_id == "r�"
    assert turn.tool_calls[0].arguments == {"value": None, "text": "�"}


def test_stream_preserves_surrogate_pairs_across_interleaved_logical_channels() -> None:
    high, low = "\ud83d", "\ude00"
    adapter = StreamingAdapter(
        chunks=[
            TextDelta(high),
            ReasoningDelta(high),
            ToolCallDelta(index=0, id="c", name="t", arguments_fragment='{"emoji":"' + high),
            TextDelta(low),
            ReasoningDelta(low),
            ToolCallDelta(index=0, arguments_fragment=low + '"}'),
            TurnComplete(),
        ]
    )
    seen: list[Any] = []

    async def run() -> ModelTurn:
        return (await ModelCallRunner(adapter=adapter).acall(REQUEST, delta_consumer=seen.append))[
            0
        ]

    turn = asyncio.run(run())

    assert "".join(chunk.text for chunk in seen if isinstance(chunk, TextDelta)) == "😀"
    assert "".join(chunk.text for chunk in seen if isinstance(chunk, ReasoningDelta)) == "😀"
    assert (
        "".join(chunk.arguments_fragment for chunk in seen if isinstance(chunk, ToolCallDelta))
        == '{"emoji":"😀"}'
    )
    assert turn.final_text == "😀"
    assert turn.tool_calls[0].arguments == {"emoji": "😀"}


def test_stream_flushes_lone_surrogates_in_original_global_order() -> None:
    adapter = StreamingAdapter(
        chunks=[
            ReasoningDelta("\ud800"),
            TextDelta("\ud800"),
            ToolCallDelta(index=2, arguments_fragment="\ud800"),
            TurnComplete(),
        ]
    )
    seen: list[Any] = []

    async def run() -> None:
        await ModelCallRunner(adapter=adapter).acall(REQUEST, delta_consumer=seen.append)

    with pytest.raises(ModelAdapterError):
        asyncio.run(run())
    suffix = [
        chunk
        for chunk in seen
        if getattr(chunk, "text", "") == "�" or getattr(chunk, "arguments_fragment", "") == "�"
    ]
    assert [type(chunk) for chunk in suffix] == [ReasoningDelta, TextDelta, ToolCallDelta]


def test_structural_turn_and_custom_init_subclasses_keep_compatibility() -> None:
    class CustomRequest(ModelRequest):
        def __init__(self, instruction: str) -> None:
            super().__init__(instruction=instruction, system_prompt="sys", tools=())

    class CustomTurn(ModelTurn):
        def __init__(self, text: str) -> None:
            super().__init__(final_text=text)

    class Adapter:
        request: ModelRequest | None = None

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            self.request = request
            return CustomTurn("answer\ud800")

    adapter = Adapter()
    turn, _receipt = asyncio.run(ModelCallRunner(adapter=adapter).acall(CustomRequest("hi\ud800")))

    assert isinstance(adapter.request, CustomRequest)
    assert adapter.request.instruction == "hi�"
    assert isinstance(turn, CustomTurn)
    assert turn.final_text == "answer�"

    class StructuralTurn:
        final_text = "duck\ud800"
        response_id = None
        tool_calls = ()
        usage = {}
        raw = {}
        reasoning = ()
        stop_reason = "stop\ud800"
        provider_retried = False

    class StructuralAdapter:
        def next_turn(self, request: ModelRequest) -> Any:
            del request
            return StructuralTurn()

    structural, receipt = asyncio.run(ModelCallRunner(adapter=StructuralAdapter()).acall(REQUEST))
    assert structural.final_text == "duck�"
    assert receipt.stop_reason == "stop�"


def test_ingress_rejection_keeps_an_attempt_zero_receipt_and_boundary_precedence() -> None:
    class CountingAdapter:
        calls = 0

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            self.calls += 1
            return ModelTurn(final_text="never")

    request = replace(REQUEST, messages=({chr(0xD800): 1, chr(0xFFFD): 2},))
    adapter = CountingAdapter()
    observer = RecordingObserver()
    runner = ModelCallRunner(
        adapter=adapter,
        subscriptions=(
            ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
        ),
    )

    with pytest.raises(ValueError, match="keys collide"):
        asyncio.run(runner.acall(request))
    assert adapter.calls == 0
    assert observer.captures[0].receipt.attempts == 0

    token = CancellationToken()
    token.cancel()
    cancelled_observer = RecordingObserver()
    cancelled_runner = ModelCallRunner(
        adapter=adapter,
        current_cancellation_token=lambda: token,
        subscriptions=(
            ModelIOSubscription(
                observer=cancelled_observer,
                policy=CapturePolicy(mode="digest"),
            ),
        ),
    )
    with pytest.raises(RunCancelled):
        asyncio.run(cancelled_runner.acall(request))
    assert cancelled_observer.captures[0].receipt.error_code == "cancelled"


def test_nonfinite_model_controls_are_rejected_before_dispatch() -> None:
    class CountingAdapter:
        calls = 0

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            self.calls += 1
            return ModelTurn(final_text="never")

    adapter = CountingAdapter()
    retry = replace(ModelConfig().retry, max_attempts=float("nan"))  # type: ignore[arg-type]
    request = replace(REQUEST, model=replace(ModelConfig(), retry=retry))

    with pytest.raises(ValueError, match="max_attempts"):
        asyncio.run(ModelCallRunner(adapter=adapter).acall(request))
    assert adapter.calls == 0


@pytest.mark.parametrize(
    "config",
    [
        replace(ModelConfig(), provider=float("nan")),  # type: ignore[arg-type]
        replace(
            ModelConfig(),
            retry=replace(ModelConfig().retry, retry_on=(float("nan"),)),  # type: ignore[arg-type]
        ),
    ],
)
def test_nonfinite_model_text_controls_leave_a_portable_attempt_zero_receipt(
    config: ModelConfig,
) -> None:
    class CountingAdapter:
        calls = 0

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            self.calls += 1
            return ModelTurn(final_text="never")

    adapter = CountingAdapter()
    observer = RecordingObserver()
    runner = ModelCallRunner(
        adapter=adapter,
        subscriptions=(
            ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
        ),
    )

    with pytest.raises(ValueError):
        asyncio.run(runner.acall(replace(REQUEST, model=config)))

    assert adapter.calls == 0
    assert observer.captures[0].receipt.attempts == 0
    json.dumps(observer.captures[0].receipt.to_json(), allow_nan=False)


@pytest.mark.parametrize(
    "retry_on",
    [
        ("",),
        "gateway_timeout",
    ],
)
def test_invalid_direct_retry_codes_are_rejected_before_dispatch(retry_on: object) -> None:
    class CountingAdapter:
        calls = 0

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            self.calls += 1
            return ModelTurn(final_text="never")

    adapter = CountingAdapter()
    retry = replace(ModelConfig().retry, retry_on=retry_on)  # type: ignore[arg-type]
    request = replace(REQUEST, model=replace(ModelConfig(), retry=retry))

    with pytest.raises(ValueError, match="model.retry.retry_on"):
        asyncio.run(ModelCallRunner(adapter=adapter).acall(request))
    assert adapter.calls == 0


def test_normalized_adapter_fallback_config_is_the_config_actually_dispatched() -> None:
    class Adapter:
        config = replace(ModelConfig(), model="fallback\ud800")
        seen_model: ModelConfig | None = None

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            self.seen_model = request.model
            return ModelTurn(final_text="done")

    adapter = Adapter()

    _turn, receipt = asyncio.run(ModelCallRunner(adapter=adapter).acall(REQUEST))

    assert adapter.seen_model is not None
    assert adapter.seen_model.model == "fallback�"
    assert receipt.model.model == "fallback�"


def test_unconfigured_adapter_keeps_the_optional_request_model_absent() -> None:
    class Adapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            if request.model is not None:
                raise RuntimeError("an absent opt-in config was synthesized")
            return ModelTurn(final_text="done")

    turn, _receipt = asyncio.run(ModelCallRunner(adapter=Adapter()).acall(REQUEST))

    assert turn.final_text == "done"


def test_model_turn_envelope_canonicalization_preserves_a_paid_answer() -> None:
    class Adapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(
                response_id=float("nan"),  # type: ignore[arg-type]
                final_text="done",
                tool_calls=None,  # type: ignore[arg-type]
                usage=None,  # type: ignore[arg-type]
                raw=None,  # type: ignore[arg-type]
                reasoning=None,  # type: ignore[arg-type]
                stop_reason=float("nan"),  # type: ignore[arg-type]
            )

    turn, receipt = asyncio.run(ModelCallRunner(adapter=Adapter()).acall(REQUEST))

    assert turn.final_text == "done"
    assert turn.response_id is None
    assert turn.tool_calls == ()
    assert turn.usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    assert turn.raw == {}
    assert turn.reasoning == ()
    assert turn.stop_reason is None
    json.dumps(receipt.to_json(), allow_nan=False)


def test_nonfinite_required_model_fields_fail_before_persistence() -> None:
    class Adapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(
                tool_calls=(
                    ToolCall(float("nan"), "tool", {}),  # type: ignore[arg-type]
                )
            )

    with pytest.raises(ModelAdapterError, match="non-portable response"):
        asyncio.run(ModelCallRunner(adapter=Adapter()).acall(REQUEST))


@pytest.mark.parametrize("malformed", [None, object(), {"final_text": "not an attribute"}])
def test_non_turn_provider_values_are_rejected(malformed: Any) -> None:
    class Adapter:
        def next_turn(self, request: ModelRequest) -> Any:
            del request
            return malformed

    with pytest.raises(ModelAdapterError, match="non-portable response"):
        asyncio.run(ModelCallRunner(adapter=Adapter()).acall(REQUEST))


def test_nonfinite_stream_text_is_rejected_before_delta_delivery() -> None:
    class Adapter:
        async def astream_turn(self, request: ModelRequest):  # noqa: ANN201
            del request
            yield TextDelta(float("nan"))  # type: ignore[arg-type]

    delivered: list[Any] = []

    with pytest.raises(ModelAdapterError, match="non-portable stream fragment"):
        asyncio.run(
            ModelCallRunner(adapter=Adapter()).acall(
                REQUEST,
                delta_consumer=delivered.append,
            )
        )

    assert delivered == []


@pytest.mark.parametrize(
    "chunk",
    [
        TextDelta("text", provider_retried="false"),  # type: ignore[arg-type]
        ToolCallDelta(index=True, arguments_fragment="{}"),  # type: ignore[arg-type]
        ToolCallDelta(index=0, id=123, arguments_fragment="{}"),  # type: ignore[arg-type]
        ToolCallDelta(index=0, name=False, arguments_fragment="{}"),  # type: ignore[arg-type]
        TurnComplete(stop_reason=123),  # type: ignore[arg-type]
        TurnComplete(provider_retried="false"),  # type: ignore[arg-type]
    ],
)
def test_assemble_streamed_turn_rejects_inexact_raw_chunk_controls(chunk: Any) -> None:
    with pytest.raises(ModelAdapterError, match="non-portable stream fragment"):
        assemble_streamed_turn([chunk])


def test_unknown_stream_fragment_is_rejected_before_delta_delivery() -> None:
    class Adapter:
        async def astream_turn(self, request: ModelRequest):  # noqa: ANN201
            del request
            yield {"bad": float("nan")}  # type: ignore[misc]
            yield TurnComplete()

    delivered: list[Any] = []

    with pytest.raises(ModelAdapterError, match="non-portable stream fragment"):
        asyncio.run(
            ModelCallRunner(adapter=Adapter()).acall(
                REQUEST,
                delta_consumer=delivered.append,
            )
        )

    assert delivered == []


@pytest.mark.parametrize("adapter_factory", [CallableObjectAdapter, AwaitableReturningAdapter])
def test_an_adapter_the_function_predicate_misses_is_still_actually_invoked(
    adapter_factory: Any,
) -> None:
    """The sharper half: the provider has to have been *called*.

    A coroutine handed back unawaited means the adapter body never ran -- no request was ever sent --
    and the receipt still said the call succeeded. That is the worst shape a failure can take here:
    an audit trail recording a provider call that did not happen.
    """
    adapter = adapter_factory()

    async def run() -> tuple[ModelTurn, Any]:
        return await ModelCallRunner(adapter=adapter).acall(REQUEST)

    turn, receipt = asyncio.run(run())
    invoked = getattr(adapter, "calls", None)
    if invoked is None:
        invoked = adapter.next_turn.calls
    assert invoked == 1, "the adapter body never ran, so nothing was asked of the provider"
    assert turn.final_text == "answer"
    assert receipt.succeeded is True


@pytest.mark.parametrize("boundary", ["cancelled", "deadline"])
def test_a_call_refused_before_the_adapter_reports_no_attempt(boundary: str) -> None:
    """`attempts` counts calls the kernel made to the adapter, so a refused call counts none.

    A run already cancelled or past its deadline when the call is requested never reaches the
    adapter. The receipt for that carried the default `attempts=1`, so a consumer summing the field
    counted provider work that provably never happened -- against the receipt's own stated contract.

    The receipt is still written, which is the point: a refused call belongs in the audit trail, and
    the alternative fix -- publishing nothing -- would have removed the only record that a call was
    requested and turned down.
    """

    class CountingAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            self.calls += 1
            return ModelTurn(final_text="answer")

    adapter = CountingAdapter()
    observer = RecordingObserver()
    token = CancellationToken()
    if boundary == "cancelled":
        token.cancel()

    async def run() -> None:
        await ModelCallRunner(
            adapter=adapter,
            current_cancellation_token=lambda: token,
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
            ),
        ).acall(REQUEST, deadline=None if boundary == "cancelled" else time.time() - 1.0)

    with pytest.raises((RunCancelled, RunTimeout)):
        asyncio.run(run())

    assert adapter.calls == 0, "the adapter must not have been reached at all"
    receipt = observer.captures[0].receipt
    assert receipt.attempts == 0, "a call that never reached the adapter reported an adapter call"
    assert receipt.succeeded is False
    assert receipt.error_code in {"cancelled", "run_timeout"}


@pytest.mark.parametrize("outcome", ["succeeded", "failed"])
def test_a_call_that_did_reach_the_adapter_still_counts_one(outcome: str) -> None:
    """The counterweight, so `attempts` does not simply become 0 everywhere.

    Both outcomes, because the downgrade lives on the *failure* path: a counterweight that only
    checked a successful call left "0 attempts for every failure" indistinguishable from the rule,
    and that mutant passed. A provider that was called and then failed was still called.
    """

    class FailingAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            raise ModelAdapterError("provider said no")

    observer = RecordingObserver()
    adapter: Any = SyncAdapter() if outcome == "succeeded" else FailingAdapter()

    async def run() -> None:
        await ModelCallRunner(
            adapter=adapter,
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
            ),
        ).acall(REQUEST)

    if outcome == "succeeded":
        asyncio.run(run())
    else:
        with pytest.raises(ModelAdapterError):
            asyncio.run(run())

    receipt = observer.captures[0].receipt
    assert receipt.succeeded is (outcome == "succeeded")
    assert receipt.attempts == 1, "the adapter was called, whatever it then did"


@pytest.mark.parametrize(
    ("adapter_factory", "expect_worker"),
    [(CallableObjectAdapter, False), (SyncAdapter, True)],
)
def test_an_async_callable_adapter_is_not_sent_to_a_worker_thread(
    monkeypatch: Any, adapter_factory: Any, expect_worker: bool
) -> None:
    """Recognising the async callable, distinct from surviving it.

    The awaitable fallback below the dispatch is what makes a missed async callable *correct*, so a
    test that only checks the turn cannot tell the two defences apart. This one observes the dispatch
    itself: an async callable must never be handed to `start_abandonable_sync_call`, which spawns a
    dedicated daemon thread per call and hands its work back through a bridge the call does not need.
    The synchronous shape is the counterweight -- the worker is where it belongs.
    """
    from monoid_agent_kernel import model_call as model_call_module

    real = model_call_module.start_abandonable_sync_call
    dispatched: list[str] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        dispatched.append(kwargs.get("thread_name", "?"))
        return real(*args, **kwargs)

    monkeypatch.setattr(model_call_module, "start_abandonable_sync_call", spy)

    async def run() -> ModelTurn:
        turn, _receipt = await ModelCallRunner(adapter=adapter_factory()).acall(REQUEST)
        return turn

    turn = asyncio.run(run())
    assert isinstance(turn, ModelTurn)
    assert (
        bool(dispatched) is expect_worker
    ), f"worker dispatch was {'skipped' if not dispatched else 'used'} for this shape"


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

    The predicate counts its own polls, and must not be phrased in terms of what the consumer has
    seen. `len(seen) >= 2` was self-referential: swapping the deliver and the poll shifted both sides
    together and produced the same two chunks, so the mutant that stops the in-flight chunk survived
    the whole suite. Counting polls is the one formulation the swap cannot compensate for.
    """
    adapter = StreamingAdapter(chunks=[TextDelta("one"), TextDelta("two"), TextDelta("three")])
    seen: list[Any] = []
    polls = {"n": 0}

    def should_abort() -> bool:
        polls["n"] += 1
        return polls["n"] >= 2

    async def run() -> None:
        await ModelCallRunner(adapter=adapter).acall(
            REQUEST, delta_consumer=seen.append, should_abort=should_abort
        )

    with pytest.raises(ModelCallAborted):
        asyncio.run(run())
    assert [chunk.text for chunk in seen] == [
        "one",
        "two",
    ], "the chunk whose arrival triggered the stop was retracted instead of delivered"
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


def test_lease_loss_supersedes_an_earlier_stop_before_standalone_publication() -> None:
    token = CancellationToken()
    token.cancel(InterruptionCause.USER_CANCEL)
    token.cancel(InterruptionCause.LEASE_LOST)
    observer = RecordingObserver()
    sidecar: list[SettledModelCall] = []

    with pytest.raises(RunCancelled) as caught:
        asyncio.run(
            ModelCallRunner(
                adapter=SyncAdapter(),
                current_cancellation_token=lambda: token,
                subscriptions=(
                    ModelIOSubscription(
                        observer=observer,
                        policy=CapturePolicy(mode="digest"),
                    ),
                ),
                settled_sink=sidecar.append,
            ).acall(REQUEST)
        )

    assert token.cause is InterruptionCause.USER_CANCEL
    assert caught.value.interruption_cause is InterruptionCause.LEASE_LOST
    assert observer.captures == []
    assert sidecar == []


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


def test_an_adapter_that_cancels_itself_is_reported_as_an_adapter_failure() -> None:
    """A callee's own cancellation is a failure of the call, not the run being stopped.

    The shared race raises ``CalleeCancelled`` so each of its two callers can name it; the tool path
    calls it ``tool_handler_cancelled``. Untranslated here it fell to the loop's generic handler,
    which rewraps with ``str(exc)`` -- and ``CalleeCancelled`` carries no message, so the run failed
    with an empty one. Checked on every dispatch shape because one funnel serves all of them.
    """

    class SelfCancellingAsync:
        async def anext_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            raise asyncio.CancelledError

    class SelfCancellingSync:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            raise asyncio.CancelledError

    class SelfCancellingStream:
        async def astream_turn(self, request: ModelRequest) -> Any:
            del request
            raise asyncio.CancelledError
            yield  # pragma: no cover -- makes this an async generator

    for label, adapter, kwargs in (
        ("anext_turn", SelfCancellingAsync(), {}),
        ("next_turn", SelfCancellingSync(), {}),
        ("astream_turn", SelfCancellingStream(), {"delta_consumer": lambda chunk: None}),
    ):

        async def run(adapter: Any = adapter, kwargs: Any = kwargs) -> None:
            await ModelCallRunner(adapter=adapter).acall(REQUEST, **kwargs)

        with pytest.raises(ModelAdapterError) as caught:
            asyncio.run(run())
        assert str(caught.value), f"{label}: the failure must say something"
        assert caught.value.error_code == "model_adapter_cancelled", label
        assert isinstance(
            caught.value.__cause__, CalleeCancelled
        ), f"{label}: the original cancellation must stay on the chain"


def test_the_run_being_cancelled_is_not_reported_as_an_adapter_failure() -> None:
    """The counterweight: only the *callee's* cancellation becomes an adapter error.

    Without this, translating every cancellation would pass -- and would have turned a run the host
    stopped into a provider's fault.
    """
    token = CancellationToken()

    class SlowAdapter:
        async def anext_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            token.cancel()
            await asyncio.sleep(30)
            return ModelTurn(final_text="never")  # pragma: no cover

    async def run() -> None:
        await ModelCallRunner(
            adapter=SlowAdapter(),
            current_cancellation_token=lambda: token,
            cancel_grace_s=0.05,
        ).acall(REQUEST)

    with pytest.raises(RunCancelled):
        asyncio.run(run())


# --- receipts -----------------------------------------------------------------------------------


def test_the_prompt_digest_ignores_what_surrounds_the_prompt() -> None:
    """``prompt_digest`` answers "did the model see the same conversation twice".

    Adding a tool to the surface or changing a generation setting must not perturb it, or the
    question it answers becomes unanswerable across ordinary configuration drift.
    """
    spec = _spec("t.one", "d", {"type": "object"})
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


def _spec(tool_id: str, description: str, schema: dict[str, Any]) -> ToolSpec:
    return ToolSpec(
        id=tool_id,
        description=description,
        input_schema=schema,
        capability="read",
        side_effect="read",
        handler=lambda **kwargs: None,
    )


async def _receipt_for(request: ModelRequest) -> Any:
    _turn, receipt = await ModelCallRunner(adapter=SyncAdapter()).acall(request)
    return receipt


def test_the_replay_key_distinguishes_tools_sharing_an_id() -> None:
    """Two requests offering the same tool id with different wire definitions are different calls.

    The provider is sent the description and the input schema, so reducing a tool to its id made
    the "exact replay key" hand back a call the model never made.
    """
    plain = ModelRequest(
        instruction="hi", system_prompt="s", tools=(_spec("t.x", "alpha", {"type": "object"}),)
    )
    renamed = ModelRequest(
        instruction="hi", system_prompt="s", tools=(_spec("t.x", "BETA", {"type": "object"}),)
    )
    reschemad = ModelRequest(
        instruction="hi",
        system_prompt="s",
        tools=(_spec("t.x", "alpha", {"type": "object", "required": ["q"]}),),
    )

    keys = {
        asyncio.run(_receipt_for(request)).request_digest for request in (plain, renamed, reschemad)
    }
    assert len(keys) == 3


@pytest.mark.parametrize(
    ("label", "request_"),
    [
        (
            "a value JSON has no form for, in messages",
            ModelRequest(
                instruction="hi",
                system_prompt="s",
                tools=(),
                messages=({"role": "user", "x": object()},),
            ),
        ),
    ],
)
def test_a_payload_the_serializer_cannot_carry_does_not_kill_the_call(
    label: str, request_: ModelRequest
) -> None:
    """A digest is bookkeeping about a call, never a precondition for making one.

    Digests are computed before the adapter is reached, so a value the canonical serializer chokes
    on stopped the call from happening at all. `{1: "x", "kind": "y"}` is the sharp case: plain
    `json.dumps` accepts it, but canonical JSON sorts keys and sorting `int` against `str` raises.

    Parametrized across all three payload sources on purpose. The first guard covered only tool
    fields and left `messages` and `observations` — both caller-filled — able to kill a call.
    """
    del label

    class Adapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(final_text="answer")

    async def run() -> Any:
        turn, receipt = await ModelCallRunner(adapter=Adapter()).acall(request_)
        return turn, receipt

    turn, receipt = asyncio.run(run())
    # The call is what must survive; whether the payload earns a key is a separate question, covered
    # by the digest tests above. Asserting both here is what made this test wrong twice.
    assert turn.final_text == "answer"
    assert isinstance(receipt.request_digest, str)


@pytest.mark.parametrize(
    "request_",
    [
        ModelRequest(
            instruction="hi",
            system_prompt="s",
            tools=(
                ToolSpec(
                    id="t",
                    description="d",
                    input_schema={"type": "object"},
                    capability="read",
                    side_effect="read",
                    handler=lambda **kwargs: None,
                    guidance={1: "x", "kind": "y"},
                ),
            ),
        ),
        ModelRequest(
            instruction="hi",
            system_prompt="s",
            tools=(),
            messages=({"deep": {"a": {2: "b"}}},),
        ),
    ],
)
def test_model_request_rejects_non_string_json_object_keys_before_dispatch(
    request_: ModelRequest,
) -> None:
    calls = 0

    class Adapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            nonlocal calls
            del request
            calls += 1
            return ModelTurn(final_text="answer")

    with pytest.raises(ValueError, match="JSON object keys must be strings"):
        asyncio.run(ModelCallRunner(adapter=Adapter()).acall(request_))

    assert calls == 0


def _self_cycle() -> dict[str, Any]:
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    return cyclic


def _branching_cycle() -> list[Any]:
    cyclic: list[Any] = []
    cyclic.extend((cyclic, cyclic))
    return cyclic


def _shared_acyclic_graph() -> Any:
    """Exponential to traverse, but no reference repeats on any single path."""
    level: Any = "leaf"
    for _ in range(40):
        level = [level, level]
    return level


@pytest.mark.parametrize(
    ("label", "factory"),
    [
        ("self cycle", _self_cycle),
        ("cycle reached through two references", _branching_cycle),
        ("acyclic but exponentially shared", _shared_acyclic_graph),
    ],
)
def test_a_pathological_payload_gets_no_key_and_does_not_hang(label: str, factory: Any) -> None:
    """None of these can be sent to a provider as JSON, so none of them is a replayable call.

    Earlier versions reshaped each into something hashable and handed it a key. That is where the
    collisions came from: a cycle marker two different graphs shared, then a marker a caller could
    type verbatim. Refusing a key is both simpler and the only answer that cannot be wrong.

    pytest-timeout is the net for a genuine hang; asserting wall-clock here would only measure the
    machine, which is how this file broke CI once already.
    """
    del label

    assert _digest({"value": factory()}) == ""


def test_two_different_cyclic_graphs_cannot_be_confused() -> None:
    """`root -> child -> root` and `root -> child -> child` once shared a non-empty key.

    Neither has one now, which is the safe resolution: an empty digest is *no key*, so a consumer
    cannot match them with each other or with anything else.
    """
    back_to_root: list[Any] = []
    child: list[Any] = [back_to_root]
    back_to_root.append(child)

    self_looping: list[Any] = []
    self_looping.append(self_looping)

    assert _digest({"v": back_to_root}) == ""
    assert _digest({"v": self_looping}) == ""


def test_a_key_json_coerces_digests_as_the_provider_would_receive_it() -> None:
    """A single non-string mapping key encodes fine -- sorting one key compares nothing -- and json
    renders it as a string.

    So `{2: "b"}` and `{"2": "b"}` share a digest. That is correct rather than a collision: a
    provider receives the same bytes for both, and this digest is defined as the identity of what
    went over the wire. Only *mixed* key types in one mapping fail to sort, and those get no key.
    """
    assert _digest({"v": {2: "b"}}) == _digest({"v": {"2": "b"}}) != ""
    assert _digest({"v": {1: "x", "kind": "y"}}) == ""


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda: type("HostileDict", (dict,), {"items": _raises})(a=1), id="dict-items"
        ),
        pytest.param(
            lambda: type("HostileList", (list,), {"__iter__": _raises})([1]), id="list-iter"
        ),
    ],
)
def test_a_container_hook_that_raises_costs_the_key_not_the_call(factory: Any) -> None:
    """A `dict` or `list` subclass can raise from inside the encoder, from a type it accepts.

    The guard used to name four exception types and this was a fifth. Which exception the encoder
    chose is never the question -- only whether it finished -- so the clause catches `Exception`.
    `BaseException` deliberately still escapes: a cancellation is not a statement about the payload.
    """
    assert _digest({"v": factory()}) == ""


def test_the_receipt_separates_calls_to_different_destinations_without_naming_them() -> None:
    """Two adapters with identical configs can address different services -- and now say so.

    `GatewayModelAdapter` lets a per-instance `gateway_url` outrank the config, so two calls with
    identical content can go to different hosts. That fact used to live *inside* the replay key,
    where it made the key unreproducible: the destination is deliberately never recorded, so
    nothing a record holds could reconstruct it and a miss could not even be diagnosed. It is now
    beside the key instead -- a keyed digest a consumer can compare and no one can read a hostname
    out of -- and the key itself describes only what was asked for.
    """
    config = ModelConfig(model="m", gateway_url="http://shared.invalid/x")

    def observed(url: str) -> Any:
        adapter = GatewayModelAdapter(config=config, gateway_url=url, token="t")
        observer = RecordingObserver()

        async def run() -> None:
            with contextlib.suppress(Exception):
                await ModelCallRunner(
                    adapter=adapter,
                    subscriptions=(
                        ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
                    ),
                ).acall(REQUEST)

        asyncio.run(run())
        return observer.captures[0].receipt

    first = observed("http://tenant-a.invalid/x")
    second = observed("http://tenant-b.invalid/x")

    assert first.request_digest == second.request_digest, "same request, same key"
    assert first.destination_digest != second.destination_digest, "different service, said so"
    assert first.destination_digest == observed("http://tenant-a.invalid/x").destination_digest
    assert "tenant-a" not in first.destination_digest
    assert first.destination_status == "resolved"


def test_an_adapter_that_names_no_destination_is_told_apart_from_one_that_cannot() -> None:
    """Declining and failing are two facts, and `""` used to be the answer to both.

    An adapter that routes on config alone has no destination concept; one whose resolver raises
    is misconfigured and every call it makes is about to fail. `_resolve_gateway_url` raises
    deterministically when no URL is configured anywhere, so this is not a transient-vs-absent
    distinction -- it is a working deployment against a broken one, and both used to mint the same
    valid-looking key. Neither costs the call its key: refusing one whenever the destination is
    unknown would refuse one for every adapter that routes on config alone, which is most of them.
    """

    class Silent:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(final_text="answer")

    class Unroutable:
        def resolve_destination(self, config: ModelConfig) -> str:
            del config
            raise RuntimeError("no route")

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(final_text="answer")

    async def receipt_for(adapter: Any) -> Any:
        _turn, receipt = await ModelCallRunner(adapter=adapter).acall(REQUEST)
        return receipt

    silent = asyncio.run(receipt_for(Silent()))
    unroutable = asyncio.run(receipt_for(Unroutable()))

    assert silent.request_digest != ""
    assert unroutable.request_digest != ""
    assert silent.destination_status == "not_declared"
    assert unroutable.destination_status == "unavailable"
    assert silent.destination_digest == ""
    assert unroutable.destination_digest == ""


def test_a_marker_shaped_string_is_ordinary_caller_text() -> None:
    """No sentinel lives in the caller's string domain any more, so none can be forged.

    `["<cycle:1>"]` used to normalize exactly like a list containing itself. It is now just a list
    holding a string, and it keeps a real key.
    """
    literal = _digest({"v": ["<cycle:1>"]})

    assert literal != ""
    assert literal != _digest({"v": ["<cycle:0>"]})


def test_objects_sharing_a_repr_do_not_share_a_key() -> None:
    """Two unrelated objects whose `__repr__` agrees were reduced to the same text and keyed alike.

    Nothing is reduced to `repr` now: a value canonical JSON cannot carry gets no key at all.
    """

    class Alpha:
        def __repr__(self) -> str:
            return "<opaque>"

    class Beta:
        def __repr__(self) -> str:
            return "<opaque>"

    assert _digest({"v": Alpha()}) == ""
    assert _digest({"v": Beta()}) == ""


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda: chr(0xD800), id="lone-surrogate"),
        # Passed as a factory: pytest builds a parametrize id with ``str(val)``, which trips the very
        # integer-conversion limit this case is about.
        pytest.param(lambda: 10**5000, id="int-past-str-conversion-limit"),
    ],
)
def test_a_serializer_hostile_primitive_costs_the_key_not_the_call(factory: Any) -> None:
    """These pass an `isinstance` check and then fail inside the encoder.

    That is why the type-by-type guard was the wrong shape: the authority on what canonical JSON can
    carry is the encoder, so it is now the thing consulted.
    """
    assert _digest({"v": factory()}) == ""


def test_an_object_shared_between_siblings_is_not_treated_as_a_cycle() -> None:
    """Only the path counts. Replacing every repeat would make the digest depend on whether the
    caller happened to share an object, so two logically equal payloads would digest differently."""
    shared = {"k": 1}

    assert _digest({"a": shared, "b": shared}) == _digest({"a": {"k": 1}, "b": {"k": 1}})


def test_normalization_does_not_disturb_an_ordinary_digest() -> None:
    """Counterweight: a normalizer that flattened everything would pass the tests above."""
    first = ModelRequest(instruction="hi", system_prompt="s", tools=())
    second = ModelRequest(instruction="bye", system_prompt="s", tools=())

    assert _digest(_prompt_payload(first)) == _digest(_prompt_payload(first))
    assert _digest(_prompt_payload(first)) != _digest(_prompt_payload(second))


def test_the_prompt_digest_distinguishes_by_reference_continuations() -> None:
    """A request may carry its history as `messages` or as a handle plus new observations.

    In the second shape those fields *are* the prompt, so hashing only `messages` made every
    by-reference continuation collide with every other -- the ordinary case for a gateway client.
    """
    first = ModelRequest(
        instruction=None, system_prompt="s", tools=(), previous_turn_handle="turn_AAA"
    )
    second = ModelRequest(
        instruction=None, system_prompt="s", tools=(), previous_turn_handle="turn_ZZZ"
    )
    assert (
        asyncio.run(_receipt_for(first)).prompt_digest
        != asyncio.run(_receipt_for(second)).prompt_digest
    )

    one = ToolObservation(call_id="c1", tool_name="t", output={"answer": "yes"})
    other = ToolObservation(call_id="c1", tool_name="t", output={"answer": "no"})
    assert (
        asyncio.run(
            _receipt_for(
                ModelRequest(instruction=None, system_prompt="s", tools=(), observations=(one,))
            )
        ).prompt_digest
        != asyncio.run(
            _receipt_for(
                ModelRequest(instruction=None, system_prompt="s", tools=(), observations=(other,))
            )
        ).prompt_digest
    )


def test_an_absent_message_log_is_not_an_empty_one() -> None:
    """`messages=None` and `messages=()` are different requests, so they get different keys.

    Both shipped adapters pick the wire shape with `messages is not None`: an empty tuple sends an
    empty conversation and drops the instruction, `None` sends the instruction. `or ()` asked
    whether the field was empty when its meaning is whether it is present, so two requests the
    provider answers differently were handed one replay key.

    The wire halves are asserted too, not assumed. Without them this test would keep passing if the
    adapters stopped distinguishing the two -- still green, but no longer testing what it says.
    """
    absent = ModelRequest(instruction="hi", system_prompt="s", tools=(), messages=None)
    empty = ModelRequest(instruction="hi", system_prompt="s", tools=(), messages=())

    sent = [GatewayModelAdapter(config=ModelConfig())._payload(r) for r in (absent, empty)]
    assert "instruction" in sent[0] and "messages" not in sent[0]
    assert sent[1]["messages"] == [] and "instruction" not in sent[1]

    assert _digest(_prompt_payload(absent)) != _digest(_prompt_payload(empty))
    keys = [
        _digest(_request_payload(r, ModelConfig(), provider="p"))
        for r in (absent, empty)
    ]
    assert keys[0] != keys[1] and "" not in keys


def test_tool_results_reach_the_redaction_policy() -> None:
    """Observations are model *input*, so a policy has to be handed them to be able to mask them.

    Omitting them from the capture did not merely give a `full` observer an incomplete picture -- it
    routed tool output around redaction entirely. This is the disclosure half; the assertion below
    it is the completeness half.
    """
    observation = ToolObservation(
        call_id="c1", tool_name="lookup", output={"api_key": "sk-live-secret"}
    )
    request = ModelRequest(
        instruction=None, system_prompt="s", tools=(), observations=(observation,)
    )
    observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=SyncAdapter(),
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="redacted")),
            ),
        ).acall(request)

    asyncio.run(run())
    capture = observer.captures[0]
    assert capture.content is not None
    assert "observations" in capture.content
    # The default policy calls ``api_key`` a secret, and it can only have masked it if it was given
    # the field at all.
    assert "sk-live-secret" not in json.dumps(capture.content, default=str)


def test_custom_redactor_output_is_normalized_before_capture_delivery() -> None:
    class Redactor:
        def redact(self, value: Any, *, policy: Any) -> Any:
            del value, policy
            return {"text": "\ud800", "score": float("nan")}

    observer = RecordingObserver()

    async def run() -> Any:
        _turn, receipt = await ModelCallRunner(
            adapter=SyncAdapter(),
            subscriptions=(
                ModelIOSubscription(
                    observer=observer,
                    policy=CapturePolicy(mode="redacted", redactor=Redactor()),
                ),
            ),
        ).acall(REQUEST)
        return receipt

    receipt = asyncio.run(run())

    assert receipt.capture_downgrades == 0
    assert observer.captures[0].content == {"text": "�", "score": None}
    json.dumps(observer.captures[0].content, allow_nan=False)


def test_custom_redactor_key_collision_downgrades_to_digest() -> None:
    class Redactor:
        def redact(self, value: Any, *, policy: Any) -> Any:
            del value, policy
            result = {"\ud800": 1}
            result["�"] = 2
            return result

    observer = RecordingObserver()

    async def run() -> Any:
        _turn, receipt = await ModelCallRunner(
            adapter=SyncAdapter(),
            subscriptions=(
                ModelIOSubscription(
                    observer=observer,
                    policy=CapturePolicy(mode="redacted", redactor=Redactor()),
                ),
            ),
        ).acall(REQUEST)
        return receipt

    receipt = asyncio.run(run())

    assert receipt.capture_downgrades == 1
    assert observer.captures[0].mode == "digest"
    assert observer.captures[0].downgraded_from == "redacted"
    assert observer.captures[0].content is None


def test_an_adapter_that_retried_internally_says_so_in_the_receipt() -> None:
    """`attempts` and `provider_retried` are not the same fact.

    The kernel makes one adapter call per turn however many attempts happen inside it, so a call
    that failed twice and succeeded on the third try would otherwise be recorded as a clean single
    attempt.
    """

    class RetriedAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(final_text="answer", provider_retried=True)

    async def run() -> Any:
        _turn, receipt = await ModelCallRunner(adapter=RetriedAdapter()).acall(REQUEST)
        return receipt

    receipt = asyncio.run(run())
    assert receipt.provider_retried is True
    assert receipt.attempts == 1, "the kernel still made exactly one adapter call"

    # Counterweight: an adapter with no retry loop reports False, which is true of it.
    assert asyncio.run(_receipt_for(REQUEST)).provider_retried is False


class _ConfiguredAdapter:
    """Stands in for the shipped adapters: falls back to `self.config` when the request omits one."""

    def __init__(self, model: str) -> None:
        self.config = ModelConfig(model=model)

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        del request
        return ModelTurn(final_text="answer")


def test_the_receipt_records_the_model_the_adapter_actually_ran() -> None:
    """`ModelRequest.model` is optional and adapters fall back to their own config.

    Reading only the request stamped every such receipt with `ModelConfig()`'s default model — a
    fabricated audit field rather than a missing one, and one that happens to look plausible.
    """

    async def receipt_for(model: str) -> Any:
        _turn, receipt = await ModelCallRunner(adapter=_ConfiguredAdapter(model)).acall(REQUEST)
        return receipt

    first = asyncio.run(receipt_for("gpt-5.5"))
    second = asyncio.run(receipt_for("claude-opus-5"))

    assert first.model.model == "gpt-5.5"
    assert second.model.model == "claude-opus-5"
    # ...and two calls that ran under different models are not the same call.
    assert first.request_digest != second.request_digest


def test_an_explicit_request_model_still_wins_over_the_adapter_config() -> None:
    """The counterweight: resolving the fallback must not start ignoring what the caller asked for."""
    request = ModelRequest(
        instruction="hi", system_prompt="sys", tools=(), model=ModelConfig(model="explicit")
    )

    async def run() -> Any:
        _turn, receipt = await ModelCallRunner(adapter=_ConfiguredAdapter("fallback")).acall(
            request
        )
        return receipt

    assert asyncio.run(run()).model.model == "explicit"


def test_an_adapter_that_exhausted_its_retries_says_so_on_the_failure() -> None:
    """The failed call is the one most likely to have been retried.

    Recording the marker only on success denied retries in exactly the exhausted-budget case — the
    one an audit trail is for.
    """

    class ExhaustedAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            raise ModelAdapterError(
                "all attempts failed",
                provider_error_code="gateway_timeout",
                retryable=True,
                provider_retried=True,
            )

    observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=ExhaustedAdapter(),
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
            ),
        ).acall(REQUEST)

    with pytest.raises(ModelAdapterError):
        asyncio.run(run())
    receipt = observer.captures[0].receipt
    assert receipt.succeeded is False
    assert receipt.provider_retried is True
    # ``retryable`` is a forecast about a future attempt; ``provider_retried`` is a fact about
    # attempts already made. Independent, and both recorded.
    assert receipt.retryable is True


def test_a_failure_without_retries_does_not_claim_any() -> None:
    """Counterweight: "always true on failure" would pass the test above."""

    class FailingAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            raise ModelAdapterError("one and done", provider_error_code="gateway_bad_request")

    observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=FailingAdapter(),
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
            ),
        ).acall(REQUEST)

    with pytest.raises(ModelAdapterError):
        asyncio.run(run())
    assert observer.captures[0].receipt.provider_retried is False


def test_a_retried_stream_reports_the_retry_through_the_terminal_chunk() -> None:
    """On the streaming path the turn is assembled from chunks, so the fact has to ride one."""
    adapter = StreamingAdapter(
        chunks=[TextDelta("ok"), TurnComplete(response_id="r", provider_retried=True)]
    )

    async def run() -> Any:
        _turn, receipt = await ModelCallRunner(adapter=adapter).acall(
            REQUEST, delta_consumer=lambda chunk: None
        )
        return receipt

    assert asyncio.run(run()).provider_retried is True


def test_a_retried_stream_keeps_its_evidence_when_the_call_never_completes() -> None:
    """A call that never completes produces no turn, so the retry has to ride the exception.

    The terminal chunk used to be the only carrier, which meant the evidence existed only for calls
    that finished -- and a failed call is the one an audit trail is for. `RunCancelled` is the
    sharpest case: it is raised by the cancel/deadline race *around* the stream and never passes
    through the adapter, so it is precisely the exception no provider can stamp for itself.
    """

    def hanging(*, retried: bool) -> Any:
        token = CancellationToken()

        class HangingStream:
            async def astream_turn(self, request: ModelRequest) -> Any:
                del request
                yield TextDelta("partial", provider_retried=retried)
                token.cancel()
                await asyncio.sleep(30)
                yield TurnComplete(response_id="never reached")  # pragma: no cover

        observer = RecordingObserver()

        async def run() -> None:
            await ModelCallRunner(
                adapter=HangingStream(),
                current_cancellation_token=lambda: token,
                subscriptions=(
                    ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
                ),
            ).acall(REQUEST, delta_consumer=lambda chunk: None)

        with pytest.raises(RunCancelled):
            asyncio.run(run())
        return observer.captures[0].receipt

    receipt = hanging(retried=True)
    assert receipt.succeeded is False
    assert receipt.error_code == "cancelled"
    assert receipt.provider_retried is True

    # Counterweight: "always true once a stream is cancelled" would pass the assertion above.
    assert hanging(retried=False).provider_retried is False


def test_a_boundary_already_crossed_is_never_paid_for() -> None:
    """A run that has already stopped must not issue the call it decided not to make.

    The cancel/deadline race reported the boundary correctly, but only after the request had been
    handed to the adapter -- so the provider ran it and would bill for it. All three dispatch shapes
    did this, and the receipt digests are built immediately before, which is time a deadline can
    expire in.
    """
    calls: list[str] = []

    class Sync:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            calls.append("sync")
            return ModelTurn(final_text="billed")

    class Async:
        async def anext_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            calls.append("async")
            return ModelTurn(final_text="billed")

    class Stream:
        async def astream_turn(self, request: ModelRequest) -> Any:
            del request
            calls.append("stream")
            yield TurnComplete(response_id="r")

    async def attempt(adapter: Any, *, expired: bool) -> None:
        token = CancellationToken()
        if not expired:
            token.cancel()
        extra = {"delta_consumer": (lambda chunk: None)} if isinstance(adapter, Stream) else {}
        with pytest.raises(RunTimeout if expired else RunCancelled):
            await ModelCallRunner(adapter=adapter, current_cancellation_token=lambda: token).acall(
                REQUEST, deadline=(time.time() - 5) if expired else None, **extra
            )

    async def run() -> None:
        for adapter in (Sync(), Async(), Stream()):
            for expired in (True, False):
                await attempt(adapter, expired=expired)

    asyncio.run(run())
    assert calls == []

    # Counterweight: a runner that dispatched nothing at all would pass the assertion above.
    async def unblocked() -> None:
        await ModelCallRunner(adapter=Sync()).acall(REQUEST, deadline=time.time() + 30)

    asyncio.run(unblocked())
    assert calls == ["sync"]


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
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
            ),
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
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="full")),
            ),
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
                ModelIOSubscription(
                    observer=ExplodingObserver(), policy=CapturePolicy(mode="full")
                ),
                ModelIOSubscription(observer=healthy, policy=CapturePolicy(mode="full")),
            ),
        ).acall(REQUEST)
        return turn

    assert asyncio.run(run()).final_text == "answer"
    assert len(healthy.captures) == 1


def test_a_malformed_recognized_usage_count_fails_closed() -> None:
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

    with pytest.raises(ModelAdapterError, match="non-portable response") as caught:
        asyncio.run(run())

    assert isinstance(caught.value.__cause__, ValueError)
    assert "output_tokens" in str(caught.value.__cause__)


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


def test_a_truncated_payload_gets_no_replay_key_rather_than_a_misleading_one() -> None:
    """Bounding the work created a way to collide, which is the failure this digest must not have.

    Two requests differing only past the cut normalize to the same thing, so an ordinary-looking
    digest would stand for "everything up to here" and a replay consumer would hand back the wrong
    call. Refusing to issue a key is the safe half: the call still happens, it is just not
    replayable.
    """
    # Sized past MAX_MODEL_PAYLOAD_BYTES -- the cap is the wire's since W6-2, so the old
    # million-int payload (~6.9 MB) is now legitimately keyable and no longer exercises this.
    huge = ModelRequest(
        instruction="hi", system_prompt="s", tools=(), messages=({"v": list(range(1_300_000))},)
    )
    huge_but_different = ModelRequest(
        instruction="hi",
        system_prompt="s",
        tools=(),
        messages=({"v": list(range(1_299_999)) + [-999]},),
    )

    assert _digest(_prompt_payload(huge)) == ""
    assert _digest(_prompt_payload(huge_but_different)) == ""

    # Counterweight: ordinary payloads still get a key, and it still discriminates. "return empty
    # always" would pass the assertions above.
    ordinary = ModelRequest(instruction="hi", system_prompt="s", tools=())
    other = ModelRequest(instruction="bye", system_prompt="s", tools=())
    assert _digest(_prompt_payload(ordinary)) not in {"", _digest(_prompt_payload(other))}


def test_an_unbounded_expansion_costs_the_key_rather_than_the_process() -> None:
    """The output cap is what stops a payload built from shared references from expanding forever.

    Asserted on the outcome, not on elapsed time: a bound that stopped working would not make this
    slower, it would make it never finish, and pytest-timeout is the net for that.
    """
    level: Any = "leaf"
    for _ in range(40):
        level = [level, level]

    assert _digest({"v": level}) == ""

    # Counterweight: a large payload that genuinely encodes still gets a key, so the cap is not
    # simply refusing everything big.
    realistic = {"messages": [{"role": "assistant", "content": "x" * 200} for _ in range(2000)]}
    assert _digest(realistic) != ""


def test_no_subscriptions_means_no_capture_work() -> None:
    """Delivery is opt-in; identifying the call is not.

    The digests are computed either way, because they describe the call whether or not anyone is
    watching. The CHANGELOG claimed otherwise and this test is what contradicted it.
    """

    async def run() -> Any:
        _turn, receipt = await ModelCallRunner(adapter=SyncAdapter()).acall(REQUEST)
        return receipt

    receipt = asyncio.run(run())
    assert receipt.capture_downgrades == 0
    assert receipt.prompt_digest != ""


# --- retry reported by an adapter whose call never returns an outcome -----------------------------


def test_a_retry_survives_a_blocking_adapter_the_run_abandons() -> None:
    """The one carrier that crosses abandonment.

    A blocking ``next_turn`` keeps running on a thread nobody reads once the run stops waiting, and
    the receipt is built from the ``RunTimeout`` the race raised -- which the adapter never touched.
    A run that timed out *because* the provider was retrying is the case most likely to matter, and
    it recorded a clean single attempt.
    """

    class RetryingBlockingAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            report_provider_retried()  # entering its second attempt
            time.sleep(3)  # which never finishes
            return ModelTurn(final_text="never")  # pragma: no cover

    async def run() -> None:
        await ModelCallRunner(adapter=RetryingBlockingAdapter(), cancel_grace_s=0.05).acall(
            REQUEST, deadline=time.time() + 0.1
        )

    with pytest.raises(RunTimeout) as caught:
        asyncio.run(run())
    assert getattr(caught.value, "provider_retried", False) is True


def test_a_blocking_adapter_that_did_not_retry_claims_nothing() -> None:
    """The counterweight: the channel reports what happened, not that a call was abandoned."""

    class BlockingAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            time.sleep(3)
            return ModelTurn(final_text="never")  # pragma: no cover

    async def run() -> None:
        await ModelCallRunner(adapter=BlockingAdapter(), cancel_grace_s=0.05).acall(
            REQUEST, deadline=time.time() + 0.1
        )

    with pytest.raises(RunTimeout) as caught:
        asyncio.run(run())
    assert getattr(caught.value, "provider_retried", False) is False


def test_a_reported_retry_reaches_a_successful_receipt_too() -> None:
    """Honoured whatever the call returns.

    Read only on failure it would be a seam that silently stops working for adapters that retry and
    then succeed -- which is most of the time a retry loop runs.
    """

    class RecoveringAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            report_provider_retried()
            return ModelTurn(final_text="answer")  # the turn itself does not say so

    async def run() -> Any:
        _turn, receipt = await ModelCallRunner(adapter=RecoveringAdapter()).acall(REQUEST)
        return receipt

    assert asyncio.run(run()).provider_retried is True


def test_the_retry_channel_does_not_leak_between_calls() -> None:
    """One call's report must not colour the next. The channel is per-call, not per-runner."""
    runner = ModelCallRunner(adapter=SyncAdapter())

    class OnceRetryingAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            report_provider_retried()
            return ModelTurn(final_text="answer")

    async def run() -> tuple[Any, Any]:
        _t, first = await ModelCallRunner(adapter=OnceRetryingAdapter()).acall(REQUEST)
        _t2, second = await runner.acall(REQUEST)
        return first, second

    first, second = asyncio.run(run())
    assert (first.provider_retried, second.provider_retried) == (True, False)


def test_reporting_a_retry_outside_a_runner_is_inert() -> None:
    """An adapter used directly is not broken by calling the seam with nobody listening."""
    report_provider_retried()  # must not raise


def test_the_retry_channel_counts_every_report() -> None:
    """`retried` answers "did any loop below run" and is monotone, so a second report is
    invisible to it -- which is exactly what per-attempt attribution cannot live with. `count`
    carries what the bool discards; the bool stays, derived, so every existing reader keeps
    meaning what it meant."""

    with collect_retry_reports() as progress:
        assert (progress.count, progress.retried) == (0, False)
        report_provider_retried()
        assert (progress.count, progress.retried) == (1, True)
        report_provider_retried()
        assert (progress.count, progress.retried) == (2, True)


# --- a broken adapter must not cost the call, the receipt, or the run ----------------------------


@pytest.mark.parametrize(
    "turn",
    [
        ModelTurn(final_text="answer", usage=None),  # type: ignore[arg-type]
        None,
        {"final_text": "answer"},
    ],
    ids=["usage-is-None", "returns-None", "returns-a-dict"],
)
def test_a_turn_shaped_result_still_produces_a_receipt(turn: Any) -> None:
    """A receipt is produced whether the call succeeded or failed -- including this way.

    Read as hard attributes, a `usage=None` (which `examples/custom_model_adapter.py` invites by
    calling usage "optional") raised from inside `_publish`'s argument list, so *no* receipt was
    produced at all and an answer the provider had already been paid for was discarded over a token
    counter.
    """

    class OddAdapter:
        def next_turn(self, request: ModelRequest) -> Any:
            del request
            return turn

    observer = RecordingObserver()
    runner = ModelCallRunner(
        adapter=OddAdapter(),
        subscriptions=(ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="full")),),
    )

    async def run() -> Any:
        with contextlib.suppress(Exception):
            return await runner.acall(REQUEST)
        return None

    asyncio.run(run())
    assert len(observer.captures) == 1, "the call happened, so the audit trail must record it"


def test_a_tool_call_the_adapter_built_oddly_costs_its_own_entry() -> None:
    """A display surface must not fail a call that already happened."""

    class Slotted:
        __slots__ = ("id",)

        def __init__(self) -> None:
            self.id = "c1"

    class OddAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(final_text="answer", tool_calls=(Slotted(),))  # type: ignore[arg-type]

    observer = RecordingObserver()
    runner = ModelCallRunner(
        adapter=OddAdapter(),
        subscriptions=(ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="full")),),
    )

    async def run() -> Any:
        turn, _receipt = await runner.acall(REQUEST)
        return turn

    assert asyncio.run(run()).final_text == "answer"
    assert len(observer.captures) == 1
    # The half this test is named for, and did not check. Asserting only that the call survived
    # made "the odd entry is dropped" indistinguishable from "the odd entry is preserved" -- and
    # dropping it is worse than the crash it replaced: the record then claims the model made fewer
    # tool calls than it made, silently, on the surface an audit reads.
    tool_calls = observer.captures[0].content["tool_calls"]
    assert len(tool_calls) == 1, f"the odd tool call vanished from the audit surface: {tool_calls}"
    assert "repr" in tool_calls[0], "an entry that cannot be walked must still say what it was"


def test_a_tool_call_that_refuses_to_describe_itself_still_costs_only_its_own_entry() -> None:
    """The fallback needs a fallback, or it is not one.

    The test above uses a `__slots__` object, which `vars()` rejects and `repr()` handles -- so the
    fallback was only ever exercised on an object that cooperates with it. One that refuses *both*
    took the exception out through `_publish`, and that is not a lost display entry: the turn had
    already been produced, so a provider answer the run had been paid for was discarded by the code
    whose whole purpose is to keep that from happening.
    """

    class Hostile:
        __slots__ = ()

        def __repr__(self) -> str:
            raise RuntimeError("repr exploded")

    class HostileAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(final_text="answer", tool_calls=(Hostile(),))  # type: ignore[arg-type]

    observer = RecordingObserver()

    async def run() -> Any:
        turn, _receipt = await ModelCallRunner(
            adapter=HostileAdapter(),
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="full")),
            ),
        ).acall(REQUEST)
        return turn

    assert asyncio.run(run()).final_text == "answer", "the provider's answer was discarded"
    assert len(observer.captures) == 1, "the capture was lost with it"
    tool_calls = observer.captures[0].content["tool_calls"]
    assert len(tool_calls) == 1, f"the entry vanished instead of degrading: {tool_calls}"
    # Degraded, not silent: the record says something was there and that it could not be described.
    assert tool_calls[0]["repr"] == "<unrepresentable Hostile>"


def test_a_raising_probe_does_not_lose_the_call() -> None:
    """`provider_name` and `config` answer bookkeeping, so neither can cost the call.

    Undefended, a property that raised took the call down before the adapter was ever invoked.
    """

    class HostileAdapter:
        @property
        def provider_name(self) -> str:
            raise RuntimeError("provider_name exploded")

        @property
        def config(self) -> ModelConfig:
            raise RuntimeError("config exploded")

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(final_text="answer")

    async def run() -> Any:
        turn, receipt = await ModelCallRunner(adapter=HostileAdapter()).acall(REQUEST)
        return turn, receipt

    turn, receipt = asyncio.run(run())
    assert turn.final_text == "answer"
    assert receipt.provider_name == ""


def test_capture_failing_does_not_replace_the_providers_failure() -> None:
    """Turning capture on must not change how a provider failure is classified.

    The docstring promises the receipt is delivered *before* the exception is re-raised; when
    delivery itself blew up it was delivered *instead of* it, and a `ModelAdapterError` carrying
    `retryable` and `http_status` reached the loop as capture's exception, losing the classification
    the loop's own `except ModelAdapterError` depends on.

    Injected where the per-observer guard cannot help: `content_digest` runs over the whole content
    before any observer is called, and `_jsonish` falls through to `str(value)`.
    """

    class Unprintable:
        def __str__(self) -> str:
            raise ValueError("no str")

    class FailingAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            raise ModelAdapterError("provider 503", retryable=True, http_status=503)

    runner = ModelCallRunner(
        adapter=FailingAdapter(),
        subscriptions=(
            ModelIOSubscription(observer=RecordingObserver(), policy=CapturePolicy(mode="digest")),
        ),
    )
    request = replace(REQUEST, messages=[{"role": "user", "content": Unprintable()}])

    async def run() -> None:
        await runner.acall(request)

    with pytest.raises(ModelAdapterError) as caught:
        asyncio.run(run())
    assert caught.value.http_status == 503
    assert caught.value.retryable is True

    # Counterweight: with no subscriptions the same call already reported correctly, so this test
    # must be failing on the capture path rather than on the adapter's own error.
    async def uncaptured() -> None:
        await ModelCallRunner(adapter=FailingAdapter()).acall(request)

    with pytest.raises(ModelAdapterError):
        asyncio.run(uncaptured())


@pytest.mark.parametrize("mode", ["raises", "hangs"], ids=["close-raises", "close-hangs"])
def test_a_cleanup_that_misbehaves_does_not_become_the_calls_outcome(mode: str) -> None:
    """The stream's `aclose()` runs in a `finally`, so what it does replaces the call's outcome.

    A provider whose close raised turned a caller's abort into a terminal failure -- killing the
    session that `ModelCallAborted` exists to keep parked. One whose close hung hung the run
    outright: the abort is raised *inside* the awaited task, so no run boundary is pending and no
    grace interval applies to it.

    Only reachable when the stream is still live, which is exactly when a caller stops it early --
    a drained generator's `aclose()` is a no-op.
    """

    class HostileCloseAdapter:
        async def astream_turn(self, request: ModelRequest):  # noqa: ANN202
            del request
            try:
                yield TextDelta("ans")
                yield TextDelta("wer")
            except GeneratorExit:
                if mode == "hangs":
                    await asyncio.sleep(30)
                raise RuntimeError("close exploded") from None

    seen: list[Any] = []

    async def run() -> None:
        await ModelCallRunner(adapter=HostileCloseAdapter(), cancel_grace_s=0.05).acall(
            REQUEST, delta_consumer=seen.append, should_abort=lambda: len(seen) >= 1
        )

    started = time.monotonic()
    with pytest.raises(ModelCallAborted):
        asyncio.run(run())
    assert time.monotonic() - started < 5.0, "a stuck close must not outlast the grace interval"


def test_a_sync_adapter_raising_stopiteration_does_not_hang_the_run() -> None:
    """`Future.set_exception` refuses `StopIteration`, and the refusal used to strand the awaiter.

    The TypeError surfaced inside a `call_soon_threadsafe` callback where nothing awaited it, so the
    future stayed pending forever and no deadline could end the run. A callee raising it is
    ordinary: `next(...)` on an exhausted iterator does.
    """

    class ExhaustedAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return next(iter(()))  # type: ignore[call-overload]

    async def run() -> None:
        # Given a deadline deliberately, so a regression fails this test instead of hanging it: the
        # stranded awaiter is only ever released by a boundary, and `RunTimeout` here means the
        # callee's failure never arrived.
        await ModelCallRunner(adapter=ExhaustedAdapter(), cancel_grace_s=0.05).acall(
            REQUEST, deadline=time.time() + 1.0
        )

    with pytest.raises(RuntimeError, match="StopIteration"):
        asyncio.run(run())


# --- receipt fields nothing was pinning ----------------------------------------------------------


def test_usage_normalization_keeps_recognized_counts_and_drops_unknown_entries() -> None:
    class OddUsageAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(
                final_text="answer",
                usage={"input_tokens": 5, "cached_tokens": 0, "bad": -3, "flag": True, 7: 1},
            )

    async def run() -> Any:
        _turn, receipt = await ModelCallRunner(adapter=OddUsageAdapter()).acall(REQUEST)
        return receipt

    assert dict(asyncio.run(run()).usage) == {
        "input_tokens": 5,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def test_an_adapter_that_reports_no_stop_reason_records_an_empty_one() -> None:
    """`ModelTurn.stop_reason` is `None` when the adapter does not report one.

    Written through `str()` unguarded, that becomes the literal `"None"` in an audit field -- a
    value indistinguishable from an adapter that really said "None".
    """

    async def run() -> Any:
        _turn, receipt = await ModelCallRunner(adapter=SyncAdapter()).acall(REQUEST)
        return receipt

    assert asyncio.run(run()).stop_reason == ""


def test_a_by_reference_call_shows_an_empty_message_log_not_a_null_one() -> None:
    """The display surface keeps its container shape even when the request carried none."""
    observer = RecordingObserver()
    runner = ModelCallRunner(
        adapter=SyncAdapter(),
        subscriptions=(ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="full")),),
    )

    async def run() -> None:
        await runner.acall(replace(REQUEST, messages=None, previous_turn_handle="prev"))

    asyncio.run(run())
    content = observer.captures[0].content
    assert content["messages"] == []
    # `previous_turn_handle` is normalized the same way and was left unbound, so only one of the two
    # sibling guards was held. Checked on the *by-value* shape, where the request carries no handle
    # and the surface must still show the empty string the code intends rather than `None`.
    by_value = RecordingObserver()
    asyncio.run(
        ModelCallRunner(
            adapter=SyncAdapter(),
            subscriptions=(
                ModelIOSubscription(observer=by_value, policy=CapturePolicy(mode="full")),
            ),
        ).acall(REQUEST)
    )
    assert by_value.captures[0].content["previous_turn_handle"] == ""


def test_a_resolver_that_answers_nothing_is_declined_not_absent() -> None:
    """The third way an adapter declines a destination: answering, with nothing.

    Unguarded, `str(None)` recorded the text `"None"` -- a destination no adapter has, shared by
    every adapter that returns one. It is now `declined`, which is a different fact from the
    `not_declared` of an adapter that never offered the member and from the `unavailable` of one
    whose probe raised.
    """

    class Vague:
        def resolve_destination(self, config: ModelConfig) -> Any:
            del config
            return None

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(final_text="answer")

    async def run() -> Any:
        _turn, receipt = await ModelCallRunner(adapter=Vague()).acall(REQUEST)
        return receipt

    receipt = asyncio.run(run())
    assert receipt.destination_status == "declined"
    assert receipt.destination_digest == ""
    # The key does not move with any of it: the endpoint left the payload entirely.
    assert receipt.request_digest == _digest(
        _request_payload(REQUEST, ModelConfig(), provider="gateway")
    )


def test_a_config_of_the_wrong_type_is_not_written_into_the_receipt() -> None:
    """`config` is probed, so an adapter may expose anything under that name."""

    class MisConfigured:
        config = "gpt-5.5"  # a string, not a ModelConfig

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(final_text="answer")

    async def run() -> Any:
        _turn, receipt = await ModelCallRunner(adapter=MisConfigured()).acall(REQUEST)
        return receipt

    assert asyncio.run(run()).model == ModelConfig()


def test_a_receipt_that_already_recorded_a_retry_keeps_it_through_a_failure() -> None:
    """`with_error` combines rather than assigns. Guards a second caller, so tested directly."""
    from monoid_agent_kernel.core.model_io import ModelCallReceipt

    receipt = ModelCallReceipt(provider_retried=True)
    assert receipt.with_error(RuntimeError("boom")).provider_retried is True
    assert ModelCallReceipt().with_error(RuntimeError("boom")).provider_retried is False


@pytest.mark.parametrize("invalid_control", [float("nan"), 1, "yes"])
def test_error_receipt_rejects_truthy_non_boolean_controls(invalid_control: Any) -> None:
    from monoid_agent_kernel.core.model_io import ModelCallReceipt

    error = RuntimeError("boom")
    error.retryable = invalid_control  # type: ignore[attr-defined]
    error.provider_retried = invalid_control  # type: ignore[attr-defined]

    receipt = ModelCallReceipt().with_error(error)

    assert receipt.retryable is False
    assert receipt.provider_retried is False


def test_an_abandoned_async_call_is_reported_the_way_an_abandoned_thread_is(
    caplog: Any,
) -> None:
    """Both halves of the bridge report; only one used to.

    An async callee whose cleanup outran the grace was detached in silence, and it has the same
    unbounded shape as the sync one -- one task, and everything it holds, per abandonment, on a loop
    that may run for days. Measured before this fix: 400 abandonments, 400 pending tasks, 400 live
    generators, zero log lines, while the sync half emitted 400.
    """

    class StubbornCleanupAdapter:
        async def astream_turn(self, request: ModelRequest):  # noqa: ANN202
            del request
            try:
                yield TextDelta("ans")
                await asyncio.sleep(30)
            finally:
                await asyncio.sleep(30)  # cleanup that outruns any grace

    async def run() -> None:
        await ModelCallRunner(adapter=StubbornCleanupAdapter(), cancel_grace_s=0.02).acall(
            REQUEST, delta_consumer=lambda chunk: None, deadline=time.time() + 0.05
        )

    with caplog.at_level("WARNING", logger="monoid_agent_kernel.core.sync_bridge"):
        with pytest.raises(RunTimeout):
            asyncio.run(run())

    assert any(
        "abandoned an asynchronous call" in record.message for record in caplog.records
    ), "an abandoned async call must be as visible as an abandoned thread"


class _CancelSuppressingCloseAdapter:
    """A provider whose stream cleanup ignores cancellation until it is released.

    The adversarial shape the grace exists to bound. A close doing its own blocking teardown --
    draining a socket, retrying a release call -- can swallow the cancellation meant to stop it.
    """

    def __init__(self, release: asyncio.Event) -> None:
        self._release = release

    async def astream_turn(self, request: ModelRequest):  # noqa: ANN202
        del request
        try:
            yield TextDelta("ans")
        except GeneratorExit:
            while not self._release.is_set():
                with contextlib.suppress(asyncio.CancelledError):
                    await self._release.wait()


async def _time_a_suppressed_close(grace: float, rescue_after: float) -> float:
    """How long `acall` takes when the stream's close refuses to be cancelled.

    `rescue_after` releases the close from a *separate* task so a regression fails on the elapsed
    assertion instead of hanging: the awaiting path is exactly the one that stops being bounded.
    """

    release = asyncio.Event()
    runner = ModelCallRunner(adapter=_CancelSuppressingCloseAdapter(release), cancel_grace_s=grace)

    async def rescue() -> None:
        await asyncio.sleep(rescue_after)
        release.set()

    rescuer = asyncio.ensure_future(rescue())
    started = time.monotonic()
    try:
        with pytest.raises(ModelCallAborted):
            await runner.acall(
                REQUEST, delta_consumer=lambda chunk: None, should_abort=lambda: True
            )
        return time.monotonic() - started
    finally:
        release.set()
        rescuer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await rescuer


def test_a_stream_close_that_suppresses_cancellation_cannot_outlast_the_grace() -> None:
    """The grace is only a bound if outrunning it means being *detached*.

    `asyncio.wait_for` reads like the bound and is not one: on timeout it cancels the close and then
    awaits that cancellation, so a close that suppresses `CancelledError` holds the run for as long
    as it likes. Measured with that spelling and a 0.05s grace: 4.59s, ~90x over.

    The abort matters here. On the cancel and deadline paths a boundary is already pending and
    `detach_unfinished_call` bounds the whole task; on abort and on ordinary completion nothing else
    is watching, so this is the only bound there is.
    """

    elapsed = asyncio.run(_time_a_suppressed_close(grace=0.05, rescue_after=3.0))
    assert elapsed < 0.6, (
        f"a 0.05s grace let a cancel-suppressing close hold the call for {elapsed:.2f}s; "
        "the close is being awaited rather than detached"
    )


def test_an_abandoned_stream_close_says_what_it_leaves_behind(caplog: Any) -> None:
    """Abandoning is the lesser harm, not a free one, so it is visible.

    Same reason the sync and async halves of the bridge warn: one generator and its connection per
    abandoned stream, on a loop that may run for days, is growth an operator has to be able to see.
    """

    with caplog.at_level("WARNING", logger="monoid_agent_kernel.model_call"):
        asyncio.run(_time_a_suppressed_close(grace=0.05, rescue_after=3.0))

    assert any(
        "outran the" in record.message for record in caplog.records
    ), "an abandoned stream close must be as visible as an abandoned call"


def test_a_close_that_finishes_in_time_is_not_reported_as_abandoned(caplog: Any) -> None:
    """Counterweight: the warning must not fire for every streamed call.

    A rule that reports the ordinary case teaches operators to filter it out, which costs exactly
    the abandonment the previous test pins.
    """

    class PromptCloseAdapter:
        async def astream_turn(self, request: ModelRequest):  # noqa: ANN202
            del request
            yield TextDelta("ans")

    async def run() -> None:
        await ModelCallRunner(adapter=PromptCloseAdapter(), cancel_grace_s=0.05).acall(
            REQUEST, delta_consumer=lambda chunk: None
        )

    with caplog.at_level("WARNING", logger="monoid_agent_kernel.model_call"):
        asyncio.run(run())

    assert not caplog.records, f"an ordinary close was reported as abandoned: {caplog.records}"


def test_a_probe_that_raises_on_lookup_is_unavailable_not_absent() -> None:
    """The probe is tolerant at the lookup, not only at the call -- and now says which it was.

    `resolve_destination` is opt-in, so an adapter may expose it as a property, and a property that
    raised took the whole call down over a replay key. The sibling probes guarded the `getattr`;
    this one guarded only the invocation, which is the half a `def` happens to exercise.

    Tolerating it is right; recording it as "no destination" was not. That collapsed a working
    deployment and a broken one into one answer, and the collapse was invisible because both
    produced a key that looked fine.
    """

    class RaisingLookup:
        @property
        def resolve_destination(self) -> Any:
            raise RuntimeError("probe exploded")

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(final_text="answer")

    async def run() -> Any:
        turn, receipt = await ModelCallRunner(adapter=RaisingLookup()).acall(REQUEST)
        return turn, receipt

    turn, receipt = asyncio.run(run())
    assert turn.final_text == "answer"
    # See the sibling above on `provider="gateway"`: the slot resolves through the config when an
    # adapter declares nothing, and this adapter declares nothing but a raising property.
    assert receipt.request_digest == _digest(
        _request_payload(REQUEST, ModelConfig(), provider="gateway")
    )
    assert receipt.destination_status == "unavailable"


def test_a_host_whose_adapter_changes_is_read_once_per_call_and_not_once_per_probe() -> None:
    """The seam that lets a host swap adapters, and the limit on how far that goes.

    Read *per call*, so a swap between calls takes effect -- the loop's `model_adapter` is public and
    mutable, and everything around the runner reads it live. Read *once*, so one call cannot be
    answered by one adapter and attributed to another: the receipt names a provider, a model and a
    destination, and three probes reading a moving field would describe a mixture of adapters that
    never ran.
    """

    class Marked:
        def __init__(self, tag: str) -> None:
            self.provider_name = tag
            self.requests = 0

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            self.requests += 1
            return ModelTurn(final_text=f"FROM-{self.provider_name}")

    first, second = Marked("FIRST"), Marked("SECOND")
    live = [first]
    reads: list[int] = []

    def current() -> Any:
        reads.append(len(reads))
        return live[0]

    async def run() -> Any:
        runner = ModelCallRunner(adapter=first, current_adapter=current)
        one = await runner.acall(REQUEST)
        live[0] = second
        two = await runner.acall(REQUEST)
        return one, two

    (turn_one, receipt_one), (turn_two, receipt_two) = asyncio.run(run())

    assert turn_one.final_text == "FROM-FIRST"
    assert turn_two.final_text == "FROM-SECOND", "a swap between calls must take effect"
    assert receipt_one.provider_name == "FIRST"
    assert receipt_two.provider_name == "SECOND", "the receipt must name the adapter that answered"
    assert (first.requests, second.requests) == (1, 1)
    assert reads == [
        0,
        1,
    ], f"the adapter must be read exactly once per call, was read {len(reads)}x"


def test_the_key_names_the_provider_the_receipt_records() -> None:
    """One read of the declaration per call, for the same reason the adapter itself gets one.

    The sibling above binds "one adapter per call" and stops there. The *declaration on that
    adapter* was still read twice -- once for `ModelCallReceipt.provider_name`, once for the replay
    key -- and a `provider_name` property that answers and then stops answering made the two
    disagree. The receipt then said `openai` while the key had been taken under the config's
    `gateway`, which is precisely the failure W6-0 exists to remove: a key whose preimage the
    record cannot reconstruct cannot be recomputed, cannot be verified, and a miss cannot be told
    apart from a defect.

    The property is checked both ways round. Agreeing with a *stable* adapter's key is the part
    that says which of the two values won -- an implementation that resolved both to the fallback
    would agree with itself and still be wrong.
    """

    class OnceThenUnreadable:
        """A declaration that answers the first read and raises after it -- a cached property
        whose refresh fails, a proxy that loses its upstream, a lazily-resolved client."""

        def __init__(self) -> None:
            self.reads = 0

        @property
        def provider_name(self) -> str:
            self.reads += 1
            if self.reads > 1:
                raise RuntimeError("declaration went unreadable")
            return "openai"

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(final_text="answer")

    class Stable:
        provider_name = "openai"

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(final_text="answer")

    request = replace(REQUEST, model=ModelConfig(provider="gateway"))

    async def run(adapter: Any) -> Any:
        return (await ModelCallRunner(adapter=adapter).acall(request))[1]

    unstable = OnceThenUnreadable()
    flickering = asyncio.run(run(unstable))
    steady = asyncio.run(run(Stable()))

    assert flickering.provider_name == "openai"
    assert flickering.request_digest == steady.request_digest, (
        "the key must name the provider the receipt records, not a second read of it"
    )
    assert flickering.request_digest == _digest(
        _request_payload(request, ModelConfig(provider="gateway"), provider="openai")
    )
    assert unstable.reads == 1, f"the declaration was read {unstable.reads}x, must be once"


def test_a_close_is_granted_the_grace_and_the_grace_is_read_live(caplog: Any) -> None:
    """Two halves of one rule: cleanup is *given* the interval, and the interval is the live one.

    Bounding a close is only half of it -- a bound of zero also bounds it. Nothing pinned that the
    close gets any time at all, so `timeout=0.0` passed the entire suite while releasing no pooled
    connection anywhere. The side effect is the assertion: a warning-only check cannot tell "the
    cleanup finished" from "the cleanup never started".

    The generator is deliberately left **suspended at a yield**. An earlier version of this test
    drained it, which runs the `finally` during iteration and leaves `aclose()` a no-op -- it
    exercised nothing, and passed. Only an abort mid-stream makes the close do real work.
    """

    released: list[str] = []

    class ReleasingCloseAdapter:
        async def astream_turn(self, request: ModelRequest):  # noqa: ANN202
            del request
            try:
                yield TextDelta("ans")
                yield TextDelta("never consumed")
            except GeneratorExit:
                await asyncio.sleep(0.05)  # a pooled connection being handed back
                released.append("released")

    async def run() -> None:
        runner = ModelCallRunner(
            adapter=ReleasingCloseAdapter(),
            cancel_grace_s=0.0,  # a snapshot that would grant the cleanup nothing
            current_cancel_grace_s=lambda: 0.5,
        )
        with pytest.raises(ModelCallAborted):
            await runner.acall(
                REQUEST, delta_consumer=lambda chunk: None, should_abort=lambda: True
            )

    with caplog.at_level("WARNING", logger="monoid_agent_kernel.model_call"):
        asyncio.run(run())

    assert released == ["released"], (
        "the close was cut off before it could release anything: the grace is bounding cleanup to "
        "nothing, or is being read from the constructed value rather than the live one"
    )
    assert (
        not caplog.records
    ), f"a close that finished inside the grace was reported as abandoned: {caplog.records}"


def test_an_abandoned_call_is_granted_the_live_grace_not_the_constructed_one(caplog: Any) -> None:
    """The other place the grace is spent, and the other half of the same live read.

    `_aclose_within_grace` is not the only spender: a call the *boundary* ends is detached by
    `detach_unfinished_call`, which waits on the task for the interval `_aawait` hands it. That site
    read the same field and nothing observed which value it read, so the snapshot spelling passed --
    on the path where the grace matters most, since a deadline or a cancel is the ordinary way a call
    outlives the run.

    The callee suppresses the cancellation and needs 50ms to finish. A zero grace waits none of it
    and reports the abandonment; the live 0.5s covers it, and reports nothing. The absent warning is
    the assertion because it separates "the cleanup finished" from "the cleanup never started".
    """

    released: list[str] = []

    class CancelSuppressingAdapter:
        async def anext_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                await asyncio.sleep(0.05)  # a pooled connection being handed back
                released.append("released")
            return ModelTurn(final_text="too late")

    async def run() -> None:
        runner = ModelCallRunner(
            adapter=CancelSuppressingAdapter(),
            cancel_grace_s=0.0,  # a snapshot that would grant the cleanup nothing
            current_cancel_grace_s=lambda: 0.5,
        )
        # A deadline in the *near future*, not an expired one: a task cancelled before it has run
        # never enters the adapter, so there would be no cleanup to grant anything to.
        with pytest.raises(RunTimeout):
            await runner.acall(REQUEST, deadline=time.time() + 0.05)

    with caplog.at_level("WARNING", logger="monoid_agent_kernel.core.sync_bridge"):
        asyncio.run(run())

    assert released == ["released"], "the callee's cleanup was cut off before it could finish"
    assert not caplog.records, (
        "a callee that finished inside the grace was reported as abandoned, so the grace being "
        f"spent is not the live one: {[r.message for r in caplog.records]}"
    )


def test_a_bookkeeping_failure_does_not_orphan_the_call_it_already_started(caplog: Any) -> None:
    """Registration runs *after* the call is live, so its failure must not skip the cleanup.

    `start_abandonable_sync_call` starts the worker thread before `await_abandonable_call` is even
    entered. With `add_cancel_callback` outside the `try`, a token that raised there skipped the
    `finally` entirely: the call was neither cancelled, detached, nor consumed, and ran to completion
    behind a run that had already reported a failure -- writing into a workspace nobody was waiting
    on. The fix shipped with no test at all; nothing in the suite injects a raising registration, so
    reverting it passed all 1987 tests.

    The warning is the observable: reaching it means the `finally` ran and the worker was detached
    and reported, rather than silently left behind.
    """

    class HostileToken(CancellationToken):
        def add_cancel_callback(self, callback: Any) -> Any:
            del callback
            raise RuntimeError("registration exploded")

    started = threading.Event()
    finished = threading.Event()

    class BlockingAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            started.set()
            time.sleep(0.3)  # outlasts the grace, so an abandoned worker is reported
            finished.set()
            return ModelTurn(final_text="nobody is waiting for this")

    runner = ModelCallRunner(
        adapter=BlockingAdapter(),
        current_cancellation_token=HostileToken,
        cancel_grace_s=0.02,
    )

    async def run() -> None:
        await runner.acall(REQUEST)

    with caplog.at_level("WARNING", logger="monoid_agent_kernel.core.sync_bridge"):
        with pytest.raises(RuntimeError, match="registration exploded"):
            asyncio.run(run())

    assert started.is_set(), "this test is meaningless unless the call really was already running"
    assert any("abandoned a synchronous call" in record.message for record in caplog.records), (
        "a call that was already running must still be detached and reported when the bookkeeping "
        "around it fails"
    )
    assert finished.wait(timeout=5.0), "the worker should still finish; it is abandoned, not killed"


@pytest.mark.parametrize(
    ("accessor", "starts_the_call"),
    [("current_cancel_grace_s", True), ("current_cancellation_token", False)],
)
def test_a_race_accessor_that_raises_does_not_orphan_a_live_call(
    caplog: Any, accessor: str, starts_the_call: bool
) -> None:
    """The sibling of the test above: the *other* thing that runs after the call is live.

    `_aawait`'s race arguments were resolved in its own argument list, and `pending` is already a
    running worker by then. An accessor raising there landed between starting the call and entering
    the wait that owns the cleanup, so the worker was left running with nobody holding its outcome --
    and silently, which is the one thing this module claims never to do. Measured before the fix:
    `abandonment reported: False`.

    The two accessors are genuinely not symmetric, which is why both are here. The pre-dispatch
    boundary check resolves the *token* before anything is dispatched, so a token that raises fails
    with no call to orphan -- `starts_the_call` is the assertion that pins that, and it breaks if the
    token ever stops being read before dispatch. The grace has no such earlier reader. Guarding both
    anyway: which accessor is touched first is an implementation detail, not a contract.
    """

    started = threading.Event()
    release = threading.Event()

    class BlockingAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            started.set()
            release.wait(timeout=5)
            return ModelTurn(final_text="nobody is waiting for this")

    def hostile() -> Any:
        raise RuntimeError("accessor exploded")

    runner = ModelCallRunner(
        adapter=BlockingAdapter(),
        cancel_grace_s=0.02,
        thread_name="nar-model-call-accessor-test",
        **{accessor: hostile},
    )

    async def run() -> None:
        await runner.acall(REQUEST)

    try:
        with caplog.at_level("WARNING", logger="monoid_agent_kernel.core.sync_bridge"):
            with pytest.raises(RuntimeError, match="accessor exploded"):
                asyncio.run(run())

        assert (
            started.is_set() is starts_the_call
        ), "the accessor was resolved on the wrong side of dispatch"
        reported = any("abandoned a synchronous call" in r.message for r in caplog.records)
        assert reported is starts_the_call, (
            "a call left running has to be detached and reported; one that never started has "
            f"nothing to report. started={started.is_set()} reported={reported}"
        )
    finally:
        release.set()
        for thread in threading.enumerate():
            if thread.name == "nar-model-call-accessor-test":
                thread.join(timeout=5)


# --- a stream the run gave up on must stop talking to the call that started it -------------------


class _CancellationSurvivingStreamAdapter:
    """A provider whose stream keeps producing after the run has cancelled it.

    The same callee shape `detach_unfinished_call` and `_aclose_within_grace` are both built for:
    the kernel can stop *waiting* for a provider, it cannot stop one. Spelled here as a suppressed
    `CancelledError` around the inner read, which is what a client that treats a dropped read as a
    hiccup and reconnects does to the cancellation aimed at it.
    """

    provider_name = "survivor"

    def __init__(self, tag: str, chunks: int = 40, per_chunk_s: float = 0.02) -> None:
        self._tag = tag
        self._chunks = chunks
        self._per_chunk_s = per_chunk_s

    async def astream_turn(self, request: ModelRequest):  # noqa: ANN202
        del request
        for index in range(self._chunks):
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(self._per_chunk_s)
            yield TextDelta(text=f"{self._tag}-{index}")


def test_a_stream_the_run_gave_up_on_stops_reaching_its_consumer() -> None:
    """`delta_consumer` belongs to one call, so nothing may reach it once that call has ended.

    `consume` runs as its own task, and that is what lets the two exist at once: the boundary
    releases `acall`, but a generator that survives the cancellation delivered to that task keeps
    yielding, and every later chunk was handed straight to the consumer of a call that had already
    raised `RunTimeout`. Nothing notices serially -- with the awaiting side gone there is no second
    call to confuse it with -- so the whole defect lives in the overlap.

    Measured before the guard: 19 of 25 chunks arrived after `acall` returned, 5 runs out of 5.

    Which side of the boundary a delivery fell on is decided by a flag flipped when `acall` returns,
    not by comparing timestamps to it. This test used to stamp `time.monotonic()` on each delivery
    and keep those `> ended`, and that filter cannot see a *single* late chunk: Windows'
    `time.monotonic()` advances in 15.625ms steps, so the leaked chunk carried exactly `ended` and a
    strict `>` dropped it. The version of the guard that delivers one chunk too many passed this test
    5/5 while its sibling caught it.
    """

    during: list[str] = []
    after: list[str] = []
    call_over = {"yes": False}

    def consume(chunk: Any) -> None:
        (after if call_over["yes"] else during).append(chunk.text)

    async def run() -> None:
        runner = ModelCallRunner(
            adapter=_CancellationSurvivingStreamAdapter("gone"), cancel_grace_s=0.02
        )
        with pytest.raises(RunTimeout):
            await runner.acall(REQUEST, deadline=time.time() + 0.10, delta_consumer=consume)
        call_over["yes"] = True
        await asyncio.sleep(0.4)  # long enough for many more chunks to be produced

    asyncio.run(run())

    assert during, "this test is meaningless unless the stream reached the consumer at all"
    assert not after, (
        f"{len(after)} chunk(s) reached the consumer after the call raised: {after[:5]}; an "
        "abandoned stream is still driving a consumer that belongs to a finished call"
    )


def test_one_turns_tokens_cannot_arrive_in_the_next_turns_stream() -> None:
    """The consequence that makes the rule above more than tidiness.

    `AgentLoop` passes `QueueEventSink.push_delta` as the consumer, and one sink is reused for every
    turn of a run -- `activate` rebinds it to the next turn's queue. So an abandoned stream that
    went on calling its consumer was not talking to nobody: its chunks were routed into the *next*
    turn's live stream, and a caller of `astream` saw the previous turn's tokens presented as this
    turn's output. Measured before the guard: 19 foreign chunks in the second turn's queue, 5/5.
    """

    class TurnTwoAdapter:
        async def astream_turn(self, request: ModelRequest):  # noqa: ANN202
            del request
            for index in range(3):
                await asyncio.sleep(0.05)
                yield TextDelta(text=f"turn-two-{index}")

    async def run() -> list[str]:
        loop = asyncio.get_running_loop()
        sink = QueueEventSink()

        first_queue: asyncio.Queue[Any] = asyncio.Queue()
        sink.activate(first_queue, loop)
        abandoned = ModelCallRunner(
            adapter=_CancellationSurvivingStreamAdapter("turn-one"), cancel_grace_s=0.02
        )
        with pytest.raises(RunTimeout):
            await abandoned.acall(
                REQUEST, deadline=time.time() + 0.10, delta_consumer=sink.push_delta
            )

        # The next turn reuses the sink, rebound to its own queue -- the loop's own wiring.
        second_queue: asyncio.Queue[Any] = asyncio.Queue()
        sink.activate(second_queue, loop)
        await ModelCallRunner(adapter=TurnTwoAdapter(), cancel_grace_s=0.02).acall(
            REQUEST, delta_consumer=sink.push_delta
        )
        await asyncio.sleep(0.3)  # the abandoned stream's remaining chunks would land in here
        sink.deactivate()

        drained: list[str] = []
        while not second_queue.empty():
            drained.append(getattr(second_queue.get_nowait(), "text", ""))
        return drained

    drained = asyncio.run(run())

    assert any(
        text.startswith("turn-two") for text in drained
    ), "this test is meaningless unless the second turn's own tokens reached its queue"
    foreign = [text for text in drained if text.startswith("turn-one")]
    assert (
        not foreign
    ), f"the abandoned turn's tokens were delivered into the next turn's stream: {foreign[:5]}"


def test_interruption_is_mid_turn_only_while_something_consumes_deltas() -> None:
    """Pins the documented cost of `--no-output-deltas` / `MONOID_OUTPUT_DELTAS=0`.

    `_adrive` takes the streaming path only when `delta_consumer is not None`, and the loop wires
    `should_abort` only in the same branch that wires the consumer. So switching delta publication
    off also moves Stop from "aborts within a token" to "waits for the in-flight call" -- which the
    release notes, the `--help` text, the `StudioConfig` comment and `docs/OBSERVABILITY.md` all
    claimed cost "live token rendering and nothing else".

    The coupling itself is deliberate and predates the switch: `emit_output_deltas` is `False` by
    default in the kernel, so this is its default behaviour, not a degradation past it. What was
    wrong was four documents saying otherwise. This test is here so that if the coupling is ever
    broken deliberately, the docs are what fails first.
    """
    stop_now: ShouldAbort = lambda: True  # noqa: E731

    async def with_consumer() -> ModelTurn:
        seen: list[Any] = []
        return (
            await ModelCallRunner(
                adapter=StreamingAdapter(chunks=[TextDelta(f"t{i}") for i in range(20)])
            ).acall(REQUEST, delta_consumer=seen.append, should_abort=stop_now)
        )[0]

    async def without_consumer() -> ModelTurn:
        return (
            await ModelCallRunner(adapter=StreamingAdapter()).acall(REQUEST, should_abort=stop_now)
        )[0]

    with pytest.raises(ModelCallAborted):
        asyncio.run(with_consumer())

    # No consumer: dispatch leaves the streaming branch entirely -- `one-shot fallback` is
    # `StreamingAdapter.next_turn`, so the generator that would have been polled between chunks was
    # never entered -- the abort predicate is never consulted, and the call runs to completion.
    assert asyncio.run(without_consumer()).final_text == "one-shot fallback"


# --- the receipt sink: a seam the caller's return value cannot reach ------------------------------


class _RaisingAdapter:
    def next_turn(self, request: ModelRequest) -> ModelTurn:
        del request
        raise ModelAdapterError(
            "upstream refused",
            provider_error_code="rate_limit",
            retryable=True,
            http_status=429,
        )


def test_a_failed_call_reaches_the_settled_sink_the_caller_never_receives_one_from() -> None:
    """The sink exists because `acall`'s return value is not a delivery channel for a failure.

    A failed call publishes its receipt and then re-raises (see `acall`), and it does not stamp the
    receipt onto the exception -- so the caller holding `turn, receipt = await runner.acall(...)`
    gets an exception and nothing else. A durable ledger wired to that return value would record
    only the calls that succeeded, which is the opposite of what an audit trail is for.

    Both halves are asserted from one runner: the success arm proves the sink is not a
    failure-only hook (and hands over the turn), and the failure arm proves the receipt carries
    the classification the exception did -- and no turn, because there is none.
    """
    recorded: list[SettledModelCall] = []
    runner = ModelCallRunner(adapter=SyncAdapter(), settled_sink=recorded.append)
    turn, returned = asyncio.run(runner.acall(REQUEST))

    assert turn.final_text == "answer"
    assert [call.receipt.request_digest for call in recorded] == [returned.request_digest]
    assert recorded[0].receipt.error_code == ""
    assert recorded[0].turn is turn

    failing = ModelCallRunner(adapter=_RaisingAdapter(), settled_sink=recorded.append)
    with pytest.raises(ModelAdapterError):
        asyncio.run(failing.acall(REQUEST))

    assert len(recorded) == 2
    failed = recorded[1].receipt
    assert recorded[1].turn is None
    assert failed.error_code != ""
    assert failed.provider_error_code == "rate_limit"
    assert failed.http_status == 429
    assert failed.retryable is True
    # The key is taken before dispatch, so a failed call is still identifiable -- which is the
    # whole reason to record it beside the successful ones rather than in a separate lane.
    assert failed.request_digest
    assert failed.digest_status == "ok"


def test_the_sink_is_handed_the_settled_receipt_and_not_the_one_before_dispatch() -> None:
    """`capture_downgrades` is resolved by delivery, so recording before it is recording a zero.

    `latency_ms` is stamped before delivery and `capture_downgrades` after it. A sink placed either
    side of `dispatch_model_call` sees one of them unset, and the field that goes quiet is the one
    that says a consumer was denied the content it asked for -- an audit record that always reads
    "nothing was withheld" is worse than one that omits the question.
    """
    recorded: list[SettledModelCall] = []
    observer = RecordingObserver()
    runner = ModelCallRunner(
        adapter=SyncAdapter(),
        # A policy that knows it *had* a redactor and no longer has one: the pipeline treats the
        # missing machinery as a redaction failure and downgrades this consumer to `digest`.
        subscriptions=(
            ModelIOSubscription(
                observer=observer,
                policy=CapturePolicy(mode="redacted", restored_without_redactor=True),
            ),
        ),
        settled_sink=recorded.append,
    )
    _turn, returned = asyncio.run(runner.acall(REQUEST))

    assert observer.captures[0].was_downgraded is True
    assert returned.capture_downgrades == 1
    assert len(recorded) == 1
    assert recorded[0].receipt.capture_downgrades == 1
    assert recorded[0].receipt.latency_ms == returned.latency_ms


def test_a_raising_sink_does_not_fail_a_call_the_provider_was_paid_for() -> None:
    """The containment rule observers already have, spelled for the sink.

    A recorder that cannot write is not a reason to discard an answer the provider has already
    billed for, and it is not a reason to reclassify a failure either: on the failure arm the
    `ModelAdapterError` must escape carrying the taxonomy it arrived with, not whatever the sink
    threw. That second half is the one a bare try/except around the call would not catch.
    """

    def explode(call: SettledModelCall) -> None:
        del call
        raise RuntimeError("ledger is on fire")

    turn, receipt = asyncio.run(
        ModelCallRunner(adapter=SyncAdapter(), settled_sink=explode).acall(REQUEST)
    )
    assert turn.final_text == "answer"
    assert receipt.error_code == ""

    with pytest.raises(ModelAdapterError) as failure:
        asyncio.run(
            ModelCallRunner(adapter=_RaisingAdapter(), settled_sink=explode).acall(REQUEST)
        )
    assert failure.value.provider_error_code == "rate_limit"
    assert failure.value.http_status == 429
    assert failure.value.retryable is True


def test_a_dispatch_that_raises_still_records_the_call_it_could_not_deliver(
    monkeypatch: Any,
) -> None:
    """Recording is in a `finally`, not after the delivery it does not depend on.

    `dispatch_model_call` contains its own observers, so this is a defence against the pipeline
    itself breaking -- and the two call sites fail in opposite directions when it does. The failure
    path publishes inside `contextlib.suppress(Exception)`, so a record placed after delivery is
    lost with no trace; the success path publishes unguarded, so the same placement fails a paid
    call. A naive "record after dispatch" implementation passes the success arm of every other test
    in this section and dies here.
    """
    from monoid_agent_kernel import model_call as model_call_module

    def explode(**kwargs: Any) -> Any:
        del kwargs
        raise RuntimeError("dispatch is broken")

    monkeypatch.setattr(model_call_module, "dispatch_model_call", explode)
    recorded: list[SettledModelCall] = []
    observer = RecordingObserver()
    runner = ModelCallRunner(
        adapter=SyncAdapter(),
        subscriptions=(
            ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
        ),
        settled_sink=recorded.append,
    )

    with pytest.raises(RuntimeError, match="dispatch is broken"):
        asyncio.run(runner.acall(REQUEST))

    # Delivery never happened, and the record still names the call.
    assert observer.captures == []
    assert len(recorded) == 1
    assert recorded[0].receipt.request_digest
    # The receipt a broken dispatch leaves behind is the pre-delivery one, so the count it could
    # not resolve stays at its floor rather than being invented.
    assert recorded[0].receipt.capture_downgrades == 0


# --- One size gate, three named refusals (W6-2) ---------------------------------------------------


def test_a_payload_between_the_old_digest_cap_and_the_wire_cap_gets_a_key() -> None:
    """4 MiB was the digest cap while 8,000,000 bytes was the message-log cap, so the band between
    them shipped calls that transmitted successfully and silently had no replay key. One constant
    now gates both: what can be sent can be keyed."""
    _turn, receipt = asyncio.run(
        ModelCallRunner(adapter=SyncAdapter()).acall(
            ModelRequest(instruction="hi", system_prompt="x" * (5 * 1024 * 1024), tools=())
        )
    )
    assert receipt.request_digest != ""
    assert receipt.digest_status == "ok"


def test_a_payload_over_the_ceiling_is_a_named_condition_not_a_missing_key() -> None:
    """`absent` used to cover both "cannot be encoded" (a defect in the payload) and "over the cap"
    (an operational condition, answered by raising the cap or offloading). A consumer holding a
    keyless record could not tell which one it was looking at."""
    _turn, receipt = asyncio.run(
        ModelCallRunner(adapter=SyncAdapter()).acall(
            ModelRequest(instruction="hi", system_prompt="x" * 8_000_001, tools=())
        )
    )
    assert receipt.request_digest == ""
    assert receipt.digest_status == "too_large"


def test_an_unencodable_payload_is_still_absent_so_the_distinction_is_real() -> None:
    """The split would be cosmetic if every refusal drifted to the new value."""
    hostile = _encoded_digest({"v": 10**5000})
    assert (hostile.digest, hostile.status) == ("", "absent")

    oversized = _encoded_digest({"v": "x" * 8_000_001})
    assert (oversized.digest, oversized.status) == ("", "too_large")


def test_a_payload_under_the_old_cap_keeps_the_digest_it_always_had() -> None:
    """Witness, not red: the cap moved up, which only turns refusals into keys. A payload that had
    a digest under the old cap must keep it byte-for-byte, or the raise would rekey every recorded
    corpus. Recomputed through `canonical_sha256` rather than compared to a golden constant, so the
    thing pinned is the encoding itself."""
    payload = _request_payload(REQUEST, ModelConfig(), provider="fake")

    assert _digest(payload) == canonical_sha256(payload)


# --- The preimage the sink receives (W6-2) --------------------------------------------------------


def test_the_preimage_the_sink_receives_is_the_bytes_the_key_was_hashed_over() -> None:
    """D-a's whole content: the sink is handed the encoder's own output, not a re-derivation it
    would have to reconstruct from the receipt -- which it cannot, because the key's provider term
    came from an adapter resolution only the call itself performed."""
    import hashlib

    recorded: list[SettledModelCall] = []
    runner = ModelCallRunner(
        adapter=SyncAdapter(), settled_sink=recorded.append, capture_request_preimage=True
    )
    _turn, receipt = asyncio.run(runner.acall(REQUEST))

    call = recorded[0]
    assert call.request_preimage is not None
    assert hashlib.sha256(call.request_preimage).hexdigest() == receipt.request_digest
    assert receipt.digest_status == "ok"


def test_a_failed_call_still_hands_the_sink_its_preimage() -> None:
    """The corpus wants "what was asked" for failures too; the request side of a failed call is
    exactly as recordable as a successful one's, because the key is taken before dispatch."""
    import hashlib

    recorded: list[SettledModelCall] = []
    failing = ModelCallRunner(
        adapter=_RaisingAdapter(), settled_sink=recorded.append, capture_request_preimage=True
    )
    with pytest.raises(ModelAdapterError):
        asyncio.run(failing.acall(REQUEST))

    call = recorded[0]
    assert call.turn is None
    assert call.request_preimage is not None
    assert hashlib.sha256(call.request_preimage).hexdigest() == call.receipt.request_digest


def test_a_call_refused_before_its_key_hands_the_sink_no_preimage() -> None:
    """A boundary crossed before the digest means no key and therefore no preimage -- `None` here
    is truthful, not a capture failure."""
    recorded: list[SettledModelCall] = []
    runner = ModelCallRunner(
        adapter=SyncAdapter(), settled_sink=recorded.append, capture_request_preimage=True
    )
    with pytest.raises(RunTimeout):
        asyncio.run(runner.acall(REQUEST, deadline=time.monotonic() - 1.0))

    assert len(recorded) == 1
    assert recorded[0].request_preimage is None
    assert recorded[0].receipt.digest_status == "not_reached"


def test_the_preimage_is_not_captured_unless_asked_for() -> None:
    """The knob exists so a ledger-only run does not hold up to 8 MB per in-flight call for bytes
    its sink never reads. The digests themselves are computed either way."""
    recorded: list[SettledModelCall] = []
    runner = ModelCallRunner(adapter=SyncAdapter(), settled_sink=recorded.append)
    _turn, receipt = asyncio.run(runner.acall(REQUEST))

    assert recorded[0].request_preimage is None
    assert receipt.request_digest != ""
    assert receipt.digest_status == "ok"


# --- the kernel retry layer ---------------------------------------------------------------------


def _kernel_model(max_attempts: int = 3) -> ModelConfig:
    """A kernel-layer policy with a zero schedule, so tests never actually wait."""

    return ModelConfig(
        retry=ModelRetryConfig(
            layer="kernel", max_attempts=max_attempts, initial_delay_s=0.0, jitter_s=0.0
        )
    )


class _FlakyAdapter:
    """Fails `failures` times through `error_factory`, then answers with billed usage."""

    def __init__(self, failures: int, error_factory: Any = None) -> None:
        self.calls = 0
        self.failures = failures
        self.error_factory = error_factory or (
            lambda: ModelAdapterError("transient", retryable=True)
        )

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        del request
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error_factory()
        return ModelTurn(final_text="answer", usage={"output_tokens": 7})


def _acall(adapter: Any, request: ModelRequest, **kwargs: Any) -> RecordingObserver:
    observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=adapter,
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
            ),
        ).acall(request, **kwargs)

    asyncio.run(run())
    return observer


def test_the_kernel_layer_pays_for_another_attempt_and_the_receipt_counts_it() -> None:
    """Under `layer="kernel"` the runner re-dispatches a retryable failure; `attempts` counts it.

    `provider_retried` stays False on purpose: kernel attempts are the kernel's own fact,
    carried by `attempts`, and the adapter flag keeps meaning what it always meant -- a loop
    BELOW the adapter boundary ran. One settled capture for the whole logical call, because
    retry is transport: the request was keyed once and answered once.
    """

    adapter = _FlakyAdapter(failures=1)
    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())
    observer = _acall(adapter, request)

    assert adapter.calls == 2
    assert len(observer.captures) == 1
    receipt = observer.captures[0].receipt
    assert receipt.succeeded is True
    assert receipt.attempts == 2
    assert receipt.provider_retried is False


def test_the_default_layer_still_makes_exactly_one_attempt() -> None:
    adapter = _FlakyAdapter(failures=1)
    request = ModelRequest(instruction="hi", system_prompt="sys", tools=())

    with pytest.raises(ModelAdapterError):
        _acall(adapter, request)

    assert adapter.calls == 1


# --- the idempotency key: issued at keying, constant across dispatches --------------------------


class _KeyCapturingAdapter:
    """Fails ``failures`` times, capturing the request's idempotency key at every dispatch."""

    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.seen: list[str] = []

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        self.seen.append(request.idempotency_key)
        if len(self.seen) <= self.failures:
            raise ModelAdapterError("transient", retryable=True)
        return ModelTurn(final_text="answer")


def test_the_runner_issues_one_key_per_call_and_every_dispatch_carries_it() -> None:
    """Issued in the keying block -- per call, before the first dispatch -- and constant across
    kernel re-dispatches: the loop reuses the request rather than rebuilding it, so the key the
    receipt records is the key every attempt presented."""

    adapter = _KeyCapturingAdapter(failures=2)
    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())
    receipt = _acall(adapter, request).captures[0].receipt

    assert receipt.attempts == 3
    assert receipt.idempotency_key.startswith("idem_")
    assert adapter.seen == [receipt.idempotency_key] * 3


def test_every_call_gets_its_own_key_even_over_the_same_request_object() -> None:
    """Two calls are two retry scopes even when their content is byte-identical. Identical
    requests share a replay slot by design -- content cannot separate them -- which is exactly
    why the token that separates their provider work is issued per call, not derived. Issuance
    is uniform: this adapter never opens a socket and its calls are keyed all the same."""

    adapter = _KeyCapturingAdapter()
    first = _acall(adapter, REQUEST).captures[0].receipt
    second = _acall(adapter, REQUEST).captures[0].receipt

    assert first.idempotency_key.startswith("idem_")
    assert second.idempotency_key.startswith("idem_")
    assert first.idempotency_key != second.idempotency_key


def test_the_runner_is_the_single_issuer_and_overwrites_a_caller_value() -> None:
    """A respected caller value would let one request object hand two calls the same scope --
    the collision the per-call issuer exists to prevent -- so the field is a carriage channel,
    not an input."""

    adapter = _KeyCapturingAdapter()
    request = ModelRequest(
        instruction="hi", system_prompt="sys", tools=(), idempotency_key="caller-chosen"
    )
    receipt = _acall(adapter, request).captures[0].receipt

    assert receipt.idempotency_key != "caller-chosen"
    assert receipt.idempotency_key.startswith("idem_")
    assert adapter.seen == [receipt.idempotency_key]


def test_the_minted_key_satisfies_the_rule_its_transports_enforce() -> None:
    """The mint and the validator must not drift: every edge omits or refuses a key outside
    the token shape, so a mint that ever left it would silently stop being carried.

    Read off ``providers.base``, where the minter lives because the runner is not its only
    caller -- the reference gateway keys the upstream hop it drives with the same function.
    Two copies of the expression is how the two issuers would come to differ.
    """

    from monoid_agent_kernel.core.model_io import is_valid_idempotency_key
    from monoid_agent_kernel.providers.base import new_idempotency_key

    for _ in range(64):
        assert is_valid_idempotency_key(new_idempotency_key())


class _NeverUnequal(str):
    """A ``str`` that answers *for* its value instead of about it: ``__ne__`` returns False,
    so any guard spelling "is this the in-band absence?" as ``value != ""`` skips the check
    behind it while the transport goes on reading the underlying string."""

    def __ne__(self, other: object) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return True

    __hash__ = str.__hash__


@pytest.mark.parametrize(
    "hostile",
    [
        pytest.param("ok\r\n X-Injected: yes", id="obs-fold"),
        pytest.param("ok\r\nX-Injected: yes", id="bare-crlf"),
        pytest.param("ok\nforged", id="lf"),
        pytest.param("A" * 129, id="too-long"),
        pytest.param("-leading-punctuation", id="bad-first-character"),
        pytest.param("key with spaces", id="space"),
        pytest.param("ké", id="non-ascii"),
        # Not strings at all. Only the empty string spells absence here; every other falsy
        # value is a caller who supplied *something* and would otherwise have watched it
        # vanish at the transport, which omits what it cannot validate. A truthiness
        # pre-filter reads all six as "no key given" -- the absence-vs-value conflation this
        # field has now produced three times, at three different types.
        pytest.param(None, id="none"),
        pytest.param(False, id="false"),
        pytest.param(0, id="zero"),
        pytest.param(0.0, id="zero-float"),
        pytest.param([], id="empty-list"),
        pytest.param({}, id="empty-dict"),
        # The seventh, and the only one of them that IS a string: a subclass whose ``__ne__``
        # returns False answers the ``!= ""`` pre-filter for the value rather than about it,
        # so the pattern check behind it never runs. The ledger mint carried the same shape
        # and the same hole; both ask through one predicate now.
        pytest.param(_NeverUnequal("bad\nkey"), id="equality-overload"),
    ],
)
def test_request_ingress_refuses_a_key_that_could_not_go_on_a_header(hostile: object) -> None:
    """Refused where this repo refuses a non-finite control or a malformed output_schema.

    The runner mints after normalization so a run-driven call never reaches this branch; it
    exists for the direct integrator, whose bad key would otherwise reach a transport that --
    probed -- neither `http.client` nor `httpx` defends against when it is obs-folded.
    """

    with pytest.raises(ValueError, match="idempotency_key"):
        normalize_model_request(
            ModelRequest(instruction="hi", system_prompt="sys", tools=(), idempotency_key=hostile)
        )

    # Counterweight: the shape the runner mints survives ingress untouched.
    kept = normalize_model_request(
        ModelRequest(
            instruction="hi", system_prompt="sys", tools=(), idempotency_key="idem_abc123"
        )
    )
    assert kept.idempotency_key == "idem_abc123"


def test_a_call_refused_before_keying_records_no_key() -> None:
    """The keying block sits past the cancel/deadline check, so a refused call was never keyed:
    ``""`` beside ``attempts == 0``, the receipt's own two-armed audit shape."""

    adapter = _KeyCapturingAdapter()
    observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=adapter,
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
            ),
        ).acall(REQUEST, deadline=time.time() - 1.0)

    with pytest.raises(RunTimeout):
        asyncio.run(run())

    receipt = observer.captures[0].receipt
    assert receipt.attempts == 0
    assert receipt.idempotency_key == ""
    assert adapter.seen == []


@pytest.mark.parametrize(
    "error_factory",
    [
        pytest.param(lambda: ModelAdapterError("terminal", retryable=False), id="not-retryable"),
        pytest.param(
            lambda: ModelAdapterError("fix config", retryable=True, config_recoverable=True),
            id="config-recoverable",
        ),
        pytest.param(lambda: RuntimeError("not an adapter error"), id="untyped"),
    ],
)
def test_the_kernel_loop_refuses_what_the_taxonomy_refuses(error_factory: Any) -> None:
    """The kernel judges by the taxonomy alone: no `ModelAdapterError`, no retry; marked
    non-retryable or config-recoverable, no retry -- re-sending cannot help a call whose
    config must change first."""

    adapter = _FlakyAdapter(failures=99, error_factory=error_factory)
    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())

    with pytest.raises((ModelAdapterError, RuntimeError)):
        _acall(adapter, request)

    assert adapter.calls == 1


def test_exhaustion_re_raises_the_last_error_with_the_attempts_it_cost() -> None:
    adapter = _FlakyAdapter(failures=99)
    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())
    observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=adapter,
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
            ),
        ).acall(request)

    with pytest.raises(ModelAdapterError) as caught:
        asyncio.run(run())

    assert caught.value.retryable is True
    assert adapter.calls == 3
    receipt = observer.captures[0].receipt
    assert receipt.attempts == 3
    assert receipt.succeeded is False


def test_swallowed_attempts_usage_reaches_the_receipt() -> None:
    """An attempt the loop absorbed still cost tokens, and the receipt is the only carrier left.

    The turn keeps the final answer's own usage -- it describes what the model said -- while
    the receipt sums what the whole logical call cost, which is what an audit or a meter
    reads. The run's cumulative token budget still counts settled turns; that boundary is
    documented, not accidental.
    """

    def billed_failure() -> ModelAdapterError:
        error = ModelAdapterError("billed refusal", retryable=True)
        mark_provider_usage(error, {"output_tokens": 5})
        return error

    adapter = _FlakyAdapter(failures=1, error_factory=billed_failure)
    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())
    observer = _acall(adapter, request)

    receipt = observer.captures[0].receipt
    assert receipt.usage["output_tokens"] == 12
    assert receipt.succeeded is True


def test_an_exhausted_call_still_accounts_every_billed_attempt() -> None:
    def billed_failure() -> ModelAdapterError:
        error = ModelAdapterError("billed refusal", retryable=True)
        mark_provider_usage(error, {"output_tokens": 5})
        return error

    adapter = _FlakyAdapter(failures=99, error_factory=billed_failure)
    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())
    observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=adapter,
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
            ),
        ).acall(request)

    with pytest.raises(ModelAdapterError):
        asyncio.run(run())

    # Two swallowed attempts at 5 plus the final error's own stamp, which `with_error` reads.
    assert observer.captures[0].receipt.usage["output_tokens"] == 15


def test_a_turn_the_normalizer_refuses_still_reports_what_it_was_billed() -> None:
    """The refusal is about the turn's shape; the counts beside it were still charged.

    ``normalize_model_turn`` builds a *fresh* ``ModelAdapterError``, so the escaping error
    carried no ``provider_usage`` and ``with_error`` -- which reads exactly that stamp -- put an
    empty usage on the receipt, while the raw turn holding well-formed counts was still in
    scope one frame away. A paid call then left no trace in the metrics or in the cumulative
    token budget, which is the failure ``mark_provider_usage`` exists to prevent and already
    prevents for the applied-parameters refusals that parse a turn, read its usage, and only
    then reject it.

    The malformed field is deliberately NOT the usage: a bad count is dropped by
    ``_recordable_usage`` and there is nothing to carry. This is the shape where the bill is
    intact and something else about the turn is not.

    The entry the log gets for this dispatch carries the same counts, because the sum invariant
    is what would otherwise refuse the receipt -- one attempt, and the receipt's total is that
    attempt's.
    """

    class BilledButMalformedAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(
                final_text="answer",
                usage={"input_tokens": 7, "output_tokens": 11},
                provider_retried="yes",  # type: ignore[arg-type]
            )

    observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=BilledButMalformedAdapter(),
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
            ),
        ).acall(REQUEST)

    with pytest.raises(ModelAdapterError, match="non-portable response") as caught:
        asyncio.run(run())

    receipt = observer.captures[0].receipt
    assert receipt.succeeded is False
    assert dict(receipt.usage) == {"input_tokens": 7, "output_tokens": 11}
    # The stamp itself, so a receipt that got the counts some other way cannot pass this.
    assert provider_usage_of(caught.value) == {"input_tokens": 7, "output_tokens": 11}
    # And the ledger still adds up to the bill it is a breakdown of.
    assert [dict(entry.usage) for entry in receipt.attempt_log] == [
        {"input_tokens": 7, "output_tokens": 11}
    ]
    # The refused turn's own retry claim is still not consulted -- only its counts travel.
    assert receipt.attempt_log[0].provider_retried is False


def test_every_arm_of_the_turn_normalizer_carries_what_the_turn_was_billed() -> None:
    """Derived census, widened off ``loop.py`` to the function every caller goes through.

    This is the third appearance of one shape -- a fresh exception built where a stamped one was
    available. ``loop.py``'s wrap was the first, its boundary arms the second, and the existing
    census in ``test_loop.py`` is scoped to that file, so it walked straight past this one. The
    widening is deliberately not "the same AST rule over more files": the carrier here is a
    single function, so the instrument asserts that *it* carries on every arm that can leave,
    and that no caller re-implements the carrying.

    Scoped to the function rather than to its four call sites because one of them --
    ``normalize_model_turn(adapter.next_turn(...))`` in the gateway service -- never binds the
    raw turn to a name, so a per-caller stamp is not merely duplicated there, it is impossible.

    What this census still cannot see, stated because a green census that is blind is worse than
    no census:

    - It reads ``providers/base.py`` only. A *different* function that refuses a billed turn
      somewhere else is a fourth appearance of the shape, and nothing here enumerates it.
    - It proves each arm calls the stamper, not that the stamper was handed the right value: a
      site passing ``{}`` or the wrong object still satisfies it. The behavioural test above is
      what pins the value, and only for the one arm it drives.
    - ``_refused_turn_usage`` returning ``{}`` for a shape it does not recognize is invisible
      here and would look identical to a turn that reported nothing.
    - It does not forbid a caller from stamping too. It cannot: ``model_call.py`` legitimately
      stamps the *cumulative* whole-call bill on the escaping error, which is a different fact
      about a different total, and no syntactic rule separates that from a re-implementation of
      this one. An earlier draft of this census asserted "no caller stamps" and reddened on that
      correct site -- an instrument needing an exemption list for a legitimate caller is
      measuring the wrong thing, so the clause was removed rather than given one.
    """

    import ast

    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "monoid_agent_kernel"
        / "providers"
        / "base.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    normalizer = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "normalize_model_turn"
    )

    # Every handler that lets an exception leave -- re-raised or rebuilt -- stamps first.
    unstamped = {}
    for handler in ast.walk(normalizer):
        if not isinstance(handler, ast.ExceptHandler):
            continue
        leaves = any(isinstance(node, ast.Raise) for node in ast.walk(handler))
        stamps = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "mark_provider_usage"
            for node in ast.walk(handler)
        )
        if leaves and not stamps:
            unstamped[handler.lineno] = ast.unparse(handler).splitlines()[0]

    assert unstamped == {}, {
        "arms_that_leave_without_carrying_the_bill": unstamped,
        "hint": "mark_provider_usage(<escaping error>, _refused_turn_usage(turn))",
    }

    # Both arms exist, so deleting one cannot pass by deleting the shape above.
    stamp_sites = [
        node.lineno
        for node in ast.walk(normalizer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "mark_provider_usage"
    ]
    assert len(stamp_sites) == 2, {"stamp_sites": stamp_sites}

    # Every caller reaches the normalizer by that name, so the carrying above is on the path of
    # all of them. Enumerated rather than asserted about one, because the count is the part that
    # goes stale: a fifth caller is covered by construction, and a caller that stopped going
    # through this function would drop out of this set and be visible here.
    repo_src = Path(__file__).resolve().parents[1] / "src"
    callers = sorted(
        str(path.relative_to(repo_src)).replace("\\", "/")
        for path in repo_src.rglob("*.py")
        if path != source_path
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "normalize_model_turn"
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        )
    )
    assert callers == [
        "monoid_agent_kernel/model_call.py",
        "monoid_agent_kernel/model_lifecycle.py",
        "monoid_agent_kernel/reference/llm_gateway/service.py",
    ], {"callers": callers}


def test_a_refused_backoff_does_not_bill_its_attempt_twice(monkeypatch: Any) -> None:
    """The attempt whose backoff cannot fit the deadline is the terminal one, billed once.

    `with_error` reads the escaping error's own `provider_usage` stamp, so an attempt belongs
    in `spent_usage` only once the loop has committed to ABSORBING it. Recorded any earlier,
    the one error the deadline check then re-raises is both the swallowed attempt and the
    terminal outcome, and the receipt carries its cost twice -- a meter reading double for a
    call that reached the provider once. The counter-arm is the test above: attempts the loop
    really did absorb still sum, 15 for three billed calls at 5.
    """

    def billed_failure() -> ModelAdapterError:
        error = ModelAdapterError("billed refusal", retryable=True)
        mark_provider_usage(error, {"output_tokens": 5})
        return error

    monkeypatch.setattr("monoid_agent_kernel.model_call.retry_delay_s", lambda *_a: 10.0)
    adapter = _FlakyAdapter(failures=99, error_factory=billed_failure)
    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())
    observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=adapter,
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
            ),
        ).acall(request, deadline=time.time() + 5.0)

    with pytest.raises(ModelAdapterError):
        asyncio.run(run())

    assert adapter.calls == 1
    receipt = observer.captures[0].receipt
    assert receipt.attempts == 1
    assert receipt.usage["output_tokens"] == 5


def test_the_attempt_log_names_every_dispatch_and_sums_to_the_receipt() -> None:
    """One entry per kernel dispatch, in order, whose usage totals are the receipt's usage.

    The absorbed attempt keeps its own taxonomy and its own bill; the answering attempt keeps
    the turn's. The receipt's `usage` is exactly their sum on the success exit -- the invariant
    that makes the log auditable instead of decorative.
    """

    def billed_failure() -> ModelAdapterError:
        error = ModelAdapterError("billed refusal", retryable=True)
        mark_provider_usage(error, {"output_tokens": 5})
        return error

    adapter = _FlakyAdapter(failures=1, error_factory=billed_failure)
    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())
    observer = _acall(adapter, request)

    receipt = observer.captures[0].receipt
    assert receipt.attempts == 2
    assert [entry.index for entry in receipt.attempt_log] == [1, 2]
    first, second = receipt.attempt_log
    assert first.succeeded is False
    assert first.retryable is True
    assert dict(first.usage) == {"output_tokens": 5}
    assert first.stream_committed is False
    assert second.succeeded is True
    # The answering entry mirrors the receipt's own normalized reading (`_recordable_usage`
    # zero-fills the core three) -- the same shape, or the sum below could not close.
    assert second.usage["output_tokens"] == 7
    assert all(entry.elapsed_ms >= 0 for entry in receipt.attempt_log)
    summed: dict[str, int] = {}
    for entry in receipt.attempt_log:
        for key, value in entry.usage.items():
            summed[key] = summed.get(key, 0) + value
    assert summed == dict(receipt.usage)


def test_the_backoff_wait_lands_on_the_entry_it_delayed(monkeypatch: Any) -> None:
    """`backoff_ms` is the wait BEFORE that entry's dispatch: 0 on the first entry, and the
    sleep an absorbed failure earned lands on the entry it delayed, not the one that caused it.

    Measured around the wait rather than copied from the schedule, so a capped or interrupted
    sleep records what happened rather than what was asked for. The lower bound is generous on
    purpose -- the pin is "a real wait was recorded on the right entry", not a timer-precision
    claim.
    """

    monkeypatch.setattr("monoid_agent_kernel.model_call.retry_delay_s", lambda *_a: 0.03)
    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())

    observer = _acall(_FlakyAdapter(failures=1), request)

    receipt = observer.captures[0].receipt
    first, second = receipt.attempt_log
    assert first.backoff_ms == 0
    assert second.backoff_ms is not None
    assert second.backoff_ms >= 25


def test_a_zero_schedule_records_a_zero_wait_not_an_absent_one() -> None:
    """0 and None are different answers on this field: the runner always knows the wait it
    imposed, so a zero-schedule run records 0 -- absence stays reserved for entries parsed
    from lines written before the field existed."""

    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())

    observer = _acall(_FlakyAdapter(failures=1), request)

    receipt = observer.captures[0].receipt
    assert [entry.backoff_ms for entry in receipt.attempt_log] == [0, 0]


def test_an_exhausted_calls_waits_are_logged_and_fit_inside_its_latency(monkeypatch: Any) -> None:
    """The failure exit threads the wait too, and the timeline algebra closes: dispatch times
    plus recorded waits never exceed the whole call's `latency_ms` -- the remainder is keying
    and settle overhead. Every duration is floored from the same monotonic clock, and floors
    sum to at most the floor of the sum, so the inequality is exact rather than statistical.
    """

    monkeypatch.setattr("monoid_agent_kernel.model_call.retry_delay_s", lambda *_a: 0.03)
    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())
    observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=_FlakyAdapter(failures=99),
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
            ),
        ).acall(request)

    with pytest.raises(ModelAdapterError):
        asyncio.run(run())

    receipt = observer.captures[0].receipt
    assert receipt.attempts == 3
    waits = [entry.backoff_ms for entry in receipt.attempt_log]
    assert waits[0] == 0
    assert all(wait is not None and wait >= 25 for wait in waits[1:])
    assert (
        sum(entry.elapsed_ms for entry in receipt.attempt_log)
        + sum(wait or 0 for wait in waits)
        <= receipt.latency_ms
    )


def test_a_refused_turns_entry_carries_the_wait_that_preceded_it(monkeypatch: Any) -> None:
    """The third construction site holds the rule the other two hold: the terminal entry for a
    turn the normalizer refuses is built outside the dispatch loop, and it still names the
    backoff that delayed its dispatch."""

    class _FailsThenRefuses:
        def next_turn(self, request: ModelRequest) -> Any:
            del request
            self.calls = getattr(self, "calls", 0) + 1
            if self.calls == 1:
                raise ModelAdapterError("transient", retryable=True)
            return ModelTurn(final_text="answer", usage={"output_tokens": "seven"})  # type: ignore[dict-item]

    monkeypatch.setattr("monoid_agent_kernel.model_call.retry_delay_s", lambda *_a: 0.03)
    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())
    observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=_FailsThenRefuses(),
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
            ),
        ).acall(request)

    with pytest.raises(Exception):
        asyncio.run(run())

    receipt = observer.captures[0].receipt
    assert receipt.attempts == 2
    first, second = receipt.attempt_log
    assert first.backoff_ms == 0
    assert second.backoff_ms is not None
    assert second.backoff_ms >= 25


def test_a_single_dispatch_call_logs_one_entry_and_a_refused_call_logs_none() -> None:
    """The log is not a kernel-layer exclusive: every settled call names its dispatches.

    Under the default layer a call is one dispatch, so the log is one entry mirroring the
    receipt. A call refused before the adapter was reached made no dispatch, and its empty
    log says so beside `attempts == 0` -- the two-armed invariant, exercised on both arms.
    """

    observer = _acall(SyncAdapter(), REQUEST)
    receipt = observer.captures[0].receipt
    assert receipt.attempts == 1
    (entry,) = receipt.attempt_log
    assert entry.index == 1
    assert entry.succeeded is True
    assert dict(entry.usage) == dict(receipt.usage)

    refused_observer = RecordingObserver()

    async def refuse() -> None:
        await ModelCallRunner(
            adapter=SyncAdapter(),
            subscriptions=(
                ModelIOSubscription(
                    observer=refused_observer, policy=CapturePolicy(mode="digest")
                ),
            ),
        ).acall(REQUEST, deadline=time.time() - 1.0)

    with pytest.raises(RunTimeout):
        asyncio.run(refuse())

    refused = refused_observer.captures[0].receipt
    assert refused.attempts == 0
    assert refused.attempt_log == ()


def test_an_exhausted_call_logs_every_attempt_and_restamps_the_cumulative_cost() -> None:
    """The terminal error leaves carrying what the whole logical call cost.

    The loop's failure accounting reads the escaping error's stamp (`_billed_usage`), and a
    stamp naming only the last attempt under-counts every absorbed one. Restamped after the
    failed receipt is built -- `with_error` reads this same stamp, and a cumulative stamp read
    back there would land the absorbed spend on the receipt twice. The log names all three
    dispatches with their own bills; the receipt and the stamp agree on the sum.
    """

    def billed_failure() -> ModelAdapterError:
        error = ModelAdapterError("billed refusal", retryable=True)
        mark_provider_usage(error, {"output_tokens": 5})
        return error

    adapter = _FlakyAdapter(failures=99, error_factory=billed_failure)
    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())
    observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=adapter,
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
            ),
        ).acall(request)

    with pytest.raises(ModelAdapterError) as caught:
        asyncio.run(run())

    receipt = observer.captures[0].receipt
    assert receipt.attempts == 3
    assert [entry.index for entry in receipt.attempt_log] == [1, 2, 3]
    assert all(not entry.succeeded for entry in receipt.attempt_log)
    assert all(dict(entry.usage) == {"output_tokens": 5} for entry in receipt.attempt_log)
    assert receipt.usage["output_tokens"] == 15
    assert provider_usage_of(caught.value) == {"output_tokens": 15}


def test_a_refused_backoff_logs_its_attempt_once_and_keeps_the_stamp_it_earned(
    monkeypatch: Any,
) -> None:
    """The deadline-refusal arm of the absorb commit point, at log level.

    The attempt whose backoff cannot fit the deadline is the terminal one: one entry, its own
    bill, and the escaping error still carries its own stamp untouched -- nothing was absorbed,
    so there is nothing cumulative to say.
    """

    def billed_failure() -> ModelAdapterError:
        error = ModelAdapterError("billed refusal", retryable=True)
        mark_provider_usage(error, {"output_tokens": 5})
        return error

    monkeypatch.setattr("monoid_agent_kernel.model_call.retry_delay_s", lambda *_a: 10.0)
    adapter = _FlakyAdapter(failures=99, error_factory=billed_failure)
    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())
    observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=adapter,
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
            ),
        ).acall(request, deadline=time.time() + 5.0)

    with pytest.raises(ModelAdapterError) as caught:
        asyncio.run(run())

    receipt = observer.captures[0].receipt
    assert receipt.attempts == 1
    (entry,) = receipt.attempt_log
    assert dict(entry.usage) == {"output_tokens": 5}
    assert receipt.usage["output_tokens"] == 5
    assert provider_usage_of(caught.value) == {"output_tokens": 5}


def test_the_channel_report_lands_on_the_attempt_that_made_it() -> None:
    """Per-attempt attribution: the adapter's own loop reported during dispatch two, so entry
    two carries it and entry one does not. The receipt's flag stays the whole call's."""

    class SecondDispatchReports:
        def __init__(self) -> None:
            self.calls = 0

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            self.calls += 1
            if self.calls == 1:
                raise ModelAdapterError("transient", retryable=True)
            report_provider_retried()
            return ModelTurn(final_text="answer", usage={"output_tokens": 7})

    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())
    observer = _acall(SecondDispatchReports(), request)

    receipt = observer.captures[0].receipt
    first, second = receipt.attempt_log
    assert first.provider_retried is False
    assert second.provider_retried is True
    assert receipt.provider_retried is True


def test_a_turn_declared_retry_lands_on_the_attempt_that_answered() -> None:
    """The outcome-carried flag is attempt-scoped too: the turn that declares it belongs to
    exactly one dispatch."""

    class TurnDeclares:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(final_text="answer", provider_retried=True)

    observer = _acall(TurnDeclares(), REQUEST)

    receipt = observer.captures[0].receipt
    (entry,) = receipt.attempt_log
    assert entry.provider_retried is True
    assert receipt.provider_retried is True


def test_a_turn_the_normalizer_refuses_still_logs_its_dispatch() -> None:
    """The fallback arm: the failure happened between the dispatch's return and the settle,
    so no attempt-scoped probe exists and the entry mirrors the receipt -- driven, not
    assumed, because the two arms build the entry from different sources."""

    class MalformedTurn:
        def next_turn(self, request: ModelRequest) -> Any:
            del request
            return ModelTurn(final_text="answer", usage={"output_tokens": "seven"})  # type: ignore[dict-item]

    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())
    observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=MalformedTurn(),
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
            ),
        ).acall(request)

    with pytest.raises(Exception):
        asyncio.run(run())

    receipt = observer.captures[0].receipt
    assert receipt.attempts == 1
    (entry,) = receipt.attempt_log
    assert entry.succeeded is False
    assert entry.error_code == receipt.error_code


def test_a_refused_turn_does_not_inherit_an_earlier_attempts_retry_report() -> None:
    """The fallback arm's retry flag is attempt-scoped too, and it was the whole call's fold.

    The entry for a turn the normalizer refuses is built from the failed receipt, because that
    is where the exception's facts were already extracted. ``provider_retried`` is the one field
    on that receipt which is NOT this attempt's: the settle handler folds the call's whole retry
    history onto the escaping error first (``if progress.retried: mark_provider_retried(exc)``),
    and ``with_error`` reads it back. So an adapter that retried internally during dispatch one
    and then returned an unusable turn on dispatch two marked dispatch TWO as having reported a
    retry it never made -- an audit log that misattributes the fact it exists to attribute.

    The sibling arms already avoid exactly this: the in-loop entry probes the exception before
    the fold is applied, and the success entry counts channel reports across its own dispatch.
    This is the third construction site, holding the rule the other two hold.
    """

    class ReportsThenRefuses:
        def __init__(self) -> None:
            self.calls = 0

        def next_turn(self, request: ModelRequest) -> Any:
            del request
            self.calls += 1
            if self.calls == 1:
                report_provider_retried()
                raise ModelAdapterError("transient", retryable=True)
            return ModelTurn(final_text="answer", usage={"output_tokens": "seven"})  # type: ignore[dict-item]

    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())
    observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=ReportsThenRefuses(),
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
            ),
        ).acall(request)

    with pytest.raises(Exception):
        asyncio.run(run())

    receipt = observer.captures[0].receipt
    assert receipt.attempts == 2
    first, second = receipt.attempt_log
    assert first.provider_retried is True
    assert second.provider_retried is False
    # The receipt keeps the fold: one call did see a provider retry. Only the per-attempt row
    # narrows, which is the whole reason the log exists beside the count.
    assert receipt.provider_retried is True


def test_every_attempt_entry_scopes_its_retry_flag_to_its_own_dispatch() -> None:
    """Derived census over all three construction sites, not a pin on the arm that was wrong.

    ``ModelCallAttempt`` is built in three places -- the absorbed-attempt probe, the
    normalizer-refusal fallback, and the answering attempt -- and ``provider_retried`` is the
    field with a wrong source lying right next to the right one: the receipt in scope carries the
    whole CALL's fold, so reading it is a one-word slip that produces a plausible, wrong log.
    Two sites had the rule and the third did not, which is precisely the asymmetry this
    repository keeps re-earning, so the instrument enumerates the sites instead of asserting
    against the one that got fixed.

    The rule is spelled as provenance rather than value: every entry's ``provider_retried`` is
    computed from ``reports_before`` -- the channel count snapshotted when THIS dispatch began --
    and none of them reaches into ``failed``, the settled receipt whose flag the outer handler
    has already folded. A fourth construction site, or a fold creeping back into any of the
    three, reddens here before any behaviour test would notice.
    """

    import ast
    from pathlib import Path as _Path

    source = (
        _Path(__file__).resolve().parents[1] / "src" / "monoid_agent_kernel" / "model_call.py"
    ).read_text(encoding="utf-8")
    sites = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelCallAttempt"
    ]
    assert len(sites) == 3, {"construction_sites": [node.lineno for node in sites]}

    expressions: dict[int, str] = {}
    for node in sites:
        flags = [keyword for keyword in node.keywords if keyword.arg == "provider_retried"]
        assert len(flags) == 1, {
            "line": node.lineno,
            "hint": "an entry that omits the flag defaults it, which is a claim not a reading",
        }
        expressions[node.lineno] = ast.unparse(flags[0].value)

    unscoped = {
        line: expression
        for line, expression in expressions.items()
        if "reports_before" not in expression or "failed." in expression
    }
    assert unscoped == {}, {
        "sites_not_scoped_to_their_own_dispatch": unscoped,
        "hint": "count this dispatch's own reports: progress.count > reports_before",
    }


class _FlakyStream:
    """A stream that dies once -- before its first chunk, or after delivering one."""

    def __init__(self, *, fail_before_first: bool) -> None:
        self.opens = 0
        self.fail_before_first = fail_before_first

    async def astream_turn(self, request: ModelRequest):  # noqa: ANN201
        del request
        self.opens += 1
        if self.opens == 1:
            if self.fail_before_first:
                raise ModelAdapterError("dead before the first frame", retryable=True)
            yield TextDelta("partial")
            raise ModelAdapterError("dead mid-stream", retryable=True)
        yield TextDelta("whole")
        yield TurnComplete(response_id="r2", usage={"output_tokens": 2}, stop_reason="stop")


def test_a_delivered_chunk_closes_the_retry_window() -> None:
    """Once the consumer holds a chunk, a retry would replay it downstream; the loop refuses.

    The counter-arm retries a stream that died before delivering anything, and the consumer
    sees only the second attempt's chunks -- the same commit line every lower loop already
    draws (the gateway's `committed`, the SDK's pre-stream retry window).
    """

    delivered: list[Any] = []
    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())

    mid_stream = _FlakyStream(fail_before_first=False)
    with pytest.raises(ModelAdapterError):
        _acall(mid_stream, request, delta_consumer=delivered.append)
    assert mid_stream.opens == 1

    delivered.clear()
    pre_first = _FlakyStream(fail_before_first=True)
    observer = _acall(pre_first, request, delta_consumer=delivered.append)
    assert pre_first.opens == 2
    assert observer.captures[0].receipt.attempts == 2
    texts = [chunk.text for chunk in delivered if isinstance(chunk, TextDelta)]
    assert texts == ["whole"]


def test_delivery_is_marked_before_the_consumer_runs() -> None:
    """The ordering proof, promoted from the W7-0 reassessment probe.

    The consumer receives the chunk and then raises an error dressed retryable. If `delivered`
    were marked AFTER the inner consumer ran, the predicate would see an undelivered retryable
    failure and reopen the stream -- replaying the side effect the consumer already performed.
    Correct order: one open, one side effect, the failure propagates.
    """

    class OneChunkStream:
        def __init__(self) -> None:
            self.opens = 0

        async def astream_turn(self, request: ModelRequest):  # noqa: ANN201
            del request
            self.opens += 1
            yield TextDelta("side-effectful chunk")
            yield TurnComplete(response_id="r", stop_reason="stop")

    stream = OneChunkStream()
    side_effects: list[Any] = []

    def poisoned_consumer(chunk: Any) -> None:
        side_effects.append(chunk)
        raise ModelAdapterError("consumer failure dressed as retryable", retryable=True)

    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())
    with pytest.raises(ModelAdapterError):
        _acall(stream, request, delta_consumer=poisoned_consumer)

    assert stream.opens == 1
    assert len(side_effects) == 1


def test_a_consumer_exception_is_not_retried_and_keeps_its_type() -> None:
    """A consumer bug must not masquerade as a provider failure.

    The RuntimeError propagates untouched -- retrying it would replay the delivered chunk, and
    reclassifying it would blame a provider for the caller's own consumer.
    """

    class OneChunkStream:
        def __init__(self) -> None:
            self.opens = 0

        async def astream_turn(self, request: ModelRequest):  # noqa: ANN201
            del request
            self.opens += 1
            yield TextDelta("chunk")
            yield TurnComplete(response_id="r", stop_reason="stop")

    stream = OneChunkStream()

    def raising_consumer(chunk: Any) -> None:
        del chunk
        raise RuntimeError("consumer bug")

    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())
    with pytest.raises(RuntimeError, match="consumer bug"):
        _acall(stream, request, delta_consumer=raising_consumer)

    assert stream.opens == 1


def test_the_kernel_loop_retries_the_anext_and_awaitable_shapes_too() -> None:
    """The two dispatch shapes W7-0's tests did not drive, promoted from the reassessment.

    All four shapes share the one dispatch point the kernel loop wraps, but a rule proven on
    two of four parallel halves is this codebase's house defect -- so the other two are driven,
    not assumed.
    """

    class AnextFlaky:
        def __init__(self) -> None:
            self.calls = 0

        async def anext_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            self.calls += 1
            if self.calls == 1:
                raise ModelAdapterError("transient", retryable=True)
            return ModelTurn(final_text="ok")

    class AwaitableFlaky:
        def __init__(self) -> None:
            self.calls = 0

        def next_turn(self, request: ModelRequest) -> Any:
            del request
            self.calls += 1
            me = self

            async def answer() -> ModelTurn:
                if me.calls == 1:
                    raise ModelAdapterError("transient", retryable=True)
                return ModelTurn(final_text="ok")

            return answer()

    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())

    coroutine_shape = AnextFlaky()
    observer = _acall(coroutine_shape, request)
    assert coroutine_shape.calls == 2
    assert observer.captures[0].receipt.attempts == 2

    awaitable_shape = AwaitableFlaky()
    observer = _acall(awaitable_shape, request)
    assert awaitable_shape.calls == 2
    assert observer.captures[0].receipt.attempts == 2


def test_kernel_retry_rides_the_replay_fallthrough_without_spinning_the_corpus() -> None:
    """The composite the reassessment drove by hand: miss -> flaky inner -> retry -> miss ->
    inner answers. The miss itself is never retried (`replay_miss` pins retryable=False), so
    the kernel's second attempt walks the same miss-then-inner path instead of spinning on
    the corpus, and the answer is the inner's."""

    from monoid_agent_kernel.providers.replay import ReplayModelAdapter

    class FlakyInner:
        provider_name = "flaky"

        def __init__(self) -> None:
            self.calls = 0

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            self.calls += 1
            if self.calls == 1:
                raise ModelAdapterError("transient under fallthrough", retryable=True)
            return ModelTurn(final_text="served live", usage={"total_tokens": 7})

    inner = FlakyInner()
    adapter = ReplayModelAdapter([], inner=inner)
    request = ModelRequest(
        instruction="never recorded", system_prompt="sys", tools=(), model=_kernel_model()
    )
    observer = _acall(adapter, request)

    receipt = observer.captures[0].receipt
    assert inner.calls == 2
    assert receipt.attempts == 2
    assert receipt.succeeded is True
    assert receipt.usage["total_tokens"] == 7


def test_stream_committed_marks_only_the_attempt_that_delivered() -> None:
    """The commit flag is per-entry and can only be true on the final one.

    A stream that died before its first chunk logs an uncommitted failure; the attempt that
    answered logs committed. And on the mid-stream death -- delivery already made -- the one
    terminal entry records that the window was closed when it settled.
    """

    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())
    sink: list[Any] = []

    pre_first = _FlakyStream(fail_before_first=True)
    observer = _acall(pre_first, request, delta_consumer=sink.append)
    first, second = observer.captures[0].receipt.attempt_log
    assert first.stream_committed is False
    assert second.stream_committed is True

    sink.clear()
    mid_stream = _FlakyStream(fail_before_first=False)
    mid_observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=mid_stream,
            subscriptions=(
                ModelIOSubscription(observer=mid_observer, policy=CapturePolicy(mode="digest")),
            ),
        ).acall(request, delta_consumer=sink.append)

    with pytest.raises(ModelAdapterError):
        asyncio.run(run())

    (terminal,) = mid_observer.captures[0].receipt.attempt_log
    assert terminal.succeeded is False
    assert terminal.stream_committed is True


class _StreamingAdapter:
    """Delivers a chunk and settles -- no failure, so the call takes exactly one dispatch.

    Answers the non-streaming door too, so the same adapter can be called with and without a
    consumer: the point of the second call is a settled turn that delivered nothing.
    """

    async def astream_turn(self, request: ModelRequest):  # noqa: ANN201
        del request
        yield TextDelta("whole")
        yield TurnComplete(response_id="r1", usage={"output_tokens": 2}, stop_reason="stop")

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        del request
        return ModelTurn(response_id="r1", final_text="whole", usage={"output_tokens": 2})


@pytest.mark.parametrize(
    "model",
    [ModelConfig(), _kernel_model()],
    ids=["adapter_layer_default", "kernel_layer"],
)
def test_stream_committed_reports_delivery_under_either_retry_layer(model: ModelConfig) -> None:
    """The field says "was a chunk delivered", and it was answered only where a window existed.

    ``delivered`` was tracked by wrapping the consumer, and the wrapper was installed only when
    the kernel owns the retry loop -- because that is where the flag is *used*, to refuse a retry
    that would replay a chunk the consumer already holds. But the flag is also *recorded*, on
    every call, and ``layer`` defaults to ``"adapter"``: every shipped streaming call wrote
    ``stream_committed: false`` onto its ledger line while the consumer was holding its chunks.
    A definite ``false`` is not "the question does not apply here" -- the key is present and the
    reader has no way to tell those apart.

    The sibling arm was the one the earlier per-entry test drove, and it was right the whole
    time; the parametrization is the point, not the second row.
    """

    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=model)
    sink: list[Any] = []

    observer = _acall(_StreamingAdapter(), request, delta_consumer=sink.append)

    (entry,) = observer.captures[0].receipt.attempt_log
    assert sink, "the adapter delivered nothing, so this proves nothing"
    assert entry.stream_committed is True

    # And a call with no consumer at all still says False: nothing was delivered *to anyone*.
    silent = _acall(_StreamingAdapter(), request)
    assert silent.captures[0].receipt.attempt_log[0].stream_committed is False


def test_the_backoff_respects_the_deadline_instead_of_sleeping_into_it(
    monkeypatch: Any,
) -> None:
    """Sleeping into a certain timeout wastes wall clock and masks the provider's own error.

    When the remaining deadline cannot fit the scheduled backoff, the loop re-raises the
    transient failure itself -- the actual problem -- rather than waiting to convert it into
    a `RunTimeout` that names nothing.
    """

    monkeypatch.setattr("monoid_agent_kernel.model_call.retry_delay_s", lambda *_a: 10.0)
    adapter = _FlakyAdapter(failures=99)
    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())
    observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=adapter,
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
            ),
        ).acall(request, deadline=time.time() + 5.0)

    with pytest.raises(ModelAdapterError) as caught:
        asyncio.run(run())

    assert caught.value.retryable is True
    assert adapter.calls == 1
    assert observer.captures[0].receipt.attempts == 1


def test_an_extreme_schedule_does_not_replace_the_provider_error_with_an_arithmetic_one() -> None:
    """The kernel loop's backoff may not lose the failure taxonomy to its own arithmetic.

    Every field here passes `ModelRetryConfig.from_json`: `backoff_multiplier` is checked for
    finiteness and positivity, never for an upper bound, and `max_attempts` only for being an
    integer above zero. The fourth attempt's schedule therefore raises the exponent to two, and
    a cap applied only to the product would let `1e308 ** 2` overflow INSIDE the handler for the
    retryable `ModelAdapterError` -- so the caller would see an `OverflowError` carrying no
    `retryable`, no `code`, and no `http_status`, and the receipt would stop at three attempts.
    """

    adapter = _FlakyAdapter(failures=99)
    request = ModelRequest(
        instruction="hi",
        system_prompt="sys",
        tools=(),
        model=ModelConfig(
            retry=ModelRetryConfig(
                layer="kernel",
                max_attempts=4,
                initial_delay_s=0.001,
                max_delay_s=0.001,
                backoff_multiplier=1e308,
                jitter_s=0.0,
            )
        ),
    )
    observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=adapter,
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
            ),
        ).acall(request)

    with pytest.raises(ModelAdapterError) as caught:
        asyncio.run(run())

    assert caught.value.retryable is True
    assert adapter.calls == 4
    assert observer.captures[0].receipt.attempts == 4


def test_a_cancellation_interrupts_the_backoff_wait(monkeypatch: Any) -> None:
    """The backoff runs under the same race as the attempts, so a cancel wakes it."""

    monkeypatch.setattr("monoid_agent_kernel.model_call.retry_delay_s", lambda *_a: 30.0)
    adapter = _FlakyAdapter(failures=99)
    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())
    token = CancellationToken()
    timer = threading.Timer(0.2, token.cancel)
    started = time.monotonic()

    async def run() -> None:
        timer.start()
        await ModelCallRunner(adapter=adapter, current_cancellation_token=lambda: token).acall(
            request, deadline=time.time() + 300.0
        )

    try:
        with pytest.raises(RunCancelled):
            asyncio.run(run())
    finally:
        timer.cancel()

    assert time.monotonic() - started < 10.0
    assert adapter.calls == 1


def test_the_kernel_hands_the_adapter_a_neutralized_policy() -> None:
    """The dispatch copy carries `max_attempts=1` with the layer preserved.

    `max_attempts=1` silences any config-honoring loop -- even a third-party adapter that
    never learned `layer` exists -- while the preserved layer value lets an adapter whose
    loop lives outside the config (the OpenAI SDK) comply on its own. The receipt keeps the
    caller's policy: it describes the call as configured, not the neutralized dispatch copy.
    """

    seen: list[ModelConfig] = []

    class RecordingAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            assert request.model is not None
            seen.append(request.model)
            return ModelTurn(final_text="answer")

    request = ModelRequest(instruction="hi", system_prompt="sys", tools=(), model=_kernel_model())
    observer = _acall(RecordingAdapter(), request)

    assert seen[0].retry.max_attempts == 1
    assert seen[0].retry.layer == "kernel"
    receipt = observer.captures[0].receipt
    assert receipt.model.retry.max_attempts == 3
    assert receipt.model.retry.layer == "kernel"


class _TenantRetryConfig(ModelRetryConfig):
    """An extension retry policy whose convenience constructor is narrower than its fields.

    The kernel supports these deliberately: `providers/base._copy_with_fields` (and
    `model_call`'s own copy of it) exist so an ingress boundary rewrites a config by copying
    fields instead of calling `dataclasses.replace`, which would dispatch back through this
    narrower `__init__` with every inherited field.
    """

    def __init__(self, layer: str = "kernel") -> None:
        super().__init__(layer=layer, initial_delay_s=0.0, jitter_s=0.0)


class _TenantModelConfig(ModelConfig):
    """The `ModelConfig` twin of the probe above."""

    def __init__(self, retry: ModelRetryConfig) -> None:
        super().__init__(retry=retry)


def test_the_neutralized_policy_does_not_re_run_an_extension_constructor() -> None:
    """The kernel layer's dispatch copy obeys the rule every other config rewrite obeys.

    `normalize_model_config` copies fields precisely so a public subclass with a smaller
    constructor survives ingress; the layer's neutralization is another rewrite of the same
    object and must copy too. `dataclasses.replace` would re-run both constructors with every
    inherited field and raise `TypeError` before the adapter is ever reached -- a config the
    kernel accepts on every other path refused by the one layer that rewrites it, and only
    under `layer="kernel"`.
    """

    seen: list[ModelConfig] = []

    class RecordingAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            assert request.model is not None
            seen.append(request.model)
            return ModelTurn(final_text="answer")

    request = ModelRequest(
        instruction="hi",
        system_prompt="sys",
        tools=(),
        model=_TenantModelConfig(retry=_TenantRetryConfig()),
    )
    observer = _acall(RecordingAdapter(), request)

    dispatched = seen[0]
    assert isinstance(dispatched, _TenantModelConfig)
    assert isinstance(dispatched.retry, _TenantRetryConfig)
    assert dispatched.retry.max_attempts == 1
    assert dispatched.retry.layer == "kernel"
    # The receipt still describes the call as configured, extension type included.
    receipt_retry = observer.captures[0].receipt.model.retry
    assert isinstance(receipt_retry, _TenantRetryConfig)
    assert receipt_retry.max_attempts == 3


def test_the_retry_layer_leaves_the_replay_key_where_it_was() -> None:
    """Neither the layer nor the neutralized dispatch copy may reach the request identity."""

    kernel_request = ModelRequest(
        instruction="hi", system_prompt="sys", tools=(), model=_kernel_model()
    )
    plain_request = ModelRequest(
        instruction="hi", system_prompt="sys", tools=(), model=ModelConfig()
    )

    kernel_receipt = _acall(_FlakyAdapter(failures=1), kernel_request).captures[0].receipt
    plain_receipt = _acall(SyncAdapter(), plain_request).captures[0].receipt

    assert kernel_receipt.request_digest == plain_receipt.request_digest
    assert kernel_receipt.digest_status == "ok"


def test_the_two_layers_do_not_multiply(monkeypatch: Any) -> None:
    """The end-to-end dedup pin: kernel loop x gateway adapter = kernel attempts, not the product.

    Three configured attempts under `layer="kernel"` must reach the wire exactly three times
    -- the gateway's own loop, which would have made three per dispatch, answers one under
    the kernel's layer. `provider_retried` stays False because the adapter's loop never ran;
    the three attempts are the kernel's, on `attempts`.
    """

    calls: list[int] = []

    def _refused(*_args: Any, **_kwargs: Any) -> Any:
        calls.append(1)
        raise URLError("unreachable")

    monkeypatch.setattr("monoid_agent_kernel.providers.gateway.urlopen", _refused)
    monkeypatch.setattr("monoid_agent_kernel.providers.gateway._retry_delay", lambda *_a: 0.0)
    monkeypatch.setattr("monoid_agent_kernel.model_call.retry_delay_s", lambda *_a: 0.0)

    adapter = GatewayModelAdapter(
        ModelConfig(
            gateway_url="http://gateway.local/internal/llm/turns",
            retry=ModelRetryConfig(layer="kernel", max_attempts=3),
        ),
        token="run-token",
    )
    observer = RecordingObserver()

    async def run() -> None:
        await ModelCallRunner(
            adapter=adapter,
            subscriptions=(
                ModelIOSubscription(observer=observer, policy=CapturePolicy(mode="digest")),
            ),
        ).acall(ModelRequest(instruction="hi", system_prompt="sys", tools=()))

    with pytest.raises(ModelAdapterError):
        asyncio.run(run())

    assert len(calls) == 3
    receipt = observer.captures[0].receipt
    assert receipt.attempts == 3
    assert receipt.provider_retried is False


# --- durable dispatch lifecycle ----------------------------------------------------------------


class _HardLifecycleCrash(BaseException):
    """A process-stop failpoint the runner must not turn into a compensation write."""


class _JournalLifecycle:
    """The PR3 standalone hook backed by the PR2 deterministic fenced journal."""

    def __init__(
        self,
        harness: DeterministicFencedRunHarness,
        *,
        run_id: str = "run-durable-call",
        failpoint: str = "",
    ) -> None:
        self.harness = harness
        self.run_id = run_id
        self.token = harness._writers.get(run_id) or harness.claim_writer(run_id, "worker-1")
        self.failpoint = failpoint
        self.states: list[str] = []

    def _loaded(self, logical_call_id: str):  # noqa: ANN202 - test fixture seam
        return self.harness.sink.load_invocation(self.run_id, logical_call_id)

    def _head(self, logical_call_id: str) -> DurableModelInvocation | None:
        loaded = self._loaded(logical_call_id)
        return loaded.value.invocation if loaded.value is not None else None

    def _commit(
        self,
        invocation: DurableModelInvocation,
        blobs: dict[str, bytes] | None = None,
    ) -> None:
        result = self.harness.sink.commit_invocation(
            invocation,
            blobs or {},
            writer_token=self.token,
        )
        if result.status not in {"committed", "already_committed"}:
            raise RuntimeError(f"invocation commit failed: {result.status}")
        if result.status == "committed":
            self.states.append(invocation.dispatch_state)

    def reserve(self, proposed: ModelDispatchReservation) -> ModelDispatchReservation:
        if self.failpoint == "before_reserve":
            raise _HardLifecycleCrash()
        head = self._head(proposed.logical_call_id)
        if head is not None and head.dispatch_state == "reserved":
            effective = replace(proposed, idempotency_key=head.idempotency_key)
            if (
                head.dispatch_attempt != effective.dispatch_attempt
                or head.dispatch_id != effective.dispatch_id
                or head.request_digest != effective.request_digest
            ):
                raise DurableModelCallError(
                    "restored reservation conflicts with the current request",
                    error_code="durable_invocation_request_conflict",
                )
            return effective
        if head is not None and head.dispatch_state == "dispatch_started":
            self._commit(
                replace(
                    head,
                    revision=head.revision + 1,
                    dispatch_state="unknown",
                    failure_code="dispatch_unknown",
                )
            )
            raise DurableModelCallError(
                "restored started dispatch is unknown",
                error_code="dispatch_unknown",
            )
        if head is not None and head.dispatch_state == "unknown":
            raise DurableModelCallError(
                "restored dispatch remains unknown",
                error_code="dispatch_unknown",
            )
        if head is not None and not (
            head.dispatch_state == "settled"
            and head.failure_code
            and head.receipt is not None
            and head.receipt.get("retryable") is True
        ):
            raise DurableModelCallError(
                "settled model dispatch cannot be reserved again",
                error_code="durable_invocation_already_settled",
            )
        revision = 1 if head is None else head.revision + 1
        idempotency_key = proposed.idempotency_key if head is None else head.idempotency_key
        effective = replace(proposed, idempotency_key=idempotency_key)
        self._commit(
            DurableModelInvocation(
                run_id=self.run_id,
                logical_call_id=effective.logical_call_id,
                revision=revision,
                dispatch_id=effective.dispatch_id,
                dispatch_attempt=effective.dispatch_attempt,
                idempotency_key=effective.idempotency_key,
                dispatch_state="reserved",
                request_digest=effective.request_digest,
                digest_generation=effective.digest_generation,
            )
        )
        if self.failpoint == "after_reserve":
            raise _HardLifecycleCrash()
        return effective

    def dispatch_started(self, reservation: ModelDispatchReservation) -> None:
        head = self._head(reservation.logical_call_id)
        assert head is not None
        self._commit(
            replace(
                head,
                revision=head.revision + 1,
                dispatch_state="dispatch_started",
            )
        )
        if self.failpoint == "after_start":
            raise _HardLifecycleCrash()

    def settled(self, settlement: ModelDispatchSettlement) -> None:
        if self.failpoint == "before_settle":
            raise _HardLifecycleCrash()
        if self.failpoint == "settle_error":
            raise RuntimeError("injected settlement failure")
        head = self._head(settlement.reservation.logical_call_id)
        assert head is not None
        blobs: dict[str, bytes] = {}
        result_ref = ""
        if settlement.result_blob is not None:
            sha256 = hashlib.sha256(settlement.result_blob).hexdigest()
            blobs[sha256] = settlement.result_blob
            result_ref = f"blob:{sha256}"
        self._commit(
            replace(
                head,
                revision=head.revision + 1,
                dispatch_state="settled",
                receipt=settlement.receipt,
                result_ref=result_ref,
                failure_code=settlement.failure_code,
            ),
            blobs,
        )

    def unknown(self, unknown: UnknownModelDispatch) -> None:
        if self.failpoint == "unknown_error":
            raise RuntimeError("injected unknown transition failure")
        head = self._head(unknown.reservation.logical_call_id)
        assert head is not None
        self._commit(
            replace(
                head,
                revision=head.revision + 1,
                dispatch_state="unknown",
                failure_code=unknown.failure_code,
            )
        )


class _CountingDurableAdapter:
    def __init__(self) -> None:
        self.calls = 0
        self.keys: list[str] = []

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        self.calls += 1
        self.keys.append(request.idempotency_key)
        return ModelTurn(
            response_id="response-1",
            final_text="durable answer",
            usage={"input_tokens": 3, "output_tokens": 4},
            raw={"private": "provider body"},
            reasoning=({"type": "encrypted_reasoning", "id": "reasoning-1"},),
            stop_reason="stop",
        )


def _durable_call(
    adapter: Any,
    lifecycle: _JournalLifecycle,
    *,
    request: ModelRequest = REQUEST,
) -> tuple[ModelTurn, Any]:
    return asyncio.run(
        ModelCallRunner(adapter=adapter, lifecycle_hook=lifecycle).acall(
            request,
            logical_call_id="call-durable-1",
        )
    )


def test_durable_runner_requires_an_explicit_logical_call_id_before_dispatch() -> None:
    harness = DeterministicFencedRunHarness()
    lifecycle = _JournalLifecycle(harness)
    adapter = _CountingDurableAdapter()

    with pytest.raises(DurableModelCallError) as caught:
        asyncio.run(ModelCallRunner(adapter=adapter, lifecycle_hook=lifecycle).acall(REQUEST))

    assert caught.value.error_code == "durable_invocation_identity_required"
    assert adapter.calls == 0
    assert lifecycle._loaded("call-durable-1").status == "missing"


@pytest.mark.parametrize(
    ("lose_after", "expected_states"),
    (
        ("reserve", ["reserved"]),
        ("dispatch_started", ["reserved", "dispatch_started"]),
    ),
)
def test_lease_loss_after_each_pre_dispatch_commit_blocks_provider_entry(
    lose_after: str,
    expected_states: list[str],
) -> None:
    harness = DeterministicFencedRunHarness()
    token = CancellationToken()

    class LeaseLosingLifecycle(_JournalLifecycle):
        def _lose_authority(self, transition: str) -> None:
            if transition == lose_after:
                token.cancel(InterruptionCause.USER_CANCEL)
                token.cancel(InterruptionCause.LEASE_LOST)

        def reserve(self, proposed: ModelDispatchReservation) -> ModelDispatchReservation:
            effective = super().reserve(proposed)
            self._lose_authority("reserve")
            return effective

        def dispatch_started(self, reservation: ModelDispatchReservation) -> None:
            super().dispatch_started(reservation)
            self._lose_authority("dispatch_started")

    lifecycle = LeaseLosingLifecycle(harness)
    adapter = _CountingDurableAdapter()
    observer = RecordingObserver()
    sidecar: list[SettledModelCall] = []

    with pytest.raises(RunCancelled) as caught:
        asyncio.run(
            ModelCallRunner(
                adapter=adapter,
                lifecycle_hook=lifecycle,
                current_cancellation_token=lambda: token,
                subscriptions=(
                    ModelIOSubscription(
                        observer=observer,
                        policy=CapturePolicy(mode="digest"),
                    ),
                ),
                settled_sink=sidecar.append,
            ).acall(REQUEST, logical_call_id="call-durable-1")
        )

    assert caught.value.interruption_cause is InterruptionCause.LEASE_LOST
    assert token.cause is InterruptionCause.USER_CANCEL
    assert lifecycle.states == expected_states
    assert adapter.calls == 0
    assert observer.captures == []
    assert sidecar == []


def test_durable_resume_abort_gate_runs_after_recovery_probe_before_dispatch() -> None:
    harness = DeterministicFencedRunHarness()
    lifecycle = _JournalLifecycle(harness)
    adapter = _CountingDurableAdapter()
    polls: list[bool] = []

    async def run() -> None:
        await ModelCallRunner(adapter=adapter, lifecycle_hook=lifecycle).acall(
            REQUEST,
            logical_call_id="call-durable-1",
            should_abort=lambda: polls.append(True) or True,
            abort_after_recovery_probe=True,
        )

    with pytest.raises(ModelCallAborted, match="before provider dispatch"):
        asyncio.run(run())

    assert polls == [True]
    assert adapter.calls == 0
    assert lifecycle._loaded("call-durable-1").status == "missing"


def test_terminal_boundary_precedes_durable_resume_abort_gate() -> None:
    harness = DeterministicFencedRunHarness()
    lifecycle = _JournalLifecycle(harness)
    adapter = _CountingDurableAdapter()
    token = CancellationToken()
    token.cancel()
    polls: list[bool] = []

    async def run() -> None:
        await ModelCallRunner(
            adapter=adapter,
            lifecycle_hook=lifecycle,
            current_cancellation_token=lambda: token,
        ).acall(
            REQUEST,
            logical_call_id="call-durable-1",
            should_abort=lambda: polls.append(True) or True,
            abort_after_recovery_probe=True,
        )

    with pytest.raises(RunCancelled):
        asyncio.run(run())

    assert polls == []
    assert adapter.calls == 0
    assert lifecycle._loaded("call-durable-1").status == "missing"


def test_durable_runner_refuses_an_unkeyable_request_before_reservation(
    monkeypatch: Any,
) -> None:
    harness = DeterministicFencedRunHarness()
    lifecycle = _JournalLifecycle(harness)
    adapter = _CountingDurableAdapter()
    monkeypatch.setattr(
        "monoid_agent_kernel.model_call._encoded_digest",
        lambda **_kwargs: _DigestResult(status="too_large"),
    )

    with pytest.raises(DurableModelCallError) as caught:
        _durable_call(adapter, lifecycle)

    assert caught.value.error_code == "durable_invocation_unkeyable"
    assert adapter.calls == 0
    assert lifecycle._loaded("call-durable-1").status == "missing"


def test_durable_runner_rejects_reservation_identity_drift_before_dispatch() -> None:
    class DriftingLifecycle(_JournalLifecycle):
        def reserve(self, proposed: ModelDispatchReservation) -> ModelDispatchReservation:
            effective = super().reserve(proposed)
            return replace(effective, dispatch_id="different-dispatch")

    harness = DeterministicFencedRunHarness()
    lifecycle = DriftingLifecycle(harness)
    adapter = _CountingDurableAdapter()

    with pytest.raises(DurableModelCallError) as caught:
        _durable_call(adapter, lifecycle)

    assert caught.value.error_code == "durable_invocation_reservation_conflict"
    assert adapter.calls == 0
    head = lifecycle._head("call-durable-1")
    assert head is not None and head.dispatch_state == "reserved"


def test_only_the_typed_refusal_marks_definite_dispatch_evidence() -> None:
    class LookalikeError(ModelAdapterError):
        dispatch_evidence = "refused"

    assert dispatch_evidence(ModelAdapterError("ambiguous")) == "unknown"
    assert dispatch_evidence(LookalikeError("lookalike")) == "unknown"
    assert dispatch_evidence(ModelDispatchRefused("refused")) == "refused"


def test_durable_success_commits_the_canonical_private_result_before_passive_delivery() -> None:
    harness = DeterministicFencedRunHarness()
    lifecycle = _JournalLifecycle(harness)
    adapter = _CountingDurableAdapter()
    delivered_states: list[str] = []

    def passive_sink(_call: SettledModelCall) -> None:
        head = lifecycle._head("call-durable-1")
        delivered_states.append(head.dispatch_state if head is not None else "missing")

    turn, receipt = asyncio.run(
        ModelCallRunner(
            adapter=adapter,
            lifecycle_hook=lifecycle,
            settled_sink=passive_sink,
        ).acall(REQUEST, logical_call_id="call-durable-1")
    )

    loaded = lifecycle._loaded("call-durable-1")
    assert loaded.value is not None
    invocation = loaded.value.invocation
    assert lifecycle.states == ["reserved", "dispatch_started", "settled"]
    assert invocation.dispatch_state == "settled"
    assert invocation.failure_code == ""
    assert invocation.receipt is not None
    assert invocation.receipt["request_digest"] == receipt.request_digest
    assert delivered_states == ["settled"]
    sha256 = invocation.result_ref.removeprefix("blob:")
    body = json.loads(loaded.value.blob(sha256))
    assert body["final_text"] == turn.final_text == "durable answer"
    assert body["reasoning"] == [{"type": "encrypted_reasoning", "id": "reasoning-1"}]
    assert "raw" not in body
    assert adapter.keys == [invocation.idempotency_key]


def test_lease_loss_at_durable_settlement_blocks_receipt_observers_and_sidecars() -> None:
    harness = DeterministicFencedRunHarness()
    token = CancellationToken()

    class LeaseLosingLifecycle(_JournalLifecycle):
        def settled(self, settlement: ModelDispatchSettlement) -> None:
            super().settled(settlement)
            token.cancel(InterruptionCause.GRACEFUL_DRAIN)
            token.cancel(InterruptionCause.LEASE_LOST)

    lifecycle = LeaseLosingLifecycle(harness)
    adapter = _CountingDurableAdapter()
    observer = RecordingObserver()
    sidecar: list[SettledModelCall] = []

    with pytest.raises(RunCancelled) as caught:
        asyncio.run(
            ModelCallRunner(
                adapter=adapter,
                lifecycle_hook=lifecycle,
                current_cancellation_token=lambda: token,
                subscriptions=(
                    ModelIOSubscription(
                        observer=observer,
                        policy=CapturePolicy(mode="digest"),
                    ),
                ),
                settled_sink=sidecar.append,
            ).acall(REQUEST, logical_call_id="call-durable-1")
        )

    assert caught.value.interruption_cause is InterruptionCause.LEASE_LOST
    assert token.cause is InterruptionCause.GRACEFUL_DRAIN
    assert token.lease_lost is True
    assert lifecycle.states == ["reserved", "dispatch_started", "settled"]
    head = lifecycle._head("call-durable-1")
    assert head is not None and head.dispatch_state == "settled"
    assert observer.captures == []
    assert sidecar == []


def test_lease_loss_during_receipt_subscriber_stops_fanout_and_sidecar() -> None:
    token = CancellationToken()
    entered, release = threading.Event(), threading.Event()
    first_captures: list[ModelCallCapture] = []

    class BlockingObserver:
        def on_model_call(self, capture: ModelCallCapture) -> None:
            first_captures.append(capture)
            entered.set()
            assert release.wait(5)

    second = RecordingObserver()
    sidecar: list[SettledModelCall] = []

    def lose_authority() -> None:
        assert entered.wait(5)
        token.cancel(InterruptionCause.GRACEFUL_DRAIN)
        token.cancel(InterruptionCause.LEASE_LOST)
        release.set()

    racer = threading.Thread(target=lose_authority)
    racer.start()
    with pytest.raises(RunCancelled) as caught:
        asyncio.run(
            ModelCallRunner(
                adapter=SyncAdapter(),
                current_cancellation_token=lambda: token,
                subscriptions=(
                    ModelIOSubscription(
                        observer=BlockingObserver(),
                        policy=CapturePolicy(mode="digest"),
                    ),
                    ModelIOSubscription(
                        observer=second,
                        policy=CapturePolicy(mode="digest"),
                    ),
                ),
                settled_sink=sidecar.append,
            ).acall(REQUEST)
        )
    racer.join(5)

    assert not racer.is_alive()
    assert caught.value.interruption_cause is InterruptionCause.LEASE_LOST
    assert len(first_captures) == 1
    assert second.captures == []
    assert sidecar == []


def test_sticky_lease_loss_supersedes_an_older_cancel_before_dispatch_compensation() -> None:
    harness = DeterministicFencedRunHarness()
    token = CancellationToken()
    lifecycle = _JournalLifecycle(harness)

    class RacingAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            token.cancel(InterruptionCause.GRACEFUL_DRAIN)
            stale_exception = RunCancelled(
                "run cancelled",
                interruption_cause=InterruptionCause.GRACEFUL_DRAIN,
            )
            token.cancel(InterruptionCause.LEASE_LOST)
            raise stale_exception

    observer = RecordingObserver()
    sidecar: list[SettledModelCall] = []

    with pytest.raises(RunCancelled) as caught:
        asyncio.run(
            ModelCallRunner(
                adapter=RacingAdapter(),
                lifecycle_hook=lifecycle,
                current_cancellation_token=lambda: token,
                subscriptions=(
                    ModelIOSubscription(
                        observer=observer,
                        policy=CapturePolicy(mode="digest"),
                    ),
                ),
                settled_sink=sidecar.append,
            ).acall(REQUEST, logical_call_id="call-durable-1")
        )

    assert caught.value.interruption_cause is InterruptionCause.LEASE_LOST
    assert token.cause is InterruptionCause.GRACEFUL_DRAIN
    assert token.lease_lost is True
    assert lifecycle.states == ["reserved", "dispatch_started"]
    head = lifecycle._head("call-durable-1")
    assert head is not None and head.dispatch_state == "dispatch_started"
    assert observer.captures == []
    assert sidecar == []


@pytest.mark.parametrize(
    ("failpoint", "expected_state", "calls"),
    (
        ("before_reserve", "missing", 0),
        ("after_reserve", "reserved", 0),
        ("after_start", "dispatch_started", 0),
        ("before_settle", "dispatch_started", 1),
    ),
)
def test_hard_crash_failpoints_leave_the_last_committed_head_without_compensation(
    failpoint: str,
    expected_state: str,
    calls: int,
) -> None:
    harness = DeterministicFencedRunHarness()
    lifecycle = _JournalLifecycle(harness, failpoint=failpoint)
    adapter = _CountingDurableAdapter()

    with pytest.raises(_HardLifecycleCrash):
        _durable_call(adapter, lifecycle)

    head = lifecycle._head("call-durable-1")
    assert (head.dispatch_state if head is not None else "missing") == expected_state
    assert adapter.calls == calls


def test_reserved_crash_reuses_the_committed_key_and_dispatches_once_after_restore() -> None:
    harness = DeterministicFencedRunHarness()
    crashed = _JournalLifecycle(harness, failpoint="after_reserve")
    adapter = _CountingDurableAdapter()

    with pytest.raises(_HardLifecycleCrash):
        _durable_call(adapter, crashed)
    reserved = crashed._head("call-durable-1")
    assert reserved is not None

    restored = _JournalLifecycle(harness)
    _durable_call(adapter, restored)

    settled = restored._head("call-durable-1")
    assert settled is not None
    assert settled.dispatch_state == "settled"
    assert settled.idempotency_key == reserved.idempotency_key
    assert adapter.keys == [reserved.idempotency_key]
    assert adapter.calls == 1


@pytest.mark.parametrize("failpoint", ("after_start", "before_settle"))
def test_started_crash_restores_as_unknown_without_another_provider_call(
    failpoint: str,
) -> None:
    harness = DeterministicFencedRunHarness()
    crashed = _JournalLifecycle(harness, failpoint=failpoint)
    adapter = _CountingDurableAdapter()

    with pytest.raises(_HardLifecycleCrash):
        _durable_call(adapter, crashed)
    calls_at_crash = adapter.calls

    restored = _JournalLifecycle(harness)
    with pytest.raises(DurableModelCallError) as caught:
        _durable_call(adapter, restored)

    assert caught.value.error_code == "dispatch_unknown"
    assert adapter.calls == calls_at_crash
    head = restored._head("call-durable-1")
    assert head is not None and head.dispatch_state == "unknown"


def test_ambiguous_transport_failure_becomes_unknown_and_disables_kernel_retry() -> None:
    class AmbiguousAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            self.calls += 1
            raise ModelAdapterError("connection dropped", retryable=True)

    harness = DeterministicFencedRunHarness()
    lifecycle = _JournalLifecycle(harness)
    adapter = AmbiguousAdapter()
    request = replace(REQUEST, model=_kernel_model())

    with pytest.raises(DurableModelCallError) as caught:
        _durable_call(adapter, lifecycle, request=request)

    assert caught.value.error_code == "dispatch_unknown"
    assert adapter.calls == 1
    head = lifecycle._head("call-durable-1")
    assert head is not None and head.dispatch_state == "unknown"


def test_malformed_provider_terminal_becomes_unknown() -> None:
    class MalformedTerminalAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            self.calls += 1
            return ModelTurn(final_text="answer", provider_retried="yes")  # type: ignore[arg-type]

    harness = DeterministicFencedRunHarness()
    lifecycle = _JournalLifecycle(harness)
    adapter = MalformedTerminalAdapter()

    with pytest.raises(DurableModelCallError) as caught:
        _durable_call(adapter, lifecycle)

    assert caught.value.error_code == "dispatch_unknown"
    assert adapter.calls == 1
    head = lifecycle._head("call-durable-1")
    assert head is not None and head.dispatch_state == "unknown"


def test_unknown_transition_failure_still_forbids_paid_call_retry() -> None:
    class AmbiguousAdapter:
        def __init__(self) -> None:
            self.calls = 0

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            self.calls += 1
            raise ModelAdapterError("connection dropped", retryable=True)

    harness = DeterministicFencedRunHarness()
    lifecycle = _JournalLifecycle(harness, failpoint="unknown_error")
    adapter = AmbiguousAdapter()

    with pytest.raises(DurableModelCallError) as caught:
        _durable_call(adapter, lifecycle, request=replace(REQUEST, model=_kernel_model()))

    assert caught.value.error_code == "dispatch_unknown"
    assert adapter.calls == 1
    head = lifecycle._head("call-durable-1")
    assert head is not None and head.dispatch_state == "dispatch_started"


def test_explicit_retryable_refusal_settles_then_reuses_the_key_on_the_next_dispatch() -> None:
    class RefusesThenAnswers:
        def __init__(self) -> None:
            self.calls = 0
            self.keys: list[str] = []

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            self.calls += 1
            self.keys.append(request.idempotency_key)
            if self.calls == 1:
                raise ModelDispatchRefused(
                    "provider refused",
                    provider_error_code="rate_limited",
                    retryable=True,
                    http_status=429,
                )
            return ModelTurn(final_text="answer", usage={"output_tokens": 2})

    harness = DeterministicFencedRunHarness()
    lifecycle = _JournalLifecycle(harness)
    adapter = RefusesThenAnswers()

    _durable_call(adapter, lifecycle, request=replace(REQUEST, model=_kernel_model()))

    assert lifecycle.states == [
        "reserved",
        "dispatch_started",
        "settled",
        "reserved",
        "dispatch_started",
        "settled",
    ]
    assert adapter.calls == 2
    assert len(set(adapter.keys)) == 1
    head = lifecycle._head("call-durable-1")
    assert head is not None
    assert head.dispatch_state == "settled"
    assert head.dispatch_attempt == 2


def test_settlement_write_failure_closes_the_started_dispatch_as_unknown() -> None:
    harness = DeterministicFencedRunHarness()
    lifecycle = _JournalLifecycle(harness, failpoint="settle_error")
    adapter = _CountingDurableAdapter()

    with pytest.raises(DurableModelCallError) as caught:
        _durable_call(adapter, lifecycle)

    assert caught.value.error_code == "dispatch_unknown"
    assert adapter.calls == 1
    head = lifecycle._head("call-durable-1")
    assert head is not None and head.dispatch_state == "unknown"
