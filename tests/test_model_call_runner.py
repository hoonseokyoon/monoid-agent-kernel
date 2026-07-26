"""``ModelCallRunner`` -- adapter dispatch, the cancel/deadline race, and the capture receipt."""

from __future__ import annotations

import asyncio
import json
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
from monoid_agent_kernel.errors import (
    ModelAdapterError,
    ModelCallAborted,
    RunCancelled,
    RunTimeout,
)
from monoid_agent_kernel import model_call
from monoid_agent_kernel.model_call import (
    ModelCallRunner,
    _canonical_ready,
    _digest,
    _prompt_payload,
)
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelTurn,
    TextDelta,
    ToolCallDelta,
    ToolObservation,
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

    keys = {asyncio.run(_receipt_for(request)).request_digest for request in (plain, renamed, reschemad)}
    assert len(keys) == 3


@pytest.mark.parametrize(
    ("label", "request_"),
    [
        (
            "mixed mapping keys in a tool field",
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
        ),
        (
            "a value JSON has no form for, in messages",
            ModelRequest(
                instruction="hi", system_prompt="s", tools=(), messages=({"role": "user", "x": object()},)
            ),
        ),
        (
            "mixed keys nested deep in messages",
            ModelRequest(
                instruction="hi", system_prompt="s", tools=(), messages=({"deep": {"a": {2: "b"}}},)
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
    assert turn.final_text == "answer"
    assert receipt.request_digest != ""


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
    ("label", "factory", "keyed"),
    [
        ("self cycle", _self_cycle, True),
        ("cycle reached through two references", _branching_cycle, True),
        ("acyclic but exponentially shared", _shared_acyclic_graph, False),
    ],
)
def test_a_pathological_payload_terminates_quickly(
    label: str, factory: Any, keyed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three shapes, because they defeat three different bounds — and two outcomes.

    An earlier version used only the self cycle, which a depth cap alone terminates in `depth`
    steps, so it certified "cycles are safe" while a cycle reached through two references per level
    still expanded `2**depth` nodes — four billion at the real cap. One shape stood in for a claim
    about all of them.

    The third has no cycle at all: nothing repeats on any single path, so ancestor tracking cannot
    see it and only the work budget bounds it.

    `keyed` is the distinction that matters downstream. A cycle marker loses nothing — it states
    exactly what that edge is — so a cyclic payload keeps a real replay key. Truncation drops
    content, so it must not.
    """
    del label
    # A small budget makes the exponential shape terminate in thousands of nodes instead of a
    # million, so the assertion below is about behaviour rather than about how fast this machine is.
    # The earlier version asserted wall-clock < 5s and failed CI at 5.79s under coverage
    # instrumentation -- a timing assertion is a machine-speed assertion wearing a correctness
    # costume. If a bound is ever removed the traversal does not slow down, it fails to terminate,
    # and pytest-timeout is what catches that.
    monkeypatch.setattr(model_call, "_MAX_DIGEST_NODES", 5_000)

    digest = _digest({"value": factory()})

    assert (digest != "") is keyed


def test_two_different_cyclic_graphs_do_not_share_a_replay_key() -> None:
    """A bare cycle marker said *that* an edge looped without saying *where*.

    `root -> child -> root` and `root -> child -> child` both became `[["<cycle>"]]` and were handed
    the same non-empty replay key, so a consumer could return the wrong call. Encoding the depth the
    edge points back to identifies the target exactly, because a path holds one object per position.
    """
    back_to_root: list[Any] = []
    child: list[Any] = [back_to_root]
    back_to_root.append(child)

    self_looping: list[Any] = []
    self_looping.append(self_looping)
    holding: list[Any] = [self_looping]

    assert _digest({"v": back_to_root}) != _digest({"v": holding})
    # Both keep a key: a depth-tagged marker is the whole truth about the edge, so nothing is lost.
    assert _digest({"v": back_to_root}) != ""
    assert _digest({"v": holding}) != ""


def test_a_repr_that_raises_does_not_abort_the_call() -> None:
    """The normalizer exists so digest bookkeeping cannot become a precondition for the call.

    `repr` was the one remaining way for a caller's value to break that: a custom `__repr__` that
    raises propagated out while the receipt was being built, before the adapter was ever reached.
    """

    class Hostile:
        def __repr__(self) -> str:
            raise RuntimeError("repr exploded")

    request = ModelRequest(
        instruction="hi", system_prompt="s", tools=(), messages=({"x": Hostile()},)
    )

    class Adapter:
        def next_turn(self, model_request: ModelRequest) -> ModelTurn:
            del model_request
            return ModelTurn(final_text="answer")

    async def run() -> Any:
        turn, receipt = await ModelCallRunner(adapter=Adapter()).acall(request)
        return turn, receipt

    turn, receipt = asyncio.run(run())
    assert turn.final_text == "answer"
    # Nothing is known about the value, so the payload is lossy and gets no replay key.
    assert receipt.prompt_digest == ""


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
        _turn, receipt = await ModelCallRunner(adapter=_ConfiguredAdapter("fallback")).acall(request)
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


def test_a_truncated_payload_gets_no_replay_key_rather_than_a_misleading_one() -> None:
    """Bounding the work created a way to collide, which is the failure this digest must not have.

    Two requests differing only past the cut normalize to the same thing, so an ordinary-looking
    digest would stand for "everything up to here" and a replay consumer would hand back the wrong
    call. Refusing to issue a key is the safe half: the call still happens, it is just not
    replayable.
    """
    huge = ModelRequest(
        instruction="hi", system_prompt="s", tools=(), messages=({"v": list(range(1_000_100))},)
    )
    huge_but_different = ModelRequest(
        instruction="hi",
        system_prompt="s",
        tools=(),
        messages=({"v": list(range(1_000_099)) + [-999]},),
    )

    assert _digest(_prompt_payload(huge)) == ""
    assert _digest(_prompt_payload(huge_but_different)) == ""

    # Counterweight: ordinary payloads still get a key, and it still discriminates. "return empty
    # always" would pass the assertions above.
    ordinary = ModelRequest(instruction="hi", system_prompt="s", tools=())
    other = ModelRequest(instruction="bye", system_prompt="s", tools=())
    assert _digest(_prompt_payload(ordinary)) not in {"", _digest(_prompt_payload(other))}


def test_a_cycle_still_gets_a_real_digest() -> None:
    """A cycle marker loses nothing -- "this points back at an ancestor" is the whole truth about
    that edge -- so unlike truncation it must not cost the call its replay key."""
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic

    assert _digest({"value": cyclic}) != ""


def test_the_work_budget_stops_the_traversal_not_merely_the_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returning a marker per element still walked every element of a very wide payload.

    Asserted on the produced structure rather than on elapsed time: the budget's promise is that
    traversal *stops*, and a normalized list shorter than its source is direct evidence of that,
    where a stopwatch only ever measures the machine.
    """
    monkeypatch.setattr(model_call, "_MAX_DIGEST_NODES", 500)
    normalized, lost_content = _canonical_ready({"messages": [list(range(100_000))]})

    kept = normalized["messages"][0]
    assert len(kept) < 500, "traversal must stop at the budget, not run the full width"
    assert lost_content is True


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
