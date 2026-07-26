"""``ModelCallRunner`` -- adapter dispatch, the cancel/deadline race, and the capture receipt."""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from dataclasses import replace
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
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.errors import (
    ModelAdapterError,
    ModelCallAborted,
    RunCancelled,
    RunTimeout,
)
from monoid_agent_kernel.model_call import (
    ModelCallRunner,
    _digest,
    _prompt_payload,
    _request_payload,
)
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelTurn,
    TextDelta,
    ToolCallDelta,
    ToolObservation,
    TurnComplete,
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
        assert isinstance(caught.value.__cause__, CalleeCancelled), (
            f"{label}: the original cancellation must stay on the chain"
        )


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
    # The call is what must survive; whether the payload earns a key is a separate question, covered
    # by the digest tests above. Asserting both here is what made this test wrong twice.
    assert turn.final_text == "answer"
    assert isinstance(receipt.request_digest, str)


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
        pytest.param(lambda: type("HostileDict", (dict,), {"items": _raises})(a=1), id="dict-items"),
        pytest.param(lambda: type("HostileList", (list,), {"__iter__": _raises})([1]), id="list-iter"),
    ],
)
def test_a_container_hook_that_raises_costs_the_key_not_the_call(factory: Any) -> None:
    """A `dict` or `list` subclass can raise from inside the encoder, from a type it accepts.

    The guard used to name four exception types and this was a fifth. Which exception the encoder
    chose is never the question -- only whether it finished -- so the clause catches `Exception`.
    `BaseException` deliberately still escapes: a cancellation is not a statement about the payload.
    """
    assert _digest({"v": factory()}) == ""


def test_the_replay_key_separates_calls_to_different_destinations() -> None:
    """Two adapters with identical configs can address different services.

    `GatewayModelAdapter` lets a per-instance `gateway_url` outrank the config, so a config-only key
    matched calls that went to different hosts and could return different answers. The destination
    is hashed rather than recorded, so an internal hostname stays internal.
    """
    config = ModelConfig(model="m", gateway_url="http://shared.invalid/x")

    def keyed(url: str) -> str:
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
        return observer.captures[0].receipt.request_digest

    first = keyed("http://tenant-a.invalid/x")
    second = keyed("http://tenant-b.invalid/x")

    assert first != second
    assert first == keyed("http://tenant-a.invalid/x")
    assert "tenant-a" not in first


def test_an_adapter_that_names_no_destination_still_gets_a_key() -> None:
    """The member is opt-in, and a resolver that raises must not cost the call its key either.

    Refusing a key whenever the destination is unknown would refuse one for every adapter that
    routes on config alone, which is most of them.
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

    async def key(adapter: Any) -> str:
        _turn, receipt = await ModelCallRunner(adapter=adapter).acall(REQUEST)
        return receipt.request_digest

    assert asyncio.run(key(Silent())) != ""
    assert asyncio.run(key(Unroutable())) != ""


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
        _digest(_request_payload(r, ModelConfig(), provider="p", destination="d"))
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
            await ModelCallRunner(
                adapter=adapter, current_cancellation_token=lambda: token
            ).acall(REQUEST, deadline=(time.time() - 5) if expired else None, **extra)

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
            ModelIOSubscription(
                observer=RecordingObserver(), policy=CapturePolicy(mode="digest")
            ),
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


def test_a_zero_count_is_a_count_and_a_bad_key_costs_only_its_entry() -> None:
    """`_recordable_usage` drops what the receipt would refuse, and nothing else.

    A zero is a real count -- an accounting consumer must see `0`, not a missing key -- and a
    non-string key would raise from `ModelCallReceipt.__post_init__`, failing a call the provider
    has already been paid for, which is the one thing this helper exists to prevent.
    """

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

    assert dict(asyncio.run(run()).usage) == {"input_tokens": 5, "cached_tokens": 0}


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


def test_a_resolver_that_answers_nothing_still_leaves_the_key_empty() -> None:
    """The third way an adapter declines a destination: answering, with nothing.

    Unguarded, `str(None)` puts the text `"None"` into the replay key -- a destination no adapter
    has, shared by every adapter that returns `None`.
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

    empty = _digest(_request_payload(REQUEST, ModelConfig(), provider="", destination=""))
    stringified = _digest(_request_payload(REQUEST, ModelConfig(), provider="", destination="None"))
    assert asyncio.run(run()).request_digest == empty
    assert empty != stringified, "the two must be distinguishable for this test to mean anything"


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

    assert any("abandoned an asynchronous call" in record.message for record in caplog.records), (
        "an abandoned async call must be as visible as an abandoned thread"
    )


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
            await runner.acall(REQUEST, delta_consumer=lambda chunk: None, should_abort=lambda: True)
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

    assert any("outran the" in record.message for record in caplog.records), (
        "an abandoned stream close must be as visible as an abandoned call"
    )


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


def test_a_destination_probe_that_raises_on_lookup_still_keeps_the_call() -> None:
    """The probe is tolerant at the lookup, not only at the call.

    `resolve_destination` is opt-in, so an adapter may expose it as a property -- and a property that
    raised took the whole call down, over a replay key. The sibling probes guarded the `getattr`;
    this one guarded only the invocation, which is the half a `def` happens to exercise.
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
    assert receipt.request_digest == _digest(
        _request_payload(REQUEST, ModelConfig(), provider="", destination="")
    )


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
    assert reads == [0, 1], f"the adapter must be read exactly once per call, was read {len(reads)}x"


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
    assert not caplog.records, (
        f"a close that finished inside the grace was reported as abandoned: {caplog.records}"
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
