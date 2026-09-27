from __future__ import annotations

import asyncio
import json
from pathlib import Path
from threading import Event, Thread
from typing import Any

import pytest

from support.runtime import runtime_config, runtime_provider, tool_binding

from monoid_agent_kernel.core.agents import AgentRuntimeConfig, PromptSpec, SubagentDefinition
from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.model_stream import (
    ModelStreamContext,
    ModelStreamDelta,
    ModelStreamOutcome,
    ModelStreamStatus,
)
from monoid_agent_kernel.core.outcome import InterruptionCause
from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelTurn,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    TurnComplete,
    mark_provider_usage,
)
from monoid_agent_kernel.recorder import MemoryEventSink


class _RecordingWriter:
    def __init__(self) -> None:
        self.deltas: list[ModelStreamDelta] = []
        self.outcomes: list[ModelStreamOutcome] = []
        self.dispatches = 0

    def begin_dispatch(self) -> None:
        self.dispatches += 1

    def push(self, delta: ModelStreamDelta) -> None:
        self.deltas.append(delta)

    def close(self, outcome: ModelStreamOutcome) -> None:
        self.outcomes.append(outcome)


class _RecordingObserver:
    def __init__(self) -> None:
        self.contexts: list[ModelStreamContext] = []
        self.writers: list[_RecordingWriter] = []

    def open(self, context: ModelStreamContext) -> _RecordingWriter:
        writer = _RecordingWriter()
        self.contexts.append(context)
        self.writers.append(writer)
        return writer


class _ScriptedStreamAdapter:
    supports_multimodal = False
    provider_name = "test-provider"

    def __init__(self, chunks: list[object]) -> None:
        self.chunks = chunks
        self.stream_calls = 0
        self.one_shot_calls = 0

    async def astream_turn(self, request: ModelRequest):  # noqa: ANN202
        del request
        self.stream_calls += 1
        for chunk in self.chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        del request
        self.one_shot_calls += 1
        return ModelTurn(response_id="one-shot", final_text="one shot")


def _loop(
    tmp_path: Path,
    adapter: object,
    *,
    observer_factories: tuple = (),
    event_sink: MemoryEventSink | None = None,
    stream_model_calls: bool = False,
    emit_output_deltas: bool = False,
    model_content_file: bool = False,
    metadata: dict[str, object] | None = None,
) -> AgentLoop:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    return AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace,
            run_root=tmp_path / "runs",
            metadata={} if metadata is None else metadata,
        ),
        model_adapter=adapter,  # type: ignore[arg-type]
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
        model_stream_observer_factories=observer_factories,
        event_sinks=(() if event_sink is None else (event_sink,)),
        stream_model_calls=stream_model_calls,
        emit_output_deltas=emit_output_deltas,
        model_content_file=model_content_file,
    )


def test_observer_gets_filtered_content_context_and_completed_outcome(tmp_path: Path) -> None:
    instances: list[_RecordingObserver] = []

    def factory() -> _RecordingObserver:
        observer = _RecordingObserver()
        instances.append(observer)
        return observer

    adapter = _ScriptedStreamAdapter(
        [
            ReasoningDelta("thinking"),
            TextDelta("Hel"),
            TextDelta("lo"),
            TurnComplete(
                response_id="response-1",
                usage={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
            ),
        ]
    )
    sink = MemoryEventSink()
    loop = _loop(tmp_path, adapter, observer_factories=(factory,), event_sink=sink)

    assert instances == []  # factories are activation-scoped, not constructor-scoped
    result = loop.run_once("hello")

    assert result.final_text == "Hello"
    assert adapter.stream_calls == 1
    assert adapter.one_shot_calls == 0
    assert len(instances) == 1
    observer = instances[0]
    assert len(observer.contexts) == 1
    context = observer.contexts[0]
    assert context.run_id == result.run_id
    assert context.root_run_id == result.run_id
    assert context.turn_id == "turn_0001"
    assert context.stream_id.startswith("stream_")
    assert context.step == 1
    assert context.provider == "test-provider"
    assert context.model == "gpt-5.5"
    assert context.started_at.endswith("Z")
    assert observer.writers[0].deltas == [
        ModelStreamDelta(channel="reasoning", text="thinking"),
        ModelStreamDelta(channel="output", text="Hel"),
        ModelStreamDelta(channel="output", text="lo"),
    ]
    assert observer.writers[0].dispatches == 1
    assert observer.writers[0].outcomes == [
        ModelStreamOutcome(
            status="completed",
            final_text="Hello",
            usage={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        )
    ]
    # The new observer channel does not imply the legacy durable content channel.
    assert not [event for event in sink.events if event.type.endswith(".delta")]
    started = next(event for event in sink.events if event.type == "model.turn.started")
    settled = next(event for event in sink.events if event.type == "turn.settled")
    assert settled.turn_id == started.turn_id
    assert settled.parent_id == started.event_id


def test_dispatch_preparation_failure_aborts_before_provider_entry(tmp_path: Path) -> None:
    class DispatchFailureWriter:
        def __init__(self) -> None:
            self.dispatches = 0
            self.aborts = 0
            self.closes = 0

        def begin_dispatch(self) -> None:
            self.dispatches += 1
            raise OSError("durable stream reset unavailable")

        def push(self, delta: ModelStreamDelta) -> None:
            del delta
            raise AssertionError("provider must not emit")

        def abort(self) -> None:
            self.aborts += 1

        def close(self, outcome: ModelStreamOutcome) -> None:
            del outcome
            self.closes += 1

    class DispatchFailureObserver:
        def __init__(self) -> None:
            self.writer = DispatchFailureWriter()

        def open(self, context: ModelStreamContext) -> DispatchFailureWriter:
            del context
            return self.writer

    observer = DispatchFailureObserver()
    adapter = _ScriptedStreamAdapter(
        [TextDelta("unreachable"), TurnComplete(response_id="unreachable")]
    )
    loop = _loop(tmp_path, adapter, observer_factories=(lambda: observer,))

    loop.open()
    try:
        suspension = loop.run_until_suspended("go")
    finally:
        loop.close()

    assert suspension.reason == "terminal"
    assert adapter.stream_calls == 0
    assert observer.writer.dispatches == 1
    assert observer.writer.aborts == 1
    assert observer.writer.closes == 0


@pytest.mark.parametrize("losing_open", ("private", "observer"))
def test_lease_loss_during_each_stream_writer_open_fences_later_opens_and_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    losing_open: str,
) -> None:
    token = CancellationToken()
    first = _RecordingObserver()
    second = _RecordingObserver()
    adapter = _ScriptedStreamAdapter([TextDelta("unreachable"), TurnComplete(response_id="r1")])
    loop = _loop(
        tmp_path,
        adapter,
        observer_factories=(lambda: first, lambda: second),
        model_content_file=losing_open == "private",
    )
    loop.cancellation_token = token
    loop.open()
    assert loop._session is not None

    if losing_open == "private":
        recorder = loop._session.res.recorder
        original_open = recorder.open_model_stream

        def lose_after_private_open(context: ModelStreamContext):  # noqa: ANN202
            writer = original_open(context)
            loop.lose_writer_authority()
            return writer

        monkeypatch.setattr(recorder, "open_model_stream", lose_after_private_open)
    else:
        original_open = first.open

        def lose_after_observer_open(context: ModelStreamContext):  # noqa: ANN202
            writer = original_open(context)
            loop.lose_writer_authority()
            return writer

        monkeypatch.setattr(first, "open", lose_after_observer_open)

    try:
        suspension = loop.run_until_suspended("go")
    finally:
        loop.discard_uncommitted()

    assert suspension.reason == "interrupted"
    assert suspension.interruption_cause is InterruptionCause.LEASE_LOST
    assert adapter.stream_calls == 0
    assert second.contexts == []
    if losing_open == "private":
        assert first.contexts == []
    else:
        assert len(first.contexts) == 1


def test_one_shot_adapter_still_closes_observer_with_settled_output(tmp_path: Path) -> None:
    class OneShotAdapter:
        supports_multimodal = False
        provider_name = "one-shot-provider"

        def __init__(self) -> None:
            self.calls = 0

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            self.calls += 1
            return ModelTurn(
                response_id="one-shot-1",
                final_text="whole answer",
                usage={"output_tokens": 2},
            )

    observer = _RecordingObserver()
    adapter = OneShotAdapter()
    loop = _loop(tmp_path, adapter, observer_factories=(lambda: observer,))

    result = loop.run_once("go")

    assert adapter.calls == 1
    assert result.final_text == "whole answer"
    assert observer.writers[0].deltas == []
    assert observer.writers[0].outcomes == [
        ModelStreamOutcome(
            status="completed",
            final_text="whole answer",
            usage={"input_tokens": 0, "output_tokens": 2, "total_tokens": 0},
        )
    ]


def test_lease_loss_during_first_stream_close_stops_remaining_writers(tmp_path: Path) -> None:
    token = CancellationToken()
    entered, release = Event(), Event()

    class BlockingCloseWriter(_RecordingWriter):
        def close(self, outcome: ModelStreamOutcome) -> None:
            entered.set()
            assert release.wait(5)
            super().close(outcome)

    class FixedObserver:
        def __init__(self, writer: _RecordingWriter) -> None:
            self.writer = writer

        def open(self, context: ModelStreamContext) -> _RecordingWriter:
            del context
            return self.writer

    class OneShotAdapter:
        def next_turn(self, request: ModelRequest) -> ModelTurn:
            del request
            return ModelTurn(response_id="r-close", final_text="done")

    first, second = BlockingCloseWriter(), _RecordingWriter()
    loop = _loop(
        tmp_path,
        OneShotAdapter(),
        observer_factories=(lambda: FixedObserver(first), lambda: FixedObserver(second)),
    )
    loop.cancellation_token = token
    loop.open()

    def lose_authority() -> None:
        assert entered.wait(5)
        token.cancel(InterruptionCause.GRACEFUL_DRAIN)
        loop.lose_writer_authority()
        release.set()

    racer = Thread(target=lose_authority)
    racer.start()
    try:
        suspension = loop.run_until_suspended("go")
    finally:
        racer.join(5)
        loop.discard_uncommitted()

    assert not racer.is_alive()
    assert suspension.interruption_cause is InterruptionCause.LEASE_LOST
    assert len(first.outcomes) == 1
    assert second.outcomes == []


def test_lease_loss_during_first_stream_push_stops_remaining_writers(tmp_path: Path) -> None:
    token = CancellationToken()
    entered, release = Event(), Event()

    class BlockingPushWriter(_RecordingWriter):
        def push(self, delta: ModelStreamDelta) -> None:
            entered.set()
            assert release.wait(5)
            super().push(delta)

    class FixedObserver:
        def __init__(self, writer: _RecordingWriter) -> None:
            self.writer = writer

        def open(self, context: ModelStreamContext) -> _RecordingWriter:
            del context
            return self.writer

    first, second = BlockingPushWriter(), _RecordingWriter()
    loop = _loop(
        tmp_path,
        _ScriptedStreamAdapter(
            [TextDelta("partial"), TurnComplete(response_id="unreachable")]
        ),
        observer_factories=(lambda: FixedObserver(first), lambda: FixedObserver(second)),
    )
    loop.cancellation_token = token
    loop.open()

    def lose_authority() -> None:
        assert entered.wait(5)
        token.cancel(InterruptionCause.GRACEFUL_DRAIN)
        loop.lose_writer_authority()
        release.set()

    racer = Thread(target=lose_authority)
    racer.start()
    try:
        suspension = loop.run_until_suspended("go")
    finally:
        racer.join(5)
        loop.discard_uncommitted()

    assert not racer.is_alive()
    assert suspension.interruption_cause is InterruptionCause.LEASE_LOST
    assert first.deltas == [ModelStreamDelta(channel="output", text="partial")]
    assert second.deltas == []
    assert first.outcomes == []
    assert second.outcomes == []


def test_root_stream_context_ignores_forged_request_metadata(tmp_path: Path) -> None:
    observer = _RecordingObserver()
    adapter = _ScriptedStreamAdapter([TextDelta("safe"), TurnComplete(response_id="r1")])
    loop = _loop(
        tmp_path,
        adapter,
        observer_factories=(lambda: observer,),
        metadata={"root_run_id": "victim-root"},
    )

    result = loop.run_once("go")

    assert observer.contexts[0].run_id == result.run_id
    assert observer.contexts[0].root_run_id == result.run_id


def test_stream_model_calls_forces_streaming_and_token_boundary_interrupt(tmp_path: Path) -> None:
    sink = MemoryEventSink()

    class InterruptingAdapter(_ScriptedStreamAdapter):
        loop: AgentLoop

        async def astream_turn(self, request: ModelRequest):  # noqa: ANN202
            del request
            self.stream_calls += 1
            yield TextDelta("part 1")
            self.loop.interrupt_turn()
            yield TextDelta("part 2")
            yield TextDelta("unreachable")
            yield TurnComplete(response_id="late")

    adapter = InterruptingAdapter([])
    loop = _loop(tmp_path, adapter, event_sink=sink, stream_model_calls=True)
    adapter.loop = loop
    loop.open()
    try:
        suspension = loop.run_until_suspended("go")
        assert suspension.reason == "interrupted"
        assert adapter.stream_calls == 1
        assert adapter.one_shot_calls == 0
        # Streaming for responsiveness alone publishes no durable content events.
        assert not [event for event in sink.events if event.type.endswith(".delta")]
        started = next(event for event in sink.events if event.type == "model.turn.started")
        interrupted = next(event for event in sink.events if event.type == "turn.interrupted")
        assert interrupted.turn_id == started.turn_id
        assert interrupted.parent_id == started.event_id
    finally:
        loop.close()


@pytest.mark.parametrize(
    ("cause", "suspension_reason", "stream_status", "stream_error_code"),
    (
        (InterruptionCause.DEADLINE, "terminal", "timed_out", "run_timeout"),
        (
            InterruptionCause.GRACEFUL_DRAIN,
            "interrupted",
            "interrupted",
            "graceful_drain",
        ),
        (
            InterruptionCause.HOST_SHUTDOWN,
            "interrupted",
            "interrupted",
            "host_shutdown",
        ),
    ),
)
def test_typed_run_cancellation_closes_the_model_stream_with_the_same_classification(
    tmp_path: Path,
    cause: InterruptionCause,
    suspension_reason: str,
    stream_status: ModelStreamStatus,
    stream_error_code: str,
) -> None:
    token = CancellationToken()
    observer = _RecordingObserver()

    class CancellingAdapter(_ScriptedStreamAdapter):
        async def astream_turn(self, request: ModelRequest):  # noqa: ANN202
            del request
            self.stream_calls += 1
            yield TextDelta("partial")
            token.cancel(cause)
            await asyncio.Event().wait()

    loop = _loop(
        tmp_path,
        CancellingAdapter([]),
        observer_factories=(lambda: observer,),
    )
    loop.cancellation_token = token
    loop.open()
    try:
        suspension = loop.run_until_suspended("go")
    finally:
        loop.discard_uncommitted()

    assert suspension.reason == suspension_reason
    assert suspension.interruption_cause is cause
    assert observer.writers[0].outcomes == [
        ModelStreamOutcome(
            status=stream_status,
            final_text="partial",
            error_code=stream_error_code,
        )
    ]


def test_partial_output_closes_observer_as_failed(tmp_path: Path) -> None:
    observer = _RecordingObserver()
    adapter = _ScriptedStreamAdapter(
        [TextDelta("partial"), ModelAdapterError("provider broke", retryable=True)]
    )
    loop = _loop(tmp_path, adapter, observer_factories=(lambda: observer,))

    loop.open()
    try:
        suspension = loop.run_until_suspended("go")
    finally:
        loop.close()

    assert suspension.reason == "turn_failed"
    assert suspension.retryable is True
    assert observer.writers[0].outcomes == [
        ModelStreamOutcome(
            status="failed",
            final_text="partial",
            error_code="model_error",
            retryable=True,
        )
    ]


def test_a_config_fixable_stream_failure_says_so_on_the_outcome(tmp_path: Path) -> None:
    """The live lane classified a park with half the vocabulary the park itself carries.

    ``retryable`` reached the outcome from the raised ``ModelAdapterError``; the sibling fact on
    the same exception did not, so the stream_closed record could say "waiting will not help" and
    never "changing the configuration will".
    """
    observer = _RecordingObserver()
    adapter = _ScriptedStreamAdapter(
        [
            TextDelta("partial"),
            ModelAdapterError(
                "the configured model is not available to this account",
                retryable=False,
                config_recoverable=True,
            ),
        ]
    )
    loop = _loop(tmp_path, adapter, observer_factories=(lambda: observer,))

    loop.open()
    try:
        suspension = loop.run_until_suspended("go")
    finally:
        loop.close()

    assert (suspension.retryable, suspension.config_recoverable) == (False, True)
    assert observer.writers[0].outcomes == [
        ModelStreamOutcome(
            status="failed",
            final_text="partial",
            error_code="model_error",
            retryable=False,
            config_recoverable=True,
        )
    ]


def test_model_stream_outcome_requires_boolean_retryability() -> None:
    with pytest.raises(ValueError, match="retryable must be a boolean"):
        ModelStreamOutcome(status="failed", retryable=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="config_recoverable must be a boolean"):
        ModelStreamOutcome(status="failed", config_recoverable=1)  # type: ignore[arg-type]


def test_observer_factory_open_push_and_close_failures_are_isolated(tmp_path: Path) -> None:
    class BrokenWriter:
        def push(self, delta: ModelStreamDelta) -> None:
            del delta
            raise RuntimeError("push failed")

        def close(self, outcome: ModelStreamOutcome) -> None:
            del outcome
            raise RuntimeError("close failed")

    class BrokenObserver:
        def open(self, context: ModelStreamContext) -> BrokenWriter:
            del context
            return BrokenWriter()

    class OpenFailureObserver:
        def open(self, context: ModelStreamContext) -> BrokenWriter:
            del context
            raise RuntimeError("open failed")

    def broken_factory() -> _RecordingObserver:
        raise RuntimeError("factory failed")

    adapter = _ScriptedStreamAdapter([TextDelta("ok"), TurnComplete(response_id="r1")])
    loop = _loop(
        tmp_path,
        adapter,
        observer_factories=(broken_factory, BrokenObserver, OpenFailureObserver),
    )

    result = loop.run_once("go")

    assert result.status == "completed"
    assert result.final_text == "ok"
    assert adapter.stream_calls == 1


def test_legacy_delta_events_and_observer_fan_out_together(tmp_path: Path) -> None:
    observer = _RecordingObserver()
    sink = MemoryEventSink()
    adapter = _ScriptedStreamAdapter(
        [ReasoningDelta("why"), TextDelta("answer"), TurnComplete(response_id="r1")]
    )
    loop = _loop(
        tmp_path,
        adapter,
        observer_factories=(lambda: observer,),
        event_sink=sink,
        emit_output_deltas=True,
    )

    result = loop.run_once("go")

    assert result.final_text == "answer"
    assert [
        event.data["text"] for event in sink.events if event.type == "model.reasoning.delta"
    ] == ["why"]
    assert [event.data["text"] for event in sink.events if event.type == "model.output.delta"] == [
        "answer"
    ]
    assert observer.writers[0].deltas == [
        ModelStreamDelta(channel="reasoning", text="why"),
        ModelStreamDelta(channel="output", text="answer"),
    ]


def test_subagent_materializes_fresh_observer_with_root_lineage(tmp_path: Path) -> None:
    parent_marker = "[[parent-stream-test]]"
    child_marker = "[[child-stream-test]]"
    instances: list[_RecordingObserver] = []

    def factory() -> _RecordingObserver:
        observer = _RecordingObserver()
        instances.append(observer)
        return observer

    class RoutingAdapter:
        supports_multimodal = False
        parent_calls = 0

        async def astream_turn(self, request: ModelRequest):  # noqa: ANN202
            if child_marker in request.system_prompt:
                yield TextDelta("child done")
                yield TurnComplete(response_id="child-1")
                return
            assert parent_marker in request.system_prompt
            self.parent_calls += 1
            if self.parent_calls == 1:
                yield ToolCallDelta(
                    index=0,
                    id="spawn-1",
                    name="agent_spawn",
                    arguments_fragment='{"subagent_type":"child","prompt":"work"}',
                )
                yield TurnComplete(response_id="parent-1")
                return
            yield TextDelta("parent done")
            yield TurnComplete(response_id="parent-2")

        def next_turn(self, request: ModelRequest) -> ModelTurn:  # pragma: no cover
            del request
            raise AssertionError("parent and child must both inherit streaming")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = RoutingAdapter()
    parent_config = AgentRuntimeConfig(
        definition_id="parent",
        prompt=PromptSpec(persona_segments=(parent_marker,)),
        tools=(tool_binding("agent.spawn"),),
    )
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,  # type: ignore[arg-type]
        runtime_config_provider=runtime_provider(parent_config),
        subagent_definitions={
            "child": SubagentDefinition(prompt=PromptSpec(persona_segments=(child_marker,)))
        },
        model_stream_observer_factories=(factory,),
        stream_model_calls=True,
    )

    result = loop.run_once("delegate")

    assert result.final_text == "parent done"
    assert len(instances) == 2  # one activation-local instance for parent, one for child
    parent_observer, child_observer = instances
    assert len(parent_observer.contexts) == 2
    assert len(child_observer.contexts) == 1
    child_context = child_observer.contexts[0]
    assert child_context.run_id != result.run_id
    assert child_context.root_run_id == result.run_id
    assert {context.root_run_id for context in parent_observer.contexts} == {result.run_id}
    stream_ids = {context.stream_id for observer in instances for context in observer.contexts}
    assert len(stream_ids) == 3


def test_restore_materializes_a_fresh_observer_snapshot(tmp_path: Path) -> None:
    instances: list[_RecordingObserver] = []

    def factory() -> _RecordingObserver:
        observer = _RecordingObserver()
        instances.append(observer)
        return observer

    adapter = _ScriptedStreamAdapter([TextDelta("unused"), TurnComplete(response_id="r1")])
    original = _loop(tmp_path, adapter, observer_factories=(factory,))
    original.open()
    checkpoint = original.snapshot()
    assert checkpoint is not None
    original.discard_uncommitted()

    restored = AgentLoop(
        spec=original.spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
        model_stream_observer_factories=(factory,),
    )
    restored.restore(checkpoint)
    try:
        assert len(instances) == 2
        assert instances[0] is not instances[1]
    finally:
        restored.discard_uncommitted()


def test_private_content_file_streams_without_durable_delta_events(tmp_path: Path) -> None:
    sink = MemoryEventSink()
    adapter = _ScriptedStreamAdapter(
        [ReasoningDelta("reason"), TextDelta("private answer"), TurnComplete(response_id="r1")]
    )
    loop = _loop(tmp_path, adapter, event_sink=sink, model_content_file=True)

    result = loop.run_once("go")

    records = [
        json.loads(line)
        for line in (result.run_dir / "model-content.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["kind"] for record in records] == [
        "stream_opened",
        "stream_segment",
        "stream_segment",
        "stream_closed",
        "settled_text",
    ]
    assert not [event for event in sink.events if event.type.endswith(".delta")]


def test_run_stream_alone_does_not_create_private_sidecar(tmp_path: Path) -> None:
    adapter = _ScriptedStreamAdapter([TextDelta("live only"), TurnComplete(response_id="r1")])
    loop = _loop(tmp_path, adapter)

    async def drive() -> tuple[list[object], object]:
        await loop.aopen()
        items: list[object] = []
        async with loop.astream("go") as stream:
            async for item in stream:
                items.append(item)
        return items, await loop.aclose()

    items, result = asyncio.run(drive())

    assert TextDelta("live only") in items
    assert not (result.run_dir / "model-content.jsonl").exists()


def test_run_stream_and_passive_observer_receive_their_distinct_chunk_sets(
    tmp_path: Path,
) -> None:
    observer = _RecordingObserver()
    sink = MemoryEventSink()
    tool_delta = ToolCallDelta(
        index=0,
        id="finish-1",
        name="run_finish",
        arguments_fragment='{"summary":"done"}',
    )
    adapter = _ScriptedStreamAdapter(
        [
            ReasoningDelta("thinking"),
            TextDelta("visible"),
            tool_delta,
            TurnComplete(response_id="r1"),
        ]
    )
    loop = _loop(
        tmp_path,
        adapter,
        observer_factories=(lambda: observer,),
        event_sink=sink,
        emit_output_deltas=True,
    )

    async def drive() -> tuple[list[object], object, object]:
        await loop.aopen()
        items: list[object] = []
        async with loop.astream("go") as stream:
            async for item in stream:
                items.append(item)
            result = stream.result
            suspension = stream.suspension
        await loop.aclose()
        return items, result, suspension

    items, result, suspension = asyncio.run(drive())

    assert suspension is None
    assert result.status == "completed"
    assert [item for item in items if isinstance(item, ReasoningDelta)] == [
        ReasoningDelta("thinking")
    ]
    assert [item for item in items if isinstance(item, TextDelta)] == [TextDelta("visible")]
    assert tool_delta in items
    assert any(isinstance(item, TurnComplete) for item in items)
    assert not [event for event in sink.events if event.type.endswith(".delta")]
    assert not [
        item for item in items if isinstance(item, AgentEvent) and item.type.endswith(".delta")
    ]
    assert observer.writers[0].deltas == [
        ModelStreamDelta(channel="reasoning", text="thinking"),
        ModelStreamDelta(channel="output", text="visible"),
    ]
    assert observer.writers[0].outcomes[0].status == "completed"


def test_run_stream_interrupt_drains_the_call_then_surfaces_a_suspension(tmp_path: Path) -> None:
    observer = _RecordingObserver()

    class InterruptingRunStreamAdapter(_ScriptedStreamAdapter):
        loop: AgentLoop

        async def astream_turn(self, request: ModelRequest):  # noqa: ANN202
            del request
            self.stream_calls += 1
            yield TextDelta("before ")
            self.loop.interrupt_turn()
            yield TextDelta("after")
            yield TurnComplete(response_id="r1")

    adapter = InterruptingRunStreamAdapter([])
    loop = _loop(tmp_path, adapter, observer_factories=(lambda: observer,))
    adapter.loop = loop

    async def drive() -> tuple[list[object], object, object]:
        await loop.aopen()
        items: list[object] = []
        async with loop.astream("go") as stream:
            async for item in stream:
                items.append(item)
            result = stream.result
            suspension = stream.suspension
        await loop.aclose()
        return items, result, suspension

    items, result, suspension = asyncio.run(drive())

    assert result is None
    assert suspension.reason == "interrupted"
    assert [item.text for item in items if isinstance(item, TextDelta)] == ["before ", "after"]
    assert observer.writers[0].deltas == [
        ModelStreamDelta(channel="output", text="before "),
        ModelStreamDelta(channel="output", text="after"),
    ]
    assert observer.writers[0].outcomes == [
        ModelStreamOutcome(
            status="completed",
            final_text="before after",
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
    ]


# --- abort drain (v0.24) ------------------------------------------------------------------------

_DRAINED_BILL = {"input_tokens": 6, "output_tokens": 3, "total_tokens": 9}


def test_interrupt_with_abort_drain_delivers_nothing_after_stop_to_observers_or_delta_events(
    tmp_path: Path,
) -> None:
    sink = MemoryEventSink()
    observer = _RecordingObserver()

    class InterruptingAdapter(_ScriptedStreamAdapter):
        loop: AgentLoop
        yielded = 0

        async def astream_turn(self, request: ModelRequest):  # noqa: ANN202
            del request
            self.stream_calls += 1
            for index, chunk in enumerate(
                (
                    TextDelta("part 1"),
                    TextDelta("part 2"),
                    TextDelta("drained"),
                    ReasoningDelta("drained thought"),
                    TurnComplete(response_id="late", usage=_DRAINED_BILL),
                )
            ):
                if index == 1:
                    self.loop.interrupt_turn()
                self.yielded += 1
                yield chunk

    adapter = InterruptingAdapter([])
    loop = _loop(
        tmp_path,
        adapter,
        event_sink=sink,
        observer_factories=(lambda: observer,),
        emit_output_deltas=True,
    )
    loop.async_model_abort_drain_s = 5.0
    adapter.loop = loop
    loop.open()
    try:
        suspension = loop.run_until_suspended("go")
        assert suspension.reason == "interrupted"
        assert adapter.yielded == 5, "the stream was not drained"
        texts = [e.data["text"] for e in sink.events if e.type == "model.output.delta"]
        assert texts == ["part 1", "part 2"]
        assert not [e for e in sink.events if e.type == "model.reasoning.delta"]
        assert [delta.text for delta in observer.writers[0].deltas] == ["part 1", "part 2"]
        assert observer.writers[0].outcomes == [
            ModelStreamOutcome(
                status="interrupted", final_text="part 1part 2", error_code="interrupted"
            )
        ]
        assert loop._session is not None
        assert dict(loop._session.state.total_usage) == _DRAINED_BILL
    finally:
        loop.close()


def test_a_user_cancel_with_abort_drain_bills_the_cancelled_run(tmp_path: Path) -> None:
    """The run's cancel token (the RunStream/host Stop) drains too, and stays a cancellation."""

    token = CancellationToken()
    observer = _RecordingObserver()

    class CancellingAdapter(_ScriptedStreamAdapter):
        yielded = 0

        async def astream_turn(self, request: ModelRequest):  # noqa: ANN202
            del request
            self.stream_calls += 1
            self.yielded += 1
            yield TextDelta("partial")
            token.cancel(InterruptionCause.USER_CANCEL)
            for chunk in (TextDelta("late"), TurnComplete(response_id="r", usage=_DRAINED_BILL)):
                self.yielded += 1
                yield chunk

    adapter = CancellingAdapter([])
    loop = _loop(tmp_path, adapter, observer_factories=(lambda: observer,), stream_model_calls=True)
    loop.cancellation_token = token
    loop.async_model_abort_drain_s = 5.0
    loop.open()
    try:
        suspension = loop.run_until_suspended("go")
        assert suspension.interruption_cause is InterruptionCause.USER_CANCEL
        assert adapter.yielded == 3
        assert [delta.text for delta in observer.writers[0].deltas] == ["partial"]
        assert [outcome.status for outcome in observer.writers[0].outcomes] == ["cancelled"]
        assert loop._session is not None
        assert dict(loop._session.state.total_usage) == _DRAINED_BILL
    finally:
        loop.discard_uncommitted()


def _stamped_drop() -> ModelAdapterError:
    failure = ModelAdapterError("gateway stream dropped", error_code="model_stream_dropped")
    mark_provider_usage(failure, _DRAINED_BILL)
    return failure


@pytest.mark.parametrize(
    ("drain_s", "make_failure", "expected_usage"),
    [
        (0.0, _stamped_drop, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
        (5.0, _stamped_drop, _DRAINED_BILL),
        (
            5.0,
            lambda: ConnectionError("connection reset"),
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        ),
    ],
    ids=["drain_off", "drain_on_stamped_failure", "drain_on_raw_exception"],
)
def test_a_read_failing_after_a_user_cancel_still_settles_the_run_as_that_cancel(
    tmp_path: Path, drain_s: float, make_failure: Any, expected_usage: dict[str, int]
) -> None:
    """The Stop's loop outcome does not depend on the drain: ``limited``/``user_cancel``.

    The cancel lands while the provider read is in flight and that read then fails with no
    further chunk. Without the drain the race ends the call at the cancel; with it, the failure
    comes after the stop was decided and must not turn the run ``failed``. A failure's own usage
    stamp is the drained bill.
    """

    token = CancellationToken()

    class FailingAfterCancel(_ScriptedStreamAdapter):
        async def astream_turn(self, request: ModelRequest):  # noqa: ANN202
            del request
            self.stream_calls += 1
            yield TextDelta("partial")
            token.cancel(InterruptionCause.USER_CANCEL)
            await asyncio.sleep(0.05)
            raise make_failure()

    loop = _loop(tmp_path, FailingAfterCancel([]), stream_model_calls=True)
    loop.cancellation_token = token
    loop.async_model_abort_drain_s = drain_s
    loop.open()
    try:
        suspension = loop.run_until_suspended("go")
        assert suspension.interruption_cause is InterruptionCause.USER_CANCEL
        assert suspension.status == "limited"
        assert loop._session is not None
        assert dict(loop._session.state.total_usage) == expected_usage
    finally:
        loop.discard_uncommitted()


def test_an_early_break_from_astream_waits_out_the_drain_and_keeps_its_bill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leaving the ``astream`` block early cancels the run; the drain that starts must finish.

    ``RunStream``'s own hard-cancel fallback is shrunk below the drain here (it is 8 s in
    production) to stand in for a drain longer than that fallback: the loop, not the fallback's
    default, must size the wait, or the host cancel cuts the drain and loses the bill.
    """

    from monoid_agent_kernel.core.streaming import RunStream

    monkeypatch.setitem(RunStream.__init__.__kwdefaults__, "cancel_grace_s", 0.3)

    class SlowTerminalAdapter(_ScriptedStreamAdapter):
        yielded = 0

        async def astream_turn(self, request: ModelRequest):  # noqa: ANN202
            del request
            self.stream_calls += 1
            self.yielded += 1
            yield TextDelta("partial")
            await asyncio.sleep(0.6)
            for chunk in (TextDelta("late"), TurnComplete(response_id="r", usage=_DRAINED_BILL)):
                self.yielded += 1
                yield chunk

    adapter = SlowTerminalAdapter([])
    loop = _loop(tmp_path, adapter, stream_model_calls=True)
    loop.async_model_abort_drain_s = 1.0
    loop.async_model_cancel_grace_s = 0.2

    async def drive() -> tuple[list[object], object]:
        await loop.aopen()
        items: list[object] = []
        async with loop.astream("go") as stream:
            async for item in stream:
                items.append(item)
                if isinstance(item, TextDelta):
                    break
        result = stream.result
        await loop.aclose()
        return items, result

    items, result = asyncio.run(drive())

    assert [item.text for item in items if isinstance(item, TextDelta)] == ["partial"]
    assert adapter.yielded == 3, "the host cancel cut the drain"
    assert result is not None
    assert result.interruption_cause is InterruptionCause.USER_CANCEL
    assert result.metrics["total_tokens"] == _DRAINED_BILL["total_tokens"]


@pytest.mark.parametrize("knob", ["5", True], ids=["numeric_string", "bool"])
def test_a_drain_knob_that_is_not_a_real_number_means_no_drain_at_the_loop(
    tmp_path: Path, knob: Any
) -> None:
    """Not a positive finite number means no drain -- a string or a bool is not one.

    Both readers of the loop's knob agree: the runner drains nothing and ``astream`` keeps the
    8 s early-exit wait it has with the drain off.
    """

    loop = _loop(tmp_path, _ScriptedStreamAdapter([]), stream_model_calls=True)

    async def read() -> tuple[float, float]:
        await loop.aopen()
        try:
            loop.async_model_abort_drain_s = knob
            runner = loop._bootstrap_resources.model_runner
            return runner._abort_drain_s(), loop.astream("go")._cancel_grace_s
        finally:
            await loop.aclose()

    assert asyncio.run(read()) == (0.0, 8.0)


def test_astream_sizes_its_early_exit_wait_from_the_drain_read_as_it_opens(tmp_path: Path) -> None:
    """8 s with the drain off (today's value); drain + model cancel grace on top with it on."""

    loop = _loop(tmp_path, _ScriptedStreamAdapter([]), stream_model_calls=True)

    async def waits() -> list[float]:
        await loop.aopen()
        try:
            seen = [loop.astream("go")._cancel_grace_s]
            loop.async_model_abort_drain_s = 3.0
            loop.async_model_cancel_grace_s = 0.5
            seen.append(loop.astream("go")._cancel_grace_s)
            loop.async_model_abort_drain_s = float("nan")
            seen.append(loop.astream("go")._cancel_grace_s)
            return seen
        finally:
            await loop.aclose()

    assert asyncio.run(waits()) == [8.0, 11.5, 8.0]
