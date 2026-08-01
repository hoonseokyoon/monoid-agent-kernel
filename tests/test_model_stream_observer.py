from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from support.runtime import runtime_config, runtime_provider, tool_binding

from monoid_agent_kernel.core.agents import AgentRuntimeConfig, PromptSpec, SubagentDefinition
from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.model_stream import (
    ModelStreamContext,
    ModelStreamDelta,
    ModelStreamOutcome,
)
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
)
from monoid_agent_kernel.recorder import MemoryEventSink


class _RecordingWriter:
    def __init__(self) -> None:
        self.deltas: list[ModelStreamDelta] = []
        self.outcomes: list[ModelStreamOutcome] = []

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


def test_model_stream_outcome_requires_boolean_retryability() -> None:
    with pytest.raises(ValueError, match="retryable must be a boolean"):
        ModelStreamOutcome(status="failed", retryable=1)  # type: ignore[arg-type]


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
