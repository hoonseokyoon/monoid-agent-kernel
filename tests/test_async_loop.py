"""P1 async-native core: async entry points, native async adapters, sync facade guard.

Tests drive the async API with ``asyncio.run`` from plain sync test functions, so no
pytest-asyncio plugin is needed. The existing sync suite already covers the facade ->
async-core path transitively.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from support.process import python_command as _python_command
from support.runtime import runtime_config, runtime_provider, tool_binding

from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.spec import AgentRunSpec, RunLimits
from monoid_agent_kernel.errors import NativeAgentError
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import (
    ModelRequest,
    ModelTurn,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    TurnComplete,
    assemble_streamed_turn,
)
from monoid_agent_kernel.providers.fake import (
    FakeModelAdapter,
    FakeStreamingModelAdapter,
    fake_tool_call,
)


def _spec(tmp_path: Path, *, mode: str = "apply", limits: RunLimits | None = None) -> AgentRunSpec:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    return AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        mode=mode,
        limits=limits or RunLimits(),
    )


def _loop(
    tmp_path: Path,
    adapter: object,
    *tool_ids: str,
    limits: RunLimits | None = None,
    mode: str = "apply",
) -> AgentLoop:
    return AgentLoop(
        spec=_spec(tmp_path, mode=mode, limits=limits),
        model_adapter=adapter,  # type: ignore[arg-type]
        runtime_config_provider=runtime_provider(runtime_config(*(tool_ids or ("fs.write",)))),
    )


def test_arun_once_writes_and_completes(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="1", tool_calls=(fake_tool_call("fs_write", {"path": "A.md", "content": "x"}, "c1"),)),
            ModelTurn(response_id="2", final_text="done"),
        ]
    )
    loop = _loop(tmp_path, adapter, "fs.write", "run.finish", limits=RunLimits(max_steps=4))
    result = asyncio.run(loop.arun_once("go"))

    assert result.status == "completed"
    assert result.final_text == "done"
    assert (tmp_path / "workspace" / "A.md").read_text() == "x"


def test_native_async_adapter_is_awaited(tmp_path: Path) -> None:
    class AsyncAdapter:
        supports_multimodal = False
        awaited = False

        def next_turn(self, request: ModelRequest) -> ModelTurn:  # pragma: no cover
            raise AssertionError("anext_turn should be preferred over next_turn")

        async def anext_turn(self, request: ModelRequest) -> ModelTurn:
            await asyncio.sleep(0)
            type(self).awaited = True
            return ModelTurn(response_id="a", final_text="native-async")

    loop = _loop(tmp_path, AsyncAdapter())
    result = asyncio.run(loop.arun_once("go"))

    assert result.status == "completed"
    assert result.final_text == "native-async"
    assert AsyncAdapter.awaited is True


def test_sync_api_inside_running_loop_raises(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    loop = _loop(tmp_path, adapter)

    async def call_sync_in_loop() -> object:
        try:
            loop.run_once("go")
        except NativeAgentError as exc:
            return exc.error_code
        return None

    assert asyncio.run(call_sync_in_loop()) == "sync_in_async_loop"


def test_async_and_sync_paths_agree(tmp_path: Path) -> None:
    def make_adapter() -> FakeModelAdapter:
        return FakeModelAdapter(turns=[ModelTurn(response_id="1", final_text="settled")])

    sync_loop = _loop(tmp_path / "s", make_adapter())
    async_loop = _loop(tmp_path / "a", make_adapter())

    sync_result = sync_loop.run_once("go")
    async_result = asyncio.run(async_loop.arun_once("go"))

    assert sync_result.status == async_result.status == "completed"
    assert sync_result.final_text == async_result.final_text == "settled"


def test_background_shell_job_completes_on_run_loop_and_reenters(tmp_path: Path) -> None:
    # B2: a background (resume_on_exit) shell job runs its asyncio subprocess monitor on the
    # run's always-on loop. The run parks, the monitor completes the subprocess on that loop
    # (no per-job thread, no 20ms poll), and the result reenters as an observation so the
    # next turn settles.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call(
                        "shell_exec",
                        {
                            "command": _python_command(
                                "import time; time.sleep(0.2); print('bg-done')"
                            ),
                            "background": True,
                        },
                        "c1",
                    ),
                ),
            ),
            # The job is still running here, so this turn parks (awaiting_tasks) rather than
            # settling; its final_text is discarded.
            ModelTurn(response_id="r2", final_text="still running"),
            # Reentry turn: the job result has been delivered, so this settles.
            ModelTurn(response_id="r3", final_text="done"),
        ]
    )
    config = runtime_config(
        bindings=(
            tool_binding(
                "shell.exec",
                runtime={"shell": {"approval_mode": "auto-approve", "default_timeout_s": 30}},
            ),
            tool_binding("run.finish"),
        )
    )
    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
    ).run_once("Run a background job.")

    assert result.status == "completed"
    assert result.final_text == "done"
    # The run parked and resumed: the model saw a later turn carrying the job result.
    assert len(adapter.requests) >= 3
    job_results = [
        obs
        for request in adapter.requests
        for obs in request.observations
        if obs.output.get("type") == "background_job_result"
    ]
    assert job_results and job_results[0].output["status"] == "exited"
    events = result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8")
    assert "job.started" in events and "job.finished" in events


def test_concurrent_runs_interleave_on_one_loop(tmp_path: Path) -> None:
    class SlowAsyncAdapter:
        supports_multimodal = False

        def next_turn(self, request: ModelRequest) -> ModelTurn:  # pragma: no cover
            raise AssertionError("anext_turn should be used")

        async def anext_turn(self, request: ModelRequest) -> ModelTurn:
            await asyncio.sleep(0.05)
            return ModelTurn(response_id="x", final_text="ok")

    async def run_both() -> list[str]:
        a = _loop(tmp_path / "a", SlowAsyncAdapter())
        b = _loop(tmp_path / "b", SlowAsyncAdapter())
        results = await asyncio.gather(a.arun_once("go"), b.arun_once("go"))
        return [r.status for r in results]

    assert asyncio.run(run_both()) == ["completed", "completed"]


# --- P4a: astream live streaming -------------------------------------------------------


async def _drain_stream(loop: AgentLoop, user_input: str) -> tuple[list[object], object, object]:
    """Open the run, drain one astream turn, then close — returning (items, result, suspension)."""
    await loop.aopen()
    items: list[object] = []
    async with loop.astream(user_input) as stream:
        async for item in stream:
            items.append(item)
        result = stream.result
        suspension = stream.suspension
    await loop.aclose()
    return items, result, suspension


def _event_types(items: list[object]) -> list[str]:
    return [it.type for it in items if isinstance(it, AgentEvent)]


def test_assemble_streamed_turn_folds_text_and_tool_args() -> None:
    turn = assemble_streamed_turn(
        [
            TextDelta("Hel"),
            TextDelta("lo"),
            ToolCallDelta(index=0, arguments_fragment='{"path":"A', id="c1", name="fs_write"),
            ToolCallDelta(index=0, arguments_fragment='.md"}'),
            TurnComplete(response_id="r1", usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}),
        ]
    )
    assert turn.final_text == "Hello"
    assert turn.response_id == "r1"
    assert turn.tool_calls == (ToolCall(id="c1", name="fs_write", arguments={"path": "A.md"}),)
    assert turn.usage == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}


def test_astream_streams_text_deltas_then_settles(tmp_path: Path) -> None:
    adapter = FakeStreamingModelAdapter(
        chunk_turns=[[TextDelta("Hel"), TextDelta("lo"), TurnComplete(response_id="r1")]]
    )
    loop = _loop(tmp_path, adapter)
    items, result, suspension = asyncio.run(_drain_stream(loop, "hi"))

    assert suspension is None
    assert result.status == "completed"
    assert result.final_text == "Hello"
    # Token deltas concatenate to the settled text.
    assert "".join(it.text for it in items if isinstance(it, TextDelta)) == "Hello"
    # Orchestration events stream too, bracketing the deltas in order.
    started = next(i for i, it in enumerate(items) if isinstance(it, AgentEvent) and it.type == "model.turn.started")
    finished = next(i for i, it in enumerate(items) if isinstance(it, AgentEvent) and it.type == "model.turn.finished")
    delta_idxs = [i for i, it in enumerate(items) if isinstance(it, TextDelta)]
    assert started < min(delta_idxs) <= max(delta_idxs) < finished


def test_astream_accumulates_streamed_tool_call(tmp_path: Path) -> None:
    adapter = FakeStreamingModelAdapter(
        chunk_turns=[
            [
                ToolCallDelta(index=0, arguments_fragment='{"path":"A', id="c1", name="fs_write"),
                ToolCallDelta(index=0, arguments_fragment='.md","content":"x"}'),
                TurnComplete(response_id="r1"),
            ],
            [TextDelta("done"), TurnComplete(response_id="r2")],
        ]
    )
    loop = _loop(tmp_path, adapter, "fs.write", "run.finish", limits=RunLimits(max_steps=4))
    items, result, _ = asyncio.run(_drain_stream(loop, "go"))

    assert result.status == "completed"
    assert result.final_text == "done"
    assert (tmp_path / "workspace" / "A.md").read_text() == "x"
    assert any(isinstance(it, ToolCallDelta) for it in items)
    types = _event_types(items)
    assert "tool.call.started" in types and "tool.call.finished" in types


def test_astream_nonstreaming_adapter_yields_orchestration_only(tmp_path: Path) -> None:
    # An adapter without astream_turn falls back to the one-shot path: orchestration events
    # still stream, but no token deltas are produced.
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="done")])
    loop = _loop(tmp_path, adapter)
    items, result, _ = asyncio.run(_drain_stream(loop, "go"))

    assert result.final_text == "done"
    assert not any(isinstance(it, (TextDelta, ToolCallDelta)) for it in items)
    assert "model.turn.started" in _event_types(items)
    assert "model.turn.finished" in _event_types(items)


def test_astream_early_break_cancels_cooperatively(tmp_path: Path) -> None:
    class InfiniteStreamAdapter:
        supports_multimodal = False

        def __init__(self) -> None:
            self.calls = 0

        def next_turn(self, request: ModelRequest) -> ModelTurn:  # pragma: no cover
            raise AssertionError("astream_turn should be preferred")

        async def astream_turn(self, request: ModelRequest):
            self.calls += 1
            await asyncio.sleep(0)
            yield ToolCallDelta(
                index=0,
                arguments_fragment=json.dumps({"path": "A.md", "content": str(self.calls)}),
                id=f"c{self.calls}",
                name="fs_write",
            )
            yield TurnComplete(response_id=f"r{self.calls}")

    adapter = InfiniteStreamAdapter()
    # The script never settles; only cooperative cancel (on break) stops it well short of the
    # huge step budget — proving early break tears the run down, not the limit.
    loop = _loop(tmp_path, adapter, "fs.write", limits=RunLimits(max_steps=100_000, max_tool_calls=100_000))

    async def go() -> tuple[object, int, bool]:
        await loop.aopen()
        async with loop.astream("go") as stream:
            async for _item in stream:
                break
        active_after = loop._stream_sink.active if loop._stream_sink else True  # type: ignore[union-attr]
        result = await loop.aclose()
        return result, adapter.calls, active_after

    result, calls, active_after = asyncio.run(go())
    assert active_after is False
    assert calls < 1_000  # cut short by cooperative cancel, not the 100k budget
    assert result.status == "limited"


def test_astream_parks_on_external_task(tmp_path: Path) -> None:
    adapter = FakeStreamingModelAdapter(
        chunk_turns=[
            [
                ToolCallDelta(index=0, arguments_fragment=json.dumps({"prompt": "Pick"}), id="c1", name="hitl_request"),
                TurnComplete(response_id="r1"),
            ]
        ]
    )
    loop = _loop(tmp_path, adapter, "hitl.request")
    items, result, suspension = asyncio.run(_drain_stream(loop, "ask the human"))

    assert result is None  # parked, not settled
    assert suspension is not None
    assert suspension.has_external is True
    assert len(suspension.awaiting_task_ids) == 1
    assert "run.awaiting_input" in _event_types(items)
