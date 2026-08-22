from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import pytest

from support.runtime import runtime_config, runtime_provider, tool_binding

from monoid_agent_kernel import tool
from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.capability import AutoGrantBroker, CapabilityLease
from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore
from monoid_agent_kernel.core.outcome import InterruptionCause
from monoid_agent_kernel.core.spec import AgentRunSpec, RunLimits
from monoid_agent_kernel.core.tool_surface import ToolScope
from monoid_agent_kernel.errors import RunCancelled, ToolExecutionError
from monoid_agent_kernel.core._sync_bridge import dispose_unawaited, start_abandonable_sync_call
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.reference.capability import HumanEscalationBroker
from monoid_agent_kernel.tools.base import ToolContext, ToolResult, ToolSpec


def _spec(tmp_path: Path, *, limits: RunLimits | None = None) -> AgentRunSpec:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    return AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        limits=limits or RunLimits(),
    )


_TENANT: ContextVar[str] = ContextVar("test_tenant", default="unset")


def _event_types(run_dir: Path) -> list[str]:
    return [
        json.loads(line)["type"]
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_async_and_sync_tool_handlers_use_native_and_worker_paths_sequentially(
    tmp_path: Path,
) -> None:
    seen: list[tuple[str, int]] = []
    active = 0
    peak = 0

    @tool(id="async.capture")
    async def async_capture(value: str) -> dict:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        seen.append((f"async-start:{value}", threading.get_ident()))
        await asyncio.sleep(0)
        seen.append((f"async-end:{value}", threading.get_ident()))
        active -= 1
        return {"value": value}

    @tool(id="sync.capture")
    def sync_capture(value: str) -> dict:
        seen.append((f"sync:{value}", threading.get_ident()))
        return {"value": value}

    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                tool_calls=(
                    fake_tool_call("async_capture", {"value": "one"}, "c1"),
                    fake_tool_call("async_capture", {"value": "two"}, "c2"),
                    fake_tool_call("sync_capture", {"value": "three"}, "c3"),
                )
            ),
            ModelTurn(final_text="done"),
        ]
    )

    async def run() -> tuple[object, int]:
        loop_thread = threading.get_ident()
        result = await AgentLoop.from_tools(
            _spec(tmp_path), adapter, [async_capture, sync_capture]
        ).arun_once("go")
        return result, loop_thread

    result, loop_thread = asyncio.run(run())

    assert result.status == "completed"
    assert peak == 1
    assert [label for label, _thread in seen] == [
        "async-start:one",
        "async-end:one",
        "async-start:two",
        "async-end:two",
        "sync:three",
    ]
    assert all(thread_id == loop_thread for label, thread_id in seen if label.startswith("async"))
    assert next(thread_id for label, thread_id in seen if label.startswith("sync")) != loop_thread
    lifecycle = [
        event
        for event in _event_types(result.run_dir)
        if event in {"tool.call.started", "tool.call.finished"}
    ]
    assert lifecycle == ["tool.call.started", "tool.call.finished"] * 3


def test_a_sync_tool_handler_may_hand_back_an_awaitable(tmp_path: Path) -> None:
    """The second defence on the tool half, and the twin of the model half's.

    A handler can be an ordinary function and still return something awaitable -- it delegates to an
    async client, or it is a callable object no predicate over functions recognises. The dispatch
    awaits what comes back rather than treating it as the result. Unbound until now: dropping the
    fallback passed the whole tool suite, because the shape it breaks is one nothing exercised.

    Its failure mode differs from the model half's, which is why this is worth its own test rather
    than an argument from the shared predicate: here `isinstance(result, ToolResult)` rejects the
    coroutine loudly, where the model dispatch had no such check and recorded a clean success for a
    provider it never called.
    """
    ran: list[str] = []

    class Provider:
        def get_tools(self, context: ToolContext) -> list[ToolSpec]:
            del context

            def handler(ctx: ToolContext, args: dict) -> Any:
                del ctx, args

                async def finish() -> ToolResult:
                    ran.append("awaited")
                    return ToolResult(ok=True, content={"done": True})

                return finish()

            return [
                ToolSpec(
                    id="sync.awaitable",
                    description="a sync handler that returns an awaitable",
                    input_schema={"type": "object", "additionalProperties": True},
                    capability="",
                    side_effect="read",
                    handler=handler,
                )
            ]

    result = asyncio.run(
        AgentLoop(
            spec=_spec(tmp_path),
            model_adapter=FakeModelAdapter(
                turns=[
                    ModelTurn(tool_calls=(fake_tool_call("sync_awaitable", {}, "c1"),)),
                    ModelTurn(final_text="done"),
                ]
            ),
            runtime_config_provider=runtime_provider(
                runtime_config(bindings=(tool_binding("sync.awaitable"),))
            ),
            tool_providers=(Provider(),),
        ).arun_once("go")
    )

    assert result.status == "completed"
    assert ran == ["awaited"], "the awaitable was taken as the result instead of being awaited"


def test_async_tool_controlled_error_becomes_ordered_failed_observation(tmp_path: Path) -> None:
    @tool(id="async.fail")
    async def fail() -> dict:
        await asyncio.sleep(0)
        raise ToolExecutionError("try again", error_code="async_tool_retry")

    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("async_fail", {}, "c1"),)),
            ModelTurn(final_text="recovered"),
        ]
    )
    result = asyncio.run(AgentLoop.from_tools(_spec(tmp_path), adapter, [fail]).arun_once("go"))

    assert result.status == "completed"
    error = adapter.requests[1].observations[0].output["error"]
    assert error["code"] == "async_tool_retry"
    assert error["retryable"] is True
    assert "tool.call.failed" in _event_types(result.run_dir)


def test_tool_local_cancelled_error_becomes_failed_observation(tmp_path: Path) -> None:
    @tool(id="async.self_cancel")
    async def self_cancel() -> dict:
        raise asyncio.CancelledError

    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("async_self_cancel", {}, "c1"),)),
            ModelTurn(final_text="continued"),
        ]
    )
    result = asyncio.run(
        AgentLoop.from_tools(_spec(tmp_path), adapter, [self_cancel]).arun_once("go")
    )

    assert result.status == "completed"
    error = adapter.requests[1].observations[0].output["error"]
    assert error["code"] == "tool_handler_cancelled"
    assert error["retryable"] is True
    assert "tool.call.failed" in _event_types(result.run_dir)


def test_host_cancelling_the_run_task_is_not_reported_as_a_tool_failure(tmp_path: Path) -> None:
    """The other kind of ``CancelledError``: the host cancelling the task that drives the run.

    ``tool_handler_cancelled`` belongs to a handler that cancelled *itself* -- the test above.
    Cancellation delivered to the awaiting task means the host stopped the run, and catching it
    alongside the handler's own turned it into one failed tool observation with the run carrying on
    to the next model call: work the host had already stopped.
    """

    @tool(id="async.block")
    async def block() -> dict:
        started.set()
        await asyncio.sleep(30)
        return {"late": True}

    started = asyncio.Event()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("async_block", {}, "c1"),)),
            ModelTurn(final_text="a step the run must never take"),
        ]
    )

    async def run() -> None:
        pending = asyncio.create_task(
            AgentLoop.from_tools(_spec(tmp_path), adapter, [block]).arun_once("go")
        )
        await asyncio.wait_for(started.wait(), timeout=10)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending

    asyncio.run(run())

    # The second turn is on the adapter and was never asked for: the run stopped where it was
    # cancelled rather than resuming past a tool call it recorded as failed.
    assert len(adapter.requests) == 1


def test_unexpected_async_tool_error_fails_run_and_clears_call_context(tmp_path: Path) -> None:
    cleared = asyncio.Event()

    @tool(id="async.boom")
    async def boom() -> dict:
        try:
            raise RuntimeError("unexpected async failure")
        finally:
            cleared.set()

    adapter = FakeModelAdapter(
        turns=[ModelTurn(tool_calls=(fake_tool_call("async_boom", {}, "c1"),))]
    )
    result = asyncio.run(AgentLoop.from_tools(_spec(tmp_path), adapter, [boom]).arun_once("go"))

    assert result.status == "failed"
    assert "unexpected async failure" in result.error
    assert cleared.is_set()


def test_run_cancellation_cancels_native_async_tool(tmp_path: Path) -> None:
    token = CancellationToken()

    async def run() -> tuple[object, bool]:
        started = asyncio.Event()
        cleaned_up = asyncio.Event()

        @tool(id="async.block")
        async def block() -> dict:
            started.set()
            try:
                await asyncio.Future()
            finally:
                cleaned_up.set()

        adapter = FakeModelAdapter(
            turns=[ModelTurn(tool_calls=(fake_tool_call("async_block", {}, "c1"),))]
        )
        loop = AgentLoop.from_tools(_spec(tmp_path), adapter, [block], cancellation_token=token)
        pending = asyncio.create_task(loop.arun_once("go"))
        await started.wait()
        token.cancel()
        result = await asyncio.wait_for(pending, timeout=2)
        return result, cleaned_up.is_set()

    result, cleaned_up = asyncio.run(run())

    assert result.status == "limited"
    assert result.error_code == "cancelled"
    assert cleaned_up is True


def test_run_deadline_cancels_native_async_tool(tmp_path: Path) -> None:
    async def run() -> tuple[object, bool]:
        cleaned_up = asyncio.Event()

        @tool(id="async.slow")
        async def slow() -> dict:
            try:
                await asyncio.Future()
            finally:
                cleaned_up.set()

        adapter = FakeModelAdapter(
            turns=[ModelTurn(tool_calls=(fake_tool_call("async_slow", {}, "c1"),))]
        )
        result = await AgentLoop.from_tools(
            _spec(tmp_path, limits=RunLimits(max_duration_s=1)), adapter, [slow]
        ).arun_once("go")
        return result, cleaned_up.is_set()

    result, cleaned_up = asyncio.run(run())

    assert result.status == "limited"
    assert result.error_code == "run_timeout"
    assert cleaned_up is True


def _block_like_a_stuck_handler() -> None:
    """Block the way a wedged sync handler does: with no way for the run to release it.

    Deliberately not an event the test can set -- releasing the worker would hide whether the run
    was freed on its own. The bound only keeps a regression to a failure instead of a hung session;
    it is far above the ~1s these runs take, so it never fires in a passing run and the daemon
    worker simply outlives the test.
    """

    threading.Event().wait(timeout=20)


def test_run_deadline_abandons_blocking_sync_tool(tmp_path: Path) -> None:
    """A blocking sync handler observes the run deadline like a native async handler.

    The worker cannot be interrupted, so the run abandons it after the grace interval. That has to
    survive ``asyncio.run`` returning, which joins the default executor's workers and would leave
    the deadline enforced internally but unobservable from outside -- hence the liveness assertion.
    """
    workers: list[threading.Thread] = []

    @tool(id="sync.block")
    def block() -> dict:
        workers.append(threading.current_thread())
        _block_like_a_stuck_handler()
        return {"late": True}

    adapter = FakeModelAdapter(
        turns=[ModelTurn(tool_calls=(fake_tool_call("sync_block", {}, "c1"),))]
    )
    result = asyncio.run(
        AgentLoop.from_tools(
            _spec(tmp_path, limits=RunLimits(max_duration_s=1)),
            adapter,
            [block],
            async_tool_cancel_grace_s=0.2,
        ).arun_once("go")
    )

    assert result.status == "limited"
    assert result.error_code == "run_timeout"
    assert workers[0].is_alive() is True


def test_run_cancellation_abandons_blocking_sync_tool(tmp_path: Path) -> None:
    token = CancellationToken()
    started = threading.Event()
    workers: list[threading.Thread] = []

    @tool(id="sync.block_forever")
    def block_forever() -> dict:
        workers.append(threading.current_thread())
        started.set()
        _block_like_a_stuck_handler()
        return {"late": True}

    async def run() -> object:
        adapter = FakeModelAdapter(
            turns=[ModelTurn(tool_calls=(fake_tool_call("sync_block_forever", {}, "c1"),))]
        )
        loop = AgentLoop.from_tools(
            _spec(tmp_path),
            adapter,
            [block_forever],
            cancellation_token=token,
            async_tool_cancel_grace_s=0.2,
        )
        pending = asyncio.create_task(loop.arun_once("go"))
        await asyncio.to_thread(started.wait, 5)
        token.cancel()
        return await asyncio.wait_for(pending, timeout=10)

    result = asyncio.run(run())

    assert result.status == "limited"
    # The run boundary is reported, not a tool failure: ``tool_handler_cancelled`` stays reserved
    # for a handler-local ``CancelledError``.
    assert result.error_code == "cancelled"
    assert workers[0].is_alive() is True


def test_lease_loss_fences_late_artifact_from_abandoned_sync_tool(tmp_path: Path) -> None:
    token = CancellationToken()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    late_errors: list[BaseException] = []
    spec = _spec(tmp_path)
    spec.workspace_root.joinpath("artifact.txt").write_text("late", encoding="utf-8")

    class Provider:
        def get_tools(self, context: ToolContext) -> list[ToolSpec]:
            del context

            def handler(ctx: ToolContext, args: dict) -> ToolResult:
                del args
                started.set()
                assert release.wait(timeout=10)
                try:
                    ctx.emit_artifact("artifact.txt", "text/plain", None, {})
                except BaseException as exc:
                    late_errors.append(exc)
                finally:
                    finished.set()
                return ToolResult(ok=True, content={})

            return [
                ToolSpec(
                    id="sync.late_artifact",
                    description="sync tool that emits after its activation loses the lease",
                    input_schema={"type": "object", "additionalProperties": True},
                    capability="",
                    side_effect="write",
                    handler=handler,
                )
            ]

    loop = AgentLoop(
        spec=spec,
        model_adapter=FakeModelAdapter(
            turns=[ModelTurn(tool_calls=(fake_tool_call("sync_late_artifact", {}, "c1"),))]
        ),
        runtime_config_provider=runtime_provider(
            runtime_config(bindings=(tool_binding("sync.late_artifact"),))
        ),
        tool_providers=(Provider(),),
        cancellation_token=token,
        async_tool_cancel_grace_s=0.05,
    )
    loop.open()
    assert loop._session is not None
    recorder = loop._session.res.recorder

    async def drive() -> object:
        pending = asyncio.create_task(loop.arun_until_suspended("go"))
        assert await asyncio.to_thread(started.wait, 5)
        loop.lose_writer_authority()
        suspension = await asyncio.wait_for(pending, timeout=10)
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        return suspension

    try:
        suspension = asyncio.run(drive())
        assert suspension.interruption_cause is InterruptionCause.LEASE_LOST
        assert len(late_errors) == 1
        assert isinstance(late_errors[0], RunCancelled)
        assert not list(recorder.artifacts_dir.glob("artifact_*"))
        assert recorder.artifacts == []
    finally:
        loop.discard_uncommitted()


@pytest.mark.parametrize("mutation", ("plan", "finish"))
def test_lease_loss_fences_context_mutation_from_abandoned_sync_tool(
    tmp_path: Path,
    mutation: str,
) -> None:
    token = CancellationToken()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    late_errors: list[BaseException] = []

    class Provider:
        def get_tools(self, context: ToolContext) -> list[ToolSpec]:
            del context

            def handler(ctx: ToolContext, args: dict) -> ToolResult:
                del args
                started.set()
                assert release.wait(timeout=10)
                try:
                    if mutation == "plan":
                        ctx.update_plan([{"step": "late", "status": "pending"}])
                    else:
                        ctx.finish("late", [], None)
                except BaseException as exc:
                    late_errors.append(exc)
                finally:
                    finished.set()
                return ToolResult(ok=True, content={})

            return [
                ToolSpec(
                    id="sync.late_context_mutation",
                    description="sync tool that mutates context after lease loss",
                    input_schema={"type": "object", "additionalProperties": True},
                    capability="",
                    side_effect="write",
                    handler=handler,
                )
            ]

    loop = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=FakeModelAdapter(
            turns=[
                ModelTurn(
                    tool_calls=(fake_tool_call("sync_late_context_mutation", {}, "c1"),)
                )
            ]
        ),
        runtime_config_provider=runtime_provider(
            runtime_config(bindings=(tool_binding("sync.late_context_mutation"),))
        ),
        tool_providers=(Provider(),),
        cancellation_token=token,
        async_tool_cancel_grace_s=0.05,
    )
    loop.open()
    assert loop._session is not None
    context = loop._session.res.context

    async def drive() -> object:
        pending = asyncio.create_task(loop.arun_until_suspended("go"))
        assert await asyncio.to_thread(started.wait, 5)
        loop.lose_writer_authority()
        suspension = await asyncio.wait_for(pending, timeout=10)
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        return suspension

    try:
        suspension = asyncio.run(drive())
        assert suspension.interruption_cause is InterruptionCause.LEASE_LOST
        assert len(late_errors) == 1
        assert isinstance(late_errors[0], RunCancelled)
        assert context.plan == []
        assert context.pending_finish is None
        assert not loop.spec.workspace_root.joinpath("late.txt").exists()
    finally:
        loop.discard_uncommitted()


def test_extension_context_exposes_methods_without_mutable_engine_capabilities(
    tmp_path: Path,
) -> None:
    forbidden = {
        "workspace",
        "recorder",
        "job_manager",
        "outbox",
        "plan",
        "pending_finish",
        "subagent_usage",
        "skills_activated",
    }
    provider_leaks: list[str] = []
    handler_leaks: list[str] = []

    class Provider:
        def get_tools(self, context: ToolContext) -> list[ToolSpec]:
            provider_leaks.extend(name for name in forbidden if hasattr(context, name))

            def handler(ctx: ToolContext, args: dict) -> ToolResult:
                del args
                handler_leaks.extend(name for name in forbidden if hasattr(ctx, name))
                ctx.update_plan([{"step": "safe façade", "status": "completed"}])
                return ToolResult(ok=True, content={})

            return [
                ToolSpec(
                    id="context.inspect",
                    description="inspect the extension context boundary",
                    input_schema={"type": "object", "additionalProperties": True},
                    capability="",
                    side_effect="read",
                    handler=handler,
                )
            ]

    loop = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=FakeModelAdapter(
            turns=[
                ModelTurn(tool_calls=(fake_tool_call("context_inspect", {}, "c1"),)),
                ModelTurn(final_text="done"),
            ]
        ),
        runtime_config_provider=runtime_provider(
            runtime_config(bindings=(tool_binding("context.inspect"),))
        ),
        tool_providers=(Provider(),),
    )
    loop.open()
    assert loop._session is not None
    workspace = loop._session.res.workspace

    try:
        assert not hasattr(workspace, "root")
        assert not hasattr(workspace, "resolve_existing_or_parent")
        assert loop.run_until_suspended("go").reason == "settled"
        loop.close()
    finally:
        if loop._session is not None:
            loop.discard_uncommitted()

    assert provider_leaks == []
    assert handler_leaks == []


def test_retained_builtin_workspace_handler_is_fenced_without_exporting_a_path(
    tmp_path: Path,
) -> None:
    loop = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=FakeModelAdapter(turns=[]),
        runtime_config_provider=runtime_provider(runtime_config("fs.write")),
    )
    loop.open()
    assert loop._session is not None
    resources = loop._session.res
    write_tool = next(spec for spec in resources.base_tool_specs if spec.id == "fs.write")
    loop.lose_writer_authority()

    try:
        with pytest.raises(RunCancelled):
            write_tool.handler(
                resources.context._extension_context,
                {"path": "late.txt", "content": "stale"},
            )
        assert not loop.spec.workspace_root.joinpath("late.txt").exists()
    finally:
        loop.discard_uncommitted()


def test_external_effect_may_finish_after_revoke_but_its_result_is_not_published(
    tmp_path: Path,
) -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    external_effect: list[str] = []

    class Provider:
        def get_tools(self, context: ToolContext) -> list[ToolSpec]:
            del context

            def handler(ctx: ToolContext, args: dict) -> ToolResult:
                del ctx, args
                external_effect.append("started")
                started.set()
                assert release.wait(timeout=10)
                external_effect.append("completed")
                finished.set()
                return ToolResult(ok=True, content={"external": "completed"})

            return [
                ToolSpec(
                    id="sync.external_effect",
                    description="external effect that can outlive its activation",
                    input_schema={"type": "object", "additionalProperties": True},
                    capability="",
                    side_effect="write",
                    handler=handler,
                )
            ]

    loop = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=FakeModelAdapter(
            turns=[ModelTurn(tool_calls=(fake_tool_call("sync_external_effect", {}, "c1"),))]
        ),
        runtime_config_provider=runtime_provider(
            runtime_config(bindings=(tool_binding("sync.external_effect"),))
        ),
        tool_providers=(Provider(),),
        async_tool_cancel_grace_s=0.05,
    )
    loop.open()
    assert loop._session is not None
    recorder = loop._session.res.recorder

    async def drive() -> object:
        pending = asyncio.create_task(loop.arun_until_suspended("go"))
        assert await asyncio.to_thread(started.wait, 5)
        loop.lose_writer_authority()
        suspension = await asyncio.wait_for(pending, timeout=10)
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        return suspension

    try:
        suspension = asyncio.run(drive())
        event_types = {
            json.loads(line)["type"]
            for line in recorder.run_dir.joinpath("events.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        }
        assert suspension.interruption_cause is InterruptionCause.LEASE_LOST
        assert external_effect == ["started", "completed"]
        assert "tool.call.finished" not in event_types
        assert "tool.call.failed" not in event_types
        assert LocalFsCheckpointStore(loop.spec.run_root).latest(loop.spec.run_id) is None
    finally:
        loop.discard_uncommitted()


def test_sync_tool_finishing_within_the_grace_is_not_abandoned(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The cancel grace applies to the worker thread, not just to its cancellable waiter.

    Cancelling a sync call's waiter completes it instantly -- there is no coroutine to throw
    ``CancelledError`` into -- so waiting on the waiter after cancelling it would grant no grace at
    all and abandon every in-flight handler on the spot. The grace exists so a handler that is
    about to finish lands its writes before the run finalizes instead of racing it.

    The run still reports ``run_timeout``: the grace is not an extension of the deadline. What it
    buys is a settled worker and no abandonment.
    """
    workers: list[threading.Thread] = []

    @tool(id="sync.almost_done")
    def almost_done() -> dict:
        workers.append(threading.current_thread())
        # Outlasts the run deadline, finishes well inside the grace below.
        time.sleep(1.3)
        return {"late": True}

    adapter = FakeModelAdapter(
        turns=[ModelTurn(tool_calls=(fake_tool_call("sync_almost_done", {}, "c1"),))]
    )
    with caplog.at_level(logging.WARNING, logger="monoid_agent_kernel.core.sync_bridge"):
        result = asyncio.run(
            AgentLoop.from_tools(
                _spec(tmp_path, limits=RunLimits(max_duration_s=1)),
                adapter,
                [almost_done],
                async_tool_cancel_grace_s=5.0,
            ).arun_once("go")
        )

    assert result.status == "limited"
    assert result.error_code == "run_timeout"
    assert workers[0].is_alive() is False
    assert [
        record for record in caplog.records if "abandoned a synchronous call" in record.message
    ] == []


def test_late_task_from_abandoned_sync_tool_is_cancelled() -> None:
    """A late awaitable that is a task or future is disposed, not only a coroutine.

    The normal path accepts any awaitable a sync handler returns, so the late path has to handle the
    same shapes. Left alone in a persistent backend loop, a returned task keeps running after the
    run was cancelled, and a future that completes with an exception is never consumed.
    """
    ran: list[str] = []

    async def scenario() -> asyncio.Future[None]:
        loop = asyncio.get_running_loop()

        async def keeps_running() -> None:
            await asyncio.sleep(0.3)
            ran.append("finished")

        task = loop.create_task(keeps_running())
        release = threading.Event()

        def call() -> asyncio.Future[None]:
            release.wait(timeout=5)
            return task

        pending = start_abandonable_sync_call(call, thread_name="nar-test-late-task")
        pending.result.cancel()  # the run gave up on the call
        release.set()  # ... and only now does the handler return its task
        await asyncio.wait({pending.settled}, timeout=5)
        await asyncio.sleep(0.5)  # long enough for the task to have finished, had it survived
        return task

    task = asyncio.run(scenario())

    assert task.cancelled() is True
    assert ran == []


def test_sync_tool_handler_sees_caller_context_variables(tmp_path: Path) -> None:
    """A sync handler runs in a copy of the caller's context, as ``asyncio.to_thread`` did.

    Hosts put request-scoped state -- credentials, tenant identity, tracing -- in ``ContextVar``s, so
    a worker started with an empty context would silently read defaults.
    """
    seen: list[str] = []

    @tool(id="sync.tenant")
    def read_tenant() -> dict:
        seen.append(_TENANT.get())
        return {"tenant": _TENANT.get()}

    async def run() -> object:
        _TENANT.set("acme")
        adapter = FakeModelAdapter(
            turns=[
                ModelTurn(tool_calls=(fake_tool_call("sync_tenant", {}, "c1"),)),
                ModelTurn(final_text="done"),
            ]
        )
        return await AgentLoop.from_tools(_spec(tmp_path), adapter, [read_tenant]).arun_once("go")

    result = asyncio.run(run())

    assert result.status == "completed"
    assert seen == ["acme"]


def test_abandoned_sync_tool_keeps_its_call_authorization(tmp_path: Path) -> None:
    """An abandoned handler keeps the authorization of the call it was invoked for.

    The run clears the current call in a ``finally`` as soon as it stops waiting, but the worker is
    still running. Were that state shared, the worker's later ``path_allowed`` / shell / web calls
    would read an empty scope -- and an empty scope applies no allow-list or deny-list narrowing at
    all, so the abandoned worker would widen to the run-level permission policy.
    """
    workers: list[threading.Thread] = []
    late_scope_results: list[tuple[bool, bool]] = []

    class Provider:
        def get_tools(self, context: ToolContext) -> list[ToolSpec]:
            del context

            def handler(ctx: ToolContext, args: dict) -> ToolResult:
                del args
                workers.append(threading.current_thread())
                # Outlive the deadline and the grace window, then exercise the retained scope.
                threading.Event().wait(timeout=1.5)
                late_scope_results.append(
                    (
                        ctx.path_allowed("notes/kept.txt", "read"),
                        ctx.path_allowed("secrets/key.txt", "read"),
                    )
                )
                return ToolResult(ok=True, content={})

            return [
                ToolSpec(
                    id="sync.late_scope",
                    description="sync tool that outlives its run",
                    input_schema={"type": "object", "additionalProperties": True},
                    capability="",
                    side_effect="read",
                    handler=handler,
                )
            ]

    result = asyncio.run(
        AgentLoop(
            spec=_spec(tmp_path, limits=RunLimits(max_duration_s=1)),
            model_adapter=FakeModelAdapter(
                turns=[ModelTurn(tool_calls=(fake_tool_call("sync_late_scope", {}, "c1"),))]
            ),
            runtime_config_provider=runtime_provider(
                runtime_config(
                    bindings=(
                        tool_binding(
                            "sync.late_scope", scope=ToolScope(allowed_paths=("notes/*",))
                        ),
                    )
                )
            ),
            tool_providers=(Provider(),),
            async_tool_cancel_grace_s=0.05,
        ).arun_once("go")
    )

    assert result.status == "limited"
    assert result.error_code == "run_timeout"
    workers[0].join(timeout=10)
    assert late_scope_results == [(True, False)]


def test_failed_tool_outcome_is_consumed_when_a_run_boundary_wins_the_turn(tmp_path: Path) -> None:
    """A handler that fails in the same loop turn a run boundary becomes observable is consumed.

    ``_check_run_boundary`` runs before anything reads the outcome, so it raises the run-level error
    first. The future is already done by then, so the detach path is skipped too -- leaving the
    handler's exception unretrieved, which asyncio reports as "Future exception was never retrieved"
    when the future is collected. Nothing downstream will read it, so this boundary has to.
    """
    token = CancellationToken()
    token.cancel()
    loop = AgentLoop.from_tools(
        _spec(tmp_path), FakeModelAdapter(turns=[]), [], cancellation_token=token
    )

    async def scenario() -> asyncio.Future[ToolResult]:
        failed: asyncio.Future[ToolResult] = asyncio.get_running_loop().create_future()
        failed.set_exception(RuntimeError("handler blew up"))
        with pytest.raises(RunCancelled):
            await loop._await_native_tool_handler(failed, None)
        return failed

    failed = asyncio.run(scenario())

    # ``_log_traceback`` is the flag that drives the warning: asyncio sets it on ``set_exception``
    # and clears it once the exception is retrieved.
    assert failed._log_traceback is False  # type: ignore[attr-defined]


def test_sync_tool_child_thread_keeps_the_call_authorization(tmp_path: Path) -> None:
    """A thread the handler starts itself is still bound by the call's scope.

    Delegating a ``ToolContext`` operation to a joined child thread is a normal handler shape. A new
    ``threading.Thread`` starts with an empty context, so a ``ContextVar`` alone reads unset there --
    and an empty scope applies no allow-list narrowing at all, so the child would widen to the
    run-level permission policy while its parent is still under a restricted authorization.
    """
    seen: list[bool] = []

    class Provider:
        def get_tools(self, context: ToolContext) -> list[ToolSpec]:
            del context

            def handler(ctx: ToolContext, args: dict) -> ToolResult:
                del args

                def in_child_thread() -> None:
                    # Outside the binding's allowed_paths, so a scoped call must refuse it.
                    seen.append(ctx.path_allowed("secrets/key.txt", "read"))

                child = threading.Thread(target=in_child_thread)
                child.start()
                child.join(timeout=5)
                return ToolResult(ok=True, content={})

            return [
                ToolSpec(
                    id="sync.child_thread",
                    description="sync tool that delegates to a joined child thread",
                    input_schema={"type": "object", "additionalProperties": True},
                    capability="",
                    side_effect="read",
                    handler=handler,
                )
            ]

    result = asyncio.run(
        AgentLoop(
            spec=_spec(tmp_path),
            model_adapter=FakeModelAdapter(
                turns=[
                    ModelTurn(tool_calls=(fake_tool_call("sync_child_thread", {}, "c1"),)),
                    ModelTurn(final_text="done"),
                ]
            ),
            runtime_config_provider=runtime_provider(
                runtime_config(
                    bindings=(
                        tool_binding(
                            "sync.child_thread", scope=ToolScope(allowed_paths=("notes/*",))
                        ),
                    )
                )
            ),
            tool_providers=(Provider(),),
        ).arun_once("go")
    )

    assert result.status == "completed"
    assert seen == [False]


def test_child_thread_of_an_abandoned_handler_is_refused_not_widened(tmp_path: Path) -> None:
    """Once the run has given up on a handler, its descendant threads lose authorization.

    The fallback that carries a call's scope into a handler-spawned thread is a single shared slot,
    and a descendant thread cannot be told apart from the live call's own child -- Python exposes no
    thread-to-creator link. So the fallback is cleared when the call ends, and an absent call now
    *refuses* scoped operations. Returning an empty ``CallContext`` there would read as "this call
    narrows nothing", handing the descendant the full run-level permission policy: a strictly wider
    authorization than its parent ever held.
    """
    seen: list[bool] = []
    child_started = threading.Event()
    release_child = threading.Event()
    child_finished = threading.Event()

    class Provider:
        def get_tools(self, context: ToolContext) -> list[ToolSpec]:
            del context

            def handler(ctx: ToolContext, args: dict) -> ToolResult:
                del args

                def in_child_thread() -> None:
                    child_started.set()
                    release_child.wait(timeout=10)
                    # Allowed by the run-level permission policy, denied by the binding's scope --
                    # so ``True`` here would mean the scope check was skipped entirely.
                    seen.append(ctx.path_allowed("secrets/key.txt", "read"))
                    child_finished.set()

                threading.Thread(target=in_child_thread, daemon=True).start()
                child_started.wait(timeout=10)
                # Outlive the deadline and the grace, so the run abandons this worker.
                threading.Event().wait(timeout=10)
                return ToolResult(ok=True, content={})  # pragma: no cover - never reached

            return [
                ToolSpec(
                    id="sync.abandoned_parent",
                    description="sync tool abandoned while a child thread is still parked",
                    input_schema={"type": "object", "additionalProperties": True},
                    capability="",
                    side_effect="read",
                    handler=handler,
                )
            ]

    result = asyncio.run(
        AgentLoop(
            spec=_spec(tmp_path, limits=RunLimits(max_duration_s=1)),
            model_adapter=FakeModelAdapter(
                turns=[ModelTurn(tool_calls=(fake_tool_call("sync_abandoned_parent", {}, "c1"),))]
            ),
            runtime_config_provider=runtime_provider(
                runtime_config(
                    bindings=(
                        tool_binding(
                            "sync.abandoned_parent", scope=ToolScope(allowed_paths=("notes/*",))
                        ),
                    )
                )
            ),
            tool_providers=(Provider(),),
            async_tool_cancel_grace_s=0.05,
        ).arun_once("go")
    )

    assert result.error_code == "run_timeout"
    release_child.set()
    assert child_finished.wait(timeout=10) is True
    assert seen == [False]


def _late_awaitable_provider(workers: list[threading.Thread], returned: list[object]) -> type:
    """A raw-``ToolSpec`` provider whose sync handler returns an awaitable only after its run has
    given up on it. Raw, because that is the shape that can return an awaitable at all -- the
    ``@tool`` decorator wraps whatever the function returns in a ``ToolResult``."""

    class Provider:
        def get_tools(self, context: ToolContext) -> list[ToolSpec]:
            del context

            async def late_result() -> ToolResult:
                # pragma: no cover - abandoned before anything awaits it
                return ToolResult(ok=True, content={"late": True})

            def handler(_ctx: ToolContext, args: dict) -> ToolResult:
                del args
                workers.append(threading.current_thread())
                threading.Event().wait(timeout=1.5)
                coroutine = late_result()
                returned.append(coroutine)
                return coroutine  # type: ignore[return-value]

            return [
                ToolSpec(
                    id="sync.late_awaitable",
                    description="sync tool returning an awaitable",
                    input_schema={"type": "object", "additionalProperties": True},
                    capability="",
                    side_effect="read",
                    handler=handler,
                )
            ]

    return Provider


def _late_awaitable_loop(tmp_path: Path, provider: type) -> AgentLoop:
    return AgentLoop(
        spec=_spec(tmp_path, limits=RunLimits(max_duration_s=1)),
        model_adapter=FakeModelAdapter(
            turns=[ModelTurn(tool_calls=(fake_tool_call("sync_late_awaitable", {}, "c1"),))]
        ),
        runtime_config_provider=runtime_provider(
            runtime_config(bindings=(tool_binding("sync.late_awaitable"),))
        ),
        tool_providers=(provider(),),
        async_tool_cancel_grace_s=0.05,
    )


def test_awaitable_from_abandoned_sync_tool_is_closed(tmp_path: Path) -> None:
    """A sync handler may return an awaitable. If the run abandoned the call before it did, that
    awaitable is closed rather than left for the collector to warn about.

    Here the loop is still running when it arrives -- the deployed shape, as the reference backend
    keeps one loop across many runs.
    """
    workers: list[threading.Thread] = []
    returned: list[object] = []

    async def run() -> object:
        result = await _late_awaitable_loop(
            tmp_path, _late_awaitable_provider(workers, returned)
        ).arun_once("go")
        # Wait on the abandoned worker without blocking the loop, then let its queued callback run.
        await asyncio.to_thread(workers[0].join, 10)
        await asyncio.sleep(0)
        return result

    result = asyncio.run(run())

    assert result.status == "limited"
    assert result.error_code == "run_timeout"
    assert inspect.getcoroutinestate(returned[0]) == inspect.CORO_CLOSED


def test_awaitable_from_abandoned_sync_tool_is_closed_after_loop_shutdown(tmp_path: Path) -> None:
    """The same awaitable, arriving after ``asyncio.run`` closed the loop.

    Delivery cannot be scheduled at all then, so the close has to happen on the worker's own thread
    instead -- otherwise this is exactly the case that leaks an unawaited coroutine.
    """
    workers: list[threading.Thread] = []
    returned: list[object] = []

    result = asyncio.run(
        _late_awaitable_loop(tmp_path, _late_awaitable_provider(workers, returned)).arun_once("go")
    )

    assert result.status == "limited"
    assert result.error_code == "run_timeout"
    workers[0].join(timeout=10)
    assert inspect.getcoroutinestate(returned[0]) == inspect.CORO_CLOSED


def test_late_failed_future_from_abandoned_sync_call_is_consumed_after_loop_shutdown() -> None:
    """A settled future arriving after the loop closed still has its outcome read.

    Cancelling a future needs the loop, so the closed-loop path cannot dispose a *pending* one -- but
    such a future can no longer run either, so there is nothing to stop and no outcome to read.
    A future that already carries an exception is the case that matters: reading it touches no loop,
    and leaving it unread is precisely what asyncio reports at collection.
    """
    release = threading.Event()
    holder: list[asyncio.Future[None]] = []

    async def scenario() -> None:
        failed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        failed.set_exception(RuntimeError("late failure"))
        holder.append(failed)

        def call() -> asyncio.Future[None]:
            release.wait(timeout=5)
            return failed

        start_abandonable_sync_call(call, thread_name="nar-test-late-failed").result.cancel()

    asyncio.run(scenario())  # the loop is closed once this returns
    release.set()  # ... and only now does the call return its future

    failed = holder[0]
    deadline = time.time() + 5
    # ``_log_traceback`` is the flag that drives the warning: set on ``set_exception``, cleared once
    # the exception is retrieved.
    while time.time() < deadline and failed._log_traceback:  # type: ignore[attr-defined]
        time.sleep(0.02)
    assert failed._log_traceback is False  # type: ignore[attr-defined]


def test_stubborn_async_tool_cleanup_cannot_block_run_cancellation(tmp_path: Path) -> None:
    token = CancellationToken()

    async def run() -> tuple[object, bool]:
        started = asyncio.Event()
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        cleanup_finished = asyncio.Event()

        @tool(id="async.stubborn")
        async def stubborn() -> dict:
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cleanup_started.set()
                await release_cleanup.wait()
                cleanup_finished.set()
                return {"late": True}

        adapter = FakeModelAdapter(
            turns=[ModelTurn(tool_calls=(fake_tool_call("async_stubborn", {}, "c1"),))]
        )
        loop = AgentLoop.from_tools(
            _spec(tmp_path),
            adapter,
            [stubborn],
            cancellation_token=token,
            async_tool_cancel_grace_s=0.01,
        )
        pending = asyncio.create_task(loop.arun_once("go"))
        await started.wait()
        token.cancel()
        result = await asyncio.wait_for(pending, timeout=1)
        assert cleanup_started.is_set()
        assert not cleanup_finished.is_set()
        release_cleanup.set()
        await asyncio.wait_for(cleanup_finished.wait(), timeout=1)
        return result, cleanup_finished.is_set()

    result, cleanup_finished = asyncio.run(run())

    assert result.status == "limited"
    assert result.error_code == "cancelled"
    assert cleanup_finished is True


def test_async_tool_preserves_capability_gate_and_token_context(tmp_path: Path) -> None:
    seen_tokens: list[str | None] = []

    class Provider:
        def get_tools(self, context: ToolContext) -> list[ToolSpec]:
            del context

            async def handler(ctx: ToolContext, args: dict) -> ToolResult:
                await asyncio.sleep(0)
                seen_tokens.append(ctx.capability_token("demo.secure"))
                return ToolResult(ok=True, content={"value": args["value"]})

            return [
                ToolSpec(
                    id="async.secure",
                    description="secure async tool",
                    input_schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                    capability="demo.secure",
                    side_effect="read",
                    handler=handler,
                )
            ]

    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("async_secure", {"value": "ok"}, "c1"),)),
            ModelTurn(final_text="done"),
        ]
    )
    config = runtime_config(
        bindings=(tool_binding("async.secure", runtime={"requires_lease": True}),)
    )
    loop = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
        tool_providers=(Provider(),),
        capability_broker=AutoGrantBroker(),
    )

    result = asyncio.run(loop.arun_once("go"))

    assert result.status == "completed"
    assert seen_tokens and seen_tokens[0]


def test_approved_async_tool_replay_executes_once(tmp_path: Path) -> None:
    calls = 0

    class Provider:
        def get_tools(self, context: ToolContext) -> list[ToolSpec]:
            del context

            async def handler(_ctx: ToolContext, args: dict) -> ToolResult:
                nonlocal calls
                await asyncio.sleep(0)
                calls += 1
                return ToolResult(ok=True, content={"value": args["value"]})

            return [
                ToolSpec(
                    id="async.approval",
                    description="approved async tool",
                    input_schema={"type": "object", "additionalProperties": True},
                    capability="",
                    side_effect="write",
                    handler=handler,
                )
            ]

    async def run() -> tuple[object, object]:
        adapter = FakeModelAdapter(
            turns=[
                ModelTurn(tool_calls=(fake_tool_call("async_approval", {"value": "ok"}, "c1"),)),
                ModelTurn(final_text="park"),
                ModelTurn(final_text="done"),
            ]
        )
        config = runtime_config(bindings=(tool_binding("async.approval", authorization="ask"),))
        loop = AgentLoop(
            spec=_spec(tmp_path),
            model_adapter=adapter,
            runtime_config_provider=runtime_provider(config),
            tool_providers=(Provider(),),
        )
        await loop.aopen()
        parked = await loop.arun_until_suspended("go")
        loop.report_task_result(parked.awaiting_task_ids[0], {"approved": True})
        resumed = await loop.arun_until_suspended(None)
        result = await loop.aclose()
        return resumed, result

    resumed, result = asyncio.run(run())

    assert resumed.reason == "settled"
    assert result.status == "completed"
    assert calls == 1
    events = _event_types(result.run_dir)
    assert events.index("tool.approval.requested") < events.index("tool.call.finished")


def test_capability_grant_replays_async_tool_once(tmp_path: Path) -> None:
    calls = 0

    class Provider:
        def get_tools(self, context: ToolContext) -> list[ToolSpec]:
            del context

            async def handler(ctx: ToolContext, _args: dict) -> ToolResult:
                nonlocal calls
                await asyncio.sleep(0)
                calls += 1
                return ToolResult(
                    ok=True,
                    content={"token_ref": ctx.capability_token("demo.secure")},
                )

            return [
                ToolSpec(
                    id="async.secure",
                    description="secure async tool",
                    input_schema={"type": "object"},
                    capability="demo.secure",
                    side_effect="read",
                    handler=handler,
                )
            ]

    async def run() -> tuple[object, object]:
        adapter = FakeModelAdapter(
            turns=[
                ModelTurn(tool_calls=(fake_tool_call("async_secure", {}, "c1"),)),
                ModelTurn(final_text="park"),
                ModelTurn(final_text="done"),
            ]
        )
        config = runtime_config(
            bindings=(tool_binding("async.secure", runtime={"requires_lease": True}),)
        )
        loop = AgentLoop(
            spec=_spec(tmp_path),
            model_adapter=adapter,
            runtime_config_provider=runtime_provider(config),
            tool_providers=(Provider(),),
            capability_broker=HumanEscalationBroker(),
        )
        await loop.aopen()
        parked = await loop.arun_until_suspended("go")
        lease = CapabilityLease(
            capability="demo.secure",
            token_ref="approved:demo.secure",
            expires_at=time.time() + 60,
            durable=True,
        )
        loop.report_task_result(
            parked.awaiting_task_ids[0],
            {"granted": True, "lease": lease.to_json()},
        )
        resumed = await loop.arun_until_suspended(None)
        result = await loop.aclose()
        return resumed, result

    resumed, result = asyncio.run(run())

    assert resumed.reason == "settled"
    assert result.status == "completed"
    assert calls == 1
    events = _event_types(result.run_dir)
    replay_finished = len(events) - 1 - events[::-1].index("tool.call.finished")
    assert events.index("capability.granted") < replay_finished


def test_the_disposal_rule_handles_every_shape_it_claims_to() -> None:
    """The rule the abandonment path and the replay wrapper now share, driven directly.

    It was a closure over one caller's outcome tuple, so the second caller that needed it --
    a synchronous wrapper refusing an awaitable its inner handed back -- could not reach it
    and would have grown a twin. Extracted, it needs its own gate: a closure that only ever
    ran behind an abandonment had no test that named it.
    """

    async def _never_awaited() -> None:  # pragma: no cover - closed before it runs
        raise AssertionError("the coroutine body must never run")

    coroutine = _never_awaited()
    dispose_unawaited(coroutine)
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED

    async def _futures() -> None:
        loop = asyncio.get_running_loop()
        pending: asyncio.Future[None] = loop.create_future()
        dispose_unawaited(pending)
        assert pending.cancelled(), "a live-loop future is cancelled so it stops running"

        settled: asyncio.Future[None] = loop.create_future()
        settled.set_exception(RuntimeError("nobody read this"))
        dispose_unawaited(settled, on_live_loop=False)
        # Consumed rather than skipped: an unretrieved exception is what warns at collection,
        # and reading a settled future's outcome touches no loop.
        assert settled.exception() is not None

        dead: asyncio.Future[None] = loop.create_future()
        dispose_unawaited(dead, on_live_loop=False)
        assert not dead.cancelled(), "a pending future on a closed loop is left alone"
        dead.cancel()

    asyncio.run(_futures())

    class _ExoticAwaitable:
        def __await__(self):  # pragma: no cover - never awaited
            yield

    exotic = _ExoticAwaitable()
    dispose_unawaited(exotic)  # no generic disposal exists; leaving it alone is the contract
    assert inspect.isawaitable(exotic)
