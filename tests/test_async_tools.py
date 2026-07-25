from __future__ import annotations

import asyncio
import inspect
import json
import logging
import threading
import time
from contextvars import ContextVar
from pathlib import Path

import pytest

from support.runtime import runtime_config, runtime_provider, tool_binding

from monoid_agent_kernel import tool
from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.capability import AutoGrantBroker, CapabilityLease
from monoid_agent_kernel.core.spec import AgentRunSpec, RunLimits
from monoid_agent_kernel.errors import ToolExecutionError
from monoid_agent_kernel.loop import AgentLoop, _start_abandonable_sync_call
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
    with caplog.at_level(logging.WARNING, logger="monoid_agent_kernel.loop"):
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
    assert [record for record in caplog.records if "abandoned a synchronous call" in record.message] == []


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

        pending = _start_abandonable_sync_call(call, thread_name="nar-test-late-task")
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
    late_tool_id: list[str] = []

    class Provider:
        def get_tools(self, context: ToolContext) -> list[ToolSpec]:
            del context

            def handler(ctx: ToolContext, args: dict) -> ToolResult:
                del args
                workers.append(threading.current_thread())
                # Outlive the deadline and the grace window, then read the call back.
                threading.Event().wait(timeout=1.5)
                late_tool_id.append(ctx._current_call.tool_id)  # type: ignore[attr-defined]
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
                runtime_config(bindings=(tool_binding("sync.late_scope"),))
            ),
            tool_providers=(Provider(),),
            async_tool_cancel_grace_s=0.05,
        ).arun_once("go")
    )

    assert result.status == "limited"
    assert result.error_code == "run_timeout"
    workers[0].join(timeout=10)
    assert late_tool_id == ["sync.late_scope"]


def _late_awaitable_provider(
    workers: list[threading.Thread], returned: list[object]
) -> type:
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
