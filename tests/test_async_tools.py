from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

from support.runtime import runtime_config, runtime_provider, tool_binding

from monoid_agent_kernel import tool
from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.capability import AutoGrantBroker, CapabilityLease
from monoid_agent_kernel.core.spec import AgentRunSpec, RunLimits
from monoid_agent_kernel.errors import ToolExecutionError
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
