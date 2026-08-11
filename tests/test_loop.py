from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from support.runtime import runtime_config, runtime_provider, tool_binding

from monoid_agent_kernel.core.agents import AgentRuntimeConfig
from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.checkpoint import (
    CheckpointRecord,
    CheckpointStore,
    LocalFsCheckpointStore,
    RunCheckpoint,
)
from monoid_agent_kernel.core.schemas import validate_run_dir
from monoid_agent_kernel.core.spec import AgentRunSpec, RunLimits
from monoid_agent_kernel.core.tool_surface import ToolQuota
from monoid_agent_kernel.errors import (
    ModelAdapterError,
    ModelCallAborted,
    RunCancelled,
    RunTimeout,
)
from monoid_agent_kernel.loop import AgentLoop, _recoverable_turn_error
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.providers.base import ModelTurn, ReasoningDelta, TextDelta, TurnComplete
from monoid_agent_kernel.providers.fake import (
    FakeModelAdapter,
    FakeStreamingModelAdapter,
    fake_tool_call,
)
from monoid_agent_kernel.tools.base import ToolContext, ToolResult, ToolSpec
from monoid_agent_kernel.recorder import MemoryEventSink
from monoid_agent_kernel.workspace.local import default_local_workspace_factory, sha256_bytes
from support.process import python_command as _python_command


DEFAULT_TOOLS = (
    "fs.read",
    "fs.write",
    "fs.patch",
    "fs.list",
    "fs.tree",
    "fs.stat",
    "fs.glob",
    "fs.copy",
    "fs.move",
    "fs.delete",
    "run.finish",
)


def _provider(*tool_ids: str):
    return runtime_provider(runtime_config(*(tool_ids or DEFAULT_TOOLS)))


def _finish_only_adapter() -> FakeModelAdapter:
    return FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("run_finish", {"summary": "done"}, "call_finish"),),
            ),
        ]
    )


def test_message_log_cap_settles_run_as_limited(tmp_path: Path) -> None:
    # A by-value conversation log that outgrows max_message_log_bytes settles the run as
    # ``limited`` (a safe stop, not a drop) before the over-limit log is ever sent.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = _finish_only_adapter()
    spec = AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        limits=RunLimits(max_message_log_bytes=10),
    )

    result = AgentLoop(
        spec=spec, model_adapter=adapter, runtime_config_provider=_provider("run.finish")
    ).run_once("This instruction is clearly longer than ten bytes.")

    assert result.status == "limited"
    assert result.error_code == "message_log_bytes_exceeded"
    assert adapter.requests == []  # the over-limit log is never sent to the model


def test_workspace_delta_cap_settles_run_as_limited(tmp_path: Path) -> None:
    # A workspace delta that outgrows the cap settles the run ``limited`` at the next
    # turn's start, before the over-cap delta is persisted into a checkpoint.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call("fs_write", {"path": "big.txt", "content": "x" * 50}, "c1"),
                ),
            ),
            ModelTurn(
                response_id="r2",
                tool_calls=(fake_tool_call("run_finish", {"summary": "done"}, "c2"),),
            ),
        ]
    )
    spec = AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        limits=RunLimits(max_delta_file_bytes=10),
    )

    result = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=_provider("fs.write", "run.finish"),
    ).run_once("write a big file")

    assert result.status == "limited"
    assert result.error_code == "workspace_delta_file_bytes_exceeded"
    assert len(adapter.requests) == 1  # turn 2 is never sent (settled at its start)


def test_default_system_prompt_is_composed_base(tmp_path: Path) -> None:
    from monoid_agent_kernel.core.prompt import compose_system_prompt

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = _finish_only_adapter()
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")

    AgentLoop(
        spec=spec, model_adapter=adapter, runtime_config_provider=_provider("run.finish")
    ).run_once("Inspect.")

    assert adapter.requests[0].system_prompt == compose_system_prompt()


def test_run_finish_surfaces_outputs_and_notes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("rough notes\n", encoding="utf-8")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call(
                        "run_finish",
                        {
                            "summary": "Reviewed the notes",
                            "outputs": ["notes.md", "SUMMARY.md"],
                            "notes": "No changes were necessary.",
                        },
                        "call_finish",
                    ),
                ),
            ),
        ]
    )

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=_provider("run.finish"),
    ).run_once("Review.")

    assert result.status == "completed"
    assert result.final_text == "Reviewed the notes"
    assert result.final_outputs == ("notes.md", "SUMMARY.md")
    assert result.final_notes == "No changes were necessary."


def test_loop_read_write_finish_happy_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("rough notes\n", encoding="utf-8")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call("fs_read", {"path": "notes.md"}, "call_read"),
                    fake_tool_call(
                        "fs_write",
                        {"path": "SUMMARY.md", "content": "Clean summary\n", "create_dirs": False},
                        "call_write",
                    ),
                ),
            ),
            ModelTurn(
                response_id="r2",
                tool_calls=(
                    fake_tool_call("run_finish", {"summary": "Created SUMMARY.md"}, "call_finish"),
                ),
            ),
        ]
    )

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=_provider(),
    ).run_once("Clean.")

    assert result.status == "completed"
    assert not workspace.joinpath("SUMMARY.md").exists()
    assert "+Clean summary" in result.diff_path.read_text(encoding="utf-8")
    proposal = json.loads(result.proposal_path.read_text(encoding="utf-8"))
    assert proposal["files"][0]["path"] == "SUMMARY.md"
    manifest = json.loads(result.run_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert manifest["agent_config"]["definition_id"] == "test-agent"
    assert any(tool["id"] == "fs.write" for tool in manifest["tool_specs"])
    assert validate_run_dir(result.run_dir) == []


def test_loop_staging_backend_records_base_hash(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_bytes(b"old\n")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call(
                        "fs_write",
                        {"path": "notes.md", "content": "new\n", "create_dirs": False},
                        "call_write",
                    ),
                    fake_tool_call("run_finish", {"summary": "Updated notes."}, "call_finish"),
                ),
            ),
        ]
    )
    result = AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace,
            run_root=tmp_path / "runs",
            workspace_backend="staging",
        ),
        model_adapter=adapter,
        runtime_config_provider=_provider("fs.write", "run.finish"),
    ).run_once("Update.")

    assert result.status == "completed"
    assert workspace.joinpath("notes.md").read_text(encoding="utf-8") == "new\n"
    file_info = json.loads(result.proposal_path.read_text(encoding="utf-8"))["files"][0]
    assert file_info["base_sha256"] == sha256_bytes(b"old\n")


def test_loop_uses_injected_workspace_factory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")
    seen: list[AgentRunSpec] = []

    def factory(run_spec: AgentRunSpec):
        seen.append(run_spec)
        return default_local_workspace_factory(run_spec)

    result = AgentLoop(
        spec=spec,
        model_adapter=_finish_only_adapter(),
        workspace_factory=factory,
        runtime_config_provider=_provider("run.finish"),
    ).run_once("noop")

    assert result.status == "completed"
    assert seen == [spec]


def test_unknown_tool_is_recorded_as_observation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1", tool_calls=(fake_tool_call("missing_tool", {}, "call_missing"),)
            ),
            ModelTurn(final_text="done"),
        ]
    )

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=_provider("run.finish"),
    ).run_once("Do it.")

    assert result.status == "completed"
    assert "unknown tool" in result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")


def test_absent_binding_means_tool_unavailable(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_write", {"path": "x.md", "content": "x"}, "c1"),),
            ),
            ModelTurn(final_text="done"),
        ]
    )

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=_provider("fs.read", "run.finish"),
    ).run_once("Write.")

    assert result.status == "completed"
    assert "fs.write" not in {tool.id for tool in adapter.requests[0].tools}
    assert "unknown tool" in result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")


def test_binding_authorization_and_quota_are_enforced(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("old.md").write_text("old\n", encoding="utf-8")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call("fs_read", {"path": "old.md"}, "read1"),
                    fake_tool_call("fs_read", {"path": "old.md"}, "read2"),
                    fake_tool_call("fs_delete", {"path": "old.md"}, "delete1"),
                ),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    config = runtime_config(
        bindings=(
            tool_binding("fs.read", quota=ToolQuota(max_calls_per_run=1)),
            tool_binding("fs.delete", authorization="deny"),
            tool_binding("run.finish"),
        )
    )

    result = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
    ).run_once("Try tools.")

    transcript = result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")
    assert "tool_quota_exceeded" in transcript
    assert "tool_binding_denied" in transcript
    assert workspace.joinpath("old.md").exists()


class _ApprovalToolProvider:
    def __init__(self) -> None:
        self.calls = 0

    def get_tools(self, context: ToolContext | None = None) -> list[ToolSpec]:
        del context

        def handler(ctx: ToolContext, args: dict) -> ToolResult:
            del ctx
            self.calls += 1
            return ToolResult(ok=True, content={"value": args.get("value")})

        return [
            ToolSpec(
                id="demo.approval",
                description="approval demo",
                input_schema={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "additionalProperties": True,
                },
                capability="",
                side_effect="write",
                handler=handler,
            )
        ]


class _ReplayProcessExit(BaseException):
    """Crash sentinel that bypasses the loop's controlled Exception boundary."""


class _ReplayApprovalToolProvider:
    def __init__(self, *, crash_on_value: str | None = None) -> None:
        self.crash_on_value = crash_on_value
        self.calls: list[str] = []

    def get_tools(self, context: ToolContext | None = None) -> list[ToolSpec]:
        del context

        def handler(ctx: ToolContext, args: dict) -> ToolResult:
            del ctx
            value = str(args.get("value") or "")
            self.calls.append(value)
            if value == self.crash_on_value:
                raise _ReplayProcessExit()
            return ToolResult(ok=True, content={"value": value})

        return [
            ToolSpec(
                id="demo.approval",
                description="approval replay crash demo",
                input_schema={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "additionalProperties": True,
                },
                capability="",
                side_effect="write",
                handler=handler,
            )
        ]


def _seed_approved_replays(
    tmp_path: Path, values: tuple[str, ...]
) -> tuple[Path, AgentRuntimeConfig, CheckpointStore, CheckpointRecord]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = _ReplayApprovalToolProvider()
    config = runtime_config(
        bindings=(tool_binding("demo.approval", authorization="ask"), tool_binding("run.finish"))
    )
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "seed-runs"),
        model_adapter=FakeModelAdapter(
            turns=[
                ModelTurn(
                    tool_calls=tuple(
                        fake_tool_call("demo_approval", {"value": value}, f"call_{index}")
                        for index, value in enumerate(values, start=1)
                    )
                )
            ]
        ),
        runtime_config_provider=runtime_provider(config),
        tool_providers=(provider,),
    )
    loop.open()
    suspended = loop.run_until_suspended("approve the calls")
    assert suspended.reason == "awaiting_tasks"
    assert len(suspended.awaiting_task_ids) == len(values)
    manager = loop._require_open().res.context.job_manager
    task_by_call_id = {
        str(getattr(task, "request").get("call_id") or ""): task_id
        for task_id, task in manager.jobs.items()
        if getattr(task, "request", {}).get("call_id")
    }
    for index in range(1, len(values) + 1):
        loop.report_task_result(task_by_call_id[f"call_{index}"], {"approved": True})
    assert loop.checkpoint_store is not None
    store = loop.checkpoint_store
    source = store.latest(loop.spec.run_id)
    assert source is not None
    assert source.checkpoint.pending_tool_approval_replays == []
    loop.release_parked()
    return workspace, config, store, source


def test_ask_authorization_parks_and_approved_call_executes_once(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = _ApprovalToolProvider()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("demo_approval", {"value": "ok"}, "call_1"),)),
            ModelTurn(final_text="park"),
            ModelTurn(final_text="done"),
        ]
    )
    config = runtime_config(
        bindings=(tool_binding("demo.approval", authorization="ask"), tool_binding("run.finish"))
    )
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
        tool_providers=(provider,),
    )
    loop.open()

    suspended = loop.run_until_suspended("use the approval tool")
    assert suspended.reason == "awaiting_tasks"
    assert suspended.awaiting_task_ids
    assert provider.calls == 0

    first = loop.report_task_result(suspended.awaiting_task_ids[0], {"approved": True})
    second = loop.report_task_result(suspended.awaiting_task_ids[0], {"approved": True})
    assert first["delivered"] is True
    assert second["duplicate"] is True
    assert loop.checkpoint_store is not None
    latest = loop.checkpoint_store.latest(loop.spec.run_id)
    assert latest is not None
    checkpoint_task = next(
        task
        for task in latest.checkpoint.hosted_tasks
        if task["task_id"] == suspended.awaiting_task_ids[0]
    )
    assert checkpoint_task["result"]["approved"] is True
    assert checkpoint_task["ready_for_reentry"] is True
    resumed = loop.run_until_suspended(None)
    result = loop.close()

    assert resumed.turn is not None
    assert provider.calls == 1
    assert result.status == "completed"


def test_approval_replay_refreshes_surface_after_quota_consumed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = _ApprovalToolProvider()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("demo_approval", {"value": "ok"}, "call_1"),)),
            ModelTurn(final_text="done"),
        ]
    )
    config = runtime_config(
        bindings=(
            tool_binding(
                "demo.approval",
                authorization="ask",
                quota=ToolQuota(max_calls_per_run=1),
            ),
            tool_binding("run.finish"),
        )
    )
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
        tool_providers=(provider,),
    )
    loop.open()

    suspended = loop.run_until_suspended("use the approval tool")
    assert suspended.reason == "awaiting_tasks"
    loop.report_task_result(suspended.awaiting_task_ids[0], {"approved": True})
    resumed = loop.run_until_suspended(None)
    result = loop.close()

    assert resumed.reason == "settled"
    assert provider.calls == 1
    assert "demo.approval" in {tool.id for tool in adapter.requests[0].tools}
    assert "demo.approval" not in {tool.id for tool in adapter.requests[-1].tools}
    transcript = result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")
    assert '"hidden_tool_ids": ["demo.approval"]' in transcript
    assert result.status == "completed"


def test_ask_authorization_reported_result_survives_restore_before_replay(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = _ApprovalToolProvider()
    config = runtime_config(
        bindings=(tool_binding("demo.approval", authorization="ask"), tool_binding("run.finish"))
    )
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=FakeModelAdapter(
            turns=[
                ModelTurn(tool_calls=(fake_tool_call("demo_approval", {"value": "ok"}, "call_1"),)),
                ModelTurn(final_text="park"),
            ]
        ),
        runtime_config_provider=runtime_provider(config),
        tool_providers=(provider,),
    )
    loop.open()
    suspended = loop.run_until_suspended("use the approval tool")

    loop.report_task_result(suspended.awaiting_task_ids[0], {"approved": True})
    assert loop.checkpoint_store is not None
    latest = loop.checkpoint_store.latest(loop.spec.run_id)
    assert latest is not None

    restored = AgentLoop(
        spec=AgentRunSpec(
            run_id=loop.spec.run_id,
            workspace_root=workspace,
            run_root=tmp_path / "restored-runs",
        ),
        model_adapter=FakeModelAdapter(turns=[ModelTurn(final_text="restored")]),
        runtime_config_provider=runtime_provider(config),
        tool_providers=(provider,),
    )
    restored.restore(latest.checkpoint, blobs=latest.blob)
    manager = restored._require_open().res.context.job_manager
    task = manager.jobs[suspended.awaiting_task_ids[0]]

    assert getattr(task, "result")["approved"] is True
    assert getattr(task, "ready_for_reentry") is True


def test_ask_authorization_replay_consumed_checkpoint_precedes_handler(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    observed: dict[str, object] = {}
    task_holder: dict[str, str] = {}
    loop_holder: dict[str, AgentLoop] = {}

    class Provider:
        calls = 0

        def get_tools(self, context: ToolContext | None = None) -> list[ToolSpec]:
            del context

            def handler(ctx: ToolContext, args: dict) -> ToolResult:
                del ctx, args
                self.calls += 1
                loop = loop_holder["loop"]
                assert loop.checkpoint_store is not None
                latest = loop.checkpoint_store.latest(loop.spec.run_id)
                assert latest is not None
                observed["delivered"] = list(latest.checkpoint.delivered_reentry_jobs)
                observed["pending_replays"] = list(latest.checkpoint.pending_tool_approval_replays)
                return ToolResult(ok=True, content={"value": "ok"})

            return [
                ToolSpec(
                    id="demo.approval",
                    description="approval demo",
                    input_schema={"type": "object", "additionalProperties": True},
                    capability="",
                    side_effect="write",
                    handler=handler,
                )
            ]

    provider = Provider()
    config = runtime_config(
        bindings=(tool_binding("demo.approval", authorization="ask"), tool_binding("run.finish"))
    )
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=FakeModelAdapter(
            turns=[
                ModelTurn(tool_calls=(fake_tool_call("demo_approval", {"value": "ok"}, "call_1"),)),
                ModelTurn(final_text="park"),
                ModelTurn(final_text="done"),
            ]
        ),
        runtime_config_provider=runtime_provider(config),
        tool_providers=(provider,),
    )
    loop_holder["loop"] = loop
    loop.open()

    suspended = loop.run_until_suspended("use the approval tool")
    task_holder["task_id"] = suspended.awaiting_task_ids[0]
    loop.report_task_result(task_holder["task_id"], {"approved": True})
    loop.run_until_suspended(None)
    result = loop.close()

    assert result.status == "completed"
    assert provider.calls == 1
    assert task_holder["task_id"] in observed["delivered"]
    assert observed["pending_replays"] == []


@pytest.mark.parametrize("crash_point", ("before_first_handler", "during_first_handler"))
def test_approval_replay_crash_preserves_unstarted_tail_once(
    tmp_path: Path, crash_point: str
) -> None:
    workspace, config, store, source = _seed_approved_replays(tmp_path, ("first", "second"))
    persisted = False

    def persist_then_maybe_crash(checkpoint: RunCheckpoint, blobs: Mapping[str, bytes]) -> None:
        nonlocal persisted
        store.put(checkpoint, blobs)
        if crash_point == "before_first_handler" and not persisted:
            persisted = True
            raise _ReplayProcessExit()
        persisted = True

    crashing_provider = _ReplayApprovalToolProvider(
        crash_on_value="first" if crash_point == "during_first_handler" else None
    )
    crashed = AgentLoop(
        spec=AgentRunSpec(
            run_id=source.checkpoint.run_id,
            workspace_root=workspace,
            run_root=tmp_path / "crashed-runs",
        ),
        model_adapter=FakeModelAdapter(turns=[ModelTurn(final_text="unreachable")]),
        runtime_config_provider=runtime_provider(config),
        tool_providers=(crashing_provider,),
        checkpoint_store=store,
        checkpoint_persist_callback=persist_then_maybe_crash,
    )
    crashed.restore(source.checkpoint, blobs=source.blob)
    try:
        with pytest.raises(_ReplayProcessExit):
            crashed.run_until_suspended(None)
    finally:
        crashed.discard_uncommitted()

    committed = store.latest(source.checkpoint.run_id)
    assert committed is not None
    assert [
        replay["arguments"]["value"]
        for replay in committed.checkpoint.pending_tool_approval_replays
    ] == ["second"]
    assert crashing_provider.calls == ([] if crash_point == "before_first_handler" else ["first"])

    recovery_provider = _ReplayApprovalToolProvider()
    recovered = AgentLoop(
        spec=AgentRunSpec(
            run_id=source.checkpoint.run_id,
            workspace_root=workspace,
            run_root=tmp_path / "recovered-runs",
        ),
        model_adapter=FakeModelAdapter(turns=[ModelTurn(final_text="recovered")]),
        runtime_config_provider=runtime_provider(config),
        tool_providers=(recovery_provider,),
        checkpoint_store=store,
    )
    recovered.restore(committed.checkpoint, blobs=committed.blob)
    try:
        resumed = recovered.run_until_suspended(None)
        assert resumed.reason == "settled"
    finally:
        recovered.discard_uncommitted()

    assert recovery_provider.calls == ["second"]
    recovered_checkpoint = store.latest(source.checkpoint.run_id)
    assert recovered_checkpoint is not None
    assert recovered_checkpoint.checkpoint.pending_tool_approval_replays == []


def test_approval_replay_next_head_checkpoint_carries_prior_observation(tmp_path: Path) -> None:
    workspace, config, store, source = _seed_approved_replays(tmp_path, ("first", "second"))
    committed: list[RunCheckpoint] = []

    def record_checkpoint(checkpoint: RunCheckpoint, blobs: Mapping[str, bytes]) -> None:
        store.put(checkpoint, blobs)
        committed.append(checkpoint)

    provider = _ReplayApprovalToolProvider()
    restored = AgentLoop(
        spec=AgentRunSpec(
            run_id=source.checkpoint.run_id,
            workspace_root=workspace,
            run_root=tmp_path / "observation-runs",
        ),
        model_adapter=FakeModelAdapter(turns=[ModelTurn(final_text="done")]),
        runtime_config_provider=runtime_provider(config),
        tool_providers=(provider,),
        checkpoint_store=store,
        checkpoint_persist_callback=record_checkpoint,
    )
    restored.restore(source.checkpoint, blobs=source.blob)
    try:
        resumed = restored.run_until_suspended(None)
        assert resumed.reason == "settled"
    finally:
        restored.discard_uncommitted()

    assert provider.calls == ["first", "second"]
    assert len(committed) >= 2
    first_barrier = committed[0]
    second_barrier = committed[1]
    assert [
        replay["arguments"]["value"] for replay in first_barrier.pending_tool_approval_replays
    ] == ["second"]
    assert second_barrier.pending_tool_approval_replays == []
    assert any(
        observation["call_id"] == "tool_approval_replay:call_1"
        for observation in second_barrier.pending_observations
    )


def test_ask_authorization_denial_never_invokes_handler(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = _ApprovalToolProvider()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("demo_approval", {"value": "no"}, "call_1"),)),
            ModelTurn(final_text="park"),
            ModelTurn(final_text="denied"),
        ]
    )
    config = runtime_config(
        bindings=(tool_binding("demo.approval", authorization="ask"), tool_binding("run.finish"))
    )
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
        tool_providers=(provider,),
    )
    loop.open()

    suspended = loop.run_until_suspended("use the approval tool")
    assert suspended.reason == "awaiting_tasks"
    loop.report_task_result(
        suspended.awaiting_task_ids[0],
        {
            "approved": False,
            "reason": "policy",
            "lease": {"token_ref": "secret-ref://lease"},
            "token_ref": "secret-ref://lease",
        },
    )
    loop.run_until_suspended(None)
    result = loop.close()

    assert provider.calls == 0
    transcript = result.run_dir.joinpath("transcript.jsonl").read_text(encoding="utf-8")
    assert "secret-ref://lease" not in transcript


def test_ask_authorization_non_answered_approval_never_invokes_handler(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = _ApprovalToolProvider()
    sink = MemoryEventSink()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                tool_calls=(fake_tool_call("demo_approval", {"value": "cancelled"}, "call_1"),)
            ),
            ModelTurn(final_text="park"),
            ModelTurn(final_text="cancelled approval denied"),
        ]
    )
    config = runtime_config(
        bindings=(tool_binding("demo.approval", authorization="ask"), tool_binding("run.finish"))
    )
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
        tool_providers=(provider,),
        event_sinks=(sink,),
    )
    loop.open()

    suspended = loop.run_until_suspended("use the approval tool")
    assert suspended.reason == "awaiting_tasks"
    loop.report_task_result(
        suspended.awaiting_task_ids[0],
        {"approved": True, "reason": "reported after cancellation"},
        status="cancelled",
    )
    loop.run_until_suspended(None)
    result = loop.close()

    assert result.status == "completed"
    assert provider.calls == 0
    assert any(event.type == "tool.approval.denied" for event in sink.events)
    assert not any(event.type == "tool.approval.approved" for event in sink.events)


def test_ask_authorization_replay_rejects_approval_key_mismatch(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = _ApprovalToolProvider()
    sink = MemoryEventSink()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("demo_approval", {"value": "ok"}, "call_1"),)),
            ModelTurn(final_text="park"),
            ModelTurn(final_text="stale approval rejected"),
        ]
    )
    config = runtime_config(
        bindings=(tool_binding("demo.approval", authorization="ask"), tool_binding("run.finish"))
    )
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
        tool_providers=(provider,),
        event_sinks=(sink,),
    )
    loop.open()

    suspended = loop.run_until_suspended("use the approval tool")
    loop.report_task_result(suspended.awaiting_task_ids[0], {"approved": True})
    manager = loop._require_open().res.context.job_manager
    manager.jobs[suspended.awaiting_task_ids[0]].request["approval_key"] = "tampered"
    loop.run_until_suspended(None)
    result = loop.close()

    assert result.status == "completed"
    assert provider.calls == 0
    assert any(
        event.type == "permission.denied" and event.data.get("error_code") == "tool_approval_stale"
        for event in sink.events
    )


def test_strict_external_side_effect_denies_unsafe_tool_before_handler(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class Provider:
        calls = 0

        def get_tools(self, context: ToolContext | None = None) -> list[ToolSpec]:
            del context

            def handler(ctx: ToolContext, args: dict) -> ToolResult:
                del ctx, args
                self.calls += 1
                return ToolResult(ok=True, content={"ran": True})

            return [
                ToolSpec(
                    id="demo.external",
                    description="external",
                    input_schema={"type": "object", "additionalProperties": True},
                    capability="",
                    side_effect="write",
                    handler=handler,
                )
            ]

    provider = Provider()
    sink = MemoryEventSink()
    config = AgentRuntimeConfig(
        definition_id="test-agent",
        tools=(
            tool_binding("demo.external", runtime={"external_side_effect": True}),
            tool_binding("run.finish"),
        ),
        metadata={"tool_side_effect_policy": {"mode": "strict"}},
    )
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=FakeModelAdapter(
            turns=[
                ModelTurn(tool_calls=(fake_tool_call("demo_external", {}, "external_1"),)),
                ModelTurn(final_text="done"),
            ]
        ),
        runtime_config_provider=runtime_provider(config),
        tool_providers=(provider,),
        event_sinks=(sink,),
    )

    result = loop.run_once("go")

    assert result.status == "completed"
    assert provider.calls == 0
    assert any(
        event.type == "permission.denied"
        and event.data.get("error_code") == "tool_side_effect_policy_denied"
        for event in sink.events
    )


def test_strict_outbox_side_effect_fails_when_handler_stages_no_request(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class Provider:
        def get_tools(self, context: ToolContext | None = None) -> list[ToolSpec]:
            del context

            def handler(ctx: ToolContext, args: dict) -> ToolResult:
                del ctx, args
                return ToolResult(ok=True, content={"ran": True})

            return [
                ToolSpec(
                    id="demo.outbox_missing",
                    description="external",
                    input_schema={"type": "object", "additionalProperties": True},
                    capability="",
                    side_effect="write",
                    handler=handler,
                )
            ]

    sink = MemoryEventSink()
    config = AgentRuntimeConfig(
        definition_id="test-agent",
        tools=(
            tool_binding(
                "demo.outbox_missing",
                runtime={"external_side_effect": True, "side_effect_delivery": "outbox"},
            ),
            tool_binding("run.finish"),
        ),
        metadata={"tool_side_effect_policy": {"mode": "strict"}},
    )
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=FakeModelAdapter(
            turns=[
                ModelTurn(tool_calls=(fake_tool_call("demo_outbox_missing", {}, "external_1"),)),
                ModelTurn(final_text="done"),
            ]
        ),
        runtime_config_provider=runtime_provider(config),
        tool_providers=(Provider(),),
        event_sinks=(sink,),
    )

    result = loop.run_once("go")

    assert result.status == "completed"
    assert any(
        event.type == "tool.call.failed"
        and event.data.get("error_code") == "tool_side_effect_outbox_missing"
        for event in sink.events
    )


def test_shell_binding_auto_approve_updates_proposal(tmp_path: Path) -> None:
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
                                "from pathlib import Path; Path('SHELL.md').write_text('shell\\n', encoding='utf-8')"
                            )
                        },
                        "c1",
                    ),
                ),
            ),
            ModelTurn(final_text="done"),
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
    ).run_once("Use shell.")

    assert result.status == "completed"
    assert not workspace.joinpath("SHELL.md").exists()
    assert (
        result.run_dir.joinpath("proposal", "files", "SHELL.md").read_text(encoding="utf-8")
        == "shell\n"
    )
    events = result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8")
    assert "tool.approval.approved" in events
    assert "shell.exec.finished" in events


def test_loop_limits_and_cancellation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    limited = AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace,
            run_root=tmp_path / "limited",
            limits=RunLimits(max_steps=1),
        ),
        model_adapter=FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="r1", tool_calls=(fake_tool_call("fs_list", {"path": "."}, "c1"),)
                ),
                ModelTurn(response_id="r2"),
            ]
        ),
        runtime_config_provider=_provider("fs.list", "run.finish"),
    ).run_once("Loop.")
    assert limited.status == "limited"
    assert limited.error_code == "max_steps_exceeded"

    token = CancellationToken()
    token.cancel()
    cancelled = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "cancelled"),
        model_adapter=FakeModelAdapter(turns=[ModelTurn(final_text="done")]),
        runtime_config_provider=_provider("run.finish"),
        cancellation_token=token,
    ).run_once("Finish.")
    assert cancelled.status == "limited"
    assert cancelled.error_code == "cancelled"


# --- recoverable turn errors -----------------------------------------------------------


class _ScriptedAdapter:
    """Drives a script of turns/exceptions: a ModelTurn is returned, a BaseException is raised."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.requests: list = []

    def next_turn(self, request):  # noqa: ANN001
        self.requests.append(request)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _loop_with(tmp_path: Path, adapter, *tool_ids: str) -> tuple[AgentLoop, MemoryEventSink, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    run_root = tmp_path / "runs"
    sink = MemoryEventSink()
    spec = AgentRunSpec(workspace_root=workspace, run_root=run_root)
    loop = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=_provider(*(tool_ids or ("run.finish",))),
        event_sinks=(sink,),
    )
    return loop, sink, run_root


def test_metrics_surface_reasoning_tokens_when_reported(tmp_path: Path) -> None:
    # R10: reasoning tokens reach the metrics.updated event when the adapter reports them, so the
    # studio meter can show the reasoning share.
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                final_text="done",
                usage={
                    "input_tokens": 5,
                    "output_tokens": 9,
                    "total_tokens": 14,
                    "reasoning_tokens": 7,
                },
            )
        ]
    )
    loop, sink, _ = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        loop.run_until_suspended("hi")
        metrics = [e for e in sink.events if e.type == "metrics.updated"]
        assert metrics and metrics[-1].data["reasoning_tokens"] == 7
    finally:
        loop.close()


def test_metrics_omit_reasoning_tokens_when_absent(tmp_path: Path) -> None:
    # A non-reasoning model reports none → the key is omitted (no "🧠0" noise in the meter).
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                final_text="done", usage={"input_tokens": 5, "output_tokens": 9, "total_tokens": 14}
            )
        ]
    )
    loop, sink, _ = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        loop.run_until_suspended("hi")
        metrics = [e for e in sink.events if e.type == "metrics.updated"]
        assert metrics and "reasoning_tokens" not in metrics[-1].data
    finally:
        loop.close()


def test_metrics_surface_every_priced_sub_count_when_reported(tmp_path: Path) -> None:
    """``reasoning_tokens`` was the only one of the four priced sub-counts on this event, so a
    cache-heavy run's priced detail never reached a live consumer -- and a cache read, a cache
    write and an audio token are each billed differently from a plain input token."""

    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                final_text="done",
                usage={
                    "input_tokens": 5,
                    "output_tokens": 9,
                    "total_tokens": 14,
                    "cache_read_tokens": 1_200,
                    "cache_creation_tokens": 300,
                    "reasoning_tokens": 7,
                    "audio_tokens": 4,
                },
            )
        ]
    )
    loop, sink, _ = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        loop.run_until_suspended("hi")
        metrics = [e for e in sink.events if e.type == "metrics.updated"]
        assert metrics
        data = metrics[-1].data
        assert data["cache_read_tokens"] == 1_200
        assert data["cache_creation_tokens"] == 300
        assert data["reasoning_tokens"] == 7
        assert data["audio_tokens"] == 4
    finally:
        loop.close()


def test_metrics_omit_every_priced_sub_count_the_adapter_did_not_report(tmp_path: Path) -> None:
    """The other half of the conditional, on all four: a run that used no cache must not read
    as one whose cache saved nothing."""

    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                final_text="done",
                usage={
                    "input_tokens": 5,
                    "output_tokens": 9,
                    "total_tokens": 14,
                    "reasoning_tokens": 7,
                },
            )
        ]
    )
    loop, sink, _ = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        loop.run_until_suspended("hi")
        metrics = [e for e in sink.events if e.type == "metrics.updated"]
        assert metrics
        data = metrics[-1].data
        assert data["reasoning_tokens"] == 7
        assert "cache_read_tokens" not in data
        assert "cache_creation_tokens" not in data
        assert "audio_tokens" not in data
    finally:
        loop.close()


def test_recoverable_turn_error_classifier() -> None:
    assert _recoverable_turn_error(ModelAdapterError("x", http_status=400))
    assert _recoverable_turn_error(ModelAdapterError("x", http_status=401))
    assert _recoverable_turn_error(ModelAdapterError("x", http_status=429, retryable=True))
    assert _recoverable_turn_error(ModelAdapterError("x", retryable=True))  # any status
    assert _recoverable_turn_error(ModelAdapterError("x", config_recoverable=True))  # no status
    assert not _recoverable_turn_error(ModelAdapterError("x", http_status=500))
    assert not _recoverable_turn_error(RuntimeError("x"))


def test_a_config_recoverable_refusal_ends_the_turn_not_the_run(tmp_path: Path) -> None:
    """A client-side proof refusal (gateway_generation_not_applied) carries no HTTP status,
    so the classifier saw an unflagged non-retryable error and terminalized the run — while
    the identical condition reported by a gateway server as HTTP 400 ended only the turn.
    The error's own remedy ('set on_unsupported=\"omit\"') is config the user fixes and
    resends against, which is this classifier's definition of recoverable."""

    adapter = _ScriptedAdapter(
        [
            ModelAdapterError(
                "the gateway sent no generation_applied echo",
                provider_error_code="gateway_generation_not_applied",
                retryable=False,
                config_recoverable=True,
            )
        ]
    )
    loop, sink, run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        susp = loop.run_until_suspended("hello")
        assert susp.reason == "turn_failed"
        # The classification the loop decided on must be observable by the driver that
        # decides what to do next: the suspension and the turn.failed event both carry it.
        assert susp.config_recoverable is True
        failed = [e for e in sink.events if e.type == "turn.failed"]
        assert failed and failed[-1].data["config_recoverable"] is True
        assert loop._session is not None and loop._session.terminal is False
        types = [e.type for e in sink.events]
        assert "run.failed" not in types
        assert list(run_root.rglob("failure.json")) == []
    finally:
        loop.close()


def test_submit_surfaces_a_recoverable_turn_failure_typed(tmp_path: Path) -> None:
    """submit/asubmit are the blocking twins of run_until_suspended, and a turn that parked
    without settling has no AgentTurnResult to return. The assert this pins the replacement
    of (``suspension.turn is not None``) crashed submit() with a message-less AssertionError
    on the first recoverable turn failure — any provider 4xx, an exhausted retryable error,
    or W5's proof refusal — and under ``python -O`` silently returned None. The rule was
    bound on the astream half only (``_astream_drive`` returns the suspension)."""

    from monoid_agent_kernel.errors import TurnNotSettled

    adapter = _ScriptedAdapter(
        [
            ModelAdapterError("bad config", http_status=400, error_code="model_error"),
            ModelTurn(response_id="r2", final_text="recovered"),
        ]
    )
    loop, _sink, _run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        with pytest.raises(TurnNotSettled) as parked:
            loop.submit("hello")
        assert parked.value.reason == "turn_failed"
        assert parked.value.suspension.http_status == 400
        assert loop._session is not None and loop._session.terminal is False
        # The session is alive: re-issuing the turn settles.
        again = loop.run_until_suspended(None)
        assert again.reason == "settled"
        assert again.final_text == "recovered"
    finally:
        loop.close()


def test_interrupt_during_submit_surfaces_typed_not_assert(tmp_path: Path) -> None:
    """interrupt_turn() is documented as a thread-safe stop whose park keeps the session
    alive — fired while a caller blocks in submit(), the park must reach that caller as a
    typed outcome, not an AssertionError."""

    from monoid_agent_kernel.errors import TurnNotSettled

    adapter = _SelfInterruptingAdapter()
    loop, _sink, _run_root = _loop_with(tmp_path, adapter, "fs.list", "run.finish")
    adapter.loop = loop
    loop.open()
    try:
        with pytest.raises(TurnNotSettled) as parked:
            loop.submit("go")
        assert parked.value.reason == "interrupted"
        again = loop.run_until_suspended(None)
        assert again.reason == "settled"
        assert again.final_text == "resumed ok"
    finally:
        loop.close()


def test_run_once_returns_the_promoted_failure_instead_of_escaping(tmp_path: Path) -> None:
    """run_once is one-shot: its own finally closes the run, and close() promotes an
    unrecovered turn_failed park to the terminal failure record. That record IS the call's
    result — escaping past the close that wrote it skipped the fork-subagent roll-up and
    terminal event and left the CLI with a raw traceback, while the record itself claimed a
    clean success and the completed-run cleanup deleted the park's checkpoints."""

    adapter = _ScriptedAdapter(
        [ModelAdapterError("bad config", http_status=400, error_code="model_error")]
    )
    loop, sink, run_root = _loop_with(tmp_path, adapter)
    result = loop.run_once("hello")
    assert result.status == "failed"
    assert result.error_code == "model_error"
    assert list(run_root.rglob("failure.json"))
    types = [e.type for e in sink.events]
    assert "run.failed" in types


def test_closing_on_an_unrecovered_turn_failure_records_a_failed_run(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        [ModelAdapterError("bad config", http_status=400, error_code="model_error")]
    )
    loop, sink, run_root = _loop_with(tmp_path, adapter)
    loop.open()
    susp = loop.run_until_suspended("hello")
    assert susp.reason == "turn_failed"
    result = loop.close()
    assert result.status == "failed"
    assert list(run_root.rglob("failure.json"))
    assert "run.failed" in [e.type for e in sink.events]


def test_a_recovered_turn_failure_still_closes_completed(tmp_path: Path) -> None:
    """The promotion keys on the LAST park: a re-attempt that settles supersedes it."""

    adapter = _ScriptedAdapter(
        [
            ModelAdapterError("transient", http_status=400, error_code="model_error"),
            ModelTurn(response_id="r2", final_text="recovered"),
        ]
    )
    loop, sink, run_root = _loop_with(tmp_path, adapter)
    loop.open()
    assert loop.run_until_suspended("hello").reason == "turn_failed"
    assert loop.run_until_suspended(None).reason == "settled"
    result = loop.close()
    assert result.status == "completed"
    assert list(run_root.rglob("failure.json")) == []
    assert "run.failed" not in [e.type for e in sink.events]


def test_a_turn_failed_run_dir_still_validates(tmp_path: Path) -> None:
    """The new turn.failed data key must be bound on its third twin — the pinned event-data
    schema — or `mak validate` rejects every run containing a recoverable turn failure."""

    from monoid_agent_kernel.core.schemas import validate_run_dir

    adapter = _ScriptedAdapter(
        [
            ModelAdapterError("bad config", http_status=400, error_code="model_error"),
            ModelTurn(response_id="r2", final_text="recovered"),
        ]
    )
    loop, _sink, run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        assert loop.run_until_suspended("hello").reason == "turn_failed"
        assert loop.run_until_suspended(None).reason == "settled"
    finally:
        loop.close()
    run_dir = next(run_root.rglob("manifest.json")).parent
    assert validate_run_dir(run_dir) == []


def test_turn_failed_suspension_is_non_terminal(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        [ModelAdapterError("bad effort", http_status=400, error_code="model_error")]
    )
    loop, sink, run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        susp = loop.run_until_suspended("hello")
        assert susp.reason == "turn_failed"
        assert susp.retryable is False
        assert susp.http_status == 400
        assert loop._session is not None and loop._session.terminal is False
        types = [e.type for e in sink.events]
        assert "turn.failed" in types
        assert "run.failed" not in types
        assert list(run_root.rglob("failure.json")) == []  # not a terminal failure
    finally:
        loop.close()


def test_turn_failed_is_idempotent_on_reentry(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        [
            ModelAdapterError("transient", http_status=503, retryable=True),
            ModelTurn(response_id="r2", final_text="recovered"),
        ]
    )
    loop, _sink, _run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        first = loop.run_until_suspended("hi")
        assert first.reason == "turn_failed"
        assert loop._session is not None and loop._session.state.pending_observations == ()
        second = loop.run_until_suspended(None)  # re-issue the same turn
        assert second.reason == "settled"
        # The re-attempt sent the identical message log — no duplicated user message.
        assert adapter.requests[0].messages == adapter.requests[1].messages
    finally:
        loop.close()


def test_non_recoverable_model_error_is_terminal(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        [ModelAdapterError("server boom", http_status=500, provider_error_code="server_error")]
    )
    loop, sink, run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        susp = loop.run_until_suspended("hi")
        assert susp.reason == "terminal"
        assert susp.status == "failed"
        assert loop._session is not None and loop._session.terminal is True
        failed = [e for e in sink.events if e.type == "run.failed"]
        assert failed, "run.failed event emitted"
        # The public failure event carries the provider detail (not just a generic message), so
        # logs and the UI can see the real cause.
        assert failed[0].data["provider_error_code"] == "server_error"
        assert failed[0].data["http_status"] == 500
        # ...and the terminal park a driver reads carries the same classification the event
        # does. The Suspension had the fields and the terminal construction dropped them, so a
        # backend promoting "what the park knew" onto its record promoted defaults over the
        # truth its own event log carried.
        assert susp.provider_error_code == "server_error"
        assert susp.http_status == 500
        assert susp.retryable is False
        assert susp.config_recoverable is False
        assert list(run_root.rglob("failure.json"))
    finally:
        loop.close()


def test_generic_model_error_is_terminal(tmp_path: Path) -> None:
    # A raw exception is wrapped into a non-retryable ModelAdapterError -> still terminal.
    adapter = _ScriptedAdapter([RuntimeError("kaboom")])
    loop, sink, _run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        susp = loop.run_until_suspended("hi")
        assert susp.reason == "terminal" and susp.status == "failed"
        assert loop._session is not None and loop._session.terminal is True
    finally:
        loop.close()


def test_turn_failed_after_tool_round_clears_observations(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        [
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_write", {"path": "a.md", "content": "x"}, "c1"),),
            ),
            ModelAdapterError("transient", http_status=503, retryable=True),
            ModelTurn(response_id="r3", final_text="done"),
        ]
    )
    loop, _sink, _run_root = _loop_with(tmp_path, adapter, "fs.write", "run.finish")
    loop.open()
    try:
        first = loop.run_until_suspended("write a.md")  # tool runs, then turn 2 model call fails
        assert first.reason == "turn_failed"
        assert loop._session is not None and loop._session.state.pending_observations == ()
        second = loop.run_until_suspended(None)
        assert second.reason == "settled"
        # The post-tool message log is re-sent verbatim — the function_call_output isn't duplicated.
        assert adapter.requests[1].messages == adapter.requests[2].messages
    finally:
        loop.close()


def test_fail_recoverable_promotes_to_terminal(tmp_path: Path) -> None:
    adapter = _ScriptedAdapter(
        [ModelAdapterError("bad", http_status=400, config_recoverable=True)]
    )
    loop, sink, run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        assert loop.run_until_suspended("hi").reason == "turn_failed"
        loop.fail_recoverable("gave up after retries", error_code="model_error")
        assert loop._session is not None and loop._session.terminal is True
        assert "run.failed" in [e.type for e in sink.events]
        assert list(run_root.rglob("failure.json"))
        # The durable observation of the terminal park keeps the inherited classification the
        # run.failed event beside it carries — the checkpoint is where a post-restart reader
        # learns what this run died of.
        assert loop.checkpoint_store is not None
        stored = loop.checkpoint_store.latest(loop.spec.run_id)
        assert stored is not None and stored.checkpoint.last_suspension is not None
        assert stored.checkpoint.last_suspension["config_recoverable"] is True
        assert stored.checkpoint.last_suspension["http_status"] == 400
    finally:
        loop.close()


def test_metrics_json_reports_the_failure_classification(tmp_path: Path) -> None:
    """metrics.json carried provider_error_code/provider_http_status and dropped the verdict.

    The pair beside them — retryable / config_recoverable — is what an operator reading only
    the metrics artifact needs to decide "resend after a config fix" vs "it will fail the same
    way". Written on failed runs, from the same state the run.failed event reads.
    """
    adapter = _ScriptedAdapter(
        [
            ModelAdapterError(
                "quota exhausted",
                http_status=429,
                provider_error_code="insufficient_quota",
                config_recoverable=True,
            )
        ]
    )
    loop, _sink, run_root = _loop_with(tmp_path, adapter)
    loop.open()
    assert loop.run_until_suspended("hi").reason == "turn_failed"
    loop.fail_recoverable("gave up after retries", error_code="model_error")
    loop.close()

    metrics = json.loads(
        (run_root / loop.spec.run_id / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["status"] == "failed"
    assert metrics["provider_error_code"] == "insufficient_quota"
    assert metrics["provider_http_status"] == 429
    assert metrics["retryable"] is False
    assert metrics["config_recoverable"] is True


def test_promotion_preserves_provider_details_from_turn_failed(tmp_path: Path) -> None:
    # A recoverable provider failure records provider detail on the turn.failed; promoting it with
    # fail_recoverable() (a fresh error with no provider fields) must NOT blank that detail.
    adapter = _ScriptedAdapter(
        [
            ModelAdapterError(
                "bad request", http_status=400, provider_error_code="invalid_request_error"
            )
        ]
    )
    loop, sink, run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        assert loop.run_until_suspended("hi").reason == "turn_failed"
        loop.fail_recoverable("gave up after retries", error_code="model_error")
        failed = [e for e in sink.events if e.type == "run.failed"]
        assert failed, "run.failed emitted"
        assert failed[0].data["provider_error_code"] == "invalid_request_error"
        assert failed[0].data["http_status"] == 400
    finally:
        loop.close()


def test_fresh_terminal_failure_clears_stale_provider_details(tmp_path: Path) -> None:
    # A recoverable turn.failed records provider detail; if the re-issued turn then fails terminally
    # for an UNRELATED reason, run.failed must reflect that new cause, not the stale detail.
    adapter = _ScriptedAdapter(
        [
            ModelAdapterError(
                "rate limited",
                http_status=429,
                provider_error_code="rate_limit_exceeded",
                retryable=True,
            ),
            ModelAdapterError("server boom", http_status=500),  # terminal, no provider code
        ]
    )
    loop, sink, _run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        assert loop.run_until_suspended("hi").reason == "turn_failed"
        assert loop.run_until_suspended(None).reason == "terminal"
        failed = [e for e in sink.events if e.type == "run.failed"]
        assert failed, "run.failed emitted"
        assert failed[0].data["http_status"] == 500
        assert failed[0].data["provider_error_code"] == ""  # not the stale rate_limit_exceeded
        # Same rule for the classification beside it: the 429 was retryable and this 500 is not.
        assert failed[0].data["retryable"] is False
        assert failed[0].data["config_recoverable"] is False
    finally:
        loop.close()


def test_the_park_records_the_provider_code_and_the_retry_it_was_classified_by(
    tmp_path: Path,
) -> None:
    """The park a driver reads must carry the two facts the decision actually turns on.

    ``retryable``/``http_status`` cannot separate an ``insufficient_quota`` (a human fixes the
    billing) from a ``rate_limit_exceeded`` (back off and re-issue), and an exhausted adapter
    retry budget reads as an untried call. Both lived only inside the live exception before
    v0.21 — so a checkpoint restore handed the recovery driver a park with no reason on it.
    """

    from monoid_agent_kernel.core.result import (
        suspension_checkpoint_payload,
        suspension_from_checkpoint_payload,
    )
    from monoid_agent_kernel.providers.base import mark_provider_usage

    exc = ModelAdapterError(
        "rate limited",
        http_status=429,
        provider_error_code="rate_limit_exceeded",
        retryable=True,
        provider_retried=True,
    )
    mark_provider_usage(exc, {"input_tokens": 11, "output_tokens": 0, "total_tokens": 11})
    loop, sink, _run_root = _loop_with(tmp_path, _ScriptedAdapter([exc]))
    loop.open()
    try:
        susp = loop.run_until_suspended("hi")
        assert susp.reason == "turn_failed"
        assert susp.provider_error_code == "rate_limit_exceeded"
        assert susp.provider_retried is True
        # ...and the durable park observation carries both across a restart.
        restored = suspension_from_checkpoint_payload(suspension_checkpoint_payload(susp))
        assert restored.provider_error_code == "rate_limit_exceeded"
        assert restored.provider_retried is True
        # ...and the event beside it states the retry and what the refused call already cost.
        failed = [e for e in sink.events if e.type == "turn.failed"][-1]
        assert failed.data["provider_retried"] is True
        assert failed.data["provider_usage"] == {
            "input_tokens": 11,
            "output_tokens": 0,
            "total_tokens": 11,
        }
    finally:
        loop.close()


def test_the_blocking_facade_hands_the_driver_the_whole_classification(tmp_path: Path) -> None:
    """``TurnNotSettled`` is a driver's only handle on a park it never sees as a Suspension."""

    from monoid_agent_kernel.errors import TurnNotSettled

    exc = ModelAdapterError(
        "quota exhausted",
        http_status=429,
        provider_error_code="insufficient_quota",
        retryable=True,
        provider_retried=True,
        config_recoverable=True,
    )
    loop, _sink, _run_root = _loop_with(tmp_path, _ScriptedAdapter([exc]))
    loop.open()
    try:
        with pytest.raises(TurnNotSettled) as parked:
            loop.submit("hi")
        assert parked.value.provider_error_code == "insufficient_quota"
        assert parked.value.provider_retried is True
        assert parked.value.config_recoverable is True
    finally:
        loop.close()


def test_the_terminal_record_keeps_the_classification_the_park_carried(tmp_path: Path) -> None:
    """``fail_recoverable`` promotes a classified park into the record of having given up.

    Both writers of that record are built from one state in one breath, so both must say it:
    the log an operator tails and the bundle they restore from.
    """

    adapter = _ScriptedAdapter(
        [
            ModelAdapterError(
                "the gateway sent no generation_applied echo",
                provider_error_code="gateway_generation_not_applied",
                retryable=False,
                config_recoverable=True,
            )
        ]
    )
    loop, sink, run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        assert loop.run_until_suspended("hi").reason == "turn_failed"
        loop.fail_recoverable("gave up after retries", error_code="model_error")
        failed = [e for e in sink.events if e.type == "run.failed"]
        assert failed and failed[0].data["config_recoverable"] is True
        assert failed[0].data["retryable"] is False
        bundles = list(run_root.rglob("failure.json"))
        assert len(bundles) == 1
        bundle = json.loads(bundles[0].read_text(encoding="utf-8"))
        assert bundle["config_recoverable"] is True
        assert bundle["retryable"] is False
    finally:
        loop.close()


def test_a_billed_refusal_publishes_the_cost_it_added_to_the_totals(tmp_path: Path) -> None:
    """A call that fails *after* the provider billed for it moved the totals and published
    nothing: the success path emitted one ``metrics.updated`` per turn and this arm emitted
    none, so a run whose only model call failed billed never reported its cost at all."""

    from monoid_agent_kernel.providers.base import mark_provider_usage

    exc = ModelAdapterError(
        "refused after a complete answer",
        http_status=400,
        provider_error_code="gateway_generation_not_applied",
    )
    mark_provider_usage(
        exc,
        {"input_tokens": 7, "output_tokens": 5, "total_tokens": 12, "reasoning_tokens": 3},
    )
    loop, sink, _run_root = _loop_with(tmp_path, _ScriptedAdapter([exc]))
    loop.open()
    try:
        assert loop.run_until_suspended("hi").reason == "turn_failed"
        metrics = [e for e in sink.events if e.type == "metrics.updated"]
        assert len(metrics) == 1, [e.type for e in sink.events]
        assert metrics[-1].data["total_tokens"] == 12
        assert metrics[-1].data["input_tokens"] == 7
        assert metrics[-1].data["reasoning_tokens"] == 3
    finally:
        loop.close()


def test_a_refusal_that_cost_nothing_publishes_no_meter(tmp_path: Path) -> None:
    """The other half of the conditional: an error raised before the provider was reached adds
    nothing to the totals, so it must not publish an unchanged meter as if a turn had run."""

    exc = ModelAdapterError("bad request", http_status=400)
    loop, sink, _run_root = _loop_with(tmp_path, _ScriptedAdapter([exc]))
    loop.open()
    try:
        assert loop.run_until_suspended("hi").reason == "turn_failed"
        assert [e for e in sink.events if e.type == "metrics.updated"] == []
    finally:
        loop.close()


def test_the_transcript_records_the_retry_on_a_successful_call(tmp_path: Path) -> None:
    """The private replay artifact of a retried-then-successful call read as a clean single
    attempt, which is exactly the case where the retry evidence matters."""

    adapter = _ScriptedAdapter(
        [
            ModelTurn(
                response_id="r1",
                final_text="done",
                usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                provider_retried=True,
            )
        ]
    )
    loop, _sink, run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        loop.run_until_suspended("hi")
    finally:
        loop.close()
    transcripts = list(run_root.rglob("transcript.jsonl"))
    assert len(transcripts) == 1
    records = [
        json.loads(line)
        for line in transcripts[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model_turns = [record for record in records if record["kind"] == "model_turn"]
    assert model_turns and all(record["provider_retried"] is True for record in model_turns)


# --- DX-9: turn-level interrupt (a "stop" that keeps the session alive) -----------------


class _SelfInterruptingAdapter:
    """First turn calls a tool and flips the loop's interrupt flag, so the next step boundary
    (before the second model call) trips — simulating a user "stop" mid-turn. A later call
    settles, proving the session survived the stop."""

    def __init__(self) -> None:
        self.loop = None
        self.calls = 0

    def next_turn(self, request):  # noqa: ANN001
        self.calls += 1
        if self.calls == 1:
            self.loop.interrupt_turn()
            return ModelTurn(
                response_id="r1", tool_calls=(fake_tool_call("fs_list", {"path": "."}, "c1"),)
            )
        return ModelTurn(response_id="r2", final_text="resumed ok")


def test_interrupt_parks_turn_without_terminating(tmp_path: Path) -> None:
    adapter = _SelfInterruptingAdapter()
    loop, sink, run_root = _loop_with(tmp_path, adapter, "fs.list", "run.finish")
    adapter.loop = loop
    loop.open()
    try:
        susp = loop.run_until_suspended("go")
        assert susp.reason == "interrupted"
        assert loop._session is not None and loop._session.terminal is False
        assert adapter.calls == 1  # the second model call never ran — the turn was stopped
        types = [e.type for e in sink.events]
        assert "turn.interrupted" in types
        assert "run.failed" not in types
        assert list(run_root.rglob("failure.json")) == []  # not a terminal failure
        # The session is alive: re-issuing the turn (the interrupt flag is consumed) settles.
        again = loop.run_until_suspended(None)
        assert again.reason == "settled"
        assert again.final_text == "resumed ok"
        assert adapter.calls == 2
    finally:
        loop.close()


def test_stale_interrupt_does_not_kill_next_turn(tmp_path: Path) -> None:
    # interrupt_turn() with no turn in flight is a no-op: the next submit clears the flag.
    adapter = _ScriptedAdapter([ModelTurn(response_id="r1", final_text="ok")])
    loop, _sink, _run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        loop.interrupt_turn()  # stale stop
        susp = loop.run_until_suspended("hi")
        assert susp.reason == "settled"
        assert susp.final_text == "ok"
    finally:
        loop.close()


# --- DX-8: autonomous-drive token streaming (model.output.delta) ------------------------


def _streaming_loop(tmp_path: Path, adapter, *, emit: bool):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    sink = MemoryEventSink()
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=_provider("run.finish"),
        event_sinks=(sink,),
        emit_output_deltas=emit,
    )
    return loop, sink


def test_autonomous_drive_emits_output_deltas(tmp_path: Path) -> None:
    adapter = FakeStreamingModelAdapter(
        chunk_turns=[
            [
                TextDelta("Hel"),
                TextDelta("lo"),
                TurnComplete(response_id="r1", usage={"total_tokens": 3}),
            ]
        ]
    )
    loop, sink = _streaming_loop(tmp_path, adapter, emit=True)
    loop.open()
    try:
        susp = loop.run_until_suspended("hi")
        assert susp.reason == "settled"
        assert susp.final_text == "Hello"  # assembled identically to the one-shot path
        deltas = [e for e in sink.events if e.type == "model.output.delta"]
        assert [d.data["text"] for d in deltas] == ["Hel", "lo"]
    finally:
        loop.close()


def test_autonomous_drive_emits_reasoning_deltas(tmp_path: Path) -> None:
    # DX-13b: reasoning summary fragments surface as model.reasoning.delta (display-only) and
    # are NOT folded into the assembled final_text (that stays the answer text alone).
    adapter = FakeStreamingModelAdapter(
        chunk_turns=[
            [
                ReasoningDelta("thinking… "),
                ReasoningDelta("almost there"),
                TextDelta("Answer"),
                TurnComplete(response_id="r1"),
            ]
        ]
    )
    loop, sink = _streaming_loop(tmp_path, adapter, emit=True)
    loop.open()
    try:
        susp = loop.run_until_suspended("hi")
        assert susp.reason == "settled"
        assert susp.final_text == "Answer"  # reasoning is not part of the answer
        reasoning = [e.data["text"] for e in sink.events if e.type == "model.reasoning.delta"]
        assert reasoning == ["thinking… ", "almost there"]
        answer = [e.data["text"] for e in sink.events if e.type == "model.output.delta"]
        assert answer == ["Answer"]
    finally:
        loop.close()


def test_no_output_deltas_when_disabled(tmp_path: Path) -> None:
    # Off by default: the same streaming adapter falls back to next_turn (no delta events).
    adapter = FakeStreamingModelAdapter(chunk_turns=[[TextDelta("hi"), TurnComplete()]])
    loop, sink = _streaming_loop(tmp_path, adapter, emit=False)
    loop.open()
    try:
        susp = loop.run_until_suspended("hi")
        assert susp.final_text == "hi"
        assert not [e for e in sink.events if e.type == "model.output.delta"]
    finally:
        loop.close()


class _StreamThenStopAdapter:
    """Streams text fragments and flips the loop's interrupt flag after the first one, so the
    next post-yield check aborts the stream mid-generation (immediate stop)."""

    def __init__(self) -> None:
        self.loop = None

    async def astream_turn(self, request):  # noqa: ANN001
        yield TextDelta("part1 ")
        self.loop.interrupt_turn()  # a "stop" arrives mid-stream
        yield TextDelta("part2 ")
        yield TextDelta("part3 ")  # must NOT be reached — the stream is aborted first
        yield TurnComplete(response_id="r1")

    def next_turn(self, request):  # noqa: ANN001
        return ModelTurn(final_text="unused")


def test_interrupt_aborts_stream_mid_generation(tmp_path: Path) -> None:
    adapter = _StreamThenStopAdapter()
    loop, sink = _streaming_loop(tmp_path, adapter, emit=True)
    adapter.loop = loop
    loop.open()
    try:
        susp = loop.run_until_suspended("go")
        assert susp.reason == "interrupted"
        assert loop._session is not None and loop._session.terminal is False
        texts = [e.data["text"] for e in sink.events if e.type == "model.output.delta"]
        assert texts == ["part1 ", "part2 "]  # part3 never streamed: aborted mid-generation
        assert "turn.interrupted" in [e.type for e in sink.events]
    finally:
        loop.close()


def test_from_tools_wires_a_custom_tool_end_to_end(tmp_path: Path) -> None:
    from monoid_agent_kernel.tools.decorator import tool

    @tool(id="custom.echo", side_effect="read")
    def echo(text: str) -> dict:
        """Echo the input text."""
        return {"echoed": text}

    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("custom_echo", {"text": "hello"}, "c1"),),
            ),
            ModelTurn(response_id="r2", final_text="done"),
        ]
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop.from_tools(spec, adapter, [echo]).run_once("echo hello")

    assert result.status == "completed"
    assert result.final_text == "done"
    # The custom tool was exposed to the model under its derived exported name...
    exported = {t.exported_name for t in adapter.requests[0].tools}
    assert "custom_echo" in exported
    # ...and its result came back as an observation.
    observations = [obs for req in adapter.requests for obs in req.observations]
    assert any(obs.output.get("result") == {"echoed": "hello"} for obs in observations)


def test_from_tools_normalizes_specs_before_binding_and_surface_hashes(tmp_path: Path) -> None:
    def handler(_context: ToolContext, _arguments: dict) -> ToolResult:
        return ToolResult(ok=True, content={})

    hostile = ToolSpec(
        id="custom.\ud800",
        description="description \ud800",
        input_schema={"type": "object", "default": float("nan")},
        capability="",
        side_effect="read",
        handler=handler,
        provider_name="custom_\ud800",
        annotations={"score": float("nan")},
    )
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="done")])
    workspace = tmp_path / "ws"
    workspace.mkdir()
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")

    result = AgentLoop.from_tools(spec, adapter, [hostile]).run_once("inspect tools")

    assert result.status == "completed"
    exposed = next(tool for tool in adapter.requests[0].tools if tool.id == "custom.�")
    assert exposed.provider_name == "custom_�"
    assert exposed.description.startswith("description �")
    # ``annotations`` is model-visible content, so a non-finite value is substituted; the
    # ``input_schema`` is a control document delivered verbatim, so it is not -- the provider
    # boundary refuses such a request (tests/test_tool_schema_delivery.py) rather than sending
    # a constraint the author did not write. The run still reaches the adapter: the transcript
    # record substitutes what portable JSON cannot carry, so the schema does not fail the run
    # at a durability boundary.
    assert math.isnan(exposed.input_schema["default"])
    assert exposed.annotations["score"] is None


def test_custom_tool_result_is_normalized_before_observation_and_persistence(
    tmp_path: Path,
) -> None:
    from monoid_agent_kernel.tools.decorator import tool

    @tool(id="custom.hostile", side_effect="read")
    def hostile() -> dict:
        """Return values Python accepts and portable JSON does not."""

        return {
            "text": "\ud800",
            "numbers": [float("nan"), float("inf"), -float("inf")],
        }

    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("custom_hostile", {}, "c1"),),
            ),
            ModelTurn(response_id="r2", final_text="done\ud800"),
        ]
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")

    loop = AgentLoop.from_tools(spec, adapter, [hostile])
    result = loop.run_once("run hostile tool")

    expected = {"text": "�", "numbers": [None, None, None]}
    observations = [obs for request in adapter.requests for obs in request.observations]
    assert any(observation.output.get("result") == expected for observation in observations)
    assert result.final_text == "done�"

    def reject_constant(value: str) -> None:
        pytest.fail(f"non-finite JSON constant persisted: {value}")

    run_dir = spec.run_root / result.run_id
    for name in ("events.jsonl", "transcript.jsonl"):
        for line in run_dir.joinpath(name).read_text(encoding="utf-8").splitlines():
            if line:
                json.loads(line, parse_constant=reject_constant)
    for name in ("status.json", "metrics.json"):
        json.loads(
            run_dir.joinpath(name).read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    assert list(run_dir.glob("*.tmp")) == []


@pytest.mark.parametrize(
    "hostile_result",
    [
        ToolResult(ok=float("nan"), content={}),  # type: ignore[arg-type]
        ToolResult(ok=False, retryable=float("nan")),  # type: ignore[arg-type]
    ],
)
def test_invalid_tool_result_control_fields_become_observable_tool_failures(
    tmp_path: Path,
    hostile_result: ToolResult,
) -> None:
    def handler(_context: ToolContext, _arguments: dict) -> ToolResult:
        return hostile_result

    spec = ToolSpec(
        id="custom.hostile_control",
        description="Return an invalid control field.",
        input_schema={"type": "object"},
        capability="",
        side_effect="read",
        handler=handler,
    )
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("custom_hostile_control", {}, "c1"),),
            ),
            ModelTurn(response_id="r2", final_text="recovered"),
        ]
    )
    run_spec = AgentRunSpec(workspace_root=tmp_path / "ws", run_root=tmp_path / "runs")
    run_spec.workspace_root.mkdir()

    result = AgentLoop.from_tools(run_spec, adapter, [spec]).run_once("run hostile tool")

    assert result.status == "completed"
    assert result.final_text == "recovered"
    assert len(adapter.requests) == 2
    observation = adapter.requests[1].observations[0].output
    assert observation["ok"] is False
    assert observation["error"]["code"] == "tool_handler_error"
    assert observation["error"]["retryable"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("blob", b"\x00\x01\x02", id="bytes"),
        pytest.param("count", 10**4700, id="int-past-the-digit-limit"),
    ],
)
def test_an_unportable_tool_result_costs_the_call_and_not_the_run(
    tmp_path: Path, field: str, value: object
) -> None:
    """A value portable JSON cannot carry is refused where the wrapper can classify it.

    Before this rule, both of these passed ``normalize_tool_result`` untouched — the normalizer
    deliberately leaves non-container scalars alone — and crashed ``json.dumps`` at the *transcript*
    write, which sits before ``tool.call.finished``: measured as ``status='failed'``,
    ``error_code=internal_error``, thirteen events and no finished/failed record for the call. The
    transcript stays raw by contract, so the fix is refusal at the boundary: the model gets a
    failed observation it can correct, and the run keeps going.
    """

    def handler(_context: ToolContext, _arguments: dict) -> ToolResult:
        return ToolResult(ok=True, content={field: value})

    spec = ToolSpec(
        id="custom.unportable",
        description="Return a value portable JSON cannot carry.",
        input_schema={"type": "object"},
        capability="",
        side_effect="read",
        handler=handler,
    )
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("custom_unportable", {}, "c1"),),
            ),
            ModelTurn(response_id="r2", final_text="recovered"),
        ]
    )
    run_spec = AgentRunSpec(workspace_root=tmp_path / "ws", run_root=tmp_path / "runs")
    run_spec.workspace_root.mkdir()

    result = AgentLoop.from_tools(run_spec, adapter, [spec]).run_once("run unportable tool")

    assert result.status == "completed"
    assert result.final_text == "recovered"
    observation = adapter.requests[1].observations[0].output
    assert observation["ok"] is False
    assert observation["error"]["code"] == "tool_result_unportable"
    events_text = (run_spec.run_root / result.run_id / "events.jsonl").read_text(encoding="utf-8")
    events = [json.loads(line) for line in events_text.splitlines() if line]
    failed = [event for event in events if event["type"] == "tool.call.failed"]
    assert failed and failed[0]["data"]["error_code"] == "tool_result_unportable"


def test_unportable_artifact_metadata_costs_the_emitting_call(tmp_path: Path) -> None:
    """The same rule at its census twin: ``ToolContext.emit_artifact`` takes Python objects too.

    A custom handler's ``metadata`` dict never crosses a JSON parse, so it can carry ``bytes`` into
    the same downstream writes the tool-result route reached — the artifact's metadata rides the
    observation back to the model and the transcript. One predicate, every Python-object ingress.
    """

    def handler(context: ToolContext, _arguments: dict) -> ToolResult:
        context.emit_artifact("art.txt", "report", None, {"raw": b"\x00"})
        return ToolResult(ok=True, content={"emitted": True})

    spec = ToolSpec(
        id="custom.emitter",
        description="Emit an artifact with unportable metadata.",
        input_schema={"type": "object"},
        capability="",
        side_effect="read",
        handler=handler,
    )
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("custom_emitter", {}, "c1"),),
            ),
            ModelTurn(response_id="r2", final_text="recovered"),
        ]
    )
    run_spec = AgentRunSpec(workspace_root=tmp_path / "ws", run_root=tmp_path / "runs")
    run_spec.workspace_root.mkdir()
    (run_spec.workspace_root / "art.txt").write_text("body", encoding="utf-8")

    result = AgentLoop.from_tools(run_spec, adapter, [spec]).run_once("emit")

    assert result.status == "completed"
    observation = adapter.requests[1].observations[0].output
    assert observation["ok"] is False
    assert observation["error"]["code"] == "artifact_metadata_unportable"


def test_invalid_success_flag_cannot_emit_success_side_effect_evidence(tmp_path: Path) -> None:
    def handler(_context: ToolContext, _arguments: dict) -> ToolResult:
        return ToolResult(
            ok=1,  # type: ignore[arg-type]
            content={"changed_paths": ["claimed.txt"]},
        )

    spec = ToolSpec(
        id="custom.invalid_success",
        description="Return a malformed success flag.",
        input_schema={"type": "object"},
        capability="",
        side_effect="write",
        handler=handler,
        emits_workspace_diff=True,
        changed_paths_source="result_content",
    )
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("custom_invalid_success", {}, "c1"),),
            ),
            ModelTurn(response_id="r2", final_text="recovered"),
        ]
    )
    run_spec = AgentRunSpec(workspace_root=tmp_path / "ws", run_root=tmp_path / "runs")
    run_spec.workspace_root.mkdir()

    result = AgentLoop.from_tools(run_spec, adapter, [spec]).run_once("run malformed tool")

    events = [
        json.loads(line)
        for line in result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert not any(event["type"] == "workspace.file.changed" for event in events)
    assert adapter.requests[1].observations[0].output["ok"] is False


def test_tool_result_is_normalized_before_background_event_decision(tmp_path: Path) -> None:
    def handler(_context: ToolContext, _arguments: dict) -> ToolResult:
        return ToolResult(
            ok=True,
            content={
                "job_id": float("nan"),
                "status": "running",
                "changed_paths": ["normalized.txt"],
            },
        )

    spec = ToolSpec(
        id="custom.background_shape",
        description="Return a non-finite background identifier.",
        input_schema={"type": "object"},
        capability="",
        side_effect="write",
        handler=handler,
        emits_workspace_diff=True,
        changed_paths_source="result_content",
        skip_emit_if_background=True,
    )
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("custom_background_shape", {}, "c1"),),
            ),
            ModelTurn(response_id="r2", final_text="done"),
        ]
    )
    run_spec = AgentRunSpec(workspace_root=tmp_path / "ws", run_root=tmp_path / "runs")
    run_spec.workspace_root.mkdir()

    result = AgentLoop.from_tools(run_spec, adapter, [spec]).run_once("run normalized tool")

    events = [
        json.loads(line)
        for line in result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    changed = [event for event in events if event["type"] == "workspace.file.changed"]
    assert len(changed) == 1
    assert changed[0]["data"]["paths"] == ["normalized.txt"]
    assert adapter.requests[1].observations[0].output["result"]["job_id"] is None


def _spec_for(tmp_path: Path) -> AgentRunSpec:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")


def _loop_for(spec: AgentRunSpec, adapter, sink: MemoryEventSink) -> AgentLoop:
    return AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=_provider("run.finish"),
        event_sinks=(sink,),
    )


def _restored_loop(spec: AgentRunSpec, adapter, sink: MemoryEventSink) -> AgentLoop:
    """A fresh loop over the same run, rehydrated from the latest durable checkpoint."""

    record = LocalFsCheckpointStore(spec.run_root).latest(spec.run_id)
    assert record is not None
    loop = _loop_for(spec, adapter, sink)
    loop.restore(record.checkpoint, blobs=record.blob)
    return loop


def test_a_restored_turn_failed_park_still_closes_as_the_failure_it_is(tmp_path: Path) -> None:
    """The promotion must survive the process boundary, or it only exists in memory.

    ``close()`` promotes an unrecovered ``turn_failed`` park from a session field, and
    ``restore()`` rebuilt the session without it — so a crash-and-recover of exactly the run
    the park exists for (a non-retryable config failure, recovered, left idle, then closed)
    finalized ``completed``, wrote no ``failure.json``, and let the completed-run cleanup
    delete the very checkpoints the park preserves for an operator restore. The durable
    ``last_suspension`` is the evidence, and on this path it is the only evidence there is —
    it demonstrably committed, since the restore is reading it."""

    spec = _spec_for(tmp_path)
    loop = _loop_for(
        spec,
        _ScriptedAdapter(
            [
                ModelAdapterError(
                    "bad config",
                    http_status=400,
                    error_code="model_error",
                    config_recoverable=True,
                )
            ]
        ),
        MemoryEventSink(),
    )
    loop.open()
    assert loop.run_until_suspended("hello").reason == "turn_failed"
    del loop  # process death: no close()

    sink = MemoryEventSink()
    restored = _restored_loop(spec, _ScriptedAdapter([]), sink)
    result = restored.close()

    assert result.status == "failed"
    assert list(spec.run_root.rglob("failure.json"))
    failed = [e for e in sink.events if e.type == "run.failed"]
    assert failed
    # ...and the classification survives with it. The live RunState twins are not checkpoint
    # fields; the durable park observation is their one authority, read back at restore — so a
    # promotion after a crash records a config-fixable failure as the config-fixable one it is.
    assert failed[0].data["config_recoverable"] is True
    # The failure detail survives the hop rather than degrading to a generic message.
    assert "bad config" in (result.error or "")


def test_a_restored_park_recovered_by_a_later_turn_still_closes_completed(
    tmp_path: Path,
) -> None:
    """The rehydrated field is a park, not a verdict: a re-attempt that settles clears it,
    exactly as it does for a park that never left memory."""

    spec = _spec_for(tmp_path)
    loop = _loop_for(
        spec,
        _ScriptedAdapter(
            [ModelAdapterError("transient", http_status=400, error_code="model_error")]
        ),
        MemoryEventSink(),
    )
    loop.open()
    assert loop.run_until_suspended("hello").reason == "turn_failed"
    del loop

    sink = MemoryEventSink()
    restored = _restored_loop(
        spec, _ScriptedAdapter([ModelTurn(response_id="r2", final_text="recovered")]), sink
    )
    assert restored.run_until_suspended(None).reason == "settled"
    result = restored.close()

    assert result.status == "completed"
    assert list(spec.run_root.rglob("failure.json")) == []
    assert "run.failed" not in [e.type for e in sink.events]


def test_a_restored_settled_park_is_not_promoted_into_a_failure(tmp_path: Path) -> None:
    """Only ``turn_failed`` rehydrates. A restore of any other park must close normally, or
    the fix would invent a failure for every recovered run."""

    spec = _spec_for(tmp_path)
    loop = _loop_for(
        spec, _ScriptedAdapter([ModelTurn(response_id="r1", final_text="done")]), MemoryEventSink()
    )
    loop.open()
    assert loop.run_until_suspended("hello").reason == "settled"
    del loop

    sink = MemoryEventSink()
    restored = _restored_loop(spec, _ScriptedAdapter([]), sink)
    assert restored.close().status == "completed"
    assert list(spec.run_root.rglob("failure.json")) == []


def test_run_once_does_not_report_an_interrupted_run_as_a_success(tmp_path: Path) -> None:
    """``run_once`` absorbs a non-settling park because ``close()`` turns it into the record
    that IS the call's result — but only ``turn_failed`` is absorbed. An interrupt (and a
    pause) still surfaces as ``TurnNotSettled`` after the same close, and that close now
    records the honest outcome underneath the raise: ``close()``'s unsettled-close
    promotion finalizes ``limited``/``closed_unsettled`` (the turn never settled) and keeps
    the checkpoints — it used to finalize a clean ``completed`` with an empty answer and
    delete them."""

    from monoid_agent_kernel.errors import TurnNotSettled

    adapter = _SelfInterruptingAdapter()
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")
    loop = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=_provider("fs.list", "run.finish"),
        event_sinks=(MemoryEventSink(),),
    )
    adapter.loop = loop

    with pytest.raises(TurnNotSettled) as parked:
        loop.run_once("go")
    assert parked.value.reason == "interrupted"

    # The record beneath the typed raise is honest now (deliberate pin move: this used to
    # finalize completed and delete the checkpoints the park preserved).
    events_path = spec.run_root / spec.run_id / "events.jsonl"
    finished = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["type"] == "run.finished"
    ]
    assert finished and finished[-1]["data"]["status"] == "limited"
    assert finished[-1]["data"]["error_code"] == "closed_unsettled"
    from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore

    assert LocalFsCheckpointStore(spec.run_root).latest(spec.run_id) is not None


def test_run_once_still_returns_the_promoted_failure_for_a_turn_failure(
    tmp_path: Path,
) -> None:
    """The half that must not change: ``turn_failed`` is the park ``close()`` promotes, so it
    is still absorbed and returned as the failed result rather than raised."""

    adapter = _ScriptedAdapter(
        [ModelAdapterError("bad config", http_status=400, error_code="model_error")]
    )
    loop, _sink, run_root = _loop_with(tmp_path, adapter)
    result = loop.run_once("hello")
    assert result.status == "failed"
    assert list(run_root.rglob("failure.json"))


def test_a_refused_turns_tokens_still_reach_the_run_budget(tmp_path: Path) -> None:
    """A call can fail *after* the provider produced and billed a complete answer — that is
    exactly the shape of the applied-parameters proof refusals. The loop's accumulation runs
    only on the returned-turn path, so those tokens never reached ``total_usage``: the metrics
    reported a run cheaper than it was, and the cumulative token budget under-counted every
    refused call, which makes it a bound that does not hold."""

    from monoid_agent_kernel.providers.base import mark_provider_usage

    refusal = ModelAdapterError(
        "did not apply the requested generation parameters",
        provider_error_code="gateway_generation_not_applied",
        error_code="model_error",
        config_recoverable=True,
    )
    mark_provider_usage(refusal, {"input_tokens": 120, "output_tokens": 340, "total_tokens": 460})

    adapter = _ScriptedAdapter([refusal, ModelTurn(response_id="r2", final_text="recovered")])
    loop, sink, _run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        assert loop.run_until_suspended("hello").reason == "turn_failed"
        totals = dict(loop._session.state.total_usage)  # type: ignore[union-attr]
        assert totals == {"input_tokens": 120, "output_tokens": 340, "total_tokens": 460}
        # A later settle adds to the refused call's cost rather than replacing it, so the
        # next metrics event — the first one this run emits — already carries it.
        assert loop.run_until_suspended(None).reason == "settled"
        assert loop._session.state.total_usage["total_tokens"] >= 460  # type: ignore[union-attr]
        metrics = [e for e in sink.events if e.type == "metrics.updated"]
        assert metrics and metrics[-1].data["total_tokens"] >= 460
    finally:
        loop.close()


def test_a_refused_gateway_body_puts_its_billed_tokens_in_the_run_budget(tmp_path: Path) -> None:
    """The same rule end to end, off a real wire rather than a hand-stamped exception.

    Two halves, and a pin on either one alone passes while the other is missing: the gateway
    reader reads the cost off the payload it is refusing (``providers/gateway.py``), and the
    loop adds it to ``total_usage`` (the arm above). The refusal here is produced by driving the
    shipped parser over a billed 200 body with one malformed key -- the shape a gateway relaying
    a reasoning-capable upstream actually produces -- so a stamp that stops covering that key
    shows up as a run whose budget forgot a paid call.
    """

    from monoid_agent_kernel.providers.gateway import _parse_gateway_response

    billed = {"input_tokens": 120, "output_tokens": 340, "total_tokens": 460}
    with pytest.raises(ModelAdapterError) as refused:
        _parse_gateway_response(
            {
                "protocol": "monoid.llm-turn-result.v1",
                "turn_handle": "turn_1",
                "final_text": "answered",
                "tool_calls": [],
                "usage": dict(billed),
                "stop_reason": "stop",
                "provider_retried": False,
                "reasoning": "not-an-array",
            }
        )
    assert refused.value.provider_error_code == "gateway_bad_response"

    adapter = _ScriptedAdapter([refused.value])
    loop, sink, _run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        # A malformed body carries no HTTP status and no config remedy, so this park is the
        # terminal one rather than ``turn_failed`` -- and the cost has to survive that arm too,
        # which is the arm a gateway's own malformed answer actually lands on.
        assert loop.run_until_suspended("hello").reason == "terminal"
        assert dict(loop._session.state.total_usage) == billed  # type: ignore[union-attr]
        metrics = [event for event in sink.events if event.type == "metrics.updated"]
        assert metrics and metrics[-1].data["total_tokens"] == 460
    finally:
        loop.close()


def test_a_refused_openai_body_puts_its_billed_tokens_in_the_run_budget(tmp_path: Path) -> None:
    """The SOURCE reader's end of the rule above, through the loop's own accumulation arm.

    Before the refusal was classified, ``_parse_response`` refused this shape with a raw
    ``ValueError``: the stamp rode it out of the adapter, but the loop's blanket
    ``except Exception`` arm re-wrapped it in a bare ``ModelAdapterError`` and nothing read the
    original's ``provider_usage`` — so the run budget, the transcript and the metrics event all
    recorded zero for a call the provider billed, while only the receipt (getattr-based)
    carried the cost. Classified, the refusal takes the loop's ``ModelAdapterError`` arm and
    the billed tokens reach ``total_usage``.
    """

    from monoid_agent_kernel.providers.openai import _parse_response

    billed = {"input_tokens": 120, "output_tokens": 340, "total_tokens": 460}
    with pytest.raises(ModelAdapterError) as refused:
        _parse_response(
            {
                "id": "resp_1",
                "status": "completed",
                "usage": {**billed, "input_tokens_details": "nope"},
                "output": [],
            }
        )
    assert refused.value.provider_error_code == "openai_bad_response"

    adapter = _ScriptedAdapter([refused.value])
    loop, sink, _run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        assert loop.run_until_suspended("hello").reason == "terminal"
        assert dict(loop._session.state.total_usage) == billed  # type: ignore[union-attr]
        metrics = [event for event in sink.events if event.type == "metrics.updated"]
        assert metrics and metrics[-1].data["total_tokens"] == 460
    finally:
        loop.close()


def test_a_failure_that_reports_no_usage_adds_nothing(tmp_path: Path) -> None:
    """The counterweight: an ordinary provider failure produced nothing and must cost
    nothing, or every failed call would inflate the budget."""

    adapter = _ScriptedAdapter(
        [ModelAdapterError("bad config", http_status=400, error_code="model_error")]
    )
    loop, _sink, _run_root = _loop_with(tmp_path, adapter)
    loop.open()
    try:
        assert loop.run_until_suspended("hello").reason == "turn_failed"
        # The run's zeroed counters, untouched — a failed call that produced nothing costs
        # nothing, or every provider error would inflate the budget.
        assert dict(loop._session.state.total_usage) == {  # type: ignore[union-attr]
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    finally:
        loop.close()


def _kernel_retry_loop(
    tmp_path: Path, adapter: _ScriptedAdapter
) -> tuple[AgentLoop, MemoryEventSink, Path]:
    """A loop whose runtime config assigns the retry loop to the kernel, zero schedule."""

    from monoid_agent_kernel.core.spec import ModelConfig, ModelRetryConfig

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    run_root = tmp_path / "runs"
    sink = MemoryEventSink()
    loop = AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=run_root),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(
            runtime_config(
                "run.finish",
                model=ModelConfig(
                    retry=ModelRetryConfig(
                        layer="kernel", max_attempts=2, initial_delay_s=0.0, jitter_s=0.0
                    )
                ),
            )
        ),
        event_sinks=(sink,),
    )
    return loop, sink, run_root


def test_a_kernel_absorbed_attempts_bill_reaches_the_run_budget(tmp_path: Path) -> None:
    """The receipt was the only carrier of absorbed spend; now the run's totals read it.

    Attempt one fails retryable with a billed body; the kernel absorbs it and attempt two
    answers. The turn still says what the model said -- the transcript row keeps the final
    attempt's own usage -- while ``total_usage``, the metrics event, and therefore the token
    budget and the child roll-up carry what the whole logical call cost. Before this, the
    loop discarded the receipt at the call site and accumulated ``turn.usage``: metrics
    reported a run cheaper than it was, and the cumulative budget was a bound that did not
    hold across absorbed attempts.
    """

    from monoid_agent_kernel.providers.base import mark_provider_usage

    absorbed = ModelAdapterError("transient overload", retryable=True)
    mark_provider_usage(absorbed, {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5})
    adapter = _ScriptedAdapter(
        [
            absorbed,
            ModelTurn(
                response_id="r2",
                final_text="recovered",
                usage={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
            ),
        ]
    )
    loop, sink, run_root = _kernel_retry_loop(tmp_path, adapter)
    loop.open()
    try:
        assert loop.run_until_suspended("hello").reason == "settled"
        totals = dict(loop._session.state.total_usage)  # type: ignore[union-attr]
        assert totals == {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}
        metrics = [event for event in sink.events if event.type == "metrics.updated"]
        assert metrics and metrics[-1].data["total_tokens"] == 12
    finally:
        loop.close()

    # The transcript's model_turn row keeps the turn's own usage -- the model's statement,
    # not the call's bill. The ledger (receipt) and the metrics carry the call-level truth;
    # reconciliation is: metrics == transcript rows + absorbed spend.
    run_dir = next(iter(run_root.iterdir()))
    turn_rows = [
        json.loads(line)
        for line in (run_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("kind") == "model_turn"
    ]
    assert turn_rows[0]["usage"]["total_tokens"] == 7


def test_a_kernel_exhausted_calls_whole_bill_reaches_the_run_budget(tmp_path: Path) -> None:
    """The failure arm of the rule above, green by design and guarded here.

    Both attempts fail billed; the kernel exhausts its budget and re-raises the second
    error restamped with the merged total, so the loop's existing failure accounting
    (``_billed_usage`` at the park) adds the whole call -- not the last attempt -- to
    ``total_usage``. A restamp that stops happening, or a park arm that reads the turn
    instead of the stamp, turns this red.
    """

    from monoid_agent_kernel.providers.base import mark_provider_usage

    first = ModelAdapterError("transient overload", retryable=True)
    mark_provider_usage(first, {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5})
    second = ModelAdapterError("still overloaded", retryable=True)
    mark_provider_usage(second, {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7})

    adapter = _ScriptedAdapter([first, second])
    loop, sink, _run_root = _kernel_retry_loop(tmp_path, adapter)
    loop.open()
    try:
        assert loop.run_until_suspended("hello").reason == "turn_failed"
        totals = dict(loop._session.state.total_usage)  # type: ignore[union-attr]
        assert totals == {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}
        metrics = [event for event in sink.events if event.type == "metrics.updated"]
        assert metrics and metrics[-1].data["total_tokens"] == 12
    finally:
        loop.close()


def test_a_kernel_absorbed_bill_survives_a_non_adapter_terminal_error(tmp_path: Path) -> None:
    """The twin door of the rule above: the terminal error is not a ``ModelAdapterError``.

    A third-party adapter is free to raise its own exception type, and the kernel's settle
    handler stamps the merged bill on whatever escapes -- but the loop's failure accounting
    reads that stamp in its ``except ModelAdapterError`` arm only. An exception arriving as
    anything else took the generic arm instead, which built a *fresh* ``ModelAdapterError``
    from the message and dropped the stamp with the original, so every absorbed attempt fell
    out of ``total_usage``, the metrics event and the token budget -- the exact accounting
    hole this branch exists to close, surviving on the arm nobody drove.

    Attempt one fails retryable with a billed body; attempt two raises ``RuntimeError``. The
    run ends ``terminal`` on both sides of the fix -- an exception that is not a
    ``ModelAdapterError`` carries no classification, so the error the loop speaks in its place
    is not retryable and the run does not park -- so nothing but the bill distinguishes the
    fixed code from the broken code. That is what makes the usage the assertion here, and the
    park reason a pinned constant beside it.
    """

    from monoid_agent_kernel.providers.base import mark_provider_usage

    first = ModelAdapterError("transient overload", retryable=True)
    mark_provider_usage(first, {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5})

    adapter = _ScriptedAdapter([first, RuntimeError("adapter blew up")])
    loop, sink, _run_root = _kernel_retry_loop(tmp_path, adapter)
    loop.open()
    try:
        assert loop.run_until_suspended("hello").reason == "terminal"
        totals = dict(loop._session.state.total_usage)  # type: ignore[union-attr]
        assert totals == {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
        metrics = [event for event in sink.events if event.type == "metrics.updated"]
        assert metrics and metrics[-1].data["total_tokens"] == 5
    finally:
        loop.close()


@pytest.mark.parametrize(
    ("boundary", "reason"),
    [
        (RunCancelled("run cancelled"), "terminal"),
        (RunTimeout("run exceeded max duration"), "terminal"),
        (ModelCallAborted("aborted"), "interrupted"),
    ],
    ids=["cancelled", "timed_out", "aborted"],
)
def test_a_kernel_absorbed_bill_survives_every_boundary_exit(
    tmp_path: Path, boundary: BaseException, reason: str
) -> None:
    """The remaining twins of the accounting arm: a call the run's own boundary ended.

    The generic-exception door was closed by routing it through the translator, which lands the
    failure in ``except ModelAdapterError`` where the bill is read. A run boundary lands nowhere
    near it: ``RunCancelled``, ``RunTimeout`` and the ``ModelCallAborted`` the loop turns into
    ``TurnInterrupted`` are all ``NativeAgentError``, so they take the arm one line below --
    which re-raised and accounted for nothing. The absorbed attempts on the other side of that
    boundary are completed, billed wire calls; they finished before anything was cancelled, and
    the ledger has them while ``total_usage`` did not.

    The abort arm loses them twice over: it replaces ``ModelCallAborted`` with a freshly built
    ``TurnInterrupted``, and a fresh exception carries no stamps -- the same rebuild-from-nothing
    that cost the generic arm its bill, in the one place the loop still spelled it by hand.

    Attempt one fails retryable with a billed body; attempt two raises the boundary. Each row
    ends the run differently -- a cancelled or timed-out run is terminal, an aborted turn parks
    alive for the next message -- and the bill is the same on all three.
    """

    from monoid_agent_kernel.providers.base import mark_provider_usage

    absorbed = ModelAdapterError("transient overload", retryable=True)
    mark_provider_usage(absorbed, {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5})

    adapter = _ScriptedAdapter([absorbed, boundary])
    loop, sink, _run_root = _kernel_retry_loop(tmp_path, adapter)
    loop.open()
    try:
        assert loop.run_until_suspended("hello").reason == reason
        totals = dict(loop._session.state.total_usage)  # type: ignore[union-attr]
        assert totals == {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
        metrics = [event for event in sink.events if event.type == "metrics.updated"]
        assert metrics and metrics[-1].data["total_tokens"] == 5
    finally:
        loop.close()


class _StallsAfterAbsorbing:
    """Fails once, billed and retryable, then parks forever inside the second dispatch.

    The stall is what makes the cancellation a *host* cancellation: the second attempt is
    suspended at an ``await`` when the driving task is cancelled, so the ``CancelledError`` is
    delivered into the runner rather than raised by the adapter. The runner tells those two
    apart on purpose -- an adapter that cancels its own call becomes a
    ``model_adapter_cancelled`` ``ModelAdapterError`` (`CalleeCancelled`), which lands in the
    accounting arm and always did. Only the delivered one escapes as itself.
    """

    def __init__(self, absorbed: BaseException) -> None:
        self.absorbed = absorbed
        self.calls = 0
        self.stalled = asyncio.Event()

    async def anext_turn(self, request: Any) -> ModelTurn:
        del request
        self.calls += 1
        if self.calls == 1:
            raise self.absorbed
        self.stalled.set()
        await asyncio.Event().wait()  # never returns; the host cancels us here
        raise AssertionError("unreachable")


def test_a_kernel_absorbed_bill_survives_a_host_cancellation(tmp_path: Path) -> None:
    """The arms below `Exception`: a stop that ends the call without being a failure.

    `asyncio.CancelledError` has inherited straight from `BaseException` since 3.8, so it is
    neither a `NativeAgentError` nor an `Exception` and matches none of the three arms that
    answer for a model call. It does not fall through them -- it never reaches them, and an
    arm that does not exist is invisible to a census that enumerates arms. The runner had
    already stamped the cumulative bill onto it (every attempt a kernel retry absorbed), and
    nothing read the stamp.

    The run object outlives the cancellation: the host that cancelled the task still finalizes
    the run and still reports its totals, and the spend it drops completed *before* the cancel
    arrived. Accounting for it must not swallow it -- a coroutine that absorbs a
    `CancelledError` is a broken coroutine -- so the cancellation is re-raised and asserted.
    """

    from monoid_agent_kernel.providers.base import mark_provider_usage

    absorbed = ModelAdapterError("transient overload", retryable=True)
    mark_provider_usage(absorbed, {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5})
    adapter = _StallsAfterAbsorbing(absorbed)
    loop, sink, _run_root = _kernel_retry_loop(tmp_path, adapter)

    async def drive() -> None:
        loop.open()
        task = asyncio.ensure_future(loop.arun_until_suspended("hello"))
        await asyncio.wait_for(adapter.stalled.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(drive())
        assert adapter.calls == 2
        totals = dict(loop._session.state.total_usage)  # type: ignore[union-attr]
        assert totals == {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}
        metrics = [event for event in sink.events if event.type == "metrics.updated"]
        assert metrics and metrics[-1].data["total_tokens"] == 5
    finally:
        loop.close()


def test_a_model_failure_the_adapter_did_not_classify_keeps_its_stamps() -> None:
    """The translator's own contract, stated as the split it makes: stamps travel, taxonomy does not.

    ``mark_provider_usage`` and ``mark_provider_retried`` exist to put a fact on an error so it
    survives the error's escape; rebuilding the error from its message alone is exactly the event
    they were written for, and it dropped them. Classification is the other half and moves the
    other way: ``retryable`` decides whether the run parks or terminalizes, so reading it off
    whatever attribute an arbitrary exception happens to expose would let a third-party error
    change a run's fate through a name collision.
    """

    from monoid_agent_kernel.loop import _as_model_adapter_error
    from monoid_agent_kernel.providers.base import (
        mark_provider_retried,
        mark_provider_usage,
        provider_usage_of,
    )

    raw = RuntimeError("adapter blew up")
    mark_provider_usage(raw, {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5})
    mark_provider_retried(raw)

    translated = _as_model_adapter_error(raw)
    assert isinstance(translated, ModelAdapterError)
    assert str(translated) == "adapter blew up"
    assert provider_usage_of(translated) == {
        "input_tokens": 2,
        "output_tokens": 3,
        "total_tokens": 5,
    }
    assert translated.provider_retried is True
    assert translated.retryable is False
    assert translated.http_status is None

    # Already the loop's vocabulary: returned unchanged, so the caller re-raises bare and the
    # original traceback and classification survive untouched.
    already = ModelAdapterError("classified", retryable=True, http_status=503)
    assert _as_model_adapter_error(already) is already
    # And not an ``Exception`` at all: a KeyboardInterrupt is not a model failure.
    interrupt = KeyboardInterrupt()
    assert _as_model_adapter_error(interrupt) is interrupt


def test_every_model_failure_the_loop_re_raises_goes_through_one_translator() -> None:
    """Derived census: no second spelling of "replace this exception with a model error".

    The bill was lost because two doors into the same handler disagreed -- one arm carried what
    the runner stamped, its twin rebuilt the error from the message -- so the instrument that
    keeps it closed is not a test of the fixed arm but an enumeration of the shape. Every
    ``raise ModelAdapterError(...) from <cause>`` in ``loop.py`` is a site that replaces one
    exception with another, and each must be the translator, which is where the carrying rule
    lives. A new wrap site anywhere in the file reddens this until it goes through the same
    function.

    Bare ``raise ModelAdapterError(...)`` with no cause is deliberately out of scope: it
    originates a failure rather than replacing one, so it has no stamps to carry (two such sites
    exist today, and they stay legal).

    Enumerating the ``raise ... from`` form alone leaves one spelling open, and it is the one
    this very file makes idiomatic: ``_acall_model`` builds its failure into a name and raises
    *that*, so ``failure = ModelAdapterError(str(exc)); raise failure from exc`` is a single
    assignment away from restoring the defect under a green census. The second clause below
    therefore refuses the construction itself anywhere a caught error is in scope. Every site
    that legitimately builds one sits outside every ``except`` -- a turn that settled with
    neither text nor tool calls, a usage value that is not a count, a driver's give-up promoted
    to the terminal record -- because there is no stamped original there to lose.
    """

    import ast

    loop_source = (
        Path(__file__).resolve().parents[1] / "src" / "monoid_agent_kernel" / "loop.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(loop_source)

    rebuilt = [
        (node.lineno, ast.unparse(node))
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
        and node.cause is not None
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "ModelAdapterError"
    ]
    assert rebuilt == [], {
        "sites_rebuilding_an_error_instead_of_translating_it": rebuilt,
        "hint": "raise _as_model_adapter_error(exc) from exc -- one translator, every door",
    }

    # And the translator is actually reached, from both doors: the boundary that raises what the
    # runner let escape, and the caller's generic arm around it. A census that only refused the
    # bad shape would stay green if the good shape were deleted with it.
    translations = sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_as_model_adapter_error"
    )
    assert len(translations) == 2, {"call_sites": translations}

    # The assignment spelling, refused at the construction rather than at the raise.
    constructed_where_an_error_was_caught = sorted(
        (node.lineno, ast.unparse(node))
        for handler in ast.walk(tree)
        if isinstance(handler, ast.ExceptHandler)
        for node in ast.walk(handler)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelAdapterError"
    )
    assert constructed_where_an_error_was_caught == [], {
        "built_with_a_caught_error_in_scope": constructed_where_an_error_was_caught,
        "hint": "_as_model_adapter_error is the only thing that turns a caught error into one",
    }

    # Naming ``ModelAdapterError`` is what the three clauses above have in common, and it is the
    # part of the rule that was never the rule: the boundary also replaces ``ModelCallAborted``
    # with a ``TurnInterrupted``, a different class doing the identical thing, and every clause
    # above walks past it. Scoped to the function that makes the model call rather than to a
    # class, so a fourth replacement of a fifth type is covered on the day it is written.
    boundary = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
        and any(
            isinstance(call.func, ast.Attribute) and call.func.attr == "acall"
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
        )
    )
    carried = {"_carrying_stamps", "_as_model_adapter_error"}
    replacements: dict[int, str] = {}
    for node in ast.walk(boundary):
        if not isinstance(node, ast.Raise) or node.cause is None or node.exc is None:
            continue
        raised = node.exc
        if isinstance(raised, ast.Name):
            # ``failure = <expr>`` ... ``raise failure from exc``: resolve the one assignment
            # that binds the name, so the spelling this file prefers is judged by what it built.
            raised = next(
                (
                    assigned.value
                    for assigned in ast.walk(boundary)
                    if isinstance(assigned, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == raised.id  # type: ignore[union-attr]
                        for target in assigned.targets
                    )
                ),
                raised,
            )
        reached = {
            call.func.id
            for call in ast.walk(raised)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        }
        if not reached & carried:
            replacements[node.lineno] = ast.unparse(node)
    assert replacements == {}, {
        "replacements_that_drop_what_was_stamped_on_the_original": replacements,
        "hint": "_carrying_stamps(<replacement>, exc) -- the bill is on the exception, not the type",
    }

    # And the carrying rule is reached from both replacements, so deleting one cannot pass by
    # deleting the shape the clause above looks for.
    carriers = sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_carrying_stamps"
    )
    assert len(carriers) == 2, {"call_sites": carriers}


def test_every_arm_that_re_raises_a_model_call_accounts_for_what_it_billed() -> None:
    """Derived census: the accounting is a property of the ``try``, not of one of its handlers.

    The bill reaches ``total_usage`` from inside an ``except`` clause, which binds it to the types
    that clause catches. ``except ModelAdapterError`` was the arm someone drove, so it was the arm
    that accounted; the ``except NativeAgentError`` beside it -- where every run boundary lands,
    cancel, timeout, and the translated abort -- re-raised an exception the runner had just stamped
    with the whole call's bill and read nothing off it. Same shape as the generic-arm defect this
    file already carries an instrument for, one arm further along, and invisible to that instrument
    because nothing here is a ``ModelAdapterError``.

    Enumerated from the ``try`` that wraps the model call: every handler that re-raises what it
    caught reaches the accounting. A handler that raises something *else* is out of scope and must
    be -- it is a replacement, which the carrying census above governs, and the exception it lets
    escape is not the one the runner stamped.

    What this test could not see, and now can
    -----------------------------------------
    The first version enumerated ``guarded[0].handlers`` and required each to account. That reads
    as "every arm answers for its bill" and means something strictly narrower: *of the arms that
    exist*, each accounts. A class with no arm contributes **no row** -- it is not an unaccounted
    handler, it is zero handlers -- so the census could only ever catch an arm that answers wrongly
    and never a class that reaches no arm at all. The arms stopped at ``Exception``, and
    ``asyncio.CancelledError`` has inherited straight from ``BaseException`` since 3.8: a host
    cancelling the driving task walked past every arm, carrying the stamp the runner had just
    written, and this test stayed green because there was nothing for it to enumerate. It was
    green *because* the defect was total rather than partial.

    The rule is about escape routes; the encoding was about handlers, and the two agree only where
    an arm exists. So the coverage clause comes first now: the handler set must be **total** --
    some arm must catch ``BaseException`` -- which makes "no arm" unrepresentable and forces every
    class outside ``Exception`` to be a decision someone wrote down rather than an omission nobody
    could see. The split between the classes that account and the ones that must not is then a
    policy this test states, because no static reading of the source can derive it.
    """

    import ast

    tree = ast.parse(
        (Path(__file__).resolve().parents[1] / "src" / "monoid_agent_kernel" / "loop.py").read_text(
            encoding="utf-8"
        )
    )

    def _calls(node: ast.AST, name: str) -> list[ast.Call]:
        return [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == name
        ]

    def _caught(handler: ast.ExceptHandler) -> frozenset[str]:
        if handler.type is None:
            return frozenset({"BaseException"})
        if isinstance(handler.type, ast.Tuple):
            return frozenset(ast.unparse(element) for element in handler.type.elts)
        return frozenset({ast.unparse(handler.type)})

    guarded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(_calls(statement, "_acall_model") for statement in node.body)
    ]
    assert len(guarded) == 1, {"try_blocks_wrapping_the_model_call": [n.lineno for n in guarded]}

    arms = [(_caught(handler), handler) for handler in guarded[0].handlers]
    caught = frozenset[str]().union(*(types for types, _ in arms))

    # (a) Coverage, first, because the clauses after it can only speak about arms that exist.
    assert "BaseException" in caught, {
        "widest_arm_in_the_handler_chain": sorted(caught),
        "hint": (
            "an exception class with no arm is invisible to an enumeration of arms; "
            "catch BaseException so every class outside Exception is a written decision"
        ),
    }

    # (b) The stops that the run outlives are named by an arm of their own rather than left to the
    #     widest one. Stated here because no reading of the source can derive which BaseException
    #     subclasses leave a live recorder behind: a cancelled task and a Ctrl-C do, interpreter
    #     and generator teardown do not.
    stopping = {"asyncio.CancelledError", "KeyboardInterrupt"}
    assert stopping <= caught, {
        "stopping_classes_with_no_arm_of_their_own": sorted(stopping - caught),
        "hint": "these two carry a stamped bill and the run outlives them, so they account",
    }

    # (c) Every arm that re-raises what it caught accounts -- except the teardown arm, which is
    #     exactly the arm catching nothing but ``BaseException``: what reaches it is SystemExit,
    #     GeneratorExit, and anything else raised outside the Exception hierarchy, all of which
    #     mean "do not run ordinary cleanup".
    teardown = frozenset({"BaseException"})
    unaccounted = {
        ", ".join(sorted(types)): handler.lineno
        for types, handler in arms
        if types != teardown
        and any(isinstance(node, ast.Raise) and node.exc is None for node in ast.walk(handler))
        and not _calls(handler, "_account_billed_model_call")
    }
    assert unaccounted == {}, {
        "arms_re_raising_a_billed_failure_without_counting_it": unaccounted,
        "hint": "self._account_billed_model_call(exc, ...) -- one route into the totals",
    }
    # And the teardown arm stays silent on purpose: if it ever starts accounting, that is a
    # decision to publish a meter through a closing recorder, and it should not pass unread.
    assert not [
        handler.lineno
        for types, handler in arms
        if types == teardown and _calls(handler, "_account_billed_model_call")
    ], {"hint": "teardown must not publish through a recorder whose sinks are closing"}

    # (d) All three answering arms reached it, so deleting the accounting everywhere cannot pass by
    #     emptying the file of the shape this census looks for.
    accounting = [
        handler.lineno for _types, handler in arms if _calls(handler, "_account_billed_model_call")
    ]
    assert len(accounting) == 3, {"arms_that_account": accounting}


# --- the wire prunes reasoning the provider can no longer replay ------------------------------

_ITEMS_A: tuple[dict, ...] = (
    {"type": "reasoning", "id": "rs_a", "summary": [], "encrypted_content": "enc_a"},
    {"type": "function_call", "call_id": "c1", "name": "fs_write", "arguments": "{}"},
)
_ITEMS_B: tuple[dict, ...] = (
    {"type": "reasoning", "id": "rs_b", "summary": [], "encrypted_content": "enc_b"},
    {"type": "message", "id": "msg_b", "content": [{"type": "output_text", "text": "done"}]},
)


class _ReasoningWireAdapter:
    """Records the wire ``messages`` of every request and tags its turns like OpenAI's adapter.

    ``supports_multimodal`` is an instance attribute so one script can drive both wire-build
    branches: the plain copy and the media-resolving one that also enforces the byte cap.
    """

    provider_name = "openai"

    def __init__(self, turns: list[ModelTurn], *, multimodal: bool = False) -> None:
        self.turns = list(turns)
        self.requests: list = []
        self.supports_multimodal = multimodal

    def next_turn(self, request):  # noqa: ANN001
        self.requests.append(request)
        return self.turns.pop(0)


def _reasoning_wire_script() -> list[ModelTurn]:
    """One tool-loop turn inside the first user's window, then a second user turn."""
    return [
        ModelTurn(
            response_id="r1",
            tool_calls=(fake_tool_call("fs_write", {"path": "A.md", "content": "x"}, "c1"),),
            reasoning=_ITEMS_A,
        ),
        ModelTurn(response_id="r2", final_text="done", reasoning=_ITEMS_B, stop_reason="stop"),
        ModelTurn(response_id="r3", final_text="again", stop_reason="stop"),
    ]


def _assistants(messages) -> list[dict]:  # noqa: ANN001
    return [m for m in (messages or ()) if m.get("role") == "assistant"]


@pytest.mark.parametrize("multimodal", (False, True))
def test_the_wire_keeps_in_window_reasoning_and_drops_the_historical_kind(
    tmp_path: Path, multimodal: bool
) -> None:
    """Replay is only possible inside the active window, so only that is worth sending.

    The upstream rule (``_reasoning_replay_flags``) replays a captured block only while it sits
    after the last user message; once a new user message lands, that block is outside the window
    forever, because the window only moves forward. The loop appended one to every assistant
    turn and pruned none, so a long conversation re-sent every dead block on every request —
    bytes that count against the wire cap and the provider's body limit and buy nothing.
    """

    adapter = _ReasoningWireAdapter(_reasoning_wire_script(), multimodal=multimodal)
    loop, _sink, _run_root = _loop_with(tmp_path, adapter, "fs.write")
    loop.open()
    try:
        assert loop.run_until_suspended("u1").reason == "settled"
        assert loop.run_until_suspended("u2").reason == "settled"
        durable = list(loop._session.state.messages)  # type: ignore[union-attr]
    finally:
        loop.close()

    assert len(adapter.requests) == 3
    # Request 2 is still inside the first user's window: the block must ride, verbatim.
    in_window = _assistants(adapter.requests[1].messages)
    assert in_window and in_window[0]["reasoning"]["items"] == [dict(i) for i in _ITEMS_A]
    assert in_window[0]["reasoning"]["provider"] == "openai"

    # Request 3 follows a new user message: every earlier block is unreachable → not on the wire.
    after_new_user = adapter.requests[2].messages
    roles = [m.get("role") for m in after_new_user]
    assert roles == ["user", "assistant", "tool", "assistant", "user"]
    assert all("reasoning" not in m for m in _assistants(after_new_user))
    wire_json = json.dumps(list(after_new_user))
    assert "enc_a" not in wire_json and "enc_b" not in wire_json

    # The durable log is the record and keeps everything, verbatim — the prune copies, never mutates.
    durable_assistants = [m for m in durable if m.get("role") == "assistant"]
    assert [a["reasoning"]["items"] for a in durable_assistants[:2]] == [
        [dict(item) for item in _ITEMS_A],
        [dict(item) for item in _ITEMS_B],
    ]
    # Everything except that one key is the same message the log holds.
    for wire, logged in zip(_assistants(after_new_user), durable_assistants):
        assert wire == {k: v for k, v in logged.items() if k != "reasoning"}
    assert len(wire_json) < len(json.dumps(durable))


def test_the_reasoning_the_wire_carries_stops_growing_with_the_conversation(
    tmp_path: Path,
) -> None:
    """The cost shape, not just the single-hop behaviour.

    The defect was O(user_turns): every settled turn added a block and none ever left, so
    request N re-sent N-1 blocks the upstream discards. Bounded to the active window, the
    reasoning the wire carries is O(1) in the number of user turns — here every turn settles
    immediately, so each fresh window is empty and the wire carries none at all while the
    durable log accumulates all three.
    """

    fat = "z" * 2000
    script = [
        ModelTurn(
            response_id=f"r{n}",
            final_text=f"a{n}",
            reasoning=({"type": "reasoning", "id": f"rs_{n}", "encrypted_content": f"{fat}{n}"},),
            stop_reason="stop",
        )
        for n in range(3)
    ]
    adapter = _ReasoningWireAdapter(script, multimodal=True)
    loop, _sink, _run_root = _loop_with(tmp_path, adapter, "fs.write")
    loop.open()
    try:
        for turn in range(3):
            assert loop.run_until_suspended(f"u{turn}").reason == "settled"
        durable = list(loop._session.state.messages)  # type: ignore[union-attr]
    finally:
        loop.close()

    assert len(adapter.requests) == 3
    for request in adapter.requests:
        assert fat not in json.dumps(list(request.messages or ()))
    assert len([m for m in durable if "reasoning" in m]) == 3
    # The last request sees the longest log; its reasoning payload is still nothing.
    assert all("reasoning" not in m for m in (adapter.requests[-1].messages or ()))


# --- the operator's policy reaches the loop, whatever class carries the loop's own ---------
#
# The bootstrap adopts ``spec.permission_policy`` only when the loop is still carrying a
# default one, and it asked that question with ``==`` against ``PermissionPolicy()``. A
# generated dataclass ``__eq__`` is class-exact, so a public extension subclass with every
# kernel field at its default answered "not default" -- and the operator's deny/redact patterns
# were then silently not adopted. The kernel supports such subclasses deliberately (the spec
# validator gates on ``isinstance``), so this is a security control lost to a class check: the
# loop enforced the empty policy at every ``self.permission_policy`` site while the subagent
# sites two thousand lines away read ``self.spec.permission_policy`` directly -- half the run
# honouring the operator's policy and half not.


@dataclass(frozen=True)
class _TenantPermissionPolicy(PermissionPolicy):
    """A deployment's extension of the policy dataclass: kernel fields, plus one of its own."""

    tenant_id: str = ""


def _denied_read_loop(
    tmp_path: Path, *, loop_policy: PermissionPolicy, spec_policy: PermissionPolicy
) -> tuple[Any, MemoryEventSink]:
    """One run that tries to read two files, under whichever policy the bootstrap settles on."""

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / ".env").write_text("operator secret", encoding="utf-8")
    (workspace / "tenant.txt").write_text("tenant secret", encoding="utf-8")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call("fs_read", {"path": ".env"}, "call_env"),
                    fake_tool_call("fs_read", {"path": "tenant.txt"}, "call_tenant"),
                ),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    sink = MemoryEventSink()
    result = AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace,
            run_root=tmp_path / "runs",
            permission_policy=spec_policy,
        ),
        model_adapter=adapter,
        permission_policy=loop_policy,
        runtime_config_provider=_provider(),
        event_sinks=(sink,),
    ).run_once("read both")
    return result, sink


def _denied_call_ids(sink: MemoryEventSink) -> list[str]:
    return [
        event.data["call_id"]
        for event in sink.events
        if event.type == "tool.call.failed" and event.data.get("error_code") == "permission_denied"
    ]


def test_the_operator_policy_is_adopted_over_an_extension_at_the_kernel_defaults(
    tmp_path: Path,
) -> None:
    """Every kernel field default means default, whichever class declares the extra ones."""

    _result, sink = _denied_read_loop(
        tmp_path,
        loop_policy=_TenantPermissionPolicy(tenant_id="acme"),
        spec_policy=PermissionPolicy(deny_patterns=(".env",)),
    )

    assert _denied_call_ids(sink) == ["call_env"], {
        "denied": _denied_call_ids(sink),
        "hint": "the operator's spec policy was not adopted, so the loop enforced nothing",
    }


def test_a_configured_extension_policy_is_not_overwritten_by_the_spec(tmp_path: Path) -> None:
    """The counterweight, and the half the class check got right: adoption is for defaults only.

    A subclass that *does* set a kernel field is a configured policy, and the loop's own
    argument outranks the spec's exactly as it always has for a plain one.
    """

    _result, sink = _denied_read_loop(
        tmp_path,
        loop_policy=_TenantPermissionPolicy(deny_patterns=("tenant.txt",), tenant_id="acme"),
        spec_policy=PermissionPolicy(deny_patterns=(".env",)),
    )

    assert _denied_call_ids(sink) == ["call_tenant"], {
        "denied": _denied_call_ids(sink),
        "hint": "adoption overwrote a policy the caller had configured",
    }


def test_the_policy_gate_reads_the_two_kernel_fields_and_nothing_else() -> None:
    """The third member of the ``is_default`` family, pinned directly like its two siblings.

    An extension field is invisible to enforcement — ``check_paths`` and the redaction readers
    match ``deny_patterns`` and ``redact_patterns`` and nothing else — so it must be invisible
    to the gate. A kernel field that *is* set stays non-default on a subclass exactly as it does
    on the base.
    """

    assert PermissionPolicy().is_default is True
    assert PermissionPolicy(deny_patterns=(".env",)).is_default is False
    assert PermissionPolicy(redact_patterns=("*.key",)).is_default is False

    assert _TenantPermissionPolicy().is_default is True
    assert _TenantPermissionPolicy(tenant_id="acme").is_default is True
    assert _TenantPermissionPolicy(deny_patterns=(".env",), tenant_id="acme").is_default is False
    assert _TenantPermissionPolicy(redact_patterns=("*.key",)).is_default is False
