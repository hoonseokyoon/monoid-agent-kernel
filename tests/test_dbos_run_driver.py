from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from support.runtime import runtime_config, runtime_provider, tool_binding

from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore
from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.reference.dbos import DbosResumeCommand, DbosRunConfig, DbosRunDriver
from monoid_agent_kernel.tools.base import ToolContext, ToolResult, ToolSpec


class _RetryableAdapter:
    def next_turn(self, request: Any) -> ModelTurn:
        del request
        raise ModelAdapterError("retry", error_code="provider_unavailable", retryable=True)


class _FaultingCheckpointStore:
    def __init__(
        self,
        inner: LocalFsCheckpointStore,
        *,
        fault_call: int,
        commit_before_raise: bool,
    ) -> None:
        self.inner = inner
        self.fault_call = fault_call
        self.commit_before_raise = commit_before_raise
        self.put_calls = 0
        self._faulted = False

    def put(self, checkpoint, blobs=()):  # noqa: ANN001, ANN201
        self.put_calls += 1
        if self.put_calls == self.fault_call and not self._faulted:
            self._faulted = True
            if self.commit_before_raise:
                self.inner.put(checkpoint, blobs)
            raise TimeoutError("secret checkpoint transport detail")
        self.inner.put(checkpoint, blobs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class _ApprovalProvider:
    def __init__(self) -> None:
        self.calls = 0

    def get_tools(self, context: ToolContext | None = None) -> list[ToolSpec]:
        del context

        def handler(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            del ctx, args
            self.calls += 1
            return ToolResult(ok=True, content={"applied": True})

        return [
            ToolSpec(
                id="demo.approval",
                description="Approval replay durability probe.",
                input_schema={"type": "object", "additionalProperties": True},
                capability="",
                side_effect="write",
                handler=handler,
            )
        ]


def _bare_driver(
    store: _FaultingCheckpointStore,
    loop_factory,
    *,
    fault_hook=None,
) -> DbosRunDriver:
    driver = object.__new__(DbosRunDriver)
    driver.config = DbosRunConfig(
        system_database_url="sqlite:///unused.sqlite",
        checkpoint_retry_interval_s=0.001,
    )
    driver._checkpoint_store = store
    driver._loop_factory = loop_factory
    driver._fault_hook = fault_hook
    driver._state_lock = threading.Lock()
    driver._accepting = True
    return driver


def _seed_retryable_checkpoint(
    tmp_path: Path,
) -> tuple[LocalFsCheckpointStore, AgentRunSpec]:
    spec = AgentRunSpec(
        run_id="run_dbos_commit_reconcile",
        workspace_root=_workspace(tmp_path / "workspace"),
        run_root=tmp_path / "runs",
    )
    store = LocalFsCheckpointStore(spec.run_root)
    source = AgentLoop(
        spec=spec,
        model_adapter=_RetryableAdapter(),
        runtime_config_provider=runtime_provider(runtime_config("fs.write")),
        checkpoint_store=store,
    )
    source.open()
    assert source.run_until_suspended("continue after recovery").reason == "turn_failed"
    source.release_parked()
    return store, spec


def _seed_approved_replay(
    tmp_path: Path,
) -> tuple[LocalFsCheckpointStore, AgentRunSpec, _ApprovalProvider, object]:
    spec = AgentRunSpec(
        run_id="run_dbos_internal_reconcile",
        workspace_root=_workspace(tmp_path / "workspace"),
        run_root=tmp_path / "runs",
    )
    store = LocalFsCheckpointStore(spec.run_root)
    provider = _ApprovalProvider()
    config = runtime_config(bindings=(tool_binding("demo.approval", authorization="ask"),))
    source = AgentLoop(
        spec=spec,
        model_adapter=FakeModelAdapter(
            turns=[
                ModelTurn(
                    tool_calls=(
                        fake_tool_call("demo_approval", {"value": "ok"}, "call_approval"),
                    )
                )
            ]
        ),
        runtime_config_provider=runtime_provider(config),
        tool_providers=(provider,),
        checkpoint_store=store,
    )
    source.open()
    parked = source.run_until_suspended("run approved effect")
    assert parked.reason == "awaiting_tasks"
    source.report_task_result(parked.awaiting_task_ids[0], {"approved": True})
    source.release_parked()
    return store, spec, provider, config


def test_boundary_commit_then_raise_reconciles_to_completed_receipt(tmp_path: Path) -> None:
    inner, spec = _seed_retryable_checkpoint(tmp_path)
    store = _FaultingCheckpointStore(inner, fault_call=1, commit_before_raise=True)

    def loop_factory(command: DbosResumeCommand) -> AgentLoop:
        return AgentLoop(
            spec=AgentRunSpec(
                run_id=command.run_id,
                workspace_root=spec.workspace_root,
                run_root=spec.run_root,
            ),
            model_adapter=FakeModelAdapter(turns=[ModelTurn(final_text="done")]),
            runtime_config_provider=runtime_provider(runtime_config("fs.write")),
        )

    driver = _bare_driver(store, loop_factory)
    command = DbosResumeCommand(spec.run_id, "resume_once", 1)

    receipt = driver._drive_one(command)

    assert receipt.status == "completed"
    assert receipt.checkpoint_seq == 2
    assert store.put_calls == 1
    latest = inner.latest(spec.run_id)
    assert latest is not None
    assert latest.checkpoint.applied_input_ids == [command.checkpoint_marker]


@pytest.mark.parametrize(
    ("fault_call", "commit_before_raise", "expected_put_calls"),
    ((1, True, 2), (2, False, 3)),
)
def test_internal_and_final_checkpoint_uncertainty_cannot_strand_approved_replay(
    tmp_path: Path,
    fault_call: int,
    commit_before_raise: bool,
    expected_put_calls: int,
) -> None:
    inner, spec, provider, config = _seed_approved_replay(tmp_path)
    source = inner.latest(spec.run_id)
    assert source is not None and source.seq == 2
    store = _FaultingCheckpointStore(
        inner,
        fault_call=fault_call,
        commit_before_raise=commit_before_raise,
    )

    def loop_factory(command: DbosResumeCommand) -> AgentLoop:
        return AgentLoop(
            spec=AgentRunSpec(
                run_id=command.run_id,
                workspace_root=spec.workspace_root,
                run_root=spec.run_root,
            ),
            model_adapter=FakeModelAdapter(turns=[ModelTurn(final_text="done")]),
            runtime_config_provider=runtime_provider(config),
            tool_providers=(provider,),
        )

    driver = _bare_driver(store, loop_factory)
    command = DbosResumeCommand(spec.run_id, "resume_approval", source.seq)

    receipt = driver._drive_one(command)

    assert receipt.status == "completed"
    assert receipt.checkpoint_seq == 4
    assert provider.calls == 1
    assert store.put_calls == expected_put_calls
    latest = inner.latest(spec.run_id)
    assert latest is not None and latest.seq == 4
    assert latest.checkpoint.active_input == {
        "input_id": command.checkpoint_marker,
        "source_seq": 2,
        "phase": "completed",
    }
    assert latest.checkpoint.pending_tool_approval_replays == []


def test_boundary_fault_keeps_durably_committed_hosted_task_live(tmp_path: Path) -> None:
    inner, spec = _seed_retryable_checkpoint(tmp_path)
    store = _FaultingCheckpointStore(inner, fault_call=99, commit_before_raise=False)

    def loop_factory(command: DbosResumeCommand) -> AgentLoop:
        return AgentLoop(
            spec=AgentRunSpec(
                run_id=command.run_id,
                workspace_root=spec.workspace_root,
                run_root=spec.run_root,
            ),
            model_adapter=FakeModelAdapter(
                turns=[
                    ModelTurn(
                        tool_calls=(
                            fake_tool_call("hitl_request", {"prompt": "continue?"}, "call_hitl"),
                        )
                    )
                ]
            ),
            runtime_config_provider=runtime_provider(runtime_config("hitl.request")),
        )

    def boundary_fault(phase: str, command: DbosResumeCommand) -> None:
        del command
        if phase == "boundary_committed":
            raise RuntimeError("post-boundary fault")

    driver = _bare_driver(store, loop_factory, fault_hook=boundary_fault)
    command = DbosResumeCommand(spec.run_id, "resume_to_hitl", 1)

    with pytest.raises(RuntimeError, match="post-boundary fault"):
        driver._drive_one(command)

    latest = inner.latest(spec.run_id)
    assert latest is not None and len(latest.checkpoint.hosted_tasks) == 1
    task_id = latest.checkpoint.hosted_tasks[0]["task_id"]
    cancel_path = spec.run_root / spec.run_id / "artifacts" / "tasks" / task_id / "cancel.requested"
    assert cancel_path.exists() is False


def _workspace(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
