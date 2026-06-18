"""Durable-persistence serializers, checkpoint I/O, and snapshot/restore."""

from __future__ import annotations

import json
import time
from pathlib import Path

from conftest import runtime_config, runtime_provider

from native_agent_runner.core.checkpoint import (
    SCHEMA_VERSION,
    RunCheckpoint,
    read_checkpoint,
    write_checkpoint,
)
from native_agent_runner.core.spec import AgentRunSpec, RunLimits
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelTurn, ToolObservation
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call
from native_agent_runner.shell import ShellExecutionOptions
from native_agent_runner.tasks import HostedTask


def _python_command(code: str) -> str:
    return f'python -c "{code.replace(chr(34), chr(92) + chr(34))}"'


def _hitl_parked_loop(spec: AgentRunSpec) -> tuple[AgentLoop, str, Path, Path]:
    """Open a loop, drive a turn that requests human input, and leave it parked on the
    hosted task (no close). Returns the loop, the parked task id, the run dir, and the
    artifacts dir — the setup shared by the restore tests."""
    provider = runtime_provider(runtime_config("hitl.request"))
    adapter = FakeModelAdapter(
        turns=[ModelTurn(response_id="r1", tool_calls=(fake_tool_call("hitl_request", {"prompt": "Pick"}, "c1"),))]
    )
    loop = AgentLoop(spec=spec, model_adapter=adapter, runtime_config_provider=provider)
    loop.open()
    suspension = loop.run_until_suspended("ask the human")
    assert suspension.reason == "awaiting_tasks"
    recorder = loop._session.res.recorder  # type: ignore[union-attr]
    return loop, suspension.awaiting_task_ids[0], recorder.run_dir, recorder.artifacts_dir


def test_tool_observation_round_trip() -> None:
    obs = ToolObservation(
        call_id="task_abc",
        tool_name="human_input",
        output={"type": "human_input_result", "answer": "Ada"},
        is_background=True,
    )
    assert ToolObservation.from_json(obs.to_json()) == obs


def test_hosted_task_checkpoint_round_trip(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    task = HostedTask(
        job_id="task_123",
        kind="automation",
        prompt="run the pipeline",
        status="running",
        started_at=100.0,
        resume_on_exit=True,
        job_path=artifacts / "tasks" / "task_123" / "task.json",
        cancel_path=artifacts / "tasks" / "task_123" / "cancel.requested",
        created_by="backend",
        choices=("a", "b"),
        request={"trigger": "nightly"},
        ready_for_reentry=False,
    )
    restored = HostedTask.from_checkpoint(task.checkpoint_json(), artifacts)

    assert restored.job_id == "task_123"
    assert restored.kind == "automation"
    assert restored.choices == ("a", "b")
    assert restored.request == {"trigger": "nightly"}
    assert restored.resume_on_exit is True
    # job_path/cancel_path are derived from artifacts_dir, matching HostedTaskExecutor.start.
    assert restored.job_path == artifacts / "tasks" / "task_123" / "task.json"
    assert restored.cancel_path == artifacts / "tasks" / "task_123" / "cancel.requested"


def test_run_checkpoint_round_trip_via_disk(tmp_path: Path) -> None:
    cp = RunCheckpoint(
        run_id="run_1",
        status="running",
        previous_turn_handle="turn_xyz",
        pending_observations=[{"call_id": "c1", "tool_name": "t", "output": {}, "is_background": False}],
        tool_call_counts={"fs.write": 2},
        total_tool_calls=2,
        total_usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        session_step=3,
        submit_local_step=1,
        terminal=False,
        hosted_tasks=[{"task_id": "task_1", "kind": "hitl"}],
        reentry_queue=["task_1"],
        delivered_reentry_jobs=[],
        remaining_duration_s=120.0,
        cancellation_requested=False,
        queued_messages=["next please"],
    )
    write_checkpoint(tmp_path, cp)
    restored = read_checkpoint(tmp_path)

    assert restored is not None
    assert restored == cp
    assert restored.schema_version == SCHEMA_VERSION


def test_read_checkpoint_missing_returns_none(tmp_path: Path) -> None:
    assert read_checkpoint(tmp_path) is None


def test_read_checkpoint_schema_mismatch_returns_none(tmp_path: Path) -> None:
    cp = RunCheckpoint(run_id="run_1")
    write_checkpoint(tmp_path, cp)
    # Corrupt the schema version on disk -> treated as no checkpoint, never raises.
    path = tmp_path / "checkpoint.json"
    path.write_text(path.read_text(encoding="utf-8").replace(SCHEMA_VERSION, "bogus.v0"), encoding="utf-8")
    assert read_checkpoint(tmp_path) is None


def test_snapshot_writes_checkpoint_at_hosted_park(tmp_path: Path) -> None:
    spec = AgentRunSpec(workspace_root=_mk(tmp_path / "ws"), run_root=tmp_path / "runs")
    loop, _task_id, run_dir, _artifacts = _hitl_parked_loop(spec)

    # The persist hook wrote a non-terminal checkpoint at the awaiting_tasks park.
    cp = read_checkpoint(run_dir)
    assert cp is not None
    assert cp.terminal is False
    assert cp.previous_turn_handle == "r1"
    assert len(cp.hosted_tasks) == 1
    # snapshot() is a pure read: calling it again yields an equal payload, save the
    # wall-clock remaining_duration_s (a derived countdown, not run state).
    again = loop.snapshot().to_json()  # type: ignore[union-attr]
    again.pop("remaining_duration_s")
    expected = cp.to_json()
    expected.pop("remaining_duration_s")
    assert again == expected


def test_restore_resumes_parked_hitl_in_fresh_loop(tmp_path: Path) -> None:
    spec = AgentRunSpec(workspace_root=_mk(tmp_path / "ws"), run_root=tmp_path / "runs")
    loop1, task_id, run_dir, _artifacts = _hitl_parked_loop(spec)
    cp = read_checkpoint(run_dir)
    assert cp is not None
    del loop1  # simulate process death WITHOUT close()

    # Fresh "process": brand-new loop + adapter over the same run dir.
    provider = runtime_provider(runtime_config("hitl.request"))
    adapter2 = FakeModelAdapter(turns=[ModelTurn(response_id="r2", final_text="thanks")])
    loop2 = AgentLoop(spec=spec, model_adapter=adapter2, runtime_config_provider=provider)
    loop2.restore(cp)

    # The hosted task is re-registered, so an external report wakes the parked run.
    loop2.report_task_result(task_id, {"answer": "Ada"})
    suspension = loop2.run_until_suspended(None)
    loop2.close()

    assert suspension.reason == "settled"
    assert suspension.turn is not None and suspension.turn.final_text == "thanks"
    # The hitl answer reached the model, and the conversation continued by reference
    # from the pre-restart turn handle (no transcript replay).
    hitl_obs = [
        obs for req in adapter2.requests for obs in req.observations if obs.tool_name == "human_input"
    ]
    assert hitl_obs and hitl_obs[0].output["answer"] == "Ada"
    assert adapter2.requests[0].previous_turn_handle == "r1"
    # A finalized run leaves no checkpoint behind.
    assert read_checkpoint(run_dir) is None


def test_restore_folds_crashed_shell_job_as_failed_observation(tmp_path: Path) -> None:
    spec = AgentRunSpec(workspace_root=_mk(tmp_path / "ws"), run_root=tmp_path / "runs")
    loop1, _task_id, run_dir, artifacts = _hitl_parked_loop(spec)
    cp = read_checkpoint(run_dir)
    assert cp is not None
    del loop1

    # Plant a shell job left "running" on disk — its subprocess was lost on the crash.
    job_dir = artifacts / "jobs" / "job_dead"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        json.dumps({"job_id": "job_dead", "status": "running", "command_preview": "sleep 999"}),
        encoding="utf-8",
    )

    provider = runtime_provider(runtime_config("hitl.request"))
    adapter2 = FakeModelAdapter(turns=[ModelTurn(response_id="r2", final_text="ok")])
    loop2 = AgentLoop(spec=spec, model_adapter=adapter2, runtime_config_provider=provider)
    loop2.restore(cp)

    pending = loop2._session.state.pending_observations  # type: ignore[union-attr]
    failed = [obs for obs in pending if obs.output.get("status") == "failed"]
    assert failed and failed[0].output["job_id"] == "job_dead"
    assert failed[0].output["error"] == "process lost on restart"
    loop2.close()


def test_snapshot_refuses_while_in_process_shell_job_runs(tmp_path: Path) -> None:
    spec = AgentRunSpec(
        workspace_root=_mk(tmp_path / "ws"),
        run_root=tmp_path / "runs",
        workspace_backend="staging",
    )
    provider = runtime_provider(runtime_config("fs.write"))
    loop = AgentLoop(
        spec=spec,
        model_adapter=FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="done")]),
        runtime_config_provider=provider,
    )
    loop.open()
    manager = loop._session.res.context.job_manager  # type: ignore[union-attr]
    job = manager.start_shell_job(
        shell_options=ShellExecutionOptions(enabled=True, approval_mode="auto-approve"),
        command=_python_command("import time; time.sleep(0.6)"),
        cwd=".",
        timeout_s=10,
        max_output_bytes=100_000,
        startup_wait_s=1,
        env={},
        requested_timeout_s=None,
        requested_max_output_bytes=None,
        requested_startup_wait_s=None,
        execution_workspace="direct",
        resume_on_exit=True,
    )
    # A live shell subprocess can't be restored, so the park is not durable yet.
    assert loop.snapshot() is None
    manager.wait(job.job_id)  # let the job finish
    # Once only a finished (hosted-free) park remains, a snapshot is allowed.
    assert loop.snapshot() is not None
    loop.close()


def test_restore_carries_remaining_deadline(tmp_path: Path) -> None:
    spec = AgentRunSpec(
        workspace_root=_mk(tmp_path / "ws"),
        run_root=tmp_path / "runs",
        limits=RunLimits(max_duration_s=1000),
    )
    loop = AgentLoop(
        spec=spec,
        model_adapter=FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="done")]),
        runtime_config_provider=runtime_provider(runtime_config("fs.write")),
    )
    cp = RunCheckpoint(run_id=spec.run_id, status="completed", remaining_duration_s=300.0)
    loop.restore(cp)
    res = loop._session.res  # type: ignore[union-attr]
    # Downtime does not count against max_duration_s: the resumed deadline is ~now+remaining,
    # so the run is not immediately limited even though it was parked for a long time.
    assert 290.0 < (res.deadline - time.time()) < 300.5
    loop.close()


def _mk(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
