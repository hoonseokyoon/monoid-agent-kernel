"""Durable-persistence serializers and checkpoint I/O (Phase G)."""

from __future__ import annotations

from pathlib import Path

from native_agent_runner.core.checkpoint import (
    SCHEMA_VERSION,
    RunCheckpoint,
    read_checkpoint,
    write_checkpoint,
)
from native_agent_runner.providers.base import ToolObservation
from native_agent_runner.tasks import HostedTask


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
