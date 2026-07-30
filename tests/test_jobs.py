"""Characterization tests pinning current TaskManager behavior.

These lock the shell background-job contract (status transitions, the
result_observation byte shape, artifact layout, reentry idempotency, and
terminal events) so the upcoming Task/TaskExecutor refactor stays behavior
preserving.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import pytest

import monoid_agent_kernel.tasks as tasks_module
from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.tasks import BackgroundJob, SubagentTaskExecutor, TaskManager
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.recorder import AgentRecorder
from monoid_agent_kernel.shell import ShellExecutionOptions
from monoid_agent_kernel.workspace.local import LocalWorkspaceBackend
from support.process import python_command as _python_command

pytestmark = pytest.mark.integration


class _CaptureSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def emit(self, event: AgentEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        return None


def _manager(tmp_path: Path) -> tuple[TaskManager, AgentRecorder, _CaptureSink]:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = LocalWorkspaceBackend(workspace_root, mode="propose", backend_kind="staging")
    sink = _CaptureSink()
    recorder = AgentRecorder(tmp_path / "runs", "run_jobs", extra_event_sinks=(sink,), status_file=False)
    manager = TaskManager(
        run_id="run_jobs",
        workspace=workspace,
        recorder=recorder,
        permission_policy=PermissionPolicy(),
    )
    return manager, recorder, sink


def _start(
    manager: TaskManager,
    command: str,
    *,
    timeout_s: int = 10,
    max_output_bytes: int = 100_000,
    resume_on_exit: bool = True,
) -> BackgroundJob:
    return manager.start_shell_job(
        shell_options=ShellExecutionOptions(enabled=True, approval_mode="auto-approve"),
        command=command,
        cwd=".",
        timeout_s=timeout_s,
        max_output_bytes=max_output_bytes,
        startup_wait_s=0,
        env={},
        requested_timeout_s=None,
        requested_max_output_bytes=None,
        requested_startup_wait_s=None,
        execution_workspace="direct",
        resume_on_exit=resume_on_exit,
    )


_RESULT_OBSERVATION_KEYS = {
    "type",
    "job_id",
    "kind",
    "command_preview",
    "status",
    "exit_code",
    "duration_s",
    "stdout_tail",
    "stderr_tail",
    "stdout_path",
    "stderr_path",
    "stdout_bytes",
    "stderr_bytes",
    "timed_out",
    "output_truncated",
    "effective_timeout_s",
    "effective_max_output_bytes",
    "changed_paths",
    "error",
}


def test_background_job_lifecycle_and_result_observation(tmp_path: Path) -> None:
    manager, recorder, _sink = _manager(tmp_path)
    job = _start(manager, _python_command("print('hello-job')"))
    manager.wait(job.job_id, timeout_s=10)

    assert job.status == "exited"
    obs = job.result_observation(recorder.run_dir)
    assert set(obs) == _RESULT_OBSERVATION_KEYS
    assert obs["type"] == "background_job_result"
    assert obs["job_id"] == job.job_id
    assert obs["status"] == "exited"
    assert obs["exit_code"] == 0
    assert "hello-job" in obs["stdout_tail"]
    assert obs["timed_out"] is False
    assert obs["output_truncated"] is False


def test_job_artifact_layout_and_schema(tmp_path: Path) -> None:
    manager, recorder, _sink = _manager(tmp_path)
    job = _start(manager, _python_command("print('art')"))
    manager.wait(job.job_id, timeout_s=10)

    job_dir = recorder.run_dir / "artifacts" / "jobs" / job.job_id
    assert job.job_path == job_dir / "job.json"
    assert job.stdout_path == job_dir / "stdout.log"
    assert job.job_path.exists()
    assert job.stdout_path.exists()

    data: dict[str, Any] = json.loads(job.job_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "monoid.background-job.v1"
    assert data["job_id"] == job.job_id
    assert data["status"] == "exited"
    assert data["stdout_path"] == f"artifacts/jobs/{job.job_id}/stdout.log"


def test_reentry_is_idempotent_and_clears_has_resume(tmp_path: Path) -> None:
    manager, _recorder, _sink = _manager(tmp_path)
    job = _start(manager, _python_command("print('reentry')"))
    manager.wait(job.job_id, timeout_s=10)

    assert manager.has_resume_jobs() is True
    first = manager.pop_reentry_observations()
    # Reentry renders through the ShellResultInjector: a background ToolObservation.
    assert [obs.output["job_id"] for obs in first] == [job.job_id]
    assert first[0].output["type"] == "background_job_result"
    assert first[0].tool_name == "background_job"
    assert first[0].is_background is True

    # Draining is idempotent: a second pop yields nothing and clears the flag.
    assert manager.pop_reentry_observations() == []
    assert manager.has_resume_jobs() is False


def test_clock_rollback_cannot_block_job_reentry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, recorder, sink = _manager(tmp_path)
    job_id = "job_clock_rollback"
    job_dir = recorder.artifacts_dir / "jobs" / job_id
    job_dir.mkdir(parents=True)
    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"
    stdout_path.write_bytes(b"")
    stderr_path.write_bytes(b"")
    job = BackgroundJob(
        job_id=job_id,
        kind="shell",
        command="python -c pass",
        command_preview="python -c pass",
        cwd=".",
        status="running",
        started_at=200.0,
        timeout_s=10,
        max_output_bytes=100_000,
        startup_wait_s=0,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        job_path=job_dir / "job.json",
        cancel_path=job_dir / "cancel.requested",
        execution_workspace="direct",
        resume_on_exit=True,
    )

    # A rollback can happen while the job is live, so started/output/status projections must
    # remain valid before there is a terminal timestamp as well as after one is recorded.
    monkeypatch.setattr(tasks_module.time, "time", lambda: 100.0)
    assert manager._public_job_payload(job)["duration_s"] == 0.0
    job.status = "exited"
    job.finished_at = 100.0
    job.exit_code = 0
    manager.jobs[job_id] = job

    waiter_entered = threading.Event()
    wait_result: list[bool] = []
    condition_wait = manager._condition.wait

    def observed_wait(timeout: float | None = None) -> bool:
        waiter_entered.set()
        return condition_wait(timeout)

    monkeypatch.setattr(manager._condition, "wait", observed_wait)
    waiter = threading.Thread(
        target=lambda: wait_result.append(manager.wait_for_reentry(5.0)),
        daemon=True,
    )
    waiter.start()
    assert waiter_entered.wait(1.0) is True

    manager.mark_ready(job)

    waiter.join(timeout=1.0)
    woke_promptly = not waiter.is_alive()
    if not woke_promptly:
        # Keep a failing mutation from leaking the waiter into later tests.
        with manager._condition:
            manager._condition.notify_all()
        waiter.join(timeout=1.0)
    assert woke_promptly is True
    assert wait_result == [True]
    persisted = json.loads(job.job_path.read_text(encoding="utf-8"))
    assert persisted["duration_s"] == 0.0
    assert job.ready_for_reentry is True
    assert [obs.output["job_id"] for obs in manager.pop_reentry_observations()] == [job_id]
    terminal = [event for event in sink.events if event.type == "job.finished"]
    assert len(terminal) == 1
    assert terminal[0].data["duration_s"] == 0.0


def test_hosted_task_duration_is_nonnegative_after_clock_rollback(tmp_path: Path) -> None:
    manager, _recorder, sink = _manager(tmp_path)
    task = manager.start_task("hitl", {"prompt": "Approve?", "resume_on_exit": True})
    task.started_at = 200.0
    task.finished_at = 100.0
    task.status = "answered"
    task.result = {"status": "answered"}

    manager.mark_ready(task)

    persisted = json.loads(task.job_path.read_text(encoding="utf-8"))
    assert persisted["duration_s"] == 0.0
    terminal = [event for event in sink.events if event.type == "task.finished"]
    assert len(terminal) == 1
    assert terminal[0].data["duration_s"] == 0.0


def test_terminal_event_failure_does_not_strand_reentry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, _recorder, _sink = _manager(tmp_path)
    task = manager.start_task("hitl", {"prompt": "Approve?", "resume_on_exit": True})
    task.status = "answered"
    task.finished_at = task.started_at
    task.result = {"status": "answered"}

    def fail_terminal_event(_job: Any) -> None:
        raise RuntimeError("event sink failed")

    monkeypatch.setattr(manager, "_emit_terminal_event", fail_terminal_event)
    with pytest.raises(RuntimeError, match="event sink failed"):
        manager.mark_ready(task)

    assert task.ready_for_reentry is True
    assert manager.has_resume_jobs() is True
    assert [obs.output["task_id"] for obs in manager.pop_reentry_observations()] == [task.job_id]


def test_hosted_task_cancel_marks_ready_for_reentry(tmp_path: Path) -> None:
    manager, _recorder, sink = _manager(tmp_path)
    task = manager.start_task("hitl", {"prompt": "Approve?", "resume_on_exit": True})

    result = manager.cancel(task.job_id)

    assert result["status"] == "cancelled"
    assert task.ready_for_reentry is True
    assert task.result is not None and task.result["status"] == "cancelled"
    observations = manager.pop_reentry_observations()
    assert observations
    assert observations[0].output["status"] == "cancelled"
    assert any(event.type == "task.cancelled" and event.data["task_id"] == task.job_id for event in sink.events)


def test_mark_ready_is_idempotent_for_cancelled_task(tmp_path: Path) -> None:
    manager, _recorder, sink = _manager(tmp_path)
    task = manager.start_task("hitl", {"prompt": "Approve?", "resume_on_exit": True})
    manager.cancel(task.job_id)

    manager.mark_ready(task)

    observations = manager.pop_reentry_observations()
    assert [obs.output["task_id"] for obs in observations] == [task.job_id]
    assert manager.pop_reentry_observations() == []
    cancelled_events = [
        event
        for event in sink.events
        if event.type == "task.cancelled" and event.data["task_id"] == task.job_id
    ]
    assert len(cancelled_events) == 1


def test_subagent_cancel_waits_for_child_coroutine_to_stop_before_reentry(tmp_path: Path) -> None:
    manager, _recorder, sink = _manager(tmp_path)
    started = threading.Event()
    cancellation_seen = threading.Event()

    async def run_child(_manager: TaskManager, _task) -> None:
        started.set()
        try:
            while True:
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            cancellation_seen.set()
            await asyncio.sleep(0.2)
            raise

    manager.executors["subagent"] = SubagentTaskExecutor(run_child=run_child)
    manager.injectors["subagent"] = manager.injectors["hitl"]
    task = manager.executors["subagent"].start(
        manager,
        definition_id="reviewer",
        prompt="keep working",
        background=True,
    )

    try:
        assert started.wait(timeout=2)
        result = manager.cancel(task.job_id)

        assert result["cancel_requested"] is True
        assert cancellation_seen.wait(timeout=2)
        assert task.ready_for_reentry is False
        assert manager.pop_reentry_observations() == []

        manager.wait(task.job_id, timeout_s=5)

        assert task.status == "cancelled"
        assert task.ready_for_reentry is True
        assert task.result is not None and task.result["status"] == "cancelled"
        observations = manager.pop_reentry_observations()
        assert observations
        assert observations[0].output["status"] == "cancelled"
        assert any(event.type == "task.cancelled" and event.data["task_id"] == task.job_id for event in sink.events)
    finally:
        manager._shutdown_task_loop()


def test_non_resume_job_is_not_offered_for_reentry(tmp_path: Path) -> None:
    manager, _recorder, _sink = _manager(tmp_path)
    job = _start(manager, _python_command("print('quiet')"), resume_on_exit=False)
    manager.wait(job.job_id, timeout_s=10)

    assert job.status == "exited"
    assert manager.has_resume_jobs() is False
    assert manager.pop_reentry_observations() == []


def test_terminal_event_emitted_on_completion(tmp_path: Path) -> None:
    manager, _recorder, sink = _manager(tmp_path)
    job = _start(manager, _python_command("print('evt')"))
    manager.wait(job.job_id, timeout_s=10)

    finished = [e for e in sink.events if e.type == "job.finished"]
    assert len(finished) == 1
    assert finished[0].data["job_id"] == job.job_id
    assert finished[0].data["status"] == "exited"
    # The public payload never leaks the raw command.
    assert "command" not in finished[0].data


@pytest.mark.slow
def test_timeout_transitions_to_timed_out(tmp_path: Path) -> None:
    manager, recorder, _sink = _manager(tmp_path)
    job = _start(manager, _python_command("import time; time.sleep(5)"), timeout_s=1)
    manager.wait(job.job_id, timeout_s=10)

    assert job.status == "timed_out"
    obs = job.result_observation(recorder.run_dir)
    assert obs["timed_out"] is True
    assert obs["status"] == "timed_out"


@pytest.mark.slow
def test_cancel_transitions_to_cancelled(tmp_path: Path) -> None:
    manager, _recorder, _sink = _manager(tmp_path)
    job = _start(manager, _python_command("import time; time.sleep(5)"), timeout_s=10)
    manager.cancel(job.job_id)
    manager.wait(job.job_id, timeout_s=10)

    assert job.status == "cancelled"
