"""Characterization tests pinning current TaskManager behavior.

These lock the shell background-job contract (status transitions, the
result_observation byte shape, artifact layout, reentry idempotency, and
terminal events) so the upcoming Task/TaskExecutor refactor stays behavior
preserving.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from native_agent_runner.core.events import AgentEvent
from native_agent_runner.tasks import BackgroundJob, TaskManager
from native_agent_runner.permissions import PermissionPolicy
from native_agent_runner.recorder import AgentRecorder
from native_agent_runner.shell import ShellExecutionOptions
from native_agent_runner.workspace.local import LocalWorkspaceBackend
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
    assert data["schema_version"] == "native-agent-runner.background-job.v1"
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
