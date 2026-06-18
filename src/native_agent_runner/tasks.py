from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from native_agent_runner._proc import file_size, spawn_process, terminate_process
from native_agent_runner.core._util import write_json_atomic
from native_agent_runner.errors import ToolExecutionError, WorkspaceError
from native_agent_runner.permissions import PermissionPolicy
from native_agent_runner.providers.base import ToolObservation
from native_agent_runner.public_view import public_path
from native_agent_runner.recorder import AgentRecorder
from native_agent_runner.shell import (
    ShellExecutionOptions,
    ResolvedShellExecutionWorkspace,
)
from native_agent_runner.core.workspace import Workspace
from native_agent_runner.workspace.paths import is_within, normalize_workspace_path

import native_agent_runner.shell as shell_runtime

BackgroundJobStatus = Literal[
    "running",
    "exited",
    "timed_out",
    "cancelled",
    "output_limited",
    "failed",
]


class TaskExecutor(Protocol):
    """Pluggable per-kind executor seam.

    The manager owns the queue/lifecycle/reentry; an executor owns "how does a
    task of this kind run and when is it done". The in-process shell executor
    monitors a subprocess; future executors (hitl, automation) may be driven
    entirely by an external reporter calling ``TaskManager.mark_ready``.
    """

    kind: str

    def cancel(self, manager: TaskManager, job: Task) -> None:
        ...


class ResultInjector(Protocol):
    """Pluggable per-kind seam deciding HOW a finished task is injected into the
    model: as a tool observation (``is_background=False``, keyed to a tool call)
    or as a new user message (``is_background=True``). This is the "appropriate
    way, defined by the backend developer"."""

    kind: str

    def observations(self, job: BackgroundJob, run_dir: Path) -> list[ToolObservation]:
        ...


class TaskReporter(Protocol):
    """The seam a backend uses to drive tasks in a running run: create a task and
    report its terminal result. Transport-agnostic — only plain ``(task_id, dict)``
    cross the boundary, so an in-process reporter (the live manager) and a future
    durable/cross-process reporter share the same shape."""

    def create_task(self, kind: str, request: dict[str, Any]) -> str:
        ...

    def report_result(self, task_id: str, result: dict[str, Any], *, status: str = "answered") -> dict[str, Any]:
        ...


@dataclass
class BackgroundJob:
    job_id: str
    kind: str
    command: str
    command_preview: str
    cwd: str
    status: BackgroundJobStatus
    started_at: float
    timeout_s: int
    max_output_bytes: int
    startup_wait_s: int
    stdout_path: Path
    stderr_path: Path
    job_path: Path
    cancel_path: Path
    execution_workspace: ResolvedShellExecutionWorkspace
    resume_on_exit: bool
    requested_timeout_s: int | None = None
    requested_max_output_bytes: int | None = None
    requested_startup_wait_s: int | None = None
    process: subprocess.Popen[bytes] | None = field(default=None, repr=False)
    finished_at: float | None = None
    exit_code: int | None = None
    timed_out: bool = False
    output_truncated: bool = False
    error: str = ""
    changed_paths: tuple[str, ...] = ()
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    tmp_root: Path | None = None
    before_snapshot: Any = field(default=None, repr=False)
    ready_for_reentry: bool = field(default=False, repr=False)
    _last_output_event_at: float = field(default=0.0, repr=False)
    _last_output_event_bytes: int = field(default=0, repr=False)

    @property
    def duration_s(self) -> float:
        return (self.finished_at or time.time()) - self.started_at

    def public_paths(self, permission_policy: PermissionPolicy) -> list[str]:
        return [public_path(path, permission_policy) for path in self.changed_paths]

    def stdout_relpath(self, run_dir: Path) -> str:
        return self.stdout_path.relative_to(run_dir).as_posix()

    def stderr_relpath(self, run_dir: Path) -> str:
        return self.stderr_path.relative_to(run_dir).as_posix()

    def to_json(self, run_dir: Path) -> dict[str, Any]:
        return {
            "schema_version": "native-agent-runner.background-job.v1",
            "job_id": self.job_id,
            "command": self.command,
            "command_preview": self.command_preview,
            "cwd": self.cwd,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "output_truncated": self.output_truncated,
            "error": self.error,
            "changed_paths": list(self.changed_paths),
            "stdout_path": self.stdout_relpath(run_dir),
            "stderr_path": self.stderr_relpath(run_dir),
            "stdout_bytes": self.stdout_bytes,
            "stderr_bytes": self.stderr_bytes,
            "requested_timeout_s": self.requested_timeout_s,
            "effective_timeout_s": self.timeout_s,
            "requested_max_output_bytes": self.requested_max_output_bytes,
            "effective_max_output_bytes": self.max_output_bytes,
            "requested_startup_wait_s": self.requested_startup_wait_s,
            "effective_startup_wait_s": self.startup_wait_s,
            "execution_workspace": self.execution_workspace,
            "resume_on_exit": self.resume_on_exit,
        }

    def started_content(self, run_dir: Path) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "command_preview": self.command_preview,
            "cwd": self.cwd,
            "stdout_path": self.stdout_relpath(run_dir),
            "stderr_path": self.stderr_relpath(run_dir),
            "resume_on_exit": self.resume_on_exit,
            "requested_timeout_s": self.requested_timeout_s,
            "effective_timeout_s": self.timeout_s,
            "requested_max_output_bytes": self.requested_max_output_bytes,
            "effective_max_output_bytes": self.max_output_bytes,
            "requested_startup_wait_s": self.requested_startup_wait_s,
            "effective_startup_wait_s": self.startup_wait_s,
            "execution_workspace": self.execution_workspace,
        }

    def terminal_event(self) -> tuple[str, str]:
        event_type = {
            "exited": "job.finished",
            "timed_out": "job.timed_out",
            "cancelled": "job.cancelled",
            "output_limited": "job.output_limited",
            "failed": "job.failed",
        }.get(self.status, "job.failed")
        level = "info" if self.status == "exited" else "warning"
        return event_type, level

    def public_payload(self, run_dir: Path, permission_policy: PermissionPolicy) -> dict[str, Any]:
        payload = self.to_json(run_dir)
        payload["changed_paths"] = self.public_paths(permission_policy)
        payload.pop("command", None)
        return payload

    def result_observation(self, run_dir: Path, *, tail_bytes: int = 8192) -> dict[str, Any]:
        stdout = read_job_log_text(run_dir, self.job_id, stream="stdout", tail_bytes=tail_bytes)
        stderr = read_job_log_text(run_dir, self.job_id, stream="stderr", tail_bytes=tail_bytes)
        return {
            "type": "background_job_result",
            "job_id": self.job_id,
            "command_preview": self.command_preview,
            "status": self.status,
            "exit_code": self.exit_code,
            "duration_s": self.duration_s,
            "stdout_tail": stdout["content"],
            "stderr_tail": stderr["content"],
            "stdout_path": self.stdout_relpath(run_dir),
            "stderr_path": self.stderr_relpath(run_dir),
            "stdout_bytes": self.stdout_bytes,
            "stderr_bytes": self.stderr_bytes,
            "timed_out": self.timed_out,
            "output_truncated": self.output_truncated,
            "effective_timeout_s": self.timeout_s,
            "effective_max_output_bytes": self.max_output_bytes,
            "changed_paths": list(self.changed_paths),
            "error": self.error,
        }


@dataclass
class ShellTaskExecutor:
    """In-process executor for shell tasks: spawn a subprocess, monitor it, and
    publish completion through ``TaskManager.mark_ready``."""

    kind: str = "shell"

    def start(
        self,
        manager: TaskManager,
        *,
        shell_options: ShellExecutionOptions,
        command: str,
        cwd: str,
        timeout_s: int,
        max_output_bytes: int,
        startup_wait_s: int,
        env: dict[str, Any],
        requested_timeout_s: int | None,
        requested_max_output_bytes: int | None,
        requested_startup_wait_s: int | None,
        execution_workspace: ResolvedShellExecutionWorkspace,
        resume_on_exit: bool,
    ) -> BackgroundJob:
        if not shell_options.enabled:
            raise ToolExecutionError("shell is disabled", error_code="shell_disabled")
        if not command.strip():
            raise ToolExecutionError("shell command is required", error_code="shell_exec_error")
        shell_options.check_command(command)
        cwd_rel = shell_runtime.validate_cwd(manager.workspace, cwd, manager.permission_policy)
        safe_env = shell_runtime.build_env(shell_options, env)
        argv = shell_runtime.shell_argv(shell_options.effective_shell(), command)
        cwd_abs, tmp_root, before_snapshot = self._prepare_workspace(manager, cwd_rel, execution_workspace)

        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job_dir = manager.recorder.artifacts_dir / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        job = BackgroundJob(
            job_id=job_id,
            kind=self.kind,
            command=command,
            command_preview=shell_runtime.preview_command(command),
            cwd=cwd_rel,
            status="running",
            started_at=time.time(),
            timeout_s=timeout_s,
            max_output_bytes=max_output_bytes,
            startup_wait_s=startup_wait_s,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            job_path=job_dir / "job.json",
            cancel_path=job_dir / "cancel.requested",
            execution_workspace=execution_workspace,
            resume_on_exit=resume_on_exit,
            requested_timeout_s=requested_timeout_s,
            requested_max_output_bytes=requested_max_output_bytes,
            requested_startup_wait_s=requested_startup_wait_s,
            tmp_root=tmp_root,
            before_snapshot=before_snapshot,
        )
        try:
            job.process = _spawn_process(argv, cwd=cwd_abs, env=safe_env, stdout_path=stdout_path, stderr_path=stderr_path)
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = time.time()
            manager._write_job(job)
            self._cleanup_tmp(job)
            raise ToolExecutionError(str(exc), error_code="shell_exec_error") from exc

        manager._register(job)
        manager.recorder.emit("job.started", data=manager._public_job_payload(job))
        thread = threading.Thread(
            target=self._monitor_job,
            args=(manager, job_id),
            name=f"native-agent-job-{job_id}",
            daemon=True,
        )
        thread.start()
        if startup_wait_s > 0:
            manager._wait_startup(job_id, startup_wait_s)
        return job

    def cancel(self, manager: TaskManager, job: BackgroundJob) -> None:
        # Called under manager._condition. Terminate the live subprocess.
        if job.process is not None and job.status == "running":
            job.status = "cancelled"
            terminate_process(job.process)

    def _prepare_workspace(
        self,
        manager: TaskManager,
        cwd_rel: str,
        execution_workspace: ResolvedShellExecutionWorkspace,
    ) -> tuple[Path, Path | None, Any]:
        if execution_workspace == "direct":
            cwd_abs = (manager.workspace.root / cwd_rel).resolve()
            if not is_within(manager.workspace.root, cwd_abs):
                raise WorkspaceError(f"shell cwd escapes workspace: {cwd_rel}")
            if not cwd_abs.exists() or not cwd_abs.is_dir():
                raise WorkspaceError(f"shell cwd is not a directory: {cwd_rel}")
            return cwd_abs, None, None

        tmp_root = Path(tempfile.mkdtemp(prefix="native-agent-shell-job-")).resolve()
        before = shell_runtime.materialize_workspace(manager.workspace, tmp_root, manager.permission_policy)
        cwd_abs = (tmp_root / cwd_rel).resolve()
        if not is_within(tmp_root, cwd_abs):
            raise WorkspaceError(f"shell cwd escapes workspace: {cwd_rel}")
        if not cwd_abs.exists() or not cwd_abs.is_dir():
            raise WorkspaceError(f"shell cwd is not a directory: {cwd_rel}")
        return cwd_abs, tmp_root, before

    def _monitor_job(self, manager: TaskManager, job_id: str) -> None:
        job = manager.get_job(job_id)
        try:
            self._monitor_process(manager, job)
            if job.execution_workspace == "isolated-copy" and job.status == "exited":
                after = shell_runtime.scan_materialized_workspace(job.tmp_root, manager.permission_policy)
                changed = shell_runtime.sync_workspace_changes(manager.workspace, job.before_snapshot, after)
                job.changed_paths = tuple(changed)
            elif job.execution_workspace == "direct":
                job.changed_paths = tuple(manager.workspace.changed_paths())
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = time.time()
        finally:
            job.stdout_bytes = file_size(job.stdout_path)
            job.stderr_bytes = file_size(job.stderr_path)
            if job.finished_at is None:
                job.finished_at = time.time()
            self._cleanup_tmp(job)
            manager.mark_ready(job)

    def _monitor_process(self, manager: TaskManager, job: BackgroundJob) -> None:
        process = job.process
        if process is None:
            job.status = "failed"
            job.error = "process was not started"
            return
        while process.poll() is None:
            now = time.time()
            stdout_bytes = file_size(job.stdout_path)
            stderr_bytes = file_size(job.stderr_path)
            total_bytes = stdout_bytes + stderr_bytes
            if job.cancel_path.exists():
                job.status = "cancelled"
                terminate_process(process)
                break
            if now - job.started_at >= job.timeout_s:
                job.status = "timed_out"
                job.timed_out = True
                terminate_process(process)
                break
            if total_bytes > job.max_output_bytes:
                job.status = "output_limited"
                job.output_truncated = True
                terminate_process(process)
                break
            if total_bytes != job._last_output_event_bytes and now - job._last_output_event_at >= 0.25:
                job.stdout_bytes = stdout_bytes
                job.stderr_bytes = stderr_bytes
                job._last_output_event_at = now
                job._last_output_event_bytes = total_bytes
                manager.recorder.emit("job.output.updated", data=manager._public_job_payload(job))
                manager._write_job(job)
            time.sleep(0.02)
        try:
            job.exit_code = process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            job.exit_code = process.wait(timeout=2)
        if job.status == "running":
            job.status = "exited"
        job.stdout_bytes = file_size(job.stdout_path)
        job.stderr_bytes = file_size(job.stderr_path)
        if job.stdout_bytes + job.stderr_bytes > job.max_output_bytes:
            job.output_truncated = True
            if job.status == "exited":
                job.status = "output_limited"

    def _cleanup_tmp(self, job: BackgroundJob) -> None:
        if job.tmp_root is not None:
            shutil.rmtree(job.tmp_root, ignore_errors=True)
            job.tmp_root = None


@dataclass
class ShellResultInjector:
    """Inject a finished shell job as a background observation (user-message
    shape: ``is_background=True``), preserving the legacy ``background_job``
    observation contract."""

    kind: str = "shell"

    def observations(self, job: BackgroundJob, run_dir: Path) -> list[ToolObservation]:
        return [
            ToolObservation(
                call_id=f"background:{job.job_id}",
                tool_name="background_job",
                output=job.result_observation(run_dir),
                is_background=True,
            )
        ]


@dataclass
class HitlTask:
    """A human-in-the-loop task: created when the agent requests human input,
    parked until an external reporter calls ``report_result``. It has no
    in-process monitor — the reporter (backend/test) is the only completion writer.
    Carries ``job_id`` (not ``task_id``) so it flows through the manager's generic
    queue unchanged; the rename happens in the final pass."""

    job_id: str
    kind: str
    prompt: str
    status: str
    started_at: float
    resume_on_exit: bool
    job_path: Path
    cancel_path: Path
    created_by: str = "model"
    choices: tuple[str, ...] = ()
    finished_at: float | None = None
    error: str = ""
    result: dict[str, Any] | None = None
    ready_for_reentry: bool = field(default=False, repr=False)

    @property
    def duration_s(self) -> float:
        return (self.finished_at or time.time()) - self.started_at

    def to_json(self, run_dir: Path) -> dict[str, Any]:
        del run_dir
        return {
            "schema_version": "native-agent-runner.task.v1",
            "task_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "created_by": self.created_by,
            "prompt": self.prompt,
            "choices": list(self.choices),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "error": self.error,
            "result": self.result,
        }

    def started_content(self, run_dir: Path) -> dict[str, Any]:
        del run_dir
        return {
            "task_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "prompt": self.prompt,
            "choices": list(self.choices),
        }

    def terminal_event(self) -> tuple[str, str]:
        event_type = {
            "answered": "task.finished",
            "cancelled": "task.cancelled",
            "timed_out": "task.timed_out",
        }.get(self.status, "task.failed")
        level = "info" if self.status == "answered" else "warning"
        return event_type, level

    def public_payload(self, run_dir: Path, permission_policy: PermissionPolicy) -> dict[str, Any]:
        del permission_policy
        return self.to_json(run_dir)

    def result_observation(self, run_dir: Path, *, tail_bytes: int = 8192) -> dict[str, Any]:
        del run_dir, tail_bytes
        return self.result or {"type": "human_input_result", "task_id": self.job_id, "status": self.status}


@dataclass
class HitlTaskExecutor:
    """In-process executor for human-in-the-loop tasks: register a parked task
    and wait for an external ``report_result``. No monitor thread."""

    kind: str = "hitl"

    def start(
        self,
        manager: TaskManager,
        *,
        prompt: str,
        choices: tuple[str, ...] = (),
        created_by: str = "model",
        resume_on_exit: bool = True,
    ) -> HitlTask:
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        task_dir = manager.recorder.artifacts_dir / "tasks" / task_id
        task_dir.mkdir(parents=True, exist_ok=False)
        task = HitlTask(
            job_id=task_id,
            kind=self.kind,
            prompt=str(prompt),
            choices=tuple(str(choice) for choice in choices),
            status="running",
            started_at=time.time(),
            resume_on_exit=resume_on_exit,
            created_by=created_by,
            job_path=task_dir / "task.json",
            cancel_path=task_dir / "cancel.requested",
        )
        manager._register(task)
        manager.recorder.emit("task.started", data=manager._public_job_payload(task))
        return task

    def cancel(self, manager: TaskManager, job: HitlTask) -> None:
        del manager
        if job.status == "running":
            job.status = "cancelled"
            job.finished_at = time.time()


@dataclass
class HitlResultInjector:
    """Inject a human answer. Default ``is_background=True`` renders it as a new
    user message; flip to ``False`` to deliver it as a tool result keyed to the
    originating ``hitl.request`` call (both shapes supported, per the backend)."""

    kind: str = "hitl"
    as_user_message: bool = True

    def observations(self, job: HitlTask, run_dir: Path) -> list[ToolObservation]:
        del run_dir
        result = dict(job.result or {})
        answer = str(result.get("answer", ""))
        message = result.get("message") or (
            f"Human responded to your request ({job.prompt!r}): {answer}"
            if answer
            else f"Human input task {job.job_id} finished with status {job.status}."
        )
        output = {
            "type": "human_input_result",
            "task_id": job.job_id,
            "status": job.status,
            "message": message,
            **result,
        }
        call_id = f"task:{job.job_id}" if self.as_user_message else f"hitl:{job.job_id}"
        return [
            ToolObservation(
                call_id=call_id,
                tool_name="human_input",
                output=output,
                is_background=self.as_user_message,
            )
        ]


Task = BackgroundJob | HitlTask


@dataclass
class TaskManager:
    run_id: str
    workspace: Workspace
    recorder: AgentRecorder
    permission_policy: PermissionPolicy
    jobs: dict[str, Task] = field(default_factory=dict)
    executors: dict[str, TaskExecutor] = field(default_factory=dict, init=False, repr=False)
    injectors: dict[str, ResultInjector] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _condition: threading.Condition = field(init=False, repr=False)
    _reentry_queue: list[str] = field(default_factory=list, init=False, repr=False)
    _delivered_reentry_jobs: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self._condition = threading.Condition(self._lock)
        self.executors = {"shell": ShellTaskExecutor(), "hitl": HitlTaskExecutor()}
        self.injectors = {"shell": ShellResultInjector(), "hitl": HitlResultInjector()}

    def _register(self, job: Task) -> None:
        with self._condition:
            self.jobs[job.job_id] = job
            self._write_job(job)

    def start_task(self, kind: str, request: dict[str, Any]) -> Task:
        """Generic task creation, callable from a tool handler or the backend.

        The executor for ``kind`` decides how the task runs (in-process monitor or
        parked for an external reporter)."""
        executor = self.executors.get(kind)
        if executor is None:
            raise ToolExecutionError(f"no executor for task kind: {kind}", error_code="task_kind_unknown")
        return executor.start(self, **request)  # type: ignore[attr-defined]

    def create_task(self, kind: str, request: dict[str, Any]) -> str:
        """TaskReporter entry: create a task and return its id (backend-initiated)."""
        return self.start_task(kind, request).job_id

    def report_result(self, task_id: str, result: dict[str, Any], *, status: str = "answered") -> dict[str, Any]:
        """External completion entry for hosted tasks (hitl/automation): set the
        terminal status/result and publish it through the shared reentry pipe."""
        task = self.get_job(task_id)
        task.status = status  # type: ignore[assignment]
        task.finished_at = time.time()
        task.result = result  # type: ignore[attr-defined]
        self.mark_ready(task)
        return {"task_id": task_id, "status": status, "delivered": True}

    def mark_ready(self, job: BackgroundJob) -> None:
        """Single completion entry: publish a finished task to the reentry queue.

        Called by the in-process shell monitor and (for hosted kinds) by an
        external reporter that has already set ``status``/``result``/``finished_at``."""
        job.ready_for_reentry = True
        with self._condition:
            self._write_job(job)
            self._emit_terminal_event(job)
            if job.resume_on_exit:
                self._reentry_queue.append(job.job_id)
            self._condition.notify_all()

    def start_shell_job(
        self,
        *,
        shell_options: ShellExecutionOptions,
        command: str,
        cwd: str,
        timeout_s: int,
        max_output_bytes: int,
        startup_wait_s: int,
        env: dict[str, Any],
        requested_timeout_s: int | None,
        requested_max_output_bytes: int | None,
        requested_startup_wait_s: int | None,
        execution_workspace: ResolvedShellExecutionWorkspace,
        resume_on_exit: bool,
    ) -> BackgroundJob:
        executor = self.executors["shell"]
        return executor.start(  # type: ignore[attr-defined]
            self,
            shell_options=shell_options,
            command=command,
            cwd=cwd,
            timeout_s=timeout_s,
            max_output_bytes=max_output_bytes,
            startup_wait_s=startup_wait_s,
            env=env,
            requested_timeout_s=requested_timeout_s,
            requested_max_output_bytes=requested_max_output_bytes,
            requested_startup_wait_s=requested_startup_wait_s,
            execution_workspace=execution_workspace,
            resume_on_exit=resume_on_exit,
        )

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._public_job_payload(job) for job in self.jobs.values()]

    def get_job(self, job_id: str) -> BackgroundJob:
        with self._lock:
            try:
                return self.jobs[job_id]
            except KeyError as exc:
                raise KeyError(f"unknown job: {job_id}") from exc

    def status(self, job_id: str) -> dict[str, Any]:
        return self._public_job_payload(self.get_job(job_id))

    def logs(
        self,
        job_id: str,
        *,
        stream: Literal["stdout", "stderr"] = "stdout",
        tail_bytes: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        self.get_job(job_id)
        return read_job_log_text(
            self.recorder.run_dir,
            job_id,
            stream=stream,
            tail_bytes=tail_bytes,
            offset=offset,
        )

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        job.cancel_path.write_text("cancel requested\n", encoding="utf-8")
        with self._condition:
            executor = self.executors.get(job.kind)
            if executor is not None:
                executor.cancel(self, job)
            self._condition.notify_all()
        return {"job_id": job_id, "cancel_requested": True, "status": job.status}

    def wait(self, job_id: str, timeout_s: int | None = None) -> dict[str, Any]:
        deadline = None if timeout_s is None else time.time() + max(0, timeout_s)
        with self._condition:
            if job_id not in self.jobs:
                raise KeyError(f"unknown job: {job_id}")
            while self.jobs[job_id].status == "running":
                remaining = None if deadline is None else deadline - time.time()
                if remaining is not None and remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            job = self.jobs[job_id]
        return job.result_observation(self.recorder.run_dir)

    def has_resume_jobs(self) -> bool:
        with self._lock:
            return bool(self._reentry_queue) or any(
                job.resume_on_exit
                and job.job_id not in self._delivered_reentry_jobs
                for job in self.jobs.values()
            )

    def pop_reentry_observations(self) -> list[ToolObservation]:
        """Drain finished resume-tasks and render them through their per-kind
        ResultInjector. The injector decides the injection shape (tool observation
        vs user message via ``is_background``)."""
        with self._condition:
            for job in self.jobs.values():
                if (
                    job.resume_on_exit
                    and job.ready_for_reentry
                    and job.job_id not in self._delivered_reentry_jobs
                    and job.job_id not in self._reentry_queue
                ):
                    self._reentry_queue.append(job.job_id)
            job_ids = list(dict.fromkeys(self._reentry_queue))
            self._reentry_queue.clear()
            observations: list[ToolObservation] = []
            for job_id in job_ids:
                job = self.jobs.get(job_id)
                if job is None:
                    continue
                injector = self.injectors.get(job.kind)
                if injector is None:
                    continue
                observations.extend(injector.observations(job, self.recorder.run_dir))
            self._delivered_reentry_jobs.update(job_id for job_id in job_ids if job_id in self.jobs)
            return observations

    def wait_for_reentry(self, timeout_s: float) -> bool:
        with self._condition:
            if self._reentry_queue:
                return True
            self._condition.wait(timeout=max(0.0, timeout_s))
            return bool(self._reentry_queue)

    def cancel_all(self) -> None:
        with self._condition:
            job_ids = list(self.jobs)
        for job_id in job_ids:
            job = self.get_job(job_id)
            if job.status == "running":
                self.cancel(job_id)
        deadline = time.time() + 5
        while time.time() < deadline:
            with self._condition:
                if all(job.status != "running" for job in self.jobs.values()):
                    return
            time.sleep(0.05)

    def _wait_startup(self, job_id: str, startup_wait_s: int) -> None:
        deadline = time.time() + startup_wait_s
        while time.time() < deadline:
            with self._lock:
                job = self.jobs[job_id]
                if job.status != "running":
                    return
                if file_size(job.stdout_path) + file_size(job.stderr_path) > 0:
                    return
            time.sleep(0.02)

    def _emit_terminal_event(self, job: BackgroundJob) -> None:
        event_type, level = job.terminal_event()
        self.recorder.emit(event_type, data=self._public_job_payload(job), level=level)

    def _public_job_payload(self, job: BackgroundJob) -> dict[str, Any]:
        return job.public_payload(self.recorder.run_dir, self.permission_policy)

    def _write_job(self, job: BackgroundJob) -> None:
        write_json_atomic(job.job_path, job.to_json(self.recorder.run_dir))


def list_job_artifacts(run_dir: Path) -> list[dict[str, Any]]:
    jobs_dir = run_dir / "artifacts" / "jobs"
    if not jobs_dir.exists():
        return []
    jobs: list[dict[str, Any]] = []
    for path in sorted(jobs_dir.glob("*/job.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            jobs.append(payload)
    return jobs


def get_job_artifact(run_dir: Path, job_id: str) -> dict[str, Any]:
    path = _job_dir(run_dir, job_id) / "job.json"
    if not path.exists():
        raise KeyError(f"unknown job: {job_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("job artifact is invalid")
    return payload


def read_job_log_text(
    run_dir: Path,
    job_id: str,
    *,
    stream: Literal["stdout", "stderr"] = "stdout",
    tail_bytes: int | None = None,
    offset: int | None = None,
) -> dict[str, Any]:
    if stream not in {"stdout", "stderr"}:
        raise ValueError("stream must be stdout or stderr")
    path = _job_dir(run_dir, job_id) / f"{stream}.log"
    if not path.exists():
        raise KeyError(f"{stream} log not found for job: {job_id}")
    size = path.stat().st_size
    start = 0
    if offset is not None:
        start = max(0, min(int(offset), size))
    elif tail_bytes is not None:
        start = max(0, size - max(0, int(tail_bytes)))
    data = path.read_bytes()[start:]
    return {
        "job_id": job_id,
        "stream": stream,
        "offset": start,
        "next_offset": size,
        "bytes": len(data),
        "total_bytes": size,
        "content": data.decode("utf-8", errors="replace"),
    }


def request_job_cancel(run_dir: Path, job_id: str) -> dict[str, Any]:
    job_dir = _job_dir(run_dir, job_id)
    if not job_dir.exists():
        raise KeyError(f"unknown job: {job_id}")
    cancel_path = job_dir / "cancel.requested"
    cancel_path.write_text("cancel requested\n", encoding="utf-8")
    return {"job_id": job_id, "cancel_requested": True}


def _job_dir(run_dir: Path, job_id: str) -> Path:
    rel = normalize_workspace_path(job_id)
    if "/" in rel or rel.startswith("."):
        raise ValueError("invalid job id")
    path = (run_dir / "artifacts" / "jobs" / rel).resolve()
    if not is_within(run_dir.resolve(), path):
        raise ValueError("job path escapes run directory")
    return path


def _spawn_process(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> subprocess.Popen[bytes]:
    stdout_handle = stdout_path.open("ab")
    stderr_handle = stderr_path.open("ab")
    try:
        return spawn_process(
            argv, cwd=cwd, env=env, stdout=stdout_handle, stderr=stderr_handle
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()
