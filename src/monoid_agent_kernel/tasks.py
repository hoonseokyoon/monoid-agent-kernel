from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from monoid_agent_kernel._proc import file_size, proc_group_kwargs, terminate_process
from monoid_agent_kernel.core._util import write_json_atomic
from monoid_agent_kernel.errors import ToolExecutionError, WorkspaceError
from monoid_agent_kernel.identifiers import namespaced_id
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.providers.base import ToolObservation
from monoid_agent_kernel.public_view import public_path
from monoid_agent_kernel.recorder import AgentRecorder
from monoid_agent_kernel.shell import (
    ShellExecutionOptions,
    ResolvedShellExecutionWorkspace,
)
from monoid_agent_kernel.core.workspace import Workspace
from monoid_agent_kernel.workspace.paths import is_within, normalize_workspace_path

import monoid_agent_kernel.shell as shell_runtime

# Upper bound on awaiting a process exit after termination, so a Windows reap race can never
# block the shared job loop indefinitely (see ``ShellTaskExecutor._terminate_and_reap``).
_REAP_TIMEOUT_S = 10.0

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
    # True for kinds that complete themselves in-process (the shell monitor
    # thread); False for hosted kinds (hitl/automation) an external reporter
    # completes. The loop uses this to know whether a parked run awaits external
    # input vs an in-process job that will finish on its own.
    in_process: bool

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
            "schema_version": namespaced_id("background-job.v1"),
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
    in_process: bool = True

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
        # Spawn the subprocess on the always-on job loop and wait (in this offloaded worker
        # thread) for it to start, so a spawn failure still surfaces synchronously as a
        # ToolExecutionError. The monitor then runs as a background coroutine on that loop.
        try:
            job.process = manager.schedule_job_coroutine(
                self._aspawn(argv, cwd_abs, safe_env, stdout_path, stderr_path)
            ).result()
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = time.time()
            manager._write_job(job)
            self._cleanup_tmp(job)
            raise ToolExecutionError(str(exc), error_code="shell_exec_error") from exc

        manager._register(job)
        manager.recorder.emit("job.started", data=manager._public_job_payload(job))
        # Monitor the subprocess on the same always-on loop — no per-job thread, no poll.
        manager.schedule_job_coroutine(self._amonitor(manager, job_id))
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

        tmp_root = Path(tempfile.mkdtemp(prefix="monoid-shell-job-")).resolve()
        before = shell_runtime.materialize_workspace(manager.workspace, tmp_root, manager.permission_policy)
        cwd_abs = (tmp_root / cwd_rel).resolve()
        if not is_within(tmp_root, cwd_abs):
            raise WorkspaceError(f"shell cwd escapes workspace: {cwd_rel}")
        if not cwd_abs.exists() or not cwd_abs.is_dir():
            raise WorkspaceError(f"shell cwd is not a directory: {cwd_rel}")
        return cwd_abs, tmp_root, before

    async def _aspawn(
        self,
        argv: list[str],
        cwd: Path,
        env: dict[str, str],
        stdout_path: Path,
        stderr_path: Path,
    ) -> asyncio.subprocess.Process:
        """Create the subprocess on the job loop, redirecting output to the job's log files
        (the child keeps its own fds, so the handles are closed once it has started)."""
        stdout_handle = stdout_path.open("ab")
        stderr_handle = stderr_path.open("ab")
        try:
            return await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(cwd),
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                **proc_group_kwargs(),  # type: ignore[arg-type]
            )
        finally:
            stdout_handle.close()
            stderr_handle.close()

    async def _amonitor(self, manager: TaskManager, job_id: str) -> None:
        job = manager.get_job(job_id)
        try:
            proc = job.process
            if proc is None:
                job.status = "failed"
                job.error = "process was not started"
            else:
                await self._await_completion(manager, job, proc)
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

    async def _await_completion(
        self, manager: TaskManager, job: BackgroundJob, proc: asyncio.subprocess.Process
    ) -> None:
        # Await exit, waking every 0.25s only to check cancel/timeout/output (event-driven on
        # exit — no 20ms busy poll). terminate_process is offloaded so taskkill/killpg never
        # blocks the shared job loop.
        while True:
            try:
                job.exit_code = await asyncio.wait_for(proc.wait(), timeout=0.25)
                break
            except asyncio.TimeoutError:
                pass
            now = time.time()
            stdout_bytes = file_size(job.stdout_path)
            stderr_bytes = file_size(job.stderr_path)
            total_bytes = stdout_bytes + stderr_bytes
            if job.cancel_path.exists():
                job.status = "cancelled"
                job.exit_code = await self._terminate_and_reap(proc)
                break
            if now - job.started_at >= job.timeout_s:
                job.status = "timed_out"
                job.timed_out = True
                job.exit_code = await self._terminate_and_reap(proc)
                break
            if total_bytes > job.max_output_bytes:
                job.status = "output_limited"
                job.output_truncated = True
                job.exit_code = await self._terminate_and_reap(proc)
                break
            if total_bytes != job._last_output_event_bytes and now - job._last_output_event_at >= 0.25:
                job.stdout_bytes = stdout_bytes
                job.stderr_bytes = stderr_bytes
                job._last_output_event_at = now
                job._last_output_event_bytes = total_bytes
                manager.recorder.emit("job.output.updated", data=manager._public_job_payload(job))
                manager._write_job(job)
        if job.status == "running":
            job.status = "exited"
        job.stdout_bytes = file_size(job.stdout_path)
        job.stderr_bytes = file_size(job.stderr_path)
        if job.stdout_bytes + job.stderr_bytes > job.max_output_bytes:
            job.output_truncated = True
            if job.status == "exited":
                job.status = "output_limited"

    async def _terminate_and_reap(self, proc: asyncio.subprocess.Process) -> int | None:
        """Terminate the process group and wait for it, bounded. ``terminate_process`` is
        offloaded so a stalled killer never blocks the shared job loop; the reap itself is
        capped, escalating to a direct ``kill`` and finally giving up (returns the last-known
        ``returncode``) rather than awaiting ``proc.wait()`` forever — a Windows reap race that
        otherwise leaves this monitor pending indefinitely."""
        await asyncio.to_thread(terminate_process, proc)
        for _ in range(2):
            try:
                return await asyncio.wait_for(proc.wait(), timeout=_REAP_TIMEOUT_S)
            except asyncio.TimeoutError:
                with contextlib.suppress(Exception):
                    proc.kill()
        return proc.returncode

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
class HostedTask:
    """A hosted task (human-in-the-loop or automation): work delegated outside the
    kernel and parked until an external reporter calls ``report_result``. No
    in-process monitor — the reporter (operator/external system) is the only
    completion writer. Carries ``job_id`` so it flows through the manager's generic
    queue alongside shell ``BackgroundJob``s."""

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
    request: dict[str, Any] = field(default_factory=dict)
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
            "schema_version": namespaced_id("task.v1"),
            "task_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "created_by": self.created_by,
            "prompt": self.prompt,
            "choices": list(self.choices),
            "request": self.request,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "error": self.error,
            "result": self.result,
        }

    def checkpoint_json(self) -> dict[str, Any]:
        """Durable snapshot for ``run_dir/checkpoint.json``. Unlike ``to_json``
        (which feeds the public ``task.json``), this carries the fields needed to
        rebuild a parked task on restore: ``resume_on_exit`` and ``ready_for_reentry``.
        ``job_path``/``cancel_path`` are derived from ``artifacts_dir`` in
        ``from_checkpoint`` rather than stored (they are absolute and host-relative)."""
        return {
            "task_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "created_by": self.created_by,
            "prompt": self.prompt,
            "choices": list(self.choices),
            "request": self.request,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "resume_on_exit": self.resume_on_exit,
            "error": self.error,
            "result": self.result,
            "ready_for_reentry": self.ready_for_reentry,
        }

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any], artifacts_dir: Path) -> HostedTask:
        """Rebuild a parked hosted task from a checkpoint payload. The task dir
        layout matches ``HostedTaskExecutor.start``: ``artifacts_dir/tasks/<id>/``."""
        task_id = str(payload.get("task_id") or "")
        task_dir = artifacts_dir / "tasks" / task_id
        return cls(
            job_id=task_id,
            kind=str(payload.get("kind") or ""),
            prompt=str(payload.get("prompt") or ""),
            status=str(payload.get("status") or "running"),
            started_at=float(payload.get("started_at") or 0.0),
            resume_on_exit=bool(payload.get("resume_on_exit", True)),
            job_path=task_dir / "task.json",
            cancel_path=task_dir / "cancel.requested",
            created_by=str(payload.get("created_by") or "model"),
            choices=tuple(str(choice) for choice in payload.get("choices") or ()),
            request=dict(payload.get("request") or {}),
            finished_at=payload.get("finished_at"),
            error=str(payload.get("error") or ""),
            result=payload.get("result"),
            ready_for_reentry=bool(payload.get("ready_for_reentry", False)),
        )

    def started_content(self, run_dir: Path) -> dict[str, Any]:
        del run_dir
        return {
            "task_id": self.job_id,
            "kind": self.kind,
            "status": self.status,
            "prompt": self.prompt,
            "choices": list(self.choices),
            "request": self.request,
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
        return self.result or {"type": f"{self.kind}_result", "task_id": self.job_id, "status": self.status}


@dataclass
class HostedTaskExecutor:
    """In-process executor for hosted (hitl/automation) tasks: register a parked
    task and wait for an external ``report_result``. No monitor thread."""

    kind: str
    in_process: bool = False

    def start(
        self,
        manager: TaskManager,
        *,
        prompt: str = "",
        choices: tuple[str, ...] = (),
        created_by: str = "model",
        resume_on_exit: bool = True,
        **request: Any,
    ) -> HostedTask:
        # Known fields are explicit; any other keys (e.g. an automation trigger
        # payload) are folded into the task's request.
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        task_dir = manager.recorder.artifacts_dir / "tasks" / task_id
        task_dir.mkdir(parents=True, exist_ok=False)
        task = HostedTask(
            job_id=task_id,
            kind=self.kind,
            prompt=str(prompt),
            choices=tuple(str(choice) for choice in choices),
            request=dict(request),
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

    def cancel(self, manager: TaskManager, job: HostedTask) -> None:
        del manager
        if job.status == "running":
            job.status = "cancelled"
            job.finished_at = time.time()


@dataclass
class HostedResultInjector:
    """Inject a hosted-task result. ``as_user_message=True`` (default) renders it as
    a new user message; ``False`` delivers it as a tool result keyed to the
    originating call (both shapes supported, chosen per kind by the integrator).
    hitl defaults to a user message; automation reads as an async tool result."""

    kind: str
    tool_name: str
    result_type: str
    as_user_message: bool = True

    def observations(self, job: HostedTask, run_dir: Path) -> list[ToolObservation]:
        del run_dir
        result = dict(job.result or {})
        answer = str(result.get("answer", ""))
        detail = job.prompt or str(job.request.get("description") or "") or job.job_id
        message = result.get("message") or (
            f"Result for {self.kind} task ({detail!r}): {answer}"
            if answer
            else f"{self.kind} task {job.job_id} finished with status {job.status}."
        )
        output = {
            "type": self.result_type,
            "task_id": job.job_id,
            "status": job.status,
            "message": message,
            **result,
        }
        call_id = f"task:{job.job_id}" if self.as_user_message else f"{self.kind}:{job.job_id}"
        return [
            ToolObservation(
                call_id=call_id,
                tool_name=self.tool_name,
                output=output,
                is_background=self.as_user_message,
            )
        ]


@dataclass
class SubagentTaskExecutor:
    """In-process executor for agent-as-tool delegation. Registers a parked
    ``HostedTask`` and schedules ``run_child`` (a closure built by the parent
    ``AgentLoop`` that runs an isolated child run) on the always-on job loop. On
    completion the coroutine sets the task's status/result and publishes it through
    ``TaskManager.mark_ready`` — the same reentry pipe shell/hosted tasks use.

    Foreground spawns pass ``resume_on_exit=False`` and the tool handler blocks on
    ``TaskManager.wait`` to read the child's final message directly; background
    spawns pass ``resume_on_exit=True`` and the result is injected later as a user
    message via ``HostedResultInjector`` (the reentry queue drains many at once, so
    several background subagents run concurrently for free)."""

    run_child: Callable[[TaskManager, HostedTask], Awaitable[None]]
    definition_ids: tuple[str, ...] = ()
    max_depth: int = 5
    max_subagents: int = 8
    kind: str = "subagent"
    in_process: bool = True

    def start(
        self,
        manager: TaskManager,
        *,
        definition_id: str = "",
        prompt: str = "",
        depth: int = 0,
        background: bool = False,
        resume_on_exit: bool | None = None,
        created_by: str = "model",
        **request: Any,
    ) -> HostedTask:
        if not definition_id:
            raise ToolExecutionError(
                "subagent definition_id is required", error_code="subagent_invalid"
            )
        if self.definition_ids and definition_id not in self.definition_ids:
            raise ToolExecutionError(
                f"unknown subagent: {definition_id}", error_code="subagent_unknown"
            )
        if depth >= self.max_depth:
            raise ToolExecutionError(
                f"subagent depth cap reached (max {self.max_depth})",
                error_code="subagent_depth_exceeded",
            )
        active = sum(1 for job in manager.jobs.values() if job.kind == self.kind)
        if self.max_subagents and active >= self.max_subagents:
            raise ToolExecutionError(
                f"subagent fan-out cap reached (max {self.max_subagents})",
                error_code="subagent_fanout_exceeded",
            )
        resume = bool(background) if resume_on_exit is None else bool(resume_on_exit)
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        task_dir = manager.recorder.artifacts_dir / "tasks" / task_id
        task_dir.mkdir(parents=True, exist_ok=False)
        task = HostedTask(
            job_id=task_id,
            kind=self.kind,
            prompt=str(prompt),
            status="running",
            started_at=time.time(),
            resume_on_exit=resume,
            request={
                "definition_id": definition_id,
                "depth": int(depth),
                "background": bool(background),
                **request,
            },
            created_by=created_by,
            job_path=task_dir / "task.json",
            cancel_path=task_dir / "cancel.requested",
        )
        manager._register(task)
        manager.recorder.emit("task.started", data=manager._public_job_payload(task))
        # Run the child on the always-on job loop; the worker thread that called
        # start() (a tool handler offloaded via to_thread) is free to block on
        # wait() for a foreground spawn without stalling the loop.
        manager.schedule_job_coroutine(self._arun(manager, task))
        return task

    async def _arun(self, manager: TaskManager, task: HostedTask) -> None:
        try:
            await self.run_child(manager, task)
            if task.status == "running":
                task.status = "answered"
        except Exception as exc:  # noqa: BLE001 - surface as a failed subagent result
            task.status = "failed"
            task.error = str(exc)
            if task.result is None:
                task.result = {
                    "type": "subagent_result",
                    "task_id": task.job_id,
                    "status": "failed",
                    "error": str(exc),
                    "message": f"subagent failed: {exc}",
                }
        finally:
            task.finished_at = time.time()
            manager.mark_ready(task)

    def cancel(self, manager: TaskManager, job: HostedTask) -> None:
        del manager
        if job.status == "running":
            job.status = "cancelled"
            job.finished_at = time.time()


Task = BackgroundJob | HostedTask


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
    # The always-on event loop background shell jobs run their asyncio subprocess monitors
    # on. In production it is the run's own loop, bound by AgentLoop._apump_turn via
    # bind_loop(); when TaskManager is used standalone (unit tests) a private daemon-thread
    # loop is started lazily as a fallback. Either way one loop, always running, hosts the
    # subprocess coroutines so they progress while the run is parked between turns.
    _run_loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    _task_loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    _task_loop_thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._condition = threading.Condition(self._lock)
        self.executors = {
            "shell": ShellTaskExecutor(),
            "hitl": HostedTaskExecutor(kind="hitl"),
            "automation": HostedTaskExecutor(kind="automation"),
            # A scoped-capability request: the run parks awaiting an external grant (a credential
            # lease / tool-enable decision), resolved through the same report_result -> reentry
            # path as hitl/automation. The grant is injected back as an async tool result.
            "capability": HostedTaskExecutor(kind="capability"),
        }
        self.injectors = {
            "shell": ShellResultInjector(),
            "hitl": HostedResultInjector(kind="hitl", tool_name="human_input", result_type="human_input_result"),
            "automation": HostedResultInjector(
                kind="automation", tool_name="automation", result_type="automation_result"
            ),
            "capability": HostedResultInjector(
                kind="capability", tool_name="capability", result_type="capability_grant"
            ),
        }

    def _register(self, job: Task) -> None:
        with self._condition:
            self.jobs[job.job_id] = job
            self._write_job(job)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the run's event loop (called by AgentLoop._apump_turn while running on it).
        Background shell jobs schedule their subprocess monitors onto this loop so they
        progress on the same always-on loop that drives the run."""
        self._run_loop = loop

    def _job_loop(self) -> asyncio.AbstractEventLoop:
        """The loop to run subprocess monitors on: the bound run loop in production, else a
        lazily-started private daemon-thread loop (standalone/unit-test fallback)."""
        if self._run_loop is not None:
            return self._run_loop
        if self._task_loop is None:
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever, name=f"nar-tasks-{self.run_id}", daemon=True
            )
            thread.start()
            self._task_loop = loop
            self._task_loop_thread = thread
        return self._task_loop

    def schedule_job_coroutine(self, coro: Any) -> Any:
        """Schedule a job coroutine on the job loop from any thread; returns a
        concurrent.futures.Future. Used by ShellTaskExecutor (which runs in an offloaded
        worker thread) to spawn/monitor a subprocess on the always-on loop."""
        return asyncio.run_coroutine_threadsafe(coro, self._job_loop())

    def _shutdown_task_loop(self) -> None:
        """Stop the private fallback loop if one was started. The bound run loop is owned by
        AgentLoop and is never stopped here.

        Cancel and await any still-pending monitor coroutines first: ``cancel_all`` sets a
        job's terminal status from *inside* ``_amonitor`` before that coroutine returns, so
        without draining we could close the loop while a monitor is in its tail — the
        "Task was destroyed but it is pending" warning. Draining makes teardown deterministic."""
        loop, thread = self._task_loop, self._task_loop_thread
        if loop is None:
            return

        async def _drain() -> None:
            pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(_drain(), loop).result(timeout=_REAP_TIMEOUT_S + 2)
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)
        loop.close()
        self._task_loop = None
        self._task_loop_thread = None

    def checkpoint_payload(self) -> dict[str, Any]:
        """Read-only durable snapshot of the hosted tasks and reentry bookkeeping for
        ``run_dir/checkpoint.json``. All hosted tasks are captured (not just running
        ones) so an answered-but-undelivered task's result is not lost on restore.
        Shell ``BackgroundJob``s are intentionally omitted — they are never restored
        live (a subprocess can't cross a process boundary)."""
        with self._lock:
            return {
                "hosted_tasks": [
                    task.checkpoint_json()
                    for task in self.jobs.values()
                    if isinstance(task, HostedTask)
                ],
                "reentry_queue": list(self._reentry_queue),
                "delivered_reentry_jobs": sorted(self._delivered_reentry_jobs),
            }

    def restore_state(
        self,
        hosted_tasks: list[HostedTask],
        *,
        reentry_queue: list[str],
        delivered_reentry_jobs: list[str],
    ) -> None:
        """Rehydrate parked hosted tasks and reentry bookkeeping into a freshly
        bootstrapped manager (durable restore). The task.json files already exist on
        disk, so this only re-registers the in-memory objects — it does not re-write
        them. Shell jobs are never restored live; see AgentLoop._rehydrate for the
        crashed-shell -> failed-observation path."""
        with self._condition:
            for task in hosted_tasks:
                self.jobs[task.job_id] = task
            self._reentry_queue = list(reentry_queue)
            self._delivered_reentry_jobs = set(delivered_reentry_jobs)

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
        terminal status/result and publish it through the shared reentry pipe.

        Idempotent — first report wins. A duplicate report (a callback retry) is a safe no-op: it
        neither clobbers the recorded result nor re-publishes to the reentry queue (which would make
        the agent observe the result twice). The dedup signal is the already-persisted+rehydrated
        ``ready_for_reentry``/``finished_at`` job state, so it holds across a restart with no extra
        bookkeeping. Mirrors the inbox's dedup-by-id (effectively-once result ingestion)."""
        task = self.get_job(task_id)
        if task.ready_for_reentry or task.finished_at is not None:
            return {"task_id": task_id, "status": task.status, "delivered": False, "duplicate": True}
        task.status = status  # type: ignore[assignment]
        task.finished_at = time.time()
        task.result = result  # type: ignore[attr-defined]
        self.mark_ready(task)
        return {"task_id": task_id, "status": status, "delivered": True, "duplicate": False}

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

    def external_pending_task_ids(self) -> list[str]:
        """Undelivered resume-tasks whose executor is NOT in-process — i.e. hosted
        (hitl/automation) tasks the run is parked on awaiting an external report."""
        with self._lock:
            out: list[str] = []
            for job in self.jobs.values():
                if (
                    job.status == "running"
                    and getattr(job, "resume_on_exit", False)
                    and job.job_id not in self._delivered_reentry_jobs
                ):
                    executor = self.executors.get(job.kind)
                    if executor is not None and not getattr(executor, "in_process", True):
                        out.append(job.job_id)
            return out

    def outstanding_resume_task_ids(self) -> set[str]:
        """Read-only: ids of every undelivered resume-task still running, regardless
        of executor kind. The difference against ``external_pending_task_ids`` is the
        set of live in-process (shell) tasks — AgentLoop.snapshot() uses it to refuse a
        snapshot while a shell job is still alive (a live subprocess can't be restored)."""
        with self._lock:
            return {
                job.job_id
                for job in self.jobs.values()
                if getattr(job, "resume_on_exit", False)
                and job.status == "running"
                and job.job_id not in self._delivered_reentry_jobs
            }

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
        # cancel() above synchronously terminates each subprocess, so the children are
        # already dying. The wait below only lets the monitor coroutines observe the deaths
        # and settle status. If we are ON the job loop thread (run_once's close() runs there),
        # we must NOT block it — the monitors need this very loop to run — so skip the wait;
        # the processes are already terminated.
        try:
            on_job_loop = asyncio.get_running_loop() is self._run_loop and self._run_loop is not None
        except RuntimeError:
            on_job_loop = False
        try:
            if not on_job_loop:
                deadline = time.time() + 5
                while time.time() < deadline:
                    with self._condition:
                        if all(job.status != "running" for job in self.jobs.values()):
                            return
                    time.sleep(0.05)
        finally:
            self._shutdown_task_loop()

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


