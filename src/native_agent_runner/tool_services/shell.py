from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from native_agent_runner.core.workspace import Workspace
from native_agent_runner.errors import ToolExecutionError, error_code_for_exception
from native_agent_runner.jobs import BackgroundJobManager
from native_agent_runner.permissions import PermissionPolicy
from native_agent_runner.public_view import public_error_message, public_path
from native_agent_runner.recorder import AgentRecorder
from native_agent_runner.shell import (
    AutoApproveShellApprovalProvider,
    DenyShellApprovalProvider,
    ShellApprovalDecision,
    ShellApprovalProvider,
    ShellApprovalRequest,
    ShellCommandRule,
    ShellExecutionOptions,
    execute_shell,
)
from native_agent_runner.tool_services.base import CallContext


@dataclass
class ShellService:
    """Orchestrates a shell tool call: approval, events, execution, counters.

    Wraps the low-level ``shell.execute_shell`` with approval gating, event
    emission, and background-job dispatch. Holds its own call counters, exposed
    via ``metrics()`` for the run summary.
    """

    run_id: str
    workspace: Workspace
    recorder: AgentRecorder
    job_manager: BackgroundJobManager
    permission_policy: PermissionPolicy
    approval_provider: ShellApprovalProvider | None = None
    shell_calls: int = 0
    failed_shell_calls: int = 0
    total_shell_duration_s: float = 0.0

    def metrics(self) -> dict[str, Any]:
        return {
            "shell_calls": self.shell_calls,
            "failed_shell_calls": self.failed_shell_calls,
            "total_shell_duration_s": self.total_shell_duration_s,
        }

    def execute(self, args: dict[str, Any], call: CallContext) -> dict[str, Any]:
        command = str(args["command"])
        cwd = str(args.get("cwd") or ".")
        shell_options = _shell_options_from_call(call)
        requested_timeout_s = args.get("timeout_s")
        requested_max_output_bytes = args.get("max_output_bytes")
        requested_startup_wait_s = args.get("startup_wait_s")
        timeout_s = shell_options.effective_timeout(requested_timeout_s)
        max_output_bytes = shell_options.effective_output_limit(requested_max_output_bytes)
        startup_wait_s = shell_options.effective_startup_wait(requested_startup_wait_s)
        execution_workspace = shell_options.effective_execution_workspace(self.workspace.backend_kind)
        background = bool(args.get("background", False))
        resume_on_exit = bool(args.get("resume_on_exit", True))
        env = args.get("env") or {}
        if not isinstance(env, dict):
            raise ToolExecutionError("shell env must be an object", error_code="tool_args_invalid")
        request = ShellApprovalRequest(
            run_id=self.run_id,
            tool_call_id=call.tool_call_id,
            command=command,
            cwd=cwd,
            requested_timeout_s=int(requested_timeout_s) if requested_timeout_s is not None else None,
            effective_timeout_s=timeout_s,
            requested_max_output_bytes=int(requested_max_output_bytes) if requested_max_output_bytes is not None else None,
            effective_max_output_bytes=max_output_bytes,
            execution_workspace=execution_workspace,
            requested_startup_wait_s=int(requested_startup_wait_s) if requested_startup_wait_s is not None else None,
            effective_startup_wait_s=startup_wait_s,
            background=background,
            resume_on_exit=resume_on_exit,
            env_keys=tuple(sorted(str(key) for key in env)),
        )
        approval_parent = call.tool_event_id
        self.recorder.emit(
            "tool.approval.requested",
            turn_id=call.turn_id,
            parent_id=approval_parent,
            data=request.to_public_json(),
        )
        provider = self.approval_provider or _approval_provider_for_options(shell_options)
        if provider is None:
            decision = ShellApprovalDecision(
                approved=False,
                reason="shell approval provider unavailable",
                approver_id="none",
            )
        else:
            decision = provider.approve_shell(request)
        approval_event_type = "tool.approval.approved" if decision.approved else "tool.approval.denied"
        self.recorder.emit(
            approval_event_type,
            turn_id=call.turn_id,
            parent_id=approval_parent,
            data={**request.to_public_json(), **decision.to_public_json()},
            level="info" if decision.approved else "warning",
        )
        if not decision.approved:
            raise ToolExecutionError(decision.reason or "shell approval denied", error_code="tool_approval_denied")

        shell_started = self.recorder.emit(
            "shell.exec.started",
            turn_id=call.turn_id,
            parent_id=approval_parent,
            data=request.to_public_json(),
        )
        if background:
            try:
                job = self.job_manager.start_shell_job(
                    shell_options=shell_options,
                    command=command,
                    cwd=cwd,
                    timeout_s=timeout_s,
                    max_output_bytes=max_output_bytes,
                    startup_wait_s=startup_wait_s,
                    env=env,
                    requested_timeout_s=request.requested_timeout_s,
                    requested_max_output_bytes=request.requested_max_output_bytes,
                    requested_startup_wait_s=request.requested_startup_wait_s,
                    execution_workspace=execution_workspace,
                    resume_on_exit=resume_on_exit,
                )
            except Exception as exc:
                self.failed_shell_calls += 1
                self.recorder.emit(
                    "shell.exec.failed",
                    turn_id=call.turn_id,
                    parent_id=shell_started.event_id,
                    data={
                        **request.to_public_json(),
                        "error": public_error_message(str(exc)),
                        "error_code": error_code_for_exception(exc),
                    },
                    level="warning",
                )
                raise
            self.shell_calls += 1
            content = job.started_content(self.recorder.run_dir)
            self.recorder.emit(
                "shell.exec.finished",
                turn_id=call.turn_id,
                parent_id=shell_started.event_id,
                data={
                    **request.to_public_json(),
                    "job_id": job.job_id,
                    "status": job.status,
                    "stdout_path": content["stdout_path"],
                    "stderr_path": content["stderr_path"],
                },
            )
            return content
        try:
            result = execute_shell(
                workspace=self.workspace,
                policy=shell_options,
                permission_policy=self.permission_policy,
                command=command,
                cwd=cwd,
                timeout_s=timeout_s,
                max_output_bytes=max_output_bytes,
                env=env,
                requested_timeout_s=request.requested_timeout_s,
                requested_max_output_bytes=request.requested_max_output_bytes,
                execution_workspace=execution_workspace,
            )
        except Exception as exc:
            self.failed_shell_calls += 1
            self.recorder.emit(
                "shell.exec.failed",
                turn_id=call.turn_id,
                parent_id=shell_started.event_id,
                data={
                    **request.to_public_json(),
                    "error": public_error_message(str(exc)),
                    "error_code": error_code_for_exception(exc),
                },
                level="warning",
            )
            raise
        self.shell_calls += 1
        self.total_shell_duration_s += result.duration_s
        if result.timed_out or result.output_truncated:
            self.failed_shell_calls += 1
        self.recorder.emit(
            "shell.exec.failed" if result.timed_out or result.output_truncated else "shell.exec.finished",
            turn_id=call.turn_id,
            parent_id=shell_started.event_id,
            data={
                **request.to_public_json(),
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "output_truncated": result.output_truncated,
                "duration_s": result.duration_s,
                "stdout_bytes": result.stdout_bytes,
                "stderr_bytes": result.stderr_bytes,
                "requested_timeout_s": result.requested_timeout_s,
                "effective_timeout_s": result.effective_timeout_s,
                "requested_max_output_bytes": result.requested_max_output_bytes,
                "effective_max_output_bytes": result.effective_max_output_bytes,
                "execution_workspace": result.execution_workspace,
                "changed_paths": [public_path(path, self.permission_policy) for path in result.changed_paths],
            },
            level="warning" if result.timed_out or result.output_truncated else "info",
        )
        return result.to_tool_content()


def _shell_options_from_call(call: CallContext) -> ShellExecutionOptions:
    runtime = call.runtime or {}
    shell_runtime = runtime.get("shell", runtime) if isinstance(runtime, dict) else {}
    if not isinstance(shell_runtime, dict):
        shell_runtime = {}
    command_rules = tuple(
        ShellCommandRule(action="allow", prefix=prefix)
        for prefix in call.scope.command_allow_prefixes
    ) + tuple(
        ShellCommandRule(action="deny", prefix=prefix)
        for prefix in call.scope.command_deny_prefixes
    )
    return ShellExecutionOptions(
        enabled=True,
        approval_mode=str(shell_runtime.get("approval_mode") or "backend"),  # type: ignore[arg-type]
        shell=str(shell_runtime.get("shell") or "auto"),  # type: ignore[arg-type]
        default_timeout_s=int(shell_runtime.get("default_timeout_s", 120)),
        max_timeout_s=int(shell_runtime.get("max_timeout_s", 900)),
        default_startup_wait_s=int(shell_runtime.get("default_startup_wait_s", 0)),
        max_startup_wait_s=int(shell_runtime.get("max_startup_wait_s", 30)),
        default_max_output_bytes=int(shell_runtime.get("default_max_output_bytes", 100_000)),
        max_output_bytes=int(shell_runtime.get("max_output_bytes", 1_000_000)),
        execution_workspace=str(shell_runtime.get("execution_workspace") or "auto"),  # type: ignore[arg-type]
        env_allowlist=call.scope.env_allowlist,
        command_rules=command_rules,
    ).validated()


def _approval_provider_for_options(options: ShellExecutionOptions) -> ShellApprovalProvider | None:
    if options.approval_mode == "auto-approve":
        return AutoApproveShellApprovalProvider(approver_id="tool-binding")
    if options.approval_mode == "deny":
        return DenyShellApprovalProvider()
    return None
