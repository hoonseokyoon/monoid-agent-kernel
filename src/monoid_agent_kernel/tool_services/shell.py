from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from monoid_agent_kernel.core.workspace import Workspace
from monoid_agent_kernel.core.json_ingress import normalize_unicode_scalars
from monoid_agent_kernel.core.runtime_controls import validate_shell_runtime
from monoid_agent_kernel.errors import ToolExecutionError, error_code_for_exception
from monoid_agent_kernel.tasks import TaskManager
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.public_view import public_error_message, public_path
from monoid_agent_kernel.recorder import AgentRecorder
from monoid_agent_kernel.shell import (
    AutoApproveShellApprovalProvider,
    DenyShellApprovalProvider,
    ShellApprovalDecision,
    ShellApprovalProvider,
    ShellApprovalRequest,
    ShellCommandRule,
    ShellExecutionOptions,
    execute_shell,
    validate_shell_approval_decision,
    validate_shell_execution_result,
)
from monoid_agent_kernel.tool_services.base import CallContext


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
    job_manager: TaskManager
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

    def execute(
        self, args: dict[str, Any], call: CallContext, *, argv_override: list[str] | None = None
    ) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ToolExecutionError(
                "shell arguments must be an object", error_code="tool_args_invalid"
            )
        command = args.get("command")
        if type(command) is not str or not command.strip():
            raise ToolExecutionError(
                "shell command must be a non-empty string",
                error_code="tool_args_invalid",
            )
        command = normalize_unicode_scalars(command)
        cwd = args.get("cwd", ".")
        if type(cwd) is not str or not cwd:
            raise ToolExecutionError("shell cwd must be a string", error_code="tool_args_invalid")
        cwd = normalize_unicode_scalars(cwd)
        if argv_override is not None and (
            type(argv_override) is not list
            or not all(type(argument) is str for argument in argv_override)
        ):
            raise ToolExecutionError(
                "shell argv override must be an array of strings",
                error_code="tool_args_invalid",
            )
        shell_options = _shell_options_from_call(call)
        requested_timeout_s = args.get("timeout_s")
        requested_max_output_bytes = args.get("max_output_bytes")
        requested_startup_wait_s = args.get("startup_wait_s")
        timeout_s = shell_options.effective_timeout(requested_timeout_s)
        max_output_bytes = shell_options.effective_output_limit(requested_max_output_bytes)
        startup_wait_s = shell_options.effective_startup_wait(requested_startup_wait_s)
        execution_workspace = shell_options.effective_execution_workspace(
            self.workspace.backend_kind
        )
        # A pre-built argv (skill.run_script) runs foreground-only: the background job path
        # builds its own argv from the command string and has no argv-override seam.
        background_requested = args.get("background", False)
        resume_on_exit = args.get("resume_on_exit", True)
        if type(background_requested) is not bool or type(resume_on_exit) is not bool:
            raise ToolExecutionError(
                "shell background and resume_on_exit must be booleans",
                error_code="tool_args_invalid",
            )
        background = background_requested and argv_override is None
        env = args.get("env", {})
        if not isinstance(env, dict):
            raise ToolExecutionError("shell env must be an object", error_code="tool_args_invalid")
        if not all(type(key) is str and type(value) is str for key, value in env.items()):
            raise ToolExecutionError(
                "shell env keys and values must be strings",
                error_code="tool_args_invalid",
            )
        env = {
            normalize_unicode_scalars(key): normalize_unicode_scalars(value)
            for key, value in env.items()
        }
        request = ShellApprovalRequest(
            run_id=self.run_id,
            tool_call_id=call.tool_call_id,
            command=command,
            cwd=cwd,
            requested_timeout_s=requested_timeout_s,
            effective_timeout_s=timeout_s,
            requested_max_output_bytes=requested_max_output_bytes,
            effective_max_output_bytes=max_output_bytes,
            execution_workspace=execution_workspace,
            requested_startup_wait_s=requested_startup_wait_s,
            effective_startup_wait_s=startup_wait_s,
            background=background,
            resume_on_exit=resume_on_exit,
            env_keys=tuple(sorted(env)),
        )
        approval_parent = call.tool_event_id
        self.recorder.emit(
            "tool.approval.requested",
            turn_id=call.turn_id,
            parent_id=approval_parent,
            data=request.to_public_json(self.permission_policy),
        )
        provider = self.approval_provider or _approval_provider_for_options(shell_options)
        if provider is None:
            decision = ShellApprovalDecision(
                approved=False,
                reason="shell approval provider unavailable",
                approver_id="none",
            )
        else:
            try:
                decision = validate_shell_approval_decision(provider.approve_shell(request))
            except (TypeError, ValueError) as exc:
                raise ToolExecutionError(
                    "shell approval provider returned an invalid decision",
                    error_code="tool_approval_invalid",
                ) from exc
        approval_event_type = (
            "tool.approval.approved" if decision.approved else "tool.approval.denied"
        )
        self.recorder.emit(
            approval_event_type,
            turn_id=call.turn_id,
            parent_id=approval_parent,
            data={**request.to_public_json(self.permission_policy), **decision.to_public_json()},
            level="info" if decision.approved else "warning",
        )
        if not decision.approved:
            raise ToolExecutionError(
                decision.reason or "shell approval denied", error_code="tool_approval_denied"
            )

        shell_started = self.recorder.emit(
            "shell.exec.started",
            turn_id=call.turn_id,
            parent_id=approval_parent,
            data=request.to_public_json(self.permission_policy),
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
                        **request.to_public_json(self.permission_policy),
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
                    **request.to_public_json(self.permission_policy),
                    "job_id": job.job_id,
                    "status": job.status,
                    "stdout_path": content["stdout_path"],
                    "stderr_path": content["stderr_path"],
                },
            )
            return content
        try:
            result = validate_shell_execution_result(
                execute_shell(
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
                    argv_override=argv_override,
                )
            )
        except Exception as exc:
            self.failed_shell_calls += 1
            self.recorder.emit(
                "shell.exec.failed",
                turn_id=call.turn_id,
                parent_id=shell_started.event_id,
                data={
                    **request.to_public_json(self.permission_policy),
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
            "shell.exec.failed"
            if result.timed_out or result.output_truncated
            else "shell.exec.finished",
            turn_id=call.turn_id,
            parent_id=shell_started.event_id,
            data={
                **request.to_public_json(self.permission_policy),
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
                "changed_paths": [
                    public_path(path, self.permission_policy) for path in result.changed_paths
                ],
            },
            level="warning" if result.timed_out or result.output_truncated else "info",
        )
        return result.to_tool_content()


def _shell_options_from_call(call: CallContext) -> ShellExecutionOptions:
    runtime = {} if call.runtime is None else call.runtime
    runtime_payload = validate_shell_runtime(runtime)
    runtime_payload.setdefault("enabled", True)
    runtime_options = ShellExecutionOptions.from_json(runtime_payload)
    scoped_command_rules = tuple(
        ShellCommandRule(action="allow", prefix=prefix)
        for prefix in call.scope.command_allow_prefixes
    ) + tuple(
        ShellCommandRule(action="deny", prefix=prefix)
        for prefix in call.scope.command_deny_prefixes
    )
    return ShellExecutionOptions(
        enabled=runtime_options.enabled,
        approval_mode=runtime_options.approval_mode,
        shell=runtime_options.shell,
        default_timeout_s=runtime_options.default_timeout_s,
        max_timeout_s=runtime_options.max_timeout_s,
        default_startup_wait_s=runtime_options.default_startup_wait_s,
        max_startup_wait_s=runtime_options.max_startup_wait_s,
        default_max_output_bytes=runtime_options.default_max_output_bytes,
        max_output_bytes=runtime_options.max_output_bytes,
        cwd_root=runtime_options.cwd_root,
        execution_workspace=runtime_options.execution_workspace,
        env_allowlist=_dedupe((*runtime_options.env_allowlist, *call.scope.env_allowlist)),
        inherit_env_allowlist=runtime_options.inherit_env_allowlist,
        command_rules=(*runtime_options.command_rules, *scoped_command_rules),
    ).validated()


def _dedupe(items: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def _approval_provider_for_options(options: ShellExecutionOptions) -> ShellApprovalProvider | None:
    if options.approval_mode == "auto-approve":
        return AutoApproveShellApprovalProvider(approver_id="tool-binding")
    if options.approval_mode == "deny":
        return DenyShellApprovalProvider()
    return None
