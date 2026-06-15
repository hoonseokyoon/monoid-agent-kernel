from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from native_agent_runner.errors import PermissionDenied, ToolExecutionError, WorkspaceError
from native_agent_runner.permissions import PermissionPolicy
from native_agent_runner.workspace.paths import is_within, normalize_workspace_path

if TYPE_CHECKING:
    from native_agent_runner.workspace.local import LocalWorkspaceBackend

ShellApprovalMode = Literal["backend", "auto-approve", "deny"]
ShellKind = Literal["auto", "bash", "powershell"]
ShellRuleAction = Literal["allow", "deny"]
ShellExecutionWorkspace = Literal["auto", "isolated-copy", "direct"]
ResolvedShellExecutionWorkspace = Literal["isolated-copy", "direct"]

_DEFAULT_INHERIT_ENV = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "SystemRoot",
    "COMSPEC",
    "PATHEXT",
)

_SENSITIVE_ENV_FRAGMENTS = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "AUTHORIZATION",
    "API_KEY",
    "APIKEY",
    "PRIVATE_KEY",
)

@dataclass(frozen=True)
class ShellCommandRule:
    action: ShellRuleAction
    prefix: str

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ShellCommandRule:
        action = str(payload.get("action") or "")
        prefix = str(payload.get("prefix") or "")
        if action not in {"allow", "deny"}:
            raise ValueError("shell command rule action must be allow or deny")
        if not prefix.strip():
            raise ValueError("shell command rule prefix is required")
        return cls(action=action, prefix=prefix)

    def matches(self, command: str) -> bool:
        return command.strip().startswith(self.prefix)

    def to_json(self) -> dict[str, str]:
        return {"action": self.action, "prefix": self.prefix}


@dataclass(frozen=True)
class ShellPolicy:
    enabled: bool = False
    approval_mode: ShellApprovalMode = "backend"
    shell: ShellKind = "auto"
    default_timeout_s: int = 120
    max_timeout_s: int = 900
    default_startup_wait_s: int = 0
    max_startup_wait_s: int = 30
    default_max_output_bytes: int = 100_000
    max_output_bytes: int = 1_000_000
    cwd_root: str = "workspace"
    execution_workspace: ShellExecutionWorkspace = "auto"
    env_allowlist: tuple[str, ...] = ()
    inherit_env_allowlist: tuple[str, ...] = _DEFAULT_INHERIT_ENV
    command_rules: tuple[ShellCommandRule, ...] = ()

    @classmethod
    def from_json(cls, payload: dict[str, Any] | None) -> ShellPolicy:
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("shell_policy must be an object")
        rules = tuple(
            ShellCommandRule.from_json(item)
            for item in payload.get("command_rules") or ()
        )
        return cls(
            enabled=bool(payload.get("enabled", False)),
            approval_mode=_approval_mode(str(payload.get("approval_mode") or "backend")),
            shell=_shell_kind(str(payload.get("shell") or "auto")),
            default_timeout_s=int(payload.get("default_timeout_s", 120)),
            max_timeout_s=int(payload.get("max_timeout_s", 900)),
            default_startup_wait_s=int(payload.get("default_startup_wait_s", 0)),
            max_startup_wait_s=int(payload.get("max_startup_wait_s", 30)),
            default_max_output_bytes=int(payload.get("default_max_output_bytes", 100_000)),
            max_output_bytes=int(payload.get("max_output_bytes", 1_000_000)),
            cwd_root=str(payload.get("cwd_root") or "workspace"),
            execution_workspace=_execution_workspace(
                str(payload.get("execution_workspace") or "auto")
            ),
            env_allowlist=_string_tuple(payload.get("env_allowlist") or ()),
            inherit_env_allowlist=_string_tuple(
                payload.get("inherit_env_allowlist") or _DEFAULT_INHERIT_ENV
            ),
            command_rules=rules,
        ).validated()

    def to_json(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "approval_mode": self.approval_mode,
            "shell": self.shell,
            "default_timeout_s": self.default_timeout_s,
            "max_timeout_s": self.max_timeout_s,
            "default_startup_wait_s": self.default_startup_wait_s,
            "max_startup_wait_s": self.max_startup_wait_s,
            "default_max_output_bytes": self.default_max_output_bytes,
            "max_output_bytes": self.max_output_bytes,
            "cwd_root": self.cwd_root,
            "execution_workspace": self.execution_workspace,
            "env_allowlist": list(self.env_allowlist),
            "inherit_env_allowlist": list(self.inherit_env_allowlist),
            "command_rules": [rule.to_json() for rule in self.command_rules],
        }

    def merged(
        self,
        *,
        enabled: bool | None = None,
        approval_mode: str | None = None,
        timeout_s: int | None = None,
        max_output_bytes: int | None = None,
        execution_workspace: str | None = None,
        env_allowlist: tuple[str, ...] = (),
    ) -> ShellPolicy:
        return ShellPolicy(
            enabled=self.enabled if enabled is None else enabled,
            approval_mode=self.approval_mode if approval_mode is None else _approval_mode(approval_mode),
            shell=self.shell,
            default_timeout_s=self.default_timeout_s if timeout_s is None else timeout_s,
            max_timeout_s=self.max_timeout_s,
            default_startup_wait_s=self.default_startup_wait_s,
            max_startup_wait_s=self.max_startup_wait_s,
            default_max_output_bytes=self.default_max_output_bytes
            if max_output_bytes is None
            else max_output_bytes,
            max_output_bytes=self.max_output_bytes,
            cwd_root=self.cwd_root,
            execution_workspace=self.execution_workspace
            if execution_workspace is None
            else _execution_workspace(execution_workspace),
            env_allowlist=tuple(dict.fromkeys((*self.env_allowlist, *env_allowlist))),
            inherit_env_allowlist=self.inherit_env_allowlist,
            command_rules=self.command_rules,
        ).validated()

    def validated(self) -> ShellPolicy:
        if self.approval_mode not in {"backend", "auto-approve", "deny"}:
            raise ValueError(f"unsupported shell approval mode: {self.approval_mode}")
        if self.shell not in {"auto", "bash", "powershell"}:
            raise ValueError(f"unsupported shell: {self.shell}")
        if self.cwd_root != "workspace":
            raise ValueError("v0.8 supports only cwd_root='workspace'")
        if self.execution_workspace not in {"auto", "isolated-copy", "direct"}:
            raise ValueError(f"unsupported shell execution workspace: {self.execution_workspace}")
        if self.default_timeout_s < 1 or self.max_timeout_s < 1:
            raise ValueError("shell timeout values must be positive")
        if self.default_timeout_s > self.max_timeout_s:
            raise ValueError("default_timeout_s cannot exceed max_timeout_s")
        if self.default_startup_wait_s < 0 or self.max_startup_wait_s < 0:
            raise ValueError("shell startup wait values must be non-negative")
        if self.default_startup_wait_s > self.max_startup_wait_s:
            raise ValueError("default_startup_wait_s cannot exceed max_startup_wait_s")
        if self.default_max_output_bytes < 1 or self.max_output_bytes < 1:
            raise ValueError("shell output byte limits must be positive")
        if self.default_max_output_bytes > self.max_output_bytes:
            raise ValueError("default_max_output_bytes cannot exceed max_output_bytes")
        return self

    def effective_timeout(self, requested: Any) -> int:
        value = self.default_timeout_s if requested is None else int(requested)
        return max(1, min(value, self.max_timeout_s))

    def effective_output_limit(self, requested: Any) -> int:
        value = self.default_max_output_bytes if requested is None else int(requested)
        return max(1, min(value, self.max_output_bytes))

    def effective_startup_wait(self, requested: Any) -> int:
        value = self.default_startup_wait_s if requested is None else int(requested)
        return max(0, min(value, self.max_startup_wait_s))

    def effective_shell(self) -> Literal["bash", "powershell"]:
        if self.shell == "auto":
            return "powershell" if os.name == "nt" else "bash"
        return self.shell

    def effective_execution_workspace(self, workspace_backend: str) -> ResolvedShellExecutionWorkspace:
        if self.execution_workspace == "auto":
            return "direct" if workspace_backend == "staging" else "isolated-copy"
        return self.execution_workspace

    def check_command(self, command: str) -> None:
        deny_matches = [rule for rule in self.command_rules if rule.action == "deny" and rule.matches(command)]
        if deny_matches:
            raise ToolExecutionError("shell command denied by policy", error_code="shell_policy_denied")
        allow_rules = [rule for rule in self.command_rules if rule.action == "allow"]
        if allow_rules and not any(rule.matches(command) for rule in allow_rules):
            raise ToolExecutionError("shell command not allowed by policy", error_code="shell_policy_denied")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "approval_mode": self.approval_mode,
            "shell": self.shell,
            "effective_shell": self.effective_shell(),
            "default_timeout_s": self.default_timeout_s,
            "max_timeout_s": self.max_timeout_s,
            "default_startup_wait_s": self.default_startup_wait_s,
            "max_startup_wait_s": self.max_startup_wait_s,
            "default_max_output_bytes": self.default_max_output_bytes,
            "max_output_bytes": self.max_output_bytes,
            "cwd_root": self.cwd_root,
            "execution_workspace": self.execution_workspace,
            "env_allowlist": list(self.env_allowlist),
            "inherit_env_allowlist": list(self.inherit_env_allowlist),
            "command_rules": [rule.to_json() for rule in self.command_rules],
        }


@dataclass(frozen=True)
class ShellApprovalRequest:
    run_id: str
    tool_call_id: str
    command: str
    cwd: str
    requested_timeout_s: int | None
    effective_timeout_s: int
    requested_max_output_bytes: int | None
    effective_max_output_bytes: int
    execution_workspace: ResolvedShellExecutionWorkspace
    requested_startup_wait_s: int | None = None
    effective_startup_wait_s: int = 0
    background: bool = False
    resume_on_exit: bool = True
    env_keys: tuple[str, ...] = ()

    @property
    def command_preview(self) -> str:
        return _preview_command(self.command)

    @property
    def timeout_s(self) -> int:
        return self.effective_timeout_s

    @property
    def max_output_bytes(self) -> int:
        return self.effective_max_output_bytes

    def to_public_json(self) -> dict[str, Any]:
        return {
            "tool": "shell.exec",
            "tool_call_id": self.tool_call_id,
            "command_preview": self.command_preview,
            "cwd": self.cwd,
            "requested_timeout_s": self.requested_timeout_s,
            "effective_timeout_s": self.effective_timeout_s,
            "requested_startup_wait_s": self.requested_startup_wait_s,
            "effective_startup_wait_s": self.effective_startup_wait_s,
            "requested_max_output_bytes": self.requested_max_output_bytes,
            "effective_max_output_bytes": self.effective_max_output_bytes,
            "timeout_s": self.effective_timeout_s,
            "max_output_bytes": self.effective_max_output_bytes,
            "execution_workspace": self.execution_workspace,
            "background": self.background,
            "resume_on_exit": self.resume_on_exit,
            "env_keys": list(self.env_keys),
        }


@dataclass(frozen=True)
class ShellApprovalDecision:
    approved: bool
    reason: str = ""
    approver_id: str = ""

    def to_public_json(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "reason": self.reason,
            "approver_id": self.approver_id,
        }


class ShellApprovalProvider(Protocol):
    def approve_shell(self, request: ShellApprovalRequest) -> ShellApprovalDecision:
        ...


@dataclass(frozen=True)
class AutoApproveShellApprovalProvider:
    approver_id: str = "reference-backend"

    def approve_shell(self, request: ShellApprovalRequest) -> ShellApprovalDecision:
        del request
        return ShellApprovalDecision(approved=True, reason="auto-approved", approver_id=self.approver_id)


@dataclass(frozen=True)
class DenyShellApprovalProvider:
    reason: str = "shell approval denied"

    def approve_shell(self, request: ShellApprovalRequest) -> ShellApprovalDecision:
        del request
        return ShellApprovalDecision(approved=False, reason=self.reason, approver_id="policy")


@dataclass(frozen=True)
class ShellExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    output_truncated: bool
    duration_s: float
    stdout_bytes: int
    stderr_bytes: int
    changed_paths: tuple[str, ...]
    requested_timeout_s: int | None = None
    effective_timeout_s: int = 0
    requested_max_output_bytes: int | None = None
    effective_max_output_bytes: int = 0
    execution_workspace: ResolvedShellExecutionWorkspace = "isolated-copy"

    def to_tool_content(self) -> dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "output_truncated": self.output_truncated,
            "duration_s": self.duration_s,
            "stdout_bytes": self.stdout_bytes,
            "stderr_bytes": self.stderr_bytes,
            "changed_paths": list(self.changed_paths),
            "requested_timeout_s": self.requested_timeout_s,
            "effective_timeout_s": self.effective_timeout_s,
            "requested_max_output_bytes": self.requested_max_output_bytes,
            "effective_max_output_bytes": self.effective_max_output_bytes,
            "execution_workspace": self.execution_workspace,
        }


@dataclass(frozen=True)
class _WorkspaceSnapshot:
    files: dict[str, bytes]
    dirs: set[str]


def execute_shell(
    *,
    workspace: LocalWorkspaceBackend,
    policy: ShellPolicy,
    permission_policy: PermissionPolicy,
    command: str,
    cwd: str,
    timeout_s: int,
    max_output_bytes: int,
    env: Mapping[str, Any],
    requested_timeout_s: int | None = None,
    requested_max_output_bytes: int | None = None,
    execution_workspace: ResolvedShellExecutionWorkspace | None = None,
) -> ShellExecutionResult:
    if not policy.enabled:
        raise ToolExecutionError("shell is disabled", error_code="shell_disabled")
    if not command.strip():
        raise ToolExecutionError("shell command is required", error_code="shell_exec_error")
    policy.check_command(command)
    cwd_rel = _validate_cwd(workspace, cwd, permission_policy)
    safe_env = _build_env(policy, env)
    shell_argv = _shell_argv(policy.effective_shell(), command)
    resolved_execution_workspace = execution_workspace or policy.effective_execution_workspace(workspace.backend_kind)

    if resolved_execution_workspace == "direct":
        cwd_abs = (workspace.root / cwd_rel).resolve()
        if not is_within(workspace.root, cwd_abs):
            raise WorkspaceError(f"shell cwd escapes workspace: {cwd_rel}")
        if not cwd_abs.exists() or not cwd_abs.is_dir():
            raise WorkspaceError(f"shell cwd is not a directory: {cwd_rel}")
        proc_result = _run_subprocess(
            shell_argv,
            cwd=cwd_abs,
            env=safe_env,
            timeout_s=timeout_s,
            max_output_bytes=max_output_bytes,
        )
        return _execution_result_from_proc(
            proc_result,
            changed_paths=tuple(workspace.changed_paths()),
            requested_timeout_s=requested_timeout_s,
            effective_timeout_s=timeout_s,
            requested_max_output_bytes=requested_max_output_bytes,
            effective_max_output_bytes=max_output_bytes,
            execution_workspace=resolved_execution_workspace,
        )

    with tempfile.TemporaryDirectory(prefix="native-agent-shell-") as tmp:
        tmp_root = Path(tmp)
        before = _materialize_workspace(workspace, tmp_root, permission_policy)
        cwd_abs = (tmp_root / cwd_rel).resolve()
        if not is_within(tmp_root.resolve(), cwd_abs):
            raise WorkspaceError(f"shell cwd escapes workspace: {cwd_rel}")
        if not cwd_abs.exists() or not cwd_abs.is_dir():
            raise WorkspaceError(f"shell cwd is not a directory: {cwd_rel}")

        proc_result = _run_subprocess(
            shell_argv,
            cwd=cwd_abs,
            env=safe_env,
            timeout_s=timeout_s,
            max_output_bytes=max_output_bytes,
        )
        if proc_result["timed_out"]:
            return _execution_result_from_proc(
                proc_result,
                changed_paths=(),
                requested_timeout_s=requested_timeout_s,
                effective_timeout_s=timeout_s,
                requested_max_output_bytes=requested_max_output_bytes,
                effective_max_output_bytes=max_output_bytes,
                execution_workspace=resolved_execution_workspace,
            )
        if proc_result["output_truncated"]:
            return _execution_result_from_proc(
                proc_result,
                changed_paths=(),
                requested_timeout_s=requested_timeout_s,
                effective_timeout_s=timeout_s,
                requested_max_output_bytes=requested_max_output_bytes,
                effective_max_output_bytes=max_output_bytes,
                execution_workspace=resolved_execution_workspace,
            )
        after = _scan_materialized_workspace(tmp_root, permission_policy)
        changed_paths = _sync_workspace_changes(workspace, before, after)
        return _execution_result_from_proc(
            proc_result,
            changed_paths=tuple(changed_paths),
            requested_timeout_s=requested_timeout_s,
            effective_timeout_s=timeout_s,
            requested_max_output_bytes=requested_max_output_bytes,
            effective_max_output_bytes=max_output_bytes,
            execution_workspace=resolved_execution_workspace,
        )


def _execution_result_from_proc(
    proc_result: dict[str, Any],
    *,
    changed_paths: tuple[str, ...],
    requested_timeout_s: int | None,
    effective_timeout_s: int,
    requested_max_output_bytes: int | None,
    effective_max_output_bytes: int,
    execution_workspace: ResolvedShellExecutionWorkspace,
) -> ShellExecutionResult:
    return ShellExecutionResult(
        exit_code=int(proc_result["exit_code"]),
        stdout=str(proc_result["stdout"]),
        stderr=str(proc_result["stderr"]),
        timed_out=bool(proc_result["timed_out"]),
        output_truncated=bool(proc_result["output_truncated"]),
        duration_s=float(proc_result["duration_s"]),
        stdout_bytes=int(proc_result["stdout_bytes"]),
        stderr_bytes=int(proc_result["stderr_bytes"]),
        changed_paths=changed_paths,
        requested_timeout_s=requested_timeout_s,
        effective_timeout_s=effective_timeout_s,
        requested_max_output_bytes=requested_max_output_bytes,
        effective_max_output_bytes=effective_max_output_bytes,
        execution_workspace=execution_workspace,
    )


def _validate_cwd(
    workspace: LocalWorkspaceBackend,
    cwd: str,
    permission_policy: PermissionPolicy,
) -> str:
    rel = normalize_workspace_path(cwd or ".")
    _check_shell_path_allowed(rel, permission_policy)
    resolved_rel, abs_path = workspace.resolve_existing_or_parent(rel)
    kind = workspace._effective_kind(resolved_rel, abs_path)
    if kind != "dir":
        raise WorkspaceError(f"shell cwd is not a directory: {resolved_rel}")
    return resolved_rel


def _materialize_workspace(
    workspace: LocalWorkspaceBackend,
    target_root: Path,
    permission_policy: PermissionPolicy,
) -> _WorkspaceSnapshot:
    snapshot = _WorkspaceSnapshot(files={}, dirs=set())
    entries = workspace.list_entries(".", recursive=True, max_entries=10000)
    for entry in entries:
        rel = normalize_workspace_path(entry.path)
        if _shell_path_denied(rel, permission_policy):
            continue
        target = target_root / rel
        if entry.kind == "dir":
            target.mkdir(parents=True, exist_ok=True)
            snapshot.dirs.add(rel)
            continue
        if entry.kind != "file":
            continue
        data, _digest = workspace.read_bytes(rel, max_bytes=500_000_000)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        snapshot.files[rel] = data
    return snapshot


def _scan_materialized_workspace(
    root: Path,
    permission_policy: PermissionPolicy,
) -> _WorkspaceSnapshot:
    root = root.resolve()
    files: dict[str, bytes] = {}
    dirs: set[str] = set()
    for item in sorted(root.rglob("*"), key=lambda child: child.as_posix()):
        resolved = item.resolve()
        if not is_within(root, resolved):
            raise WorkspaceError(f"shell produced a path escaping workspace: {item}")
        rel = resolved.relative_to(root).as_posix()
        _check_shell_path_allowed(rel, permission_policy)
        if item.is_dir():
            dirs.add(rel)
        elif item.is_file():
            files[rel] = item.read_bytes()
    return _WorkspaceSnapshot(files=files, dirs=dirs)


def _sync_workspace_changes(
    workspace: LocalWorkspaceBackend,
    before: _WorkspaceSnapshot,
    after: _WorkspaceSnapshot,
) -> list[str]:
    changed: set[str] = set()
    for rel in sorted(before.files):
        if rel not in after.files:
            workspace.delete_path(rel)
            changed.add(rel)

    for rel in sorted(after.dirs - before.dirs, key=lambda item: (item.count("/"), item)):
        has_file_descendant = any(path.startswith(rel.rstrip("/") + "/") for path in after.files)
        if not has_file_descendant:
            workspace.mkdir(rel)
            changed.add(rel)

    missing_dirs = before.dirs - after.dirs
    top_missing = [
        rel
        for rel in missing_dirs
        if not any(parent in missing_dirs for parent in _ancestor_paths(rel))
    ]
    for rel in sorted(top_missing, key=lambda item: (item.count("/"), item), reverse=True):
        if workspace.exists(rel):
            workspace.delete_path(rel, recursive=True)
            changed.add(rel)
    for rel, data in sorted(after.files.items()):
        if before.files.get(rel) != data:
            workspace.write_bytes(rel, data, create_dirs=True)
            changed.add(rel)
    return sorted(changed)


def _run_subprocess(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_s: int,
    max_output_bytes: int,
) -> dict[str, Any]:
    started = time.time()
    stdout_file = tempfile.NamedTemporaryFile(prefix="native-agent-shell-stdout-", delete=False)
    stderr_file = tempfile.NamedTemporaryFile(prefix="native-agent-shell-stderr-", delete=False)
    stdout_path = Path(stdout_file.name)
    stderr_path = Path(stderr_file.name)
    stdout_file.close()
    stderr_file.close()
    creationflags = 0
    preexec_fn = None
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        preexec_fn = os.setsid

    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=creationflags,
            preexec_fn=preexec_fn,
        )
        timed_out = False
        output_truncated = False
        while process.poll() is None:
            if time.time() - started >= timeout_s:
                timed_out = True
                _terminate_process(process)
                break
            if _file_size(stdout_path) + _file_size(stderr_path) > max_output_bytes:
                output_truncated = True
                _terminate_process(process)
                break
            time.sleep(0.02)
        try:
            exit_code = process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            exit_code = process.wait(timeout=2)

    stdout_bytes = _file_size(stdout_path)
    stderr_bytes = _file_size(stderr_path)
    if stdout_bytes + stderr_bytes > max_output_bytes:
        output_truncated = True
    stdout = _read_output(stdout_path, max_output_bytes)
    remaining = max(1, max_output_bytes - len(stdout.encode("utf-8", errors="replace")))
    stderr = _read_output(stderr_path, remaining)
    stdout_path.unlink(missing_ok=True)
    stderr_path.unlink(missing_ok=True)
    return {
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "output_truncated": output_truncated,
        "duration_s": time.time() - started,
        "stdout_bytes": stdout_bytes,
        "stderr_bytes": stderr_bytes,
    }


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        process.kill()


def _build_env(policy: ShellPolicy, requested: Mapping[str, Any]) -> dict[str, str]:
    env: dict[str, str] = {}
    inherited = set(policy.inherit_env_allowlist)
    for key in inherited:
        if key in os.environ and not _sensitive_env_key(key):
            env[key] = os.environ[key]
    allowed = set(policy.env_allowlist)
    for key, value in requested.items():
        key_text = str(key)
        if key_text in allowed and not _sensitive_env_key(key_text):
            env[key_text] = str(value)
    return env


def _shell_argv(shell: Literal["bash", "powershell"], command: str) -> list[str]:
    if shell == "powershell":
        return [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ]
    return ["bash", "-lc", command]


def _check_shell_path_allowed(rel: str, permission_policy: PermissionPolicy) -> None:
    if _shell_path_denied(rel, permission_policy):
        raise PermissionDenied(f"shell denied by path policy: {rel}", error_code="shell_policy_denied")


def _shell_path_denied(rel: str, permission_policy: PermissionPolicy) -> bool:
    normalized = normalize_workspace_path(rel)
    return permission_policy.is_path_denied(normalized)


def _ancestor_paths(rel: str) -> list[str]:
    parts = rel.split("/")
    return ["/".join(parts[:index]) for index in range(1, len(parts))]


def _sensitive_env_key(key: str) -> bool:
    upper = key.upper()
    return any(fragment in upper for fragment in _SENSITIVE_ENV_FRAGMENTS)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        raise ValueError("expected an array of strings")
    result = tuple(str(item) for item in value)
    if any(not item.strip() for item in result):
        raise ValueError("empty string is not allowed")
    return result


def _approval_mode(value: str) -> ShellApprovalMode:
    if value not in {"backend", "auto-approve", "deny"}:
        raise ValueError(f"unsupported shell approval mode: {value}")
    return value  # type: ignore[return-value]


def _shell_kind(value: str) -> ShellKind:
    if value not in {"auto", "bash", "powershell"}:
        raise ValueError(f"unsupported shell: {value}")
    return value  # type: ignore[return-value]


def _execution_workspace(value: str) -> ShellExecutionWorkspace:
    if value not in {"auto", "isolated-copy", "direct"}:
        raise ValueError(f"unsupported shell execution workspace: {value}")
    return value  # type: ignore[return-value]


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _read_output(path: Path, max_bytes: int) -> str:
    data = path.read_bytes()[:max_bytes]
    return data.decode("utf-8", errors="replace")


def _preview_command(command: str) -> str:
    single_line = " ".join(command.split())
    if len(single_line.encode("utf-8")) <= 240:
        return single_line
    return single_line[:200] + "..."
