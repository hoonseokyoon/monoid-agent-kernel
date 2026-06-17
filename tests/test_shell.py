from __future__ import annotations

from pathlib import Path

import pytest

from native_agent_runner.permissions import PermissionPolicy
from native_agent_runner.shell import ShellExecutionOptions, execute_shell
from native_agent_runner.workspace.local import LocalWorkspaceBackend


def _python_command(code: str) -> str:
    escaped = code.replace('"', '\\"')
    return f'python -c "{escaped}"'


def test_shell_policy_defaults_json_and_manifest() -> None:
    policy = ShellExecutionOptions.from_json(
        {
            "enabled": True,
            "approval_mode": "auto-approve",
            "env_allowlist": ["SAFE_VAR"],
            "command_rules": [{"action": "deny", "prefix": "rm "}],
        }
    )

    assert policy.enabled is True
    assert policy.approval_mode == "auto-approve"
    assert policy.env_allowlist == ("SAFE_VAR",)
    manifest = policy.to_manifest()
    assert manifest["enabled"] is True
    assert manifest["effective_shell"] in {"bash", "powershell"}
    assert manifest["execution_workspace"] == "auto"
    assert manifest["command_rules"] == [{"action": "deny", "prefix": "rm "}]


def test_shell_exec_materializes_workspace_and_syncs_to_propose_overlay(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace_root.joinpath("notes.md").write_text("notes\n", encoding="utf-8")
    workspace = LocalWorkspaceBackend(workspace_root, mode="propose")

    result = execute_shell(
        workspace=workspace,
        policy=ShellExecutionOptions(enabled=True, approval_mode="auto-approve"),
        permission_policy=PermissionPolicy(),
        command=_python_command("from pathlib import Path; Path('SUMMARY.md').write_text('summary\\n', encoding='utf-8')"),
        cwd=".",
        timeout_s=5,
        max_output_bytes=100_000,
        env={},
    )

    assert result.exit_code == 0
    assert result.changed_paths == ("SUMMARY.md",)
    assert not workspace_root.joinpath("SUMMARY.md").exists()
    assert "+summary" in workspace.diff_patch()
    assert workspace.changed_paths() == ["SUMMARY.md"]


def test_shell_exec_direct_workspace_for_staging_backend(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace_root.joinpath("notes.md").write_text("notes\n", encoding="utf-8")
    workspace = LocalWorkspaceBackend(workspace_root, mode="propose", backend_kind="staging")
    policy = ShellExecutionOptions(enabled=True, approval_mode="auto-approve")

    result = execute_shell(
        workspace=workspace,
        policy=policy,
        permission_policy=PermissionPolicy(),
        command=_python_command("from pathlib import Path; Path('SUMMARY.md').write_text('summary\\n', encoding='utf-8')"),
        cwd=".",
        timeout_s=policy.effective_timeout(None),
        max_output_bytes=policy.effective_output_limit(None),
        env={},
    )

    assert result.exit_code == 0
    assert result.execution_workspace == "direct"
    assert workspace_root.joinpath("SUMMARY.md").read_text(encoding="utf-8") == "summary\n"
    assert result.changed_paths == ("SUMMARY.md",)
    assert workspace.changed_paths() == ["SUMMARY.md"]


def test_shell_requested_limits_are_recorded_separately_from_effective_limits(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = LocalWorkspaceBackend(workspace_root, mode="propose", backend_kind="staging")
    policy = ShellExecutionOptions(
        enabled=True,
        approval_mode="auto-approve",
        default_timeout_s=5,
        max_timeout_s=10,
        default_max_output_bytes=100,
        max_output_bytes=200,
    )

    result = execute_shell(
        workspace=workspace,
        policy=policy,
        permission_policy=PermissionPolicy(),
        command=_python_command("print('ok')"),
        cwd=".",
        timeout_s=policy.effective_timeout(999),
        max_output_bytes=policy.effective_output_limit(999),
        env={},
        requested_timeout_s=999,
        requested_max_output_bytes=999,
    )

    assert result.requested_timeout_s == 999
    assert result.effective_timeout_s == 10
    assert result.requested_max_output_bytes == 999
    assert result.effective_max_output_bytes == 200
    assert result.to_tool_content()["effective_timeout_s"] == 10


def test_direct_shell_leaves_generated_paths_for_backend_policy(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = LocalWorkspaceBackend(workspace_root, mode="propose", backend_kind="staging")

    result = execute_shell(
        workspace=workspace,
        policy=ShellExecutionOptions(enabled=True, approval_mode="auto-approve"),
        permission_policy=PermissionPolicy(),
        command=_python_command("from pathlib import Path; Path('.env').write_text('secret', encoding='utf-8')"),
        cwd=".",
        timeout_s=5,
        max_output_bytes=100_000,
        env={},
    )

    assert result.exit_code == 0
    assert workspace_root.joinpath(".env").read_text(encoding="utf-8") == "secret"
    assert result.changed_paths == (".env",)


def test_isolated_copy_shell_allows_secret_looking_paths_by_default(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = LocalWorkspaceBackend(workspace_root, mode="propose")

    result = execute_shell(
        workspace=workspace,
        policy=ShellExecutionOptions(enabled=True, approval_mode="auto-approve"),
        permission_policy=PermissionPolicy(),
        command=_python_command("from pathlib import Path; Path('.env').write_text('secret', encoding='utf-8')"),
        cwd=".",
        timeout_s=5,
        max_output_bytes=100_000,
        env={},
    )

    assert result.exit_code == 0
    assert result.changed_paths == (".env",)
    assert not workspace_root.joinpath(".env").exists()
    assert workspace.read_bytes(".env")[0] == b"secret"


def test_shell_env_filters_provider_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("SAFE_VAR", "host-safe")
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = LocalWorkspaceBackend(workspace_root, mode="propose")
    policy = ShellExecutionOptions(
        enabled=True,
        approval_mode="auto-approve",
        env_allowlist=("SAFE_VAR", "OPENAI_API_KEY"),
        inherit_env_allowlist=("PATH", "SystemRoot", "COMSPEC", "PATHEXT", "OPENAI_API_KEY", "SAFE_VAR"),
    )

    result = execute_shell(
        workspace=workspace,
        policy=policy,
        permission_policy=PermissionPolicy(),
        command=_python_command(
            "import os; print((os.getenv('SAFE_VAR') or '') + ':' + str(os.getenv('OPENAI_API_KEY')))"
        ),
        cwd=".",
        timeout_s=5,
        max_output_bytes=100_000,
        env={"SAFE_VAR": "model-safe", "OPENAI_API_KEY": "model-secret"},
    )

    assert result.stdout.strip() == "model-safe:None"
    assert "provider-secret" not in result.stdout
    assert "model-secret" not in result.stdout


def test_shell_rejects_bad_cwd_and_explicitly_denied_output_path(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = LocalWorkspaceBackend(workspace_root, mode="propose")
    policy = ShellExecutionOptions(enabled=True, approval_mode="auto-approve")

    with pytest.raises(Exception):
        execute_shell(
            workspace=workspace,
            policy=policy,
            permission_policy=PermissionPolicy(),
            command="echo nope",
            cwd="..",
            timeout_s=5,
            max_output_bytes=100_000,
            env={},
        )

    with pytest.raises(Exception):
        execute_shell(
            workspace=workspace,
            policy=policy,
            permission_policy=PermissionPolicy(deny_patterns=(".env",)),
            command=_python_command("from pathlib import Path; Path('.env').write_text('x', encoding='utf-8')"),
            cwd=".",
            timeout_s=5,
            max_output_bytes=100_000,
            env={},
        )


def test_shell_timeout_and_output_cap(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = LocalWorkspaceBackend(workspace_root, mode="propose")
    policy = ShellExecutionOptions(enabled=True, approval_mode="auto-approve")

    timeout_result = execute_shell(
        workspace=workspace,
        policy=policy,
        permission_policy=PermissionPolicy(),
        command=_python_command("import time; time.sleep(5)"),
        cwd=".",
        timeout_s=1,
        max_output_bytes=100_000,
        env={},
    )
    assert timeout_result.timed_out is True

    output_result = execute_shell(
        workspace=workspace,
        policy=policy,
        permission_policy=PermissionPolicy(),
        command=_python_command("print('x' * 10000)"),
        cwd=".",
        timeout_s=5,
        max_output_bytes=100,
        env={},
    )
    assert output_result.output_truncated is True
    assert len(output_result.stdout.encode("utf-8")) <= 100
