from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from native_agent_runner.cli import main
from native_agent_runner.core.spec import (
    AgentRunSpec,
    ModelConfig,
    ModelRetryConfig,
    ReasoningConfig,
    RunLimits,
)
from native_agent_runner.permissions import PermissionPolicy
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter
from native_agent_runner.shell import ShellCommandRule, ShellPolicy
from native_agent_runner.tools.policy import ToolPolicy
from native_agent_runner.web import WebPolicy


def _populated_spec() -> AgentRunSpec:
    return AgentRunSpec(
        instruction="do the thing",
        workspace_root=Path("/ws"),
        run_root=Path("runs"),
        run_id="fixedrunid",
        mode="apply",
        workspace_backend="staging",
        model=ModelConfig(
            provider="openai",
            model="gpt-x",
            timeout_s=42,
            gateway_url="http://gw/internal/llm/turns",
            reasoning=ReasoningConfig(effort="high", summary="auto", on_unsupported="omit"),
            retry=ModelRetryConfig(max_attempts=5, retry_on=("gateway_timeout",)),
        ),
        limits=RunLimits(max_steps=7, max_tool_calls=11, max_bytes_read=123, max_duration_s=None),
        capabilities=frozenset({"fs.read", "run.control"}),
        permission_policy=PermissionPolicy(deny_patterns=(".env",), redact_patterns=("*.key",)),
        tool_policy=ToolPolicy(allowed_tools=("fs.read",), denied_tools=("shell.exec",)),
        shell_policy=ShellPolicy(
            enabled=True,
            approval_mode="auto-approve",
            default_timeout_s=30,
            command_rules=(ShellCommandRule(action="allow", prefix="ls"),),
        ),
        web_policy=WebPolicy(enabled=True, context_enabled=True, allowed_domains=("a.com",)),
        system_prompt_base="You are a custom base agent.",
        persona_segments=("Specialize in X.", "Be terse."),
        metadata={"tenant": "t1", "n": 3},
    )


def test_agent_run_spec_round_trip_is_lossless() -> None:
    spec = _populated_spec()
    blob = json.dumps(spec.to_json())  # must be JSON-serializable
    assert AgentRunSpec.from_json(json.loads(blob)) == spec


def test_default_spec_round_trip_preserves_run_id() -> None:
    spec = AgentRunSpec(instruction="x", workspace_root=Path("/w"), run_root=Path("runs"))
    restored = AgentRunSpec.from_json(spec.to_json())
    assert restored == spec
    assert restored.run_id == spec.run_id


@pytest.mark.parametrize(
    "value",
    [
        ModelConfig(provider="fake", model="m", timeout_s=5, gateway_url=None),
        ModelConfig(reasoning=ReasoningConfig(effort="minimal")),
        RunLimits(max_duration_s=None),
        RunLimits(max_duration_s=600),
        ReasoningConfig(effort="xhigh", summary="detailed"),
        ModelRetryConfig(max_attempts=2, retry_on=("gateway_rate_limited",)),
        ShellPolicy(enabled=True, command_rules=(ShellCommandRule(action="deny", prefix="rm"),)),
    ],
)
def test_sub_type_round_trip(value: object) -> None:
    cls = type(value)
    assert cls.from_json(value.to_json()) == value  # type: ignore[attr-defined]


def test_from_json_requires_instruction_and_workspace() -> None:
    with pytest.raises(ValueError):
        AgentRunSpec.from_json({"workspace_root": "/ws"})
    with pytest.raises(ValueError):
        AgentRunSpec.from_json({"instruction": "x"})


def test_cli_run_accepts_spec_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = AgentRunSpec(
        instruction="Finish.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        mode="apply",
    )
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps(spec.to_json()), encoding="utf-8")

    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    monkeypatch.setattr("native_agent_runner.cli._model_adapter", lambda *_args, **_kwargs: adapter)
    runner = CliRunner()

    result = runner.invoke(main, ["run", "--spec", str(spec_file)])

    assert result.exit_code == 0, result.output
    run_id_line = next(line for line in result.output.splitlines() if line.startswith("run_id: "))
    run_id = run_id_line.removeprefix("run_id: ")
    manifest = json.loads((tmp_path / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "apply"


def test_cli_spec_and_workspace_are_mutually_exclusive(tmp_path: Path) -> None:
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps({"instruction": "x", "workspace_root": str(tmp_path)}), encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(main, ["run", "--spec", str(spec_file), "--workspace", str(tmp_path)])

    assert result.exit_code != 0
    assert "cannot be combined" in result.output
