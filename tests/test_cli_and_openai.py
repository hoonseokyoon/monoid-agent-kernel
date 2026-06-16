from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from native_agent_runner.cli import main
from native_agent_runner.core.spec import ModelConfig, ReasoningConfig
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter
from native_agent_runner.providers.openai import OpenAIModelAdapter


def _openai_responses_available() -> bool:
    try:
        from openai import OpenAI
    except ImportError:
        return False
    return hasattr(OpenAI(api_key="test"), "responses")


def test_cli_requires_instruction(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runner = CliRunner()

    result = runner.invoke(main, ["run", "--workspace", str(workspace)])

    assert result.exit_code != 0
    assert "--instruction or --instruction-file is required" in result.output


def test_cli_run_accepts_tool_policy_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])

    monkeypatch.setattr("native_agent_runner.cli._model_adapter", lambda *_args, **_kwargs: adapter)
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "run",
            "--workspace",
            str(workspace),
            "--instruction",
            "Finish.",
            "--run-root",
            str(tmp_path / "runs"),
            "--allow-tool",
            "fs.read",
            "--allow-tool",
            "run.finish",
        ],
    )

    assert result.exit_code == 0, result.output
    assert {tool.id for tool in adapter.requests[0].tools} == {"fs.read", "run.finish"}
    run_id_line = next(line for line in result.output.splitlines() if line.startswith("run_id: "))
    run_id = run_id_line.removeprefix("run_id: ")
    manifest = json.loads((tmp_path / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tool_policy"]["allowed_tools"] == ["fs.read", "run.finish"]


def test_cli_tool_policy_file_merges_cli_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy_file = tmp_path / "policy.json"
    policy_file.write_text(
        json.dumps({"allowed_tools": ["fs.*", "run.finish"]}),
        encoding="utf-8",
    )
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])

    monkeypatch.setattr("native_agent_runner.cli._model_adapter", lambda *_args, **_kwargs: adapter)
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "run",
            "--workspace",
            str(workspace),
            "--instruction",
            "Finish.",
            "--run-root",
            str(tmp_path / "runs"),
            "--tool-policy-file",
            str(policy_file),
            "--deny-tool",
            "fs.write",
        ],
    )

    assert result.exit_code == 0, result.output
    exposed = {tool.id for tool in adapter.requests[0].tools}
    assert "fs.write" not in exposed
    run_id_line = next(line for line in result.output.splitlines() if line.startswith("run_id: "))
    run_id = run_id_line.removeprefix("run_id: ")
    manifest = json.loads((tmp_path / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
    assert {"tool": "fs.write", "reason": "denied_by_tool_policy"} in manifest["tool_policy"]["hidden_tools"]


def test_cli_run_accepts_permission_policy_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])

    monkeypatch.setattr("native_agent_runner.cli._model_adapter", lambda *_args, **_kwargs: adapter)
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "run",
            "--workspace",
            str(workspace),
            "--instruction",
            "Finish.",
            "--run-root",
            str(tmp_path / "runs"),
            "--deny-path",
            ".env",
            "--redact-path",
            "*.key",
        ],
    )

    assert result.exit_code == 0, result.output
    run_id_line = next(line for line in result.output.splitlines() if line.startswith("run_id: "))
    run_id = run_id_line.removeprefix("run_id: ")
    manifest = json.loads((tmp_path / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["permission_policy"] == {
        "deny_patterns": [".env"],
        "redact_patterns": ["*.key"],
    }


def test_cli_permission_policy_file_merges_cli_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy_file = tmp_path / "permission-policy.json"
    policy_file.write_text(json.dumps({"deny_patterns": ["*.key"]}), encoding="utf-8")
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])

    monkeypatch.setattr("native_agent_runner.cli._model_adapter", lambda *_args, **_kwargs: adapter)
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "run",
            "--workspace",
            str(workspace),
            "--instruction",
            "Finish.",
            "--run-root",
            str(tmp_path / "runs"),
            "--permission-policy-file",
            str(policy_file),
            "--redact-path",
            ".env",
        ],
    )

    assert result.exit_code == 0, result.output
    run_id_line = next(line for line in result.output.splitlines() if line.startswith("run_id: "))
    run_id = run_id_line.removeprefix("run_id: ")
    manifest = json.loads((tmp_path / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["permission_policy"] == {
        "deny_patterns": ["*.key"],
        "redact_patterns": [".env"],
    }


def test_cli_run_accepts_shell_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])

    monkeypatch.setattr("native_agent_runner.cli._model_adapter", lambda *_args, **_kwargs: adapter)
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "run",
            "--workspace",
            str(workspace),
            "--instruction",
            "Finish.",
            "--run-root",
            str(tmp_path / "runs"),
            "--enable-shell",
            "--shell-approval-mode",
            "auto-approve",
            "--shell-timeout-s",
            "30",
            "--shell-max-output-bytes",
            "4096",
            "--shell-env",
            "SAFE_VAR",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "shell.exec" in {tool.id for tool in adapter.requests[0].tools}
    run_id_line = next(line for line in result.output.splitlines() if line.startswith("run_id: "))
    run_id = run_id_line.removeprefix("run_id: ")
    manifest = json.loads((tmp_path / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["shell_policy"]["enabled"] is True
    assert manifest["shell_policy"]["approval_mode"] == "auto-approve"
    assert manifest["shell_policy"]["default_timeout_s"] == 30
    assert manifest["shell_policy"]["default_max_output_bytes"] == 4096
    assert manifest["shell_policy"]["env_allowlist"] == ["SAFE_VAR"]


def test_cli_run_accepts_web_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])

    monkeypatch.setattr("native_agent_runner.cli._model_adapter", lambda *_args, **_kwargs: adapter)
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "run",
            "--workspace",
            str(workspace),
            "--instruction",
            "Finish.",
            "--run-root",
            str(tmp_path / "runs"),
            "--enable-web",
            "--web-gateway-url",
            "http://web-gateway.internal",
            "--web-allow-domain",
            "docs.example.test",
            "--web-block-domain",
            "*.private.test",
            "--web-max-searches",
            "2",
            "--web-max-fetches",
            "3",
            "--enable-web-context",
            "--web-max-contexts",
            "5",
            "--web-max-results",
            "4",
            "--web-context-max-tokens",
            "6000",
            "--web-context-max-urls",
            "6",
            "--web-context-max-snippets",
            "7",
            "--web-max-response-bytes",
            "4096",
            "--web-timeout-s",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    assert {"web.search", "web.fetch", "web.context"}.issubset({tool.id for tool in adapter.requests[0].tools})
    run_id_line = next(line for line in result.output.splitlines() if line.startswith("run_id: "))
    run_id = run_id_line.removeprefix("run_id: ")
    manifest = json.loads((tmp_path / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["web_policy"]["enabled"] is True
    assert manifest["web_policy"]["context_enabled"] is True
    assert manifest["web_policy"]["allowed_domains"] == ["docs.example.test"]
    assert manifest["web_policy"]["blocked_domains"] == ["*.private.test"]
    assert manifest["web_policy"]["max_search_calls"] == 2
    assert manifest["web_policy"]["max_fetch_calls"] == 3
    assert manifest["web_policy"]["max_context_calls"] == 5
    assert manifest["web_policy"]["max_results"] == 4
    assert manifest["web_policy"]["max_context_tokens"] == 6000
    assert manifest["web_policy"]["max_context_urls"] == 6
    assert manifest["web_policy"]["max_context_snippets"] == 7
    assert manifest["web_policy"]["default_max_response_bytes"] == 4096
    assert manifest["web_policy"]["default_timeout_s"] == 10


def test_cli_enable_web_requires_gateway_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])

    monkeypatch.setattr("native_agent_runner.cli._model_adapter", lambda *_args, **_kwargs: adapter)
    result = CliRunner().invoke(
        main,
        [
            "run",
            "--workspace",
            str(workspace),
            "--instruction",
            "Finish.",
            "--enable-web",
        ],
    )

    assert result.exit_code != 0
    assert "--web-gateway-url is required" in result.output


def test_openai_payload_uses_reasoning_effort() -> None:
    adapter = OpenAIModelAdapter(
        ModelConfig(model="gpt-5.5", reasoning=ReasoningConfig(effort="high", summary="detailed"))
    )
    payload = adapter._payload(  # intentionally testing adapter serialization boundary
        request=type(
            "Request",
            (),
            {
                "system_prompt": "sys",
                "tools": (),
                "previous_turn_handle": None,
                "instruction": "hello",
                "observations": (),
            },
        )()
    )

    assert payload["model"] == "gpt-5.5"
    assert payload["reasoning"] == {"effort": "high", "summary": "detailed"}


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or not _openai_responses_available(),
    reason="OPENAI_API_KEY or OpenAI Responses SDK support not available",
)
def test_openai_smoke_example_workspace(tmp_path: Path) -> None:
    from native_agent_runner.core.spec import AgentRunSpec
    from native_agent_runner.loop import AgentLoop

    example = Path(__file__).resolve().parents[1] / "examples" / "workspaces" / "edit_markdown_notes"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text(
        example.joinpath("notes.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    spec = AgentRunSpec(
        instruction="Read notes.md and create a clearer summary in SUMMARY.md.",
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        model=ModelConfig(reasoning=ReasoningConfig(effort="low")),
    )

    result = AgentLoop(
        spec=spec,
        model_adapter=OpenAIModelAdapter(spec.model, allow_direct_provider_api=True),
    ).run()

    if result.status == "failed" and (
        "Connection error" in result.error
        or "insufficient_quota" in result.error
        or "rate_limit" in result.error.lower()
    ):
        pytest.skip("OpenAI smoke dependency unavailable")
    assert result.status in {"completed", "limited"}
    assert result.run_dir.joinpath("events.jsonl").exists()
    assert result.run_dir.joinpath("transcript.jsonl").exists()
    assert result.run_dir.joinpath("metrics.json").exists()


def _run_with_profile(monkeypatch, tmp_path, *extra_args):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    monkeypatch.setattr("native_agent_runner.cli._model_adapter", lambda *_a, **_k: adapter)
    result = CliRunner().invoke(
        main,
        [
            "run",
            "--workspace",
            str(workspace),
            "--instruction",
            "Finish.",
            "--run-root",
            str(tmp_path / "runs"),
            *extra_args,
        ],
    )
    return result, adapter


def test_cli_profile_lightweight_exposes_read_only_tools(monkeypatch, tmp_path: Path) -> None:
    result, adapter = _run_with_profile(monkeypatch, tmp_path, "--profile", "lightweight")
    assert result.exit_code == 0, result.output
    exposed = {tool.id for tool in adapter.requests[0].tools}
    assert "fs.read" in exposed
    assert "fs.write" not in exposed
    assert "shell.exec" not in exposed


def test_cli_profile_heavyweight_exposes_shell_and_web(monkeypatch, tmp_path: Path) -> None:
    # heavyweight enables web, which requires a gateway URL.
    result, adapter = _run_with_profile(
        monkeypatch, tmp_path, "--profile", "heavyweight", "--web-gateway-url", "http://localhost"
    )
    assert result.exit_code == 0, result.output
    exposed = {tool.id for tool in adapter.requests[0].tools}
    assert {"fs.write", "shell.exec", "web.search"}.issubset(exposed)


def test_cli_explicit_flag_overrides_profile(monkeypatch, tmp_path: Path) -> None:
    # lightweight is read-only; --mode propose should re-enable writes.
    result, adapter = _run_with_profile(
        monkeypatch, tmp_path, "--profile", "lightweight", "--mode", "propose"
    )
    assert result.exit_code == 0, result.output
    exposed = {tool.id for tool in adapter.requests[0].tools}
    assert "fs.write" in exposed


def test_cli_profile_conflicts_with_spec(tmp_path: Path) -> None:
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps({"instruction": "x", "workspace_root": "ws"}), encoding="utf-8")
    result = CliRunner().invoke(main, ["run", "--spec", str(spec_file), "--profile", "standard"])
    assert result.exit_code != 0
    assert "--profile cannot be combined with --spec" in result.output
