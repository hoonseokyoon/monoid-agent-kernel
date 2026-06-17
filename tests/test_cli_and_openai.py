from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from conftest import runtime_config

from native_agent_runner.cli import main
from native_agent_runner.core.spec import ModelConfig, ReasoningConfig
from native_agent_runner.providers.base import ModelRequest, ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter
from native_agent_runner.providers.openai import OpenAIModelAdapter


def _openai_responses_available() -> bool:
    try:
        from openai import OpenAI
    except ImportError:
        return False
    return hasattr(OpenAI(api_key="test"), "responses")


def _write_config(path: Path, *tool_ids: str, model: ModelConfig | None = None) -> Path:
    path.write_text(
        json.dumps(runtime_config(*tool_ids, model=model).to_json()),
        encoding="utf-8",
    )
    return path


def test_cli_requires_runtime_config(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = CliRunner().invoke(main, ["run", "--workspace", str(workspace), "--instruction", "Finish."])

    assert result.exit_code != 0
    assert "--runtime-config-file or --agent-definition-file is required" in result.output


def test_cli_run_accepts_runtime_config_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    monkeypatch.setattr("native_agent_runner.cli._model_adapter", lambda *_args, **_kwargs: adapter)
    config_file = _write_config(tmp_path / "runtime.json", "fs.read", "run.finish")

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
            "--runtime-config-file",
            str(config_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert {tool.id for tool in adapter.requests[0].tools} == {"fs.read", "run.finish"}
    run_id = next(line for line in result.output.splitlines() if line.startswith("run_id: ")).removeprefix("run_id: ")
    manifest = json.loads((tmp_path / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["agent_config"]["definition_id"] == "test-agent"


def test_cli_spec_file_pairs_with_runtime_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(
        json.dumps({"instruction": "Finish.", "workspace_root": str(workspace), "run_root": str(tmp_path / "runs")}),
        encoding="utf-8",
    )
    config_file = _write_config(tmp_path / "runtime.json", "run.finish")
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    monkeypatch.setattr("native_agent_runner.cli._model_adapter", lambda *_args, **_kwargs: adapter)

    result = CliRunner().invoke(main, ["run", "--spec", str(spec_file), "--runtime-config-file", str(config_file)])

    assert result.exit_code == 0, result.output
    assert {tool.id for tool in adapter.requests[0].tools} == {"run.finish"}


def test_cli_permission_policy_flags_remain_run_boundary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    monkeypatch.setattr("native_agent_runner.cli._model_adapter", lambda *_args, **_kwargs: adapter)
    config_file = _write_config(tmp_path / "runtime.json", "run.finish")

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
            "--runtime-config-file",
            str(config_file),
            "--deny-path",
            ".env",
            "--redact-path",
            "*.key",
        ],
    )

    assert result.exit_code == 0, result.output
    run_id = next(line for line in result.output.splitlines() if line.startswith("run_id: ")).removeprefix("run_id: ")
    manifest = json.loads((tmp_path / "runs" / run_id / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["permission_policy"] == {
        "deny_patterns": [".env"],
        "redact_patterns": ["*.key"],
    }


def test_cli_requires_web_gateway_for_web_bindings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="done")])
    monkeypatch.setattr("native_agent_runner.cli._model_adapter", lambda *_args, **_kwargs: adapter)
    config_file = _write_config(tmp_path / "runtime.json", "web.search", "run.finish")

    result = CliRunner().invoke(
        main,
        [
            "run",
            "--workspace",
            str(workspace),
            "--instruction",
            "Finish.",
            "--runtime-config-file",
            str(config_file),
        ],
    )

    assert result.exit_code != 0
    assert "runtime config binds web tools; --web-gateway-url is required" in result.output


def test_openai_payload_uses_turn_model_config() -> None:
    adapter = OpenAIModelAdapter(ModelConfig(model="fallback"))
    request = ModelRequest(
        instruction="hello",
        system_prompt="sys",
        tools=(),
        model=ModelConfig(model="gpt-5.5", reasoning=ReasoningConfig(effort="high", summary="detailed")),
    )

    payload = adapter._payload(request)

    assert payload["model"] == "gpt-5.5"
    assert payload["reasoning"] == {"effort": "high", "summary": "detailed"}


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or not _openai_responses_available(),
    reason="OPENAI_API_KEY or OpenAI Responses SDK support not available",
)
def test_openai_smoke_payload_only() -> None:
    adapter = OpenAIModelAdapter(ModelConfig(), allow_direct_provider_api=True)
    request = ModelRequest(instruction="Say ok.", system_prompt="sys", tools=())

    payload = adapter._payload(request)

    assert payload["input"] == [{"role": "user", "content": "Say ok."}]
