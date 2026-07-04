from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from support.runtime import runtime_config, tool_binding

from monoid_agent_kernel.cli import main
from monoid_agent_kernel.core.agents import AgentDefinition
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.skills.definition import SKILL_FILENAME


def _write_runtime_config(path: Path, *tool_ids: str) -> Path:
    path.write_text(json.dumps(runtime_config(*tool_ids).to_json()), encoding="utf-8")
    return path


def _write_custom_tools(path: Path) -> Path:
    path.write_text(
        """
from monoid_agent_kernel import tool


@tool(id="custom.echo", side_effect="read")
def echo(text: str) -> dict:
    return {"text": text}


def get_tools(context):
    del context
    return [echo]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_skill(root: Path, name: str = "commit-msg") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / SKILL_FILENAME).write_text(
        "---\n"
        f"name: {name}\n"
        "description: Write commit messages\n"
        "---\n"
        "Write a concise commit message.\n",
        encoding="utf-8",
    )
    return skill_dir


def _write_fork_skill(root: Path, name: str = "reviewer") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / SKILL_FILENAME).write_text(
        "---\n"
        f"name: {name}\n"
        "description: Review code changes\n"
        "context: fork\n"
        "allowed-tools: fs.read skill.read_file\n"
        "---\n"
        "Review the change.\n",
        encoding="utf-8",
    )
    return skill_dir


def test_builder_init_writes_minimal_run_files(tmp_path: Path) -> None:
    target = tmp_path / "agent"

    result = CliRunner().invoke(main, ["builder", "init", "--target", str(target)])

    assert result.exit_code == 0, result.output
    runtime_payload = json.loads((target / "runtime-config.json").read_text(encoding="utf-8"))
    spec_payload = json.loads((target / "run-spec.json").read_text(encoding="utf-8"))
    assert runtime_payload["definition_id"] == "builder-agent"
    assert [item["ref"]["tool_id"] for item in runtime_payload["tools"]] == [
        "fs.read",
        "fs.write",
        "run.finish",
    ]
    assert spec_payload["workspace_root"] == "."
    assert spec_payload["run_root"] == "runs"
    assert "created:" in result.output


def test_builder_init_refuses_existing_files_without_force(tmp_path: Path) -> None:
    target = tmp_path / "agent"
    target.mkdir()
    (target / "runtime-config.json").write_text("{}", encoding="utf-8")

    result = CliRunner().invoke(main, ["builder", "init", "--target", str(target)])

    assert result.exit_code != 0
    assert "pass --force" in result.output


def test_builder_init_can_write_custom_tool_template(tmp_path: Path) -> None:
    target = tmp_path / "agent"

    result = CliRunner().invoke(
        main,
        ["builder", "init", "--target", str(target), "--custom-tool-template"],
    )

    assert result.exit_code == 0, result.output
    assert "custom.word_count" in (target / "tools.py").read_text(encoding="utf-8")


def test_builder_config_validate_accepts_runtime_config_and_custom_tool(tmp_path: Path) -> None:
    config = tmp_path / "runtime-config.json"
    config.write_text(
        json.dumps(
            runtime_config(
                bindings=(tool_binding("custom.echo"), tool_binding("run.finish")),
            ).to_json()
        ),
        encoding="utf-8",
    )
    custom_tools = _write_custom_tools(tmp_path / "tools.py")

    result = CliRunner().invoke(
        main,
        [
            "builder",
            "config",
            "validate",
            "--runtime-config-file",
            str(config),
            "--tool-module",
            f"{custom_tools}:get_tools",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["issues"] == []


def test_builder_config_validate_reports_unknown_tool(tmp_path: Path) -> None:
    config = _write_runtime_config(tmp_path / "runtime-config.json", "missing.tool")

    result = CliRunner().invoke(
        main,
        [
            "builder",
            "config",
            "validate",
            "--runtime-config-file",
            str(config),
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert any("missing.tool" in issue for issue in payload["issues"])


def test_builder_config_validate_accepts_agent_definition_file(tmp_path: Path) -> None:
    definition = AgentDefinition(
        id="defined-agent",
        model=ModelConfig(model="gpt-5.5"),
        tools=(tool_binding("fs.read"), tool_binding("run.finish")),
    )
    definition_file = tmp_path / "agent-definition.json"
    definition_file.write_text(json.dumps(definition.to_json()), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "builder",
            "config",
            "validate",
            "--agent-definition-file",
            str(definition_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "valid"


def test_builder_config_validate_registers_skill_tools(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir)
    config = _write_runtime_config(tmp_path / "runtime-config.json", "run.finish")

    result = CliRunner().invoke(
        main,
        [
            "builder",
            "config",
            "validate",
            "--runtime-config-file",
            str(config),
            "--skills-directory",
            str(skills_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["bound_tool_count"] >= 4


def test_builder_config_validate_registers_fork_skill_subagents(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write_fork_skill(skills_dir)
    config = _write_runtime_config(tmp_path / "runtime-config.json", "agent.spawn", "run.finish")

    result = CliRunner().invoke(
        main,
        [
            "builder",
            "config",
            "validate",
            "--runtime-config-file",
            str(config),
            "--skills-directory",
            str(skills_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["issues"] == []


def test_builder_tools_list_marks_runtime_bound_tools(tmp_path: Path) -> None:
    config = _write_runtime_config(tmp_path / "runtime-config.json", "fs.read", "run.finish")

    result = CliRunner().invoke(
        main,
        [
            "builder",
            "tools",
            "list",
            "--runtime-config-file",
            str(config),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    tools = {item["id"]: item for item in payload["tools"]}
    assert tools["fs.read"]["bound"] is True
    assert tools["fs.read"]["binding_ids"] == ["fs.read"]
    assert tools["fs.write"]["bound"] is False
    assert tools["tool.search"]["bound"] is True


def test_builder_tools_list_includes_custom_and_skill_tools(tmp_path: Path) -> None:
    custom_tools = _write_custom_tools(tmp_path / "tools.py")
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir)

    result = CliRunner().invoke(
        main,
        [
            "builder",
            "tools",
            "list",
            "--tool-module",
            f"{custom_tools}:get_tools",
            "--skills-directory",
            str(skills_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    tool_ids = {item["id"] for item in payload["tools"]}
    assert {"custom.echo", "skill", "skill.read_file", "skill.run_script"} <= tool_ids
