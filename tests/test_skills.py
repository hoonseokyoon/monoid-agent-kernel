from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import runtime_config, runtime_provider

from native_agent_runner.core.spec import AgentRunSpec
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call
from native_agent_runner.skills import SkillDefinition, SkillProvider, load_skill_definitions
from native_agent_runner.skills.definition import SKILL_FILENAME


# --- fixtures / helpers ------------------------------------------------------------


def _write_skill(
    root: Path,
    name: str,
    *,
    description: str = "",
    allowed_tools: str = "",
    body: str = "Do the thing.",
    resources: dict[str, str] | None = None,
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {name}"]
    if description:
        lines.append(f"description: {description}")
    if allowed_tools:
        lines.append(f"allowed-tools: {allowed_tools}")
    lines += ["---", "", body, ""]
    (skill_dir / SKILL_FILENAME).write_text("\n".join(lines), encoding="utf-8")
    for rel, content in (resources or {}).items():
        target = skill_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return skill_dir


def _specs(provider: SkillProvider) -> dict[str, object]:
    return {spec.id: spec for spec in provider.get_tools()}  # type: ignore[attr-defined]


# --- loader (directory discovery) --------------------------------------------------


def test_loader_parses_skill_directories(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "pdf-fill",
        description="Fill PDF forms",
        allowed_tools="Read Glob",
        body="Use the bundled script.",
        resources={"references/FORMS.md": "FORMS_CONTENT"},
    )
    _write_skill(tmp_path, "commit-msg", description="Write commit messages")

    definitions = load_skill_definitions(tmp_path)

    assert set(definitions) == {"pdf-fill", "commit-msg"}
    pdf = definitions["pdf-fill"]
    assert pdf.description == "Fill PDF forms"
    assert pdf.instructions == "Use the bundled script."
    assert pdf.allowed_tools == ("Read", "Glob")
    assert pdf.directory == tmp_path / "pdf-fill"


def test_loader_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="skills directory not found"):
        load_skill_definitions(tmp_path / "nope")


def test_loader_name_falls_back_to_directory_name(tmp_path: Path) -> None:
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / SKILL_FILENAME).write_text("---\ndescription: d\n---\nbody\n", encoding="utf-8")

    definitions = load_skill_definitions(tmp_path)

    assert set(definitions) == {"my-skill"}
    assert definitions["my-skill"].description == "d"


def test_loader_duplicate_name_first_wins(tmp_path: Path) -> None:
    _write_skill(tmp_path / "a", "dup", description="first")
    _write_skill(tmp_path / "b", "dup", description="second")

    definitions = load_skill_definitions(tmp_path)

    # sorted path order: a/ before b/, so "first" wins.
    assert definitions["dup"].description == "first"


def test_from_frontmatter_allowed_tools_inline_list() -> None:
    definition = SkillDefinition.from_frontmatter(
        {"name": "x", "allowed-tools": ["fs.read", "shell.exec"]}, "body"
    )
    assert definition.allowed_tools == ("fs.read", "shell.exec")


# --- L1: catalog (static_segment) --------------------------------------------------


def test_static_segment_lists_catalog(tmp_path: Path) -> None:
    _write_skill(tmp_path, "pdf-fill", description="Fill PDF forms")
    _write_skill(tmp_path, "commit-msg", description="Write commit messages")
    provider = SkillProvider(load_skill_definitions(tmp_path))

    segment = provider.static_segment()

    assert segment is not None
    assert "- pdf-fill: Fill PDF forms" in segment
    assert "- commit-msg: Write commit messages" in segment
    assert "`skill`" in segment  # tells the model how to load one


def test_empty_provider_is_inert() -> None:
    provider = SkillProvider({})
    assert provider.static_segment() is None
    assert list(provider.get_tools()) == []
    assert provider.tool_bindings() == ()


# --- L2: skill tool (load instructions) --------------------------------------------


def test_skill_tool_returns_instructions_resources_and_advisory(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "pdf-fill",
        description="Fill PDF forms",
        allowed_tools="Read Glob",
        body="Step 1. Step 2.",
        resources={"references/FORMS.md": "x", "scripts/fill.py": "print(1)"},
    )
    provider = SkillProvider(load_skill_definitions(tmp_path))
    skill = _specs(provider)["skill"]

    result = skill.handler(None, {"name": "pdf-fill"})

    assert result.ok
    assert result.content["instructions"] == "Step 1. Step 2."
    assert result.content["allowed_tools"] == ["Read", "Glob"]  # advisory
    assert set(result.content["resources"]) == {"references/FORMS.md", "scripts/fill.py"}


def test_skill_tool_unknown_name(tmp_path: Path) -> None:
    _write_skill(tmp_path, "pdf-fill", description="d")
    provider = SkillProvider(load_skill_definitions(tmp_path))
    skill = _specs(provider)["skill"]

    result = skill.handler(None, {"name": "ghost"})

    assert not result.ok
    assert result.error_code == "skill_unknown"


def test_skill_tool_enum_constrains_names(tmp_path: Path) -> None:
    _write_skill(tmp_path, "pdf-fill", description="d")
    provider = SkillProvider(load_skill_definitions(tmp_path))
    skill = _specs(provider)["skill"]
    assert skill.input_schema["properties"]["name"]["enum"] == ["pdf-fill"]


# --- L3: skill.read_file (read a bundled resource) ---------------------------------


def test_read_file_returns_bundled_content(tmp_path: Path) -> None:
    _write_skill(tmp_path, "pdf-fill", resources={"references/FORMS.md": "FORMS_CONTENT"})
    provider = SkillProvider(load_skill_definitions(tmp_path))
    read = _specs(provider)["skill.read_file"]

    result = read.handler(None, {"name": "pdf-fill", "path": "references/FORMS.md"})

    assert result.ok
    assert result.content["content"] == "FORMS_CONTENT"


def test_read_file_rejects_traversal(tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("TOPSECRET", encoding="utf-8")
    _write_skill(tmp_path, "pdf-fill")
    provider = SkillProvider(load_skill_definitions(tmp_path))
    read = _specs(provider)["skill.read_file"]

    result = read.handler(None, {"name": "pdf-fill", "path": "../secret.txt"})

    assert not result.ok
    assert result.error_code == "skill_path_invalid"


def test_read_file_rejects_skill_md(tmp_path: Path) -> None:
    _write_skill(tmp_path, "pdf-fill")
    provider = SkillProvider(load_skill_definitions(tmp_path))
    read = _specs(provider)["skill.read_file"]

    result = read.handler(None, {"name": "pdf-fill", "path": SKILL_FILENAME})

    assert not result.ok
    assert result.error_code == "skill_path_invalid"


def test_read_file_missing_resource(tmp_path: Path) -> None:
    _write_skill(tmp_path, "pdf-fill")
    provider = SkillProvider(load_skill_definitions(tmp_path))
    read = _specs(provider)["skill.read_file"]

    result = read.handler(None, {"name": "pdf-fill", "path": "references/NOPE.md"})

    assert not result.ok
    assert result.error_code == "skill_resource_missing"


# --- bindings ----------------------------------------------------------------------


def test_tool_bindings_cover_both_tools(tmp_path: Path) -> None:
    _write_skill(tmp_path, "pdf-fill")
    provider = SkillProvider(load_skill_definitions(tmp_path))
    bound = {b.ref.tool_id for b in provider.tool_bindings()}
    assert bound == {"skill", "skill.read_file"}


# --- E2E: progressive disclosure through a real run --------------------------------


def test_e2e_model_loads_skill_then_reads_resource(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    _write_skill(
        skills_root,
        "pdf-fill",
        description="Fill PDF forms",
        body="INSTRUCTIONS_BODY",
        resources={"references/FORMS.md": "FORMS_CONTENT"},
    )
    provider = SkillProvider(load_skill_definitions(skills_root))

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(tool_calls=(fake_tool_call("skill", {"name": "pdf-fill"}, "c1"),)),
            ModelTurn(
                tool_calls=(
                    fake_tool_call(
                        "skill_read_file",
                        {"name": "pdf-fill", "path": "references/FORMS.md"},
                        "c2",
                    ),
                )
            ),
            ModelTurn(final_text="done"),
        ]
    )
    AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        context_providers=(provider,),
        tool_providers=(provider,),
        runtime_config_provider=runtime_provider(runtime_config(bindings=provider.tool_bindings())),
    ).run_once("go")

    # L1: the catalog is in the very first system prompt.
    assert "pdf-fill: Fill PDF forms" in adapter.requests[0].system_prompt
    # L2 + L3: the instructions and the on-demand resource both reached the model.
    outputs = json.dumps([obs.output for req in adapter.requests for obs in req.observations])
    assert "INSTRUCTIONS_BODY" in outputs
    assert "FORMS_CONTENT" in outputs
