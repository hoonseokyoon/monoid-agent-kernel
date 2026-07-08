from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import click

from monoid_agent_kernel.core.agents import (
    AgentDefinition,
    AgentRuntimeConfig,
    ToolBinding,
    collect_runtime_config_issues,
)
from monoid_agent_kernel.skills import SkillProvider, load_skill_definitions
from monoid_agent_kernel.subagent_loader import load_subagent_definitions
from monoid_agent_kernel.tool_loader import load_tool_provider
from monoid_agent_kernel.tools.base import ToolRegistry, ToolSpec
from monoid_agent_kernel.tools.builtin import agent_spawn_tool
from monoid_agent_kernel.tools.defaults import default_tool_bindings
from monoid_agent_kernel.tools.tool_ids import list_builtin_tools


@click.group("builder")
def builder_group() -> None:
    """Scaffold and inspect Monoid builder configuration."""


@builder_group.command("init")
@click.option("--target", type=click.Path(path_type=Path), required=True)
@click.option("--force", is_flag=True, help="Overwrite existing scaffold files.")
@click.option("--custom-tool-template", is_flag=True, help="Also write a tools.py provider template.")
def builder_init(target: Path, force: bool, custom_tool_template: bool) -> None:
    """Create minimal files for a local monoid run."""
    target.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {
        "runtime-config.json": _pretty_json(_default_runtime_config()),
        "run-spec.json": _pretty_json(_default_run_spec()),
    }
    if custom_tool_template:
        files["tools.py"] = _CUSTOM_TOOLS_TEMPLATE

    paths = {name: target / name for name in files}
    if not force:
        collisions = [path for path in paths.values() if path.exists()]
        if collisions:
            names = ", ".join(str(path) for path in collisions)
            raise click.ClickException(f"{names} already exists; pass --force to overwrite")

    written: list[Path] = []
    for name, content in files.items():
        path = paths[name]
        path.write_text(content, encoding="utf-8")
        written.append(path)

    for path in written:
        click.echo(f"created: {path}")


@builder_group.group("config")
def builder_config_group() -> None:
    """Inspect and validate builder config files."""


@builder_config_group.command("validate")
@click.option("--runtime-config-file", type=click.Path(path_type=Path), default=None)
@click.option("--agent-definition-file", type=click.Path(path_type=Path), default=None)
@click.option("--tool-module", multiple=True, help="Load custom tools from path.py:function.")
@click.option(
    "--skills-directory",
    type=click.Path(path_type=Path),
    default=None,
    help="Load Agent Skills from a directory.",
)
@click.option(
    "--agents-directory",
    type=click.Path(path_type=Path),
    default=None,
    help="Load subagent definitions from a directory.",
)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
@click.pass_context
def builder_config_validate(
    ctx: click.Context,
    *,
    runtime_config_file: Path | None,
    agent_definition_file: Path | None,
    tool_module: tuple[str, ...],
    skills_directory: Path | None,
    agents_directory: Path | None,
    json_output: bool,
) -> None:
    """Validate a runtime config against the tools the run would register."""
    try:
        config = _load_agent_runtime_config(runtime_config_file, agent_definition_file)
        registry, registration_issues, skill_bindings = _build_tool_registry(
            tool_modules=tool_module,
            skills_directory=skills_directory,
            agents_directory=agents_directory,
        )
        effective_config = (
            replace(config, tools=config.tools + skill_bindings) if skill_bindings else config
        )
        issues = registration_issues + collect_runtime_config_issues(effective_config, registry)
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    payload = {
        "ok": not issues,
        "definition_id": config.definition_id,
        "tool_count": len(registry.specs()),
        "bound_tool_count": len(effective_config.tools)
        + (1 if effective_config.tool_search.enabled else 0),
        "issues": issues,
    }
    if json_output:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    elif issues:
        click.echo("invalid")
        for issue in issues:
            click.echo(f"- {issue}")
    else:
        click.echo("valid")

    if issues:
        ctx.exit(1)


@builder_group.group("tools")
def builder_tools_group() -> None:
    """Inspect available builder tools."""


@builder_tools_group.command("list")
@click.option("--tool-module", multiple=True, help="Load custom tools from path.py:function.")
@click.option(
    "--skills-directory",
    type=click.Path(path_type=Path),
    default=None,
    help="Load Agent Skills from a directory.",
)
@click.option(
    "--runtime-config-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Mark tools bound by a runtime config.",
)
@click.option("--json", "json_output", is_flag=True, help="Print machine-readable JSON.")
def builder_tools_list(
    *,
    tool_module: tuple[str, ...],
    skills_directory: Path | None,
    runtime_config_file: Path | None,
    json_output: bool,
) -> None:
    """List builtin, custom, and skill tools."""
    try:
        registry, registration_issues, skill_bindings = _build_tool_registry(
            tool_modules=tool_module,
            skills_directory=skills_directory,
            agents_directory=None,
        )
        if registration_issues:
            raise click.ClickException("; ".join(registration_issues))
        config = _load_runtime_config_file(runtime_config_file) if runtime_config_file else None
        effective_config = (
            replace(config, tools=config.tools + skill_bindings)
            if config is not None and skill_bindings
            else config
        )
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    binding_ids_by_tool = _binding_ids_by_tool(effective_config)
    tools = [
        _tool_payload(spec, binding_ids_by_tool.get(spec.id, ()))
        for spec in sorted(registry.specs(), key=lambda item: item.id)
    ]
    payload = {"tools": tools}
    if json_output:
        click.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return

    for tool in tools:
        marker = " [bound]" if tool["bound"] else ""
        click.echo(f"{tool['id']}{marker} - {tool['description']}")


def _load_agent_runtime_config(
    runtime_config_file: Path | None,
    agent_definition_file: Path | None,
) -> AgentRuntimeConfig:
    if runtime_config_file is not None and agent_definition_file is not None:
        raise click.ClickException("--runtime-config-file and --agent-definition-file are mutually exclusive")
    if runtime_config_file is None and agent_definition_file is None:
        raise click.ClickException("--runtime-config-file or --agent-definition-file is required")
    if runtime_config_file is not None:
        return _load_runtime_config_file(runtime_config_file)

    assert agent_definition_file is not None
    try:
        payload = json.loads(agent_definition_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"invalid agent config JSON: {exc.msg}") from exc
    try:
        return AgentRuntimeConfig.from_definition(AgentDefinition.from_json(payload))
    except Exception as exc:
        raise click.ClickException(f"failed to load agent runtime config: {exc}") from exc


def _load_runtime_config_file(path: Path) -> AgentRuntimeConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"invalid agent config JSON: {exc.msg}") from exc
    try:
        return AgentRuntimeConfig.from_json(payload)
    except Exception as exc:
        raise click.ClickException(f"failed to load agent runtime config: {exc}") from exc


def _build_tool_registry(
    *,
    tool_modules: tuple[str, ...],
    skills_directory: Path | None,
    agents_directory: Path | None,
) -> tuple[ToolRegistry, list[str], tuple[ToolBinding, ...]]:
    registry = ToolRegistry()
    issues: list[str] = []
    skill_bindings: tuple[ToolBinding, ...] = ()
    subagents: dict[str, Any] = {}

    _register_many(registry, list_builtin_tools(), issues)

    for item in tool_modules:
        provider = load_tool_provider(item)
        _register_many(registry, provider.get_tools(None), issues)  # type: ignore[arg-type]

    if skills_directory is not None:
        skill_definitions = load_skill_definitions(skills_directory)
        if skill_definitions:
            skill_provider = SkillProvider(skill_definitions)
            skill_tools = tuple(skill_provider.get_tools(None))
            _register_many(registry, skill_tools, issues)
            skill_bindings = skill_provider.tool_bindings()
            subagents.update(skill_provider.subagent_definitions())

    if agents_directory is not None:
        subagents.update(load_subagent_definitions(agents_directory))
    if subagents:
        _register_many(
            registry,
            [agent_spawn_tool({name: definition.description for name, definition in subagents.items()})],
            issues,
        )

    return registry, issues, skill_bindings


def _register_many(registry: ToolRegistry, specs: Any, issues: list[str]) -> None:
    for spec in specs:
        try:
            registry.register(spec)
        except ValueError as exc:
            issues.append(str(exc))


def _binding_ids_by_tool(config: AgentRuntimeConfig | None) -> dict[str, tuple[str, ...]]:
    if config is None:
        return {}
    bindings: dict[str, list[str]] = {}
    for binding in config.tools:
        bindings.setdefault(binding.ref.tool_id, []).append(binding.binding_id)
    if config.tool_search.enabled:
        bindings.setdefault("tool.search", []).append(config.tool_search.binding_id)
    return {tool_id: tuple(ids) for tool_id, ids in bindings.items()}


def _tool_payload(spec: ToolSpec, binding_ids: tuple[str, ...]) -> dict[str, Any]:
    return {
        "id": spec.id,
        "exported_name": spec.exported_name,
        "capability": spec.capability,
        "side_effect": spec.side_effect,
        "description": spec.description.split("\n", 1)[0],
        "bound": bool(binding_ids),
        "binding_ids": list(binding_ids),
    }


def _default_runtime_config() -> dict[str, Any]:
    default_bindings = [
        *(binding.to_json() for binding in default_tool_bindings("read")),
        *(binding.to_json() for binding in default_tool_bindings("write")),
        ToolBinding.for_tool("run.finish", binding_id="finish", model_name="finish").to_json(),
    ]
    return {
        "definition_id": "builder-agent",
        "config_version": 1,
        "model": {
            "provider": "gateway",
            "model": "gpt-5.5",
            "gateway_url": "http://127.0.0.1:8080/internal/llm/turns",
            "reasoning": {"effort": "low", "summary": "off"},
        },
        "prompt": {
            "persona_segments": ["Work directly in the workspace and keep changes focused."],
            "runtime_segments": [],
        },
        "tools": default_bindings,
        "tool_search": {"enabled": True, "top_k": 5},
    }


def _default_run_spec() -> dict[str, Any]:
    return {
        "workspace_root": ".",
        "run_root": "runs",
        "mode": "propose",
        "workspace_backend": "overlay",
        "limits": {"max_steps": 30},
    }


def _pretty_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


_CUSTOM_TOOLS_TEMPLATE = '''from __future__ import annotations

from monoid_agent_kernel import tool


@tool(id="custom.word_count", side_effect="read")
def word_count(text: str) -> dict:
    """Count words in a text string."""
    words = [part for part in text.split() if part]
    return {"words": len(words)}


def get_tools(context) -> list:
    del context
    return [word_count]
'''
