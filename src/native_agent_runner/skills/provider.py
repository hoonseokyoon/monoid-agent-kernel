"""A set of Agent Skills exposed to the engine through existing extension seams.

:class:`SkillProvider` implements **both** the ``ContextProvider`` and ``ToolProvider``
protocols, so a single instance attaches to ``AgentLoop`` without touching the core loop
(the same zero-core-change pattern MCP uses):

* as a ``ContextProvider`` — :meth:`static_segment` advertises the skill catalog (L1);
* as a ``ToolProvider`` — :meth:`get_tools` yields the ``skill`` tool (load a skill's
  instructions, L2) and ``skill.read_file`` (read a bundled resource on demand, L3).

Register the one instance in both ``AgentLoop.context_providers`` and
``AgentLoop.tool_providers``. Provider tools are not auto-bound; use :meth:`tool_bindings`
to merge bindings into the runtime config (mirrors ``McpToolProvider.tool_bindings``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from native_agent_runner.core.agents import RegistryToolRef, ToolBinding
from native_agent_runner.core.context import TurnContext
from native_agent_runner.skills.definition import SKILL_FILENAME, SkillDefinition
from native_agent_runner.tools.base import ToolResult, ToolSpec

SKILL_TOOL_ID = "skill"
SKILL_READ_FILE_TOOL_ID = "skill.read_file"

_MAX_RESOURCE_FILES = 100
_MAX_RESOURCE_BYTES = 1_000_000


class SkillProvider:
    """Progressive-disclosure Agent Skills as a context + tool provider."""

    def __init__(self, definitions: Mapping[str, SkillDefinition]) -> None:
        self._definitions: dict[str, SkillDefinition] = dict(definitions)

    # -- ContextProvider (L1: always-resident catalog) ---------------------------------

    def static_segment(self) -> str | None:
        if not self._definitions:
            return None
        lines = [
            "# Available Skills",
            "",
            "You have access to the skills below — focused instructions for specific "
            f"tasks. When a request matches a skill's purpose, call the `{SKILL_TOOL_ID}` "
            "tool with its name to load the full instructions before proceeding. Only "
            "load a skill when it is relevant.",
            "",
        ]
        lines.extend(self._catalog_lines())
        return "\n".join(lines)

    def dynamic_segment(self, turn: TurnContext) -> str | None:  # noqa: ARG002 - no per-turn context
        # Once activated, a skill's L2 instructions live in tool-result history, so there
        # is nothing to inject per turn.
        return None

    # -- ToolProvider (L2: load instructions, L3: read a bundled resource) --------------

    def get_tools(self, context: Any = None) -> Iterable[ToolSpec]:  # noqa: ARG002 - context unused
        if not self._definitions:
            return
        names = sorted(self._definitions)
        catalog = "\n".join(self._catalog_lines())
        yield ToolSpec(
            id=SKILL_TOOL_ID,
            description=(
                "Load an Agent Skill's full instructions into context. Choose 'name' from "
                "the available skills; the tool returns the skill's instructions plus a "
                "manifest of bundled resource files (read those on demand with "
                f"'{SKILL_READ_FILE_TOOL_ID}'). Only load a skill when it is relevant.\n\n"
                f"Available skills:\n{catalog}"
            ),
            input_schema=_object_schema(
                {"name": {"type": "string", "enum": names}},
                required=["name"],
            ),
            capability="skill",
            side_effect="read",
            handler=self._make_skill_handler(),
        )
        yield ToolSpec(
            id=SKILL_READ_FILE_TOOL_ID,
            description=(
                "Read a bundled resource file from a skill's directory (Level 3 of a "
                "skill). Provide the skill 'name' and the resource 'path' relative to the "
                "skill directory, as listed in that skill's 'resources' manifest."
            ),
            input_schema=_object_schema(
                {
                    "name": {"type": "string", "enum": names},
                    "path": {"type": "string"},
                },
                required=["name", "path"],
            ),
            capability="skill",
            side_effect="read",
            handler=self._make_read_file_handler(),
        )

    def tool_bindings(self) -> tuple[ToolBinding, ...]:
        """Bindings for the skill tools, to merge into the runtime config so the run can
        see them (provider tools are not auto-bound)."""
        return tuple(
            ToolBinding(binding_id=spec.id, ref=RegistryToolRef(tool_id=spec.id), authorization="allow")
            for spec in self.get_tools()
        )

    # -- internals ---------------------------------------------------------------------

    def _catalog_lines(self) -> list[str]:
        lines: list[str] = []
        for name in sorted(self._definitions):
            description = self._definitions[name].description.strip()
            lines.append(f"- {name}: {description}" if description else f"- {name}")
        return lines

    def _make_skill_handler(self):
        def handler(_context: Any, args: dict[str, Any]) -> ToolResult:
            name = str(args.get("name") or "")
            definition = self._definitions.get(name)
            if definition is None:
                return ToolResult(ok=False, error=f"unknown skill: {name}", error_code="skill_unknown")
            content: dict[str, Any] = {"name": definition.name, "instructions": definition.instructions}
            if definition.allowed_tools:
                # Advisory (Claude parity): a hint, not an enforced restriction.
                content["allowed_tools"] = list(definition.allowed_tools)
            resources = _list_resources(definition.directory)
            if resources:
                content["resources"] = resources
            # Best-effort observability: record the activation if the context exposes the
            # hook (the engine's AgentToolContext does; bare test stubs do not). Duck-typed
            # so skills stay decoupled from the core ToolContext contract.
            record = getattr(_context, "record_skill_activation", None)
            if callable(record):
                record(definition.name, resource_count=len(resources))
            return ToolResult(ok=True, content=content)

        return handler

    def _make_read_file_handler(self):
        def handler(_context: Any, args: dict[str, Any]) -> ToolResult:
            name = str(args.get("name") or "")
            rel = str(args.get("path") or "")
            definition = self._definitions.get(name)
            if definition is None:
                return ToolResult(ok=False, error=f"unknown skill: {name}", error_code="skill_unknown")
            if definition.directory is None:
                return ToolResult(
                    ok=False, error=f"skill has no bundled resources: {name}", error_code="skill_no_resources"
                )
            try:
                text = _read_resource(definition.directory, rel)
            except _ResourceError as exc:
                return ToolResult(ok=False, error=str(exc), error_code=exc.code)
            return ToolResult(ok=True, content={"name": name, "path": rel, "content": text})

        return handler


def _object_schema(properties: dict[str, Any], *, required: list[str] | None = None) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _list_resources(directory: Path | None) -> list[str]:
    """Relative posix paths of bundled resource files under ``directory`` (the skill's
    own ``SKILL.md`` is excluded — it is the L2 payload, not a resource)."""
    if directory is None or not directory.is_dir():
        return []
    out: list[str] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == SKILL_FILENAME:
            continue
        out.append(path.relative_to(directory).as_posix())
        if len(out) >= _MAX_RESOURCE_FILES:
            break
    return out


class _ResourceError(Exception):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


def _read_resource(directory: Path, rel: str) -> str:
    """Read a bundled resource, guarding against path traversal (the resolved path must
    stay within the skill directory) and non-text content."""
    root = directory.resolve()
    candidate = (root / rel).resolve()
    if root != candidate and root not in candidate.parents:
        raise _ResourceError("path escapes the skill directory", "skill_path_invalid")
    if candidate.name == SKILL_FILENAME and candidate.parent == root:
        raise _ResourceError("SKILL.md is loaded via the skill tool, not as a resource", "skill_path_invalid")
    if not candidate.is_file():
        raise _ResourceError(f"resource not found: {rel}", "skill_resource_missing")
    data = candidate.read_bytes()
    if len(data) > _MAX_RESOURCE_BYTES:
        raise _ResourceError(f"resource too large: {rel}", "skill_resource_too_large")
    if b"\x00" in data:
        raise _ResourceError(f"binary resource cannot be read as text: {rel}", "skill_resource_binary")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _ResourceError(f"resource is not utf-8 text: {rel}", "skill_resource_binary") from exc
