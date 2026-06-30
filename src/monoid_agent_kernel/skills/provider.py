"""A set of Agent Skills exposed to the engine through existing extension seams.

:class:`SkillProvider` implements **both** the ``ContextProvider`` and ``ToolProvider``
protocols, so a single instance attaches to ``AgentLoop`` without touching the core loop
(the same zero-core-change pattern MCP uses):

* as a ``ContextProvider`` — :meth:`dynamic_segment` advertises the skill catalog (L1),
  gated on the ``skill`` tool being bound this turn (so it tracks a capability hot-swap);
* as a ``ToolProvider`` — :meth:`get_tools` yields the ``skill`` tool (load a skill's
  instructions, L2) and ``skill.read_file`` (read a bundled resource on demand, L3).

Register the one instance in both ``AgentLoop.context_providers`` and
``AgentLoop.tool_providers``. Provider tools are not auto-bound; use :meth:`tool_bindings`
to merge bindings into the runtime config (mirrors ``McpToolProvider.tool_bindings``).
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core.agents import PromptSpec, RegistryToolRef, SubagentDefinition, ToolBinding
from monoid_agent_kernel.core.context import TurnContext
from monoid_agent_kernel.skills.definition import SKILL_FILENAME, SkillDefinition
from monoid_agent_kernel.tools.base import ToolResult, ToolSpec

SKILL_TOOL_ID = "skill"
SKILL_READ_FILE_TOOL_ID = "skill.read_file"
SKILL_RUN_SCRIPT_TOOL_ID = "skill.run_script"

# Fork skills are exposed to the subagent machine under this namespace so their synthesized
# SubagentDefinitions never collide with operator-defined subagents (``--agents-directory``).
SKILL_SUBAGENT_PREFIX = "skill:"

_MAX_RESOURCE_FILES = 100
_MAX_RESOURCE_BYTES = 1_000_000

# Map a bundled script's extension to the argv prefix that runs it. ``sys.executable`` (the
# runner's own Python) backs ``.py`` so a Python script always has an interpreter on hand;
# the rest rely on the interpreter being on PATH. Scripts are run by argv — never through a
# shell — so the source never enters context and the args are never re-parsed by a shell.
_INTERPRETERS: dict[str, list[str]] = {
    ".py": [sys.executable],
    ".sh": ["bash"],
    ".bash": ["bash"],
    ".js": ["node"],
    ".mjs": ["node"],
    ".rb": ["ruby"],
    ".ps1": ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"],
}


class SkillProvider:
    """Progressive-disclosure Agent Skills as a context + tool provider."""

    def __init__(self, definitions: Mapping[str, SkillDefinition]) -> None:
        self._definitions: dict[str, SkillDefinition] = dict(definitions)

    # -- ContextProvider (L1: catalog, gated on the skill tool being bound) -------------

    def static_segment(self) -> str | None:
        # The catalog is config-gated (it must vanish if the skill tool is unbound, e.g. a
        # capability toggled off mid-run), which a once-at-bootstrap static segment can't
        # express — so it is emitted per-turn by dynamic_segment instead. A catalog advertising
        # a skill the model has no tool to activate would be misleading.
        return None

    def dynamic_segment(self, turn: TurnContext) -> str | None:
        if not self._definitions or SKILL_TOOL_ID not in turn.bound_tools:
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

    # -- ToolProvider (L2: load instructions, L3: read a bundled resource) --------------

    def get_tools(self, context: Any = None) -> Iterable[ToolSpec]:  # noqa: ARG002 - context unused
        if not self._definitions:
            return
        names = sorted(self._definitions)
        catalog = "\n".join(self._catalog_lines())
        yield ToolSpec(
            id=SKILL_TOOL_ID,
            description=(
                "Activate an Agent Skill. For most skills this loads the skill's full "
                "instructions into context (plus a manifest of bundled resource files, read "
                f"on demand with '{SKILL_READ_FILE_TOOL_ID}'). Some skills run in a separate "
                "context instead: for those, provide 'task' describing what to accomplish and "
                "the tool returns only the skill's final result. Choose 'name' from the "
                "available skills; only activate a skill when it is relevant.\n\n"
                f"Available skills:\n{catalog}"
            ),
            input_schema=_object_schema(
                {
                    "name": {"type": "string", "enum": names},
                    "task": {"type": "string"},
                },
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
        yield ToolSpec(
            id=SKILL_RUN_SCRIPT_TOOL_ID,
            description=(
                "Run a bundled executable script from a skill's directory (Level 3) and "
                "get back only its stdout/stderr/exit code — the script's source never "
                "enters context. Provide the skill 'name', the script 'path' relative to "
                "the skill directory, and optional 'args' (passed to the script verbatim, "
                "never through a shell). The interpreter is chosen by file extension "
                "(.py/.sh/.js/.rb/.ps1). The script runs in the workspace under the same "
                "approval and permission rules as a shell command."
            ),
            input_schema=_object_schema(
                {
                    "name": {"type": "string", "enum": names},
                    "path": {"type": "string"},
                    "args": {"type": "array", "items": {"type": "string"}, "default": []},
                },
                required=["name", "path"],
            ),
            capability="skill",
            side_effect="shell",  # executes code → gated like shell.exec (mode + approval)
            handler=self._make_run_script_handler(),
        )

    def tool_bindings(self) -> tuple[ToolBinding, ...]:
        """Bindings for the skill tools, to merge into the runtime config so the run can
        see them (provider tools are not auto-bound)."""
        return tuple(
            ToolBinding(binding_id=spec.id, ref=RegistryToolRef(tool_id=spec.id), authorization="allow")
            for spec in self.get_tools()
        )

    def subagent_definitions(self) -> dict[str, SubagentDefinition]:
        """SubagentDefinitions for fork skills (``context: fork``), keyed by a namespaced id.
        Merge these into ``AgentLoop(subagent_definitions=...)`` so the ``skill`` tool can
        spawn them. A fork skill runs as a FRESH subagent whose persona is the skill's
        instructions and whose tool allowlist is the skill's ``allowed_tools`` — resolved
        against the parent's bindings, so it is a hard ceiling (the subagent can never exceed
        the parent). Empty ``allowed_tools`` inherits all of the parent's tools."""
        out: dict[str, SubagentDefinition] = {}
        for name, definition in self._definitions.items():
            if definition.context != "fork":
                continue
            prompt = (
                PromptSpec(persona_segments=(definition.instructions,))
                if definition.instructions
                else PromptSpec()
            )
            out[SKILL_SUBAGENT_PREFIX + name] = SubagentDefinition(
                description=definition.description,
                prompt=prompt,
                tools=definition.allowed_tools or None,
                context="fresh",
            )
        return out

    def catalog(self) -> list[dict[str, str]]:
        """Plain ``[{name, description}]`` of available skills (the L1 catalog as data, for a UI
        list) — no instructions/resources, which stay behind progressive disclosure."""
        return [
            {"name": name, "description": self._definitions[name].description.strip()}
            for name in sorted(self._definitions)
        ]

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
            if definition.context == "fork":
                return _activate_fork(_context, definition, str(args.get("task") or ""))
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

    def _make_run_script_handler(self):
        def handler(context: Any, args: dict[str, Any]) -> ToolResult:
            name = str(args.get("name") or "")
            rel = str(args.get("path") or "")
            script_args = [str(a) for a in (args.get("args") or ())]
            definition = self._definitions.get(name)
            if definition is None:
                return ToolResult(ok=False, error=f"unknown skill: {name}", error_code="skill_unknown")
            if definition.directory is None:
                return ToolResult(
                    ok=False, error=f"skill has no bundled resources: {name}", error_code="skill_no_resources"
                )
            try:
                script = _resolve_resource(definition.directory, rel)
            except _ResourceError as exc:
                return ToolResult(ok=False, error=str(exc), error_code=exc.code)
            interpreter = _INTERPRETERS.get(script.suffix.lower())
            if interpreter is None:
                return ToolResult(
                    ok=False,
                    error=f"no interpreter for script extension '{script.suffix}': {rel}",
                    error_code="skill_script_unsupported",
                )
            # argv is executed directly (no shell); ``command`` is only a readable label for
            # the approval preview and scope check. The script source is never read.
            argv = [*interpreter, str(script), *script_args]
            label = " ".join([Path(interpreter[0]).name, rel, *script_args]).strip()
            run = getattr(context, "run_script", None)
            if not callable(run):
                return ToolResult(
                    ok=False,
                    error="this context cannot run scripts",
                    error_code="skill_run_unsupported",
                )
            result = run({"command": label, "argv": argv, "cwd": "."})
            if result.get("timed_out"):
                return ToolResult(
                    ok=False, content=result, error="script timed out", error_code="skill_script_timeout"
                )
            if result.get("output_truncated"):
                return ToolResult(
                    ok=False,
                    content=result,
                    error="script exceeded output limit",
                    error_code="skill_script_output_limit",
                )
            return ToolResult(ok=True, content=result)

        return handler


def _activate_fork(context: Any, definition: SkillDefinition, task: str) -> ToolResult:
    """Run a fork skill as a subagent and return its final message. Delegates to the tool
    context's ``spawn_subagent`` (the engine registered the skill's SubagentDefinition under
    the namespaced id); the subagent's persona is the skill instructions and ``task`` is its
    first user message."""
    spawn = getattr(context, "spawn_subagent", None)
    if not callable(spawn):
        return ToolResult(
            ok=False, error="this context cannot run fork skills", error_code="skill_fork_unsupported"
        )
    prompt = task.strip() or definition.description or "Follow your instructions for this task."
    result = spawn({"subagent_type": SKILL_SUBAGENT_PREFIX + definition.name, "prompt": prompt})
    failed = str(result.get("status") or "") == "failed"
    return ToolResult(
        ok=not failed,
        content=result,
        error="" if not failed else str(result.get("error") or "skill subagent failed"),
        error_code="" if not failed else "skill_fork_failed",
    )


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


def _resolve_resource(directory: Path, rel: str) -> Path:
    """Resolve ``rel`` against the skill directory and return the absolute path, guarding
    against traversal (the resolved path must stay within the skill dir) and refusing the
    skill's own ``SKILL.md`` (it is the L2 payload, not a resource). Raises ``_ResourceError``."""
    root = directory.resolve()
    candidate = (root / rel).resolve()
    if root != candidate and root not in candidate.parents:
        raise _ResourceError("path escapes the skill directory", "skill_path_invalid")
    if candidate.name == SKILL_FILENAME and candidate.parent == root:
        raise _ResourceError("SKILL.md is loaded via the skill tool, not as a resource", "skill_path_invalid")
    if not candidate.is_file():
        raise _ResourceError(f"resource not found: {rel}", "skill_resource_missing")
    return candidate


def _read_resource(directory: Path, rel: str) -> str:
    """Read a bundled resource as utf-8 text (traversal-guarded, non-text rejected)."""
    candidate = _resolve_resource(directory, rel)
    data = candidate.read_bytes()
    if len(data) > _MAX_RESOURCE_BYTES:
        raise _ResourceError(f"resource too large: {rel}", "skill_resource_too_large")
    if b"\x00" in data:
        raise _ResourceError(f"binary resource cannot be read as text: {rel}", "skill_resource_binary")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _ResourceError(f"resource is not utf-8 text: {rel}", "skill_resource_binary") from exc
