"""Load Agent Skills from a directory of ``SKILL.md`` files (YAML frontmatter + body).

Parallel to ``subagent_loader`` — an external source the CLI scans and injects into a
:class:`~native_agent_runner.skills.provider.SkillProvider`. Each skill is a directory
containing a ``SKILL.md`` (the Anthropic ``<skills>/<skill-name>/SKILL.md`` convention);
that directory is the skill's bundle root for Level-3 resources. No PyYAML dependency.
"""

from __future__ import annotations

import logging
from pathlib import Path

from native_agent_runner.core.frontmatter import parse_frontmatter
from native_agent_runner.skills.definition import SKILL_FILENAME, SkillDefinition

__all__ = ["load_skill_definitions"]

_log = logging.getLogger(__name__)


def load_skill_definitions(directory: Path) -> dict[str, SkillDefinition]:
    """Scan ``directory`` recursively for ``SKILL.md`` files and return a name ->
    definition map. The skill's bundle root is the SKILL.md's parent directory.

    Raises ``ValueError`` if the directory is missing or a file fails to parse (with the
    offending path), so a misconfigured skills dir fails loudly. A duplicate name keeps the
    first file (sorted by path) and logs a WARNING naming the dropped file."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"skills directory not found: {directory}")
    definitions: dict[str, SkillDefinition] = {}
    sources: dict[str, Path] = {}
    for path in sorted(root.rglob(SKILL_FILENAME)):
        if not path.is_file():
            continue
        skill_dir = path.parent
        try:
            meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
            definition = SkillDefinition.from_frontmatter(meta, body, directory=skill_dir)
        except Exception as exc:  # noqa: BLE001 - re-raise with the offending file
            raise ValueError(f"failed to load skill {path}: {exc}") from exc
        if not definition.name:
            raise ValueError(f"skill file has no name: {path}")
        if definition.name in definitions:
            _log.warning(
                "duplicate skill name %r: keeping %s, skipping %s",
                definition.name,
                sources[definition.name],
                path,
            )
            continue
        definitions[definition.name] = definition
        sources[definition.name] = path
    return definitions
