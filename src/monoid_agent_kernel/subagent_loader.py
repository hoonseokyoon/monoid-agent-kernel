"""Load subagent definitions from a directory of ``.claude/agents``-style markdown
files (YAML frontmatter + body). Parallel to ``tool_loader`` — an external source the
CLI scans and injects into ``AgentLoop(subagent_definitions=...)``.

Each ``*.md`` file is one subagent. Its id is the frontmatter ``name`` (falling back to
the file stem). Files are scanned recursively; on a duplicate id the first one wins and
later ones are skipped (deterministic by sorted path) — the skip is logged at WARNING so a
silent first-wins collision is debuggable. The frontmatter format is the
``SubagentDefinition.from_frontmatter`` contract; no PyYAML dependency.
"""

from __future__ import annotations

import logging
from pathlib import Path

from monoid_agent_kernel.core.agents import SubagentDefinition
from monoid_agent_kernel.core.frontmatter import parse_frontmatter

__all__ = ["load_subagent_definitions"]

_log = logging.getLogger(__name__)


def load_subagent_definitions(directory: Path) -> dict[str, SubagentDefinition]:
    """Scan ``directory`` recursively for ``*.md`` subagent files and return an id ->
    definition map. Raises ``ValueError`` if the directory is missing or a file fails to
    parse (with the offending path), so a misconfigured agents dir fails loudly. A duplicate
    id keeps the first file (sorted by path) and logs a WARNING naming the dropped file."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"agents directory not found: {directory}")
    definitions: dict[str, SubagentDefinition] = {}
    sources: dict[str, Path] = {}
    for path in sorted(root.rglob("*.md")):
        if not path.is_file():
            continue
        try:
            meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
            definition = SubagentDefinition.from_frontmatter(meta, body)
        except Exception as exc:  # noqa: BLE001 - re-raise with the offending file
            raise ValueError(f"failed to load subagent {path}: {exc}") from exc
        sub_id = str(meta.get("name") or path.stem).strip()
        if not sub_id:
            raise ValueError(f"subagent file has no name: {path}")
        if sub_id in definitions:
            _log.warning(
                "duplicate subagent id %r: keeping %s, skipping %s",
                sub_id,
                sources[sub_id],
                path,
            )
            continue
        definitions[sub_id] = definition
        sources[sub_id] = path
    return definitions
