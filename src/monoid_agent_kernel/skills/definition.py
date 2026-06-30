"""The data model for a single Agent Skill (Anthropic ``SKILL.md`` model).

A skill is *procedural knowledge* delivered by **progressive disclosure**, so the three
levels of disclosure map directly onto the fields here (see :class:`SkillDefinition`).
This module is pure data + parsing; the engine attachment lives in ``skills/provider.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

SKILL_FILENAME = "SKILL.md"

# How a skill's L2 payload is delivered when activated:
# - "inline": the SKILL.md body is returned by the ``skill`` tool into the conversation.
# - "fork":   the body runs as an isolated *subagent* (reusing the subagent machine); only
#             its final message returns. Heavy skills keep their working noise out of context.
SkillContext = Literal["inline", "fork"]


@dataclass(frozen=True)
class SkillDefinition:
    """One Agent Skill, loaded by progressive disclosure.

    The three disclosure levels map onto this data:

    * **L1 (always resident, ~100 tokens)** — ``name`` + ``description`` are advertised in
      the system prompt by :meth:`SkillProvider.static_segment` so the model knows the
      skill exists and when to reach for it.
    * **L2 (on trigger)** — ``instructions`` (the SKILL.md body) is returned by the
      ``skill`` tool when the model activates the skill, entering the conversation.
    * **L3 (on demand)** — bundled resource files under ``directory`` are read one at a
      time via ``skill.read_file``; their content stays out of context until read.

    ``allowed_tools`` is **advisory** (Claude parity): it is surfaced to the model as a
    hint about which tools the skill expects, but it does not restrict the tool registry.
    """

    name: str
    description: str = ""
    instructions: str = ""
    allowed_tools: tuple[str, ...] = ()
    context: SkillContext = "inline"
    directory: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_frontmatter(
        cls, meta: Mapping[str, Any], body: str, *, directory: Path | None = None
    ) -> SkillDefinition:
        """Build a definition from a parsed ``SKILL.md``: the YAML frontmatter ``meta``
        plus the markdown ``body`` (which becomes the L2 ``instructions``). Mirrors the
        SKILL.md spec field names. ``name`` falls back to the skill directory name.

        ``allowed-tools`` accepts the spec's space-separated string form
        (``allowed-tools: Read Glob``) or an inline list; either way it is advisory.
        """
        name = str(meta.get("name") or (directory.name if directory is not None else "")).strip()

        allowed_raw = meta.get("allowed-tools", meta.get("allowed_tools", ()))
        if isinstance(allowed_raw, str):
            allowed = tuple(token for token in allowed_raw.split() if token)
        else:
            allowed = tuple(str(item) for item in allowed_raw or ())

        metadata_raw = meta.get("metadata")
        metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}

        context: SkillContext = "fork" if str(meta.get("context") or "inline") == "fork" else "inline"

        return cls(
            name=name,
            description=str(meta.get("description") or ""),
            instructions=body.strip(),
            allowed_tools=allowed,
            context=context,
            directory=directory,
            metadata=metadata,
        )
