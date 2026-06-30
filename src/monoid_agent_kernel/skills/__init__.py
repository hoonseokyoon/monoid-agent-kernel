"""Agent Skills (Anthropic ``SKILL.md`` progressive-disclosure model).

Skills are a *knowledge layer*: procedural how-to instructions delivered to the model by
progressive disclosure (catalog → instructions → bundled resources). They attach to the
engine through the existing ``ContextProvider`` and ``ToolProvider`` seams with zero core
changes — complementary to subagents (execution layer) and MCP (integration layer).
"""

from __future__ import annotations

from monoid_agent_kernel.skills.definition import SKILL_FILENAME, SkillDefinition
from monoid_agent_kernel.skills.loader import load_skill_definitions
from monoid_agent_kernel.skills.provider import SkillProvider

__all__ = [
    "SKILL_FILENAME",
    "SkillDefinition",
    "SkillProvider",
    "load_skill_definitions",
]
