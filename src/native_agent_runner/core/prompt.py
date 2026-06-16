"""System-prompt composition.

The agent's identity prompt is composed, not hardcoded: a general-purpose
``BASE_SYSTEM_PROMPT`` provided by the core, plus zero or more persona/role
segments supplied per instance (via ``AgentRunSpec`` or a profile). This is how
the runner stays a *general* agent (goal 2) while specialization — a coding
agent, a research agent — is layered in as a module rather than baked into core.

Static context segments contributed by ``ContextProvider``s (see
``core/context.py``) are folded in here at bootstrap as additional persona-level
segments; per-turn dynamic context is appended separately by the loop.
"""

from __future__ import annotations

from collections.abc import Iterable

# Deliberately general: it grounds the agent in a sandboxed workspace and the
# provided tools without narrowing to files/diffs, so non-file work (shell, web,
# artifacts) and persona specializations read naturally on top of it.
BASE_SYSTEM_PROMPT = """You are a general-purpose agent operating in a sandboxed workspace.
Use only the provided tools to inspect the workspace or act on it; do not invent inputs you have not observed.
Respect tool errors and permissions. Finish by calling run.finish with a concise summary.
"""


def compose_system_prompt(
    base: str = BASE_SYSTEM_PROMPT,
    persona_segments: Iterable[str] = (),
) -> str:
    """Join the base prompt with persona/context segments into one system prompt.

    Empty/blank segments are dropped. With no segments the result is the base
    prompt (trailing newline preserved), so the no-persona path is unchanged.
    """
    parts = [base.strip()]
    parts.extend(segment.strip() for segment in persona_segments if segment and segment.strip())
    return "\n\n".join(parts) + "\n"
