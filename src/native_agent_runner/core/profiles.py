"""Named capability/limits presets for configuring an agent's "weight class".

A profile bundles the knobs that together define a lightweight↔heavyweight
instance (write mode, shell/web policies, run limits). Capabilities are NOT set
here — ``AgentRunSpec.effective_capabilities()`` derives them from ``mode`` plus
the shell/web policies, so a profile only needs to choose those.

A profile is a base; any field a caller specifies explicitly overrides it.
``standard`` matches the no-profile defaults, so omitting a profile is identical
to selecting ``standard``.
"""

from __future__ import annotations

from dataclasses import dataclass

from native_agent_runner.core.spec import RunLimits, RunMode
from native_agent_runner.shell import ShellPolicy
from native_agent_runner.web import WebPolicy


@dataclass(frozen=True)
class AgentProfile:
    name: str
    mode: RunMode
    shell_policy: ShellPolicy
    web_policy: WebPolicy
    limits: RunLimits
    # Persona/role segments appended to the base system prompt. The three built-in
    # weight-class profiles leave this empty (they tune capability, not identity);
    # a specialization profile (e.g. a coding agent) carries its persona here.
    persona_segments: tuple[str, ...] = ()


AGENT_PROFILES: dict[str, AgentProfile] = {
    "lightweight": AgentProfile(
        name="lightweight",
        mode="read-only",
        shell_policy=ShellPolicy(),
        web_policy=WebPolicy(),
        limits=RunLimits(max_steps=15, max_tool_calls=40, max_duration_s=300),
    ),
    "standard": AgentProfile(
        name="standard",
        mode="propose",
        shell_policy=ShellPolicy(),
        web_policy=WebPolicy(),
        limits=RunLimits(),
    ),
    "heavyweight": AgentProfile(
        name="heavyweight",
        mode="propose",
        shell_policy=ShellPolicy(enabled=True),
        web_policy=WebPolicy(enabled=True),
        limits=RunLimits(max_steps=60, max_tool_calls=200, max_duration_s=1800),
    ),
}

PROFILE_NAMES = tuple(AGENT_PROFILES)


def resolve_profile(name: str) -> AgentProfile:
    try:
        return AGENT_PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown profile: {name!r}; choose one of {', '.join(PROFILE_NAMES)}"
        ) from None
