"""Phase 1S conformance profile metadata."""

from __future__ import annotations

from ._metadata import ProfileMetadata
from .capability_security import PROFILE as CAPABILITY_SECURITY
from .control_plane import PROFILE as CONTROL_PLANE
from .durable_runner import PROFILE as DURABLE_RUNNER
from .minimal_agent import PROFILE as MINIMAL_AGENT
from .multi_agent import PROFILE as MULTI_AGENT
from .provider_gateway import PROFILE as PROVIDER_GATEWAY
from .reference_full import PROFILE as REFERENCE_FULL
from .tool_agent import PROFILE as TOOL_AGENT

PROFILES: tuple[ProfileMetadata, ...] = (
    MINIMAL_AGENT,
    TOOL_AGENT,
    DURABLE_RUNNER,
    CONTROL_PLANE,
    CAPABILITY_SECURITY,
    PROVIDER_GATEWAY,
    MULTI_AGENT,
    REFERENCE_FULL,
)
PROFILE_BY_ID: dict[str, ProfileMetadata] = {profile.profile_id: profile for profile in PROFILES}


def get_profile(profile_id: str) -> ProfileMetadata:
    """Return profile metadata by stable id."""
    try:
        return PROFILE_BY_ID[profile_id]
    except KeyError as exc:
        raise KeyError(f"unknown conformance profile: {profile_id}") from exc


__all__ = [
    "CAPABILITY_SECURITY",
    "CONTROL_PLANE",
    "DURABLE_RUNNER",
    "MINIMAL_AGENT",
    "MULTI_AGENT",
    "PROFILES",
    "PROFILE_BY_ID",
    "PROVIDER_GATEWAY",
    "ProfileMetadata",
    "REFERENCE_FULL",
    "TOOL_AGENT",
    "get_profile",
]
