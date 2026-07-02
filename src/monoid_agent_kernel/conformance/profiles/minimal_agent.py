"""Minimal local agent profile metadata."""

from __future__ import annotations

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="minimal-agent",
    title="Minimal Agent",
    summary="Local loop or chatbot-style integration with basic lifecycle and model adapter behavior.",
    rule_ids=(),
    harnesses=("backend",),
)
