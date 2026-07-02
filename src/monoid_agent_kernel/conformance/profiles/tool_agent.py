"""Tool-using agent profile metadata."""

from __future__ import annotations

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="tool-agent",
    title="Tool Agent",
    summary="Agent integration that executes tools with bindings, permissions, and output validation.",
    rule_ids=(),
    harnesses=("backend",),
)
