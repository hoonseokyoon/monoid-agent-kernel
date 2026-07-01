"""Multi-agent profile metadata."""

from __future__ import annotations

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="multi-agent",
    title="Multi Agent",
    summary="Subagent runtime with identity, capability isolation, shared revocation, and trace linkage.",
    rule_ids=("OR-04-REVOCATION-SCOPE", "OR-09-SUBAGENT-BOUNDARY"),
    harnesses=("backend", "capability"),
)
