"""Multi-agent profile metadata."""

from __future__ import annotations

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="multi-agent",
    title="Multi Agent",
    summary="Subagent runtime with identity, capability isolation, shared revocation, and trace linkage.",
    rule_ids=("PH1S-R4", "PH1S-R9"),
    harnesses=("backend", "capability"),
)
