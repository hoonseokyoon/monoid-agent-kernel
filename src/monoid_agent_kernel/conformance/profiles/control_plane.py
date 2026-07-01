"""Control-plane profile metadata."""

from __future__ import annotations

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="control-plane",
    title="Control Plane",
    summary="Backend with external control commands, audit events, lifecycle policy, and stable results.",
    rule_ids=("PH1S-R3", "PH1S-R5", "PH1S-R6", "PH1S-R7"),
    harnesses=("backend", "capability"),
)
