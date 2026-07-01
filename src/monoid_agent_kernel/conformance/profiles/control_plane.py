"""Control-plane profile metadata."""

from __future__ import annotations

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="control-plane",
    title="Control Plane",
    summary="Backend with external control commands, audit events, lifecycle policy, and stable results.",
    rule_ids=(
        "OR-03-LEASE-ADMISSION",
        "OR-05-EVENT-SEQUENCING",
        "OR-06-CONTROL-AUDIT",
        "OR-07-DURABLE-METADATA",
    ),
    harnesses=("backend", "capability"),
)
