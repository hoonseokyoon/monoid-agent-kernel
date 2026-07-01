"""Capability-security profile metadata."""

from __future__ import annotations

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="capability-security",
    title="Capability Security",
    summary="Capability-gated runtime with scope narrowing, lease admission, denial, and revocation rules.",
    rule_ids=("PH1S-R1", "PH1S-R2", "PH1S-R3", "PH1S-R4", "PH1S-R6", "PH1S-R9"),
    harnesses=("backend", "capability", "gateway"),
)
