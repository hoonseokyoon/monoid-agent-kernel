"""Capability-security profile metadata."""

from __future__ import annotations

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="capability-security",
    title="Capability Security",
    summary="Capability-gated runtime with scope narrowing, lease admission, denial, and revocation rules.",
    rule_ids=(
        "OR-01-SCOPE-RELATION",
        "OR-02-CAPABILITY-BOUNDARY",
        "OR-03-LEASE-ADMISSION",
        "OR-04-REVOCATION-SCOPE",
        "OR-06-CONTROL-AUDIT",
        "OR-09-SUBAGENT-BOUNDARY",
    ),
    harnesses=("backend", "capability", "gateway"),
)
