"""Reference-full profile metadata."""

from __future__ import annotations

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="reference-full",
    title="Reference Full",
    summary="Bundled Reference services and Studio smoke path across all Phase 1S rules.",
    rule_ids=(
        "OR-01-SCOPE-RELATION",
        "OR-02-CAPABILITY-BOUNDARY",
        "OR-03-LEASE-ADMISSION",
        "OR-04-REVOCATION-SCOPE",
        "OR-05-EVENT-SEQUENCING",
        "OR-06-CONTROL-AUDIT",
        "OR-07-DURABLE-METADATA",
        "OR-08-PROVIDER-CAPS",
        "OR-09-SUBAGENT-BOUNDARY",
    ),
    harnesses=("backend", "capability", "gateway"),
)
