"""Reference-full profile metadata."""

from __future__ import annotations

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="reference-full",
    title="Reference Full",
    summary="Bundled Reference services and Studio smoke path across all Phase 1S rules.",
    rule_ids=(
        "PH1S-R1",
        "PH1S-R2",
        "PH1S-R3",
        "PH1S-R4",
        "PH1S-R5",
        "PH1S-R6",
        "PH1S-R7",
        "PH1S-R8",
        "PH1S-R9",
    ),
    harnesses=("backend", "capability", "gateway"),
)
