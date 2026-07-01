"""Provider-gateway profile metadata."""

from __future__ import annotations

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="provider-gateway",
    title="Provider Gateway",
    summary="Gateway runtime with signed scopes, domain boundaries, redirect checks, and effective caps.",
    rule_ids=("PH1S-R1", "PH1S-R2", "PH1S-R8"),
    harnesses=("gateway",),
)
