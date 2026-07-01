"""Provider-gateway profile metadata."""

from __future__ import annotations

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="provider-gateway",
    title="Provider Gateway",
    summary="Gateway runtime with signed scopes, domain boundaries, redirect checks, and effective caps.",
    rule_ids=("OR-01-SCOPE-RELATION", "OR-02-CAPABILITY-BOUNDARY", "OR-08-PROVIDER-CAPS"),
    harnesses=("gateway",),
)
