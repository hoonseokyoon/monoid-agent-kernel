"""Durable runner profile metadata."""

from __future__ import annotations

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="durable-runner",
    title="Durable Runner",
    summary="Backend that preserves run state, event sequence, diagnostics, and recovery metadata.",
    rule_ids=("PH1S-R5", "PH1S-R7", "PH1S-R9"),
    harnesses=("backend",),
)
