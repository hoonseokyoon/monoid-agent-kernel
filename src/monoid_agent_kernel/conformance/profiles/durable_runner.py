"""Durable runner profile metadata."""

from __future__ import annotations

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="durable-runner",
    title="Durable Runner",
    summary="Backend that preserves run state, event sequence, diagnostics, and recovery metadata.",
    rule_ids=("OR-05-EVENT-SEQUENCING", "OR-07-DURABLE-METADATA", "OR-09-SUBAGENT-BOUNDARY"),
    harnesses=("backend",),
)
