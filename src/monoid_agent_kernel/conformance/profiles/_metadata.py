"""Shared conformance profile metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileMetadata:
    """Static description of one conformance profile."""

    profile_id: str
    title: str
    summary: str
    rule_ids: tuple[str, ...]
    harnesses: tuple[str, ...]
