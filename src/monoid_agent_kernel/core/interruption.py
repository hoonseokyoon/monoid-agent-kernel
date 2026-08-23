"""Portable causes for run and turn interruption boundaries."""

from __future__ import annotations

from enum import StrEnum


class InterruptionCause(StrEnum):
    USER_CANCEL = "user_cancel"
    GRACEFUL_DRAIN = "graceful_drain"
    LEASE_LOST = "lease_lost"
    DEADLINE = "deadline"
    HOST_SHUTDOWN = "host_shutdown"
    PROVIDER_FAILURE = "provider_failure"
    VALIDATION_FAILURE = "validation_failure"
    UNKNOWN = "unknown"


def parse_interruption_cause(value: object) -> InterruptionCause | None:
    """Parse the portable cause vocabulary at an untrusted projection boundary.

    Event schema v1 deliberately accepts arbitrary strings for rolling-version and retained-log
    compatibility. Every current reader still needs the same closed portable vocabulary; keeping
    that decision here prevents live, offline, and backend projections from drifting.
    """

    if not isinstance(value, str) or not value:
        return None
    try:
        return InterruptionCause(value)
    except ValueError:
        return None


__all__ = ["InterruptionCause", "parse_interruption_cause"]
