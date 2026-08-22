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


__all__ = ["InterruptionCause"]
