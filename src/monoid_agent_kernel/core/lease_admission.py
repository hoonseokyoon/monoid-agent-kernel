"""Lease admission helpers for capability approval boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from monoid_agent_kernel.core.scope import scope_within


@dataclass(frozen=True)
class LeaseAdmissionError(ValueError):
    """Raised when a capability lease cannot be admitted for a request."""

    reason: str
    detail: str

    def __str__(self) -> str:
        return self.detail


def validate_lease_admission(
    request_capability: str,
    request_scope: Mapping[str, Any],
    lease_capability: str,
    lease_scope: Mapping[str, Any],
) -> None:
    """Validate that a granted lease is no broader than the original request."""
    requested = str(request_capability)
    granted = str(lease_capability)
    if granted != requested:
        raise LeaseAdmissionError(
            "capability_mismatch",
            f"broker granted capability {granted!r} for request {requested!r}",
        )
    if not scope_within(lease_scope, request_scope):
        raise LeaseAdmissionError(
            "scope_widened",
            f"broker granted a wider scope than requested for {requested!r}",
        )


def sanitize_denied_capability_result(
    result: Mapping[str, Any],
    *,
    reason: str = "",
    answer: str = "Deny",
) -> dict[str, Any]:
    """Return a denial result with grant material removed."""
    denied = dict(result)
    denied["answer"] = answer
    denied["approved"] = False
    denied["granted"] = False
    denied.pop("lease", None)
    denied.pop("token_ref", None)
    denied["reason"] = reason or str(denied.get("reason") or "denied")
    return denied
