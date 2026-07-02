"""Capability revocation helpers shared by vaults and conformance tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class CapabilityRevocationState:
    lease_ids: set[str] = field(default_factory=set)
    capabilities: set[str] = field(default_factory=set)
    before: float = 0.0
    all_revoked: bool = False


class RevocableLease(Protocol):
    lease_id: str
    capability: str
    issued_at: float


def apply_capability_revocation(
    state: CapabilityRevocationState,
    *,
    capability: str | None = None,
    lease_id: str | None = None,
    before: float | None = None,
) -> dict[str, object]:
    """Record a monotonic revocation and return the public summary."""
    revoked_caps: list[str] = ["*"] if capability == "*" else []
    if capability and capability != "*":
        state.capabilities.add(capability)
        revoked_caps = [capability]
    elif capability == "*":
        state.all_revoked = True
    if lease_id:
        state.lease_ids.add(lease_id)
    if before is not None:
        state.before = max(state.before, before)
    return {
        "capabilities": revoked_caps,
        "lease_id": lease_id or "",
        "revoked_before": state.before,
    }


def is_lease_revoked(state: CapabilityRevocationState, lease: RevocableLease) -> bool:
    """Return true when a concrete lease is covered by revocation state."""
    return (
        state.all_revoked
        or lease.lease_id in state.lease_ids
        or lease.capability in state.capabilities
        or lease.issued_at < state.before
    )


def is_capability_revoked(state: CapabilityRevocationState, capability: str) -> bool:
    """Return true when a capability is under an authoritative re-broker stop."""
    return state.all_revoked or capability in state.capabilities


def export_revocation_state(state: CapabilityRevocationState) -> dict[str, object]:
    """Serialize revocation records for checkpoint/conformance transfer."""
    capabilities = sorted(state.capabilities)
    if state.all_revoked and "*" not in capabilities:
        capabilities = ["*", *capabilities]
    return {
        "revoked_lease_ids": sorted(state.lease_ids),
        "revoked_capabilities": capabilities,
        "revoked_before": state.before,
        "revoked_all": state.all_revoked,
    }


def import_revocation_state(
    state: CapabilityRevocationState,
    *,
    lease_ids: list[str] | None = None,
    capabilities: list[str] | None = None,
    before: float = 0.0,
    all_revoked: bool = False,
) -> None:
    """Merge serialized revocation records into an existing state."""
    state.lease_ids.update(lease_ids or ())
    imported_capabilities = set(capabilities or ())
    if "*" in imported_capabilities:
        all_revoked = True
        imported_capabilities.discard("*")
    state.capabilities.update(imported_capabilities)
    state.before = max(state.before, before)
    state.all_revoked = state.all_revoked or all_revoked
