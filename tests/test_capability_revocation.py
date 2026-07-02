from __future__ import annotations

from monoid_agent_kernel.core.capability import CapabilityLease, CapabilityRequest, CapabilityVault
from monoid_agent_kernel.core.capability_revocation import (
    CapabilityRevocationState,
    apply_capability_revocation,
    export_revocation_state,
    import_revocation_state,
    is_capability_revoked,
    is_lease_revoked,
)


def test_apply_capability_revocation_blocks_capability() -> None:
    state = CapabilityRevocationState()
    lease = CapabilityLease(capability="web.search", token_ref="t", expires_at=9e9)

    summary = apply_capability_revocation(state, capability="web.search")

    assert summary["capabilities"] == ["web.search"]
    assert is_capability_revoked(state, "web.search")
    assert is_lease_revoked(state, lease)


def test_apply_capability_revocation_blocks_one_lease_id() -> None:
    state = CapabilityRevocationState()
    revoked = CapabilityLease(capability="cap.a", token_ref="a", expires_at=9e9, lease_id="lease_a")
    live = CapabilityLease(capability="cap.b", token_ref="b", expires_at=9e9, lease_id="lease_b")

    apply_capability_revocation(state, lease_id="lease_a")

    assert is_lease_revoked(state, revoked)
    assert not is_lease_revoked(state, live)
    assert not is_capability_revoked(state, "cap.a")


def test_apply_capability_revocation_uses_monotonic_watermark() -> None:
    state = CapabilityRevocationState()
    old = CapabilityLease(capability="cap.old", token_ref="old", expires_at=9e9, issued_at=100.0)
    new = CapabilityLease(capability="cap.new", token_ref="new", expires_at=9e9, issued_at=200.0)

    apply_capability_revocation(state, before=150.0)
    apply_capability_revocation(state, before=50.0)

    assert state.before == 150.0
    assert is_lease_revoked(state, old)
    assert not is_lease_revoked(state, new)


def test_apply_capability_revocation_wildcard_blocks_all() -> None:
    state = CapabilityRevocationState()
    lease = CapabilityLease(capability="cap.any", token_ref="t", expires_at=9e9)

    summary = apply_capability_revocation(state, capability="*")

    assert summary["capabilities"] == ["*"]
    assert is_capability_revoked(state, "cap.any")
    assert is_lease_revoked(state, lease)


def test_export_import_revocation_state_roundtrip() -> None:
    state = CapabilityRevocationState()
    apply_capability_revocation(state, capability="web.search", lease_id="lease_x", before=42.0)
    exported = export_revocation_state(state)

    fresh = CapabilityRevocationState()
    import_revocation_state(
        fresh,
        lease_ids=exported["revoked_lease_ids"],
        capabilities=exported["revoked_capabilities"],
        before=exported["revoked_before"],
        all_revoked=exported["revoked_all"],
    )

    assert export_revocation_state(fresh) == exported


def test_vault_child_split_live_leases_and_shared_revocations() -> None:
    parent = CapabilityVault()
    parent.admit(
        CapabilityRequest(capability="web.search"),
        CapabilityLease(capability="web.search", token_ref="parent-live", expires_at=9e9),
    )
    parent.admit(
        CapabilityRequest(capability="web.fetch"),
        CapabilityLease(capability="web.fetch", token_ref="parent-durable", expires_at=9e9, durable=True),
    )

    child = parent.fork_for_child()

    assert child.token_for("web.search", now=0.0) is None
    assert child.token_for("web.fetch", now=0.0) == "parent-durable"

    child.admit(
        CapabilityRequest(capability="web.search"),
        CapabilityLease(capability="web.search", token_ref="child-live", expires_at=9e9),
    )
    parent.revoke(capability="web.search")

    assert parent.token_for("web.search", now=0.0) is None
    assert child.token_for("web.search", now=0.0) is None
