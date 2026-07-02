"""Capability-security profile metadata."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from monoid_agent_kernel.conformance.harness import CapabilityHarness

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="capability-security",
    title="Capability Security",
    summary="Capability-gated runtime with scope narrowing, lease admission, denial, and revocation rules.",
    rule_ids=(
        "OR-01-SCOPE-RELATION",
        "OR-02-CAPABILITY-BOUNDARY",
        "OR-03-LEASE-ADMISSION",
        "OR-04-REVOCATION-SCOPE",
        "OR-06-CONTROL-AUDIT",
        "OR-09-SUBAGENT-BOUNDARY",
    ),
    harnesses=("backend", "capability", "gateway"),
)


def assert_capability_security_lease_admission(harness: CapabilityHarness) -> None:
    """Run the Phase 1S lease-admission conformance smoke matrix."""
    request = harness.request_capability(
        {
            "capability": "web.search",
            "scope": {"allowed_domains": ["*.example.test"], "max_results": 5},
        }
    )
    request_id = str(request["request_id"])
    lease = {
        "lease_id": "lease_profile_ok",
        "capability": "web.search",
        "token_ref": "approved:web.search",
        "expires_at": 4_102_444_800.0,
        "issued_at": 1_700_000_000.0,
        "max_expires_at": 4_102_445_000.0,
        "durable": True,
        "scope": {"allowed_domains": ["docs.example.test"], "max_results": 3},
    }

    admitted = harness.grant_capability(request_id, lease)
    assert admitted["lease_id"] == lease["lease_id"]
    assert admitted["capability"] == lease["capability"]
    assert admitted["token_ref"] == lease["token_ref"]
    assert admitted["issued_at"] == lease["issued_at"]
    assert admitted["max_expires_at"] == lease["max_expires_at"]
    assert admitted["durable"] is True
    assert admitted["scope"] == lease["scope"]

    widened = harness.request_capability(
        {"capability": "web.search", "scope": {"allowed_domains": ["docs.example.test"], "max_results": 2}}
    )
    _assert_raises(
        lambda: harness.grant_capability(
            str(widened["request_id"]),
            {
                "capability": "web.search",
                "token_ref": "approved:web.search",
                "expires_at": 4_102_444_800.0,
                "scope": {"allowed_domains": ["*.example.test"], "max_results": 5},
            },
        ),
        "wider scope",
    )

    mismatch = harness.request_capability({"capability": "web.search", "scope": {}})
    _assert_raises(
        lambda: harness.grant_capability(
            str(mismatch["request_id"]),
            {
                "capability": "web.fetch",
                "token_ref": "approved:web.fetch",
                "expires_at": 4_102_444_800.0,
                "scope": {},
            },
        ),
        "web.fetch",
    )

    denial = harness.deny_capability(
        request_id,
        {
            "answer": "Approve",
            "approved": True,
            "granted": True,
            "lease": {"capability": "web.search", "token_ref": "secret-ref://lease"},
            "token_ref": "secret-ref://lease",
        },
    )
    assert denial["answer"] == "Deny"
    assert denial["approved"] is False
    assert denial["granted"] is False
    assert denial["reason"]
    assert "lease" not in denial
    assert "token_ref" not in denial


def assert_capability_security_revocation_profile(harness: CapabilityHarness) -> None:
    """Run the Phase 1S capability revocation conformance smoke matrix."""
    _admit_profile_lease(
        harness,
        capability="web.search",
        lease_id="lease_revoke_cap",
        token_ref="token:web.search",
        issued_at=100.0,
        scope={"allowed_domains": ["docs.example.test"]},
    )
    assert harness.token_for("web.search", now=200.0) == "token:web.search"
    assert harness.valid_lease("web.search", {"allowed_domains": ["docs.example.test"]}, now=200.0)
    harness.revoke_capability({"capability": "web.search"})
    assert harness.token_for("web.search", now=200.0) is None
    assert harness.valid_lease("web.search", {"allowed_domains": ["docs.example.test"]}, now=200.0) is None

    _admit_profile_lease(
        harness,
        capability="web.fetch",
        lease_id="lease_revoke_id",
        token_ref="token:web.fetch",
        issued_at=200.0,
    )
    _admit_profile_lease(
        harness,
        capability="web.context",
        lease_id="lease_revoke_other",
        token_ref="token:web.context",
        issued_at=200.0,
    )
    harness.revoke_capability({"lease_id": "lease_revoke_id"})
    assert harness.token_for("web.fetch", now=300.0) is None
    assert harness.token_for("web.context", now=300.0) == "token:web.context"

    _admit_profile_lease(
        harness,
        capability="cap.old",
        lease_id="lease_old",
        token_ref="token:old",
        issued_at=100.0,
    )
    _admit_profile_lease(
        harness,
        capability="cap.new",
        lease_id="lease_new",
        token_ref="token:new",
        issued_at=200.0,
    )
    harness.revoke_capability({"before": 150.0})
    assert harness.token_for("cap.old", now=300.0) is None
    assert harness.token_for("cap.new", now=300.0) == "token:new"
    harness.revoke_capability({"before": 50.0})
    assert harness.export_revocations()["revoked_before"] == 150.0

    _admit_profile_lease(
        harness,
        capability="cap.persist",
        lease_id="lease_persist",
        token_ref="token:persist",
        issued_at=200.0,
    )
    harness.revoke_capability({"capability": "cap.persist", "lease_id": "lease_extra"})
    exported = harness.export_revocations()
    assert "cap.persist" in exported["revoked_capabilities"]
    assert "lease_extra" in exported["revoked_lease_ids"]
    assert harness.import_revocations(exported) == exported

    _admit_profile_lease(
        harness,
        capability="cap.star",
        lease_id="lease_star",
        token_ref="token:star",
        issued_at=200.0,
    )
    wildcard = harness.revoke_capability({"capability": "*"})
    assert wildcard["capabilities"] == ["*"]
    assert harness.token_for("cap.star", now=300.0) is None


def _admit_profile_lease(
    harness: CapabilityHarness,
    *,
    capability: str,
    lease_id: str,
    token_ref: str,
    issued_at: float,
    scope: dict[str, Any] | None = None,
    durable: bool = False,
) -> dict[str, Any]:
    request = harness.request_capability({"capability": capability, "scope": scope or {}})
    return harness.grant_capability(
        str(request["request_id"]),
        {
            "lease_id": lease_id,
            "capability": capability,
            "token_ref": token_ref,
            "expires_at": 4_102_444_800.0,
            "issued_at": issued_at,
            "durable": durable,
            "scope": scope or {},
        },
    )


def _assert_raises(operation: Callable[[], Any], message: str) -> None:
    try:
        operation()
    except Exception as exc:
        if message not in str(exc):
            raise AssertionError(f"expected error containing {message!r}, got {exc!r}") from exc
        return
    raise AssertionError(f"expected error containing {message!r}")
