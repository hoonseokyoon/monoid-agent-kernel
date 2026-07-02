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


def _assert_raises(operation: Callable[[], Any], message: str) -> None:
    try:
        operation()
    except Exception as exc:
        if message not in str(exc):
            raise AssertionError(f"expected error containing {message!r}, got {exc!r}") from exc
        return
    raise AssertionError(f"expected error containing {message!r}")
