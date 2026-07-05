"""Multi-agent profile metadata."""

from __future__ import annotations

from typing import Any

from monoid_agent_kernel.conformance.harness import CapabilityHarness, MultiAgentBackendHarness

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="multi-agent",
    title="Multi Agent",
    summary="Subagent runtime with identity, capability isolation, shared revocation, and trace linkage.",
    rule_ids=("OR-04-REVOCATION-SCOPE", "OR-09-SUBAGENT-BOUNDARY"),
    harnesses=("multi-agent-backend", "capability"),
)


def assert_multi_agent_backend_boundary_profile(harness: MultiAgentBackendHarness) -> None:
    """Run the backend subagent identity and diagnostics smoke matrix."""
    result = harness.run_subagent_boundary_case()
    assert result["parent_stream_isolated"] is True
    assert result["child_identity_valid"] is True
    assert result["descendant_events_readable"] is True
    assert result["non_descendant_rejected"] is True
    assert result["path_traversal_rejected"] is True
    assert result["trace_linked"] is True
    assert result["task_result_linked"] is True
    assert result["diagnostics_summary_present"] is True
    assert result["usage_rollup_matches"] is True


def assert_multi_agent_backend_capability_boundary_profile(harness: MultiAgentBackendHarness) -> None:
    """Run the child capability-boundary smoke matrix."""
    result = harness.run_subagent_capability_boundary_case()
    assert result["revoked_event_observed"] is True
    assert result["child_result_observed_revoked"] is True
    assert result["gated_handler_not_called"] is True


def assert_multi_agent_shared_revocation_profile(harness: CapabilityHarness) -> None:
    """Run the child-vault revocation sharing smoke matrix."""
    _admit_profile_lease(
        harness,
        capability="web.search",
        lease_id="lease_parent_live",
        token_ref="token:parent-live",
        durable=False,
    )
    _admit_profile_lease(
        harness,
        capability="web.fetch",
        lease_id="lease_parent_durable",
        token_ref="token:parent-durable",
        durable=True,
    )

    child = harness.fork_child()
    assert child.token_for("web.search", now=200.0) is None
    assert child.token_for("web.fetch", now=200.0) == "token:parent-durable"

    _admit_profile_lease(
        child,
        capability="web.search",
        lease_id="lease_child_live",
        token_ref="token:child-live",
        durable=False,
    )
    assert harness.token_for("web.search", now=200.0) == "token:parent-live"
    assert child.token_for("web.search", now=200.0) == "token:child-live"

    harness.revoke_capability({"capability": "web.search"})
    assert harness.token_for("web.search", now=200.0) is None
    assert child.token_for("web.search", now=200.0) is None

    _admit_profile_lease(
        child,
        capability="web.context",
        lease_id="lease_child_only",
        token_ref="token:child-only",
        durable=False,
    )
    assert child.token_for("web.context", now=200.0) == "token:child-only"
    assert harness.token_for("web.context", now=200.0) is None

    child.revoke_capability({"capability": "*"})
    assert harness.token_for("web.fetch", now=200.0) is None
    assert child.token_for("web.context", now=200.0) is None


def _admit_profile_lease(
    harness: CapabilityHarness,
    *,
    capability: str,
    lease_id: str,
    token_ref: str,
    durable: bool,
) -> dict[str, Any]:
    request = harness.request_capability({"capability": capability, "scope": {}})
    return harness.grant_capability(
        str(request["request_id"]),
        {
            "lease_id": lease_id,
            "capability": capability,
            "token_ref": token_ref,
            "expires_at": 4_102_444_800.0,
            "issued_at": 100.0,
            "durable": durable,
            "scope": {},
        },
    )
