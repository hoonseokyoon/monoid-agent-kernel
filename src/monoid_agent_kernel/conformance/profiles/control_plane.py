"""Control-plane profile metadata."""

from __future__ import annotations

from monoid_agent_kernel.conformance.harness import ControlPlaneHarness

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="control-plane",
    title="Control Plane",
    summary="Backend with external control commands, audit events, lifecycle policy, and stable results.",
    rule_ids=(
        "OR-03-LEASE-ADMISSION",
        "OR-05-EVENT-SEQUENCING",
        "OR-06-CONTROL-AUDIT",
        "OR-07-DURABLE-METADATA",
    ),
    harnesses=("control-plane",),
)


def assert_control_plane_decision_profile(harness: ControlPlaneHarness) -> None:
    """Run the approve/deny control decision smoke matrix."""
    result = harness.run_control_decision_case()
    assert result["approve_delivered"] is True
    assert result["deny_delivered"] is True
    assert result["denied_result_sanitized"] is True
    assert result["approve_audit_recorded"] is True
    assert result["deny_audit_recorded"] is True


def assert_control_plane_audit_sequence_profile(harness: ControlPlaneHarness) -> None:
    """Run the control audit sequencing smoke matrix."""
    result = harness.run_control_audit_sequence_case()
    assert result["authorized_completed"] is True
    assert result["failed_audit_recorded"] is True
    assert result["unauthorized_excluded"] is True
    assert result["terminal_audit_appended"] is True
    assert result["sequence_monotonic"] is True
    assert result["secrets_redacted"] is True
