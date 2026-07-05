"""Tool-using agent profile metadata and assertions."""

from __future__ import annotations

from monoid_agent_kernel.conformance.harness import ToolAgentHarness

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="tool-agent",
    title="Tool Agent",
    summary="Agent integration that executes tools with bindings, permissions, and output validation.",
    rule_ids=(
        "OR-10-TOOL-SURFACE-ADMISSION",
        "OR-11-GENERIC-ASK-APPROVAL",
    ),
    harnesses=("tool-agent",),
)


def assert_tool_agent_surface_admission_profile(harness: ToolAgentHarness) -> None:
    """Run the Phase 2 tool surface admission smoke matrix."""
    result = harness.run_tool_surface_admission_case()
    assert result["permission_denied"] is True
    assert result["denial_code"] == "tool_quota_exceeded"
    assert result["run_completed"] is True


def assert_tool_agent_generic_ask_approval_profile(harness: ToolAgentHarness) -> None:
    """Run the Phase 2 generic authorization='ask' approval smoke matrix."""
    result = harness.run_generic_ask_approval_case()
    assert result["approved_requested"] is True
    assert result["approved_recorded"] is True
    assert result["approved_replayed_once"] is True
    assert result["denied_requested"] is True
    assert result["denied_recorded"] is True
    assert result["denied_replayed"] is False
    assert result["stale_approved_recorded"] is True
    assert result["stale_replay_rejected"] is True
