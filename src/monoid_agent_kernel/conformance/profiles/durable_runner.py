"""Durable runner profile metadata."""

from __future__ import annotations

from monoid_agent_kernel.conformance.harness import DurableRunnerHarness

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="durable-runner",
    title="Durable Runner",
    summary="Backend that preserves run state, event sequence, diagnostics, and recovery metadata.",
    rule_ids=("OR-05-EVENT-SEQUENCING", "OR-07-DURABLE-METADATA", "OR-09-SUBAGENT-BOUNDARY"),
    harnesses=("durable-runner",),
)


def assert_durable_runner_event_sequence_profile(harness: DurableRunnerHarness) -> None:
    """Run the event sequence and diagnostics smoke matrix."""
    result = harness.run_event_sequence_case()
    assert result["sequence_monotonic"] is True
    assert result["control_completed"] is True
    assert result["diagnostics_latest_seq_projected"] is True


def assert_durable_runner_recovery_metadata_profile(harness: DurableRunnerHarness) -> None:
    """Run the recovery metadata smoke matrix."""
    result = harness.run_recovery_metadata_case()
    assert result["same_restart_resumed"] is True
    assert result["same_runtime_config_recovered"] is True
    assert result["empty_restart_resumed"] is True
    assert result["empty_runtime_config_recovered"] is True


def assert_durable_runner_subagent_diagnostics_profile(harness: DurableRunnerHarness) -> None:
    """Run the subagent diagnostics projection smoke matrix."""
    result = harness.run_subagent_diagnostics_case()
    assert result["diagnostics_summary_present"] is True
    assert result["identity_projected"] is True
    assert result["trace_linked"] is True
    assert result["status_completed"] is True
    assert result["usage_rollup_matches"] is True
