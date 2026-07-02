"""External side-effect tool profile metadata and assertions."""

from __future__ import annotations

from monoid_agent_kernel.conformance.harness import SideEffectHarness

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="side-effect-tool-agent",
    title="Side-Effect Tool Agent",
    summary="Agent integration that runs external side-effect tools with durable outbox or idempotency admission.",
    rule_ids=("OR-12-DURABLE-SIDE-EFFECT",),
    harnesses=("side-effect",),
)


def assert_side_effect_tool_agent_profile(harness: SideEffectHarness) -> None:
    """Run the Phase 2 durable side-effect/outbox smoke matrix."""

    dispatched = harness.run_outbox_dispatched_case()
    assert dispatched["requested"] is True
    assert dispatched["dispatched"] is True

    pending = harness.run_pending_recovery_case()
    assert pending["initial_status"] == "pending"
    assert pending["recovered_status"] == "pending"
    assert pending["request_id"] == pending["recovered_request_id"]

    rejected = harness.run_strict_rejected_case()
    assert rejected["denied"] is True
    assert rejected["handler_finished"] is False

    idempotent = harness.run_idempotent_inline_case()
    assert idempotent["missing_denied"] is True
    assert idempotent["valid_finished"] is True
