"""External agent message-fabric profile metadata and assertions."""

from __future__ import annotations

from monoid_agent_kernel.conformance.harness import MessageFabricHarness

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="message-fabric",
    title="Message Fabric",
    summary="Agent integration that exchanges external-agent messages through durable inbox/outbox envelopes.",
    rule_ids=("OR-13-EXTERNAL-AGENT-ENVELOPE",),
    harnesses=("message-fabric",),
)


def assert_message_fabric_profile(harness: MessageFabricHarness) -> None:
    """Run the external-agent envelope and durable message-fabric smoke matrix."""

    two_peer = harness.run_two_peer_exchange_case()
    assert two_peer["planner_dispatched"] is True
    assert two_peer["worker_replied"] is True
    assert two_peer["planner_trace_preserved"] is True
    assert two_peer["worker_trace_preserved"] is True

    malformed = harness.run_malformed_envelope_case()
    assert malformed["status"] == "rejected"
    assert malformed["error_code"] == "external_agent_envelope_invalid"

    duplicate = harness.run_duplicate_after_restart_case()
    assert duplicate["first_status"] == "queued"
    assert duplicate["duplicate_status_after_restart"] == "duplicate"
    assert duplicate["message_id"] in duplicate["seen_inbox_ids_after_restart"]

    unavailable = harness.run_peer_unavailable_case()
    assert unavailable["pending"] is True
    assert unavailable["status"] == "pending"
    assert unavailable["attempts"] >= 1


__all__ = ["PROFILE", "assert_message_fabric_profile"]
