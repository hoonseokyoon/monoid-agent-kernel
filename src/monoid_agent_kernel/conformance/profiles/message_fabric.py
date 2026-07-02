"""External agent message-fabric profile metadata and assertions."""

from __future__ import annotations

from monoid_agent_kernel.conformance.harness import MessageFabricHarness
from monoid_agent_kernel.core.external_agent_envelope import EXTERNAL_AGENT_ENVELOPE_VERSION

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

    two_peer = harness.submit_run({"scenario": "message-fabric-two-peer"})
    planner_events = list(harness.events(str(two_peer["run_id"]), str(two_peer["token"]), limit=200)["events"])
    worker_events = list(
        harness.events(str(two_peer["peer_run_id"]), str(two_peer["peer_token"]), limit=200)["events"]
    )
    assert _has_event(planner_events, "outbox.dispatched", destination="worker")
    assert _has_event(worker_events, "outbox.dispatched", destination="planner")
    assert _event_with_trace(planner_events, "outbox.dispatched", destination="worker")
    assert _event_with_trace(worker_events, "outbox.dispatched", destination="planner")

    receiver = harness.submit_run({"scenario": "message-fabric-receiver"})
    malformed = harness.deliver_external_agent_message(
        str(receiver["run_id"]),
        str(receiver["token"]),
        {"protocol": EXTERNAL_AGENT_ENVELOPE_VERSION, "peer_id": "planner"},
    )
    assert malformed["status"] == "rejected"
    assert malformed["error_code"] == "external_agent_envelope_invalid"
    _cancel(harness, receiver)

    duplicate = harness.submit_run({"scenario": "message-fabric-duplicate-restart"})
    assert duplicate["first_status"] == "queued"
    assert duplicate["duplicate_status_after_restart"] == "duplicate"
    assert duplicate["message_id"] in duplicate["seen_inbox_ids_after_restart"]

    unavailable = harness.submit_run({"scenario": "message-fabric-peer-unavailable"})
    state = harness.message_fabric_state(str(unavailable["run_id"]), str(unavailable["token"]))
    pending = [request for request in state["requests"] if request["destination"] == "missing-worker"]
    assert len(pending) == 1
    assert pending[0]["status"] == "pending"
    assert pending[0]["attempts"] >= 1


def _has_event(events: list[dict], event_type: str, **data: str) -> bool:
    for event in events:
        if event.get("type") != event_type:
            continue
        event_data = event.get("data") or {}
        if all(event_data.get(key) == value for key, value in data.items()):
            return True
    return False


def _event_with_trace(events: list[dict], event_type: str, **data: str) -> bool:
    for event in events:
        if event.get("type") != event_type:
            continue
        event_data = event.get("data") or {}
        if not all(event_data.get(key) == value for key, value in data.items()):
            continue
        return bool(event_data.get("traceparent"))
    return False


def _cancel(harness: MessageFabricHarness, submitted: dict) -> None:
    try:
        harness.dispatch(
            {
                "type": "cancel",
                "run_id": str(submitted["run_id"]),
                "args": {"token": str(submitted["token"])},
                "issuer": "message-fabric-profile",
            }
        )
    except Exception:
        pass


__all__ = ["PROFILE", "assert_message_fabric_profile"]
