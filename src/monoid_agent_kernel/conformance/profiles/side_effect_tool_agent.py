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

    dispatched = harness.submit_run({"scenario": "tool-side-effect-outbox-dispatched"})
    dispatched_events = list(harness.events(str(dispatched["run_id"]), str(dispatched["token"]))["events"])
    assert _has_event(dispatched_events, "outbox.requested", destination="email")
    assert _has_event(dispatched_events, "outbox.dispatched", destination="email")

    pending = harness.submit_run({"scenario": "tool-side-effect-pending-recovery"})
    pending_effects = harness.side_effects(str(pending["run_id"]), str(pending["token"]))
    pending_requests = list(pending_effects["requests"])
    assert len(pending_requests) == 1
    assert pending_requests[0]["status"] == "pending"
    restarted = harness.restart(local_state="same")
    recovered_effects = restarted.side_effects(str(pending["run_id"]), str(pending["token"]))
    recovered_requests = list(recovered_effects["requests"])
    assert len(recovered_requests) == 1
    assert recovered_requests[0]["request_id"] == pending_requests[0]["request_id"]
    assert recovered_requests[0]["status"] == "pending"
    _cancel(harness, pending)

    rejected = harness.submit_run({"scenario": "tool-side-effect-strict-rejected"})
    rejected_events = list(harness.events(str(rejected["run_id"]), str(rejected["token"]))["events"])
    assert _has_event(
        rejected_events,
        "permission.denied",
        call_id="unsafe_1",
        error_code="tool_side_effect_policy_denied",
    )
    assert not _has_event(rejected_events, "tool.call.finished", call_id="unsafe_1")

    idempotent = harness.submit_run({"scenario": "tool-side-effect-idempotent-inline"})
    idempotent_events = list(harness.events(str(idempotent["run_id"]), str(idempotent["token"]))["events"])
    assert _has_event(
        idempotent_events,
        "permission.denied",
        call_id="idempotent_missing",
        error_code="tool_side_effect_policy_denied",
    )
    assert _has_event(
        idempotent_events,
        "tool.call.finished",
        call_id="idempotent_ok",
        tool="demo_external_idempotent",
    )


def _has_event(events: list[dict], event_type: str, **data: str) -> bool:
    for event in events:
        if event.get("type") != event_type:
            continue
        event_data = event.get("data") or {}
        if all(event_data.get(key) == value for key, value in data.items()):
            return True
    return False


def _cancel(harness: SideEffectHarness, submitted: dict) -> None:
    try:
        harness.dispatch(
            {
                "type": "cancel",
                "run_id": str(submitted["run_id"]),
                "args": {"token": str(submitted["token"])},
                "issuer": "side-effect-tool-agent-profile",
            }
        )
    except Exception:
        pass
