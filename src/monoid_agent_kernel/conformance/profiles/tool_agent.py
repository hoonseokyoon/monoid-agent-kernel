"""Tool-using agent profile metadata and assertions."""

from __future__ import annotations

from monoid_agent_kernel.conformance.harness import BackendHarness

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="tool-agent",
    title="Tool Agent",
    summary="Agent integration that executes tools with bindings, permissions, and output validation.",
    rule_ids=(
        "OR-10-TOOL-SURFACE-ADMISSION",
        "OR-11-GENERIC-ASK-APPROVAL",
        "OR-12-DURABLE-SIDE-EFFECT",
    ),
    harnesses=("backend",),
)


def assert_tool_agent_surface_admission_profile(harness: BackendHarness) -> None:
    """Run the Phase 2 tool surface admission smoke matrix."""
    submitted = harness.submit_run({"scenario": "tool-quota-denied"})
    run_id = str(submitted["run_id"])
    token = str(submitted["token"])
    events = list(harness.events(run_id, token)["events"])
    denied = [
        event
        for event in events
        if event["type"] == "permission.denied"
        and event["data"].get("error_code") == "tool_quota_exceeded"
    ]
    assert denied
    result = harness.result(run_id, token)
    assert result["status"] == "completed"


def assert_tool_agent_generic_ask_approval_profile(harness: BackendHarness) -> None:
    """Run the Phase 2 generic authorization='ask' approval smoke matrix."""
    approved = harness.submit_run({"scenario": "tool-ask-approved"})
    approved_events = list(harness.events(str(approved["run_id"]), str(approved["token"]))["events"])
    assert _has_event(approved_events, "tool.approval.requested")
    assert _has_event(approved_events, "tool.approval.approved")
    assert _has_tool_replay_result(harness, approved)
    _cancel(harness, approved)

    denied = harness.submit_run({"scenario": "tool-ask-denied"})
    denied_events = list(harness.events(str(denied["run_id"]), str(denied["token"]))["events"])
    assert _has_event(denied_events, "tool.approval.requested")
    assert _has_event(denied_events, "tool.approval.denied")
    assert not _has_successful_approval_replay(denied_events)
    _cancel(harness, denied)

    stale = harness.submit_run({"scenario": "tool-ask-stale-denied"})
    stale_events = list(harness.events(str(stale["run_id"]), str(stale["token"]))["events"])
    assert _has_event(stale_events, "tool.approval.approved")
    assert any(
        event["data"].get("error_code") in {
            "tool_not_in_surface",
            "tool_binding_denied",
            "tool_unknown",
        }
        and event["type"] in {"permission.denied", "tool.call.failed"}
        for event in stale_events
    )
    _cancel(harness, stale)


def assert_tool_agent_durable_side_effect_profile(harness: BackendHarness) -> None:
    """Run the Phase 2 durable side-effect/outbox smoke matrix."""

    dispatched = harness.submit_run({"scenario": "tool-side-effect-outbox-dispatched"})
    dispatched_effects = harness.side_effects(str(dispatched["run_id"]), str(dispatched["token"]))
    dispatched_requests = list(dispatched_effects["requests"])
    assert len(dispatched_requests) == 1
    assert dispatched_requests[0]["destination"] == "email"
    assert dispatched_requests[0]["status"] == "dispatched"
    assert dispatched_requests[0]["token_ref_present"] is True

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
    assert any(
        event["type"] == "permission.denied"
        and event["data"].get("error_code") == "tool_side_effect_policy_denied"
        for event in rejected_events
    )
    rejected_effects = harness.side_effects(str(rejected["run_id"]), str(rejected["token"]))
    assert rejected_effects["unsafe_handler_calls"] == 0
    assert rejected_effects["requests"] == []

    idempotent = harness.submit_run({"scenario": "tool-side-effect-idempotent-inline"})
    idempotent_events = list(harness.events(str(idempotent["run_id"]), str(idempotent["token"]))["events"])
    assert any(
        event["type"] == "permission.denied"
        and event["data"].get("error_code") == "tool_side_effect_policy_denied"
        for event in idempotent_events
    )
    idempotent_effects = harness.side_effects(str(idempotent["run_id"]), str(idempotent["token"]))
    assert idempotent_effects["idempotent_handler_calls"] == 1
    assert idempotent_effects["idempotency_keys"] == ["idem-1"]


def _has_event(events: list[dict], event_type: str) -> bool:
    return any(event["type"] == event_type for event in events)


def _has_successful_approval_replay(events: list[dict]) -> bool:
    return sum(
        1
        for event in events
        if (
        event["type"] == "tool.call.finished"
        and event["data"].get("tool") == "demo_approval"
        and event["data"].get("ok") is True
        )
    ) >= 2


def _has_tool_replay_result(harness: BackendHarness, submitted: dict) -> bool:
    run_id = str(submitted["run_id"])
    token = str(submitted["token"])
    events = list(harness.events(run_id, token)["events"])
    return _has_successful_approval_replay(events)


def _cancel(harness: BackendHarness, submitted: dict) -> None:
    try:
        harness.dispatch(
            {
                "type": "cancel",
                "run_id": str(submitted["run_id"]),
                "args": {"token": str(submitted["token"])},
                "issuer": "tool-agent-profile",
            }
        )
    except Exception:
        pass
