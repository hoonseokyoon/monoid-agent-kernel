"""Tool-using agent profile metadata and assertions."""

from __future__ import annotations

from monoid_agent_kernel.conformance.harness import BackendHarness

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="tool-agent",
    title="Tool Agent",
    summary="Agent integration that executes tools with bindings, permissions, and output validation.",
    rule_ids=("OR-10-TOOL-SURFACE-ADMISSION", "OR-11-GENERIC-ASK-APPROVAL"),
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
