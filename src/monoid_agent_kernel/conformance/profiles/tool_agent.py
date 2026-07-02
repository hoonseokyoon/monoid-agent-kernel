"""Tool-using agent profile metadata and assertions."""

from __future__ import annotations

import time
from collections.abc import Callable

from monoid_agent_kernel.conformance.harness import BackendHarness

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="tool-agent",
    title="Tool Agent",
    summary="Agent integration that executes tools with bindings, permissions, and output validation.",
    rule_ids=(
        "OR-10-TOOL-SURFACE-ADMISSION",
        "OR-11-GENERIC-ASK-APPROVAL",
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
    approved_events = _wait_for_events(
        harness,
        approved,
        lambda events: _has_event(events, "tool.approval.requested")
        and _has_event(events, "tool.approval.approved")
        and _has_successful_approval_replay(events),
    )
    assert _has_event(approved_events, "tool.approval.requested")
    assert _has_event(approved_events, "tool.approval.approved")
    assert _has_successful_approval_replay(approved_events)
    _cancel(harness, approved)

    denied = harness.submit_run({"scenario": "tool-ask-denied"})
    denied_events = _wait_for_events(
        harness,
        denied,
        lambda events: _has_event(events, "tool.approval.requested")
        and _has_event(events, "tool.approval.denied"),
    )
    assert _has_event(denied_events, "tool.approval.requested")
    assert _has_event(denied_events, "tool.approval.denied")
    assert not _has_successful_approval_replay(denied_events)
    _cancel(harness, denied)

    stale = harness.submit_run({"scenario": "tool-ask-stale-denied"})
    stale_events = _wait_for_events(
        harness,
        stale,
        lambda events: _has_event(events, "tool.approval.approved")
        and _has_stale_approval_rejection(events),
    )
    assert _has_event(stale_events, "tool.approval.approved")
    assert _has_stale_approval_rejection(stale_events)
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


def _has_stale_approval_rejection(events: list[dict]) -> bool:
    return any(
        event["data"].get("error_code") in {
            "tool_not_in_surface",
            "tool_binding_denied",
            "tool_unknown",
        }
        and event["type"] in {"permission.denied", "tool.call.failed"}
        for event in events
    )


def _wait_for_events(
    harness: BackendHarness,
    submitted: dict,
    predicate: Callable[[list[dict]], bool],
    *,
    timeout_s: float = 20.0,
) -> list[dict]:
    run_id = str(submitted["run_id"])
    token = str(submitted["token"])
    deadline = time.time() + timeout_s
    events: list[dict] = []
    while time.time() < deadline:
        events = list(harness.events(run_id, token, limit=200)["events"])
        if predicate(events):
            return events
        time.sleep(0.05)
    return events


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
