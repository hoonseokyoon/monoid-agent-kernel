"""Multi-agent profile metadata."""

from __future__ import annotations

import json
from typing import Any

from monoid_agent_kernel.conformance.harness import BackendHarness, CapabilityHarness

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="multi-agent",
    title="Multi Agent",
    summary="Subagent runtime with identity, capability isolation, shared revocation, and trace linkage.",
    rule_ids=("OR-04-REVOCATION-SCOPE", "OR-09-SUBAGENT-BOUNDARY"),
    harnesses=("backend", "capability"),
)


def assert_multi_agent_backend_boundary_profile(harness: BackendHarness) -> None:
    """Run the Phase 1S backend subagent identity and diagnostics smoke matrix."""
    submitted = harness.submit_run({"scenario": "subagent-foreground"})
    run_id = str(submitted["run_id"])
    token = str(submitted["token"])
    events = list(harness.events(run_id, token)["events"])
    assert all(event["run_id"] == run_id for event in events)
    started = _one_event(events, "subagent.started")
    terminal = _one_event(events, "subagent.finished")
    started_data = dict(started["data"])
    terminal_data = dict(terminal["data"])
    child_run_id = str(started_data["child_run_id"])
    task_id = str(started_data["task_id"])
    traceparent = str(started_data["traceparent"])

    assert child_run_id.startswith(f"{run_id}.sub.")
    assert started_data["parent_run_id"] == run_id
    assert started_data["root_run_id"] == run_id
    assert terminal_data["child_run_id"] == child_run_id
    assert terminal_data["traceparent"] == traceparent
    assert terminal["parent_id"] == started["event_id"]

    child_events = list(harness.descendant_events(run_id, token, child_run_id)["events"])
    assert child_events
    assert all(event["run_id"] == child_run_id for event in child_events)
    _assert_raises(lambda: harness.descendant_events(run_id, token, "some.other.run"))
    _assert_raises(lambda: harness.descendant_events(run_id, token, f"{run_id}.sub.../escape"))

    task = harness.task_result(run_id, token, task_id)["result"]
    assert task["child_run_id"] == child_run_id
    assert task["traceparent"] == traceparent

    diagnostics = harness.diagnostics(run_id, token, event_limit=50)
    item = _diagnostic_subagent(diagnostics, child_run_id)
    assert item["task_id"] == task_id
    assert item["traceparent"] == traceparent
    assert item["status"] == "completed"
    assert item["usage"].get("total_tokens") == 10

    result = harness.result(run_id, token)
    assert result["metrics"]["subagent_count"] == 1
    assert result["metrics"]["subagent_usage"]["total_tokens"] == 10
    assert result["metrics"]["total_tokens"] == 10


def assert_multi_agent_backend_capability_boundary_profile(harness: BackendHarness) -> None:
    """Run the Phase 1S child capability-boundary smoke matrix."""
    submitted = harness.submit_run({"scenario": "subagent-capability-revoked"})
    run_id = str(submitted["run_id"])
    token = str(submitted["token"])
    events = list(harness.events(run_id, token)["events"])
    started = _one_event(events, "subagent.started")
    child_run_id = str(started["data"]["child_run_id"])
    task_id = str(started["data"]["task_id"])

    child_events = list(harness.descendant_events(run_id, token, child_run_id)["events"])
    assert any(
        event["type"] == "capability.revoked"
        and event["data"].get("capability") == "mcp.demo.gated"
        for event in child_events
    )
    task = harness.task_result(run_id, token, task_id)["result"]
    assert "capability_revoked" in json.dumps(task, sort_keys=True)


def assert_multi_agent_shared_revocation_profile(harness: CapabilityHarness) -> None:
    """Run the Phase 1S child-vault revocation sharing smoke matrix."""
    _admit_profile_lease(
        harness,
        capability="web.search",
        lease_id="lease_parent_live",
        token_ref="token:parent-live",
        durable=False,
    )
    _admit_profile_lease(
        harness,
        capability="web.fetch",
        lease_id="lease_parent_durable",
        token_ref="token:parent-durable",
        durable=True,
    )

    child = harness.fork_child()
    assert child.token_for("web.search", now=200.0) is None
    assert child.token_for("web.fetch", now=200.0) == "token:parent-durable"

    _admit_profile_lease(
        child,
        capability="web.search",
        lease_id="lease_child_live",
        token_ref="token:child-live",
        durable=False,
    )
    assert harness.token_for("web.search", now=200.0) == "token:parent-live"
    assert child.token_for("web.search", now=200.0) == "token:child-live"

    harness.revoke_capability({"capability": "web.search"})
    assert harness.token_for("web.search", now=200.0) is None
    assert child.token_for("web.search", now=200.0) is None

    _admit_profile_lease(
        child,
        capability="web.context",
        lease_id="lease_child_only",
        token_ref="token:child-only",
        durable=False,
    )
    assert child.token_for("web.context", now=200.0) == "token:child-only"
    assert harness.token_for("web.context", now=200.0) is None

    child.revoke_capability({"capability": "*"})
    assert harness.token_for("web.fetch", now=200.0) is None
    assert child.token_for("web.context", now=200.0) is None


def _one_event(events: list[dict[str, Any]], event_type: str) -> dict[str, Any]:
    matches = [event for event in events if event["type"] == event_type]
    assert matches, f"missing event {event_type}"
    return matches[0]


def _diagnostic_subagent(diagnostics: dict[str, Any], child_run_id: str) -> dict[str, Any]:
    items = list(diagnostics["subagents"]["items"])
    matches = [item for item in items if item["child_run_id"] == child_run_id]
    assert matches, f"missing diagnostics subagent {child_run_id}"
    return matches[0]


def _assert_raises(fn: Any) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError("expected operation to fail")


def _admit_profile_lease(
    harness: CapabilityHarness,
    *,
    capability: str,
    lease_id: str,
    token_ref: str,
    durable: bool,
) -> dict[str, Any]:
    request = harness.request_capability({"capability": capability, "scope": {}})
    return harness.grant_capability(
        str(request["request_id"]),
        {
            "lease_id": lease_id,
            "capability": capability,
            "token_ref": token_ref,
            "expires_at": 4_102_444_800.0,
            "issued_at": 100.0,
            "durable": durable,
            "scope": {},
        },
    )
