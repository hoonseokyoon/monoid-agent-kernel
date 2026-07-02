"""Control-plane profile metadata."""

from __future__ import annotations

from typing import Any

from monoid_agent_kernel.conformance.harness import BackendHarness

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="control-plane",
    title="Control Plane",
    summary="Backend with external control commands, audit events, lifecycle policy, and stable results.",
    rule_ids=(
        "OR-03-LEASE-ADMISSION",
        "OR-05-EVENT-SEQUENCING",
        "OR-06-CONTROL-AUDIT",
        "OR-07-DURABLE-METADATA",
    ),
    harnesses=("backend", "capability"),
)


def assert_control_plane_decision_profile(harness: BackendHarness) -> None:
    """Run the Phase 1S approve/deny control decision smoke matrix."""
    submitted = harness.submit_run({"scenario": "parked-hitl"})
    run_id = str(submitted["run_id"])
    token = str(submitted["token"])
    try:
        approve_task = _create_hitl_task(harness, run_id, token, command_id="cmd_profile_create_approve")
        approved = harness.dispatch(
            {
                "type": "approve",
                "run_id": run_id,
                "args": {"token": approve_task["callback_token"], "task_id": approve_task["task_id"]},
                "issuer": "callback_worker",
                "reason": "approved by profile",
                "command_id": "cmd_profile_approve",
            }
        )
        assert approved["status"] == "ok"
        assert approved["data"]["delivered"] is True

        deny_task = _create_hitl_task(harness, run_id, token, command_id="cmd_profile_create_deny")
        denied = harness.dispatch(
            {
                "type": "deny",
                "run_id": run_id,
                "args": {
                    "token": token,
                    "task_id": deny_task["task_id"],
                    "result": {
                        "answer": "Approve",
                        "approved": True,
                        "granted": True,
                        "lease": {"capability": "web.search", "token_ref": "secret-ref://lease"},
                        "token_ref": "secret-ref://lease",
                    },
                },
                "issuer": "operator_a",
                "reason": "policy denied",
                "command_id": "cmd_profile_deny",
            }
        )
        assert denied["status"] == "ok"
        assert denied["data"]["delivered"] is True

        result = harness.task_result(run_id, token, str(deny_task["task_id"]))["result"]
        assert result["answer"] == "Deny"
        assert result["approved"] is False
        assert result["granted"] is False
        assert result["reason"] == "policy denied"
        assert "lease" not in result
        assert "token_ref" not in result

        events = list(harness.events(run_id, token)["events"])
        by_id = {
            (event["type"], event["data"].get("command_id")): event["data"]
            for event in events
            if str(event["type"]).startswith("control.command.")
        }
        assert by_id[("control.command.received", "cmd_profile_approve")]["command"] == "approve"
        assert by_id[("control.command.completed", "cmd_profile_approve")]["result_code"] == "ok"
        assert by_id[("control.command.received", "cmd_profile_deny")]["command"] == "deny"
        assert by_id[("control.command.completed", "cmd_profile_deny")]["result_code"] == "ok"
    finally:
        try:
            harness.dispatch(
                {
                    "type": "cancel",
                    "run_id": run_id,
                    "args": {"token": token},
                    "issuer": "profile",
                    "command_id": "cmd_profile_cleanup_cancel",
                }
            )
        except Exception:
            pass


def _create_hitl_task(
    harness: BackendHarness,
    run_id: str,
    token: str,
    *,
    command_id: str,
) -> dict[str, Any]:
    created = harness.dispatch(
        {
            "type": "create_task",
            "run_id": run_id,
            "args": {
                "token": token,
                "kind": "hitl",
                "request": {"prompt": "Approve profile action?", "choices": ["Approve", "Deny"]},
            },
            "issuer": "profile",
            "command_id": command_id,
        }
    )
    assert created["status"] == "ok"
    return dict(created["data"])
