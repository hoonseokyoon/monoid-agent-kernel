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


def assert_control_plane_audit_sequence_profile(harness: BackendHarness) -> None:
    """Run the Phase 1S control audit sequencing smoke matrix."""
    submitted = harness.submit_run({"scenario": "multi-turn"})
    run_id = str(submitted["run_id"])
    token = str(submitted["token"])
    terminal = harness.submit_run({"scenario": "completed"})
    terminal_run_id = str(terminal["run_id"])
    terminal_token = str(terminal["token"])
    try:
        status = harness.dispatch(
            {
                "type": "status",
                "run_id": run_id,
                "args": {"token": token},
                "issuer": "operator_a",
                "reason": "audit ok",
                "command_id": "cmd_profile_audit_status",
            }
        )
        assert status["status"] == "ok"

        current = harness.runtime_config(run_id, token)
        failed = harness.dispatch(
            {
                "type": "replace_runtime_config",
                "run_id": run_id,
                "args": {"token": token, "expected_version": 999, "config": current["config"]},
                "issuer": "operator_a",
                "reason": "audit failure",
                "command_id": "cmd_profile_audit_bad_replace",
            }
        )
        assert failed["status"] == "error"

        try:
            rejected = harness.dispatch(
                {
                    "type": "inspect",
                    "run_id": run_id,
                    "args": {"token": "bad-token"},
                    "issuer": "operator_b",
                    "reason": "bad auth",
                    "command_id": "cmd_profile_audit_bad_auth",
                }
            )
            assert rejected["status"] != "ok"
        except Exception:
            pass

        events = list(harness.events(run_id, token)["events"])
        _assert_monotonic_unique_sequence(events)
        control = [
            event for event in events if str(event.get("type") or "").startswith("control.command.")
        ]
        by_id = {(event["type"], event["data"].get("command_id")): event["data"] for event in control}
        assert by_id[("control.command.received", "cmd_profile_audit_status")]["args_keys"] == []
        assert by_id[("control.command.completed", "cmd_profile_audit_status")]["result_code"] == "ok"
        assert by_id[("control.command.failed", "cmd_profile_audit_bad_replace")]["failure_code"]
        assert all(event["data"].get("command_id") != "cmd_profile_audit_bad_auth" for event in control)
        assert token not in repr(control)
        assert "bad-token" not in repr(control)

        terminal_status = harness.dispatch(
            {
                "type": "status",
                "run_id": terminal_run_id,
                "args": {"token": terminal_token},
                "issuer": "operator_a",
                "reason": "terminal audit",
                "command_id": "cmd_profile_audit_terminal_status",
            }
        )
        assert terminal_status["status"] == "ok"
        terminal_events = list(harness.events(terminal_run_id, terminal_token)["events"])
        _assert_monotonic_unique_sequence(terminal_events)
        assert any(
            event["type"] == "control.command.completed"
            and event["data"].get("command_id") == "cmd_profile_audit_terminal_status"
            for event in terminal_events
        )
    finally:
        _cancel(harness, run_id, token, command_id="cmd_profile_audit_cleanup_live")
        _cancel(harness, terminal_run_id, terminal_token, command_id="cmd_profile_audit_cleanup_terminal")


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


def _assert_monotonic_unique_sequence(events: list[dict[str, Any]]) -> None:
    seqs = [int(event["seq"]) for event in events]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))


def _cancel(harness: BackendHarness, run_id: str, token: str, *, command_id: str) -> None:
    try:
        harness.dispatch(
            {
                "type": "cancel",
                "run_id": run_id,
                "args": {"token": token},
                "issuer": "profile",
                "command_id": command_id,
            }
        )
    except Exception:
        pass
