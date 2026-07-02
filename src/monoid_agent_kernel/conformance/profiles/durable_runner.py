"""Durable runner profile metadata."""

from __future__ import annotations

from typing import Any

from monoid_agent_kernel.conformance.harness import BackendHarness

from ._metadata import ProfileMetadata

PROFILE = ProfileMetadata(
    profile_id="durable-runner",
    title="Durable Runner",
    summary="Backend that preserves run state, event sequence, diagnostics, and recovery metadata.",
    rule_ids=("OR-05-EVENT-SEQUENCING", "OR-07-DURABLE-METADATA", "OR-09-SUBAGENT-BOUNDARY"),
    harnesses=("backend",),
)


def assert_durable_runner_event_sequence_profile(harness: BackendHarness) -> None:
    """Run the Phase 1S event sequence and diagnostics smoke matrix."""
    submitted = harness.submit_run({"scenario": "multi-turn"})
    run_id = str(submitted["run_id"])
    token = str(submitted["token"])
    try:
        result = harness.dispatch(
            {
                "type": "status",
                "run_id": run_id,
                "args": {"token": token},
                "issuer": "profile",
                "reason": "sequence profile",
                "command_id": "cmd_profile_sequence_status",
            }
        )
        assert result["status"] == "ok"

        events = list(harness.events(run_id, token)["events"])
        _assert_monotonic_unique_sequence(events)
        control_completed = [
            event
            for event in events
            if event["type"] == "control.command.completed"
            and event["data"].get("command_id") == "cmd_profile_sequence_status"
        ]
        assert control_completed

        diagnostics = harness.diagnostics(run_id, token, event_limit=1)
        items = list(diagnostics["events"]["items"])
        assert items
        assert items[-1]["seq"] == max(int(event["seq"]) for event in events)
        assert diagnostics["events"]["next_seq"] >= items[-1]["seq"]
    finally:
        _cancel(harness, run_id, token, command_id="cmd_profile_sequence_cleanup")


def assert_durable_runner_recovery_metadata_profile(harness: BackendHarness) -> None:
    """Run the Phase 1S recovery metadata smoke matrix."""
    same_run = harness.submit_run({"scenario": "recoverable-multi-turn"})
    same_run_id = str(same_run["run_id"])
    same_token = str(same_run["token"])
    empty_run_id = ""
    empty_token = ""
    same_restart: BackendHarness | None = None
    empty_restart: BackendHarness | None = None
    try:
        same_update = _replace_with_next_config(
            harness,
            same_run_id,
            same_token,
            command_label="same",
        )
        same_restart = harness.restart(local_state="same")
        resumed = same_restart.resume_run(same_run_id, same_token)
        assert resumed["resumed"] is True
        assert same_restart.runtime_config(same_run_id, same_token)["config_hash"] == same_update["config_hash"]

        empty_run = harness.submit_run({"scenario": "recoverable-multi-turn"})
        empty_run_id = str(empty_run["run_id"])
        empty_token = str(empty_run["token"])
        empty_update = _replace_with_next_config(
            harness,
            empty_run_id,
            empty_token,
            command_label="empty",
        )
        empty_restart = harness.restart(local_state="empty")
        materialized = empty_restart.resume_run(empty_run_id, empty_token)
        assert materialized["resumed"] is True
        assert empty_restart.runtime_config(empty_run_id, empty_token)["config_hash"] == empty_update["config_hash"]
    finally:
        _cancel(harness, same_run_id, same_token, command_id="cmd_profile_recovery_cleanup_same_original")
        if same_restart is not None:
            _cancel(same_restart, same_run_id, same_token, command_id="cmd_profile_recovery_cleanup_same_restart")
        if empty_run_id:
            _cancel(harness, empty_run_id, empty_token, command_id="cmd_profile_recovery_cleanup_empty_original")
        if empty_restart is not None and empty_run_id:
            _cancel(empty_restart, empty_run_id, empty_token, command_id="cmd_profile_recovery_cleanup_empty_restart")


def _replace_with_next_config(
    harness: BackendHarness,
    run_id: str,
    token: str,
    *,
    command_label: str,
) -> dict[str, Any]:
    current = harness.runtime_config(run_id, token)
    config = dict(current["config"])
    current_version = int(current["config_version"])
    config["config_version"] = current_version + 1
    config.pop("config_hash", None)
    updated = harness.replace_runtime_config(
        run_id,
        token,
        config,
        expected_version=current_version,
        issuer="profile",
        reason=f"{command_label} recovery config",
    )
    assert updated["config_version"] == current_version + 1
    return dict(updated)


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
