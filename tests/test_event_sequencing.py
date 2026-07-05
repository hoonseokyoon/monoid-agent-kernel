from __future__ import annotations

import json

from monoid_agent_kernel.core.event_sequencing import RunEventSequencer
from monoid_agent_kernel.recorder import AgentRecorder, append_event_to_run


def test_live_record_status_requires_live_sequence_owner() -> None:
    sequencer = RunEventSequencer()

    assert sequencer.requires_live_sequence_owner("running") is True
    assert sequencer.requires_live_sequence_owner("awaiting_input") is True
    assert sequencer.is_queued_before_recorder("queued") is True
    assert sequencer.is_terminal_direct_append_status("completed") is True


def test_queued_direct_append_seeds_later_recorder(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "run_queued"
    sequencer = RunEventSequencer()

    assert sequencer.is_queued_before_recorder("queued") is True
    first = append_event_to_run(
        run_dir,
        "control.command.received",
        data={"command_id": "cmd_1", "command": "status"},
    )

    recorder = AgentRecorder(run_root, "run_queued")
    try:
        second = recorder.emit("run.started", data={"mode": "propose"})
    finally:
        recorder.close()

    assert first.seq == 1
    assert second.seq == 2


def test_terminal_direct_append_keeps_unique_increasing_sequence(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run_terminal"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps({"run_id": "run_terminal", "status": "completed", "last_event_seq": 0}),
        encoding="utf-8",
    )
    sequencer = RunEventSequencer()

    assert sequencer.run_dir_allows_direct_append(run_dir) is True
    append_event_to_run(run_dir, "control.command.received", data={"command_id": "cmd_1"})
    append_event_to_run(run_dir, "control.command.completed", data={"command_id": "cmd_1"})

    page = sequencer.read_event_page(run_dir / "events.jsonl", from_seq=0, limit=None)
    assert [event["seq"] for event in page["events"]] == [1, 2]
    assert page["next_seq"] == 3


def test_legacy_limited_status_allows_direct_append(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run_limited"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps({"run_id": "run_limited", "status": "limited", "last_event_seq": 0}),
        encoding="utf-8",
    )

    assert RunEventSequencer().run_dir_allows_direct_append(run_dir) is True


def test_live_limited_state_requires_terminal_flag_for_direct_append(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run_live_limited"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps({"run_id": "run_live_limited", "state": "limited", "last_event_seq": 0}),
        encoding="utf-8",
    )
    sequencer = RunEventSequencer()

    assert sequencer.run_dir_allows_direct_append(run_dir) is False
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "run_live_limited",
                "state": "limited",
                "terminal": True,
                "last_event_seq": 0,
            }
        ),
        encoding="utf-8",
    )
    assert sequencer.run_dir_allows_direct_append(run_dir) is True


def test_recordless_nonterminal_run_skips_direct_append(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run_live"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps({"run_id": "run_live", "status": "running", "last_event_seq": 1}),
        encoding="utf-8",
    )

    assert RunEventSequencer().run_dir_allows_direct_append(run_dir) is False


def test_diagnostics_tail_uses_newest_known_sequence() -> None:
    sequencer = RunEventSequencer()

    from_seq = sequencer.diagnostics_from_seq(
        {"last_event_seq": 2},
        {"last_event_seq": 5},
        event_limit=3,
    )

    assert from_seq == 3
