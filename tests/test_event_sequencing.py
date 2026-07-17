from __future__ import annotations

import json

import pytest

from monoid_agent_kernel.core._event_log import EventLogCorruption, inspect_event_log_tail
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


def test_page_reader_hides_uncommitted_tail_until_newline(tmp_path) -> None:
    events_path = tmp_path / "events.jsonl"
    events_path.write_bytes(b'{"seq":1}\n{"seq":2}')
    sequencer = RunEventSequencer()

    first_page = sequencer.read_event_page(events_path, from_seq=0, limit=None)
    with events_path.open("ab") as handle:
        handle.write(b"\n")
    second_page = sequencer.read_event_page(events_path, from_seq=2, limit=None)

    assert [event["seq"] for event in first_page["events"]] == [1]
    assert first_page["next_seq"] == 2
    assert [event["seq"] for event in second_page["events"]] == [2]


def test_event_log_tail_scan_uses_binary_offsets_for_crlf_and_utf8(tmp_path) -> None:
    events_path = tmp_path / "events.jsonl"
    events_path.write_bytes(
        '{"seq":1,"data":{"text":"한글"}}\r\n{"seq":2,"data":{}}\r\n'.encode()
    )

    tail = inspect_event_log_tail(events_path)

    assert tail.last_seq == 2
    assert tail.committed_end == tail.file_size
    assert tail.incomplete_size == 0


def test_event_log_tail_scan_is_bounded_by_tail_record_not_history(tmp_path) -> None:
    events_path = tmp_path / "events.jsonl"
    events_path.write_bytes(
        b"".join(f'{{"seq":{seq}}}\n'.encode() for seq in range(1, 100_001))
    )

    tail = inspect_event_log_tail(events_path)

    assert tail.file_size > 1_000_000
    assert tail.last_seq == 100_000
    assert tail.inspected_bytes < 70_000


def test_direct_append_truncates_unacknowledged_incomplete_tail(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run_partial"
    run_dir.mkdir(parents=True)
    events_path = run_dir / "events.jsonl"
    events_path.write_bytes(b'{"seq":1}\n{"seq":2')
    (run_dir / "status.json").write_text(
        json.dumps({"run_id": "run_partial", "last_event_seq": 1}),
        encoding="utf-8",
    )

    event = append_event_to_run(run_dir, "control.command.received")

    assert event.seq == 2
    records = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert [record["seq"] for record in records] == [1, 2]


def test_direct_append_treats_valid_record_without_newline_as_uncommitted(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run_no_newline"
    run_dir.mkdir(parents=True)
    events_path = run_dir / "events.jsonl"
    events_path.write_bytes(b'{"seq":1}')

    event = append_event_to_run(run_dir, "control.command.received")

    assert event.seq == 1
    assert [json.loads(line)["seq"] for line in events_path.read_text().splitlines()] == [1]


@pytest.mark.parametrize("with_matching_status", [False, True])
def test_direct_append_rejects_out_of_order_history(
    tmp_path,
    with_matching_status: bool,
) -> None:
    run_dir = tmp_path / "runs" / "run_out_of_order"
    run_dir.mkdir(parents=True)
    events_path = run_dir / "events.jsonl"
    original = b'{"seq":1}\n{"seq":3}\n{"seq":2}\n'
    events_path.write_bytes(original)
    if with_matching_status:
        (run_dir / "status.json").write_text(
            json.dumps({"run_id": "run_out_of_order", "last_event_seq": 2}),
            encoding="utf-8",
        )

    with pytest.raises(EventLogCorruption, match="sequence is not increasing"):
        append_event_to_run(run_dir, "control.command.received")

    assert events_path.read_bytes() == original


def test_incomplete_tail_fails_closed_when_status_watermark_is_ahead(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run_acknowledged_partial"
    run_dir.mkdir(parents=True)
    events_path = run_dir / "events.jsonl"
    original = b'{"seq":1}\n{"seq":2'
    events_path.write_bytes(original)
    (run_dir / "status.json").write_text(
        json.dumps({"run_id": "run_acknowledged_partial", "last_event_seq": 2}),
        encoding="utf-8",
    )

    with pytest.raises(EventLogCorruption, match="acknowledged status watermark"):
        append_event_to_run(run_dir, "control.command.received")

    assert events_path.read_bytes() == original


def test_complete_tail_fails_closed_when_status_watermark_is_ahead(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "run_missing_acknowledged_event"
    run_dir.mkdir(parents=True)
    events_path = run_dir / "events.jsonl"
    original = b'{"seq":1}\n'
    events_path.write_bytes(original)
    status_path = run_dir / "status.json"
    original_status = json.dumps(
        {"run_id": "run_missing_acknowledged_event", "last_event_seq": 2}
    )
    status_path.write_text(original_status, encoding="utf-8")

    with pytest.raises(EventLogCorruption, match="acknowledged status watermark"):
        append_event_to_run(run_dir, "control.command.received")

    assert events_path.read_bytes() == original
    assert status_path.read_text(encoding="utf-8") == original_status


@pytest.mark.parametrize(
    "invalid_status",
    [
        "{",
        "[]",
        '{"last_event_seq":true}',
        '{"last_event_seq":' + ("9" * 5_000) + "}",
    ],
)
def test_incomplete_tail_fails_closed_when_status_watermark_is_invalid(
    tmp_path,
    invalid_status: str,
) -> None:
    run_dir = tmp_path / "runs" / "run_unknown_watermark"
    run_dir.mkdir(parents=True)
    events_path = run_dir / "events.jsonl"
    original = b'{"seq":1}\npartial'
    events_path.write_bytes(original)
    (run_dir / "status.json").write_text(invalid_status, encoding="utf-8")

    with pytest.raises(EventLogCorruption, match="watermark cannot be verified"):
        append_event_to_run(run_dir, "control.command.received")

    assert events_path.read_bytes() == original


def test_incomplete_tail_fails_closed_when_status_watermark_is_unreadable(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "runs" / "run_unreadable_watermark"
    run_dir.mkdir(parents=True)
    events_path = run_dir / "events.jsonl"
    original = b'{"seq":1}\npartial'
    events_path.write_bytes(original)
    (run_dir / "status.json").write_text(
        json.dumps({"run_id": "run_unreadable_watermark", "last_event_seq": 1}),
        encoding="utf-8",
    )
    status_path = run_dir / "status.json"
    path_type = type(status_path)
    read_text = path_type.read_text

    def fail_status_read(path, *args, **kwargs):
        if path == status_path:
            raise OSError("simulated read failure")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(path_type, "read_text", fail_status_read)

    with pytest.raises(EventLogCorruption, match="watermark cannot be verified"):
        append_event_to_run(run_dir, "control.command.received")

    assert events_path.read_bytes() == original


@pytest.mark.parametrize(
    "committed_tail, message",
    [
        (b'{"seq":\n', "valid JSON"),
        (b"\xff\n", "valid UTF-8"),
        (b'[{"seq":1}]\n', "JSON object"),
        (b'{}\n', "invalid sequence"),
        (b'{"seq":null}\n', "invalid sequence"),
        (b'{"seq":true}\n', "invalid sequence"),
        (b'{"seq":-1}\n', "invalid sequence"),
        (b'{"seq":"2"}\n', "invalid sequence"),
        (b'{"seq":1.5}\n', "invalid sequence"),
        (b'{"seq":[1]}\n', "invalid sequence"),
    ],
)
def test_committed_malformed_tail_blocks_append_without_mutation(
    tmp_path,
    committed_tail: bytes,
    message: str,
) -> None:
    run_dir = tmp_path / "runs" / "run_corrupt"
    run_dir.mkdir(parents=True)
    events_path = run_dir / "events.jsonl"
    original = b'{"seq":1}\n' + committed_tail
    events_path.write_bytes(original)

    with pytest.raises(EventLogCorruption, match=message):
        append_event_to_run(run_dir, "control.command.received")

    assert events_path.read_bytes() == original


def test_recorder_reopen_repairs_safe_partial_tail_before_opening_sink(tmp_path) -> None:
    run_root = tmp_path / "runs"
    run_dir = run_root / "run_reopen"
    (run_dir / "artifacts").mkdir(parents=True)
    events_path = run_dir / "events.jsonl"
    events_path.write_bytes(b'{"seq":1}\npartial')
    (run_dir / "status.json").write_text(
        json.dumps({"run_id": "run_reopen", "last_event_seq": 1}),
        encoding="utf-8",
    )

    recorder = AgentRecorder(run_root, "run_reopen", reopen=True)
    try:
        event = recorder.emit("run.started")
    finally:
        recorder.close()

    assert event.seq == 2
    assert [json.loads(line)["seq"] for line in events_path.read_text().splitlines()] == [1, 2]


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
