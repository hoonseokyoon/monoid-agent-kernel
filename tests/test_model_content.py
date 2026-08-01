from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core.model_content import (
    MODEL_CONTENT_SCHEMA_VERSION,
    ModelContentStore,
    read_model_content,
)
from monoid_agent_kernel.core.model_io import content_digest
from monoid_agent_kernel.core.model_stream import (
    ModelStreamContext,
    ModelStreamDelta,
    ModelStreamOutcome,
    safe_open_model_stream,
)
from monoid_agent_kernel.recorder import AgentRecorder


def _context(
    stream_id: str = "stream-1",
    *,
    run_id: str = "run-1",
) -> ModelStreamContext:
    return ModelStreamContext(
        run_id=run_id,
        root_run_id=run_id,
        turn_id="turn-1",
        stream_id=stream_id,
        step=1,
        provider="test-provider",
        model="test-model",
        started_at="2026-08-01T00:00:00Z",
    )


def _records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def test_first_segment_is_immediate_and_channel_switch_flushes(tmp_path: Path) -> None:
    path = tmp_path / "model-content.jsonl"
    store = ModelContentStore(path, run_id="run-1")
    writer = store.open(_context())

    writer.push(ModelStreamDelta("output", "a"))
    assert [record["kind"] for record in _records(path)] == [
        "stream_opened",
        "stream_segment",
    ]
    writer.push(ModelStreamDelta("output", "b"))
    # The second same-channel delta is waiting in the bounded batch.
    assert [record.get("text") for record in _records(path) if "text" in record] == ["a"]

    writer.push(ModelStreamDelta("reasoning", "thinking"))
    assert [record.get("text") for record in _records(path) if "text" in record] == ["a", "b"]
    writer.close(
        ModelStreamOutcome(
            "completed",
            final_text="ab",
            usage={"input_tokens": 2, "output_tokens": 2},
        )
    )
    store.close()

    records = _records(path)
    assert all(record["schema_version"] == MODEL_CONTENT_SCHEMA_VERSION for record in records)
    assert [record["kind"] for record in records] == [
        "stream_opened",
        "stream_segment",
        "stream_segment",
        "stream_segment",
        "stream_closed",
    ]
    assert [record["segment_index"] for record in records if "segment_index" in record] == [
        0,
        1,
        2,
    ]

    recovered = read_model_content(path)
    assert recovered.skipped_records == 0
    assert len(recovered.snapshots) == 1
    snapshot = recovered.snapshots[0]
    assert snapshot.status == "completed"
    assert snapshot.output_text == "ab"
    assert snapshot.reasoning_text == "thinking"
    assert snapshot.final_text == "ab"
    assert snapshot.best_output_text == "ab"
    assert snapshot.usage == {"input_tokens": 2, "output_tokens": 2}


def test_segments_are_bounded_by_utf8_bytes_and_flush_on_size(tmp_path: Path) -> None:
    path = tmp_path / "model-content.jsonl"
    store = ModelContentStore(path, run_id="run-1", max_segment_bytes=4)
    writer = store.open(_context())

    writer.push(ModelStreamDelta("output", "한글a"))
    writer.close(ModelStreamOutcome("interrupted", final_text="한글a"))
    store.close()

    segments = [record for record in _records(path) if record.get("kind") == "stream_segment"]
    assert "".join(record["text"] for record in segments) == "한글a"
    assert all(len(record["text"].encode("utf-8")) <= 4 for record in segments)


def test_same_channel_batch_flushes_on_its_deadline(tmp_path: Path) -> None:
    path = tmp_path / "model-content.jsonl"
    store = ModelContentStore(path, run_id="run-1", batch_interval_s=0.02)
    writer = store.open(_context())
    writer.push(ModelStreamDelta("output", "first"))
    writer.push(ModelStreamDelta("output", "second"))

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        segments = [record for record in _records(path) if record.get("kind") == "stream_segment"]
        if len(segments) == 2:
            break
        time.sleep(0.005)
    else:
        raise AssertionError("the buffered segment did not flush on its deadline")

    writer.close(ModelStreamOutcome("completed", final_text="firstsecond"))
    store.close()


def test_store_close_flushes_partial_and_reader_marks_stream_abandoned(tmp_path: Path) -> None:
    path = tmp_path / "model-content.jsonl"
    store = ModelContentStore(path, run_id="run-1")
    writer = store.open(_context())
    writer.push(ModelStreamDelta("output", "first"))
    writer.push(ModelStreamDelta("output", " buffered"))

    store.close()

    snapshot = read_model_content(path).snapshots[0]
    assert snapshot.status == "abandoned"
    assert snapshot.output_text == "first buffered"
    assert not any(record.get("kind") == "stream_closed" for record in _records(path))


def test_torn_tail_is_isolated_before_append_and_reader_skips_it(tmp_path: Path) -> None:
    path = tmp_path / "model-content.jsonl"
    path.write_bytes(b'{"kind":"stream_segment","text":"torn')
    store = ModelContentStore(path, run_id="run-1")
    store.open(_context())
    store.close()

    raw = path.read_bytes()
    assert b'torn\n{"kind":"stream_opened"' in raw
    result = read_model_content(path)
    assert result.skipped_records == 1
    assert result.snapshots[0].status == "abandoned"


def test_reader_accepts_a_run_directory_whose_name_ends_in_jsonl(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-name.jsonl"
    path = run_dir / "model-content.jsonl"
    store = ModelContentStore(path, run_id="run-name.jsonl")
    writer = store.open(_context(run_id="run-name.jsonl"))
    writer.push(ModelStreamDelta("output", "recovered"))
    writer.close(ModelStreamOutcome("completed", final_text="recovered"))
    store.close()

    result = read_model_content(run_dir)

    assert result.snapshots[0].best_output_text == "recovered"


def test_reader_skips_schema_invalid_stream_metadata(tmp_path: Path) -> None:
    path = tmp_path / "model-content.jsonl"
    records = [
        {
            "schema_version": MODEL_CONTENT_SCHEMA_VERSION,
            "kind": "stream_opened",
            "run_id": "run-1",
            "root_run_id": "run-1",
            "turn_id": "turn-1",
            "stream_id": "bad-open",
            "step": 0,
            "provider": None,
            "model": None,
            "started_at": "2026-08-01T00:00:00Z",
        },
        {
            "schema_version": MODEL_CONTENT_SCHEMA_VERSION,
            "kind": "stream_opened",
            "run_id": "run-1",
            "root_run_id": "run-1",
            "turn_id": "turn-1",
            "stream_id": "valid-open",
            "step": 1,
            "provider": None,
            "model": None,
            "started_at": "2026-08-01T00:00:00Z",
        },
        {
            "schema_version": MODEL_CONTENT_SCHEMA_VERSION,
            "kind": "stream_segment",
            "run_id": "run-1",
            "stream_id": "valid-open",
            "segment_index": 0,
            "channel": "output",
            "text": "must be skipped",
            "text_len": 15,
        },
        {
            "schema_version": MODEL_CONTENT_SCHEMA_VERSION,
            "kind": "stream_closed",
            "run_id": "run-1",
            "stream_id": "valid-open",
            "status": "completed",
            "final_text": "must be skipped",
            "usage": None,
            "error_code": None,
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    result = read_model_content(path)

    assert result.skipped_records == 3
    assert len(result.snapshots) == 1
    assert result.snapshots[0].status == "abandoned"
    assert result.snapshots[0].output_text == ""


def test_reader_recovers_after_malformed_lines_and_rejects_bad_settled_digest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model-content.jsonl"
    store = ModelContentStore(path, run_id="run-1")
    writer = store.open(_context())
    writer.push(ModelStreamDelta("output", "partial"))
    writer.close(ModelStreamOutcome("cancelled", error_code="user_cancelled"))
    store.settled_text("good", content_digest("good"), 4)
    store.close()
    with path.open("ab") as handle:
        handle.write(b"\xff\n")
        handle.write(
            json.dumps(
                {
                    "schema_version": MODEL_CONTENT_SCHEMA_VERSION,
                    "kind": "settled_text",
                    "run_id": "run-1",
                    "final_text": "tampered",
                    "final_text_digest": content_digest("different"),
                    "final_text_len": 8,
                    "recorded_at": "2026-08-01T00:00:01Z",
                }
            ).encode("utf-8")
            + b"\n"
        )

    result = read_model_content(tmp_path)
    assert result.snapshots[0].status == "cancelled"
    assert result.snapshots[0].error_code == "user_cancelled"
    assert result.settled_texts == {content_digest("good"): "good"}
    assert result.skipped_records == 2


def test_reader_does_not_fabricate_text_from_invalid_utf8_inside_json(tmp_path: Path) -> None:
    path = tmp_path / "model-content.jsonl"
    store = ModelContentStore(path, run_id="run-1")
    writer = store.open(_context())
    writer.push(ModelStreamDelta("output", "valid"))

    invalid_segment = (
        b'{"schema_version":"monoid.model-content.v1","kind":"stream_segment",'
        b'"run_id":"run-1","stream_id":"stream-1","segment_index":1,'
        b'"channel":"output","text":"bad\xfftext","text_len":8,'
        b'"emitted_at":"2026-08-01T00:00:01Z"}\n'
    )
    with path.open("ab") as handle:
        handle.write(invalid_segment)
    writer.close(ModelStreamOutcome("completed", final_text="valid"))
    store.close()

    result = read_model_content(path)

    assert result.skipped_records == 1
    assert result.snapshots[0].output_text == "valid"


class _FailingWriter:
    def __init__(self) -> None:
        self.push_calls = 0
        self.close_calls = 0

    def push(self, delta: ModelStreamDelta) -> None:
        self.push_calls += 1
        raise RuntimeError("exporter unavailable")

    def close(self, outcome: ModelStreamOutcome) -> None:
        self.close_calls += 1


class _Observer:
    def __init__(self, writer: _FailingWriter) -> None:
        self.writer = writer

    def open(self, context: ModelStreamContext) -> _FailingWriter:
        return self.writer


class _FailingOpenObserver:
    def open(self, context: ModelStreamContext) -> _FailingWriter:
        raise OSError("unavailable")


def test_observer_failures_are_shielded_and_a_broken_writer_is_disabled() -> None:
    failing = _FailingWriter()
    writer = safe_open_model_stream(_Observer(failing), _context())

    writer.push(ModelStreamDelta("output", "one"))
    writer.push(ModelStreamDelta("output", "two"))
    writer.close(ModelStreamOutcome("failed", error_code="provider_error"))
    writer.close(ModelStreamOutcome("failed", error_code="provider_error"))

    assert failing.push_calls == 1
    assert failing.close_calls == 1
    unavailable = safe_open_model_stream(_FailingOpenObserver(), _context())
    unavailable.push(ModelStreamDelta("output", "safe"))
    unavailable.close(ModelStreamOutcome("completed"))


def test_recorder_sidecar_is_opt_in_and_dual_writes_settled_text(tmp_path: Path) -> None:
    disabled = AgentRecorder(tmp_path, "disabled", status_file=False)
    disabled.settled_text("private")
    disabled.close()
    assert not (disabled.run_dir / "model-content.jsonl").exists()

    enabled = AgentRecorder(
        tmp_path,
        "enabled",
        status_file=False,
        model_content_file=True,
    )
    digest = enabled.settled_text("private")
    # Repeating the content stays deduplicated in both private artifacts.
    assert enabled.settled_text("private") == digest
    enabled.close()

    transcript = _records(enabled.run_dir / "transcript.jsonl")
    sidecar = _records(enabled.run_dir / "model-content.jsonl")
    assert [record["kind"] for record in transcript] == ["settled_text"]
    assert [record["kind"] for record in sidecar] == ["settled_text"]
    assert read_model_content(enabled.run_dir).settled_texts == {digest: "private"}
