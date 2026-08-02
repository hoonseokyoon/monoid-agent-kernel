from __future__ import annotations

import gc
import json
import os
import stat
import threading
import time
import weakref
from pathlib import Path
from typing import Any

import pytest

from monoid_agent_kernel.core.model_content import (
    MODEL_CONTENT_FILENAME,
    MODEL_CONTENT_SCHEMA_VERSION,
    ActiveModelContentState,
    ModelContentStore,
    active_model_content_state,
    active_model_content_stream_ids,
    flush_active_model_content,
    model_content_file_identity,
    read_model_content,
    watch_active_model_content,
    _model_content_file_path,
)
from monoid_agent_kernel.core.model_io import content_digest
from monoid_agent_kernel.core.model_stream import (
    ModelStreamContext,
    ModelStreamDelta,
    ModelStreamOutcome,
    safe_open_model_stream,
)
from monoid_agent_kernel.recorder import AgentRecorder
from monoid_agent_kernel.reference.backend.model_stream import LiveModelStreamBroker


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
    assert snapshot.retryable is False
    closed = next(record for record in records if record["kind"] == "stream_closed")
    assert closed["retryable"] is False


def test_retryable_failed_stream_round_trips_through_private_content(tmp_path: Path) -> None:
    path = tmp_path / "model-content.jsonl"
    store = ModelContentStore(path, run_id="run-1")
    writer = store.open(_context())
    writer.push(ModelStreamDelta("output", "retry prefix"))
    writer.close(
        ModelStreamOutcome(
            "failed",
            final_text="retry prefix",
            error_code="gateway_timeout",
            retryable=True,
        )
    )
    store.close()

    closed = next(record for record in _records(path) if record["kind"] == "stream_closed")
    snapshot = read_model_content(path).snapshots[0]

    assert closed["retryable"] is True
    assert snapshot.status == "failed"
    assert snapshot.retryable is True


def test_reader_defaults_missing_stream_retryability_to_false(tmp_path: Path) -> None:
    path = tmp_path / "model-content.jsonl"
    store = ModelContentStore(path, run_id="run-1")
    writer = store.open(_context())
    writer.push(ModelStreamDelta("output", "old prefix"))
    writer.close(ModelStreamOutcome("failed", final_text="old prefix"))
    store.close()

    records = _records(path)
    closed = next(record for record in records if record["kind"] == "stream_closed")
    closed.pop("retryable")
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    snapshot = read_model_content(path).snapshots[0]

    assert snapshot.status == "failed"
    assert snapshot.retryable is False


def test_explicit_live_hydration_flush_commits_the_current_batch(tmp_path: Path) -> None:
    path = tmp_path / "model-content.jsonl"
    store = ModelContentStore(path, run_id="run-1", batch_interval_s=60.0)
    writer = store.open(_context())
    writer.push(ModelStreamDelta("output", "persisted"))
    writer.push(ModelStreamDelta("output", " buffered tail"))

    assert read_model_content(path).snapshots[0].output_text == "persisted"
    assert active_model_content_stream_ids(tmp_path) == frozenset({"stream-1"})
    assert flush_active_model_content(tmp_path) == 1
    assert read_model_content(path).snapshots[0].output_text == "persisted buffered tail"

    store.close()
    assert active_model_content_stream_ids(path) == frozenset()
    assert flush_active_model_content(path) == 0


def test_open_publishes_an_active_writer_only_after_its_descriptor_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model-content.jsonl"
    store = ModelContentStore(path, run_id="run-1")
    append_entered = threading.Event()
    continue_append = threading.Event()
    state_started = threading.Event()
    state_done = threading.Event()
    opened: list[object] = []
    states: list[ActiveModelContentState | None] = []
    original_append = store._append

    def blocked_append(payload: dict[str, Any]) -> bool:
        append_entered.set()
        assert continue_append.wait(2)
        return original_append(payload)

    monkeypatch.setattr(store, "_append", blocked_append)

    def open_writer() -> None:
        opened.append(store.open(_context()))

    def inspect_state() -> None:
        state_started.set()
        states.append(store.active_state())
        state_done.set()

    open_thread = threading.Thread(target=open_writer)
    open_thread.start()
    assert append_entered.wait(2)
    state_thread = threading.Thread(target=inspect_state)
    state_thread.start()
    assert state_started.wait(2)
    assert not state_done.wait(0.05)

    continue_append.set()
    open_thread.join(2)
    state_thread.join(2)
    assert not open_thread.is_alive()
    assert not state_thread.is_alive()
    assert len(opened) == 1
    assert len(states) == 1
    state = states[0]
    assert state is not None
    assert state.stream_ids == frozenset({"stream-1"})
    assert state.file_identity is not None
    assert not store._disabled
    store.close()


def test_closing_writer_remains_active_until_terminal_append_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ModelContentStore(tmp_path / "model-content.jsonl", run_id="run-1")
    writer = store.open(_context())
    writer.push(ModelStreamDelta("output", "prefix"))
    close_entered = threading.Event()
    continue_close = threading.Event()
    original_append = store._append

    def blocked_terminal_append(payload: dict[str, Any]) -> bool:
        if payload.get("kind") == "stream_closed":
            close_entered.set()
            assert continue_close.wait(2)
        return original_append(payload)

    monkeypatch.setattr(store, "_append", blocked_terminal_append)
    close_thread = threading.Thread(
        target=writer.close,
        args=(ModelStreamOutcome("completed", final_text="prefix"),),
    )
    close_thread.start()
    assert close_entered.wait(2)
    try:
        state = active_model_content_state(tmp_path)
        assert state.stream_ids == frozenset({"stream-1"})
    finally:
        continue_close.set()
        close_thread.join(2)
        store.close()
    assert not close_thread.is_alive()


def test_path_mutation_watch_detects_a_complete_store_writer_lifecycle(
    tmp_path: Path,
) -> None:
    with watch_active_model_content(tmp_path) as watch:
        assert not watch.changed
        store = ModelContentStore(tmp_path / "model-content.jsonl", run_id="run-1")
        writer = store.open(_context())
        writer.push(ModelStreamDelta("output", "short-lived prefix"))
        writer.close(ModelStreamOutcome("completed", final_text="short-lived prefix"))
        store.close()
        assert watch.changed

    state = active_model_content_state(tmp_path)
    assert state.store_count == 0
    assert state.stream_ids == frozenset()


def test_path_mutation_watch_ignores_unrelated_store_lifecycle(tmp_path: Path) -> None:
    watched_dir = tmp_path / "watched"
    other_dir = tmp_path / "other"

    with watch_active_model_content(watched_dir) as watch:
        store = ModelContentStore(other_dir / "model-content.jsonl", run_id="run-2")
        writer = store.open(_context(run_id="run-2"))
        writer.close(ModelStreamOutcome("completed", final_text="done"))
        store.close()
        assert not watch.changed


def test_active_store_registry_does_not_retain_an_unclosed_writer_cycle(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model-content.jsonl"
    store = ModelContentStore(path, run_id="run-1", batch_interval_s=60.0)
    writer = store.open(_context())
    store_reference = weakref.ref(store)

    assert flush_active_model_content(path) == 1

    # The process-local lookup must not become an owner. A recorder lost during exceptional
    # teardown can otherwise leave its store/writer cycle alive and every later hydration would
    # keep flushing a stale handle for the lifetime of Studio.
    del writer
    del store
    gc.collect()

    assert store_reference() is None
    assert flush_active_model_content(path) == 0


def test_live_hydration_flush_fails_closed_when_an_active_store_is_disabled(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model-content.jsonl"
    store = ModelContentStore(path, run_id="run-1", batch_interval_s=60.0)
    writer = store.open(_context())
    writer.push(ModelStreamDelta("output", "persisted"))
    writer.push(ModelStreamDelta("output", " buffered tail"))
    store._disabled = True

    with pytest.raises(OSError, match="failed to flush 1 active model-content store"):
        flush_active_model_content(path)

    # A failed explicit flush never claims that the buffered tail reached the sidecar.
    assert read_model_content(path).snapshots[0].output_text == "persisted"
    store.close()


def test_live_hydration_flush_rejects_regular_path_replacement(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model-content.jsonl"
    store = ModelContentStore(path, run_id="run-1", batch_interval_s=60.0)
    writer = store.open(_context())
    writer.push(ModelStreamDelta("output", "persisted"))
    writer.push(ModelStreamDelta("output", " buffered tail"))
    displaced = tmp_path / "displaced-model-content.jsonl"
    try:
        path.replace(displaced)
        path.write_text("replacement must remain untouched\n", encoding="utf-8")
    except OSError as exc:
        store.close()
        pytest.skip(f"open-file replacement is unavailable: {exc}")

    assert active_model_content_stream_ids(path) == frozenset()
    with pytest.raises(OSError, match="failed to flush 1 active model-content store"):
        flush_active_model_content(path)

    assert path.read_text(encoding="utf-8") == "replacement must remain untouched\n"
    store.close()


def test_identity_bound_reader_rejects_a_regular_file_replacement(tmp_path: Path) -> None:
    path = tmp_path / "model-content.jsonl"
    store = ModelContentStore(path, run_id="run-1")
    writer = store.open(_context())
    writer.push(ModelStreamDelta("output", "trusted prefix"))
    writer.close(ModelStreamOutcome("completed", final_text="trusted prefix"))
    store.close()
    expected_identity = model_content_file_identity(path, allow_missing=False)
    assert expected_identity is not None

    displaced = tmp_path / "trusted-model-content.jsonl"
    path.replace(displaced)
    path.write_bytes(displaced.read_bytes())

    with pytest.raises(OSError, match="identity changed before read"):
        read_model_content(path, expected_identity=expected_identity)


def test_oversized_live_gap_flushes_the_exact_sidecar_prefix_before_resume(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model-content.jsonl"
    store = ModelContentStore(path, run_id="run-1", batch_interval_s=60.0)
    broker = LiveModelStreamBroker(generation="oversized-flush", max_bytes=700)
    context = _context()
    sidecar_writer = store.open(context)
    live_writer = broker.observer("run-1").open(context)

    for text in ("a", "x" * 5_001):
        delta = ModelStreamDelta("output", text)
        # AgentLoop deliberately orders the private sidecar before the passive live observer.
        sidecar_writer.push(delta)
        live_writer.push(delta)

    reset = broker.subscribe("run-1", after_cursor="oversized-flush:2").poll()
    assert len(reset) == 1
    assert reset[0].kind == "reset"
    assert reset[0].cursor.sequence == 3
    assert len(read_model_content(path).snapshots[0].output_text) < 5_002

    assert flush_active_model_content(path) == 1
    assert read_model_content(path).snapshots[0].output_text == "a" + "x" * 5_001
    store.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows path comparison is case-insensitive")
def test_active_store_registry_uses_windows_case_insensitive_file_keys(tmp_path: Path) -> None:
    path = tmp_path / "model-content.jsonl"
    store = ModelContentStore(path, run_id="run-1", batch_interval_s=60.0)
    writer = store.open(_context())
    writer.push(ModelStreamDelta("output", "persisted"))
    writer.push(ModelStreamDelta("output", " tail"))

    assert flush_active_model_content(path.with_name("MODEL-CONTENT.JSONL")) == 1
    assert read_model_content(path).snapshots[0].output_text == "persisted tail"
    store.close()


def test_model_content_reader_writer_and_registry_reject_a_planted_file_symlink(
    tmp_path: Path,
) -> None:
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    target = outside_dir / "model-content.jsonl"
    source = ModelContentStore(target, run_id="outside-run")
    source_writer = source.open(_context(run_id="outside-run"))
    source_writer.push(ModelStreamDelta("output", "outside private text"))
    source_writer.close(ModelStreamOutcome("completed", final_text="outside private text"))
    source.close()
    original = target.read_bytes()

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    link = run_dir / "model-content.jsonl"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")

    assert read_model_content(run_dir).snapshots == ()

    planted = ModelContentStore(link, run_id="run-1")
    planted_writer = planted.open(_context())
    planted_writer.push(ModelStreamDelta("output", "must not append outside"))
    with pytest.raises(OSError, match="failed to flush 1 active model-content store"):
        flush_active_model_content(run_dir)
    planted.close()

    assert target.read_bytes() == original


def test_model_content_reader_writer_and_registry_reject_a_planted_hardlink(
    tmp_path: Path,
) -> None:
    outside_dir = tmp_path / "outside-hardlink"
    outside_dir.mkdir()
    target = outside_dir / "model-content.jsonl"
    source = ModelContentStore(target, run_id="outside-run")
    source_writer = source.open(_context(run_id="outside-run"))
    source_writer.push(ModelStreamDelta("output", "outside hardlinked text"))
    source_writer.close(ModelStreamOutcome("completed", final_text="outside hardlinked text"))
    source.close()
    original = target.read_bytes()

    run_dir = tmp_path / "run-hardlink"
    run_dir.mkdir()
    link = run_dir / "model-content.jsonl"
    try:
        os.link(target, link)
    except OSError as exc:
        pytest.skip(f"hard links are unavailable: {exc}")

    assert read_model_content(run_dir).snapshots == ()

    planted = ModelContentStore(link, run_id="run-hardlink")
    planted_writer = planted.open(_context(run_id="run-hardlink"))
    planted_writer.push(ModelStreamDelta("output", "must not mutate hardlink target"))
    with pytest.raises(OSError, match="failed to flush 1 active model-content store"):
        flush_active_model_content(run_dir)
    planted.close()

    assert target.read_bytes() == original


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


def test_reserved_sidecar_basename_can_be_a_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / MODEL_CONTENT_FILENAME
    run_dir.mkdir()
    path = run_dir / MODEL_CONTENT_FILENAME
    with watch_active_model_content(run_dir) as watch:
        store = ModelContentStore(
            path,
            run_id=MODEL_CONTENT_FILENAME,
            batch_interval_s=60.0,
        )
        writer = store.open(_context(run_id=MODEL_CONTENT_FILENAME))
        writer.push(ModelStreamDelta("output", "persisted"))
        writer.push(ModelStreamDelta("output", " buffered tail"))

        assert active_model_content_stream_ids(run_dir) == frozenset({"stream-1"})
        assert flush_active_model_content(run_dir) == 1
        assert model_content_file_identity(run_dir, allow_missing=False) == (
            model_content_file_identity(path, allow_missing=False)
        )
        writer.close(ModelStreamOutcome("completed", final_text="persisted buffered tail"))
        store.close()
        assert watch.changed

    result = read_model_content(run_dir)

    assert result.snapshots[0].best_output_text == "persisted buffered tail"


def test_reserved_sidecar_basename_directory_symlink_is_not_followed(tmp_path: Path) -> None:
    outside_dir = tmp_path / "outside-run"
    outside_path = outside_dir / MODEL_CONTENT_FILENAME
    store = ModelContentStore(outside_path, run_id="outside-run")
    writer = store.open(_context(run_id="outside-run"))
    writer.push(ModelStreamDelta("output", "outside private text"))
    writer.close(ModelStreamOutcome("completed", final_text="outside private text"))
    store.close()

    link = tmp_path / MODEL_CONTENT_FILENAME
    try:
        link.symlink_to(outside_dir, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    assert read_model_content(link).snapshots == ()
    assert flush_active_model_content(link) == 0


def test_reserved_sidecar_basename_windows_junction_is_not_treated_as_run_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DirectoryReparseMetadata:
        st_mode = stat.S_IFDIR
        st_file_attributes = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)

    path = tmp_path / MODEL_CONTENT_FILENAME
    monkeypatch.setattr(Path, "lstat", lambda _path: DirectoryReparseMetadata())

    assert _model_content_file_path(path) == path


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


@pytest.mark.parametrize(
    ("missing_index", "expected_output", "expected_segment_count", "expected_last_index"),
    [
        (0, "", 0, None),
        (1, "abcd", 1, 0),
    ],
)
def test_reader_stops_both_channels_at_first_missing_segment(
    tmp_path: Path,
    missing_index: int,
    expected_output: str,
    expected_segment_count: int,
    expected_last_index: int | None,
) -> None:
    path = tmp_path / "model-content.jsonl"
    store = ModelContentStore(path, run_id="run-1", max_segment_bytes=4)
    writer = store.open(_context())
    writer.push(ModelStreamDelta("output", "abcd"))
    writer.push(ModelStreamDelta("reasoning", "wxyz"))
    writer.push(ModelStreamDelta("output", "ijkl"))
    writer.close(ModelStreamOutcome("cancelled", error_code="user_cancelled"))
    store.close()

    records = _records(path)
    missing = next(
        record
        for record in records
        if record.get("kind") == "stream_segment"
        and record.get("segment_index") == missing_index
    )
    missing["text_len"] += 1
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    result = read_model_content(path)

    assert result.skipped_records == 1
    snapshot = result.snapshots[0]
    assert snapshot.status == "cancelled"
    assert snapshot.error_code == "user_cancelled"
    assert snapshot.output_text == expected_output
    assert snapshot.reasoning_text == ""
    assert snapshot.segment_count == expected_segment_count
    assert snapshot.last_segment_index == expected_last_index


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
