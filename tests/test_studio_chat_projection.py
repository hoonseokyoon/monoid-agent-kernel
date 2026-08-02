from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

from monoid_agent_kernel.core.model_content import ModelContentStore
from monoid_agent_kernel.core.model_io import content_digest, content_length
from monoid_agent_kernel.core.model_stream import (
    ModelStreamContext,
    ModelStreamDelta,
    ModelStreamOutcome,
)
from monoid_agent_kernel.reference.studio.chat_projection import (
    CHAT_SCHEMA_V1,
    CHAT_SCHEMA_V2,
    ChatProjection,
    is_supported_chat_response,
)


def _write_terminal_model_content(
    run_dir: Path,
    *,
    status: Literal["completed", "interrupted", "failed"],
    root_run_id: str = "run-1",
    run_id: str | None = None,
    turn_id: str = "turn_0001",
    stream_id: str = "stream-partial",
    text: str = "partial answer",
    retryable: bool = False,
    step: int = 1,
    started_at: str = "2026-08-01T00:00:00Z",
) -> None:
    resolved_run_id = run_id or root_run_id
    store = ModelContentStore(run_dir / "model-content.jsonl", run_id=resolved_run_id)
    writer = store.open(
        ModelStreamContext(
            run_id=resolved_run_id,
            root_run_id=root_run_id,
            turn_id=turn_id,
            stream_id=stream_id,
            step=step,
            provider="test",
            model="test-model",
            started_at=started_at,
        )
    )
    writer.push(ModelStreamDelta(channel="output", text=text))
    writer.close(
        ModelStreamOutcome(status=status, final_text=text, retryable=retryable)
    )
    store.close()


def _write_interrupted_model_content(
    run_dir: Path,
    *,
    root_run_id: str = "run-1",
    run_id: str | None = None,
    turn_id: str = "turn_0001",
    stream_id: str = "stream-partial",
    text: str = "partial answer",
) -> None:
    _write_terminal_model_content(
        run_dir,
        status="interrupted",
        root_run_id=root_run_id,
        run_id=run_id,
        turn_id=turn_id,
        stream_id=stream_id,
        text=text,
    )


def test_chat_response_reader_distinguishes_the_v1_and_v2_shapes() -> None:
    base = {"run_id": "run-1", "messages": [], "event_cursor": -1}

    assert is_supported_chat_response({"schema_version": CHAT_SCHEMA_V1, **base})
    assert not is_supported_chat_response(
        {"schema_version": CHAT_SCHEMA_V1, **base, "event_log_error": ""}
    )
    assert is_supported_chat_response(
        {"schema_version": CHAT_SCHEMA_V2, **base, "event_log_error": ""}
    )
    assert not is_supported_chat_response({"schema_version": CHAT_SCHEMA_V2, **base})
    assert not is_supported_chat_response(
        {"schema_version": CHAT_SCHEMA_V2, **base, "event_log_error": None}
    )
    assert not is_supported_chat_response(
        {"schema_version": "studio.chat.v3", **base, "event_log_error": ""}
    )


def test_chat_response_reader_validates_required_message_fields_but_allows_extensions() -> None:
    message = {
        "id": "message-1",
        "role": "assistant",
        "content": "done",
        "attachments": [
            {"name": "result.txt", "mime": "text/plain", "future_attachment_member": True}
        ],
        "created_at": 1.25,
        "source": {"kind": "event", "future_source_member": True},
        "schema_version": "studio.chat.message.v999",
        "future_message_member": True,
    }

    def response(messages: list[object]) -> dict[str, object]:
        return {
            "schema_version": CHAT_SCHEMA_V2,
            "run_id": "run-1",
            "messages": messages,
            "event_cursor": -1,
            "event_log_error": "",
        }

    legacy_message = dict(message)
    legacy_message.pop("schema_version")
    legacy_message.pop("source")
    assert is_supported_chat_response(response([]))
    assert is_supported_chat_response(response([message]))
    assert is_supported_chat_response(response([legacy_message]))
    assert is_supported_chat_response(
        response(
            [
                {**legacy_message, "id": "message-user", "role": "user"},
                {**legacy_message, "id": "message-assistant", "role": "assistant"},
                {**legacy_message, "id": "message-error", "role": "error"},
            ]
        )
    )

    def without_required_field(field: str) -> dict[str, object]:
        candidate = dict(legacy_message)
        candidate.pop(field)
        return candidate

    for malformed in (
        None,
        [],
        {"legacy_or_future_member": True},
        without_required_field("id"),
        without_required_field("role"),
        without_required_field("content"),
        without_required_field("attachments"),
        without_required_field("created_at"),
        {**legacy_message, "id": ""},
        {**legacy_message, "id": "   "},
        {**message, "id": 7},
        {**message, "role": "system"},
        {**message, "role": 7},
        {**message, "role": []},
        {**message, "content": None},
        {**message, "attachments": {}},
        {**message, "attachments": [None]},
        {**message, "attachments": [[]]},
        {**message, "attachments": [{"mime": "text/plain"}]},
        {**message, "attachments": [{"name": 7, "mime": "text/plain"}]},
        {**message, "attachments": [{"name": "result.txt"}]},
        {**message, "attachments": [{"name": "result.txt", "mime": 7}]},
        {**message, "created_at": "1"},
        {**message, "created_at": True},
        {**message, "created_at": float("nan")},
        {**message, "created_at": float("inf")},
        {**message, "created_at": float("-inf")},
        {**message, "created_at": 10**1000},
        {**message, "source": None},
        {**message, "source": []},
        {**message, "source": "event"},
    ):
        assert not is_supported_chat_response(response([malformed]))

    assert not is_supported_chat_response(response([legacy_message, dict(legacy_message)]))


def test_chat_projection_dedupes_user_messages_and_strips_attachment_bytes(tmp_path: Path) -> None:
    projection = ChatProjection(tmp_path)

    projection.append_user(
        content="see attached",
        attachments=(
            {
                "name": "diagram.png",
                "mime": "image/png",
                "data_b64": "not persisted",
            },
        ),
        client_message_id="client-1",
    )
    projection.append_user(content="duplicate", client_message_id="client-1")

    records = projection.read()
    assert len(records) == 1
    assert records[0]["content"] == "see attached"
    assert records[0]["source"]["client_message_id"] == "client-1"
    assert records[0]["attachments"] == [{"name": "diagram.png", "mime": "image/png"}]
    assert "not persisted" not in (tmp_path / "studio.chat.jsonl").read_text(encoding="utf-8")


def test_chat_projection_projects_assistant_and_non_retryable_errors_once(tmp_path: Path) -> None:
    projection = ChatProjection(tmp_path)
    settled = {
        "type": "turn.settled",
        "event_id": "evt-final",
        "seq": 3,
        "timestamp": "2026-07-08T00:00:00Z",
        "data": {"final_text": "done"},
    }
    retryable = {
        "type": "turn.failed",
        "event_id": "evt-retry",
        "seq": 4,
        "data": {"error": "temporary", "retryable": True},
    }
    failed = {
        "type": "turn.failed",
        "event_id": "evt-failed",
        "seq": 5,
        "data": {
            "error": "unsupported effort",
            "retryable": False,
            "provider_error_code": "bad_request",
            "http_status": 400,
        },
    }

    projection.project_events([settled, retryable, failed, settled, failed])

    records = projection.read()
    assert [(record["role"], record["content"]) for record in records] == [
        ("assistant", "done"),
        ("error", "unsupported effort - bad_request · HTTP 400"),
    ]
    assert records[1]["source"]["event_type"] == "turn.failed"
    assert projection.event_cursor() == 5


def test_chat_projection_hydrates_interrupted_partial_from_matching_private_stream(
    tmp_path: Path,
) -> None:
    _write_interrupted_model_content(tmp_path)
    interrupted = {
        "type": "turn.interrupted",
        "run_id": "run-1",
        "turn_id": "turn_0001",
        "event_id": "evt-interrupted",
        "seq": 4,
        "timestamp": "2026-08-01T00:00:01Z",
        "data": {"reason": "user_stop"},
    }
    (tmp_path / "events.jsonl").write_text(
        json.dumps(interrupted) + "\n",
        encoding="utf-8",
    )
    projection = ChatProjection(tmp_path)

    first = projection.catch_up("run-1")
    second = projection.catch_up("run-1")
    hidden = projection.catch_up("run-1", include_model_stream_partials=False)

    assert first == second
    assert hidden["messages"] == []
    assert hidden["event_cursor"] == -1
    assert is_supported_chat_response(first)
    assert first["event_cursor"] == 4
    assert len(first["messages"]) == 1
    partial = first["messages"][0]
    assert partial["id"] == "assistant:model-stream:stream-partial:partial"
    assert partial["role"] == "assistant"
    assert partial["content"] == "partial answer"
    assert partial["source"] == {
        "kind": "model_stream_partial",
        "event_type": "turn.interrupted",
        "event_id": "evt-interrupted",
        "seq": 4,
        "root_run_id": "run-1",
        "run_id": "run-1",
        "turn_id": "turn_0001",
        "stream_id": "stream-partial",
        "status": "interrupted",
        "partial": True,
        "retryable": False,
    }


def test_chat_projection_preserves_non_retryable_failed_partial_after_a_later_turn(
    tmp_path: Path,
) -> None:
    _write_terminal_model_content(
        tmp_path,
        status="failed",
        turn_id="turn_0001",
        stream_id="stream-failed",
        text="useful failed prefix",
    )
    _write_terminal_model_content(
        tmp_path,
        status="completed",
        turn_id="turn_0002",
        stream_id="stream-completed",
        text="later answer",
        step=2,
        started_at="2026-08-01T00:00:03Z",
    )
    events = [
        {
            "type": "turn.failed",
            "run_id": "run-1",
            "turn_id": "turn_0001",
            "event_id": "evt-failed",
            "seq": 4,
            "timestamp": "2026-08-01T00:00:01Z",
            "data": {"error": "bad request", "retryable": False},
        },
        {
            "type": "turn.settled",
            "run_id": "run-1",
            "turn_id": "turn_0002",
            "event_id": "evt-settled",
            "seq": 8,
            "timestamp": "2026-08-01T00:00:04Z",
            "data": {"final_text": "later answer"},
        },
    ]
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    projection = ChatProjection(tmp_path)
    projection.append_user(
        content="normal next message",
        client_message_id="client-next",
        created_at=1785542402.0,
    )

    first = projection.catch_up("run-1")
    second = projection.catch_up("run-1")

    assert first == second
    assert [(message["role"], message["content"]) for message in first["messages"]] == [
        ("assistant", "useful failed prefix"),
        ("error", "bad request"),
        ("user", "normal next message"),
        ("assistant", "later answer"),
    ]
    partial = first["messages"][0]
    assert partial["source"] == {
        "kind": "model_stream_partial",
        "event_type": "turn.failed",
        "event_id": "evt-failed",
        "seq": 4,
        "root_run_id": "run-1",
        "run_id": "run-1",
        "turn_id": "turn_0001",
        "stream_id": "stream-failed",
        "status": "failed",
        "partial": True,
        "retryable": False,
    }
    assert first["event_cursor"] == 8


def test_chat_projection_backfills_failed_partial_beside_an_existing_error(
    tmp_path: Path,
) -> None:
    failed = {
        "type": "turn.failed",
        "run_id": "run-1",
        "turn_id": "turn_0001",
        "event_id": "evt-failed",
        "seq": 4,
        "timestamp": "2026-08-01T00:00:01Z",
        "data": {"error": "bad request", "retryable": False},
    }
    projection = ChatProjection(tmp_path)
    projection.project_events([failed], root_run_id="run-1")
    assert [(record["role"], record["content"]) for record in projection.read()] == [
        ("error", "bad request")
    ]

    _write_terminal_model_content(
        tmp_path,
        status="failed",
        stream_id="stream-late",
        text="late durable prefix",
    )
    projection.project_events([failed], root_run_id="run-1")
    projection.project_events([failed], root_run_id="run-1")

    assert [(record["role"], record["content"]) for record in projection.read()] == [
        ("assistant", "late durable prefix"),
        ("error", "bad request"),
    ]


@pytest.mark.parametrize("prefix_before_crash", ["prefix before crash", ""])
def test_chat_projection_fails_closed_when_a_recovered_turn_reuses_its_id(
    tmp_path: Path,
    prefix_before_crash: str,
) -> None:
    for stream_id, text, started_at in (
        ("stream-before-crash", prefix_before_crash, "2026-08-01T00:00:00Z"),
        ("stream-after-restore", "prefix after restore", "2026-08-01T00:00:02Z"),
    ):
        _write_terminal_model_content(
            tmp_path,
            status="failed",
            turn_id="turn_0001",
            stream_id=stream_id,
            text=text,
            started_at=started_at,
        )
    failed_events = [
        {
            "type": "turn.failed",
            "run_id": "run-1",
            "turn_id": "turn_0001",
            "event_id": event_id,
            "seq": seq,
            "timestamp": timestamp,
            "data": {"error": error, "retryable": False},
        }
        for event_id, seq, timestamp, error in (
            ("evt-before-crash", 4, "2026-08-01T00:00:01Z", "first failure"),
            ("evt-after-restore", 8, "2026-08-01T00:00:03Z", "second failure"),
        )
    ]

    projection = ChatProjection(tmp_path)
    projection.project_events(failed_events, root_run_id="run-1")
    projection.project_events(failed_events, root_run_id="run-1")

    assert [(record["role"], record["content"]) for record in projection.read()] == [
        ("error", "first failure"),
        ("error", "second failure"),
    ]
    persisted = projection.path.read_text(encoding="utf-8")
    assert "prefix before crash" not in persisted
    assert "prefix after restore" not in persisted


def test_chat_projection_omits_retryable_and_explicitly_retried_failed_partials(
    tmp_path: Path,
) -> None:
    _write_terminal_model_content(
        tmp_path,
        status="failed",
        turn_id="turn_retryable",
        stream_id="stream-retryable",
        text="automatic retry prefix",
        retryable=True,
    )
    _write_terminal_model_content(
        tmp_path,
        status="failed",
        turn_id="turn_manual",
        stream_id="stream-manual",
        text="manual retry prefix",
    )
    events = [
        {
            "type": "turn.failed",
            "run_id": "run-1",
            "turn_id": "turn_retryable",
            "event_id": "evt-retryable",
            "seq": 2,
            "timestamp": "2026-08-01T00:00:01Z",
            "data": {"error": "temporary", "retryable": True},
        },
        {
            "type": "turn.failed",
            "run_id": "run-1",
            "turn_id": "turn_manual",
            "event_id": "evt-manual",
            "seq": 4,
            "timestamp": "2026-08-01T00:00:02Z",
            "data": {"error": "bad config", "retryable": False},
        },
        {
            "type": "run.resumed",
            "run_id": "run-1",
            "turn_id": "turn_manual",
            "event_id": "evt-resumed",
            "seq": 5,
            "timestamp": "2026-08-01T00:00:03Z",
            "data": {"reason": "studio-retry"},
        },
    ]
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    response = ChatProjection(tmp_path).catch_up("run-1")

    assert [(message["role"], message["content"]) for message in response["messages"]] == [
        ("error", "bad config")
    ]
    persisted = (tmp_path / "studio.chat.jsonl").read_text(encoding="utf-8")
    assert "manual retry prefix" in persisted
    assert "automatic retry prefix" not in persisted


def test_chat_projection_hydrates_completed_digest_from_model_content(
    tmp_path: Path,
) -> None:
    text = "completed answer retained outside the operation log"
    digest = content_digest(text)
    store = ModelContentStore(tmp_path / "model-content.jsonl", run_id="run-1")
    store.settled_text(text, digest, content_length(text))
    store.close()
    (tmp_path / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "turn.settled",
                "run_id": "run-1",
                "turn_id": "turn_completed",
                "event_id": "evt-completed",
                "seq": 5,
                "timestamp": "2026-08-01T00:00:01Z",
                "data": {
                    "status": "completed",
                    "final_text_digest": digest,
                    "final_text_len": content_length(text),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    response = ChatProjection(tmp_path).catch_up("run-1")

    assert response["event_cursor"] == 5
    assert [(message["role"], message["content"]) for message in response["messages"]] == [
        ("assistant", text)
    ]


def test_chat_projection_returns_active_sidecar_prefix_without_persisting_it(
    tmp_path: Path,
) -> None:
    store = ModelContentStore(tmp_path / "model-content.jsonl", run_id="run-1")
    older_writer = store.open(
        ModelStreamContext(
            run_id="run-1",
            root_run_id="run-1",
            turn_id="turn_0002",
            stream_id="stream-active-old",
            step=1,
            provider="test",
            model="test-model",
            started_at="2026-07-31T23:59:59Z",
        )
    )
    older_writer.push(ModelStreamDelta(channel="output", text="stale retry prefix"))
    context = ModelStreamContext(
        run_id="run-1",
        root_run_id="run-1",
        turn_id="turn_0002",
        stream_id="stream-active",
        step=2,
        provider="test",
        model="test-model",
        started_at="2026-08-01T00:00:00Z",
    )
    writer = store.open(context)
    writer.push(ModelStreamDelta(channel="output", text="durable prefix"))
    started = {
        "type": "model.turn.started",
        "run_id": "run-1",
        "turn_id": "turn_0002",
        "event_id": "evt-started",
        "seq": 8,
        "timestamp": "2026-08-01T00:00:01Z",
        "data": {"step": 2},
    }
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(json.dumps(started) + "\n", encoding="utf-8")
    projection = ChatProjection(tmp_path)
    try:
        body = projection.catch_up("run-1")
        hidden = projection.catch_up("run-1", include_model_stream_partials=False)

        assert is_supported_chat_response(body)
        assert hidden["messages"] == []
        assert len(body["messages"]) == 1
        active = body["messages"][0]
        assert active["id"] == "assistant:model-stream:stream-active:active"
        assert active["content"] == "durable prefix"
        assert active["source"]["kind"] == "model_stream_active"
        assert active["source"]["stream_id"] == "stream-active"
        # Response-only: catch-up may refresh this growing prefix, so it cannot be frozen into the
        # append-only Studio chat sidecar.
        assert not (tmp_path / "studio.chat.jsonl").exists()
        assert "stale retry prefix" not in json.dumps(body)

        writer.close(ModelStreamOutcome(status="interrupted", final_text="durable prefix"))
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "type": "turn.interrupted",
                        "run_id": "run-1",
                        "turn_id": "turn_0002",
                        "event_id": "evt-interrupted",
                        "seq": 9,
                        "timestamp": "2026-08-01T00:00:02Z",
                        "data": {"reason": "user_stop"},
                    }
                )
                + "\n"
            )
        settled = projection.catch_up("run-1")
        assert [message["id"] for message in settled["messages"]] == [
            "assistant:model-stream:stream-active:partial"
        ]
    finally:
        store.close()


def test_chat_projection_does_not_revive_a_stream_closed_during_sidecar_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ModelContentStore(tmp_path / "model-content.jsonl", run_id="run-1")
    context = ModelStreamContext(
        run_id="run-1",
        root_run_id="run-1",
        turn_id="turn_0002",
        stream_id="stream-closing",
        step=1,
        provider="test",
        model="test-model",
        started_at="2026-08-01T00:00:00Z",
    )
    writer = store.open(context)
    writer.push(ModelStreamDelta(channel="output", text="prefix before close"))
    started = {
        "type": "model.turn.started",
        "run_id": "run-1",
        "turn_id": context.turn_id,
        "event_id": "evt-started",
        "seq": 8,
        "timestamp": "2026-08-01T00:00:01Z",
        "data": {"step": 1},
    }
    (tmp_path / "events.jsonl").write_text(json.dumps(started) + "\n", encoding="utf-8")

    from monoid_agent_kernel.reference.studio import chat_projection as projection_module

    original_read = projection_module.read_model_content
    closed = False

    def close_after_read(path: Path, **kwargs):  # noqa: ANN003, ANN202
        nonlocal closed
        result = original_read(path, **kwargs)
        if not closed:
            closed = True
            writer.close(ModelStreamOutcome(status="interrupted", final_text="prefix before close"))
        return result

    monkeypatch.setattr(projection_module, "read_model_content", close_after_read)
    try:
        body = ChatProjection(tmp_path).catch_up("run-1")
        assert body["messages"] == []
    finally:
        store.close()


def test_chat_projection_does_not_revive_a_process_lost_sidecar_prefix(
    tmp_path: Path,
) -> None:
    store = ModelContentStore(tmp_path / "model-content.jsonl", run_id="run-1")
    writer = store.open(
        ModelStreamContext(
            run_id="run-1",
            root_run_id="run-1",
            turn_id="turn_crashed",
            stream_id="stream-crashed",
            step=1,
            provider="test",
            model="test-model",
            started_at="2026-08-01T00:00:00Z",
        )
    )
    writer.push(ModelStreamDelta(channel="output", text="stale process prefix"))
    (tmp_path / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "model.turn.started",
                "run_id": "run-1",
                "turn_id": "turn_crashed",
                "event_id": "evt-crashed-start",
                "seq": 1,
                "timestamp": "2026-08-01T00:00:01Z",
                "data": {"step": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # Process teardown flushes the prefix without inventing a terminal outcome. A later Studio
    # process sees the same start-only durable shape but has no process-local writer proof.
    store.close()

    response = ChatProjection(tmp_path).catch_up("run-1")

    assert response["messages"] == []


def test_chat_projection_does_not_publish_active_prefix_from_corrupt_event_log(
    tmp_path: Path,
) -> None:
    store = ModelContentStore(tmp_path / "model-content.jsonl", run_id="run-1")
    writer = store.open(
        ModelStreamContext(
            run_id="run-1",
            root_run_id="run-1",
            turn_id="turn_corrupt",
            stream_id="stream-corrupt",
            step=1,
            provider="test",
            model="test-model",
            started_at="2026-08-01T00:00:00Z",
        )
    )
    writer.push(ModelStreamDelta(channel="output", text="untrusted active prefix"))
    (tmp_path / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "model.turn.started",
                "run_id": "run-1",
                "turn_id": "turn_corrupt",
                "event_id": "evt-corrupt-start",
                "seq": 1,
                "data": {"step": 1},
            }
        )
        + "\n{not-json}\n",
        encoding="utf-8",
    )
    try:
        response = ChatProjection(tmp_path).catch_up("run-1")

        assert response["messages"] == []
        assert response["event_log_error"]
    finally:
        store.close()


def test_chat_projection_does_not_invent_partial_without_matching_private_stream(
    tmp_path: Path,
) -> None:
    _write_interrupted_model_content(
        tmp_path,
        root_run_id="run-1",
        run_id="run-1.sub.foreign",
    )
    events = [
        {
            "type": "turn.interrupted",
            "run_id": "run-1",
            "turn_id": "turn_0001",
            "event_id": "evt-mismatch",
            "seq": 1,
            "data": {"reason": "user_stop"},
        }
    ]
    projection = ChatProjection(tmp_path)

    projection.project_events(events)
    assert projection.read() == []

    (tmp_path / "model-content.jsonl").write_bytes(b'{"torn":"\xff')
    projection.project_events(events)
    assert projection.read() == []


def test_chat_projection_rejects_foreign_event_and_sidecar_context_pair(
    tmp_path: Path,
) -> None:
    """Two mutually consistent misplaced artifacts still cannot cross the requested root."""

    _write_interrupted_model_content(
        tmp_path,
        root_run_id="foreign-run",
        run_id="foreign-run",
        turn_id="turn_foreign",
        text="foreign private prefix",
    )
    (tmp_path / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "turn.interrupted",
                "run_id": "foreign-run",
                "turn_id": "turn_foreign",
                "event_id": "evt-foreign",
                "seq": 1,
                "data": {"reason": "user_stop"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    response = ChatProjection(tmp_path).catch_up("run-1")

    assert response["messages"] == []
    assert response["event_cursor"] == -1
    assert "foreign private prefix" not in json.dumps(response)


def test_chat_projection_orders_catchup_events_between_existing_user_turns(tmp_path: Path) -> None:
    projection = ChatProjection(tmp_path)
    projection.append_user(content="first", client_message_id="client-1", created_at=10.0)
    projection.append_user(content="second", client_message_id="client-2", created_at=30.0)

    projection.project_events(
        [
            {
                "type": "turn.settled",
                "event_id": "evt-first",
                "seq": 3,
                "timestamp": "1970-01-01T00:00:20Z",
                "data": {"final_text": "first answer"},
            },
            {
                "type": "turn.settled",
                "event_id": "evt-second",
                "seq": 7,
                "timestamp": "1970-01-01T00:00:40Z",
                "data": {"final_text": "second answer"},
            },
        ]
    )

    assert [(record["role"], record["content"]) for record in projection.read()] == [
        ("user", "first"),
        ("assistant", "first answer"),
        ("user", "second"),
        ("assistant", "second answer"),
    ]


def test_chat_projection_backfills_legacy_title_and_events(tmp_path: Path) -> None:
    (tmp_path / "run.json").write_text(
        json.dumps({"title": "legacy prompt", "created_at": 123.0}),
        encoding="utf-8",
    )
    (tmp_path / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "turn.settled",
                "event_id": "evt-final",
                "seq": 7,
                "timestamp": "2026-07-08T00:00:00Z",
                "data": {"final_text": "legacy answer"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    body = ChatProjection(tmp_path).catch_up("run-1")

    assert body["schema_version"] == "studio.chat.v2"
    assert body["event_cursor"] == 7
    assert [(message["role"], message["content"]) for message in body["messages"]] == [
        ("user", "legacy prompt"),
        ("assistant", "legacy answer"),
    ]
    assert body["messages"][0]["source"]["legacy"] is True


def test_chat_projection_withholds_uncommitted_event_tail(tmp_path: Path) -> None:
    event = {
        "type": "turn.settled",
        "event_id": "evt-partial",
        "seq": 1,
        "timestamp": "2026-07-08T00:00:00Z",
        "data": {"final_text": "withheld answer"},
    }
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(json.dumps(event), encoding="utf-8")
    projection = ChatProjection(tmp_path)

    before_commit = projection.catch_up("run-1")
    assert not (tmp_path / "studio.chat.jsonl").exists()
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    after_commit = projection.catch_up("run-1")

    assert before_commit["messages"] == []
    assert after_commit["messages"][0]["content"] == "withheld answer"


def test_chat_projection_backfills_legacy_title_after_event_only_projection(tmp_path: Path) -> None:
    (tmp_path / "run.json").write_text(
        json.dumps({"title": "legacy prompt", "created_at": 10.0}),
        encoding="utf-8",
    )
    projection = ChatProjection(tmp_path)
    projection.project_events(
        [
            {
                "type": "turn.settled",
                "event_id": "evt-final",
                "seq": 7,
                "timestamp": "1970-01-01T00:00:20Z",
                "data": {"final_text": "legacy answer"},
            }
        ]
    )

    body = projection.catch_up("run-1")

    assert [(message["role"], message["content"]) for message in body["messages"]] == [
        ("user", "legacy prompt"),
        ("assistant", "legacy answer"),
    ]
    assert body["messages"][0]["source"]["legacy"] is True
