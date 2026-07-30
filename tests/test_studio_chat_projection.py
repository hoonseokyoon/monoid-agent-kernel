from __future__ import annotations

import json
from pathlib import Path

from monoid_agent_kernel.reference.studio.chat_projection import (
    CHAT_SCHEMA_V1,
    CHAT_SCHEMA_V2,
    ChatProjection,
    is_supported_chat_response,
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
