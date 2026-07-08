from __future__ import annotations

import json
from pathlib import Path

from monoid_agent_kernel.reference.studio.chat_projection import ChatProjection


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

    assert body["schema_version"] == "studio.chat.v1"
    assert body["event_cursor"] == 7
    assert [(message["role"], message["content"]) for message in body["messages"]] == [
        ("user", "legacy prompt"),
        ("assistant", "legacy answer"),
    ]
    assert body["messages"][0]["source"]["legacy"] is True


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
