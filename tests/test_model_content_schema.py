from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from monoid_agent_kernel.core.model_content import ModelContentStore, read_model_content
from monoid_agent_kernel.core.model_io import content_digest, content_length
from monoid_agent_kernel.core.model_stream import (
    ModelStreamContext,
    ModelStreamDelta,
    ModelStreamOutcome,
)
from monoid_agent_kernel.core.schemas import MODEL_CONTENT_RECORD_SCHEMA, validate_run_dir


def _errors(record: dict[str, object]) -> list[str]:
    return [
        error.message
        for error in Draft202012Validator(MODEL_CONTENT_RECORD_SCHEMA).iter_errors(record)
    ]


@pytest.mark.parametrize(
    "record",
    [
        {
            "schema_version": "monoid.model-content.v1",
            "kind": "stream_opened",
            "run_id": "run-1",
            "root_run_id": "run-1",
            "turn_id": "turn-1",
            "stream_id": "stream-1",
            "step": 1,
            "provider": "openai",
            "model": "gpt-test",
            "started_at": "2026-08-01T00:00:00Z",
        },
        {
            "schema_version": "monoid.model-content.v1",
            "kind": "stream_segment",
            "run_id": "run-1",
            "stream_id": "stream-1",
            "segment_index": 0,
            "channel": "output",
            "text": "hello",
            "text_len": 5,
            "emitted_at": "2026-08-01T00:00:01Z",
        },
        {
            "schema_version": "monoid.model-content.v1",
            "kind": "stream_closed",
            "run_id": "run-1",
            "stream_id": "stream-1",
            "status": "completed",
            "final_text": "hello",
            "usage": {"output_tokens": 1},
            "error_code": None,
            "finished_at": "2026-08-01T00:00:02Z",
        },
        {
            "schema_version": "monoid.model-content.v1",
            "kind": "settled_text",
            "run_id": "run-1",
            "final_text": "hello",
            "final_text_digest": content_digest("hello"),
            "final_text_len": content_length("hello"),
            "recorded_at": "2026-08-01T00:00:03Z",
        },
    ],
)
def test_model_content_record_variants_are_schema_valid(record: dict[str, object]) -> None:
    assert _errors(record) == []


def test_model_content_schema_accepts_the_legacy_namespace() -> None:
    record = {
        "schema_version": "native-agent-runner.model-content.v1",
        "kind": "stream_segment",
        "run_id": "run-1",
        "stream_id": "stream-1",
        "segment_index": 0,
        "channel": "reasoning",
        "text": "thinking",
        "text_len": 8,
        "emitted_at": "2026-08-01T00:00:01Z",
    }

    assert _errors(record) == []


def test_model_content_schema_accepts_optional_retryability_and_rejects_non_boolean() -> None:
    record = {
        "schema_version": "monoid.model-content.v1",
        "kind": "stream_closed",
        "run_id": "run-1",
        "stream_id": "stream-1",
        "status": "failed",
        "final_text": "partial",
        "usage": None,
        "error_code": "gateway_timeout",
        "retryable": True,
        "finished_at": "2026-08-01T00:00:02Z",
    }

    assert _errors(record) == []
    assert _errors({**record, "retryable": 1})


def test_the_closed_record_carries_both_halves_of_the_classification() -> None:
    """The live stream lane used to say only whether waiting might help.

    ``additionalProperties`` is False on this branch, so the key and its schema slot had to land
    together: a writer emitting an undeclared key produces a record that fails its own schema.
    Optional, so a sidecar written before the key existed still validates.
    """
    record = {
        "schema_version": "monoid.model-content.v1",
        "kind": "stream_closed",
        "run_id": "run-1",
        "stream_id": "stream-1",
        "status": "failed",
        "final_text": "partial",
        "usage": None,
        "error_code": "model_not_found",
        "retryable": False,
        "config_recoverable": True,
        "finished_at": "2026-08-01T00:00:02Z",
    }

    assert _errors(record) == []
    assert _errors({**record, "config_recoverable": 1})
    del record["config_recoverable"]
    assert _errors(record) == []


def test_model_content_schema_rejects_unknown_fields_and_missing_versions() -> None:
    record = {
        "kind": "stream_segment",
        "run_id": "run-1",
        "stream_id": "stream-1",
        "segment_index": 0,
        "channel": "output",
        "text": "hello",
        "text_len": 5,
        "emitted_at": "2026-08-01T00:00:01Z",
        "unexpected": True,
    }

    assert _errors(record)


def test_validate_run_dir_treats_model_content_as_optional(tmp_path: Path) -> None:
    issues = validate_run_dir(tmp_path)

    assert not any(issue.path.startswith("model-content.jsonl") for issue in issues)


def test_validate_run_dir_checks_model_content_settled_text_digest(tmp_path: Path) -> None:
    record = {
        "schema_version": "monoid.model-content.v1",
        "kind": "settled_text",
        "run_id": "run-1",
        "final_text": "tampered",
        "final_text_digest": content_digest("original"),
        "final_text_len": content_length("tampered"),
        "recorded_at": "2026-08-01T00:00:03Z",
    }
    (tmp_path / "model-content.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    issues = validate_run_dir(tmp_path)

    assert any(
        issue.path == "model-content.jsonl:1"
        and issue.message == "settled_text digest does not match final_text"
        for issue in issues
    )


def test_model_content_store_writes_only_schema_valid_records(tmp_path: Path) -> None:
    path = tmp_path / "model-content.jsonl"
    store = ModelContentStore(path, run_id="run-1")
    writer = store.open(
        ModelStreamContext(
            run_id="run-1",
            root_run_id="run-1",
            turn_id="turn-1",
            stream_id="stream-1",
            step=1,
            provider="openai",
            model="gpt-test",
            started_at="2026-08-01T00:00:00Z",
        )
    )
    writer.push(ModelStreamDelta(channel="output", text="hello"))
    writer.close(
        ModelStreamOutcome(
            status="failed",
            final_text="hello",
            usage={},
            error_code="model_not_found",
            retryable=False,
            config_recoverable=True,
        )
    )
    store.settled_text("hello", content_digest("hello"), content_length("hello") or 0)
    store.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert [record["kind"] for record in records] == [
        "stream_opened",
        "stream_segment",
        "stream_closed",
        "settled_text",
    ]
    assert all(_errors(record) == [] for record in records)
    # Both halves of the classification reach the record the studio and the recovery reader see.
    closed = next(record for record in records if record["kind"] == "stream_closed")
    assert (closed["retryable"], closed["config_recoverable"]) == (False, True)
    # ...and the tolerant reader carries them back off the file.
    recovered = read_model_content(path).snapshots[0]
    assert (recovered.retryable, recovered.config_recoverable) == (False, True)
