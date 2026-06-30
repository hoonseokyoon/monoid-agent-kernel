"""Lock the per-event-type ``data`` contract.

The event envelope is validated by ``EVENT_SCHEMA``; ``EVENT_DATA_SCHEMAS`` pins
the ``data`` payload for each event type so drift between the producer (emit) and
the two consumers (``recorder.StatusJsonSink`` / ``core.projections``) is caught by
``validate_run_dir``. These tests guard completeness (every declared event type has
a data schema) and that the schemas accept valid payloads / reject malformed ones.
"""

from __future__ import annotations

from typing import get_args

from jsonschema import Draft202012Validator

from monoid_agent_kernel.core.events import AgentEventType
from monoid_agent_kernel.core.schemas import EVENT_DATA_SCHEMAS


def test_every_event_type_has_a_data_schema() -> None:
    declared = set(get_args(AgentEventType))
    covered = set(EVENT_DATA_SCHEMAS)
    assert declared == covered, {
        "missing_schema": sorted(declared - covered),
        "extra_schema": sorted(covered - declared),
    }


def test_all_data_schemas_are_valid_json_schema() -> None:
    for event_type, schema in EVENT_DATA_SCHEMAS.items():
        # Raises SchemaError if the schema itself is malformed.
        Draft202012Validator.check_schema(schema)
        assert schema["type"] == "object", event_type


def _validator(event_type: str) -> Draft202012Validator:
    return Draft202012Validator(EVENT_DATA_SCHEMAS[event_type])


def test_valid_payload_passes_strict_event() -> None:
    data = {"step": 3, "previous_turn_handle": None}
    assert list(_validator("model.turn.started").iter_errors(data)) == []


def test_missing_required_key_fails() -> None:
    # `step` is required for model.turn.started.
    errors = list(_validator("model.turn.started").iter_errors({"previous_turn_handle": "h"}))
    assert errors


def test_wrong_type_fails() -> None:
    errors = list(_validator("model.turn.started").iter_errors({"step": "not-an-int"}))
    assert errors


def test_unknown_key_fails_on_strict_event() -> None:
    # tool.call.started is additionalProperties: False.
    data = {"call_id": "c1", "tool": "fs_read", "surprise": 1}
    assert list(_validator("tool.call.started").iter_errors(data))


def test_unknown_key_allowed_on_dynamic_event() -> None:
    # job.* payloads are assembled from the public job dict (additionalProperties: True).
    data = {"job_id": "job_1", "status": "running", "exit_code": None, "anything": True}
    assert list(_validator("job.started").iter_errors(data)) == []
