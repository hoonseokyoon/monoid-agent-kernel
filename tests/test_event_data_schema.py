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


def test_studio_retry_resume_marker_uses_existing_v1_data_shape() -> None:
    assert list(
        _validator("run.resumed").iter_errors({"reason": "studio-retry"})
    ) == []


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


# --- field-set stability for the payloads that carry model text ------------------------------
#
# Adding a key to a strict schema is caught by nothing, and *removing* one is caught by nothing
# either: ``additionalProperties: False`` rejects unknown keys on the way in, but a property that
# quietly disappears just makes the schema more permissive about what it no longer describes.
# These four events are the ones that carry model output, so their field sets are pinned
# explicitly — a change here should be a decision, not a diff nobody reads.

_CONTENT_EVENT_FIELDS: dict[str, tuple[set[str], set[str]]] = {
    "run.finished": (
        {
            "status",
            "error",
            "error_code",
            "interruption_cause",
            "final_text",
            "final_text_digest",
            "final_text_len",
            "duration_s",
            "diff_path",
            "proposal_path",
            "metrics_path",
        },
        {"status"},
    ),
    "turn.settled": (
        {
            "status",
            "final_text",
            "final_text_digest",
            "final_text_len",
            "error_code",
            "interruption_cause",
            "changed_paths",
            "output_validators",
            "output_retries",
        },
        {"status"},
    ),
    # Both delta types stream raw model text to every sink, and ``text`` is required. v0.20 keeps
    # them that way deliberately: they are the only live-streaming mechanism, so they are switchable
    # and documented rather than removed. Pinned here so that decision has to be revisited on
    # purpose rather than drifted past — and this note is that revisit.
    #
    # The previous wording said "gated by ``emit_output_deltas`` (default False)", which was true of
    # this dataclass and false of the shipped product: Studio set the flag from
    # ``find_spec("httpx") is not None``, so for anyone who had installed the async extra the
    # channel was on, durable, and unfiltered. v0.20 adds a real off switch
    # (``MONOID_OUTPUT_DELTAS=0``, ``StudioConfig.stream_output_deltas``,
    # ``monoid studio serve --no-output-deltas``) instead of a default that only looked like one.
    "model.output.delta": ({"text"}, {"text"}),
    "model.reasoning.delta": ({"text"}, {"text"}),
}


def test_content_bearing_events_have_a_stable_field_set() -> None:
    for event_type, (properties, required) in _CONTENT_EVENT_FIELDS.items():
        schema = EVENT_DATA_SCHEMAS[event_type]
        assert set(schema["properties"]) == properties, event_type
        assert set(schema["required"]) == required, event_type
        assert schema["additionalProperties"] is False, event_type


def test_settle_events_still_accept_final_text() -> None:
    """v0.20 stops *emitting* model-authored ``final_text`` here; the property must stay.

    ``validate_run_dir`` replays committed ``events.jsonl`` against these schemas, so dropping the
    property would fail every run directory written before the change. Kernel-authored text also
    keeps travelling inline, so the field is still live on the emit side.
    """
    for event_type in ("run.finished", "turn.settled"):
        data = {"status": "completed", "final_text": "Stopped after reaching max steps."}
        assert list(_validator(event_type).iter_errors(data)) == [], event_type


def test_settle_events_accept_a_digest_without_the_text() -> None:
    # The shape a settle event takes once model-authored text moves to the run-dir record.
    for event_type in ("run.finished", "turn.settled"):
        data = {"status": "completed", "final_text_digest": "a" * 64, "final_text_len": 12}
        assert list(_validator(event_type).iter_errors(data)) == [], event_type


def test_settle_events_reject_each_mistyped_digest_field() -> None:
    """The counterweight: accepting the new keys must not mean accepting anything under them.

    Each field is mistyped on its own. Swapping both at once would pass even if one of the two
    had been declared with the wrong type, because the other's error alone satisfies the assert.
    """
    for event_type in ("run.finished", "turn.settled"):
        digest_as_int = {"status": "completed", "final_text_digest": 64}
        assert list(_validator(event_type).iter_errors(digest_as_int)), f"{event_type}: digest"
        length_as_str = {"status": "completed", "final_text_len": "12"}
        assert list(_validator(event_type).iter_errors(length_as_str)), f"{event_type}: len"
