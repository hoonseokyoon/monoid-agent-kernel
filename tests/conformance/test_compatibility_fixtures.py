from __future__ import annotations

from monoid_agent_kernel.adapters.temporal import (
    TemporalActivationResult,
    TemporalRunPolicy,
    TemporalRunState,
    TemporalRunStatus,
)
from monoid_agent_kernel.conformance.fixtures import load_compatibility_fixtures
from monoid_agent_kernel.conformance.report import decode_conformance_report
from monoid_agent_kernel.core.checkpoint import decode_checkpoint
from monoid_agent_kernel.core.control import ControlCommand
from monoid_agent_kernel.core.model_invocation import decode_model_invocation
from monoid_agent_kernel.core.outcome import TerminalOutcome
from monoid_agent_kernel.hosting.activation import ActivationCommand, ActivationReceipt
from monoid_agent_kernel.hosting.admission import (
    AdmissionReceipt,
    AdmissionRequest,
    AdmittedCommand,
)


def test_packaged_compatibility_fixtures_have_stable_unique_ids() -> None:
    fixtures = load_compatibility_fixtures()

    assert len(fixtures) == 19
    assert len({fixture.fixture_id for fixture in fixtures}) == len(fixtures)


def test_checkpoint_compatibility_fixtures_match_checked_reader_outcomes() -> None:
    for fixture in load_compatibility_fixtures():
        if fixture.artifact != "checkpoint":
            continue
        assert decode_checkpoint(fixture.payload).status == fixture.expected_status


def test_legacy_control_command_fixture_is_readable() -> None:
    fixture = next(
        item
        for item in load_compatibility_fixtures()
        if item.fixture_id == "control-command-legacy-v1"
    )

    command = ControlCommand.from_json(fixture.payload)
    assert command.command_id == "fixture_command"
    assert command.type == "cancel"


def test_v021_checkpoint_fixture_defaults_v022_additive_fields() -> None:
    fixture = next(
        item
        for item in load_compatibility_fixtures()
        if item.fixture_id == "checkpoint-current-v1"
    )

    checkpoint = decode_checkpoint(fixture.payload).value

    assert checkpoint is not None
    assert checkpoint.last_model_invocation is None
    assert checkpoint.interruption_cause == ""


def test_v021_cancelled_checkpoint_fixture_leaves_cause_migration_to_restore() -> None:
    fixture = next(
        item
        for item in load_compatibility_fixtures()
        if item.fixture_id == "checkpoint-v021-cancelled-v1"
    )

    checkpoint = decode_checkpoint(fixture.payload).value

    assert checkpoint is not None
    assert checkpoint.cancellation_requested is True
    assert checkpoint.interruption_cause == ""


def test_v022_additive_checkpoint_fixture_carries_invocation_and_cause() -> None:
    fixture = next(
        item
        for item in load_compatibility_fixtures()
        if item.fixture_id == "checkpoint-v022-additive-v1"
    )

    checkpoint = decode_checkpoint(fixture.payload).value

    assert checkpoint is not None
    assert checkpoint.last_model_invocation is not None
    assert checkpoint.last_model_invocation["dispatch_state"] == "unknown"
    assert checkpoint.interruption_cause == "lease_lost"


def test_terminal_outcome_fixture_matches_strict_reader() -> None:
    fixture = next(
        item
        for item in load_compatibility_fixtures()
        if item.fixture_id == "terminal-outcome-current-v1"
    )

    outcome = TerminalOutcome.from_json(fixture.payload)

    assert outcome.kind == "dispatch_unknown"
    assert outcome.retry_eligibility == "after_reconciliation"


def test_model_invocation_fixtures_match_checked_reader_outcomes() -> None:
    for fixture in load_compatibility_fixtures():
        if fixture.artifact != "model-invocation":
            continue
        assert decode_model_invocation(fixture.payload).status == fixture.expected_status


def test_v1_conformance_report_fixture_matches_checked_reader_outcome() -> None:
    fixture = next(
        item
        for item in load_compatibility_fixtures()
        if item.fixture_id == "conformance-report-v1"
    )

    assert decode_conformance_report(fixture.payload).status == fixture.expected_status


def test_v023_portable_record_fixtures_match_strict_readers() -> None:
    readers = {
        "admission-request": AdmissionRequest.from_json,
        "admitted-command": AdmittedCommand.from_json,
        "admission-receipt": AdmissionReceipt.from_json,
        "activation-command": ActivationCommand.from_json,
        "activation-receipt": ActivationReceipt.from_json,
        "temporal-run-policy": TemporalRunPolicy.from_json,
        "temporal-run-state": TemporalRunState.from_json,
        "temporal-activation-result": TemporalActivationResult.from_json,
        "temporal-run-status": TemporalRunStatus.from_json,
    }
    fixtures = {
        fixture.artifact: fixture
        for fixture in load_compatibility_fixtures()
        if fixture.artifact in readers
    }

    assert set(fixtures) == set(readers)
    for artifact, reader in readers.items():
        fixture = fixtures[artifact]
        assert fixture.expected_status == "loaded"
        assert reader(fixture.payload).to_json() == fixture.payload
