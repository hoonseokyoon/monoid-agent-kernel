from __future__ import annotations

from monoid_agent_kernel.conformance.fixtures import load_compatibility_fixtures
from monoid_agent_kernel.conformance.report import decode_conformance_report
from monoid_agent_kernel.core.checkpoint import decode_checkpoint
from monoid_agent_kernel.core.control import ControlCommand


def test_packaged_compatibility_fixtures_have_stable_unique_ids() -> None:
    fixtures = load_compatibility_fixtures()

    assert len(fixtures) == 5
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


def test_v1_conformance_report_fixture_matches_checked_reader_outcome() -> None:
    fixture = next(
        item
        for item in load_compatibility_fixtures()
        if item.fixture_id == "conformance-report-v1"
    )

    assert decode_conformance_report(fixture.payload).status == fixture.expected_status
