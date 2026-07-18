from __future__ import annotations

import copy
import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from monoid_agent_kernel.conformance.fixtures import load_compatibility_fixtures
from monoid_agent_kernel.conformance.provenance import (
    ConformanceEvent,
    ConformanceEvidenceBundle,
    ConformanceEvidenceReference,
    ConformanceTarget,
    build_evidence_reference,
    case_id_sha256,
)
from monoid_agent_kernel.conformance.report import (
    CONFORMANCE_REPORT_READER_VERSION,
    CONFORMANCE_REPORT_V1,
    CONFORMANCE_REPORT_V2,
    CONFORMANCE_REPORT_VERSION,
    MAX_CONFORMANCE_REPORT_BYTES,
    SUPPORTED_CONFORMANCE_REPORT_VERSIONS,
    ConformanceObservation,
    ConformanceReport,
    ConformanceRuleOutcome,
    decode_conformance_report,
    observation,
    read_conformance_report,
)


def _target() -> ConformanceTarget:
    return ConformanceTarget(
        implementation_id="vendor.runtime",
        implementation_version="1.2.3",
        adapter_id="vendor.monoid-adapter",
        adapter_version="2.0.0",
    )


def _reference() -> tuple[ConformanceTarget, ConformanceEvidenceReference]:
    target = _target()
    bundle = ConformanceEvidenceBundle(
        profile_id="minimal-agent",
        target=target,
        case_id_sha256=case_id_sha256("run_1"),
        run_id_present=True,
        submitted=True,
        states=("submitted", "running", "completed"),
        result_run_id_matches=True,
        result_status="completed",
        events=(ConformanceEvent(seq=1, event_type="run.finished"),),
        events_complete=True,
        next_seq=2,
    )
    return target, build_evidence_reference(
        bundle,
        evidence_id="minimal-agent.lifecycle",
        artifact_name="minimal-agent.evidence.json",
    )


def _outcome(
    rule_id: str = "MIN-04-EVENT-SEQUENCE",
    *,
    evidence_refs: tuple[str, ...] = ("minimal-agent.lifecycle",),
) -> ConformanceRuleOutcome:
    return ConformanceRuleOutcome(
        rule_id=rule_id,
        profile_id="minimal-agent",
        status="passed",
        observations=(
            ConformanceObservation(
                observation_id="events_present",
                passed=True,
                expected=True,
                actual=True,
            ),
        ),
        evidence_refs=evidence_refs,
    )


def _v2_report(
    *,
    outcomes: tuple[ConformanceRuleOutcome, ...] | None = None,
) -> ConformanceReport:
    target, reference = _reference()
    return ConformanceReport(
        harness_id="external-adapter",
        profile_id="minimal-agent",
        outcomes=outcomes or (_outcome(),),
        started_at=1.0,
        duration_s=0.25,
        schema_version=CONFORMANCE_REPORT_V2,
        runner_version="0.19.2",
        provenance_status="available",
        target=target,
        evidence=(reference,),
    )


def test_reader_first_versions_keep_v1_writer_and_support_v2() -> None:
    assert CONFORMANCE_REPORT_VERSION == CONFORMANCE_REPORT_V1
    assert CONFORMANCE_REPORT_READER_VERSION == CONFORMANCE_REPORT_V2
    assert SUPPORTED_CONFORMANCE_REPORT_VERSIONS == (
        CONFORMANCE_REPORT_V1,
        CONFORMANCE_REPORT_V2,
    )


def test_v2_report_round_trips_through_checked_reader() -> None:
    report = _v2_report()
    payload = report.to_json()
    checked = decode_conformance_report(payload)

    assert payload["target"] == _target().to_json()
    assert payload["provenance_status"] == "available"
    assert payload["outcomes"][0]["evidence_refs"] == ["minimal-agent.lifecycle"]
    assert checked.status == "loaded"
    assert checked.observed_schema == CONFORMANCE_REPORT_V2
    assert checked.value == report
    assert checked.value is not None
    assert checked.value.to_json() == payload


def test_packaged_v1_report_migrates_purely_to_explicit_unavailable_provenance() -> None:
    fixture = next(
        item
        for item in load_compatibility_fixtures()
        if item.fixture_id == "conformance-report-v1"
    )
    source = copy.deepcopy(fixture.payload)

    checked = decode_conformance_report(fixture.payload)

    assert fixture.payload == source
    assert checked.status == fixture.expected_status == "migrated"
    assert checked.observed_schema == CONFORMANCE_REPORT_V1
    assert checked.migrations == ("v1->v2",)
    assert checked.value is not None
    assert checked.value.schema_version == CONFORMANCE_REPORT_V2
    assert checked.value.provenance_status == "unavailable"
    assert checked.value.target is None
    assert checked.value.evidence == ()
    assert all(outcome.evidence_refs == () for outcome in checked.value.outcomes)
    migrated = checked.value.to_json()
    assert migrated["target"] is None
    assert migrated["evidence"] == []
    assert migrated["provenance_status"] == "unavailable"


@pytest.mark.parametrize(
    ("schema_version", "status"),
    [
        ("monoid.conformance-report.v3", "unsupported_version"),
        ("native-agent-runner.conformance-report.v1", "unsupported_version"),
        ("monoid.other-report.v2", "corrupt"),
        ("monoid.conformance-report.vx", "corrupt"),
    ],
)
def test_checked_reader_classifies_future_namespace_family_and_malformed_versions(
    schema_version: str,
    status: str,
) -> None:
    payload = _v2_report().to_json()
    payload["schema_version"] = schema_version

    assert decode_conformance_report(payload).status == status


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update(extra="field"),
        lambda payload: payload.update(passed=False),
        lambda payload: payload["summary"].update(total=99),
        lambda payload: payload["outcomes"][0].update(passed=False),
        lambda payload: payload["outcomes"][0]["observations"][0].update(passed=False),
        lambda payload: payload["target"].update(secret="value"),
        lambda payload: payload["evidence"][0].update(size_bytes=-1),
    ],
)
def test_v2_reader_rejects_unknown_fields_and_inconsistent_derived_values(
    mutate: object,
) -> None:
    payload = _v2_report().to_json()
    mutate(payload)  # type: ignore[operator]

    assert decode_conformance_report(payload).status == "corrupt"


def test_v1_migration_rejects_reserved_v2_fields() -> None:
    fixture = next(
        item
        for item in load_compatibility_fixtures()
        if item.fixture_id == "conformance-report-v1"
    )
    payload = copy.deepcopy(fixture.payload)
    payload["target"] = None

    assert decode_conformance_report(payload).status == "corrupt"


def test_report_rejects_duplicate_and_dangling_evidence_references() -> None:
    report = _v2_report()
    reference = report.evidence[0]

    with pytest.raises(ValueError, match="evidence ids must be unique"):
        replace(report, evidence=(reference, reference))
    with pytest.raises(ValueError, match="must be unique"):
        replace(report, outcomes=(replace(report.outcomes[0], evidence_refs=("x", "x")),))
    with pytest.raises(ValueError, match="dangling"):
        replace(report, outcomes=(replace(report.outcomes[0], evidence_refs=("missing",)),))


def test_report_allows_cross_outcome_evidence_reuse() -> None:
    report = _v2_report(
        outcomes=(
            _outcome("MIN-02-LIFECYCLE"),
            _outcome("MIN-04-EVENT-SEQUENCE"),
        )
    )

    assert report.passed
    assert [item["evidence_refs"] for item in report.to_json()["outcomes"]] == [
        ["minimal-agent.lifecycle"],
        ["minimal-agent.lifecycle"],
    ]


def test_report_requires_target_for_available_or_referenced_provenance() -> None:
    report = _v2_report()

    with pytest.raises(ValueError, match="requires a target"):
        replace(report, target=None)
    with pytest.raises(ValueError, match="must be empty"):
        replace(report, provenance_status="unavailable")


def test_provenance_fields_are_keyword_only_additions() -> None:
    outcome_parameter = inspect.signature(ConformanceRuleOutcome).parameters["evidence_refs"]
    report_parameters = inspect.signature(ConformanceReport).parameters

    assert outcome_parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert report_parameters["provenance_status"].kind is inspect.Parameter.KEYWORD_ONLY
    assert report_parameters["target"].kind is inspect.Parameter.KEYWORD_ONLY
    assert report_parameters["evidence"].kind is inspect.Parameter.KEYWORD_ONLY


def test_file_reader_rejects_duplicate_keys_and_nonfinite_numbers(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    payload = json.dumps(_v2_report().to_json(), sort_keys=True)
    path.write_text(payload.replace('"profile_id":', '"profile_id":"x","profile_id":', 1))

    assert read_conformance_report(path).status == "corrupt"

    path.write_text(payload.replace('"duration_s": 0.25', '"duration_s": NaN', 1))
    assert read_conformance_report(path).status == "corrupt"

    path.write_text(payload)
    assert read_conformance_report(path, max_bytes=len(payload) - 1).status == "corrupt"
    assert MAX_CONFORMANCE_REPORT_BYTES > len(payload)

    with pytest.raises(ValueError, match="max_bytes"):
        read_conformance_report(path, max_bytes=-1)


def test_file_reader_uses_a_bounded_stream_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "report.json"
    payload = json.dumps(_v2_report().to_json(), sort_keys=True)
    path.write_text(payload)
    original_open = Path.open
    requested_sizes: list[int] = []

    class _Reader:
        def __init__(self, stream: Any) -> None:
            self._stream = stream

        def __enter__(self) -> _Reader:
            self._stream.__enter__()
            return self

        def __exit__(self, *args: Any) -> None:
            self._stream.__exit__(*args)

        def read(self, size: int = -1) -> bytes:
            requested_sizes.append(size)
            return self._stream.read(size)

    def _open(candidate: Path, *args: Any, **kwargs: Any) -> _Reader:
        return _Reader(original_open(candidate, *args, **kwargs))

    monkeypatch.setattr(Path, "open", _open)

    assert read_conformance_report(path, max_bytes=len(payload)).status == "loaded"
    assert requested_sizes == [len(payload) + 1, 1]

    requested_sizes.clear()
    assert read_conformance_report(path, max_bytes=2**40).status == "loaded"
    assert requested_sizes == [64 * 1024, 64 * 1024]


def test_observation_derives_passed_from_normalized_json_values() -> None:
    normalized = observation(
        "sequence",
        expected=(1, 2),
        actual=[1, 2],
    )

    assert normalized.passed is True
    assert normalized.expected == (1, 2)
    assert normalized.actual == (1, 2)
    report = _v2_report(
        outcomes=(
            ConformanceRuleOutcome(
                rule_id="MIN-04-EVENT-SEQUENCE",
                profile_id="minimal-agent",
                status="passed",
                observations=(normalized,),
                evidence_refs=("minimal-agent.lifecycle",),
            ),
        )
    )
    decoded = decode_conformance_report(report.to_json())

    assert decoded.status == "loaded"
    assert decoded.value == report
    with pytest.raises(ValueError, match="passed value is inconsistent"):
        ConformanceObservation("sequence", False, (1, 2), [1, 2])


def test_v2_reader_accepts_json_serializable_tuple_observations() -> None:
    report = _v2_report(
        outcomes=(
            ConformanceRuleOutcome(
                rule_id="MIN-04-EVENT-SEQUENCE",
                profile_id="minimal-agent",
                status="passed",
                observations=(
                    ConformanceObservation(
                        observation_id="event_sequence_contiguous",
                        passed=True,
                        expected=(1, 2),
                        actual=(1, 2),
                    ),
                ),
                evidence_refs=("minimal-agent.lifecycle",),
            ),
        )
    )

    payload = report.to_json()
    assert payload["outcomes"][0]["observations"][0]["actual"] == [1, 2]
    assert decode_conformance_report(payload).status == "loaded"
