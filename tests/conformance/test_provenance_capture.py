from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import monoid_agent_kernel.reference.conformance as reference_conformance

from monoid_agent_kernel._version import package_version
from monoid_agent_kernel.conformance.harness import MinimalAgentEvidenceCapture
from monoid_agent_kernel.conformance.profiles.minimal_agent import (
    MINIMAL_AGENT_RULE_IDS,
    execute_minimal_agent_profile,
    run_minimal_agent_profile,
)
from monoid_agent_kernel.conformance.provenance import (
    ConformanceEvent,
    ConformanceEvidenceBundle,
    ConformanceTarget,
    build_evidence_reference,
    case_id_sha256,
    serialize_conformance_evidence,
    verify_conformance_evidence,
)
from monoid_agent_kernel.conformance.report import CONFORMANCE_REPORT_VERSION
from monoid_agent_kernel.conformance.runner import run_conformance
from monoid_agent_kernel.reference.conformance import ReferenceBackendHarness


def _case(*, event_seqs: tuple[int, ...] = (1, 2)) -> dict[str, Any]:
    return {
        "submitted": True,
        "run_id": "run_external",
        "states": ("submitted", "running", "completed"),
        "result": {
            "run_id": "run_external",
            "status": "completed",
            "provider_error": "Authorization: Bearer raw-result-secret",
        },
        "event_seqs": event_seqs,
    }


def _evidence(*, event_seqs: tuple[int, ...] = (1, 2)) -> ConformanceEvidenceBundle:
    return ConformanceEvidenceBundle(
        profile_id="minimal-agent",
        target=ConformanceTarget(
            implementation_id="vendor.runtime",
            implementation_version="1.0.0",
            adapter_id="vendor.monoid-adapter",
            adapter_version="2.0.0",
        ),
        case_id_sha256=case_id_sha256("run_external"),
        run_id_present=True,
        submitted=True,
        states=("submitted", "running", "completed"),
        result_run_id_matches=True,
        result_status="completed",
        events=tuple(
            ConformanceEvent(seq=seq, event_type="run.observed") for seq in event_seqs
        ),
        events_complete=True,
        next_seq=(event_seqs[-1] + 1 if event_seqs else 0),
    )


class _LegacyHarness:
    harness_id = "legacy-external"
    supported_profiles = ("minimal-agent",)

    def __init__(self) -> None:
        self.calls = 0

    def run_minimal_lifecycle_case(self) -> dict[str, Any]:
        self.calls += 1
        return _case()


class _EvidenceHarness(_LegacyHarness):
    def __init__(
        self,
        *,
        case_event_seqs: tuple[int, ...] = (1, 2),
        evidence_event_seqs: tuple[int, ...] = (1, 2),
    ) -> None:
        super().__init__()
        self.case_event_seqs = case_event_seqs
        self.evidence_event_seqs = evidence_event_seqs
        self.evidence_calls = 0

    def run_minimal_lifecycle_case(self) -> dict[str, Any]:
        self.calls += 1
        raise AssertionError("legacy method must not run for an evidence harness")

    def run_minimal_lifecycle_case_with_evidence(self) -> MinimalAgentEvidenceCapture:
        self.evidence_calls += 1
        return MinimalAgentEvidenceCapture(
            case=_case(event_seqs=self.case_event_seqs),
            evidence=_evidence(event_seqs=self.evidence_event_seqs),
        )


def test_legacy_harness_executes_once_without_evidence() -> None:
    harness = _LegacyHarness()

    execution = execute_minimal_agent_profile(harness)

    assert harness.calls == 1
    assert execution.evidence == ()
    assert tuple(outcome.rule_id for outcome in execution.outcomes) == MINIMAL_AGENT_RULE_IDS
    assert all(outcome.passed for outcome in execution.outcomes)

    wrapper_harness = _LegacyHarness()
    wrapper_outcomes = run_minimal_agent_profile(wrapper_harness)
    assert wrapper_harness.calls == 1
    assert tuple(outcome.rule_id for outcome in wrapper_outcomes) == MINIMAL_AGENT_RULE_IDS
    assert all(outcome.passed for outcome in wrapper_outcomes)


def test_enhanced_harness_executes_one_case_and_preserves_failed_sequence() -> None:
    harness = _EvidenceHarness(
        case_event_seqs=(1, 3),
        evidence_event_seqs=(1, 3),
    )

    execution = execute_minimal_agent_profile(harness)

    assert harness.calls == 0
    assert harness.evidence_calls == 1
    assert len(execution.evidence) == 1
    assert tuple(event.seq for event in execution.evidence[0].events) == (1, 3)
    failed = [outcome for outcome in execution.outcomes if not outcome.passed]
    assert [outcome.rule_id for outcome in failed] == ["MIN-04-EVENT-SEQUENCE"]
    assert b"raw-result-secret" not in serialize_conformance_evidence(execution.evidence[0])


def test_mismatched_capture_becomes_safe_error_outcomes() -> None:
    execution = execute_minimal_agent_profile(
        _EvidenceHarness(case_event_seqs=(1, 3), evidence_event_seqs=(1, 2))
    )

    assert execution.evidence == ()
    assert all(outcome.status == "error" for outcome in execution.outcomes)
    assert {outcome.error for outcome in execution.outcomes} == {
        "ValueError: details redacted"
    }
    assert "run_external" not in repr(execution.outcomes)


def test_enhanced_result_accepts_generic_mapping_consistently() -> None:
    harness = _EvidenceHarness()
    capture = harness.run_minimal_lifecycle_case_with_evidence()
    case = dict(capture.case)
    case["result"] = MappingProxyType(dict(case["result"]))

    class MappingHarness(_EvidenceHarness):
        def run_minimal_lifecycle_case_with_evidence(self) -> MinimalAgentEvidenceCapture:
            return MinimalAgentEvidenceCapture(case=case, evidence=capture.evidence)

    execution = execute_minimal_agent_profile(MappingHarness())

    assert all(outcome.passed for outcome in execution.outcomes)
    assert execution.evidence == (capture.evidence,)


@pytest.mark.parametrize(
    "evidence",
    [
        replace(_evidence(), events_complete=False),
        replace(_evidence(), next_seq=99),
    ],
)
def test_incomplete_or_inconsistent_capture_becomes_safe_errors(
    evidence: ConformanceEvidenceBundle,
) -> None:
    class InvalidEvidenceHarness(_EvidenceHarness):
        def run_minimal_lifecycle_case_with_evidence(self) -> MinimalAgentEvidenceCapture:
            return MinimalAgentEvidenceCapture(case=_case(), evidence=evidence)

    execution = execute_minimal_agent_profile(InvalidEvidenceHarness())

    assert execution.evidence == ()
    assert all(outcome.status == "error" for outcome in execution.outcomes)


def test_runner_keeps_v1_report_and_junit_shape_for_enhanced_harness(
    tmp_path: Path,
) -> None:
    report = run_conformance(_EvidenceHarness(), "minimal-agent")
    payload = report.to_json()
    junit_path = tmp_path / "conformance.xml"
    report.write_junit(junit_path)
    suite = ET.parse(junit_path).getroot()

    assert payload["schema_version"] == CONFORMANCE_REPORT_VERSION
    assert set(payload) == {
        "schema_version",
        "runner_version",
        "harness_id",
        "profile_id",
        "passed",
        "started_at",
        "duration_s",
        "summary",
        "outcomes",
    }
    assert "target" not in payload
    assert "evidence" not in payload
    assert suite.find("properties") is None
    assert suite.attrib["hostname"] == "legacy-external"


def test_reference_collector_rechecks_terminal_high_water_after_exhausted_page() -> None:
    page_calls: list[int] = []
    status_calls = 0

    def read_page(from_seq: int, limit: int) -> dict[str, Any]:
        del limit
        page_calls.append(from_seq)
        if from_seq == 0:
            return {
                "events": [
                    {"seq": 1, "type": "run.started", "data": {"secret": "one"}},
                    {"seq": 2, "type": "run.running", "data": {"secret": "two"}},
                ],
                "next_seq": 3,
                "has_more": False,
            }
        assert from_seq == 3
        return {
            "events": [{"seq": 3, "type": "run.finished", "data": {"secret": "three"}}],
            "next_seq": 4,
            "has_more": False,
        }

    def read_status() -> dict[str, Any]:
        nonlocal status_calls
        status_calls += 1
        return {
            "terminal": True,
            "last_event_seq": 3,
        }

    events, next_seq = reference_conformance._collect_complete_minimal_events(
        read_page=read_page,
        read_status=read_status,
        page_size=2,
    )

    assert page_calls == [0, 3]
    assert status_calls == 2
    assert tuple((event.seq, event.event_type) for event in events) == (
        (1, "run.started"),
        (2, "run.running"),
        (3, "run.finished"),
    )
    assert next_seq == 4


def test_reference_collector_preserves_gaps_and_rejects_nonprogress() -> None:
    events, next_seq = reference_conformance._collect_complete_minimal_events(
        read_page=lambda from_seq, limit: {
            "events": [{"seq": 1, "type": "run.started"}, {"seq": 3, "type": "run.finished"}],
            "next_seq": 4,
            "has_more": False,
        },
        read_status=lambda: {"terminal": True, "last_event_seq": 3},
    )

    assert tuple(event.seq for event in events) == (1, 3)
    assert next_seq == 4

    with pytest.raises(ValueError, match="make progress"):
        reference_conformance._collect_complete_minimal_events(
            read_page=lambda from_seq, limit: {
                "events": [],
                "next_seq": from_seq,
                "has_more": True,
            },
            read_status=lambda: {"terminal": True, "last_event_seq": 0},
        )

    with pytest.raises(ValueError, match="inconsistent cursor"):
        reference_conformance._collect_complete_minimal_events(
            read_page=lambda from_seq, limit: {
                "events": [{"seq": 1, "type": "run.started"}],
                "next_seq": 100,
                "has_more": False,
            },
            read_status=lambda: {"terminal": True, "last_event_seq": 3},
        )

    with pytest.raises(ValueError, match="sequence did not advance"):
        reference_conformance._collect_complete_minimal_events(
            read_page=lambda from_seq, limit: {
                "events": [
                    {"seq": 2, "type": "run.running"},
                    {"seq": 1, "type": "run.started"},
                ],
                "next_seq": 2,
                "has_more": False,
            },
            read_status=lambda: {"terminal": True, "last_event_seq": 2},
        )


def test_reference_capture_paginates_and_excludes_raw_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reference_conformance, "_MINIMAL_EVIDENCE_PAGE_SIZE", 2)
    harness = ReferenceBackendHarness(tmp_path / "reference")
    original_events = harness.events
    calls: list[int] = []
    observed_run_ids: list[str] = []
    observed_tokens: list[str] = []
    secret = "Authorization: Bearer event-provider-secret"

    def paged_events(
        run_id: str,
        token: str,
        *,
        from_seq: int = 0,
        limit: int | None = None,
    ) -> dict[str, Any]:
        calls.append(from_seq)
        observed_run_ids.append(run_id)
        observed_tokens.append(token)
        page = original_events(run_id, token, from_seq=from_seq, limit=limit)
        for event in page["events"]:
            event.setdefault("data", {})["provider_error"] = secret
        return page

    monkeypatch.setattr(harness, "events", paged_events)
    try:
        execution = execute_minimal_agent_profile(harness)
    finally:
        harness.close()

    assert len(calls) > 1
    assert calls[0] == 0
    assert all(right > left for left, right in zip(calls, calls[1:]))
    assert len(execution.evidence) == 1
    bundle = execution.evidence[0]
    assert bundle.events_complete is True
    assert bundle.next_seq == bundle.events[-1].seq + 1
    assert bundle.target == ConformanceTarget(
        implementation_id="monoid-agent-kernel.reference-backend",
        implementation_version=package_version(),
        adapter_id="monoid-agent-kernel.reference-conformance",
        adapter_version=package_version(),
    )
    data = serialize_conformance_evidence(bundle)
    assert secret.encode() not in data
    assert observed_run_ids[0].encode() not in data
    assert observed_tokens[0].encode() not in data
    assert b'"data"' not in data
    assert str(tmp_path).encode() not in data
    reference = build_evidence_reference(
        bundle,
        evidence_id="minimal-agent.lifecycle",
        artifact_name="minimal-agent.evidence.json",
    )
    assert verify_conformance_evidence(reference, data) == bundle
    event_outcome = next(
        outcome
        for outcome in execution.outcomes
        if outcome.rule_id == "MIN-04-EVENT-SEQUENCE"
    )
    assert event_outcome.observations[-1].actual == tuple(
        event.seq for event in bundle.events
    )
