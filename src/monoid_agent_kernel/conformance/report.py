"""Typed, versioned conformance observations and report serializers."""

from __future__ import annotations

import json
import math
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from monoid_agent_kernel._version import package_version
from monoid_agent_kernel.conformance.provenance import (
    ConformanceEvidenceReference,
    ConformanceTarget,
)
from monoid_agent_kernel.core.durable_codec import DurableCodec, DurableLoadResult
from monoid_agent_kernel.identifiers import namespaced_id

CONFORMANCE_REPORT_V1 = namespaced_id("conformance-report.v1")
CONFORMANCE_REPORT_V2 = namespaced_id("conformance-report.v2")
# Reader-first rollout: the runner remains a v1 writer until evidence wiring flips this alias.
CONFORMANCE_REPORT_VERSION = CONFORMANCE_REPORT_V1
CONFORMANCE_REPORT_READER_VERSION = CONFORMANCE_REPORT_V2
SUPPORTED_CONFORMANCE_REPORT_VERSIONS = (
    CONFORMANCE_REPORT_V1,
    CONFORMANCE_REPORT_V2,
)
MAX_CONFORMANCE_REPORT_BYTES = 16 * 1024 * 1024
_REPORT_READ_CHUNK_BYTES = 64 * 1024
RuleStatus = Literal["passed", "failed", "error", "skipped"]
ReportProvenanceStatus = Literal["available", "unavailable"]


def safe_exception_summary(exc: BaseException) -> str:
    """Return a diagnostic category without copying exception text into CI artifacts."""

    category = next(
        (
            cls.__name__
            for cls in type(exc).__mro__
            if cls.__module__ == "builtins" and issubclass(cls, BaseException)
        ),
        "Exception",
    )
    return f"{category}: details redacted"


@dataclass(frozen=True)
class ConformanceObservation:
    observation_id: str
    passed: bool
    expected: Any
    actual: Any
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, str) or not self.observation_id:
            raise ValueError("conformance observation_id must be a non-empty string")
        if type(self.passed) is not bool:
            raise ValueError("conformance observation passed must be a boolean")
        if not isinstance(self.detail, str):
            raise ValueError("conformance observation detail must be a string")
        expected = _canonical_json_value(
            self.expected,
            "conformance expected observation",
        )
        actual = _canonical_json_value(
            self.actual,
            "conformance actual observation",
        )
        if self.passed != (actual == expected):
            raise ValueError("conformance observation passed value is inconsistent")
        object.__setattr__(self, "expected", expected)
        object.__setattr__(self, "actual", actual)

    def to_json(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "passed": self.passed,
            "expected": _json_value(self.expected, "conformance expected observation"),
            "actual": _json_value(self.actual, "conformance actual observation"),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ConformanceRuleOutcome:
    rule_id: str
    profile_id: str
    status: RuleStatus
    observations: tuple[ConformanceObservation, ...] = ()
    duration_s: float = 0.0
    error: str = ""
    evidence_refs: tuple[str, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not self.rule_id:
            raise ValueError("conformance rule_id must be a non-empty string")
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("conformance profile_id must be a non-empty string")
        if self.status not in {"passed", "failed", "error", "skipped"}:
            raise ValueError("unsupported conformance rule status")
        if not isinstance(self.error, str):
            raise ValueError("conformance rule error must be a string")
        _nonnegative_number(self.duration_s, "conformance rule duration_s")
        if not isinstance(self.observations, (list, tuple)):
            raise TypeError("conformance observations must be a list or tuple")
        observations = tuple(self.observations)
        if any(not isinstance(item, ConformanceObservation) for item in observations):
            raise TypeError("conformance observations must be typed")
        if self.status == "passed" and any(not item.passed for item in observations):
            raise ValueError("passed conformance outcomes require passed observations")
        if not isinstance(self.evidence_refs, (list, tuple)):
            raise TypeError("conformance evidence references must be a list or tuple")
        evidence_refs = tuple(self.evidence_refs)
        if any(not isinstance(item, str) or not item for item in evidence_refs):
            raise ValueError("conformance evidence references must be non-empty strings")
        if len(set(evidence_refs)) != len(evidence_refs):
            raise ValueError("conformance outcome evidence references must be unique")
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "evidence_refs", evidence_refs)

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_json(self, *, include_evidence_refs: bool = False) -> dict[str, Any]:
        payload = {
            "rule_id": self.rule_id,
            "profile_id": self.profile_id,
            "status": self.status,
            "passed": self.passed,
            "duration_s": self.duration_s,
            "error": self.error,
            "observations": [observation.to_json() for observation in self.observations],
        }
        if include_evidence_refs:
            payload["evidence_refs"] = list(self.evidence_refs)
        return payload


@dataclass(frozen=True)
class ConformanceReport:
    harness_id: str
    profile_id: str
    outcomes: tuple[ConformanceRuleOutcome, ...]
    started_at: float = field(default_factory=time.time)
    duration_s: float = 0.0
    schema_version: str = CONFORMANCE_REPORT_VERSION
    runner_version: str = field(default_factory=package_version)
    provenance_status: ReportProvenanceStatus = field(
        default="unavailable",
        kw_only=True,
    )
    target: ConformanceTarget | None = field(default=None, kw_only=True)
    evidence: tuple[ConformanceEvidenceReference, ...] = field(
        default=(),
        kw_only=True,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.harness_id, str) or not self.harness_id:
            raise ValueError("conformance report harness_id must be a non-empty string")
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("conformance report profile_id must be a non-empty string")
        if not isinstance(self.runner_version, str) or not self.runner_version:
            raise ValueError("conformance report runner_version must be a non-empty string")
        _nonnegative_number(self.started_at, "conformance report started_at")
        _nonnegative_number(self.duration_s, "conformance report duration_s")
        if self.schema_version not in SUPPORTED_CONFORMANCE_REPORT_VERSIONS:
            raise ValueError("unsupported conformance report schema")
        if not isinstance(self.outcomes, (list, tuple)):
            raise TypeError("conformance report outcomes must be a list or tuple")
        if not isinstance(self.evidence, (list, tuple)):
            raise TypeError("conformance report evidence must be a list or tuple")
        outcomes = tuple(self.outcomes)
        evidence = tuple(self.evidence)
        if any(not isinstance(item, ConformanceRuleOutcome) for item in outcomes):
            raise TypeError("conformance report outcomes must be typed")
        if any(not isinstance(item, ConformanceEvidenceReference) for item in evidence):
            raise TypeError("conformance report evidence must be typed")
        if any(item.profile_id != self.profile_id for item in outcomes):
            raise ValueError("conformance outcome profile does not match its report")
        evidence_ids = tuple(item.evidence_id for item in evidence)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("conformance report evidence ids must be unique")
        known_evidence = set(evidence_ids)
        if any(
            reference not in known_evidence
            for outcome in outcomes
            for reference in outcome.evidence_refs
        ):
            raise ValueError("conformance outcome contains a dangling evidence reference")
        if self.schema_version == CONFORMANCE_REPORT_V1:
            if (
                self.provenance_status != "unavailable"
                or self.target is not None
                or evidence
                or any(outcome.evidence_refs for outcome in outcomes)
            ):
                raise ValueError("v1 conformance reports cannot contain provenance fields")
        else:
            if self.provenance_status not in {"available", "unavailable"}:
                raise ValueError("unsupported conformance provenance status")
            if self.provenance_status == "available" and self.target is None:
                raise ValueError("available conformance provenance requires a target")
            if self.provenance_status == "unavailable" and (
                self.target is not None or evidence or any(outcome.evidence_refs for outcome in outcomes)
            ):
                raise ValueError("unavailable conformance provenance must be empty")
            if self.target is not None and not isinstance(self.target, ConformanceTarget):
                raise TypeError("conformance report target must be typed")
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "evidence", evidence)

    @property
    def passed(self) -> bool:
        return bool(self.outcomes) and all(outcome.passed for outcome in self.outcomes)

    def to_json(self) -> dict[str, Any]:
        include_provenance = self.schema_version == CONFORMANCE_REPORT_V2
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "runner_version": self.runner_version,
            "harness_id": self.harness_id,
            "profile_id": self.profile_id,
            "passed": self.passed,
            "started_at": self.started_at,
            "duration_s": self.duration_s,
            "summary": _report_summary(self.outcomes),
            "outcomes": [
                outcome.to_json(include_evidence_refs=include_provenance)
                for outcome in self.outcomes
            ],
        }
        if include_provenance:
            payload["provenance_status"] = self.provenance_status
            payload["target"] = self.target.to_json() if self.target is not None else None
            payload["evidence"] = [item.to_json() for item in self.evidence]
        return payload

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def write_junit(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        suite = ET.Element(
            "testsuite",
            {
                "name": f"monoid-conformance:{self.profile_id}",
                "tests": str(len(self.outcomes)),
                "failures": str(sum(outcome.status == "failed" for outcome in self.outcomes)),
                "errors": str(sum(outcome.status == "error" for outcome in self.outcomes)),
                "skipped": str(sum(outcome.status == "skipped" for outcome in self.outcomes)),
                "time": f"{self.duration_s:.6f}",
            },
        )
        suite.set("hostname", self.harness_id)
        for outcome in self.outcomes:
            case = ET.SubElement(
                suite,
                "testcase",
                {
                    "classname": self.profile_id,
                    "name": outcome.rule_id,
                    "time": f"{outcome.duration_s:.6f}",
                },
            )
            if outcome.status == "failed":
                failure = ET.SubElement(
                    case, "failure", {"message": outcome.error or "rule failed"}
                )
                failure.text = json.dumps(outcome.to_json(), sort_keys=True)
            elif outcome.status == "error":
                error = ET.SubElement(case, "error", {"message": outcome.error or "rule error"})
                error.text = json.dumps(outcome.to_json(), sort_keys=True)
            elif outcome.status == "skipped":
                ET.SubElement(case, "skipped")
            system_out = ET.SubElement(case, "system-out")
            system_out.text = json.dumps(
                [observation.to_json() for observation in outcome.observations], sort_keys=True
            )
        tree = ET.ElementTree(suite)
        ET.indent(tree, space="  ")
        tree.write(path, encoding="utf-8", xml_declaration=True)


def observation(
    observation_id: str,
    *,
    expected: Any,
    actual: Any,
    detail: str = "",
) -> ConformanceObservation:
    normalized_expected = _canonical_json_value(
        expected,
        "conformance expected observation",
    )
    normalized_actual = _canonical_json_value(
        actual,
        "conformance actual observation",
    )
    return ConformanceObservation(
        observation_id=observation_id,
        passed=normalized_actual == normalized_expected,
        expected=expected,
        actual=actual,
        detail=detail,
    )


def outcome_from_observations(
    rule_id: str,
    profile_id: str,
    observations: tuple[ConformanceObservation, ...],
    *,
    duration_s: float = 0.0,
) -> ConformanceRuleOutcome:
    passed = all(item.passed for item in observations)
    failed_ids = [item.observation_id for item in observations if not item.passed]
    return ConformanceRuleOutcome(
        rule_id=rule_id,
        profile_id=profile_id,
        status="passed" if passed else "failed",
        observations=observations,
        duration_s=duration_s,
        error="" if passed else f"failed observations: {', '.join(failed_ids)}",
    )


def decode_conformance_report(payload: object) -> DurableLoadResult[ConformanceReport]:
    """Decode v1/v2 report data into the current v2 typed model."""

    return _CONFORMANCE_REPORT_CODEC.decode(payload, _load_report_v2)


def read_conformance_report(
    path: Path,
    *,
    max_bytes: int = MAX_CONFORMANCE_REPORT_BYTES,
) -> DurableLoadResult[ConformanceReport]:
    """Read report JSON with duplicate-key rejection and checked version classification."""

    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("conformance report max_bytes must be a non-negative integer")
    try:
        if path.stat().st_size > max_bytes:
            return _CONFORMANCE_REPORT_CODEC.corrupt(
                "conformance report exceeds the byte limit"
            )
        with path.open("rb") as stream:
            data = bytearray()
            while len(data) <= max_bytes:
                request_size = min(
                    _REPORT_READ_CHUNK_BYTES,
                    max_bytes - len(data) + 1,
                )
                chunk = stream.read(request_size)
                if not chunk:
                    break
                data.extend(chunk)
        if len(data) > max_bytes:
            return _CONFORMANCE_REPORT_CODEC.corrupt(
                "conformance report exceeds the byte limit"
            )
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, RecursionError):
        return _CONFORMANCE_REPORT_CODEC.corrupt("conformance report JSON is unreadable")
    return decode_conformance_report(payload)


def _migrate_report_v1_to_v2(payload: dict[str, Any]) -> dict[str, Any]:
    if any(key in payload for key in {"provenance_status", "target", "evidence"}):
        raise ValueError("v1 conformance report contains reserved v2 fields")
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, list):
        raise ValueError("v1 conformance report outcomes must be a list")
    migrated_outcomes: list[dict[str, Any]] = []
    for outcome in outcomes:
        if not isinstance(outcome, Mapping):
            raise ValueError("v1 conformance report outcome must be an object")
        migrated = dict(outcome)
        if "evidence_refs" in migrated:
            raise ValueError("v1 conformance report outcome contains a reserved v2 field")
        migrated["evidence_refs"] = []
        migrated_outcomes.append(migrated)
    migrated_payload = dict(payload)
    migrated_payload["provenance_status"] = "unavailable"
    migrated_payload["target"] = None
    migrated_payload["evidence"] = []
    migrated_payload["outcomes"] = migrated_outcomes
    return migrated_payload


def _load_report_v2(payload: dict[str, Any]) -> ConformanceReport:
    parsed = _closed_object(
        payload,
        "conformance report",
        {
            "schema_version",
            "runner_version",
            "harness_id",
            "profile_id",
            "passed",
            "started_at",
            "duration_s",
            "summary",
            "outcomes",
            "provenance_status",
            "target",
            "evidence",
        },
    )
    if parsed["schema_version"] != CONFORMANCE_REPORT_V2:
        raise ValueError("conformance report loader requires v2")
    summary_payload = _closed_object(
        parsed["summary"],
        "conformance report summary",
        {"total", "passed", "failed", "errors", "skipped"},
    )
    summary = {
        key: _nonnegative_int(value, f"conformance summary {key}")
        for key, value in summary_payload.items()
    }
    outcome_payloads = _list(parsed["outcomes"], "conformance report outcomes")
    outcomes = tuple(_load_outcome(item) for item in outcome_payloads)
    target_payload = parsed["target"]
    target = (
        ConformanceTarget.from_json(_mapping(target_payload, "conformance report target"))
        if target_payload is not None
        else None
    )
    evidence = tuple(
        ConformanceEvidenceReference.from_json(
            _mapping(item, "conformance report evidence reference")
        )
        for item in _list(parsed["evidence"], "conformance report evidence")
    )
    provenance_status = _string(
        parsed["provenance_status"],
        "conformance report provenance_status",
    )
    if provenance_status not in {"available", "unavailable"}:
        raise ValueError("unsupported conformance report provenance status")
    report = ConformanceReport(
        harness_id=_string(parsed["harness_id"], "conformance report harness_id"),
        profile_id=_string(parsed["profile_id"], "conformance report profile_id"),
        outcomes=outcomes,
        started_at=_nonnegative_number(
            parsed["started_at"],
            "conformance report started_at",
        ),
        duration_s=_nonnegative_number(
            parsed["duration_s"],
            "conformance report duration_s",
        ),
        schema_version=CONFORMANCE_REPORT_V2,
        runner_version=_string(
            parsed["runner_version"],
            "conformance report runner_version",
        ),
        provenance_status=provenance_status,
        target=target,
        evidence=evidence,
    )
    if _boolean(parsed["passed"], "conformance report passed") != report.passed:
        raise ValueError("conformance report passed value is inconsistent")
    if summary != _report_summary(report.outcomes):
        raise ValueError("conformance report summary is inconsistent")
    return report


def _load_outcome(payload: Any) -> ConformanceRuleOutcome:
    parsed = _closed_object(
        payload,
        "conformance rule outcome",
        {
            "rule_id",
            "profile_id",
            "status",
            "passed",
            "duration_s",
            "error",
            "observations",
            "evidence_refs",
        },
    )
    status = _string(parsed["status"], "conformance rule status")
    if status not in {"passed", "failed", "error", "skipped"}:
        raise ValueError("unsupported conformance rule status")
    passed = _boolean(parsed["passed"], "conformance rule passed")
    if passed != (status == "passed"):
        raise ValueError("conformance rule passed value is inconsistent")
    return ConformanceRuleOutcome(
        rule_id=_string(parsed["rule_id"], "conformance rule_id"),
        profile_id=_string(parsed["profile_id"], "conformance profile_id"),
        status=status,
        observations=tuple(
            _load_observation(item)
            for item in _list(parsed["observations"], "conformance observations")
        ),
        duration_s=_nonnegative_number(
            parsed["duration_s"],
            "conformance rule duration_s",
        ),
        error=_optional_string(parsed["error"], "conformance rule error"),
        evidence_refs=tuple(
            _string(item, "conformance evidence reference id")
            for item in _list(parsed["evidence_refs"], "conformance evidence references")
        ),
    )


def _load_observation(payload: Any) -> ConformanceObservation:
    parsed = _closed_object(
        payload,
        "conformance observation",
        {"observation_id", "passed", "expected", "actual", "detail"},
    )
    expected = _json_value(parsed["expected"], "conformance expected observation")
    actual = _json_value(parsed["actual"], "conformance actual observation")
    passed = _boolean(parsed["passed"], "conformance observation passed")
    if passed != (actual == expected):
        raise ValueError("conformance observation passed value is inconsistent")
    return ConformanceObservation(
        observation_id=_string(
            parsed["observation_id"],
            "conformance observation_id",
        ),
        passed=passed,
        expected=expected,
        actual=actual,
        detail=_optional_string(parsed["detail"], "conformance observation detail"),
    )


def _report_summary(
    outcomes: tuple[ConformanceRuleOutcome, ...],
) -> dict[str, int]:
    return {
        "total": len(outcomes),
        "passed": sum(outcome.status == "passed" for outcome in outcomes),
        "failed": sum(outcome.status == "failed" for outcome in outcomes),
        "errors": sum(outcome.status == "error" for outcome in outcomes),
        "skipped": sum(outcome.status == "skipped" for outcome in outcomes),
    }


def _closed_object(value: Any, label: str, keys: set[str]) -> dict[str, Any]:
    payload = dict(_mapping(value, label))
    if set(payload) != keys:
        raise ValueError(f"{label} fields do not match the schema")
    return payload


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with string keys")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _nonnegative_number(value: Any, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return float(value)


def _json_value(value: Any, label: str) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, label) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{label} contains a non-string object key")
        return {key: _json_value(item, label) for key, item in value.items()}
    raise ValueError(f"{label} contains a non-JSON value")


def _canonical_json_value(value: Any, label: str) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_json_value(item, label) for item in value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{label} contains a non-string object key")
        return {key: _canonical_json_value(item, label) for key, item in value.items()}
    raise ValueError(f"{label} contains a non-JSON value")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("conformance report JSON contains duplicate keys")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> Any:
    del value
    raise ValueError("conformance report JSON contains a non-finite number")


_CONFORMANCE_REPORT_CODEC = DurableCodec[ConformanceReport](
    family="conformance-report",
    current_schema=CONFORMANCE_REPORT_READER_VERSION,
    accepted_namespaces=("monoid",),
    migrations={1: _migrate_report_v1_to_v2},
)
