"""Typed, versioned conformance observations and report serializers."""

from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from monoid_agent_kernel._version import package_version
from monoid_agent_kernel.identifiers import namespaced_id

CONFORMANCE_REPORT_VERSION = namespaced_id("conformance-report.v1")
RuleStatus = Literal["passed", "failed", "error", "skipped"]


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

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConformanceRuleOutcome:
    rule_id: str
    profile_id: str
    status: RuleStatus
    observations: tuple[ConformanceObservation, ...] = ()
    duration_s: float = 0.0
    error: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_json(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "profile_id": self.profile_id,
            "status": self.status,
            "passed": self.passed,
            "duration_s": self.duration_s,
            "error": self.error,
            "observations": [observation.to_json() for observation in self.observations],
        }


@dataclass(frozen=True)
class ConformanceReport:
    harness_id: str
    profile_id: str
    outcomes: tuple[ConformanceRuleOutcome, ...]
    started_at: float = field(default_factory=time.time)
    duration_s: float = 0.0
    schema_version: str = CONFORMANCE_REPORT_VERSION
    runner_version: str = field(default_factory=package_version)

    @property
    def passed(self) -> bool:
        return bool(self.outcomes) and all(outcome.passed for outcome in self.outcomes)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "runner_version": self.runner_version,
            "harness_id": self.harness_id,
            "profile_id": self.profile_id,
            "passed": self.passed,
            "started_at": self.started_at,
            "duration_s": self.duration_s,
            "summary": {
                "total": len(self.outcomes),
                "passed": sum(outcome.status == "passed" for outcome in self.outcomes),
                "failed": sum(outcome.status == "failed" for outcome in self.outcomes),
                "errors": sum(outcome.status == "error" for outcome in self.outcomes),
                "skipped": sum(outcome.status == "skipped" for outcome in self.outcomes),
            },
            "outcomes": [outcome.to_json() for outcome in self.outcomes],
        }

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
    return ConformanceObservation(
        observation_id=observation_id,
        passed=actual == expected,
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
