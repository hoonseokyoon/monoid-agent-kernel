from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from monoid_agent_kernel.conformance import runner
from monoid_agent_kernel.conformance.profiles.minimal_agent import MINIMAL_AGENT_RULE_IDS
from monoid_agent_kernel.conformance.report import CONFORMANCE_REPORT_VERSION
from monoid_agent_kernel.conformance.runner import run_conformance


class _MinimalHarness:
    harness_id = "external-test"
    supported_profiles = ("minimal-agent",)

    def __init__(self, *, contiguous: bool = True) -> None:
        self._contiguous = contiguous

    def run_minimal_lifecycle_case(self) -> dict[str, object]:
        return {
            "submitted": True,
            "run_id": "run_external",
            "states": ["submitted", "running", "completed"],
            "result": {"run_id": "run_external", "status": "completed"},
            "event_seqs": [1, 2, 3] if self._contiguous else [1, 3],
        }


def test_external_runner_returns_typed_stable_outcomes() -> None:
    report = run_conformance(_MinimalHarness(), "minimal-agent")

    assert report.schema_version == CONFORMANCE_REPORT_VERSION
    assert report.passed
    assert tuple(outcome.rule_id for outcome in report.outcomes) == MINIMAL_AGENT_RULE_IDS
    assert report.to_json()["summary"] == {
        "total": 4,
        "passed": 4,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
    }


def test_external_runner_reports_rule_failure_without_hiding_observations() -> None:
    report = run_conformance(_MinimalHarness(contiguous=False), "minimal-agent")

    failed = [outcome for outcome in report.outcomes if outcome.status == "failed"]
    assert not report.passed
    assert [outcome.rule_id for outcome in failed] == ["MIN-04-EVENT-SEQUENCE"]
    assert failed[0].observations[-1].actual == (1, 3)


def test_module_runner_writes_json_and_junit_for_reference_backend(tmp_path: Path) -> None:
    json_path = tmp_path / "conformance.json"
    junit_path = tmp_path / "conformance.xml"
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(root / "src"), env.get("PYTHONPATH", "")))
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "monoid_agent_kernel.conformance.runner",
            "--harness",
            "monoid_agent_kernel.reference.conformance:create_minimal_harness",
            "--profile",
            "minimal-agent",
            "--json-out",
            str(json_path),
            "--junit-out",
            str(junit_path),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout_report = json.loads(result.stdout)
    file_report = json.loads(json_path.read_text(encoding="utf-8"))
    assert stdout_report == file_report
    assert file_report["passed"] is True
    suite = ET.parse(junit_path).getroot()
    assert suite.attrib["tests"] == "4"
    assert suite.attrib["failures"] == "0"
    assert [case.attrib["name"] for case in suite.findall("testcase")] == list(
        MINIMAL_AGENT_RULE_IDS
    )


def test_runner_reports_close_failure_without_overriding_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class CloseFailingHarness(_MinimalHarness):
        def close(self) -> None:
            raise OSError("cleanup failed")

    monkeypatch.setattr(runner, "load_harness", lambda factory_ref: CloseFailingHarness())

    exit_code = runner.main(["--harness", "external.module:create_harness"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out)["passed"] is True
    assert "conformance runner close error: OSError: cleanup failed" in captured.err
