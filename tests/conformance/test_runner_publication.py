from __future__ import annotations

import builtins
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from monoid_agent_kernel.conformance import runner
from monoid_agent_kernel.conformance.harness import MinimalAgentEvidenceCapture
from monoid_agent_kernel.conformance.provenance import (
    ConformanceEvent,
    ConformanceEvidenceBundle,
    ConformanceTarget,
    case_id_sha256,
)
from monoid_agent_kernel.conformance.report import (
    CONFORMANCE_REPORT_V1,
    CONFORMANCE_REPORT_V2,
    read_conformance_report,
)
from monoid_agent_kernel.conformance.runner import (
    ConformanceEvidenceArtifact,
    ConformanceExecution,
    execute_conformance,
    run_conformance,
)
from monoid_agent_kernel.conformance.verification import verify_conformance_report

_RAW_RUN_ID = "run-private-123"
_SECRET = "Authorization: Bearer raw-provider-secret"


def _case(*, event_seqs: tuple[int, ...] = (1, 2)) -> dict[str, Any]:
    return {
        "submitted": True,
        "run_id": _RAW_RUN_ID,
        "states": ("submitted", "running", "completed"),
        "result": {
            "run_id": _RAW_RUN_ID,
            "status": "completed",
            "provider_error": _SECRET,
        },
        "event_seqs": event_seqs,
    }


def _bundle(*, event_seqs: tuple[int, ...] = (1, 2)) -> ConformanceEvidenceBundle:
    return ConformanceEvidenceBundle(
        profile_id="minimal-agent",
        target=ConformanceTarget(
            implementation_id="vendor.runtime",
            implementation_version="1.0.0",
            adapter_id="vendor.monoid-adapter",
            adapter_version="2.0.0",
        ),
        case_id_sha256=case_id_sha256(_RAW_RUN_ID),
        run_id_present=True,
        submitted=True,
        states=("submitted", "running", "completed"),
        result_run_id_matches=True,
        result_status="completed",
        events=tuple(ConformanceEvent(seq=seq, event_type="run.observed") for seq in event_seqs),
        events_complete=True,
        next_seq=(event_seqs[-1] + 1 if event_seqs else 0),
    )


class _EvidenceHarness:
    harness_id = "external-evidence"
    supported_profiles = ("minimal-agent",)

    def __init__(self, *, event_seqs: tuple[int, ...] = (1, 2)) -> None:
        self.event_seqs = event_seqs
        self.legacy_calls = 0
        self.evidence_calls = 0
        self.close_calls = 0

    def run_minimal_lifecycle_case(self) -> dict[str, Any]:
        self.legacy_calls += 1
        raise AssertionError("enhanced harness must execute through its evidence method")

    def run_minimal_lifecycle_case_with_evidence(self) -> MinimalAgentEvidenceCapture:
        self.evidence_calls += 1
        return MinimalAgentEvidenceCapture(
            case=_case(event_seqs=self.event_seqs),
            evidence=_bundle(event_seqs=self.event_seqs),
        )

    def close(self) -> None:
        self.close_calls += 1


class _LegacyHarness:
    harness_id = "legacy external/harness"
    supported_profiles = ("minimal-agent",)

    def __init__(self) -> None:
        self.calls = 0
        self.close_calls = 0

    def run_minimal_lifecycle_case(self) -> dict[str, Any]:
        self.calls += 1
        return _case()

    def close(self) -> None:
        self.close_calls += 1


def test_report_only_wrapper_never_claims_discarded_evidence() -> None:
    harness = _EvidenceHarness()

    report = run_conformance(harness, "minimal-agent")

    assert harness.evidence_calls == 1
    assert harness.legacy_calls == 0
    assert report.schema_version == CONFORMANCE_REPORT_V1
    assert report.provenance_status == "unavailable"
    assert report.target is None
    assert report.evidence == ()
    assert all(outcome.evidence_refs == () for outcome in report.outcomes)


def test_retained_execution_self_verifies_from_one_harness_invocation() -> None:
    harness = _EvidenceHarness()

    execution = execute_conformance(
        harness,
        "minimal-agent",
        retain_evidence=True,
    )

    assert harness.evidence_calls == 1
    assert harness.legacy_calls == 0
    assert execution.report.provenance_status == "available"
    assert len(execution.evidence_artifacts) == 1
    artifact = execution.evidence_artifacts[0]
    digest = artifact.reference.resource.digest("sha256")
    assert artifact.reference.resource.name == (f"minimal-agent.lifecycle.sha256-{digest}.json")
    assert all(
        outcome.evidence_refs == ("minimal-agent.lifecycle",)
        for outcome in execution.report.outcomes
    )
    verified = verify_conformance_report(
        execution.report,
        {artifact.reference.evidence_id: artifact.data},
    )
    assert verified.status == "verified"
    assert verified.report_passed is True


def test_public_execution_carriers_enforce_typed_evidence_ownership() -> None:
    available = execute_conformance(
        _EvidenceHarness(),
        "minimal-agent",
        retain_evidence=True,
    )
    unavailable = execute_conformance(_EvidenceHarness(), "minimal-agent")
    artifact = available.evidence_artifacts[0]

    with pytest.raises(TypeError, match="reference must be typed"):
        ConformanceEvidenceArtifact(
            reference=object(),  # type: ignore[arg-type]
            data=artifact.data,
        )
    with pytest.raises(TypeError, match="data must be bytes"):
        ConformanceEvidenceArtifact(
            reference=artifact.reference,
            data=bytearray(artifact.data),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="digest mismatch"):
        ConformanceEvidenceArtifact(
            reference=artifact.reference,
            data=artifact.data[:-1] + b" ",
        )
    with pytest.raises(TypeError, match="report must be typed"):
        ConformanceExecution(report=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="artifacts must be typed"):
        ConformanceExecution(
            report=unavailable.report,
            evidence_artifacts=(object(),),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="retain every"):
        ConformanceExecution(report=available.report)
    with pytest.raises(ValueError, match="cannot retain"):
        ConformanceExecution(
            report=unavailable.report,
            evidence_artifacts=(artifact,),
        )
    with pytest.raises(TypeError, match="retain_evidence"):
        execute_conformance(
            _EvidenceHarness(),
            "minimal-agent",
            retain_evidence=1,  # type: ignore[arg-type]
        )


def test_cli_publishes_verified_available_outputs_in_one_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = _EvidenceHarness()
    monkeypatch.setattr(runner, "load_harness", lambda factory_ref: harness)
    json_path = tmp_path / "conformance.json"
    junit_path = tmp_path / "conformance.xml"
    evidence_dir = tmp_path / "evidence"

    exit_code = runner.main(
        [
            "--harness",
            "external.module:create_harness",
            "--json-out",
            str(json_path),
            "--junit-out",
            str(junit_path),
            "--evidence-dir",
            str(evidence_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0, captured.err
    assert harness.evidence_calls == 1
    assert harness.legacy_calls == 0
    assert harness.close_calls == 1
    stdout_report = json.loads(captured.out)
    file_report = json.loads(json_path.read_text(encoding="utf-8"))
    assert stdout_report == file_report
    checked = read_conformance_report(json_path)
    assert checked.status == "loaded" and checked.value is not None
    report = checked.value
    reference = report.evidence[0]
    evidence_path = evidence_dir / reference.resource.name
    evidence_data = evidence_path.read_bytes()
    verified = verify_conformance_report(
        report,
        {reference.evidence_id: evidence_data},
    )
    assert verified.status == "verified"
    assert verified.report_passed is True

    suite = ET.parse(junit_path).getroot()
    suite_properties = {
        item.attrib["name"]: item.attrib["value"] for item in suite.findall("./properties/property")
    }
    assert suite.attrib["hostname"] == harness.harness_id
    assert suite_properties["monoid.report.schema_version"] == CONFORMANCE_REPORT_V2
    assert suite_properties["monoid.provenance.status"] == "available"
    assert json.loads(suite_properties["monoid.target"]) == report.target.to_json()
    assert (
        json.loads(suite_properties["monoid.evidence.minimal-agent.lifecycle"])
        == reference.to_json()
    )
    assert json.loads(suite_properties["monoid.rule_evidence_refs"]) == [
        {
            "rule_id": outcome.rule_id,
            "evidence_refs": ["minimal-agent.lifecycle"],
        }
        for outcome in report.outcomes
    ]
    for case in suite.findall("testcase"):
        assert case.find("properties") is None
        assert {child.tag for child in case} <= {
            "error",
            "failure",
            "skipped",
            "system-out",
            "system-err",
        }
        assert isinstance(json.loads(case.findtext("system-out", default="")), list)

    serialized = b"\n".join(
        (
            captured.out.encode(),
            captured.err.encode(),
            json_path.read_bytes(),
            junit_path.read_bytes(),
            evidence_data,
        )
    )
    assert _RAW_RUN_ID.encode() not in serialized
    assert _SECRET.encode() not in serialized
    assert str(tmp_path).encode() not in serialized
    assert b'"provider_error"' not in serialized


def test_legacy_harness_with_evidence_directory_preserves_v1_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = _LegacyHarness()
    evidence_dir = tmp_path / "evidence"
    monkeypatch.setattr(runner, "load_harness", lambda factory_ref: harness)

    exit_code = runner.main(
        [
            "--harness",
            "external.module:create_harness",
            "--evidence-dir",
            str(evidence_dir),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert harness.calls == 1
    assert harness.close_calls == 1
    assert payload["schema_version"] == CONFORMANCE_REPORT_V1
    assert "provenance_status" not in payload
    assert "target" not in payload
    assert "evidence" not in payload
    assert not evidence_dir.exists()


def test_content_addressed_publish_reuses_exact_bytes_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evidence" / "artifact.json"
    data = b'{"safe":true}\n'

    runner._publish_content_addressed(path, data)
    original = path.stat()
    runner._publish_content_addressed(path, data)
    reused = path.stat()

    assert path.read_bytes() == data
    assert reused.st_mtime_ns == original.st_mtime_ns
    assert reused.st_ino == original.st_ino
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []

    conflict = tmp_path / "evidence" / "conflict.json"
    conflict.write_bytes(b"different")
    with pytest.raises(ValueError, match="conflicts"):
        runner._publish_content_addressed(conflict, data)
    assert conflict.read_bytes() == b"different"
    assert list(conflict.parent.glob(f".{conflict.name}.*.tmp")) == []


def test_content_addressed_race_never_overwrites_the_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "evidence" / "artifact.json"
    desired = b"desired"
    winner = b"racing-writer"
    original_link = os.link

    def _racing_link(source: Path, destination: Path) -> None:
        destination.write_bytes(winner)
        raise FileExistsError(destination)

    monkeypatch.setattr(os, "link", _racing_link)
    with pytest.raises(ValueError, match="conflicts"):
        runner._publish_content_addressed(path, desired)
    monkeypatch.setattr(os, "link", original_link)

    assert path.read_bytes() == winner


def test_content_addressed_publish_requires_an_atomic_no_link_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "evidence" / "artifact.json"
    data = b"portable-filesystem"

    def _unsupported_link(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("hard links unsupported")

    monkeypatch.setattr(os, "link", _unsupported_link)

    if os.name == "nt":
        runner._publish_content_addressed(path, data)
        runner._publish_content_addressed(path, data)
        assert path.read_bytes() == data
    else:
        with pytest.raises(OSError, match="atomic no-replace support"):
            runner._publish_content_addressed(path, data)
        assert not path.exists()
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


@pytest.mark.skipif(os.name != "nt", reason="Windows rename is the no-link fallback")
def test_windows_no_link_fallback_never_replaces_a_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "evidence" / "artifact.json"
    path.parent.mkdir()
    path.write_bytes(b"winner")

    def _unsupported_link(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("hard links unsupported")

    monkeypatch.setattr(os, "link", _unsupported_link)

    with pytest.raises(ValueError, match="conflicts"):
        runner._publish_content_addressed(path, b"challenger")

    assert path.read_bytes() == b"winner"
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_content_addressed_publish_rejects_symlink_targets(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"safe")
    link = tmp_path / "evidence" / "artifact.json"
    link.parent.mkdir()
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match="regular file"):
        runner._publish_content_addressed(link, b"safe")

    assert target.read_bytes() == b"safe"


@pytest.mark.parametrize(
    "outputs",
    [
        ("--json-out", "same", "--junit-out", "same"),
        ("--json-out", "nested/../same", "--junit-out", "same"),
        ("--json-out", "parent", "--junit-out", "parent/child.xml"),
        ("--json-out", "parent", "--evidence-dir", "parent/evidence"),
    ],
)
def test_known_output_conflicts_fail_before_loading_the_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outputs: tuple[str, str, str, str],
) -> None:
    calls = 0

    def _load(factory_ref: str) -> _EvidenceHarness:
        nonlocal calls
        calls += 1
        return _EvidenceHarness()

    monkeypatch.setattr(runner, "load_harness", _load)
    arguments = ["--harness", "external.module:create_harness"]
    for option, value in zip(outputs[::2], outputs[1::2], strict=True):
        arguments.extend((option, str(tmp_path / value)))

    assert runner.main(arguments) == 2

    captured = capsys.readouterr()
    assert calls == 0
    assert captured.out == ""
    assert "ValueError: details redacted" in captured.err


@pytest.mark.skipif(os.name != "nt", reason="Win32 filename aliases are platform-specific")
@pytest.mark.parametrize(
    "outputs",
    [
        ("--json-out", "Report.json", "--junit-out", "report.json"),
        ("--json-out", "report.json.", "--junit-out", "other.json"),
        ("--json-out", "report.json ", "--junit-out", "other.json"),
        ("--json-out", "report.json::$DATA", "--junit-out", "other.json"),
        ("--json-out", "NUL.txt", "--junit-out", "other.json"),
    ],
)
def test_windows_output_aliases_fail_before_loading_the_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    outputs: tuple[str, str, str, str],
) -> None:
    calls = 0

    def _load(factory_ref: str) -> _EvidenceHarness:
        nonlocal calls
        calls += 1
        return _EvidenceHarness()

    monkeypatch.setattr(runner, "load_harness", _load)
    arguments = ["--harness", "external.module:create_harness"]
    for option, value in zip(outputs[::2], outputs[1::2], strict=True):
        arguments.extend((option, str(tmp_path / value)))

    assert runner.main(arguments) == 2
    assert calls == 0
    assert capsys.readouterr().out == ""


def test_existing_hardlink_alias_fails_before_loading_the_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"old")
    os.link(first, second)
    calls = 0

    def _load(factory_ref: str) -> _EvidenceHarness:
        nonlocal calls
        calls += 1
        return _EvidenceHarness()

    monkeypatch.setattr(runner, "load_harness", _load)

    result = runner.main(
        [
            "--harness",
            "external.module:create_harness",
            "--json-out",
            str(first),
            "--junit-out",
            str(second),
        ]
    )

    assert result == 2
    assert calls == 0
    assert capsys.readouterr().out == ""


def test_symlink_parent_alias_fails_before_loading_the_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    actual = tmp_path / "actual"
    alias = tmp_path / "alias"
    actual.mkdir()
    try:
        alias.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    calls = 0

    def _load(factory_ref: str) -> _EvidenceHarness:
        nonlocal calls
        calls += 1
        return _EvidenceHarness()

    monkeypatch.setattr(runner, "load_harness", _load)

    result = runner.main(
        [
            "--harness",
            "external.module:create_harness",
            "--json-out",
            str(actual / "report"),
            "--junit-out",
            str(alias / "report"),
        ]
    )

    assert result == 2
    assert calls == 0
    assert capsys.readouterr().out == ""


def test_digest_derived_output_collision_fails_before_any_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    preview = execute_conformance(
        _EvidenceHarness(),
        "minimal-agent",
        retain_evidence=True,
    )
    name = preview.evidence_artifacts[0].reference.resource.name
    evidence_dir = tmp_path / "evidence"
    harness = _EvidenceHarness()
    monkeypatch.setattr(runner, "load_harness", lambda factory_ref: harness)

    result = runner.main(
        [
            "--harness",
            "external.module:create_harness",
            "--json-out",
            str(evidence_dir / name),
            "--evidence-dir",
            str(evidence_dir),
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert harness.evidence_calls == 1
    assert harness.close_calls == 1
    assert captured.out == ""
    assert not evidence_dir.exists()


@pytest.mark.skipif(os.name != "nt", reason="8.3 aliases are Win32-specific")
def test_generated_evidence_recheck_catches_a_new_8dot3_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import ctypes

    preview = execute_conformance(
        _EvidenceHarness(),
        "minimal-agent",
        retain_evidence=True,
    )
    artifact = preview.evidence_artifacts[0]
    evidence_dir = tmp_path / "evidence"
    artifact_path = evidence_dir / artifact.reference.resource.name

    def _unsupported_link(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("hard links unsupported")

    monkeypatch.setattr(os, "link", _unsupported_link)
    runner._publish_content_addressed(artifact_path, artifact.data)
    buffer = ctypes.create_unicode_buffer(32768)
    get_short_path = ctypes.windll.kernel32.GetShortPathNameW  # type: ignore[attr-defined]
    get_short_path.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
    get_short_path.restype = ctypes.c_uint32
    length = get_short_path(str(artifact_path), buffer, len(buffer))
    if length == 0 or length >= len(buffer):
        pytest.skip("the volume does not expose a usable 8.3 alias")
    short_path = Path(buffer.value)
    if os.path.normcase(os.fspath(short_path)) == os.path.normcase(os.fspath(artifact_path)):
        pytest.skip("8.3 aliases are disabled on this volume")
    artifact_path.unlink()
    harness = _EvidenceHarness()
    monkeypatch.setattr(runner, "load_harness", lambda factory_ref: harness)

    result = runner.main(
        [
            "--harness",
            "external.module:create_harness",
            "--json-out",
            str(short_path),
            "--evidence-dir",
            str(evidence_dir),
        ]
    )

    captured = capsys.readouterr()
    if result == 0:
        assert artifact_path.exists(), "the mutable output replaced generated evidence"
        actual_buffer = ctypes.create_unicode_buffer(32768)
        get_short_path(str(artifact_path), actual_buffer, len(actual_buffer))
        if os.path.normcase(actual_buffer.value) != os.path.normcase(os.fspath(short_path)):
            pytest.skip("the volume reassigned the short filename after the probe")
    assert result == 2
    assert harness.evidence_calls == 1
    assert harness.close_calls == 1
    assert captured.out == ""
    assert artifact_path.read_bytes() == artifact.data


def test_publication_order_is_evidence_then_junit_then_json_then_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    order: list[str] = []
    original_publish = runner._publish_content_addressed
    original_atomic = runner._atomic_write_bytes
    original_print = builtins.print

    def _publish(path: Path, data: bytes) -> None:
        order.append("evidence")
        original_publish(path, data)

    def _atomic(path: Path, data: bytes) -> None:
        order.append("junit" if path.suffix == ".xml" else "json")
        original_atomic(path, data)

    def _print(*args: Any, **kwargs: Any) -> None:
        order.append("stdout")
        original_print(*args, **kwargs)

    monkeypatch.setattr(runner, "load_harness", lambda factory_ref: _EvidenceHarness())
    monkeypatch.setattr(runner, "_publish_content_addressed", _publish)
    monkeypatch.setattr(runner, "_atomic_write_bytes", _atomic)
    monkeypatch.setattr(builtins, "print", _print)

    result = runner.main(
        [
            "--harness",
            "external.module:create_harness",
            "--json-out",
            str(tmp_path / "report.json"),
            "--junit-out",
            str(tmp_path / "report.xml"),
            "--evidence-dir",
            str(tmp_path / "evidence"),
        ]
    )

    assert result == 0
    assert order == ["evidence", "junit", "json", "stdout"]
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("failure_stage", ["evidence", "junit", "json"])
def test_publication_failures_preserve_the_last_authoritative_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_stage: str,
) -> None:
    secret = "publication-provider-secret"
    json_path = tmp_path / "report.json"
    junit_path = tmp_path / "report.xml"
    evidence_dir = tmp_path / "evidence"
    old_json = b"old-authoritative-json"
    old_junit = b"old-secondary-junit"
    json_path.write_bytes(old_json)
    junit_path.write_bytes(old_junit)
    original_publish = runner._publish_content_addressed
    original_atomic = runner._atomic_write_bytes
    harness = _EvidenceHarness()

    def _publish(path: Path, data: bytes) -> None:
        if failure_stage == "evidence":
            raise OSError(secret)
        original_publish(path, data)

    def _atomic(path: Path, data: bytes) -> None:
        stage = "junit" if path.suffix == ".xml" else "json"
        if failure_stage == stage:
            raise OSError(secret)
        original_atomic(path, data)

    monkeypatch.setattr(runner, "load_harness", lambda factory_ref: harness)
    monkeypatch.setattr(runner, "_publish_content_addressed", _publish)
    monkeypatch.setattr(runner, "_atomic_write_bytes", _atomic)

    result = runner.main(
        [
            "--harness",
            "external.module:create_harness",
            "--json-out",
            str(json_path),
            "--junit-out",
            str(junit_path),
            "--evidence-dir",
            str(evidence_dir),
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert secret not in captured.err
    assert "OSError: details redacted" in captured.err
    assert harness.evidence_calls == 1
    assert harness.close_calls == 1
    assert json_path.read_bytes() == old_json
    if failure_stage in {"evidence", "junit"}:
        assert junit_path.read_bytes() == old_junit
    else:
        assert junit_path.read_bytes() != old_junit
        assert ET.parse(junit_path).getroot().attrib["tests"] == "4"
    evidence_files = list(evidence_dir.glob("*.json")) if evidence_dir.exists() else []
    assert len(evidence_files) == (0 if failure_stage == "evidence" else 1)
    assert list(tmp_path.rglob("*.tmp")) == []


def test_failed_rule_still_publishes_verifiable_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        runner,
        "load_harness",
        lambda factory_ref: _EvidenceHarness(event_seqs=(1, 3)),
    )
    json_path = tmp_path / "report.json"
    evidence_dir = tmp_path / "evidence"

    result = runner.main(
        [
            "--harness",
            "external.module:create_harness",
            "--json-out",
            str(json_path),
            "--evidence-dir",
            str(evidence_dir),
        ]
    )

    assert result == 1
    assert json.loads(capsys.readouterr().out)["passed"] is False
    checked = read_conformance_report(json_path)
    assert checked.value is not None
    reference = checked.value.evidence[0]
    verified = verify_conformance_report(
        checked.value,
        {reference.evidence_id: (evidence_dir / reference.resource.name).read_bytes()},
    )
    assert verified.status == "verified"
    assert verified.report_passed is False
