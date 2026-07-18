"""External conformance runner with JSON and JUnit reports."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib
import json
import os
import stat
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from monoid_agent_kernel.conformance.harness import ConformanceHarness, MinimalAgentHarness
from monoid_agent_kernel.conformance.profiles.minimal_agent import (
    execute_minimal_agent_profile,
)
from monoid_agent_kernel.conformance.provenance import (
    ConformanceEvidenceReference,
    build_evidence_reference,
    serialize_conformance_evidence,
    verify_conformance_evidence,
)
from monoid_agent_kernel.conformance.report import (
    CONFORMANCE_REPORT_V2,
    CONFORMANCE_REPORT_VERSION,
    ConformanceReport,
    safe_exception_summary,
)
from monoid_agent_kernel.conformance.verification import verify_conformance_report

SUPPORTED_RUNNER_PROFILES = ("minimal-agent",)
_MINIMAL_AGENT_EVIDENCE_ID = "minimal-agent.lifecycle"
_WINDOWS_INVALID_COMPONENT_CHARS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED_COMPONENTS = frozenset(
    {
        "AUX",
        "CLOCK$",
        "CON",
        "CONIN$",
        "CONOUT$",
        "NUL",
        "PRN",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
        *(f"COM{number}" for number in "¹²³"),
        *(f"LPT{number}" for number in "¹²³"),
    }
)


@dataclass(frozen=True, kw_only=True)
class ConformanceEvidenceArtifact:
    """One verified exact-byte artifact retained alongside a report."""

    reference: ConformanceEvidenceReference
    data: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.reference) is not ConformanceEvidenceReference:
            raise TypeError("conformance execution evidence reference must be typed")
        if type(self.data) is not bytes:
            raise TypeError("conformance execution evidence data must be bytes")
        verify_conformance_evidence(self.reference, self.data)


@dataclass(frozen=True, kw_only=True)
class ConformanceExecution:
    """One report plus the exact evidence artifacts still owned by its caller."""

    report: ConformanceReport
    evidence_artifacts: tuple[ConformanceEvidenceArtifact, ...] = ()

    def __post_init__(self) -> None:
        if type(self.report) is not ConformanceReport:
            raise TypeError("conformance execution report must be typed")
        artifacts = tuple(self.evidence_artifacts)
        if any(type(item) is not ConformanceEvidenceArtifact for item in artifacts):
            raise TypeError("conformance execution evidence artifacts must be typed")
        references = tuple(item.reference for item in artifacts)
        if self.report.provenance_status == "available":
            if references != self.report.evidence or not artifacts:
                raise ValueError(
                    "available conformance execution must retain every evidence artifact"
                )
            verified = verify_conformance_report(
                self.report,
                {artifact.reference.evidence_id: artifact.data for artifact in artifacts},
            )
            if not verified.verified:
                raise ValueError("generated conformance report evidence did not verify")
        elif artifacts:
            raise ValueError("unavailable conformance execution cannot retain report evidence")
        object.__setattr__(self, "evidence_artifacts", artifacts)


def run_conformance(harness: ConformanceHarness, profile_id: str) -> ConformanceReport:
    """Run one supported profile without claiming discarded evidence."""

    return execute_conformance(harness, profile_id).report


def execute_conformance(
    harness: ConformanceHarness,
    profile_id: str,
    *,
    retain_evidence: bool = False,
) -> ConformanceExecution:
    """Execute once and optionally return a report bound to retained exact evidence."""

    if type(retain_evidence) is not bool:
        raise TypeError("retain_evidence must be a boolean")
    if profile_id not in SUPPORTED_RUNNER_PROFILES:
        raise ValueError(f"profile {profile_id!r} is not executable by the external runner")
    harness_id = _public_harness_id(harness.harness_id)
    if profile_id not in harness.supported_profiles:
        raise ValueError("harness does not declare the requested profile")
    started_at = time.time()
    started = time.perf_counter()
    if profile_id == "minimal-agent":
        if not isinstance(harness, MinimalAgentHarness):
            raise TypeError("minimal-agent requires MinimalAgentHarness")
        profile_execution = execute_minimal_agent_profile(harness)
    else:  # pragma: no cover - guarded by SUPPORTED_RUNNER_PROFILES
        raise AssertionError(profile_id)

    artifacts: tuple[ConformanceEvidenceArtifact, ...] = ()
    outcomes = profile_execution.outcomes
    provenance_status = "unavailable"
    target = None
    evidence_references: tuple[ConformanceEvidenceReference, ...] = ()
    if retain_evidence and profile_execution.evidence:
        if len(profile_execution.evidence) != 1:
            raise ValueError("minimal-agent execution produced unsupported evidence cardinality")
        bundle = profile_execution.evidence[0]
        data = serialize_conformance_evidence(bundle)
        digest = hashlib.sha256(data).hexdigest()
        artifact_name = f"{_MINIMAL_AGENT_EVIDENCE_ID}.sha256-{digest}.json"
        reference = build_evidence_reference(
            bundle,
            evidence_id=_MINIMAL_AGENT_EVIDENCE_ID,
            artifact_name=artifact_name,
        )
        artifact = ConformanceEvidenceArtifact(reference=reference, data=data)
        evidence_references = (reference,)
        outcomes = tuple(
            replace(outcome, evidence_refs=(_MINIMAL_AGENT_EVIDENCE_ID,)) for outcome in outcomes
        )
        provenance_status = "available"
        target = bundle.target
        artifacts = (artifact,)
    report = ConformanceReport(
        harness_id=harness_id,
        profile_id=profile_id,
        outcomes=outcomes,
        started_at=started_at,
        duration_s=time.perf_counter() - started,
        schema_version=(CONFORMANCE_REPORT_V2 if artifacts else CONFORMANCE_REPORT_VERSION),
        provenance_status=provenance_status,
        target=target,
        evidence=evidence_references,
    )
    return ConformanceExecution(
        report=report,
        evidence_artifacts=artifacts,
    )


def load_harness(factory_ref: str) -> ConformanceHarness:
    """Load ``module:factory`` and construct an external harness."""

    module_name, separator, attribute = factory_ref.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("harness must use module:factory syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"harness factory is not callable: {factory_ref}")
    harness = factory()
    if not isinstance(harness, ConformanceHarness):
        raise TypeError(f"factory did not return ConformanceHarness: {factory_ref}")
    return harness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m monoid_agent_kernel.conformance.runner")
    parser.add_argument(
        "--harness", required=True, help="External harness factory as module:factory"
    )
    parser.add_argument("--profile", choices=SUPPORTED_RUNNER_PROFILES, default="minimal-agent")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--junit-out", type=Path)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        help=(
            "Retain content-addressed evidence and emit available provenance when the adapter "
            "supplies it"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    harness: ConformanceHarness | None = None
    try:
        _require_output_roles(args.json_out, args.junit_out, args.evidence_dir)
        if args.evidence_dir is not None and (
            args.evidence_dir.exists() and not args.evidence_dir.is_dir()
        ):
            raise ValueError("conformance evidence output must be a directory")
        harness = load_harness(args.harness)
        execution = execute_conformance(
            harness,
            args.profile,
            retain_evidence=args.evidence_dir is not None,
        )
        report = execution.report
        evidence_outputs = _evidence_output_paths(execution, args.evidence_dir)
        output_paths = (
            args.json_out,
            args.junit_out,
            *(path for _, path in evidence_outputs),
        )
        _require_distinct_file_paths(*output_paths)
        console_json = json.dumps(
            report.to_json(),
            sort_keys=True,
            allow_nan=False,
        )
        json_data = report.to_json_bytes() if args.json_out is not None else None
        junit_data = report.to_junit_bytes() if args.junit_out is not None else None
        for artifact, path in evidence_outputs:
            _publish_content_addressed(path, artifact.data)
            _require_distinct_file_paths(*output_paths)
        if args.junit_out is not None and junit_data is not None:
            _atomic_write_bytes(args.junit_out, junit_data)
            _require_distinct_file_paths(*output_paths)
        if args.json_out is not None:
            if json_data is None:  # pragma: no cover - guarded by the serializer branch
                raise AssertionError("missing conformance JSON projection")
            _atomic_write_bytes(args.json_out, json_data)
            _require_distinct_file_paths(*output_paths)
        print(console_json)
        return 0 if report.passed else 1
    except Exception as exc:
        print(f"conformance runner error: {safe_exception_summary(exc)}", file=sys.stderr)
        return 2
    finally:
        close = getattr(harness, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                print(
                    f"conformance runner close error: {safe_exception_summary(exc)}",
                    file=sys.stderr,
                )


def _public_harness_id(value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError("harness_id must be a non-empty public string")
    return value


def _evidence_output_paths(
    execution: ConformanceExecution,
    evidence_dir: Path | None,
) -> tuple[tuple[ConformanceEvidenceArtifact, Path], ...]:
    if evidence_dir is None:
        if execution.evidence_artifacts:
            raise ValueError("retained evidence requires an output directory")
        return ()
    return tuple(
        (artifact, evidence_dir / artifact.reference.resource.name)
        for artifact in execution.evidence_artifacts
    )


def _require_output_roles(
    json_out: Path | None,
    junit_out: Path | None,
    evidence_dir: Path | None,
) -> None:
    _require_distinct_file_paths(json_out, junit_out)
    if evidence_dir is None:
        return
    _require_portable_output_path(evidence_dir)
    directory = evidence_dir.resolve(strict=False)
    for file_path in (json_out, junit_out):
        if file_path is None:
            continue
        output = file_path.resolve(strict=False)
        if _is_same_or_ancestor(output, directory):
            raise ValueError("conformance output file and evidence directory conflict")


def _require_distinct_file_paths(*paths: Path | None) -> None:
    present = [path for path in paths if path is not None]
    for path in present:
        _require_portable_output_path(path)
    resolved = [path.resolve(strict=False) for path in present]
    if len({_path_key(path) for path in resolved}) != len(resolved):
        raise ValueError("conformance output paths must be distinct")
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            if _is_same_or_ancestor(left, right) or _is_same_or_ancestor(
                right,
                left,
            ):
                raise ValueError("conformance output file paths cannot contain each other")
            if left.exists() and right.exists():
                try:
                    aliases = os.path.samefile(left, right)
                except OSError:
                    aliases = False
                if aliases:
                    raise ValueError("conformance output paths must not alias")


def _require_portable_output_path(path: Path) -> None:
    if os.name != "nt":
        return
    for component in path.parts:
        if component == path.anchor or component in {".", ".."}:
            continue
        if component.endswith((" ", ".")):
            raise ValueError("conformance output path has a non-portable component")
        if any(character in _WINDOWS_INVALID_COMPONENT_CHARS for character in component):
            raise ValueError("conformance output path has a non-portable component")
        if any(ord(character) < 32 for character in component):
            raise ValueError("conformance output path has a non-portable component")
        device_name = component.split(".", maxsplit=1)[0].upper()
        if device_name in _WINDOWS_RESERVED_COMPONENTS:
            raise ValueError("conformance output path has a reserved component")


def _path_key(path: Path) -> str:
    return os.path.normcase(os.fspath(path))


def _path_parts_key(path: Path) -> tuple[str, ...]:
    return tuple(os.path.normcase(part) for part in path.parts)


def _is_same_or_ancestor(ancestor: Path, descendant: Path) -> bool:
    ancestor_parts = _path_parts_key(ancestor)
    descendant_parts = _path_parts_key(descendant)
    return (
        len(ancestor_parts) <= len(descendant_parts)
        and descendant_parts[: len(ancestor_parts)] == ancestor_parts
    )


def _publish_content_addressed(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _write_temporary_bytes(path, data)
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
        except OSError as exc:
            if os.name != "nt":
                raise OSError(
                    "content-addressed conformance evidence requires atomic no-replace support"
                ) from exc
            try:
                os.rename(temporary, path)
            except FileExistsError:
                pass
    finally:
        temporary.unlink(missing_ok=True)
    _require_exact_regular_file(path, data)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _write_temporary_bytes(path, data)
    try:
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_temporary_bytes(path: Path, data: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _require_exact_regular_file(path: Path, data: bytes) -> None:
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("content-addressed conformance evidence must be a regular file")
    if metadata.st_size != len(data):
        raise ValueError("content-addressed conformance evidence conflicts")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode) or opened.st_size != len(data):
            raise ValueError("content-addressed conformance evidence conflicts")
        existing = handle.read(len(data) + 1)
    if not hmac.compare_digest(existing, data):
        raise ValueError("content-addressed conformance evidence conflicts")
    published = path.lstat()
    if path.is_symlink() or (published.st_dev, published.st_ino) != (
        opened.st_dev,
        opened.st_ino,
    ):
        raise ValueError("content-addressed conformance evidence changed during verification")


def _replace_with_retry(
    source: Path,
    destination: Path,
    *,
    attempts: int = 10,
    backoff_s: float = 0.01,
) -> None:
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(backoff_s)


__all__ = [
    "ConformanceEvidenceArtifact",
    "ConformanceExecution",
    "SUPPORTED_RUNNER_PROFILES",
    "build_parser",
    "execute_conformance",
    "load_harness",
    "main",
    "run_conformance",
]


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main())
