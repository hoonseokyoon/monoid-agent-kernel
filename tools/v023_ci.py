"""Validate and report the reproducible v0.23 CI/service campaign configuration."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import sysconfig
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "tests/service/campaign-lock.json"
COMPOSE_PATH = ROOT / "tests/service/compose.yml"
PYPROJECT_PATH = ROOT / "pyproject.toml"
ALLOWED_PROFILES = frozenset({"core", "postgres", "objectstore", "temporal", "combined"})
PROFILE_LABELS = frozenset(f"ci:{profile}" for profile in ALLOWED_PROFILES)
BRANCH_PATTERN = re.compile(r"(?:^|/)v0[.]23-pr(?P<number>[0-9]{2})(?:-|$)")
CAMPAIGN_TERMINAL_PROFILES = {
    "codex/v0.23-production-adapters": "combined",
    "develop": "combined",
}
SDK_IMPORT_MODULES = {
    "psycopg": "psycopg",
    "psycopg-pool": "psycopg_pool",
    "boto3": "boto3",
    "botocore": "botocore",
    "temporalio": "temporalio",
}
MAX_TEMPORAL_ARCHIVE_BYTES = 256 * 1024 * 1024


def _load_lock() -> dict[str, Any]:
    loaded = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("campaign lock root must be an object")
    return loaded


def _lock_digest(lock: dict[str, Any]) -> str:
    canonical = json.dumps(lock, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_lock() -> dict[str, Any]:
    """Fail closed when tracked dependency, service, or PR-profile locks drift."""
    import tomllib

    lock = _load_lock()
    if lock.get("schema_version") != 1 or lock.get("campaign") != "v0.23":
        raise ValueError("campaign lock identity is invalid")

    dependencies = lock.get("python_dependencies")
    if not isinstance(dependencies, dict) or set(dependencies) != {
        "psycopg",
        "psycopg-pool",
        "boto3",
        "botocore",
        "temporalio",
    }:
        raise ValueError("campaign Python dependency set is invalid")
    for name, entry in dependencies.items():
        if not isinstance(entry, dict) or not entry.get("requirement") or not entry.get("exact"):
            raise ValueError(f"dependency lock is incomplete: {name}")

    pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    expected_extras = {
        "postgres": [dependencies["psycopg"]["requirement"], dependencies["psycopg-pool"]["requirement"]],
        "object-store-s3": [dependencies["boto3"]["requirement"]],
        "temporal": [dependencies["temporalio"]["requirement"]],
        "durable-host": [
            dependencies["psycopg"]["requirement"],
            dependencies["psycopg-pool"]["requirement"],
            dependencies["boto3"]["requirement"],
            dependencies["temporalio"]["requirement"],
        ],
    }
    for extra, expected in expected_extras.items():
        if extras.get(extra) != expected:
            raise ValueError(f"pyproject extra {extra!r} differs from campaign lock")

    services = lock.get("services")
    if not isinstance(services, dict):
        raise ValueError("service lock is missing")
    compose = COMPOSE_PATH.read_text(encoding="utf-8")
    for service in ("postgres16", "postgres18", "minio"):
        image = services.get(service, {}).get("image")
        if not isinstance(image, str) or "@sha256:" not in image or image not in compose:
            raise ValueError(f"compose service image is not digest-locked: {service}")

    temporal = services.get("temporal_cli")
    if not isinstance(temporal, dict) or not str(temporal.get("version", "")).startswith("v"):
        raise ValueError("Temporal CLI version must include its v prefix")
    if not re.fullmatch(r"[0-9]+[.][0-9]+[.][0-9]+", str(temporal.get("embedded_server", ""))):
        raise ValueError("Temporal embedded server version must be an exact semantic version")
    checksums = temporal.get("archive_sha256")
    required_archives = {
        "darwin_amd64",
        "darwin_arm64",
        "linux_amd64",
        "linux_arm64",
        "windows_amd64",
        "windows_arm64",
    }
    if not isinstance(checksums, dict) or set(checksums) != required_archives:
        raise ValueError("Temporal CLI archive checksum matrix is incomplete")
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in checksums.values()):
        raise ValueError("Temporal CLI archive checksum is invalid")

    profiles = lock.get("pull_request_profiles")
    if not isinstance(profiles, dict) or set(profiles) != {f"{number:02d}" for number in range(1, 14)}:
        raise ValueError("PR profile matrix must cover PR 01 through PR 13")
    if any(profile not in ALLOWED_PROFILES for profile in profiles.values()):
        raise ValueError("PR profile matrix contains an unknown profile")
    return lock


def exact_requirements(lock: dict[str, Any]) -> list[str]:
    dependencies = lock["python_dependencies"]
    return [
        f"psycopg[binary]=={dependencies['psycopg']['exact']}",
        f"psycopg-pool=={dependencies['psycopg-pool']['exact']}",
        f"boto3=={dependencies['boto3']['exact']}",
        f"botocore=={dependencies['botocore']['exact']}",
        f"temporalio=={dependencies['temporalio']['exact']}",
    ]


def verify_installed(lock: dict[str, Any]) -> dict[str, str]:
    """Verify that the service process imports the exact SDK campaign, not a broad-extra drift."""
    expected = {
        name: str(entry["exact"]) for name, entry in lock["python_dependencies"].items()
    }
    actual: dict[str, str] = {}
    for name in expected:
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(f"campaign dependency is not installed: {name}") from exc
    if actual != expected:
        raise ValueError(f"installed SDK campaign differs from lock: {actual}, expected {expected}")
    for distribution, module_name in SDK_IMPORT_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            raise ValueError(f"campaign dependency cannot be imported: {distribution}") from exc
    return actual


def temporal_archive_spec(lock: dict[str, Any]) -> dict[str, str]:
    """Select the exact Temporal CLI archive and checksum for the current platform."""
    system_names = {"darwin": "darwin", "linux": "linux", "windows": "windows"}
    system = system_names.get(platform.system().lower())
    machine = (
        platform.machine()
        or os.environ.get("PROCESSOR_ARCHITEW6432")
        or os.environ.get("PROCESSOR_ARCHITECTURE")
        or ""
    ).lower()
    if not machine:
        interpreter_platform = sysconfig.get_platform().lower()
        if "amd64" in interpreter_platform or "x86_64" in interpreter_platform:
            machine = "amd64"
        elif "arm64" in interpreter_platform or "aarch64" in interpreter_platform:
            machine = "arm64"
    architecture = {
        "amd64": "amd64",
        "x86_64": "amd64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }.get(machine)
    if system is None or architecture is None:
        raise ValueError(
            f"Temporal CLI campaign does not support platform {platform.system()}/{machine}"
        )

    temporal = lock["services"]["temporal_cli"]
    version = str(temporal["version"])
    platform_key = f"{system}_{architecture}"
    checksum = temporal["archive_sha256"].get(platform_key)
    if not checksum:
        raise ValueError(f"Temporal CLI checksum is missing for {platform_key}")
    filename = f"temporal_cli_{version.removeprefix('v')}_{system}_{architecture}.tar.gz"
    return {
        "platform": platform_key,
        "version": version,
        "embedded_server": str(temporal["embedded_server"]),
        "filename": filename,
        "sha256": str(checksum),
        "url": f"https://github.com/temporalio/cli/releases/download/{version}/{filename}",
        "executable_name": "temporal.exe" if system == "windows" else "temporal",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _download_checked_archive(*, url: str, destination: Path, expected_sha256: str) -> None:
    temporary = destination.with_name(f"{destination.name}.part")
    request = urllib.request.Request(url, headers={"User-Agent": "monoid-v0.23-ci"})
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_TEMPORAL_ARCHIVE_BYTES:
                    raise ValueError("Temporal CLI archive exceeds the bounded download size")
                digest.update(chunk)
                output.write(chunk)
        observed = digest.hexdigest()
        if observed != expected_sha256:
            raise ValueError(
                f"Temporal CLI archive checksum mismatch: {observed}, expected {expected_sha256}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _extract_temporal_executable(
    *, archive: Path, destination: Path, executable_name: str
) -> None:
    with tarfile.open(archive, mode="r:gz") as bundle:
        candidates = [
            member
            for member in bundle.getmembers()
            if member.isfile() and Path(member.name).name == executable_name
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Temporal CLI archive must contain exactly one regular executable; "
                f"found {[member.name for member in candidates]}"
            )
        member = candidates[0]
        if member.size > MAX_TEMPORAL_ARCHIVE_BYTES:
            raise ValueError("Temporal CLI executable exceeds the bounded extraction size")
        source = bundle.extractfile(member)
        if source is None:
            raise ValueError("Temporal CLI executable cannot be read from the verified archive")
        payload = source.read(MAX_TEMPORAL_ARCHIVE_BYTES + 1)
        if len(payload) != member.size or len(payload) > MAX_TEMPORAL_ARCHIVE_BYTES:
            raise ValueError("Temporal CLI executable size differs from its archive metadata")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.part")
    try:
        temporary.write_bytes(payload)
        if os.name != "nt":
            temporary.chmod(0o755)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_temporal_cli(lock: dict[str, Any], cache_dir: Path) -> dict[str, str]:
    """Download, verify, and extract the exact platform Temporal CLI campaign artifact."""
    spec = temporal_archive_spec(lock)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / spec["filename"]
    if archive.exists():
        observed = _sha256_file(archive)
        if observed != spec["sha256"]:
            raise ValueError(
                f"cached Temporal CLI archive checksum mismatch: {observed}, "
                f"expected {spec['sha256']}"
            )
    else:
        _download_checked_archive(
            url=spec["url"],
            destination=archive,
            expected_sha256=spec["sha256"],
        )

    destination = cache_dir / spec["platform"] / spec["version"] / spec["executable_name"]
    _extract_temporal_executable(
        archive=archive,
        destination=destination,
        executable_name=spec["executable_name"],
    )
    return {
        **spec,
        "archive": str(archive.resolve()),
        "executable": str(destination.resolve()),
    }


def resolve_profile(
    lock: dict[str, Any],
    *,
    head_ref: str,
    labels: Sequence[str],
    requested_profile: str | None = None,
) -> str:
    selected_labels = sorted(PROFILE_LABELS.intersection(labels))
    matched = BRANCH_PATTERN.search(head_ref)
    terminal_expected = CAMPAIGN_TERMINAL_PROFILES.get(head_ref)
    if requested_profile:
        if requested_profile not in ALLOWED_PROFILES:
            raise ValueError(f"unknown manually requested profile: {requested_profile}")
        selected = requested_profile
    elif matched is not None:
        if len(selected_labels) != 1:
            raise ValueError(
                "ready PR must carry exactly one service profile label; "
                f"found {selected_labels}"
            )
        selected = selected_labels[0].split(":", 1)[1]
    elif terminal_expected is not None:
        if len(selected_labels) > 1:
            raise ValueError(
                "campaign terminal PR accepts at most one service profile label; "
                f"found {selected_labels}"
            )
        selected = (
            selected_labels[0].split(":", 1)[1] if selected_labels else terminal_expected
        )
    else:
        if len(selected_labels) > 1:
            raise ValueError(
                "non-campaign PR accepts at most one service profile label; "
                f"found {selected_labels}"
            )
        selected = selected_labels[0].split(":", 1)[1] if selected_labels else "core"

    expected = (
        lock["pull_request_profiles"][matched.group("number")]
        if matched is not None
        else terminal_expected
    )
    if expected is not None and selected != expected:
        raise ValueError(
            f"{head_ref!r} requires ci:{expected}, received ci:{selected}"
        )
    return selected


def _profile_flags(profile: str) -> dict[str, bool]:
    return {
        "run_postgres": profile in {"postgres", "objectstore", "combined"},
        "run_postgres18": profile == "combined",
        "run_objectstore": profile in {"objectstore", "combined"},
        "run_temporal": profile in {"temporal", "combined"},
    }


def _write_github_output(path: Path, values: dict[str, str | bool]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for key, value in values.items():
            rendered = str(value).lower() if isinstance(value, bool) else value
            handle.write(f"{key}={rendered}\n")


def _parse_labels(raw: str) -> list[str]:
    if not raw.strip():
        return []
    loaded = json.loads(raw)
    if loaded is None:
        return []
    if not isinstance(loaded, list) or any(not isinstance(item, str) for item in loaded):
        raise ValueError("labels JSON must be an array of strings")
    return loaded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-lock")
    commands.add_parser("exact-requirements")
    commands.add_parser("verify-installed")

    temporal_cli = commands.add_parser("prepare-temporal-cli")
    temporal_cli.add_argument("--cache-dir", type=Path, required=True)

    resolve = commands.add_parser("resolve-profile")
    resolve.add_argument("--head-ref", required=True)
    resolve.add_argument("--labels-json", default="[]")
    resolve.add_argument("--requested-profile")
    resolve.add_argument("--github-output", type=Path)

    evidence = commands.add_parser("write-evidence")
    evidence.add_argument("--head-sha", required=True)
    evidence.add_argument("--merge-sha", required=True)
    evidence.add_argument("--profile", choices=sorted(ALLOWED_PROFILES), required=True)
    evidence.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    lock = validate_lock()
    if arguments.command == "validate-lock":
        print(f"v0.23 campaign lock valid: {_lock_digest(lock)}")
        return 0
    if arguments.command == "exact-requirements":
        print(" ".join(exact_requirements(lock)))
        return 0
    if arguments.command == "verify-installed":
        print(json.dumps(verify_installed(lock), sort_keys=True))
        return 0
    if arguments.command == "prepare-temporal-cli":
        print(json.dumps(prepare_temporal_cli(lock, arguments.cache_dir), sort_keys=True))
        return 0
    if arguments.command == "resolve-profile":
        profile = resolve_profile(
            lock,
            head_ref=arguments.head_ref,
            labels=_parse_labels(arguments.labels_json),
            requested_profile=arguments.requested_profile or None,
        )
        values: dict[str, str | bool] = {
            "profile": profile,
            "lock_sha256": _lock_digest(lock),
            **_profile_flags(profile),
        }
        if arguments.github_output:
            _write_github_output(arguments.github_output, values)
        print(json.dumps(values, sort_keys=True))
        return 0
    if arguments.command == "write-evidence":
        evidence = {
            "schema_version": 1,
            "campaign": "v0.23",
            "head_sha": arguments.head_sha,
            "merge_sha": arguments.merge_sha,
            "profile": arguments.profile,
            "campaign_lock_sha256": _lock_digest(lock),
            "python_dependencies": {
                name: entry["exact"] for name, entry in lock["python_dependencies"].items()
            },
            "services": lock["services"],
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote v0.23 CI evidence: {arguments.output}")
        return 0
    raise AssertionError(f"unhandled command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
