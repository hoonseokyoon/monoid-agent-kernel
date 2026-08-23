"""Validate and report the reproducible v0.23 CI/service campaign configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "tests/service/campaign-lock.json"
COMPOSE_PATH = ROOT / "tests/service/compose.yml"
PYPROJECT_PATH = ROOT / "pyproject.toml"
ALLOWED_PROFILES = frozenset({"core", "postgres", "objectstore", "temporal", "combined"})
PROFILE_LABELS = frozenset(f"ci:{profile}" for profile in ALLOWED_PROFILES)
BRANCH_PATTERN = re.compile(r"(?:^|/)v0[.]23-pr(?P<number>[0-9]{2})(?:-|$)")


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
        f"temporalio=={dependencies['temporalio']['exact']}",
    ]


def verify_installed(lock: dict[str, Any]) -> dict[str, str]:
    """Verify that the service process imports the exact SDK campaign, not a broad-extra drift."""
    import importlib.metadata

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
    return actual


def resolve_profile(
    lock: dict[str, Any],
    *,
    head_ref: str,
    labels: Sequence[str],
    requested_profile: str | None = None,
) -> str:
    selected_labels = sorted(PROFILE_LABELS.intersection(labels))
    if requested_profile:
        if requested_profile not in ALLOWED_PROFILES:
            raise ValueError(f"unknown manually requested profile: {requested_profile}")
        selected = requested_profile
    else:
        if len(selected_labels) != 1:
            raise ValueError(
                "ready PR must carry exactly one service profile label; "
                f"found {selected_labels}"
            )
        selected = selected_labels[0].split(":", 1)[1]

    matched = BRANCH_PATTERN.search(head_ref)
    if matched is None:
        if not requested_profile:
            raise ValueError(f"cannot derive v0.23 PR sequence from head ref: {head_ref!r}")
        return selected
    expected = lock["pull_request_profiles"][matched.group("number")]
    if selected != expected:
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
