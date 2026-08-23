"""Validate the built release wheel with only the Python standard library."""

from __future__ import annotations

import json
import re
import sys
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile


EXPECTED_VERSION = "0.22.0"
EXPECTED_BASE_DEPENDENCIES = {"click", "jsonschema", "pathspec", "pydantic"}
REQUIRED_MEMBERS = {
    "monoid_agent_kernel/core/authority.py",
    "monoid_agent_kernel/core/model_invocation.py",
    "monoid_agent_kernel/core/outcome.py",
    "monoid_agent_kernel/hosting/__init__.py",
    "monoid_agent_kernel/hosting/commit_results.py",
    "monoid_agent_kernel/hosting/contracts.py",
    "monoid_agent_kernel/adapters/__init__.py",
    "monoid_agent_kernel/adapters/postgres/__init__.py",
    "monoid_agent_kernel/adapters/object_store/__init__.py",
    "monoid_agent_kernel/adapters/temporal/__init__.py",
    "monoid_agent_kernel/conformance/fixtures/compatibility-v1.json",
}
FORBIDDEN_VENDORED_PACKAGES = {
    "boto3",
    "botocore",
    "dbos",
    "psycopg",
    "psycopg2",
    "redis",
    "temporalio",
}
EXPECTED_DURABLE_EXTRA_DEPENDENCIES = {
    "postgres": {"psycopg", "psycopg-pool"},
    "object-store-s3": {"boto3"},
    "temporal": {"temporalio"},
    "durable-host": {"psycopg", "psycopg-pool", "boto3", "temporalio"},
}


def _requirement_name(requirement: str) -> str:
    matched = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    if matched is None:
        raise ValueError(f"cannot parse wheel requirement: {requirement!r}")
    return matched.group(1).lower().replace("_", "-")


def audit_wheel(wheel_path: Path) -> None:
    with ZipFile(wheel_path) as archive:
        members = set(archive.namelist())
        missing = sorted(REQUIRED_MEMBERS - members)
        if missing:
            raise ValueError(f"release wheel is missing required members: {missing}")

        top_level = {member.split("/", 1)[0].lower() for member in members if "/" in member}
        vendored = sorted(FORBIDDEN_VENDORED_PACKAGES & top_level)
        if vendored:
            raise ValueError(f"release wheel vendors platform packages: {vendored}")

        metadata_members = sorted(
            member for member in members if member.endswith(".dist-info/METADATA")
        )
        if len(metadata_members) != 1:
            raise ValueError(f"release wheel must contain one METADATA file: {metadata_members}")
        metadata = BytesParser().parsebytes(archive.read(metadata_members[0]))
        if metadata["Version"] != EXPECTED_VERSION:
            raise ValueError(
                f"release wheel version is {metadata['Version']!r}, expected {EXPECTED_VERSION!r}"
            )

        requirements = metadata.get_all("Requires-Dist", [])
        base_requirements = [item for item in requirements if "extra ==" not in item.lower()]
        base_names = {_requirement_name(item) for item in base_requirements}
        if base_names != EXPECTED_BASE_DEPENDENCIES:
            raise ValueError(
                "release wheel base dependencies differ from the v0.22 boundary: "
                f"{sorted(base_names)}"
            )
        forbidden_base = sorted(FORBIDDEN_VENDORED_PACKAGES & base_names)
        if forbidden_base:
            raise ValueError(f"platform packages became base dependencies: {forbidden_base}")

        provided_extras = set(metadata.get_all("Provides-Extra", []))
        missing_extras = sorted(EXPECTED_DURABLE_EXTRA_DEPENDENCIES.keys() - provided_extras)
        if missing_extras:
            raise ValueError(f"release wheel is missing durable extras: {missing_extras}")
        durable_requirements: dict[str, set[str]] = {
            extra: set() for extra in EXPECTED_DURABLE_EXTRA_DEPENDENCIES
        }
        for requirement in requirements:
            matched_extra = re.search(r"extra\s*==\s*['\"]([^'\"]+)['\"]", requirement)
            if matched_extra and matched_extra.group(1) in durable_requirements:
                durable_requirements[matched_extra.group(1)].add(_requirement_name(requirement))
        if durable_requirements != EXPECTED_DURABLE_EXTRA_DEPENDENCIES:
            raise ValueError(
                "release wheel durable extra dependencies differ from the v0.23 boundary: "
                f"{durable_requirements}"
            )

        fixture = json.loads(
            archive.read(
                "monoid_agent_kernel/conformance/fixtures/compatibility-v1.json"
            ).decode("utf-8")
        )
        fixture_ids = {item["fixture_id"] for item in fixture["fixtures"]}
        required_fixture_ids = {
            "checkpoint-current-v1",
            "checkpoint-v021-cancelled-v1",
            "checkpoint-v022-additive-v1",
            "terminal-outcome-current-v1",
            "model-invocation-current-v1",
        }
        missing_fixtures = sorted(required_fixture_ids - fixture_ids)
        if missing_fixtures:
            raise ValueError(f"release wheel is missing compatibility fixtures: {missing_fixtures}")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise SystemExit("usage: release_wheel_audit.py WHEEL_OR_DIRECTORY")
    target = Path(arguments[0])
    wheels = sorted(target.glob("*.whl")) if target.is_dir() else [target]
    if len(wheels) != 1 or not wheels[0].is_file():
        raise SystemExit(f"expected exactly one release wheel, found: {wheels}")
    try:
        audit_wheel(wheels[0])
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"release wheel audit passed: {wheels[0].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
