"""Packaged historical compatibility fixtures for external conformance suites."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Any


@dataclass(frozen=True)
class CompatibilityFixture:
    fixture_id: str
    artifact: str
    expected_status: str
    payload: dict[str, Any]


def load_compatibility_fixtures() -> tuple[CompatibilityFixture, ...]:
    resource = files("monoid_agent_kernel.conformance").joinpath(
        "fixtures", "compatibility-v1.json"
    )
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "monoid.conformance-fixtures.v1":
        raise ValueError("unsupported conformance fixture schema")
    return tuple(
        CompatibilityFixture(
            fixture_id=str(item["fixture_id"]),
            artifact=str(item["artifact"]),
            expected_status=str(item["expected_status"]),
            payload=dict(item["payload"]),
        )
        for item in payload["fixtures"]
    )
