from __future__ import annotations

import pytest

from monoid_agent_kernel.conformance import (
    BackendHarness,
    CapabilityHarness,
    GatewayHarness,
    PROFILES,
    PROFILE_BY_ID,
    get_profile,
)


EXPECTED_PROFILE_IDS = (
    "minimal-agent",
    "tool-agent",
    "durable-runner",
    "control-plane",
    "capability-security",
    "provider-gateway",
    "multi-agent",
    "reference-full",
)
EXPECTED_RULE_IDS = {
    "OR-01-SCOPE-RELATION",
    "OR-02-CAPABILITY-BOUNDARY",
    "OR-03-LEASE-ADMISSION",
    "OR-04-REVOCATION-SCOPE",
    "OR-05-EVENT-SEQUENCING",
    "OR-06-CONTROL-AUDIT",
    "OR-07-DURABLE-METADATA",
    "OR-08-PROVIDER-CAPS",
    "OR-09-SUBAGENT-BOUNDARY",
    "OR-10-TOOL-SURFACE-ADMISSION",
    "OR-11-GENERIC-ASK-APPROVAL",
    "OR-12-DURABLE-SIDE-EFFECT",
}


def test_phase_1s_profile_metadata_is_registered() -> None:
    assert tuple(profile.profile_id for profile in PROFILES) == EXPECTED_PROFILE_IDS
    assert set(PROFILE_BY_ID) == set(EXPECTED_PROFILE_IDS)


def test_phase_1s_profile_rule_ids_are_known() -> None:
    declared_rule_ids = {rule_id for profile in PROFILES for rule_id in profile.rule_ids}

    assert declared_rule_ids == EXPECTED_RULE_IDS


@pytest.mark.parametrize("profile_id", EXPECTED_PROFILE_IDS)
def test_get_profile_returns_registered_metadata(profile_id: str) -> None:
    profile = get_profile(profile_id)

    assert profile.profile_id == profile_id
    assert profile.title
    assert profile.summary
    assert profile.harnesses


def test_harness_protocols_are_importable() -> None:
    assert BackendHarness.__name__ == "BackendHarness"
    assert GatewayHarness.__name__ == "GatewayHarness"
    assert CapabilityHarness.__name__ == "CapabilityHarness"
