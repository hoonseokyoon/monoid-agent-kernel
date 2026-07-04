from __future__ import annotations

from pathlib import Path

from monoid_agent_kernel.conformance.profiles.message_fabric import assert_message_fabric_profile
from monoid_agent_kernel.reference.conformance import ReferenceBackendHarness


def test_message_fabric_profile_passes_against_reference_backend(tmp_path: Path) -> None:
    with ReferenceBackendHarness(tmp_path) as harness:
        assert_message_fabric_profile(harness)
