from __future__ import annotations

from pathlib import Path

from monoid_agent_kernel.conformance.profiles.multi_agent import (
    assert_multi_agent_backend_boundary_profile,
    assert_multi_agent_backend_capability_boundary_profile,
    assert_multi_agent_shared_revocation_profile,
)
from monoid_agent_kernel.reference.conformance import ReferenceBackendHarness, ReferenceCapabilityHarness


def test_reference_capability_vault_satisfies_multi_agent_revocation_profile() -> None:
    assert_multi_agent_shared_revocation_profile(ReferenceCapabilityHarness())


def test_reference_backend_satisfies_multi_agent_boundary_profile(tmp_path: Path) -> None:
    assert_multi_agent_backend_boundary_profile(ReferenceBackendHarness(tmp_path))


def test_reference_backend_satisfies_multi_agent_capability_boundary_profile(tmp_path: Path) -> None:
    harness = ReferenceBackendHarness(tmp_path)
    assert_multi_agent_backend_capability_boundary_profile(harness)
    assert harness.gated_provider.calls == 0
