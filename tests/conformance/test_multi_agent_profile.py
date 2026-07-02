from __future__ import annotations

from conformance.capability_harness import ReferenceCapabilityHarness
from monoid_agent_kernel.conformance.profiles.multi_agent import (
    assert_multi_agent_shared_revocation_profile,
)


def test_reference_capability_vault_satisfies_multi_agent_revocation_profile() -> None:
    assert_multi_agent_shared_revocation_profile(ReferenceCapabilityHarness())
