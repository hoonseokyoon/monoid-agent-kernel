from __future__ import annotations

from monoid_agent_kernel.conformance.profiles.capability_security import (
    assert_capability_security_revocation_profile,
)
from monoid_agent_kernel.reference.conformance import ReferenceCapabilityHarness


def test_reference_capability_vault_satisfies_revocation_profile() -> None:
    assert_capability_security_revocation_profile(ReferenceCapabilityHarness())
