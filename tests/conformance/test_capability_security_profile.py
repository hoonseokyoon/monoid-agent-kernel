from __future__ import annotations

from conformance.capability_harness import ReferenceCapabilityHarness
from monoid_agent_kernel.conformance.profiles.capability_security import (
    assert_capability_security_lease_admission,
)


def test_reference_capability_vault_satisfies_lease_admission_profile() -> None:
    assert_capability_security_lease_admission(ReferenceCapabilityHarness())
