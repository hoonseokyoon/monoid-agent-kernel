from __future__ import annotations

from monoid_agent_kernel.conformance.profiles.provider_gateway import assert_provider_gateway_profile
from monoid_agent_kernel.reference.conformance import ReferenceGatewayHarness


def test_reference_web_gateway_satisfies_provider_gateway_profile() -> None:
    assert_provider_gateway_profile(ReferenceGatewayHarness())
