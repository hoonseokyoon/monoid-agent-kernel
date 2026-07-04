from __future__ import annotations

from monoid_agent_kernel.conformance.profiles.side_effect_tool_agent import (
    assert_side_effect_tool_agent_profile,
)
from monoid_agent_kernel.reference.conformance import ReferenceBackendHarness


def test_reference_backend_satisfies_side_effect_tool_agent_profile(tmp_path) -> None:
    with ReferenceBackendHarness(tmp_path) as harness:
        assert_side_effect_tool_agent_profile(harness)
