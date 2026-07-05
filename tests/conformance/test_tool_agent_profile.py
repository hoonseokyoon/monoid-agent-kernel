from __future__ import annotations

from monoid_agent_kernel.conformance.profiles.tool_agent import (
    assert_tool_agent_generic_ask_approval_profile,
    assert_tool_agent_surface_admission_profile,
)
from monoid_agent_kernel.reference.conformance import ReferenceBackendHarness


def test_reference_backend_satisfies_tool_agent_surface_admission_profile(tmp_path) -> None:
    with ReferenceBackendHarness(tmp_path) as harness:
        assert_tool_agent_surface_admission_profile(harness)


def test_reference_backend_satisfies_tool_agent_generic_ask_approval_profile(tmp_path) -> None:
    with ReferenceBackendHarness(tmp_path) as harness:
        assert_tool_agent_generic_ask_approval_profile(harness)
