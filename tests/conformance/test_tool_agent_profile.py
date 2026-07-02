from __future__ import annotations

from monoid_agent_kernel.conformance.profiles.tool_agent import (
    assert_tool_agent_durable_side_effect_profile,
    assert_tool_agent_generic_ask_approval_profile,
    assert_tool_agent_surface_admission_profile,
)
from monoid_agent_kernel.reference.conformance import ReferenceBackendHarness


def test_reference_backend_satisfies_tool_agent_surface_admission_profile(tmp_path) -> None:
    assert_tool_agent_surface_admission_profile(ReferenceBackendHarness(tmp_path))


def test_reference_backend_satisfies_tool_agent_generic_ask_approval_profile(tmp_path) -> None:
    assert_tool_agent_generic_ask_approval_profile(ReferenceBackendHarness(tmp_path))


def test_reference_backend_satisfies_tool_agent_durable_side_effect_profile(tmp_path) -> None:
    assert_tool_agent_durable_side_effect_profile(ReferenceBackendHarness(tmp_path))
