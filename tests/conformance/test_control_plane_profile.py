from __future__ import annotations

from pathlib import Path

from monoid_agent_kernel.conformance.profiles.control_plane import (
    assert_control_plane_audit_sequence_profile,
    assert_control_plane_decision_profile,
)
from monoid_agent_kernel.reference.conformance import ReferenceBackendHarness


def test_reference_backend_satisfies_control_plane_decision_profile(tmp_path: Path) -> None:
    assert_control_plane_decision_profile(ReferenceBackendHarness(tmp_path))


def test_reference_backend_satisfies_control_plane_audit_sequence_profile(tmp_path: Path) -> None:
    assert_control_plane_audit_sequence_profile(ReferenceBackendHarness(tmp_path))
