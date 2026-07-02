from __future__ import annotations

from pathlib import Path

from monoid_agent_kernel.conformance.profiles.reference_full import assert_reference_full_profile
from monoid_agent_kernel.reference.conformance import ReferenceConformanceFactory


def test_reference_full_profile_passes_against_bundled_reference(tmp_path: Path) -> None:
    assert_reference_full_profile(ReferenceConformanceFactory(tmp_path))
