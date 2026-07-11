from __future__ import annotations

from pathlib import Path

import pytest

from support.test_tiers import primary_tier_for_path

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/test_tokens.py", "unit"),
        ("tests/test_workspace_contract.py", "contract"),
        ("tests/test_public_surface.py", "contract"),
        ("tests/conformance/test_reference_full_profile.py", "contract"),
        ("tests/test_backend.py", "integration"),
        ("tests/test_studio_sessions.py", "integration"),
        ("tests/test_mcp_gateway.py", "integration"),
    ],
)
def test_primary_tier_policy(path: str, expected: str) -> None:
    assert primary_tier_for_path(Path(path)) == expected
