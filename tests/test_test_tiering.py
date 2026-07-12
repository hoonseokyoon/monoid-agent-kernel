from __future__ import annotations

from pathlib import Path

import pytest

from support.test_tiers import primary_tier_for_path, requires_serial_execution

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
        ("tests/test_capability.py", "integration"),
        ("tests/test_inbox.py", "integration"),
        ("tests/test_outbox.py", "integration"),
        ("tests/test_memory.py", "integration"),
        ("tests/test_proposal_package.py", "integration"),
    ],
)
def test_primary_tier_policy(path: str, expected: str) -> None:
    assert primary_tier_for_path(Path(path)) == expected


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_backend.py",
        "tests/test_capability.py",
        "tests/test_checkpoint_store_contract.py",
        "tests/test_command_store_contract.py",
        "tests/conformance/test_reference_full_profile.py",
    ],
)
def test_threaded_and_service_backed_modules_are_serial(path: str) -> None:
    assert requires_serial_execution(Path(path)) is True
