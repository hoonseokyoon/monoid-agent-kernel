"""Central, enforced primary-tier classification for the repository test suite.

Primary tiers describe the broadest boundary crossed by a test.  They are assigned
from the test module so new tests inherit a stable tier without hundreds of repeated
decorators.  Orthogonal traits (``slow``, ``live``, and ``serial``) remain ordinary
pytest markers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

PRIMARY_TIERS = frozenset({"unit", "contract", "integration"})

_CONTRACT_MODULES = frozenset(
    {
        "test_event_data_schema.py",
        "test_external_agent_envelope_properties.py",
        "test_inbox_outbox_properties.py",
        "test_namespace_compat.py",
        "test_public_surface.py",
        "test_wire_validation.py",
    }
)

_INTEGRATION_MODULES = frozenset(
    {
        "test_async_loop.py",
        "test_capability.py",
        "test_cli_and_openai.py",
        "test_examples.py",
        "test_gateway_provider.py",
        "test_hitl.py",
        "test_inbox.py",
        "test_jobs.py",
        "test_llm_gateway_backend.py",
        "test_llm_gateway_stream.py",
        "test_mcp.py",
        "test_mcp_gateway.py",
        "test_memory.py",
        "test_outbox.py",
        "test_proposal_package.py",
        "test_shell.py",
        "test_web_gateway.py",
    }
)

_SERIAL_CONTRACT_MODULES = frozenset(
    {
        "test_checkpoint_store_contract.py",
        "test_lease_store_contract.py",
    }
)


def primary_tier_for_path(path: Path) -> str:
    """Return the single primary tier assigned to a collected test module."""
    normalized = path.as_posix()
    name = path.name
    if "/tests/conformance/" in f"/{normalized}" or name.endswith("_contract.py"):
        return "contract"
    if name.startswith("test_backend") or name.startswith("test_studio"):
        return "integration"
    if name in _CONTRACT_MODULES:
        return "contract"
    if name in _INTEGRATION_MODULES:
        return "integration"
    return "unit"


def requires_serial_execution(path: Path) -> bool:
    """Return whether a module owns threaded/process/service lifecycle."""
    normalized = path.as_posix()
    return (
        primary_tier_for_path(path) == "integration"
        or "/tests/conformance/" in f"/{normalized}"
        or path.name in _SERIAL_CONTRACT_MODULES
    )


def classify_items(items: list[Any]) -> list[str]:
    """Assign tiers and return collection errors for conflicting declarations."""
    errors: list[str] = []
    for item in items:
        expected = primary_tier_for_path(Path(str(item.path)))
        declared = {name for name in PRIMARY_TIERS if list(item.iter_markers(name=name))}
        if len(declared) > 1:
            errors.append(f"{item.nodeid}: multiple primary tiers declared: {sorted(declared)}")
            continue
        if declared and expected not in declared:
            errors.append(
                f"{item.nodeid}: declares {next(iter(declared))!r}, "
                f"but module policy assigns {expected!r}"
            )
            continue
        if not declared:
            item.add_marker(getattr(pytest.mark, expected))
        # Integration tests cross real component/process/thread boundaries.  Keep the
        # required PR shard deterministic; parallelism belongs to unit/contract tests.
        if requires_serial_execution(Path(str(item.path))) and not list(
            item.iter_markers(name="serial")
        ):
            item.add_marker(pytest.mark.serial)
    return errors
