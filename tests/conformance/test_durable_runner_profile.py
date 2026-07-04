from __future__ import annotations

from pathlib import Path

from monoid_agent_kernel.conformance.profiles.durable_runner import (
    assert_durable_runner_event_sequence_profile,
    assert_durable_runner_recovery_metadata_profile,
    assert_durable_runner_subagent_diagnostics_profile,
)
from monoid_agent_kernel.reference.conformance import ReferenceBackendHarness


def test_reference_backend_satisfies_durable_runner_event_sequence_profile(tmp_path: Path) -> None:
    with ReferenceBackendHarness(tmp_path) as harness:
        assert_durable_runner_event_sequence_profile(harness)


def test_reference_backend_satisfies_durable_runner_recovery_metadata_profile(tmp_path: Path) -> None:
    with ReferenceBackendHarness(tmp_path) as harness:
        assert_durable_runner_recovery_metadata_profile(harness)


def test_reference_backend_satisfies_subagent_diagnostics_profile(tmp_path: Path) -> None:
    with ReferenceBackendHarness(tmp_path) as harness:
        assert_durable_runner_subagent_diagnostics_profile(harness)
