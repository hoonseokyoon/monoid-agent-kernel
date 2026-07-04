"""Reference-full profile metadata."""

from __future__ import annotations

from typing import Protocol
from typing import Any, Callable

from monoid_agent_kernel.conformance.harness import (
    BackendHarness,
    CapabilityHarness,
    GatewayHarness,
    MessageFabricHarness,
    SideEffectHarness,
)

from .capability_security import (
    assert_capability_security_lease_admission,
    assert_capability_security_revocation_profile,
)
from .control_plane import (
    assert_control_plane_audit_sequence_profile,
    assert_control_plane_decision_profile,
)
from .durable_runner import (
    assert_durable_runner_event_sequence_profile,
    assert_durable_runner_recovery_metadata_profile,
    assert_durable_runner_subagent_diagnostics_profile,
)
from ._metadata import ProfileMetadata
from .message_fabric import assert_message_fabric_profile
from .multi_agent import (
    assert_multi_agent_backend_boundary_profile,
    assert_multi_agent_backend_capability_boundary_profile,
    assert_multi_agent_shared_revocation_profile,
)
from .provider_gateway import assert_provider_gateway_profile
from .side_effect_tool_agent import assert_side_effect_tool_agent_profile
from .tool_agent import (
    assert_tool_agent_generic_ask_approval_profile,
    assert_tool_agent_surface_admission_profile,
)

PROFILE = ProfileMetadata(
    profile_id="reference-full",
    title="Reference Full",
    summary="Bundled Reference services and Studio smoke path across current operational rules.",
    rule_ids=(
        "OR-01-SCOPE-RELATION",
        "OR-02-CAPABILITY-BOUNDARY",
        "OR-03-LEASE-ADMISSION",
        "OR-04-REVOCATION-SCOPE",
        "OR-05-EVENT-SEQUENCING",
        "OR-06-CONTROL-AUDIT",
        "OR-07-DURABLE-METADATA",
        "OR-08-PROVIDER-CAPS",
        "OR-09-SUBAGENT-BOUNDARY",
        "OR-10-TOOL-SURFACE-ADMISSION",
        "OR-11-GENERIC-ASK-APPROVAL",
        "OR-12-DURABLE-SIDE-EFFECT",
        "OR-13-EXTERNAL-AGENT-ENVELOPE",
    ),
    harnesses=("backend", "capability", "gateway", "side-effect", "message-fabric", "studio"),
)


class ReferenceFullFactory(Protocol):
    def new_backend(self) -> BackendHarness:
        """Return a fresh Reference backend harness."""

    def new_side_effect(self) -> SideEffectHarness:
        """Return a fresh Reference side-effect harness."""

    def new_message_fabric(self) -> MessageFabricHarness:
        """Return a fresh Reference message-fabric harness."""

    def new_capability(self) -> CapabilityHarness:
        """Return a fresh Reference capability harness."""

    def new_gateway(self) -> GatewayHarness:
        """Return a fresh Reference gateway harness."""

    def run_studio_smoke(self) -> dict:
        """Run the Reference Studio smoke path."""


def assert_reference_full_profile(factory: ReferenceFullFactory) -> None:
    """Run the bundled Reference implementation across the current profile set."""
    assert_provider_gateway_profile(factory.new_gateway())

    assert_capability_security_lease_admission(factory.new_capability())
    assert_capability_security_revocation_profile(factory.new_capability())
    assert_multi_agent_shared_revocation_profile(factory.new_capability())

    _run_with_close(factory.new_backend(), assert_control_plane_decision_profile)
    _run_with_close(factory.new_backend(), assert_control_plane_audit_sequence_profile)

    _run_with_close(factory.new_backend(), assert_tool_agent_surface_admission_profile)
    _run_with_close(factory.new_backend(), assert_tool_agent_generic_ask_approval_profile)
    _run_with_close(factory.new_side_effect(), assert_side_effect_tool_agent_profile)
    _run_with_close(factory.new_message_fabric(), assert_message_fabric_profile)

    _run_with_close(factory.new_backend(), assert_durable_runner_event_sequence_profile)
    _run_with_close(factory.new_backend(), assert_durable_runner_recovery_metadata_profile)
    _run_with_close(factory.new_backend(), assert_durable_runner_subagent_diagnostics_profile)

    _run_with_close(factory.new_backend(), assert_multi_agent_backend_boundary_profile)
    _run_with_close(factory.new_backend(), assert_multi_agent_backend_capability_boundary_profile)

    smoke = factory.run_studio_smoke()
    assert smoke["run_id"]


def _run_with_close(harness: Any, assertion: Callable[[Any], None]) -> None:
    try:
        assertion(harness)
    finally:
        close = getattr(harness, "close", None)
        if callable(close):
            close()
