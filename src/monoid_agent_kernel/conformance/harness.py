"""Protocols implemented by conformance test adapters."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

JsonObject = Mapping[str, Any]


@runtime_checkable
class ConformanceHarness(Protocol):
    """Common metadata exposed by every conformance harness."""

    @property
    def harness_id(self) -> str:
        """Stable identifier used in test output."""

    @property
    def supported_profiles(self) -> Sequence[str]:
        """Profile ids the harness intends to run."""


@runtime_checkable
class BackendHarness(ConformanceHarness, Protocol):
    """Raw backend operations kept for compatibility and custom harnesses."""

    def submit_run(self, request: JsonObject) -> JsonObject:
        """Submit a run and return a handle containing at least run id and token."""

    def status(self, run_id: str, token: str) -> JsonObject:
        """Return the backend status projection for a run."""

    def events(self, run_id: str, token: str, *, from_seq: int = 0, limit: int | None = None) -> JsonObject:
        """Return a page of run events."""

    def descendant_events(
        self,
        run_id: str,
        token: str,
        descendant_run_id: str,
        *,
        from_seq: int = 0,
        limit: int | None = None,
    ) -> JsonObject:
        """Return a page of events for one authorized descendant run."""

    def diagnostics(self, run_id: str, token: str, *, event_limit: int = 50) -> JsonObject:
        """Return the diagnostics aggregate for a run."""

    def result(self, run_id: str, token: str) -> JsonObject:
        """Return the backend run result projection."""

    def runtime_config(self, run_id: str, token: str) -> JsonObject:
        """Return the current runtime config projection for a run."""

    def replace_runtime_config(
        self,
        run_id: str,
        token: str,
        config: JsonObject,
        *,
        expected_version: int,
        issuer: str,
        reason: str,
    ) -> JsonObject:
        """Replace the runtime config for a live run."""

    def resume_run(self, run_id: str, token: str) -> JsonObject:
        """Materialize and resume one run from durable recovery metadata."""

    def recover_runs(self) -> Sequence[str]:
        """Recover all discoverable parked runs for this harness instance."""

    def restart(self, *, local_state: str = "same") -> BackendHarness:
        """Return a fresh harness instance over the same durable state."""

    def task_result(self, run_id: str, token: str, task_id: str) -> JsonObject:
        """Return the stored result for one backend task."""

    def dispatch(self, command: JsonObject) -> JsonObject:
        """Dispatch one backend control command."""


@runtime_checkable
class ToolAgentHarness(ConformanceHarness, Protocol):
    """Tool-agent behavior cases."""

    def run_tool_surface_admission_case(self) -> JsonObject:
        """Run a tool surface admission case with a denied unavailable/quota-limited tool."""

    def run_generic_ask_approval_case(self) -> JsonObject:
        """Run generic authorization='ask' approval, denial, and stale replay cases."""


@runtime_checkable
class ControlPlaneHarness(ConformanceHarness, Protocol):
    """Control-plane behavior cases."""

    def run_control_decision_case(self) -> JsonObject:
        """Run approve/deny decision handling and decision audit behavior."""

    def run_control_audit_sequence_case(self) -> JsonObject:
        """Run authorized, failed, unauthorized, and terminal audit sequencing behavior."""


@runtime_checkable
class DurableRunnerHarness(ConformanceHarness, Protocol):
    """Durable-runner behavior cases."""

    def run_event_sequence_case(self) -> JsonObject:
        """Run event sequencing and diagnostics projection behavior."""

    def run_recovery_metadata_case(self) -> JsonObject:
        """Run recovery metadata and restart materialization behavior."""

    def run_subagent_diagnostics_case(self) -> JsonObject:
        """Run subagent diagnostics summary behavior."""


@runtime_checkable
class MultiAgentBackendHarness(ConformanceHarness, Protocol):
    """Backend-visible multi-agent behavior cases."""

    def run_subagent_boundary_case(self) -> JsonObject:
        """Run subagent identity, trace, diagnostics, and accounting behavior."""

    def run_subagent_capability_boundary_case(self) -> JsonObject:
        """Run child capability-boundary behavior after parent revocation."""


@runtime_checkable
class SideEffectHarness(ConformanceHarness, Protocol):
    """Backend operations used by side-effect-tool-agent profiles."""

    def run_outbox_dispatched_case(self) -> JsonObject:
        """Run a strict outbox side-effect case that stages and dispatches one request."""

    def run_pending_recovery_case(self) -> JsonObject:
        """Run a strict outbox side-effect case whose pending request survives restart."""

    def run_strict_rejected_case(self) -> JsonObject:
        """Run a strict external side-effect case rejected before handler execution."""

    def run_idempotent_inline_case(self) -> JsonObject:
        """Run an idempotent inline external side-effect case."""


@runtime_checkable
class MessageFabricHarness(ConformanceHarness, Protocol):
    """Backend operations used by external agent message-fabric profiles."""

    def run_two_peer_exchange_case(self) -> JsonObject:
        """Run a two-peer exchange over the external-agent message fabric."""

    def run_malformed_envelope_case(self) -> JsonObject:
        """Run a malformed external-agent envelope rejection case."""

    def run_duplicate_after_restart_case(self) -> JsonObject:
        """Run a duplicate message case that survives restart."""

    def run_peer_unavailable_case(self) -> JsonObject:
        """Run a peer-unavailable case that leaves a retryable pending request."""


@runtime_checkable
class GatewayHarness(ConformanceHarness, Protocol):
    """Gateway operation used by provider-gateway profiles."""

    def call_gateway(
        self,
        capability: str,
        payload: JsonObject,
        *,
        signed_capability: str | None = None,
        signed_scope: JsonObject | None = None,
    ) -> JsonObject:
        """Call one gateway capability with a normalized payload."""


@runtime_checkable
class CapabilityHarness(ConformanceHarness, Protocol):
    """Capability operations used by capability-security profiles."""

    def request_capability(self, payload: JsonObject) -> JsonObject:
        """Create or simulate one capability request."""

    def grant_capability(self, request_id: str, lease: JsonObject) -> JsonObject:
        """Grant a capability request with a lease payload."""

    def deny_capability(self, request_id: str, result: JsonObject) -> JsonObject:
        """Deny a capability request with a result payload."""

    def revoke_capability(self, payload: JsonObject) -> JsonObject:
        """Revoke capabilities according to the payload."""

    def token_for(self, capability: str, *, now: float) -> str | None:
        """Return the currently usable token handle for one capability."""

    def valid_lease(self, capability: str, scope: JsonObject, *, now: float) -> JsonObject | None:
        """Return a currently valid lease for one capability and scope."""

    def export_revocations(self) -> JsonObject:
        """Return serialized revocation state."""

    def import_revocations(self, payload: JsonObject) -> JsonObject:
        """Merge serialized revocation state and return the current export."""

    def fork_child(self) -> CapabilityHarness:
        """Return a child capability harness sharing revocation state."""
