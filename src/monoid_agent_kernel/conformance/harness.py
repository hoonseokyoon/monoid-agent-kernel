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
    """Backend operations used by durable-runner and control-plane profiles."""

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
