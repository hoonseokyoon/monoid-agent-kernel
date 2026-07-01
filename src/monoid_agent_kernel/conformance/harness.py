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

    def diagnostics(self, run_id: str, token: str, *, event_limit: int = 50) -> JsonObject:
        """Return the diagnostics aggregate for a run."""

    def dispatch(self, command: JsonObject) -> JsonObject:
        """Dispatch one backend control command."""


@runtime_checkable
class GatewayHarness(ConformanceHarness, Protocol):
    """Gateway operation used by provider-gateway profiles."""

    def call_gateway(self, capability: str, payload: JsonObject) -> JsonObject:
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
