"""Capability request/lease: the agent asks for scoped, short-lived access; secrets stay out.

The runtime never holds a raw credential. When a tool needs external access (web, email, a
cloud API), it carries a *capability* requirement (declared on its binding). At call time the
loop asks a :class:`CapabilityBroker` for a lease — a scoped, expiring handle (``token_ref``,
never the secret) — and only then runs the tool. This generalizes the gateway-token pattern
(LLM/web access already keep the provider key behind a gateway) into one contract any
capability can use, and makes acquisition on-demand and brokered (auto-grant, policy, or
human escalation) rather than only statically provisioned at run start.

Protocols:
  ``native-agent-runner.capability-request.v1`` / ``...capability-lease.v1``

Security invariants the core enforces (see ``CapabilityVault.admit``):
  - the secret never enters the core (a lease carries ``token_ref``, a handle);
  - a grant may only NARROW the requested scope, never widen it (fail-closed);
  - a lease is checked for expiry before every use; an expired lease is re-requested;
  - leases are NOT checkpointed — on restart they are re-brokered (no stale secret on disk).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

CAPABILITY_REQUEST_VERSION = "native-agent-runner.capability-request.v1"
CAPABILITY_LEASE_VERSION = "native-agent-runner.capability-lease.v1"


@dataclass(frozen=True)
class CapabilityRequest:
    """A scoped, time-boxed request for a capability, issued by the core when a tool needs
    access it does not yet hold a lease for."""

    capability: str
    scope: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    binding_id: str = ""
    ttl_seconds: int = 600
    reason: str = ""
    request_id: str = field(default_factory=lambda: f"cap_req_{uuid.uuid4().hex[:12]}")

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": CAPABILITY_REQUEST_VERSION,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "binding_id": self.binding_id,
            "capability": self.capability,
            "scope": dict(self.scope),
            "ttl_seconds": self.ttl_seconds,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CapabilityLease:
    """A granted lease: a scoped, expiring handle to a secret the broker manages. ``token_ref``
    is a reference (e.g. ``secret-ref://…`` or a gateway token), never the raw secret —
    resolution happens at the edge (the gateway/tool), not in the core."""

    capability: str
    token_ref: str
    expires_at: float  # epoch seconds; checked before every use
    scope: dict[str, Any] = field(default_factory=dict)
    lease_id: str = field(default_factory=lambda: f"lease_{uuid.uuid4().hex[:12]}")

    def is_valid(self, now: float) -> bool:
        return now < self.expires_at

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": CAPABILITY_LEASE_VERSION,
            "lease_id": self.lease_id,
            "capability": self.capability,
            "scope": dict(self.scope),
            "expires_at": self.expires_at,
            "token_ref": self.token_ref,
        }


@dataclass(frozen=True)
class CapabilityDenial:
    """A broker's refusal to grant. ``retryable`` hints whether a later attempt might succeed
    (e.g. a transient policy backend) versus a hard no."""

    capability: str
    reason: str = ""
    retryable: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "reason": self.reason,
            "retryable": self.retryable,
        }


@dataclass(frozen=True)
class CapabilityPending:
    """The broker cannot grant synchronously — the request must be escalated (e.g. human/Daemon
    approval). The loop parks the run on a ``capability`` hosted-task carrying ``request``; when the
    grant is reported (``report_task_result``) the lease is admitted to the vault and the model
    retries the gated tool. ``prompt`` is a human-facing description for the approval UI."""

    request: CapabilityRequest
    prompt: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "capability": self.request.capability,
            "request_id": self.request.request_id,
            "prompt": self.prompt,
        }


CapabilityGrant = CapabilityLease | CapabilityDenial | CapabilityPending


@runtime_checkable
class CapabilityBroker(Protocol):
    """The seam an integrator (an Agent Daemon / Cell) implements to decide capability access.
    The core only ever *requests*; the broker grants a scoped lease or denies. Transport-neutral:
    an in-process policy object, a gateway-token minter, or a human-escalation broker all fit."""

    def request(self, req: CapabilityRequest) -> CapabilityGrant: ...


def scope_within(inner: dict[str, Any], outer: dict[str, Any]) -> bool:
    """True if ``inner`` scope is no broader than ``outer`` — the least-privilege check the core
    applies to a grant (grant.scope must be ⊆ request.scope). List-valued constraints (e.g.
    ``allowed_domains``) must be a subset; scalar constraints must be equal; a key absent from
    ``outer`` means *unconstrained* there, so any ``inner`` value is within."""
    for key, inner_val in inner.items():
        if key not in outer:
            continue  # outer is unconstrained on this key -> inner is within
        outer_val = outer[key]
        if isinstance(inner_val, (list, tuple, set)) and isinstance(outer_val, (list, tuple, set)):
            if not set(inner_val) <= set(outer_val):
                return False
        elif inner_val != outer_val:
            return False
    return True


@dataclass
class CapabilityVault:
    """Per-run, in-memory cache of granted leases. Holds only handles (``token_ref``), never
    secrets, and is intentionally NOT serialized into checkpoints — on restart leases are
    re-brokered (so a stale handle never survives on disk). ``admit`` is the core's fail-closed
    gate: a grant that widens the requested scope is rejected."""

    _leases: dict[str, CapabilityLease] = field(default_factory=dict)

    def get_valid(self, capability: str, scope: dict[str, Any], *, now: float) -> CapabilityLease | None:
        """Return a cached, non-expired lease that COVERS ``scope`` (the requested constraints
        are within the lease's scope), else ``None``."""
        lease = self._leases.get(capability)
        if lease is None or not lease.is_valid(now):
            return None
        # The cached lease must be at least as broad as what this call needs.
        if not scope_within(scope, lease.scope):
            return None
        return lease

    def token_for(self, capability: str, *, now: float) -> str | None:
        """The ``token_ref`` (access handle) of a currently-valid lease for ``capability``, or
        ``None``. A tool handler reads this (via ``ToolContext.capability_token``) to obtain the
        handle the gate acquired — the handle, never the secret; the edge resolves it."""
        lease = self._leases.get(capability)
        if lease is None or not lease.is_valid(now):
            return None
        return lease.token_ref

    def admit(self, request: CapabilityRequest, lease: CapabilityLease) -> CapabilityLease:
        """Store a granted lease after enforcing least-privilege (grant scope ⊆ request scope).
        Raises ``ValueError`` if the broker tried to widen scope (fail-closed)."""
        if not scope_within(lease.scope, request.scope):
            raise ValueError(
                f"broker granted a wider scope than requested for {request.capability!r}"
            )
        self._leases[lease.capability] = lease
        return lease


@dataclass
class AutoGrantBroker:
    """The zero-config default broker: grants any request, scoped exactly to what was asked,
    with a fixed TTL. Intended for local development and tests — NOT for production (it applies
    no policy). ``token_ref`` is a non-secret placeholder."""

    ttl_seconds: int = 600
    now: Any = None  # optional injectable clock for tests: a callable() -> float

    def request(self, req: CapabilityRequest) -> CapabilityGrant:
        import time

        clock = self.now if callable(self.now) else time.time
        ttl = req.ttl_seconds or self.ttl_seconds
        return CapabilityLease(
            capability=req.capability,
            token_ref=f"auto:{req.capability}",
            expires_at=clock() + ttl,
            scope=dict(req.scope),
        )
