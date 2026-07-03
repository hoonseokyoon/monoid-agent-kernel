"""Capability request/lease: the agent asks for scoped, short-lived access; secrets stay out.

The runtime never holds a raw credential. When a tool needs external access (web, email, a
cloud API), it carries a *capability* requirement (declared on its binding). At call time the
loop asks a :class:`CapabilityBroker` for a lease — a scoped, expiring handle (``token_ref``,
never the secret) — and only then runs the tool. This generalizes the gateway-token pattern
(LLM/web access already keep the provider key behind a gateway) into one contract any
capability can use, and makes acquisition on-demand and brokered (auto-grant, policy, or
human escalation) rather than only statically provisioned at run start.

Protocols:
  ``monoid.capability-request.v1`` / ``...capability-lease.v1``

Security invariants the core enforces (see ``CapabilityVault.admit``):
  - the secret never enters the core (a lease carries ``token_ref``, a handle);
  - a grant may only NARROW the requested scope, never widen it (fail-closed);
  - a lease is checked for expiry before every use; an expired lease is re-requested;
  - leases are NOT checkpointed — on restart they are re-brokered (no stale secret on disk).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from monoid_agent_kernel.core.capability_revocation import (
    CapabilityRevocationState,
    apply_capability_revocation,
    export_revocation_state,
    import_revocation_state,
    is_capability_revoked,
    is_lease_revoked,
)
from monoid_agent_kernel.core.lease_admission import validate_lease_admission
from monoid_agent_kernel.core.scope import scope_within
from monoid_agent_kernel.core.wire_validation import (
    parse_bool,
    parse_float,
    parse_str,
    require_object,
)
from monoid_agent_kernel.identifiers import namespaced_id

CAPABILITY_REQUEST_VERSION = namespaced_id("capability-request.v1")
CAPABILITY_LEASE_VERSION = namespaced_id("capability-lease.v1")


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
    # Whether this lease should survive a restart (checkpointed). Sync auto-grants stay ephemeral
    # (False) — re-brokering is cheap and no handle touches disk. A human/policy-approved lease is
    # marked durable so a restart does not re-prompt the approver. The handle (token_ref), never a
    # secret, is what persists.
    durable: bool = False
    # When the lease was minted (epoch seconds). Backs the per-run "revoke everything issued before
    # T" watermark (a bulk cohort kill, à la AWS STS ``aws:TokenIssueTime``). Old checkpoint payloads
    # without it decode to ``0.0`` — safely *before* any watermark, so they fail closed.
    issued_at: float = field(default_factory=time.time)
    # Absolute lifetime ceiling (epoch seconds). Rotation may refresh the lease repeatedly, but never
    # past this — so a one-time human approval cannot be silently auto-extended forever. ``None`` =
    # no ceiling (the default for ephemeral sync grants); a policy/approval broker sets it.
    max_expires_at: float | None = None

    def is_valid(self, now: float) -> bool:
        return now < self.expires_at

    def can_rotate(self, now: float, skew: float) -> bool:
        """True if this lease should be refreshed now: still valid, within ``skew`` seconds of
        expiry, and not yet at its absolute ceiling. Past the ceiling it is left to expire (then the
        normal re-broker / re-escalation path applies) rather than auto-extended."""
        if not self.is_valid(now) or now < self.expires_at - skew:
            return False
        return self.max_expires_at is None or now < self.max_expires_at

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": CAPABILITY_LEASE_VERSION,
            "lease_id": self.lease_id,
            "capability": self.capability,
            "scope": dict(self.scope),
            "expires_at": self.expires_at,
            "token_ref": self.token_ref,
            "durable": self.durable,
            "issued_at": self.issued_at,
            "max_expires_at": self.max_expires_at,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> CapabilityLease:
        payload = require_object(payload, "capability lease")
        protocol = parse_str(payload, "protocol")
        if protocol and protocol != CAPABILITY_LEASE_VERSION:
            raise ValueError("unsupported capability lease protocol")
        max_expires_at = parse_float(
            payload,
            "max_expires_at",
            default=0.0,
            allow_none=True,
        ) if "max_expires_at" in payload else None
        scope = require_object(payload["scope"], "scope") if "scope" in payload else {}
        kwargs: dict[str, Any] = {
            "capability": parse_str(payload, "capability"),
            "token_ref": parse_str(payload, "token_ref"),
            "expires_at": parse_float(payload, "expires_at", default=0.0) or 0.0,
            "scope": dict(scope),
            "durable": parse_bool(payload, "durable", default=False),
            "issued_at": parse_float(payload, "issued_at", default=0.0) or 0.0,
            "max_expires_at": max_expires_at,
        }
        lease_id = parse_str(payload, "lease_id")
        if lease_id:
            kwargs["lease_id"] = lease_id
        return cls(**kwargs)


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


@dataclass
class CapabilityVault:
    """Per-run, in-memory cache of granted leases. Holds only handles (``token_ref``), never
    secrets. Durable (human/policy-approved) leases are checkpointed; ephemeral sync grants are
    not, so they re-broker on restart and no handle for them survives on disk. ``admit`` is the
    core's fail-closed gate: a grant that widens the requested scope is rejected.

    Revocation is an *object-capability caretaker* move: because a tool only ever holds a handle
    that it re-fetches per call (via :meth:`token_for`), revoking is simply the vault refusing to
    hand the handle back. The read path (:meth:`get_valid` / :meth:`token_for`) is **fail-closed**
    against three revocation records — a per-lease set, a per-capability set, and a per-run
    ``issued_before`` watermark (a bulk cohort kill). The gate additionally refuses to *re-broker*
    a revoked capability (see ``AgentLoop._ensure_capability_lease``) so revocation survives even a
    permissive broker."""

    _leases: dict[str, CapabilityLease] = field(default_factory=dict)
    _revocations: CapabilityRevocationState = field(default_factory=CapabilityRevocationState)

    @property
    def _revoked_lease_ids(self) -> set[str]:
        return self._revocations.lease_ids

    @_revoked_lease_ids.setter
    def _revoked_lease_ids(self, value: set[str]) -> None:
        self._revocations.lease_ids = value

    @property
    def _revoked_capabilities(self) -> set[str]:
        return self._revocations.capabilities

    @_revoked_capabilities.setter
    def _revoked_capabilities(self, value: set[str]) -> None:
        self._revocations.capabilities = value

    @property
    def _revoked_before(self) -> float:
        return self._revocations.before

    @_revoked_before.setter
    def _revoked_before(self, value: float) -> None:
        self._revocations.before = value

    @property
    def _revoked_all(self) -> bool:
        return self._revocations.all_revoked

    @_revoked_all.setter
    def _revoked_all(self, value: bool) -> None:
        self._revocations.all_revoked = value

    def _is_revoked(self, lease: CapabilityLease) -> bool:
        return is_lease_revoked(self._revocations, lease)

    def get_valid(self, capability: str, scope: dict[str, Any], *, now: float) -> CapabilityLease | None:
        """Return a cached, non-expired, non-revoked lease that COVERS ``scope`` (the requested
        constraints are within the lease's scope), else ``None``."""
        lease = self._leases.get(capability)
        if lease is None or not lease.is_valid(now) or self._is_revoked(lease):
            return None
        # The cached lease must be at least as broad as what this call needs.
        if not scope_within(scope, lease.scope):
            return None
        return lease

    def token_for(self, capability: str, *, now: float) -> str | None:
        """The ``token_ref`` (access handle) of a currently-valid, non-revoked lease for
        ``capability``, or ``None``. A tool handler reads this (via ``ToolContext.capability_token``)
        to obtain the handle the gate acquired — the handle, never the secret; the edge resolves it.
        Returns ``None`` once revoked: the caretaker has cleared its slot."""
        lease = self._leases.get(capability)
        if lease is None or not lease.is_valid(now) or self._is_revoked(lease):
            return None
        return lease.token_ref

    def admit(self, request: CapabilityRequest, lease: CapabilityLease) -> CapabilityLease:
        """Store a granted lease after enforcing least-privilege (grant scope ⊆ request scope).
        Raises ``ValueError`` if the broker tried to widen scope or grant another capability."""
        validate_lease_admission(request.capability, request.scope, lease.capability, lease.scope)
        self._leases[lease.capability] = lease
        return lease

    def revoke(
        self,
        *,
        capability: str | None = None,
        lease_id: str | None = None,
        before: float | None = None,
    ) -> dict[str, Any]:
        """Record a revocation and return a summary of what was revoked. Three granularities,
        composable in one call:
          - ``capability`` — block this capability for the run, authoritatively (the gate will not
            re-broker it). The primary operator kill switch.
          - ``lease_id`` — invalidate one specific grant (a compromised lease).
          - ``before`` — a watermark: every lease issued before this epoch time is rejected in O(1)
            (a bulk cohort kill).
        Revocation is monotonic and additive — there is no un-revoke (start a fresh lease cohort)."""
        return apply_capability_revocation(
            self._revocations,
            capability=capability,
            lease_id=lease_id,
            before=before,
        )

    def is_capability_revoked(self, capability: str) -> bool:
        """True if this capability is under a per-capability revocation — the gate's hard stop that
        refuses to even re-broker (so revocation cannot be undone by a permissive broker)."""
        return is_capability_revoked(self._revocations, capability)

    def export_durable(self) -> list[dict[str, Any]]:
        """Serialize the leases marked ``durable`` (e.g. human/policy-approved) for the checkpoint.
        Ephemeral sync grants are intentionally excluded — they re-broker on restart, so no handle
        for them ever lands on disk. Expiry is re-checked on use, so an expired lease here is
        harmless (it is filtered by ``get_valid`` after restore)."""
        return [lease.to_json() for lease in self._leases.values() if lease.durable]

    def export_revocations(self) -> dict[str, Any]:
        """Serialize the revocation records for the checkpoint, so a revoked durable lease stays
        dead across a restart (the kill switch must not be forgotten when the run resumes)."""
        return export_revocation_state(self._revocations)

    def fork_for_child(self) -> CapabilityVault:
        """Create a child-run vault with isolated live lease slots and shared revocations.

        Durable grants are copied into the child so approved access survives delegation, while
        ephemeral live leases stay local to each run. Revocations share one state object so an
        operator kill switch in the parent is immediately visible to already-running children.
        """
        child = CapabilityVault(_revocations=self._revocations)
        for lease in self._leases.values():
            if lease.durable:
                child.install(lease)
        return child

    def import_revocations(
        self,
        *,
        lease_ids: list[str] | None = None,
        capabilities: list[str] | None = None,
        before: float = 0.0,
        all_revoked: bool = False,
    ) -> None:
        """Rehydrate revocation records on restore (paired with :meth:`export_revocations`)."""
        import_revocation_state(
            self._revocations,
            lease_ids=lease_ids,
            capabilities=capabilities,
            before=before,
            all_revoked=all_revoked,
        )

    def install(self, lease: CapabilityLease) -> None:
        """Directly install a lease (no scope re-check) — used on restore to rehydrate durable
        leases from a trusted checkpoint. The lease was already scope-checked at grant time."""
        self._leases[lease.capability] = lease


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
