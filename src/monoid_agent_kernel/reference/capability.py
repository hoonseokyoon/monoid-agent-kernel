"""Reference capability brokers.

These show how the `CapabilityBroker` contract is satisfied by real strategies. They are
reference examples like the other `reference.*` services.

- `GatewayCapabilityBroker` — mints a scoped, short-lived gateway token as the lease's
  ``token_ref``. This is the "absorb the gateway" path: the LLM/web gateway tokens the kernel
  already uses ARE capability leases; this broker issues the same kind of token for any
  capability, so the proven "secret behind a gateway, kernel holds a scoped token" pattern
  generalizes under one contract. The gateway (not the core) resolves the token to the secret.
- `DenyAllBroker` — refuses everything (useful as a safe default / for tests).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from monoid_agent_kernel.core.capability import (
    CapabilityDenial,
    CapabilityGrant,
    CapabilityLease,
    CapabilityPending,
    CapabilityRequest,
)
from monoid_agent_kernel.reference._shared.tokens import TokenKind, TokenManager

# Capabilities whose lease token must be accepted by an EXISTING gateway (not the generic capability
# gateway): the kernel's web tools post to the web gateway, which verifies a ``web_gateway``/``csp.web-
# gateway`` token. Minting that exact token as the lease handle is the "the gateway token IS a
# capability lease" absorption, made concrete — the web path needs no separate credential.
_GATEWAY_TOKEN_KINDS: dict[str, tuple[TokenKind, str]] = {
    "web.search": ("web_gateway", "csp.web-gateway"),
    "web.fetch": ("web_gateway", "csp.web-gateway"),
    "web.context": ("web_gateway", "csp.web-gateway"),
}


@dataclass
class GatewayCapabilityBroker:
    """Grants a capability by minting a scoped gateway token (the lease ``token_ref``). The
    token encodes the run identity + the capability/scope in its claims; a capability gateway
    verifies it and applies the secret. The core never sees the secret."""

    token_manager: TokenManager
    tenant_id: str
    user_id: str
    audience: str = "csp.capability-gateway"
    # Per-capability override of the minted token's (kind, audience), so a lease for a capability
    # backed by an existing gateway (e.g. ``web.*`` -> the web gateway) is a token that gateway
    # already accepts. Defaults to the web mapping; other capabilities use ``("capability", audience)``.
    gateway_token_kinds: dict[str, tuple[TokenKind, str]] = field(
        default_factory=lambda: dict(_GATEWAY_TOKEN_KINDS)
    )

    def request(self, req: CapabilityRequest) -> CapabilityGrant:
        ttl = req.ttl_seconds or 600
        kind, audience = self.gateway_token_kinds.get(req.capability, ("capability", self.audience))
        scope = dict(req.scope)
        if req.binding_id and req.capability in self.gateway_token_kinds:
            scope.setdefault("binding_id", req.binding_id)
        token = self.token_manager.issue(
            kind=kind,
            audience=audience,
            run_id=req.run_id,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            ttl_s=ttl,
            metadata={"capability": req.capability, "scope": scope},
        )
        return CapabilityLease(
            capability=req.capability,
            token_ref=token,
            expires_at=time.time() + ttl,
            scope=scope,
        )


@dataclass
class DenyAllBroker:
    """Refuses every request. A safe default before a real policy broker is wired."""

    reason: str = "no capability broker policy configured"

    def request(self, req: CapabilityRequest) -> CapabilityGrant:
        return CapabilityDenial(capability=req.capability, reason=self.reason)


@dataclass
class HumanEscalationBroker:
    """Escalates every request to an external (human/Daemon) decision instead of granting
    synchronously: returns `CapabilityPending`, so the loop parks the run on a `capability`
    hosted-task and resumes once the grant is reported via `report_task_result`. The simplest
    async broker — a real policy broker would auto-grant low-risk capabilities, deny forbidden
    ones, and escalate only the sensitive ones (the three-way outcome is the point)."""

    prompt_template: str = "Approve capability '{capability}' (scope={scope})?"

    def request(self, req: CapabilityRequest) -> CapabilityGrant:
        prompt = self.prompt_template.format(capability=req.capability, scope=req.scope)
        return CapabilityPending(request=req, prompt=prompt)
