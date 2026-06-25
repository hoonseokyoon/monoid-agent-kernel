"""Reference capability brokers.

These show how the `CapabilityBroker` contract is satisfied by real strategies. They are
examples, not part of the supported surface (like the other `reference.*` services).

- `GatewayCapabilityBroker` — mints a scoped, short-lived gateway token as the lease's
  ``token_ref``. This is the "absorb the gateway" path: the LLM/web gateway tokens the runner
  already uses ARE capability leases; this broker issues the same kind of token for any
  capability, so the proven "secret behind a gateway, runner holds a scoped token" pattern
  generalizes under one contract. The gateway (not the core) resolves the token to the secret.
- `DenyAllBroker` — refuses everything (useful as a safe default / for tests).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from native_agent_runner.core.capability import (
    CapabilityDenial,
    CapabilityGrant,
    CapabilityLease,
    CapabilityPending,
    CapabilityRequest,
)
from native_agent_runner.reference._shared.tokens import TokenManager


@dataclass
class GatewayCapabilityBroker:
    """Grants a capability by minting a scoped gateway token (the lease ``token_ref``). The
    token encodes the run identity + the capability/scope in its claims; a capability gateway
    verifies it and applies the secret. The core never sees the secret."""

    token_manager: TokenManager
    tenant_id: str
    user_id: str
    audience: str = "csp.capability-gateway"

    def request(self, req: CapabilityRequest) -> CapabilityGrant:
        ttl = req.ttl_seconds or 600
        token = self.token_manager.issue(
            kind="capability",
            audience=self.audience,
            run_id=req.run_id,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            ttl_s=ttl,
            metadata={"capability": req.capability, "scope": dict(req.scope)},
        )
        return CapabilityLease(
            capability=req.capability,
            token_ref=token,
            expires_at=time.time() + ttl,
            scope=dict(req.scope),
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
