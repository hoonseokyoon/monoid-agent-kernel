from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from monoid_agent_kernel.core.capability import CapabilityLease, CapabilityRequest, CapabilityVault
from monoid_agent_kernel.core.lease_admission import sanitize_denied_capability_result


@dataclass
class ReferenceCapabilityHarness:
    vault: CapabilityVault = field(default_factory=CapabilityVault)
    requests: dict[str, CapabilityRequest] = field(default_factory=dict)

    @property
    def harness_id(self) -> str:
        return "reference-capability-vault"

    @property
    def supported_profiles(self) -> tuple[str, ...]:
        return ("capability-security", "multi-agent")

    def request_capability(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = CapabilityRequest(
            capability=str(payload.get("capability") or ""),
            scope=dict(payload.get("scope") or {}),
        )
        self.requests[request.request_id] = request
        return request.to_json()

    def grant_capability(self, request_id: str, lease: dict[str, Any]) -> dict[str, Any]:
        admitted = self.vault.admit(self.requests[request_id], CapabilityLease.from_json(dict(lease)))
        return admitted.to_json()

    def deny_capability(self, request_id: str, result: dict[str, Any]) -> dict[str, Any]:
        del request_id
        return sanitize_denied_capability_result(result, reason="denied by profile")

    def revoke_capability(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.vault.revoke(
            capability=(str(payload["capability"]) if payload.get("capability") else None),
            lease_id=(str(payload["lease_id"]) if payload.get("lease_id") else None),
            before=(float(payload["before"]) if payload.get("before") is not None else None),
        )

    def token_for(self, capability: str, *, now: float) -> str | None:
        return self.vault.token_for(capability, now=now)

    def valid_lease(self, capability: str, scope: dict[str, Any], *, now: float) -> dict[str, Any] | None:
        lease = self.vault.get_valid(capability, dict(scope), now=now)
        return lease.to_json() if lease is not None else None

    def export_revocations(self) -> dict[str, Any]:
        return self.vault.export_revocations()

    def import_revocations(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.vault.import_revocations(
            lease_ids=list(payload.get("revoked_lease_ids") or ()),
            capabilities=list(payload.get("revoked_capabilities") or ()),
            before=float(payload.get("revoked_before") or 0.0),
            all_revoked=bool(payload.get("revoked_all", False)),
        )
        return self.vault.export_revocations()

    def fork_child(self) -> ReferenceCapabilityHarness:
        return ReferenceCapabilityHarness(vault=self.vault.fork_for_child())
