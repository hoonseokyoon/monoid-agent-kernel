from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from monoid_agent_kernel.conformance.profiles.capability_security import (
    assert_capability_security_lease_admission,
)
from monoid_agent_kernel.core.capability import CapabilityLease, CapabilityRequest, CapabilityVault
from monoid_agent_kernel.core.lease_admission import sanitize_denied_capability_result


@dataclass
class _ReferenceCapabilityHarness:
    vault: CapabilityVault = field(default_factory=CapabilityVault)
    requests: dict[str, CapabilityRequest] = field(default_factory=dict)

    @property
    def harness_id(self) -> str:
        return "reference-capability-vault"

    @property
    def supported_profiles(self) -> tuple[str, ...]:
        return ("capability-security",)

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


def test_reference_capability_vault_satisfies_lease_admission_profile() -> None:
    assert_capability_security_lease_admission(_ReferenceCapabilityHarness())
