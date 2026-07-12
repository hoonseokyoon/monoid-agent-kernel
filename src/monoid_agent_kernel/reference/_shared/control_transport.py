"""Credential-free durable control transport models shared by Reference profiles.

The legacy SQLite command inbox and optional durable-workflow profiles use the same
principal, receipt, redaction, and command-identity rules without depending on one
another's orchestration machinery.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from monoid_agent_kernel.errors import NativeAgentError
from monoid_agent_kernel.identifiers import namespaced_id

COMMAND_RECEIPT_VERSION = namespaced_id("command-receipt.v1")
CommandStatus = Literal["pending", "claimed", "completed", "failed"]

_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "access_token",
        "authorization",
        "bearer_token",
        "callback_token",
        "credential",
        "password",
        "refresh_token",
        "secret",
        "token",
    }
)
_SENSITIVE_COMPACT_KEYS = frozenset(key.replace("_", "") for key in _SENSITIVE_KEYS)


class CommandQueueFull(NativeAgentError):
    error_code = "command_queue_full"


class CommandConflict(NativeAgentError):
    error_code = "command_id_conflict"


@dataclass(frozen=True)
class CommandPrincipal:
    tenant_id: str
    user_id: str
    issuer: str = ""

    @property
    def actor(self) -> str:
        authenticated = f"{self.tenant_id}/{self.user_id}"
        return f"{authenticated} ({self.issuer})" if self.issuer else authenticated

    def to_json(self) -> dict[str, str]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "issuer": self.issuer,
        }


@dataclass(frozen=True)
class CommandReceipt:
    run_id: str
    command_id: str
    status: CommandStatus
    result: dict[str, Any] | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    transient_result: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": COMMAND_RECEIPT_VERSION,
            "run_id": self.run_id,
            "command_id": self.command_id,
            "status": self.status,
            "result": dict(self.result) if self.result is not None else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def sanitize_command_data(value: Any, *, key: str = "") -> Any:
    """Return JSON-safe persisted data with credential-shaped fields redacted."""

    lowered = key.lower()
    compact = "".join(character for character in lowered if character.isalnum())
    sensitive_suffix = compact.endswith(("password", "secret", "secretkey"))
    if key and (compact in _SENSITIVE_COMPACT_KEYS or sensitive_suffix):
        return "[redacted]"
    if isinstance(value, dict):
        return {
            str(item_key): sanitize_command_data(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_command_data(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def redact_command_credential(value: Any, credential: str) -> Any:
    """Remove the authenticated bearer value if a caller repeated it in payload text."""

    if not credential:
        return value
    if isinstance(value, dict):
        return {
            str(item_key).replace(credential, "[redacted]"): redact_command_credential(
                item, credential
            )
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_command_credential(item, credential) for item in value]
    if isinstance(value, str):
        return value.replace(credential, "[redacted]")
    return value


def command_identity_sha256(
    *,
    command_type: str,
    args: dict[str, Any],
    principal: CommandPrincipal,
    reason: str,
) -> str:
    """Canonical semantic identity used to reject conflicting command-ID reuse."""

    payload = {
        "type": command_type,
        "args": sanitize_command_data(args),
        "principal": principal.to_json(),
        "reason": reason,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
