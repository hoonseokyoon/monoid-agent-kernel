"""Control-plane audit payload helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

CALLBACK_TOKEN_COMMANDS = frozenset({"approve", "deny", "report_task_result"})


@dataclass(frozen=True)
class ControlAuditPolicy:
    """Build redacted audit envelopes for backend control commands."""

    callback_token_commands: frozenset[str] = field(default_factory=lambda: CALLBACK_TOKEN_COMMANDS)

    def accepts_callback_token(self, command_type: str) -> bool:
        return command_type in self.callback_token_commands

    def args_keys(self, args: Mapping[str, Any]) -> list[str]:
        return sorted(key for key in args if key != "token")

    def received_payload(
        self,
        *,
        command_id: str,
        command_type: str,
        target_run_id: str,
        actor: str,
        reason: str,
        token_sha256: str,
        idempotency_key: str,
        args: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "command_id": command_id,
            "command": command_type,
            "target_run_id": target_run_id,
            "actor": actor,
            "reason": reason,
            "token_sha256": token_sha256,
            "idempotency_key": idempotency_key,
            "args_keys": self.args_keys(args),
        }

    def completed_payload(
        self,
        *,
        command_id: str,
        command_type: str,
        target_run_id: str,
        actor: str,
        idempotency_key: str,
        token_sha256: str,
        status: str,
        result_code: str,
        state: str | None,
        duration_ms: float,
    ) -> dict[str, Any]:
        return {
            "command_id": command_id,
            "command": command_type,
            "target_run_id": target_run_id,
            "actor": actor,
            "idempotency_key": idempotency_key,
            "token_sha256": token_sha256,
            "status": status,
            "result_code": result_code,
            "state": state,
            "duration_ms": duration_ms,
        }

    def failed_payload(
        self,
        *,
        command_id: str,
        command_type: str,
        target_run_id: str,
        actor: str,
        idempotency_key: str,
        token_sha256: str,
        status: str,
        error: str,
        error_code: str,
        duration_ms: float,
    ) -> dict[str, Any]:
        return {
            "command_id": command_id,
            "command": command_type,
            "target_run_id": target_run_id,
            "actor": actor,
            "idempotency_key": idempotency_key,
            "token_sha256": token_sha256,
            "status": status,
            "error": error,
            "error_code": error_code,
            "failure_code": error_code,
            "duration_ms": duration_ms,
        }
