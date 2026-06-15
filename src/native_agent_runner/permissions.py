from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal

from native_agent_runner._policy_util import dedupe, str_tuple
from native_agent_runner.errors import PermissionDenied
from native_agent_runner.workspace.paths import normalize_workspace_path

PermissionOperation = Literal["read", "write", "artifact", "run"]


def matches_path_patterns(rel: str, patterns: tuple[str, ...]) -> bool:
    if not patterns:
        return False
    pure = PurePosixPath(normalize_workspace_path(rel))
    return any(pure.match(pattern) for pattern in patterns)


@dataclass(frozen=True)
class PermissionPolicy:
    deny_patterns: tuple[str, ...] = ()
    redact_patterns: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, payload: dict[str, Any] | None) -> PermissionPolicy:
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("permission_policy must be an object")
        return cls(
            deny_patterns=str_tuple(
                payload.get("deny_patterns") or (),
                type_error="expected an array of path patterns",
                empty_error="empty path pattern is not allowed",
            ),
            redact_patterns=str_tuple(
                payload.get("redact_patterns") or (),
                type_error="expected an array of path patterns",
                empty_error="empty path pattern is not allowed",
            ),
        )

    def to_json(self) -> dict[str, list[str]]:
        return {
            "deny_patterns": list(self.deny_patterns),
            "redact_patterns": list(self.redact_patterns),
        }

    def merged(
        self,
        *,
        deny_patterns: tuple[str, ...] = (),
        redact_patterns: tuple[str, ...] = (),
    ) -> PermissionPolicy:
        return PermissionPolicy(
            deny_patterns=dedupe((*self.deny_patterns, *deny_patterns)),
            redact_patterns=dedupe((*self.redact_patterns, *redact_patterns)),
        )

    def check_capability(self, capability: str, capabilities: frozenset[str]) -> None:
        if capability not in capabilities:
            raise PermissionDenied(f"missing capability: {capability}")

    def check_paths(self, operation: PermissionOperation, paths: tuple[str, ...]) -> None:
        if operation in {"artifact", "run"}:
            return
        for raw in paths:
            rel = normalize_workspace_path(raw)
            if self.is_path_denied(rel):
                raise PermissionDenied(f"{operation} denied for path: {rel}")

    def is_path_denied(self, rel: str) -> bool:
        return matches_path_patterns(rel, self.deny_patterns)

    def is_path_redacted(self, rel: str) -> bool:
        return matches_path_patterns(rel, self.redact_patterns)
