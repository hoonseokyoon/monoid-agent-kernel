from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from native_agent_runner.permissions import PermissionPolicy
from native_agent_runner.shell import ShellPolicy
from native_agent_runner.tools.policy import ToolPolicy
from native_agent_runner.web import WebPolicy

RunMode = Literal["read-only", "propose", "apply"]
WorkspaceBackendKind = Literal["overlay", "staging"]
ReasoningEffort = Literal["default", "none", "minimal", "low", "medium", "high", "xhigh"]
ReasoningSummary = Literal["off", "auto", "detailed"]


@dataclass(frozen=True)
class ReasoningConfig:
    effort: ReasoningEffort = "medium"
    summary: ReasoningSummary = "off"
    on_unsupported: Literal["fail", "omit"] = "fail"

    @classmethod
    def from_json(cls, payload: dict[str, Any] | None) -> ReasoningConfig:
        if payload is None:
            return cls()
        defaults = cls()
        return cls(
            effort=payload.get("effort", defaults.effort),
            summary=payload.get("summary", defaults.summary),
            on_unsupported=payload.get("on_unsupported", defaults.on_unsupported),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "effort": self.effort,
            "summary": self.summary,
            "on_unsupported": self.on_unsupported,
        }


@dataclass(frozen=True)
class ModelRetryConfig:
    max_attempts: int = 3
    initial_delay_s: float = 0.5
    max_delay_s: float = 4.0
    backoff_multiplier: float = 2.0
    jitter_s: float = 0.1
    retry_on: tuple[str, ...] = (
        "gateway_timeout",
        "gateway_network_error",
        "gateway_rate_limited",
        "gateway_server_error",
    )

    @classmethod
    def from_json(cls, payload: dict[str, Any] | None) -> ModelRetryConfig:
        if payload is None:
            return cls()
        defaults = cls()
        return cls(
            max_attempts=int(payload.get("max_attempts", defaults.max_attempts)),
            initial_delay_s=float(payload.get("initial_delay_s", defaults.initial_delay_s)),
            max_delay_s=float(payload.get("max_delay_s", defaults.max_delay_s)),
            backoff_multiplier=float(payload.get("backoff_multiplier", defaults.backoff_multiplier)),
            jitter_s=float(payload.get("jitter_s", defaults.jitter_s)),
            retry_on=tuple(str(code) for code in payload.get("retry_on", defaults.retry_on)),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "initial_delay_s": self.initial_delay_s,
            "max_delay_s": self.max_delay_s,
            "backoff_multiplier": self.backoff_multiplier,
            "jitter_s": self.jitter_s,
            "retry_on": list(self.retry_on),
        }


@dataclass(frozen=True)
class ModelConfig:
    provider: Literal["gateway", "openai", "fake"] = "gateway"
    model: str = "gpt-5.5"
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    timeout_s: int = 600
    gateway_url: str | None = None
    retry: ModelRetryConfig = field(default_factory=ModelRetryConfig)

    @classmethod
    def from_json(cls, payload: dict[str, Any] | None) -> ModelConfig:
        if payload is None:
            return cls()
        defaults = cls()
        gateway_url = payload.get("gateway_url")
        return cls(
            provider=payload.get("provider", defaults.provider),
            model=str(payload.get("model", defaults.model)),
            reasoning=ReasoningConfig.from_json(payload.get("reasoning")),
            timeout_s=int(payload.get("timeout_s", defaults.timeout_s)),
            gateway_url=None if gateway_url is None else str(gateway_url),
            retry=ModelRetryConfig.from_json(payload.get("retry")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "reasoning": self.reasoning.to_json(),
            "timeout_s": self.timeout_s,
            "gateway_url": self.gateway_url,
            "retry": self.retry.to_json(),
        }


@dataclass(frozen=True)
class RunLimits:
    max_steps: int = 30
    max_tool_calls: int = 100
    max_bytes_read: int = 1_000_000
    max_duration_s: int | None = 900

    @classmethod
    def from_json(cls, payload: dict[str, Any] | None) -> RunLimits:
        if payload is None:
            return cls()
        defaults = cls()
        max_duration_raw = payload.get("max_duration_s", defaults.max_duration_s)
        return cls(
            max_steps=int(payload.get("max_steps", defaults.max_steps)),
            max_tool_calls=int(payload.get("max_tool_calls", defaults.max_tool_calls)),
            max_bytes_read=int(payload.get("max_bytes_read", defaults.max_bytes_read)),
            max_duration_s=None if max_duration_raw is None else int(max_duration_raw),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "max_bytes_read": self.max_bytes_read,
            "max_duration_s": self.max_duration_s,
        }


def default_capabilities(mode: RunMode) -> frozenset[str]:
    base = {
        "fs.read",
        "text.search",
        "artifact.control",
        "run.control",
    }
    if mode in {"propose", "apply"}:
        base.update(
            {
                "fs.write",
                "fs.patch",
                "fs.mkdir",
                "fs.copy",
                "fs.move",
                "fs.delete",
            }
        )
    return frozenset(base)


@dataclass(frozen=True)
class AgentRunSpec:
    instruction: str
    workspace_root: Path
    run_root: Path
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    mode: RunMode = "propose"
    workspace_backend: WorkspaceBackendKind = "overlay"
    model: ModelConfig = field(default_factory=ModelConfig)
    limits: RunLimits = field(default_factory=RunLimits)
    capabilities: frozenset[str] | None = None
    permission_policy: PermissionPolicy = field(default_factory=PermissionPolicy)
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)
    shell_policy: ShellPolicy = field(default_factory=ShellPolicy)
    web_policy: WebPolicy = field(default_factory=WebPolicy)
    metadata: dict[str, object] = field(default_factory=dict)

    def effective_capabilities(self) -> frozenset[str]:
        if self.capabilities is not None:
            capabilities = set(self.capabilities)
        else:
            capabilities = set(default_capabilities(self.mode))
        if self.shell_policy.enabled:
            capabilities.add("shell.exec")
            capabilities.add("job.control")
        if self.web_policy.enabled:
            if self.web_policy.search_enabled:
                capabilities.add("web.search")
            if self.web_policy.fetch_enabled:
                capabilities.add("web.fetch")
            if self.web_policy.context_enabled:
                capabilities.add("web.context")
        return frozenset(capabilities)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> AgentRunSpec:
        if not isinstance(payload, dict):
            raise ValueError("spec must be an object")
        if not payload.get("instruction"):
            raise ValueError("spec.instruction is required")
        if not payload.get("workspace_root"):
            raise ValueError("spec.workspace_root is required")
        # Lazy import avoids a module-load cycle (profiles imports spec types).
        from native_agent_runner.core.profiles import resolve_profile

        profile_name = payload.get("profile")
        profile = resolve_profile(str(profile_name)) if profile_name else None
        metadata = dict(payload.get("metadata") or {})
        if profile is not None:
            metadata.setdefault("profile", profile.name)
        capabilities = payload.get("capabilities")
        kwargs: dict[str, Any] = {
            "instruction": str(payload["instruction"]),
            "workspace_root": Path(str(payload["workspace_root"])),
            "run_root": Path(str(payload.get("run_root") or "runs")),
            "mode": payload["mode"] if "mode" in payload else (profile.mode if profile else "propose"),
            "workspace_backend": payload.get("workspace_backend", "overlay"),
            "model": ModelConfig.from_json(payload.get("model")),
            "limits": (
                RunLimits.from_json(payload["limits"])
                if "limits" in payload
                else (profile.limits if profile else RunLimits())
            ),
            "capabilities": None if capabilities is None else frozenset(str(c) for c in capabilities),
            "permission_policy": PermissionPolicy.from_json(payload.get("permission_policy")),
            "tool_policy": ToolPolicy.from_json(payload.get("tool_policy")),
            "shell_policy": (
                ShellPolicy.from_json(payload["shell_policy"])
                if "shell_policy" in payload
                else (profile.shell_policy if profile else ShellPolicy())
            ),
            "web_policy": (
                WebPolicy.from_json(payload["web_policy"])
                if "web_policy" in payload
                else (profile.web_policy if profile else WebPolicy())
            ),
            "metadata": metadata,
        }
        run_id = payload.get("run_id")
        if run_id:
            kwargs["run_id"] = str(run_id)
        return cls(**kwargs)

    def to_json(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "workspace_root": str(self.workspace_root),
            "run_root": str(self.run_root),
            "run_id": self.run_id,
            "mode": self.mode,
            "workspace_backend": self.workspace_backend,
            "model": self.model.to_json(),
            "limits": self.limits.to_json(),
            "capabilities": None if self.capabilities is None else sorted(self.capabilities),
            "permission_policy": self.permission_policy.to_json(),
            "tool_policy": self.tool_policy.to_json(),
            "shell_policy": self.shell_policy.to_json(),
            "web_policy": self.web_policy.to_json(),
            "metadata": dict(self.metadata),
        }
