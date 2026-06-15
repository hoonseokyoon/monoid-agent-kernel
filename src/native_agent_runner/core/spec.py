from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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


@dataclass(frozen=True)
class ModelConfig:
    provider: Literal["gateway", "openai", "fake"] = "gateway"
    model: str = "gpt-5.5"
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    timeout_s: int = 600
    gateway_url: str | None = None
    retry: ModelRetryConfig = field(default_factory=ModelRetryConfig)


@dataclass(frozen=True)
class RunLimits:
    max_steps: int = 30
    max_tool_calls: int = 100
    max_bytes_read: int = 1_000_000
    max_duration_s: int | None = 900


def default_capabilities(mode: RunMode) -> frozenset[str]:
    base = {
        "filesystem.read",
        "text.search",
        "artifact.emit",
        "run.control",
    }
    if mode in {"propose", "apply"}:
        base.update(
            {
                "filesystem.write",
                "filesystem.patch",
                "filesystem.mkdir",
                "filesystem.copy",
                "filesystem.move",
                "filesystem.delete",
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
