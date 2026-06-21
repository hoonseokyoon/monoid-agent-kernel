from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from native_agent_runner.core.content import (
    ContentPart,
    TextPart,
    content_part_from_json,
    content_part_to_json,
    non_text_part_types,
)
from native_agent_runner.permissions import PermissionPolicy

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


def _optional_int(payload: dict[str, Any], key: str, default: int | None) -> int | None:
    """Parse an optional int field: missing → ``default``; explicit ``null`` → ``None``."""
    if key not in payload:
        return default
    value = payload[key]
    return None if value is None else int(value)


@dataclass(frozen=True)
class RunLimits:
    max_steps: int = 30
    max_tool_calls: int = 100
    max_bytes_read: int = 1_000_000
    max_duration_s: int | None = 900
    # Bounds on the by-value conversation log so a long multi-turn run cannot grow it
    # without limit (it is resent every turn and persisted in every checkpoint). Defaults
    # are generous backstops; exceeding either settles the run as ``limited``.
    max_messages: int = 100_000
    max_message_log_bytes: int = 8_000_000
    # Bounds on the workspace delta a checkpoint may carry, so a runaway/huge/malicious
    # delta cannot bloat the checkpoint store (capture) or fill the disk (restore). Generous
    # backstops; exceeding either on capture settles the run ``limited`` (the prior good
    # checkpoint stays the recovery point), and exceeding on restore refuses the checkpoint.
    max_workspace_delta_bytes: int = 100_000_000
    max_delta_file_bytes: int = 50_000_000
    # Keep only the N most-recent tool-result images on the wire (older ones evicted,
    # cache-aligned) to bound replay growth in screenshot-heavy loops. ``None`` = keep all.
    # Default off: under gateway-side prompt caching, evicting images is uneconomical
    # (cache reads are ~0.1x), so enable only when not caching image-bearing turns.
    keep_recent_tool_images: int | None = None
    # Token budget on the run's accumulated API-reported usage (the authoritative actuals,
    # not an estimate). Checked before each turn against the running totals: once a prior
    # turn pushes a count past its cap the run settles ``limited`` instead of starting
    # another turn. ``None`` = unbounded. These bound the cost dimension that bytes/steps
    # can't (a single turn can be cheap in bytes yet huge in tokens).
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None

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
            max_messages=int(payload.get("max_messages", defaults.max_messages)),
            max_message_log_bytes=int(payload.get("max_message_log_bytes", defaults.max_message_log_bytes)),
            max_workspace_delta_bytes=int(
                payload.get("max_workspace_delta_bytes", defaults.max_workspace_delta_bytes)
            ),
            max_delta_file_bytes=int(payload.get("max_delta_file_bytes", defaults.max_delta_file_bytes)),
            keep_recent_tool_images=(
                None
                if payload.get("keep_recent_tool_images", defaults.keep_recent_tool_images) is None
                else int(payload["keep_recent_tool_images"])
            ),
            max_input_tokens=_optional_int(payload, "max_input_tokens", defaults.max_input_tokens),
            max_output_tokens=_optional_int(payload, "max_output_tokens", defaults.max_output_tokens),
            max_total_tokens=_optional_int(payload, "max_total_tokens", defaults.max_total_tokens),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "max_bytes_read": self.max_bytes_read,
            "max_duration_s": self.max_duration_s,
            "max_messages": self.max_messages,
            "max_message_log_bytes": self.max_message_log_bytes,
            "max_workspace_delta_bytes": self.max_workspace_delta_bytes,
            "max_delta_file_bytes": self.max_delta_file_bytes,
            "keep_recent_tool_images": self.keep_recent_tool_images,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_total_tokens": self.max_total_tokens,
        }


def text_from_parts(parts: tuple[ContentPart, ...]) -> str:
    """Join the text of the text parts in ``parts`` for text-only model adapters.

    Non-text parts (images, documents) are not forwarded yet (see core/content.py),
    so only ``TextPart`` content is extracted.
    """
    text_segments = [
        part.text.strip()
        for part in parts
        if isinstance(part, TextPart) and part.text.strip()
    ]
    return "\n\n".join(text_segments)


def input_to_parts(user_input: str | tuple[ContentPart, ...]) -> tuple[ContentPart, ...]:
    """Normalize a ``submit()`` argument into content parts."""
    if isinstance(user_input, str):
        return (TextPart(user_input),)
    return tuple(user_input)


def user_message_from_parts(parts: tuple[ContentPart, ...]) -> dict[str, Any] | None:
    """Build the durable by-value user message for ``parts``.

    All-text input keeps the legacy ``{"role": "user", "content": <str>}`` shape (and
    returns ``None`` when the text is empty, so an empty turn is not logged). When any
    non-text part is present, the message carries the parts **by reference** as a list of
    ``content_part_to_json`` dicts — lossless and JSON-round-trippable, so it survives
    checkpoint/resume. Resolution to bytes happens later, at wire-build time.
    """
    if non_text_part_types(parts):
        return {"role": "user", "content": [content_part_to_json(part) for part in parts]}
    text = text_from_parts(parts)
    if not text:
        return None
    return {"role": "user", "content": text}


@dataclass(frozen=True)
class AgentRunSpec:
    """Session descriptor: where and under what constraints a run executes.

    It carries no user input — the instruction(s) flow in through
    ``AgentLoop.submit()`` / ``run_once()``. ``input`` remains as the (contract-only)
    multimodal surface; see core/content.py.
    """

    workspace_root: Path
    run_root: Path
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    mode: RunMode = "propose"
    workspace_backend: WorkspaceBackendKind = "overlay"
    limits: RunLimits = field(default_factory=RunLimits)
    permission_policy: PermissionPolicy = field(default_factory=PermissionPolicy)
    input: tuple[ContentPart, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def effective_input(self) -> tuple[ContentPart, ...]:
        """The explicit input parts, if any."""
        return self.input

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> AgentRunSpec:
        if not isinstance(payload, dict):
            raise ValueError("spec must be an object")
        if not payload.get("workspace_root"):
            raise ValueError("spec.workspace_root is required")
        metadata = dict(payload.get("metadata") or {})
        kwargs: dict[str, Any] = {
            "workspace_root": Path(str(payload["workspace_root"])),
            "run_root": Path(str(payload.get("run_root") or "runs")),
            "mode": payload.get("mode", "propose"),
            "workspace_backend": payload.get("workspace_backend", "overlay"),
            "limits": (
                RunLimits.from_json(payload["limits"])
                if "limits" in payload
                else RunLimits()
            ),
            "permission_policy": PermissionPolicy.from_json(payload.get("permission_policy")),
            "input": (
                tuple(content_part_from_json(p) for p in payload["input"])
                if "input" in payload
                else ()
            ),
            "metadata": metadata,
        }
        run_id = payload.get("run_id")
        if run_id:
            kwargs["run_id"] = str(run_id)
        return cls(**kwargs)

    def to_json(self) -> dict[str, Any]:
        return {
            "workspace_root": str(self.workspace_root),
            "run_root": str(self.run_root),
            "run_id": self.run_id,
            "mode": self.mode,
            "workspace_backend": self.workspace_backend,
            "limits": self.limits.to_json(),
            "permission_policy": self.permission_policy.to_json(),
            "input": [content_part_to_json(p) for p in self.input],
            "metadata": dict(self.metadata),
        }
