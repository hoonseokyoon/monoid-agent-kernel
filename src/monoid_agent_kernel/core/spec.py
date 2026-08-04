from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

from monoid_agent_kernel.core.content import (
    ContentPart,
    TextPart,
    normalize_content_part,
    content_part_from_json,
    content_part_to_json,
    non_text_part_types,
)
from monoid_agent_kernel.core.json_ingress import normalize_json_ingress, normalize_unicode_scalars
from monoid_agent_kernel.permissions import PermissionPolicy

RunMode = Literal["read-only", "propose", "apply"]
WorkspaceBackendKind = Literal["overlay", "staging"]
ReasoningEffort = Literal["default", "none", "minimal", "low", "medium", "high", "xhigh"]
ReasoningSummary = Literal["off", "auto", "detailed"]

_REASONING_EFFORTS = get_args(ReasoningEffort)
_REASONING_SUMMARIES = get_args(ReasoningSummary)
# Shared by reasoning and generation: what to do when a transport cannot prove the
# setting was applied.
_MODEL_FALLBACK_MODES = ("fail", "omit")


def _model_choice(value: Any, field_name: str, choices: tuple[str, ...]) -> Any:
    if value not in choices:
        rendered = ", ".join(choices)
        raise ValueError(f"{field_name} must be one of: {rendered}")
    return value


@dataclass(frozen=True)
class ReasoningConfig:
    effort: ReasoningEffort = "medium"
    summary: ReasoningSummary = "off"
    on_unsupported: Literal["fail", "omit"] = "fail"

    @property
    def is_default(self) -> bool:
        """Whether the caller configured reasoning at all — the gate the applied echo rides.

        The generation twin can read its projected payload for this (every default field
        projects to nothing), but the default reasoning config projects a non-empty provider
        block (``{"effort": "medium"}``), so payload truthiness would claim every call
        configured reasoning. Dataclass equality is the honest sentinel; the one thing it
        cannot see — an explicit ``effort="medium"`` — is exactly as invisible as an explicit
        ``temperature=None`` is to generation's gate.
        """

        return self == ReasoningConfig()

    @classmethod
    def from_json(cls, payload: dict[str, Any] | None) -> ReasoningConfig:
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("model reasoning config must be an object or null")
        defaults = cls()
        return validate_reasoning_config(
            cls(
                effort=payload.get("effort", defaults.effort),
                summary=payload.get("summary", defaults.summary),
                on_unsupported=payload.get("on_unsupported", defaults.on_unsupported),
            )
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "effort": self.effort,
            "summary": self.summary,
            "on_unsupported": self.on_unsupported,
        }


def validate_reasoning_config(reasoning: ReasoningConfig) -> ReasoningConfig:
    """Fail-closed check shared by the JSON codec and direct-Python normalization.

    The reasoning twin of :func:`validate_generation_config`, and for the same reason: one
    rule source for both ingresses, so a value accepted from JSON can never diverge from a
    value accepted from a constructor. Before this existed, the codec and the gateway server
    both rejected an unknown effort while a Python-constructed one sailed through
    ``normalize_model_config`` to fail mid-run as a provider 400.
    """

    if not isinstance(reasoning, ReasoningConfig):
        raise ValueError("model.reasoning must be a ReasoningConfig")
    _model_choice(reasoning.effort, "model.reasoning.effort", _REASONING_EFFORTS)
    _model_choice(reasoning.summary, "model.reasoning.summary", _REASONING_SUMMARIES)
    _model_choice(
        reasoning.on_unsupported,
        "model.reasoning.on_unsupported",
        _MODEL_FALLBACK_MODES,
    )
    return reasoning


def _model_control_number(
    value: Any,
    field_name: str,
    *,
    allow_zero: bool,
) -> int | float:
    """Validate a model timing control before any numeric coercion can change its meaning."""

    if type(value) not in (int, float):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        finite = False
    if not finite:
        raise ValueError(f"{field_name} must be a finite number")
    if value < 0 or (not allow_zero and value == 0):
        requirement = "non-negative" if allow_zero else "greater than zero"
        raise ValueError(f"{field_name} must be {requirement}")
    return value


def _model_retry_codes(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("model.retry.retry_on must be an array of non-empty strings")
    codes: list[str] = []
    for code in value:
        if not isinstance(code, str) or not code:
            raise ValueError("model.retry.retry_on entries must be non-empty strings")
        codes.append(code)
    return tuple(codes)


def _model_text(value: Any, field_name: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if type(value) is not str:
        suffix = " or null" if allow_none else ""
        raise ValueError(f"{field_name} must be a string{suffix}")
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _generation_number(
    value: Any,
    field_name: str,
    *,
    minimum: int | float,
    maximum: int | float,
    exclusive_minimum: bool = False,
) -> int | float | None:
    """Validate an optional sampling control; ``None`` delegates to the provider default."""

    if value is None:
        return None
    if type(value) not in (int, float):
        raise ValueError(f"{field_name} must be a finite number or null")
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        finite = False
    if not finite:
        raise ValueError(f"{field_name} must be a finite number or null")
    below = value <= minimum if exclusive_minimum else value < minimum
    if below or value > maximum:
        lower = f"greater than {minimum}" if exclusive_minimum else f"at least {minimum}"
        raise ValueError(f"{field_name} must be {lower} and at most {maximum}")
    return value


@dataclass(frozen=True)
class GenerationConfig:
    """Per-call sampling controls. ``None`` on a value field delegates to the provider default.

    ``on_unsupported`` is enforced where non-application is detectable: the gateway transport
    echoes what it applied, so ``"fail"`` rejects a turn whose parameters were silently dropped
    by an older server. A direct provider call has no echo; there the provider's own error is
    the only signal, so ``"fail"`` and ``"omit"`` behave identically.
    """

    temperature: int | float | None = None
    top_p: int | float | None = None
    max_output_tokens: int | None = None
    on_unsupported: Literal["fail", "omit"] = "fail"

    @property
    def is_default(self) -> bool:
        return self == GenerationConfig()

    @classmethod
    def from_json(cls, payload: dict[str, Any] | None) -> GenerationConfig:
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("model generation config must be an object or null")
        defaults = cls()
        return validate_generation_config(
            cls(
                temperature=payload.get("temperature", defaults.temperature),
                top_p=payload.get("top_p", defaults.top_p),
                max_output_tokens=payload.get("max_output_tokens", defaults.max_output_tokens),
                on_unsupported=payload.get("on_unsupported", defaults.on_unsupported),
            )
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "on_unsupported": self.on_unsupported,
        }


def validate_generation_config(generation: GenerationConfig) -> GenerationConfig:
    """Fail-closed check shared by the JSON codec and direct-Python normalization.

    One rule source for both ingresses, so a range accepted from JSON can never diverge from
    the range accepted from a constructor.
    """

    if not isinstance(generation, GenerationConfig):
        raise ValueError("model.generation must be a GenerationConfig")
    _generation_number(
        generation.temperature,
        "model.generation.temperature",
        minimum=0,
        maximum=2,
    )
    _generation_number(
        generation.top_p,
        "model.generation.top_p",
        minimum=0,
        maximum=1,
        exclusive_minimum=True,
    )
    if generation.max_output_tokens is not None and (
        type(generation.max_output_tokens) is not int or generation.max_output_tokens < 1
    ):
        raise ValueError(
            "model.generation.max_output_tokens must be an integer greater than zero or null"
        )
    _model_choice(
        generation.on_unsupported,
        "model.generation.on_unsupported",
        _MODEL_FALLBACK_MODES,
    )
    return generation


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
        if not isinstance(payload, dict):
            raise ValueError("model retry config must be an object or null")
        defaults = cls()
        max_attempts = payload.get("max_attempts", defaults.max_attempts)
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("model.retry.max_attempts must be an integer greater than zero")
        return cls(
            max_attempts=max_attempts,
            initial_delay_s=_model_control_number(
                payload.get("initial_delay_s", defaults.initial_delay_s),
                "model.retry.initial_delay_s",
                allow_zero=True,
            ),
            max_delay_s=_model_control_number(
                payload.get("max_delay_s", defaults.max_delay_s),
                "model.retry.max_delay_s",
                allow_zero=True,
            ),
            backoff_multiplier=_model_control_number(
                payload.get("backoff_multiplier", defaults.backoff_multiplier),
                "model.retry.backoff_multiplier",
                allow_zero=False,
            ),
            jitter_s=_model_control_number(
                payload.get("jitter_s", defaults.jitter_s),
                "model.retry.jitter_s",
                allow_zero=True,
            ),
            retry_on=_model_retry_codes(payload.get("retry_on", defaults.retry_on)),
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
    timeout_s: int | float = 600
    gateway_url: str | None = None
    retry: ModelRetryConfig = field(default_factory=ModelRetryConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @classmethod
    def from_json(cls, payload: dict[str, Any] | None) -> ModelConfig:
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("model config must be an object or null")
        defaults = cls()
        gateway_url = payload.get("gateway_url")
        return cls(
            provider=payload.get("provider", defaults.provider),
            model=_model_text(payload.get("model", defaults.model), "model.model"),
            reasoning=ReasoningConfig.from_json(payload.get("reasoning")),
            timeout_s=_model_control_number(
                payload.get("timeout_s", defaults.timeout_s),
                "model.timeout_s",
                allow_zero=False,
            ),
            gateway_url=_model_text(
                gateway_url,
                "model.gateway_url",
                allow_none=True,
            ),
            retry=ModelRetryConfig.from_json(payload.get("retry")),
            generation=GenerationConfig.from_json(payload.get("generation")),
        )

    def to_json(self) -> dict[str, Any]:
        payload = {
            "provider": self.provider,
            "model": self.model,
            "reasoning": self.reasoning.to_json(),
            "timeout_s": self.timeout_s,
            "gateway_url": self.gateway_url,
            "retry": self.retry.to_json(),
        }
        # The one key emitted only when configured, unlike every sibling: this dict feeds the
        # request digest (replay key), the runtime-config semantic hash (durable recovery
        # compares it across versions), and the gateway wire, so a never-configured block must
        # serialize byte-identically to a config that predates the field.
        if not self.generation.is_default:
            payload["generation"] = self.generation.to_json()
        return payload


def _validate_run_limit(
    field_name: str,
    value: object,
    *,
    allow_none: bool = False,
    minimum: int = 0,
) -> None:
    """Validate one run-budget control without coercing its security semantics."""
    if value is None:
        if allow_none:
            return
        raise ValueError(f"run limit {field_name} must be an integer >= {minimum}")
    if type(value) is not int or value < minimum:
        suffix = " or null" if allow_none else ""
        raise ValueError(f"run limit {field_name} must be an integer >= {minimum}{suffix}")


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
    # Bounds on agent-as-tool delegation. ``max_subagents`` caps how many subagent
    # tasks a single run may spawn (fan-out backstop); ``max_subagent_depth`` caps
    # nesting (a child at this depth cannot spawn further children). Enforced at
    # spawn time in ``SubagentTaskExecutor``; the depth Claude Code uses is 5.
    max_subagents: int = 8
    max_subagent_depth: int = 5
    # How many times the loop re-prompts after a failed output-validator check before settling
    # the run ``limited`` (``output_validator_unsatisfied``). A repair turn may call tools and
    # shares the global step/tool/token budget, so this bounds settle attempts, not total cost.
    max_output_retries: int = 1

    def __post_init__(self) -> None:
        """Make every construction path enforce the same exact budget types and ranges."""
        for field_name in (
            "max_steps",
            "max_tool_calls",
            "max_bytes_read",
            "max_messages",
            "max_message_log_bytes",
            "max_workspace_delta_bytes",
            "max_delta_file_bytes",
            "max_subagent_depth",
            "max_output_retries",
        ):
            _validate_run_limit(field_name, getattr(self, field_name))
        # A zero fan-out value is treated as unbounded by the executor, so accepting it
        # would disable the documented safety cap instead of disabling delegation.
        _validate_run_limit("max_subagents", self.max_subagents, minimum=1)
        for field_name in (
            "max_duration_s",
            "keep_recent_tool_images",
            "max_input_tokens",
            "max_output_tokens",
            "max_total_tokens",
        ):
            _validate_run_limit(field_name, getattr(self, field_name), allow_none=True)

    @classmethod
    def from_json(cls, payload: dict[str, Any] | None) -> RunLimits:
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("run limits must be an object or null")
        defaults = cls()
        return cls(
            max_steps=payload.get("max_steps", defaults.max_steps),
            max_tool_calls=payload.get("max_tool_calls", defaults.max_tool_calls),
            max_bytes_read=payload.get("max_bytes_read", defaults.max_bytes_read),
            max_duration_s=payload.get("max_duration_s", defaults.max_duration_s),
            max_messages=payload.get("max_messages", defaults.max_messages),
            max_message_log_bytes=payload.get(
                "max_message_log_bytes", defaults.max_message_log_bytes
            ),
            max_workspace_delta_bytes=payload.get(
                "max_workspace_delta_bytes", defaults.max_workspace_delta_bytes
            ),
            max_delta_file_bytes=payload.get("max_delta_file_bytes", defaults.max_delta_file_bytes),
            keep_recent_tool_images=payload.get(
                "keep_recent_tool_images", defaults.keep_recent_tool_images
            ),
            max_input_tokens=payload.get("max_input_tokens", defaults.max_input_tokens),
            max_output_tokens=payload.get("max_output_tokens", defaults.max_output_tokens),
            max_total_tokens=payload.get("max_total_tokens", defaults.max_total_tokens),
            max_subagents=payload.get("max_subagents", defaults.max_subagents),
            max_subagent_depth=payload.get("max_subagent_depth", defaults.max_subagent_depth),
            max_output_retries=payload.get("max_output_retries", defaults.max_output_retries),
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
            "max_subagents": self.max_subagents,
            "max_subagent_depth": self.max_subagent_depth,
            "max_output_retries": self.max_output_retries,
        }


def text_from_parts(parts: tuple[ContentPart, ...]) -> str:
    """Join the text of the text parts in ``parts`` for text-only model adapters.

    Used only for adapters without multimodal support: such an adapter cannot carry
    images/documents, so only ``TextPart`` content is extracted. Multimodal adapters keep the
    full by-reference parts list (see ``user_message_from_parts`` / core/content.py).
    """
    text_segments = [
        part.text.strip() for part in parts if isinstance(part, TextPart) and part.text.strip()
    ]
    return "\n\n".join(text_segments)


def input_to_parts(user_input: str | tuple[ContentPart, ...]) -> tuple[ContentPart, ...]:
    """Normalize a ``submit()`` argument into content parts."""
    if isinstance(user_input, str):
        return (normalize_content_part(TextPart(user_input)),)
    return tuple(normalize_content_part(part) for part in user_input)


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

    def __post_init__(self) -> None:
        if not isinstance(self.workspace_root, Path) or not isinstance(self.run_root, Path):
            raise ValueError("spec workspace_root and run_root must be Path values")
        if type(self.run_id) is not str or not self.run_id:
            raise ValueError("spec.run_id must be a non-empty string")
        if type(self.mode) is not str or self.mode not in {"read-only", "propose", "apply"}:
            raise ValueError("spec.mode must be read-only, propose, or apply")
        if type(self.workspace_backend) is not str or self.workspace_backend not in {
            "overlay",
            "staging",
        }:
            raise ValueError("spec.workspace_backend must be overlay or staging")
        if not isinstance(self.limits, RunLimits):
            raise ValueError("spec.limits must be RunLimits")
        # Re-run the authoritative guard at the containing spec boundary. This keeps
        # manually assembled specs from carrying an unchecked or tampered limits object
        # into AgentLoop.
        RunLimits.__post_init__(self.limits)
        if not isinstance(self.permission_policy, PermissionPolicy):
            raise ValueError("spec.permission_policy must be PermissionPolicy")
        if not isinstance(self.input, (list, tuple)):
            raise ValueError("spec.input must be an array of content parts")
        if not isinstance(self.metadata, dict):
            raise ValueError("spec.metadata must be an object")
        normalized_metadata = normalize_json_ingress(self.metadata)
        if not isinstance(normalized_metadata, dict):  # pragma: no cover - guarded above
            raise ValueError("spec.metadata must be an object")
        object.__setattr__(
            self,
            "workspace_root",
            Path(normalize_unicode_scalars(str(self.workspace_root))),
        )
        object.__setattr__(self, "run_root", Path(normalize_unicode_scalars(str(self.run_root))))
        object.__setattr__(self, "run_id", normalize_unicode_scalars(self.run_id))
        object.__setattr__(
            self, "input", tuple(normalize_content_part(part) for part in self.input)
        )
        object.__setattr__(self, "metadata", normalized_metadata)
        subagent_depth = normalized_metadata.get("subagent_depth", 0)
        if type(subagent_depth) is not int or subagent_depth < 0:
            raise ValueError("spec.metadata.subagent_depth must be a non-negative integer")

    @property
    def effective_input(self) -> tuple[ContentPart, ...]:
        """The explicit input parts, if any."""
        return self.input

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> AgentRunSpec:
        if not isinstance(payload, dict):
            raise ValueError("spec must be an object")
        workspace_root = payload.get("workspace_root")
        if type(workspace_root) is not str or not workspace_root:
            raise ValueError("spec.workspace_root is required")
        run_root = payload.get("run_root", "runs")
        if type(run_root) is not str or not run_root:
            raise ValueError("spec.run_root must be a non-empty string")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("spec.metadata must be an object")
        input_payload = payload.get("input", ())
        if not isinstance(input_payload, (list, tuple)):
            raise ValueError("spec.input must be an array of content parts")
        kwargs: dict[str, Any] = {
            "workspace_root": Path(workspace_root),
            "run_root": Path(run_root),
            "mode": payload.get("mode", "propose"),
            "workspace_backend": payload.get("workspace_backend", "overlay"),
            "limits": (
                RunLimits.from_json(payload["limits"]) if "limits" in payload else RunLimits()
            ),
            "permission_policy": PermissionPolicy.from_json(payload.get("permission_policy")),
            "input": tuple(content_part_from_json(p) for p in input_payload),
            "metadata": dict(metadata),
        }
        run_id = payload.get("run_id")
        if run_id is not None:
            if type(run_id) is not str or not run_id:
                raise ValueError("spec.run_id must be a non-empty string")
            kwargs["run_id"] = run_id
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
