"""Typed direct-Python ingress for per-turn agent runtime configuration.

Runtime config providers are Python seams, so callers can construct the frozen dataclasses with
values that their JSON readers would never produce.  Normalize text and open JSON content before
the config is hashed or compiled into a tool surface, while rejecting controls whose meaning
cannot be recovered without silently changing policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from typing import Any

from monoid_agent_kernel.core.agents import (
    AgentRuntimeConfig,
    OutputValidatorBinding,
    PromptSpec,
    RegistryToolRef,
    ToolBinding,
    ToolSearchConfig,
)
from monoid_agent_kernel.core.json_ingress import normalize_json_ingress, normalize_unicode_scalars
from monoid_agent_kernel.core.runtime_controls import (
    validate_shell_runtime,
    validate_web_runtime,
)
from monoid_agent_kernel.core.spec import (
    GenerationConfig,
    ModelConfig,
    ModelRetryConfig,
    ReasoningConfig,
)
from monoid_agent_kernel.core.tool_surface import ToolGuidance, ToolQuota, ToolScope
from monoid_agent_kernel.permissions import validate_internal_path_patterns
from monoid_agent_kernel.providers.base import normalize_model_config

_MODEL_PROVIDERS = frozenset({"fake", "gateway", "openai"})
_TOOL_EXPOSURES = frozenset({"immediate", "searchable", "hidden"})
_TOOL_AUTHORIZATIONS = frozenset({"allow", "ask", "deny"})


def _copy_with_fields(value: Any, /, **changes: Any) -> Any:
    """Preserve extension subclasses whose convenience constructor omits base fields."""

    cloned = copy(value)
    for name, replacement in changes.items():
        object.__setattr__(cloned, name, replacement)
    return cloned


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return normalize_unicode_scalars(value)


def _optional_content_text(value: Any, field_name: str) -> str | None:
    normalized = normalize_json_ingress(value)
    if normalized is None:
        return None
    if not isinstance(normalized, str):
        raise ValueError(f"{field_name} must be a string or null")
    return normalized


def _optional_identity_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _text_array(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be an array of strings")
    return tuple(_required_text(item, f"{field_name} item") for item in value)


def _enum(value: Any, choices: frozenset[str], field_name: str) -> str:
    normalized = _required_text(value, field_name)
    if normalized not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{field_name} must be one of: {expected}")
    return normalized


def _exact_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _optional_bool(value: Any, field_name: str) -> bool | None:
    if value is None:
        return None
    return _exact_bool(value, field_name)


def _integer_at_least(value: Any, minimum: int, field_name: str) -> int:
    if type(value) is not int or value < minimum:
        requirement = (
            "a positive integer"
            if minimum == 1
            else f"an integer greater than or equal to {minimum}"
        )
        raise ValueError(f"{field_name} must be {requirement}")
    return value


def _optional_nonnegative_integer(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _integer_at_least(value, 0, field_name)


def _json_object(value: Any, field_name: str) -> dict[Any, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    normalized = normalize_json_ingress(value if isinstance(value, dict) else dict(value))
    if not isinstance(normalized, dict):  # defensive: the Mapping check guarantees this today
        raise ValueError(f"{field_name} must be an object")
    return normalized


def _tool_runtime(value: Any, *, tool_id: str) -> dict[Any, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("tool binding runtime must be an object")
    if "requires_lease" in value:
        requirement = value["requires_lease"]
        if type(requirement) is not bool and requirement != "optional":
            raise ValueError(
                "tool binding runtime.requires_lease must be a boolean or 'optional'"
            )
    if tool_id == "shell.exec":
        validate_shell_runtime(value)
    if tool_id.startswith("web."):
        validate_web_runtime(value)
    return _json_object(value, "tool binding runtime")


def _normalize_prompt(prompt: PromptSpec) -> PromptSpec:
    if not isinstance(prompt, PromptSpec):
        raise ValueError("runtime config prompt must be a PromptSpec")
    return _copy_with_fields(
        prompt,
        system_prompt_base=_optional_content_text(
            prompt.system_prompt_base,
            "runtime config prompt.system_prompt_base",
        ),
        persona_segments=_text_array(
            prompt.persona_segments,
            "runtime config prompt.persona_segments",
        ),
        runtime_segments=_text_array(
            prompt.runtime_segments,
            "runtime config prompt.runtime_segments",
        ),
    )


def _normalize_model(model: ModelConfig | None) -> ModelConfig | None:
    if model is None:
        return None
    if not isinstance(model, ModelConfig):
        raise ValueError("runtime config model must be a ModelConfig or null")
    if not isinstance(model.reasoning, ReasoningConfig):
        raise ValueError("model.reasoning must be a ReasoningConfig")
    if not isinstance(model.retry, ModelRetryConfig):
        raise ValueError("model.retry must be a ModelRetryConfig")
    if not isinstance(model.retry.retry_on, (list, tuple)):
        raise ValueError("model.retry.retry_on must be an array of strings")
    if not isinstance(model.generation, GenerationConfig):
        raise ValueError("model.generation must be a GenerationConfig")

    normalized = normalize_model_config(model)
    assert normalized is not None
    _enum(normalized.provider, _MODEL_PROVIDERS, "model.provider")
    # The reasoning enums are enforced inside normalize_model_config via
    # validate_reasoning_config -- the same single rule source the JSON codec uses. The local
    # frozensets this file used to re-declare were a third, hand-copied edition of that rule,
    # one new ReasoningEffort value away from silently diverging.
    _integer_at_least(normalized.retry.max_attempts, 1, "model.retry.max_attempts")
    return normalized


def _normalize_tool_guidance(guidance: ToolGuidance) -> ToolGuidance:
    if not isinstance(guidance, ToolGuidance):
        raise ValueError("tool binding guidance must be a ToolGuidance")
    if not isinstance(guidance.examples, (list, tuple)):
        raise ValueError("tool binding guidance.examples must be an array")
    return _copy_with_fields(
        guidance,
        summary=_required_text(guidance.summary, "tool binding guidance.summary"),
        policy=_required_text(guidance.policy, "tool binding guidance.policy"),
        examples=tuple(
            _json_object(item, "tool binding guidance example") for item in guidance.examples
        ),
        annotations=_json_object(
            guidance.annotations,
            "tool binding guidance.annotations",
        ),
    )


def _normalize_tool_scope(scope: ToolScope) -> ToolScope:
    if not isinstance(scope, ToolScope):
        raise ValueError("tool binding scope must be a ToolScope")
    allowed_paths = validate_internal_path_patterns(
        _text_array(scope.allowed_paths, "tool binding scope.allowed_paths")
    )
    denied_paths = validate_internal_path_patterns(
        _text_array(scope.denied_paths, "tool binding scope.denied_paths")
    )
    return _copy_with_fields(
        scope,
        allowed_paths=allowed_paths,
        denied_paths=denied_paths,
        allowed_domains=_text_array(
            scope.allowed_domains,
            "tool binding scope.allowed_domains",
        ),
        blocked_domains=_text_array(
            scope.blocked_domains,
            "tool binding scope.blocked_domains",
        ),
        command_allow_prefixes=_text_array(
            scope.command_allow_prefixes,
            "tool binding scope.command_allow_prefixes",
        ),
        command_deny_prefixes=_text_array(
            scope.command_deny_prefixes,
            "tool binding scope.command_deny_prefixes",
        ),
        env_allowlist=_text_array(
            scope.env_allowlist,
            "tool binding scope.env_allowlist",
        ),
    )


def _normalize_tool_quota(quota: ToolQuota) -> ToolQuota:
    if not isinstance(quota, ToolQuota):
        raise ValueError("tool binding quota must be a ToolQuota")
    return _copy_with_fields(
        quota,
        max_calls_per_run=_optional_nonnegative_integer(
            quota.max_calls_per_run,
            "tool binding quota.max_calls_per_run",
        ),
    )


def _normalize_registry_ref(ref: RegistryToolRef) -> RegistryToolRef:
    if not isinstance(ref, RegistryToolRef):
        raise ValueError("tool binding ref must be a RegistryToolRef")
    return _copy_with_fields(
        ref,
        tool_id=_required_text(ref.tool_id, "tool binding ref.tool_id"),
        kind=_enum(ref.kind, frozenset({"registry"}), "tool binding ref.kind"),
    )


def _normalize_tool_binding(binding: ToolBinding) -> ToolBinding:
    if not isinstance(binding, ToolBinding):
        raise ValueError("runtime config tools must contain ToolBinding values")
    return _copy_with_fields(
        binding,
        binding_id=_required_text(binding.binding_id, "tool binding binding_id"),
        ref=_normalize_registry_ref(binding.ref),
        model_name=_optional_identity_text(binding.model_name, "tool binding model_name"),
        exposure=_enum(binding.exposure, _TOOL_EXPOSURES, "tool binding exposure"),
        authorization=_enum(
            binding.authorization,
            _TOOL_AUTHORIZATIONS,
            "tool binding authorization",
        ),
        guidance=_normalize_tool_guidance(binding.guidance),
        scope=_normalize_tool_scope(binding.scope),
        quota=_normalize_tool_quota(binding.quota),
        title=_required_text(binding.title, "tool binding title"),
        summary=_required_text(binding.summary, "tool binding summary"),
        risk=_required_text(binding.risk, "tool binding risk"),
        requires_approval=_optional_bool(
            binding.requires_approval,
            "tool binding requires_approval",
        ),
        reason=_required_text(binding.reason, "tool binding reason"),
        runtime=_tool_runtime(binding.runtime, tool_id=binding.ref.tool_id),
        metadata=_json_object(binding.metadata, "tool binding metadata"),
    )


def _normalize_tool_search(config: ToolSearchConfig) -> ToolSearchConfig:
    if not isinstance(config, ToolSearchConfig):
        raise ValueError("runtime config tool_search must be a ToolSearchConfig")
    return _copy_with_fields(
        config,
        enabled=_exact_bool(config.enabled, "runtime config tool_search.enabled"),
        top_k=_integer_at_least(config.top_k, 1, "runtime config tool_search.top_k"),
        binding_id=_required_text(
            config.binding_id,
            "runtime config tool_search.binding_id",
        ),
        model_name=_required_text(
            config.model_name,
            "runtime config tool_search.model_name",
        ),
    )


def _normalize_output_validator(binding: OutputValidatorBinding) -> OutputValidatorBinding:
    if not isinstance(binding, OutputValidatorBinding):
        raise ValueError(
            "runtime config output_validators must contain OutputValidatorBinding values"
        )
    return _copy_with_fields(
        binding,
        validator_id=_required_text(
            binding.validator_id,
            "output validator binding validator_id",
        ),
        enabled=_exact_bool(binding.enabled, "output validator binding enabled"),
    )


def normalize_runtime_config(config: AgentRuntimeConfig) -> AgentRuntimeConfig:
    """Copy a runtime config into its portable typed domain before any use or hashing."""

    if not isinstance(config, AgentRuntimeConfig):
        raise ValueError("runtime config provider must return an AgentRuntimeConfig")
    if not isinstance(config.tools, (list, tuple)):
        raise ValueError("runtime config tools must be an array")
    if not isinstance(config.output_validators, (list, tuple)):
        raise ValueError("runtime config output_validators must be an array")
    return _copy_with_fields(
        config,
        definition_id=_required_text(config.definition_id, "runtime config definition_id"),
        config_version=_integer_at_least(
            config.config_version,
            1,
            "runtime config config_version",
        ),
        model=_normalize_model(config.model),
        prompt=_normalize_prompt(config.prompt),
        tools=tuple(_normalize_tool_binding(binding) for binding in config.tools),
        tool_search=_normalize_tool_search(config.tool_search),
        output_validators=tuple(
            _normalize_output_validator(binding) for binding in config.output_validators
        ),
        metadata=_json_object(config.metadata, "runtime config metadata"),
    )


def preflight_runtime_config(
    config: AgentRuntimeConfig,
) -> tuple[AgentRuntimeConfig | None, list[str]]:
    """Normalize a preflight config through the same ingress as execution."""

    try:
        return normalize_runtime_config(config), []
    except (TypeError, ValueError) as exc:
        return None, [str(exc)]
