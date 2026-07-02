"""Policy helpers for durable external tool side effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from monoid_agent_kernel.tools.base import ToolSpec

ToolSideEffectPolicyMode = Literal["compat", "strict"]
SideEffectDelivery = Literal["outbox", "idempotent"]

SIDE_EFFECT_POLICY_DENIED = "tool_side_effect_policy_denied"
SIDE_EFFECT_OUTBOX_MISSING = "tool_side_effect_outbox_missing"


@dataclass(frozen=True)
class ToolSideEffectPolicy:
    """Runtime policy for externally visible tool side effects."""

    mode: ToolSideEffectPolicyMode = "compat"


@dataclass(frozen=True)
class SideEffectDeclaration:
    """A tool/binding declaration for an external side effect."""

    external: bool = False
    delivery: SideEffectDelivery | str = ""
    idempotency_key_arg: str = "idempotency_key"


@dataclass(frozen=True)
class SideEffectAdmission:
    """Admission decision for one tool call under the active policy."""

    allowed: bool = True
    declaration: SideEffectDeclaration = SideEffectDeclaration()
    requires_outbox: bool = False
    idempotency_key: str = ""
    error: str = ""
    error_code: str = ""


def side_effect_policy_from_config(config: Any) -> ToolSideEffectPolicy:
    """Read the side-effect policy from ``AgentRuntimeConfig.metadata``-like objects."""

    metadata = getattr(config, "metadata", {}) or {}
    raw = metadata.get("tool_side_effect_policy") if isinstance(metadata, Mapping) else None
    if isinstance(raw, Mapping):
        mode = str(raw.get("mode") or "compat").strip().lower()
    else:
        mode = str(raw or "compat").strip().lower()
    if mode == "strict":
        return ToolSideEffectPolicy(mode="strict")
    return ToolSideEffectPolicy(mode="compat")


def side_effect_declaration_from_tool(
    spec: ToolSpec,
    binding_runtime: Mapping[str, Any] | None,
) -> SideEffectDeclaration:
    """Return the effective external side-effect declaration.

    Binding runtime is the most specific declaration and overrides tool annotations. Tool authors
    can still put defaults in ``ToolSpec.annotations`` for generated bindings.
    """

    runtime = binding_runtime or {}
    annotations = spec.annotations or {}
    external = _bool_setting("external_side_effect", runtime, annotations)
    delivery = _str_setting("side_effect_delivery", runtime, annotations)
    idempotency_key_arg = _str_setting("idempotency_key_arg", runtime, annotations) or "idempotency_key"
    return SideEffectDeclaration(
        external=external,
        delivery=delivery,
        idempotency_key_arg=idempotency_key_arg,
    )


def admit_tool_side_effect(
    spec: ToolSpec,
    binding_runtime: Mapping[str, Any] | None,
    arguments: Mapping[str, Any],
    policy: ToolSideEffectPolicy,
) -> SideEffectAdmission:
    """Check whether a tool call may run under the active side-effect policy."""

    declaration = side_effect_declaration_from_tool(spec, binding_runtime)
    if policy.mode != "strict" or not declaration.external:
        return SideEffectAdmission(declaration=declaration)

    if declaration.delivery == "outbox":
        return SideEffectAdmission(declaration=declaration, requires_outbox=True)

    if declaration.delivery == "idempotent":
        key_arg = declaration.idempotency_key_arg
        idempotency_key = str(arguments.get(key_arg) or "").strip()
        if idempotency_key:
            return SideEffectAdmission(declaration=declaration, idempotency_key=idempotency_key)
        return SideEffectAdmission(
            allowed=False,
            declaration=declaration,
            error=f"external side-effect tool requires idempotency key argument: {key_arg}",
            error_code=SIDE_EFFECT_POLICY_DENIED,
        )

    return SideEffectAdmission(
        allowed=False,
        declaration=declaration,
        error="external side-effect tool requires outbox or idempotent delivery",
        error_code=SIDE_EFFECT_POLICY_DENIED,
    )


def verify_outbox_side_effect(
    admission: SideEffectAdmission,
    before_count: int,
    after_count: int,
) -> SideEffectAdmission:
    """Verify that an outbox-declared tool staged at least one durable request."""

    if not admission.requires_outbox or after_count > before_count:
        return SideEffectAdmission(declaration=admission.declaration)
    return SideEffectAdmission(
        allowed=False,
        declaration=admission.declaration,
        requires_outbox=True,
        error="external side-effect tool declared outbox delivery but staged no outbox request",
        error_code=SIDE_EFFECT_OUTBOX_MISSING,
    )


def _bool_setting(key: str, runtime: Mapping[str, Any], annotations: Mapping[str, Any]) -> bool:
    if key in runtime:
        return _as_bool(runtime.get(key))
    if key in annotations:
        return _as_bool(annotations.get(key))
    return False


def _str_setting(key: str, runtime: Mapping[str, Any], annotations: Mapping[str, Any]) -> str:
    if key in runtime:
        return str(runtime.get(key) or "").strip()
    if key in annotations:
        return str(annotations.get(key) or "").strip()
    return ""


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
