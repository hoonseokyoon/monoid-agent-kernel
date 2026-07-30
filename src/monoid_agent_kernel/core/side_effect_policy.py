"""Policy helpers for durable external tool side effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping

if TYPE_CHECKING:
    from monoid_agent_kernel.tools.base import ToolSpec

ToolSideEffectPolicyMode = Literal["compat", "strict"]
SideEffectDelivery = Literal["outbox", "idempotent"]

SIDE_EFFECT_POLICY_DENIED = "tool_side_effect_policy_denied"
SIDE_EFFECT_OUTBOX_MISSING = "tool_side_effect_outbox_missing"


@dataclass(frozen=True)
class ToolSideEffectPolicy:
    """Runtime policy for externally visible tool side effects."""

    mode: ToolSideEffectPolicyMode = "compat"

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in {"compat", "strict"}:
            raise ValueError("tool side-effect policy mode must be compat or strict")


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
    error: str = ""
    error_code: str = ""


def side_effect_policy_from_config(config: Any) -> ToolSideEffectPolicy:
    """Read the side-effect policy from ``AgentRuntimeConfig.metadata``-like objects."""

    metadata = getattr(config, "metadata", {})
    if not isinstance(metadata, Mapping) or "tool_side_effect_policy" not in metadata:
        return ToolSideEffectPolicy(mode="compat")

    raw = metadata["tool_side_effect_policy"]
    if isinstance(raw, Mapping):
        if "mode" not in raw:
            return ToolSideEffectPolicy(mode="compat")
        mode = raw["mode"]
    elif isinstance(raw, str):
        mode = raw
    else:
        raise ValueError("tool side-effect policy must be an object or mode string")
    if not isinstance(mode, str) or mode not in {"compat", "strict"}:
        raise ValueError("tool side-effect policy mode must be compat or strict")
    return ToolSideEffectPolicy(mode=mode)


def validate_side_effect_settings(
    settings: Mapping[str, Any],
    *,
    source: str = "side-effect declaration",
) -> None:
    """Validate reserved side-effect declaration fields without coercion."""

    if "external_side_effect" in settings and type(settings["external_side_effect"]) is not bool:
        raise ValueError(f"{source} external_side_effect must be a boolean")
    if "side_effect_delivery" in settings:
        delivery = settings["side_effect_delivery"]
        if not isinstance(delivery, str) or delivery not in {"outbox", "idempotent"}:
            raise ValueError(f"{source} side_effect_delivery must be outbox or idempotent")
    if "idempotency_key_arg" in settings:
        key_arg = settings["idempotency_key_arg"]
        if not isinstance(key_arg, str) or not key_arg.strip():
            raise ValueError(f"{source} idempotency_key_arg must be a non-empty string")


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
    validate_side_effect_settings(runtime, source="binding runtime")
    validate_side_effect_settings(annotations, source="tool annotations")
    external = _setting("external_side_effect", runtime, annotations, False)
    delivery = _setting("side_effect_delivery", runtime, annotations, "")
    idempotency_key_arg = _setting(
        "idempotency_key_arg",
        runtime,
        annotations,
        "idempotency_key",
    )
    return SideEffectDeclaration(
        external=external,
        delivery=delivery,
        idempotency_key_arg=idempotency_key_arg.strip(),
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
        idempotency_key = arguments.get(key_arg)
        if type(idempotency_key) is str and idempotency_key.strip():
            return SideEffectAdmission(declaration=declaration)
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


def _setting(
    key: str,
    runtime: Mapping[str, Any],
    annotations: Mapping[str, Any],
    default: Any,
) -> Any:
    if key in runtime:
        return runtime[key]
    if key in annotations:
        return annotations[key]
    return default
