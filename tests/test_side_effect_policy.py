from __future__ import annotations

from monoid_agent_kernel.core.agents import AgentRuntimeConfig
from monoid_agent_kernel.core.side_effect_policy import (
    SIDE_EFFECT_OUTBOX_MISSING,
    SIDE_EFFECT_POLICY_DENIED,
    ToolSideEffectPolicy,
    admit_tool_side_effect,
    side_effect_declaration_from_tool,
    side_effect_policy_from_config,
    verify_outbox_side_effect,
)
from monoid_agent_kernel.tools.base import ToolContext, ToolResult, ToolSpec


def _tool(*, annotations: dict | None = None) -> ToolSpec:
    def handler(ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        return ToolResult(ok=True)

    return ToolSpec(
        id="demo.external",
        description="demo",
        input_schema={"type": "object", "additionalProperties": True},
        capability="",
        side_effect="write",
        handler=handler,
        annotations=annotations or {},
    )


def test_side_effect_policy_from_config_defaults_to_compat() -> None:
    config = AgentRuntimeConfig(definition_id="demo")

    assert side_effect_policy_from_config(config).mode == "compat"


def test_side_effect_policy_from_config_reads_strict_metadata() -> None:
    config = AgentRuntimeConfig(
        definition_id="demo",
        metadata={"tool_side_effect_policy": {"mode": "strict"}},
    )

    assert side_effect_policy_from_config(config).mode == "strict"


def test_compat_mode_does_not_block_external_side_effect() -> None:
    admission = admit_tool_side_effect(
        _tool(),
        {"external_side_effect": True},
        {},
        ToolSideEffectPolicy(mode="compat"),
    )

    assert admission.allowed is True
    assert admission.requires_outbox is False


def test_strict_mode_rejects_external_side_effect_without_delivery() -> None:
    admission = admit_tool_side_effect(
        _tool(),
        {"external_side_effect": True},
        {},
        ToolSideEffectPolicy(mode="strict"),
    )

    assert admission.allowed is False
    assert admission.error_code == SIDE_EFFECT_POLICY_DENIED


def test_strict_mode_accepts_outbox_declaration_and_verifies_staging() -> None:
    admission = admit_tool_side_effect(
        _tool(),
        {"external_side_effect": True, "side_effect_delivery": "outbox"},
        {},
        ToolSideEffectPolicy(mode="strict"),
    )

    assert admission.allowed is True
    assert admission.requires_outbox is True
    assert verify_outbox_side_effect(admission, 1, 2).allowed is True
    missing = verify_outbox_side_effect(admission, 1, 1)
    assert missing.allowed is False
    assert missing.error_code == SIDE_EFFECT_OUTBOX_MISSING


def test_strict_mode_requires_configured_idempotency_key_arg() -> None:
    missing = admit_tool_side_effect(
        _tool(),
        {
            "external_side_effect": True,
            "side_effect_delivery": "idempotent",
            "idempotency_key_arg": "request_id",
        },
        {"idempotency_key": "wrong-key"},
        ToolSideEffectPolicy(mode="strict"),
    )
    admitted = admit_tool_side_effect(
        _tool(),
        {
            "external_side_effect": True,
            "side_effect_delivery": "idempotent",
            "idempotency_key_arg": "request_id",
        },
        {"request_id": "req-1"},
        ToolSideEffectPolicy(mode="strict"),
    )

    assert missing.allowed is False
    assert missing.error_code == SIDE_EFFECT_POLICY_DENIED
    assert admitted.allowed is True
    assert admitted.idempotency_key == "req-1"


def test_binding_runtime_overrides_tool_annotations() -> None:
    declaration = side_effect_declaration_from_tool(
        _tool(
            annotations={
                "external_side_effect": True,
                "side_effect_delivery": "outbox",
            }
        ),
        {
            "external_side_effect": False,
            "side_effect_delivery": "idempotent",
        },
    )

    assert declaration.external is False
    assert declaration.delivery == "idempotent"
