from __future__ import annotations

import json
from pathlib import Path

import pytest

from support.runtime import runtime_config, tool_binding

from native_agent_runner.core.agents import AgentRuntimeConfig, PromptSpec, ToolSearchConfig
from native_agent_runner.core.content import DocumentPart, ImagePart, TextPart
from native_agent_runner.core.spec import AgentRunSpec, ModelConfig, ReasoningConfig, RunLimits
from native_agent_runner.permissions import PermissionPolicy
from native_agent_runner.tools.base import ToolResult, ToolSpec

pytestmark = pytest.mark.unit


def _ok_tool(*_args: object) -> ToolResult:
    return ToolResult(ok=True)


def test_agent_run_spec_round_trip_is_run_specific() -> None:
    spec = AgentRunSpec(
        workspace_root=Path("/workspace"),
        run_root=Path("runs"),
        run_id="run_123",
        mode="propose",
        workspace_backend="staging",
        limits=RunLimits(max_steps=7, max_tool_calls=11, max_bytes_read=1234, max_duration_s=99),
        permission_policy=PermissionPolicy(deny_patterns=(".env",), redact_patterns=("*.key",)),
        input=(TextPart(text="hello"),),
        metadata={"tenant": "a"},
    )

    restored = AgentRunSpec.from_json(json.loads(json.dumps(spec.to_json())))

    assert restored == spec
    payload = restored.to_json()
    assert "model" not in payload
    assert "tools" not in payload
    assert "tool_policy" not in payload
    assert "shell_policy" not in payload
    assert "web_policy" not in payload


def test_runtime_config_round_trip_hash_and_model() -> None:
    config = AgentRuntimeConfig(
        definition_id="coding",
        config_version=3,
        model=ModelConfig(model="gpt-x", reasoning=ReasoningConfig(effort="high", summary="auto")),
        prompt=PromptSpec(persona_segments=("Be direct.",), runtime_segments=("Use concise edits.",)),
        tools=(
            tool_binding("fs.read", guidance="Read before writing."),
            tool_binding("run.finish"),
        ),
        tool_search=ToolSearchConfig(enabled=True, top_k=3),
        metadata={"owner": "platform"},
    )

    restored = AgentRuntimeConfig.from_json(json.loads(json.dumps(config.to_json())))

    assert restored == config
    assert restored.config_hash == config.config_hash
    assert restored.to_json()["config_hash"] == config.config_hash


def test_content_parts_json_round_trip() -> None:
    spec = AgentRunSpec(
        workspace_root=Path("/workspace"),
        run_root=Path("runs"),
        input=(
            TextPart(text="hello"),
            ImagePart(source_ref="workspace://image.png", mime_type="image/png"),
            DocumentPart(source_ref="workspace://doc.pdf", mime_type="application/pdf"),
        ),
    )

    restored = AgentRunSpec.from_json(spec.to_json())

    assert restored.input == spec.input
    assert restored.effective_input == spec.input


def test_runtime_config_rejects_duplicate_binding_ids() -> None:
    config = runtime_config(
        bindings=(
            tool_binding("fs.read", binding_id="read"),
            tool_binding("fs.stat", binding_id="read"),
        )
    )

    try:
        from native_agent_runner.core.agents import validate_runtime_config

        validate_runtime_config(
            config,
            (
                ToolSpec(
                    id="fs.read",
                    description="minimal read spec",
                    input_schema={"type": "object"},
                    capability="workspace.read",
                    side_effect="read",
                    handler=_ok_tool,
                ),
            ),
        )
    except Exception as exc:
        assert "duplicate tool binding_id" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("duplicate binding id was accepted")
