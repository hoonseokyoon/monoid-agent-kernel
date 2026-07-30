from __future__ import annotations

import json
from pathlib import Path

import pytest

from support.runtime import runtime_config, tool_binding

from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.core.agents import AgentRuntimeConfig, PromptSpec, ToolSearchConfig
from monoid_agent_kernel.core.content import DocumentPart, ImagePart, TextPart
from monoid_agent_kernel.core.spec import AgentRunSpec, ModelConfig, ReasoningConfig, RunLimits
from monoid_agent_kernel.core.tool_surface import ToolScope
from monoid_agent_kernel.permissions import PermissionPolicy, matches_path_patterns
from monoid_agent_kernel.tools.base import ToolResult, ToolSpec

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
        permission_policy=PermissionPolicy(
            deny_patterns=(".env", "!odd"), redact_patterns=("*.key", "!private")
        ),
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
    assert payload["permission_policy"]["deny_patterns"] == [".env", "!odd"]
    assert payload["permission_policy"]["redact_patterns"] == ["*.key", "!private"]
    assert payload["permission_policy"]["path_pattern_encoding"] == "monoid.literal-bang.v1"


def test_runtime_config_round_trip_hash_and_model() -> None:
    config = AgentRuntimeConfig(
        definition_id="coding",
        config_version=3,
        model=ModelConfig(model="gpt-x", reasoning=ReasoningConfig(effort="high", summary="auto")),
        prompt=PromptSpec(persona_segments=("Be direct.",), runtime_segments=("Use concise edits.",)),
        tools=(
            tool_binding(
                "fs.read",
                guidance="Read before writing.",
                scope=ToolScope(allowed_paths=("!odd",), denied_paths=("!private",)),
            ),
            tool_binding("run.finish"),
        ),
        tool_search=ToolSearchConfig(enabled=True, top_k=3),
        metadata={"owner": "platform"},
    )

    restored = AgentRuntimeConfig.from_json(json.loads(json.dumps(config.to_json())))

    assert restored == config
    assert restored.config_hash == config.config_hash
    assert restored.to_json()["config_hash"] == config.config_hash


def test_runtime_config_durable_decoder_migrates_legacy_tool_scopes() -> None:
    config = runtime_config(
        bindings=(
            tool_binding(
                "fs.read",
                scope=ToolScope(allowed_paths=("!odd",), denied_paths=("./!private",)),
            ),
        )
    )
    legacy_payload = json.loads(json.dumps(config.to_json()))
    legacy_scope = legacy_payload["tools"][0]["scope"]
    legacy_scope.pop("path_pattern_encoding")
    legacy_scope["allowed_paths"] = ["!odd"]
    legacy_scope["denied_paths"] = ["./!private"]

    with pytest.raises(ValueError, match="negated path patterns"):
        AgentRuntimeConfig.from_json(legacy_payload)

    restored = AgentRuntimeConfig.from_durable_json(legacy_payload)

    assert restored.tools[0].scope == config.tools[0].scope
    assert restored.to_json()["tools"][0]["scope"]["allowed_paths"] == ["!odd"]
    assert restored.to_json()["tools"][0]["scope"]["denied_paths"] == ["./!private"]
    assert restored.to_json()["tools"][0]["scope"]["path_pattern_encoding"] == (
        "monoid.literal-bang.v1"
    )
    assert restored.config_hash == config.config_hash == legacy_payload["config_hash"]


def test_runtime_config_hash_keeps_v019_literal_bang_projection_compatible() -> None:
    config = runtime_config(
        bindings=(
            tool_binding(
                "fs.read",
                scope=ToolScope(allowed_paths=("!odd",), denied_paths=("./!private",)),
            ),
        ),
    )
    current_payload = json.loads(json.dumps(config.to_json()))
    legacy_projection = json.loads(json.dumps(current_payload))
    legacy_projection["tools"][0]["scope"].pop("path_pattern_encoding")
    legacy_projection.pop("config_hash")

    assert config.config_hash == canonical_sha256(legacy_projection)
    assert current_payload["config_hash"] == config.config_hash


def test_runtime_config_hash_omits_only_the_scope_encoding_marker() -> None:
    config = runtime_config(
        bindings=(
            tool_binding("fs.read", scope=ToolScope(allowed_paths=("!odd",))),
        ),
    )
    marked = json.loads(json.dumps(config.to_json()))
    unmarked = json.loads(json.dumps(marked))
    unmarked["tools"][0]["scope"].pop("path_pattern_encoding")

    assert AgentRuntimeConfig.from_durable_json(marked).config_hash == config.config_hash
    assert AgentRuntimeConfig.from_durable_json(unmarked).config_hash == config.config_hash

    legacy_inert = json.loads(json.dumps(unmarked))
    legacy_inert["tools"][0]["scope"]["allowed_paths"] = [r"\!odd"]
    assert AgentRuntimeConfig.from_durable_json(legacy_inert).config_hash != config.config_hash

    changed_path = json.loads(json.dumps(marked))
    changed_path["tools"][0]["scope"]["allowed_paths"] = ["!other"]
    assert AgentRuntimeConfig.from_durable_json(changed_path).config_hash != config.config_hash

    wrong_level = json.loads(json.dumps(marked))
    wrong_level["tools"][0]["runtime"]["path_pattern_encoding"] = "wrong-level"
    assert AgentRuntimeConfig.from_json(wrong_level).config_hash != config.config_hash

    unsupported = json.loads(json.dumps(marked))
    unsupported["tools"][0]["scope"]["path_pattern_encoding"] = "future.encoding.v2"
    with pytest.raises(ValueError, match="unsupported path pattern encoding"):
        AgentRuntimeConfig.from_durable_json(unsupported)


def test_runtime_config_durable_decoder_preserves_legacy_purepath_scope() -> None:
    payload = runtime_config(
        bindings=(
            tool_binding("fs.read", scope=ToolScope(allowed_paths=("secret/file",))),
        )
    ).to_json()
    payload["tools"][0]["scope"]["allowed_paths"] = ["secret//file"]

    with pytest.raises(ValueError, match="workspace path"):
        AgentRuntimeConfig.from_json(payload)

    restored = AgentRuntimeConfig.from_durable_json(payload)

    assert matches_path_patterns("secret/file", restored.tools[0].scope.allowed_paths)
    assert restored.to_json()["tools"][0]["scope"]["allowed_paths"] == ["secret//file"]


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
        from monoid_agent_kernel.core.agents import validate_runtime_config

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
