from __future__ import annotations

import json
from pathlib import Path

import pytest

from support.runtime import runtime_config, tool_binding

from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.core.agents import AgentRuntimeConfig, PromptSpec, ToolSearchConfig
from monoid_agent_kernel.core.content import DocumentPart, ImagePart, TextPart
from monoid_agent_kernel.core.spec import (
    AgentRunSpec,
    ModelConfig,
    ModelRetryConfig,
    ReasoningConfig,
    RunLimits,
)
from monoid_agent_kernel.core.tool_surface import ToolScope
from monoid_agent_kernel.permissions import PermissionPolicy, matches_path_patterns
from monoid_agent_kernel.providers.base import normalize_model_config
from monoid_agent_kernel.tools.base import ToolResult, ToolSpec

pytestmark = pytest.mark.unit


_RUN_LIMIT_FIELDS = (
    "max_steps",
    "max_tool_calls",
    "max_bytes_read",
    "max_duration_s",
    "max_messages",
    "max_message_log_bytes",
    "max_workspace_delta_bytes",
    "max_delta_file_bytes",
    "keep_recent_tool_images",
    "max_input_tokens",
    "max_output_tokens",
    "max_total_tokens",
    "max_subagents",
    "max_subagent_depth",
    "max_output_retries",
)

_OPTIONAL_RUN_LIMIT_FIELDS = (
    "max_duration_s",
    "keep_recent_tool_images",
    "max_input_tokens",
    "max_output_tokens",
    "max_total_tokens",
)

_REQUIRED_RUN_LIMIT_FIELDS = tuple(
    field_name for field_name in _RUN_LIMIT_FIELDS if field_name not in _OPTIONAL_RUN_LIMIT_FIELDS
)

_ZERO_RUN_LIMIT_FIELDS = tuple(
    field_name for field_name in _RUN_LIMIT_FIELDS if field_name != "max_subagents"
)


def _ok_tool(*_args: object) -> ToolResult:
    return ToolResult(ok=True)


@pytest.mark.parametrize("payload", ([], True, 1, "model"))
def test_model_config_json_requires_object_or_null(payload: object) -> None:
    with pytest.raises(ValueError, match="model config must be an object or null"):
        ModelConfig.from_json(payload)  # type: ignore[arg-type]


@pytest.mark.parametrize("payload", ([], True, 1, "retry"))
def test_model_retry_json_requires_object_or_null(payload: object) -> None:
    with pytest.raises(ValueError, match="model retry config must be an object or null"):
        ModelRetryConfig.from_json(payload)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (True, 1.0, "2", 0, -1, None, float("nan")))
def test_model_retry_json_rejects_non_exact_positive_max_attempts(value: object) -> None:
    with pytest.raises(ValueError, match="model.retry.max_attempts"):
        ModelRetryConfig.from_json({"max_attempts": value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("timeout_s", True),
        ("timeout_s", "1"),
        ("timeout_s", 0),
        ("timeout_s", -1),
        ("timeout_s", float("nan")),
        ("timeout_s", float("inf")),
    ],
)
def test_model_config_json_rejects_invalid_timeout_control(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="model.timeout_s"):
        ModelConfig.from_json({field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("initial_delay_s", True),
        ("initial_delay_s", "0"),
        ("initial_delay_s", -1),
        ("max_delay_s", float("nan")),
        ("max_delay_s", float("inf")),
        ("backoff_multiplier", True),
        ("backoff_multiplier", "2"),
        ("backoff_multiplier", 0),
        ("backoff_multiplier", -1),
        ("jitter_s", -1),
        ("jitter_s", float("nan")),
        ("jitter_s", float("-inf")),
    ],
)
def test_model_retry_json_rejects_invalid_timing_controls(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=f"model.retry.{field_name}"):
        ModelRetryConfig.from_json({field_name: value})


@pytest.mark.parametrize(
    "retry_on",
    ("gateway_timeout", {"gateway_timeout": True}, {"gateway_timeout"}, None),
)
def test_model_retry_json_requires_an_explicit_retry_code_sequence(retry_on: object) -> None:
    with pytest.raises(ValueError, match="model.retry.retry_on"):
        ModelRetryConfig.from_json({"retry_on": retry_on})


@pytest.mark.parametrize("entry", ("", 1, True, None))
def test_model_retry_json_requires_nonempty_string_codes(entry: object) -> None:
    with pytest.raises(ValueError, match="model.retry.retry_on entries"):
        ModelRetryConfig.from_json({"retry_on": ["gateway_timeout", entry]})


def test_model_retry_json_reads_the_layer_and_defaults_it_to_adapter() -> None:
    assert ModelRetryConfig.from_json(None).layer == "adapter"
    assert ModelRetryConfig.from_json({}).layer == "adapter"
    assert ModelRetryConfig.from_json({"layer": "adapter"}).layer == "adapter"
    assert ModelRetryConfig.from_json({"layer": "kernel"}).layer == "kernel"


@pytest.mark.parametrize("value", ("provider", "", 1, True, None))
def test_model_retry_json_rejects_an_unknown_layer(value: object) -> None:
    with pytest.raises(ValueError, match="model.retry.layer"):
        ModelRetryConfig.from_json({"layer": value})


def test_model_retry_layer_serializes_only_when_it_departs_the_default() -> None:
    # The exact key set of the default serialization: this dict feeds the runtime-config
    # semantic hash, so a config that never chose a layer must serialize byte-identically to
    # one written before the field existed -- and the NEXT field added here must land in this
    # pin rather than silently widening that surface.
    assert set(ModelRetryConfig().to_json()) == {
        "max_attempts",
        "initial_delay_s",
        "max_delay_s",
        "backoff_multiplier",
        "jitter_s",
        "retry_on",
    }
    kernel = ModelRetryConfig(layer="kernel")
    assert kernel.to_json()["layer"] == "kernel"
    assert ModelRetryConfig.from_json(kernel.to_json()) == kernel


def test_direct_python_retry_layer_is_validated_like_every_other_control() -> None:
    normalized = normalize_model_config(ModelConfig(retry=ModelRetryConfig(layer="kernel")))
    assert normalized is not None and normalized.retry.layer == "kernel"
    with pytest.raises(ValueError, match="model.retry.layer"):
        normalize_model_config(ModelConfig(retry=ModelRetryConfig(layer="upstream")))


def test_model_json_controls_preserve_valid_numeric_semantics() -> None:
    model = ModelConfig.from_json({"timeout_s": 0.5})
    retry = ModelRetryConfig.from_json(
        {
            "max_attempts": 1,
            "initial_delay_s": 0,
            "max_delay_s": 0.0,
            "backoff_multiplier": 0.5,
            "jitter_s": 0,
            "retry_on": ("gateway_timeout",),
        }
    )

    assert model.timeout_s == 0.5
    assert retry == ModelRetryConfig(
        max_attempts=1,
        initial_delay_s=0,
        max_delay_s=0.0,
        backoff_multiplier=0.5,
        jitter_s=0,
        retry_on=("gateway_timeout",),
    )


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


@pytest.mark.parametrize("field_name", _RUN_LIMIT_FIELDS)
@pytest.mark.parametrize(
    "invalid_value",
    (True, 1.0, float("nan"), float("inf"), "1", -1),
    ids=("bool", "float", "nan", "infinity", "string", "negative"),
)
def test_run_limits_direct_construction_rejects_untyped_or_invalid_budgets(
    field_name: str, invalid_value: object
) -> None:
    with pytest.raises(ValueError, match=rf"run limit {field_name}"):
        RunLimits(**{field_name: invalid_value})  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", _RUN_LIMIT_FIELDS)
def test_agent_run_spec_json_cannot_bypass_run_limit_validation(field_name: str) -> None:
    with pytest.raises(ValueError, match=rf"run limit {field_name}"):
        AgentRunSpec.from_json(
            {
                "workspace_root": "/workspace",
                "run_root": "runs",
                "limits": {field_name: float("nan")},
            }
        )


@pytest.mark.parametrize("field_name", _OPTIONAL_RUN_LIMIT_FIELDS)
def test_run_limits_accept_documented_unbounded_null(field_name: str) -> None:
    limits = RunLimits(**{field_name: None})  # type: ignore[arg-type]

    assert getattr(limits, field_name) is None


@pytest.mark.parametrize("field_name", _REQUIRED_RUN_LIMIT_FIELDS)
def test_run_limits_reject_null_for_required_budgets(field_name: str) -> None:
    with pytest.raises(ValueError, match=rf"run limit {field_name}"):
        RunLimits(**{field_name: None})  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", _ZERO_RUN_LIMIT_FIELDS)
def test_run_limits_preserve_zero_budget_semantics(field_name: str) -> None:
    limits = RunLimits(**{field_name: 0})  # type: ignore[arg-type]

    assert getattr(limits, field_name) == 0


def test_run_limits_reject_zero_fanout_that_would_disable_the_cap() -> None:
    with pytest.raises(ValueError, match="run limit max_subagents"):
        RunLimits(max_subagents=0)


@pytest.mark.parametrize("payload", ([], True, 1, "limits"))
def test_run_limits_json_requires_object_or_null(payload: object) -> None:
    with pytest.raises(ValueError, match="run limits must be an object or null"):
        RunLimits.from_json(payload)  # type: ignore[arg-type]


def test_agent_run_spec_direct_limits_must_be_validated_run_limits() -> None:
    with pytest.raises(ValueError, match="spec.limits must be RunLimits"):
        AgentRunSpec(
            workspace_root=Path("/workspace"),
            run_root=Path("runs"),
            limits={"max_duration_s": float("nan")},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_depth",
    [True, 1.5, float("nan"), "0", -1],
    ids=("bool", "float", "nan", "string", "negative"),
)
def test_agent_run_spec_rejects_untyped_or_negative_reserved_subagent_depth(
    invalid_depth: object,
) -> None:
    with pytest.raises(ValueError, match="metadata.subagent_depth"):
        AgentRunSpec(
            workspace_root=Path("/workspace"),
            run_root=Path("runs"),
            metadata={"subagent_depth": invalid_depth},
        )


def test_runtime_config_round_trip_hash_and_model() -> None:
    config = AgentRuntimeConfig(
        definition_id="coding",
        config_version=3,
        model=ModelConfig(model="gpt-x", reasoning=ReasoningConfig(effort="high", summary="auto")),
        prompt=PromptSpec(
            persona_segments=("Be direct.",), runtime_segments=("Use concise edits.",)
        ),
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
        bindings=(tool_binding("fs.read", scope=ToolScope(allowed_paths=("!odd",))),),
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
        bindings=(tool_binding("fs.read", scope=ToolScope(allowed_paths=("secret/file",))),)
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


def test_agent_run_spec_normalizes_direct_python_json_content() -> None:
    spec = AgentRunSpec(
        workspace_root=Path("/workspace"),
        run_root=Path("runs"),
        run_id="run\ud800",
        input=(TextPart(text="prompt\ud800"),),
        metadata={"score": float("nan"), "text": "metadata\ud800"},
    )

    assert spec.run_id == "run�"
    assert spec.input == (TextPart(text="prompt�"),)
    assert spec.metadata == {"score": None, "text": "metadata�"}
    json.dumps(spec.to_json(), allow_nan=False)


@pytest.mark.parametrize(
    "field_name,value,error",
    (
        ("mode", False, "spec.mode"),
        ("workspace_backend", False, "spec.workspace_backend"),
        ("metadata", False, "spec.metadata"),
        ("run_id", False, "spec.run_id"),
    ),
)
def test_agent_run_spec_rejects_coercible_operational_fields(
    field_name: str,
    value: object,
    error: str,
) -> None:
    payload: dict[str, object] = {"workspace_root": "/workspace", field_name: value}

    with pytest.raises(ValueError, match=error):
        AgentRunSpec.from_json(payload)  # type: ignore[arg-type]


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
