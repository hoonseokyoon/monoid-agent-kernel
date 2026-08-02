"""GenerationConfig: fail-closed ingress, serialization omission, and identity stability.

The omission and stability pins encode W5's compatibility contract (dx-note
2026-08-02-w5-implementation-plan.md): a config that never sets a generation value must
serialize byte-identically to one that predates the field, because ``ModelConfig.to_json``
feeds the request digest (replay key), the runtime-config semantic hash (durable recovery),
and the gateway wire all at once.
"""

from __future__ import annotations

import pytest

from monoid_agent_kernel._runtime_config_ingress import normalize_runtime_config
from monoid_agent_kernel.core.agents import AgentRuntimeConfig
from monoid_agent_kernel.core.spec import (
    GenerationConfig,
    ModelConfig,
    ReasoningConfig,
    validate_generation_config,
)
from monoid_agent_kernel.model_call import _digest, _request_payload
from monoid_agent_kernel.providers.base import ModelRequest, normalize_model_config

# Captured on develop @ v0.20.1 (6eb9fcf), before GenerationConfig existed. These literals
# are the contract: regenerating them after a serialization change defeats the pin.
_PRE_W5_CONFIG_HASH_NO_MODEL = "83dab782014d676f0b646421a32c2e41b2befc06efd620d7ae9afd22cb0c3b2c"
_PRE_W5_CONFIG_HASH_DEFAULT_MODEL = (
    "182b10bcd89a7e08517a6022479ad2cf9b6e0c8cd269bfc2341c6ad5a041f792"
)
_PRE_W5_REQUEST_DIGEST = "54c2cb6d143ab5716cd942f584e34a3100d87dad5e85c48bfeadc767a43ed9c6"


# --- fail-closed ingress -------------------------------------------------------------


@pytest.mark.parametrize("payload", ([], True, 1, "generation"))
def test_generation_json_requires_object_or_null(payload: object) -> None:
    with pytest.raises(ValueError, match="model generation config must be an object or null"):
        GenerationConfig.from_json(payload)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("temperature", True),
        ("temperature", "0.7"),
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("top_p", True),
        ("top_p", "1"),
        ("top_p", 0),
        ("top_p", 0.0),
        ("top_p", -1),
        ("top_p", 1.5),
        ("top_p", float("nan")),
    ],
)
def test_generation_json_rejects_invalid_sampling_controls(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=f"model.generation.{field_name}"):
        GenerationConfig.from_json({field_name: value})


@pytest.mark.parametrize("value", (True, 1.0, "128", 0, -1, float("nan")))
def test_generation_json_rejects_non_exact_positive_max_output_tokens(value: object) -> None:
    with pytest.raises(ValueError, match="model.generation.max_output_tokens"):
        GenerationConfig.from_json({"max_output_tokens": value})


@pytest.mark.parametrize("value", ("FAIL", "ignore", 1, None, True))
def test_generation_json_rejects_unknown_fallback_mode(value: object) -> None:
    with pytest.raises(ValueError, match="model.generation.on_unsupported"):
        GenerationConfig.from_json({"on_unsupported": value})


def test_generation_json_preserves_valid_boundary_semantics() -> None:
    generation = GenerationConfig.from_json(
        {
            "temperature": 0,
            "top_p": 1,
            "max_output_tokens": 1,
            "on_unsupported": "omit",
        }
    )
    assert generation == GenerationConfig(
        temperature=0,
        top_p=1,
        max_output_tokens=1,
        on_unsupported="omit",
    )
    assert generation.temperature == 2 - 2  # exact value, no coercion
    assert GenerationConfig.from_json(None) == GenerationConfig()
    assert GenerationConfig.from_json({"temperature": 2}).temperature == 2


# --- D-e: ReasoningConfig codec joins the fail-closed contract ------------------------


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("effort", "extreme"),
        ("effort", 1),
        ("effort", True),
        ("summary", "verbose"),
        ("summary", None),
        ("on_unsupported", "ignore"),
        ("on_unsupported", False),
    ],
)
def test_reasoning_json_rejects_unknown_enum_values(field_name: str, value: object) -> None:
    with pytest.raises(ValueError, match=f"model.reasoning.{field_name}"):
        ReasoningConfig.from_json({field_name: value})


def test_reasoning_json_still_accepts_every_documented_value() -> None:
    for effort in ("default", "none", "minimal", "low", "medium", "high", "xhigh"):
        assert ReasoningConfig.from_json({"effort": effort}).effort == effort
    for summary in ("off", "auto", "detailed"):
        assert ReasoningConfig.from_json({"summary": summary}).summary == summary
    for mode in ("fail", "omit"):
        assert ReasoningConfig.from_json({"on_unsupported": mode}).on_unsupported == mode


# --- the omission rule ---------------------------------------------------------------


def test_default_generation_is_absent_from_model_config_json() -> None:
    """A never-configured generation block must not appear on the wire, in the digest
    payload, or in the semantic hash -- key absence IS the compatibility mechanism."""

    payload = ModelConfig().to_json()
    assert "generation" not in payload

    round_tripped = ModelConfig.from_json(payload)
    assert round_tripped.generation == GenerationConfig()


def test_configured_generation_round_trips_through_model_config_json() -> None:
    config = ModelConfig(
        generation=GenerationConfig(temperature=0.3, max_output_tokens=512),
    )
    payload = config.to_json()
    assert payload["generation"] == {
        "temperature": 0.3,
        "top_p": None,
        "max_output_tokens": 512,
        "on_unsupported": "fail",
    }
    assert ModelConfig.from_json(payload) == config


def test_explicit_omit_mode_alone_still_serializes() -> None:
    """on_unsupported="omit" with no values set is an explicit configuration -- dropping it
    would break round-trip fidelity, so only the true default is omitted."""

    config = ModelConfig(generation=GenerationConfig(on_unsupported="omit"))
    payload = config.to_json()
    assert payload["generation"]["on_unsupported"] == "omit"
    assert ModelConfig.from_json(payload) == config


# --- identity stability pins (pre-W5 literals) ----------------------------------------


def test_config_hash_is_unchanged_for_generation_free_configs() -> None:
    no_model = AgentRuntimeConfig(definition_id="w5-pin")
    with_model = AgentRuntimeConfig(definition_id="w5-pin", model=ModelConfig())

    assert no_model.config_hash == _PRE_W5_CONFIG_HASH_NO_MODEL
    assert with_model.config_hash == _PRE_W5_CONFIG_HASH_DEFAULT_MODEL


def test_request_digest_is_unchanged_for_generation_free_requests() -> None:
    request = ModelRequest(instruction="hi", system_prompt="sys", tools=())
    payload = _request_payload(request, ModelConfig(), provider="fake", destination="")

    assert _digest(payload) == _PRE_W5_REQUEST_DIGEST


def test_setting_generation_changes_the_request_digest() -> None:
    request = ModelRequest(instruction="hi", system_prompt="sys", tools=())
    configured = _request_payload(
        request,
        ModelConfig(generation=GenerationConfig(temperature=0.1)),
        provider="fake",
        destination="",
    )

    assert _digest(configured) != _PRE_W5_REQUEST_DIGEST


# --- direct-Python normalization threading --------------------------------------------


@pytest.mark.parametrize(
    "generation",
    [
        GenerationConfig(temperature=3),
        GenerationConfig(top_p=0),
        GenerationConfig(max_output_tokens=0),
        GenerationConfig(temperature=float("nan")),
        GenerationConfig(on_unsupported="ignore"),  # type: ignore[arg-type]
    ],
)
def test_normalize_model_config_rejects_invalid_direct_generation(
    generation: GenerationConfig,
) -> None:
    with pytest.raises(ValueError, match="model.generation"):
        normalize_model_config(ModelConfig(generation=generation))


def test_normalize_model_config_passes_valid_generation_through_unchanged() -> None:
    config = ModelConfig(generation=GenerationConfig(temperature=1, top_p=0.9))
    normalized = normalize_model_config(config)
    assert normalized is not None
    assert normalized.generation == config.generation


def test_runtime_config_ingress_rejects_non_generation_config_type() -> None:
    config = AgentRuntimeConfig(
        definition_id="w5-pin",
        model=ModelConfig(generation={"temperature": 0.5}),  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="model.generation must be a GenerationConfig"):
        normalize_runtime_config(config)


def test_validate_generation_config_requires_the_type() -> None:
    with pytest.raises(ValueError, match="model.generation must be a GenerationConfig"):
        validate_generation_config({"temperature": 0.5})  # type: ignore[arg-type]


# --- the reasoning twin of direct-Python normalization --------------------------------


@pytest.mark.parametrize(
    ("reasoning", "field_name"),
    [
        (ReasoningConfig(effort="turbo"), "effort"),  # type: ignore[arg-type]
        (ReasoningConfig(summary="verbose"), "summary"),  # type: ignore[arg-type]
        (ReasoningConfig(on_unsupported="ignore"), "on_unsupported"),  # type: ignore[arg-type]
    ],
)
def test_normalize_model_config_rejects_invalid_direct_reasoning(
    reasoning: ReasoningConfig, field_name: str
) -> None:
    """The generation half of this function fails closed through validate_generation_config;
    a direct-Python ReasoningConfig was the one construction route left open (the codec and
    the gateway server both reject) — exactly the "retained and direct-Python controls fail
    closed" contract, unbound on its twin."""

    with pytest.raises(ValueError, match=f"model.reasoning.{field_name}"):
        normalize_model_config(ModelConfig(reasoning=reasoning))


def test_normalize_model_config_passes_valid_reasoning_through_unchanged() -> None:
    config = ModelConfig(reasoning=ReasoningConfig(effort="low", summary="auto"))
    normalized = normalize_model_config(config)
    assert normalized is not None
    assert normalized.reasoning == config.reasoning


def test_validate_reasoning_config_is_the_single_rule_source() -> None:
    from monoid_agent_kernel.core.spec import validate_reasoning_config

    with pytest.raises(ValueError, match="model.reasoning.effort"):
        validate_reasoning_config(ReasoningConfig(effort="turbo"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model.reasoning must be a ReasoningConfig"):
        validate_reasoning_config({"effort": "low"})  # type: ignore[arg-type]
    valid = ReasoningConfig(effort="low", summary="auto", on_unsupported="omit")
    assert validate_reasoning_config(valid) == valid
