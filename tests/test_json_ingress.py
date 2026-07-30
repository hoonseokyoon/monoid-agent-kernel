from __future__ import annotations

import json
import math

import pytest

from monoid_agent_kernel.core.json_ingress import (
    is_finite_json_number,
    loads_json_ingress,
    loads_model_envelope_json_ingress,
    loads_model_json_ingress,
    normalize_json_ingress,
    normalize_unicode_scalars,
)
from monoid_agent_kernel.reference.conformance import ReferenceCapabilityHarness
from monoid_agent_kernel.tools.base import (
    ToolResult,
    ToolSpec,
    normalize_tool_result,
    normalize_tool_spec,
)


def test_unicode_normalization_preserves_scalars_and_repairs_surrogates() -> None:
    ordinary = "한글 😀"

    assert normalize_unicode_scalars(ordinary) is ordinary
    assert normalize_unicode_scalars("\ud83d\ude00") == "😀"
    assert normalize_unicode_scalars("a\ud800b\udc00c") == "a�b�c"


def test_finite_json_number_check_is_total_for_arbitrarily_large_integers() -> None:
    class IntegerSubclass(int):
        pass

    assert is_finite_json_number(0)
    assert is_finite_json_number(1.5)
    assert not is_finite_json_number(10**400)
    assert not is_finite_json_number(float("inf"))
    assert not is_finite_json_number(True)
    assert not is_finite_json_number(IntegerSubclass(1))


def test_json_ingress_normalizes_nested_keys_values_and_nonfinite_numbers() -> None:
    source = {
        "\ud800-key": [float("nan"), float("inf"), -float("inf")],
        "nested": {"text": "\ud83d\ude00"},
    }

    normalized = normalize_json_ingress(source)

    assert normalized == {
        "�-key": [None, None, None],
        "nested": {"text": "😀"},
    }
    assert math.isnan(source["\ud800-key"][0])
    assert source["nested"]["text"] == "\ud83d\ude00"


def test_json_ingress_refuses_key_collisions_after_normalization() -> None:
    lone_surrogate = "\ud800"
    replacement_character = "�"
    with pytest.raises(ValueError, match="keys collide"):
        normalize_json_ingress({lone_surrogate: 1, replacement_character: 2})


def test_json_ingress_is_iterative_and_preserves_graph_topology() -> None:
    leaf: list[object] = []
    deep = leaf
    for _ in range(1_500):
        child: list[object] = []
        deep.append(child)
        deep = child

    shared = {"value": float("nan")}
    cycle: list[object] = []
    cycle.append(cycle)
    source = {"deep": leaf, "left": shared, "right": shared, "cycle": cycle}

    normalized = normalize_json_ingress(source)

    cursor = normalized["deep"]
    for _ in range(1_500):
        cursor = cursor[0]
    assert cursor == []
    assert normalized["left"] is normalized["right"]
    assert normalized["left"] == {"value": None}
    assert normalized["cycle"][0] is normalized["cycle"]


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_external_json_rejects_nonfinite_constants(constant: str) -> None:
    with pytest.raises(ValueError, match="non-finite number"):
        loads_json_ingress('{"value": ' + constant + "}")


@pytest.mark.parametrize("number", ["1e9999", "-1e9999"])
def test_external_json_rejects_float_overflow(number: str) -> None:
    with pytest.raises(ValueError, match="non-finite number"):
        loads_json_ingress('{"max_duration_s": ' + number + "}")


def test_external_json_reports_excessive_nesting_as_invalid_json() -> None:
    document = "[" * 2_000 + "0" + "]" * 2_000

    with pytest.raises(ValueError, match="JSON nesting exceeds the parser limit"):
        loads_json_ingress(document)


def test_json_nesting_limit_cannot_be_bypassed_by_a_string_subclass() -> None:
    class EmptyIteratorString(str):
        def __iter__(self):
            return iter(())

    accepted = "[" * 512 + "0" + "]" * 512
    document = EmptyIteratorString("[" * 513 + "0" + "]" * 513)

    assert loads_json_ingress(accepted) is not None
    assert loads_model_json_ingress(accepted) is not None

    with pytest.raises(ValueError, match="JSON nesting exceeds the parser limit"):
        loads_json_ingress(document)
    with pytest.raises(ValueError, match="model JSON nesting exceeds the parser limit"):
        loads_model_json_ingress(document)


def test_external_json_preserves_the_decoder_limit_diagnostic() -> None:
    with pytest.raises(ValueError, match="decoder limit exceeded"):
        loads_json_ingress("1" * 5_000)


def test_json_nesting_limit_ignores_delimiters_inside_strings() -> None:
    payload = {"text": '\\"[{]}' * 1_000}

    assert loads_json_ingress(json.dumps(payload)) == payload
    assert loads_model_json_ingress(json.dumps(payload)) == payload


def test_model_json_substitutes_nonfinite_values_and_rejects_excessive_nesting() -> None:
    assert loads_model_json_ingress(
        '{"values": [NaN, Infinity, -Infinity, 1e9999], "text": "\\ud800"}'
    ) == {"values": [None, None, None, None], "text": "�"}

    document = "[" * 2_000 + "0" + "]" * 2_000
    with pytest.raises(ValueError, match="model JSON nesting exceeds the parser limit"):
        loads_model_json_ingress(document)


def test_model_envelope_parser_preserves_nonfinite_markers_for_control_validation() -> None:
    parsed = loads_model_envelope_json_ingress(
        '{"stop_reason": NaN, "arguments": {"score": 1e9999}}'
    )

    assert math.isnan(parsed["stop_reason"])
    assert math.isinf(parsed["arguments"]["score"])


@pytest.mark.parametrize("key", [None, 1, 1.5])
def test_json_ingress_rejects_non_string_object_keys(key: object) -> None:
    with pytest.raises(ValueError, match="JSON object keys must be strings"):
        normalize_json_ingress({key: "value"})


def test_external_json_repairs_surrogates_after_parse() -> None:
    assert loads_json_ingress('{"value": "\\ud800"}') == {"value": "�"}


@pytest.mark.parametrize("invalid", [True, "1", float("nan"), float("inf"), 10**400])
def test_reference_capability_harness_rejects_invalid_revocation_epochs(
    invalid: object,
) -> None:
    harness = ReferenceCapabilityHarness()

    with pytest.raises(ValueError, match="before must be a finite number"):
        harness.revoke_capability({"before": invalid})
    with pytest.raises(ValueError, match="revoked_before must be a finite number"):
        harness.import_revocations({"revoked_before": invalid})


def test_tool_normalizers_preserve_custom_init_extension_types() -> None:
    class CustomResult(ToolResult):
        def __init__(self) -> None:
            super().__init__(ok=True, content={"text": "\ud800", "number": float("nan")})

    class CustomSpec(ToolSpec):
        def __init__(self) -> None:
            super().__init__(
                id="custom.\ud800",
                description="description\ud800",
                input_schema={"example": "\ud800"},
                capability="custom.read",
                side_effect="read",
                handler=lambda _context, _arguments: ToolResult(ok=True),
            )

    result = normalize_tool_result(CustomResult())
    spec = normalize_tool_spec(CustomSpec())

    assert type(result) is CustomResult
    assert result.content == {"text": "�", "number": None}
    assert type(spec) is CustomSpec
    assert spec.id == "custom.�"
    assert spec.description == "description�"
    assert spec.input_schema == {"example": "�"}
