from __future__ import annotations

import pytest

from monoid_agent_kernel.core.wire_validation import (
    WireValidationError,
    optional_object,
    parse_bool,
    parse_float,
    parse_int,
    parse_literal,
    parse_required_str,
    parse_str,
    require_list,
    require_object,
)


def test_require_object_rejects_non_objects() -> None:
    with pytest.raises(WireValidationError):
        require_object([], "payload")


def test_optional_object_preserves_missing_default_and_rejects_wrong_type() -> None:
    assert optional_object({}, "metadata") == {}
    assert optional_object({}, "metadata", default={"source": "test"}) == {"source": "test"}

    with pytest.raises(WireValidationError):
        optional_object({"metadata": []}, "metadata")


def test_require_list_rejects_non_lists() -> None:
    with pytest.raises(WireValidationError):
        require_list({}, "items")


def test_parse_str_does_not_coerce_scalars() -> None:
    assert parse_str({"name": "agent"}, "name") == "agent"

    with pytest.raises(WireValidationError):
        parse_str({"name": 1}, "name")


def test_parse_required_str_rejects_missing_and_empty_values() -> None:
    with pytest.raises(WireValidationError):
        parse_required_str({}, "id")
    with pytest.raises(WireValidationError):
        parse_required_str({"id": "   "}, "id", strip=True)


def test_parse_bool_does_not_use_truthiness() -> None:
    assert parse_bool({"flag": False}, "flag") is False

    with pytest.raises(WireValidationError):
        parse_bool({"flag": "false"}, "flag")


def test_parse_int_and_float_reject_bool_and_strings() -> None:
    assert parse_int({"count": 3}, "count") == 3
    assert parse_float({"limit": 3}, "limit") == 3.0

    with pytest.raises(WireValidationError):
        parse_int({"count": True}, "count")
    with pytest.raises(WireValidationError):
        parse_float({"limit": "3"}, "limit")


def test_parse_literal_rejects_unknown_values() -> None:
    assert parse_literal({"status": "pending"}, "status", ("pending", "done")) == "pending"

    with pytest.raises(WireValidationError):
        parse_literal({"status": "oops"}, "status", ("pending", "done"))
