"""Strict helpers for JSON-native wire payload parsing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypeVar

from pydantic import TypeAdapter, ValidationError

from monoid_agent_kernel.core.json_ingress import is_finite_json_number


class WireValidationError(ValueError):
    """Raised when a wire payload field is present with the wrong type."""


_T = TypeVar("_T")
_MISSING = object()

_OBJECT_ADAPTER = TypeAdapter(dict[str, Any])
_LIST_ADAPTER = TypeAdapter(list[Any])
_STR_ADAPTER = TypeAdapter(str)
_BOOL_ADAPTER = TypeAdapter(bool)


def require_object(value: Any, name: str = "payload") -> dict[str, Any]:
    """Return ``value`` as a JSON object or raise."""

    return _validate(_OBJECT_ADAPTER, value, name, "must be an object")


def require_only_fields(
    payload: Mapping[str, Any],
    allowed: Iterable[str],
    name: str = "payload",
) -> None:
    """Reject fields outside a versioned closed schema without echoing their names."""

    allowed_fields = frozenset(allowed)
    if any(key not in allowed_fields for key in payload):
        raise WireValidationError(f"{name} contains fields outside its closed schema")


def optional_object(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an optional object field, preserving missing-field defaults."""

    value = payload.get(key, _MISSING)
    if value is _MISSING:
        return dict(default or {})
    return require_object(value, key)


def require_list(value: Any, name: str = "value") -> list[Any]:
    """Return ``value`` as a JSON array or raise."""

    return _validate(_LIST_ADAPTER, value, name, "must be a list")


def optional_list(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: Sequence[Any] | None = None,
) -> list[Any]:
    """Return an optional list field, preserving missing-field defaults."""

    value = payload.get(key, _MISSING)
    if value is _MISSING:
        return list(default or ())
    return require_list(value, key)


def parse_str(payload: Mapping[str, Any], key: str, *, default: str = "") -> str:
    """Parse an optional string field without coercion."""

    value = payload.get(key, _MISSING)
    if value is _MISSING:
        return default
    return _validate(_STR_ADAPTER, value, key, "must be a string")


def parse_required_str(
    payload: Mapping[str, Any],
    key: str,
    *,
    strip: bool = False,
    non_empty: bool = True,
) -> str:
    """Parse a required string field without coercion."""

    if key not in payload:
        raise WireValidationError(f"{key} is required")
    value = parse_str(payload, key)
    if strip:
        value = value.strip()
    if non_empty and not value:
        raise WireValidationError(f"{key} is required")
    return value


def parse_bool(payload: Mapping[str, Any], key: str, *, default: bool = False) -> bool:
    """Parse an optional boolean field without truthiness coercion."""

    value = payload.get(key, _MISSING)
    if value is _MISSING:
        return default
    return _validate(_BOOL_ADAPTER, value, key, "must be a boolean")


def parse_int(payload: Mapping[str, Any], key: str, *, default: int = 0) -> int:
    """Parse an optional integer field without bool/string coercion."""

    value = payload.get(key, _MISSING)
    if value is _MISSING:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise WireValidationError(f"{key} must be an integer")
    return value


def parse_float(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: float = 0.0,
    allow_none: bool = False,
) -> float | None:
    """Parse an optional JSON number field without bool/string coercion."""

    value = payload.get(key, _MISSING)
    if value is _MISSING:
        return default
    if value is None and allow_none:
        return None
    if not is_finite_json_number(value):
        raise WireValidationError(f"{key} must be a finite number")
    return float(value)


def parse_literal(
    payload: Mapping[str, Any],
    key: str,
    allowed: Iterable[_T],
    *,
    default: _T | object = _MISSING,
) -> _T:
    """Parse a field and require exact type-and-value membership in ``allowed``."""

    allowed_tuple = tuple(allowed)
    value = payload.get(key, _MISSING)
    if value is _MISSING:
        if default is _MISSING:
            raise WireValidationError(f"{key} is required")
        value = default
    for allowed_value in allowed_tuple:
        if type(value) is type(allowed_value) and value == allowed_value:
            return allowed_value
    allowed_text = ", ".join(str(item) for item in allowed_tuple)
    raise WireValidationError(f"{key} must be one of: {allowed_text}")


def _validate(adapter: TypeAdapter[_T], value: Any, field: str, message: str) -> _T:
    try:
        return adapter.validate_python(value, strict=True)
    except ValidationError as exc:
        raise WireValidationError(f"{field} {message}") from exc
