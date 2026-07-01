"""Shared scope relation helpers for capabilities and gateways."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from monoid_agent_kernel.web import domain_matches

DEFAULT_NUMERIC_SCOPE_CAP_KEYS = frozenset(
    {
        "max_calls",
        "max_results",
        "max_bytes",
        "timeout_s",
        "max_tokens",
        "max_urls",
        "max_snippets",
    }
)


@dataclass(frozen=True)
class ScopePolicyError(ValueError):
    """Raised when a requested scope violates a signed or outer scope."""

    key: str
    reason: str
    detail: str

    def __str__(self) -> str:
        return self.detail


def scope_within(
    inner: Mapping[str, Any],
    outer: Mapping[str, Any],
    *,
    numeric_keys: Iterable[str] = DEFAULT_NUMERIC_SCOPE_CAP_KEYS,
) -> bool:
    """Return true when ``inner`` is no broader than ``outer``.

    A missing outer key is unconstrained. Numeric caps narrow by choosing a smaller
    value, list constraints narrow by subset, ``allowed_domains`` narrows by domain
    pattern relation, and scalar constraints narrow by equality.
    """
    numeric_key_set = frozenset(numeric_keys)
    for key, inner_val in inner.items():
        if key not in outer:
            continue
        outer_val = outer[key]
        if key == "allowed_domains" and _is_sequence(inner_val) and _is_sequence(outer_val):
            if not domain_patterns_within(_string_tuple(inner_val), _string_tuple(outer_val)):
                return False
        elif _is_sequence(inner_val) and _is_sequence(outer_val):
            if not set(inner_val) <= set(outer_val):
                return False
        elif key in numeric_key_set and _numeric_cap_within(inner_val, outer_val):
            continue
        elif inner_val != outer_val:
            return False
    return True


def domain_patterns_within(requested: Iterable[str], signed: Iterable[str]) -> bool:
    """Return true when every requested domain pattern is covered by signed patterns."""
    signed_patterns = _domain_tuple(signed)
    if "*" in signed_patterns:
        return True
    for pattern in _domain_tuple(requested):
        if not any(_domain_pattern_within(pattern, signed_pattern) for signed_pattern in signed_patterns):
            return False
    return True


def effective_signed_scope(
    scope: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    numeric_keys: Iterable[str] = DEFAULT_NUMERIC_SCOPE_CAP_KEYS,
) -> dict[str, Any]:
    """Apply signed scope as the effective payload ceiling.

    The returned payload preserves caller-provided narrower values and fills omitted
    signed constraints as defaults. Violations raise ``ScopePolicyError``.
    """
    effective = dict(payload)
    _apply_binding_id(scope, effective)
    _apply_allowed_domains(scope, effective)
    _apply_blocked_domains(scope, effective)
    for key in numeric_keys:
        _apply_numeric_cap(scope, effective, key)
    return effective


def _apply_binding_id(scope: Mapping[str, Any], payload: dict[str, Any]) -> None:
    signed = str(scope.get("binding_id") or "").strip()
    if not signed:
        return
    requested = str(payload.get("binding_id") or "").strip()
    if requested and requested != signed:
        raise ScopePolicyError("binding_id", "exceeds", "binding_id exceeds signed scope")
    payload["binding_id"] = signed


def _apply_allowed_domains(scope: Mapping[str, Any], payload: dict[str, Any]) -> None:
    if "allowed_domains" not in scope:
        return
    signed = _domain_tuple(scope.get("allowed_domains") or ())
    requested = _domain_tuple(payload.get("allowed_domains") or ())
    if signed and requested and not domain_patterns_within(requested, signed):
        raise ScopePolicyError("allowed_domains", "exceeds", "allowed_domains exceeds signed scope")
    if signed:
        payload["allowed_domains"] = list(requested or signed)


def _apply_blocked_domains(scope: Mapping[str, Any], payload: dict[str, Any]) -> None:
    if "blocked_domains" not in scope:
        return
    signed = _domain_tuple(scope.get("blocked_domains") or ())
    requested = _domain_tuple(payload.get("blocked_domains") or ())
    payload["blocked_domains"] = list(dict.fromkeys((*signed, *requested)))


def _apply_numeric_cap(scope: Mapping[str, Any], payload: dict[str, Any], key: str) -> None:
    if key not in scope or scope.get(key) is None:
        return
    signed = _positive_number(scope[key], key=key, reason="signed_not_positive")
    requested_raw = payload.get(key)
    if requested_raw is None:
        payload[key] = _json_number(signed)
        return
    requested = _positive_number(requested_raw, key=key, reason="request_not_positive")
    if requested > signed:
        raise ScopePolicyError(key, "exceeds", f"{key} exceeds signed scope")


def _positive_number(value: Any, *, key: str, reason: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ScopePolicyError(key, reason, f"{key} must be positive")
    number = float(value)
    if number <= 0:
        raise ScopePolicyError(key, reason, f"{key} must be positive")
    return number


def _json_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _numeric_cap_within(inner_val: Any, outer_val: Any) -> bool:
    if isinstance(inner_val, bool) or isinstance(outer_val, bool):
        return False
    if not isinstance(inner_val, int | float) or not isinstance(outer_val, int | float):
        return False
    return float(inner_val) <= float(outer_val)


def _domain_pattern_within(pattern: str, signed_pattern: str) -> bool:
    pattern = pattern.lower().strip()
    signed_pattern = signed_pattern.lower().strip()
    if signed_pattern == "*":
        return True
    if pattern == "*":
        return False
    if pattern == signed_pattern:
        return True
    if pattern.startswith("*."):
        suffix = pattern[2:].strip(".")
        return bool(suffix) and signed_pattern.startswith("*.") and domain_matches(suffix, signed_pattern)
    return domain_matches(pattern, signed_pattern)


def _domain_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ScopePolicyError("domains", "invalid", "domain filters must be arrays")
    return tuple(str(item).strip().lower() for item in value if str(item).strip())


def _string_tuple(value: Any) -> tuple[str, ...]:
    return tuple(str(item).strip() for item in value if str(item).strip())


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple, set))
