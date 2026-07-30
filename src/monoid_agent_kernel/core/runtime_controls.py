"""Exact validation for reserved per-binding runtime controls.

Binding ``runtime`` objects deliberately remain extensible.  The fields interpreted by the
kernel's built-in shell and web services are different: they are policy and resource controls,
so accepting a coercible value would silently change the configured authority or budget.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_WEB_INTEGER_MINIMUMS: dict[str, int] = {
    "max_calls": 0,
    "max_search_calls": 0,
    "max_fetch_calls": 0,
    "max_context_calls": 0,
    "default_max_results": 1,
    "max_results": 1,
    "default_timeout_s": 1,
    "max_timeout_s": 1,
    "timeout_s": 1,
    "default_max_response_bytes": 1,
    "max_response_bytes": 1,
    "max_bytes": 1,
    "default_max_context_tokens": 1,
    "max_context_tokens": 1,
    "max_tokens": 1,
    "default_max_context_urls": 1,
    "max_context_urls": 1,
    "max_urls": 1,
    "default_max_context_snippets": 1,
    "max_context_snippets": 1,
    "max_snippets": 1,
}

_WEB_DEFAULT_MAX_PAIRS = (
    ("default_max_results", "max_results"),
    ("default_timeout_s", "max_timeout_s"),
    ("default_max_response_bytes", "max_response_bytes"),
    ("default_max_context_tokens", "max_context_tokens"),
    ("default_max_context_urls", "max_context_urls"),
    ("default_max_context_snippets", "max_context_snippets"),
)


def runtime_section(
    runtime: Any,
    section: str,
    *,
    source: str,
) -> Mapping[str, Any]:
    """Return a flat or explicitly nested runtime section without fallback coercion."""

    if not isinstance(runtime, Mapping):
        raise ValueError(f"{source} must be an object")
    if section not in runtime:
        return runtime
    nested = runtime[section]
    if not isinstance(nested, Mapping):
        raise ValueError(f"{source} must be an object")
    return nested


def validate_shell_runtime(
    runtime: Any,
    *,
    source: str = "shell binding runtime",
) -> dict[str, Any]:
    """Validate every shell control and return the selected flat section."""

    from monoid_agent_kernel.shell import ShellExecutionOptions

    section = dict(runtime_section(runtime, "shell", source=source))
    ShellExecutionOptions.from_json(section)
    return section


def validate_web_runtime(
    runtime: Any,
    *,
    source: str = "web binding runtime",
) -> dict[str, Any]:
    """Validate all web budget keys, including aliases shadowed by another spelling."""

    section = dict(runtime_section(runtime, "web", source=source))
    for key, minimum in _WEB_INTEGER_MINIMUMS.items():
        if key not in section:
            continue
        value = section[key]
        if type(value) is not int or value < minimum:
            qualifier = "non-negative" if minimum == 0 else "positive"
            raise ValueError(f"{source} {key} must be a {qualifier} integer")
    for default_key, max_key in _WEB_DEFAULT_MAX_PAIRS:
        if (
            default_key in section
            and max_key in section
            and section[default_key] > section[max_key]
        ):
            raise ValueError(f"{source} {default_key} cannot exceed {max_key}")
    return section


def exact_runtime_integer(
    value: Any,
    *,
    field_name: str,
    minimum: int,
) -> int:
    """Validate a runtime/request integer without bool, float, or string coercion."""

    if type(value) is not int or value < minimum:
        qualifier = "non-negative" if minimum == 0 else "positive"
        raise ValueError(f"{field_name} must be a {qualifier} integer")
    return value
