"""Shared coercion helpers for execution-boundary dataclasses.

Used by permission and runtime execution parsers to keep JSON array handling
consistent. Internal only; not part of the supported public surface.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable


def str_tuple(
    value: Any,
    *,
    type_error: str,
    empty_error: str | None = None,
    normalize: bool = False,
    error: Callable[[str], Exception] = ValueError,
) -> tuple[str, ...]:
    """Coerce a JSON array into a tuple of strings.

    A bare string (or any non-array) is rejected with ``type_error``. With
    ``normalize=True`` each item is stripped, lowercased, and empties are
    dropped (domain lists). Otherwise, when ``empty_error`` is given, an
    empty/whitespace item raises it. ``error`` selects the exception type.
    """
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise error(type_error)
    items = tuple(str(item) for item in value)
    if normalize:
        return tuple(item.strip().lower() for item in items if item.strip())
    if empty_error is not None and any(not item.strip() for item in items):
        raise error(empty_error)
    return items


def dedupe(values: Iterable[str]) -> tuple[str, ...]:
    """Order-preserving de-duplication."""
    return tuple(dict.fromkeys(values))
