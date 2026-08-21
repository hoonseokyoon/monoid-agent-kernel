"""Shared scalar validators for content-free durable and wire evidence."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

_OPAQUE_REF_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+/-]{0,255}\Z", re.ASCII)
_TAXONOMY_CODE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)
_UTC_TIMESTAMP_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z",
    re.ASCII,
)


def is_safe_opaque_ref(value: Any) -> bool:
    """Return whether ``value`` is a bounded, control-free opaque address or identity."""

    return type(value) is str and _OPAQUE_REF_PATTERN.fullmatch(value) is not None


def is_safe_taxonomy_code(value: Any) -> bool:
    """Return whether ``value`` is a bounded machine taxonomy value, not free-form text."""

    return type(value) is str and _TAXONOMY_CODE_PATTERN.fullmatch(value) is not None


def is_safe_utc_timestamp(value: Any) -> bool:
    """Return whether ``value`` is a real UTC RFC3339 instant with bounded precision."""

    if type(value) is not str or _UTC_TIMESTAMP_PATTERN.fullmatch(value) is None:
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


__all__ = ["is_safe_opaque_ref", "is_safe_taxonomy_code", "is_safe_utc_timestamp"]
