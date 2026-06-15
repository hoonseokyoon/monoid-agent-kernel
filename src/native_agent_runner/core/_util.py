"""Internal shared utilities for core modules (timestamps, atomic JSON writes).

Core-internal only; not part of the supported public surface and intentionally
not re-exported from ``native_agent_runner.contracts`` or the package root.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    """Current UTC time as ISO-8601 with a ``Z`` suffix (e.g. ``...T12:00:00Z``)."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def canonical_sha256(payload: dict[str, Any], *, drop: tuple[str, ...] = ()) -> str:
    """SHA-256 hex digest of ``payload`` serialized as canonical JSON.

    ``drop`` lists top-level keys removed before hashing (e.g. the hash field
    itself). Serialization is deterministic — ``sort_keys=True`` and compact
    separators — so the digest is stable across processes and must stay
    byte-identical to remain compatible with already-recorded hashes.
    """
    canonical = dict(payload)
    for key in drop:
        canonical.pop(key, None)
    data = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as pretty JSON, replacing ``path`` atomically via a temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)
