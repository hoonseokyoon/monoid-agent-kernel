"""Lock the canonical-JSON SHA-256 contract shared across core modules.

These hashes are persisted in proposal/package/approval/apply artifacts and
verified at every boundary, so the serialization must stay byte-identical. This
guards the consolidation of the previously-duplicated ``_canonical_sha256`` /
``_canonical_payload_sha256`` implementations into ``core._util.canonical_sha256``.
"""

from __future__ import annotations

import hashlib
import json

from native_agent_runner.core._util import canonical_sha256


def _reference(payload: dict, drop: tuple[str, ...] = ()) -> str:
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


def test_canonical_sha256_matches_reference_formula() -> None:
    payload = {"b": 1, "a": "x", "nested": {"z": [1, 2], "y": "é"}}
    assert canonical_sha256(payload) == _reference(payload)


def test_canonical_sha256_is_key_order_independent() -> None:
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})


def test_canonical_sha256_drop_excludes_keys() -> None:
    payload = {"value": 1, "proposal_hash": "stale"}
    assert canonical_sha256(payload, drop=("proposal_hash",)) == canonical_sha256({"value": 1})
    # The hash field's prior value must not influence the digest.
    other = {"value": 1, "proposal_hash": "different"}
    assert canonical_sha256(payload, drop=("proposal_hash",)) == canonical_sha256(
        other, drop=("proposal_hash",)
    )


def test_canonical_sha256_no_drop_keeps_all_keys() -> None:
    payload = {"value": 1, "package_hash": "abc"}
    assert canonical_sha256(payload) == _reference(payload)
    assert canonical_sha256(payload, drop=("package_hash",)) != canonical_sha256(payload)
