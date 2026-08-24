"""Reusable conformance rules for content-addressed blob stores."""

from __future__ import annotations

import hashlib
from typing import Protocol

from monoid_agent_kernel.conformance.report import (
    ConformanceRuleOutcome,
    observation,
    outcome_from_observations,
    safe_exception_summary,
)
from monoid_agent_kernel.hosting.blobs import (
    BlobNotFound,
    BlobPutResult,
    BlobStat,
    ContentAddressedBlobStore,
)


CONTENT_ADDRESSED_BLOB_STORE_PROFILE = "content-addressed-blob-store-contract"
_CONTRACT_BYTES = b"monoid content-addressed blob conformance bytes\n"
_CONTRACT_SHA256 = hashlib.sha256(_CONTRACT_BYTES).hexdigest()
_MISSING_SHA256 = hashlib.sha256(b"monoid deliberately absent blob\n").hexdigest()


class ContentAddressedBlobStoreFactory(Protocol):
    def __call__(self) -> ContentAddressedBlobStore: ...


def _error(rule_id: str, exc: Exception) -> ConformanceRuleOutcome:
    return ConformanceRuleOutcome(
        rule_id=rule_id,
        profile_id=CONTENT_ADDRESSED_BLOB_STORE_PROFILE,
        status="error",
        error=safe_exception_summary(exc),
    )


def run_content_addressed_blob_store_contract(
    factory: ContentAddressedBlobStoreFactory,
) -> tuple[ConformanceRuleOutcome, ...]:
    """Verify write-once identity, checked reads, missing values, and fail-closed input checks."""

    outcomes: list[ConformanceRuleOutcome] = []
    try:
        store = factory()
        first = store.put_if_absent(_CONTRACT_SHA256, _CONTRACT_BYTES)
        second = store.put_if_absent(_CONTRACT_SHA256, _CONTRACT_BYTES)
        stat = store.stat(_CONTRACT_SHA256)
        loaded = store.get_checked(_CONTRACT_SHA256)
        outcomes.append(
            outcome_from_observations(
                "BLOB-01-WRITE-ONCE-CHECKED-READ",
                CONTENT_ADDRESSED_BLOB_STORE_PROFILE,
                (
                    observation(
                        "typed-first-result",
                        expected=True,
                        actual=isinstance(first, BlobPutResult),
                    ),
                    observation(
                        "portable-first-status",
                        expected=True,
                        actual=getattr(first, "status", "") in {"stored", "already_present"},
                    ),
                    observation(
                        "idempotent-second-status",
                        expected="already_present",
                        actual=getattr(second, "status", ""),
                    ),
                    observation(
                        "typed-stat",
                        expected=True,
                        actual=isinstance(stat, BlobStat),
                    ),
                    observation(
                        "stat-identity",
                        expected=[_CONTRACT_SHA256, len(_CONTRACT_BYTES)],
                        actual=(
                            [stat.sha256, stat.size_bytes]
                            if isinstance(stat, BlobStat)
                            else ["", -1]
                        ),
                    ),
                    observation(
                        "checked-bytes",
                        expected=_CONTRACT_SHA256,
                        actual=hashlib.sha256(loaded).hexdigest(),
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("BLOB-01-WRITE-ONCE-CHECKED-READ", exc))

    try:
        store = factory()
        missing_stat = store.stat(_MISSING_SHA256)
        missing_typed = False
        try:
            store.get_checked(_MISSING_SHA256)
        except BlobNotFound:
            missing_typed = True
        outcomes.append(
            outcome_from_observations(
                "BLOB-02-TYPED-MISSING",
                CONTENT_ADDRESSED_BLOB_STORE_PROFILE,
                (
                    observation("missing-stat", expected=None, actual=missing_stat),
                    observation("missing-read", expected=True, actual=missing_typed),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("BLOB-02-TYPED-MISSING", exc))

    try:
        store = factory()
        malformed_rejected = False
        mismatch_rejected = False
        nonbytes_rejected = False
        try:
            store.put_if_absent("A" * 64, _CONTRACT_BYTES)
        except ValueError:
            malformed_rejected = True
        try:
            store.put_if_absent(_CONTRACT_SHA256, b"different")
        except ValueError:
            mismatch_rejected = True
        try:
            store.put_if_absent(_CONTRACT_SHA256, bytearray(_CONTRACT_BYTES))  # type: ignore[arg-type]
        except TypeError:
            nonbytes_rejected = True
        outcomes.append(
            outcome_from_observations(
                "BLOB-03-FAIL-CLOSED-INPUT",
                CONTENT_ADDRESSED_BLOB_STORE_PROFILE,
                (
                    observation("malformed-digest", expected=True, actual=malformed_rejected),
                    observation("digest-mismatch", expected=True, actual=mismatch_rejected),
                    observation("exact-bytes", expected=True, actual=nonbytes_rejected),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("BLOB-03-FAIL-CLOSED-INPUT", exc))
    return tuple(outcomes)


__all__ = [
    "CONTENT_ADDRESSED_BLOB_STORE_PROFILE",
    "ContentAddressedBlobStoreFactory",
    "run_content_addressed_blob_store_contract",
]
