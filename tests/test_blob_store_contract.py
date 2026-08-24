from __future__ import annotations

import hashlib
from typing import get_type_hints

import pytest

from monoid_agent_kernel.conformance import run_content_addressed_blob_store_contract
from monoid_agent_kernel.hosting import (
    BlobCorrupt,
    BlobNotFound,
    BlobPutResult,
    BlobStat,
    BlobStoreConflict,
    BlobStoreError,
    BlobTooLarge,
    ContentAddressedBlobStore,
    is_content_sha256,
)


class _MemoryBlobStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_if_absent(self, sha256: str, data: bytes) -> BlobPutResult:
        if not is_content_sha256(sha256):
            raise ValueError("invalid digest")
        if type(data) is not bytes:
            raise TypeError("data must be bytes")
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError("digest mismatch")
        status = "already_present" if sha256 in self.objects else "stored"
        self.objects.setdefault(sha256, data)
        return BlobPutResult(status=status, stat=self._stat(sha256))

    def _stat(self, sha256: str) -> BlobStat:
        return BlobStat(
            sha256=sha256,
            size_bytes=len(self.objects[sha256]),
            locator=f"memory:{sha256}",
        )

    def stat(self, sha256: str) -> BlobStat | None:
        if not is_content_sha256(sha256):
            raise ValueError("invalid digest")
        return self._stat(sha256) if sha256 in self.objects else None

    def get_checked(self, sha256: str) -> bytes:
        if not is_content_sha256(sha256):
            raise ValueError("invalid digest")
        try:
            return self.objects[sha256]
        except KeyError as exc:
            raise BlobNotFound("missing") from exc


def test_content_addressed_blob_store_contract_is_reusable() -> None:
    store = _MemoryBlobStore()
    outcomes = run_content_addressed_blob_store_contract(lambda: store)

    assert [outcome.rule_id for outcome in outcomes] == [
        "BLOB-01-WRITE-ONCE-CHECKED-READ",
        "BLOB-02-TYPED-MISSING",
        "BLOB-03-FAIL-CLOSED-INPUT",
    ]
    assert all(outcome.passed for outcome in outcomes)


def test_blob_values_are_closed_and_public_safe() -> None:
    sha256 = "a" * 64
    stat = BlobStat(sha256=sha256, size_bytes=3, locator=f"memory:{sha256}")
    stored = BlobPutResult(status="stored", stat=stat)
    present = BlobPutResult(status="already_present", stat=stat)

    assert stored.created is True
    assert present.created is False
    assert stat.logical_ref == f"blob:{sha256}"

    with pytest.raises(ValueError, match="status"):
        BlobPutResult(status="ok", stat=stat)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="BlobStat"):
        BlobPutResult(status="stored", stat=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lowercase"):
        BlobStat(sha256="A" * 64, size_bytes=3, locator="memory:value")
    with pytest.raises(ValueError, match="non-negative"):
        BlobStat(sha256=sha256, size_bytes=-1, locator="memory:value")
    with pytest.raises(ValueError, match="printable ASCII"):
        BlobStat(sha256=sha256, size_bytes=3, locator="memory:line\nbreak")


def test_blob_error_taxonomy_is_typed() -> None:
    assert issubclass(BlobNotFound, BlobStoreError)
    assert issubclass(BlobCorrupt, BlobStoreError)
    assert issubclass(BlobStoreConflict, BlobStoreError)
    assert issubclass(BlobTooLarge, ValueError)


def test_blob_protocol_annotations_resolve() -> None:
    put = get_type_hints(ContentAddressedBlobStore.put_if_absent)
    stat = get_type_hints(ContentAddressedBlobStore.stat)
    get = get_type_hints(ContentAddressedBlobStore.get_checked)

    assert put == {"sha256": str, "data": bytes, "return": BlobPutResult}
    assert stat == {"sha256": str, "return": BlobStat | None}
    assert get == {"sha256": str, "return": bytes}


@pytest.mark.parametrize("value", ["a" * 64, "0" * 64, hashlib.sha256(b"").hexdigest()])
def test_content_sha256_accepts_only_canonical_spelling(value: str) -> None:
    assert is_content_sha256(value)


@pytest.mark.parametrize("value", ["", "A" * 64, "a" * 63, "g" * 64, 1, True, None])
def test_content_sha256_rejects_noncanonical_values(value: object) -> None:
    assert not is_content_sha256(value)
