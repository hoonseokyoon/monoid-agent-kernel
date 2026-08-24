"""Provider-neutral content-addressed blob storage contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol, get_args


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_PutStatus = Literal["stored", "already_present"]


def is_content_sha256(value: object) -> bool:
    """Return whether ``value`` is the canonical lowercase SHA-256 spelling."""

    return type(value) is str and _SHA256_PATTERN.fullmatch(value) is not None


@dataclass(frozen=True, kw_only=True)
class BlobStat:
    """Public-safe immutable object metadata returned by a blob store."""

    sha256: str
    size_bytes: int
    locator: str

    def __post_init__(self) -> None:
        if not is_content_sha256(self.sha256):
            raise ValueError("blob stat sha256 must be a lowercase SHA-256 digest")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("blob stat size_bytes must be a non-negative integer")
        if (
            type(self.locator) is not str
            or not self.locator
            or len(self.locator) > 2048
            or not self.locator.isascii()
            or not all(character.isprintable() for character in self.locator)
        ):
            raise ValueError("blob stat locator must be bounded printable ASCII")

    @property
    def logical_ref(self) -> str:
        return f"blob:{self.sha256}"


@dataclass(frozen=True, kw_only=True)
class BlobPutResult:
    """Conditional immutable publication result."""

    status: _PutStatus
    stat: BlobStat

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in get_args(_PutStatus):
            raise ValueError("blob put status is outside the portable vocabulary")
        if not isinstance(self.stat, BlobStat):
            raise TypeError("blob put result stat must be BlobStat")

    @property
    def created(self) -> bool:
        return self.status == "stored"


class BlobStoreError(RuntimeError):
    """Base class for typed content-addressed storage failures."""


class BlobNotFound(BlobStoreError):
    """The requested content address has no physical object."""


class BlobCorrupt(BlobStoreError):
    """Stored metadata or bytes disagree with the immutable content address."""


class BlobStoreConflict(BlobStoreError):
    """A bounded conditional-write race could not be reconciled."""


class BlobTooLarge(ValueError):
    """The caller supplied bytes outside an adapter's configured object bound."""


class ContentAddressedBlobStore(Protocol):
    """Write-once global bytes keyed by their independently calculated SHA-256."""

    def put_if_absent(self, sha256: str, data: bytes) -> BlobPutResult: ...

    def stat(self, sha256: str) -> BlobStat | None: ...

    def get_checked(self, sha256: str) -> bytes: ...


__all__ = [
    "BlobStat",
    "BlobPutResult",
    "BlobStoreError",
    "BlobNotFound",
    "BlobCorrupt",
    "BlobStoreConflict",
    "BlobTooLarge",
    "ContentAddressedBlobStore",
    "is_content_sha256",
]
