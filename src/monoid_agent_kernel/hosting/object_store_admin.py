"""Provider-neutral bounded object inventory and garbage-collection values."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, get_args

from .blobs import is_content_sha256


_DeleteStatus = Literal["deleted", "already_missing", "precondition_failed"]
_AbortStatus = Literal["aborted", "already_missing", "precondition_failed"]
_GcStatus = Literal[
    "deleted",
    "already_missing",
    "skipped_associated",
    "skipped_generation",
    "precondition_failed",
]


def _aware_datetime(value: object, field_name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _bounded_ascii(value: object, field_name: str, *, maximum: int = 4096) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > maximum
        or not value.isascii()
        or not all(character.isprintable() for character in value)
    ):
        raise ValueError(f"{field_name} must be bounded printable ASCII")


def _optional_token(value: object, field_name: str) -> None:
    if value is not None:
        _bounded_ascii(value, field_name, maximum=8192)


@dataclass(frozen=True, kw_only=True)
class ObjectInventoryEntry:
    """One content-addressed object observed by an explicitly bounded admin listing."""

    sha256: str
    size_bytes: int
    locator: str
    last_modified: datetime
    delete_token: str

    def __post_init__(self) -> None:
        if not is_content_sha256(self.sha256):
            raise ValueError("object inventory sha256 must be a lowercase SHA-256 digest")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("object inventory size_bytes must be a non-negative integer")
        _bounded_ascii(self.locator, "object inventory locator", maximum=2048)
        _aware_datetime(self.last_modified, "object inventory last_modified")
        _bounded_ascii(self.delete_token, "object inventory delete_token")


@dataclass(frozen=True, kw_only=True)
class ObjectInventoryPage:
    entries: tuple[ObjectInventoryEntry, ...]
    next_token: str | None = None

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple or any(
            not isinstance(entry, ObjectInventoryEntry) for entry in self.entries
        ):
            raise TypeError("object inventory entries must be a tuple of ObjectInventoryEntry")
        _optional_token(self.next_token, "object inventory next_token")


@dataclass(frozen=True, kw_only=True)
class ObjectDeleteResult:
    status: _DeleteStatus

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in get_args(_DeleteStatus):
            raise ValueError("object delete status is outside the portable vocabulary")


@dataclass(frozen=True, kw_only=True)
class IncompleteMultipartUpload:
    sha256: str
    upload_id: str
    initiated_at: datetime

    def __post_init__(self) -> None:
        if not is_content_sha256(self.sha256):
            raise ValueError("multipart upload sha256 must be a lowercase SHA-256 digest")
        _bounded_ascii(self.upload_id, "multipart upload_id", maximum=4096)
        _aware_datetime(self.initiated_at, "multipart initiated_at")


@dataclass(frozen=True, kw_only=True)
class IncompleteMultipartPage:
    uploads: tuple[IncompleteMultipartUpload, ...]
    next_token: str | None = None

    def __post_init__(self) -> None:
        if type(self.uploads) is not tuple or any(
            not isinstance(upload, IncompleteMultipartUpload) for upload in self.uploads
        ):
            raise TypeError("multipart uploads must be a tuple of IncompleteMultipartUpload")
        _optional_token(self.next_token, "multipart next_token")


@dataclass(frozen=True, kw_only=True)
class MultipartAbortResult:
    status: _AbortStatus

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in get_args(_AbortStatus):
            raise ValueError("multipart abort status is outside the portable vocabulary")


class ObjectStoreAdmin(Protocol):
    """Privileged bounded inventory/delete operations kept off the runtime store contract."""

    def inventory_page(
        self,
        *,
        continuation_token: str | None = None,
        limit: int = 1000,
    ) -> ObjectInventoryPage: ...

    def delete_if_match(self, sha256: str, delete_token: str) -> ObjectDeleteResult: ...

    def incomplete_multipart_page(
        self,
        *,
        continuation_token: str | None = None,
        limit: int = 1000,
    ) -> IncompleteMultipartPage: ...

    def abort_incomplete_multipart(
        self,
        upload: IncompleteMultipartUpload,
    ) -> MultipartAbortResult: ...


@dataclass(frozen=True, kw_only=True)
class ObjectGcCandidate:
    sha256: str
    size_bytes: int
    locator: str
    last_modified: datetime
    delete_token: str
    generation: int

    def __post_init__(self) -> None:
        ObjectInventoryEntry(
            sha256=self.sha256,
            size_bytes=self.size_bytes,
            locator=self.locator,
            last_modified=self.last_modified,
            delete_token=self.delete_token,
        )
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("GC candidate generation must be a non-negative integer")


@dataclass(frozen=True, kw_only=True)
class ObjectGcPlan:
    plan_id: str
    created_at: datetime
    grace_before: datetime
    candidates: tuple[ObjectGcCandidate, ...]
    next_token: str | None = None

    def __post_init__(self) -> None:
        if not is_content_sha256(self.plan_id):
            raise ValueError("GC plan_id must be a lowercase SHA-256 digest")
        _aware_datetime(self.created_at, "GC plan created_at")
        _aware_datetime(self.grace_before, "GC plan grace_before")
        if self.grace_before > self.created_at:
            raise ValueError("GC plan grace_before cannot be after created_at")
        if type(self.candidates) is not tuple or any(
            not isinstance(candidate, ObjectGcCandidate) for candidate in self.candidates
        ):
            raise TypeError("GC candidates must be a tuple of ObjectGcCandidate")
        _optional_token(self.next_token, "GC plan next_token")


@dataclass(frozen=True, kw_only=True)
class ObjectGcReceipt:
    plan_id: str
    sha256: str
    candidate_generation: int
    observed_generation: int
    status: _GcStatus
    recorded_at: datetime

    def __post_init__(self) -> None:
        if not is_content_sha256(self.plan_id):
            raise ValueError("GC receipt plan_id must be a lowercase SHA-256 digest")
        if not is_content_sha256(self.sha256):
            raise ValueError("GC receipt sha256 must be a lowercase SHA-256 digest")
        if type(self.candidate_generation) is not int or self.candidate_generation < 0:
            raise ValueError("GC receipt candidate_generation must be non-negative")
        if type(self.observed_generation) is not int or self.observed_generation < 0:
            raise ValueError("GC receipt observed_generation must be non-negative")
        if type(self.status) is not str or self.status not in get_args(_GcStatus):
            raise ValueError("GC receipt status is outside the portable vocabulary")
        _aware_datetime(self.recorded_at, "GC receipt recorded_at")


__all__ = [
    "ObjectInventoryEntry",
    "ObjectInventoryPage",
    "ObjectDeleteResult",
    "IncompleteMultipartUpload",
    "IncompleteMultipartPage",
    "MultipartAbortResult",
    "ObjectStoreAdmin",
    "ObjectGcCandidate",
    "ObjectGcPlan",
    "ObjectGcReceipt",
]
