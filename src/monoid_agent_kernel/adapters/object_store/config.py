"""Validated S3-compatible object-store configuration without optional imports."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit


_BUCKET_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{1,253}[A-Za-z0-9]\Z", re.ASCII)
_REGION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)
_PREFIX_SEGMENT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z", re.ASCII)
_ACCOUNT_PATTERN = re.compile(r"[0-9]{12}\Z", re.ASCII)
_MIN_MULTIPART_PART_BYTES = 5 * 1024 * 1024
_MAX_MULTIPART_PART_BYTES = 5 * 1024 * 1024 * 1024
_MAX_S3_OBJECT_BYTES = 5 * 1024 * 1024 * 1024 * 1024
_MAX_MULTIPART_PARTS = 10_000


def _positive_timeout(value: object, field_name: str) -> None:
    if (
        type(value) not in {int, float}
        or isinstance(value, bool)
        or not 0 < float(value) <= 3600
    ):
        raise ValueError(f"{field_name} must be in the range (0, 3600]")


@dataclass(frozen=True, kw_only=True)
class S3ObjectStoreConfig:
    """Location, integrity, retry, and bounded-upload policy for one S3 bucket."""

    bucket: str
    prefix: str = ""
    region_name: str = "us-east-1"
    endpoint_url: str | None = field(default=None, repr=False)
    addressing_style: Literal["auto", "path", "virtual"] = "auto"
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 60.0
    sdk_max_attempts: int = 4
    max_conflict_retries: int = 3
    multipart_threshold_bytes: int = 8 * 1024 * 1024
    multipart_part_bytes: int = 8 * 1024 * 1024
    max_object_bytes: int = 5 * 1024 * 1024 * 1024
    expected_bucket_owner: str | None = None
    server_side_encryption: Literal["AES256", "aws:kms"] | None = None
    sse_kms_key_id: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.bucket) is not str or _BUCKET_PATTERN.fullmatch(self.bucket) is None:
            raise ValueError("S3 bucket must be a bounded ASCII bucket identity")
        if type(self.prefix) is not str or len(self.prefix) > 512:
            raise ValueError("S3 object prefix must be a bounded string")
        if self.prefix:
            if self.prefix.startswith("/") or self.prefix.endswith("/"):
                raise ValueError("S3 object prefix cannot start or end with a slash")
            segments = self.prefix.split("/")
            if any(_PREFIX_SEGMENT_PATTERN.fullmatch(segment) is None for segment in segments):
                raise ValueError("S3 object prefix must contain bounded safe path segments")
        if type(self.region_name) is not str or _REGION_PATTERN.fullmatch(self.region_name) is None:
            raise ValueError("S3 region_name must be a bounded ASCII region identity")
        if self.endpoint_url is not None:
            if type(self.endpoint_url) is not str or len(self.endpoint_url) > 2048:
                raise ValueError("S3 endpoint_url must be a bounded HTTP(S) URL")
            parsed = urlsplit(self.endpoint_url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("S3 endpoint_url cannot contain credentials, query, or fragment")
        if self.addressing_style not in {"auto", "path", "virtual"}:
            raise ValueError("S3 addressing_style is outside the supported vocabulary")
        _positive_timeout(self.connect_timeout_s, "S3 connect_timeout_s")
        _positive_timeout(self.read_timeout_s, "S3 read_timeout_s")
        if type(self.sdk_max_attempts) is not int or not 1 <= self.sdk_max_attempts <= 20:
            raise ValueError("S3 sdk_max_attempts must be between 1 and 20")
        if (
            type(self.max_conflict_retries) is not int
            or not 0 <= self.max_conflict_retries <= 20
        ):
            raise ValueError("S3 max_conflict_retries must be between 0 and 20")
        if (
            type(self.multipart_part_bytes) is not int
            or not _MIN_MULTIPART_PART_BYTES
            <= self.multipart_part_bytes
            <= _MAX_MULTIPART_PART_BYTES
        ):
            raise ValueError("S3 multipart_part_bytes must be between 5 MiB and 5 GiB")
        if (
            type(self.max_object_bytes) is not int
            or not 1 <= self.max_object_bytes <= _MAX_S3_OBJECT_BYTES
        ):
            raise ValueError("S3 max_object_bytes must be between 1 byte and 5 TiB")
        if (
            type(self.multipart_threshold_bytes) is not int
            or not 1
            <= self.multipart_threshold_bytes
            <= min(self.max_object_bytes, _MAX_MULTIPART_PART_BYTES)
        ):
            raise ValueError(
                "S3 multipart_threshold_bytes must fit within max_object_bytes and 5 GiB"
            )
        if self.max_object_bytes > self.multipart_part_bytes * _MAX_MULTIPART_PARTS:
            raise ValueError("S3 max_object_bytes would exceed the 10,000 multipart part limit")
        if self.expected_bucket_owner is not None and (
            type(self.expected_bucket_owner) is not str
            or _ACCOUNT_PATTERN.fullmatch(self.expected_bucket_owner) is None
        ):
            raise ValueError("S3 expected_bucket_owner must be a 12-digit account id")
        if self.server_side_encryption not in {None, "AES256", "aws:kms"}:
            raise ValueError("S3 server_side_encryption is outside the supported vocabulary")
        if self.server_side_encryption == "aws:kms":
            if (
                type(self.sse_kms_key_id) is not str
                or not self.sse_kms_key_id
                or len(self.sse_kms_key_id) > 2048
                or not self.sse_kms_key_id.isascii()
                or not all(character.isprintable() for character in self.sse_kms_key_id)
            ):
                raise ValueError("S3 aws:kms encryption requires a bounded printable key id")
        elif self.sse_kms_key_id is not None:
            raise ValueError("S3 sse_kms_key_id requires aws:kms encryption")

    def object_key(self, sha256: str) -> str:
        base = f"sha256/{sha256[:2]}/{sha256}"
        return f"{self.prefix}/{base}" if self.prefix else base


__all__ = ["S3ObjectStoreConfig"]
