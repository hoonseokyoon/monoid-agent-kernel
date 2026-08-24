"""S3-compatible immutable content-addressed blob storage."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from monoid_agent_kernel.core.json_ingress import portable_type_name
from monoid_agent_kernel.hosting.blobs import (
    BlobCorrupt,
    BlobNotFound,
    BlobPutResult,
    BlobStat,
    BlobStoreConflict,
    BlobStoreError,
    BlobTooLarge,
    is_content_sha256,
)
from monoid_agent_kernel.hosting.object_store_admin import (
    IncompleteMultipartPage,
    IncompleteMultipartUpload,
    MultipartAbortResult,
    ObjectDeleteResult,
    ObjectInventoryEntry,
    ObjectInventoryPage,
)

from .config import S3ObjectStoreConfig


_METADATA_SCHEMA = "content-addressed-blob-v1"


class S3DependencyMissing(RuntimeError):
    """The optional boto3 dependency is unavailable."""


class S3ObjectStoreFailure(BlobStoreError):
    """An S3 request failed without a portable storage classification."""

    def __init__(self, operation: str, *, http_status: int | None = None, error_code: str = "") -> None:
        safe_code = (
            error_code
            if error_code
            and len(error_code) <= 128
            and error_code.isascii()
            and all(character.isalnum() or character in "._-" for character in error_code)
            else ""
        )
        safe_status = http_status if type(http_status) is int and 100 <= http_status <= 599 else None
        detail = ""
        if safe_status is not None:
            detail = f" (HTTP {safe_status}"
            detail += f", {safe_code})" if safe_code else ")"
        elif safe_code:
            detail = f" ({safe_code})"
        super().__init__(f"S3 {operation} failed{detail}")
        self.operation = operation
        self.http_status = safe_status
        self.error_code = safe_code


@dataclass(frozen=True, kw_only=True)
class S3ObjectStoreDoctorReport:
    """Public-safe reachability and capability report with no location or credential fields."""

    ok: bool
    reachable: bool
    versioning_enabled: bool
    conditional_create_supported: bool = True
    checked_read_supported: bool = True
    object_inventory_supported: bool = True
    versioned_delete_supported: bool = True
    multipart_cleanup_supported: bool = True
    encryption_configured: bool = False
    errors: tuple[str, ...] = ()


def _base64_sha256(data: bytes | memoryview) -> str:
    return base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")


def _composite_sha256(part_digests: list[bytes]) -> str:
    composite = base64.b64encode(hashlib.sha256(b"".join(part_digests)).digest()).decode("ascii")
    return f"{composite}-{len(part_digests)}"


def _error_details(exc: BaseException) -> tuple[int | None, str]:
    response = getattr(exc, "response", None)
    if not isinstance(response, Mapping):
        return None, ""
    metadata = response.get("ResponseMetadata")
    error = response.get("Error")
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
    code = error.get("Code") if isinstance(error, Mapping) else ""
    return (status if type(status) is int else None, code if type(code) is str else "")


def _is_missing(exc: BaseException) -> bool:
    _, code = _error_details(exc)
    return code in {"404", "NoSuchKey", "NotFound"}


def _is_precondition(exc: BaseException) -> bool:
    status, code = _error_details(exc)
    return status == 412 or code == "PreconditionFailed"


def _is_conditional_conflict(exc: BaseException) -> bool:
    status, code = _error_details(exc)
    return status == 409 or code in {"ConditionalRequestConflict", "OperationAborted"}


class _S3Client:
    def __init__(self, config: S3ObjectStoreConfig, *, client: object | None = None) -> None:
        if not isinstance(config, S3ObjectStoreConfig):
            raise TypeError("S3 adapter config must be S3ObjectStoreConfig")
        self.config = config
        self._client = client if client is not None else self._build_client()

    def _build_client(self) -> object:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover - exercised in installed import smoke
            raise S3DependencyMissing(
                "S3 object-store support requires the 'object-store-s3' extra"
            ) from exc
        sdk_config = Config(
            signature_version="s3v4",
            connect_timeout=float(self.config.connect_timeout_s),
            read_timeout=float(self.config.read_timeout_s),
            retries={"max_attempts": self.config.sdk_max_attempts, "mode": "standard"},
            s3={"addressing_style": self.config.addressing_style},
            user_agent_extra="monoid-agent-kernel-v0.23",
        )
        return boto3.client(
            "s3",
            endpoint_url=self.config.endpoint_url,
            region_name=self.config.region_name,
            config=sdk_config,
        )

    def _key(self, sha256: str) -> str:
        return self.config.object_key(sha256)

    def _inventory_prefix(self) -> str:
        return f"{self.config.prefix}/sha256/" if self.config.prefix else "sha256/"

    def _digest_from_key(self, key: object) -> str | None:
        if type(key) is not str:
            return None
        prefix = self._inventory_prefix()
        if not key.startswith(prefix):
            return None
        suffix = key[len(prefix) :]
        parts = suffix.split("/")
        if len(parts) != 2:
            return None
        shard, sha256 = parts
        if not is_content_sha256(sha256) or shard != sha256[:2]:
            return None
        return sha256

    def _locator(self, key: str) -> str:
        return f"s3://{self.config.bucket}/{key}"

    def _location(self, sha256: str) -> dict[str, object]:
        kwargs: dict[str, object] = {"Bucket": self.config.bucket, "Key": self._key(sha256)}
        if self.config.expected_bucket_owner is not None:
            kwargs["ExpectedBucketOwner"] = self.config.expected_bucket_owner
        return kwargs

    @staticmethod
    def _raise_failure(operation: str, exc: BaseException) -> None:
        http_status, error_code = _error_details(exc)
        raise S3ObjectStoreFailure(
            operation,
            http_status=http_status,
            error_code=error_code,
        ) from None


class S3ContentAddressedBlobStore(_S3Client):
    """Write-once SHA-256 objects using S3 conditional single and multipart requests."""

    def _encryption(self) -> dict[str, object]:
        kwargs: dict[str, object] = {}
        if self.config.server_side_encryption is not None:
            kwargs["ServerSideEncryption"] = self.config.server_side_encryption
        if self.config.sse_kms_key_id is not None:
            kwargs["SSEKMSKeyId"] = self.config.sse_kms_key_id
        return kwargs

    def _metadata(
        self,
        sha256: str,
        size_bytes: int,
        *,
        upload: str,
        part_count: int,
        composite_sha256: str = "",
    ) -> dict[str, str]:
        metadata = {
            "monoid-schema": _METADATA_SCHEMA,
            "monoid-sha256": sha256,
            "monoid-size": str(size_bytes),
            "monoid-upload": upload,
            "monoid-part-count": str(part_count),
        }
        if composite_sha256:
            metadata["monoid-composite-sha256"] = composite_sha256
        return metadata

    def _stat_from_response(
        self,
        sha256: str,
        response: object,
    ) -> BlobStat:
        if not isinstance(response, Mapping):
            raise BlobCorrupt("S3 object metadata response is malformed")
        size_bytes = response.get("ContentLength")
        metadata = response.get("Metadata")
        checksum = response.get("ChecksumSHA256")
        checksum_type = response.get("ChecksumType")
        if (
            type(size_bytes) is not int
            or size_bytes < 0
            or not isinstance(metadata, Mapping)
            or any(type(key) is not str or type(value) is not str for key, value in metadata.items())
        ):
            raise BlobCorrupt("S3 object metadata response is malformed")
        normalized = {str(key).casefold(): str(value) for key, value in metadata.items()}
        if (
            normalized.get("monoid-schema") != _METADATA_SCHEMA
            or normalized.get("monoid-sha256") != sha256
            or normalized.get("monoid-size") != str(size_bytes)
        ):
            raise BlobCorrupt("S3 object metadata disagrees with its content address")
        upload = normalized.get("monoid-upload")
        part_count = normalized.get("monoid-part-count")
        if upload == "single":
            expected_checksum = base64.b64encode(bytes.fromhex(sha256)).decode("ascii")
            if part_count != "1" or checksum != expected_checksum:
                raise BlobCorrupt("S3 single object checksum evidence is missing or invalid")
            if checksum_type not in {None, "FULL_OBJECT"}:
                raise BlobCorrupt("S3 single object checksum type is invalid")
        elif upload == "multipart":
            expected_checksum = normalized.get("monoid-composite-sha256")
            if (
                type(expected_checksum) is not str
                or not expected_checksum
                or type(part_count) is not str
                or not part_count.isascii()
                or not part_count.isdecimal()
                or not 2 <= int(part_count) <= 10_000
                or checksum != expected_checksum
                or not expected_checksum.endswith(f"-{int(part_count)}")
            ):
                raise BlobCorrupt("S3 multipart object checksum evidence is missing or invalid")
            encoded_checksum = expected_checksum.rsplit("-", 1)[0]
            try:
                decoded_checksum = base64.b64decode(encoded_checksum, validate=True)
            except (ValueError, TypeError):
                raise BlobCorrupt("S3 multipart object checksum evidence is malformed") from None
            if len(decoded_checksum) != hashlib.sha256().digest_size:
                raise BlobCorrupt("S3 multipart object checksum evidence is malformed")
            if checksum_type not in {None, "COMPOSITE"}:
                raise BlobCorrupt("S3 multipart object checksum type is invalid")
        else:
            raise BlobCorrupt("S3 object upload profile is missing or invalid")
        return BlobStat(
            sha256=sha256,
            size_bytes=size_bytes,
            locator=self._locator(self._key(sha256)),
        )

    def stat(self, sha256: str) -> BlobStat | None:
        if not is_content_sha256(sha256):
            raise ValueError("S3 stat sha256 must be a lowercase SHA-256 digest")
        try:
            response = self._client.head_object(  # type: ignore[attr-defined]
                **self._location(sha256),
                ChecksumMode="ENABLED",
            )
        except Exception as exc:
            if _is_missing(exc):
                return None
            self._raise_failure("head_object", exc)
        return self._stat_from_response(sha256, response)

    def _require_stat(self, sha256: str) -> BlobStat:
        stat = self.stat(sha256)
        if stat is None:
            raise BlobNotFound("S3 object disappeared before checked metadata readback")
        return stat

    def get_checked(self, sha256: str) -> bytes:
        if not is_content_sha256(sha256):
            raise ValueError("S3 get sha256 must be a lowercase SHA-256 digest")
        try:
            response = self._client.get_object(  # type: ignore[attr-defined]
                **self._location(sha256),
                ChecksumMode="ENABLED",
            )
        except Exception as exc:
            if _is_missing(exc):
                raise BlobNotFound("S3 content address was not found") from None
            self._raise_failure("get_object", exc)
        body = response.get("Body") if isinstance(response, Mapping) else None
        if body is None:
            raise BlobCorrupt("S3 checked read response has no byte stream")
        try:
            if not callable(getattr(body, "read", None)):
                raise BlobCorrupt("S3 checked read response has no byte stream")
            stat = self._stat_from_response(sha256, response)
            try:
                data = body.read(self.config.max_object_bytes + 1)
            except Exception as exc:
                self._raise_failure("get_object body read", exc)
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        if type(data) is not bytes:
            raise BlobCorrupt("S3 checked read did not return bytes")
        if (
            len(data) != stat.size_bytes
            or len(data) > self.config.max_object_bytes
            or hashlib.sha256(data).hexdigest() != sha256
        ):
            raise BlobCorrupt("S3 object bytes failed content-address verification")
        return data

    def _existing_result(self, sha256: str) -> BlobPutResult:
        self.get_checked(sha256)
        return BlobPutResult(status="already_present", stat=self._require_stat(sha256))

    def _try_existing_result(self, sha256: str) -> BlobPutResult | None:
        try:
            return self._existing_result(sha256)
        except BlobNotFound:
            return None

    def put_if_absent(self, sha256: str, data: bytes) -> BlobPutResult:
        if not is_content_sha256(sha256):
            raise ValueError("S3 put sha256 must be a lowercase SHA-256 digest")
        if type(data) is not bytes:
            raise TypeError("S3 put data must be bytes")
        if len(data) > self.config.max_object_bytes:
            raise BlobTooLarge("S3 object exceeds the configured maximum size")
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError("S3 put bytes do not match the supplied content address")
        if (
            len(data) < self.config.multipart_threshold_bytes
            or len(data) <= self.config.multipart_part_bytes
        ):
            return self._put_single(sha256, data)
        return self._put_multipart(sha256, data)

    def _put_single(self, sha256: str, data: bytes) -> BlobPutResult:
        checksum = _base64_sha256(data)
        for _ in range(self.config.max_conflict_retries + 1):
            try:
                response = self._client.put_object(  # type: ignore[attr-defined]
                    **self._location(sha256),
                    **self._encryption(),
                    Body=data,
                    ContentLength=len(data),
                    ContentType="application/octet-stream",
                    IfNoneMatch="*",
                    ChecksumSHA256=checksum,
                    Metadata=self._metadata(
                        sha256,
                        len(data),
                        upload="single",
                        part_count=1,
                    ),
                )
            except Exception as exc:
                if _is_precondition(exc):
                    existing = self._try_existing_result(sha256)
                    if existing is not None:
                        return existing
                    continue
                if _is_conditional_conflict(exc):
                    existing = self._try_existing_result(sha256)
                    if existing is not None:
                        return existing
                    continue
                existing = self._try_existing_result(sha256)
                if existing is not None:
                    return existing
                self._raise_failure("put_object", exc)
            if not isinstance(response, Mapping):
                raise BlobCorrupt("S3 put response is malformed")
            returned_checksum = response.get("ChecksumSHA256")
            if returned_checksum is not None and returned_checksum != checksum:
                raise BlobCorrupt("S3 put response checksum disagrees with caller bytes")
            return BlobPutResult(status="stored", stat=self._require_stat(sha256))
        raise BlobStoreConflict("S3 conditional single PUT exhausted its retry budget") from None

    def _multipart_parts(self, data: bytes) -> tuple[list[bytes], str]:
        view = memoryview(data)
        part_digests: list[bytes] = []
        for start in range(0, len(data), self.config.multipart_part_bytes):
            end = min(start + self.config.multipart_part_bytes, len(data))
            part_digests.append(hashlib.sha256(view[start:end]).digest())
        if not 2 <= len(part_digests) <= 10_000:
            raise BlobTooLarge("S3 multipart upload requires between 2 and 10,000 parts")
        return part_digests, _composite_sha256(part_digests)

    def _abort(self, sha256: str, upload_id: str) -> None:
        try:
            self._client.abort_multipart_upload(  # type: ignore[attr-defined]
                **self._location(sha256),
                UploadId=upload_id,
            )
        except Exception:
            return

    def _put_multipart(self, sha256: str, data: bytes) -> BlobPutResult:
        part_digests, composite_checksum = self._multipart_parts(data)
        for _ in range(self.config.max_conflict_retries + 1):
            upload_id = ""
            completion_started = False
            try:
                created = self._client.create_multipart_upload(  # type: ignore[attr-defined]
                    **self._location(sha256),
                    **self._encryption(),
                    ContentType="application/octet-stream",
                    ChecksumAlgorithm="SHA256",
                    ChecksumType="COMPOSITE",
                    Metadata=self._metadata(
                        sha256,
                        len(data),
                        upload="multipart",
                        part_count=len(part_digests),
                        composite_sha256=composite_checksum,
                    ),
                )
                upload_id_value = created.get("UploadId") if isinstance(created, Mapping) else None
                if type(upload_id_value) is not str or not upload_id_value:
                    raise BlobCorrupt("S3 multipart create response has no upload id")
                upload_id = upload_id_value
                completed_parts: list[dict[str, object]] = []
                for index, part_digest in enumerate(part_digests, start=1):
                    start = (index - 1) * self.config.multipart_part_bytes
                    end = min(start + self.config.multipart_part_bytes, len(data))
                    part = data[start:end]
                    part_checksum = base64.b64encode(part_digest).decode("ascii")
                    uploaded = self._client.upload_part(  # type: ignore[attr-defined]
                        **self._location(sha256),
                        UploadId=upload_id,
                        PartNumber=index,
                        Body=part,
                        ContentLength=len(part),
                        ChecksumSHA256=part_checksum,
                    )
                    etag = uploaded.get("ETag") if isinstance(uploaded, Mapping) else None
                    returned_checksum = (
                        uploaded.get("ChecksumSHA256") if isinstance(uploaded, Mapping) else None
                    )
                    if type(etag) is not str or not etag:
                        raise BlobCorrupt("S3 multipart upload response has no ETag")
                    if returned_checksum is not None and returned_checksum != part_checksum:
                        raise BlobCorrupt("S3 multipart part checksum disagrees with caller bytes")
                    completed_parts.append(
                        {
                            "ETag": etag,
                            "PartNumber": index,
                            "ChecksumSHA256": part_checksum,
                        }
                    )
                completion_started = True
                completed = self._client.complete_multipart_upload(  # type: ignore[attr-defined]
                    **self._location(sha256),
                    UploadId=upload_id,
                    MultipartUpload={"Parts": completed_parts},
                    IfNoneMatch="*",
                    ChecksumType="COMPOSITE",
                    MpuObjectSize=len(data),
                )
                if not isinstance(completed, Mapping):
                    raise BlobCorrupt("S3 multipart completion response is malformed")
                returned_checksum = completed.get("ChecksumSHA256")
                returned_type = completed.get("ChecksumType")
                if returned_checksum is not None and returned_checksum != composite_checksum:
                    raise BlobCorrupt("S3 multipart completion checksum is invalid")
                if returned_type not in {None, "COMPOSITE"}:
                    raise BlobCorrupt("S3 multipart completion checksum type is invalid")
                return BlobPutResult(status="stored", stat=self._require_stat(sha256))
            except Exception as exc:
                if upload_id:
                    self._abort(sha256, upload_id)
                if _is_precondition(exc):
                    existing = self._try_existing_result(sha256)
                    if existing is not None:
                        return existing
                    continue
                if _is_conditional_conflict(exc):
                    existing = self._try_existing_result(sha256)
                    if existing is not None:
                        return existing
                    continue
                if completion_started:
                    existing = self._try_existing_result(sha256)
                    if existing is not None:
                        return existing
                if isinstance(exc, (BlobCorrupt, BlobTooLarge, BlobStoreError)):
                    raise
                self._raise_failure("multipart upload", exc)
        raise BlobStoreConflict("S3 conditional multipart PUT exhausted its retry budget") from None


class S3ObjectStoreAdmin(_S3Client):
    """Privileged bounded inventory, safe deletion, and multipart cleanup surface.

    Object inventory and deletion require bucket versioning and target the exact inventoried
    version. This prevents same-content recreation from satisfying a stale garbage-collection
    token. Multipart cleanup remains available independently of object versioning.
    """

    def doctor(self) -> S3ObjectStoreDoctorReport:
        """Probe bucket reachability/versioning without writing an object or naming its location."""

        errors: list[str] = []
        reachable = False
        versioning_enabled = False
        kwargs: dict[str, object] = {"Bucket": self.config.bucket}
        if self.config.expected_bucket_owner is not None:
            kwargs["ExpectedBucketOwner"] = self.config.expected_bucket_owner
        try:
            self._client.head_bucket(**kwargs)  # type: ignore[attr-defined]
            reachable = True
        except Exception as exc:  # noqa: BLE001 - doctor emits only the portable type
            errors.append(f"reachability: {portable_type_name(exc)}")
        if reachable:
            try:
                response = self._client.get_bucket_versioning(**kwargs)  # type: ignore[attr-defined]
                versioning_enabled = (
                    isinstance(response, Mapping) and response.get("Status") == "Enabled"
                )
                if not versioning_enabled:
                    errors.append("versioning: disabled")
            except Exception as exc:  # noqa: BLE001 - doctor emits only the portable type
                errors.append(f"versioning: {portable_type_name(exc)}")
        return S3ObjectStoreDoctorReport(
            ok=reachable and versioning_enabled,
            reachable=reachable,
            versioning_enabled=versioning_enabled,
            versioned_delete_supported=versioning_enabled,
            encryption_configured=self.config.server_side_encryption is not None,
            errors=tuple(errors),
        )

    def _require_enabled_versioning(self) -> None:
        kwargs: dict[str, object] = {"Bucket": self.config.bucket}
        if self.config.expected_bucket_owner is not None:
            kwargs["ExpectedBucketOwner"] = self.config.expected_bucket_owner
        try:
            response = self._client.get_bucket_versioning(**kwargs)  # type: ignore[attr-defined]
        except Exception as exc:
            self._raise_failure("get_bucket_versioning", exc)
        if not isinstance(response, Mapping) or response.get("Status") != "Enabled":
            raise BlobCorrupt("S3 object administration requires enabled bucket versioning")

    @staticmethod
    def _page_limit(limit: object) -> int:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("S3 admin page limit must be between 1 and 1000")
        return limit

    @staticmethod
    def _token(value: object, field_name: str) -> str | None:
        if value is None:
            return None
        if (
            type(value) is not str
            or not value
            or len(value) > 8192
            or not value.isascii()
            or not all(character.isprintable() for character in value)
        ):
            raise ValueError(f"S3 {field_name} must be bounded printable ASCII")
        return value

    def inventory_page(
        self,
        *,
        continuation_token: str | None = None,
        limit: int = 1000,
    ) -> ObjectInventoryPage:
        self._require_enabled_versioning()
        checked_limit = self._page_limit(limit)
        checked_token = self._token(continuation_token, "inventory continuation_token")
        kwargs: dict[str, object] = {
            "Bucket": self.config.bucket,
            "Prefix": self._inventory_prefix(),
            "MaxKeys": checked_limit,
        }
        if checked_token is not None:
            kwargs["ContinuationToken"] = checked_token
        if self.config.expected_bucket_owner is not None:
            kwargs["ExpectedBucketOwner"] = self.config.expected_bucket_owner
        try:
            response = self._client.list_objects_v2(**kwargs)  # type: ignore[attr-defined]
        except Exception as exc:
            self._raise_failure("list_objects_v2", exc)
        if not isinstance(response, Mapping):
            raise BlobCorrupt("S3 object inventory response is malformed")
        raw_entries = response.get("Contents", ())
        if not isinstance(raw_entries, (list, tuple)):
            raise BlobCorrupt("S3 object inventory contents are malformed")
        entries: list[ObjectInventoryEntry] = []
        for raw in raw_entries:
            if not isinstance(raw, Mapping):
                raise BlobCorrupt("S3 object inventory entry is malformed")
            sha256 = self._digest_from_key(raw.get("Key"))
            if sha256 is None:
                continue
            size_bytes = raw.get("Size")
            last_modified = raw.get("LastModified")
            delete_token = raw.get("ETag")
            if (
                type(size_bytes) is not int
                or size_bytes < 0
                or not isinstance(last_modified, datetime)
                or type(delete_token) is not str
            ):
                raise BlobCorrupt("S3 object inventory entry is malformed")
            try:
                current = self._client.head_object(  # type: ignore[attr-defined]
                    **self._location(sha256)
                )
            except Exception as exc:
                if _is_missing(exc):
                    continue
                self._raise_failure("head_object for versioned inventory", exc)
            if not isinstance(current, Mapping):
                raise BlobCorrupt("S3 versioned inventory metadata is malformed")
            current_size = current.get("ContentLength")
            current_modified = current.get("LastModified")
            current_etag = current.get("ETag")
            current_version = current.get("VersionId")
            if (
                type(current_size) is not int
                or current_size < 0
                or not isinstance(current_modified, datetime)
                or type(current_etag) is not str
                or type(current_version) is not str
                or not current_version
                or current_version == "null"
            ):
                raise BlobCorrupt("S3 versioned inventory metadata is malformed")
            size_bytes = current_size
            last_modified = current_modified
            delete_token = self._encode_versioned_delete_token(
                current_etag,
                current_version,
            )
            entries.append(
                ObjectInventoryEntry(
                    sha256=sha256,
                    size_bytes=size_bytes,
                    locator=self._locator(self._key(sha256)),
                    last_modified=last_modified,
                    delete_token=delete_token,
                )
            )
        next_token = response.get("NextContinuationToken")
        if response.get("IsTruncated") is True:
            if type(next_token) is not str or not next_token:
                raise BlobCorrupt("S3 truncated object inventory has no continuation token")
            checked_next = self._token(next_token, "inventory next_token")
        else:
            checked_next = None
        return ObjectInventoryPage(entries=tuple(entries), next_token=checked_next)

    @staticmethod
    def _encode_versioned_delete_token(etag: str, version_id: str) -> str:
        raw = json.dumps(
            {"etag": etag, "version_id": version_id},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_versioned_delete_token(token: str) -> tuple[str, str]:
        try:
            padding = "=" * (-len(token) % 4)
            value = json.loads(base64.urlsafe_b64decode(token + padding).decode("ascii"))
        except Exception as exc:
            raise ValueError("S3 versioned delete_token is invalid") from exc
        if (
            not isinstance(value, dict)
            or set(value) != {"etag", "version_id"}
            or type(value["etag"]) is not str
            or not value["etag"]
            or type(value["version_id"]) is not str
            or not value["version_id"]
        ):
            raise ValueError("S3 versioned delete_token is invalid")
        return value["etag"], value["version_id"]

    def _target_version_etag(self, sha256: str, version_id: str) -> str | None:
        kwargs = self._location(sha256)
        kwargs["VersionId"] = version_id
        try:
            response = self._client.head_object(**kwargs)  # type: ignore[attr-defined]
        except Exception as exc:
            if _is_missing(exc) or _error_details(exc)[1] == "NoSuchVersion":
                return None
            self._raise_failure("head_object for version delete", exc)
        token = response.get("ETag") if isinstance(response, Mapping) else None
        observed_version = response.get("VersionId") if isinstance(response, Mapping) else None
        if (
            type(token) is not str
            or not token
            or type(observed_version) is not str
            or observed_version != version_id
        ):
            raise BlobCorrupt("S3 version delete metadata is malformed")
        return token

    def _version_exists(self, sha256: str, version_id: str) -> bool:
        return self._target_version_etag(sha256, version_id) is not None

    def delete_if_match(self, sha256: str, delete_token: str) -> ObjectDeleteResult:
        self._require_enabled_versioning()
        if not is_content_sha256(sha256):
            raise ValueError("S3 delete sha256 must be a lowercase SHA-256 digest")
        checked_token = self._token(delete_token, "delete_token")
        assert checked_token is not None
        expected_etag, expected_version = self._decode_versioned_delete_token(checked_token)
        current_etag = self._target_version_etag(sha256, expected_version)
        if current_etag is None:
            return ObjectDeleteResult(status="already_missing")
        if current_etag != expected_etag:
            return ObjectDeleteResult(status="precondition_failed")
        delete_kwargs = self._location(sha256)
        delete_kwargs["VersionId"] = expected_version
        try:
            self._client.delete_object(**delete_kwargs)  # type: ignore[attr-defined]
        except Exception as exc:
            if _is_missing(exc) or _error_details(exc)[1] == "NoSuchVersion":
                return ObjectDeleteResult(status="already_missing")
            if _is_precondition(exc):
                return ObjectDeleteResult(status="precondition_failed")
            if not self._version_exists(sha256, expected_version):
                return ObjectDeleteResult(status="deleted")
            self._raise_failure("delete_object", exc)
        return ObjectDeleteResult(status="deleted")

    @staticmethod
    def _encode_multipart_token(
        key_marker: str,
        upload_id_marker: str,
        *,
        bucket_scope: bool,
    ) -> str:
        raw = json.dumps(
            {"bucket_scope": bucket_scope, "key": key_marker, "upload": upload_id_marker},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_multipart_token(token: str) -> tuple[str, str, bool]:
        try:
            padding = "=" * (-len(token) % 4)
            value = json.loads(base64.urlsafe_b64decode(token + padding).decode("ascii"))
        except Exception as exc:
            raise ValueError("S3 multipart continuation_token is invalid") from exc
        if (
            not isinstance(value, dict)
            or set(value) != {"bucket_scope", "key", "upload"}
            or type(value["bucket_scope"]) is not bool
            or type(value["key"]) is not str
            or type(value["upload"]) is not str
            or not value["key"]
            or not value["upload"]
        ):
            raise ValueError("S3 multipart continuation_token is invalid")
        return value["key"], value["upload"], value["bucket_scope"]

    def incomplete_multipart_page(
        self,
        *,
        continuation_token: str | None = None,
        limit: int = 1000,
    ) -> IncompleteMultipartPage:
        checked_limit = self._page_limit(limit)
        checked_token = self._token(continuation_token, "multipart continuation_token")
        kwargs: dict[str, object] = {
            "Bucket": self.config.bucket,
            "Prefix": self._inventory_prefix(),
            "MaxUploads": checked_limit,
        }
        bucket_scope = False
        if checked_token is not None:
            key_marker, upload_marker, bucket_scope = self._decode_multipart_token(checked_token)
            kwargs["KeyMarker"] = key_marker
            kwargs["UploadIdMarker"] = upload_marker
            if bucket_scope:
                kwargs.pop("Prefix")
        if self.config.expected_bucket_owner is not None:
            kwargs["ExpectedBucketOwner"] = self.config.expected_bucket_owner
        try:
            response = self._client.list_multipart_uploads(**kwargs)  # type: ignore[attr-defined]
        except Exception as exc:
            self._raise_failure("list_multipart_uploads", exc)
        if not isinstance(response, Mapping):
            raise BlobCorrupt("S3 multipart inventory response is malformed")
        if (
            not bucket_scope
            and checked_token is None
            and not response.get("Uploads")
            and response.get("IsTruncated") is not True
        ):
            # Some S3-compatible services accept Prefix but return an empty multipart listing.
            # A bounded bucket-scoped page preserves correctness; exact key parsing below keeps
            # other namespaces out of the result.
            bucket_scope = True
            kwargs.pop("Prefix")
            try:
                response = self._client.list_multipart_uploads(  # type: ignore[attr-defined]
                    **kwargs
                )
            except Exception as exc:
                self._raise_failure("bucket-scoped list_multipart_uploads", exc)
            if not isinstance(response, Mapping):
                raise BlobCorrupt("S3 multipart inventory response is malformed")
        raw_uploads = response.get("Uploads", ())
        if not isinstance(raw_uploads, (list, tuple)):
            raise BlobCorrupt("S3 multipart inventory uploads are malformed")
        uploads: list[IncompleteMultipartUpload] = []
        for raw in raw_uploads:
            if not isinstance(raw, Mapping):
                raise BlobCorrupt("S3 multipart inventory entry is malformed")
            sha256 = self._digest_from_key(raw.get("Key"))
            if sha256 is None:
                continue
            upload_id = raw.get("UploadId")
            initiated = raw.get("Initiated")
            if type(upload_id) is not str or not isinstance(initiated, datetime):
                raise BlobCorrupt("S3 multipart inventory entry is malformed")
            uploads.append(
                IncompleteMultipartUpload(
                    sha256=sha256,
                    upload_id=upload_id,
                    initiated_at=initiated,
                )
            )
        if response.get("IsTruncated") is True:
            next_key = response.get("NextKeyMarker")
            next_upload = response.get("NextUploadIdMarker")
            if type(next_key) is not str or type(next_upload) is not str:
                raise BlobCorrupt("S3 truncated multipart inventory has no continuation markers")
            next_token = self._encode_multipart_token(
                next_key,
                next_upload,
                bucket_scope=bucket_scope,
            )
        else:
            next_token = None
        return IncompleteMultipartPage(uploads=tuple(uploads), next_token=next_token)

    def abort_incomplete_multipart(
        self,
        upload: IncompleteMultipartUpload,
    ) -> MultipartAbortResult:
        if not isinstance(upload, IncompleteMultipartUpload):
            raise TypeError("S3 multipart abort requires IncompleteMultipartUpload")
        for _ in range(self.config.max_conflict_retries + 1):
            try:
                self._client.abort_multipart_upload(  # type: ignore[attr-defined]
                    **self._location(upload.sha256),
                    UploadId=upload.upload_id,
                )
            except Exception as exc:
                if _is_missing(exc) or _error_details(exc)[1] == "NoSuchUpload":
                    return MultipartAbortResult(status="already_missing")
                if _is_precondition(exc):
                    return MultipartAbortResult(status="precondition_failed")
                self._raise_failure("abort_multipart_upload", exc)
            try:
                self._client.list_parts(  # type: ignore[attr-defined]
                    **self._location(upload.sha256),
                    UploadId=upload.upload_id,
                    MaxParts=1,
                )
            except Exception as exc:
                if _is_missing(exc) or _error_details(exc)[1] == "NoSuchUpload":
                    return MultipartAbortResult(status="aborted")
                self._raise_failure("list_parts after abort", exc)
        raise BlobStoreConflict("S3 multipart abort exhausted its retry budget")


__all__ = [
    "S3DependencyMissing",
    "S3ObjectStoreFailure",
    "S3ObjectStoreDoctorReport",
    "S3ContentAddressedBlobStore",
    "S3ObjectStoreAdmin",
]
