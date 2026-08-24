"""S3-compatible immutable content-addressed blob storage."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Mapping

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
    status, code = _error_details(exc)
    return status == 404 or code in {"404", "NoSuchKey", "NotFound"}


def _is_precondition(exc: BaseException) -> bool:
    status, code = _error_details(exc)
    return status == 412 or code == "PreconditionFailed"


def _is_conditional_conflict(exc: BaseException) -> bool:
    status, code = _error_details(exc)
    return status == 409 or code in {"ConditionalRequestConflict", "OperationAborted"}


class S3ContentAddressedBlobStore:
    """Write-once SHA-256 objects using S3 conditional single and multipart requests."""

    def __init__(self, config: S3ObjectStoreConfig, *, client: object | None = None) -> None:
        if not isinstance(config, S3ObjectStoreConfig):
            raise TypeError("S3 content-addressed store config must be S3ObjectStoreConfig")
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

    def _locator(self, key: str) -> str:
        return f"s3://{self.config.bucket}/{key}"

    def _location(self, sha256: str) -> dict[str, object]:
        kwargs: dict[str, object] = {"Bucket": self.config.bucket, "Key": self._key(sha256)}
        if self.config.expected_bucket_owner is not None:
            kwargs["ExpectedBucketOwner"] = self.config.expected_bucket_owner
        return kwargs

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

    @staticmethod
    def _raise_failure(operation: str, exc: BaseException) -> None:
        http_status, error_code = _error_details(exc)
        raise S3ObjectStoreFailure(
            operation,
            http_status=http_status,
            error_code=error_code,
        ) from None

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
        stat = self._stat_from_response(sha256, response)
        body = response.get("Body") if isinstance(response, Mapping) else None
        if body is None or not callable(getattr(body, "read", None)):
            raise BlobCorrupt("S3 checked read response has no byte stream")
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


__all__ = [
    "S3DependencyMissing",
    "S3ObjectStoreFailure",
    "S3ContentAddressedBlobStore",
]
