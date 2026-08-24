from __future__ import annotations

import base64
import hashlib
import subprocess
import sys
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import pytest

from monoid_agent_kernel.adapters.object_store import (
    S3ContentAddressedBlobStore,
    S3ObjectStoreConfig,
    S3ObjectStoreAdmin,
    S3ObjectStoreFailure,
)
from monoid_agent_kernel.hosting import (
    BlobCorrupt,
    BlobNotFound,
    BlobStoreConflict,
    BlobTooLarge,
)


_MIB = 1024 * 1024


class _S3Error(Exception):
    def __init__(self, status: int, code: str) -> None:
        super().__init__("simulated S3 failure")
        self.response = {
            "ResponseMetadata": {"HTTPStatusCode": status},
            "Error": {"Code": code},
        }


class _FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.uploads: dict[str, dict[str, Any]] = {}
        self.put_faults: list[str] = []
        self.complete_faults: list[str] = []
        self.put_calls = 0
        self.create_calls = 0
        self.upload_part_calls = 0
        self.complete_calls = 0
        self.abort_calls = 0
        self.list_calls = 0
        self.multipart_list_calls = 0
        self.last_body: BytesIO | None = None

    @staticmethod
    def _location(kwargs: dict[str, Any]) -> tuple[str, str]:
        return kwargs["Bucket"], kwargs["Key"]

    def _response(self, value: dict[str, Any], *, body: bool = False) -> dict[str, Any]:
        response = {
            "ContentLength": len(value["data"]),
            "Metadata": dict(value["metadata"]),
            "ChecksumSHA256": value["checksum"],
            "ChecksumType": value["checksum_type"],
            "ETag": value.get("etag", f'"{hashlib.md5(value["data"]).hexdigest()}"'),  # noqa: S324
            "LastModified": datetime(2026, 8, 23, tzinfo=UTC),
        }
        if body:
            self.last_body = BytesIO(value["data"])
            response["Body"] = self.last_body
        if "version_id" in value:
            response["VersionId"] = value["version_id"]
        return response

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        value = self.objects.get(self._location(kwargs))
        if value is None:
            raise _S3Error(404, "NoSuchKey")
        if kwargs.get("VersionId") not in {None, value.get("version_id")}:
            raise _S3Error(404, "NoSuchVersion")
        return self._response(value)

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        value = self.objects.get(self._location(kwargs))
        if value is None:
            raise _S3Error(404, "NoSuchKey")
        return self._response(value, body=True)

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.put_calls += 1
        location = self._location(kwargs)
        fault = self.put_faults.pop(0) if self.put_faults else ""
        if fault == "409":
            raise _S3Error(409, "ConditionalRequestConflict")
        if location in self.objects and kwargs.get("IfNoneMatch") == "*":
            raise _S3Error(412, "PreconditionFailed")
        data = kwargs["Body"]
        checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
        assert kwargs["ChecksumSHA256"] == checksum
        self.objects[location] = {
            "data": data,
            "metadata": dict(kwargs["Metadata"]),
            "checksum": checksum,
            "checksum_type": "FULL_OBJECT",
            "version_id": f"version-{self.put_calls}",
        }
        if fault == "response_lost":
            raise ConnectionError("simulated response loss")
        return {"ChecksumSHA256": checksum}

    def create_multipart_upload(self, **kwargs: Any) -> dict[str, Any]:
        self.create_calls += 1
        upload_id = f"upload-{self.create_calls}"
        assert kwargs["ChecksumAlgorithm"] == "SHA256"
        assert kwargs["ChecksumType"] == "COMPOSITE"
        self.uploads[upload_id] = {
            "location": self._location(kwargs),
            "metadata": dict(kwargs["Metadata"]),
            "parts": {},
            "initiated": datetime(2026, 8, 24, tzinfo=UTC),
        }
        return {"UploadId": upload_id}

    def upload_part(self, **kwargs: Any) -> dict[str, Any]:
        self.upload_part_calls += 1
        upload = self.uploads[kwargs["UploadId"]]
        data = kwargs["Body"]
        checksum = base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")
        assert kwargs["ChecksumSHA256"] == checksum
        upload["parts"][kwargs["PartNumber"]] = data
        return {"ETag": f'"part-{kwargs["PartNumber"]}"', "ChecksumSHA256": checksum}

    def complete_multipart_upload(self, **kwargs: Any) -> dict[str, Any]:
        self.complete_calls += 1
        upload = self.uploads[kwargs["UploadId"]]
        location = upload["location"]
        fault = self.complete_faults.pop(0) if self.complete_faults else ""
        if fault == "409":
            raise _S3Error(409, "ConditionalRequestConflict")
        if location in self.objects and kwargs.get("IfNoneMatch") == "*":
            raise _S3Error(412, "PreconditionFailed")
        assert kwargs["ChecksumType"] == "COMPOSITE"
        data = b"".join(upload["parts"][number] for number in sorted(upload["parts"]))
        part_digests = [
            hashlib.sha256(upload["parts"][number]).digest()
            for number in sorted(upload["parts"])
        ]
        checksum = (
            base64.b64encode(hashlib.sha256(b"".join(part_digests)).digest()).decode("ascii")
            + f"-{len(part_digests)}"
        )
        self.objects[location] = {
            "data": data,
            "metadata": upload["metadata"],
            "checksum": checksum,
            "checksum_type": "COMPOSITE",
            "version_id": f"multipart-version-{self.complete_calls}",
        }
        self.uploads.pop(kwargs["UploadId"], None)
        if fault == "response_lost":
            raise ConnectionError("simulated completion response loss")
        return {"ChecksumSHA256": checksum, "ChecksumType": "COMPOSITE"}

    def abort_multipart_upload(self, **kwargs: Any) -> None:
        self.abort_calls += 1
        if self.uploads.pop(kwargs["UploadId"], None) is None:
            raise _S3Error(404, "NoSuchUpload")

    def list_parts(self, **kwargs: Any) -> dict[str, Any]:
        upload = self.uploads.get(kwargs["UploadId"])
        if upload is None:
            raise _S3Error(404, "NoSuchUpload")
        return {"Parts": list(upload["parts"].values())}

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        self.list_calls += 1
        prefix = kwargs.get("Prefix", "")
        keys = sorted(key for bucket, key in self.objects if bucket == kwargs["Bucket"])
        keys = [key for key in keys if key.startswith(prefix)]
        token = kwargs.get("ContinuationToken")
        if token is not None:
            keys = [key for key in keys if key > token]
        selected = keys[: kwargs["MaxKeys"]]
        contents = []
        for key in selected:
            value = self.objects[(kwargs["Bucket"], key)]
            contents.append(
                {
                    "Key": key,
                    "Size": len(value["data"]),
                    "LastModified": datetime(2026, 8, 23, tzinfo=UTC),
                    "ETag": value.get(
                        "etag",
                        f'"{hashlib.md5(value["data"]).hexdigest()}"',  # noqa: S324
                    ),
                }
            )
        truncated = len(keys) > len(selected)
        return {
            "Contents": contents,
            "IsTruncated": truncated,
            "NextContinuationToken": selected[-1] if truncated else None,
        }

    def delete_object(self, **kwargs: Any) -> dict[str, Any]:
        location = self._location(kwargs)
        value = self.objects.get(location)
        if value is None:
            raise _S3Error(404, "NoSuchKey")
        if kwargs.get("VersionId") not in {None, value.get("version_id")}:
            raise _S3Error(404, "NoSuchVersion")
        etag = value.get("etag", f'"{hashlib.md5(value["data"]).hexdigest()}"')  # noqa: S324
        if kwargs.get("IfMatch") != etag:
            raise _S3Error(412, "PreconditionFailed")
        del self.objects[location]
        return {"ResponseMetadata": {"HTTPStatusCode": 204}}

    def list_multipart_uploads(self, **kwargs: Any) -> dict[str, Any]:
        self.multipart_list_calls += 1
        prefix = kwargs.get("Prefix", "")
        values = sorted(
            (
                upload["location"][1],
                upload_id,
                upload,
            )
            for upload_id, upload in self.uploads.items()
            if upload["location"][0] == kwargs["Bucket"]
            and upload["location"][1].startswith(prefix)
        )
        key_marker = kwargs.get("KeyMarker")
        upload_marker = kwargs.get("UploadIdMarker")
        if key_marker is not None:
            values = [
                value
                for value in values
                if (value[0], value[1]) > (key_marker, upload_marker)
            ]
        selected = values[: kwargs["MaxUploads"]]
        truncated = len(values) > len(selected)
        return {
            "Uploads": [
                {"Key": key, "UploadId": upload_id, "Initiated": upload["initiated"]}
                for key, upload_id, upload in selected
            ],
            "IsTruncated": truncated,
            "NextKeyMarker": selected[-1][0] if truncated else None,
            "NextUploadIdMarker": selected[-1][1] if truncated else None,
        }


class _SecretFailureS3(_FakeS3):
    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise RuntimeError("credential=must-never-reach-public-error")


class _MissingBucketS3(_FakeS3):
    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise _S3Error(404, "NoSuchBucket")

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise _S3Error(404, "NoSuchBucket")


def _config(**changes: Any) -> S3ObjectStoreConfig:
    values = {
        "bucket": "monoid-object-test",
        "prefix": "tenant-a/blobs",
        "multipart_threshold_bytes": 6 * _MIB,
        "multipart_part_bytes": 5 * _MIB,
        "max_object_bytes": 20 * _MIB,
    }
    values.update(changes)
    return S3ObjectStoreConfig(**values)


def _store(client: _FakeS3 | None = None, **changes: Any) -> tuple[S3ContentAddressedBlobStore, _FakeS3]:
    client = client or _FakeS3()
    return S3ContentAddressedBlobStore(_config(**changes), client=client), client


def _admin(
    client: _FakeS3 | None = None,
    **changes: Any,
) -> tuple[S3ObjectStoreAdmin, _FakeS3]:
    client = client or _FakeS3()
    return S3ObjectStoreAdmin(_config(**changes), client=client), client


def test_s3_config_is_bounded_and_hides_sensitive_values() -> None:
    config = _config(
        endpoint_url="https://objects.example.test",
        server_side_encryption="aws:kms",
        sse_kms_key_id="arn:aws:kms:us-east-1:123456789012:key/example",
    )

    assert "objects.example.test" not in repr(config)
    assert "key/example" not in repr(config)
    assert config.object_key("a" * 64) == f"tenant-a/blobs/sha256/aa/{'a' * 64}"

    with pytest.raises(ValueError, match="credentials"):
        _config(endpoint_url="https://user:secret@objects.example.test")
    with pytest.raises(ValueError, match="prefix"):
        _config(prefix="../private")
    with pytest.raises(ValueError, match="5 MiB"):
        _config(multipart_part_bytes=4 * _MIB)
    with pytest.raises(ValueError, match="10,000"):
        _config(multipart_part_bytes=5 * _MIB, max_object_bytes=50_001 * _MIB)
    with pytest.raises(ValueError, match="requires aws:kms"):
        _config(sse_kms_key_id="key-id")
    with pytest.raises(ValueError, match="admin_delete_mode"):
        _config(admin_delete_mode="unsafe")


def test_object_store_namespace_import_does_not_load_optional_sdks() -> None:
    script = """
import importlib.abc
import sys

class BlockOptionalSdk(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.', 1)[0] in {'boto3', 'botocore'}:
            raise ImportError(f'blocked optional SDK: {fullname}')
        return None

sys.meta_path.insert(0, BlockOptionalSdk())
import monoid_agent_kernel.adapters.object_store
"""

    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603


def test_single_put_is_conditional_checked_and_idempotent_without_list() -> None:
    store, client = _store()
    data = b"single object bytes"
    sha256 = hashlib.sha256(data).hexdigest()

    first = store.put_if_absent(sha256, data)
    second = store.put_if_absent(sha256, data)

    assert first.status == "stored"
    assert second.status == "already_present"
    assert first.stat == second.stat == store.stat(sha256)
    assert store.get_checked(sha256) == data
    assert first.stat.locator.endswith(f"/sha256/{sha256[:2]}/{sha256}")
    assert client.put_calls == 2
    assert client.list_calls == client.multipart_list_calls == 0


def test_single_put_retries_409_and_reconciles_response_loss() -> None:
    data = b"retryable single object"
    sha256 = hashlib.sha256(data).hexdigest()
    client = _FakeS3()
    client.put_faults = ["409"]
    store, _ = _store(client)

    assert store.put_if_absent(sha256, data).status == "stored"
    assert client.put_calls == 2

    lost_data = b"single response loss"
    lost_sha = hashlib.sha256(lost_data).hexdigest()
    client.put_faults = ["response_lost"]
    assert store.put_if_absent(lost_sha, lost_data).status == "already_present"
    assert store.get_checked(lost_sha) == lost_data


def test_single_put_exhausts_a_bounded_conflict_budget() -> None:
    data = b"persistent conflict"
    sha256 = hashlib.sha256(data).hexdigest()
    client = _FakeS3()
    client.put_faults = ["409", "409"]
    store, _ = _store(client, max_conflict_retries=1)

    with pytest.raises(BlobStoreConflict, match="retry budget"):
        store.put_if_absent(sha256, data)
    assert client.put_calls == 2


def test_adapter_failure_is_typed_and_does_not_chain_raw_sdk_text() -> None:
    data = b"safe failure projection"
    sha256 = hashlib.sha256(data).hexdigest()
    store, _ = _store(_SecretFailureS3())

    with pytest.raises(S3ObjectStoreFailure) as raised:
        store.put_if_absent(sha256, data)

    assert "credential" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.operation == "put_object"


def test_bucket_level_404_remains_an_adapter_failure() -> None:
    store, _ = _store(_MissingBucketS3())
    sha256 = hashlib.sha256(b"missing bucket").hexdigest()

    for operation in (store.stat, store.get_checked):
        with pytest.raises(S3ObjectStoreFailure) as raised:
            operation(sha256)
        assert raised.value.http_status == 404
        assert raised.value.error_code == "NoSuchBucket"


def test_multipart_put_checks_parts_restarts_409_and_reconciles_response_loss() -> None:
    data = b"a" * (10 * _MIB + 17)
    sha256 = hashlib.sha256(data).hexdigest()
    client = _FakeS3()
    client.complete_faults = ["409"]
    store, _ = _store(client)

    result = store.put_if_absent(sha256, data)

    assert result.status == "stored"
    assert store.get_checked(sha256) == data
    assert client.create_calls == 2
    assert client.upload_part_calls == 6
    assert client.complete_calls == 2
    assert client.abort_calls == 1

    lost_data = b"b" * (6 * _MIB + 11)
    lost_sha = hashlib.sha256(lost_data).hexdigest()
    client.complete_faults = ["response_lost"]
    assert store.put_if_absent(lost_sha, lost_data).status == "already_present"
    assert store.get_checked(lost_sha) == lost_data


def test_multipart_existing_object_converges_after_checked_read() -> None:
    data = b"c" * (6 * _MIB + 5)
    sha256 = hashlib.sha256(data).hexdigest()
    store, client = _store()

    assert store.put_if_absent(sha256, data).status == "stored"
    assert store.put_if_absent(sha256, data).status == "already_present"
    assert client.abort_calls == 1


def test_checked_read_classifies_missing_metadata_size_checksum_and_body_corruption() -> None:
    store, client = _store()
    missing = hashlib.sha256(b"missing").hexdigest()
    assert store.stat(missing) is None
    with pytest.raises(BlobNotFound):
        store.get_checked(missing)

    data = b"checked corruption"
    sha256 = hashlib.sha256(data).hexdigest()
    store.put_if_absent(sha256, data)
    location = (store.config.bucket, store.config.object_key(sha256))

    client.objects[location]["metadata"]["monoid-size"] = "999"
    with pytest.raises(BlobCorrupt, match="metadata"):
        store.stat(sha256)
    with pytest.raises(BlobCorrupt, match="metadata"):
        store.get_checked(sha256)
    assert client.last_body is not None and client.last_body.closed
    client.objects[location]["metadata"]["monoid-size"] = str(len(data))

    client.objects[location]["checksum"] = "wrong"
    with pytest.raises(BlobCorrupt, match="checksum"):
        store.stat(sha256)
    client.objects[location]["checksum"] = base64.b64encode(
        hashlib.sha256(data).digest()
    ).decode("ascii")

    client.objects[location]["data"] = b"corrupt body value"
    with pytest.raises(BlobCorrupt, match="bytes"):
        store.get_checked(sha256)


def test_put_validates_digest_exact_bytes_and_maximum_before_network() -> None:
    store, client = _store(max_object_bytes=7 * _MIB)
    data = b"value"
    sha256 = hashlib.sha256(data).hexdigest()

    with pytest.raises(ValueError, match="lowercase"):
        store.put_if_absent(sha256.upper(), data)
    with pytest.raises(TypeError, match="bytes"):
        store.put_if_absent(sha256, bytearray(data))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="do not match"):
        store.put_if_absent(sha256, b"different")
    oversized = b"x" * (7 * _MIB + 1)
    with pytest.raises(BlobTooLarge, match="maximum"):
        store.put_if_absent(hashlib.sha256(oversized).hexdigest(), oversized)
    assert client.put_calls == client.create_calls == 0


def test_threshold_at_or_below_part_size_keeps_one_part_values_on_single_put() -> None:
    store, client = _store(
        multipart_threshold_bytes=5 * _MIB,
        multipart_part_bytes=5 * _MIB,
    )
    data = b"z" * (5 * _MIB)

    assert store.put_if_absent(hashlib.sha256(data).hexdigest(), data).status == "stored"
    assert client.put_calls == 1
    assert client.create_calls == 0


def test_admin_inventory_is_bounded_parses_only_exact_content_keys_and_conditionally_deletes() -> None:
    admin, client = _admin()
    runtime = S3ContentAddressedBlobStore(admin.config, client=client)
    assert not hasattr(runtime, "inventory_page")
    assert not hasattr(admin, "put_if_absent")
    values = (b"admin-left", b"admin-right")
    digests = tuple(hashlib.sha256(value).hexdigest() for value in values)
    for sha256, value in zip(digests, values, strict=True):
        runtime.put_if_absent(sha256, value)
    client.objects[(admin.config.bucket, f"{admin.config.prefix}/sha256/not-a-digest")] = {
        "data": b"ignored",
        "metadata": {},
        "checksum": "",
        "checksum_type": "FULL_OBJECT",
    }

    first = admin.inventory_page(limit=1)
    second = admin.inventory_page(continuation_token=first.next_token, limit=2)
    entries = first.entries + second.entries

    assert {entry.sha256 for entry in entries} == set(digests)
    victim = next(entry for entry in entries if entry.sha256 == digests[0])
    assert admin.delete_if_match(victim.sha256, '"wrong"').status == "precondition_failed"
    assert admin.delete_if_match(victim.sha256, victim.delete_token).status == "deleted"
    assert admin.delete_if_match(victim.sha256, victim.delete_token).status == "already_missing"


def test_admin_incomplete_multipart_inventory_token_and_abort_are_explicit() -> None:
    admin, client = _admin()
    digests = (
        hashlib.sha256(b"upload-left").hexdigest(),
        hashlib.sha256(b"upload-right").hexdigest(),
    )
    for sha256 in digests:
        client.create_multipart_upload(
            Bucket=admin.config.bucket,
            Key=admin.config.object_key(sha256),
            Metadata={},
            ChecksumAlgorithm="SHA256",
            ChecksumType="COMPOSITE",
        )

    first = admin.incomplete_multipart_page(limit=1)
    second = admin.incomplete_multipart_page(continuation_token=first.next_token, limit=1)
    uploads = first.uploads + second.uploads

    assert {upload.sha256 for upload in uploads} == set(digests)
    assert admin.abort_incomplete_multipart(uploads[0]).status == "aborted"
    assert admin.abort_incomplete_multipart(uploads[0]).status == "already_missing"


def test_admin_version_id_mode_rejects_a_recreated_current_version() -> None:
    admin, client = _admin(admin_delete_mode="version_id")
    runtime = S3ContentAddressedBlobStore(admin.config, client=client)
    data = b"versioned conditional deletion"
    sha256 = hashlib.sha256(data).hexdigest()
    runtime.put_if_absent(sha256, data)
    entry = admin.inventory_page(limit=1).entries[0]
    location = (admin.config.bucket, admin.config.object_key(sha256))
    client.objects[location]["version_id"] = "replacement-version"

    assert admin.delete_if_match(sha256, entry.delete_token).status == "precondition_failed"
    current = admin.inventory_page(limit=1).entries[0]
    assert admin.delete_if_match(sha256, current.delete_token).status == "deleted"
