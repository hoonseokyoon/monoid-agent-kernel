from __future__ import annotations

import base64
import hashlib
import os
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.serial, pytest.mark.service]


if os.environ.get("MONOID_SERVICE_PROFILE") not in {"objectstore", "combined"}:
    pytest.skip("object-store service profile is not selected", allow_module_level=True)

import boto3  # noqa: E402
from botocore import config as botocore_config  # noqa: E402
from botocore import exceptions as botocore_exceptions  # noqa: E402

from monoid_agent_kernel.adapters.object_store import (  # noqa: E402
    S3ContentAddressedBlobStore,
    S3ObjectStoreConfig,
)
from monoid_agent_kernel.conformance import (  # noqa: E402
    run_content_addressed_blob_store_contract,
)
from monoid_agent_kernel.hosting import BlobCorrupt  # noqa: E402


_MIB = 1024 * 1024


class _ClientProxy:
    def __init__(self, delegate: object) -> None:
        self.delegate = delegate

    def __getattr__(self, name: str) -> object:
        return getattr(self.delegate, name)


class _PutBarrierClient(_ClientProxy):
    def __init__(self, delegate: object, barrier: threading.Barrier) -> None:
        super().__init__(delegate)
        self.barrier = barrier

    def put_object(self, **kwargs: object) -> object:
        self.barrier.wait(timeout=15)
        return self.delegate.put_object(**kwargs)  # type: ignore[attr-defined]


class _CompleteBarrierClient(_ClientProxy):
    def __init__(self, delegate: object, barrier: threading.Barrier) -> None:
        super().__init__(delegate)
        self.barrier = barrier

    def complete_multipart_upload(self, **kwargs: object) -> object:
        self.barrier.wait(timeout=30)
        return self.delegate.complete_multipart_upload(**kwargs)  # type: ignore[attr-defined]


class _ConflictOnceClient(_ClientProxy):
    def __init__(self, delegate: object, *, operation: str) -> None:
        super().__init__(delegate)
        self.operation = operation
        self.injected = False
        self.calls = 0

    @staticmethod
    def _conflict(operation: str) -> botocore_exceptions.ClientError:
        return botocore_exceptions.ClientError(
            {
                "ResponseMetadata": {"HTTPStatusCode": 409},
                "Error": {"Code": "ConditionalRequestConflict", "Message": "injected"},
            },
            operation,
        )

    def put_object(self, **kwargs: object) -> object:
        self.calls += 1
        if self.operation == "put" and not self.injected:
            self.injected = True
            raise self._conflict("PutObject")
        return self.delegate.put_object(**kwargs)  # type: ignore[attr-defined]

    def complete_multipart_upload(self, **kwargs: object) -> object:
        self.calls += 1
        if self.operation == "complete" and not self.injected:
            self.injected = True
            raise self._conflict("CompleteMultipartUpload")
        return self.delegate.complete_multipart_upload(**kwargs)  # type: ignore[attr-defined]


class _ResponseLossOnceClient(_ClientProxy):
    def __init__(self, delegate: object, *, operation: str) -> None:
        super().__init__(delegate)
        self.operation = operation
        self.injected = False

    def put_object(self, **kwargs: object) -> object:
        response = self.delegate.put_object(**kwargs)  # type: ignore[attr-defined]
        if self.operation == "put" and not self.injected:
            self.injected = True
            raise ConnectionError("injected response loss")
        return response

    def complete_multipart_upload(self, **kwargs: object) -> object:
        response = self.delegate.complete_multipart_upload(**kwargs)  # type: ignore[attr-defined]
        if self.operation == "complete" and not self.injected:
            self.injected = True
            raise ConnectionError("injected response loss")
        return response


class _ListDeniedClient(_ClientProxy):
    def list_objects_v2(self, **kwargs: object) -> object:
        del kwargs
        raise AssertionError("runtime object-store path attempted to list the bucket")

    def list_multipart_uploads(self, **kwargs: object) -> object:
        del kwargs
        raise AssertionError("runtime object-store path attempted to list multipart uploads")


def _client() -> object:
    endpoint = os.environ.get("MONOID_MINIO_ENDPOINT")
    if not endpoint:
        pytest.fail("MONOID_MINIO_ENDPOINT is required for the selected service profile")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("MONOID_MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("MONOID_MINIO_SECRET_KEY"),
        region_name="us-east-1",
        config=botocore_config.Config(
            signature_version="s3v4",
            connect_timeout=10,
            read_timeout=30,
            retries={"max_attempts": 5, "mode": "standard"},
            s3={"addressing_style": "path"},
        ),
    )


@pytest.fixture
def object_store_bucket() -> Iterator[tuple[object, str]]:
    client = _client()
    bucket = f"monoid-v023-{uuid.uuid4().hex}"
    client.create_bucket(Bucket=bucket)  # type: ignore[attr-defined]
    try:
        yield client, bucket
    finally:
        uploads = client.list_multipart_uploads(Bucket=bucket).get("Uploads", [])  # type: ignore[attr-defined]
        for upload in uploads:
            client.abort_multipart_upload(  # type: ignore[attr-defined]
                Bucket=bucket,
                Key=upload["Key"],
                UploadId=upload["UploadId"],
            )
        objects = client.list_objects_v2(Bucket=bucket).get("Contents", [])  # type: ignore[attr-defined]
        for value in objects:
            client.delete_object(Bucket=bucket, Key=value["Key"])  # type: ignore[attr-defined]
        client.delete_bucket(Bucket=bucket)  # type: ignore[attr-defined]


def _store(client: object, bucket: str, *, prefix: str) -> S3ContentAddressedBlobStore:
    return S3ContentAddressedBlobStore(
        S3ObjectStoreConfig(
            bucket=bucket,
            prefix=prefix,
            endpoint_url=os.environ["MONOID_MINIO_ENDPOINT"],
            addressing_style="path",
            multipart_threshold_bytes=6 * _MIB,
            multipart_part_bytes=5 * _MIB,
            max_object_bytes=32 * _MIB,
        ),
        client=client,
    )


def test_pinned_minio_supports_checked_conditional_put() -> None:
    endpoint = os.environ.get("MONOID_MINIO_ENDPOINT")
    if not endpoint:
        pytest.fail("MONOID_MINIO_ENDPOINT is required for the selected service profile")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("MONOID_MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("MONOID_MINIO_SECRET_KEY"),
        region_name="us-east-1",
        config=botocore_config.Config(
            signature_version="s3v4",
            connect_timeout=10,
            read_timeout=10,
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )
    bucket = f"monoid-v023-{uuid.uuid4().hex}"
    key = "sha256/service-smoke"
    checksum_mismatch_key = "sha256/checksum-mismatch"
    body = b"monoid-v0.23-object-store-service-smoke"
    checksum = base64.b64encode(hashlib.sha256(body).digest()).decode("ascii")

    client.create_bucket(Bucket=bucket)
    try:
        wrong_checksum = base64.b64encode(hashlib.sha256(b"wrong-digest").digest()).decode(
            "ascii"
        )
        with pytest.raises(botocore_exceptions.ClientError) as checksum_mismatch:
            client.put_object(
                Bucket=bucket,
                Key=checksum_mismatch_key,
                Body=body,
                IfNoneMatch="*",
                ChecksumSHA256=wrong_checksum,
            )
        checksum_error = checksum_mismatch.value.response
        assert checksum_error["ResponseMetadata"]["HTTPStatusCode"] == 400
        assert checksum_error["Error"]["Code"] == "XAmzContentChecksumMismatch"

        result = client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            IfNoneMatch="*",
            ChecksumSHA256=checksum,
        )
        assert result["ResponseMetadata"]["HTTPStatusCode"] == 200
        replacement = b"different-bytes-must-not-replace-the-first-writer"
        replacement_checksum = base64.b64encode(hashlib.sha256(replacement).digest()).decode(
            "ascii"
        )
        with pytest.raises(botocore_exceptions.ClientError) as raised:
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=replacement,
                IfNoneMatch="*",
                ChecksumSHA256=replacement_checksum,
            )
        error = raised.value.response
        assert error["ResponseMetadata"]["HTTPStatusCode"] == 412
        assert error["Error"]["Code"] == "PreconditionFailed"
        loaded = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        assert loaded == body
        assert hashlib.sha256(loaded).digest() == hashlib.sha256(body).digest()
    finally:
        client.delete_object(Bucket=bucket, Key=key)
        client.delete_object(Bucket=bucket, Key=checksum_mismatch_key)
        client.delete_bucket(Bucket=bucket)


def test_s3_adapter_single_multipart_and_neutral_contract_on_pinned_minio(
    object_store_bucket: tuple[object, str],
) -> None:
    client, bucket = object_store_bucket
    store = _store(client, bucket, prefix="roundtrip")
    single = b"actual MinIO single object bytes"
    single_sha = hashlib.sha256(single).hexdigest()
    multipart = b"m" * (10 * _MIB + 31)
    multipart_sha = hashlib.sha256(multipart).hexdigest()

    assert store.put_if_absent(single_sha, single).status == "stored"
    assert store.put_if_absent(single_sha, single).status == "already_present"
    assert store.get_checked(single_sha) == single
    assert store.put_if_absent(multipart_sha, multipart).status == "stored"
    assert store.put_if_absent(multipart_sha, multipart).status == "already_present"
    assert store.get_checked(multipart_sha) == multipart

    contract_store = _store(client, bucket, prefix="contract")
    outcomes = run_content_addressed_blob_store_contract(lambda: contract_store)
    assert all(outcome.passed for outcome in outcomes), [outcome.to_json() for outcome in outcomes]


def test_s3_adapter_builds_a_lazy_client_from_the_standard_credential_chain(
    object_store_bucket: tuple[object, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, bucket = object_store_bucket
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", os.environ["MONOID_MINIO_ACCESS_KEY"])
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", os.environ["MONOID_MINIO_SECRET_KEY"])
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    store = S3ContentAddressedBlobStore(
        S3ObjectStoreConfig(
            bucket=bucket,
            prefix="owned-client",
            endpoint_url=os.environ["MONOID_MINIO_ENDPOINT"],
            addressing_style="path",
        )
    )
    data = b"standard boto3 credential provider chain"
    sha256 = hashlib.sha256(data).hexdigest()

    assert store.put_if_absent(sha256, data).status == "stored"
    assert store.get_checked(sha256) == data


@pytest.mark.parametrize("multipart", [False, True])
def test_s3_adapter_conditional_race_converges_on_pinned_minio(
    object_store_bucket: tuple[object, str],
    *,
    multipart: bool,
) -> None:
    first_client, bucket = object_store_bucket
    second_client = _client()
    barrier = threading.Barrier(2)
    if multipart:
        first_client = _CompleteBarrierClient(first_client, barrier)
        second_client = _CompleteBarrierClient(second_client, barrier)
        data = b"r" * (10 * _MIB + 23)
        prefix = "multipart-race"
    else:
        first_client = _PutBarrierClient(first_client, barrier)
        second_client = _PutBarrierClient(second_client, barrier)
        data = b"actual single conditional race"
        prefix = "single-race"
    sha256 = hashlib.sha256(data).hexdigest()
    stores = (
        _store(first_client, bucket, prefix=prefix),
        _store(second_client, bucket, prefix=prefix),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda store: store.put_if_absent(sha256, data), stores))

    assert sorted(result.status for result in results) == ["already_present", "stored"]
    assert all(store.get_checked(sha256) == data for store in stores)


def test_s3_adapter_retries_real_requests_after_injected_409(
    object_store_bucket: tuple[object, str],
) -> None:
    client, bucket = object_store_bucket
    single_client = _ConflictOnceClient(client, operation="put")
    single_store = _store(single_client, bucket, prefix="single-409")
    single = b"single 409 request"
    single_sha = hashlib.sha256(single).hexdigest()

    assert single_store.put_if_absent(single_sha, single).status == "stored"
    assert single_client.calls == 2

    multipart_client = _ConflictOnceClient(client, operation="complete")
    multipart_store = _store(multipart_client, bucket, prefix="multipart-409")
    multipart = b"k" * (10 * _MIB + 29)
    multipart_sha = hashlib.sha256(multipart).hexdigest()

    assert multipart_store.put_if_absent(multipart_sha, multipart).status == "stored"
    assert multipart_client.calls == 2
    assert client.list_multipart_uploads(Bucket=bucket).get("Uploads", []) == []  # type: ignore[attr-defined]


def test_s3_adapter_reconciles_actual_commit_after_response_loss(
    object_store_bucket: tuple[object, str],
) -> None:
    client, bucket = object_store_bucket
    single_store = _store(
        _ResponseLossOnceClient(client, operation="put"),
        bucket,
        prefix="single-response-loss",
    )
    single = b"single committed response loss"
    single_sha = hashlib.sha256(single).hexdigest()
    assert single_store.put_if_absent(single_sha, single).status == "already_present"
    assert single_store.get_checked(single_sha) == single

    multipart_store = _store(
        _ResponseLossOnceClient(client, operation="complete"),
        bucket,
        prefix="multipart-response-loss",
    )
    multipart = b"l" * (6 * _MIB + 37)
    multipart_sha = hashlib.sha256(multipart).hexdigest()
    assert multipart_store.put_if_absent(multipart_sha, multipart).status == "already_present"
    assert multipart_store.get_checked(multipart_sha) == multipart


def test_s3_adapter_runtime_path_is_list_free_and_detects_forged_content(
    object_store_bucket: tuple[object, str],
) -> None:
    client, bucket = object_store_bucket
    store = _store(_ListDeniedClient(client), bucket, prefix="list-free")
    data = b"list-free runtime bytes"
    sha256 = hashlib.sha256(data).hexdigest()

    assert store.put_if_absent(sha256, data).status == "stored"
    assert store.stat(sha256) is not None
    assert store.get_checked(sha256) == data

    forged = b"forged bytes"
    key = store.config.object_key(hashlib.sha256(b"expected bytes").hexdigest())
    expected_sha = hashlib.sha256(b"expected bytes").hexdigest()
    forged_checksum = base64.b64encode(hashlib.sha256(forged).digest()).decode("ascii")
    client.put_object(  # type: ignore[attr-defined]
        Bucket=bucket,
        Key=key,
        Body=forged,
        ChecksumSHA256=forged_checksum,
        Metadata={
            "monoid-schema": "content-addressed-blob-v1",
            "monoid-sha256": expected_sha,
            "monoid-size": str(len(forged)),
            "monoid-upload": "single",
            "monoid-part-count": "1",
        },
    )
    with pytest.raises(BlobCorrupt, match="checksum"):
        store.stat(expected_sha)
