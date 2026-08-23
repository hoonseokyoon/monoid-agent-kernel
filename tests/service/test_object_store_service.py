from __future__ import annotations

import base64
import hashlib
import os
import uuid

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.serial, pytest.mark.service]


if os.environ.get("MONOID_SERVICE_PROFILE") not in {"objectstore", "combined"}:
    pytest.skip("object-store service profile is not selected", allow_module_level=True)

import boto3  # noqa: E402
from botocore import config as botocore_config  # noqa: E402
from botocore import exceptions as botocore_exceptions  # noqa: E402


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
