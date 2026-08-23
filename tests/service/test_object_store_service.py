from __future__ import annotations

import base64
import hashlib
import os
import uuid

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.serial, pytest.mark.service]


@pytest.mark.skipif(
    os.environ.get("MONOID_SERVICE_PROFILE") not in {"objectstore", "combined"},
    reason="object-store service profile is not selected",
)
def test_pinned_minio_supports_checked_conditional_put() -> None:
    boto3 = pytest.importorskip("boto3")
    botocore_config = pytest.importorskip("botocore.config")
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
    body = b"monoid-v0.23-object-store-service-smoke"
    checksum = base64.b64encode(hashlib.sha256(body).digest()).decode("ascii")

    client.create_bucket(Bucket=bucket)
    try:
        result = client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            IfNoneMatch="*",
            ChecksumSHA256=checksum,
        )
        assert result["ResponseMetadata"]["HTTPStatusCode"] == 200
        loaded = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        assert loaded == body
        assert hashlib.sha256(loaded).digest() == hashlib.sha256(body).digest()
    finally:
        client.delete_object(Bucket=bucket, Key=key)
        client.delete_bucket(Bucket=bucket)
