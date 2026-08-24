"""S3-compatible content-addressed object storage with a lazy boto3 boundary."""

from .config import S3ObjectStoreConfig
from .s3 import S3ContentAddressedBlobStore, S3DependencyMissing, S3ObjectStoreFailure

__all__ = [
    "S3ObjectStoreConfig",
    "S3DependencyMissing",
    "S3ObjectStoreFailure",
    "S3ContentAddressedBlobStore",
]
