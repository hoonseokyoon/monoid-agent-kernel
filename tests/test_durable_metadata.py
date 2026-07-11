from __future__ import annotations

import pytest

from support.runtime import runtime_config

from monoid_agent_kernel.core.durable_metadata import (
    ACCEPTED_RUN_METADATA_SCHEMA_VERSIONS,
    RUN_METADATA_SCHEMA_VERSION,
    DurableMetadataCommitter,
    decode_run_metadata,
    read_run_metadata,
    read_run_metadata_checked,
    runtime_config_from_metadata,
    validate_run_metadata,
)
from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore


def _metadata(run_id: str = "run_1", *, version: int = 1) -> dict:
    config = runtime_config("fs.read", "run.finish", version=version)
    return {
        "schema_version": RUN_METADATA_SCHEMA_VERSION,
        "run_id": run_id,
        "tenant_id": "tenant_a",
        "user_id": "user_a",
        "runtime_config": config.to_json(),
        "runtime_config_version": config.config_version,
        "runtime_config_hash": config.config_hash,
    }


def test_run_metadata_validation_accepts_current_and_legacy_schema() -> None:
    current = _metadata()
    legacy = {**current, "schema_version": ACCEPTED_RUN_METADATA_SCHEMA_VERSIONS[1]}

    assert validate_run_metadata(current) == current
    assert validate_run_metadata(legacy) == legacy
    assert validate_run_metadata({**current, "schema_version": "future.backend-run.v99"}) is None
    assert decode_run_metadata(current).status == "loaded"
    assert decode_run_metadata(legacy).status == "loaded"
    assert decode_run_metadata({**current, "schema_version": "monoid.backend-run.v99"}).status == (
        "unsupported_version"
    )


def test_runtime_config_hash_mismatch_is_rejected() -> None:
    meta = {**_metadata(), "runtime_config_hash": "not-the-config-hash"}

    with pytest.raises(ValueError, match="runtime config hash mismatch"):
        runtime_config_from_metadata(meta)


def test_shared_metadata_materializes_local_recovery_descriptor(tmp_path) -> None:
    store = LocalFsCheckpointStore(tmp_path / "shared")
    committer = DurableMetadataCommitter(store)
    run_dir = tmp_path / "local" / "run_1"
    meta = _metadata()
    store.put_run_metadata("run_1", meta)

    recovered = committer.read_recovery_metadata(run_dir, "run_1")

    assert recovered == meta
    assert read_run_metadata(run_dir) == meta
    assert committer.read_recovery_metadata_checked(run_dir, "run_1").status == "loaded"


def test_metadata_writer_canonicalizes_accepted_legacy_namespace(tmp_path) -> None:
    run_dir = tmp_path / "run_1"
    legacy = {**_metadata(), "schema_version": ACCEPTED_RUN_METADATA_SCHEMA_VERSIONS[1]}

    written = DurableMetadataCommitter(None).write_initial_metadata(run_dir, "run_1", legacy)

    assert written["schema_version"] == RUN_METADATA_SCHEMA_VERSION
    assert read_run_metadata(run_dir)["schema_version"] == RUN_METADATA_SCHEMA_VERSION  # type: ignore[index]


def test_unsupported_shared_schema_is_ignored_without_materialization(tmp_path) -> None:
    store = LocalFsCheckpointStore(tmp_path / "shared")
    committer = DurableMetadataCommitter(store)
    run_dir = tmp_path / "local" / "run_1"
    store.put_run_metadata("run_1", {**_metadata(), "schema_version": "future.backend-run.v99"})

    checked = committer.read_recovery_metadata_checked(run_dir, "run_1")
    assert checked.status == "unsupported_version"
    assert committer.read_recovery_metadata(run_dir, "run_1") is None
    assert not (run_dir / "run.json").exists()


def test_corrupt_local_metadata_is_not_treated_as_missing_or_replaced_from_shared(tmp_path) -> None:
    store = LocalFsCheckpointStore(tmp_path / "shared")
    committer = DurableMetadataCommitter(store)
    run_dir = tmp_path / "local" / "run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text("{", encoding="utf-8")
    store.put_run_metadata("run_1", _metadata())

    checked = committer.read_recovery_metadata_checked(run_dir, "run_1")

    assert checked.status == "corrupt"
    assert read_run_metadata_checked(run_dir).status == "corrupt"
    assert (run_dir / "run.json").read_text(encoding="utf-8") == "{"


def test_runtime_config_metadata_store_failure_keeps_local_descriptor_unchanged(tmp_path) -> None:
    class FailingStore(LocalFsCheckpointStore):
        def put_run_metadata(self, run_id: str, metadata: dict) -> None:
            del run_id, metadata
            raise OSError("shared metadata unavailable")

    run_dir = tmp_path / "runs" / "run_1"
    initial = _metadata(version=1)
    DurableMetadataCommitter(None).write_initial_metadata(run_dir, "run_1", initial)
    replacement = runtime_config("fs.read", "run.finish", version=2)

    with pytest.raises(OSError, match="shared metadata unavailable"):
        DurableMetadataCommitter(FailingStore(tmp_path / "shared")).commit_runtime_config_update(
            run_dir,
            "run_1",
            replacement,
            issuer="test",
            reason="replace guidance",
            committed_at=123.0,
        )

    assert read_run_metadata(run_dir) == initial
