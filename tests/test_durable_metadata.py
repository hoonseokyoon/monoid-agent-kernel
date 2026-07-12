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
        "workspace_root": "/workspace",
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
    assert decode_run_metadata({**current, "schema_version": "monoid.backend-run.v²"}).status == (
        "corrupt"
    )
    assert (
        decode_run_metadata(
            {"schema_version": RUN_METADATA_SCHEMA_VERSION, "run_id": "minimal"}
        ).status
        == "loaded"
    )
    assert decode_run_metadata(
        {**current, "limits": {"max_duration_s": None}}
    ).status == "loaded"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("run_id", []),
        ("tenant_id", []),
        ("multi_turn", "false"),
        ("runtime_config", "invalid"),
        ("runtime_config_version", True),
        ("metadata_generation", 0),
        ("limits", []),
        ("created_at", "now"),
    ),
)
def test_run_metadata_decoder_rejects_wrong_known_field_types(field: str, value: object) -> None:
    payload = _metadata()
    payload[field] = value

    assert decode_run_metadata(payload).status == "corrupt"


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
    assert written["metadata_generation"] == 1
    assert read_run_metadata(run_dir)["schema_version"] == RUN_METADATA_SCHEMA_VERSION  # type: ignore[index]


def test_metadata_update_increments_generation(tmp_path) -> None:
    run_dir = tmp_path / "run_1"
    committer = DurableMetadataCommitter(None)
    initial = committer.write_initial_metadata(run_dir, "run_1", _metadata())

    updated = committer.commit_metadata_update(run_dir, "run_1", {**initial, "title": "updated"})

    assert initial["metadata_generation"] == 1
    assert updated["metadata_generation"] == 2
    assert read_run_metadata(run_dir) == updated


def test_higher_shared_generation_wins_and_materializes_locally(tmp_path) -> None:
    run_dir = tmp_path / "local" / "run_1"
    local = DurableMetadataCommitter(None).write_initial_metadata(run_dir, "run_1", _metadata())
    store = LocalFsCheckpointStore(tmp_path / "shared")
    shared = {**local, "metadata_generation": 2, "title": "shared-newer"}
    store.put_run_metadata("run_1", shared)

    checked = DurableMetadataCommitter(store).read_recovery_metadata_checked(run_dir, "run_1")

    assert checked.status == "loaded"
    assert checked.value == shared
    assert read_run_metadata(run_dir) == shared


def test_update_reconciles_newer_shared_generation_before_incrementing(tmp_path) -> None:
    run_dir = tmp_path / "local" / "run_1"
    local = DurableMetadataCommitter(None).write_initial_metadata(run_dir, "run_1", _metadata())
    store = LocalFsCheckpointStore(tmp_path / "shared")
    shared = {**local, "metadata_generation": 2, "title": "shared-newer"}
    store.put_run_metadata("run_1", shared)

    updated = DurableMetadataCommitter(store).commit_metadata_update(
        run_dir, "run_1", {**shared, "title": "updated"}
    )

    assert updated["metadata_generation"] == 3
    assert read_run_metadata(run_dir) == updated
    assert store.run_metadata("run_1") == updated


def test_versioned_local_metadata_heals_missing_shared_copy(tmp_path) -> None:
    run_dir = tmp_path / "local" / "run_1"
    local = DurableMetadataCommitter(None).write_initial_metadata(run_dir, "run_1", _metadata())
    store = LocalFsCheckpointStore(tmp_path / "shared")

    checked = DurableMetadataCommitter(store).read_recovery_metadata_checked(run_dir, "run_1")

    assert checked.status == "loaded"
    assert checked.value == local
    assert store.run_metadata("run_1") == local


def test_same_generation_divergence_is_corrupt(tmp_path) -> None:
    run_dir = tmp_path / "local" / "run_1"
    local = DurableMetadataCommitter(None).write_initial_metadata(run_dir, "run_1", _metadata())
    store = LocalFsCheckpointStore(tmp_path / "shared")
    store.put_run_metadata("run_1", {**local, "title": "diverged"})

    checked = DurableMetadataCommitter(store).read_recovery_metadata_checked(run_dir, "run_1")

    assert checked.status == "corrupt"
    assert "same generation" in checked.message


def test_corrupt_shared_copy_blocks_stale_local_recovery(tmp_path) -> None:
    run_dir = tmp_path / "local" / "run_1"
    DurableMetadataCommitter(None).write_initial_metadata(run_dir, "run_1", _metadata())
    store = LocalFsCheckpointStore(tmp_path / "shared")
    shared_path = store._dir("run_1") / "run_meta.json"
    shared_path.parent.mkdir(parents=True)
    shared_path.write_text("{", encoding="utf-8")

    checked = DurableMetadataCommitter(store).read_recovery_metadata_checked(run_dir, "run_1")

    assert checked.status == "corrupt"


def test_unavailable_shared_copy_defers_even_with_valid_local_metadata(tmp_path) -> None:
    class UnavailableStore(LocalFsCheckpointStore):
        def run_metadata_checked(self, run_id: str):
            del run_id
            raise OSError("metadata store unavailable")

    run_dir = tmp_path / "local" / "run_1"
    DurableMetadataCommitter(None).write_initial_metadata(run_dir, "run_1", _metadata())

    with pytest.raises(OSError, match="metadata store unavailable"):
        DurableMetadataCommitter(
            UnavailableStore(tmp_path / "shared")
        ).read_recovery_metadata_checked(run_dir, "run_1")


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


def test_legacy_shared_metadata_read_failure_is_retryable(tmp_path) -> None:
    class UnavailableLegacyStore(LocalFsCheckpointStore):
        run_metadata_checked = None

        def run_metadata(self, run_id: str):
            del run_id
            raise OSError("metadata store unavailable")

    committer = DurableMetadataCommitter(UnavailableLegacyStore(tmp_path / "shared"))

    with pytest.raises(OSError, match="metadata store unavailable"):
        committer.read_recovery_metadata_checked(tmp_path / "local" / "run_1", "run_1")


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

    assert read_run_metadata(run_dir) == {**initial, "metadata_generation": 1}
