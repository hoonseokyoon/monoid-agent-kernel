from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier

import pytest
from support.runtime import runtime_config

from monoid_agent_kernel.core.capability import (
    CapabilityLease,
    CapabilityRequest,
    CapabilityVault,
)
from monoid_agent_kernel.core.checkpoint import (
    CheckpointStore,
    LocalFsCheckpointStore,
    RunCheckpoint,
    checkpoint_payload_for_write,
    load_latest_checked,
)
from monoid_agent_kernel.core.durable_metadata import (
    RUN_METADATA_SCHEMA_VERSION,
    DurableMetadataCommitter,
)
from monoid_agent_kernel.core.outbox import Outbox, OutboxRequest
from monoid_agent_kernel.reference.stores.lease import LeaseStore, LocalFsLeaseStore
from monoid_agent_kernel.reference.stores.sqlite import SqliteCheckpointStore, SqliteLeaseStore


@dataclass
class _FaultStore:
    name: str
    store: CheckpointStore
    root: Path

    def replace_manifest(self, run_id: str, seq: int, text: str) -> None:
        if isinstance(self.store, LocalFsCheckpointStore):
            path = self.store._dir(run_id) / str(seq) / "manifest.json"
            path.write_text(text, encoding="utf-8")
            return
        assert isinstance(self.store, SqliteCheckpointStore)
        with sqlite3.connect(self.store._db_path) as conn:
            conn.execute(
                "UPDATE checkpoints SET manifest=? WHERE run_id=? AND seq=?",
                (text, run_id, seq),
            )

    def point_latest(self, run_id: str, seq: int) -> None:
        if isinstance(self.store, LocalFsCheckpointStore):
            (self.store._dir(run_id) / "LATEST").write_text(
                json.dumps({"seq": seq}), encoding="utf-8"
            )
            return
        assert isinstance(self.store, SqliteCheckpointStore)
        with sqlite3.connect(self.store._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO checkpoint_latest(run_id, seq) VALUES (?, ?)",
                (run_id, seq),
            )

    def remove_blob(self, run_id: str, digest: str) -> None:
        if isinstance(self.store, LocalFsCheckpointStore):
            (self.store._dir(run_id) / "blobs" / digest).unlink()
            return
        assert isinstance(self.store, SqliteCheckpointStore)
        with sqlite3.connect(self.store._db_path) as conn:
            conn.execute("DELETE FROM blobs WHERE sha=?", (digest,))

    def replace_metadata(self, run_id: str, text: str) -> None:
        if isinstance(self.store, LocalFsCheckpointStore):
            (self.store._dir(run_id) / "run_meta.json").write_text(text, encoding="utf-8")
            return
        assert isinstance(self.store, SqliteCheckpointStore)
        with sqlite3.connect(self.store._db_path) as conn:
            conn.execute(
                "UPDATE run_metadata SET metadata=? WHERE run_id=?",
                (text, run_id),
            )

    def stage_interrupted_checkpoint(self, checkpoint: RunCheckpoint) -> None:
        payload = json.dumps(checkpoint_payload_for_write(checkpoint))
        if isinstance(self.store, LocalFsCheckpointStore):
            path = self.store._dir(checkpoint.run_id) / str(checkpoint.seq) / "manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")
            return
        assert isinstance(self.store, SqliteCheckpointStore)
        conn = sqlite3.connect(self.store._db_path, isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO checkpoints(run_id, seq, manifest) VALUES (?, ?, ?)",
                (checkpoint.run_id, checkpoint.seq, payload),
            )
            conn.execute("ROLLBACK")
        finally:
            conn.close()


@pytest.fixture(params=("localfs", "sqlite"))
def fault_store(request: pytest.FixtureRequest, tmp_path: Path) -> _FaultStore:
    if request.param == "localfs":
        return _FaultStore("localfs", LocalFsCheckpointStore(tmp_path / "local"), tmp_path)
    return _FaultStore("sqlite", SqliteCheckpointStore(tmp_path / "checkpoints.db"), tmp_path)


@pytest.mark.parametrize(
    ("manifest", "expected"),
    (
        ("{", "corrupt"),
        (
            json.dumps(
                {
                    "schema_version": "monoid.checkpoint.v99",
                    "run_id": "run_bad",
                    "seq": 1,
                }
            ),
            "unsupported_version",
        ),
    ),
)
def test_corrupt_truncated_and_future_checkpoint_matrix(
    fault_store: _FaultStore, manifest: str, expected: str
) -> None:
    fault_store.store.put(RunCheckpoint(run_id="run_bad", seq=1))
    fault_store.replace_manifest("run_bad", 1, manifest)

    checked = load_latest_checked(fault_store.store, "run_bad")
    assert checked.status == expected
    assert checked.sequence == 1


@pytest.mark.parametrize(("field", "value"), (("run_id", "run_other"), ("seq", 999)))
def test_checkpoint_manifest_identity_must_match_lookup_and_committed_sequence(
    fault_store: _FaultStore, field: str, value: object
) -> None:
    checkpoint = RunCheckpoint(run_id="run_bound", seq=1)
    fault_store.store.put(checkpoint)
    payload = checkpoint_payload_for_write(checkpoint)
    payload[field] = value
    fault_store.replace_manifest("run_bound", 1, json.dumps(payload))

    checked = load_latest_checked(fault_store.store, "run_bound")

    assert checked.status == "corrupt"
    assert checked.sequence == 1
    assert fault_store.store.latest("run_bound") is None


def test_stale_latest_and_missing_blob_matrix(fault_store: _FaultStore) -> None:
    store = fault_store.store
    store.put(RunCheckpoint(run_id="run_pointer", seq=1, final_text="last-good"))
    fault_store.point_latest("run_pointer", 2)

    stale = load_latest_checked(store, "run_pointer")
    assert stale.status == "corrupt"
    assert stale.sequence == 2

    digest = store.put_blob("run_blob", b"durable payload")
    fault_store.remove_blob("run_blob", digest)
    with pytest.raises(KeyError):
        store.get_blob("run_blob", digest)


def test_interrupted_publication_preserves_last_good_and_stale_writer_loses(
    fault_store: _FaultStore,
) -> None:
    store = fault_store.store
    store.put(RunCheckpoint(run_id="run_publish", seq=1, final_text="committed"))
    fault_store.stage_interrupted_checkpoint(
        RunCheckpoint(run_id="run_publish", seq=2, final_text="interrupted")
    )

    assert store.latest("run_publish").checkpoint.final_text == "committed"  # type: ignore[union-attr]
    store.put(RunCheckpoint(run_id="run_publish", seq=3, final_text="newest"))
    store.put(RunCheckpoint(run_id="run_publish", seq=2, final_text="stale-writer"))
    latest = store.latest("run_publish")
    assert latest is not None
    assert latest.seq == 3
    assert latest.checkpoint.final_text == "newest"


def test_interrupted_metadata_and_run_shared_divergence_matrix(
    fault_store: _FaultStore, tmp_path: Path
) -> None:
    run_id = "run_metadata"
    config = runtime_config("fs.read", "run.finish")
    shared = {
        "schema_version": RUN_METADATA_SCHEMA_VERSION,
        "run_id": run_id,
        "tenant_id": "tenant_a",
        "user_id": "user_a",
        "workspace_root": str(tmp_path),
        "runtime_config": config.to_json(),
        "runtime_config_hash": config.config_hash,
        "source": "shared",
    }
    fault_store.store.put_run_metadata(run_id, shared)  # type: ignore[attr-defined]
    fault_store.replace_metadata(run_id, "{")
    checked_reader = fault_store.store.run_metadata_checked  # type: ignore[attr-defined]
    assert checked_reader(run_id).status == "corrupt"

    fault_store.store.put_run_metadata(run_id, shared)  # type: ignore[attr-defined]
    run_dir = tmp_path / "local-run"
    local = {**shared, "source": "local"}
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps(local), encoding="utf-8")
    recovered = DurableMetadataCommitter(fault_store.store).read_recovery_metadata_checked(
        run_dir, run_id
    )
    assert recovered.status == "loaded"
    assert recovered.value is not None and recovered.value["source"] == "local"


@pytest.mark.parametrize("kind", ("localfs", "sqlite"))
def test_restart_lease_and_cancellation_race_has_one_recovery_owner(
    kind: str, tmp_path: Path
) -> None:
    checkpoint: CheckpointStore
    lease: LeaseStore
    if kind == "localfs":
        checkpoint = LocalFsCheckpointStore(tmp_path / "runs")
        lease = LocalFsLeaseStore(tmp_path / "leases")
    else:
        db = tmp_path / "shared.db"
        checkpoint = SqliteCheckpointStore(db)
        lease = SqliteLeaseStore(db)
    checkpoint.put(RunCheckpoint(run_id="run_cancel", seq=1, cancellation_requested=True))
    lease.heartbeat("run_cancel", "crashed-worker", ttl_s=-1.0)
    barrier = Barrier(2)

    def claim(worker: str) -> tuple[str, bool]:
        barrier.wait()
        return worker, lease.try_claim("run_cancel", worker, ttl_s=60.0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ("recovery-a", "recovery-b")))

    winners = [worker for worker, won in results if won]
    assert len(winners) == 1
    assert lease.owner("run_cancel") == winners[0]
    restored = checkpoint.latest("run_cancel")
    assert restored is not None and restored.checkpoint.cancellation_requested


def test_side_effect_staging_ack_and_restart_state_are_idempotent() -> None:
    staged = Outbox()
    request = staged.append(
        OutboxRequest(
            id="outbox-stable",
            idempotency_key="effect-stable",
            destination="mail",
            expect_ack=True,
            correlation_id="correlation-stable",
        )
    )
    restarted = Outbox()
    restarted.import_(staged.export())
    recovered = restarted.get(request.id)
    assert recovered is not None
    assert recovered.status == "pending"
    assert recovered.idempotency_key == "effect-stable"

    restarted.mark(request.id, status="dispatched", attempts=1, reference="ack-1")
    after_ack_restart = Outbox()
    after_ack_restart.import_(restarted.export())
    acknowledged = after_ack_restart.get(request.id)
    assert acknowledged is not None
    assert acknowledged.status == "dispatched"
    assert acknowledged.reference == "ack-1"
    assert after_ack_restart.pending() == []


def test_capability_revocation_survives_restart_and_blocks_durable_handle() -> None:
    lease = CapabilityLease(
        capability="web.search",
        token_ref="durable-handle",
        expires_at=9e9,
        durable=True,
    )
    before_restart = CapabilityVault()
    before_restart.admit(CapabilityRequest(capability="web.search"), lease)
    before_restart.revoke(capability="web.search")

    revocations = before_restart.export_revocations()
    restarted = CapabilityVault()
    restarted.install(CapabilityLease.from_json(before_restart.export_durable()[0]))
    restarted.import_revocations(
        lease_ids=revocations["revoked_lease_ids"],
        capabilities=revocations["revoked_capabilities"],
        before=revocations["revoked_before"],
        all_revoked=revocations["revoked_all"],
    )

    assert restarted.token_for("web.search", now=0.0) is None
    assert restarted.is_capability_revoked("web.search")
