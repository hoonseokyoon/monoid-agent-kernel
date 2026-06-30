"""Backend-agnostic CheckpointStore contract.

Every store implementation (LocalFs, SQLite, and any future object-store/DB) must honor
the same invariants the loop relies on: atomic last-good commit, monotonic ``latest``,
content-addressed write-once blobs, and run isolation. The suite is parametrized over a
store factory so a new backend is verified by adding one ``pytest.param`` — if it passes
here it is a drop-in for ``RunnerBackend(checkpoint_store=...)``.

Backend-specific crash simulations (poking a half-written manifest on disk, blob ``.tmp``
GC) stay in ``test_checkpoint.py`` because how you forge a torn write differs per backend.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from monoid_agent_kernel.core.checkpoint import CheckpointStore, LocalFsCheckpointStore, RunCheckpoint
from monoid_agent_kernel.reference.stores.sqlite import SqliteCheckpointStore

StoreFactory = Callable[[Path], CheckpointStore]


def _local_fs(tmp_path: Path) -> CheckpointStore:
    return LocalFsCheckpointStore(tmp_path)


def _sqlite(tmp_path: Path) -> CheckpointStore:
    return SqliteCheckpointStore(tmp_path / "checkpoints.db")


# New backends append a pytest.param here; the whole suite then runs against them.
STORE_FACTORIES = [
    pytest.param(_local_fs, id="local_fs"),
    pytest.param(_sqlite, id="sqlite"),
]


@pytest.fixture(params=STORE_FACTORIES)
def store(request: pytest.FixtureRequest, tmp_path: Path) -> CheckpointStore:
    factory: StoreFactory = request.param
    return factory(tmp_path)


def test_put_latest_seq_isolation_and_delete(store: CheckpointStore) -> None:
    assert store.latest("run_1") is None

    store.put(RunCheckpoint(run_id="run_1", seq=1, previous_turn_handle="a"))
    store.put(RunCheckpoint(run_id="run_1", seq=2, previous_turn_handle="b"))
    record = store.latest("run_1")
    assert record is not None and record.seq == 2
    assert record.checkpoint.previous_turn_handle == "b"

    # Runs are isolated; deleting one leaves the other intact.
    store.put(RunCheckpoint(run_id="run_2", seq=1))
    store.delete("run_1")
    assert store.latest("run_1") is None
    assert store.latest("run_2") is not None


def test_latest_is_monotonic(store: CheckpointStore) -> None:
    # A late writer with a lower seq (e.g. a reclaim racing a slow original worker) must
    # never regress latest() and unpublish a newer committed checkpoint.
    store.put(RunCheckpoint(run_id="run_1", seq=2, final_text="new"))
    store.put(RunCheckpoint(run_id="run_1", seq=1, final_text="stale"))
    record = store.latest("run_1")
    assert record is not None and record.seq == 2 and record.checkpoint.final_text == "new"


def test_blob_round_trips_and_is_write_once(store: CheckpointStore) -> None:
    sha = "a" * 64
    store.put(RunCheckpoint(run_id="run_1", seq=1), blobs={sha: b"created\n"})
    record = store.latest("run_1")
    assert record is not None and record.blob(sha) == b"created\n"

    # A later checkpoint can reference an already-stored blob (content-addressed dedup):
    # re-putting the same sha is a no-op write, and the bytes are still readable.
    store.put(RunCheckpoint(run_id="run_1", seq=2), blobs={sha: b"created\n"})
    assert store.latest("run_1").blob(sha) == b"created\n"  # type: ignore[union-attr]


def test_put_blob_is_content_addressed_and_readable(store: CheckpointStore) -> None:
    # The standalone blob API (for on-demand artifacts like an exported package): put_blob returns
    # the sha256 digest = the retrieval handle, get_blob round-trips, and it is content-addressed
    # (identical bytes dedup to the same digest).
    import hashlib

    data = b"proposal-tar-bytes\x00\x01"
    digest = store.put_blob("run_art", data)
    assert digest == hashlib.sha256(data).hexdigest()
    assert store.get_blob("run_art", digest) == data
    # Idempotent / content-addressed: storing the same bytes again yields the same handle.
    assert store.put_blob("run_art", data) == digest
    # An unknown digest raises KeyError (→ 404 at the API boundary).
    with pytest.raises(KeyError):
        store.get_blob("run_art", "f" * 64)
