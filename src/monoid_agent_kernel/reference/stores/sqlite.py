"""SQLite-backed durable stores (stdlib ``sqlite3`` — no external dependencies).

A single DB file is the durable "save store": a transaction gives the CheckpointStore
seam's invariants for free — ``put`` commits atomically (a crash mid-put rolls back, so
``latest`` never sees a torn checkpoint), the latest pointer advances monotonically via a
conditional UPSERT, and content-addressed blobs are write-once. The same db also hosts the
``SqliteLeaseStore`` (a transactional CAS lease), so one shared db is the "shared board"
that lets a worker on another process/host reclaim a crashed peer's run.

SQLite itself is single-host; a real cross-host deployment swaps this for a networked DB or
object store behind the same seams (documented follow-up). What it proves here, with zero
dependencies, is that the seams are sufficient and the transactional commit/CAS pattern is
correct.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from monoid_agent_kernel.core._util import sha256_bytes
from monoid_agent_kernel.core.checkpoint import (
    CHECKPOINT_CODEC,
    CheckpointRecord,
    RunCheckpoint,
    checkpoint_payload_for_write,
    decode_checkpoint,
)
from monoid_agent_kernel.core.durable_codec import DurableLoadResult
from monoid_agent_kernel.core.durable_metadata import RUN_METADATA_CODEC, decode_run_metadata

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    run_id   TEXT    NOT NULL,
    seq      INTEGER NOT NULL,
    manifest TEXT    NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS checkpoint_latest (
    run_id TEXT    PRIMARY KEY,
    seq    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS run_metadata (
    run_id   TEXT PRIMARY KEY,
    metadata TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS blobs (
    sha  TEXT PRIMARY KEY,
    data BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS leases (
    run_id       TEXT PRIMARY KEY,
    worker_id    TEXT NOT NULL,
    heartbeat_at REAL NOT NULL,
    ttl_s        REAL NOT NULL
);
"""


def _connect(db_path: str) -> sqlite3.Connection:
    # isolation_level=None -> autocommit; we drive BEGIN IMMEDIATE / COMMIT explicitly so a
    # write takes the db write-lock up front (cross-process serialization). busy_timeout +
    # WAL let readers and a writer coexist without spurious SQLITE_BUSY.
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _ensure_schema(db_path: str) -> None:
    conn = _connect(db_path)
    try:
        conn.executescript(_SCHEMA)
    finally:
        conn.close()


class SqliteCheckpointStore:
    """A durable ``CheckpointStore`` backed by one SQLite db. Honors the same contract as
    ``LocalFsCheckpointStore`` (atomic last-good, monotonic ``latest``, write-once blobs)."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()  # serialize this instance's writers; cross-process is BEGIN IMMEDIATE
        _ensure_schema(self._db_path)

    def put(self, checkpoint: RunCheckpoint, blobs: Mapping[str, bytes] = {}) -> None:
        manifest = json.dumps(checkpoint_payload_for_write(checkpoint), ensure_ascii=False)
        with self._lock:
            conn = _connect(self._db_path)
            try:
                conn.execute("BEGIN IMMEDIATE")
                for sha, data in blobs.items():
                    # Content-addressed, write-once: an already-stored blob is left as-is.
                    conn.execute("INSERT OR IGNORE INTO blobs(sha, data) VALUES (?, ?)", (sha, data))
                conn.execute(
                    "INSERT OR REPLACE INTO checkpoints(run_id, seq, manifest) VALUES (?, ?, ?)",
                    (checkpoint.run_id, checkpoint.seq, manifest),
                )
                # Monotonic latest flip: advance only, never regress to a lower seq.
                conn.execute(
                    "INSERT INTO checkpoint_latest(run_id, seq) VALUES (?, ?) "
                    "ON CONFLICT(run_id) DO UPDATE SET seq=excluded.seq "
                    "WHERE excluded.seq > checkpoint_latest.seq",
                    (checkpoint.run_id, checkpoint.seq),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def latest_checked(self, run_id: str) -> DurableLoadResult[CheckpointRecord]:
        conn = _connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT seq FROM checkpoint_latest WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                return CHECKPOINT_CODEC.missing().map(
                    lambda checkpoint: CheckpointRecord(seq=checkpoint.seq, checkpoint=checkpoint)
                )
            seq = int(row[0])
            mrow = conn.execute(
                "SELECT manifest FROM checkpoints WHERE run_id=? AND seq=?", (run_id, seq)
            ).fetchone()
        finally:
            conn.close()
        if mrow is None:
            return CHECKPOINT_CODEC.corrupt(
                "checkpoint latest pointer references a missing manifest", sequence=seq
            ).map(lambda checkpoint: CheckpointRecord(seq=seq, checkpoint=checkpoint))
        try:
            payload = json.loads(mrow[0])
        except ValueError:
            return CHECKPOINT_CODEC.corrupt(
                "checkpoint manifest is not valid JSON", sequence=seq
            ).map(lambda checkpoint: CheckpointRecord(seq=seq, checkpoint=checkpoint))
        decoded = replace(decode_checkpoint(payload), sequence=seq)
        return decoded.map(
            lambda checkpoint: CheckpointRecord(
                seq=seq, checkpoint=checkpoint, _blob_reader=self._read_blob
            )
        )

    def latest(self, run_id: str) -> CheckpointRecord | None:
        return self.latest_checked(run_id).value

    def put_blob(self, run_id: str, data: bytes) -> str:
        """Store a standalone content-addressed blob (write-once, shared ``blobs`` table — the same
        namespace ``put`` fills for checkpoints) and return its sha256 digest."""
        del run_id  # blobs are global/content-addressed in this store; run scope is the token's job
        sha = sha256_bytes(data)
        with self._lock:
            conn = _connect(self._db_path)
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("INSERT OR IGNORE INTO blobs(sha, data) VALUES (?, ?)", (sha, data))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()
        return sha

    def get_blob(self, run_id: str, sha256: str) -> bytes:
        del run_id  # content-addressed lookup; the token already authorized the run
        return self._read_blob(sha256)

    def put_run_metadata(self, run_id: str, metadata: Mapping[str, object]) -> None:
        payload = json.dumps(dict(metadata), ensure_ascii=False)
        with self._lock:
            conn = _connect(self._db_path)
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT OR REPLACE INTO run_metadata(run_id, metadata) VALUES (?, ?)",
                    (run_id, payload),
                )
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def run_metadata(self, run_id: str) -> dict[str, object] | None:
        conn = _connect(self._db_path)
        try:
            row = conn.execute("SELECT metadata FROM run_metadata WHERE run_id=?", (run_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        payload = json.loads(row[0])
        return dict(payload) if isinstance(payload, dict) else None

    def run_metadata_checked(self, run_id: str) -> DurableLoadResult[dict[str, object]]:
        conn = _connect(self._db_path)
        try:
            row = conn.execute("SELECT metadata FROM run_metadata WHERE run_id=?", (run_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            return RUN_METADATA_CODEC.missing()
        try:
            payload = json.loads(row[0])
        except ValueError:
            return RUN_METADATA_CODEC.corrupt("backend-run metadata is not valid JSON")
        return decode_run_metadata(payload)

    def _read_blob(self, sha256: str) -> bytes:
        conn = _connect(self._db_path)
        try:
            row = conn.execute("SELECT data FROM blobs WHERE sha=?", (sha256,)).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(sha256)
        return bytes(row[0])

    def delete(self, run_id: str) -> None:
        with self._lock:
            conn = _connect(self._db_path)
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("DELETE FROM checkpoints WHERE run_id=?", (run_id,))
                conn.execute("DELETE FROM checkpoint_latest WHERE run_id=?", (run_id,))
                conn.execute("DELETE FROM run_metadata WHERE run_id=?", (run_id,))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()
        # Content-addressed blobs may be shared across runs, so they are not deleted here
        # (a production store would refcount/GC them); leftover blobs are harmless.


class SqliteLeaseStore:
    """A durable ``LeaseStore`` backed by one SQLite db (typically the same db as the
    checkpoint store). ``try_claim`` is a transactional CAS — ``BEGIN IMMEDIATE`` takes the
    write lock, so concurrent claimers (even in other processes/hosts) serialize and exactly
    one sees the lease stale. This is the "shared board" that crosses the instance boundary
    a per-host ``lease.json`` cannot."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        _ensure_schema(self._db_path)

    def candidate_run_ids(self) -> list[str]:
        conn = _connect(self._db_path)
        try:
            rows = conn.execute("SELECT run_id FROM leases").fetchall()
        finally:
            conn.close()
        return [str(row[0]) for row in rows]

    def heartbeat(self, run_id: str, worker_id: str, ttl_s: float) -> None:
        self._write(run_id, worker_id, ttl_s)

    def is_stale(self, run_id: str) -> bool:
        conn = _connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT heartbeat_at, ttl_s FROM leases WHERE run_id=?", (run_id,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return True
        return (time.time() - float(row[0])) > float(row[1])

    def try_claim(self, run_id: str, worker_id: str, ttl_s: float) -> bool:
        with self._lock:
            conn = _connect(self._db_path)
            try:
                conn.execute("BEGIN IMMEDIATE")  # take the write lock -> serialize claimers
                row = conn.execute(
                    "SELECT heartbeat_at, ttl_s FROM leases WHERE run_id=?", (run_id,)
                ).fetchone()
                stale = row is None or (time.time() - float(row[0])) > float(row[1])
                if not stale:
                    conn.execute("COMMIT")
                    return False
                conn.execute(
                    "INSERT OR REPLACE INTO leases(run_id, worker_id, heartbeat_at, ttl_s) "
                    "VALUES (?, ?, ?, ?)",
                    (run_id, worker_id, time.time(), ttl_s),
                )
                conn.execute("COMMIT")
                return True
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def owner(self, run_id: str) -> str | None:
        conn = _connect(self._db_path)
        try:
            row = conn.execute("SELECT worker_id FROM leases WHERE run_id=?", (run_id,)).fetchone()
        finally:
            conn.close()
        return str(row[0]) if row is not None else None

    def release(self, run_id: str) -> None:
        with self._lock:
            conn = _connect(self._db_path)
            try:
                conn.execute("DELETE FROM leases WHERE run_id=?", (run_id,))
            finally:
                conn.close()

    def _write(self, run_id: str, worker_id: str, ttl_s: float) -> None:
        with self._lock:
            conn = _connect(self._db_path)
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO leases(run_id, worker_id, heartbeat_at, ttl_s) "
                    "VALUES (?, ?, ?, ?)",
                    (run_id, worker_id, time.time(), ttl_s),
                )
            finally:
                conn.close()
