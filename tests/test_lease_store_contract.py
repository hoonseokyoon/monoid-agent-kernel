"""Backend-agnostic LeaseStore contract.

Every lease store (LocalFs, SQLite, and any future shared DB / object store) must give the
same guarantee the watchdog relies on: ``try_claim`` is an atomic compare-and-set, so two
workers racing the same stale run produce exactly one winner. Parametrized over a store
factory — a new backend is verified by adding one ``pytest.param``.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from monoid_agent_kernel.reference.stores.lease import LeaseStore, LocalFsLeaseStore
from monoid_agent_kernel.reference.stores.sqlite import SqliteLeaseStore

LeaseFactory = Callable[[Path], LeaseStore]


def _local_fs(tmp_path: Path) -> LeaseStore:
    return LocalFsLeaseStore(tmp_path)


def _sqlite(tmp_path: Path) -> LeaseStore:
    return SqliteLeaseStore(tmp_path / "leases.db")


LEASE_FACTORIES = [
    pytest.param(_local_fs, id="local_fs"),
    pytest.param(_sqlite, id="sqlite"),
]


@pytest.fixture(params=LEASE_FACTORIES)
def lease_store(request: pytest.FixtureRequest, tmp_path: Path) -> LeaseStore:
    factory: LeaseFactory = request.param
    return factory(tmp_path)


def test_heartbeat_owner_and_release(lease_store: LeaseStore) -> None:
    assert lease_store.owner("r") is None
    assert lease_store.is_stale("r") is True  # absent -> reclaimable

    lease_store.heartbeat("r", "w1", ttl_s=30.0)
    assert lease_store.owner("r") == "w1"
    assert lease_store.is_stale("r") is False
    assert "r" in lease_store.candidate_run_ids()

    lease_store.release("r")
    assert lease_store.owner("r") is None
    assert lease_store.is_stale("r") is True


def test_try_claim_denied_when_live_and_granted_when_stale(lease_store: LeaseStore) -> None:
    lease_store.heartbeat("r", "live", ttl_s=30.0)
    assert lease_store.try_claim("r", "thief", ttl_s=30.0) is False  # a live peer holds it
    assert lease_store.owner("r") == "live"

    lease_store.heartbeat("r", "live", ttl_s=0.0)  # expire it
    time.sleep(0.02)
    assert lease_store.is_stale("r") is True
    assert lease_store.try_claim("r", "thief", ttl_s=30.0) is True  # now claimable
    assert lease_store.owner("r") == "thief"
    assert lease_store.is_stale("r") is False  # the claim refreshed the heartbeat


def test_concurrent_claim_has_single_winner(lease_store: LeaseStore) -> None:
    # An absent lease is stale, so every worker races to claim it; the CAS must let exactly
    # one win (the rest see a now-fresh lease).
    results: list[bool] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(4)

    def claim(worker_id: str) -> None:
        barrier.wait()
        won = lease_store.try_claim("r", worker_id, ttl_s=30.0)
        with results_lock:
            results.append(won)

    threads = [threading.Thread(target=claim, args=(f"w{i}",)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count(True) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("heartbeat_at", None),
        ("heartbeat_at", True),
        ("heartbeat_at", "1"),
        ("heartbeat_at", 1e309),
        ("lease_ttl_s", None),
        ("lease_ttl_s", True),
        ("lease_ttl_s", "30"),
        ("lease_ttl_s", 0),
        ("lease_ttl_s", 1e309),
        ("worker_id", 7),
        ("worker_id", ""),
        ("run_id", "other"),
    ],
)
def test_local_fs_malformed_lease_is_reclaimable(tmp_path: Path, field: str, value: object) -> None:
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    payload = {
        "run_id": "r",
        "worker_id": "worker",
        "heartbeat_at": time.time(),
        "lease_ttl_s": 30,
    }
    payload[field] = value
    (run_dir / "lease.json").write_text(json.dumps(payload), encoding="utf-8")

    store = LocalFsLeaseStore(tmp_path)
    assert store.owner("r") is None
    assert store.is_stale("r") is True


def test_local_fs_overflowing_json_number_cannot_abort_recovery(tmp_path: Path) -> None:
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    (run_dir / "lease.json").write_text(
        '{"run_id":"r","worker_id":"worker","heartbeat_at":1e309,"lease_ttl_s":30}',
        encoding="utf-8",
    )

    store = LocalFsLeaseStore(tmp_path)
    assert store.owner("r") is None
    assert store.is_stale("r") is True

    huge = str(10**400)
    (run_dir / "lease.json").write_text(
        f'{{"run_id":"r","worker_id":"worker","heartbeat_at":{huge},"lease_ttl_s":30}}',
        encoding="utf-8",
    )
    assert store.owner("r") is None
    assert store.is_stale("r") is True
