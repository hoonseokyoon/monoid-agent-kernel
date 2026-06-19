"""The ``LeaseStore`` seam: who owns a run right now.

The watchdog answers "is this run's worker still alive, and may I take it over?" The
*policy* (heartbeat, stale detection, reclaim) lives in ``RunnerBackend``; the *storage and
the claim's atomicity* live behind this seam. The default ``LocalFsLeaseStore`` keeps a
``lease.json`` per run dir with an ``O_EXCL`` file-lock CAS — correct for one host / one
shared filesystem. A ``SqliteLeaseStore`` (transactional CAS) puts the lease in a shared db
instead, so a worker on another process/host can see it and reclaim across that boundary —
the "shared board" that local files cannot be.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from native_agent_runner.core._util import file_lock, write_json_atomic


class LeaseStore(Protocol):
    """How a run's ownership lease is stored and claimed. ``try_claim`` MUST be an atomic
    compare-and-set so two watchdogs racing the same stale run produce exactly one winner."""

    def candidate_run_ids(self) -> list[str]:
        """Run ids that might need reclaiming (the watchdog filters by ``is_stale``)."""
        ...

    def heartbeat(self, run_id: str, worker_id: str, ttl_s: float) -> None:
        """Refresh (or create) the lease the owning worker holds for its live run."""
        ...

    def is_stale(self, run_id: str) -> bool:
        """True if no lease exists or its heartbeat is older than its ttl (worker crashed)."""
        ...

    def try_claim(self, run_id: str, worker_id: str, ttl_s: float) -> bool:
        """Atomically claim a *stale* lease for ``worker_id``; return False if a live peer
        already holds it (or won the race). On success the lease now names this worker."""
        ...

    def owner(self, run_id: str) -> str | None:
        """The worker_id currently holding the lease, or None."""
        ...

    def release(self, run_id: str) -> None:
        """Drop the lease (terminal run, or a failed reclaim that must be retried)."""
        ...


@dataclass
class LocalFsLeaseStore:
    """Default lease store: ``run_root/<run_id>/lease.json`` + an ``O_EXCL`` ``.reclaim.lock``
    CAS. Same on-disk shape the watchdog used before the seam was extracted (no behavior
    change). Durable/visible only within one host or a shared coherent filesystem."""

    run_root: Path
    lock_timeout_s: float = 5.0

    def _lease_path(self, run_id: str) -> Path:
        return self.run_root / run_id / "lease.json"

    def _read(self, run_id: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(self._lease_path(run_id).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return None
        return payload if isinstance(payload, dict) else None

    def candidate_run_ids(self) -> list[str]:
        if not self.run_root.is_dir():
            return []
        return [path.name for path in self.run_root.iterdir() if path.is_dir()]

    def heartbeat(self, run_id: str, worker_id: str, ttl_s: float) -> None:
        write_json_atomic(
            self._lease_path(run_id),
            {
                "run_id": run_id,
                "worker_id": worker_id,
                "pid": os.getpid(),
                "heartbeat_at": time.time(),
                "lease_ttl_s": ttl_s,
            },
        )

    def is_stale(self, run_id: str) -> bool:
        lease = self._read(run_id)
        if lease is None:
            return True  # no lease (crashed before writing, or a legacy run) -> reclaimable
        ttl = float(lease.get("lease_ttl_s", 0.0))
        return (time.time() - float(lease.get("heartbeat_at", 0.0))) > ttl

    def try_claim(self, run_id: str, worker_id: str, ttl_s: float) -> bool:
        with file_lock(
            self.run_root / run_id / ".reclaim.lock",
            timeout_s=self.lock_timeout_s,
            stale_s=ttl_s * 2 + 5.0,
        ):
            if not self.is_stale(run_id):
                return False  # a peer reclaimed first -> back off
            self.heartbeat(run_id, worker_id, ttl_s)
            return True

    def owner(self, run_id: str) -> str | None:
        lease = self._read(run_id)
        return str(lease["worker_id"]) if lease and "worker_id" in lease else None

    def release(self, run_id: str) -> None:
        try:
            self._lease_path(run_id).unlink(missing_ok=True)
        except OSError:  # pragma: no cover - best effort
            pass
