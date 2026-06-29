"""Internal shared utilities for core modules (timestamps, atomic JSON writes).

Core-internal only; not part of the supported public surface and intentionally
not re-exported from ``native_agent_runner.contracts`` or the package root.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    """Current UTC time as ISO-8601 with a ``Z`` suffix (e.g. ``...T12:00:00Z``)."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(payload: dict[str, Any], *, drop: tuple[str, ...] = ()) -> str:
    """SHA-256 hex digest of ``payload`` serialized as canonical JSON.

    ``drop`` lists top-level keys removed before hashing (e.g. the hash field
    itself). Serialization is deterministic — ``sort_keys=True`` and compact
    separators — so the digest is stable across processes and must stay
    byte-identical to remain compatible with already-recorded hashes.
    """
    canonical = dict(payload)
    for key in drop:
        canonical.pop(key, None)
    data = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` as pretty JSON, replacing ``path`` atomically via a temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _atomic_replace(tmp_path, path)


def _atomic_replace(src: Path, dst: Path, *, attempts: int = 10, backoff_s: float = 0.01) -> None:
    """``os.replace`` with a bounded retry for Windows.

    On POSIX, replacing a destination that a reader currently has open is atomic and succeeds
    immediately. On Windows the same replace fails with ``PermissionError`` (ERROR_ACCESS_DENIED /
    sharing violation) while another handle holds the destination open — which happens whenever a
    status/checkpoint reader polls the file the run is updating. Readers open-read-close in
    microseconds, so a few short retries close the race; POSIX never reaches the retry path.
    """
    for attempt in range(attempts):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(backoff_s)


def read_text_resilient(path: Path, *, attempts: int = 10, backoff_s: float = 0.01) -> str:
    """``Path.read_text`` with a bounded retry for the Windows replace-vs-reader race.

    The symmetric counterpart of :func:`_atomic_replace`: that flips a destination via
    ``os.replace``; on Windows a reader that opens the destination mid-replace fails with
    ``PermissionError`` (sharing violation). A polling status/checkpoint reader is exactly that
    reader, so it needs the same short retry the writer already has. POSIX never reaches the
    retry path (replacing an open file is atomic there)."""
    for attempt in range(attempts):
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(backoff_s)
    raise AssertionError("unreachable")  # the loop returns or raises


@contextmanager
def file_lock(lock_path: Path, *, timeout_s: float = 10.0, stale_s: float = 30.0) -> Iterator[None]:
    """Best-effort cross-process advisory lock via an ``O_EXCL`` lock file.

    A waiter steals a lock older than ``stale_s`` (its holder presumably crashed) and,
    after ``timeout_s`` of waiting on a live holder, steals it anyway — so a stuck holder
    can never deadlock a caller. Released by unlinking the lock file on exit. Cross-process
    safe on both Windows and POSIX (``O_EXCL`` create is atomic on both)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    fd: int | None = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue  # holder released between our attempt and the stat — retry
            if age > stale_s or time.monotonic() > deadline:
                try:
                    lock_path.unlink()  # steal a stale lock or one we have waited out
                except FileNotFoundError:
                    pass
                continue
            time.sleep(0.02)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:  # pragma: no cover - stolen by a peer after timeout
            pass
