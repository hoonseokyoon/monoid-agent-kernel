"""Collecting what no reader of one run's replay corpus can resolve.

The ``model_payloads/`` directory holds content-addressed offload chunks, and two kinds of litter
can survive a run: an **orphan chunk** whose referencing record never landed (a crash between the
chunk's ``os.replace`` and the line append, or an append the disk refused), and a **dead
temporary** from a writer that died mid-write in another process (the recorder's own sweep is
deliberately pid-scoped, because a pid in a filename cannot prove its writer dead). This module
is the collector those sites defer to, and it keeps one invariant above everything:
``validate_run_dir`` reports exactly the same issues after a sweep as before it, because the only
thing ever deleted is what no record lets any reader resolve.

Concurrency is a contract, not a mechanism: **never run this beside a live writer of the same run
directory.** The writer takes no cross-process lock, and nothing on disk distinguishes a dead
writer from a live one, so liveness is the operator's knowledge -- the same trust model
``monoid validate`` runs under. Two belts make even a violated contract non-destructive for
anything touched within their window: a candidate must be older than ``min_age_s``, and adoption
-- a resumed writer re-deriving a chunk that already exists -- refreshes the file's times
(:func:`~monoid_agent_kernel.core._verified_file.write_verified_bytes_once`), so recent use looks
recent. The belts bound the damage of a broken contract; they are not permission to break it.

Judging chunk-shaped files needs the corpus. When ``model_payloads.jsonl`` is absent or cannot be
opened as this run's own regular file, those files are reported ``unjudged`` and left alone: a
mutilated directory and a first-call crash whose very first chunk was directory-sized leave the
same state (the chunk file lands before the corpus file's lazy create), and an invented empty
keep-set would purge a corpus that merely lost its index. Temporaries need no corpus -- no record
ever references one -- so the litter half runs regardless. Damaged *lines* inside a readable
corpus are the opposite case: every reader skips them (the validator's lenient loop, mirrored
here line for line), so what only they referenced is unreachable by construction, collectable,
and the report names the line numbers so the damage itself is never silent.

The collector never opens a chunk file -- classification is by name, keep-set membership and
``lstat`` alone, so there is no read to bound -- and never builds a path from corpus content:
names come from ``os.scandir``, and the keep-set is only ever tested for membership.
"""

from __future__ import annotations

import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core._verified_file import (
    open_verified_regular_fd,
    verified_directory_is_safe,
    write_once_temp_stem,
)
from monoid_agent_kernel.core.json_ingress import loads_json_ingress
from monoid_agent_kernel.core.model_payloads import (
    MODEL_PAYLOADS_DIRNAME,
    MODEL_PAYLOADS_FILENAME,
    corpus_keep_set,
    is_chunk_sha256,
)


@dataclass(frozen=True)
class PayloadGcEntry:
    """One chunk-directory entry as the collector judged it.

    ``classification``: ``kept`` (a record resolves it), ``orphan`` (chunk-shaped, nothing
    resolves it), ``temp`` (the write-once temporary shape over a sha stem), ``foreign``
    (anything else -- never touched), or ``unjudged`` (chunk-shaped, but the corpus needed to
    judge it was absent or unreadable -- never touched). ``age_s`` is against the single clock
    reading the whole pass used, and it is the quantity the ``min_age_s`` gate consumed.
    ``error`` is empty unless an applied deletion failed or was withheld.
    """

    name: str
    classification: str
    size: int
    age_s: float
    deleted: bool
    error: str


@dataclass(frozen=True)
class PayloadGcReport:
    """What one pass saw, and what it did.

    ``candidate_bytes`` is what ``--apply`` would reclaim (orphans and temps past the age gate),
    counted identically in both modes; ``reclaimed_bytes`` is what an applied pass actually
    deleted. ``damaged_lines`` are the 1-based corpus line numbers no reader parses.
    """

    run_dir: str
    chunk_dir_state: str
    corpus_state: str
    applied: bool
    min_age_s: float
    entries: tuple[PayloadGcEntry, ...]
    damaged_lines: tuple[int, ...]
    candidate_bytes: int
    reclaimed_bytes: int


def _corpus_records(path: Path) -> tuple[str, list[dict[str, Any]], list[int]]:
    """(state, parseable records, damaged line numbers).

    The read goes through the verified opener because this reader's conclusions *delete*: a
    corpus reached through a planted link is not this run's corpus, and judging from it would
    turn the swap into a purge. A hard link is accepted (``require_single_link=False``) for the
    reason the chunk reader accepts one -- a hardlink-deduplicated archive is still these bytes.
    The line loop mirrors ``_validate_model_payload_digests`` exactly: blank lines skip
    silently, a line that fails ingress parsing or is not an object is damaged, the rest count.
    """

    try:
        path.lstat()
    except FileNotFoundError:
        return "absent", [], []
    except OSError:
        return "unreadable", [], []
    descriptor = open_verified_regular_fd(path, os.O_RDONLY, require_single_link=False)
    if descriptor is None:
        return "unreadable", [], []
    handle = None
    try:
        handle = os.fdopen(descriptor, "rb")
        descriptor = None  # owned by ``handle`` from here
        data = handle.read()
    except (OSError, ValueError):
        return "unreadable", [], []
    finally:
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        elif descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    records: list[dict[str, Any]] = []
    damaged: list[int] = []
    for index, raw_line in enumerate(data.split(b"\n"), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = loads_json_ingress(raw_line.decode("utf-8"))
        except Exception:  # noqa: BLE001 - unparseable is a classification here, not a failure
            damaged.append(index)
            continue
        if not isinstance(payload, dict):
            damaged.append(index)
            continue
        records.append(payload)
    return "ok", records, damaged


def _classification(
    name: str, metadata: os.stat_result | None, *, keep: set[str], corpus_ok: bool
) -> str:
    if metadata is None or not stat.S_ISREG(metadata.st_mode):
        return "foreign"
    if is_chunk_sha256(name):
        if not corpus_ok:
            return "unjudged"
        return "kept" if name in keep else "orphan"
    stem = write_once_temp_stem(name)
    if stem is not None and is_chunk_sha256(stem):
        return "temp"
    return "foreign"


def collect_payload_garbage(
    run_dir: Path, *, min_age_s: float, apply: bool, now: float | None = None
) -> PayloadGcReport:
    """One pass over one run directory's chunk directory: judge everything, delete candidates
    only when asked.

    ``apply=False`` judges and counts; ``apply=True`` also unlinks each candidate, re-checking
    it by ``lstat`` immediately first -- still a regular file, still past the gate against the
    same clock reading -- so a name that changed since the scan is withheld with a per-entry
    error rather than deleted on stale evidence. A deletion the platform refuses (Windows holds
    a file someone has open) is likewise a per-entry error and the sweep finishes; a failed
    entry is loud precisely because it usually means the concurrency contract is being tested.

    ``now`` is injectable so tests own the clock. Ages are ``now - st_mtime``; a file dated in
    the future is younger than any gate, which is the protective direction.
    """

    moment = time.time() if now is None else now
    corpus_state, records, damaged = _corpus_records(run_dir / MODEL_PAYLOADS_FILENAME)

    def report(
        chunk_dir_state: str,
        entries: tuple[PayloadGcEntry, ...] = (),
        candidate_bytes: int = 0,
        reclaimed_bytes: int = 0,
    ) -> PayloadGcReport:
        return PayloadGcReport(
            run_dir=str(run_dir),
            chunk_dir_state=chunk_dir_state,
            corpus_state=corpus_state,
            applied=apply,
            min_age_s=min_age_s,
            entries=entries,
            damaged_lines=tuple(damaged),
            candidate_bytes=candidate_bytes,
            reclaimed_bytes=reclaimed_bytes,
        )

    chunk_dir = run_dir / MODEL_PAYLOADS_DIRNAME
    try:
        chunk_dir.lstat()
    except FileNotFoundError:
        return report("absent")
    except OSError:
        return report("unsafe")
    if not verified_directory_is_safe(chunk_dir):
        # The gate the writer's own sweep runs behind, for the same reason: enumeration and
        # deletion through a redirection are operations in a directory of somebody else's
        # choosing. Nothing is listed, nothing is touched.
        return report("unsafe")

    snapshot: list[tuple[str, os.stat_result | None]] = []
    try:
        with os.scandir(chunk_dir) as listing:
            for entry in listing:
                try:
                    snapshot.append((entry.name, entry.stat(follow_symlinks=False)))
                except OSError:
                    snapshot.append((entry.name, None))
    except OSError:
        return report("unsafe")
    snapshot.sort(key=lambda pair: pair[0])

    keep = corpus_keep_set(records) if corpus_state == "ok" else set()
    entries: list[PayloadGcEntry] = []
    candidate_bytes = 0
    reclaimed_bytes = 0
    for name, metadata in snapshot:
        classification = _classification(
            name, metadata, keep=keep, corpus_ok=corpus_state == "ok"
        )
        size = int(metadata.st_size) if metadata is not None else 0
        age_s = (moment - metadata.st_mtime) if metadata is not None else 0.0
        deleted = False
        error = "" if metadata is not None else "could not stat"
        if classification in ("orphan", "temp") and age_s >= min_age_s:
            candidate_bytes += size
            if apply:
                target = chunk_dir / name
                try:
                    current = target.lstat()
                    if not stat.S_ISREG(current.st_mode) or (
                        moment - current.st_mtime
                    ) < min_age_s:
                        error = "changed since the scan; left in place"
                    else:
                        os.unlink(target)
                        deleted = True
                        reclaimed_bytes += size
                except OSError as exc:
                    error = f"{type(exc).__name__}: {exc}"
        entries.append(
            PayloadGcEntry(
                name=name,
                classification=classification,
                size=size,
                age_s=age_s,
                deleted=deleted,
                error=error,
            )
        )
    return report(
        "ok",
        entries=tuple(entries),
        candidate_bytes=candidate_bytes,
        reclaimed_bytes=reclaimed_bytes,
    )
