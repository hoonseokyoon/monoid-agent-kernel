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
``monoid validate`` runs under. Two belts narrow what a violated contract can cost: a candidate's
age must have reached ``min_age_s``, and adoption -- any writer accepting a chunk file that
already exists -- refreshes that file's times
(:func:`~monoid_agent_kernel.core._verified_file.write_verified_bytes_once`), so recent use looks
recent. Neither is a guarantee. The second is best-effort by construction (a touch the platform
refuses is swallowed, because the chunk *is* stored), and a writer that stalls past ``min_age_s``
between storing a chunk and appending the line that references it outlives both. The belts bound
the damage of a broken contract; they are not permission to break it.

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

import math
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core._verified_file import (
    VerifiedFileIdentity,
    file_identity,
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


# How many damaged corpus line numbers one report will name. Every other quantity in a report is
# bounded by files on disk; this one tracked corpus content one-to-one, so a million-line corpus
# put a million integers on one terminal line and into the JSON. The count is what an operator
# acts on -- "this corpus is torn" -- and the list is a sample to start reading from.
MAX_REPORTED_DAMAGED_LINES = 100


class UnusableAgeGate(ValueError):
    """The age gate cannot act as one, so no pass is attempted.

    Its own type, rather than a bare ``ValueError``, because the caller that renders it has to
    know it came from the *argument* and not from the sweep. This neighbourhood signals with
    ``ValueError`` in several places (the reassembly budget, chunk resolution), and a sweep that
    raised one mid-pass would otherwise be reported as a bad flag -- after deletions had already
    happened, which is the failure the check exists to prevent.
    """


@dataclass(frozen=True)
class PayloadGcEntry:
    """One chunk-directory entry as the collector judged it.

    ``classification``: ``kept`` (the keep-set names it -- which is deliberately more than "a
    record resolves it"; see :func:`~monoid_agent_kernel.core.model_payloads.corpus_keep_set`),
    ``orphan`` (chunk-shaped, outside the keep-set, so no reader can resolve it), ``temp`` (the
    write-once temporary shape over a sha stem), ``foreign`` (anything else -- never touched), or
    ``unjudged`` (chunk-shaped, but the corpus needed to judge it was absent or unreadable --
    never touched). ``age_s`` is against the single clock reading the whole pass used, and it is
    the quantity the ``min_age_s`` gate consumed. ``error`` names whatever stopped this entry
    from being handled as classified: a deletion that failed or was withheld, or -- in either
    mode -- a scan whose ``stat`` raised.
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

    ``chunk_dir_state``: ``absent``, ``unsafe`` (a redirection or a non-directory wearing the
    name -- somebody put it there), ``unreadable`` (the platform refused the listing, which on
    Windows is the shape of an antivirus pass, the indexer or a sync engine), or ``ok``. The two
    refusals are separated because they call for opposite responses, and the corpus half already
    made that distinction.

    ``candidate_bytes`` is the size of what ``--apply`` would remove (orphans and temps past the
    age gate), counted identically in both modes. It is an **upper bound** on bytes returned to
    the volume, not a promise of them: a name whose inode has other links frees nothing, and a
    scan cannot see a link count on every platform (Windows serves ``scandir`` stats from the
    directory listing, with ``st_nlink`` zero). ``reclaimed_bytes`` counts only files whose last
    name this pass removed, which is knowable, because the pre-unlink ``lstat`` is a real one.

    ``damaged_line_count`` is how many corpus lines no reader parses; ``damaged_lines`` names the
    first :data:`MAX_REPORTED_DAMAGED_LINES` of them.
    """

    run_dir: str
    chunk_dir_state: str
    corpus_state: str
    applied: bool
    min_age_s: float
    entries: tuple[PayloadGcEntry, ...]
    damaged_lines: tuple[int, ...]
    damaged_line_count: int
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


def _directory_still_approved(chunk_dir: Path, approved: VerifiedFileIdentity) -> str:
    """``""`` when ``chunk_dir`` is still the directory the gate approved, else why not.

    Identity is ``(st_dev, st_ino)``, and an inode number is only evidence *if the platform
    supplies one*: Python's own documentation says ``st_ino`` "if non-zero, uniquely identifies
    the file", CPython's Windows ``lstat`` leaves it zero when it falls back to directory
    attributes (an access-denied or sharing-violation open), and SMB/FAT volumes may have no
    stable file index at all. Comparing two zeroed identities is a check that passes always --
    a guard that silently becomes a no-op while the documentation still advertises it, which is
    worse than an absent one. A deleter cannot spend that: an unprovable identity withholds the
    deletion and says so. The cost is uncollected garbage on such a volume, reported per entry
    and exit-1 visible; the alternative cost is deleting in a directory nobody vouched for.

    Deliberately narrower than :func:`~monoid_agent_kernel.core._verified_file.file_identity`
    itself, whose other consumers compare an identity they captured from a descriptor they hold
    open -- and where failing closed on a zeroed inode would refuse ordinary writes on those same
    volumes rather than merely decline to delete.
    """

    if not approved.inode:
        return "chunk directory identity is not provable on this filesystem; left in place"
    current = file_identity(chunk_dir.lstat())
    if not current.inode or current != approved:
        return "chunk directory changed since the scan; left in place"
    return ""


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
    it by ``lstat`` immediately first -- still in the directory the gate approved, still a
    regular file, still past the gate against the same clock reading -- so an entry that stopped
    meeting those three tests since the scan is withheld with a per-entry error rather than
    deleted on stale evidence. Those are the three things the re-check can see; a same-name swap
    to a *different* old regular file inside the approved directory reads as unchanged, and is
    reachable only once the concurrency contract above is already broken. A deletion the platform
    refuses is likewise a per-entry error and the sweep finishes; a failed entry is loud precisely
    because it usually means the contract is being tested. That refusal is a Windows property: it
    holds an open file against unlink, so a writer racing this pass costs a loud error there and
    nothing at all on POSIX, where the unlink succeeds and the writer keeps filling an inode with
    no name. Two collectors overlapping is the same shape -- the loser reports one
    ``FileNotFoundError`` per entry and exits non-zero although the directory reached the state
    it asked for. The contract that forbids the first case forbids this one too: one sweep at a
    time, and none beside a writer.

    ``now`` is injectable so tests own the clock. Ages are ``now - st_mtime``; a file dated in
    the future is younger than any non-negative gate, which is the protective direction.

    ``min_age_s`` must be finite and non-negative, and a value that is not is refused here --
    before anything is read, so a refusal can never follow a sweep. The gate is the only safety
    belt this function has, and each unusable value breaks it a different way: infinity spares
    everything and then cannot be reported (``json.dumps`` refuses non-finite numbers), a NaN
    makes every comparison false so an applied pass becomes a silent no-op, and a negative gate
    deletes exactly what the belt exists to protect -- future-dated entries, and a candidate
    freshened between the scan and the unlink.
    """

    if not math.isfinite(min_age_s) or min_age_s < 0:
        raise UnusableAgeGate("must be a finite, non-negative number of seconds")
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
            damaged_lines=tuple(damaged[:MAX_REPORTED_DAMAGED_LINES]),
            damaged_line_count=len(damaged),
            candidate_bytes=candidate_bytes,
            reclaimed_bytes=reclaimed_bytes,
        )

    chunk_dir = run_dir / MODEL_PAYLOADS_DIRNAME
    try:
        approved = chunk_dir.lstat()
    except FileNotFoundError:
        return report("absent")
    except OSError:
        # Not ``unsafe``: the platform declining to answer is not somebody having planted
        # something. On Windows this is the shape of an antivirus pass, the search indexer or a
        # sync engine holding the directory for a moment, and telling an operator they were
        # attacked when they were merely unlucky sends them the wrong way.
        return report("unreadable")
    if not verified_directory_is_safe(chunk_dir):
        # The gate the writer's own sweep runs behind, for the same reason: enumeration and
        # deletion through a redirection are operations in a directory of somebody else's
        # choosing. Nothing is listed, nothing is touched.
        return report("unsafe")
    # Which directory the gate approved, so every unlink can re-prove it is standing in that one.
    # A gate that runs once governs a pathname, and each deletion re-resolves that pathname: a
    # redirection planted after the scan would aim the rest of the pass elsewhere, and elsewhere
    # a name that is garbage here can be a referenced chunk (tool-definition chunks are
    # byte-identical across runs, so one run's orphan sha is another run's live one).
    approved_directory = file_identity(approved)

    snapshot: list[tuple[str, os.stat_result | None]] = []
    try:
        with os.scandir(chunk_dir) as listing:
            for entry in listing:
                try:
                    # Mode, size and mtime only. Windows serves these from the directory listing,
                    # where ``st_dev``/``st_ino``/``st_nlink`` come back zero for every ordinary
                    # file -- so nothing here may be handed to ``file_identity`` or read as a
                    # link count. Both of those questions get a real ``lstat`` below.
                    snapshot.append((entry.name, entry.stat(follow_symlinks=False)))
                except OSError:
                    snapshot.append((entry.name, None))
    except OSError:
        return report("unreadable")
    snapshot.sort(key=lambda pair: pair[0])

    keep = corpus_keep_set(records) if corpus_state == "ok" else set()
    # The corpus was the largest thing this pass allocated and its records are spent: the keep-set
    # is all the directory pass consults. Releasing them here keeps a big corpus from being
    # resident for the whole sweep.
    records = []
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
                    # The directory first, and by itself: every fact gathered after this one is
                    # read through the pathname it governs. Asking about the *entry* first put the
                    # likelier half of a swap -- a planted directory that does not happen to
                    # mirror this sha -- on the ``FileNotFoundError`` arm, which tells the
                    # operator "a file vanished, probably a writer" when the truth is "every
                    # remaining line of this report describes somebody else's directory".
                    error = _directory_still_approved(chunk_dir, approved_directory)
                    if not error:
                        current = target.lstat()
                        if not stat.S_ISREG(current.st_mode) or (
                            moment - current.st_mtime
                        ) < min_age_s:
                            error = "changed since the scan; left in place"
                        else:
                            # Counted only when this name was the inode's last one. A
                            # hardlink-deduplicated archive of a run directory is a supported
                            # shape (``docs/OBSERVABILITY.md`` blesses ``cp -al`` for these very
                            # files), and there an unlink returns nothing to the volume while the
                            # archive's name goes on holding the bytes. A count of zero is the
                            # honest answer for a capacity script; the file is still gone, and
                            # ``deleted`` says so. ``st_nlink == 0`` means the platform did not
                            # answer, which is treated as one name -- the pre-change behavior.
                            last_name = current.st_nlink <= 1
                            os.unlink(target)
                            deleted = True
                            if last_name:
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
