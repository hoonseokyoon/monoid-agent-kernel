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
from datetime import UTC, datetime
from dataclasses import dataclass
from pathlib import Path

from monoid_agent_kernel.core._verified_file import (
    VerifiedFileIdentity,
    directory_metadata_is_safe,
    file_identity,
    write_once_temp_stem,
)
from monoid_agent_kernel.core.json_ingress import portable_type_name
from monoid_agent_kernel.core.model_payloads import (
    MODEL_PAYLOADS_DIRNAME,
    MODEL_PAYLOADS_FILENAME,
    corpus_keep_set,
    is_chunk_sha256,
    read_corpus_records,
)


# How many damaged corpus line numbers one report will name. Every other quantity in a report is
# bounded by files on disk; this one tracked corpus content one-to-one, so a million-line corpus
# put a million integers on one terminal line and into the JSON. The count is what an operator
# acts on -- "this corpus is torn" -- and the list is a sample to start reading from.
MAX_REPORTED_DAMAGED_LINES = 100


def utc_timestamp_of(moment: float) -> str:
    """``moment`` -- the epoch reading every ``age_s`` in a report is relative to -- as ISO-8601.

    Derived from the pass's own clock reading rather than taken freshly, so the stamp and the ages
    describe one instant. Spelled the way ``core._util.utc_timestamp`` spells now, which is what
    every other artifact in a run directory carries.
    """

    return datetime.fromtimestamp(moment, UTC).isoformat().replace("+00:00", "Z")


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
    the quantity the ``min_age_s`` gate consumed. ``reclaimed`` is the bytes this entry returned
    to the volume -- ``size`` when the sweep removed the inode's last name, ``0`` otherwise, so a
    consumer summing it gets the report's own total and can name the entry a hardlink accounted
    for rather than only observe the aggregate gap. ``error`` names whatever stopped this entry
    from being handled as classified: a deletion that failed or was withheld, or -- in either
    mode -- a scan whose ``stat`` raised.
    """

    name: str
    classification: str
    size: int
    age_s: float
    deleted: bool
    reclaimed: int
    error: str


@dataclass(frozen=True)
class PayloadGcReport:
    """What one pass saw, and what it did.

    ``chunk_dir_state``: ``absent``; ``unsafe`` (a redirection or a non-directory wearing the
    name -- somebody put it there); ``unreadable`` (the platform refused, which on Windows is the
    shape of an antivirus pass, the indexer or a sync engine); ``unprovable`` (a volume that
    supplies no stable file ids, so no deletion here could be re-proved and none is attempted);
    ``swapped`` (the gate approved a directory and something else was standing in its place
    before the pass finished -- every entry below it describes whatever was there at the time);
    or ``ok``. The refusals are separated because they call for opposite responses, and the
    corpus half already made that distinction.

    ``candidate_bytes`` is the size of what ``--apply`` would remove (orphans and temps past the
    age gate), counted identically in both modes. It is an **upper bound** on bytes returned to
    the volume, not a promise of them: a name whose inode has other links frees nothing, and a
    scan cannot see a link count on every platform (Windows serves ``scandir`` stats from the
    directory listing, with ``st_nlink`` zero). ``reclaimed_bytes`` counts only files whose last
    name this pass removed -- knowable because the pre-unlink ``lstat`` is a real one, except
    where that ``lstat`` itself reports no link count (the same Windows attribute fallback), which
    counts as one name and is the one way this number can over-report.

    ``damaged_line_count`` is how many corpus lines no reader parses; ``damaged_lines`` names the
    first :data:`MAX_REPORTED_DAMAGED_LINES` of them.

    ``swept_at`` is when the pass read its clock. Every ``age_s`` is relative to that instant, so
    without it a saved report cannot be placed in time -- and this report is the only record the
    verb leaves, since it writes nothing to the run's event log.
    """

    run_dir: str
    swept_at: str
    chunk_dir_state: str
    corpus_state: str
    applied: bool
    min_age_s: float
    entries: tuple[PayloadGcEntry, ...]
    damaged_lines: tuple[int, ...]
    damaged_line_count: int
    candidate_bytes: int
    reclaimed_bytes: int


# The collector's line reader IS the shared one (W6-4b moved it whole to model_payloads so the
# replay reader consumes the same function instead of a mirror). The alias keeps this module's
# vocabulary -- its callers and comments say "corpus records" in collector terms -- and the
# reader tests pin the identity, so a re-definition here would fail loudly rather than drift.
_corpus_records = read_corpus_records


def _directory_still_approved(chunk_dir: Path, approved: VerifiedFileIdentity) -> str:
    """``""`` when ``chunk_dir`` is still the directory the gate approved, else why not.

    Only reached when the gate proved an identity worth comparing: identity is ``(st_dev,
    st_ino)``, and an inode number is evidence only *if the platform supplies one*. Python's own
    documentation says ``st_ino`` "if non-zero, uniquely identifies the file", CPython's Windows
    ``lstat`` leaves it zero when it falls back to directory attributes, and SMB/FAT volumes may
    have no stable file index at all -- where two zeroed identities compare equal and this check
    would pass always, a guard silently becoming a no-op while the documentation advertises it.
    That question is settled once, at the gate, so both modes give the same answer; here the
    comparison can be taken at face value.

    Deliberately narrower than :func:`~monoid_agent_kernel.core._verified_file.file_identity`
    itself, whose other consumers compare an identity they captured from a descriptor they hold
    open -- and where failing closed on a zeroed inode would refuse ordinary writes on those same
    volumes rather than merely decline to delete.
    """

    try:
        current = file_identity(chunk_dir.lstat())
    except OSError:
        # The directory being gone or unreadable *is* the answer, and it must not fall through to
        # the caller's per-entry ``except OSError``, which would name the failure after the entry
        # -- the same misattribution the ordering fix removed, surviving on its sibling half.
        return "the chunk directory changed since the scan; left in place"
    if current != approved:
        return "the chunk directory changed since the scan; left in place"
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
    swept_at = utc_timestamp_of(moment)
    corpus_state, records, damaged = _corpus_records(run_dir / MODEL_PAYLOADS_FILENAME)

    def report(
        chunk_dir_state: str,
        entries: tuple[PayloadGcEntry, ...] = (),
        candidate_bytes: int = 0,
        reclaimed_bytes: int = 0,
    ) -> PayloadGcReport:
        return PayloadGcReport(
            run_dir=str(run_dir),
            swept_at=swept_at,
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
    if not directory_metadata_is_safe(approved):
        # The gate the writer's own sweep runs behind, for the same reason: enumeration and
        # deletion through a redirection are operations in a directory of somebody else's
        # choosing. Nothing is listed, nothing is touched. Asked of the stat already in hand,
        # never of the path: the path-taking form answers ``False`` for an unreadable directory
        # too, which would put the platform declining back under "somebody planted this" -- the
        # third of the three refusal sites, and the one a boolean hid.
        return report("unsafe")
    # Which directory the gate approved, so every unlink can re-prove it is standing in that one.
    # A gate that runs once governs a pathname, and each deletion re-resolves that pathname: a
    # redirection planted after the scan would aim the rest of the pass elsewhere, and elsewhere
    # a name that is garbage here can be a referenced chunk (tool-definition chunks are
    # byte-identical across runs, so one run's orphan sha is another run's live one).
    approved_directory = file_identity(approved)
    # Decided once, here, rather than per candidate, so both modes answer the same. An inode
    # number is evidence only where the platform supplies one, and without it no deletion in this
    # directory can ever be re-proved -- so nothing is a candidate, ``--apply`` has nothing to do,
    # and a report-only pass says so instead of promising bytes it could not have removed. Per
    # candidate it produced N copies of one sentence in one mode and silence in the other.
    provable = bool(approved_directory.inode)

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
    except FileNotFoundError:
        return report("absent")
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
    swapped = False
    for name, metadata in snapshot:
        classification = _classification(
            name, metadata, keep=keep, corpus_ok=corpus_state == "ok"
        )
        size = int(metadata.st_size) if metadata is not None else 0
        age_s = (moment - metadata.st_mtime) if metadata is not None else 0.0
        deleted = False
        reclaimed = 0
        error = "" if metadata is not None else "could not stat"
        if provable and classification in ("orphan", "temp") and age_s >= min_age_s:
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
                    if error:
                        swapped = True
                    else:
                        current = target.lstat()
                        if not stat.S_ISREG(current.st_mode) or (
                            moment - current.st_mtime
                        ) < min_age_s:
                            error = "this entry changed since the scan; left in place"
                        else:
                            # Counted only when this name was the inode's last one. A
                            # hardlink-deduplicated archive of a run directory is a supported
                            # shape (``docs/OBSERVABILITY.md`` blesses ``cp -al`` for these very
                            # files), and there an unlink returns nothing to the volume while the
                            # archive's name goes on holding the bytes. A count of zero is the
                            # honest answer for a capacity script; the file is still gone, and
                            # ``deleted`` says so. ``st_nlink == 0`` means the platform did not
                            # answer, which is treated as one name -- the pre-change behavior,
                            # and the one place this number can over-report.
                            last_name = current.st_nlink <= 1
                            os.unlink(target)
                            deleted = True
                            if last_name:
                                # The fresh size, not the scan's: the two are the same file only
                                # if nothing rewrote it, which is what the mtime test above
                                # establishes -- so take the measurement from the stat that made
                                # the decision.
                                reclaimed = int(current.st_size)
                                reclaimed_bytes += reclaimed
                except OSError as exc:
                    error = f"{portable_type_name(exc)}: {exc}"
        entries.append(
            PayloadGcEntry(
                name=name,
                classification=classification,
                size=size,
                age_s=age_s,
                deleted=deleted,
                reclaimed=reclaimed,
                error=error,
            )
        )
    # A swap discovered mid-pass has to reach the top of the report. ``chunk_dir_state`` said
    # "ok" -- true of the gate, and stale by the time it was printed -- so a consumer reading
    # states alone saw a healthy sweep, and the only carrier was a per-entry string one of whose
    # siblings is a substring of it.
    return report(
        "unprovable" if not provable else "swapped" if swapped else "ok",
        entries=tuple(entries),
        candidate_bytes=candidate_bytes,
        reclaimed_bytes=reclaimed_bytes,
    )
