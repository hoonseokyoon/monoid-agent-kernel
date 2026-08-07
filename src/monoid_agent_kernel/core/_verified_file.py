"""Opening a run-directory artifact without accepting an indirection planted in its place.

A run directory is a privacy boundary, and everything the kernel appends inside it is supposed to
be a file this process created. A *pathname* is not evidence of that. ``open(path, "a")`` follows a
symlink, and a hard link is a second name for an inode that may live anywhere on the volume with
anyone's permissions -- so either one silently turns "append a line to my own artifact" into
"append it to a file someone else chose", written with the agent's credentials. The exposure is
real wherever a run directory can be touched between runs, which includes every reopened durable
run.

The rule these functions enforce is therefore not "the path looks right" but "the bytes I touch
belong to a regular file in a real directory, and to the same inode I just verified":
``O_NOFOLLOW`` where the platform has it, ``lstat`` before the open for the platforms that do not,
``S_ISREG`` on both the named entry and the opened descriptor, and ``os.path.samestat`` to close
the window between them.

That rule covers the **final** component. The two chunk primitives also check their parent
(:func:`verified_directory_is_safe`), because the corpus put a subdirectory inside the run
directory and ``mkdir(exist_ok=True)`` follows a link planted at it. The append opener does not,
because its parent *is* the run directory, and a symlinked run root is an ordinary deployment
choice rather than an attack -- an indirection planted at the run directory itself is the
deployment's boundary to defend, not this module's.

``st_nlink == 1`` is part of that rule only for the artifacts these functions *append to*: a hard
link is a second name for an inode a writer is about to mutate. It is deliberately not required of
a content-addressed chunk, whose bytes nothing rewrites (adoption refreshes only its times; see
:func:`write_verified_bytes_once`) and which the reader authenticates by re-hashing -- requiring
it there would fail every hardlink-deduplicated archive of a run directory.

These primitives were written for ``model-content.jsonl`` and live here because they were never
about that artifact. ``model_calls.jsonl`` is the second append-only sidecar in the same directory
with exactly the same exposure, and this package's recurring defect is a rule proven on one of two
parallel sites and never bound on its twin. One function, both callers.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class VerifiedFileIdentity:
    """Stable identity for one verified inode."""

    device: int
    inode: int


def file_identity(metadata: os.stat_result) -> VerifiedFileIdentity:
    return VerifiedFileIdentity(device=metadata.st_dev, inode=metadata.st_ino)


def verified_file_is_safe(
    path: Path, *, allow_missing: bool = True, require_single_link: bool = True
) -> bool:
    """Whether a path is absent or an ordinary file this process may use, without following links.

    A missing file is safe for a lazy writer to create unless ``allow_missing`` is false. Existing
    links, directories, FIFOs, devices, and other special files fail closed for readers and writers.

    ``require_single_link`` is the append artifacts' rule: a hard link is a second name for the
    inode a writer is about to *mutate*, so an appender that accepted one would be writing into
    somebody else's file. It does not follow for reading a **content-addressed** file, where the
    bytes are authenticated by re-hashing them and the link count says nothing about them -- and
    where insisting on it turns any hardlink-deduplicated archive of a run directory (``cp -al``,
    ``rsync --link-dest``, a restored backup) into a corpus-wide integrity failure.
    """

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return allow_missing
    except OSError:
        return False
    if not stat.S_ISREG(metadata.st_mode):
        return False
    return metadata.st_nlink == 1 or not require_single_link


def directory_metadata_is_safe(metadata: os.stat_result) -> bool:
    """Whether an ``lstat`` result describes a real directory rather than a redirection wearing
    one's shape.

    Split out from :func:`verified_directory_is_safe` so a caller that already holds the stat can
    ask the question without a second ``lstat`` -- and, more importantly, without folding its own
    ``OSError`` into the same ``False``. A boolean cannot tell "somebody planted this" apart from
    "the platform declined to answer", and a caller that reports those differently (the collector
    does; they call for opposite responses) has to keep the two questions separate. One authoring
    site for the rule either way: the path-taking form below is this one plus a lookup.
    """

    if getattr(metadata, "st_reparse_tag", 0):
        return False  # a junction lstats as a directory; only the tag distinguishes it
    return stat.S_ISDIR(metadata.st_mode)


def verified_directory_is_safe(path: Path) -> bool:
    """Whether ``path`` is absent or a real directory, not a redirection wearing one's shape.

    The final path component was the only one the file primitives checked, and a run directory has
    subdirectories now. ``Path.mkdir(exist_ok=True)`` asks ``is_dir()``, which *follows*, so a
    symlink -- or a Windows junction, which needs no privilege to create -- planted at
    ``model_payloads/`` accepted every chunk write and sent it out of the run directory, with the
    writer told it had succeeded.

    Fails closed on an unreadable path, which is right for a *writer* -- it declines to write --
    and is why a reporting caller wants :func:`directory_metadata_is_safe` instead.
    """

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return directory_metadata_is_safe(metadata)


def open_verified_regular_fd(
    path: Path,
    flags: int,
    *,
    expected_identity: VerifiedFileIdentity | None = None,
    require_single_link: bool = True,
) -> int | None:
    """Open ``path`` without accepting a link/special-file swap before the first I/O."""

    if not verified_file_is_safe(path, require_single_link=require_single_link):
        return None
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o666)
    except OSError:
        return None
    try:
        opened = os.fstat(descriptor)
        named = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or (require_single_link and (opened.st_nlink != 1 or named.st_nlink != 1))
            or not os.path.samestat(opened, named)
            or (expected_identity is not None and file_identity(opened) != expected_identity)
        ):
            os.close(descriptor)
            return None
    except OSError:
        try:
            os.close(descriptor)
        except OSError:
            pass
        return None
    return descriptor


def open_verified_append_text(path: Path) -> TextIO | None:
    """Open one JSONL artifact for verified UTF-8 appending, or refuse it entirely.

    Returns ``None`` for every refusal -- a planted link, a special file, a directory that cannot be
    created, a descriptor that could not be wrapped -- so a caller has one thing to check and one
    thing to do about it: stop writing this artifact. Callers must treat ``None`` as terminal rather
    than retrying, because the reason is a property of the path, not a transient.

    The torn last line is closed off here, on the descriptor that was just verified, rather than by
    reopening the pathname: a second ``open`` of the same name is a second chance to be handed a
    different file, which is the very substitution the verified open exists to refuse. Appending
    after a line that lacks its newline concatenates the remnant and the new record into one
    unparseable line and loses **both**, so a crash would otherwise cost the next activation's first
    record as well as its own.
    """

    descriptor: int | None = None
    handle: TextIO | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = open_verified_regular_fd(path, os.O_RDWR | os.O_CREAT | os.O_APPEND)
        if descriptor is None:
            return None
        size = os.fstat(descriptor).st_size
        torn_tail = False
        if size:
            os.lseek(descriptor, size - 1, os.SEEK_SET)
            torn_tail = os.read(descriptor, 1) != b"\n"
        handle = os.fdopen(descriptor, "a", encoding="utf-8", newline="\n")
        descriptor = None  # owned by ``handle`` from here
        if torn_tail:
            handle.write("\n")
            handle.flush()
        return handle
    except (OSError, ValueError):
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
        return None


def read_verified_bytes(path: Path, *, max_bytes: int) -> bytes | None:
    """Read one run-directory file this process is entitled to read, or refuse it entirely.

    The reading twin of :func:`open_verified_append_text`, and it exists because the writer's
    guarantees are worth nothing if the reader re-establishes none of them. ``Path.read_bytes`` was
    the defect: it follows a link out of the run directory, it blocks forever on a FIFO, and it has
    no size bound, so a content-addressed name planted by anyone with write access to the directory
    turns a validation pass into an arbitrary read, a hang, or an out-of-memory.

    ``None`` for every refusal -- a symlink, a special file, a redirected parent directory, a file
    larger than ``max_bytes``, an I/O error. A *hard* link is accepted; see
    :func:`verified_file_is_safe`. The caller re-hashes what it gets, which is what makes the *content* trustworthy; this
    function is only responsible for the bytes having come from a file the run directory owns.
    """

    if not verified_directory_is_safe(path.parent):
        return None
    # Content-addressed, so the link count is not the reader's business: the caller re-hashes what
    # it gets, and refusing a multiply-linked file would fail every hardlink-deduplicated archive.
    descriptor = open_verified_regular_fd(path, os.O_RDONLY, require_single_link=False)
    if descriptor is None:
        return None
    try:
        if os.fstat(descriptor).st_size > max_bytes:
            return None
        blocks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                return b"".join(blocks)
            total += len(block)
            if total > max_bytes:
                # The size was checked above; a file still growing under us does not get to
                # decide how much memory this pass uses.
                return None
            blocks.append(block)
    except OSError:
        return None
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


# The temporary-name shape ``write_verified_bytes_once`` mints, matched where it is minted. The
# pid segment is matched, never trusted: pids are reused, so it identifies a writer's *naming*
# and nothing about its liveness -- freshness is the caller's own filter.
#
# ``[0-9]`` rather than ``\d``, which in a Python regex accepts every Unicode decimal digit: an
# Arabic-Indic pid matched a shape ``os.getpid()`` cannot produce, and matching here is a licence
# to delete.
_WRITE_ONCE_TEMP_NAME = re.compile(r"(.+)\.[0-9]+\.[0-9a-f]{12}\.tmp")


def write_once_temp_stem(name: str) -> str | None:
    """The stored name a :func:`write_verified_bytes_once` temporary was carrying bytes for, or
    ``None`` when ``name`` was never one of its temporaries. One authoring site: this predicate
    sits beside the f-string that mints the shape (``{name}.{pid}.{12 hex}.tmp``), so a collector
    matching crash litter and the writer creating it cannot drift apart. What the stem *means* is
    the caller's question -- this module does not know it stores content-addressed names.

    Case-sensitive, deliberately, and therefore narrower than the recorder's own pid-scoped
    ``Path.glob`` sweep, which folds case on Windows. A temporary whose name reached the directory
    with any letter re-cased -- a restore through a case-mangling path -- is classified foreign
    and never collected. That is the safe end of the asymmetry: the wider matcher is the one with
    no age gate and no directory re-check behind it, and widening a *delete* predicate to accept
    spellings the writer cannot mint is the wrong direction.
    """

    match = _WRITE_ONCE_TEMP_NAME.fullmatch(name)
    return match.group(1) if match is not None else None


def write_verified_bytes_once(path: Path, data: bytes) -> bool:
    """Create one write-once content-addressed file, or report that it could not be done.

    The write-once half: if ``path`` already holds a regular file **of exactly this length**,
    no bytes are written, the accepted name's times are refreshed (adoption; the comment in the
    body says why), and the answer is ``True`` -- content addressing means a file of the right
    name and the right size is almost certainly this content, and :func:`read_verified_bytes`
    re-hashes what it reads, so the remaining lie is caught at resolution. Anything else under that
    name is refused, and a refusal here is terminal for the caller's artifact: "already written",
    "someone planted an indirection" and "these are not those bytes" are three different answers,
    and stopping loudly beats handing a reader a reference to content that was never stored.

    The verified half mirrors :func:`open_verified_append_text` for the *creation* path: the bytes
    land in a uniquely named temporary sibling opened with ``O_CREAT | O_EXCL | O_NOFOLLOW`` --
    exclusive creation cannot be redirected by a planted link, and a link planted at the temporary
    name fails the open outright -- then take the final name via ``os.replace``, which replaces a
    link *itself* rather than writing through it (the same shape the checkpoint store's blob
    writer uses). A crash between the two leaves an orphaned ``*.tmp`` -- swept at the owner's
    next open under the same pid, and by ``monoid gc`` across pids, behind its age gate -- never
    a half-written file under a content-addressed name that would poison every reader trusting
    the name.

    ``False`` is terminal for the artifact the caller is building, for the reason the append
    opener's ``None`` is: the refusal is a property of the path.
    """

    try:
        if not verified_directory_is_safe(path.parent):
            return False
        try:
            existing: os.stat_result | None = path.lstat()
        except FileNotFoundError:
            existing = None
        if existing is not None:
            # ``path.exists()`` was the defect here, and it was the module docstring's own defect:
            # it follows a link, so a name planted at a predictable content-addressed sha reported
            # "already stored" for a chunk that was never written, and left every reader resolving
            # whoever's bytes the link points at. An existing name is accepted only when it is a
            # regular file -- no symlink, no FIFO, no directory.
            #
            # Not ``st_nlink == 1``, which is the *appenders'* rule: a second name for an inode
            # this process is about to rewrite is somebody else's file, but this function rewrites
            # no bytes (``O_EXCL`` on a temp, then ``os.replace``; the adoption touch below moves
            # times only, an archive's other name included), and a link count is what every
            # hardlink-deduplicating archive of a run directory changes. A hard link here is a
            # second name for a real file inside this directory, not an escape from it.
            #
            # The size is checked instead, because it is the thing that actually separates the two
            # cases: an archive's link points at *these bytes*, a planted one almost never does.
            # Refusing on a mismatch keeps the loud write-time stop for the planted case rather
            # than deferring it to whoever next runs a validator, and costs the archive case
            # nothing. Equal size is not proof -- the reader still hashes -- but it is the only
            # cheap evidence available before the bytes are read.
            if not (stat.S_ISREG(existing.st_mode) and existing.st_size == len(data)):
                return False
            # Adoption leaves a timestamp. Accepting an existing name is the one write-path event
            # age-based collection cannot otherwise see: a chunk orphaned by a crashed activation
            # and re-derived by a resumed one is referenced from *now* with an mtime from days
            # ago, indistinguishable from the garbage ``monoid gc --min-age-s`` deletes.
            # Refreshing the times (never the bytes) on the accepted name is what makes that age
            # gate a protocol about recent use rather than a guess about fresh writes.
            # Deliberately unlike the conformance runner's ``_publish_content_addressed``, whose
            # exists-hit reuse is pinned mtime-stable (``tests/conformance/
            # test_runner_publication.py``): no collector sweeps the evidence directory, so
            # stability is the useful property there. On a multiply-linked chunk the shared
            # inode's times move too, which an incremental archiver may answer with one redundant
            # re-copy -- that costs a copy, not correctness. By name, not by
            # ``follow_symlinks=False`` or a descriptor: Windows CPython supports neither for
            # ``utime`` (measured -- ``os.utime`` is in neither ``os.supports_follow_symlinks``
            # nor ``os.supports_fd`` on 3.11), and the ``lstat`` above already proved the name a
            # regular file. What remains is a same-instant swap redirecting a *time* touch -- no
            # bytes move -- a strictly weaker residual than the unlink-after-glob the recorder's
            # own temp sweep documents and accepts. Best-effort, because the chunk IS stored and
            # ``True`` is the honest answer whether or not the touch landed.
            try:
                os.utime(path)
            except OSError:
                pass
            return True
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f"{path.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp")
        flags = (
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(temp, flags, 0o666)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
        except OSError:
            try:
                os.unlink(temp)
            except OSError:
                pass
            return False
        try:
            os.replace(temp, path)
        except OSError:
            try:
                os.unlink(temp)
            except OSError:
                pass
            return False
        return True
    except OSError:
        return False
