"""Opening a run-directory artifact without accepting an indirection planted in its place.

A run directory is a privacy boundary, and everything the kernel appends inside it is supposed to
be a file this process created. A *pathname* is not evidence of that. ``open(path, "a")`` follows a
symlink, and a hard link is a second name for an inode that may live anywhere on the volume with
anyone's permissions -- so either one silently turns "append a line to my own artifact" into
"append it to a file someone else chose", written with the agent's credentials. The exposure is
real wherever a run directory can be touched between runs, which includes every reopened durable
run.

The rule these functions enforce is therefore not "the path looks right" but "the bytes I am about
to write go to a single-link regular file, and to the same inode I just verified": ``O_NOFOLLOW``
where the platform has it, ``lstat`` before the open for the platforms that do not, ``S_ISREG`` and
``st_nlink == 1`` on both the named entry and the opened descriptor, and ``os.path.samestat`` to
close the window between them.

These primitives were written for ``model-content.jsonl`` and live here because they were never
about that artifact. ``model_calls.jsonl`` is the second append-only sidecar in the same directory
with exactly the same exposure, and this package's recurring defect is a rule proven on one of two
parallel sites and never bound on its twin. One function, both callers.
"""

from __future__ import annotations

import os
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


def verified_file_is_safe(path: Path, *, allow_missing: bool = True) -> bool:
    """Whether a path is absent or an ordinary single-link file, without following links.

    A missing file is safe for a lazy writer to create unless ``allow_missing`` is false. Existing
    links, directories, FIFOs, devices, and other special files fail closed for readers and writers.
    """

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return allow_missing
    except OSError:
        return False
    # A hard link is also an indirection across the run-directory boundary: appending here mutates
    # the same inode through every other name. These artifacts are process-created single-link
    # files.
    return stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1


def open_verified_regular_fd(
    path: Path,
    flags: int,
    *,
    expected_identity: VerifiedFileIdentity | None = None,
) -> int | None:
    """Open ``path`` without accepting a link/special-file swap before the first I/O."""

    if not verified_file_is_safe(path):
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
            or opened.st_nlink != 1
            or named.st_nlink != 1
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

    ``None`` for every refusal -- a link, a special file, a file larger than ``max_bytes``, an I/O
    error. The caller re-hashes what it gets, which is what makes the *content* trustworthy; this
    function is only responsible for the bytes having come from a file the run directory owns.
    """

    descriptor = open_verified_regular_fd(path, os.O_RDONLY)
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


def write_verified_bytes_once(path: Path, data: bytes) -> bool:
    """Create one write-once content-addressed file, or report that it could not be done.

    The write-once half: if ``path`` already holds a single-link regular file, nothing is written
    and the answer is ``True`` -- content addressing means an existing file with this name is this
    content, and :func:`read_verified_bytes` re-hashes what it reads, so a lie planted under the
    right name is caught at resolution rather than trusted here. A name that exists as anything
    *else* is refused instead: "already written" and "someone put an indirection here" are not the
    same answer, and only a lstat can tell them apart.

    The verified half mirrors :func:`open_verified_append_text` for the *creation* path: the bytes
    land in a uniquely named temporary sibling opened with ``O_CREAT | O_EXCL | O_NOFOLLOW`` --
    exclusive creation cannot be redirected by a planted link, and a link planted at the temporary
    name fails the open outright -- then take the final name via ``os.replace``, which replaces a
    link *itself* rather than writing through it (the same shape the checkpoint store's blob
    writer uses). A crash between the two leaves an orphaned ``*.tmp`` the owner sweeps at open,
    never a half-written file under a content-addressed name that would poison every reader
    trusting the name.

    ``False`` is terminal for the artifact the caller is building, for the reason the append
    opener's ``None`` is: the refusal is a property of the path.
    """

    try:
        try:
            existing: os.stat_result | None = path.lstat()
        except FileNotFoundError:
            existing = None
        if existing is not None:
            # ``path.exists()`` was the defect here, and it was the module docstring's own defect:
            # it follows a link, so a name planted at a predictable content-addressed sha reported
            # "already stored" for a chunk that was never written, and left every reader resolving
            # whoever's bytes the link points at. An existing name is accepted only when it is the
            # kind of file this function creates.
            return stat.S_ISREG(existing.st_mode) and existing.st_nlink == 1
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
