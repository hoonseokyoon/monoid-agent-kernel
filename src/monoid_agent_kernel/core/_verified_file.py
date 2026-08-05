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
