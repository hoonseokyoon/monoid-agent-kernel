from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath

from monoid_agent_kernel.errors import WorkspaceError

_WINDOWS_INVALID_CHARS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "CONIN$",
        "CONOUT$",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
        *(f"COM{index}" for index in "¹²³"),
        *(f"LPT{index}" for index in "¹²³"),
    }
)
_WINDOWS_SHORT_NAME = re.compile(r"^[^ .~]{1,6}~[0-9]+(?:\.[^ .]{1,3})?$", re.IGNORECASE)


def _validate_workspace_segment(part: str, raw: str | None) -> None:
    # C0 controls make JSONL/event boundaries ambiguous, and regex ``$`` treats a final newline as
    # an alternate end position. Reject them on every platform so an allow-list cannot authorize a
    # different POSIX filename by appending a newline. DEL follows the same portable-path rule.
    if any(ord(char) < 32 or ord(char) == 127 for char in part):
        raise WorkspaceError(f"control characters are not allowed in workspace paths: {raw!r}")
    if os.name != "nt":
        return
    if any(char in _WINDOWS_INVALID_CHARS for char in part):
        raise WorkspaceError(f"invalid Windows workspace path: {raw!r}")
    if part.endswith((".", " ")):
        raise WorkspaceError(f"ambiguous Windows workspace path: {raw!r}")
    device_stem = part.split(".", 1)[0].rstrip(" .").upper()
    if device_stem in _WINDOWS_RESERVED_NAMES:
        raise WorkspaceError(f"reserved Windows device path: {raw!r}")
    # Existing 8.3 aliases are another spelling for a long filename. The lexical policy matcher
    # cannot know that identity without a workspace handle, so reject the alias-shaped spelling
    # before any read/write or public preview. This may exclude a rare literal long filename, but
    # keeps deny and allow scopes single-valued on Windows.
    if _WINDOWS_SHORT_NAME.fullmatch(part):
        raise WorkspaceError(f"Windows short-name aliases are not allowed: {raw!r}")


def normalize_workspace_path(raw: str | None) -> str:
    value = "." if raw is None or raw == "" else raw.replace("\\", "/")
    if value.startswith("/") or (len(value) >= 2 and value[1] == ":"):
        raise WorkspaceError(f"absolute paths are not allowed: {raw!r}")
    pure = PurePosixPath(value)
    parts: list[str] = []
    for part in pure.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            raise WorkspaceError(f"parent traversal is not allowed: {raw!r}")
        _validate_workspace_segment(part, raw)
        parts.append(part)
    return "." if not parts else "/".join(parts)


def is_within(root: Path, candidate: Path) -> bool:
    try:
        os.path.commonpath([str(root), str(candidate)])
    except ValueError:
        return False
    return os.path.commonpath([str(root), str(candidate)]) == str(root)

