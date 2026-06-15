from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from native_agent_runner.errors import WorkspaceError


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
        parts.append(part)
    return "." if not parts else "/".join(parts)


def is_within(root: Path, candidate: Path) -> bool:
    try:
        os.path.commonpath([str(root), str(candidate)])
    except ValueError:
        return False
    return os.path.commonpath([str(root), str(candidate)]) == str(root)

