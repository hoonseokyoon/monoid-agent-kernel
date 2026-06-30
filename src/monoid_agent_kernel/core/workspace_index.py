from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core._util import sha256_bytes, utc_timestamp
from monoid_agent_kernel.core.workspace import Workspace
from monoid_agent_kernel.workspace.paths import is_within

WORKSPACE_INDEX_SCHEMA_VERSION = "native-agent-runner.workspace-index.v1"


def build_workspace_index(
    workspace: Workspace,
    *,
    run_id: str,
    max_entries: int = 500,
    max_hash_bytes: int = 200_000,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    root = workspace.root.resolve()
    truncated = False

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        current = Path(dirpath)
        safe_dirnames: list[str] = []
        for dirname in dirnames:
            path = current / dirname
            rel = _relative_path(root, path)
            if rel is None:
                excluded.append({"path": str(path), "reason": "path_escape"})
                continue
            if path.is_symlink() and not is_within(root, path.resolve()):
                excluded.append({"path": rel, "reason": "symlink_escape"})
                continue
            if len(entries) >= max_entries:
                truncated = True
                continue
            entries.append(
                {
                    "path": rel,
                    "kind": "dir",
                    "size": 0,
                    "sha256": None,
                    "hash_status": "not_file",
                }
            )
            safe_dirnames.append(dirname)
        dirnames[:] = safe_dirnames

        for filename in filenames:
            path = current / filename
            rel = _relative_path(root, path)
            if rel is None:
                excluded.append({"path": str(path), "reason": "path_escape"})
                continue
            if path.is_symlink() and not is_within(root, path.resolve()):
                excluded.append({"path": rel, "reason": "symlink_escape"})
                continue
            if len(entries) >= max_entries:
                truncated = True
                continue
            entries.append(_file_index_entry(path, rel, max_hash_bytes=max_hash_bytes))

    return {
        "schema_version": WORKSPACE_INDEX_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": utc_timestamp(),
        "workspace_root": str(root),
        "max_entries": max_entries,
        "max_hash_bytes": max_hash_bytes,
        "truncated": truncated,
        "entries": entries,
        "excluded": excluded,
    }


def _relative_path(root: Path, path: Path) -> str | None:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    if not is_within(root, resolved):
        return None
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _file_index_entry(path: Path, rel: str, *, max_hash_bytes: int) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": rel, "kind": "other", "size": 0, "sha256": None, "hash_status": "error"}
    if not path.is_file():
        return {"path": rel, "kind": "other", "size": 0, "sha256": None, "hash_status": "not_file"}
    if stat.st_size > max_hash_bytes:
        return {
            "path": rel,
            "kind": "file",
            "size": stat.st_size,
            "sha256": None,
            "hash_status": "too_large",
        }
    try:
        data = path.read_bytes()
    except OSError:
        return {
            "path": rel,
            "kind": "file",
            "size": stat.st_size,
            "sha256": None,
            "hash_status": "error",
        }
    return {
        "path": rel,
        "kind": "file",
        "size": len(data),
        "sha256": sha256_bytes(data),
        "hash_status": "hashed",
    }
