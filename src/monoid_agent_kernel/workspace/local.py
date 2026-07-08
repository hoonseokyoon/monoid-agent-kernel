from __future__ import annotations

import difflib
import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from monoid_agent_kernel.core._util import sha256_bytes, utc_timestamp
from monoid_agent_kernel.core.spec import AgentRunSpec, RunMode, WorkspaceBackendKind
from monoid_agent_kernel.core.workspace import ChangedEntry, FileEntry, Workspace
from monoid_agent_kernel.errors import WorkspaceError
from monoid_agent_kernel.identifiers import namespaced_id
from monoid_agent_kernel.workspace.paths import is_within, normalize_workspace_path

# Re-exported from core so existing ``workspace.local`` import sites keep working.
__all__ = [
    "ChangedEntry",
    "FileEntry",
    "LocalWorkspaceBackend",
    "default_local_workspace_factory",
    "sha256_bytes",
]

WORKSPACE_BASE_SCHEMA_VERSION = namespaced_id("workspace-base.v1")


def _glob_matches(path: str, pattern: str) -> bool:
    if pattern in {"**", "**/*"}:
        return True
    return fnmatch.fnmatch(path, pattern)


@dataclass(frozen=True)
class _TreeSnapshot:
    root: str
    kind: str
    files: dict[str, bytes]
    dirs: tuple[str, ...]
    total_bytes: int


@dataclass
class LocalWorkspaceBackend:
    root: Path
    mode: RunMode = "propose"
    max_bytes_read: int = 1_000_000
    backend_kind: WorkspaceBackendKind = "overlay"
    _overlay: dict[str, bytes] = field(default_factory=dict, init=False, repr=False)
    _overlay_dirs: set[str] = field(default_factory=set, init=False, repr=False)
    _created_dirs: set[str] = field(default_factory=set, init=False, repr=False)
    _deleted_files: set[str] = field(default_factory=set, init=False, repr=False)
    _deleted_dirs: set[str] = field(default_factory=set, init=False, repr=False)
    _originals: dict[str, bytes | None] = field(default_factory=dict, init=False, repr=False)
    _base_files: dict[str, bytes] = field(default_factory=dict, init=False, repr=False)
    _base_dirs: set[str] = field(default_factory=set, init=False, repr=False)
    _base_entries: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _base_excluded: list[dict[str, str]] = field(default_factory=list, init=False, repr=False)
    _base_excluded_paths: set[str] = field(default_factory=set, init=False, repr=False)
    _base_excluded_prefixes: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = self.root.resolve()
        if not self.root.exists():
            raise WorkspaceError(f"workspace root does not exist: {self.root}")
        if not self.root.is_dir():
            raise WorkspaceError(f"workspace root is not a directory: {self.root}")
        if self.backend_kind not in {"overlay", "staging"}:
            raise WorkspaceError(f"unsupported workspace backend: {self.backend_kind}")
        self._capture_base_snapshot()

    def normalize(self, path: str | None) -> str:
        return normalize_workspace_path(path)

    def resolve_existing_or_parent(self, path: str | None, *, for_write: bool = False) -> tuple[str, Path]:
        rel = self.normalize(path)
        candidate = self.root / Path(rel)
        if candidate.exists():
            resolved = candidate.resolve()
        else:
            parent = candidate.parent
            while not parent.exists() and parent != self.root.parent:
                parent = parent.parent
            if not parent.exists():
                raise WorkspaceError(f"no existing parent for path: {rel}")
            resolved_parent = parent.resolve()
            if not is_within(self.root, resolved_parent):
                raise WorkspaceError(f"path escapes workspace through parent symlink: {rel}")
            resolved = resolved_parent / candidate.relative_to(parent)
        if not is_within(self.root, resolved):
            raise WorkspaceError(f"path escapes workspace: {rel}")
        if for_write and rel == ".":
            raise WorkspaceError("cannot write workspace root")
        return rel, resolved

    def exists(self, path: str | None) -> bool:
        rel = self.normalize(path)
        return self._effective_kind(rel, self.root / rel) is not None

    def read_bytes(self, path: str | None, *, max_bytes: int | None = None) -> tuple[bytes, str]:
        rel, abs_path = self.resolve_existing_or_parent(path)
        if rel in self._overlay:
            data = self._overlay[rel]
        else:
            if self._is_effectively_deleted(rel):
                raise WorkspaceError(f"file does not exist: {rel}")
            if not abs_path.exists() or not abs_path.is_file():
                raise WorkspaceError(f"file does not exist: {rel}")
            limit = max_bytes or self.max_bytes_read
            size = abs_path.stat().st_size
            if size > limit:
                raise WorkspaceError(f"file exceeds max read size: {rel} ({size} bytes > {limit} bytes)")
            data = abs_path.read_bytes()
        limit = max_bytes or self.max_bytes_read
        if len(data) > limit:
            raise WorkspaceError(f"file exceeds max read size: {rel} ({len(data)} bytes > {limit} bytes)")
        return data, sha256_bytes(data)

    def write_bytes(
        self,
        path: str | None,
        data: bytes,
        *,
        create_dirs: bool = False,
        expected_sha256: str | None = None,
        overwrite: bool = True,
    ) -> str:
        if self.mode == "read-only":
            raise WorkspaceError("workspace is read-only")
        self._reject_symlink_components(self.normalize(path))
        rel, abs_path = self.resolve_existing_or_parent(path, for_write=True)
        current = self._read_optional_bytes(rel, abs_path)
        if expected_sha256 is not None and sha256_bytes(current or b"") != expected_sha256:
            raise WorkspaceError(f"expected_sha256 mismatch for {rel}")
        self._write_file(rel, data, create_dirs=create_dirs, overwrite=overwrite)
        return sha256_bytes(data)

    def mkdir(self, path: str | None) -> str:
        if self.mode == "read-only":
            raise WorkspaceError("workspace is read-only")
        self._reject_symlink_components(self.normalize(path))
        rel, abs_path = self.resolve_existing_or_parent(path, for_write=True)
        kind = self._effective_kind(rel, abs_path)
        if kind == "file":
            raise WorkspaceError(f"path is a file: {rel}")
        self._record_original(rel, None)
        self._clear_delete_marker_for(rel)
        if self._uses_overlay():
            self._overlay_dirs.add(rel)
            self._ensure_overlay_parent_dirs(rel)
        else:
            abs_path.mkdir(parents=True, exist_ok=True)
        if kind is None:
            self._created_dirs.add(rel)
        return rel

    def copy_path(
        self,
        source_path: str | None,
        destination_path: str | None,
        *,
        overwrite: bool = False,
        create_dirs: bool = False,
        recursive: bool = False,
        max_entries: int = 1000,
        max_bytes: int = 50_000_000,
        directory_mode: str = "merge",
    ) -> dict[str, int | str]:
        if self.mode == "read-only":
            raise WorkspaceError("workspace is read-only")
        snapshot = self._collect_tree(source_path, recursive=recursive, max_entries=max_entries, max_bytes=max_bytes)
        self._reject_symlink_components(self.normalize(destination_path))
        dest_rel, _dest_abs = self.resolve_existing_or_parent(destination_path, for_write=True)
        if snapshot.root == ".":
            raise WorkspaceError("cannot copy workspace root")
        if dest_rel == snapshot.root or self._is_descendant(dest_rel, snapshot.root):
            raise WorkspaceError("destination cannot be the source path or inside the source tree")
        if snapshot.kind == "file":
            data = snapshot.files[snapshot.root]
            self._write_file(dest_rel, data, create_dirs=create_dirs, overwrite=overwrite)
            return {"path": dest_rel, "files": 1, "dirs": 0, "bytes": len(data)}

        self._validate_directory_mode(directory_mode)
        if directory_mode == "replace" and overwrite and self._effective_kind(dest_rel, self.root / dest_rel) == "dir":
            self.delete_path(dest_rel, recursive=True, max_entries=max_entries, max_bytes=max_bytes)
        self._preflight_tree_destination(snapshot, dest_rel, overwrite=overwrite, create_dirs=create_dirs)
        dirs_written = 0
        for source_dir in snapshot.dirs:
            target_dir = self._mapped_tree_path(snapshot.root, dest_rel, source_dir)
            self._mkdir_for_tree(target_dir)
            dirs_written += 1
        for source_file, data in sorted(snapshot.files.items()):
            target_file = self._mapped_tree_path(snapshot.root, dest_rel, source_file)
            self._write_file(target_file, data, create_dirs=True, overwrite=True)
        return {
            "path": dest_rel,
            "files": len(snapshot.files),
            "dirs": dirs_written,
            "bytes": snapshot.total_bytes,
        }

    def move_path(
        self,
        source_path: str | None,
        destination_path: str | None,
        *,
        overwrite: bool = False,
        create_dirs: bool = False,
        recursive: bool = False,
        max_entries: int = 1000,
        max_bytes: int = 50_000_000,
        directory_mode: str = "merge",
    ) -> dict[str, int | str]:
        if self.mode == "read-only":
            raise WorkspaceError("workspace is read-only")
        snapshot = self._collect_tree(source_path, recursive=recursive, max_entries=max_entries, max_bytes=max_bytes)
        self._reject_symlink_components(self.normalize(destination_path))
        dest_rel, _dest_abs = self.resolve_existing_or_parent(destination_path, for_write=True)
        if snapshot.root == ".":
            raise WorkspaceError("cannot move workspace root")
        if dest_rel == snapshot.root or self._is_descendant(dest_rel, snapshot.root):
            raise WorkspaceError("destination cannot be the source path or inside the source tree")
        if snapshot.kind == "dir":
            self._validate_directory_mode(directory_mode)
            if not (directory_mode == "replace" and overwrite):
                self._preflight_tree_destination(snapshot, dest_rel, overwrite=overwrite, create_dirs=create_dirs)
        else:
            self._preflight_file_destination(dest_rel, create_dirs=create_dirs, overwrite=overwrite)

        copied = self.copy_path(
            source_path,
            destination_path,
            overwrite=overwrite,
            create_dirs=create_dirs,
            recursive=recursive,
            max_entries=max_entries,
            max_bytes=max_bytes,
            directory_mode=directory_mode,
        )
        self._delete_snapshot(snapshot)
        return copied

    def delete_path(
        self,
        path: str | None,
        *,
        recursive: bool = False,
        max_entries: int = 1000,
        max_bytes: int = 50_000_000,
    ) -> dict[str, int | str]:
        if self.mode == "read-only":
            raise WorkspaceError("workspace is read-only")
        snapshot = self._collect_tree(path, recursive=recursive, max_entries=max_entries, max_bytes=max_bytes)
        if snapshot.root == ".":
            raise WorkspaceError("cannot delete workspace root")
        self._delete_snapshot(snapshot)
        return {
            "path": snapshot.root,
            "files": len(snapshot.files),
            "dirs": len(snapshot.dirs) if snapshot.kind == "dir" else 0,
            "bytes": snapshot.total_bytes,
        }

    def list_entries(self, path: str | None = ".", *, recursive: bool = False, max_entries: int = 200) -> list[FileEntry]:
        rel, abs_path = self.resolve_existing_or_parent(path)
        entries: dict[str, FileEntry] = {}
        if self._effective_kind(rel, abs_path) == "dir" and abs_path.exists() and abs_path.is_dir() and not self._is_effectively_deleted(rel):
            iterator = abs_path.rglob("*") if recursive else abs_path.iterdir()
            for item in sorted(iterator, key=lambda child: child.as_posix()):
                if len(entries) >= max_entries:
                    break
                item_resolved = item.resolve()
                if not is_within(self.root, item_resolved):
                    continue
                item_rel = self._to_rel(item)
                if self._disk_hidden(item_rel):
                    continue
                if item.is_symlink():
                    entries[item_rel] = FileEntry(path=item_rel, kind="symlink", size=0)
                    continue
                entries[item_rel] = FileEntry(
                    path=item_rel,
                    kind="dir" if item.is_dir() else "file" if item.is_file() else "other",
                    size=item.stat().st_size if item.is_file() else 0,
                )
        prefix = "" if rel == "." else rel.rstrip("/") + "/"
        for item_rel in sorted(self._overlay_dirs):
            if item_rel == "." or item_rel == rel or self._is_effectively_deleted(item_rel):
                continue
            if prefix and not item_rel.startswith(prefix):
                continue
            self._add_entry(entries, prefix, item_rel, "dir", 0, recursive)
            if len(entries) >= max_entries:
                break
        for item_rel, data in sorted(self._overlay.items()):
            if item_rel == rel or self._is_effectively_deleted(item_rel):
                continue
            if prefix and not item_rel.startswith(prefix):
                continue
            self._add_entry(entries, prefix, item_rel, "file", len(data), recursive)
            if len(entries) >= max_entries:
                break
        return sorted(entries.values(), key=lambda entry: entry.path)[:max_entries]

    def glob(self, pattern: str, *, root: str | None = ".", max_matches: int = 200) -> list[str]:
        root_rel, root_abs = self.resolve_existing_or_parent(root)
        matches: set[str] = set()
        base_pattern = pattern.replace("\\", "/")
        if self._effective_kind(root_rel, root_abs) == "dir" and root_abs.exists():
            for item in sorted(root_abs.rglob("*"), key=lambda child: child.as_posix()):
                if item.is_symlink() and not is_within(self.root, item.resolve()):
                    continue
                item_rel = self._to_rel(item)
                if self._disk_hidden(item_rel):
                    continue
                local_rel = item_rel if root_rel == "." else item_rel.removeprefix(root_rel + "/")
                if _glob_matches(local_rel, base_pattern):
                    matches.add(item_rel)
                if len(matches) >= max_matches:
                    break
        for item_rel in sorted(set(self._overlay) | self._overlay_dirs):
            if item_rel == "." or self._is_effectively_deleted(item_rel):
                continue
            local_rel = item_rel if root_rel == "." else item_rel.removeprefix(root_rel + "/")
            if (root_rel == "." or item_rel.startswith(root_rel + "/")) and _glob_matches(local_rel, base_pattern):
                matches.add(item_rel)
            if len(matches) >= max_matches:
                break
        return sorted(matches)[:max_matches]

    def text_files(self, root: str | None = ".", *, file_glob: str | None = None, max_files: int = 500) -> Iterable[str]:
        pattern = file_glob or "**/*"
        count = 0
        for rel in self.glob(pattern, root=root, max_matches=max_files * 2):
            if count >= max_files:
                break
            if rel in self._overlay:
                count += 1
                yield rel
                continue
            path = self.root / rel
            if not self._is_effectively_deleted(rel) and not path.is_symlink() and path.is_file():
                count += 1
                yield rel

    def stat_path(self, path: str | None) -> dict[str, Any]:
        rel, abs_path = self.resolve_existing_or_parent(path)
        kind = self._effective_kind(rel, abs_path)
        if kind is None:
            return {"path": rel, "exists": False}
        if kind == "file":
            if rel in self._overlay:
                size = len(self._overlay[rel])
                return {"path": rel, "exists": True, "kind": "file", "size": size}
            stat = abs_path.stat()
            return {
                "path": rel,
                "exists": True,
                "kind": "file",
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            }
        if kind == "symlink":
            return {"path": rel, "exists": True, "kind": "symlink", "size": 0}
        return {"path": rel, "exists": True, "kind": "dir", "size": 0}

    def diff_patch(self) -> str:
        if self.backend_kind == "staging":
            return self._staging_diff_patch()
        changed = sorted(self._originals)
        parts: list[str] = []
        for rel in changed:
            before = self._originals.get(rel)
            after = self._proposed_bytes(rel)
            if before == after:
                continue
            if before is None and after is None:
                continue
            parts.extend(self._unified_diff(rel, before, after))
        return "".join(parts)

    def changed_paths(self) -> list[str]:
        return [entry.path for entry in self.changed_entries()]

    def changed_entries(self) -> list[ChangedEntry]:
        if self.backend_kind == "staging":
            return self._staging_changed_entries()
        entries: list[ChangedEntry] = []
        for rel in sorted(self._originals):
            before = self._originals.get(rel)
            if rel in self._deleted_files:
                if before is None:
                    continue
                entries.append(
                    ChangedEntry(
                        path=rel,
                        kind="missing",
                        base_sha256=sha256_bytes(before),
                        proposed_sha256=None,
                        change_kind="deleted",
                    )
                )
                continue
            if rel in self._deleted_dirs:
                entries.append(
                    ChangedEntry(
                        path=rel,
                        kind="dir",
                        base_sha256=None,
                        proposed_sha256=None,
                        change_kind="deleted",
                    )
                )
                continue
            if rel in self._overlay:
                data = self._overlay[rel]
                if before == data:
                    continue
                proposed_sha = sha256_bytes(data)
                entries.append(
                    ChangedEntry(
                        path=rel,
                        kind="file",
                        size=len(data),
                        sha256=proposed_sha,
                        content=data,
                        base_sha256=sha256_bytes(before) if before is not None else None,
                        proposed_sha256=proposed_sha,
                        change_kind="created" if before is None else "modified",
                    )
                )
                continue
            if rel in self._overlay_dirs:
                if before is None and not (self.root / rel).exists():
                    entries.append(
                        ChangedEntry(
                            path=rel,
                            kind="dir",
                            base_sha256=None,
                            proposed_sha256=None,
                            change_kind="directory",
                        )
                    )
                continue

            path = self.root / rel
            if path.exists() and path.is_file():
                data = path.read_bytes()
                if before == data:
                    continue
                proposed_sha = sha256_bytes(data)
                entries.append(
                    ChangedEntry(
                        path=rel,
                        kind="file",
                        size=len(data),
                        sha256=proposed_sha,
                        content=data,
                        base_sha256=sha256_bytes(before) if before is not None else None,
                        proposed_sha256=proposed_sha,
                        change_kind="created" if before is None else "modified",
                    )
                )
            elif path.exists() and path.is_dir() and before is None:
                entries.append(
                    ChangedEntry(
                        path=rel,
                        kind="dir",
                        base_sha256=None,
                        proposed_sha256=None,
                        change_kind="directory",
                    )
                )
            elif before is not None:
                entries.append(
                    ChangedEntry(
                        path=rel,
                        kind="missing",
                        base_sha256=sha256_bytes(before),
                        proposed_sha256=None,
                        change_kind="deleted",
                    )
                )
        return entries

    def snapshot_current_as_new_baseline(self) -> None:
        if self.mode == "read-only":
            raise WorkspaceError("cannot re-baseline a read-only workspace")
        if self.backend_kind == "staging":
            # Staging writes land on disk, so the current disk tree is the new base.
            files, dirs = self._scan_staging_current()
            self._base_files = dict(files)
            self._base_dirs = set(dirs)
        else:
            # Overlay reads resolve against _overlay/_deleted_* (not _base_files),
            # so we fold the proposed state into _base_files for base.json accuracy
            # while deliberately keeping the overlay/delete markers live.
            proposed_files = dict(self._base_files)
            for rel in self._deleted_files:
                proposed_files.pop(rel, None)
            proposed_files.update(self._overlay)
            self._base_files = proposed_files
            self._base_dirs = (set(self._base_dirs) | set(self._overlay_dirs)) - set(self._deleted_dirs)
        # changed_entries()/diff_patch() iterate _originals; clearing it resets the
        # reported delta to empty. Later writes re-record originals from the current
        # (re-baselined) proposed state, so only post-commit changes are reported.
        self._originals.clear()
        self._rebuild_base_entries()

    def _rebuild_base_entries(self) -> None:
        entries: list[dict[str, Any]] = []
        for rel in sorted(self._base_dirs):
            entries.append({"path": rel, "kind": "dir", "size": 0, "sha256": None})
        for rel in sorted(self._base_files):
            data = self._base_files[rel]
            entries.append(
                {"path": rel, "kind": "file", "size": len(data), "sha256": sha256_bytes(data)}
            )
        self._base_entries = entries

    def workspace_base_payload(self, run_id: str) -> dict[str, Any]:
        return {
            "schema_version": WORKSPACE_BASE_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": utc_timestamp(),
            "workspace_root": str(self.root),
            "workspace_backend": self.backend_kind,
            "entries": list(self._base_entries),
            "excluded": list(self._base_excluded),
        }

    def _capture_base_snapshot(self) -> None:
        self._base_files.clear()
        self._base_dirs.clear()
        self._base_entries.clear()
        self._base_excluded.clear()
        self._base_excluded_paths.clear()
        self._base_excluded_prefixes.clear()
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            dirnames.sort()
            filenames.sort()
            current = Path(dirpath)
            safe_dirnames: list[str] = []
            for dirname in dirnames:
                path = current / dirname
                rel = self._relative_path(path)
                if rel is None:
                    self._base_excluded.append({"path": str(path), "reason": "path_escape"})
                    continue
                if path.is_symlink() and not is_within(self.root, path.resolve()):
                    self._record_base_excluded(rel, is_dir=True, reason="symlink_escape")
                    continue
                self._base_dirs.add(rel)
                self._base_entries.append(
                    {"path": rel, "kind": "dir", "size": 0, "sha256": None}
                )
                safe_dirnames.append(dirname)
            dirnames[:] = safe_dirnames

            for filename in filenames:
                path = current / filename
                rel = self._relative_path(path)
                if rel is None:
                    self._base_excluded.append({"path": str(path), "reason": "path_escape"})
                    continue
                if path.is_symlink() and not is_within(self.root, path.resolve()):
                    self._record_base_excluded(rel, is_dir=False, reason="symlink_escape")
                    continue
                if not path.is_file():
                    self._base_entries.append(
                        {"path": rel, "kind": "other", "size": 0, "sha256": None}
                    )
                    continue
                data = path.read_bytes()
                self._base_files[rel] = data
                self._base_entries.append(
                    {
                        "path": rel,
                        "kind": "file",
                        "size": len(data),
                        "sha256": sha256_bytes(data),
                    }
                )

    def _staging_changed_entries(self) -> list[ChangedEntry]:
        current_files, current_dirs = self._scan_staging_current()
        entries: list[ChangedEntry] = []
        all_file_paths = sorted(set(self._base_files) | set(current_files))
        for rel in all_file_paths:
            before = self._base_files.get(rel)
            after = current_files.get(rel)
            if before == after:
                continue
            if after is None:
                entries.append(
                    ChangedEntry(
                        path=rel,
                        kind="missing",
                        base_sha256=sha256_bytes(before or b""),
                        proposed_sha256=None,
                        change_kind="deleted",
                    )
                )
                continue
            proposed_sha = sha256_bytes(after)
            entries.append(
                ChangedEntry(
                    path=rel,
                    kind="file",
                    size=len(after),
                    sha256=proposed_sha,
                    content=after,
                    base_sha256=sha256_bytes(before) if before is not None else None,
                    proposed_sha256=proposed_sha,
                    change_kind="created" if before is None else "modified",
                )
            )

        for rel in sorted(current_dirs - self._base_dirs):
            has_file_descendant = any(path.startswith(rel.rstrip("/") + "/") for path in current_files)
            if not has_file_descendant:
                entries.append(
                    ChangedEntry(
                        path=rel,
                        kind="dir",
                        base_sha256=None,
                        proposed_sha256=None,
                        change_kind="directory",
                    )
                )
        for rel in sorted(self._base_dirs - current_dirs, key=lambda item: (item.count("/"), item), reverse=True):
            entries.append(
                ChangedEntry(
                    path=rel,
                    kind="dir",
                    base_sha256=None,
                    proposed_sha256=None,
                    change_kind="deleted",
                )
            )
        return sorted(entries, key=lambda entry: entry.path)

    def _staging_diff_patch(self) -> str:
        current_files, _current_dirs = self._scan_staging_current()
        parts: list[str] = []
        for rel in sorted(set(self._base_files) | set(current_files)):
            before = self._base_files.get(rel)
            after = current_files.get(rel)
            if before == after:
                continue
            parts.extend(self._unified_diff(rel, before, after))
        return "".join(parts)

    def _scan_staging_current(self) -> tuple[dict[str, bytes], set[str]]:
        files: dict[str, bytes] = {}
        dirs: set[str] = set()
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            dirnames.sort()
            filenames.sort()
            current = Path(dirpath)
            safe_dirnames: list[str] = []
            for dirname in dirnames:
                path = current / dirname
                rel = self._relative_path(path)
                if rel is None:
                    raise WorkspaceError(f"path escapes workspace: {path}")
                if self._is_base_excluded(rel):
                    continue
                if path.is_symlink() and not is_within(self.root, path.resolve()):
                    raise WorkspaceError(f"path escapes workspace through symlink: {rel}")
                dirs.add(rel)
                safe_dirnames.append(dirname)
            dirnames[:] = safe_dirnames

            for filename in filenames:
                path = current / filename
                rel = self._relative_path(path)
                if rel is None:
                    raise WorkspaceError(f"path escapes workspace: {path}")
                if self._is_base_excluded(rel):
                    continue
                if path.is_symlink() and not is_within(self.root, path.resolve()):
                    raise WorkspaceError(f"path escapes workspace through symlink: {rel}")
                if path.is_file():
                    files[rel] = path.read_bytes()
        return files, dirs

    def _record_base_excluded(self, rel: str, *, is_dir: bool, reason: str) -> None:
        self._base_excluded_paths.add(rel)
        if is_dir:
            self._base_excluded_prefixes.add(rel)
        self._base_excluded.append({"path": rel, "reason": reason})

    def _is_base_excluded(self, rel: str) -> bool:
        return rel in self._base_excluded_paths or any(self._is_descendant(rel, prefix) for prefix in self._base_excluded_prefixes)

    def _relative_path(self, path: Path) -> str | None:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path.absolute()
        if not is_within(self.root, resolved):
            return None
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return None

    def _uses_overlay(self) -> bool:
        return self.backend_kind == "overlay" and self.mode == "propose"

    def _write_file(self, rel: str, data: bytes, *, create_dirs: bool, overwrite: bool) -> None:
        abs_path = self.root / rel
        kind = self._effective_kind(rel, abs_path)
        if kind == "dir":
            raise WorkspaceError(f"path is a directory: {rel}")
        current = self._read_optional_bytes(rel, abs_path)
        if current is not None and not overwrite:
            raise WorkspaceError(f"destination already exists: {rel}")
        if current is None and not create_dirs and not self._parent_exists(rel):
            raise WorkspaceError(f"parent directory does not exist: {rel}")
        self._record_original(rel, current)
        self._clear_delete_marker_for(rel)
        if self._uses_overlay():
            self._overlay[rel] = data
            self._ensure_overlay_parent_dirs(rel)
        else:
            abs_path.parent.mkdir(parents=create_dirs, exist_ok=True)
            abs_path.write_bytes(data)

    def _mkdir_for_tree(self, rel: str) -> None:
        kind = self._effective_kind(rel, self.root / rel)
        self._record_original(rel, None)
        self._clear_delete_marker_for(rel)
        if self._uses_overlay():
            self._overlay_dirs.add(rel)
            self._ensure_overlay_parent_dirs(rel)
        else:
            (self.root / rel).mkdir(parents=True, exist_ok=True)
        if kind is None:
            self._created_dirs.add(rel)

    def _delete_snapshot(self, snapshot: _TreeSnapshot) -> None:
        for rel, data in sorted(snapshot.files.items(), reverse=True):
            self._delete_file(rel, data)
        for rel in sorted(snapshot.dirs, key=lambda item: item.count("/"), reverse=True):
            self._delete_dir(rel)

    def _delete_file(self, rel: str, current: bytes) -> None:
        created_in_run = self._originals.get(rel) is None and rel in self._originals
        if created_in_run:
            if not self._uses_overlay():
                path = self.root / rel
                if path.exists() and path.is_file():
                    path.unlink()
            self._overlay.pop(rel, None)
            self._deleted_files.discard(rel)
            self._originals.pop(rel, None)
            return
        self._record_original(rel, current)
        self._overlay.pop(rel, None)
        self._deleted_files.add(rel)
        if not self._uses_overlay():
            path = self.root / rel
            if path.exists() and path.is_file():
                path.unlink()

    def _delete_dir(self, rel: str) -> None:
        if rel == ".":
            raise WorkspaceError("cannot delete workspace root")
        created_in_run = rel in self._created_dirs
        if created_in_run:
            if not self._uses_overlay():
                path = self.root / rel
                if path.exists() and path.is_dir():
                    path.rmdir()
            self._overlay_dirs.discard(rel)
            self._created_dirs.discard(rel)
            self._deleted_dirs.discard(rel)
            self._originals.pop(rel, None)
            return
        self._record_original(rel, None)
        for item in list(self._overlay):
            if self._is_descendant(item, rel):
                self._overlay.pop(item, None)
        for item in list(self._overlay_dirs):
            if item == rel or self._is_descendant(item, rel):
                self._overlay_dirs.discard(item)
        self._deleted_dirs.add(rel)
        if not self._uses_overlay():
            path = self.root / rel
            if path.exists() and path.is_dir():
                path.rmdir()

    def _collect_tree(
        self,
        path: str | None,
        *,
        recursive: bool,
        max_entries: int,
        max_bytes: int,
    ) -> _TreeSnapshot:
        raw_rel = self.normalize(path)
        self._reject_symlink_components(raw_rel)
        if raw_rel != "." and self._effective_kind(raw_rel, self.root / raw_rel) == "symlink":
            raise WorkspaceError(f"symlink file operations are not supported: {raw_rel}")
        rel, abs_path = self.resolve_existing_or_parent(path)
        if rel == ".":
            raise WorkspaceError("workspace root is not a valid file operation target")
        kind = self._effective_kind(rel, abs_path)
        if kind is None:
            raise WorkspaceError(f"path does not exist: {rel}")
        if kind == "symlink":
            raise WorkspaceError(f"symlink file operations are not supported: {rel}")
        if kind == "file":
            data, _digest = self.read_bytes(rel, max_bytes=max_bytes)
            return _TreeSnapshot(root=rel, kind="file", files={rel: data}, dirs=(), total_bytes=len(data))

        descendants = self._tree_descendants(rel, abs_path, recursive=recursive)
        if descendants and not recursive:
            raise WorkspaceError(f"directory requires recursive=true: {rel}")
        files: dict[str, bytes] = {}
        dirs: set[str] = {rel}
        total_bytes = 0
        for child_rel, child_kind in descendants:
            if child_kind == "dir":
                dirs.add(child_rel)
                continue
            if child_kind == "file":
                data, _digest = self.read_bytes(child_rel, max_bytes=max_bytes)
                total_bytes += len(data)
                if total_bytes > max_bytes:
                    raise WorkspaceError(f"operation exceeds max bytes: {max_bytes}")
                files[child_rel] = data
            elif child_kind == "symlink":
                raise WorkspaceError(f"symlink file operations are not supported: {child_rel}")
        entry_count = len(files) + max(0, len(dirs) - 1)
        if entry_count > max_entries:
            raise WorkspaceError(f"operation exceeds max entries: {max_entries}")
        return _TreeSnapshot(root=rel, kind="dir", files=files, dirs=tuple(sorted(dirs)), total_bytes=total_bytes)

    def _tree_descendants(self, rel: str, abs_path: Path, *, recursive: bool) -> list[tuple[str, str]]:
        descendants: dict[str, str] = {}
        if abs_path.exists() and abs_path.is_dir() and not self._is_effectively_deleted(rel):
            iterator = abs_path.rglob("*") if recursive else abs_path.iterdir()
            for item in sorted(iterator, key=lambda child: child.as_posix()):
                item_resolved = item.resolve()
                if not is_within(self.root, item_resolved):
                    raise WorkspaceError(f"path escapes workspace through symlink: {self._to_rel(item)}")
                item_rel = self._to_rel(item)
                if self._disk_hidden(item_rel):
                    continue
                if item.is_symlink():
                    descendants[item_rel] = "symlink"
                    continue
                descendants[item_rel] = "dir" if item.is_dir() else "file" if item.is_file() else "other"
        prefix = rel.rstrip("/") + "/"
        for item_rel in sorted(self._overlay_dirs):
            if item_rel == rel or not item_rel.startswith(prefix) or self._is_effectively_deleted(item_rel):
                continue
            if recursive or "/" not in item_rel[len(prefix) :]:
                descendants[item_rel] = "dir"
        for item_rel in sorted(self._overlay):
            if item_rel == rel or not item_rel.startswith(prefix) or self._is_effectively_deleted(item_rel):
                continue
            if recursive or "/" not in item_rel[len(prefix) :]:
                descendants[item_rel] = "file"
        return sorted(descendants.items())

    def _reject_symlink_components(self, rel: str) -> None:
        if rel == ".":
            return
        probe = self.root
        for part in Path(rel).parts:
            probe = probe / part
            if probe.is_symlink():
                raise WorkspaceError(f"symlink file operations are not supported: {rel}")
            if not probe.exists():
                return

    def _preflight_file_destination(self, rel: str, *, create_dirs: bool, overwrite: bool) -> None:
        kind = self._effective_kind(rel, self.root / rel)
        if kind == "dir":
            raise WorkspaceError(f"destination is a directory: {rel}")
        if kind == "file" and not overwrite:
            raise WorkspaceError(f"destination already exists: {rel}")
        if kind is None and not create_dirs and not self._parent_exists(rel):
            raise WorkspaceError(f"parent directory does not exist: {rel}")

    def _preflight_tree_destination(
        self,
        snapshot: _TreeSnapshot,
        dest_rel: str,
        *,
        overwrite: bool,
        create_dirs: bool,
    ) -> None:
        dest_kind = self._effective_kind(dest_rel, self.root / dest_rel)
        if dest_kind == "file":
            raise WorkspaceError(f"destination is a file: {dest_rel}")
        if dest_kind is not None and not overwrite:
            raise WorkspaceError(f"destination already exists: {dest_rel}")
        if dest_kind is None and not create_dirs and not self._parent_exists(dest_rel):
            raise WorkspaceError(f"parent directory does not exist: {dest_rel}")
        if not overwrite:
            return
        for source_file in snapshot.files:
            target = self._mapped_tree_path(snapshot.root, dest_rel, source_file)
            if self._effective_kind(target, self.root / target) == "dir":
                raise WorkspaceError(f"destination is a directory: {target}")

    def _record_original(self, rel: str, current: bytes | None) -> None:
        if rel not in self._originals:
            self._originals[rel] = current

    def _read_optional_bytes(self, rel: str, abs_path: Path) -> bytes | None:
        if rel in self._overlay:
            return self._overlay[rel]
        if self._is_effectively_deleted(rel):
            return None
        if abs_path.exists() and abs_path.is_file():
            return abs_path.read_bytes()
        return None

    def _read_disk_optional(self, rel: str) -> bytes | None:
        path = self.root / rel
        if path.exists() and path.is_file():
            return path.read_bytes()
        return None

    def _proposed_bytes(self, rel: str) -> bytes | None:
        if rel in self._deleted_files or rel in self._deleted_dirs:
            return None
        if rel in self._overlay:
            return self._overlay[rel]
        return self._read_disk_optional(rel)

    def path_kind(self, path: str | None) -> str | None:
        """Effective kind of a workspace path: ``"file"``, ``"dir"``, or ``None``.

        Accounts for staged overlay/deletes, so it reflects the proposed state
        rather than only what is on disk.
        """
        rel, abs_path = self.resolve_existing_or_parent(path)
        return self._effective_kind(rel, abs_path)

    def _effective_kind(self, rel: str, abs_path: Path) -> str | None:
        if rel in self._overlay:
            return "file"
        if rel in self._overlay_dirs:
            return "dir"
        if self._is_effectively_deleted(rel):
            return None
        if abs_path.is_symlink():
            return "symlink"
        if abs_path.exists() and abs_path.is_dir():
            return "dir"
        if abs_path.exists() and abs_path.is_file():
            return "file"
        return None

    def _is_effectively_deleted(self, rel: str) -> bool:
        if rel in self._overlay or rel in self._overlay_dirs:
            return False
        if rel in self._deleted_files or rel in self._deleted_dirs:
            return True
        return any(self._is_descendant(rel, deleted_dir) for deleted_dir in self._deleted_dirs)

    def _disk_hidden(self, rel: str) -> bool:
        return rel in self._overlay or self._is_effectively_deleted(rel)

    def _clear_delete_marker_for(self, rel: str) -> None:
        self._deleted_files.discard(rel)
        self._deleted_dirs.discard(rel)
        parts = rel.split("/")
        for index in range(1, len(parts)):
            ancestor = "/".join(parts[:index])
            self._deleted_dirs.discard(ancestor)

    def _parent_exists(self, rel: str) -> bool:
        parent = self._parent_rel(rel)
        if parent == ".":
            return True
        return self._effective_kind(parent, self.root / parent) == "dir"

    def _ensure_overlay_parent_dirs(self, rel: str) -> None:
        parent = self._parent_rel(rel)
        while parent != ".":
            self._overlay_dirs.add(parent)
            self._deleted_dirs.discard(parent)
            parent = self._parent_rel(parent)

    def _to_rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    @staticmethod
    def _parent_rel(rel: str) -> str:
        parent = Path(rel).parent.as_posix()
        return "." if parent == "." else parent

    @staticmethod
    def _is_descendant(path: str, parent: str) -> bool:
        return path.startswith(parent.rstrip("/") + "/")

    @staticmethod
    def _mapped_tree_path(source_root: str, dest_root: str, source_path: str) -> str:
        if source_path == source_root:
            return dest_root
        suffix = source_path.removeprefix(source_root.rstrip("/") + "/")
        return f"{dest_root.rstrip('/')}/{suffix}"

    @staticmethod
    def _validate_directory_mode(value: str) -> None:
        if value not in {"merge", "replace"}:
            raise WorkspaceError("directory_mode must be 'merge' or 'replace'")

    @staticmethod
    def _add_entry(
        entries: dict[str, FileEntry],
        prefix: str,
        item_rel: str,
        kind: str,
        size: int,
        recursive: bool,
    ) -> None:
        remainder = item_rel[len(prefix) :]
        if not recursive and "/" in remainder:
            first = prefix + remainder.split("/", 1)[0]
            entries.setdefault(first, FileEntry(path=first, kind="dir", size=0))
        else:
            entries[item_rel] = FileEntry(path=item_rel, kind=kind, size=size)

    @staticmethod
    def _decode_for_diff(data: bytes | None) -> list[str]:
        if data is None:
            return []
        try:
            return data.decode("utf-8").splitlines(keepends=True)
        except UnicodeDecodeError:
            digest = sha256_bytes(data)
            return [f"<binary sha256={digest} size={len(data)}>\n"]

    @classmethod
    def _unified_diff(cls, rel: str, before: bytes | None, after: bytes | None) -> list[str]:
        before_lines = cls._decode_for_diff(before)
        after_lines = cls._decode_for_diff(after)
        from_name = f"a/{rel}"
        to_name = f"b/{rel}"
        return list(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=from_name,
                tofile=to_name,
                lineterm="\n",
            )
        )


def default_local_workspace_factory(spec: AgentRunSpec) -> Workspace:
    """Build the default local-filesystem workspace from a run spec.

    This is ``AgentLoop.workspace_factory``'s default; integrators can supply
    their own factory to back the engine with a different workspace.
    """
    return LocalWorkspaceBackend(
        spec.workspace_root,
        mode=spec.mode,
        max_bytes_read=spec.limits.max_bytes_read,
        backend_kind=spec.workspace_backend,
    )
