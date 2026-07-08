"""Provider-backed persistent memory tools.

The kernel treats memory as ordinary tools. This module provides a default
filesystem-backed implementation that exposes a Claude-style ``/memories``
virtual filesystem through separate read/write tool specs.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote

from monoid_agent_kernel.core.agents import RegistryToolRef, ToolBinding
from monoid_agent_kernel.core.context import TurnContext
from monoid_agent_kernel.core.tool_surface import ToolAuthorizationDecision
from monoid_agent_kernel.tools.base import ToolResult, ToolSpec

MEMORY_ROOT = "/memories"
MEMORY_SEARCH_TOOL_ID = "memory.search"
MEMORY_VIEW_TOOL_ID = "memory.view"
MEMORY_CREATE_TOOL_ID = "memory.create"
MEMORY_STR_REPLACE_TOOL_ID = "memory.str_replace"
MEMORY_INSERT_TOOL_ID = "memory.insert"
MEMORY_DELETE_TOOL_ID = "memory.delete"
MEMORY_RENAME_TOOL_ID = "memory.rename"
MEMORY_READ_TOOL_IDS = (
    MEMORY_SEARCH_TOOL_ID,
    MEMORY_VIEW_TOOL_ID,
)
MEMORY_TOOL_IDS = (
    *MEMORY_READ_TOOL_IDS,
    MEMORY_CREATE_TOOL_ID,
    MEMORY_STR_REPLACE_TOOL_ID,
    MEMORY_INSERT_TOOL_ID,
    MEMORY_DELETE_TOOL_ID,
    MEMORY_RENAME_TOOL_ID,
)

_STARTUP_INDEX_PATH = "/memories/MEMORY.md"
_STARTUP_INDEX_MAX_LINES = 200
_STARTUP_INDEX_MAX_BYTES = 25_000
_MAX_VIEW_BYTES = 1_000_000


class MemoryToolError(Exception):
    """Structured failure returned to the model as a failed ``ToolResult``."""

    def __init__(self, message: str, *, code: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class MemoryStore(Protocol):
    """Storage contract behind the memory tools.

    Paths are virtual POSIX paths rooted at ``/memories``. A store may map them to
    local files, a database, object storage, a graph, or any other implementation.
    """

    def view(self, path: str, view_range: tuple[int, int] | None = None) -> dict[str, Any]:
        ...

    def search(
        self,
        query: str,
        namespace: str | None = None,
        *,
        limit: int = 20,
        filters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...

    def create(self, path: str, file_text: str) -> dict[str, Any]:
        ...

    def str_replace(self, path: str, old_str: str, new_str: str = "") -> dict[str, Any]:
        ...

    def insert(self, path: str, insert_line: int, insert_text: str) -> dict[str, Any]:
        ...

    def delete(self, path: str) -> dict[str, Any]:
        ...

    def rename(self, old_path: str, new_path: str) -> dict[str, Any]:
        ...

    def startup_index(
        self,
        path: str = _STARTUP_INDEX_PATH,
        *,
        max_lines: int = _STARTUP_INDEX_MAX_LINES,
        max_bytes: int = _STARTUP_INDEX_MAX_BYTES,
    ) -> str | None:
        ...


@dataclass(frozen=True)
class _Mount:
    virtual: str
    root: Path


@dataclass(frozen=True)
class _ResolvedPath:
    virtual: str
    mount: _Mount
    relative_parts: tuple[str, ...]
    path: Path


class LocalFilesystemMemoryStore:
    """Filesystem reference store for the ``/memories`` virtual path tree."""

    def __init__(
        self,
        base_path: str | Path | None = None,
        *,
        mounts: Mapping[str, str | Path] | None = None,
        root_path: str = MEMORY_ROOT,
        max_view_bytes: int = _MAX_VIEW_BYTES,
    ) -> None:
        if mounts is None:
            if base_path is None:
                raise ValueError("LocalFilesystemMemoryStore requires base_path or mounts")
            mounts = {root_path: base_path}
        if not mounts:
            raise ValueError("LocalFilesystemMemoryStore requires at least one mount")
        self.max_view_bytes = int(max_view_bytes)
        normalized_mounts: list[_Mount] = []
        for virtual, root in mounts.items():
            normalized = _normalize_memory_path(virtual)
            root_path_obj = Path(root).expanduser()
            root_path_obj.mkdir(parents=True, exist_ok=True)
            normalized_mounts.append(_Mount(normalized, root_path_obj.resolve()))
        self._mounts = tuple(sorted(normalized_mounts, key=lambda item: len(item.virtual), reverse=True))

    @property
    def mounts(self) -> tuple[dict[str, str], ...]:
        return tuple({"virtual": mount.virtual, "root": str(mount.root)} for mount in self._mounts)

    def search(
        self,
        query: str,
        namespace: str | None = None,
        *,
        limit: int = 20,
        filters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        needle = str(query or "")
        if not needle:
            raise MemoryToolError("memory search query is required", code="memory_query_required", retryable=True)
        max_matches = max(1, min(int(limit), 100))
        filter_payload = dict(filters or {})
        file_glob = filter_payload.get("file_glob")
        if file_glob is not None:
            file_glob = str(file_glob)
        root_virtual = _namespace_to_path(namespace)
        roots = self._search_roots(root_virtual)
        matches: list[dict[str, Any]] = []
        lowered_needle = needle.lower()
        for resolved in roots:
            for file_virtual, file_path in self._iter_search_files(resolved, file_glob=file_glob):
                if len(matches) >= max_matches:
                    break
                matches.extend(
                    self._search_file(
                        file_virtual,
                        file_path,
                        needle=needle,
                        lowered_needle=lowered_needle,
                        remaining=max_matches - len(matches),
                    )
                )
            if len(matches) >= max_matches:
                break
        return {
            "operation": "search",
            "status": "ok",
            "message": f"Found {len(matches)} memory match(es).",
            "query": needle,
            "namespace": root_virtual,
            "matches": matches,
            "count": len(matches),
        }

    def view(self, path: str, view_range: tuple[int, int] | None = None) -> dict[str, Any]:
        virtual = _normalize_memory_path(path)
        resolved = self._resolve_if_mounted(virtual)
        if resolved is None:
            virtual_view = self._view_virtual_namespace(virtual)
            if virtual_view is not None:
                return virtual_view
            raise MemoryToolError(
                f"Memory path is not mounted: {virtual}",
                code="memory_path_unmounted",
                retryable=True,
            )
        if not resolved.path.exists():
            virtual_view = self._view_virtual_namespace(virtual)
            if virtual_view is not None:
                return virtual_view
            raise MemoryToolError(
                f"The path {virtual} does not exist. Please provide a valid path.",
                code="memory_path_not_found",
                retryable=True,
            )
        if resolved.path.is_dir():
            return self._view_directory(resolved)
        if not resolved.path.is_file():
            raise MemoryToolError(
                f"Unsupported memory path kind: {virtual}",
                code="memory_unsupported_path",
            )
        return self._view_file(resolved, view_range)

    def create(self, path: str, file_text: str) -> dict[str, Any]:
        resolved = self._resolve(path, for_write=True)
        self._reject_root_operation(resolved, "create")
        if resolved.path.exists():
            raise MemoryToolError(
                f"File {resolved.virtual} already exists",
                code="memory_file_exists",
                retryable=True,
            )
        resolved.path.parent.mkdir(parents=True, exist_ok=True)
        data = file_text.encode("utf-8")
        resolved.path.write_bytes(data)
        return {
            "operation": "create",
            "path": resolved.virtual,
            "status": "created",
            "message": f"File created successfully at: {resolved.virtual}",
            "sha256": _sha256(data),
        }

    def str_replace(self, path: str, old_str: str, new_str: str = "") -> dict[str, Any]:
        if old_str == "":
            raise MemoryToolError("old_str must not be empty", code="memory_empty_old_str", retryable=True)
        resolved = self._resolve(path)
        text = self._read_text_file(resolved)
        occurrences = text.count(old_str)
        if occurrences == 0:
            raise MemoryToolError(
                f"No replacement was performed, old_str did not appear verbatim in {resolved.virtual}.",
                code="memory_old_str_not_found",
                retryable=True,
            )
        if occurrences > 1:
            lines = _occurrence_lines(text, old_str)
            raise MemoryToolError(
                "No replacement was performed. Multiple occurrences of old_str "
                f"in lines: {', '.join(str(line) for line in lines)}. Please ensure it is unique.",
                code="memory_ambiguous_replace",
                retryable=True,
            )
        changed = text.replace(old_str, new_str, 1)
        data = changed.encode("utf-8")
        resolved.path.write_bytes(data)
        replacement_index = changed.find(new_str) if new_str else max(0, text.find(old_str) - 1)
        return {
            "operation": "str_replace",
            "path": resolved.virtual,
            "status": "edited",
            "message": "The memory file has been edited.",
            "sha256": _sha256(data),
            "snippet": _snippet_around_index(changed, replacement_index),
        }

    def insert(self, path: str, insert_line: int, insert_text: str) -> dict[str, Any]:
        resolved = self._resolve(path)
        text = self._read_text_file(resolved)
        line_count = len(text.splitlines())
        if insert_line < 0 or insert_line > line_count:
            raise MemoryToolError(
                f"Invalid insert_line parameter: {insert_line}. It should be within "
                f"the range of lines of the file: [0, {line_count}]",
                code="memory_invalid_insert_line",
                retryable=True,
            )
        changed, inserted_at = _insert_after_line(text, insert_line, insert_text)
        data = changed.encode("utf-8")
        resolved.path.write_bytes(data)
        return {
            "operation": "insert",
            "path": resolved.virtual,
            "status": "edited",
            "message": f"The file {resolved.virtual} has been edited.",
            "sha256": _sha256(data),
            "snippet": _line_snippet(changed, max(1, inserted_at)),
        }

    def delete(self, path: str) -> dict[str, Any]:
        resolved = self._resolve(path)
        self._reject_root_operation(resolved, "delete")
        if not resolved.path.exists():
            raise MemoryToolError(
                f"The path {resolved.virtual} does not exist",
                code="memory_path_not_found",
                retryable=True,
            )
        if resolved.path.is_dir():
            shutil.rmtree(resolved.path)
        else:
            resolved.path.unlink()
        return {
            "operation": "delete",
            "path": resolved.virtual,
            "status": "deleted",
            "message": f"Successfully deleted {resolved.virtual}",
        }

    def rename(self, old_path: str, new_path: str) -> dict[str, Any]:
        source = self._resolve(old_path)
        self._reject_root_operation(source, "rename")
        if not source.path.exists():
            raise MemoryToolError(
                f"The path {source.virtual} does not exist",
                code="memory_path_not_found",
                retryable=True,
            )
        destination = self._resolve(new_path, for_write=True)
        self._reject_root_operation(destination, "rename")
        if destination.path.exists():
            raise MemoryToolError(
                f"The destination {destination.virtual} already exists",
                code="memory_destination_exists",
                retryable=True,
            )
        if source.path.is_dir() and _is_within(source.path.resolve(), destination.path):
            raise MemoryToolError(
                "Cannot rename a memory directory into itself",
                code="memory_invalid_rename",
                retryable=True,
            )
        destination.path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source.path), str(destination.path))
        return {
            "operation": "rename",
            "old_path": source.virtual,
            "new_path": destination.virtual,
            "status": "renamed",
            "message": f"Successfully renamed {source.virtual} to {destination.virtual}",
        }

    def startup_index(
        self,
        path: str = _STARTUP_INDEX_PATH,
        *,
        max_lines: int = _STARTUP_INDEX_MAX_LINES,
        max_bytes: int = _STARTUP_INDEX_MAX_BYTES,
    ) -> str | None:
        try:
            resolved = self._resolve(path)
        except MemoryToolError:
            return None
        if not resolved.path.exists() or not resolved.path.is_file():
            return None
        with resolved.path.open("rb") as handle:
            data = handle.read(max(0, int(max_bytes)))
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return None
        lines = text.splitlines()[:max_lines]
        if not lines:
            return None
        return (
            f"Memory index ({resolved.virtual}, first {len(lines)} lines or {max_bytes} bytes):\n"
            + "\n".join(lines)
        )

    def _resolve_if_mounted(self, virtual: str, *, for_write: bool = False) -> _ResolvedPath | None:
        mount = self._find_mount(virtual)
        if mount is None:
            return None
        return self._resolve_mounted(virtual, mount, for_write=for_write)

    def _resolve(self, path: str, *, for_write: bool = False) -> _ResolvedPath:
        virtual = _normalize_memory_path(path)
        resolved = self._resolve_if_mounted(virtual, for_write=for_write)
        if resolved is None:
            raise MemoryToolError(
                f"Memory path is not mounted: {virtual}",
                code="memory_path_unmounted",
                retryable=True,
            )
        return resolved

    def _find_mount(self, virtual: str) -> _Mount | None:
        for mount in self._mounts:
            if virtual == mount.virtual or virtual.startswith(mount.virtual + "/"):
                return mount
        return None

    def _resolve_mounted(self, virtual: str, mount: _Mount, *, for_write: bool) -> _ResolvedPath:
        suffix = virtual.removeprefix(mount.virtual).strip("/")
        relative_parts = tuple(part for part in suffix.split("/") if part)
        candidate = mount.root.joinpath(*relative_parts) if relative_parts else mount.root
        resolved_candidate = _resolve_candidate(mount.root, candidate)
        if for_write and resolved_candidate == mount.root and virtual == mount.virtual:
            raise MemoryToolError(
                f"Cannot write memory mount root: {virtual}",
                code="memory_root_write_rejected",
                retryable=True,
            )
        return _ResolvedPath(virtual, mount, relative_parts, resolved_candidate)

    def _view_virtual_namespace(self, virtual: str) -> dict[str, Any] | None:
        mounts = self._child_mounts(virtual)
        if not mounts:
            return None
        entries: dict[str, dict[str, Any]] = {
            virtual: {"path": virtual, "kind": "dir", "size": 0}
        }
        for mount in mounts:
            child = _first_child_under(mount.virtual, virtual)
            size = _tree_size(mount.root)
            entries[virtual]["size"] += size
            if child in entries:
                entries[child]["size"] += size
            else:
                entries[child] = {"path": child, "kind": "dir", "size": size}
        return {
            "operation": "view",
            "path": virtual,
            "status": "ok",
            "message": f"Listed memory directory: {virtual}",
            "entries": list(entries.values()),
        }

    def _view_directory(self, resolved: _ResolvedPath) -> dict[str, Any]:
        entries: dict[str, dict[str, Any]] = {
            resolved.virtual: {"path": resolved.virtual, "kind": "dir", "size": _tree_size(resolved.path)}
        }
        child_mounts = self._child_mounts(resolved.virtual)
        for item in sorted(resolved.path.rglob("*"), key=lambda child: child.as_posix()):
            rel = item.relative_to(resolved.path)
            if len(rel.parts) > 2:
                continue
            virtual = _join_virtual(resolved.virtual, rel.as_posix())
            if _is_shadowed_by_mount(virtual, child_mounts):
                continue
            if any(part.startswith(".") or part == "node_modules" for part in rel.parts):
                continue
            if item.is_symlink():
                continue
            kind = "dir" if item.is_dir() else "file" if item.is_file() else "other"
            size = item.stat().st_size if item.is_file() else _tree_size(item) if item.is_dir() else 0
            entries[virtual] = {"path": virtual, "kind": kind, "size": size}
        for mount in child_mounts:
            child = _first_child_under(mount.virtual, resolved.virtual)
            size = _tree_size(mount.root)
            if child in entries:
                entries[child]["size"] += size
            else:
                entries[child] = {"path": child, "kind": "dir", "size": size}
        return {
            "operation": "view",
            "path": resolved.virtual,
            "status": "ok",
            "message": f"Listed memory directory: {resolved.virtual}",
            "entries": list(entries.values()),
        }

    def _view_file(
        self,
        resolved: _ResolvedPath,
        view_range: tuple[int, int] | None,
    ) -> dict[str, Any]:
        text = self._read_text_file(resolved)
        lines = text.splitlines()
        start = 1
        end = len(lines)
        if view_range is not None:
            raw_start, raw_end = view_range
            if raw_start < 1:
                raise MemoryToolError(
                    "view_range start line must be >= 1",
                    code="memory_invalid_view_range",
                    retryable=True,
                )
            if raw_start > len(lines):
                raise MemoryToolError(
                    "view_range start line exceeds the file length",
                    code="memory_invalid_view_range",
                    retryable=True,
                )
            start = raw_start
            end = len(lines) if raw_end == -1 else raw_end
            if end < start:
                raise MemoryToolError(
                    "view_range end line must be >= start line or -1",
                    code="memory_invalid_view_range",
                    retryable=True,
                )
            end = min(end, len(lines))
        selected = lines[start - 1 : end] if lines else []
        return {
            "operation": "view",
            "path": resolved.virtual,
            "status": "ok",
            "message": f"Read memory file: {resolved.virtual}",
            "content": _numbered_lines(selected, start),
            "lines": {"start": start, "end": end, "total": len(lines)},
            "sha256": _sha256(text.encode("utf-8")),
        }

    def _search_roots(self, virtual: str) -> tuple[_ResolvedPath, ...]:
        child_mounts = self._child_mounts(virtual)
        roots: list[_ResolvedPath] = []
        resolved = self._resolve_if_mounted(virtual)
        if resolved is not None:
            if not resolved.path.exists():
                if child_mounts:
                    return tuple(_ResolvedPath(mount.virtual, mount, (), mount.root) for mount in child_mounts)
                raise MemoryToolError(
                    f"The path {virtual} does not exist. Please provide a valid namespace.",
                    code="memory_path_not_found",
                    retryable=True,
                )
            roots.append(resolved)
        roots.extend(_ResolvedPath(mount.virtual, mount, (), mount.root) for mount in child_mounts)
        if roots:
            return tuple(roots)
        raise MemoryToolError(
            f"Memory namespace is not mounted: {virtual}",
            code="memory_path_unmounted",
            retryable=True,
        )

    def _child_mounts(self, virtual: str) -> tuple[_Mount, ...]:
        return tuple(
            mount
            for mount in sorted(self._mounts, key=lambda item: item.virtual)
            if mount.virtual != virtual and _is_virtual_child(virtual, mount.virtual)
        )

    def _iter_search_files(
        self,
        resolved: _ResolvedPath,
        *,
        file_glob: str | None,
    ) -> Iterable[tuple[str, Path]]:
        if resolved.path.is_file():
            if _search_file_selected(resolved.path.name, file_glob):
                yield resolved.virtual, resolved.path
            return
        if not resolved.path.is_dir():
            return
        child_mounts = self._child_mounts(resolved.virtual)
        for item in sorted(resolved.path.rglob("*"), key=lambda child: child.as_posix()):
            if item.is_symlink() or not item.is_file():
                continue
            rel = item.relative_to(resolved.path)
            virtual = _join_virtual(resolved.virtual, rel.as_posix())
            if _is_shadowed_by_mount(virtual, child_mounts):
                continue
            if any(part.startswith(".") or part == "node_modules" for part in rel.parts):
                continue
            if not _search_file_selected(rel.as_posix(), file_glob):
                continue
            yield virtual, item

    def _search_file(
        self,
        virtual: str,
        path: Path,
        *,
        needle: str,
        lowered_needle: str,
        remaining: int,
    ) -> list[dict[str, Any]]:
        try:
            resolved = _ResolvedPath(virtual, self._find_mount(virtual) or self._mounts[0], (), path)
            text = self._read_text_file(resolved)
        except MemoryToolError:
            return []
        matches: list[dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if lowered_needle not in line.lower():
                continue
            matches.append(
                {
                    "path": virtual,
                    "line": line_no,
                    "text": line,
                    "snippet": _line_snippet(text, line_no, radius=1),
                }
            )
            if len(matches) >= remaining:
                break
        return matches

    def _read_text_file(self, resolved: _ResolvedPath) -> str:
        if not resolved.path.exists() or not resolved.path.is_file():
            raise MemoryToolError(
                f"The path {resolved.virtual} does not exist. Please provide a valid path.",
                code="memory_path_not_found",
                retryable=True,
            )
        size = resolved.path.stat().st_size
        if size > self.max_view_bytes:
            raise MemoryToolError(
                f"Memory file exceeds max view size: {resolved.virtual}",
                code="memory_file_too_large",
                retryable=True,
            )
        data = resolved.path.read_bytes()
        if b"\x00" in data:
            raise MemoryToolError(
                f"Unsupported binary memory file: {resolved.virtual}",
                code="memory_unsupported_media",
            )
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MemoryToolError(
                f"Unsupported non-UTF-8 memory file: {resolved.virtual}",
                code="memory_unsupported_media",
            ) from exc

    @staticmethod
    def _reject_root_operation(resolved: _ResolvedPath, operation: str) -> None:
        if resolved.virtual == MEMORY_ROOT or not resolved.relative_parts:
            raise MemoryToolError(
                f"Cannot {operation} the memory root: {resolved.virtual}",
                code="memory_root_operation_rejected",
                retryable=True,
            )


class MemoryProvider:
    """Expose a ``MemoryStore`` as operation-specific MAK tools."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        read_authorization: ToolAuthorizationDecision = "allow",
        write_authorization: ToolAuthorizationDecision = "ask",
        startup_index_path: str = _STARTUP_INDEX_PATH,
        startup_index_max_lines: int = _STARTUP_INDEX_MAX_LINES,
        startup_index_max_bytes: int = _STARTUP_INDEX_MAX_BYTES,
    ) -> None:
        self.store = store
        self.read_authorization = read_authorization
        self.write_authorization = write_authorization
        self.startup_index_path = startup_index_path
        self.startup_index_max_lines = int(startup_index_max_lines)
        self.startup_index_max_bytes = int(startup_index_max_bytes)

    def static_segment(self) -> str | None:
        return None

    def dynamic_segment(self, turn: TurnContext) -> str | None:
        bound_tools = set(turn.bound_tools)
        can_view_index = (
            MEMORY_VIEW_TOOL_ID in turn.allowed_tools
            if turn.allowed_tools is not None
            else self.read_authorization == "allow" and MEMORY_VIEW_TOOL_ID in bound_tools
        )
        if not (set(MEMORY_READ_TOOL_IDS) & bound_tools):
            return None
        lines = [
            "# Memory",
            "Persistent memory is available under /memories. Use memory tools when stored context helps the task or future sessions.",
        ]
        if can_view_index:
            index = self.store.startup_index(
                self.startup_index_path,
                max_lines=self.startup_index_max_lines,
                max_bytes=self.startup_index_max_bytes,
            )
            if index:
                lines.extend(("", index))
        return "\n".join(lines)

    def get_tools(self, context: Any = None) -> Iterable[ToolSpec]:  # noqa: ARG002 - provider seam
        return (
            self._search_tool(),
            self._view_tool(),
            self._create_tool(),
            self._str_replace_tool(),
            self._insert_tool(),
            self._delete_tool(),
            self._rename_tool(),
        )

    def tool_bindings(self) -> tuple[ToolBinding, ...]:
        return tuple(
            ToolBinding(
                binding_id=tool_id,
                ref=RegistryToolRef(tool_id=tool_id),
                authorization=(
                    self.read_authorization
                    if tool_id in {MEMORY_SEARCH_TOOL_ID, MEMORY_VIEW_TOOL_ID}
                    else self.write_authorization
                ),
            )
            for tool_id in MEMORY_TOOL_IDS
        )

    def catalog(self) -> dict[str, Any]:
        mounts = getattr(self.store, "mounts", ())
        return {
            "tools": [
                {"id": spec.id, "description": spec.description}
                for spec in self.get_tools(None)
            ],
            "mounts": list(mounts),
        }

    def _search_tool(self) -> ToolSpec:
        def handler(_context: Any, args: dict[str, Any]) -> ToolResult:
            filters = args.get("filters")
            return _call_memory_tool(
                lambda: self.store.search(
                    str(args["query"]),
                    None if args.get("namespace") is None else str(args.get("namespace")),
                    limit=int(args.get("limit", 20)),
                    filters=filters if isinstance(filters, Mapping) else None,
                )
            )

        return ToolSpec(
            id=MEMORY_SEARCH_TOOL_ID,
            description="Search UTF-8 memory files under /memories for a literal query.",
            input_schema=_object_schema(
                {
                    "query": {"type": "string"},
                    "namespace": {"type": ["string", "null"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                    "filters": {"type": ["object", "null"], "additionalProperties": True},
                },
                required=["query"],
            ),
            capability="memory.read",
            side_effect="read",
            handler=handler,
            guidance={"group": "memory", "tags": ["memory", "read", "search"]},
            annotations={"memory": True},
        )

    def _view_tool(self) -> ToolSpec:
        def handler(_context: Any, args: dict[str, Any]) -> ToolResult:
            return _call_memory_tool(
                lambda: self.store.view(str(args["path"]), _coerce_view_range(args.get("view_range")))
            )

        return ToolSpec(
            id=MEMORY_VIEW_TOOL_ID,
            description="Read a memory directory listing or UTF-8 memory file under /memories.",
            input_schema=_object_schema(
                {
                    "path": {"type": "string"},
                    "view_range": {
                        "type": ["array", "null"],
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                required=["path"],
            ),
            capability="memory.read",
            side_effect="read",
            handler=handler,
            guidance={"group": "memory", "tags": ["memory", "read"]},
            annotations={"memory": True},
        )

    def _create_tool(self) -> ToolSpec:
        def handler(_context: Any, args: dict[str, Any]) -> ToolResult:
            return _call_memory_tool(lambda: self.store.create(str(args["path"]), str(args["file_text"])))

        return ToolSpec(
            id=MEMORY_CREATE_TOOL_ID,
            description="Create a new UTF-8 memory file under /memories.",
            input_schema=_object_schema(
                {"path": {"type": "string"}, "file_text": {"type": "string"}},
                required=["path", "file_text"],
            ),
            capability="memory.write",
            side_effect="write",
            handler=handler,
            guidance={"group": "memory", "tags": ["memory", "write"]},
            annotations={"memory": True},
        )

    def _str_replace_tool(self) -> ToolSpec:
        def handler(_context: Any, args: dict[str, Any]) -> ToolResult:
            return _call_memory_tool(
                lambda: self.store.str_replace(
                    str(args["path"]),
                    str(args["old_str"]),
                    "" if args.get("new_str") is None else str(args.get("new_str")),
                )
            )

        return ToolSpec(
            id=MEMORY_STR_REPLACE_TOOL_ID,
            description="Replace one unique exact string in a UTF-8 memory file under /memories.",
            input_schema=_object_schema(
                {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": ["string", "null"]},
                },
                required=["path", "old_str"],
            ),
            capability="memory.write",
            side_effect="write",
            handler=handler,
            guidance={"group": "memory", "tags": ["memory", "write"]},
            annotations={"memory": True},
        )

    def _insert_tool(self) -> ToolSpec:
        def handler(_context: Any, args: dict[str, Any]) -> ToolResult:
            return _call_memory_tool(
                lambda: self.store.insert(
                    str(args["path"]),
                    int(args["insert_line"]),
                    str(args["insert_text"]),
                )
            )

        return ToolSpec(
            id=MEMORY_INSERT_TOOL_ID,
            description="Insert text after a line in a UTF-8 memory file under /memories.",
            input_schema=_object_schema(
                {
                    "path": {"type": "string"},
                    "insert_line": {"type": "integer", "minimum": 0},
                    "insert_text": {"type": "string"},
                },
                required=["path", "insert_line", "insert_text"],
            ),
            capability="memory.write",
            side_effect="write",
            handler=handler,
            guidance={"group": "memory", "tags": ["memory", "write"]},
            annotations={"memory": True},
        )

    def _delete_tool(self) -> ToolSpec:
        def handler(_context: Any, args: dict[str, Any]) -> ToolResult:
            return _call_memory_tool(lambda: self.store.delete(str(args["path"])))

        return ToolSpec(
            id=MEMORY_DELETE_TOOL_ID,
            description="Delete a memory file or directory under /memories. The root cannot be deleted.",
            input_schema=_object_schema({"path": {"type": "string"}}, required=["path"]),
            capability="memory.write",
            side_effect="write",
            handler=handler,
            guidance={"group": "memory", "tags": ["memory", "write"]},
            annotations={"memory": True},
        )

    def _rename_tool(self) -> ToolSpec:
        def handler(_context: Any, args: dict[str, Any]) -> ToolResult:
            return _call_memory_tool(lambda: self.store.rename(str(args["old_path"]), str(args["new_path"])))

        return ToolSpec(
            id=MEMORY_RENAME_TOOL_ID,
            description="Rename or move a memory file or directory under /memories.",
            input_schema=_object_schema(
                {"old_path": {"type": "string"}, "new_path": {"type": "string"}},
                required=["old_path", "new_path"],
            ),
            capability="memory.write",
            side_effect="write",
            handler=handler,
            guidance={"group": "memory", "tags": ["memory", "write"]},
            annotations={"memory": True},
        )


class LocalFilesystemMemoryProvider(MemoryProvider):
    """Convenience provider using ``LocalFilesystemMemoryStore``."""

    def __init__(
        self,
        base_path: str | Path | None = None,
        *,
        mounts: Mapping[str, str | Path] | None = None,
        read_authorization: ToolAuthorizationDecision = "allow",
        write_authorization: ToolAuthorizationDecision = "ask",
        startup_index_path: str = _STARTUP_INDEX_PATH,
        startup_index_max_lines: int = _STARTUP_INDEX_MAX_LINES,
        startup_index_max_bytes: int = _STARTUP_INDEX_MAX_BYTES,
    ) -> None:
        super().__init__(
            LocalFilesystemMemoryStore(base_path, mounts=mounts),
            read_authorization=read_authorization,
            write_authorization=write_authorization,
            startup_index_path=startup_index_path,
            startup_index_max_lines=startup_index_max_lines,
            startup_index_max_bytes=startup_index_max_bytes,
        )


def _object_schema(properties: dict[str, Any], *, required: list[str] | None = None) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _call_memory_tool(fn: Any) -> ToolResult:
    try:
        return ToolResult(ok=True, content=fn())
    except MemoryToolError as exc:
        return ToolResult(
            ok=False,
            error=str(exc),
            error_code=exc.code,
            retryable=exc.retryable,
        )


def _coerce_view_range(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise MemoryToolError(
            "view_range must be a two-item array",
            code="memory_invalid_view_range",
            retryable=True,
        )
    return int(value[0]), int(value[1])


def _namespace_to_path(namespace: str | None) -> str:
    if namespace is None or str(namespace).strip() == "":
        return MEMORY_ROOT
    raw = str(namespace).strip()
    if raw.startswith("/"):
        return _normalize_memory_path(raw)
    return _normalize_memory_path(MEMORY_ROOT + "/" + raw.strip("/"))


def _normalize_memory_path(path: str) -> str:
    raw = str(path or "").strip()
    decoded = unquote(raw)
    if not decoded:
        raise MemoryToolError("memory path is required", code="memory_path_invalid", retryable=True)
    if "\\" in raw or "\\" in decoded:
        raise MemoryToolError("memory paths must use POSIX separators", code="memory_path_invalid", retryable=True)
    if decoded != MEMORY_ROOT and not decoded.startswith(MEMORY_ROOT + "/"):
        raise MemoryToolError(
            f"memory path must stay under {MEMORY_ROOT}",
            code="memory_path_outside_root",
            retryable=True,
        )
    parts = [part for part in decoded.split("/") if part]
    if not parts or parts[0] != "memories":
        raise MemoryToolError(
            f"memory path must stay under {MEMORY_ROOT}",
            code="memory_path_outside_root",
            retryable=True,
        )
    if any(part in {".", ".."} for part in parts[1:]):
        raise MemoryToolError("memory path traversal is not allowed", code="memory_path_traversal", retryable=True)
    return "/" + "/".join(parts)


def _resolve_candidate(root: Path, candidate: Path) -> Path:
    root_resolved = root.resolve()
    _reject_symlink_path(root_resolved, candidate)
    if candidate.exists():
        resolved = candidate.resolve()
        if not _is_within(root_resolved, resolved):
            raise MemoryToolError("memory path escapes its mount", code="memory_path_escape", retryable=True)
        return resolved
    parent = candidate.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.exists():
        parent = root_resolved
    parent_resolved = parent.resolve()
    if not _is_within(root_resolved, parent_resolved):
        raise MemoryToolError("memory path escapes its mount", code="memory_path_escape", retryable=True)
    if not parent.is_dir():
        raise MemoryToolError(
            "memory path parent is not a directory",
            code="memory_parent_not_directory",
            retryable=True,
        )
    try:
        suffix = candidate.relative_to(parent)
    except ValueError:
        suffix = Path(candidate.name)
    resolved = parent_resolved / suffix
    if not _is_within(root_resolved, resolved):
        raise MemoryToolError("memory path escapes its mount", code="memory_path_escape", retryable=True)
    return resolved


def _reject_symlink_path(root: Path, candidate: Path) -> None:
    probe = candidate
    while True:
        if probe.is_symlink():
            raise MemoryToolError(
                "memory symlinks are not supported",
                code="memory_symlink_unsupported",
                retryable=True,
            )
        if probe == root or probe.parent == probe:
            return
        probe = probe.parent


def _is_within(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve() if candidate.exists() else candidate
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _is_virtual_child(parent: str, child: str) -> bool:
    return child.startswith(parent.rstrip("/") + "/")


def _is_shadowed_by_mount(virtual: str, mounts: tuple[_Mount, ...]) -> bool:
    return any(virtual == mount.virtual or _is_virtual_child(mount.virtual, virtual) for mount in mounts)


def _first_child_under(virtual: str, parent: str) -> str:
    suffix = virtual.removeprefix(parent.rstrip("/") + "/")
    first_part = suffix.split("/", 1)[0]
    return _join_virtual(parent, first_part)


def _join_virtual(base: str, child: str) -> str:
    return base.rstrip("/") + "/" + child.replace("\\", "/").strip("/")


def _search_file_selected(path: str, file_glob: str | None) -> bool:
    if file_glob is None or file_glob.strip() == "":
        return True
    import fnmatch

    return fnmatch.fnmatch(path, file_glob)


def _tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    if not path.exists() or path.is_symlink():
        return total
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _numbered_lines(lines: list[str], start_line: int) -> str:
    return "\n".join(f"{line_no:6}\t{line}" for line_no, line in enumerate(lines, start=start_line))


def _line_snippet(text: str, line_no: int, *, radius: int = 2) -> str:
    lines = text.splitlines()
    if not lines:
        return ""
    start = max(1, line_no - radius)
    end = min(len(lines), line_no + radius)
    return _numbered_lines(lines[start - 1 : end], start)


def _snippet_around_index(text: str, index: int) -> str:
    line_no = text[: max(index, 0)].count("\n") + 1
    return _line_snippet(text, line_no)


def _occurrence_lines(text: str, needle: str) -> tuple[int, ...]:
    lines: list[int] = []
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            break
        lines.append(text[:idx].count("\n") + 1)
        start = idx + max(1, len(needle))
    return tuple(lines)


def _insert_after_line(text: str, insert_line: int, insert_text: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    offset = 0 if insert_line == 0 else sum(len(line) for line in lines[:insert_line])
    payload = insert_text
    prefix = text[:offset]
    suffix = text[offset:]
    if insert_line > 0 and prefix and not prefix.endswith(("\n", "\r")) and payload and not payload.startswith(("\n", "\r")):
        payload = "\n" + payload
    if suffix and payload and not payload.endswith(("\n", "\r")) and not suffix.startswith(("\n", "\r")):
        payload = payload + "\n"
    return prefix + payload + suffix, insert_line + 1


__all__ = [
    "MEMORY_ROOT",
    "MEMORY_TOOL_IDS",
    "MEMORY_SEARCH_TOOL_ID",
    "MEMORY_VIEW_TOOL_ID",
    "MEMORY_CREATE_TOOL_ID",
    "MEMORY_STR_REPLACE_TOOL_ID",
    "MEMORY_INSERT_TOOL_ID",
    "MEMORY_DELETE_TOOL_ID",
    "MEMORY_RENAME_TOOL_ID",
    "MemoryToolError",
    "MemoryStore",
    "MemoryProvider",
    "LocalFilesystemMemoryStore",
    "LocalFilesystemMemoryProvider",
]
