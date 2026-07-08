from __future__ import annotations

import fnmatch
from collections.abc import Mapping
from typing import Any

from monoid_agent_kernel.core.content import DocumentPart, ImagePart
from monoid_agent_kernel.core.media import MEDIA_INPUT_CAPABILITY
from monoid_agent_kernel.core.workspace import Workspace
from monoid_agent_kernel.errors import WorkspaceError
from monoid_agent_kernel.tools.base import ToolContext, ToolResult, ToolSpec


def builtin_tools(workspace: Workspace) -> list[ToolSpec]:
    return [
        _fs_list(workspace),
        _fs_tree(workspace),
        _fs_stat(workspace),
        _fs_read(workspace),
        _fs_read_media(workspace),
        _fs_glob(workspace),
        _text_search(workspace),
        _tool_search(),
        _fs_write(workspace),
        _fs_patch(workspace),
        _fs_mkdir(workspace),
        _fs_copy(workspace),
        _fs_move(workspace),
        _fs_delete(workspace),
        _shell_exec(),
        _job_list(),
        _job_status(),
        _job_logs(),
        _job_cancel(),
        _job_wait(),
        _hitl_request(),
        _web_search(),
        _web_fetch(),
        _web_context(),
        _artifact_emit(),
        _artifact_list(),
        _run_update_plan(),
        _run_finish(),
    ]


def _object_schema(properties: dict[str, Any], *, required: list[str] | None = None) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _decode_text_or_none(data: bytes) -> str | None:
    """Decode bytes as UTF-8 text, or ``None`` when they're binary (a NUL byte) or not valid
    UTF-8 — a branchable signal so callers can fall back instead of always raising."""
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _text_from_bytes(data: bytes, path: str) -> str:
    # Reject binary/non-utf8 for the plain-text read tools (fs.read_lines / fs.read_many). The
    # primary fs.read tool falls back to the media path instead (see _fs_read); this stays strict.
    text = _decode_text_or_none(data)
    if text is not None:
        return text
    if b"\x00" in data:
        raise WorkspaceError(f"binary file cannot be read as text: {path}")
    raise WorkspaceError(f"file is not utf-8 text: {path}")


def _path_allowed(context: ToolContext, path: str, operation: str = "read") -> bool:
    checker = getattr(context, "path_allowed", None)
    if not callable(checker):
        return True
    return bool(checker(path, operation))


def _skip_reasons(permission_denied: int = 0, unreadable: int = 0) -> dict[str, int]:
    reasons: dict[str, int] = {}
    if permission_denied:
        reasons["permission_denied"] = permission_denied
    if unreadable:
        reasons["unreadable"] = unreadable
    return reasons


def _short_snippet(text: str, needle: str, *, radius: int = 80) -> str:
    index = text.find(needle)
    if index < 0:
        return ""
    start = max(0, index - radius)
    end = min(len(text), index + len(needle) + radius)
    prefix = "..." if start else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end] + suffix


def _tool_search() -> ToolSpec:
    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content=context.search_tools(args))

    return ToolSpec(
        id="tool.search",
        description=(
            "Search tools available through the current Tool Surface. "
            "Returned tools can be loaded for the next turn."
        ),
        input_schema=_object_schema(
            {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1},
                "namespace": {"type": "string"},
                "group": {"type": "string"},
                "groups": {"type": "array", "items": {"type": "string"}},
                "tag": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            required=["query"],
        ),
        capability="tool.search",
        side_effect="read",
        handler=handler,
    )


def _fs_list(workspace: Workspace) -> ToolSpec:
    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        max_entries = int(args.get("max_entries", 200))
        raw_limit = min(max(max_entries * 5, max_entries), 5000)
        raw_entries = workspace.list_entries(
            args.get("path", "."),
            recursive=bool(args.get("recursive", False)),
            max_entries=raw_limit,
        )
        entries = []
        skipped = 0
        for entry in raw_entries:
            if _path_allowed(context, entry.path, "read"):
                entries.append(entry)
            else:
                skipped += 1
        visible = entries[:max_entries]
        return ToolResult(
            ok=True,
            content={
                "entries": [entry.__dict__ for entry in visible],
                "count": len(visible),
                "limit": max_entries,
                "searched_count": len(raw_entries),
                "skipped_count": skipped,
                "skipped_reasons": _skip_reasons(permission_denied=skipped),
                "truncated": len(entries) > max_entries or len(raw_entries) >= raw_limit,
            },
        )

    return ToolSpec(
        id="fs.list",
        description="List files and directories in the workspace.",
        input_schema=_object_schema(
            {
                "path": {"type": "string", "default": "."},
                "recursive": {"type": "boolean", "default": False},
                "max_entries": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 200},
            }
        ),
        capability="fs.read",
        side_effect="read",
        handler=handler,
        path_args=("path",),
    )


def _fs_tree(workspace: Workspace) -> ToolSpec:
    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        path = args.get("path", ".")
        depth = int(args.get("depth", 2))
        max_entries = int(args.get("max_entries", 200))
        normalized_path = workspace.normalize(path)
        if workspace.path_kind(path) == "file":
            name = normalized_path.split("/")[-1]
            return ToolResult(
                ok=True,
                content={
                    "tree": name,
                    "entries": 1,
                    "limit": max_entries,
                    "searched_count": 1,
                    "skipped_count": 0,
                    "skipped_reasons": {},
                    "truncated": False,
                },
            )
        raw_limit = min(max(max_entries * 5, max_entries), 5000)
        raw_entries = workspace.list_entries(path, recursive=True, max_entries=raw_limit)
        root_depth = 0 if path in {None, "", "."} else len(normalized_path.split("/"))
        lines: list[str] = []
        skipped = 0
        searched = 0
        for entry in raw_entries:
            searched += 1
            if not _path_allowed(context, entry.path, "read"):
                skipped += 1
                continue
            entry_depth = len(entry.path.split("/")) - root_depth
            if entry_depth > depth:
                continue
            indent = "  " * max(0, entry_depth - 1)
            suffix = "/" if entry.kind == "dir" else ""
            lines.append(f"{indent}{entry.path.split('/')[-1]}{suffix}")
            if len(lines) >= max_entries:
                break
        return ToolResult(
            ok=True,
            content={
                "tree": "\n".join(lines),
                "entries": len(lines),
                "limit": max_entries,
                "searched_count": searched,
                "skipped_count": skipped,
                "skipped_reasons": _skip_reasons(permission_denied=skipped),
                "truncated": len(lines) >= max_entries or len(raw_entries) >= raw_limit,
            },
        )

    return ToolSpec(
        id="fs.tree",
        description="Return a compact directory tree for a workspace path.",
        input_schema=_object_schema(
            {
                "path": {"type": "string", "default": "."},
                "depth": {"type": "integer", "minimum": 1, "maximum": 10, "default": 2},
                "max_entries": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 200},
            }
        ),
        capability="fs.read",
        side_effect="read",
        handler=handler,
        path_args=("path",),
    )


def _fs_stat(workspace: Workspace) -> ToolSpec:
    def handler(_context: ToolContext, args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        stat_path = getattr(workspace, "stat_path", None)
        if callable(stat_path):
            return ToolResult(ok=True, content=stat_path(path))
        rel, _abs_path = workspace.resolve_existing_or_parent(path)
        kind = workspace.path_kind(path)
        if kind is None:
            return ToolResult(ok=True, content={"path": rel, "exists": False})
        if kind == "file":
            data, _digest = workspace.read_bytes(rel)
            return ToolResult(ok=True, content={"path": rel, "exists": True, "kind": "file", "size": len(data)})
        return ToolResult(ok=True, content={"path": rel, "exists": True, "kind": kind, "size": 0})

    return ToolSpec(
        id="fs.stat",
        description="Return metadata for a workspace path.",
        input_schema=_object_schema({"path": {"type": "string"}}, required=["path"]),
        capability="fs.read",
        side_effect="read",
        handler=handler,
        path_args=("path",),
    )


def _fs_read(workspace: Workspace) -> ToolSpec:
    def handler(_context: ToolContext, args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        max_bytes = int(args.get("max_bytes", workspace.max_bytes_read))
        data, digest = workspace.read_bytes(path, max_bytes=max_bytes)
        text = _decode_text_or_none(data)
        if text is None:
            # Binary / non-utf8: point the model at fs.read_media rather than reading media here.
            # fs.read_media enforces its own scope, quota, and authorization for the path; fs.read
            # has no way to honor those, so it must not return media under its own (broader) binding.
            raise WorkspaceError(
                f"{path!r} is not UTF-8 text; use fs.read_media to read images or PDFs."
            )
        lines = text.splitlines(keepends=True)
        total_lines = len(lines)
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        applied_start: int | None = None
        applied_end: int | None = None
        if start_line is not None or end_line is not None:
            start = int(start_line or 1)
            end = int(end_line or max(total_lines, start))
            if start < 1 or end < start:
                raise WorkspaceError(f"invalid line range for {workspace.normalize(path)}")
            if total_lines and start > total_lines:
                raise WorkspaceError(f"line range starts past end of file: {workspace.normalize(path)}")
            end = min(total_lines, end)
            applied_start = start
            applied_end = end
            selected = "".join(lines[start - 1 : end])
        else:
            selected = text
        return ToolResult(
            ok=True,
            content={
                "path": workspace.normalize(path),
                "content": selected,
                "sha256": digest,
                "size": len(data),
                "line_start": applied_start,
                "line_end": applied_end,
                "total_lines": total_lines,
                "truncated": False,
            },
        )

    return ToolSpec(
        id="fs.read",
        description="Read a UTF-8 text file from the workspace.",
        input_schema=_object_schema(
            {
                "path": {"type": "string"},
                "start_line": {"type": ["integer", "null"], "minimum": 1},
                "end_line": {"type": ["integer", "null"], "minimum": 1},
                "max_bytes": {"type": "integer", "minimum": 1, "maximum": 10_000_000, "default": 1_000_000},
            },
            required=["path"],
        ),
        capability="fs.read",
        side_effect="read",
        handler=handler,
        path_args=("path",),
    )


def _sniff_media_mime(data: bytes) -> str | None:
    """Identify an image or PDF by magic bytes (not extension — avoids mislabel-driven 400s)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:5] == b"%PDF-":
        return "application/pdf"
    return None


def _media_result(workspace: Workspace, path: str, data: bytes, digest: str) -> ToolResult:
    """Build a by-reference media ToolResult (ImagePart/DocumentPart) for an image or PDF. Shared
    by fs.read_media and fs.read's binary fallback. Raises WorkspaceError if the bytes are
    oversized for the run's wire-build cap or are not a supported media type."""
    # Eager guard (gap 2b): the media is forwarded by reference and re-read at wire-build under
    # the run's max_bytes_read. Reject here — adjacent to the cause — if it would not fit then,
    # so the run never produces a media reference doomed to fail mid-turn. Same threshold as the
    # wire-build read (single source of truth), with an actionable remedy.
    if len(data) > workspace.max_bytes_read:
        raise WorkspaceError(
            f"media {path!r} is {len(data)} bytes, over the run's max_bytes_read "
            f"({workspace.max_bytes_read}); it cannot be forwarded to the model. "
            f"Raise max_bytes_read or downsample the media."
        )
    mime = _sniff_media_mime(data)
    if mime is None:
        raise WorkspaceError(f"not a supported image or PDF file: {path}")
    normalized = workspace.normalize(path)
    # Media travels by reference (source_ref); the kernel resolves + forwards it so the
    # model can view it. The text content carries metadata only.
    part: ImagePart | DocumentPart = (
        DocumentPart(source_ref=normalized, mime_type=mime)
        if mime == "application/pdf"
        else ImagePart(source_ref=normalized, mime_type=mime)
    )
    return ToolResult(
        ok=True,
        content={"path": normalized, "mime_type": mime, "sha256": digest, "size": len(data)},
        media=(part,),
    )


def _fs_read_media(workspace: Workspace) -> ToolSpec:
    def handler(_context: ToolContext, args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        max_bytes = int(args.get("max_bytes", workspace.max_bytes_read))
        data, digest = workspace.read_bytes(path, max_bytes=max_bytes)
        return _media_result(workspace, path, data, digest)

    return ToolSpec(
        id="fs.read_media",
        description="Read an image (PNG/JPEG/GIF/WebP) or PDF file from the workspace so the model can view it.",
        input_schema=_object_schema(
            {
                "path": {"type": "string"},
                "max_bytes": {"type": "integer", "minimum": 1, "maximum": 10_000_000, "default": 1_000_000},
            },
            required=["path"],
        ),
        capability=MEDIA_INPUT_CAPABILITY,
        side_effect="read",
        handler=handler,
        path_args=("path",),
    )


def _fs_glob(workspace: Workspace) -> ToolSpec:
    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        max_matches = int(args.get("max_matches", 200))
        raw_limit = min(max(max_matches * 5, max_matches), 5000)
        raw_matches = workspace.glob(
            str(args["pattern"]),
            root=str(args.get("root", ".")),
            max_matches=raw_limit,
        )
        skipped = 0
        matches: list[str] = []
        for rel in raw_matches:
            if _path_allowed(context, rel, "read"):
                matches.append(rel)
            else:
                skipped += 1
        visible = matches[:max_matches]
        return ToolResult(
            ok=True,
            content={
                "matches": visible,
                "count": len(visible),
                "limit": max_matches,
                "searched_count": len(raw_matches),
                "skipped_count": skipped,
                "skipped_reasons": _skip_reasons(permission_denied=skipped),
                "truncated": len(matches) > max_matches or len(raw_matches) >= raw_limit,
            },
        )

    return ToolSpec(
        id="fs.glob",
        description="Find workspace paths matching a glob pattern.",
        input_schema=_object_schema(
            {
                "pattern": {"type": "string"},
                "root": {"type": "string", "default": "."},
                "max_matches": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 200},
            },
            required=["pattern"],
        ),
        capability="fs.read",
        side_effect="read",
        handler=handler,
        path_args=("root",),
    )


def _text_search(workspace: Workspace) -> ToolSpec:
    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        pattern = str(args["pattern"])
        root = str(args.get("root", "."))
        file_glob = args.get("file_glob")
        case_sensitive = bool(args.get("case_sensitive", False))
        max_matches = int(args.get("max_matches", 100))
        needle = pattern if case_sensitive else pattern.lower()
        matches: list[dict[str, Any]] = []
        searched = 0
        permission_skipped = 0
        unreadable = 0
        for rel in workspace.text_files(root=root, file_glob=file_glob):
            if len(matches) >= max_matches:
                break
            searched += 1
            if not _path_allowed(context, rel, "read"):
                permission_skipped += 1
                continue
            try:
                data, _digest = workspace.read_bytes(rel)
                text = _text_from_bytes(data, rel)
            except WorkspaceError:
                unreadable += 1
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                haystack = line if case_sensitive else line.lower()
                if needle in haystack or fnmatch.fnmatch(haystack, needle):
                    matches.append({"path": rel, "line": lineno, "text": line})
                    if len(matches) >= max_matches:
                        break
        return ToolResult(
            ok=True,
            content={
                "matches": matches,
                "count": len(matches),
                "limit": max_matches,
                "searched_count": searched,
                "skipped_count": permission_skipped + unreadable,
                "skipped_reasons": _skip_reasons(permission_denied=permission_skipped, unreadable=unreadable),
                "truncated": len(matches) >= max_matches,
            },
        )

    return ToolSpec(
        id="text.search",
        description="Search UTF-8 text files for a literal pattern.",
        input_schema=_object_schema(
            {
                "pattern": {"type": "string"},
                "root": {"type": "string", "default": "."},
                "file_glob": {"type": ["string", "null"]},
                "case_sensitive": {"type": "boolean", "default": False},
                "max_matches": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100},
            },
            required=["pattern"],
        ),
        capability="text.search",
        side_effect="read",
        handler=handler,
        path_args=("root",),
    )


def _fs_write(workspace: Workspace) -> ToolSpec:
    def handler(_context: ToolContext, args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        content = str(args["content"])
        if_exists = str(args.get("if_exists") or "overwrite")
        if if_exists not in {"overwrite", "fail"}:
            raise WorkspaceError("if_exists must be 'overwrite' or 'fail'")
        digest = workspace.write_bytes(
            path,
            content.encode("utf-8"),
            create_dirs=bool(args.get("create_dirs", False)),
            expected_sha256=args.get("expected_sha256"),
            overwrite=if_exists == "overwrite",
        )
        return ToolResult(ok=True, content={"path": workspace.normalize(path), "sha256": digest, "if_exists": if_exists})

    return ToolSpec(
        id="fs.write",
        description="Write a UTF-8 text file in the workspace.",
        input_schema=_object_schema(
            {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "create_dirs": {"type": "boolean", "default": False},
                "expected_sha256": {"type": ["string", "null"]},
                "if_exists": {"type": "string", "enum": ["overwrite", "fail"], "default": "overwrite"},
            },
            required=["path", "content"],
        ),
        capability="fs.write",
        side_effect="write",
        handler=handler,
        path_args=("path",),
    )


def _fs_patch(workspace: Workspace) -> ToolSpec:
    def handler(_context: ToolContext, args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        data, digest = workspace.read_bytes(path)
        expected_sha256 = args.get("expected_sha256")
        if expected_sha256 is not None and digest != expected_sha256:
            raise WorkspaceError(f"expected_sha256 mismatch for {workspace.normalize(path)}")
        text = _text_from_bytes(data, path)
        replacements = args["replacements"]
        changed = text
        applied = 0
        snippets: list[dict[str, Any]] = []
        for replacement in replacements:
            old = str(replacement["old"])
            new = str(replacement["new"])
            count = int(replacement.get("count", 1))
            if old == "":
                raise WorkspaceError(f"patch old text must not be empty in {workspace.normalize(path)}")
            occurrences = changed.count(old)
            if occurrences < count:
                raise WorkspaceError(f"patch text not found enough times in {workspace.normalize(path)}")
            snippets.append(
                {
                    "old": _short_snippet(changed, old),
                    "new": new[:200],
                    "count": count,
                    "occurrences": occurrences,
                }
            )
            changed = changed.replace(old, new, count)
            applied += count
        new_digest = workspace.write_bytes(
            path,
            changed.encode("utf-8"),
            expected_sha256=digest,
        )
        return ToolResult(
            ok=True,
            content={
                "path": workspace.normalize(path),
                "sha256": new_digest,
                "replacements": applied,
                "snippets": snippets,
            },
        )

    return ToolSpec(
        id="fs.patch",
        description="Patch a UTF-8 file using exact text replacements.",
        input_schema=_object_schema(
            {
                "path": {"type": "string"},
                "expected_sha256": {"type": ["string", "null"]},
                "replacements": {
                    "type": "array",
                    "minItems": 1,
                    "items": _object_schema(
                        {
                            "old": {"type": "string"},
                            "new": {"type": "string"},
                            "count": {"type": "integer", "minimum": 1, "default": 1},
                        },
                        required=["old", "new"],
                    ),
                },
            },
            required=["path", "replacements"],
        ),
        capability="fs.patch",
        side_effect="write",
        handler=handler,
        path_args=("path",),
    )


def _fs_mkdir(workspace: Workspace) -> ToolSpec:
    def handler(_context: ToolContext, args: dict[str, Any]) -> ToolResult:
        path = workspace.mkdir(str(args["path"]))
        return ToolResult(ok=True, content={"path": path})

    return ToolSpec(
        id="fs.mkdir",
        description="Create a directory in the workspace.",
        input_schema=_object_schema({"path": {"type": "string"}}, required=["path"]),
        capability="fs.mkdir",
        side_effect="write",
        handler=handler,
        path_args=("path",),
    )


def _file_operation_schema(required: list[str]) -> dict[str, Any]:
    return _object_schema(
        {
            "source_path": {"type": "string"},
            "destination_path": {"type": "string"},
            "overwrite": {"type": "boolean", "default": False},
            "create_dirs": {"type": "boolean", "default": False},
            "recursive": {"type": "boolean", "default": False},
            "max_entries": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 1000},
            "max_bytes": {"type": "integer", "minimum": 1, "maximum": 500_000_000, "default": 50_000_000},
            "directory_mode": {"type": "string", "enum": ["merge", "replace"], "default": "merge"},
        },
        required=required,
    )


def _fs_copy(workspace: Workspace) -> ToolSpec:
    def handler(_context: ToolContext, args: dict[str, Any]) -> ToolResult:
        result = workspace.copy_path(
            str(args["source_path"]),
            str(args["destination_path"]),
            overwrite=bool(args.get("overwrite", False)),
            create_dirs=bool(args.get("create_dirs", False)),
            recursive=bool(args.get("recursive", False)),
            max_entries=int(args.get("max_entries", 1000)),
            max_bytes=int(args.get("max_bytes", 50_000_000)),
            directory_mode=str(args.get("directory_mode") or "merge"),
        )
        return ToolResult(ok=True, content=result)

    return ToolSpec(
        id="fs.copy",
        description="Copy a file or directory within the workspace.",
        input_schema=_file_operation_schema(["source_path", "destination_path"]),
        capability="fs.copy",
        side_effect="write",
        handler=handler,
        path_args=("source_path", "destination_path"),
    )


def _fs_move(workspace: Workspace) -> ToolSpec:
    def handler(_context: ToolContext, args: dict[str, Any]) -> ToolResult:
        result = workspace.move_path(
            str(args["source_path"]),
            str(args["destination_path"]),
            overwrite=bool(args.get("overwrite", False)),
            create_dirs=bool(args.get("create_dirs", False)),
            recursive=bool(args.get("recursive", False)),
            max_entries=int(args.get("max_entries", 1000)),
            max_bytes=int(args.get("max_bytes", 50_000_000)),
            directory_mode=str(args.get("directory_mode") or "merge"),
        )
        return ToolResult(ok=True, content=result)

    return ToolSpec(
        id="fs.move",
        description="Move a file or directory within the workspace.",
        input_schema=_file_operation_schema(["source_path", "destination_path"]),
        capability="fs.move",
        side_effect="write",
        handler=handler,
        path_args=("source_path", "destination_path"),
    )


def _fs_delete(workspace: Workspace) -> ToolSpec:
    def handler(_context: ToolContext, args: dict[str, Any]) -> ToolResult:
        result = workspace.delete_path(
            str(args["path"]),
            recursive=bool(args.get("recursive", False)),
            max_entries=int(args.get("max_entries", 1000)),
            max_bytes=int(args.get("max_bytes", 50_000_000)),
        )
        return ToolResult(ok=True, content=result)

    return ToolSpec(
        id="fs.delete",
        description="Delete a file or directory from the workspace.",
        input_schema=_object_schema(
            {
                "path": {"type": "string"},
                "recursive": {"type": "boolean", "default": False},
                "max_entries": {"type": "integer", "minimum": 1, "maximum": 10000, "default": 1000},
                "max_bytes": {"type": "integer", "minimum": 1, "maximum": 500_000_000, "default": 50_000_000},
            },
            required=["path"],
        ),
        capability="fs.delete",
        side_effect="write",
        handler=handler,
        path_args=("path",),
    )


def _shell_exec() -> ToolSpec:
    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        result = context.execute_shell(args)
        if result.get("timed_out"):
            return ToolResult(
                ok=False,
                content=result,
                error="shell command timed out",
                error_code="shell_timeout",
            )
        if result.get("output_truncated"):
            return ToolResult(
                ok=False,
                content=result,
                error="shell command exceeded output limit",
                error_code="shell_output_limit_exceeded",
            )
        return ToolResult(ok=True, content=result)

    return ToolSpec(
        id="shell.exec",
        description="Run a shell command in a sanitized workspace copy.",
        input_schema=_object_schema(
            {
                "command": {"type": "string"},
                "cwd": {"type": "string", "default": "."},
                "timeout_s": {"type": ["integer", "null"], "minimum": 1},
                "max_output_bytes": {"type": ["integer", "null"], "minimum": 1},
                "startup_wait_s": {"type": ["integer", "null"], "minimum": 0},
                "background": {"type": "boolean", "default": False},
                "resume_on_exit": {"type": "boolean", "default": True},
                "env": {"type": "object", "additionalProperties": True, "default": {}},
            },
            required=["command"],
        ),
        capability="shell.exec",
        side_effect="shell",
        handler=handler,
        path_args=("cwd",),
        preview_kind="shell",
        emits_workspace_diff=True,
        changed_paths_source="result_content",
        result_payload_kind="shell_exec",
        skip_emit_if_background=True,
    )


def _job_list() -> ToolSpec:
    def handler(context: ToolContext, _args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content={"jobs": context.list_jobs()})

    return ToolSpec(
        id="job.list",
        description="List background shell jobs for this run.",
        input_schema=_object_schema({}),
        capability="job.control",
        side_effect="shell",
        handler=handler,
    )


def _job_status() -> ToolSpec:
    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content=context.job_status(args))

    return ToolSpec(
        id="job.status",
        description="Get the status of a background shell job.",
        input_schema=_object_schema({"job_id": {"type": "string"}}, required=["job_id"]),
        capability="job.control",
        side_effect="shell",
        handler=handler,
    )


def _job_logs() -> ToolSpec:
    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content=context.job_logs(args))

    return ToolSpec(
        id="job.logs",
        description="Read stdout or stderr from a background shell job.",
        input_schema=_object_schema(
            {
                "job_id": {"type": "string"},
                "stream": {"enum": ["stdout", "stderr"], "default": "stdout"},
                "tail_bytes": {"type": ["integer", "null"], "minimum": 0},
                "offset": {"type": ["integer", "null"], "minimum": 0},
            },
            required=["job_id"],
        ),
        capability="job.control",
        side_effect="shell",
        handler=handler,
    )


def _job_cancel() -> ToolSpec:
    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content=context.job_cancel(args))

    return ToolSpec(
        id="job.cancel",
        description="Cancel a background shell job.",
        input_schema=_object_schema({"job_id": {"type": "string"}}, required=["job_id"]),
        capability="job.control",
        side_effect="shell",
        handler=handler,
    )


def _job_wait() -> ToolSpec:
    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content=context.job_wait(args))

    return ToolSpec(
        id="job.wait",
        description="Wait briefly for a background shell job and return its current or final result.",
        input_schema=_object_schema(
            {
                "job_id": {"type": "string"},
                "timeout_s": {"type": ["integer", "null"], "minimum": 0},
            },
            required=["job_id"],
        ),
        capability="job.control",
        side_effect="shell",
        handler=handler,
    )


def _hitl_request() -> ToolSpec:
    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content=context.request_human_input(args))

    return ToolSpec(
        id="hitl.request",
        description=(
            "Request input from a human and pause for it. Returns a task_id immediately; "
            "the run parks until the human answers, then the answer is delivered to you."
        ),
        input_schema=_object_schema(
            {
                "prompt": {"type": "string"},
                "choices": {"type": "array", "items": {"type": "string"}},
            },
            required=["prompt"],
        ),
        capability="hitl.request",
        side_effect="run",
        handler=handler,
    )


def agent_spawn_tool(subagents: Mapping[str, str] | None = None) -> ToolSpec:
    """The agent-as-tool delegation tool. Registered in the base registry only when
    the run has ``subagent_definitions`` (see ``AgentLoop`` bootstrap); a runtime
    config still needs an explicit binding to ``agent.spawn`` to expose it.

    ``subagents`` maps available subagent id -> description; when present the tool
    advertises them (so the model picks the right one, the way Claude selects a
    subagent by its description) and constrains ``subagent_type`` to those ids."""

    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content=context.spawn_subagent(args))

    description = (
        "Delegate a focused task to a subagent that works in an isolated context "
        "and returns only its final message. Choose 'subagent_type' from the "
        "available subagents and give a self-contained 'prompt' (the subagent does "
        "not see this conversation). Set 'background': true to spawn without "
        "waiting — its result is delivered to you later; otherwise the call blocks "
        "and returns the subagent's final message directly."
    )
    subagent_type_schema: dict[str, Any] = {"type": "string"}
    if subagents:
        subagent_type_schema = {"type": "string", "enum": sorted(subagents)}
        catalog = "\n".join(
            f"- {sub_id}: {desc}" if desc else f"- {sub_id}"
            for sub_id, desc in sorted(subagents.items())
        )
        description = f"{description}\n\nAvailable subagents:\n{catalog}"

    return ToolSpec(
        id="agent.spawn",
        description=description,
        input_schema=_object_schema(
            {
                "subagent_type": subagent_type_schema,
                "prompt": {"type": "string"},
                "background": {"type": "boolean", "default": False},
            },
            required=["subagent_type", "prompt"],
        ),
        capability="agent.spawn",
        side_effect="run",
        handler=handler,
    )


def _web_search() -> ToolSpec:
    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content=context.execute_web_search(args))

    return ToolSpec(
        id="web.search",
        description="Search the web through the configured WebGateway.",
        input_schema=_object_schema(
            {
                "query": {"type": "string"},
                "max_results": {"type": ["integer", "null"], "minimum": 1},
                "allowed_domains": {"type": "array", "items": {"type": "string"}, "default": []},
                "blocked_domains": {"type": "array", "items": {"type": "string"}, "default": []},
                "recency_days": {"type": ["integer", "null"], "minimum": 1},
                "locale": {"type": ["string", "null"]},
            },
            required=["query"],
        ),
        capability="web.search",
        side_effect="read",
        handler=handler,
        preview_kind="web",
    )


def _web_fetch() -> ToolSpec:
    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content=context.execute_web_fetch(args))

    return ToolSpec(
        id="web.fetch",
        description="Fetch one web page through the configured WebGateway.",
        input_schema=_object_schema(
            {
                "url": {"type": "string"},
                "format": {"enum": ["text", "markdown"], "default": "text"},
                "timeout_s": {"type": ["integer", "null"], "minimum": 1},
                "max_bytes": {"type": ["integer", "null"], "minimum": 1},
            },
            required=["url"],
        ),
        capability="web.fetch",
        side_effect="read",
        handler=handler,
        preview_kind="web",
    )


def _web_context() -> ToolSpec:
    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content=context.execute_web_context(args))

    return ToolSpec(
        id="web.context",
        description="Build LLM-ready web grounding context through the configured WebGateway.",
        input_schema=_object_schema(
            {
                "query": {"type": "string"},
                "max_tokens": {"type": ["integer", "null"], "minimum": 1},
                "max_urls": {"type": ["integer", "null"], "minimum": 1},
                "max_snippets": {"type": ["integer", "null"], "minimum": 1},
                "allowed_domains": {"type": "array", "items": {"type": "string"}, "default": []},
                "blocked_domains": {"type": "array", "items": {"type": "string"}, "default": []},
                "recency_days": {"type": ["integer", "null"], "minimum": 1},
                "locale": {"type": ["string", "null"]},
            },
            required=["query"],
        ),
        capability="web.context",
        side_effect="read",
        handler=handler,
        preview_kind="web",
    )


def _artifact_emit() -> ToolSpec:
    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        artifact = context.emit_artifact(
            str(args["path"]),
            str(args["kind"]),
            args.get("label"),
            args.get("metadata") or {},
        )
        return ToolResult(ok=True, content={"artifact": artifact})

    return ToolSpec(
        id="artifact.emit",
        description="Register a workspace file as a run artifact.",
        input_schema=_object_schema(
            {
                "path": {"type": "string"},
                "kind": {"type": "string"},
                "label": {"type": ["string", "null"]},
                "metadata": {"type": "object", "additionalProperties": True},
            },
            required=["path", "kind"],
        ),
        capability="artifact.control",
        side_effect="artifact",
        handler=handler,
        path_args=("path",),
    )


def _artifact_list() -> ToolSpec:
    def handler(context: ToolContext, _args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content={"artifacts": context.list_artifacts()})

    return ToolSpec(
        id="artifact.list",
        description="List artifacts emitted during this run.",
        input_schema=_object_schema({}),
        capability="artifact.control",
        side_effect="artifact",
        handler=handler,
    )


def _run_update_plan() -> ToolSpec:
    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        items = list(args["items"])
        context.update_plan(items)
        return ToolResult(ok=True, content={"items": items})

    return ToolSpec(
        id="run.update_plan",
        description="Update the agent's current plan for observability.",
        input_schema=_object_schema(
            {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "string"},
                            "status": {"type": "string"},
                        },
                        "required": ["step", "status"],
                        "additionalProperties": True,
                    },
                }
            },
            required=["items"],
        ),
        capability="run.control",
        side_effect="run",
        handler=handler,
    )


def _run_finish() -> ToolSpec:
    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        outputs = list(args.get("outputs") or [])
        context.finish(str(args["summary"]), outputs, args.get("notes"))
        return ToolResult(ok=True, content={"summary": args["summary"], "outputs": outputs})

    return ToolSpec(
        id="run.finish",
        description="Finish the run with a summary and output paths.",
        input_schema=_object_schema(
            {
                "summary": {"type": "string"},
                "outputs": {"type": "array", "items": {"type": "string"}, "default": []},
                "notes": {"type": ["string", "null"]},
            },
            required=["summary"],
        ),
        capability="run.control",
        side_effect="run",
        handler=handler,
    )
