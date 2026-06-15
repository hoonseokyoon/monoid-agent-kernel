from __future__ import annotations

import fnmatch
from typing import Any

from native_agent_runner.errors import WorkspaceError
from native_agent_runner.tools.base import ToolContext, ToolResult, ToolSpec
from native_agent_runner.workspace.local import LocalWorkspaceBackend


def builtin_tools(workspace: LocalWorkspaceBackend) -> list[ToolSpec]:
    return [
        _fs_list(workspace),
        _fs_tree(workspace),
        _fs_stat(workspace),
        _fs_read(workspace),
        _fs_glob(workspace),
        _text_search(workspace),
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


def _text_from_bytes(data: bytes, path: str) -> str:
    if b"\x00" in data:
        raise WorkspaceError(f"binary file cannot be read as text: {path}")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WorkspaceError(f"file is not utf-8 text: {path}") from exc


def _fs_list(workspace: LocalWorkspaceBackend) -> ToolSpec:
    def handler(_context: ToolContext, args: dict[str, Any]) -> ToolResult:
        entries = workspace.list_entries(
            args.get("path", "."),
            recursive=bool(args.get("recursive", False)),
            max_entries=int(args.get("max_entries", 200)),
        )
        return ToolResult(
            ok=True,
            content={"entries": [entry.__dict__ for entry in entries], "count": len(entries)},
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


def _fs_tree(workspace: LocalWorkspaceBackend) -> ToolSpec:
    def handler(_context: ToolContext, args: dict[str, Any]) -> ToolResult:
        path = args.get("path", ".")
        depth = int(args.get("depth", 2))
        max_entries = int(args.get("max_entries", 200))
        entries = workspace.list_entries(path, recursive=True, max_entries=max_entries)
        root_depth = 0 if path in {None, "", "."} else len(workspace.normalize(path).split("/"))
        lines: list[str] = []
        for entry in entries:
            entry_depth = len(entry.path.split("/")) - root_depth
            if entry_depth > depth:
                continue
            indent = "  " * max(0, entry_depth - 1)
            suffix = "/" if entry.kind == "dir" else ""
            lines.append(f"{indent}{entry.path.split('/')[-1]}{suffix}")
        return ToolResult(ok=True, content={"tree": "\n".join(lines), "entries": len(lines)})

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


def _fs_stat(workspace: LocalWorkspaceBackend) -> ToolSpec:
    def handler(_context: ToolContext, args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        rel, abs_path = workspace.resolve_existing_or_parent(path)
        kind = workspace._effective_kind(rel, abs_path)
        if kind is None:
            return ToolResult(ok=True, content={"path": rel, "exists": False})
        if kind == "file":
            data, _digest = workspace.read_bytes(rel)
            return ToolResult(ok=True, content={"path": rel, "exists": True, "kind": "file", "size": len(data)})
        return ToolResult(
            ok=True,
            content={
                "path": rel,
                "exists": True,
                "kind": "dir",
                "size": 0,
            },
        )

    return ToolSpec(
        id="fs.stat",
        description="Return metadata for a workspace path.",
        input_schema=_object_schema({"path": {"type": "string"}}, required=["path"]),
        capability="fs.read",
        side_effect="read",
        handler=handler,
        path_args=("path",),
    )


def _fs_read(workspace: LocalWorkspaceBackend) -> ToolSpec:
    def handler(_context: ToolContext, args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        max_bytes = int(args.get("max_bytes", workspace.max_bytes_read))
        data, digest = workspace.read_bytes(path, max_bytes=max_bytes)
        text = _text_from_bytes(data, path)
        lines = text.splitlines()
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        if start_line is not None or end_line is not None:
            start = max(1, int(start_line or 1))
            end = min(len(lines), int(end_line or len(lines)))
            selected = "\n".join(lines[start - 1 : end])
        else:
            selected = text
        return ToolResult(
            ok=True,
            content={"path": workspace.normalize(path), "content": selected, "sha256": digest, "size": len(data)},
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


def _fs_glob(workspace: LocalWorkspaceBackend) -> ToolSpec:
    def handler(_context: ToolContext, args: dict[str, Any]) -> ToolResult:
        matches = workspace.glob(
            str(args["pattern"]),
            root=str(args.get("root", ".")),
            max_matches=int(args.get("max_matches", 200)),
        )
        return ToolResult(ok=True, content={"matches": matches, "count": len(matches)})

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


def _text_search(workspace: LocalWorkspaceBackend) -> ToolSpec:
    def handler(_context: ToolContext, args: dict[str, Any]) -> ToolResult:
        pattern = str(args["pattern"])
        root = str(args.get("root", "."))
        file_glob = args.get("file_glob")
        case_sensitive = bool(args.get("case_sensitive", False))
        max_matches = int(args.get("max_matches", 100))
        needle = pattern if case_sensitive else pattern.lower()
        matches: list[dict[str, Any]] = []
        for rel in workspace.text_files(root=root, file_glob=file_glob):
            if len(matches) >= max_matches:
                break
            try:
                data, _digest = workspace.read_bytes(rel)
                text = _text_from_bytes(data, rel)
            except WorkspaceError:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                haystack = line if case_sensitive else line.lower()
                if needle in haystack or fnmatch.fnmatch(haystack, needle):
                    matches.append({"path": rel, "line": lineno, "text": line})
                    if len(matches) >= max_matches:
                        break
        return ToolResult(ok=True, content={"matches": matches, "count": len(matches)})

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


def _fs_write(workspace: LocalWorkspaceBackend) -> ToolSpec:
    def handler(_context: ToolContext, args: dict[str, Any]) -> ToolResult:
        path = str(args["path"])
        content = str(args["content"])
        digest = workspace.write_bytes(
            path,
            content.encode("utf-8"),
            create_dirs=bool(args.get("create_dirs", False)),
            expected_sha256=args.get("expected_sha256"),
        )
        return ToolResult(ok=True, content={"path": workspace.normalize(path), "sha256": digest})

    return ToolSpec(
        id="fs.write",
        description="Write a UTF-8 text file in the workspace.",
        input_schema=_object_schema(
            {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "create_dirs": {"type": "boolean", "default": False},
                "expected_sha256": {"type": ["string", "null"]},
            },
            required=["path", "content"],
        ),
        capability="fs.write",
        side_effect="write",
        handler=handler,
        path_args=("path",),
    )


def _fs_patch(workspace: LocalWorkspaceBackend) -> ToolSpec:
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
        for replacement in replacements:
            old = str(replacement["old"])
            new = str(replacement["new"])
            count = int(replacement.get("count", 1))
            occurrences = changed.count(old)
            if occurrences < count:
                raise WorkspaceError(f"patch text not found enough times in {workspace.normalize(path)}")
            changed = changed.replace(old, new, count)
            applied += count
        new_digest = workspace.write_bytes(
            path,
            changed.encode("utf-8"),
            expected_sha256=digest,
        )
        return ToolResult(
            ok=True,
            content={"path": workspace.normalize(path), "sha256": new_digest, "replacements": applied},
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


def _fs_mkdir(workspace: LocalWorkspaceBackend) -> ToolSpec:
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
        },
        required=required,
    )


def _fs_copy(workspace: LocalWorkspaceBackend) -> ToolSpec:
    def handler(_context: ToolContext, args: dict[str, Any]) -> ToolResult:
        result = workspace.copy_path(
            str(args["source_path"]),
            str(args["destination_path"]),
            overwrite=bool(args.get("overwrite", False)),
            create_dirs=bool(args.get("create_dirs", False)),
            recursive=bool(args.get("recursive", False)),
            max_entries=int(args.get("max_entries", 1000)),
            max_bytes=int(args.get("max_bytes", 50_000_000)),
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


def _fs_move(workspace: LocalWorkspaceBackend) -> ToolSpec:
    def handler(_context: ToolContext, args: dict[str, Any]) -> ToolResult:
        result = workspace.move_path(
            str(args["source_path"]),
            str(args["destination_path"]),
            overwrite=bool(args.get("overwrite", False)),
            create_dirs=bool(args.get("create_dirs", False)),
            recursive=bool(args.get("recursive", False)),
            max_entries=int(args.get("max_entries", 1000)),
            max_bytes=int(args.get("max_bytes", 50_000_000)),
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


def _fs_delete(workspace: LocalWorkspaceBackend) -> ToolSpec:
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


def _web_search() -> ToolSpec:
    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content=context.execute_web_search(args))

    return ToolSpec(
        id="web.search",
        description="Search the web through the configured CSP WebGateway.",
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
    )


def _web_fetch() -> ToolSpec:
    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content=context.execute_web_fetch(args))

    return ToolSpec(
        id="web.fetch",
        description="Fetch one web page through the configured CSP WebGateway.",
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
    )


def _web_context() -> ToolSpec:
    def handler(context: ToolContext, args: dict[str, Any]) -> ToolResult:
        return ToolResult(ok=True, content=context.execute_web_context(args))

    return ToolSpec(
        id="web.context",
        description="Build LLM-ready web grounding context through the configured CSP WebGateway.",
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
