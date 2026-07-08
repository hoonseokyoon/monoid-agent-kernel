from __future__ import annotations

from typing import Any

from monoid_agent_kernel.core.agents import ToolBinding
from monoid_agent_kernel.core.tool_surface import ToolScope

READ_TOOL_IDS = (
    "fs.read",
    "fs.list",
    "fs.tree",
    "fs.stat",
    "fs.glob",
    "text.search",
    "fs.read_media",
)
WRITE_TOOL_IDS = ("fs.write", "fs.patch", "fs.mkdir", "fs.copy", "fs.move", "fs.delete")
SHELL_TOOL_IDS = ("shell.exec", "job.status", "job.logs", "job.cancel", "job.wait")
ARTIFACT_TOOL_IDS = ("artifact.emit", "artifact.list")
DESTRUCTIVE_WRITE_TOOL_IDS = frozenset({"fs.copy", "fs.move", "fs.delete"})

_TOOL_METADATA: dict[str, dict[str, Any]] = {
    "fs.read": {"title": "Read file", "summary": "Read UTF-8 workspace text.", "risk": "low", "tags": ("fs", "read")},
    "fs.list": {"title": "List files", "summary": "List workspace files and directories.", "risk": "low", "tags": ("fs", "read", "discover")},
    "fs.tree": {"title": "File tree", "summary": "Show a compact workspace tree.", "risk": "low", "tags": ("fs", "read", "discover")},
    "fs.stat": {"title": "File stat", "summary": "Inspect workspace path metadata.", "risk": "low", "tags": ("fs", "read", "metadata")},
    "fs.glob": {"title": "Find paths", "summary": "Find workspace paths by glob.", "risk": "low", "tags": ("fs", "read", "discover")},
    "text.search": {"title": "Search text", "summary": "Search UTF-8 workspace files.", "risk": "low", "tags": ("text", "search")},
    "fs.read_media": {"title": "Read media", "summary": "Read supported image or PDF files.", "risk": "low", "tags": ("fs", "read", "media")},
    "fs.write": {"title": "Write file", "summary": "Write UTF-8 workspace text.", "risk": "side_effect", "tags": ("fs", "write")},
    "fs.patch": {"title": "Patch file", "summary": "Apply exact text replacements.", "risk": "side_effect", "tags": ("fs", "write", "patch")},
    "fs.mkdir": {"title": "Create directory", "summary": "Create a workspace directory.", "risk": "side_effect", "tags": ("fs", "write")},
    "fs.copy": {"title": "Copy path", "summary": "Copy a workspace file or directory.", "risk": "destructive", "tags": ("fs", "write")},
    "fs.move": {"title": "Move path", "summary": "Move a workspace file or directory.", "risk": "destructive", "tags": ("fs", "write")},
    "fs.delete": {"title": "Delete path", "summary": "Delete a workspace file or directory.", "risk": "destructive", "tags": ("fs", "write", "delete")},
    "shell.exec": {"title": "Run shell", "summary": "Run a shell command in the workspace.", "risk": "high", "tags": ("shell", "execute")},
    "job.status": {"title": "Job status", "summary": "Inspect a background task.", "risk": "low", "tags": ("job", "read")},
    "job.logs": {"title": "Job logs", "summary": "Read background task logs.", "risk": "low", "tags": ("job", "read")},
    "job.cancel": {"title": "Cancel job", "summary": "Cancel a background task.", "risk": "side_effect", "tags": ("job", "write")},
    "job.wait": {"title": "Wait for job", "summary": "Wait for a background task result.", "risk": "low", "tags": ("job", "read")},
    "artifact.emit": {"title": "Emit artifact", "summary": "Register a workspace file as a run artifact.", "risk": "side_effect", "tags": ("artifact", "write")},
    "artifact.list": {"title": "List artifacts", "summary": "List run artifacts.", "risk": "low", "tags": ("artifact", "read")},
}


def default_tool_bindings(
    capability: str,
    *,
    shell_scope: ToolScope | None = None,
    shell_runtime: dict[str, Any] | None = None,
) -> tuple[ToolBinding, ...]:
    if capability == "read":
        return tuple(_binding(tool_id, group="read") for tool_id in READ_TOOL_IDS)
    if capability == "write":
        return tuple(
            _binding(
                tool_id,
                group="write",
                authorization="ask" if tool_id in DESTRUCTIVE_WRITE_TOOL_IDS else "allow",
                requires_approval=tool_id in DESTRUCTIVE_WRITE_TOOL_IDS,
            )
            for tool_id in WRITE_TOOL_IDS
        )
    if capability == "shell":
        return tuple(
            _binding(
                tool_id,
                group="shell",
                scope=shell_scope if tool_id == "shell.exec" else None,
                runtime={"shell": dict(shell_runtime or {})} if tool_id == "shell.exec" else None,
            )
            for tool_id in SHELL_TOOL_IDS
        )
    if capability == "artifact":
        return tuple(_binding(tool_id, group="artifact") for tool_id in ARTIFACT_TOOL_IDS)
    return ()


def _binding(
    tool_id: str,
    *,
    group: str,
    authorization: str = "allow",
    requires_approval: bool | None = None,
    scope: ToolScope | None = None,
    runtime: dict[str, Any] | None = None,
) -> ToolBinding:
    meta = _TOOL_METADATA.get(tool_id, {})
    tags = tuple(str(item) for item in meta.get("tags", ()))
    return ToolBinding.for_tool(
        tool_id,
        authorization=authorization,  # type: ignore[arg-type]
        scope=scope or ToolScope(),
        runtime=runtime or {},
        title=str(meta.get("title") or tool_id),
        summary=str(meta.get("summary") or ""),
        risk=str(meta.get("risk") or ""),
        requires_approval=requires_approval,
        metadata={"tool_search": {"groups": (group,), "tags": tags}},
    )
