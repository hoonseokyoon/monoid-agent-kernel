"""Named constants for the builtin tool ids, plus a discovery helper.

Configs and bindings reference tools by id (e.g. ``RegistryToolRef("fs.write")``). These
constants give one source of truth and IDE autocomplete instead of bare, typo-prone strings;
a wrong id otherwise only surfaces at config-validation time. The set is kept in lockstep with
``builtin_tools`` by ``tests/test_tool_ids.py``.
"""

from __future__ import annotations

from native_agent_runner.core.workspace import Workspace
from native_agent_runner.tools.base import ToolSpec
from native_agent_runner.tools.builtin import builtin_tools

# Filesystem
FS_LIST = "fs.list"
FS_TREE = "fs.tree"
FS_STAT = "fs.stat"
FS_READ = "fs.read"
FS_READ_MEDIA = "fs.read_media"
FS_GLOB = "fs.glob"
FS_WRITE = "fs.write"
FS_PATCH = "fs.patch"
FS_MKDIR = "fs.mkdir"
FS_COPY = "fs.copy"
FS_MOVE = "fs.move"
FS_DELETE = "fs.delete"

# Text search + tool discovery
TEXT_SEARCH = "text.search"
TOOL_SEARCH = "tool.search"

# Shell + background jobs
SHELL_EXEC = "shell.exec"
JOB_LIST = "job.list"
JOB_STATUS = "job.status"
JOB_LOGS = "job.logs"
JOB_CANCEL = "job.cancel"
JOB_WAIT = "job.wait"

# Human-in-the-loop + web (web tools require a WebGateway binding)
HITL_REQUEST = "hitl.request"
WEB_SEARCH = "web.search"
WEB_FETCH = "web.fetch"
WEB_CONTEXT = "web.context"

# Artifacts + run control
ARTIFACT_EMIT = "artifact.emit"
ARTIFACT_LIST = "artifact.list"
RUN_UPDATE_PLAN = "run.update_plan"
RUN_FINISH = "run.finish"

# Subagent delegation. Registered by AgentLoop only when subagent_definitions are loaded, so it
# is NOT part of builtin_tools(); the constant is here for binding configs that enable delegation.
AGENT_SPAWN = "agent.spawn"


def list_builtin_tools(workspace: Workspace | None = None) -> list[ToolSpec]:
    """Return the builtin tool specs. ``workspace`` is only used by tool *handlers* at call time,
    so it may be omitted for pure discovery (id/description/schema) — the same discovery path the
    reference backend uses to validate configs."""
    return builtin_tools(workspace)  # type: ignore[arg-type]
