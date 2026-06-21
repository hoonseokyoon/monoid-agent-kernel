"""Native agent runner — the contracts and the core engine that implements them.

The package is layered in three tiers (see ``docs/CONTRACTS.md`` and the README):

- **contracts** — the stable integration surface (``native_agent_runner.contracts``): the specs
  and protocols you depend on and implement.
- **core** — the engine that implements those contracts (``AgentLoop`` and friends), the default
  supported runner.
- **reference** — example services under ``native_agent_runner.reference``, reached explicitly,
  e.g. ``from native_agent_runner.reference.backend import RunnerBackend``. They are examples,
  not part of the supported public surface.

Importing this package exposes the contracts plus the core conveniences.
"""

from native_agent_runner import contracts as contracts
from native_agent_runner.contracts import *  # noqa: F401,F403

# Core conveniences (not contract types, but part of the core runner surface).
from native_agent_runner.providers.gateway import GatewayModelAdapter
from native_agent_runner.providers.fake import FakeModelAdapter, FakeStreamingModelAdapter
from native_agent_runner.providers.openai import OpenAIModelAdapter
from native_agent_runner.workspace.local import default_local_workspace_factory
from native_agent_runner.core.packages import (
    apply_package,
    create_approval,
    export_package,
    import_package,
    verify_package,
)
from native_agent_runner.core.projections import project_run_status
from native_agent_runner.narration import EventNarration, narrate_event
from native_agent_runner.observability import OtelEventSink
from native_agent_runner.mcp import McpError, McpToolProvider
from native_agent_runner.tasks import (
    TaskManager,
    get_job_artifact,
    list_job_artifacts,
    read_job_log_text,
    request_job_cancel,
)
from native_agent_runner.recorder import (
    JsonlEventSink,
    MemoryEventSink,
    StatusJsonSink,
    StdoutJsonlSink,
)

__all__ = [
    *contracts.__all__,
    # core conveniences
    "GatewayModelAdapter",
    "FakeModelAdapter",
    "FakeStreamingModelAdapter",
    "OpenAIModelAdapter",
    "default_local_workspace_factory",
    "apply_package",
    "create_approval",
    "export_package",
    "import_package",
    "verify_package",
    "project_run_status",
    "narrate_event",
    "EventNarration",
    "OtelEventSink",
    "McpToolProvider",
    "McpError",
    "TaskManager",
    "get_job_artifact",
    "list_job_artifacts",
    "read_job_log_text",
    "request_job_cancel",
    "JsonlEventSink",
    "MemoryEventSink",
    "StatusJsonSink",
    "StdoutJsonlSink",
]
