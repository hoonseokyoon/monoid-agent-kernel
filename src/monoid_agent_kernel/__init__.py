"""Monoid Agent Kernel: the contracts and core engine that implements them.

The package is layered in three tiers (see ``docs/CONTRACTS.md`` and the README):

- **contracts** — the stable integration surface (``monoid_agent_kernel.contracts``): the specs
  and protocols you depend on and implement.
- **core** — the engine that implements those contracts (``AgentLoop`` and friends), the default
  supported runner.
- **reference** — example services under ``monoid_agent_kernel.reference``, reached explicitly,
  e.g. ``from monoid_agent_kernel.reference.backend import RunnerBackend``. They are examples,
  not part of the supported public surface.

Importing this package exposes the contracts plus the core conveniences.
"""

from monoid_agent_kernel import contracts as contracts
from monoid_agent_kernel.contracts import *  # noqa: F401,F403

# Core conveniences (not contract types, but part of the core runner surface).
from monoid_agent_kernel.providers.gateway import GatewayModelAdapter
from monoid_agent_kernel.providers.fake import FakeModelAdapter, FakeStreamingModelAdapter
from monoid_agent_kernel.providers.openai import OpenAIModelAdapter
from monoid_agent_kernel.core.packages import (
    apply_package,
    create_approval,
    export_package,
    import_package,
    verify_package,
)
from monoid_agent_kernel.core.projections import project_run_status
from monoid_agent_kernel.narration import EventNarration, narrate_event
from monoid_agent_kernel.observability import OtelEventSink
from monoid_agent_kernel.mcp import McpError, McpToolProvider
from monoid_agent_kernel.tasks import (
    TaskManager,
    get_job_artifact,
    list_job_artifacts,
    read_job_log_text,
    request_job_cancel,
)
from monoid_agent_kernel.recorder import (
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
