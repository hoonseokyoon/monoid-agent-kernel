"""Native agent runner — core engine and stable integration contracts.

Importing this package exposes only the core runner and its integration contracts (see
``native_agent_runner.contracts`` and ``docs/CONTRACTS.md``).

The reference example services live under ``native_agent_runner.reference`` and are reached
explicitly, e.g. ``from native_agent_runner.reference.backend import RunnerBackend``. They are
examples, not part of the supported public surface.
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
