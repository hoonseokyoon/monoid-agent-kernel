from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


EXPECTED_CONTRACTS_ALL = [
    "AgentLoop",
    "AgentRunSpec",
    "ModelConfig",
    "ModelRetryConfig",
    "ReasoningConfig",
    "RunLimits",
    "AgentArtifact",
    "AgentRunResult",
    "AgentTurnResult",
    "Suspension",
    "RunCheckpoint",
    "CheckpointStore",
    "CheckpointRecord",
    "Workspace",
    "WorkspaceFactory",
    "FileEntry",
    "ChangedEntry",
    "SessionState",
    "AgentSession",
    "LoopSession",
    "SessionInspection",
    "SessionHealth",
    "CONTROL_PROTOCOL_VERSION",
    "ControlCommand",
    "ControlCommandType",
    "INBOX_PROTOCOL_VERSION",
    "InboxMessage",
    "OUTBOX_REQUEST_VERSION",
    "OutboxRequest",
    "OutboxReceipt",
    "OutboxSender",
    "ControlResult",
    "ControlDispatcher",
    "CAPABILITY_REQUEST_VERSION",
    "CAPABILITY_LEASE_VERSION",
    "CapabilityRequest",
    "CapabilityLease",
    "CapabilityDenial",
    "CapabilityPending",
    "CapabilityBroker",
    "AgentDefinition",
    "AgentRuntimeConfig",
    "BoundTool",
    "BoundToolCatalog",
    "PromptSpec",
    "RegistryToolRef",
    "RuntimeConfigProvider",
    "RuntimeConfigSource",
    "ToolBinding",
    "ToolSearchConfig",
    "ContextProvider",
    "TurnContext",
    "OutputValidator",
    "OutputValidatorBinding",
    "ValidationOutcome",
    "FinalOutputView",
    "OutputRetry",
    "OutputValidatorError",
    "ToolAuthorization",
    "ToolGuidance",
    "ToolQuota",
    "ToolScope",
    "ToolSearchEntry",
    "ToolSurfaceResolver",
    "ToolSurfaceSnapshot",
    "ContentPart",
    "TextPart",
    "ImagePart",
    "DocumentPart",
    "AudioPart",
    "VideoPart",
    "ModelAdapter",
    "ModelRequest",
    "ModelTurn",
    "ToolCall",
    "ToolObservation",
    "RunStream",
    "ModelStreamChunk",
    "TextDelta",
    "ToolCallDelta",
    "TurnComplete",
    "DynamicToolProvider",
    "ToolContext",
    "ToolHandler",
    "ToolProvider",
    "ToolRegistry",
    "ToolResult",
    "ToolSideEffect",
    "ToolSpec",
    "tool",
    "EVENT_SCHEMA_VERSION",
    "AgentEvent",
    "AgentEventLevel",
    "AgentEventType",
    "EventSink",
    "PermissionPolicy",
    "WebGatewayClient",
    "TaskExecutor",
    "ResultInjector",
    "TaskReporter",
    "SubagentDefinition",
    "SubagentTaskExecutor",
    "SkillDefinition",
    "SkillProvider",
]

REMOVED_PUBLIC_SURFACE_NAMES = [
    "LocalFsCheckpointStore",
    "read_checkpoint",
    "write_checkpoint",
    "default_local_workspace_factory",
    "LEGAL_TRANSITIONS",
    "can_transition",
    "assert_transition",
    "state_from_suspension",
    "to_session_state",
    "parse_traceparent",
    "new_traceparent",
    "child_traceparent",
    "trace_id_of",
    "CapabilityVault",
    "AutoGrantBroker",
    "StaticRuntimeConfigProvider",
    "static_runtime_config",
    "coerce_runtime_config_provider",
    "compile_bound_tool_catalog",
    "generated_tool_bindings",
    "validate_runtime_config",
    "collect_runtime_config_issues",
    "DefaultToolSurfaceResolver",
    "is_inbox_envelope",
    "assemble_streamed_turn",
    "tool_ids",
    "list_builtin_tools",
    "agent_spawn_tool",
    "load_subagent_definitions",
    "parse_frontmatter",
    "load_skill_definitions",
    "FakeModelAdapter",
    "FakeStreamingModelAdapter",
    "GatewayModelAdapter",
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


def test_contracts_public_surface_is_intentional() -> None:
    import monoid_agent_kernel.contracts as contracts

    assert contracts.__all__ == EXPECTED_CONTRACTS_ALL


def test_package_root_mirrors_contracts_surface() -> None:
    import monoid_agent_kernel as root
    import monoid_agent_kernel.contracts as contracts

    assert root.__all__ == contracts.__all__


def test_helpers_and_conveniences_are_not_root_or_contract_exports() -> None:
    import monoid_agent_kernel as root
    import monoid_agent_kernel.contracts as contracts

    for name in REMOVED_PUBLIC_SURFACE_NAMES:
        assert not hasattr(contracts, name), name
        assert not hasattr(root, name), name


def test_memory_surface_is_explicit_module_only() -> None:
    import monoid_agent_kernel as root
    import monoid_agent_kernel.contracts as contracts
    import monoid_agent_kernel.memory as memory

    assert memory.__all__ == [
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
    assert memory.MEMORY_TOOL_IDS == (
        "memory.search",
        "memory.view",
        "memory.create",
        "memory.str_replace",
        "memory.insert",
        "memory.delete",
        "memory.rename",
    )

    for name in memory.__all__:
        assert not hasattr(contracts, name), name
        assert not hasattr(root, name), name


def test_removed_names_remain_available_from_explicit_modules() -> None:
    from monoid_agent_kernel.core.agents import (
        coerce_runtime_config_provider,
        collect_runtime_config_issues,
        compile_bound_tool_catalog,
        generated_tool_bindings,
        static_runtime_config,
        validate_runtime_config,
    )
    from monoid_agent_kernel.core.capability import AutoGrantBroker, CapabilityVault
    from monoid_agent_kernel.core.checkpoint import (
        LocalFsCheckpointStore,
        read_checkpoint,
        write_checkpoint,
    )
    from monoid_agent_kernel.core.frontmatter import parse_frontmatter
    from monoid_agent_kernel.core.inbox import is_inbox_envelope
    from monoid_agent_kernel.core.lifecycle import (
        LEGAL_TRANSITIONS,
        assert_transition,
        can_transition,
        state_from_suspension,
        to_session_state,
    )
    from monoid_agent_kernel.core.packages import (
        apply_package,
        create_approval,
        export_package,
        import_package,
        verify_package,
    )
    from monoid_agent_kernel.core.projections import project_run_status
    from monoid_agent_kernel.core.trace_context import (
        child_traceparent,
        new_traceparent,
        parse_traceparent,
        trace_id_of,
    )
    from monoid_agent_kernel.core.tool_surface import DefaultToolSurfaceResolver
    from monoid_agent_kernel.mcp import McpError, McpToolProvider
    from monoid_agent_kernel.narration import EventNarration, narrate_event
    from monoid_agent_kernel.observability.otel import OtelEventSink
    from monoid_agent_kernel.providers.base import assemble_streamed_turn
    from monoid_agent_kernel.providers.fake import FakeModelAdapter
    from monoid_agent_kernel.providers.gateway import GatewayModelAdapter
    from monoid_agent_kernel.providers.openai import OpenAIModelAdapter
    from monoid_agent_kernel.recorder import (
        JsonlEventSink,
        MemoryEventSink,
        StatusJsonSink,
        StdoutJsonlSink,
    )
    from monoid_agent_kernel.skills import load_skill_definitions
    from monoid_agent_kernel.subagent_loader import load_subagent_definitions
    from monoid_agent_kernel.tasks import (
        TaskManager,
        get_job_artifact,
        list_job_artifacts,
        read_job_log_text,
        request_job_cancel,
    )
    from monoid_agent_kernel.tools import tool_ids
    from monoid_agent_kernel.tools.builtin import agent_spawn_tool
    from monoid_agent_kernel.tools.tool_ids import list_builtin_tools
    from monoid_agent_kernel.workspace.local import default_local_workspace_factory

    explicit_module_names = (
        LocalFsCheckpointStore,
        read_checkpoint,
        write_checkpoint,
        default_local_workspace_factory,
        LEGAL_TRANSITIONS,
        can_transition,
        assert_transition,
        state_from_suspension,
        to_session_state,
        parse_traceparent,
        new_traceparent,
        child_traceparent,
        trace_id_of,
        CapabilityVault,
        AutoGrantBroker,
        static_runtime_config,
        coerce_runtime_config_provider,
        compile_bound_tool_catalog,
        generated_tool_bindings,
        validate_runtime_config,
        collect_runtime_config_issues,
        DefaultToolSurfaceResolver,
        is_inbox_envelope,
        assemble_streamed_turn,
        tool_ids,
        list_builtin_tools,
        agent_spawn_tool,
        load_subagent_definitions,
        parse_frontmatter,
        load_skill_definitions,
        FakeModelAdapter,
        GatewayModelAdapter,
        OpenAIModelAdapter,
        apply_package,
        create_approval,
        export_package,
        import_package,
        verify_package,
        project_run_status,
        narrate_event,
        EventNarration,
        OtelEventSink,
        McpToolProvider,
        McpError,
        TaskManager,
        get_job_artifact,
        list_job_artifacts,
        read_job_log_text,
        request_job_cancel,
        JsonlEventSink,
        MemoryEventSink,
        StatusJsonSink,
        StdoutJsonlSink,
    )
    assert all(name is not None for name in explicit_module_names)


def test_audio_and_video_parts_are_public_contracts() -> None:
    import monoid_agent_kernel as root
    import monoid_agent_kernel.contracts as contracts
    from monoid_agent_kernel.core.content import AudioPart, VideoPart

    assert contracts.AudioPart is AudioPart
    assert contracts.VideoPart is VideoPart
    assert root.AudioPart is AudioPart
    assert root.VideoPart is VideoPart


def test_legacy_namespace_mirrors_narrow_contract_surface() -> None:
    for module_name in list(sys.modules):
        if module_name == "native_agent_runner" or module_name.startswith("native_agent_runner."):
            sys.modules.pop(module_name)

    import monoid_agent_kernel as root

    with pytest.warns(DeprecationWarning, match="monoid_agent_kernel"):
        legacy = importlib.import_module("native_agent_runner")

    assert legacy.__all__ == root.__all__
    assert legacy.AudioPart is root.AudioPart
    assert legacy.VideoPart is root.VideoPart
    assert not hasattr(legacy, "FakeModelAdapter")


def test_root_import_keeps_reference_and_optional_providers_lazy() -> None:
    root = Path(__file__).resolve().parents[1]
    src = str(root / "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    code = """
import sys
import monoid_agent_kernel
blocked = [
    name for name in sys.modules
    if name.startswith('monoid_agent_kernel.reference.')
    or name in {'openai', 'httpx', 'opentelemetry'}
    or name.startswith('openai.')
    or name.startswith('httpx.')
    or name.startswith('opentelemetry.')
]
if blocked:
    raise SystemExit('unexpected imports: ' + ', '.join(sorted(blocked)))
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
