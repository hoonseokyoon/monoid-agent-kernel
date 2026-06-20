"""Stable integration contracts for native-agent-runner.

This module is the single, stable surface an integrating system should depend on.
Import the engine entry point, the data specs, and the protocols you must implement
(model adapter, tools, event sink) from here.

The ``native_agent_runner.reference`` package (backend, llm_gateway, web_gateway) is a
*reference implementation* of these contracts, not part of the supported surface — real
integrators are expected to build their own services against the types exported here.

See ``docs/CONTRACTS.md`` for the full Python and HTTP wire contracts.
"""

from __future__ import annotations

# Engine entry / result
from native_agent_runner.loop import AgentLoop
from native_agent_runner.core.spec import (
    AgentRunSpec,
    ModelConfig,
    ModelRetryConfig,
    ReasoningConfig,
    RunLimits,
)
from native_agent_runner.core.result import AgentArtifact, AgentRunResult, AgentTurnResult, Suspension
from native_agent_runner.core.checkpoint import (
    CheckpointRecord,
    CheckpointStore,
    LocalFsCheckpointStore,
    RunCheckpoint,
    read_checkpoint,
    write_checkpoint,
)

# Context providers (pluggable static + per-turn system context)
from native_agent_runner.core.context import ContextProvider, TurnContext
from native_agent_runner.core.agents import (
    AgentDefinition,
    AgentRuntimeConfig,
    BoundTool,
    BoundToolCatalog,
    PromptSpec,
    RegistryToolRef,
    RuntimeConfigProvider,
    RuntimeConfigSource,
    StaticRuntimeConfigProvider,
    ToolBinding,
    ToolSearchConfig,
    coerce_runtime_config_provider,
    compile_bound_tool_catalog,
    generated_tool_bindings,
    static_runtime_config,
)
from native_agent_runner.core.tool_surface import (
    DefaultToolSurfaceResolver,
    ToolAuthorization,
    ToolGuidance,
    ToolQuota,
    ToolScope,
    ToolSearchEntry,
    ToolSurfaceResolver,
    ToolSurfaceSnapshot,
)

# Multimodal input content parts (contract-only surface; see core/content.py)
from native_agent_runner.core.content import (
    ContentPart,
    DocumentPart,
    ImagePart,
    TextPart,
)

# Model adapter contract
from native_agent_runner.providers.base import (
    ModelAdapter,
    ModelRequest,
    ModelTurn,
    ToolCall,
    ToolObservation,
)

# Tool contract
from native_agent_runner.tools.base import (
    DynamicToolProvider,
    ToolContext,
    ToolHandler,
    ToolProvider,
    ToolRegistry,
    ToolResult,
    ToolSideEffect,
    ToolSpec,
)
from native_agent_runner.tools.decorator import tool

# Event contract
from native_agent_runner.core.events import (
    EVENT_SCHEMA_VERSION,
    AgentEvent,
    AgentEventLevel,
    AgentEventType,
    EventSink,
)

# Permission boundary
from native_agent_runner.permissions import PermissionPolicy

# Gateway client seam (the contract the core calls across for web tools)
from native_agent_runner.web import WebGatewayClient

# Async task seams (executor/injector/reporter the backend plugs in)
from native_agent_runner.tasks import (
    ResultInjector,
    TaskExecutor,
    TaskReporter,
)

__all__ = [
    # engine entry / result
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
    "LocalFsCheckpointStore",
    "read_checkpoint",
    "write_checkpoint",
    # context providers
    "AgentDefinition",
    "AgentRuntimeConfig",
    "BoundTool",
    "BoundToolCatalog",
    "PromptSpec",
    "RegistryToolRef",
    "RuntimeConfigProvider",
    "RuntimeConfigSource",
    "StaticRuntimeConfigProvider",
    "static_runtime_config",
    "coerce_runtime_config_provider",
    "ToolBinding",
    "ToolSearchConfig",
    "compile_bound_tool_catalog",
    "generated_tool_bindings",
    "ContextProvider",
    "TurnContext",
    "DefaultToolSurfaceResolver",
    "ToolAuthorization",
    "ToolGuidance",
    "ToolQuota",
    "ToolScope",
    "ToolSearchEntry",
    "ToolSurfaceResolver",
    "ToolSurfaceSnapshot",
    # multimodal input content parts (contract-only)
    "ContentPart",
    "TextPart",
    "ImagePart",
    "DocumentPart",
    # model adapter contract
    "ModelAdapter",
    "ModelRequest",
    "ModelTurn",
    "ToolCall",
    "ToolObservation",
    # tool contract
    "DynamicToolProvider",
    "ToolContext",
    "ToolHandler",
    "ToolProvider",
    "ToolRegistry",
    "ToolResult",
    "ToolSideEffect",
    "ToolSpec",
    "tool",
    # event contract
    "EVENT_SCHEMA_VERSION",
    "AgentEvent",
    "AgentEventLevel",
    "AgentEventType",
    "EventSink",
    # permission boundary
    "PermissionPolicy",
    # gateway client seam
    "WebGatewayClient",
    # async task seams
    "TaskExecutor",
    "ResultInjector",
    "TaskReporter",
]
