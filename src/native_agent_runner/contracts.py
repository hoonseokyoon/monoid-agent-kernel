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

# Workspace seam (the file-storage surface the engine works through; the local backend is
# the default, integrators supply their own via AgentLoop.workspace_factory)
from native_agent_runner.core.workspace import (
    ChangedEntry,
    FileEntry,
    Workspace,
    WorkspaceFactory,
)
from native_agent_runner.workspace.local import default_local_workspace_factory

# Session lifecycle (formal FSM + AgentSession contract)
from native_agent_runner.core.lifecycle import (
    LEGAL_TRANSITIONS,
    AgentSession,
    LoopSession,
    SessionHealth,
    SessionInspection,
    SessionState,
    assert_transition,
    can_transition,
    state_from_suspension,
    to_session_state,
)

# Control protocol (transport-independent command envelope + dispatch seam)
from native_agent_runner.core.control import (
    CONTROL_PROTOCOL_VERSION,
    ControlCommand,
    ControlCommandType,
    ControlDispatcher,
    ControlResult,
)

# Inbox message envelope (provenance + idempotent ingress; an edge/transport contract)
from native_agent_runner.core.inbox import (
    INBOX_PROTOCOL_VERSION,
    InboxMessage,
    is_inbox_envelope,
)

# Outbox request (capability-gated durable egress; the edge drains via an OutboxSender)
from native_agent_runner.core.outbox import (
    OUTBOX_REQUEST_VERSION,
    OutboxReceipt,
    OutboxRequest,
    OutboxSender,
)

# W3C Trace Context helpers (observability metadata carried on the envelopes)
from native_agent_runner.core.trace_context import (
    child_traceparent,
    new_traceparent,
    parse_traceparent,
    trace_id_of,
)

# Capability request/lease (scoped, short-lived access; secrets stay outside the core)
from native_agent_runner.core.capability import (
    CAPABILITY_LEASE_VERSION,
    CAPABILITY_REQUEST_VERSION,
    AutoGrantBroker,
    CapabilityBroker,
    CapabilityDenial,
    CapabilityLease,
    CapabilityPending,
    CapabilityRequest,
    CapabilityVault,
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
    SubagentDefinition,
    ToolBinding,
    ToolSearchConfig,
    coerce_runtime_config_provider,
    collect_runtime_config_issues,
    compile_bound_tool_catalog,
    generated_tool_bindings,
    static_runtime_config,
    validate_runtime_config,
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
    ModelStreamChunk,
    ModelTurn,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolObservation,
    TurnComplete,
    assemble_streamed_turn,
)
from native_agent_runner.core.streaming import RunStream

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
from native_agent_runner.tools import tool_ids
from native_agent_runner.tools.tool_ids import list_builtin_tools

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
    SubagentTaskExecutor,
    TaskExecutor,
    TaskReporter,
)

# Subagent (agent-as-tool) delegation tool + file discovery
from native_agent_runner.core.frontmatter import parse_frontmatter
from native_agent_runner.subagent_loader import load_subagent_definitions
from native_agent_runner.tools.builtin import agent_spawn_tool

# Agent Skills (progressive disclosure) — context + tool provider, plus file discovery
from native_agent_runner.skills import SkillDefinition, SkillProvider, load_skill_definitions


class core:  # noqa: N801 - intentional lowercase: a curated namespace, not a class to instantiate
    """The must-know contracts, curated. Import this instead of wading through the full ~130-name
    surface::

        from native_agent_runner.contracts import core
        loop = core.AgentLoop.from_tools(spec, adapter, [my_tool])

    Everything here is also exported at the top level; this is the short list to learn first."""

    AgentLoop = AgentLoop
    AgentRunSpec = AgentRunSpec
    AgentRuntimeConfig = AgentRuntimeConfig
    ModelAdapter = ModelAdapter
    ToolSpec = ToolSpec
    tool = tool
    EventSink = EventSink
    Workspace = Workspace
    PermissionPolicy = PermissionPolicy


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
    # workspace seam (file-storage surface; the local backend is the default)
    "Workspace",
    "WorkspaceFactory",
    "FileEntry",
    "ChangedEntry",
    "default_local_workspace_factory",
    # session lifecycle (formal FSM + AgentSession contract)
    "SessionState",
    "LEGAL_TRANSITIONS",
    "can_transition",
    "assert_transition",
    "state_from_suspension",
    "to_session_state",
    "AgentSession",
    "LoopSession",
    "SessionInspection",
    "SessionHealth",
    # control protocol (transport-independent command envelope + dispatch seam)
    "CONTROL_PROTOCOL_VERSION",
    "ControlCommand",
    "ControlCommandType",
    # inbox message envelope (provenance + idempotent ingress)
    "INBOX_PROTOCOL_VERSION",
    "InboxMessage",
    "is_inbox_envelope",
    # outbox request (capability-gated durable egress)
    "OUTBOX_REQUEST_VERSION",
    "OutboxRequest",
    "OutboxReceipt",
    "OutboxSender",
    # W3C Trace Context helpers
    "parse_traceparent",
    "new_traceparent",
    "child_traceparent",
    "trace_id_of",
    "ControlResult",
    "ControlDispatcher",
    # capability request/lease
    "CAPABILITY_REQUEST_VERSION",
    "CAPABILITY_LEASE_VERSION",
    "CapabilityRequest",
    "CapabilityLease",
    "CapabilityDenial",
    "CapabilityPending",
    "CapabilityBroker",
    "CapabilityVault",
    "AutoGrantBroker",
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
    # streaming (astream) contract
    "RunStream",
    "ModelStreamChunk",
    "TextDelta",
    "ToolCallDelta",
    "TurnComplete",
    "assemble_streamed_turn",
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
    "tool_ids",
    "list_builtin_tools",
    # config validation
    "validate_runtime_config",
    "collect_runtime_config_issues",
    # curated must-know namespace
    "core",
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
    # subagent (agent-as-tool) delegation
    "SubagentDefinition",
    "SubagentTaskExecutor",
    "agent_spawn_tool",
    "load_subagent_definitions",
    "parse_frontmatter",
    # agent skills (progressive disclosure)
    "SkillDefinition",
    "SkillProvider",
    "load_skill_definitions",
]
