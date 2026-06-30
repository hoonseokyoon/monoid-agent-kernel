"""Stable integration contracts for Monoid Agent Kernel.

This module is the single, stable surface an integrating system should depend on.
Import the engine entry point, the data specs, and the protocols you must implement
(model adapter, tools, event sink) from here.

The ``monoid_agent_kernel.reference`` package (backend, llm_gateway, web_gateway)
contains reference implementations of these contracts. Real integrators build their
own services against the types exported here.

See ``docs/CONTRACTS.md`` for the full Python and HTTP wire contracts.
"""

from __future__ import annotations

# Engine entry / result
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.core.spec import (
    AgentRunSpec,
    ModelConfig,
    ModelRetryConfig,
    ReasoningConfig,
    RunLimits,
)
from monoid_agent_kernel.core.result import AgentArtifact, AgentRunResult, AgentTurnResult, Suspension
from monoid_agent_kernel.core.checkpoint import (
    CheckpointRecord,
    CheckpointStore,
    LocalFsCheckpointStore,
    RunCheckpoint,
    read_checkpoint,
    write_checkpoint,
)

# Workspace seam (the file-storage surface the engine works through; the local backend is
# the default, integrators supply their own via AgentLoop.workspace_factory)
from monoid_agent_kernel.core.workspace import (
    ChangedEntry,
    FileEntry,
    Workspace,
    WorkspaceFactory,
)
from monoid_agent_kernel.workspace.local import default_local_workspace_factory

# Session lifecycle (formal FSM + AgentSession contract)
from monoid_agent_kernel.core.lifecycle import (
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
from monoid_agent_kernel.core.control import (
    CONTROL_PROTOCOL_VERSION,
    ControlCommand,
    ControlCommandType,
    ControlDispatcher,
    ControlResult,
)

# Inbox message envelope (provenance + idempotent ingress; an edge/transport contract)
from monoid_agent_kernel.core.inbox import (
    INBOX_PROTOCOL_VERSION,
    InboxMessage,
    is_inbox_envelope,
)

# Outbox request (capability-gated durable egress; the edge drains via an OutboxSender)
from monoid_agent_kernel.core.outbox import (
    OUTBOX_REQUEST_VERSION,
    OutboxReceipt,
    OutboxRequest,
    OutboxSender,
)

# W3C Trace Context helpers (observability metadata carried on the envelopes)
from monoid_agent_kernel.core.trace_context import (
    child_traceparent,
    new_traceparent,
    parse_traceparent,
    trace_id_of,
)

# Capability request/lease (scoped, short-lived access; secrets stay outside the core)
from monoid_agent_kernel.core.capability import (
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
from monoid_agent_kernel.core.context import ContextProvider, TurnContext

# Output validation (post-response conformance; checked at the settle points)
from monoid_agent_kernel.core.output_validator import (
    FinalOutputView,
    OutputRetry,
    OutputValidator,
    OutputValidatorError,
    ValidationOutcome,
)
from monoid_agent_kernel.core.agents import (
    AgentDefinition,
    AgentRuntimeConfig,
    BoundTool,
    BoundToolCatalog,
    OutputValidatorBinding,
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
from monoid_agent_kernel.core.tool_surface import (
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
from monoid_agent_kernel.core.content import (
    ContentPart,
    DocumentPart,
    ImagePart,
    TextPart,
)

# Model adapter contract
from monoid_agent_kernel.providers.base import (
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
from monoid_agent_kernel.core.streaming import RunStream

# Tool contract
from monoid_agent_kernel.tools.base import (
    DynamicToolProvider,
    ToolContext,
    ToolHandler,
    ToolProvider,
    ToolRegistry,
    ToolResult,
    ToolSideEffect,
    ToolSpec,
)
from monoid_agent_kernel.tools.decorator import tool
from monoid_agent_kernel.tools import tool_ids
from monoid_agent_kernel.tools.tool_ids import list_builtin_tools

# Event contract
from monoid_agent_kernel.core.events import (
    EVENT_SCHEMA_VERSION,
    AgentEvent,
    AgentEventLevel,
    AgentEventType,
    EventSink,
)

# Permission boundary
from monoid_agent_kernel.permissions import PermissionPolicy

# Gateway client seam (the contract the core calls across for web tools)
from monoid_agent_kernel.web import WebGatewayClient

# Async task seams (executor/injector/reporter the backend plugs in)
from monoid_agent_kernel.tasks import (
    ResultInjector,
    SubagentTaskExecutor,
    TaskExecutor,
    TaskReporter,
)

# Subagent (agent-as-tool) delegation tool + file discovery
from monoid_agent_kernel.core.frontmatter import parse_frontmatter
from monoid_agent_kernel.subagent_loader import load_subagent_definitions
from monoid_agent_kernel.tools.builtin import agent_spawn_tool

# Agent Skills (progressive disclosure) — context + tool provider, plus file discovery
from monoid_agent_kernel.skills import SkillDefinition, SkillProvider, load_skill_definitions


class core:  # noqa: N801 - intentional lowercase: a curated namespace, not a class to instantiate
    """The must-know contracts, curated. Import this instead of wading through the full ~130-name
    surface::

        from monoid_agent_kernel.contracts import core
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
    "OutputValidator",
    "OutputValidatorBinding",
    "ValidationOutcome",
    "FinalOutputView",
    "OutputRetry",
    "OutputValidatorError",
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
    # NOTE: the curated `core` namespace is intentionally NOT in __all__ — re-exporting it would
    # shadow the `monoid_agent_kernel.core` package at the root. Reach it via
    # `from monoid_agent_kernel.contracts import core`.
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
