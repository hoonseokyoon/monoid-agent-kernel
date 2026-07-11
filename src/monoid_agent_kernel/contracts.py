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
    CheckedCheckpointStore,
    CheckpointRecord,
    CheckpointStore,
    RunCheckpoint,
)
from monoid_agent_kernel.core.durable_codec import ArtifactVersion, DurableLoadResult, DurableLoadStatus
from monoid_agent_kernel.core.compatibility import (
    PUBLIC_ARTIFACT_COMPATIBILITY,
    PUBLIC_COMPATIBILITY_ALIASES,
    CompatibilityAlias,
    CompatibilityArtifact,
    compatibility_artifact,
    compatibility_registry,
)

# Workspace seam (the file-storage surface the engine works through; the local backend is
# the default, integrators supply their own via AgentLoop.workspace_factory)
from monoid_agent_kernel.core.workspace import (
    ChangedEntry,
    FileEntry,
    Workspace,
    WorkspaceFactory,
)

# Session lifecycle (formal FSM + AgentSession contract)
from monoid_agent_kernel.core.lifecycle import (
    AgentSession,
    LoopSession,
    SessionHealth,
    SessionInspection,
    SessionState,
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
)

# Outbox request (capability-gated durable egress; the edge drains via an OutboxSender)
from monoid_agent_kernel.core.outbox import (
    OUTBOX_REQUEST_VERSION,
    OutboxReceipt,
    OutboxRequest,
    OutboxSender,
)

# Capability request/lease (scoped, short-lived access; secrets stay outside the core)
from monoid_agent_kernel.core.capability import (
    CAPABILITY_LEASE_VERSION,
    CAPABILITY_REQUEST_VERSION,
    CapabilityBroker,
    CapabilityDenial,
    CapabilityLease,
    CapabilityPending,
    CapabilityRequest,
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
    SubagentDefinition,
    ToolBinding,
    ToolSearchConfig,
)
from monoid_agent_kernel.core.tool_surface import (
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
    AudioPart,
    ContentPart,
    DocumentPart,
    ImagePart,
    TextPart,
    VideoPart,
)

# Model adapter contract
from monoid_agent_kernel.providers.base import (
    AsyncModelAdapter,
    ModelAdapter,
    ModelRequest,
    ModelStreamChunk,
    ModelTurn,
    StreamingModelAdapter,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolObservation,
    TurnComplete,
)
from monoid_agent_kernel.core.streaming import RunStream

# Tool contract
from monoid_agent_kernel.tools.base import (
    AsyncToolHandler,
    DynamicToolProvider,
    SyncToolHandler,
    ToolContext,
    ToolHandler,
    ToolProvider,
    ToolRegistry,
    ToolResult,
    ToolSideEffect,
    ToolSpec,
)
from monoid_agent_kernel.tools.decorator import tool

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

# Agent Skills (progressive disclosure) — context + tool provider
from monoid_agent_kernel.skills import SkillDefinition, SkillProvider


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
    "CheckedCheckpointStore",
    "CheckpointRecord",
    "ArtifactVersion",
    "DurableLoadResult",
    "DurableLoadStatus",
    "CompatibilityAlias",
    "CompatibilityArtifact",
    "PUBLIC_ARTIFACT_COMPATIBILITY",
    "PUBLIC_COMPATIBILITY_ALIASES",
    "compatibility_artifact",
    "compatibility_registry",
    # workspace seam (file-storage surface; the local backend is the default)
    "Workspace",
    "WorkspaceFactory",
    "FileEntry",
    "ChangedEntry",
    # session lifecycle (formal FSM + AgentSession contract)
    "SessionState",
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
    # outbox request (capability-gated durable egress)
    "OUTBOX_REQUEST_VERSION",
    "OutboxRequest",
    "OutboxReceipt",
    "OutboxSender",
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
    # context providers
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
    # multimodal input content parts (contract-only)
    "ContentPart",
    "TextPart",
    "ImagePart",
    "DocumentPart",
    "AudioPart",
    "VideoPart",
    # model adapter contract
    "ModelAdapter",
    "AsyncModelAdapter",
    "StreamingModelAdapter",
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
    # tool contract
    "SyncToolHandler",
    "AsyncToolHandler",
    "DynamicToolProvider",
    "ToolContext",
    "ToolHandler",
    "ToolProvider",
    "ToolRegistry",
    "ToolResult",
    "ToolSideEffect",
    "ToolSpec",
    "tool",
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
    # agent skills (progressive disclosure)
    "SkillDefinition",
    "SkillProvider",
]
