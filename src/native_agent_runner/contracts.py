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
from native_agent_runner.core.result import AgentArtifact, AgentRunResult

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
    ToolContext,
    ToolHandler,
    ToolProvider,
    ToolRegistry,
    ToolResult,
    ToolSideEffect,
    ToolSpec,
)

# Event contract
from native_agent_runner.core.events import (
    EVENT_SCHEMA_VERSION,
    AgentEvent,
    AgentEventLevel,
    AgentEventType,
    EventSink,
)

# Policy contracts
from native_agent_runner.permissions import PermissionPolicy
from native_agent_runner.shell import ShellPolicy
from native_agent_runner.tools.policy import ToolPolicy
from native_agent_runner.web import WebPolicy

# Gateway client seam (the contract the core calls across for web tools)
from native_agent_runner.web import WebGatewayClient

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
    # model adapter contract
    "ModelAdapter",
    "ModelRequest",
    "ModelTurn",
    "ToolCall",
    "ToolObservation",
    # tool contract
    "ToolContext",
    "ToolHandler",
    "ToolProvider",
    "ToolRegistry",
    "ToolResult",
    "ToolSideEffect",
    "ToolSpec",
    # event contract
    "EVENT_SCHEMA_VERSION",
    "AgentEvent",
    "AgentEventLevel",
    "AgentEventType",
    "EventSink",
    # policy contracts
    "PermissionPolicy",
    "ShellPolicy",
    "ToolPolicy",
    "WebPolicy",
    # gateway client seam
    "WebGatewayClient",
]
