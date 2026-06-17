from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from native_agent_runner.core.spec import ModelConfig
from native_agent_runner.tools.base import ToolSpec


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolObservation:
    call_id: str
    tool_name: str
    output: dict[str, Any]
    is_background: bool = False


@dataclass(frozen=True)
class ModelTurn:
    response_id: str | None = None
    final_text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRequest:
    instruction: str
    system_prompt: str
    tools: tuple[ToolSpec, ...]
    previous_turn_handle: str | None = None
    observations: tuple[ToolObservation, ...] = ()
    model: ModelConfig | None = None


class ModelAdapter(Protocol):
    # Optional capability flag. The loop reads it via
    # ``getattr(adapter, "supports_multimodal", False)``; an adapter that can
    # accept non-text content parts sets it True. Defaulting off keeps existing
    # adapters valid without declaring it. Multimodal forwarding itself is not
    # yet implemented (see core/content.py) — this is the negotiation seam.
    supports_multimodal: bool = False

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        ...

