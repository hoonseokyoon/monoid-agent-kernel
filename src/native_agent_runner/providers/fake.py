from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import ClassVar

from native_agent_runner.providers.base import (
    ModelRequest,
    ModelStreamChunk,
    ModelTurn,
    TextDelta,
    ToolCall,
    assemble_streamed_turn,
)


@dataclass
class FakeModelAdapter:
    turns: list[ModelTurn] = field(default_factory=list)
    requests: list[ModelRequest] = field(default_factory=list)

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        if not self.turns:
            return ModelTurn(final_text="fake model completed", stop_reason="stop")
        return self.turns.pop(0)


@dataclass
class FakeMultimodalModelAdapter:
    """A multimodal ``FakeModelAdapter``: declares ``supports_multimodal`` so the loop
    resolves by-reference media to wire blocks, and records the resolved ``messages`` it
    receives so a test can assert what was forwarded."""

    turns: list[ModelTurn] = field(default_factory=list)
    requests: list[ModelRequest] = field(default_factory=list)
    supports_multimodal: ClassVar[bool] = True

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        if not self.turns:
            return ModelTurn(final_text="fake model completed", stop_reason="stop")
        return self.turns.pop(0)


@dataclass
class FakeStreamingModelAdapter:
    """A test adapter that streams: each entry in ``chunk_turns`` is one turn's chunk list.

    Exposes ``astream_turn`` (preferred by ``astream`` while a stream is active) and a sync
    ``next_turn`` fallback that returns the same assembled turn — so the same script drives
    both the streaming and non-streaming paths to an identical ``ModelTurn``.
    """

    chunk_turns: list[list[ModelStreamChunk]] = field(default_factory=list)
    requests: list[ModelRequest] = field(default_factory=list)

    def _next_chunks(self) -> list[ModelStreamChunk]:
        if not self.chunk_turns:
            return [TextDelta("fake stream completed")]
        return self.chunk_turns.pop(0)

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        return assemble_streamed_turn(self._next_chunks())

    async def astream_turn(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:
        self.requests.append(request)
        for chunk in self._next_chunks():
            await asyncio.sleep(0)
            yield chunk


def fake_tool_call(name: str, arguments: dict[str, object], call_id: str = "call_fake") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=dict(arguments))
