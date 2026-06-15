from __future__ import annotations

from dataclasses import dataclass, field

from native_agent_runner.providers.base import ModelRequest, ModelTurn, ToolCall


@dataclass
class FakeModelAdapter:
    turns: list[ModelTurn] = field(default_factory=list)
    requests: list[ModelRequest] = field(default_factory=list)

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        if not self.turns:
            return ModelTurn(final_text="fake model completed")
        return self.turns.pop(0)


def fake_tool_call(name: str, arguments: dict[str, object], call_id: str = "call_fake") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=dict(arguments))

