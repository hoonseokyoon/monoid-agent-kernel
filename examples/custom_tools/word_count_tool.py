from __future__ import annotations

from native_agent_runner.tools.base import ToolContext, ToolResult, ToolSpec


def get_tools(_context: ToolContext) -> list[ToolSpec]:
    def handler(_tool_context: ToolContext, args: dict[str, object]) -> ToolResult:
        text = str(args["text"])
        return ToolResult(ok=True, content={"words": len(text.split())})

    return [
        ToolSpec(
            id="skill.word_count",
            provider_name="skill_word_count",
            description="Count whitespace-separated words in a text string.",
            input_schema={
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
            capability="run.control",
            side_effect="run",
            handler=handler,
        )
    ]

