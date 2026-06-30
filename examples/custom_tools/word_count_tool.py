"""Authoring a custom tool — two equivalent ways.

Load either with the CLI: ``--tool-module examples/custom_tools/word_count_tool.py:get_tools``.

The ``@tool`` decorator (recommended) derives the JSON Schema from the function's type
hints. The hand-written ``ToolSpec`` below is the same tool spelled out explicitly — both
are fully supported; reach for the literal form when you need schema details the decorator
doesn't express.
"""

from __future__ import annotations

from monoid_agent_kernel import tool
from monoid_agent_kernel.tools.base import ToolContext, ToolResult, ToolSpec


# --- Recommended: @tool derives input_schema from the signature --------------------------
@tool(id="skill.word_count", capability="run.control", side_effect="run")
def word_count(text: str) -> dict:
    """Count whitespace-separated words in a text string."""
    return {"words": len(text.split())}


def get_tools(_context: ToolContext) -> list[ToolSpec]:
    return [word_count]


# --- Equivalent hand-written form (for reference) ----------------------------------------
def _word_count_handwritten() -> ToolSpec:
    def handler(_tool_context: ToolContext, args: dict[str, object]) -> ToolResult:
        text = str(args["text"])
        return ToolResult(ok=True, content={"words": len(text.split())})

    return ToolSpec(
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
