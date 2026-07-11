from __future__ import annotations

import asyncio
import inspect

from monoid_agent_kernel import tool
from monoid_agent_kernel.tools.base import ToolContext, ToolRegistry, ToolResult, ToolSpec


def test_tool_builds_spec_and_schema_from_hints() -> None:
    @tool(side_effect="read")
    def word_count(text: str, top_k: int = 5) -> dict:
        """Count words and return the top_k most frequent."""
        return {"n": len(text.split()), "top_k": top_k}

    assert isinstance(word_count, ToolSpec)
    assert word_count.id == "word_count"
    assert word_count.capability == "word_count"
    assert word_count.side_effect == "read"
    assert word_count.description == "Count words and return the top_k most frequent."

    props = word_count.input_schema["properties"]
    assert props["text"]["type"] == "string"
    assert props["top_k"]["type"] == "integer"
    assert word_count.input_schema["required"] == ["text"]


def test_tool_handler_validates_and_wraps_dict() -> None:
    @tool()
    def add(a: int, b: int = 0) -> dict:
        return {"sum": a + b}

    ok = add.handler(None, {"a": 2, "b": 3})
    assert ok.ok and ok.content == {"sum": 5}

    bad = add.handler(None, {"a": "not-an-int"})
    assert not bad.ok
    assert bad.error_code == "invalid_tool_args"


def test_tool_injects_context_and_passes_through_tool_result() -> None:
    seen: dict[str, object] = {}

    @tool()
    def needs_ctx(ctx: ToolContext, value: str) -> ToolResult:
        seen["ctx"] = ctx
        return ToolResult(ok=True, content={"value": value})

    sentinel = object()
    result = needs_ctx.handler(sentinel, {"value": "hi"})
    assert result.ok and result.content == {"value": "hi"}
    assert seen["ctx"] is sentinel


def test_tool_non_dict_return_wrapped_under_result_key() -> None:
    @tool()
    def shout(text: str) -> str:
        return text.upper()

    out = shout.handler(None, {"text": "hi"})
    assert out.ok and out.content == {"result": "HI"}


def test_tool_spec_registers_and_validates_args() -> None:
    @tool(id="skill.echo")
    def echo(text: str) -> dict:
        return {"text": text}

    registry = ToolRegistry()
    registry.register(echo)
    resolved = registry.resolve("skill.echo")
    registry.validate_args(resolved, {"text": "ok"})  # no raise


def test_tool_preserves_async_function_and_normalizes_awaited_result() -> None:
    @tool(id="skill.async_add")
    async def add(a: int, b: int = 0) -> dict:
        await asyncio.sleep(0)
        return {"sum": a + b}

    assert inspect.iscoroutinefunction(add.handler)
    result = asyncio.run(add.handler(None, {"a": 2, "b": 3}))
    invalid = asyncio.run(add.handler(None, {"a": "not-an-int"}))

    assert result.ok and result.content == {"sum": 5}
    assert not invalid.ok and invalid.error_code == "invalid_tool_args"
