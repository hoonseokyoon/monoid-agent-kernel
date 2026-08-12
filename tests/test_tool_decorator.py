from __future__ import annotations

import asyncio
import inspect

import pytest

from monoid_agent_kernel import tool
from monoid_agent_kernel.tools.base import (
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    normalize_tool_spec,
)


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


class _LyingId(str):
    """A ``str`` subclass that answers a different id than the one it stores."""

    def __str__(self) -> str:
        return "innocent"


def test_a_non_string_tool_id_is_refused_rather_than_spelled() -> None:
    """`exact_text` is total, and a total normalizer in front of a validator silences it.

    `normalize_tool_spec` refuses a non-string id -- `_required_text` raises "tool id must be a
    string" -- but it only ever sees what the decorator built. Once the id was passed through
    `exact_text` alone, every value arrived as a string and that refusal could not fire again:
    `@tool(id=123)` registered as `"123"`, and `@tool(id=object())` registered as
    `"<object object at 0x...>"`. The second is the one that matters: a MEMORY ADDRESS, different
    on every run, and `capability or tool_id` carries it into the capability as well, so an
    approval policy keyed on that capability matches nothing after a restart.

    Falsy non-strings are refused here too, which is WIDER than the refusal that was lost -- `0`
    and `[]` never reached `normalize_tool_spec` at all, because `id or fn.__name__` swallowed
    them into the function name. Deliberate, and pinned rather than described: "a string or not
    provided" is a rule that can be stated, while "a string, or absent, or any falsy object in
    which case we quietly use something else" is the shape that hides the next integrator's typo.
    """
    for bad in (123, object(), b"bytes", ["a"], 0, False, []):
        with pytest.raises(TypeError, match="must be a string"):

            @tool(id=bad)
            def _f(x: int) -> dict:
                """doc."""
                return {"x": x}

    # The type name in that message goes through `portable_type_name`, so a class cannot name
    # itself into the raise -- the same rule the rest of this commit's parent enforces by census.
    class Evil(type):
        @property
        def __name__(cls) -> str:  # noqa: N805
            return "x" * 5000

    class Hostile(metaclass=Evil):
        pass

    with pytest.raises(TypeError) as caught:

        @tool(id=Hostile())
        def _g(x: int) -> dict:
            """doc."""
            return {"x": x}

    assert len(str(caught.value)) < 200, len(str(caught.value))


def test_a_tool_id_that_is_a_string_subclass_is_accepted_and_reduced_to_its_base() -> None:
    """The counter-arm, and the reason the check is `isinstance` and not `type(id) is str`.

    A subclass IS a string, so it is accepted -- and then taken at its base value, because
    `normalize_unicode_scalars` deliberately returns subclasses as they arrived (it carries
    `_LegacyPathPattern` on purpose), so nothing downstream would have reduced it. Without
    `exact_text` here the lying `__str__` reached `ToolSpec.id` intact and survived
    `normalize_tool_spec` as a `_LyingId`.
    """

    @tool(id=_LyingId("skill.real"))
    def echo(text: str) -> dict:
        """doc."""
        return {"text": text}

    assert type(echo.id) is str
    assert echo.id == "skill.real"
    # ...and it stays exact through the boundary that used to hand the subclass onward.
    assert type(normalize_tool_spec(echo).id) is str

    # An ordinary string is untouched, and an omitted id still falls back to the function name.
    @tool(id="skill.plain")
    def plain(text: str) -> dict:
        """doc."""
        return {"text": text}

    @tool()
    def defaulted(text: str) -> dict:
        """doc."""
        return {"text": text}

    assert (plain.id, defaulted.id) == ("skill.plain", "defaulted")
