from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar
from uuid import UUID

import pytest
from pydantic import BaseModel, Field, RootModel, SecretStr, model_serializer
from pydantic_core import PydanticSerializationError

from monoid_agent_kernel import tool
from monoid_agent_kernel.core.authority import WriteAuthorityRevoked
from monoid_agent_kernel.errors import RunCancelled, RunTimeout, ToolExecutionError, TurnInterrupted
from monoid_agent_kernel.tools.base import (
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    normalize_tool_result,
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


class _Tier(Enum):
    STANDARD = "standard"


class _Customer(BaseModel):
    name: str


class _Draft(BaseModel):
    customer: _Customer
    title: str = "Weekly"


class _SerializedInput(BaseModel):
    value: int
    calls: ClassVar[int] = 0
    fail: ClassVar[bool] = False

    @model_serializer
    def serialize(self) -> dict:
        type(self).calls += 1
        if self.fail:
            raise ValueError("input serialization should never run")
        return {"serialized": self.value}


def _invoke(spec: ToolSpec, args: dict, ctx: object = None) -> ToolResult:
    result = spec.handler(ctx, args)
    return asyncio.run(result) if inspect.isawaitable(result) else result


@pytest.mark.parametrize("is_async", [False, True])
def test_validated_inputs_keep_domain_types_defaults_and_context(is_async: bool) -> None:
    seen = []

    def capture(
        ctx: ToolContext,
        draft: _Draft,
        customers: list[_Customer],
        by_name: dict[str, _Customer],
        tier: _Tier = _Tier.STANDARD,
        limit: int = 7,
    ) -> dict:
        seen.append((ctx, draft, customers, by_name, tier, limit))
        return {"called": True}

    async def capture_async(
        ctx: ToolContext,
        draft: _Draft,
        customers: list[_Customer],
        by_name: dict[str, _Customer],
        tier: _Tier = _Tier.STANDARD,
        limit: int = 7,
    ) -> dict:
        return capture(ctx, draft, customers, by_name, tier, limit)

    spec = tool(capture_async if is_async else capture)
    sentinel = object()
    args = {
        "draft": {"customer": {"name": "Ada"}},
        "customers": [{"name": "Lin"}],
        "by_name": {"owner": {"name": "Grace"}},
    }
    assert _invoke(spec, args, sentinel).ok
    ctx, draft, customers, by_name, tier, limit = seen.pop()
    assert ctx is sentinel
    assert isinstance(draft, _Draft) and isinstance(draft.customer, _Customer)
    assert (draft.customer.name, draft.title) == ("Ada", "Weekly")
    assert isinstance(customers[0], _Customer) and customers[0].name == "Lin"
    assert isinstance(by_name["owner"], _Customer) and by_name["owner"].name == "Grace"
    assert tier is _Tier.STANDARD and limit == 7
    assert set(spec.input_schema["properties"]) == {
        "draft",
        "customers",
        "by_name",
        "tier",
        "limit",
    }
    assert spec.input_schema["required"] == ["draft", "customers", "by_name"]
    invalid = _invoke(spec, {**args, "draft": {"customer": {}}}, sentinel)
    assert not invalid.ok and invalid.error_code == "invalid_tool_args"
    assert seen == []


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("serializer_fails", [False, True])
def test_input_validation_never_calls_serializers(is_async: bool, serializer_fails: bool) -> None:
    _SerializedInput.calls = 0
    _SerializedInput.fail = serializer_fails

    def capture(payload: _SerializedInput) -> dict:
        assert isinstance(payload, _SerializedInput)
        return {"value": payload.value}

    async def capture_async(payload: _SerializedInput) -> dict:
        return capture(payload)

    result = _invoke(tool(capture_async if is_async else capture), {"payload": {"value": "3"}})
    assert result.content == {"value": 3}
    assert _SerializedInput.calls == 0


class _Report(BaseModel):
    customer: _Customer
    at: datetime
    reference: UUID
    tier: _Tier


class _ScalarReport(BaseModel):
    @model_serializer
    def serialize(self) -> str:
        return "ready"


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "value, content",
    [
        (
            _Report(
                customer=_Customer(name="Ada"),
                at=datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
                reference=UUID("12345678-1234-5678-1234-567812345678"),
                tier=_Tier.STANDARD,
            ),
            {
                "customer": {"name": "Ada"},
                "at": "2030-01-02T03:04:05Z",
                "reference": "12345678-1234-5678-1234-567812345678",
                "tier": "standard",
            },
        ),
        (RootModel[dict[str, int]]({"count": 3}), {"count": 3}),
        (RootModel[list[int]]([1, 2]), {"result": [1, 2]}),
        (RootModel[int](3), {"result": 3}),
        (_ScalarReport(), {"result": "ready"}),
    ],
)
def test_top_level_models_use_json_mode_and_actual_dump_shape(is_async, value, content) -> None:
    def report():
        return value

    async def report_async():
        return value

    result = normalize_tool_result(_invoke(tool(report_async if is_async else report), {}))
    assert result.to_observation() == {"ok": True, "result": content}


def test_model_serialization_respects_domain_keys_exclusion_and_actual_subclass() -> None:
    class Public(BaseModel):
        ok: bool = False
        error: str = "domain status"
        result: int = 4
        secret: SecretStr = SecretStr("private-secret")
        hidden: str = Field(default="private-hidden", exclude=True)

    class Actual(Public):
        detail: str = "published by actual return type"

    @tool
    def report() -> Public:
        return Actual()

    result = normalize_tool_result(report.handler(None, {}))
    assert result.to_observation() == {
        "ok": True,
        "result": {
            "ok": False,
            "error": "domain status",
            "result": 4,
            "secret": "**********",
            "detail": "published by actual return type",
        },
    }


@pytest.mark.parametrize("envelope", ["dict", "explicit"])
def test_model_conversion_stays_at_the_decorated_return_boundary(envelope: str) -> None:
    payload = {"customer": _Customer(name="Ada")}

    @tool
    def report():
        return ToolResult(ok=True, content=payload) if envelope == "explicit" else payload

    with pytest.raises(ToolExecutionError) as caught:
        normalize_tool_result(report.handler(None, {}))
    assert caught.value.error_code == "tool_result_unportable"
    assert caught.value.retryable is True


@pytest.mark.parametrize(
    "value",
    [
        {"ok": False, "error": "domain", "result": 1},
        "scalar",
        [1, 2],
        None,
        ToolResult(
            ok=False, content={"detail": 1}, error="retry", error_code="app_error", retryable=True
        ),
    ],
)
def test_existing_return_envelopes_are_preserved(value) -> None:
    @tool
    def report():
        return value

    result = report.handler(None, {})
    if isinstance(value, ToolResult):
        assert result is value
    else:
        assert result.ok
        assert result.content == (value if isinstance(value, dict) else {"result": value})


@pytest.mark.parametrize("failure_kind", ["unsupported", "serializer"])
def test_model_serialization_failures_have_a_fixed_tool_diagnostic(failure_kind: str) -> None:
    class Unsupported(BaseModel):
        payload: Any

    class Broken(BaseModel):
        @model_serializer
        def serialize(self):
            raise ValueError("private serializer details")

    @tool
    def report():
        return Unsupported(payload=object()) if failure_kind == "unsupported" else Broken()

    with pytest.raises(ToolExecutionError) as caught:
        report.handler(None, {})
    assert caught.value.error_code == "tool_result_unportable"
    assert caught.value.category == "tool" and caught.value.retryable is True
    assert str(caught.value) == "tool result model could not be serialized as JSON"


@pytest.mark.parametrize(
    "original",
    [
        RunCancelled("stop"),
        RunTimeout("time"),
        TurnInterrupted("interrupt"),
        WriteAuthorityRevoked(),
        asyncio.CancelledError("local"),
        KeyboardInterrupt(),
        SystemExit(1),
    ],
)
@pytest.mark.parametrize("route", ["direct", "serializer", "nested_field", "explicit_wrapper"])
def test_model_dump_preserves_available_control_exception_identity(original, route) -> None:
    class Broken(BaseModel):
        @model_serializer
        def serialize(self):
            raise original

    class Nested(BaseModel):
        payload: Any

    class Override(BaseModel):
        def model_dump(self, **kwargs):
            if route == "explicit_wrapper":
                try:
                    raise original
                except BaseException as cause:
                    raise PydanticSerializationError("wrapper") from cause
            raise original

    @tool
    def report():
        if route == "nested_field":
            return Nested(payload=Broken())
        return Broken() if route == "serializer" else Override()

    with pytest.raises(type(original)) as caught:
        report.handler(None, {})
    assert caught.value is original
