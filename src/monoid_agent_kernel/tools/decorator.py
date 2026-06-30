"""``@tool`` — build a :class:`ToolSpec` from a typed Python function.

The hand-written path (construct a ``ToolSpec`` with a literal JSON Schema and a
``(ctx, args)`` handler) stays fully supported. This decorator is sugar for the common
case: derive the ``input_schema`` from the function's parameter type hints (via pydantic,
already a dependency) and wrap the return value in a :class:`ToolResult`.

    from monoid_agent_kernel import tool

    @tool(side_effect="read")
    def word_count(text: str, top_k: int = 5) -> dict:
        '''Count words and return the top_k most frequent.'''
        ...

If the first parameter is annotated :class:`ToolContext` (or named ``ctx`` / ``context``),
the engine's tool context is injected and the remaining parameters are filled from the
validated tool arguments. The function may return a ``ToolResult`` (used as-is), a ``dict``
(wrapped as ``content``), or any other value (wrapped as ``{"result": value}``).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import ValidationError, create_model

from monoid_agent_kernel.tools.base import (
    ToolContext,
    ToolResult,
    ToolSideEffect,
    ToolSpec,
)

_CTX_NAMES = {"ctx", "context"}


def tool(
    func: Callable[..., Any] | None = None,
    *,
    id: str | None = None,
    description: str | None = None,
    capability: str | None = None,
    side_effect: ToolSideEffect = "read",
    provider_name: str | None = None,
) -> ToolSpec | Callable[[Callable[..., Any]], ToolSpec]:
    """Turn a typed function into a :class:`ToolSpec`.

    Usable bare (``@tool``) or with keywords (``@tool(side_effect="write")``). Defaults:
    ``id`` = function name, ``description`` = first docstring line, ``capability`` = ``id``.
    """

    def build(fn: Callable[..., Any]) -> ToolSpec:
        return _spec_from_function(
            fn,
            id=id,
            description=description,
            capability=capability,
            side_effect=side_effect,
            provider_name=provider_name,
        )

    return build(func) if func is not None else build


def _spec_from_function(
    fn: Callable[..., Any],
    *,
    id: str | None,
    description: str | None,
    capability: str | None,
    side_effect: ToolSideEffect,
    provider_name: str | None,
) -> ToolSpec:
    signature = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:  # annotations that can't be resolved -> treat each as Any
        hints = {}

    params = list(signature.parameters.values())
    ctx_param_name: str | None = None
    if params:
        first = params[0]
        if hints.get(first.name) is ToolContext or first.name in _CTX_NAMES:
            ctx_param_name = first.name
            params = params[1:]

    fields: dict[str, tuple[Any, Any]] = {}
    for param in params:
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(
                f"@tool does not support *args/**kwargs parameters: {fn.__name__}.{param.name}"
            )
        annotation = hints.get(param.name, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (annotation, default)

    model = create_model(f"{fn.__name__.title().replace('_', '')}Args", **fields)
    input_schema = model.model_json_schema()

    tool_id = id or fn.__name__
    tool_description = description or (inspect.getdoc(fn) or "").split("\n", 1)[0]

    def handler(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        try:
            validated = model(**(args or {}))
        except ValidationError as exc:
            return ToolResult(
                ok=False,
                error=str(exc),
                error_code="invalid_tool_args",
                category="tool",
            )
        kwargs = validated.model_dump()
        result = fn(ctx, **kwargs) if ctx_param_name is not None else fn(**kwargs)
        if isinstance(result, ToolResult):
            return result
        if isinstance(result, dict):
            return ToolResult(ok=True, content=result)
        return ToolResult(ok=True, content={"result": result})

    return ToolSpec(
        id=tool_id,
        description=tool_description,
        input_schema=input_schema,
        capability=capability or tool_id,
        side_effect=side_effect,
        handler=handler,
        provider_name=provider_name,
    )
