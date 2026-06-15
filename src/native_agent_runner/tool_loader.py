from __future__ import annotations

import importlib.util
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

from native_agent_runner.tools.base import ToolContext, ToolSpec


class FunctionToolProvider:
    def __init__(self, func: object) -> None:
        self._func = func

    def get_tools(self, context: ToolContext) -> Iterable[ToolSpec]:
        result = self._func(context)  # type: ignore[misc]
        return list(result)


def load_tool_provider(spec: str) -> FunctionToolProvider:
    if ":" not in spec:
        raise ValueError("--tool-module must use path.py:function")
    path_raw, func_name = spec.rsplit(":", 1)
    module = _load_module(Path(path_raw))
    func = getattr(module, func_name)
    return FunctionToolProvider(func)


def _load_module(path: Path) -> ModuleType:
    resolved = path.resolve()
    module_name = f"native_agent_runner_custom_{abs(hash(str(resolved)))}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load tool module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
