from __future__ import annotations

import importlib.util
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

from native_agent_runner.core.events import EventSink


def load_event_sinks(spec: str) -> tuple[EventSink, ...]:
    if ":" not in spec:
        raise ValueError("--event-sink-module must use path.py:function")
    path_raw, func_name = spec.rsplit(":", 1)
    module = _load_module(Path(path_raw))
    try:
        func = getattr(module, func_name)
    except AttributeError as exc:
        raise ValueError(f"event sink function not found: {func_name}") from exc
    result = func()
    return _coerce_sinks(result)


def _coerce_sinks(result: Any) -> tuple[EventSink, ...]:
    if _looks_like_sink(result):
        return (result,)
    if isinstance(result, Iterable) and not isinstance(result, (str, bytes, dict)):
        sinks = tuple(result)
        if all(_looks_like_sink(sink) for sink in sinks):
            return sinks
    raise ValueError("event sink function must return an EventSink or iterable of EventSink objects")


def _looks_like_sink(value: Any) -> bool:
    return callable(getattr(value, "emit", None)) and callable(getattr(value, "close", None))


def _load_module(path: Path) -> ModuleType:
    resolved = path.resolve()
    module_name = f"native_agent_runner_event_sink_{abs(hash(str(resolved)))}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load event sink module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
