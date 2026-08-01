"""Provider-independent contracts for observing one model response stream.

The model adapter owns provider chunks.  This contract exposes only the two pieces of authored
content that a presentation or private persistence layer can consume: output text and reasoning
text.  Tool-call assembly remains inside the model turn implementation.

Observers are diagnostic/content-egress integrations.  Their failure must never turn a paid model
call into a failed agent run, so :func:`safe_open_model_stream` shields ``open``, ``push``, and
``close`` and falls back to a no-op writer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

ModelStreamChannel: TypeAlias = Literal["output", "reasoning"]
ModelStreamStatus: TypeAlias = Literal[
    "completed",
    "interrupted",
    "failed",
    "cancelled",
    "timed_out",
]

_CHANNELS = frozenset({"output", "reasoning"})
_STATUSES = frozenset({"completed", "interrupted", "failed", "cancelled", "timed_out"})
_LOGGER = logging.getLogger("monoid_agent_kernel.core.model_stream")


@dataclass(frozen=True)
class ModelStreamContext:
    """Stable identity and routing metadata for one provider model call."""

    run_id: str
    root_run_id: str
    turn_id: str
    stream_id: str
    step: int
    provider: str | None
    model: str | None
    started_at: str


@dataclass(frozen=True)
class ModelStreamDelta:
    """One provider-independent piece of authored stream content."""

    channel: ModelStreamChannel
    text: str

    def __post_init__(self) -> None:
        if self.channel not in _CHANNELS:
            raise ValueError("model stream channel must be 'output' or 'reasoning'")


@dataclass(frozen=True)
class ModelStreamOutcome:
    """The terminal state of a model stream.

    ``final_text`` is the provider's settled or best available partial output.  ``usage`` remains
    provider-independent by carrying the normalized usage mapping produced by the adapter layer.
    """

    status: ModelStreamStatus
    final_text: str | None = None
    usage: Mapping[str, Any] | None = field(default=None)
    error_code: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise ValueError(
                "model stream status must be completed, interrupted, failed, cancelled, or "
                "timed_out"
            )
        if self.usage is not None:
            object.__setattr__(self, "usage", dict(self.usage))


@runtime_checkable
class ModelStreamWriter(Protocol):
    """Receives the ordered content and one terminal outcome for a model stream."""

    def push(self, delta: ModelStreamDelta) -> None: ...

    def close(self, outcome: ModelStreamOutcome) -> None: ...


@runtime_checkable
class ModelStreamObserver(Protocol):
    """Opens an isolated writer for one model call."""

    def open(self, context: ModelStreamContext) -> ModelStreamWriter: ...


ModelStreamObserverFactory: TypeAlias = Callable[[], ModelStreamObserver]


class NoopModelStreamWriter:
    """A reusable inert writer returned when observation is disabled or unavailable."""

    def push(self, delta: ModelStreamDelta) -> None:
        return None

    def close(self, outcome: ModelStreamOutcome) -> None:
        return None


NOOP_MODEL_STREAM_WRITER = NoopModelStreamWriter()


class _FailureShieldedModelStreamWriter:
    def __init__(self, writer: ModelStreamWriter) -> None:
        self._writer = writer
        self._closed = False
        self._disabled = False

    def push(self, delta: ModelStreamDelta) -> None:
        if self._closed or self._disabled:
            return
        try:
            self._writer.push(delta)
        except Exception:  # noqa: BLE001 - observer failure is deliberately isolated
            # One broken exporter must not create one log entry per provider token.  Keep close
            # available for best-effort resource release, but disable all later content delivery.
            self._disabled = True
            _LOGGER.debug("model stream observer push failed", exc_info=True)

    def close(self, outcome: ModelStreamOutcome) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._writer.close(outcome)
        except Exception:  # noqa: BLE001 - observer failure is deliberately isolated
            _LOGGER.debug("model stream observer close failed", exc_info=True)


def safe_open_model_stream(
    observer: ModelStreamObserver | None,
    context: ModelStreamContext,
) -> ModelStreamWriter:
    """Open a writer while keeping all observer failures off the model-call path."""

    if observer is None:
        return NOOP_MODEL_STREAM_WRITER
    try:
        writer = observer.open(context)
    except Exception:  # noqa: BLE001 - observer failure is deliberately isolated
        _LOGGER.debug("model stream observer open failed", exc_info=True)
        return NOOP_MODEL_STREAM_WRITER
    return _FailureShieldedModelStreamWriter(writer)
