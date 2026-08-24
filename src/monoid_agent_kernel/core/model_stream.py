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
    ``retryable`` records the provider's transient-failure signal used by a lifecycle owner to
    decide whether the call is eligible for automatic retry. An explicit user reissue can still
    replace a non-retryable call after configuration changes.

    ``config_recoverable`` is the other half of that sentence, and the reason it is a separate
    fact: it says the reissue *will* succeed once the configuration is fixed. The live stream lane
    used to classify a park with half the vocabulary the park itself carries.
    """

    status: ModelStreamStatus
    final_text: str | None = None
    usage: Mapping[str, Any] | None = field(default=None)
    error_code: str | None = None
    retryable: bool = False
    config_recoverable: bool = False

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise ValueError(
                "model stream status must be completed, interrupted, failed, cancelled, or "
                "timed_out"
            )
        if self.usage is not None:
            object.__setattr__(self, "usage", dict(self.usage))
        if type(self.retryable) is not bool:
            raise ValueError("model stream retryable must be a boolean")
        if type(self.config_recoverable) is not bool:
            raise ValueError("model stream config_recoverable must be a boolean")


@runtime_checkable
class ModelStreamWriter(Protocol):
    """Receives the ordered content and one terminal outcome for a model stream."""

    def push(self, delta: ModelStreamDelta) -> None: ...

    def close(self, outcome: ModelStreamOutcome) -> None: ...


@runtime_checkable
class ModelStreamDispatchAwareWriter(Protocol):
    """Optional writer extension notified immediately before a real provider dispatch."""

    def begin_dispatch(self) -> None: ...


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

    def begin_dispatch(self) -> None:
        if self._closed or self._disabled:
            return
        begin = getattr(self._writer, "begin_dispatch", None)
        if not callable(begin):
            return
        try:
            begin()
        except Exception:  # noqa: BLE001 - observer failure is deliberately isolated
            self._disabled = True
            _LOGGER.debug("model stream observer dispatch-start failed", exc_info=True)

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


def safe_begin_model_stream_dispatch(writer: ModelStreamWriter) -> None:
    """Notify a dispatch-aware writer without letting exporter failures reach the provider."""

    begin = getattr(writer, "begin_dispatch", None)
    if not callable(begin):
        return
    try:
        begin()
    except Exception:  # noqa: BLE001 - third-party writers may be unshielded
        _LOGGER.debug("model stream dispatch-start notification failed", exc_info=True)
