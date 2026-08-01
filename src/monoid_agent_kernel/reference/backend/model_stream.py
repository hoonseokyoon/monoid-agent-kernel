"""Bounded, passive live delivery for provider-independent model stream content.

The durable event log is intentionally absent from this path.  A broker receives the kernel's
passive :class:`~monoid_agent_kernel.core.model_stream.ModelStreamObserver` callbacks, multiplexes
root and descendant model calls into one root-scoped ring, and lets presentation clients reconnect
with an SSE-compatible ``generation:sequence`` cursor.

Subscriber lifetime never owns model execution.  Closing a subscription only wakes its waiter;
writers and other subscribers continue independently.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeAlias

from monoid_agent_kernel.core._util import utc_timestamp
from monoid_agent_kernel.core.json_ingress import normalize_json_ingress, normalize_unicode_scalars
from monoid_agent_kernel.core.model_stream import (
    NOOP_MODEL_STREAM_WRITER,
    ModelStreamChannel,
    ModelStreamContext,
    ModelStreamDelta,
    ModelStreamObserver,
    ModelStreamObserverFactory,
    ModelStreamOutcome,
    ModelStreamStatus,
    ModelStreamWriter,
)
from monoid_agent_kernel.core.subagent_runtime import is_descendant_run_id
from monoid_agent_kernel.identifiers import namespaced_id


MODEL_STREAM_LIVE_SCHEMA_VERSION = namespaced_id("model-stream.live.v1")
DEFAULT_MODEL_STREAM_RING_FRAMES = 1024
DEFAULT_MODEL_STREAM_RING_BYTES = 512 * 1024
DEFAULT_MODEL_STREAM_ROOT_RINGS = 64

LiveModelStreamKind: TypeAlias = Literal["opened", "delta", "closed", "reset", "heartbeat"]
LiveModelStreamResetReason: TypeAlias = Literal["generation_changed", "cursor_gap", "cursor_ahead"]

_GENERATION_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True, order=True)
class LiveModelStreamCursor:
    """The last root-ring sequence consumed by a subscriber."""

    generation: str
    sequence: int

    def __post_init__(self) -> None:
        if not self.generation or _GENERATION_PATTERN.fullmatch(self.generation) is None:
            raise ValueError("model stream cursor generation is invalid")
        if self.sequence < 0:
            raise ValueError("model stream cursor sequence must be non-negative")

    @classmethod
    def parse(cls, value: str) -> LiveModelStreamCursor:
        if not isinstance(value, str):
            raise ValueError("model stream cursor must be a string")
        generation, separator, raw_sequence = value.rpartition(":")
        if not separator or not raw_sequence.isascii() or not raw_sequence.isdecimal():
            raise ValueError("model stream cursor must use generation:sequence")
        return cls(generation=generation, sequence=int(raw_sequence))

    def __str__(self) -> str:
        return f"{self.generation}:{self.sequence}"


@dataclass(frozen=True)
class LiveModelStreamFrame:
    """One root-multiplexed lifecycle, content, reset, or heartbeat frame."""

    kind: LiveModelStreamKind
    cursor: LiveModelStreamCursor
    root_run_id: str
    run_id: str | None = None
    turn_id: str | None = None
    stream_id: str | None = None
    step: int | None = None
    provider: str | None = None
    model: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    channel: ModelStreamChannel | None = None
    text: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    status: ModelStreamStatus | None = None
    final_text: str | None = None
    usage: Mapping[str, Any] | None = field(default=None)
    error_code: str | None = None
    partial: bool | None = None
    content_omitted: bool = False
    reason: LiveModelStreamResetReason | None = None
    oldest_available_cursor: str | None = None
    latest_cursor: str | None = None

    def __post_init__(self) -> None:
        if self.usage is not None:
            object.__setattr__(self, "usage", dict(self.usage))
        if self.kind == "delta":
            if self.channel not in {"output", "reasoning"} or not isinstance(self.text, str):
                raise ValueError("model stream delta requires a channel and text")
            if type(self.start_offset) is not int or type(self.end_offset) is not int:
                raise ValueError("model stream delta requires UTF-8 byte offsets")
            if self.start_offset < 0 or self.end_offset < self.start_offset:
                raise ValueError("model stream delta offsets are invalid")
            if self.end_offset - self.start_offset != len(self.text.encode("utf-8")):
                raise ValueError("model stream delta offsets do not match its UTF-8 text")
        elif self.start_offset is not None or self.end_offset is not None:
            raise ValueError("model stream offsets are valid only on delta frames")
        if self.kind in {"opened", "delta", "closed"} and (
            not isinstance(self.started_at, str) or not self.started_at
        ):
            raise ValueError("model stream call frames require started_at")

    @property
    def sequence(self) -> int:
        return self.cursor.sequence

    @property
    def event_id(self) -> str:
        return "" if self.kind == "heartbeat" else str(self.cursor)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": MODEL_STREAM_LIVE_SCHEMA_VERSION,
            "cursor": str(self.cursor),
            "sequence": self.sequence,
            "kind": self.kind,
            "root_run_id": self.root_run_id,
        }
        optional = {
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "stream_id": self.stream_id,
            "step": self.step,
            "provider": self.provider,
            "model": self.model,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "channel": self.channel,
            "text": self.text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "status": self.status,
            "final_text": self.final_text,
            "usage": None if self.usage is None else dict(self.usage),
            "error_code": self.error_code,
            "partial": self.partial,
            "reason": self.reason,
            "oldest_available_cursor": self.oldest_available_cursor,
            "latest_cursor": self.latest_cursor,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        if self.content_omitted:
            payload["content_omitted"] = True
        normalized = normalize_json_ingress(payload)
        if not isinstance(normalized, dict):  # pragma: no cover - the root is statically a dict
            raise TypeError("model stream frame must normalize to an object")
        return normalized

    def to_sse(self) -> bytes:
        """Serialize this frame for a passive SSE response."""

        if self.kind == "heartbeat":
            return b": keep-alive\n\n"
        payload = json.dumps(
            self.to_json(),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        return (f"id: {self.event_id}\nevent: model-stream\ndata: {payload}\n\n").encode("utf-8")


@dataclass(frozen=True)
class LiveModelStreamBufferStats:
    """A content-free snapshot used for health checks and boundedness tests."""

    generation: str
    latest_sequence: int
    oldest_sequence: int | None
    frame_count: int
    byte_count: int


@dataclass
class _RootRing:
    generation: str
    frames: deque[tuple[LiveModelStreamFrame, int]] = field(default_factory=deque)
    latest_sequence: int = 0
    byte_count: int = 0


class LiveModelStreamBroker:
    """Thread-safe root-multiplexed live model-stream ring.

    ``observer_factory`` is suitable for ``AgentLoop.model_stream_observer_factories``.  Binding it
    to the owning root adds a second routing check beyond the lineage carried by the kernel context.
    Root content rings use a bounded LRU; reactivating an evicted root creates a new generation so
    stale subscribers receive an explicit reset instead of accepting a reused sequence.
    """

    def __init__(
        self,
        *,
        generation: str | None = None,
        max_frames: int = DEFAULT_MODEL_STREAM_RING_FRAMES,
        max_bytes: int = DEFAULT_MODEL_STREAM_RING_BYTES,
        max_roots: int = DEFAULT_MODEL_STREAM_ROOT_RINGS,
    ) -> None:
        if max_frames < 1:
            raise ValueError("model stream ring frame limit must be positive")
        if max_bytes < 1:
            raise ValueError("model stream ring byte limit must be positive")
        if max_roots < 1:
            raise ValueError("model stream root ring limit must be positive")
        resolved_generation = generation or uuid.uuid4().hex
        # Use the cursor validator as the single source of truth for generation syntax.
        LiveModelStreamCursor(resolved_generation, 0)
        self.generation = resolved_generation
        self.max_frames = max_frames
        self.max_bytes = max_bytes
        self.max_roots = max_roots
        self._condition = threading.Condition(threading.RLock())
        self._rings: OrderedDict[str, _RootRing] = OrderedDict()
        self._ring_serial = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def close(self) -> None:
        """Wake every subscriber and make later observation inert.

        This is process/server teardown, not run control: no cancellation or interruption handle is
        retained by the broker.  Writers may safely race with close and simply become no-ops.
        """

        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._rings.clear()
            self._condition.notify_all()

    def observer(self, root_run_id: str | None = None) -> LiveModelStreamObserver:
        if root_run_id is not None:
            _validate_root_run_id(root_run_id)
        return LiveModelStreamObserver(self, root_run_id=root_run_id)

    def observer_factory(self, root_run_id: str | None = None) -> ModelStreamObserverFactory:
        if root_run_id is not None:
            _validate_root_run_id(root_run_id)
        return lambda: self.observer(root_run_id)

    def subscribe(
        self,
        root_run_id: str,
        *,
        after_cursor: str | LiveModelStreamCursor | None = None,
    ) -> LiveModelStreamSubscription:
        _validate_root_run_id(root_run_id)
        with self._condition:
            ring = None if self._closed else self._ring_locked(root_run_id)
            # A first-time client may connect after execution has already started.  Replay the
            # retained ring; an evicted prefix produces a reset frame so Studio can hydrate.
            if after_cursor is None:
                cursor = LiveModelStreamCursor(
                    self.generation if ring is None else ring.generation,
                    0,
                )
            elif isinstance(after_cursor, LiveModelStreamCursor):
                cursor = after_cursor
            else:
                cursor = LiveModelStreamCursor.parse(after_cursor)
            subscription = LiveModelStreamSubscription(
                self,
                root_run_id=root_run_id,
                cursor=cursor,
            )
            if self._closed:
                subscription._closed = True
            return subscription

    def stats(self, root_run_id: str) -> LiveModelStreamBufferStats:
        _validate_root_run_id(root_run_id)
        with self._condition:
            ring = self._rings.get(root_run_id)
            if ring is None:
                return LiveModelStreamBufferStats(
                    self.generation,
                    0,
                    None,
                    0,
                    0,
                )
            self._rings.move_to_end(root_run_id)
            return LiveModelStreamBufferStats(
                generation=ring.generation,
                latest_sequence=ring.latest_sequence,
                oldest_sequence=ring.frames[0][0].sequence if ring.frames else None,
                frame_count=len(ring.frames),
                byte_count=ring.byte_count,
            )

    def drop_root(self, root_run_id: str) -> None:
        """Discard one root ring; later publications start a new cursor generation."""

        _validate_root_run_id(root_run_id)
        with self._condition:
            self._rings.pop(root_run_id, None)
            self._condition.notify_all()

    @property
    def buffered_root_count(self) -> int:
        with self._condition:
            return len(self._rings)

    def _open(
        self,
        context: ModelStreamContext,
        *,
        bound_root_run_id: str | None,
    ) -> ModelStreamWriter:
        _validate_context_lineage(context, bound_root_run_id=bound_root_run_id)
        with self._condition:
            if self._closed:
                return NOOP_MODEL_STREAM_WRITER
        self._publish(
            context.root_run_id,
            kind="opened",
            context=context,
            provider=context.provider,
            model=context.model,
            started_at=context.started_at,
        )
        return _LiveModelStreamWriter(self, context)

    def _publish(
        self,
        root_run_id: str,
        *,
        kind: Literal["opened", "delta", "closed"],
        context: ModelStreamContext,
        **values: Any,
    ) -> None:
        with self._condition:
            if self._closed:
                return
            ring = self._ring_locked(root_run_id)
            ring.latest_sequence += 1
            cursor = LiveModelStreamCursor(ring.generation, ring.latest_sequence)
            frame = LiveModelStreamFrame(
                kind=kind,
                cursor=cursor,
                root_run_id=root_run_id,
                run_id=context.run_id,
                turn_id=context.turn_id,
                stream_id=context.stream_id,
                step=context.step,
                **values,
            )
            size = _frame_size(frame)
            if size > self.max_bytes and kind == "closed" and frame.final_text is not None:
                # Preserve terminal state under the strict ring budget.  Reconnect/hydration obtains
                # full content from the private sidecar when a single settled output is enormous.
                frame = replace(frame, final_text=None, content_omitted=True)
                size = _frame_size(frame)
            if size <= self.max_bytes:
                ring.frames.append((frame, size))
                ring.byte_count += size
                while len(ring.frames) > self.max_frames or ring.byte_count > self.max_bytes:
                    _discard_oldest(ring)
            # An oversized non-terminal frame still consumes its sequence.  Subscribers receive a
            # reset before the next retained frame instead of silently joining discontinuous text.
            self._condition.notify_all()

    def _poll(
        self,
        subscription: LiveModelStreamSubscription,
        *,
        timeout_s: float,
        limit: int | None,
    ) -> tuple[LiveModelStreamFrame, ...]:
        if timeout_s < 0:
            raise ValueError("model stream poll timeout must be non-negative")
        if limit is not None and limit < 1:
            raise ValueError("model stream poll limit must be positive")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._closed or subscription._closed:
                    subscription._closed = True
                    return ()
                ring = self._ring_locked(subscription.root_run_id)
                frames = self._collect_locked(subscription, ring, limit=limit)
                if frames or timeout_s == 0:
                    return frames
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return ()
                self._condition.wait(remaining)

    def _collect_locked(
        self,
        subscription: LiveModelStreamSubscription,
        ring: _RootRing,
        *,
        limit: int | None,
    ) -> tuple[LiveModelStreamFrame, ...]:
        output: list[LiveModelStreamFrame] = []
        cursor = subscription._cursor
        oldest = ring.frames[0][0].sequence if ring.frames else ring.latest_sequence + 1
        baseline = max(0, oldest - 1) if ring.frames else ring.latest_sequence
        reset_reason: LiveModelStreamResetReason | None = None
        if cursor.generation != ring.generation:
            reset_reason = "generation_changed"
        elif cursor.sequence < baseline:
            reset_reason = "cursor_gap"
        elif cursor.sequence > ring.latest_sequence:
            reset_reason = "cursor_ahead"
            baseline = ring.latest_sequence
        if reset_reason is not None:
            cursor = LiveModelStreamCursor(ring.generation, baseline)
            subscription._cursor = cursor
            output.append(
                self._reset_frame(
                    subscription.root_run_id,
                    ring,
                    cursor=cursor,
                    reason=reset_reason,
                )
            )
            if limit is not None and len(output) >= limit:
                return tuple(output)

        expected = subscription._cursor.sequence + 1
        for frame, _size in ring.frames:
            if frame.sequence < expected:
                continue
            if frame.sequence > expected:
                # Oversized frames can create a hole even when the front of the ring is retained.
                reset_cursor = LiveModelStreamCursor(ring.generation, frame.sequence - 1)
                subscription._cursor = reset_cursor
                output.append(
                    self._reset_frame(
                        subscription.root_run_id,
                        ring,
                        cursor=reset_cursor,
                        reason="cursor_gap",
                    )
                )
                expected = frame.sequence
                if limit is not None and len(output) >= limit:
                    return tuple(output)
            output.append(frame)
            subscription._cursor = frame.cursor
            expected = frame.sequence + 1
            if limit is not None and len(output) >= limit:
                break
        if subscription._cursor.sequence < ring.latest_sequence and (
            limit is None or len(output) < limit
        ):
            # The newest publication itself may have exceeded the byte budget.  Signal that hole
            # immediately; waiting for a later retained frame would leave a live client unaware
            # that it must hydrate missing content.
            reset_cursor = LiveModelStreamCursor(ring.generation, ring.latest_sequence)
            subscription._cursor = reset_cursor
            output.append(
                self._reset_frame(
                    subscription.root_run_id,
                    ring,
                    cursor=reset_cursor,
                    reason="cursor_gap",
                )
            )
        return tuple(output)

    def _reset_frame(
        self,
        root_run_id: str,
        ring: _RootRing,
        *,
        cursor: LiveModelStreamCursor,
        reason: LiveModelStreamResetReason,
    ) -> LiveModelStreamFrame:
        oldest_cursor = (
            str(LiveModelStreamCursor(ring.generation, ring.frames[0][0].sequence))
            if ring.frames
            else None
        )
        return LiveModelStreamFrame(
            kind="reset",
            cursor=cursor,
            root_run_id=root_run_id,
            reason=reason,
            oldest_available_cursor=oldest_cursor,
            latest_cursor=str(LiveModelStreamCursor(ring.generation, ring.latest_sequence)),
        )

    def _close_subscription(self, subscription: LiveModelStreamSubscription) -> None:
        with self._condition:
            subscription._closed = True
            self._condition.notify_all()

    def _heartbeat(self, subscription: LiveModelStreamSubscription) -> LiveModelStreamFrame:
        with self._condition:
            return LiveModelStreamFrame(
                kind="heartbeat",
                cursor=subscription._cursor,
                root_run_id=subscription.root_run_id,
            )

    def _ring_locked(self, root_run_id: str) -> _RootRing:
        ring = self._rings.get(root_run_id)
        if ring is not None:
            self._rings.move_to_end(root_run_id)
            return ring
        self._ring_serial += 1
        generation = (
            self.generation
            if self._ring_serial == 1
            else f"{self.generation}.{self._ring_serial:x}"
        )
        ring = _RootRing(generation=generation)
        self._rings[root_run_id] = ring
        evicted = False
        while len(self._rings) > self.max_roots:
            self._rings.popitem(last=False)
            evicted = True
        if evicted:
            self._condition.notify_all()
        return ring


class LiveModelStreamObserver(ModelStreamObserver):
    """Passive observer backed by a shared :class:`LiveModelStreamBroker`."""

    def __init__(
        self,
        broker: LiveModelStreamBroker,
        *,
        root_run_id: str | None = None,
    ) -> None:
        self._broker = broker
        self._root_run_id = root_run_id

    def open(self, context: ModelStreamContext) -> ModelStreamWriter:
        return self._broker._open(context, bound_root_run_id=self._root_run_id)


class _LiveModelStreamWriter(ModelStreamWriter):
    def __init__(self, broker: LiveModelStreamBroker, context: ModelStreamContext) -> None:
        self._broker = broker
        self._context = context
        self._lock = threading.Lock()
        self._closed = False
        self._byte_offsets: dict[ModelStreamChannel, int] = {
            "output": 0,
            "reasoning": 0,
        }

    def push(self, delta: ModelStreamDelta) -> None:
        if not delta.text:
            return
        with self._lock:
            if self._closed:
                return
            text = normalize_unicode_scalars(delta.text)
            start_offset = self._byte_offsets[delta.channel]
            end_offset = start_offset + len(text.encode("utf-8"))
            self._byte_offsets[delta.channel] = end_offset
            self._broker._publish(
                self._context.root_run_id,
                kind="delta",
                context=self._context,
                started_at=self._context.started_at,
                channel=delta.channel,
                text=text,
                start_offset=start_offset,
                end_offset=end_offset,
            )

    def close(self, outcome: ModelStreamOutcome) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            usage = None
            if outcome.usage is not None:
                usage = _normalize_usage(outcome.usage)
            self._broker._publish(
                self._context.root_run_id,
                kind="closed",
                context=self._context,
                started_at=self._context.started_at,
                finished_at=utc_timestamp(),
                status=outcome.status,
                final_text=(
                    None
                    if outcome.final_text is None
                    else normalize_unicode_scalars(outcome.final_text)
                ),
                usage=usage,
                error_code=outcome.error_code,
                partial=outcome.status != "completed",
            )


class LiveModelStreamSubscription:
    """A replay-capable root subscription whose close is execution-independent."""

    def __init__(
        self,
        broker: LiveModelStreamBroker,
        *,
        root_run_id: str,
        cursor: LiveModelStreamCursor,
    ) -> None:
        self._broker = broker
        self.root_run_id = root_run_id
        self._cursor = cursor
        self._closed = False

    @property
    def cursor(self) -> str:
        with self._broker._condition:
            return str(self._cursor)

    @property
    def closed(self) -> bool:
        with self._broker._condition:
            return self._closed or self._broker._closed

    def poll(
        self,
        *,
        timeout_s: float = 0.0,
        limit: int | None = None,
    ) -> tuple[LiveModelStreamFrame, ...]:
        return self._broker._poll(self, timeout_s=timeout_s, limit=limit)

    async def apoll(
        self,
        *,
        timeout_s: float = 0.0,
        limit: int | None = None,
    ) -> tuple[LiveModelStreamFrame, ...]:
        try:
            return await asyncio.to_thread(self.poll, timeout_s=timeout_s, limit=limit)
        except asyncio.CancelledError:
            # Cancelling ``to_thread`` stops only the awaiting Future. Close the passive reader so
            # the actual worker leaves its condition wait instead of pinning executor shutdown.
            self.close()
            raise

    def frames(self, *, heartbeat_interval_s: float = 15.0) -> Iterator[LiveModelStreamFrame]:
        if heartbeat_interval_s <= 0:
            raise ValueError("model stream heartbeat interval must be positive")
        try:
            while not self.closed:
                frames = self.poll(timeout_s=heartbeat_interval_s)
                if frames:
                    yield from frames
                elif not self.closed:
                    yield self._broker._heartbeat(self)
        finally:
            self.close()

    async def aframes(
        self, *, heartbeat_interval_s: float = 15.0
    ) -> AsyncIterator[LiveModelStreamFrame]:
        if heartbeat_interval_s <= 0:
            raise ValueError("model stream heartbeat interval must be positive")
        try:
            while not self.closed:
                frames = await self.apoll(timeout_s=heartbeat_interval_s)
                if frames:
                    for frame in frames:
                        yield frame
                elif not self.closed:
                    yield self._broker._heartbeat(self)
        finally:
            # Cancelling the asyncio wrapper cannot cancel its in-flight worker thread. Closing the
            # passive reader wakes that thread's condition wait immediately.
            self.close()

    def close(self) -> None:
        self._broker._close_subscription(self)

    def __enter__(self) -> LiveModelStreamSubscription:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _validate_root_run_id(root_run_id: str) -> None:
    if not isinstance(root_run_id, str) or not root_run_id:
        raise ValueError("model stream root run id must be a non-empty string")
    if ".sub." in root_run_id or not is_descendant_run_id(root_run_id, root_run_id):
        raise ValueError("model stream root run id is invalid")


def _validate_context_lineage(
    context: ModelStreamContext,
    *,
    bound_root_run_id: str | None,
) -> None:
    _validate_root_run_id(context.root_run_id)
    if bound_root_run_id is not None and context.root_run_id != bound_root_run_id:
        raise ValueError("model stream context does not belong to the bound root run")
    if not context.run_id or not is_descendant_run_id(context.root_run_id, context.run_id):
        raise ValueError("model stream context run is outside its root lineage")
    if not context.turn_id or not context.stream_id:
        raise ValueError("model stream context requires turn and stream ids")
    if type(context.step) is not int or context.step < 1:
        raise ValueError("model stream context step must be a positive integer")
    if not isinstance(context.started_at, str) or not context.started_at.endswith("Z"):
        raise ValueError("model stream context started_at must be a UTC timestamp")
    if context.provider is not None and not isinstance(context.provider, str):
        raise ValueError("model stream context provider must be a string or null")
    if context.model is not None and not isinstance(context.model, str):
        raise ValueError("model stream context model must be a string or null")


def _frame_size(frame: LiveModelStreamFrame) -> int:
    return len(
        json.dumps(
            frame.to_json(),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _normalize_usage(usage: Mapping[str, Any]) -> Mapping[str, Any] | None:
    try:
        normalized = normalize_json_ingress(dict(usage))
        if not isinstance(normalized, Mapping):
            return None
        # Validate the detached value now so malformed usage cannot suppress the terminal frame.
        json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, UnicodeError, RecursionError):
        return None
    return normalized


def _discard_oldest(ring: _RootRing) -> None:
    _frame, size = ring.frames.popleft()
    ring.byte_count -= size


__all__ = [
    "DEFAULT_MODEL_STREAM_RING_BYTES",
    "DEFAULT_MODEL_STREAM_RING_FRAMES",
    "DEFAULT_MODEL_STREAM_ROOT_RINGS",
    "MODEL_STREAM_LIVE_SCHEMA_VERSION",
    "LiveModelStreamBroker",
    "LiveModelStreamBufferStats",
    "LiveModelStreamCursor",
    "LiveModelStreamFrame",
    "LiveModelStreamObserver",
    "LiveModelStreamSubscription",
]
