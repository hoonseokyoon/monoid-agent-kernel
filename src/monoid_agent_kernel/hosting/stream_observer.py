"""Bounded durable persistence for provider-independent model stream deltas."""

from __future__ import annotations

import contextvars
import hashlib
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from monoid_agent_kernel.core.authority import ActivationWriteAuthority
from monoid_agent_kernel.core.model_invocation import logical_model_call_id
from monoid_agent_kernel.core.model_stream import (
    ModelStreamContext,
    ModelStreamDelta,
    ModelStreamOutcome,
    ModelStreamWriter,
)
from monoid_agent_kernel.hosting.contracts import WriterToken
from monoid_agent_kernel.hosting.streams import (
    MAX_STREAM_CHUNK_BYTES,
    DurableStreamHead,
    DurableStreamIdentity,
    DurableStreamStore,
    durable_model_stream_id,
)


class DurableStreamWriteError(RuntimeError):
    """Base error for the bounded model-stream persistence bridge."""


class DurableStreamWriteRejected(DurableStreamWriteError):
    """The durable store rejected a mutation with a typed portable status."""

    def __init__(self, operation: str, status: str) -> None:
        super().__init__(f"durable stream {operation} was rejected: {status}")
        self.operation = operation
        self.status = status


class DurableStreamWriterTimeout(DurableStreamWriteError):
    """The bounded flush worker did not finish within its configured join budget."""


def _supports_store(value: object) -> bool:
    return all(
        callable(getattr(value, method, None))
        for method in ("open", "reset", "append", "seal", "read_after")
    )


def _positive_finite(value: object, field_name: str, *, maximum: float) -> float:
    if (
        type(value) not in {int, float}
        or not math.isfinite(float(value))
        or not (0 < float(value) <= maximum)
    ):
        raise ValueError(f"{field_name} must be finite and between zero and {maximum}")
    return float(value)


def _utf8_prefix_size(buffer: bytearray, maximum: int) -> int:
    """Return the largest complete UTF-8 prefix no longer than ``maximum``."""

    end = min(len(buffer), maximum)
    while end > 0:
        try:
            bytes(buffer[:end]).decode("utf-8")
        except UnicodeDecodeError as exc:
            if exc.reason != "unexpected end of data":
                raise DurableStreamWriteError("buffered model stream bytes are invalid UTF-8") from exc
            end -= 1
            continue
        return end
    raise DurableStreamWriteError("durable stream chunk bound cannot hold one UTF-8 scalar")


@dataclass
class _Lane:
    identity: DurableStreamIdentity
    head: DurableStreamHead
    reset_before_append: bool = False
    digest: object = field(default_factory=hashlib.sha256)
    buffer: bytearray = field(default_factory=bytearray)
    flush_at: float | None = None


class DurableModelStreamObserver:
    """Activation-bound factory target that opens one coalescing writer per model call."""

    def __init__(
        self,
        store: DurableStreamStore,
        *,
        writer_token: WriterToken,
        write_authority: ActivationWriteAuthority,
        chunk_bytes: int = 64 * 1024,
        flush_interval_s: float = 0.25,
        max_buffer_bytes: int = 1024 * 1024,
        supervisor_join_timeout_s: float = 30,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not _supports_store(store):
            raise TypeError("durable model stream store must satisfy DurableStreamStore")
        if not isinstance(writer_token, WriterToken):
            raise TypeError("durable model stream observer requires WriterToken")
        if not isinstance(write_authority, ActivationWriteAuthority):
            raise TypeError(
                "durable model stream observer requires ActivationWriteAuthority"
            )
        if type(chunk_bytes) is not int or not (4 <= chunk_bytes <= MAX_STREAM_CHUNK_BYTES):
            raise ValueError(
                "durable model stream chunk_bytes must be between 4 and MAX_STREAM_CHUNK_BYTES"
            )
        if type(max_buffer_bytes) is not int or not (
            chunk_bytes <= max_buffer_bytes <= 16 * MAX_STREAM_CHUNK_BYTES
        ):
            raise ValueError(
                "durable model stream max_buffer_bytes must be bounded and at least chunk_bytes"
            )
        if not callable(monotonic):
            raise TypeError("durable model stream monotonic clock must be callable")
        self.store = store
        self.writer_token = writer_token
        self.write_authority = write_authority
        self.chunk_bytes = chunk_bytes
        self.flush_interval_s = _positive_finite(
            flush_interval_s,
            "durable model stream flush_interval_s",
            maximum=3600,
        )
        self.max_buffer_bytes = max_buffer_bytes
        self.supervisor_join_timeout_s = _positive_finite(
            supervisor_join_timeout_s,
            "durable model stream supervisor_join_timeout_s",
            maximum=3600,
        )
        self.monotonic = monotonic

    def open(self, context: ModelStreamContext) -> ModelStreamWriter:
        if not isinstance(context, ModelStreamContext):
            raise TypeError("durable model stream open requires ModelStreamContext")
        if context.run_id != self.writer_token.run_id:
            raise DurableStreamWriteRejected("open", "fenced")
        return _DurableModelStreamWriter(self, context)


class _DurableModelStreamWriter:
    def __init__(
        self,
        observer: DurableModelStreamObserver,
        context: ModelStreamContext,
    ) -> None:
        self._observer = observer
        self._context = context
        self._condition = threading.Condition()
        self._lanes: dict[str, _Lane] = {}
        self._failure: BaseException | None = None
        self._closing = False
        self._closed = False
        self._aborting = False
        self._flush_requested = False
        self._inflight = False
        self._channel_cursor = 0
        self._worker_context = contextvars.copy_context()
        output = self._ensure_lane("output")
        if output.reset_before_append:
            # Output exists for every observer execution and therefore detects a replacement.
            # Load the other kernel-owned private lane now so the first replacement delta can
            # reset stale reasoning too, even when the replacement emits no reasoning bytes.
            self._ensure_lane("reasoning")
        self._worker = threading.Thread(
            target=lambda: self._worker_context.run(self._run),
            name=f"monoid-stream-{context.stream_id[:32]}",
            daemon=True,
        )
        self._worker.start()

    @property
    def _store(self) -> DurableStreamStore:
        return self._observer.store

    def _identity(self, channel: str) -> DurableStreamIdentity:
        return DurableStreamIdentity(
            run_id=self._context.run_id,
            stream_id=durable_model_stream_id(
                self._context.run_id,
                self._context.turn_id,
            ),
            logical_call_id=logical_model_call_id(
                self._context.run_id,
                self._context.turn_id,
            ),
            channel=channel,
        )

    def _reject(self, operation: str, status: str) -> None:
        if status == "fenced":
            self._observer.write_authority.revoke()
        raise DurableStreamWriteRejected(operation, status)

    def _ensure_lane(self, channel: str) -> _Lane:
        with self._condition:
            self._raise_failure_locked()
            if self._closing or self._closed:
                raise DurableStreamWriteError("durable model stream writer is closed")
            existing = self._lanes.get(channel)
            if existing is not None:
                return existing
            # Lane creation and hydration are serialized with ``close``. A late delta cannot
            # create an unsealed side lane after close has started, and a concurrently revoked
            # activation cannot use the background writer's still-current database token.
            identity = self._identity(channel)
            result = self._observer.write_authority.guard_external_call(
                lambda: self._store.open(
                    identity,
                    writer_token=self._observer.writer_token,
                )
            )
            if result.status not in {"opened", "already_open", "sealed"} or result.head is None:
                self._reject("open", result.status)
            lane = _Lane(
                identity=identity,
                head=result.head,
                reset_before_append=result.status in {"already_open", "sealed"},
            )
            cursor = 0
            while cursor < lane.head.cursor_bytes:
                replay = self._observer.write_authority.guard_external_call(
                    lambda: self._store.read_after(
                        identity,
                        generation=lane.head.generation,
                        cursor=cursor,
                        limit=100,
                    )
                )
                if replay.status != "ok" or not replay.chunks:
                    self._reject("hydrate", replay.status)
                for chunk in replay.chunks:
                    lane.digest.update(chunk.data)  # type: ignore[attr-defined]
                cursor = replay.next_cursor
            if cursor != lane.head.cursor_bytes:
                raise DurableStreamWriteError("durable stream hydration did not reach its head")
            self._lanes[channel] = lane
            return lane

    def _raise_failure_locked(self) -> None:
        if self._failure is not None:
            raise DurableStreamWriteError("durable stream flush worker failed") from self._failure

    def _buffered_bytes_locked(self) -> int:
        return sum(len(lane.buffer) for lane in self._lanes.values())

    def _reset_lane_locked(self, lane: _Lane) -> None:
        previous = lane.head
        result = self._observer.write_authority.guard_external_call(
            lambda: self._store.reset(
                lane.identity,
                expected_generation=previous.generation,
                reset_id=f"writer-reopen-{previous.generation + 1}",
                writer_token=self._observer.writer_token,
            )
        )
        if result.status not in {"reset", "already_reset"}:
            self._reject("reset", result.status)
        if (
            result.head is None
            or result.applied_generation is None
            or result.applied_generation != previous.generation + 1
            or result.head.identity != lane.identity
            or result.head.generation != result.applied_generation
            or result.head.cursor_bytes != 0
            or result.head.next_chunk_sequence != 1
            or result.head.state != "open"
        ):
            raise DurableStreamWriteError("durable stream reset evidence disagrees with lane")
        lane.head = result.head
        lane.digest = hashlib.sha256()
        lane.reset_before_append = False

    def _reset_prior_lanes_locked(self) -> None:
        for lane in sorted(
            self._lanes.values(),
            key=lambda candidate: candidate.identity.channel,
        ):
            if lane.reset_before_append:
                self._reset_lane_locked(lane)

    def begin_dispatch(self) -> None:
        """Move a replacement provider execution to a fresh generation before adapter entry."""

        with self._condition:
            self._raise_failure_locked()
            if self._closing or self._closed:
                raise DurableStreamWriteError("durable model stream writer is closed")
            self._reset_prior_lanes_locked()

    def prepare_settlement(self) -> None:
        """Flush every delivered delta before invocation settlement becomes recoverable."""

        # Batching may use an injected/frozen clock. The supervisor budget must still expire in
        # wall-clock time when a store blocks or the worker fails to make progress.
        deadline = time.monotonic() + self._observer.supervisor_join_timeout_s
        with self._condition:
            self._raise_failure_locked()
            if self._closing or self._closed:
                raise DurableStreamWriteError("durable model stream writer is closed")
            self._flush_requested = True
            self._condition.notify_all()
            try:
                while self._inflight or any(lane.buffer for lane in self._lanes.values()):
                    self._raise_failure_locked()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise DurableStreamWriterTimeout(
                            "durable stream flush exceeded supervisor_join_timeout_s"
                        )
                    self._condition.wait(timeout=remaining)
                self._raise_failure_locked()
            finally:
                self._flush_requested = False
                self._condition.notify_all()

    def push(self, delta: ModelStreamDelta) -> None:
        if not isinstance(delta, ModelStreamDelta):
            raise TypeError("durable model stream push requires ModelStreamDelta")
        data = delta.text.encode("utf-8")
        if not data:
            return
        lane = self._ensure_lane(delta.channel)
        position = 0
        with self._condition:
            if self._closing or self._closed:
                raise DurableStreamWriteError("durable model stream writer is closed")
            if any(value.reset_before_append for value in self._lanes.values()):
                # A recovered successful call emits no deltas and preserves its open/sealed bytes.
                # The first new delta proves that the lifecycle admitted a replacement dispatch;
                # reset every pre-existing kernel lane before recording any replacement content.
                self._reset_prior_lanes_locked()
            while position < len(data):
                self._raise_failure_locked()
                while self._buffered_bytes_locked() >= self._observer.max_buffer_bytes:
                    self._condition.notify_all()
                    self._condition.wait(timeout=self._observer.flush_interval_s)
                    self._raise_failure_locked()
                    if self._closing:
                        raise DurableStreamWriteError("durable model stream writer is closing")
                capacity = self._observer.max_buffer_bytes - self._buffered_bytes_locked()
                take = min(capacity, len(data) - position)
                # ``data`` is complete UTF-8. Avoid splitting a multi-byte scalar at the bounded
                # queue edge so every independently persisted batch remains decodable.
                while take > 0:
                    try:
                        data[position : position + take].decode("utf-8")
                    except UnicodeDecodeError as exc:
                        if exc.reason != "unexpected end of data":
                            raise DurableStreamWriteError(
                                "model stream delta contains invalid UTF-8"
                            ) from exc
                        take -= 1
                        continue
                    break
                if take == 0:
                    self._condition.notify_all()
                    self._condition.wait(timeout=self._observer.flush_interval_s)
                    continue
                if not lane.buffer:
                    lane.flush_at = self._observer.monotonic() + self._observer.flush_interval_s
                lane.buffer.extend(data[position : position + take])
                position += take
                self._condition.notify_all()

    def _next_batch_locked(self) -> tuple[_Lane, bytes] | None:
        now = self._observer.monotonic()
        channels = sorted(self._lanes)
        for step in range(len(channels)):
            index = (self._channel_cursor + step) % len(channels)
            channel = channels[index]
            lane = self._lanes[channel]
            if not lane.buffer:
                continue
            due = self._flush_requested or (lane.flush_at is not None and lane.flush_at <= now)
            if not self._closing and len(lane.buffer) < self._observer.chunk_bytes and not due:
                continue
            maximum = min(len(lane.buffer), self._observer.chunk_bytes)
            size = _utf8_prefix_size(lane.buffer, maximum)
            data = bytes(lane.buffer[:size])
            del lane.buffer[:size]
            lane.flush_at = (
                now + self._observer.flush_interval_s if lane.buffer else None
            )
            self._channel_cursor = (index + 1) % len(channels)
            return lane, data
        return None

    def _wait_timeout_locked(self) -> float | None:
        deadlines = [
            lane.flush_at
            for lane in self._lanes.values()
            if lane.buffer and lane.flush_at is not None
        ]
        if not deadlines:
            return None
        return max(0.0, min(deadlines) - self._observer.monotonic())

    def _append(self, lane: _Lane, data: bytes) -> None:
        head = lane.head
        result = self._observer.write_authority.guard_external_call(
            lambda: self._store.append(
                lane.identity,
                generation=head.generation,
                start_offset=head.cursor_bytes,
                data=data,
                writer_token=self._observer.writer_token,
            )
        )
        if result.status not in {"committed", "already_committed"}:
            self._reject("append", result.status)
        if result.head is None or result.chunk is None:
            raise DurableStreamWriteError("accepted durable stream append omitted evidence")
        if (
            result.head.identity != lane.identity
            or result.chunk.identity != lane.identity
            or result.chunk.generation != head.generation
            or result.chunk.sequence != head.next_chunk_sequence
            or result.chunk.start_offset != head.cursor_bytes
            or result.chunk.end_offset != head.cursor_bytes + len(data)
            or result.chunk.sha256 != hashlib.sha256(data).hexdigest()
            or result.head.generation != head.generation
            or result.head.cursor_bytes != result.chunk.end_offset
            or result.head.next_chunk_sequence != result.chunk.sequence + 1
            or result.head.state != "open"
        ):
            raise DurableStreamWriteError("durable stream append evidence disagrees with batch")
        lane.digest.update(data)  # type: ignore[attr-defined]
        lane.head = result.head

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    if self._aborting:
                        return
                    batch = self._next_batch_locked()
                    if batch is None:
                        if self._closing and not any(
                            lane.buffer for lane in self._lanes.values()
                        ):
                            return
                        self._condition.wait(timeout=self._wait_timeout_locked())
                        continue
                    self._inflight = True
                lane, data = batch
                self._append(lane, data)
                with self._condition:
                    self._inflight = False
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._inflight = False
                self._failure = exc
                self._condition.notify_all()

    def _reconcile_completed_output(self, outcome: ModelStreamOutcome) -> None:
        if outcome.status != "completed" or outcome.final_text is None:
            return
        lane = self._lanes["output"]
        authoritative = outcome.final_text.encode("utf-8")
        if (
            lane.head.cursor_bytes == len(authoritative)
            and lane.digest.hexdigest() == hashlib.sha256(authoritative).hexdigest()  # type: ignore[attr-defined]
        ):
            return
        # Recovery can publish the durable model result before the observer's buffered suffix is
        # flushed. Rebuild from the settled output rather than sealing a truncated open prefix.
        self._reset_lane_locked(lane)
        pending = bytearray(authoritative)
        while pending:
            size = _utf8_prefix_size(
                pending,
                min(len(pending), self._observer.chunk_bytes),
            )
            data = bytes(pending[:size])
            del pending[:size]
            self._append(lane, data)

    def close(self, outcome: ModelStreamOutcome) -> None:
        if not isinstance(outcome, ModelStreamOutcome):
            raise TypeError("durable model stream close requires ModelStreamOutcome")
        with self._condition:
            if self._closed:
                return
            self._closing = True
            self._condition.notify_all()
        self._worker.join(timeout=self._observer.supervisor_join_timeout_s)
        if self._worker.is_alive():
            raise DurableStreamWriterTimeout(
                "durable stream flush worker exceeded supervisor_join_timeout_s"
            )
        with self._condition:
            self._raise_failure_locked()
            self._reconcile_completed_output(outcome)
        for channel in sorted(self._lanes):
            lane = self._lanes[channel]
            final_sha256 = lane.digest.hexdigest()  # type: ignore[attr-defined]
            result = self._observer.write_authority.guard_external_call(
                lambda: self._store.seal(
                    lane.identity,
                    generation=lane.head.generation,
                    final_size_bytes=lane.head.cursor_bytes,
                    final_sha256=final_sha256,
                    writer_token=self._observer.writer_token,
                )
            )
            if result.status not in {"sealed", "already_sealed"}:
                self._reject("seal", result.status)
            if result.head is None:
                raise DurableStreamWriteError("accepted durable stream seal omitted its head")
            lane.head = result.head
        with self._condition:
            self._closed = True

    def abort(self) -> None:
        """Stop the worker without reconciling or sealing an unprepared generation."""

        with self._condition:
            if self._closed:
                return
            self._aborting = True
            self._closing = True
            self._condition.notify_all()
        self._worker.join(timeout=self._observer.supervisor_join_timeout_s)
        if self._worker.is_alive():
            raise DurableStreamWriterTimeout(
                "durable stream abort exceeded supervisor_join_timeout_s"
            )
        with self._condition:
            self._closed = True
            for lane in self._lanes.values():
                lane.buffer.clear()
                lane.flush_at = None
            self._condition.notify_all()


__all__ = [
    "DurableStreamWriteError",
    "DurableStreamWriteRejected",
    "DurableStreamWriterTimeout",
    "DurableModelStreamObserver",
]
