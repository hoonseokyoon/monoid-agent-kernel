"""Private, append-only model-stream content for replay and Studio hydration.

``model-content.jsonl`` is deliberately separate from the durable operation log.  It contains
authored model content and therefore inherits the run directory's private access boundary.  The
writer batches provider deltas into bounded segments; the reader treats each JSONL line as an
independent recovery unit and reports an opened stream without a valid close as ``abandoned``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TextIO, TypeAlias

from monoid_agent_kernel.core._util import utc_timestamp
from monoid_agent_kernel.core.json_ingress import (
    loads_json_ingress,
    normalize_json_ingress,
    normalize_unicode_scalars,
)
from monoid_agent_kernel.core.model_io import content_digest, content_length
from monoid_agent_kernel.core.model_stream import (
    NOOP_MODEL_STREAM_WRITER,
    ModelStreamChannel,
    ModelStreamContext,
    ModelStreamDelta,
    ModelStreamOutcome,
    ModelStreamStatus,
    ModelStreamWriter,
)
from monoid_agent_kernel.identifiers import accepts_namespaced_id, namespaced_id

MODEL_CONTENT_SCHEMA_VERSION = namespaced_id("model-content.v1")
MODEL_CONTENT_FILENAME = "model-content.jsonl"
DEFAULT_MODEL_CONTENT_BATCH_INTERVAL_S = 0.25
DEFAULT_MODEL_CONTENT_SEGMENT_BYTES = 4096

ModelContentSnapshotStatus: TypeAlias = Literal[
    "completed",
    "interrupted",
    "failed",
    "cancelled",
    "timed_out",
    "abandoned",
]


@dataclass(frozen=True)
class ModelContentSnapshot:
    """A tolerant reconstruction of one stream from valid sidecar records."""

    context: ModelStreamContext
    status: ModelContentSnapshotStatus
    output_text: str = ""
    reasoning_text: str = ""
    final_text: str | None = None
    usage: Mapping[str, Any] | None = None
    error_code: str | None = None
    segment_count: int = 0
    last_segment_index: int | None = None

    def __post_init__(self) -> None:
        if self.usage is not None:
            object.__setattr__(self, "usage", dict(self.usage))

    @property
    def best_output_text(self) -> str:
        """Settled output when available, otherwise the durable partial output."""

        return self.final_text if self.final_text is not None else self.output_text


@dataclass(frozen=True)
class ModelContentReadResult:
    """Recovered streams plus content-addressed settled text from one sidecar."""

    snapshots: tuple[ModelContentSnapshot, ...] = ()
    settled_texts: Mapping[str, str] = field(default_factory=dict)
    skipped_records: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "settled_texts", dict(self.settled_texts))


class ModelContentStore:
    """Best-effort observer that writes ``monoid.model-content.v1`` JSONL records.

    The file is opened lazily.  Its first delta is persisted immediately, then same-channel deltas
    are coalesced for at most ``batch_interval_s`` or ``max_segment_bytes``.  A channel switch
    flushes the previous channel before accepting the next one.
    """

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        batch_interval_s: float = DEFAULT_MODEL_CONTENT_BATCH_INTERVAL_S,
        max_segment_bytes: int = DEFAULT_MODEL_CONTENT_SEGMENT_BYTES,
    ) -> None:
        if batch_interval_s <= 0:
            raise ValueError("model content batch interval must be positive")
        if max_segment_bytes < 4:
            # One Unicode scalar can occupy four UTF-8 bytes and must never be split into invalid
            # text merely to satisfy a caller-selected byte limit.
            raise ValueError("model content segment size must be at least four bytes")
        self.path = path
        self.run_id = run_id
        self.batch_interval_s = batch_interval_s
        self.max_segment_bytes = max_segment_bytes
        self._lock = threading.RLock()
        self._handle: TextIO | None = None
        self._writers: set[_ModelContentWriter] = set()
        self._settled_text_digests: set[str] = set()
        self._disabled = False
        self._closing = False
        self._closed = False

    def open(self, context: ModelStreamContext) -> ModelStreamWriter:
        with self._lock:
            if self._disabled or self._closing or self._closed:
                return NOOP_MODEL_STREAM_WRITER
            writer = _ModelContentWriter(self, context)
            self._writers.add(writer)
        self._append(
            {
                "schema_version": MODEL_CONTENT_SCHEMA_VERSION,
                "kind": "stream_opened",
                "run_id": context.run_id,
                "root_run_id": context.root_run_id,
                "turn_id": context.turn_id,
                "stream_id": context.stream_id,
                "step": context.step,
                "provider": context.provider,
                "model": context.model,
                "started_at": context.started_at,
            }
        )
        return writer

    def settled_text(self, text: str, digest: str, text_len: int) -> None:
        """Write one content-addressed settled result, retrying after a failed append."""

        if content_length(text) != text_len or content_digest(text) != digest:
            return
        with self._lock:
            if digest in self._settled_text_digests:
                return
        persisted = self._append(
            {
                "schema_version": MODEL_CONTENT_SCHEMA_VERSION,
                "kind": "settled_text",
                "run_id": self.run_id,
                "final_text": text,
                "final_text_digest": digest,
                "final_text_len": text_len,
                "recorded_at": utc_timestamp(),
            }
        )
        if persisted:
            with self._lock:
                self._settled_text_digests.add(digest)

    def close(self) -> None:
        with self._lock:
            if self._closed or self._closing:
                return
            self._closing = True
            writers = tuple(self._writers)
        # Flush buffered partials, but intentionally leave streams without a terminal record.  A
        # reader then calls them abandoned instead of inventing a provider outcome.
        for writer in writers:
            writer._abandon()
        with self._lock:
            self._closed = True
            self._closing = False
            handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    def _unregister(self, writer: _ModelContentWriter) -> None:
        with self._lock:
            self._writers.discard(writer)

    def _append(self, payload: dict[str, Any]) -> bool:
        try:
            normalized = normalize_json_ingress(payload)
            line = json.dumps(
                normalized,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
            line.encode("utf-8")
        except (TypeError, ValueError, UnicodeError, RecursionError):
            return False

        with self._lock:
            if self._disabled or self._closed:
                return False
            handle = self._ensure_handle_locked()
            if handle is None:
                return False
            try:
                handle.write(line + "\n")
                handle.flush()
            except (OSError, UnicodeError):
                # A partial write may have torn the current line.  Disable this handle so a later
                # record cannot be glued to it; the next recorder instance isolates the tail.
                self._disabled = True
                return False
        return True

    def _ensure_handle_locked(self) -> TextIO | None:
        if self._handle is not None:
            return self._handle
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size:
                size = self.path.stat().st_size
                with self.path.open("rb") as existing:
                    existing.seek(size - 1)
                    torn_tail = existing.read(1) != b"\n"
            else:
                torn_tail = False
            self._handle = self.path.open("a", encoding="utf-8", newline="\n")
            if torn_tail:
                self._handle.write("\n")
                self._handle.flush()
        except OSError:
            self._disabled = True
            self._handle = None
        return self._handle


class _ModelContentWriter:
    def __init__(self, store: ModelContentStore, context: ModelStreamContext) -> None:
        self._store = store
        self._context = context
        self._lock = threading.RLock()
        self._buffer_channel: ModelStreamChannel | None = None
        self._buffer_parts: list[str] = []
        self._buffer_bytes = 0
        self._next_segment_index = 0
        self._first_segment_written = False
        self._timer: threading.Timer | None = None
        self._closed = False

    def push(self, delta: ModelStreamDelta) -> None:
        if not delta.text:
            return
        text = normalize_unicode_scalars(delta.text)
        with self._lock:
            if self._closed:
                return
            for chunk in _utf8_chunks(text, self._store.max_segment_bytes):
                if not self._first_segment_written:
                    self._write_segment_locked(delta.channel, chunk)
                    self._first_segment_written = True
                    continue
                if self._buffer_channel is not None and self._buffer_channel != delta.channel:
                    self._flush_locked()
                if self._buffer_parts and (
                    self._buffer_bytes + len(chunk.encode("utf-8")) > self._store.max_segment_bytes
                ):
                    self._flush_locked()
                self._buffer_channel = delta.channel
                self._buffer_parts.append(chunk)
                self._buffer_bytes += len(chunk.encode("utf-8"))
                if self._buffer_bytes >= self._store.max_segment_bytes:
                    self._flush_locked()
                else:
                    self._schedule_flush_locked()

    def close(self, outcome: ModelStreamOutcome) -> None:
        with self._lock:
            if self._closed:
                return
            self._flush_locked()
            self._closed = True
            self._cancel_timer_locked()
        self._store._append(
            {
                "schema_version": MODEL_CONTENT_SCHEMA_VERSION,
                "kind": "stream_closed",
                "run_id": self._context.run_id,
                "stream_id": self._context.stream_id,
                "status": outcome.status,
                "final_text": outcome.final_text,
                "usage": None if outcome.usage is None else dict(outcome.usage),
                "error_code": outcome.error_code,
                "finished_at": utc_timestamp(),
            }
        )
        self._store._unregister(self)

    def _abandon(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._flush_locked()
            self._closed = True
            self._cancel_timer_locked()
        self._store._unregister(self)

    def _schedule_flush_locked(self) -> None:
        if self._timer is not None:
            return
        timer = threading.Timer(self._store.batch_interval_s, self._flush_due)
        timer.daemon = True
        self._timer = timer
        timer.start()

    def _flush_due(self) -> None:
        with self._lock:
            self._timer = None
            if not self._closed:
                self._flush_locked()

    def _cancel_timer_locked(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()

    def _flush_locked(self) -> None:
        if not self._buffer_parts or self._buffer_channel is None:
            self._cancel_timer_locked()
            return
        channel = self._buffer_channel
        text = "".join(self._buffer_parts)
        self._buffer_channel = None
        self._buffer_parts.clear()
        self._buffer_bytes = 0
        self._cancel_timer_locked()
        self._write_segment_locked(channel, text)

    def _write_segment_locked(self, channel: ModelStreamChannel, text: str) -> None:
        index = self._next_segment_index
        self._next_segment_index += 1
        self._store._append(
            {
                "schema_version": MODEL_CONTENT_SCHEMA_VERSION,
                "kind": "stream_segment",
                "run_id": self._context.run_id,
                "stream_id": self._context.stream_id,
                "segment_index": index,
                "channel": channel,
                "text": text,
                "text_len": len(text),
                "emitted_at": utc_timestamp(),
            }
        )


def _utf8_chunks(text: str, limit: int) -> tuple[str, ...]:
    chunks: list[str] = []
    parts: list[str] = []
    size = 0
    for character in text:
        char_size = len(character.encode("utf-8"))
        if parts and size + char_size > limit:
            chunks.append("".join(parts))
            parts = []
            size = 0
        parts.append(character)
        size += char_size
    if parts:
        chunks.append("".join(parts))
    return tuple(chunks)


@dataclass
class _RecoveredStream:
    context: ModelStreamContext
    segments: dict[int, tuple[ModelStreamChannel, str]] = field(default_factory=dict)
    status: ModelStreamStatus | None = None
    final_text: str | None = None
    usage: Mapping[str, Any] | None = None
    error_code: str | None = None

    def snapshot(self) -> ModelContentSnapshot:
        ordered = sorted(self.segments.items())
        output = "".join(text for _, (channel, text) in ordered if channel == "output")
        reasoning = "".join(text for _, (channel, text) in ordered if channel == "reasoning")
        indexes = [index for index, _ in ordered]
        return ModelContentSnapshot(
            context=self.context,
            status=self.status or "abandoned",
            output_text=output,
            reasoning_text=reasoning,
            final_text=self.final_text,
            usage=self.usage,
            error_code=self.error_code,
            segment_count=len(ordered),
            last_segment_index=indexes[-1] if indexes else None,
        )


def read_model_content(path: Path) -> ModelContentReadResult:
    """Recover valid records from a sidecar path or run directory.

    Missing files, malformed lines, torn UTF-8, unknown record kinds, and invalid field shapes are
    skipped.  This function is a presentation/recovery aid and never makes a run unreadable.
    """

    # The sidecar name is fixed. A run id may itself end in ``.jsonl``, so suffix-based file
    # detection mistakes a valid run directory for a JSONL file and silently returns no content.
    if path.name != MODEL_CONTENT_FILENAME:
        path = path / MODEL_CONTENT_FILENAME
    streams: dict[str, _RecoveredStream] = {}
    order: list[str] = []
    settled_texts: dict[str, str] = {}
    skipped = 0
    try:
        handle = path.open("rb")
    except OSError:
        return ModelContentReadResult()
    with handle:
        for raw_line in handle:
            record = _read_record(raw_line)
            if record is None:
                skipped += 1
                continue
            kind = record.get("kind")
            if kind == "stream_opened":
                context = _context_from_record(record)
                if context is None or context.stream_id in streams:
                    skipped += 1
                    continue
                streams[context.stream_id] = _RecoveredStream(context)
                order.append(context.stream_id)
            elif kind == "stream_segment":
                if not _apply_segment(streams, record):
                    skipped += 1
            elif kind == "stream_closed":
                if not _apply_close(streams, record):
                    skipped += 1
            elif kind == "settled_text":
                if not _apply_settled_text(settled_texts, record):
                    skipped += 1
            else:
                skipped += 1
    return ModelContentReadResult(
        snapshots=tuple(streams[stream_id].snapshot() for stream_id in order),
        settled_texts=settled_texts,
        skipped_records=skipped,
    )


def _read_record(raw_line: bytes) -> dict[str, Any] | None:
    try:
        # Stream segments have no per-record content digest. Replacing invalid UTF-8 would turn a
        # torn byte sequence inside an otherwise valid JSON string into fabricated U+FFFD content,
        # so the whole record must be isolated instead.
        record = loads_json_ingress(raw_line.decode("utf-8").strip())
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    if not isinstance(record, dict):
        return None
    if not accepts_namespaced_id(record.get("schema_version"), "model-content.v1"):
        return None
    return record


def _context_from_record(record: dict[str, Any]) -> ModelStreamContext | None:
    string_fields = ("run_id", "root_run_id", "turn_id", "stream_id", "started_at")
    if not all(isinstance(record.get(field), str) and record[field] for field in string_fields):
        return None
    step = record.get("step")
    provider = record.get("provider")
    model = record.get("model")
    if not {"provider", "model"}.issubset(record):
        return None
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        return None
    if not record["started_at"].endswith("Z"):
        return None
    if provider is not None and not isinstance(provider, str):
        return None
    if model is not None and not isinstance(model, str):
        return None
    return ModelStreamContext(
        run_id=record["run_id"],
        root_run_id=record["root_run_id"],
        turn_id=record["turn_id"],
        stream_id=record["stream_id"],
        step=step,
        provider=provider,
        model=model,
        started_at=record["started_at"],
    )


def _apply_segment(streams: dict[str, _RecoveredStream], record: dict[str, Any]) -> bool:
    stream_id = record.get("stream_id")
    index = record.get("segment_index")
    channel = record.get("channel")
    text = record.get("text")
    text_len = record.get("text_len")
    emitted_at = record.get("emitted_at")
    recovered = streams.get(stream_id) if isinstance(stream_id, str) else None
    if (
        not isinstance(stream_id, str)
        or recovered is None
        or record.get("run_id") != recovered.context.run_id
        or recovered.status is not None
        or isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or channel not in {"output", "reasoning"}
        or not isinstance(text, str)
        or isinstance(text_len, bool)
        or not isinstance(text_len, int)
        or text_len != len(text)
        or not isinstance(emitted_at, str)
        or not emitted_at.endswith("Z")
        or index in recovered.segments
    ):
        return False
    recovered.segments[index] = (channel, text)
    return True


def _apply_close(streams: dict[str, _RecoveredStream], record: dict[str, Any]) -> bool:
    stream_id = record.get("stream_id")
    status = record.get("status")
    if not isinstance(stream_id, str) or stream_id not in streams:
        return False
    recovered = streams[stream_id]
    if (
        record.get("run_id") != recovered.context.run_id
        or recovered.status is not None
        or not {"final_text", "usage", "error_code", "finished_at"}.issubset(record)
        or status
        not in {
            "completed",
            "interrupted",
            "failed",
            "cancelled",
            "timed_out",
        }
    ):
        return False
    final_text = record.get("final_text")
    usage = record.get("usage")
    error_code = record.get("error_code")
    finished_at = record.get("finished_at")
    if final_text is not None and not isinstance(final_text, str):
        return False
    if usage is not None and not isinstance(usage, dict):
        return False
    if error_code is not None and not isinstance(error_code, str):
        return False
    if not isinstance(finished_at, str) or not finished_at.endswith("Z"):
        return False
    recovered.status = status
    recovered.final_text = final_text
    recovered.usage = usage
    recovered.error_code = error_code
    return True


def _apply_settled_text(settled: dict[str, str], record: dict[str, Any]) -> bool:
    digest = record.get("final_text_digest")
    text = record.get("final_text")
    text_len = record.get("final_text_len")
    run_id = record.get("run_id")
    recorded_at = record.get("recorded_at")
    if (
        not isinstance(run_id, str)
        or not run_id
        or not isinstance(recorded_at, str)
        or not recorded_at.endswith("Z")
        or not isinstance(digest, str)
        or not isinstance(text, str)
        or isinstance(text_len, bool)
        or not isinstance(text_len, int)
        or content_length(text) != text_len
        or content_digest(text) != digest
    ):
        return False
    settled[digest] = text
    return True
