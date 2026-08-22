"""Private, append-only model-stream content for replay and Studio hydration.

``model-content.jsonl`` is deliberately separate from the durable operation log.  It contains
authored model content and therefore inherits the run directory's private access boundary.  The
writer batches provider deltas into bounded segments; the reader treats each JSONL line as an
independent recovery unit and reports an opened stream without a valid close as ``abandoned``.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import threading
import weakref
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Literal, TextIO, TypeAlias

from monoid_agent_kernel.core._util import utc_timestamp
from monoid_agent_kernel.core._verified_file import (
    VerifiedFileIdentity,
    file_identity,
    open_verified_append_text,
    open_verified_regular_fd,
    verified_file_is_safe,
)
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
_WINDOWS_REPARSE_POINT_ATTRIBUTE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)
_LOGGER = logging.getLogger("monoid_agent_kernel.core.model_content")
"""Named under ``core`` like every other logger in this package (``core.model_stream``,
``core.sync_bridge``), so a deployment that quiets or routes ``monoid_agent_kernel.core`` reaches
this one too."""

_ACTIVE_STORE_LOCK = threading.RLock()
_ACTIVE_STORES: dict[str, set[weakref.ReferenceType[ModelContentStore]]] = {}

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
    retryable: bool = False
    config_recoverable: bool = False
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


# The sidecar was the first artifact to need a verified inode identity, so the name it was given
# is this module's. The type itself is not about model content and now lives beside the verified
# open that produces it; this alias keeps the sidecar-local vocabulary readable.
ModelContentFileIdentity = VerifiedFileIdentity


@dataclass(frozen=True)
class ActiveModelContentState:
    """Process-local active writers and the sidecar descriptor they own."""

    store_count: int = 0
    stream_ids: frozenset[str] = frozenset()
    file_identity: ModelContentFileIdentity | None = None


@dataclass(eq=False)
class ModelContentMutationWatch:
    """Request-scoped notice of registry or writer changes for one sidecar path."""

    _registry_key: str
    _changed: bool = False

    @property
    def changed(self) -> bool:
        """Whether the watched path changed before this observation point."""

        with _ACTIVE_STORE_LOCK:
            return self._changed


_ACTIVE_STORE_WATCHERS: dict[str, set[ModelContentMutationWatch]] = {}


@contextmanager
def watch_active_model_content(path: Path) -> Iterator[ModelContentMutationWatch]:
    """Watch one sidecar path for a bounded snapshot operation.

    Watch registrations live only for the duration of the caller's operation. This catches a
    complete create/open/close/remove ABA cycle without retaining per-run tombstones or making
    unrelated runs invalidate each other's snapshots.
    """

    key = _model_content_registry_key(path)
    watch = ModelContentMutationWatch(key)
    with _ACTIVE_STORE_LOCK:
        _ACTIVE_STORE_WATCHERS.setdefault(key, set()).add(watch)
    try:
        yield watch
    finally:
        with _ACTIVE_STORE_LOCK:
            watchers = _ACTIVE_STORE_WATCHERS.get(key)
            if watchers is not None:
                watchers.discard(watch)
                if not watchers:
                    _ACTIVE_STORE_WATCHERS.pop(key, None)


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
        self._registry_key = _model_content_registry_key(path)
        self._registry_ref: weakref.ReferenceType[ModelContentStore] = weakref.ref(
            self,
            lambda reference, key=self._registry_key: _remove_active_store(key, reference),
        )
        with _ACTIVE_STORE_LOCK:
            _ACTIVE_STORES.setdefault(self._registry_key, set()).add(self._registry_ref)
            _mark_active_store_mutation_locked(self._registry_key)

    def open(self, context: ModelStreamContext) -> ModelStreamWriter:
        with self._lock:
            if self._disabled or self._closing or self._closed:
                return NOOP_MODEL_STREAM_WRITER
            writer = _ModelContentWriter(self, context)
            persisted = self._append(
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
            if not persisted:
                return NOOP_MODEL_STREAM_WRITER
            # Publish the process-local writer only after its opening record and descriptor are
            # durable. active_state() can therefore never observe an active id without the exact
            # sidecar identity that owns it.
            self._writers.add(writer)
            with _ACTIVE_STORE_LOCK:
                _mark_active_store_mutation_locked(self._registry_key)
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
        _remove_active_store(self._registry_key, self._registry_ref)

    def discard(self) -> None:
        """Release owned handles and buffered writers without publishing their pending data.

        Revoked activation cleanup uses this path. Setting ``_disabled`` while holding the store
        lock makes every timer or concurrent close that has not already entered ``_append`` fail
        closed. An append already holding the lock completes before this method can return.
        """

        with self._lock:
            if self._closed:
                return
            self._disabled = True
            self._closing = True
            writers = tuple(self._writers)
        for writer in writers:
            writer._discard()
        with self._lock:
            self._closed = True
            self._closing = False
            handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        _remove_active_store(self._registry_key, self._registry_ref)

    def flush(self) -> bool:
        """Persist every currently buffered stream segment without closing its writer.

        Studio uses this process-local coordination point before reset hydration. The private
        sidecar observer runs before the live broker observer, so a reset visible to Studio has
        already returned from the corresponding sidecar ``push``; flushing here commits that exact
        prefix before the client accepts the reset cursor. ``False`` means the store could not
        persist that prefix; callers must not advance a live cursor from a stale snapshot.
        """

        with self._lock:
            if self._disabled:
                return False
            if self._closing or self._closed:
                # A registry snapshot may race close after copying this store reference. It can no
                # longer prove a live descriptor/path identity, so explicit hydration retries.
                return False
            if not self._handle_matches_path_locked():
                self._disable_locked("the sidecar was replaced after it was opened")
                return False
            writers = tuple(self._writers)
        for writer in writers:
            writer._flush_pending()
        with self._lock:
            if self._disabled or not self._handle_matches_path_locked():
                self._disable_locked("the sidecar was replaced after it was opened")
                return False
            return True

    def active_stream_ids(self) -> frozenset[str]:
        """Return stream ids with writers still owned by this live process-local store."""

        try:
            state = self.active_state()
        except OSError:
            return frozenset()
        return frozenset() if state is None else state.stream_ids

    def active_state(self) -> ActiveModelContentState | None:
        """Return active writers tied to the currently named sidecar descriptor.

        A regular-file replacement can leave this store writing a displaced inode. Such a handle
        cannot establish that a pathname reader observes the same private prefix, so the store is
        disabled and the active proof fails closed.
        """

        with self._lock:
            if self._closing or self._closed:
                return None
            if self._disabled or not self._handle_matches_path_locked():
                self._disable_locked("the sidecar was replaced after it was opened")
                raise OSError("active model-content descriptor no longer matches its path")
            # A writer stays active until its terminal append (or abandonment flush) completes and
            # unregisters it. Treating its earlier private ``_closed`` flag as authoritative would
            # expose an unmarked interval where a snapshot could revive an abandoned prefix.
            stream_ids = frozenset(writer._context.stream_id for writer in self._writers)
            identity: ModelContentFileIdentity | None = None
            if self._handle is not None:
                try:
                    identity = file_identity(os.fstat(self._handle.fileno()))
                except (OSError, ValueError) as exc:
                    self._disable_locked("its descriptor became unavailable")
                    raise OSError("active model-content descriptor is unavailable") from exc
            if stream_ids and identity is None:
                self._disable_locked("a live writer lost its verified descriptor")
                raise OSError("active model-content writer has no verified descriptor")
            return ActiveModelContentState(
                store_count=1,
                stream_ids=stream_ids,
                file_identity=identity,
            )

    def _handle_matches_path_locked(self) -> bool:
        """Whether the active descriptor still names this run directory's single-link file."""

        if self._handle is None:
            return True
        try:
            opened = os.fstat(self._handle.fileno())
            named = self.path.lstat()
        except (OSError, ValueError):
            return False
        return (
            stat.S_ISREG(opened.st_mode)
            and stat.S_ISREG(named.st_mode)
            and opened.st_nlink == 1
            and named.st_nlink == 1
            and os.path.samestat(opened, named)
        )

    def _unregister(self, writer: _ModelContentWriter) -> None:
        with self._lock:
            if writer not in self._writers:
                return
            self._writers.remove(writer)
            with _ACTIVE_STORE_LOCK:
                _mark_active_store_mutation_locked(self._registry_key)

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
                self._disable_locked("an append failed and may have torn its line")
                return False
        return True

    def _ensure_handle_locked(self) -> TextIO | None:
        if self._handle is not None:
            return self._handle
        # Refusal is terminal. The verified open declines a planted link or a special file, which is
        # a property of the path rather than a transient, so retrying on the next segment only
        # re-runs the same refusal.
        handle = open_verified_append_text(self.path)
        if handle is None:
            self._disable_locked("it could not be safely opened")
            return None
        self._handle = handle
        return self._handle

    def _disable_locked(self, reason: str) -> None:
        """Stop this store for good, and say so once. Caller holds ``_lock``.

        This is the oldest of the three sidecar writers and the one that owns its own open, so a
        rule bound where the *recorder* logs leaves it silent -- which is how it was written first.
        Its terminal state has several doors, not one: a refused open, a torn append, three
        detections that the file under the descriptor is no longer the file at the path, and two
        that the descriptor itself is gone. The substitution ones are what the open-time check
        refuses, noticed one moment later, and a reader cannot tell a truncated sidecar from a
        complete one, so silence there is worse than at the door that was already loud.

        The message names ``self.path.name`` -- the basename of the file this store was pointed
        at, never the directory holding it. Hardcoding ``MODEL_CONTENT_FILENAME`` instead would
        name a file that does not exist for a caller who chose another one. The guarantee is
        exactly "a basename, not a path": for every shipped configuration that basename is the
        sidecar filename, and a caller who points this class at a *directory* gets that
        directory's name -- which is also a caller whose open is about to fail, since the
        constructor stores the path verbatim while the registry key normalizes it.
        """

        if self._disabled:
            return
        self._disabled = True
        artifact = self.path.name
        _LOGGER.warning(
            "%s: %s; this activation records no more of it",
            artifact,
            reason,
            extra={"monoid_run_id": self.run_id, "monoid_artifact": artifact},
        )


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
                "retryable": outcome.retryable,
                "config_recoverable": outcome.config_recoverable,
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

    def _discard(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._buffer_channel = None
            self._buffer_parts.clear()
            self._buffer_bytes = 0
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

    def _flush_pending(self) -> None:
        with self._lock:
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
    retryable: bool = False
    config_recoverable: bool = False

    def snapshot(self) -> ModelContentSnapshot:
        # A valid segment after a malformed/missing record has no trustworthy placement offset.
        # Preserve only the shared output/reasoning prefix proven contiguous from index zero.
        ordered: list[tuple[int, tuple[ModelStreamChannel, str]]] = []
        for expected_index, item in enumerate(sorted(self.segments.items())):
            if item[0] != expected_index:
                break
            ordered.append(item)
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
            retryable=self.retryable,
            config_recoverable=self.config_recoverable,
            segment_count=len(ordered),
            last_segment_index=indexes[-1] if indexes else None,
        )


def read_model_content(
    path: Path,
    *,
    expected_identity: ModelContentFileIdentity | None = None,
) -> ModelContentReadResult:
    """Recover valid records from a sidecar path or run directory.

    Missing files, malformed lines, torn UTF-8, unknown record kinds, and invalid field shapes are
    skipped. This function is a presentation/recovery aid and never makes a run unreadable. When
    ``expected_identity`` is supplied, descriptor and pathname identity must remain equal to that
    verified sidecar for the whole read or the snapshot fails closed with ``OSError``.
    """

    # The sidecar name is fixed. A run id may itself end in ``.jsonl``, so suffix-based file
    # detection mistakes a valid run directory for a JSONL file and silently returns no content.
    path = _model_content_file_path(path)
    streams: dict[str, _RecoveredStream] = {}
    order: list[str] = []
    settled_texts: dict[str, str] = {}
    skipped = 0
    handle = open_model_content_for_read(path, expected_identity=expected_identity)
    if handle is None:
        if expected_identity is not None:
            raise OSError("model-content sidecar identity changed before read")
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
    result = ModelContentReadResult(
        snapshots=tuple(streams[stream_id].snapshot() for stream_id in order),
        settled_texts=settled_texts,
        skipped_records=skipped,
    )
    if expected_identity is not None and not model_content_file_matches_identity(
        path,
        expected_identity,
    ):
        raise OSError("model-content sidecar identity changed during read")
    return result


def flush_active_model_content(path: Path) -> int:
    """Flush in-process stores for ``path`` and return how many live stores were reached.

    This helper is intentionally separate from :func:`read_model_content`: ordinary artifact reads
    remain side-effect free, while a presentation layer handling an explicit live-stream reset can
    request a coordinated, up-to-date snapshot. Cross-process writers are outside the Reference
    process-local broker contract.
    """

    key = _model_content_registry_key(path)
    with _ACTIVE_STORE_LOCK:
        references = tuple(_ACTIVE_STORES.get(key, ()))
    stores = tuple(store for reference in references if (store := reference()) is not None)
    failures = 0
    for store in stores:
        try:
            if not store.flush():
                failures += 1
        except Exception:  # noqa: BLE001 - explicit hydration must fail closed
            failures += 1
    if failures:
        raise OSError(f"failed to flush {failures} active model-content store(s)")
    return len(stores)


def active_model_content_stream_ids(path: Path) -> frozenset[str]:
    """Return exact stream ids that still have process-local writers for ``path``."""

    try:
        return active_model_content_state(path).stream_ids
    except OSError:
        return frozenset()


def active_model_content_state(path: Path) -> ActiveModelContentState:
    """Return active stream ids plus their shared, verified descriptor identity.

    Every registered store for a path must either have no open descriptor or name the same inode.
    Any displaced, disabled, or conflicting store invalidates the aggregate proof.
    """

    with watch_active_model_content(path) as mutation_watch:
        key = mutation_watch._registry_key
        with _ACTIVE_STORE_LOCK:
            references = tuple(_ACTIVE_STORES.get(key, ()))
        active: set[str] = set()
        identities: set[ModelContentFileIdentity] = set()
        store_count = 0
        for reference in references:
            store = reference()
            if store is None:
                continue
            state = store.active_state()
            if state is None:
                continue
            store_count += 1
            active.update(state.stream_ids)
            if state.file_identity is not None:
                identities.add(state.file_identity)
        if len(identities) > 1:
            raise OSError("active model-content stores reference conflicting sidecar identities")
        identity = next(iter(identities), None)
        if active and identity is None:
            raise OSError("active model-content streams have no verified sidecar identity")
        if mutation_watch.changed:
            raise OSError("active model-content state changed while it was inspected")
        return ActiveModelContentState(
            store_count=store_count,
            stream_ids=frozenset(active),
            file_identity=identity,
        )


def _model_content_registry_key(path: Path) -> str:
    # Windows paths are case-insensitive. Detect the canonical filename with the same semantics as
    # the final key so a differently-cased file path is not mistaken for a run directory.
    path = _model_content_file_path(path)
    try:
        # Resolve the containing run directory, not the final artifact. Resolving the final member
        # would alias a planted symlink with its target and let a snapshot flush an unrelated store.
        resolved = path.parent.resolve(strict=False) / path.name
    except (OSError, RuntimeError):
        resolved = path.absolute()
    return os.path.normcase(str(resolved))


def _model_content_file_path(path: Path) -> Path:
    try:
        metadata = path.lstat()
    except OSError:
        metadata = None
    is_reparse_point = bool(
        metadata is not None
        and getattr(metadata, "st_file_attributes", 0) & _WINDOWS_REPARSE_POINT_ATTRIBUTE
    )
    if metadata is not None and stat.S_ISDIR(metadata.st_mode) and not is_reparse_point:
        # A valid run id can equal the reserved sidecar basename. Prefer the existing filesystem
        # type over the lexical shortcut, without following a planted symlink or Windows junction.
        return path / MODEL_CONTENT_FILENAME
    if os.path.normcase(path.name) != os.path.normcase(MODEL_CONTENT_FILENAME):
        return path / MODEL_CONTENT_FILENAME
    return path


def model_content_file_is_safe(path: Path, *, allow_missing: bool = True) -> bool:
    """Whether a sidecar path is absent or an ordinary file, without following links.

    The sidecar-named entry point for :func:`verified_file_is_safe`, kept because Studio calls it
    by this name; the rule it applies belongs to every run-directory artifact, not to this one.
    """

    return verified_file_is_safe(path, allow_missing=allow_missing)


def model_content_file_identity(
    path: Path,
    *,
    allow_missing: bool = True,
) -> ModelContentFileIdentity | None:
    """Return the identity of one verified sidecar pathname without following links."""

    path = _model_content_file_path(path)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if allow_missing:
            return None
        raise
    except OSError as exc:
        raise OSError("model-content sidecar metadata is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise OSError("model-content sidecar is not a verified single-link regular file")
    return file_identity(metadata)


def model_content_file_matches_identity(
    path: Path,
    expected_identity: ModelContentFileIdentity,
) -> bool:
    """Whether the currently named sidecar still has ``expected_identity``."""

    try:
        return model_content_file_identity(path, allow_missing=False) == expected_identity
    except OSError:
        return False


def open_model_content_for_read(
    path: Path,
    *,
    expected_identity: ModelContentFileIdentity | None = None,
) -> BinaryIO | None:
    """Open the fixed sidecar for a verified, no-indirection binary read."""

    path = _model_content_file_path(path)
    descriptor = open_verified_regular_fd(
        path,
        os.O_RDONLY,
        expected_identity=expected_identity,
    )
    if descriptor is None:
        return None
    try:
        return os.fdopen(descriptor, "rb")
    except (OSError, ValueError):
        try:
            os.close(descriptor)
        except OSError:
            pass
        return None


def _remove_active_store(
    key: str,
    reference: weakref.ReferenceType[ModelContentStore],
) -> None:
    with _ACTIVE_STORE_LOCK:
        references = _ACTIVE_STORES.get(key)
        if references is None:
            return
        if reference not in references:
            return
        references.remove(reference)
        _mark_active_store_mutation_locked(key)
        if not references:
            _ACTIVE_STORES.pop(key, None)


def _mark_active_store_mutation_locked(key: str) -> None:
    """Notify request-scoped watchers while ``_ACTIVE_STORE_LOCK`` is held."""

    for watch in tuple(_ACTIVE_STORE_WATCHERS.get(key, ())):
        watch._changed = True


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
    retryable = record.get("retryable", False)
    # Optional and defaulted, like ``retryable``: a sidecar written before this key existed is a
    # valid record, not a skipped one. Present-but-mistyped is still refused below.
    config_recoverable = record.get("config_recoverable", False)
    finished_at = record.get("finished_at")
    if final_text is not None and not isinstance(final_text, str):
        return False
    if usage is not None and not isinstance(usage, dict):
        return False
    if error_code is not None and not isinstance(error_code, str):
        return False
    if type(retryable) is not bool:
        return False
    if type(config_recoverable) is not bool:
        return False
    if not isinstance(finished_at, str) or not finished_at.endswith("Z"):
        return False
    recovered.status = status
    recovered.final_text = final_text
    recovered.usage = usage
    recovered.error_code = error_code
    recovered.retryable = retryable
    recovered.config_recoverable = config_recoverable
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
