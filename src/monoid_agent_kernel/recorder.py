from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from monoid_agent_kernel.core._event_log import (
    EventLogCorruption,
    EventLogTail,
    repair_event_log_tail_for_append,
    validate_committed_event_sequence,
)
from monoid_agent_kernel.core._util import (
    canonical_sha256,
    read_text_resilient,
    sha256_bytes,
    utc_timestamp,
    write_json_atomic,
)
from monoid_agent_kernel.core.events import AgentEvent, EventBus, EventSink, make_agent_event
from monoid_agent_kernel.core.json_ingress import (
    loads_json_ingress,
    normalize_json_ingress,
)
from monoid_agent_kernel.core.lifecycle import (
    SessionState,
    session_state_from_run_status,
    session_state_value,
)
from monoid_agent_kernel.core.manifest import RunManifest
from monoid_agent_kernel.core.model_calls import MODEL_CALLS_FILENAME, model_call_record
from monoid_agent_kernel.core.model_content import MODEL_CONTENT_FILENAME, ModelContentStore
from monoid_agent_kernel.core.model_io import ModelCallReceipt, content_digest, content_length
from monoid_agent_kernel.core.model_stream import (
    NOOP_MODEL_STREAM_WRITER,
    ModelStreamContext,
    ModelStreamWriter,
    safe_open_model_stream,
)
from monoid_agent_kernel.core.result import AgentArtifact
from monoid_agent_kernel.identifiers import namespaced_id

if TYPE_CHECKING:
    from monoid_agent_kernel.core.workspace import ChangedEntry, Workspace

_LOGGER = logging.getLogger("monoid_agent_kernel.recorder")
_PROPOSAL_LOCKS_GUARD = threading.Lock()
_PROPOSAL_LOCKS: dict[str, threading.RLock] = {}


@contextmanager
def proposal_snapshot_lock(run_dir: Path) -> Iterator[None]:
    """Serialize one run's proposal snapshot writes and live-directory consumers.

    Proposal files form one logical revision even though they occupy several paths. The reference
    backend and Studio use this same in-process lock while reading, exporting, approving, or
    applying them, so no consumer can observe the writer between ``diff.patch`` and the final
    atomic ``proposal.json`` replace.
    """
    key = str(run_dir.resolve())
    with _PROPOSAL_LOCKS_GUARD:
        lock = _PROPOSAL_LOCKS.setdefault(key, threading.RLock())
    with lock:
        yield


@dataclass
class JsonlEventSink:
    path: Path
    _handle: TextIO = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def emit(self, event: AgentEvent) -> None:
        _write_jsonl(self._handle, event.to_json())

    def close(self) -> None:
        self._handle.close()


@dataclass
class StdoutJsonlSink:
    handle: TextIO = field(default_factory=lambda: sys.stdout)

    def emit(self, event: AgentEvent) -> None:
        _write_jsonl(self.handle, event.to_json())

    def close(self) -> None:
        self.handle.flush()


# The parks a model turn starting must clear. The offline twin
# (``core/projections.py:_PARKED_STATES``) names the same set. PAUSED joined when the pause
# became visible on this surface: a resumed pump unparks it exactly like the other two.
_PARKED_STATE_VALUES = frozenset(
    {
        SessionState.AWAITING_INPUT.value,
        SessionState.AWAITING_TASKS.value,
        SessionState.PAUSED.value,
    }
)

# The failure-classification keys a parked ``turn.failed`` writes beside ``error`` /
# ``error_code``. One tuple, because three branches share one rule: the park writes them, the
# unpark clears them, and a non-failed terminal heals them.
_FAILURE_CLASSIFICATION_KEYS = (
    "provider_error_code",
    "http_status",
    "retryable",
    "config_recoverable",
    "provider_retried",
)


@dataclass
class StatusJsonSink:
    path: Path
    state: dict[str, Any] = field(default_factory=dict)

    def emit(self, event: AgentEvent) -> None:
        data = event.data
        if event.type == "run.started":
            self.state.update(
                {
                    "run_id": event.run_id,
                    "state": session_state_value(SessionState.RUNNING),
                    "terminal": False,
                    "started_at": event.timestamp,
                    "workspace": data.get("workspace"),
                    "workspace_backend": data.get("workspace_backend"),
                    "workspace_base_path": data.get("workspace_base_path"),
                    "manifest_path": data.get("manifest_path"),
                    "mode": data.get("mode"),
                    "model": data.get("model"),
                    "reasoning_effort": data.get("reasoning_effort"),
                }
            )
        elif event.type == "run.finished":
            state = session_state_from_run_status(
                str(data.get("status", "completed") or "completed"),
                error_code=str(data.get("error_code") or ""),
                terminal=True,
            )
            # No ``final_text``. This sink is a *fan-out* sink — it fires as the event is emitted,
            # before the log is on disk — so no hydration seam can reach it, and model-authored text
            # now leaves ``run.finished`` as a digest. Carrying ``data.get("final_text", "")`` would
            # therefore write ``""`` on every model-answered run: no schema failure (``STATUS_SCHEMA``
            # never declares the field and allows additional properties), no error, just a silently
            # empty answer. Removing it is the honest shape, and it is also a leak fix —
            # ``RunProjectionService.status()`` returns this file wholesale to any run-token bearer.
            # Verified to have no reader in ``src/``, ``tests/``, ``docs/``, ``studio-ui/`` or
            # ``scripts/``, and subagent runs never had one (``status_file=False``).
            self.state.update(
                {
                    "state": session_state_value(state),
                    "terminal": True,
                    "finished_at": event.timestamp,
                    "error": data.get("error", ""),
                    "error_code": data.get("error_code", ""),
                }
            )
            # The terminal heal: ASSIGNED error/error_code above, and a non-failed terminal
            # clears the parked classification — a completed run must not keep a dead turn's
            # ``retryable``. A failed terminal keeps it: ``run.finished{status:"failed"}``
            # follows the ``run.failed`` that owns the terminal classification, and popping
            # here would undo that record one event later.
            if state is not SessionState.FAILED:
                for key in _FAILURE_CLASSIFICATION_KEYS:
                    self.state.pop(key, None)
        elif event.type == "run.failed":
            self.state.update(
                {
                    "state": session_state_value(SessionState.FAILED),
                    "terminal": True,
                    "error": data.get("error", ""),
                    "error_code": data.get("error_code", ""),
                    "error_type": data.get("type", ""),
                    # Exactly what the event carries — the terminal twin of ``turn.failed``
                    # keeps the classification and deliberately drops ``provider_retried``
                    # (a per-call fact; publishing it here would publish one call's number
                    # as the run's), so this branch drops it too rather than inventing it.
                    "provider_error_code": data.get("provider_error_code", ""),
                    "http_status": data.get("http_status"),
                    "retryable": data.get("retryable", False),
                    "config_recoverable": data.get("config_recoverable", False),
                }
            )
            self.state.pop("provider_retried", None)
        elif event.type == "run.waiting":
            self.state["state"] = session_state_value(SessionState.AWAITING_TASKS)
            self.state["terminal"] = False
            self.state["waiting_for_background_jobs"] = True
            self.state["waiting_jobs"] = data.get("jobs", [])
        elif event.type == "run.resumed":
            self.state["state"] = session_state_value(SessionState.RUNNING)
            self.state["terminal"] = False
            self.state["waiting_for_background_jobs"] = False
            self.state["resumed_jobs"] = data.get("job_ids", [])
        elif event.type == "run.awaiting_input":
            self.state["state"] = session_state_value(SessionState.AWAITING_INPUT)
            self.state["terminal"] = False
            self.state["awaiting_input"] = {
                "reason": data.get("reason"),
                "task_ids": data.get("task_ids", []),
                "prompt": data.get("prompt"),
            }
        elif event.type == "agent.config.updated":
            self.state["agent_config"] = {
                "definition_id": data.get("definition_id"),
                "config_version": data.get("config_version"),
                "config_hash": data.get("config_hash"),
            }
        elif event.type == "model.turn.started":
            self.state["current_turn_id"] = event.turn_id
            self.state["current_step"] = data.get("step")
            # Every park, not one. Clearing only ``AWAITING_INPUT`` here left a run that had
            # parked on background jobs reading as parked while the turn that unparked it was
            # already running — ``run.resumed`` is emitted for the job case but not for every
            # path out of a task wait — and a resumed pause read as paused for the whole
            # resumed turn. The offline twin (``core/projections.py``) clears the same set.
            if self.state.get("state") in _PARKED_STATE_VALUES:
                self.state["state"] = session_state_value(SessionState.RUNNING)
                self.state["terminal"] = False
                self.state["waiting_for_background_jobs"] = False
                self.state.pop("awaiting_input", None)
            # The unpark clear, unconditional on purpose: a retried turn never passes through
            # a parked state (the driver re-pumps straight from ``turn_failed``), so guarding
            # this behind the parked-state check kept a dead turn's error beside
            # state="running". While PARKED the failure remains — that is the point of
            # carrying it — the model turn *starting* is what supersedes it.
            self.state.pop("error", None)
            self.state.pop("error_code", None)
            for key in _FAILURE_CLASSIFICATION_KEYS:
                self.state.pop(key, None)
        elif event.type == "turn.failed":
            # A recoverable model-turn park. The event carries the whole classification and
            # this branch used to copy two of its seven facts — ``config_recoverable`` alone
            # cannot separate an ``insufficient_quota`` (fix config) from a ``rate_limit``
            # (wait), so the full set rides. ``provider_usage`` stays out: metering, not
            # classification. State is untouched: the park that follows names it.
            self.state["error"] = data.get("error", "")
            self.state["error_code"] = data.get("error_code", "")
            self.state["provider_error_code"] = data.get("provider_error_code", "")
            self.state["http_status"] = data.get("http_status")
            self.state["retryable"] = data.get("retryable", False)
            self.state["config_recoverable"] = data.get("config_recoverable", False)
            self.state["provider_retried"] = data.get("provider_retried", False)
        elif event.type == "session.state.changed":
            # The pause park's session-lane projection. Without this branch a paused run read
            # as "running" on the one artifact an operator polls. Only the pause emits this
            # event today, and this reader binds only the state it can prove: a future
            # emitter with a new state value must decide its own projection here.
            if data.get("state") == session_state_value(SessionState.PAUSED):
                self.state["state"] = session_state_value(SessionState.PAUSED)
                self.state["terminal"] = False
        elif event.type == "tool.call.started":
            self.state["current_tool"] = data.get("tool")
            self.state["current_tool_call_id"] = data.get("call_id")
        elif event.type in {"tool.call.finished", "tool.call.failed"}:
            self.state.pop("current_tool", None)
            self.state.pop("current_tool_call_id", None)
        elif event.type == "plan.updated":
            self.state["plan"] = data.get("items", [])
            # Carry the drop count across with the list it belongs to. The event moved it out of
            # the array so a typed renderer would stop drawing a blank row for it, and copying only
            # ``items`` here would have turned that into a *silent* cap on this surface: a reader
            # would see 20 steps with nothing saying there had been 25. Reassigned rather than
            # merged, because a later shorter plan has to clear a stale count.
            truncated = data.get("truncated_items")
            if isinstance(truncated, int) and truncated > 0:
                self.state["plan_truncated_items"] = truncated
            else:
                self.state.pop("plan_truncated_items", None)
        elif event.type == "metrics.updated":
            self.state["metrics"] = data
        elif event.type == "workspace.proposal.updated":
            self.state["proposal"] = data
        elif event.type.startswith("job."):
            jobs = self.state.setdefault("jobs", {})
            if isinstance(jobs, dict):
                job_id = data.get("job_id")
                if isinstance(job_id, str) and job_id:
                    jobs[job_id] = data

        self.state.update(
            {
                "last_event_seq": event.seq,
                "last_event_type": event.type,
                "updated_at": event.timestamp,
            }
        )
        # status.json is a best-effort observability projection: a transient write failure must
        # never fail the run. write_json_atomic retries the Windows replace-vs-concurrent-reader
        # race; if it still can't be written, skip — a later event rewrites the full state.
        try:
            write_json_atomic(self.path, self.state)
        except OSError:
            _LOGGER.debug("status.json write skipped (transient fs error)", exc_info=True)

    def close(self) -> None:
        return None


@dataclass
class MemoryEventSink:
    events: list[AgentEvent] = field(default_factory=list)

    def emit(self, event: AgentEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        return None


@dataclass
class AgentRecorder:
    run_root: Path
    run_id: str
    extra_event_sinks: tuple[EventSink, ...] = ()
    status_file: bool = True
    # Durable restore: reopen an existing run dir instead of requiring a fresh one.
    # Transcript/events sinks already append, so reopening just tolerates the existing
    # artifacts dir rather than failing the run-id-collision guard.
    reopen: bool = False
    run_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    event_bus: EventBus = field(init=False)
    _transcript_file: TextIO = field(init=False, repr=False)
    started_at: float = field(default_factory=time.time)
    artifacts: list[AgentArtifact] = field(default_factory=list)
    # Private authored-content sidecar. Opt-in keeps the v0.20 run-dir shape and embedders'
    # retention assumptions unchanged unless the caller explicitly enables the new channel.
    model_content_file: bool = False
    # Private model-call ledger, opt-in for the same reason and appended after it so no positional
    # caller of this constructor is rebound.
    model_calls_file: bool = False
    # The routing root this run belongs to, or empty when the recorder stands alone. Every ledger
    # line carries it: a subagent records into its own run directory, so the root is the only
    # thing that makes a run tree joinable without walking the parent's events first.
    root_run_id: str = ""
    # Digests already written by ``settled_text``. Per-recorder, so a resumed run may re-append a
    # record whose digest is already in the file; that duplicates identical content, which the
    # content-addressed join resolves the same either way.
    _settled_text_digests: set[str] = field(default_factory=set, init=False, repr=False)
    _model_content_store: ModelContentStore | None = field(default=None, init=False, repr=False)
    _model_content_store_failed: bool = field(default=False, init=False, repr=False)
    _model_content_store_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )
    _model_calls_handle: TextIO | None = field(default=None, init=False, repr=False)
    _model_calls_failed: bool = field(default=False, init=False, repr=False)
    _model_calls_index: int = field(default=0, init=False, repr=False)
    _model_calls_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.run_dir = self.run_root / self.run_id
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=self.reopen)
        events_path = self.run_dir / "events.jsonl"
        advertised_last_seq = _read_status_last_event_seq(self.run_dir)
        tail = repair_event_log_tail_for_append(
            events_path,
            advertised_last_seq=advertised_last_seq,
        )
        initial_seq = _verified_event_sequence_seed(events_path, tail)
        self._transcript_file = (self.run_dir / "transcript.jsonl").open("a", encoding="utf-8")
        self._terminate_torn_transcript_tail()
        sinks: list[EventSink] = [JsonlEventSink(events_path)]
        if self.status_file:
            sinks.append(StatusJsonSink(self.run_dir / "status.json"))
        sinks.extend(self.extra_event_sinks)
        self.event_bus = EventBus(self.run_id, tuple(sinks), _seq=initial_seq)

    def emit(
        self,
        event_type: str,
        *,
        data: dict[str, Any] | None = None,
        level: str = "info",
        turn_id: str | None = None,
        parent_id: str | None = None,
    ) -> AgentEvent:
        return self.event_bus.emit(
            event_type,  # type: ignore[arg-type]
            data=data,
            level=level,  # type: ignore[arg-type]
            turn_id=turn_id,
            parent_id=parent_id,
        )

    def _terminate_torn_transcript_tail(self) -> None:
        """Close off ``transcript.jsonl``'s torn last line. See :meth:`_terminate_torn_tail`."""

        self._terminate_torn_tail(self.run_dir / "transcript.jsonl", self._transcript_file)

    @staticmethod
    def _terminate_torn_tail(path: Path, handle: TextIO) -> None:
        """Close off a torn last line so the next append is not glued onto it.

        ``transcript.jsonl`` gets no ``repair_event_log_tail_for_append`` the way ``events.jsonl``
        does. Appending after a line that lacks its trailing newline concatenates the remnant and
        the new record into one unparseable line, losing **both** — so a crash costs the next run's
        first record as well as its own. On the recovery path that first record can be the
        settled-text one, which a committed ``run.finished`` then names: exactly the
        "event names text that is not on disk" failure the write-before-emit ordering exists to
        prevent, with no crash during the recovered run.

        Writing the missing newline costs one byte and confines a torn write to the record it tore.
        Best-effort by design: if the file cannot be inspected, appending as before is no worse.

        Taken as a parameter rather than hard-coded to the transcript because ``model_calls.jsonl``
        reopens the same way and has the same exposure: a durable run appends to the ledger it
        already has, so a torn tail from the activation that crashed would otherwise consume the
        recovered activation's first record.
        """
        try:
            size = path.stat().st_size
            if size == 0:
                return
            with path.open("rb") as reader:
                reader.seek(size - 1)
                if reader.read(1) == b"\n":
                    return
            handle.write("\n")
            handle.flush()
        except OSError:
            return

    def transcript(self, item: dict[str, Any]) -> None:
        _write_jsonl(self._transcript_file, item)

    def record_model_call(self, receipt: ModelCallReceipt) -> None:
        """Append one settled call to the private ledger, shielding the run from every failure.

        Handed to ``ModelCallRunner.receipt_sink``, so it runs for failed calls as well as
        successful ones — the reason that seam exists rather than the loop's return value, which a
        failure never reaches. A direct method rather than a ``ModelIOObserver``, for the reason
        ``open_model_stream`` is one: the recorder owns its private artifacts and does not pretend
        to implement the external observer API, whose ``CapturePolicy`` narrowing would strip the
        digests this ledger exists to carry.

        Three containment layers, each for a failure the run must survive. The record is built and
        encoded before the handle is touched, so a caller's hostile ``InvocationContext.attributes``
        costs its own line rather than the run — and does not consume an index, since the counter
        advances only for a line that reached the file. A write error disables the handle rather
        than retrying: a partial write may have torn the current line, and appending after it would
        glue the next record onto the remnant and lose both. And nothing here raises, whoever calls
        it: an answer the provider has already been paid for is not discarded because a disk filled
        up, and this method's guarantee must not depend on ``ModelCallRunner._record`` also having
        one.

        Reserving the index and writing the line happen under **one** acquisition. Split across
        two, concurrent callers read the same counter and write two records claiming one index,
        which silently defeats the only thing ``call_index`` is for — noticing that a best-effort
        append-only file dropped something — and defeats it in the direction that reads as
        "nothing was lost". That the loop happens to call this sequentially is a property of one
        caller; the recorder holds a lock because it is shared with tool and job threads.
        """

        if not self.model_calls_file or self._model_calls_failed:
            return
        try:
            with self._model_calls_lock:
                if self._model_calls_failed:
                    return
                record = model_call_record(
                    receipt,
                    run_id=self.run_id,
                    root_run_id=self.root_run_id or self.run_id,
                    call_index=self._model_calls_index,
                    recorded_at=utc_timestamp(),
                )
                # Normalized and encoded before a handle is touched, so an unencodable value is
                # refused rather than half-written.
                line = json.dumps(
                    normalize_json_ingress(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                line.encode("utf-8")
                if self._append_model_call(line):
                    self._model_calls_index += 1
        except Exception:  # noqa: BLE001 - one unrecordable call must not cost the others
            _LOGGER.debug("model call record could not be written", exc_info=True)

    def _append_model_call(self, line: str) -> bool:
        """Write one encoded line. Caller holds ``_model_calls_lock``; returns whether it landed."""

        handle = self._ensure_model_calls_handle_locked()
        if handle is None:
            return False
        try:
            handle.write(line + "\n")
            handle.flush()
        except (OSError, UnicodeError):
            self._model_calls_failed = True
            _LOGGER.debug("model call ledger append failed", exc_info=True)
            return False
        return True

    def _ensure_model_calls_handle_locked(self) -> TextIO | None:
        if self._model_calls_handle is not None:
            return self._model_calls_handle
        path = self.run_dir / MODEL_CALLS_FILENAME
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a", encoding="utf-8", newline="\n")
        except OSError:
            self._model_calls_failed = True
            _LOGGER.debug("model call ledger could not be opened", exc_info=True)
            return None
        # A durable run reopens the directory it already has and appends to this same file, so a
        # tail torn by the activation that crashed would consume this activation's first record.
        self._terminate_torn_tail(path, handle)
        self._model_calls_handle = handle
        return handle

    def open_model_stream(self, context: ModelStreamContext) -> ModelStreamWriter:
        """Open the opt-in private content writer without exposing failures to the run."""

        store = self._get_model_content_store()
        if store is None:
            return NOOP_MODEL_STREAM_WRITER
        return safe_open_model_stream(store, context)

    def _get_model_content_store(self) -> ModelContentStore | None:
        if not self.model_content_file or self._model_content_store_failed:
            return None
        if self._model_content_store is not None:
            return self._model_content_store
        with self._model_content_store_lock:
            if self._model_content_store is not None:
                return self._model_content_store
            if self._model_content_store_failed:
                return None
            try:
                self._model_content_store = ModelContentStore(
                    self.run_dir / MODEL_CONTENT_FILENAME,
                    run_id=self.run_id,
                )
            except Exception:  # noqa: BLE001 - private content persistence is best-effort
                self._model_content_store_failed = True
                _LOGGER.debug("model content store initialization failed", exc_info=True)
                return None
        return self._model_content_store

    def settled_text(self, text: str) -> str:
        """Record model-authored settled text and return its content digest.

        The digest is the join key: a settle event carries ``final_text_digest`` and a reader
        resolves the text by matching it here. Keying on *content* rather than on the emitting
        event's identity is what lets this be written **before** the emit — the event's id and seq
        do not exist until ``emit`` returns, and writing afterwards would open a window in which a
        committed event names text that is not yet on disk.

        Identical text yields one record. ``turn.settled`` and ``run.finished`` normally carry the
        same value, so a content-addressed key makes the second write redundant by construction
        rather than by a caller remembering not to repeat itself.

        This lives on the recorder rather than beside the emit sites because the recorder owns both
        private content artifacts. ``model-content.jsonl`` is the v0.20.1 primary join source;
        ``transcript.jsonl`` retains the same record during the compatibility window so older run
        directories and readers keep working. Their contracts are registered separately as
        ``monoid.model-content.v1`` and ``monoid.transcript.v1``.

        Durability is best-effort and deliberately so: ``_write_jsonl`` flushes but does not fsync,
        and tail repair only stops a torn line from consuming the *next* record; it does not recover
        the torn one. A crash can leave a committed event whose digest resolves to nothing, so
        **content-missing is a tolerated read outcome** — hydration fills absent fields and never
        fails a read.
        """
        # ``content_digest``, never a bare sha256 of the text: it hashes canonical JSON under a
        # shape key so a text field cannot collide with a structured value's serialization. Once a
        # record persists one it is frozen (see the function's own docstring).
        digest = content_digest(text)
        if digest not in self._settled_text_digests:
            _write_jsonl(
                self._transcript_file,
                {
                    "kind": "settled_text",
                    "final_text": text,
                    "final_text_digest": digest,
                    "final_text_len": content_length(text),
                },
            )
            # Marked written only once it is. Adding before the write meant a raising write (a
            # full disk mid-flush) recorded the digest as present with nothing on disk, so a later
            # call for the same text short-circuited and returned a digest resolving to nothing.
            self._settled_text_digests.add(digest)
        store = self._get_model_content_store()
        if store is not None:
            try:
                text_len = content_length(text)
                if text_len is not None:
                    store.settled_text(text, digest, text_len)
            except Exception:  # noqa: BLE001 - private sidecar failure must not change this method
                _LOGGER.debug("settled text sidecar write failed", exc_info=True)
        return digest

    def emit_artifact_bytes(
        self,
        *,
        workspace_path: str,
        content: bytes,
        kind: str,
        label: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentArtifact:
        artifact_id = f"artifact_{len(self.artifacts) + 1:04d}"
        target = self.artifacts_dir / artifact_id / Path(workspace_path).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        artifact = AgentArtifact(
            artifact_id=artifact_id,
            path=str(target.relative_to(self.run_dir).as_posix()),
            kind=kind,
            label=label,
            metadata=dict(metadata or {}),
        )
        self.artifacts.append(artifact)
        return artifact

    def write_diff(self, diff_text: str) -> Path:
        diff_path = self.run_dir / "diff.patch"
        diff_path.write_text(diff_text, encoding="utf-8")
        return diff_path

    def write_proposal_revision(self, workspace: Workspace) -> tuple[str, Path, dict[str, Any]]:
        """Persist one internally consistent diff and proposal snapshot revision."""
        with proposal_snapshot_lock(self.run_dir):
            diff_text = workspace.diff_patch()
            diff_path = self.write_diff(diff_text)
            proposal_payload = self.write_proposal_snapshot(workspace, diff_path)
        return diff_text, diff_path, proposal_payload

    def write_manifest(self, manifest: RunManifest) -> Path:
        manifest_path = self.run_dir / "manifest.json"
        write_json_atomic(manifest_path, manifest.to_json())
        return manifest_path

    def write_workspace_index(self, payload: dict[str, Any]) -> Path:
        path = self.run_dir / "workspace.index.json"
        write_json_atomic(path, payload)
        return path

    def write_workspace_base(self, payload: dict[str, Any]) -> Path:
        path = self.run_dir / "workspace.base.json"
        write_json_atomic(path, payload)
        return path

    def write_proposal_snapshot(self, workspace: Workspace, diff_path: Path) -> dict[str, Any]:
        proposal_path = self.run_dir / "proposal.json"
        files_dir = self.run_dir / "proposal" / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        files: list[dict[str, Any]] = []
        for entry in workspace.changed_entries():
            files.append(self._write_proposal_entry(entry, files_dir))
        diff_data = diff_path.read_bytes() if diff_path.exists() else b""
        diff_bytes = diff_path.stat().st_size if diff_path.exists() else 0
        payload: dict[str, Any] = {
            "schema_version": namespaced_id("proposal.v2"),
            "run_id": self.run_id,
            "updated_at": time.time(),
            "mode": workspace.mode,
            "diff_path": str(diff_path.relative_to(self.run_dir)),
            "diff_bytes": diff_bytes,
            "diff_sha256": sha256_bytes(diff_data),
            "changed_paths": [entry["path"] for entry in files],
            "files": files,
        }
        # updated_at is wall-clock metadata, not content; excluding it makes the
        # proposal_hash a stable content identifier so repeated settle checkpoints
        # with no workspace change produce the same hash.
        payload["proposal_hash"] = canonical_sha256(payload, drop=("proposal_hash", "updated_at"))
        write_json_atomic(proposal_path, payload)
        return payload

    def write_metrics(self, metrics: dict[str, Any]) -> Path:
        path = self.run_dir / "metrics.json"
        payload = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": time.time(),
            **metrics,
        }
        write_json_atomic(path, payload)
        return path

    def write_failure(self, payload: dict[str, Any]) -> Path:
        """Write ``run_dir/failure.json`` — the operator-facing failure bundle: what
        broke plus which checkpoint to restore from. The core surfaces this; recovery
        (if any) is the integrator's call (no auto-recovery in the core)."""
        path = self.run_dir / "failure.json"
        write_json_atomic(path, payload)
        return path

    def close(self) -> None:
        try:
            self.event_bus.close()
        finally:
            try:
                # Event-sink failure must not retain the private transcript handle. TextIO close is
                # idempotent, while EventBus separately guarantees each configured sink is called
                # once.
                self._transcript_file.close()
            finally:
                try:
                    store = self._model_content_store
                    if store is not None:
                        try:
                            store.close()
                        except Exception:  # noqa: BLE001 - private persistence is best-effort
                            _LOGGER.debug("model content store close failed", exc_info=True)
                finally:
                    # Nested rather than sequential: one private artifact failing to close must
                    # not retain the other's descriptor, the same rule the transcript already has
                    # against the event bus above.
                    with self._model_calls_lock:
                        handle, self._model_calls_handle = self._model_calls_handle, None
                    if handle is not None:
                        try:
                            handle.close()
                        except OSError:
                            _LOGGER.debug("model call ledger close failed", exc_info=True)

    def _write_proposal_entry(self, entry: ChangedEntry, files_dir: Path) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": entry.path,
            "kind": entry.kind,
            "size": entry.size,
            "sha256": entry.sha256,
            "base_sha256": entry.base_sha256,
            "proposed_sha256": entry.proposed_sha256,
            "change_kind": entry.change_kind,
        }
        if entry.content is None:
            return payload
        target = files_dir.joinpath(*entry.path.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(entry.content)
        payload["snapshot_path"] = str(target.relative_to(self.run_dir).as_posix())
        payload["snapshot_sha256"] = sha256_bytes(entry.content)
        return payload


def _write_jsonl(handle: TextIO, payload: dict[str, Any]) -> None:
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
    try:
        line.encode("utf-8")
    except UnicodeEncodeError:
        # Manually constructed events can bypass semantic ingress. Repair them to valid Unicode so
        # the persisted record remains encodable after it is read and projected onto a later wire.
        normalized = normalize_json_ingress(payload)
        line = json.dumps(normalized, ensure_ascii=False, sort_keys=True, allow_nan=False)
    handle.write(line + "\n")
    handle.flush()


def append_event_to_run(
    run_dir: Path,
    event_type: str,
    *,
    data: dict[str, Any] | None = None,
    level: str = "info",
) -> AgentEvent:
    """Append through the queued or terminal sequence owner selected by the caller."""
    events_path = run_dir / "events.jsonl"
    run_id = _run_id_from_run_dir(run_dir)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    advertised_last_seq = _read_status_last_event_seq(run_dir)
    tail = repair_event_log_tail_for_append(
        events_path,
        advertised_last_seq=advertised_last_seq,
    )
    seq = _verified_event_sequence_seed(events_path, tail) + 1
    event = make_agent_event(
        run_id=run_id,
        seq=seq,
        event_type=event_type,  # type: ignore[arg-type]
        data=data,
        level=level,  # type: ignore[arg-type]
    )
    with events_path.open("a", encoding="utf-8") as handle:
        _write_jsonl(handle, event.to_json())
    _update_status_last_event(run_dir, event)
    return event


def _run_id_from_run_dir(run_dir: Path) -> str:
    status_path = run_dir / "status.json"
    if status_path.exists():
        try:
            status = loads_json_ingress(status_path.read_text(encoding="utf-8"))
            if isinstance(status, dict) and isinstance(status.get("run_id"), str):
                return status["run_id"]
        except (OSError, ValueError):
            pass
    proposal_path = run_dir / "proposal.json"
    if proposal_path.exists():
        try:
            proposal = loads_json_ingress(proposal_path.read_text(encoding="utf-8"))
            if isinstance(proposal, dict) and isinstance(proposal.get("run_id"), str):
                return proposal["run_id"]
        except (OSError, ValueError):
            pass
    return run_dir.name


def _read_status_last_event_seq(run_dir: Path) -> int | None:
    try:
        raw_status = read_text_resilient(run_dir / "status.json")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as exc:
        raise EventLogCorruption("status event watermark cannot be verified") from exc
    try:
        status = loads_json_ingress(raw_status)
    except ValueError as exc:
        raise EventLogCorruption("status event watermark cannot be verified") from exc
    if not isinstance(status, dict):
        raise EventLogCorruption("status event watermark cannot be verified")
    if "last_event_seq" not in status:
        return None
    value = status["last_event_seq"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EventLogCorruption("status event watermark cannot be verified")
    return value


def _verified_event_sequence_seed(
    events_path: Path,
    tail: EventLogTail,
) -> int:
    if not tail.exists or tail.file_size == 0:
        return 0
    validated_last_seq = validate_committed_event_sequence(events_path)
    if validated_last_seq != tail.last_seq:
        raise EventLogCorruption("event log tail changed during sequence validation")
    return validated_last_seq


def _update_status_last_event(run_dir: Path, event: AgentEvent) -> None:
    status_path = run_dir / "status.json"
    if not status_path.exists():
        return
    try:
        status = loads_json_ingress(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if not isinstance(status, dict):
        return
    status["last_event_seq"] = event.seq
    status["last_event_type"] = event.type
    status["updated_at"] = event.timestamp
    if event.type.startswith("proposal."):
        status["proposal_event"] = event.data
    write_json_atomic(status_path, status)
