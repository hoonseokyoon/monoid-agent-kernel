from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
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
from monoid_agent_kernel.core._verified_file import (
    open_verified_append_text,
    verified_directory_is_safe,
    write_verified_bytes_once,
)
from monoid_agent_kernel.core.events import AgentEvent, EventBus, EventSink, make_agent_event
from monoid_agent_kernel.core.json_ingress import (
    json_nesting_within_limit,
    loads_json_ingress,
    normalize_json_ingress,
)
from monoid_agent_kernel.core.interruption import parse_interruption_cause
from monoid_agent_kernel.core.lifecycle import (
    SessionState,
    session_state_from_run_status,
    session_state_value,
)
from monoid_agent_kernel.core.manifest import RunManifest
from monoid_agent_kernel.core.model_calls import MODEL_CALLS_FILENAME, model_call_record
from monoid_agent_kernel.core.model_payloads import (
    MODEL_PAYLOADS_DIRNAME,
    MODEL_PAYLOADS_FILENAME,
    PAYLOAD_OFFLOAD_THRESHOLD_BYTES,
    SplitRequestPayload,
    chunk_marker,
    is_chunk_sha256,
    chunk_record,
    model_request_record,
    model_response_record,
    response_record_body,
    split_request_payload,
)
from monoid_agent_kernel.core.model_content import MODEL_CONTENT_FILENAME, ModelContentStore
from monoid_agent_kernel.model_call import SettledModelCall
from monoid_agent_kernel.core.model_io import content_digest, content_length
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


@dataclass(frozen=True)
class _EncodedResponseBody:
    """``response_record_body`` plus the reader's nesting verdict on an offloaded body.

    Both halves are functions of the turn alone, which is why they belong together and why
    they belong outside ``_model_calls_lock``. ``readable`` is False only for a body that
    would be offloaded AND that the replay reader's lexical bound refuses.
    """

    value: Any
    encoded: bytes | None
    unrecorded_reason: str
    readable: bool


def encoded_response_body(turn: Any) -> _EncodedResponseBody:
    """The response record's body and whether the reader could parse it back. Pure."""

    recorded = response_record_body(turn)
    readable = True
    if (
        recorded.value is not None
        and recorded.encoded is not None
        and len(recorded.encoded) > PAYLOAD_OFFLOAD_THRESHOLD_BYTES
    ):
        readable = json_nesting_within_limit(recorded.encoded.decode("utf-8"))
    return _EncodedResponseBody(
        recorded.value, recorded.encoded, recorded.unrecorded_reason, readable
    )


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
            cause = parse_interruption_cause(data.get("interruption_cause"))
            if cause is not None:
                self.state["interruption_cause"] = cause.value
            else:
                self.state.pop("interruption_cause", None)
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
            self.state.pop("interruption_cause", None)
        elif event.type == "turn.paused":
            # The current turn park owns this projection. A cause-less pause supersedes any
            # interruption cause carried by the preceding park.
            self.state.pop("interruption_cause", None)
        elif event.type == "turn.settled":
            cause = parse_interruption_cause(data.get("interruption_cause"))
            if cause is not None:
                self.state["interruption_cause"] = cause.value
            else:
                self.state.pop("interruption_cause", None)
        elif event.type == "turn.interrupted":
            cause = parse_interruption_cause(data.get("interruption_cause"))
            if cause is not None:
                self.state["interruption_cause"] = cause.value
            else:
                # Legacy producers emit only the reason. The newest park is still authoritative,
                # so absence or malformed input clears an older typed cause instead of inheriting it.
                self.state.pop("interruption_cause", None)
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
            self.state.pop("interruption_cause", None)
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
    # Private replay corpus (request preimages + settled response bodies), opt-in like its two
    # sidecar siblings and appended after them so no positional caller is rebound.
    model_payload_file: bool = False
    # Optional host writer-authority fence. EventBus applies it around every sink callback so a
    # slow extension cannot let later projections publish after this activation loses its lease.
    check_authority: Callable[[], None] | None = None
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
    # Distinct from ``_failed``: the handle is opened lazily, so without a closed state a record
    # arriving after ``close`` silently reopens the file the recorder just released.
    _model_calls_closed: bool = field(default=False, init=False, repr=False)
    _model_calls_index: int = field(default=0, init=False, repr=False)
    # One lock for every per-call sidecar: the ledger line and the payload records of one call
    # must land under one acquisition or their shared ``call_index`` is only an aspiration.
    _model_calls_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )
    _model_payloads_handle: TextIO | None = field(default=None, init=False, repr=False)
    _model_payloads_failed: bool = field(default=False, init=False, repr=False)
    _model_payloads_closed: bool = field(default=False, init=False, repr=False)
    # Per-process dedup state. Empty again after a durable reopen, which makes duplicate chunk
    # and request records legal-by-construction: content-addressed, so a duplicate is identical.
    _payload_chunk_shas: set[str] = field(default_factory=set, init=False, repr=False)
    _payload_request_digests: set[str] = field(default_factory=set, init=False, repr=False)

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
        self.event_bus = EventBus(
            self.run_id,
            tuple(sinks),
            _seq=initial_seq,
            check_authority=self.check_authority,
        )

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

        Path-and-handle rather than handle alone because the transcript's handle is opened in text
        append mode and cannot be read through. That reopen is the weaker form of this check and the
        reason ``model_calls.jsonl`` no longer uses it: ``open_verified_append_text`` isolates the
        same torn tail on the descriptor it has already verified, so there is no second ``open`` of
        a name that could by then be something else. The transcript is a v0.20 artifact opened
        eagerly in ``__post_init__``, where refusing a planted path would fail the run outright
        rather than one sidecar; moving it to the verified opener is a separate change with its own
        failure policy to decide.
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

    def record_settled_call(self, call: SettledModelCall) -> None:
        """Record one settled call into the private sidecars, shielding the run from every failure.

        Handed to ``ModelCallRunner.settled_sink``, so it runs for failed calls as well as
        successful ones — the reason that seam exists rather than the loop's return value, which a
        failure never reaches. One entry point for however many files the recorder keeps per call
        (today the ledger; the payload corpus joins it), because one delivery under one lock is
        what keeps their per-call indices in agreement — two sink methods could only agree by
        cooperation across two acquisitions, which is the index race this recorder already fixed
        once internally. A direct method rather than a ``ModelIOObserver``, for the reason
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

        ledger_wanted = (
            self.model_calls_file and not self._model_calls_failed and not self._model_calls_closed
        )
        payloads_wanted = (
            self.model_payload_file
            and not self._model_payloads_failed
            and not self._model_payloads_closed
        )
        if not ledger_wanted and not payloads_wanted:
            return
        try:
            # Split before the lock, not under it. This is the only expensive thing either arm
            # does -- a SHA pass, a decode, a per-value re-encode, and a full re-encode-and-compare
            # of the whole preimage, measured at ~170 ms at the 8 MB ceiling -- and it is pure. The
            # lock is shared with the ledger arm and with tool and job threads, so holding it
            # across that work makes the corpus's cost the ledger's latency. The seen-set read is a
            # fast path whose only stale outcome is one redundant split; the write side re-checks.
            split = None
            try:
                # The *whole* block, not just the call. Containment has to cover everything that
                # moved out of ``_record_payloads_locked``'s handler, and the guard is part of
                # that: a receipt whose ``request_digest`` is unhashable raises on the seen-set
                # test, one statement before the call. Either raise, uncontained, would skip the
                # ledger arm and the counter too -- the call would vanish from both files with no
                # hole to show for it. Both are reachable through the public sink, whose
                # ``SettledModelCall`` is untyped at runtime.
                if (
                    payloads_wanted
                    and call.receipt.request_digest
                    and call.request_preimage is not None
                    and call.receipt.request_digest not in self._payload_request_digests
                ):
                    split = split_request_payload(
                        call.request_preimage, call.receipt.request_digest
                    )
            except Exception:  # noqa: BLE001 - one unrecordable corpus entry costs only the corpus
                _LOGGER.debug("model payload split failed", exc_info=True)
                split = None
            # The response body joins the split out here for the same reason, and it was not
            # always so: the offloaded-depth gate arrived inside the lock, where its per-character
            # scan measured 382.9 ms on a 7.9 MB body against 4.0 ms of sha256 -- a 12x critical
            # section, all of it pure work the paragraph above already ruled out holding. Both the
            # encode and the verdict are functions of the turn alone, so a concurrent recorder now
            # waits on the write, not on the scan.
            body = None
            if payloads_wanted and call.turn is not None:
                try:
                    body = encoded_response_body(call.turn)
                except Exception:  # noqa: BLE001 - as above: the corpus entry, never the ledger
                    _LOGGER.debug("model payload response encode failed", exc_info=True)
                    body = None
            with self._model_calls_lock:
                index = self._model_calls_index
                # One clock read for however many files record this call. ``call_index`` restarts
                # at zero when a durable run reopens its directory, so it cannot join the ledger to
                # the corpus across activations on its own; two independent reads of the clock
                # cannot either, and can even match a *different* call's line. The pair is a join
                # only if the pair comes from one reading.
                recorded_at = utc_timestamp()
                advanced = False
                if (
                    self.model_calls_file
                    and not self._model_calls_failed
                    and not self._model_calls_closed
                ):
                    advanced = self._record_ledger_locked(call, index, recorded_at) or advanced
                if (
                    self.model_payload_file
                    and not self._model_payloads_failed
                    and not self._model_payloads_closed
                ):
                    advanced = (
                        self._record_payloads_locked(call, index, recorded_at, split, body)
                        or advanced
                    )
                # One counter for however many files recorded this call, advanced when any of
                # them did: the index identifies the CALL, so per-file write failures surface as
                # holes in that file rather than as two files disagreeing about which call is
                # which. With a single sidecar enabled this is exactly the W6-1 rule (advance
                # only for a line that reached the file).
                if advanced:
                    self._model_calls_index += 1
        except Exception:  # noqa: BLE001 - one unrecordable call must not cost the others
            _LOGGER.debug("model call record could not be written", exc_info=True)

    def _record_ledger_locked(self, call: SettledModelCall, index: int, recorded_at: str) -> bool:
        """Build and append one ledger line. Caller holds ``_model_calls_lock``."""

        try:
            record = model_call_record(
                call.receipt,
                run_id=self.run_id,
                root_run_id=self.root_run_id or self.run_id,
                call_index=index,
                recorded_at=recorded_at,
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
        except Exception:  # noqa: BLE001 - a hostile context costs its own line, not the run
            _LOGGER.debug("model call record could not be encoded", exc_info=True)
            return False
        return self._append_model_call(line)

    def _record_payloads_locked(
        self,
        call: SettledModelCall,
        index: int,
        recorded_at: str,
        split: SplitRequestPayload | None,
        body: _EncodedResponseBody | None,
    ) -> bool:
        """Record one call's corpus entries. Caller holds ``_model_calls_lock``.

        Request side first, response side second, chunks before the record that references them:
        a failure part-way leaves unreferenced chunks -- which the validator deliberately ignores
        and ``monoid gc`` reclaims -- never a record whose references dangle.

        Returns whether this call has been *accounted for*, which is what the shared index counts.
        A landed response line is one way; a failed call is the other, and it is not a write that
        did not land -- there is no turn to record, the ledger line carries the taxonomy, and the
        accounting is complete. Getting that wrong made ``call_index`` mean something different
        depending on whether the ledger was switched on, because the ledger arm writes a line for
        every call and so advanced the counter on the corpus's behalf.

        The split is verified before anything is written (``split_request_payload`` reassembles
        and compares), so the corpus cannot contain a request record that fails its own digest --
        the validator re-proves per record what the writer proved per write. It arrives already
        computed because it is expensive and pure; see the caller.
        """

        receipt = call.receipt
        # The join key as the corpus may state it. A digest that is not a digest joins nothing, and
        # the schema says so (``^(|[0-9a-f]{64})$``); writing one through would put a line in this
        # file that its own validator rejects, which is the one thing ``_encoded_payload_line``'s
        # doctrine forbids. Empty is the honest value and already means "look at the ledger line's
        # ``digest_status``" -- the request side cannot reach this, because a split only exists for
        # a preimage that hashed to its digest.
        joinable_digest = receipt.request_digest if is_chunk_sha256(receipt.request_digest) else ""
        wrote_response = False
        envelope = {
            "run_id": self.run_id,
            "root_run_id": self.root_run_id or self.run_id,
            "recorded_at": recorded_at,
        }
        try:
            # ``digest_generation`` names the rules the key was taken under, and the schema needs
            # a non-empty one. A recipe whose generation is unknown cannot be interpreted by any
            # reader, and inventing a default would file it under rules it may not have followed --
            # the ``_digest`` doctrine again. Refusing costs the request record; the response
            # record and the ledger line still describe the call.
            if not receipt.digest_generation:
                split = None
            if split is not None and receipt.request_digest not in self._payload_request_digests:
                landed = True
                for sha, chunk in split.chunks.items():
                    if sha in self._payload_chunk_shas:
                        continue
                    if not self._store_payload_chunk_locked(sha, chunk, envelope):
                        landed = False
                        break
                    self._payload_chunk_shas.add(sha)
                if landed:
                    request_line = self._encoded_payload_line(
                        model_request_record(
                            split.payload,
                            refs=split.refs,
                            request_digest=receipt.request_digest,
                            digest_generation=receipt.digest_generation,
                            **envelope,
                        )
                    )
                    if request_line is not None and self._append_model_payload(request_line):
                        self._payload_request_digests.add(receipt.request_digest)
            # Re-checked, because the request side can have gone terminal since the gate above:
            # a response record written now would carry a ``request_digest`` naming a request
            # record that can never exist, and would append to a handle the opener just refused.
            if call.turn is not None and not self._model_payloads_failed:
                # Recomputed only if the hoist did not manage it, so a raise still lands exactly
                # where it used to -- after the ledger arm, not instead of it.
                recorded = body if body is not None else encoded_response_body(call.turn)
                reason = recorded.unrecorded_reason
                response: dict | None
                if recorded.value is not None and recorded.encoded is not None:
                    if len(recorded.encoded) > PAYLOAD_OFFLOAD_THRESHOLD_BYTES:
                        if not recorded.readable:
                            # Asked here because the line gate cannot ask it: an offloaded body
                            # is never in the line, so its brackets sit in a chunk file and the
                            # encoder below sees a shallow reference. Storing it anyway wrote a
                            # record claiming ``unrecorded_reason: ""`` -- the writer saying it
                            # recorded this answer -- that the reader then refused as
                            # ``not_recorded``, the kernel declining to read a body it had just
                            # written. The bound cannot move to the reader instead: without it
                            # the decoder's own stack limit decides, so the same corpus replays
                            # or does not depending on how deep the call stack already is.
                            response = None
                            reason = "unencodable"
                        else:
                            sha = sha256_bytes(recorded.encoded)
                            if self._store_payload_chunk_locked(sha, recorded.encoded, envelope):
                                self._payload_chunk_shas.add(sha)
                                response = chunk_marker(sha)
                            else:
                                return wrote_response
                    else:
                        response = recorded.value
                else:
                    response = None

                def line_for(body: dict | None, reason: str) -> str | None:
                    return self._encoded_payload_line(
                        model_response_record(
                            body,
                            call_index=index,
                            request_digest=joinable_digest,
                            unrecorded_reason=reason,
                            **envelope,
                        )
                    )

                response_line = line_for(response, reason)
                if response_line is None:
                    # The body itself is what the line encoder refused -- today only by being
                    # nested deeper than the corpus reader parses. Dropping the record would say
                    # the call never happened; the doctrine for a body this artifact cannot carry
                    # is a record with a typed absence, and it is the same answer whether the body
                    # was inline or offloaded. Both halves reach it now: the inline one here, the
                    # offloaded one through the depth check above, which is where an offloaded
                    # body's brackets are still visible.
                    response_line = line_for(None, "unencodable")
                if response_line is not None and self._append_model_payload(response_line):
                    wrote_response = True
        except Exception:  # noqa: BLE001 - one unrecordable call must not cost the others
            _LOGGER.debug("model payload records could not be written", exc_info=True)
        return wrote_response or call.turn is None

    def _encoded_payload_line(self, record: dict[str, Any]) -> str | None:
        """Encode one corpus record, refusing rather than half-writing.

        No ``normalize_json_ingress`` here, deliberately unlike the ledger line: a request recipe
        is decoded canonical-encoder output and a response body is a normalized turn's fields, so
        there is nothing left to normalize -- and a normalizer that ever *did* change a recipe
        value would break the byte-identity the corpus exists to keep.

        The corpus does not contain what its own validator cannot read, and the thing the validator
        reads is **this line** -- not the preimage, and not the response body. The bound belongs
        here for the same reason: it is one function all three record kinds pass through, and the
        two writers that tried to own it separately would each have had to guess the envelope's
        depth. It costs a scan of a few kilobytes; scanning the preimage instead cost 80% of the
        settle path and refused records that were perfectly readable, because a value deep enough
        to matter is also large enough to be lifted into a chunk, where its brackets live inside a
        JSON string and count for nothing.
        """

        try:
            line = json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
            line.encode("utf-8")
        except Exception:  # noqa: BLE001 - refusal is the containment
            _LOGGER.debug("model payload record could not be encoded", exc_info=True)
            return None
        if not json_nesting_within_limit(line):
            _LOGGER.debug("model payload record is deeper than the corpus reader accepts")
            return None
        return line

    def _store_payload_chunk_locked(self, sha: str, chunk: bytes, envelope: dict[str, str]) -> bool:
        """Store one chunk: inline record up to the offload threshold, directory file past it.

        A directory-write failure is terminal for the corpus arm (``_model_payloads_failed``),
        the same treatment a torn append gets: the run keeps its answer, the corpus stops rather
        than accumulating records whose references dangle.
        """

        if len(chunk) > PAYLOAD_OFFLOAD_THRESHOLD_BYTES:
            target = self.run_dir / MODEL_PAYLOADS_DIRNAME / sha
            if write_verified_bytes_once(target, chunk):
                return True
            # A chunk lands before the record that references it, so this is the one disable that
            # can leave the corpus file never created at all -- nothing on disk to notice, which is
            # why it needs the loudest of them, not the quietest.
            self._lose_model_payloads_locked("a chunk could not be stored")
            return False
        line = self._encoded_payload_line(chunk_record(chunk, **envelope))
        if line is None:
            return False
        return self._append_model_payload(line)

    def _lose_artifact(self, artifact: str, reason: str) -> None:
        """Say, once, that an artifact this run was configured to produce stops here.

        Every writer below keeps a flag meaning "this artifact records nothing for the rest of the
        activation". The two sidecar arms reach it through a refused open, a refused chunk and a
        torn append; the content flag has one door, a store that would not construct, and the
        store's own further transitions live on its ``_disabled`` and its own logger. Setting
        one *is* the event an operator needs told -- the run still exits zero, and `monoid
        validate` still calls the directory clean, because each artifact is optional -- so the
        assignment and the announcement are one call, and the three doors below are the only places
        the raw flag is written. A new failure path cannot reach the terminal state without going
        through a door that speaks; a census test refuses the file if one ever does.

        ``WARNING`` because the last-resort handler drops anything below it, so an operator who
        configured no logging -- which is what `monoid run` is -- would otherwise be told nothing at
        all. The message names the artifact and nothing else: identifiers travel in ``extra`` where
        an aggregator can key on them, so the default stderr rendering stays free of the run id and
        the paths around it, and no traceback is rendered here for the same reason.
        """

        _LOGGER.warning(
            "%s: %s; this activation records no more of it",
            artifact,
            reason,
            extra={"monoid_run_id": self.run_id, "monoid_artifact": artifact},
        )

    def _lose_model_calls_locked(self, reason: str) -> None:
        """Caller holds ``_model_calls_lock``."""

        if self._model_calls_failed:
            return
        self._model_calls_failed = True
        self._lose_artifact(MODEL_CALLS_FILENAME, reason)

    def _lose_model_payloads_locked(self, reason: str) -> None:
        """Caller holds ``_model_calls_lock`` -- both sidecar arms share it."""

        if self._model_payloads_failed:
            return
        self._model_payloads_failed = True
        self._lose_artifact(MODEL_PAYLOADS_FILENAME, reason)

    def _lose_model_content_locked(self, reason: str) -> None:
        """Caller holds ``_model_content_store_lock`` -- a DIFFERENT lock from the two above, which
        is why each door names its own rather than inheriting a file-wide convention."""

        if self._model_content_store_failed:
            return
        self._model_content_store_failed = True
        self._lose_artifact(MODEL_CONTENT_FILENAME, reason)

    def _append_model_call(self, line: str) -> bool:
        """Write one encoded line. Caller holds ``_model_calls_lock``; returns whether it landed."""

        handle = self._ensure_model_calls_handle_locked()
        if handle is None:
            return False
        try:
            handle.write(line + "\n")
            handle.flush()
        except (OSError, UnicodeError):
            # The traceback stays, at ``debug``, beside the announcement: the door says *that* the
            # ledger stopped, and only errno says whether this was a full disk, a dead handle or a
            # network mount going away. Dropping it here while the constructor site kept its own
            # was the level promotion eating the detail it was not about.
            _LOGGER.debug("model call ledger append failed", exc_info=True)
            self._lose_model_calls_locked("an append failed and may have torn its line")
            return False
        return True

    def _append_model_payload(self, line: str) -> bool:
        """Write one encoded corpus line. Caller holds ``_model_calls_lock``."""

        handle = self._ensure_model_payloads_handle_locked()
        if handle is None:
            return False
        try:
            handle.write(line + "\n")
            handle.flush()
        except (OSError, UnicodeError):
            _LOGGER.debug("model payload append failed", exc_info=True)
            self._lose_model_payloads_locked("an append failed and may have torn its line")
            return False
        return True

    def _ensure_model_payloads_handle_locked(self) -> TextIO | None:
        """The corpus twin of ``_ensure_model_calls_handle_locked`` -- same verified open, same
        terminal refusal, same reason (see that docstring; the rule was extracted to
        ``core/_verified_file.py`` precisely so the second sidecar could not skip it). Also sweeps
        orphaned ``*.tmp`` files out of the chunk directory once per activation -- **this process's**
        only, because the temporary name carries its writer's pid and a durable run reclaimed while
        its previous owner is still alive would otherwise have its in-flight chunk deleted
        mid-write, failing that writer's ``os.replace`` and killing its corpus arm over something
        it did not do. That scope is deliberately narrower than "crash litter": a process that dies
        mid-write never reopens under the same pid, so its temporary outlives every sweep. What
        this collects is an in-process reopen's own leftovers. Cross-process orphans are unreferenced
        files in a content-addressed directory, which the validator ignores by design and which
        ``monoid gc`` collects -- offline, age-gated, on the operator's word that no writer is
        live -- because identity cannot tell a dead writer from a live one, and guessing wrong in
        the other direction is the failure above."""

        if self._model_payloads_failed:
            # Terminal, not per-line. Two appends can reach this within one call, and a refusal is
            # a property of the path: re-running the verified open is a second chance to be handed
            # a different file, which is the substitution it exists to refuse.
            return None
        if self._model_payloads_handle is not None:
            return self._model_payloads_handle
        handle = open_verified_append_text(self.run_dir / MODEL_PAYLOADS_FILENAME)
        if handle is None:
            self._lose_model_payloads_locked("it could not be safely opened")
            return None
        self._model_payloads_handle = handle
        chunk_dir = self.run_dir / MODEL_PAYLOADS_DIRNAME
        if not verified_directory_is_safe(chunk_dir):
            # The sweep unlinks, and it runs before the first write -- so it reaches the chunk
            # directory ahead of the only other gate on it. ``Path.glob`` follows a redirection
            # planted there, which would make this a delete primitive in a directory of somebody
            # else's choosing.
            return handle
        try:
            for orphan in chunk_dir.glob(f"*.{os.getpid()}.*.tmp"):
                try:
                    orphan.unlink()
                except OSError:
                    continue
        except OSError:
            pass
        return handle

    def _ensure_model_calls_handle_locked(self) -> TextIO | None:
        """Open the ledger only if the path names a file this process is entitled to append to.

        ``open(path, "a")`` is the wrong primitive here and was the defect. The handle is acquired
        lazily, so between the run directory existing and the first receipt arriving there is a
        window in which ``model_calls.jsonl`` can be replaced by a link -- and a reopened durable
        run stretches that window across the whole gap since the last activation. A symlink is
        followed; a hard link is a second name for an inode that can live anywhere on the volume.
        Either one makes the agent append its own ledger, under its own credentials, to a file
        somebody else chose.

        The same function the ``model-content.jsonl`` writer uses, deliberately: two append-only
        sidecars in one directory with one exposure is exactly the shape where a rule gets proven on
        one site and forgotten on its twin. It also folds in the torn-tail isolation a durable
        reopen needs, on the descriptor it just verified rather than by reopening the pathname.

        A refusal disables the ledger rather than being retried, the same terminal treatment a torn
        write gets: the reason a verified open said no is a property of the path, so the next
        receipt would only re-run the identical refusal.
        """

        if self._model_calls_failed:
            # The corpus twin has always self-guarded, and the "once per activation" property was
            # resting on the two callers above happening to check the flag first. It is the door's
            # own business.
            return None
        if self._model_calls_handle is not None:
            return self._model_calls_handle
        handle = open_verified_append_text(self.run_dir / MODEL_CALLS_FILENAME)
        if handle is None:
            self._lose_model_calls_locked("it could not be safely opened")
            return None
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
                # The detail stays at ``debug``. Promoting the level carried this ``exc_info`` up
                # with it, and a rendered traceback names the absolute run directory -- run id and
                # whatever the deployment's parents are called -- on the stderr of every embedder,
                # which is exactly what ``_lose_artifact`` refuses to put in its message.
                _LOGGER.debug("model content store initialization failed", exc_info=True)
                self._lose_model_content_locked("it could not be opened")
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
                # idempotent, while EventBus separately owns close idempotence and stops its
                # callback fan-out when writer authority moves.
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
                    try:
                        with self._model_calls_lock:
                            # Marked closed under the same acquisition that takes the handle, so a
                            # concurrent record cannot slip between the two and reopen the file.
                            self._model_calls_closed = True
                            handle, self._model_calls_handle = self._model_calls_handle, None
                        if handle is not None:
                            try:
                                handle.close()
                            except OSError:
                                _LOGGER.debug("model call ledger close failed", exc_info=True)
                    finally:
                        # Fourth arm, nested for the same reason as the third: the corpus handle
                        # must be released even when the ledger's close raised.
                        with self._model_calls_lock:
                            self._model_payloads_closed = True
                            payloads_handle = self._model_payloads_handle
                            self._model_payloads_handle = None
                        if payloads_handle is not None:
                            try:
                                payloads_handle.close()
                            except OSError:
                                _LOGGER.debug("model payload corpus close failed", exc_info=True)

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
