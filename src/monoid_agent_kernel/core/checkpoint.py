"""Durable run checkpoint: a complete state snapshot at a safe recovery boundary.

The checkpoint carries the provider-neutral conversation, mutable run state, workspace delta,
hosted tasks, counters, and durable activation observations needed to rehydrate a fresh
``AgentLoop`` after process restart. Observable suspension boundaries and selected internal safety
barriers can both publish snapshots.

This persistence model uses state snapshots. Restore continues from the latest committed state
instead of replaying earlier model turns. An external effect can still commit
before the next snapshot, so such effects require a stable idempotency key or durable outbox.

``RunCheckpoint`` is a plain JSON container — the object<->dict conversions for
observations, content parts, runtime config and hosted tasks live in the loop's
``snapshot()``/``restore()`` so this module stays dependency-light.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from monoid_agent_kernel.core._util import file_lock, sha256_bytes, write_json_atomic
from monoid_agent_kernel.core.durable_codec import DurableCodec, DurableLoadResult
from monoid_agent_kernel.identifiers import accepted_namespaced_ids, namespaced_id

SCHEMA_VERSION = namespaced_id("checkpoint.v1")
ACCEPTED_SCHEMA_VERSIONS = accepted_namespaced_ids("checkpoint.v1")

CHECKPOINT_FILENAME = "checkpoint.json"


@dataclass
class RunCheckpoint:
    """Serialized park-point state of an open run. Field groups mirror ``RunState``
    (loop.py), ``_Session`` counters, and the parked hosted tasks. ``previous_surface_snapshot``
    is intentionally absent — it is recomputed each turn, so dropping it is safe."""

    run_id: str
    schema_version: str = SCHEMA_VERSION
    # Monotonic per-run checkpoint sequence; the store flips its "latest" pointer to
    # this only after the checkpoint is fully written (atomic commit / last-good).
    seq: int = 0
    # --- RunState (minus previous_surface_snapshot) ---
    status: str = "completed"
    error: str = ""
    error_code: str = ""
    provider_error_code: str = ""
    provider_http_status: int | None = None
    final_text: str = ""
    previous_turn_handle: str | None = None
    pending_user_input: list[dict[str, Any]] | None = None
    pending_observations: list[dict[str, Any]] = field(default_factory=list)
    pending_binding_loads: list[str] = field(default_factory=list)
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    previous_runtime_config: dict[str, Any] | None = None
    total_tool_calls: int = 0
    # Output-validator re-prompt counter, persisted so a mid-repair restart does not re-grant
    # the retry budget. (``final_output`` is NOT persisted — it is only set on success, after
    # which the run closes, so it never needs to survive a restart.)
    output_retries: int = 0
    total_usage: dict[str, int] = field(default_factory=dict)
    # By-value conversation log (vendor-independent continuation). Provider-neutral
    # user/assistant/tool messages; the system prompt is NOT here (regenerated each turn).
    messages: list[dict[str, Any]] = field(default_factory=list)

    # --- _Session counters ---
    session_step: int = 0
    submit_local_step: int = 0
    terminal: bool = False

    # --- parked hosted tasks + reentry bookkeeping ---
    hosted_tasks: list[dict[str, Any]] = field(default_factory=list)
    reentry_queue: list[str] = field(default_factory=list)
    delivered_reentry_jobs: list[str] = field(default_factory=list)

    # --- workspace delta (agent's created/modified/deleted files since the base) ---
    # Each entry: {path, kind, change_kind, base_sha256, proposed_sha256, content_sha256}.
    # File content lives in the store's content-addressed blobs (keyed by content_sha256),
    # not inline. ``workspace_base`` records which base the delta applies on top of, since
    # the agent workspace is not durable and the base is re-provisioned on restore.
    workspace_delta: list[dict[str, Any]] = field(default_factory=list)
    workspace_base: dict[str, Any] | None = None

    # --- capability leases (durable/approved only; ephemeral sync grants are not persisted) ---
    # Handles only (token_ref), never secrets; re-installed into the vault on restore so a
    # human-approved capability is not re-prompted after a restart. See core/capability.py.
    capability_leases: list[dict[str, Any]] = field(default_factory=list)
    # Gated tool calls awaiting auto-redispatch after their capability is granted (Phase ⑤).
    pending_capability_replays: list[dict[str, Any]] = field(default_factory=list)
    # Generic authorization="ask" tool calls awaiting auto-redispatch after an approval result.
    pending_tool_approval_replays: list[dict[str, Any]] = field(default_factory=list)
    # Capability revocation records (per-lease, per-capability, and an issued-before watermark) so a
    # revoked capability stays dead across a restart — the kill switch is not forgotten on resume.
    revoked_lease_ids: list[str] = field(default_factory=list)
    revoked_capabilities: list[str] = field(default_factory=list)
    revoked_before: float = 0.0
    revoked_all: bool = False

    # --- run-level bookkeeping ---
    remaining_duration_s: float | None = None
    cancellation_requested: bool = False
    # Filled by the backend driver (the message queue lives outside the loop). Each entry is
    # JSON-native: a ``str`` (text message) or a ``list[dict]`` of content-part dicts (a
    # multimodal message carried by-reference). Kept JSON-native so the checkpoint round-trips
    # without any dataclass (de)serialization here.
    queued_messages: list[Any] = field(default_factory=list)
    # Idempotency: ids of inbox messages already processed, so a redelivery after a restart is
    # recognized and dropped (effectively-once ingress). Additive; old checkpoints default to [].
    inbox_seen_ids: list[str] = field(default_factory=list)
    # Staged outbound side-effects (capability-gated). Persisted in full (handles only, never
    # secrets) so a pending request survives a restart and is (re)dispatched by the edge. Additive.
    outbox_requests: list[dict[str, Any]] = field(default_factory=list)

    # --- additive v0.18 recovery observations (kept at the tail for positional compatibility) ---
    # Portable observation of the suspension boundary that produced this checkpoint.
    # Recovery drivers use it to return the same result when an input was committed before
    # the driver's own receipt. Older checkpoints omit it and remain readable.
    last_suspension: dict[str, Any] | None = None
    # Generic activation/input identities whose resulting boundary is already committed. A
    # recovery driver returns the stored boundary for a repeated id instead of driving effects
    # again. This stays transport- and implementation-neutral; old checkpoints default to [].
    applied_input_ids: list[str] = field(default_factory=list)
    # The input currently advancing between durable boundaries. ``phase`` is driver-defined but
    # portable values are "running" and "completed"; ``source_seq`` identifies the admitted
    # source. A matching recovery can continue from an internal safety checkpoint.
    active_input: dict[str, Any] | None = None
    # Immutable boundary receipts keyed by applied input identity. They let an old duplicate return
    # its own stored observation after newer inputs have advanced the run.
    applied_input_receipts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> RunCheckpoint | None:
        """Compatibility wrapper over :func:`decode_checkpoint`."""
        return decode_checkpoint(payload).value


_CHECKPOINT_STRING_FIELDS = frozenset(
    {
        "status",
        "error",
        "error_code",
        "provider_error_code",
        "final_text",
    }
)
_CHECKPOINT_OPTIONAL_STRING_FIELDS = frozenset({"previous_turn_handle"})
_CHECKPOINT_NONNEGATIVE_INT_FIELDS = frozenset(
    {
        "seq",
        "provider_http_status",
        "total_tool_calls",
        "output_retries",
        "session_step",
        "submit_local_step",
    }
)
_CHECKPOINT_BOOL_FIELDS = frozenset({"terminal", "revoked_all", "cancellation_requested"})
_CHECKPOINT_LIST_OF_DICT_FIELDS = frozenset(
    {
        "pending_observations",
        "messages",
        "hosted_tasks",
        "workspace_delta",
        "capability_leases",
        "pending_capability_replays",
        "pending_tool_approval_replays",
        "outbox_requests",
    }
)
_CHECKPOINT_LIST_OF_STRING_FIELDS = frozenset(
    {
        "pending_binding_loads",
        "reentry_queue",
        "delivered_reentry_jobs",
        "revoked_lease_ids",
        "revoked_capabilities",
        "inbox_seen_ids",
        "applied_input_ids",
    }
)


def _require_nonempty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"checkpoint {field_name} must be a non-empty string")


def _require_nonnegative_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"checkpoint {field_name} must be a non-negative integer")


def _require_finite_nonnegative_number(value: object, field_name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError(f"checkpoint {field_name} must be a finite non-negative number")


def _require_list_of(value: object, item_type: type[object], field_name: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, item_type) for item in value):
        raise ValueError(f"checkpoint {field_name} has an invalid list shape")


def _validate_counter_mapping(value: object, field_name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint {field_name} must be an object")
    for key, count in value.items():
        if not isinstance(key, str):
            raise ValueError(f"checkpoint {field_name} keys must be strings")
        _require_nonnegative_int(count, f"{field_name}.{key}")


def _validate_active_input(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("checkpoint active_input must be an object or null")
    _require_nonempty_string(value.get("input_id"), "active_input.input_id")
    if value.get("phase") not in {"running", "completed"}:
        raise ValueError("checkpoint active_input.phase must be running or completed")
    _require_nonnegative_int(value.get("source_seq"), "active_input.source_seq")


def _validate_receipts(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("checkpoint applied_input_receipts must be an object")
    for input_id, receipt in value.items():
        _require_nonempty_string(input_id, "applied_input_receipts key")
        if not isinstance(receipt, dict):
            raise ValueError("checkpoint applied_input_receipts values must be objects")
        if "checkpoint_seq" in receipt:
            _require_nonnegative_int(
                receipt["checkpoint_seq"],
                f"applied_input_receipts.{input_id}.checkpoint_seq",
            )
        if "terminal" in receipt and not isinstance(receipt["terminal"], bool):
            raise ValueError(
                f"checkpoint applied_input_receipts.{input_id}.terminal must be boolean"
            )
        if "suspension" in receipt and not isinstance(receipt["suspension"], dict):
            raise ValueError(
                f"checkpoint applied_input_receipts.{input_id}.suspension must be an object"
            )
        for field_name in ("checkpoint_sha256", "state", "error", "error_code"):
            if field_name in receipt and not isinstance(receipt[field_name], str):
                raise ValueError(
                    f"checkpoint applied_input_receipts.{input_id}.{field_name} must be a string"
                )


def _validate_checkpoint_payload(payload: dict[str, Any]) -> None:
    _require_nonempty_string(payload.get("run_id"), "run_id")
    for field_name in _CHECKPOINT_STRING_FIELDS:
        if field_name in payload and not isinstance(payload[field_name], str):
            raise ValueError(f"checkpoint {field_name} must be a string")
    for field_name in _CHECKPOINT_OPTIONAL_STRING_FIELDS:
        if (
            field_name in payload
            and payload[field_name] is not None
            and not isinstance(payload[field_name], str)
        ):
            raise ValueError(f"checkpoint {field_name} must be a string or null")
    for field_name in _CHECKPOINT_NONNEGATIVE_INT_FIELDS:
        if field_name in payload and payload[field_name] is not None:
            _require_nonnegative_int(payload[field_name], field_name)
    for field_name in _CHECKPOINT_BOOL_FIELDS:
        if field_name in payload and not isinstance(payload[field_name], bool):
            raise ValueError(f"checkpoint {field_name} must be boolean")
    for field_name in _CHECKPOINT_LIST_OF_DICT_FIELDS:
        if field_name in payload:
            _require_list_of(payload[field_name], dict, field_name)
    for field_name in _CHECKPOINT_LIST_OF_STRING_FIELDS:
        if field_name in payload:
            _require_list_of(payload[field_name], str, field_name)
    if "pending_user_input" in payload and payload["pending_user_input"] is not None:
        _require_list_of(payload["pending_user_input"], dict, "pending_user_input")
    for field_name in ("previous_runtime_config", "workspace_base", "last_suspension"):
        if (
            field_name in payload
            and payload[field_name] is not None
            and not isinstance(payload[field_name], dict)
        ):
            raise ValueError(f"checkpoint {field_name} must be an object or null")
    for field_name in ("tool_call_counts", "total_usage"):
        if field_name in payload:
            _validate_counter_mapping(payload[field_name], field_name)
    for field_name in ("revoked_before", "remaining_duration_s"):
        if field_name in payload and payload[field_name] is not None:
            _require_finite_nonnegative_number(payload[field_name], field_name)
    if "queued_messages" in payload:
        messages = payload["queued_messages"]
        if not isinstance(messages, list):
            raise ValueError("checkpoint queued_messages must be a list")
        for message in messages:
            if isinstance(message, (str, dict)):
                continue
            if isinstance(message, list) and all(isinstance(part, dict) for part in message):
                continue
            raise ValueError(
                "checkpoint queued_messages entries must be strings, envelopes, or content lists"
            )
    if "active_input" in payload:
        _validate_active_input(payload["active_input"])
    if "applied_input_receipts" in payload:
        _validate_receipts(payload["applied_input_receipts"])


def _checkpoint_from_payload(payload: dict[str, Any]) -> RunCheckpoint:
    _validate_checkpoint_payload(payload)
    known = {name for name in RunCheckpoint.__dataclass_fields__}
    return RunCheckpoint(**{key: value for key, value in payload.items() if key in known})


CHECKPOINT_CODEC = DurableCodec[RunCheckpoint](
    family="checkpoint",
    current_schema=SCHEMA_VERSION,
)


def decode_checkpoint(payload: object) -> DurableLoadResult[RunCheckpoint]:
    return CHECKPOINT_CODEC.decode(payload, _checkpoint_from_payload)


def checkpoint_payload_for_write(checkpoint: RunCheckpoint) -> dict[str, Any]:
    """Return the canonical current writer shape regardless of a restored alias."""
    payload = checkpoint.to_json()
    payload["schema_version"] = SCHEMA_VERSION
    _validate_checkpoint_payload(payload)
    return payload


def write_checkpoint(run_dir: Path, checkpoint: RunCheckpoint) -> Path:
    """Atomically write ``checkpoint`` to ``run_dir/checkpoint.json`` and return the path."""
    path = run_dir / CHECKPOINT_FILENAME
    write_json_atomic(path, checkpoint_payload_for_write(checkpoint))
    return path


def read_checkpoint_checked(run_dir: Path) -> DurableLoadResult[RunCheckpoint]:
    """Read a single-file checkpoint with an explicit durable load outcome."""
    path = run_dir / CHECKPOINT_FILENAME
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return CHECKPOINT_CODEC.missing()
    except OSError:
        return CHECKPOINT_CODEC.corrupt("checkpoint file could not be read")
    try:
        payload = json.loads(raw)
    except ValueError:
        return CHECKPOINT_CODEC.corrupt("checkpoint file is not valid JSON")
    return decode_checkpoint(payload)


def read_checkpoint(run_dir: Path) -> RunCheckpoint | None:
    """Read ``run_dir/checkpoint.json`` if present and schema-compatible. Returns
    ``None`` for a missing, unparseable, or schema-mismatched checkpoint — never raises.

    Single-file helper retained for simple round-trips and tests; production code goes
    through a ``CheckpointStore`` (which adds seq, atomic commit, and content blobs)."""
    return read_checkpoint_checked(run_dir).value


# --- CheckpointStore seam: core defines WHAT (RunCheckpoint), the store defines HOW ---


@dataclass
class CheckpointRecord:
    """A fully-committed checkpoint read back from a store: the manifest plus lazy
    access to its content blobs (changed-file bytes, keyed by sha256). The blob accessor
    is a callable, not a directory, so a store can back it with files, a DB, or an object
    store — the loop only ever needs ``blob(sha)``."""

    seq: int
    checkpoint: RunCheckpoint
    _blob_reader: Callable[[str], bytes] | None = None

    def blob(self, sha256: str) -> bytes:
        """Read a content blob by its sha256 key (workspace delta files, Phase L)."""
        if self._blob_reader is None:
            raise KeyError(sha256)
        return self._blob_reader(sha256)


class CheckpointStore(Protocol):
    """How a checkpoint is durably stored. The core produces a ``RunCheckpoint`` (the
    WHAT) and hands it here; the integrator implements the HOW (local fs / mounted
    volume / object store / DB). ``put`` MUST commit atomically — a partially-written
    checkpoint must never be returned by ``latest``."""

    def put(self, checkpoint: RunCheckpoint, blobs: Mapping[str, bytes] = ...) -> None: ...

    def latest(self, run_id: str) -> CheckpointRecord | None: ...

    def delete(self, run_id: str) -> None: ...

    def put_blob(self, run_id: str, data: bytes) -> str:
        """Store ``data`` as a content-addressed, write-once blob and return its sha256 digest.
        The digest IS the retrieval handle (see ``get_blob``) — content-addressed, so an identical
        payload dedups and the same bytes always map to the same handle. This is the standalone
        entry to the same blob namespace ``put`` fills for checkpoints, used for on-demand
        artifacts (e.g. an exported package) that must be fetched back as data, never by path."""
        ...

    def get_blob(self, run_id: str, sha256: str) -> bytes:
        """Read a content-addressed blob by its sha256 digest. Raises ``KeyError`` if absent."""
        ...

    def put_run_metadata(self, run_id: str, metadata: Mapping[str, Any]) -> None:
        """Store backend-owned run recovery metadata beside checkpoints."""
        ...

    def run_metadata(self, run_id: str) -> dict[str, Any] | None:
        """Return backend-owned recovery metadata for ``run_id`` if present."""
        ...


class CheckedCheckpointStore(Protocol):
    """Optional store extension that preserves checked durable load outcomes."""

    def latest_checked(self, run_id: str) -> DurableLoadResult[CheckpointRecord]: ...


def bind_checkpoint_record_result(
    result: DurableLoadResult[CheckpointRecord], run_id: str
) -> DurableLoadResult[CheckpointRecord]:
    """Bind a checked record to its lookup key and committed sequence."""

    if not result.ok:
        return result
    record = result.value
    if not isinstance(record, CheckpointRecord):
        return CHECKPOINT_CODEC.corrupt(
            "checkpoint store returned an invalid record",
            sequence=result.sequence,
        ).map(lambda checkpoint: CheckpointRecord(seq=checkpoint.seq, checkpoint=checkpoint))
    if isinstance(record.seq, bool) or not isinstance(record.seq, int) or record.seq < 0:
        return CHECKPOINT_CODEC.corrupt(
            "checkpoint store returned an invalid committed sequence",
            sequence=result.sequence,
        ).map(lambda checkpoint: CheckpointRecord(seq=checkpoint.seq, checkpoint=checkpoint))
    if record.checkpoint.run_id != run_id:
        return CHECKPOINT_CODEC.corrupt(
            "checkpoint manifest run_id does not match the requested run",
            sequence=record.seq,
        ).map(lambda checkpoint: CheckpointRecord(seq=checkpoint.seq, checkpoint=checkpoint))
    if record.checkpoint.seq != record.seq:
        return CHECKPOINT_CODEC.corrupt(
            "checkpoint manifest sequence does not match the committed sequence",
            sequence=record.seq,
        ).map(lambda checkpoint: CheckpointRecord(seq=checkpoint.seq, checkpoint=checkpoint))
    if result.sequence is not None and result.sequence != record.seq:
        return CHECKPOINT_CODEC.corrupt(
            "checkpoint checked result sequence does not match the committed record",
            sequence=record.seq,
        ).map(lambda checkpoint: CheckpointRecord(seq=checkpoint.seq, checkpoint=checkpoint))
    return replace(result, sequence=record.seq)


def load_latest_checked(store: CheckpointStore, run_id: str) -> DurableLoadResult[CheckpointRecord]:
    """Use a checked store when available and adapt legacy stores without breaking them."""
    checked = getattr(store, "latest_checked", None)
    if callable(checked):
        result = checked(run_id)
        if isinstance(result, DurableLoadResult):
            return bind_checkpoint_record_result(result, run_id)
        return CHECKPOINT_CODEC.corrupt("checkpoint store returned an invalid checked result").map(
            lambda checkpoint: CheckpointRecord(seq=checkpoint.seq, checkpoint=checkpoint)
        )
    # A legacy store cannot distinguish malformed durable state from a transient
    # transport failure. Preserve its exception so recovery can defer and retry;
    # only checked stores may authoritatively classify an artifact as corrupt.
    record = store.latest(run_id)
    if record is None:
        return DurableLoadResult(
            status="missing",
            family=CHECKPOINT_CODEC.family,
            current_schema=CHECKPOINT_CODEC.current_schema,
        )
    return bind_checkpoint_record_result(
        DurableLoadResult(
            status="loaded",
            family=CHECKPOINT_CODEC.family,
            current_schema=CHECKPOINT_CODEC.current_schema,
            value=record,
            observed_schema=record.checkpoint.schema_version,
            sequence=record.seq,
        ),
        run_id,
    )

@dataclass
class LocalFsCheckpointStore:
    """Default local-filesystem store. Layout under ``run_root/<run_id>/checkpoints/``:
    ``blobs/<sha>`` (content-addressed, write-once, shared across seqs) and
    ``<seq>/manifest.json``; a ``LATEST`` pointer is flipped only after the manifest is
    committed. In a container this is durable iff ``run_root`` is a durable mount; an
    object-store/DB store is a drop-in replacement (same protocol)."""

    run_root: Path
    # Cross-process put serialization. A writer steals a lock left by a crashed peer once
    # it is older than ``lock_stale_s``; ``lock_timeout_s`` bounds how long it waits for a
    # live peer before stealing anyway (so a stuck holder can never deadlock a put).
    lock_timeout_s: float = 10.0
    lock_stale_s: float = 30.0

    def _dir(self, run_id: str) -> Path:
        return self.run_root / run_id / "checkpoints"

    def put(self, checkpoint: RunCheckpoint, blobs: Mapping[str, bytes] = {}) -> None:
        cdir = self._dir(checkpoint.run_id)
        # Serialize puts for this run across processes: without it, two writers could interleave
        # blob writes and LATEST flips and tear a checkpoint.
        cdir.mkdir(parents=True, exist_ok=True)
        with file_lock(cdir / ".put.lock", timeout_s=self.lock_timeout_s, stale_s=self.lock_stale_s):
            # 0) GC orphaned blob temp files left by a crashed prior write (no LATEST was
            #    flipped for them, so they are pure dead weight).
            self._gc_blob_tmp(cdir)
            # 1) Content blobs first — content-addressed and write-once, so a crash here
            #    only leaves harmless orphans (no LATEST flip yet).
            if blobs:
                blobs_dir = cdir / "blobs"
                blobs_dir.mkdir(parents=True, exist_ok=True)
                for sha256, data in blobs.items():
                    target = blobs_dir / sha256
                    if not target.exists():
                        tmp = target.with_suffix(".tmp")
                        tmp.write_bytes(data)
                        tmp.replace(target)
            # 2) Manifest (atomic file write into the seq dir).
            seq_dir = cdir / str(checkpoint.seq)
            seq_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(seq_dir / "manifest.json", checkpoint_payload_for_write(checkpoint))
            # 3) Flip the latest pointer last — only now is this seq considered committed.
            #    Monotonic: never regress LATEST to an older seq, so a late/lower-seq writer
            #    cannot unpublish a newer committed checkpoint (re-putting the same seq, as
            #    the backend does to fold in the message queue, is a no-op flip).
            if checkpoint.seq > self._read_latest_seq(cdir):
                write_json_atomic(cdir / "LATEST", {"seq": checkpoint.seq})

    def put_blob(self, run_id: str, data: bytes) -> str:
        """Write a standalone content-addressed blob into the run's ``blobs/`` dir (write-once,
        same namespace as checkpoint blobs) and return its sha256 digest."""
        sha = sha256_bytes(data)
        cdir = self._dir(run_id)
        blobs_dir = cdir / "blobs"
        blobs_dir.mkdir(parents=True, exist_ok=True)
        with file_lock(cdir / ".put.lock", timeout_s=self.lock_timeout_s, stale_s=self.lock_stale_s):
            target = blobs_dir / sha
            if not target.exists():
                tmp = target.with_suffix(".tmp")
                tmp.write_bytes(data)
                tmp.replace(target)
        return sha

    def get_blob(self, run_id: str, sha256: str) -> bytes:
        try:
            return (self._dir(run_id) / "blobs" / sha256).read_bytes()
        except OSError as exc:
            raise KeyError(sha256) from exc

    def put_run_metadata(self, run_id: str, metadata: Mapping[str, Any]) -> None:
        cdir = self._dir(run_id)
        cdir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(cdir / "run_meta.json", dict(metadata))

    def run_metadata(self, run_id: str) -> dict[str, Any] | None:
        try:
            payload = json.loads((self._dir(run_id) / "run_meta.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return None
        if not isinstance(payload, dict) or payload.get("run_id") != run_id:
            return None
        return dict(payload)

    def run_metadata_checked(self, run_id: str) -> DurableLoadResult[dict[str, Any]]:
        from monoid_agent_kernel.core.durable_metadata import (
            RUN_METADATA_CODEC,
            bind_run_metadata_result,
            decode_run_metadata,
        )

        path = self._dir(run_id) / "run_meta.json"
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return RUN_METADATA_CODEC.missing()
        except OSError:
            return RUN_METADATA_CODEC.corrupt("backend-run metadata could not be read")
        try:
            payload = json.loads(raw)
        except ValueError:
            return RUN_METADATA_CODEC.corrupt("backend-run metadata is not valid JSON")
        return bind_run_metadata_result(decode_run_metadata(payload), run_id)

    def _read_latest_seq(self, cdir: Path) -> int:
        try:
            return int(json.loads((cdir / "LATEST").read_text(encoding="utf-8"))["seq"])
        except (FileNotFoundError, ValueError, OSError, KeyError, TypeError):
            return -1

    def _gc_blob_tmp(self, cdir: Path) -> None:
        blobs_dir = cdir / "blobs"
        if not blobs_dir.is_dir():
            return
        for tmp in blobs_dir.glob("*.tmp"):
            try:
                tmp.unlink()
            except OSError:  # pragma: no cover - best-effort cleanup
                pass

    def latest_checked(self, run_id: str) -> DurableLoadResult[CheckpointRecord]:
        cdir = self._dir(run_id)
        # Metadata and standalone blobs share this directory and may exist before the
        # first checkpoint. Only a published LATEST pointer proves a checkpoint commit
        # was attempted; without it the checkpoint state is genuinely missing.
        if not cdir.is_dir() or not (cdir / "LATEST").exists():
            return CHECKPOINT_CODEC.missing().map(
                lambda checkpoint: CheckpointRecord(seq=checkpoint.seq, checkpoint=checkpoint)
            )
        # A reader can race a concurrent put()'s atomic LATEST/manifest replace, including across
        # processes.
        # Retry a few times so a transient read failure mid-commit is not mistaken for
        # "no checkpoint" — returning None here would wrongly skip a recoverable run.
        seq = -1
        manifest: dict[str, Any] | None = None
        for attempt in range(4):
            try:
                pointer = json.loads((cdir / "LATEST").read_text(encoding="utf-8"))
                seq = int(pointer["seq"])
                manifest = json.loads((cdir / str(seq) / "manifest.json").read_text(encoding="utf-8"))
                break
            except (ValueError, OSError, KeyError, TypeError):
                manifest = None
                if attempt < 3:
                    time.sleep(0.01)
        if manifest is None:
            return CHECKPOINT_CODEC.corrupt(
                "checkpoint latest pointer or manifest could not be read",
                sequence=seq if seq >= 0 else None,
            )
        decoded = replace(decode_checkpoint(manifest), sequence=seq)
        blobs_dir = cdir / "blobs"
        return bind_checkpoint_record_result(
            decoded.map(
                lambda checkpoint: CheckpointRecord(
                    seq=seq,
                    checkpoint=checkpoint,
                    _blob_reader=lambda sha256: (blobs_dir / sha256).read_bytes(),
                )
            ),
            run_id,
        )

    def latest(self, run_id: str) -> CheckpointRecord | None:
        return self.latest_checked(run_id).value

    def delete(self, run_id: str) -> None:
        shutil.rmtree(self._dir(run_id), ignore_errors=True)
