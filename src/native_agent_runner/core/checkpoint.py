"""Durable run checkpoint: a state-snapshot taken at a pump park point.

The harness is already half-durable — the provider holds the conversation by
reference (``turn_handle``), the workspace truth is on disk, and events/transcript
are append-only. The only volatile state worth persisting is the small mutable
``RunState`` (handle + counters + pending observations) plus the hosted tasks a run
is parked on. This module is the serialized form of exactly that, written to
``run_dir/checkpoint.json`` and read back to rehydrate a fresh ``AgentLoop`` after a
process restart.

This is a *state-snapshot* (LangGraph style), not an event-sourcing journal
(Temporal/Restate style): because the LLM transcript is by-reference, restore never
replays the model, so there is no determinism constraint and no double-side-effect
risk. Snapshots are only ever taken at clean park points, never mid-step.

``RunCheckpoint`` is a plain JSON container — the object<->dict conversions for
observations, content parts, runtime config and hosted tasks live in the loop's
``snapshot()``/``restore()`` so this module stays dependency-light.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from native_agent_runner.core._util import write_json_atomic

SCHEMA_VERSION = "native-agent-runner.checkpoint.v1"

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
    total_usage: dict[str, int] = field(default_factory=dict)

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

    # --- run-level bookkeeping ---
    remaining_duration_s: float | None = None
    cancellation_requested: bool = False
    # Filled by the backend driver (the message queue lives outside the loop).
    queued_messages: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> RunCheckpoint | None:
        """Rebuild from a checkpoint payload. Returns ``None`` on a schema mismatch
        (forward/backward incompatibility) rather than raising — the caller treats an
        unreadable checkpoint as "no checkpoint" and skips recovery."""
        if not isinstance(payload, dict):
            return None
        if payload.get("schema_version") != SCHEMA_VERSION:
            return None
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


def write_checkpoint(run_dir: Path, checkpoint: RunCheckpoint) -> Path:
    """Atomically write ``checkpoint`` to ``run_dir/checkpoint.json`` and return the path."""
    path = run_dir / CHECKPOINT_FILENAME
    write_json_atomic(path, checkpoint.to_json())
    return path


def read_checkpoint(run_dir: Path) -> RunCheckpoint | None:
    """Read ``run_dir/checkpoint.json`` if present and schema-compatible. Returns
    ``None`` for a missing, unparseable, or schema-mismatched checkpoint — never raises.

    Single-file helper retained for simple round-trips and tests; production code goes
    through a ``CheckpointStore`` (which adds seq, atomic commit, and content blobs)."""
    path = run_dir / CHECKPOINT_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return None
    return RunCheckpoint.from_json(payload)


# --- CheckpointStore seam: core defines WHAT (RunCheckpoint), the store defines HOW ---


@dataclass
class CheckpointRecord:
    """A fully-committed checkpoint read back from a store: the manifest plus lazy
    access to its content blobs (changed-file bytes, keyed by sha256)."""

    seq: int
    checkpoint: RunCheckpoint
    _blobs_dir: Path | None = None

    def blob(self, sha256: str) -> bytes:
        """Read a content blob by its sha256 key (workspace delta files, Phase L)."""
        if self._blobs_dir is None:
            raise KeyError(sha256)
        return (self._blobs_dir / sha256).read_bytes()


class CheckpointStore(Protocol):
    """How a checkpoint is durably stored. The core produces a ``RunCheckpoint`` (the
    WHAT) and hands it here; the integrator implements the HOW (local fs / mounted
    volume / object store / DB). ``put`` MUST commit atomically — a partially-written
    checkpoint must never be returned by ``latest``."""

    def put(self, checkpoint: RunCheckpoint, blobs: Mapping[str, bytes] = ...) -> None: ...

    def latest(self, run_id: str) -> CheckpointRecord | None: ...

    def delete(self, run_id: str) -> None: ...


@dataclass
class LocalFsCheckpointStore:
    """Default local-filesystem store. Layout under ``run_root/<run_id>/checkpoints/``:
    ``blobs/<sha>`` (content-addressed, write-once, shared across seqs) and
    ``<seq>/manifest.json``; a ``LATEST`` pointer is flipped only after the manifest is
    committed. In a container this is durable iff ``run_root`` is a durable mount; an
    object-store/DB store is a drop-in replacement (same protocol)."""

    run_root: Path

    def _dir(self, run_id: str) -> Path:
        return self.run_root / run_id / "checkpoints"

    def put(self, checkpoint: RunCheckpoint, blobs: Mapping[str, bytes] = {}) -> None:
        cdir = self._dir(checkpoint.run_id)
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
        write_json_atomic(seq_dir / "manifest.json", checkpoint.to_json())
        # 3) Flip the latest pointer last — only now is this seq considered committed.
        write_json_atomic(cdir / "LATEST", {"seq": checkpoint.seq})

    def latest(self, run_id: str) -> CheckpointRecord | None:
        cdir = self._dir(run_id)
        try:
            pointer = json.loads((cdir / "LATEST").read_text(encoding="utf-8"))
            seq = int(pointer["seq"])
            manifest = json.loads((cdir / str(seq) / "manifest.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError, KeyError, TypeError):
            return None
        checkpoint = RunCheckpoint.from_json(manifest)
        if checkpoint is None:
            return None
        return CheckpointRecord(seq=seq, checkpoint=checkpoint, _blobs_dir=cdir / "blobs")

    def delete(self, run_id: str) -> None:
        shutil.rmtree(self._dir(run_id), ignore_errors=True)
