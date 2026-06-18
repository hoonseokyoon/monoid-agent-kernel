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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
    ``None`` for a missing, unparseable, or schema-mismatched checkpoint — never raises."""
    path = run_dir / CHECKPOINT_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return None
    return RunCheckpoint.from_json(payload)
