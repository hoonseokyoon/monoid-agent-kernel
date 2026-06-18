from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

RunStatus = Literal["completed", "failed", "limited"]


@dataclass(frozen=True)
class AgentArtifact:
    artifact_id: str
    path: str
    kind: str
    label: str | None = None


@dataclass(frozen=True)
class AgentTurnResult:
    """Result of a single ``submit()`` (one user turn settling).

    Non-terminal: the run stays open for further submits. ``proposal_*`` reflect
    the accumulated workspace changes so far (preview), and ``turn_handle`` is the
    continuation handle to thread into the next user turn.
    """

    status: RunStatus
    final_text: str
    proposal_path: Path
    proposal_hash: str
    changed_paths: tuple[str, ...] = ()
    turn_handle: str | None = None
    error: str = ""
    error_code: str = ""
    metrics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Suspension:
    """Why a non-blocking pump (``AgentLoop.run_until_suspended``) returned control.

    ``settled`` — the model produced final text and the run awaits the next user
    message. ``awaiting_tasks`` — the run parked with no model tool work and
    pending tasks; ``awaiting_task_ids``/``has_external`` describe the hosted
    (hitl/automation) tasks a caller must wait on. ``limited`` — a per-submit or
    session budget was hit. ``terminal`` — cancelled/timed out/failed.
    For every reason except ``awaiting_tasks`` a settle checkpoint ran and
    ``turn`` carries its result.
    """

    reason: Literal["settled", "awaiting_tasks", "limited", "terminal"]
    status: RunStatus
    final_text: str = ""
    error: str = ""
    error_code: str = ""
    awaiting_task_ids: tuple[str, ...] = ()
    has_external: bool = False
    turn: AgentTurnResult | None = None


@dataclass(frozen=True)
class AgentRunResult:
    run_id: str
    status: RunStatus
    final_text: str
    run_dir: Path
    diff_path: Path
    proposal_path: Path
    artifacts: tuple[AgentArtifact, ...] = ()
    final_outputs: tuple[str, ...] = ()
    final_notes: str | None = None
    metrics: dict[str, object] = field(default_factory=dict)
    error: str = ""
    error_code: str = ""
    final_turn_handle: str | None = None
