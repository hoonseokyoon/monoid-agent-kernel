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
