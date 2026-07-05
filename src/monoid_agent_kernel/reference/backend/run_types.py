from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core.agents import AgentDefinition, AgentRuntimeConfig
from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.content import ContentPart
from monoid_agent_kernel.core.lifecycle import SessionState, session_state_value
from monoid_agent_kernel.core.result import AgentRunResult
from monoid_agent_kernel.core.spec import RunMode, WorkspaceBackendKind
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.permissions import PermissionPolicy


@dataclass(frozen=True)
class BackendRunRequest:
    tenant_id: str
    user_id: str
    workspace_root: Path
    instruction: str
    # Optional multimodal first turn: when non-empty, these content parts (text + image/document
    # references) drive the opening turn instead of ``instruction``. ``instruction`` is still used
    # for the run title / metadata, so callers pass the text alongside.
    input_parts: tuple[ContentPart, ...] = ()
    mode: RunMode = "propose"
    workspace_backend: WorkspaceBackendKind = "overlay"
    max_steps: int = 30
    max_tool_calls: int = 100
    max_bytes_read: int = 1_000_000
    max_duration_s: int | None = 900
    permission_policy: PermissionPolicy = field(default_factory=PermissionPolicy)
    agent_definition: AgentDefinition | None = None
    runtime_config: AgentRuntimeConfig | None = None
    # When False (default) the run closes after the first turn settles (one-shot).
    # When True the session stays open awaiting follow-up messages (multi-turn).
    multi_turn: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendRunSubmission:
    run_id: str
    run_token: str
    state: SessionState
    terminal: bool
    run_dir: Path
    status_url: str
    result_url: str
    events_url: str
    proposal_url: str

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_token": self.run_token,
            "state": session_state_value(self.state),
            "terminal": self.terminal,
            "run_dir": str(self.run_dir),
            "status_url": self.status_url,
            "result_url": self.result_url,
            "events_url": self.events_url,
            "proposal_url": self.proposal_url,
        }


@dataclass(frozen=True)
class _PreparedRun:
    """The shared output of run setup (validate + tokens + stored record), before the run is
    driven. Consumed by ``submit_run`` (autonomous) and ``astream_run`` (stream-driven)."""

    run_id: str
    record: BackendRunRecord
    workspace_root: Path
    run_token: str
    llm_gateway_token: str
    web_gateway_token: str


@dataclass
class BackendRunRecord:
    run_id: str
    tenant_id: str
    user_id: str
    workspace_root: Path
    run_dir: Path
    state: SessionState
    terminal: bool
    created_at: float
    run_token_sha256: str
    llm_gateway_token_sha256: str
    web_gateway_token_sha256: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    error: str = ""
    error_code: str = ""
    result: AgentRunResult | None = None
    # Latest settled turn's validated output (AgentTurnResult.final_output), captured per park so a
    # live multi-turn run can expose it via status() before the run closes (result() carries the
    # final one). Process-local — not persisted. None when no output validator produced a value.
    last_final_output: Any = None
    last_event_seq: int = 0
    last_event_type: str = ""
    cancellation_token: CancellationToken = field(default_factory=CancellationToken)
    runtime_config: AgentRuntimeConfig | None = None
    runtime_config_issuer: str = ""
    runtime_config_reason: str = ""
    runtime_config_committed_at: float = 0.0
    # Authoritative lifecycle FSM state, updated by the session driver as it observes each
    # suspension. The control protocol's inspect/health read this (a throwaway LoopSession is
    # seeded with it) since the backend drives the loop directly, not through a facade.
    loop: AgentLoop | None = None
    # Pending user messages for a multi-turn session. asyncio.Queue (not queue.Queue) so the
    # run coroutine awaits the next message WITHOUT holding a thread — a parked multi-turn
    # session is just a suspended coroutine, not a blocked worker (which would exhaust the
    # shared executor). Producers (send_message/cancel from other threads) enqueue via the
    # backend's _call_soon so the put runs on the loop. Created without a running loop (3.10+
    # binds lazily); all gets/puts happen on the shared loop.
    message_queue: asyncio.Queue[Any] = field(default_factory=asyncio.Queue, repr=False)
    # Ids of inbox messages already processed — the idempotency/dedup set. Checkpointed (restored on
    # recover) so a redelivered message is dropped once, even across a restart. Mutated only on the
    # shared loop (dequeue), so no extra lock is needed.
    seen_inbox_ids: set[str] = field(default_factory=set, repr=False)
    # The run's outbox sender (drains staged sends), or None to leave staged requests pending.
    outbox_sender: Any = field(default=None, repr=False)
