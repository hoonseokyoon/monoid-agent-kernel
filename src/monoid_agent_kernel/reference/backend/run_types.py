from __future__ import annotations

import asyncio
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from monoid_agent_kernel._runtime_config_ingress import normalize_runtime_config
from monoid_agent_kernel.core.agents import AgentDefinition, AgentRuntimeConfig
from monoid_agent_kernel.core.authority import ActivationWriteAuthority
from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.content import ContentPart, normalize_content_part
from monoid_agent_kernel.core.json_ingress import normalize_json_ingress, normalize_unicode_scalars
from monoid_agent_kernel.core.lifecycle import SessionState, session_state_value
from monoid_agent_kernel.core.outbox import OutboxSender
from monoid_agent_kernel.core.outcome import InterruptionCause
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


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return normalize_unicode_scalars(value)


def _nonnegative_integer(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _normalize_permission_policy(policy: Any) -> PermissionPolicy:
    if not isinstance(policy, PermissionPolicy):
        raise ValueError("permission_policy must be a PermissionPolicy")
    return PermissionPolicy(
        deny_patterns=tuple(
            _required_text(pattern, "permission_policy deny pattern")
            for pattern in policy.deny_patterns
        ),
        redact_patterns=tuple(
            _required_text(pattern, "permission_policy redact pattern")
            for pattern in policy.redact_patterns
        ),
    )


def _normalize_agent_definition(definition: Any) -> AgentDefinition | None:
    if definition is None:
        return None
    if not isinstance(definition, AgentDefinition):
        raise ValueError("agent_definition must be an AgentDefinition or null")
    normalized_config = normalize_runtime_config(AgentRuntimeConfig.from_definition(definition))
    normalized_metadata = normalize_json_ingress(definition.metadata)
    if not isinstance(normalized_metadata, dict):
        raise ValueError("agent_definition.metadata must be an object")
    normalized = copy(definition)
    for name, value in {
        "id": normalized_config.definition_id,
        "version": _required_text(definition.version, "agent_definition.version"),
        "description": _required_text(
            definition.description,
            "agent_definition.description",
        ),
        "model": normalized_config.model,
        "prompt": normalized_config.prompt,
        "tools": normalized_config.tools,
        "tool_search": normalized_config.tool_search,
        "metadata": normalized_metadata,
    }.items():
        object.__setattr__(normalized, name, value)
    return normalized


def normalize_backend_run_request(request: BackendRunRequest) -> BackendRunRequest:
    """Copy a direct Python run request into its portable typed domain before any side effect."""

    if not isinstance(request, BackendRunRequest):
        raise ValueError("request must be a BackendRunRequest")
    if not isinstance(request.workspace_root, Path):
        raise ValueError("workspace_root must be a Path")
    if not isinstance(request.input_parts, (list, tuple)):
        raise ValueError("input_parts must be an array of content parts")
    if type(request.multi_turn) is not bool:
        raise ValueError("multi_turn must be a boolean")
    if request.max_duration_s is not None and (
        type(request.max_duration_s) is not int or request.max_duration_s < 0
    ):
        raise ValueError("max_duration_s must be a non-negative integer or null")
    if not isinstance(request.metadata, dict):
        raise ValueError("metadata must be an object")

    normalized_metadata = normalize_json_ingress(request.metadata)
    assert isinstance(normalized_metadata, dict)
    subagent_depth = normalized_metadata.get("subagent_depth", 0)
    if type(subagent_depth) is not int or subagent_depth < 0:
        raise ValueError("metadata.subagent_depth must be a non-negative integer")
    normalized_definition = _normalize_agent_definition(request.agent_definition)
    normalized_config = (
        normalize_runtime_config(request.runtime_config)
        if request.runtime_config is not None
        else None
    )
    return BackendRunRequest(
        tenant_id=_required_text(request.tenant_id, "tenant_id"),
        user_id=_required_text(request.user_id, "user_id"),
        workspace_root=Path(normalize_unicode_scalars(str(request.workspace_root))),
        instruction=_required_text(request.instruction, "instruction"),
        input_parts=tuple(normalize_content_part(part) for part in request.input_parts),
        mode=_required_text(request.mode, "mode"),  # type: ignore[arg-type]
        workspace_backend=_required_text(  # type: ignore[arg-type]
            request.workspace_backend,
            "workspace_backend",
        ),
        max_steps=_nonnegative_integer(request.max_steps, "max_steps"),
        max_tool_calls=_nonnegative_integer(request.max_tool_calls, "max_tool_calls"),
        max_bytes_read=_nonnegative_integer(request.max_bytes_read, "max_bytes_read"),
        max_duration_s=request.max_duration_s,
        permission_policy=_normalize_permission_policy(request.permission_policy),
        agent_definition=normalized_definition,
        runtime_config=normalized_config,
        multi_turn=request.multi_turn,
        metadata=normalized_metadata,
    )


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
    interruption_cause: InterruptionCause | None = None
    # The failure classification the last observed park carried — one vocabulary, all five,
    # because ``config_recoverable`` alone cannot separate an ``insufficient_quota`` (fix the
    # config) from a ``rate_limit`` (wait). Read off the Suspension by the session driver at
    # every park (assigned, never or-ed, so a clean settle clears a stale answer) and off the
    # ``turn.failed`` event by ``record_event`` for stream-driven runs; it does NOT change
    # control flow — a single-shot run whose turn fails is still terminal — it only lets
    # status()/result() say what the park already knew, which is the whole reason the
    # classification exists. Spellings follow the Suspension/event vocabulary (``http_status``,
    # not the checkpoint's ``provider_http_status`` alias).
    config_recoverable: bool = False
    retryable: bool = False
    http_status: int | None = None
    provider_error_code: str = ""
    # Per-call fact (attempts the adapter already made inside one call). Carried while parked;
    # the terminal vocabulary deliberately drops it, exactly as ``run.failed`` does.
    provider_retried: bool = False
    result: AgentRunResult | None = None
    # Latest settled turn's validated output (AgentTurnResult.final_output), captured per park so a
    # live multi-turn run can expose it via status() before the run closes (result() carries the
    # final one). Process-local — not persisted. None when no output validator produced a value.
    last_final_output: Any = None
    last_event_seq: int = 0
    last_event_type: str = ""
    cancellation_token: CancellationToken = field(default_factory=CancellationToken)
    write_authority: ActivationWriteAuthority = field(default_factory=ActivationWriteAuthority)
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
    outbox_sender: OutboxSender | None = field(default=None, repr=False)
