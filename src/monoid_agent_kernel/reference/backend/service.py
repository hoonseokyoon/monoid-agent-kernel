from __future__ import annotations

import asyncio
import atexit
import json
import logging
import random
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import thread as _cf_thread
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

from monoid_agent_kernel.core.agents import (
    AgentDefinition,
    AgentRuntimeConfig,
    RuntimeConfigProvider,
    SubagentDefinition,
    validate_runtime_config,
)
from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.control import ControlCommand, ControlResult
from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.inbox import InboxMessage, is_inbox_envelope
from monoid_agent_kernel.core.outbox import OutboxReceipt
from monoid_agent_kernel.core.trace_context import new_traceparent, trace_id_of
from monoid_agent_kernel.core.lifecycle import (
    LoopSession,
    SessionState,
    state_from_suspension,
    to_session_state,
)
from monoid_agent_kernel.core.packages import (
    apply_package,
    create_approval,
    export_package,
    write_apply_result,
    write_approval,
)
from monoid_agent_kernel.core._util import read_text_resilient, write_json_atomic
from monoid_agent_kernel.core.checkpoint import (
    CheckpointRecord,
    CheckpointStore,
    LocalFsCheckpointStore,
)
from monoid_agent_kernel.reference.stores.lease import LeaseStore, LocalFsLeaseStore
from monoid_agent_kernel.core.proposal_file import ProposalFileError, read_proposal_file_payload
from monoid_agent_kernel.core.result import AgentRunResult, Suspension
from monoid_agent_kernel.core.content import (
    ContentPart,
    content_part_from_json,
    content_part_to_json,
)
from monoid_agent_kernel.core.media import normalize_inline_media_dicts
from monoid_agent_kernel.core.spec import (
    AgentRunSpec,
    ModelConfig,
    ModelRetryConfig,
    RunLimits,
    RunMode,
    WorkspaceBackendKind,
)
from monoid_agent_kernel.core.workspace import Workspace
from monoid_agent_kernel.errors import NativeAgentError, PermissionDenied
from monoid_agent_kernel.tasks import (
    get_job_artifact,
    list_job_artifacts,
    read_job_log_text,
    request_job_cancel,
)
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.providers.base import ModelAdapter
from monoid_agent_kernel.providers.gateway import GatewayModelAdapter
from monoid_agent_kernel.identifiers import (
    BACKEND_AUDIENCE,
    BACKEND_AUDIENCES,
    TASK_CALLBACK_AUDIENCE,
    TASK_CALLBACK_AUDIENCES,
    accepted_namespaced_ids,
    namespaced_id,
)
from monoid_agent_kernel.reference._shared.tokens import TokenError, TokenKind, TokenManager
from monoid_agent_kernel.recorder import append_event_to_run
from monoid_agent_kernel.tools.builtin import agent_spawn_tool, builtin_tools
from monoid_agent_kernel.web import WebGatewayClient
from monoid_agent_kernel.workspace.paths import is_within

BackendRunState = Literal["queued", "running", "awaiting_input", "completed", "failed", "limited"]

# Sentinels enqueued to wake/stop a session worker blocked on its message queue.
_CLOSE_SESSION = object()
# Wakes a paused worker: resume the SAME turn with no new input. Ignored (a no-op) by the other
# queue-waiting branches, which expect a real user message or _CLOSE_SESSION.
_RESUME_SESSION = object()
ModelAdapterFactory = Callable[[AgentRunSpec, str], ModelAdapter]

# A run-artifact fetch handle is a bare sha256 hex digest — validated before any store lookup so a
# crafted value can never reach the blob layer as a path.
_ARTIFACT_DIGEST_RE = re.compile(r"^[a-f0-9]{64}$")


# --- multimodal message normalization (the backend message queue stays JSON-native) ---------
# A backend message is either a ``str`` (text) or a ``list[dict]`` of content-part dicts (a
# multimodal message carried by-reference). Keeping the queue + checkpoint JSON-native means a
# parked multimodal message survives a restart with no dataclass (de)serialization in the driver.


def _normalize_inbound_message(content: str | Sequence[Any]) -> str | list[dict[str, Any]]:
    """Normalize a ``send_message`` argument into the queue's JSON-native wire form. Accepts a
    plain ``str``, a sequence of ``ContentPart`` dataclasses, or a sequence of already-serialized
    content-part dicts (the HTTP boundary sends the latter)."""
    if isinstance(content, str):
        return content
    parts: list[dict[str, Any]] = []
    for item in content:
        parts.append(item if isinstance(item, dict) else content_part_to_json(item))
    if not parts:
        raise ValueError("message has no content")
    return parts


def _queued_message_to_loop_input(message: Any) -> str | tuple[ContentPart, ...]:
    """Convert a dequeued backend message into a loop ``submit`` input. Unwraps an inbox envelope to
    its ``content`` (the provenance/id stay on the queue/checkpoint, not in the loop); a legacy raw
    ``str``/``list[dict]`` entry passes through. ``content_part_from_json`` rebuilds typed parts."""
    if is_inbox_envelope(message):
        message = InboxMessage.from_json(message).content
    if isinstance(message, list):
        return tuple(content_part_from_json(part) for part in message)
    return message  # str

# Durable recovery descriptor (run.json) — what recover_runs needs to rebuild a parked run.
_RUN_META_SCHEMA_VERSION = namespaced_id("backend-run.v1")
_ACCEPTED_RUN_META_SCHEMA_VERSIONS = accepted_namespaced_ids("backend-run.v1")

_LOGGER = logging.getLogger("monoid_agent_kernel.backend")


def _read_run_meta(run_dir: Path) -> dict[str, Any] | None:
    """Read run.json if present and schema-compatible; ``None`` otherwise (never raises)."""
    try:
        payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return None
    return _validate_run_meta(payload)


def _validate_run_meta(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or payload.get("schema_version") not in _ACCEPTED_RUN_META_SCHEMA_VERSIONS:
        return None
    return dict(payload)


def _runtime_config_from_meta(meta: Mapping[str, Any]) -> AgentRuntimeConfig:
    config_payload = meta.get("runtime_config")
    if not isinstance(config_payload, dict):
        raise ValueError("run metadata is missing runtime_config")
    config = AgentRuntimeConfig.from_json(config_payload)
    expected_hash = str(meta.get("runtime_config_hash") or config_payload.get("config_hash") or "")
    if expected_hash and expected_hash != config.config_hash:
        raise ValueError("runtime config hash mismatch in run metadata")
    return config


def _read_event_page(events_path: Path, *, from_seq: int, limit: int | None) -> dict[str, Any]:
    if from_seq < 0:
        raise ValueError("from_seq must be non-negative")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    events: list[dict[str, Any]] = []
    next_seq = from_seq
    has_more = False
    if events_path.exists():
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                seq = int(event.get("seq") or 0)
                if seq < from_seq:
                    continue
                if limit is not None and len(events) >= limit:
                    has_more = True
                    break
                events.append(event)
                next_seq = seq + 1
    return {"events": events, "next_seq": next_seq, "has_more": has_more}


_DIAGNOSTIC_EVENT_DATA_KEYS = {
    "attempts",
    "actor",
    "binding_id",
    "call_id",
    "capability",
    "child_run_id",
    "command",
    "command_id",
    "error",
    "error_code",
    "failure_code",
    "idempotency_key",
    "job_id",
    "reason",
    "request_id",
    "result_code",
    "run_id",
    "state",
    "status",
    "target_run_id",
    "task_id",
    "tool",
    "traceparent",
}


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _diagnostic_event_summary(event: dict[str, Any]) -> dict[str, Any]:
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    return {
        "seq": event.get("seq"),
        "type": event.get("type"),
        "timestamp": event.get("timestamp"),
        "level": event.get("level"),
        "turn_id": event.get("turn_id"),
        "parent_id": event.get("parent_id"),
        "data": {key: data[key] for key in sorted(_DIAGNOSTIC_EVENT_DATA_KEYS) if key in data},
    }


_DIRECT_AUDIT_APPEND_STATUSES = {"completed", "failed", "limited", "ended"}


def _run_dir_allows_direct_audit_append(run_dir: Path) -> bool:
    status = _read_optional_json(run_dir / "status.json")
    if status is None:
        return False
    return str(status.get("status") or "") in _DIRECT_AUDIT_APPEND_STATUSES


def _trace_ids_from_events(events: list[dict[str, Any]]) -> list[str]:
    trace_ids: set[str] = set()
    for event in events:
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        trace_id = trace_id_of(str(data.get("traceparent") or ""))
        if trace_id:
            trace_ids.add(trace_id)
    return sorted(trace_ids)


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
    status: BackendRunState
    run_dir: Path
    status_url: str
    result_url: str
    events_url: str
    proposal_url: str

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_token": self.run_token,
            "status": self.status,
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


def _json_safe(value: Any) -> Any:
    """Render an output validator's value safe for a JSON wire projection **at any nesting depth**.
    Round-trips through json with a ``repr`` fallback so a non-serializable value — a Pydantic
    model, dataclass, or bytes, even nested inside a dict/list — can never 500 the HTTP response
    when the edge later ``json.dumps`` it."""
    try:
        return json.loads(json.dumps(value, default=repr))
    except (TypeError, ValueError):
        return repr(value)


@dataclass
class BackendRunRecord:
    run_id: str
    tenant_id: str
    user_id: str
    workspace_root: Path
    run_dir: Path
    status: BackendRunState
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
    session_state: SessionState = SessionState.CREATED
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


@dataclass
class TenantUsage:
    tenant_id: str
    runs: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    web_search_calls: int = 0
    web_fetch_calls: int = 0
    web_context_calls: int = 0
    web_failed_calls: int = 0
    web_result_count: int = 0
    web_bytes_returned: int = 0
    web_context_source_count: int = 0
    web_context_bytes_returned: int = 0

    def add_metrics(self, metrics: dict[str, Any]) -> None:
        self.runs += 1
        self.input_tokens += int(metrics.get("input_tokens") or 0)
        self.output_tokens += int(metrics.get("output_tokens") or 0)
        self.total_tokens += int(metrics.get("total_tokens") or 0)
        self.web_search_calls += int(metrics.get("web_search_calls") or 0)
        self.web_fetch_calls += int(metrics.get("web_fetch_calls") or 0)
        self.web_context_calls += int(metrics.get("web_context_calls") or 0)
        self.web_failed_calls += int(metrics.get("web_failed_calls") or 0)
        self.web_result_count += int(metrics.get("web_result_count") or 0)
        self.web_bytes_returned += int(metrics.get("web_bytes_returned") or 0)
        self.web_context_source_count += int(metrics.get("web_context_source_count") or 0)
        self.web_context_bytes_returned += int(metrics.get("web_context_bytes_returned") or 0)

    def to_json(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "runs": self.runs,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "web_search_calls": self.web_search_calls,
            "web_fetch_calls": self.web_fetch_calls,
            "web_context_calls": self.web_context_calls,
            "web_failed_calls": self.web_failed_calls,
            "web_result_count": self.web_result_count,
            "web_bytes_returned": self.web_bytes_returned,
            "web_context_source_count": self.web_context_source_count,
            "web_context_bytes_returned": self.web_context_bytes_returned,
        }


class BackendRunStateSink:
    def __init__(self, backend: RunnerBackend, run_id: str) -> None:
        self._backend = backend
        self._run_id = run_id

    def emit(self, event: AgentEvent) -> None:
        self._backend.record_event(self._run_id, event)

    def close(self) -> None:
        return None


class BackendRuntimeConfigProvider(RuntimeConfigProvider):
    def __init__(self, backend: RunnerBackend, run_id: str) -> None:
        self._backend = backend
        self._run_id = run_id

    def current_config(self, run_id: str) -> AgentRuntimeConfig | None:
        del run_id
        return self._backend.current_runtime_config(self._run_id)


def _backend_builtin_tool_specs(
    subagent_definitions: Mapping[str, SubagentDefinition] | None = None,
    tool_providers: Sequence[Any] = (),
) -> tuple[Any, ...]:
    specs = list(builtin_tools(cast(Workspace, None)))
    # agent.spawn is registered dynamically by the loop bootstrap (only when the run carries
    # subagent_definitions), so config validation must know about it too when they're present —
    # otherwise a binding to agent.spawn looks like an unknown tool.
    if subagent_definitions:
        catalog = {sid: d.description for sid, d in subagent_definitions.items()}
        specs.append(agent_spawn_tool(catalog))
    # Provider tools (skill, skill.read_file, mcp.<server>.<tool>, …) are likewise registered by
    # the loop bootstrap from tool_providers, so validation must know them too or a binding to a
    # provider tool looks unknown (the DX-10/agent_spawn precedent). get_tools() is cheap here:
    # SkillProvider does no I/O, and McpToolProvider caches its discovery after the first call.
    for provider in tool_providers:
        specs.extend(provider.get_tools())
    return tuple(specs)


def _runtime_config_uses_web(config: AgentRuntimeConfig) -> bool:
    return any(binding.ref.tool_id.startswith("web.") for binding in config.tools)


class _DaemonDetachedExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor whose worker threads are excluded from CPython's
    interpreter-shutdown forced join.

    ``concurrent.futures`` registers an ``atexit`` hook (run inside
    ``threading._shutdown()``, BEFORE our ``atexit`` handlers) that joins EVERY
    executor worker unconditionally — daemon flag notwithstanding. A worker stuck in
    a long offloaded call (here, ``asyncio.to_thread(loop.wait_for_pending_tasks)``
    for a multi-turn/recovered run that is parked awaiting a hosted-task result that
    never arrives) therefore stalls process exit for up to ``task_wait_poll_s`` per
    such worker — the "tests pass fast but the process takes minutes to exit" hang.

    Spawned from the daemon run-loop thread, these workers are already daemon (they
    inherit it), so the OS reclaims them at exit. We only drop them from the global
    join registry so a blocked offload can never gate shutdown. New work is still
    refused after ``shutdown()``; this changes nothing about in-process behavior."""

    def _adjust_thread_count(self) -> None:  # type: ignore[override]
        before = set(self._threads)
        super()._adjust_thread_count()
        for worker in self._threads - before:
            _cf_thread._threads_queues.pop(worker, None)


def _teardown_loop(
    loop: asyncio.AbstractEventLoop,
    thread: threading.Thread,
    executor: ThreadPoolExecutor,
) -> None:
    """Stop the shared run loop, join its thread, and shut down its executor. Idempotent.
    Idle executor workers exit promptly; a worker blocked in a long offloaded wait (e.g. an
    idle multi-turn message-get) exits when that call returns."""
    if loop.is_closed():
        return
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    executor.shutdown(wait=False)
    loop.close()


# Process-wide shared run loop. One loop + one executor for the whole process, shared by
# every RunnerBackend, so creating many short-lived backends (e.g. tests) never leaks
# loop/executor threads — the previous per-backend loop accumulated threads and starved the
# scheduler. Started lazily on a daemon thread; cleaned up at process exit via atexit.
_shared_loop_lock = threading.Lock()
_shared_loop: asyncio.AbstractEventLoop | None = None


def _get_shared_loop() -> asyncio.AbstractEventLoop:
    global _shared_loop
    with _shared_loop_lock:
        if _shared_loop is None or _shared_loop.is_closed():
            loop = asyncio.new_event_loop()
            executor = _DaemonDetachedExecutor(max_workers=32, thread_name_prefix="nar-backend-io")
            loop.set_default_executor(executor)
            thread = threading.Thread(target=loop.run_forever, name="nar-backend-loop", daemon=True)
            thread.start()
            atexit.register(_teardown_loop, loop, thread, executor)
            _shared_loop = loop
        return _shared_loop


async def _async_sleep_before_retry(attempt: int, retry: ModelRetryConfig) -> None:
    """Awaitable, cancellable exponential backoff with jitter — the async counterpart of the
    gateway's sync ``_sleep_before_retry`` (used between turn-level auto-retries on the shared
    loop). ``attempt`` is 1-based."""
    delay = min(retry.max_delay_s, retry.initial_delay_s * (retry.backoff_multiplier ** max(0, attempt - 1)))
    if retry.jitter_s > 0:
        delay += random.uniform(0, retry.jitter_s)
    if delay > 0:
        await asyncio.sleep(delay)


@dataclass
class _GatewayTokenSource:
    """A callable gateway-token source that re-mints shortly before expiry. Resolved per request by
    the model adapter (``GatewayModelAdapter.token_provider``), so a run that outlives the token TTL
    stays authenticated without a restart — the same re-mint the recovery path already performs,
    applied proactively and in-process (the backend holds the signing key). Not thread-safe by design:
    a run's model calls are serialized on its loop."""

    token_manager: TokenManager
    kind: TokenKind
    audience: str
    run_id: str
    tenant_id: str
    user_id: str
    ttl_s: int
    metadata: dict[str, Any] = field(default_factory=dict)
    refresh_skew_s: int = 300
    _token: str = ""
    _expires_at: float = 0.0

    def __call__(self) -> str:
        now = time.time()
        # Re-mint near expiry; cap the skew at half the TTL so a short TTL doesn't re-mint every call.
        skew = min(self.refresh_skew_s, self.ttl_s // 2)
        if not self._token or now >= self._expires_at - skew:
            self._token = self.token_manager.issue(
                kind=self.kind,
                audience=self.audience,
                run_id=self.run_id,
                tenant_id=self.tenant_id,
                user_id=self.user_id,
                ttl_s=self.ttl_s,
                metadata=dict(self.metadata),
            )
            self._expires_at = now + self.ttl_s
        return self._token


@dataclass
class RunnerBackend:
    run_root: Path
    token_manager: TokenManager
    allowed_workspace_roots: tuple[Path, ...]
    llm_gateway_url: str
    model_adapter_factory: ModelAdapterFactory | None = None
    web_gateway_url: str | None = None
    allowed_apply_roots: tuple[Path, ...] = ()
    run_token_ttl_s: int = 3600
    llm_gateway_token_ttl_s: int = 3600
    web_gateway_token_ttl_s: int = 3600
    task_callback_token_ttl_s: int = 3600
    # Multi-turn session limits.
    idle_timeout_s: float = 300.0
    max_session_lifetime_s: float = 1800.0
    max_turns: int = 50
    task_wait_poll_s: float = 5.0
    # A recoverable model-turn failure (turn_failed) keeps the session alive: a transient one
    # is auto-retried with backoff; a config/auth 4xx parks for the user to fix + resend. A run
    # that takes this many CONSECUTIVE failed turns (no settle between) is given up as failed.
    # Only auto-retries count toward this cap; user-initiated resends are bounded by max_turns.
    max_consecutive_turn_failures: int = 5
    turn_retry: ModelRetryConfig = field(default_factory=ModelRetryConfig)
    # Opt-in token streaming for the autonomous drive: when set, runs emit model.output.delta
    # events (for adapters that support astream_turn) so an event-stream consumer renders tokens
    # live. Off by default; a UI-facing embedder (e.g. studio) turns it on.
    emit_output_deltas: bool = False
    # Agent-as-tool delegation: subagent id -> definition. When non-empty, runs can bind
    # agent.spawn (the loop bootstrap registers it). Child runs write to run_root/<child_id>/.
    subagent_definitions: Mapping[str, SubagentDefinition] = field(default_factory=dict)
    # Per-run factories for extra event sinks, appended to every run's sinks (besides the
    # backend's own state sink). A FACTORY (not a shared instance) so each run gets its own sink —
    # required for stateful sinks like OtelEventSink (per-run span state). The seam an embedder
    # uses to attach observability without a core dep — e.g. studio sets ``(OtelEventSink,)`` when
    # OTel is toggled on. Read at loop-build time so it can change at runtime. Empty → no deps.
    extra_event_sink_factories: tuple[Any, ...] = ()
    # Tool/context providers attached to every run the backend builds (Skills, MCP, custom).
    # The embedder-facing seam for the loop's tool_providers/context_providers (the CLI passes
    # these to AgentLoop directly; without these fields an out-of-process embedder could not
    # attach a provider at all). INSTANCES, not factories (unlike extra_event_sink_factories):
    # a provider holds a shared, reusable resource (MCP's live httpx client + discovery cache)
    # or is immutable (SkillProvider) — both are safe to share across concurrent runs (the MCP
    # client is documented thread-safe; SkillProvider is read-only). Read at loop-build time so
    # a parked run re-attaches them on resume/restart. Their tools must also be declared to
    # config validation — see _backend_builtin_tool_specs. Empty → no providers.
    tool_providers: tuple[Any, ...] = ()
    context_providers: tuple[Any, ...] = ()
    # Output validators attached to every run the backend builds. Default-on: each runs unless a
    # run's config disables it via OutputValidatorBinding(enabled=False). Read at loop-build time
    # so a parked run re-attaches them on resume/restart, exactly like tool/context providers.
    # Empty → no validators.
    output_validators: tuple[Any, ...] = ()
    # Per-run capability broker factory: ``(request) -> CapabilityBroker | None``. Called at
    # loop-build time so a broker can be scoped to the run's identity (tenant/user/run id) — e.g.
    # a GatewayCapabilityBroker minting per-tenant tokens. None (or a None return) leaves capability
    # gating off for that run. A factory (not an instance) because a broker is typically per-run
    # identity-bound, unlike the shared tool/context providers above.
    capability_broker_factory: Callable[[BackendRunRequest], Any] | None = None
    # Per-run outbox sender (drains staged outbound sends at the edge — see core/outbox.py). A
    # factory like capability_broker_factory; None (or a None return) leaves staged requests pending
    # (durable, never dispatched). The drain performs the actual IO; the core only stages.
    outbox_sender_factory: Callable[[BackendRunRequest], Any] | None = None
    # An outbox request is redispatched (at-least-once + idempotency_key) at most this many times on
    # a retryable failure before it is dead-lettered as failed.
    outbox_max_attempts: int = 5
    # Retry schedule for a failed outbox send: capped exponential backoff with **full jitter**
    # (delay = uniform(0, min(cap, base * factor**attempts))). The next-attempt time is stamped on
    # the request (durable), so the schedule survives a restart; the watchdog redrive tick (below)
    # dispatches a request once its time arrives, decoupling retry timing from run activity.
    outbox_retry_base_s: float = 1.0
    outbox_retry_factor: float = 2.0
    outbox_retry_cap_s: float = 300.0
    # A run whose checkpoint cannot be resumed is retried at most this many times across
    # restarts before being marked unrecoverable (a durable failure.json), so a poison
    # checkpoint never drives an unbounded restart/crash loop.
    max_recover_attempts: int = 3
    # Active watchdog (opt-in via start_watchdog). A worker heartbeats a lease.json for each
    # of its live runs; the watchdog reclaims runs whose lease has gone stale (the owning
    # worker crashed). lease_ttl_s must comfortably exceed watchdog_interval_s so a healthy
    # worker's own lease never looks stale between ticks.
    lease_ttl_s: float = 30.0
    watchdog_interval_s: float = 5.0
    # Resource bounds. A follow-up message larger than max_message_bytes is rejected, and a
    # run's pending-message queue is capped at max_message_queue_depth. max_concurrent_runs
    # caps how many runs do real work at once (0 = unbounded); excess runs stay ``queued``
    # until a slot frees, bounding CPU / memory / gateway load under a submission burst.
    max_message_bytes: int = 1_000_000
    max_message_queue_depth: int = 100
    max_concurrent_runs: int = 0
    # How checkpoints are durably stored (backend owns HOW). Defaults to a local-fs
    # store under run_root; swap for a mounted-volume path or an object-store/DB store.
    checkpoint_store: CheckpointStore | None = None
    # How run-ownership leases are stored/claimed. Defaults to local lease.json files; a
    # shared store (SqliteLeaseStore over the same db as the checkpoint store) lets a worker
    # on another process/host reclaim a crashed peer's run.
    lease_store: LeaseStore | None = None
    _records: dict[str, BackendRunRecord] = field(default_factory=dict, init=False, repr=False)
    _usage: dict[str, TenantUsage] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _worker_id: str = field(default="", init=False, repr=False)
    _watchdog_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _watchdog_stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _run_semaphore: asyncio.BoundedSemaphore | None = field(default=None, init=False, repr=False)
    # RNG for the outbox backoff jitter — a dedicated instance so a test can seed it deterministically
    # (backend._outbox_rng.seed(...)) without perturbing global random state.
    _outbox_rng: random.Random = field(default_factory=random.Random, init=False, repr=False)

    def __post_init__(self) -> None:
        self._worker_id = uuid.uuid4().hex
        if self.max_concurrent_runs > 0:
            # Bound on the shared run loop; constructed without a running loop (3.10+ binds
            # lazily) and acquired/released inside the run coroutines.
            self._run_semaphore = asyncio.BoundedSemaphore(self.max_concurrent_runs)
        self.run_root = self.run_root.resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        roots = tuple(root.resolve() for root in self.allowed_workspace_roots)
        if not roots:
            raise ValueError("at least one allowed workspace root is required")
        self.allowed_workspace_roots = roots
        self.allowed_apply_roots = tuple(root.resolve() for root in self.allowed_apply_roots)
        if self.checkpoint_store is None:
            self.checkpoint_store = LocalFsCheckpointStore(self.run_root)
        if self.lease_store is None:
            self.lease_store = LocalFsLeaseStore(self.run_root)

    # --- Shared event loop (coroutine-per-run) ------------------------------------------

    def _spawn(self, coro: Any) -> Any:
        """Schedule a coroutine on the process-shared run loop from any (sync) thread;
        returns a concurrent.futures.Future."""
        return asyncio.run_coroutine_threadsafe(coro, _get_shared_loop())

    def _call_soon(self, fn: Callable[..., Any], *args: Any) -> None:
        """Run a thread-safe callback on the process-shared run loop (fire-and-forget)."""
        _get_shared_loop().call_soon_threadsafe(fn, *args)

    def spawn_coroutine(self, coro: Any) -> Any:
        """Schedule a coroutine on the process-shared run loop; returns a
        ``concurrent.futures.Future``. The transport seam the SSE bridge uses to drive
        ``astream_run`` on the shared loop from a sync handler thread."""
        return self._spawn(coro)

    def request_stream_cancel(self, run_id: str) -> None:
        """Cooperatively cancel a stream-driven run (internal; the SSE handler calls this on
        client disconnect or backpressure). No auth — the caller created the run via
        ``astream_run``. A missing record (already drained) is a no-op."""
        with self._lock:
            record = self._records.get(run_id)
        if record is not None:
            record.cancellation_token.cancel()

    def shutdown(self, *, drain: bool = False, drain_timeout_s: float = 5.0) -> None:
        """Stop this backend's watchdog. The run loop is process-shared (one per process,
        cleaned up at exit via atexit), so there is nothing per-backend to tear down — and
        stopping it here would break other backends in the process. Idempotent.

        Pass ``drain=True`` to first cooperatively end this backend's own runs (see
        :meth:`drain`), so an embedder that stops a backend mid-session leaves no parked
        session coroutines on the shared loop (otherwise teardown logs "Task was destroyed
        but it is pending")."""
        if drain:
            self.drain(timeout_s=drain_timeout_s)
        self.stop_watchdog()

    def drain(self, *, timeout_s: float = 5.0) -> list[str]:
        """Cooperatively end every non-terminal run this backend owns: cancel it and wake any
        session parked on its message queue, then wait (bounded by ``timeout_s``) for each to
        reach a terminal state.

        Returns the run ids still non-terminal when the timeout elapsed (empty on a clean
        drain). Idempotent; safe to call before :meth:`shutdown`. This is the one-call
        counterpart to issuing a ``cancel_run`` per run and sleeping."""
        terminal = {"completed", "failed", "limited"}
        with self._lock:
            records = [record for record in self._records.values() if record.status not in terminal]
        for record in records:
            with self._lock:
                record.cancellation_token.cancel()
                if not record.error_code:
                    record.error = "run drained on shutdown"
                    record.error_code = "cancelled"
            # Wake a session parked on its message queue (put runs on the shared loop).
            self._call_soon(record.message_queue.put_nowait, _CLOSE_SESSION)
        deadline = time.time() + timeout_s
        pending: list[str] = []
        for record in records:
            while time.time() < deadline:
                if record.status in terminal:
                    break
                time.sleep(0.02)
            else:
                pending.append(record.run_id)
        return pending

    def submit_run(self, request: BackendRunRequest) -> BackendRunSubmission:
        prepared = self._prepare_run_record(request)
        # Run executes as a coroutine on the shared loop (coroutine-per-run), not a thread.
        self._spawn(
            self._run_run(
                prepared.run_id,
                request,
                prepared.workspace_root,
                prepared.llm_gateway_token,
                prepared.web_gateway_token,
            )
        )
        return self._submission_for(prepared)

    def _prepare_run_record(self, request: BackendRunRequest) -> _PreparedRun:
        """Validate the request, mint the three run tokens, and store a queued run record.

        Shared by ``submit_run`` (autonomous drive) and ``astream_run`` (stream-driven). Stops
        at "record stored under lock" — the caller owns how the run is then driven."""
        self._validate_request(request)
        workspace_root = request.workspace_root.resolve()
        self._check_workspace_allowed(workspace_root)
        run_id = uuid.uuid4().hex
        run_dir = self.run_root / run_id
        run_token = self.token_manager.issue(
            kind="run_access",
            audience=BACKEND_AUDIENCE,
            run_id=run_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            ttl_s=self.run_token_ttl_s,
        )
        tool_specs = _backend_builtin_tool_specs(self.subagent_definitions, self.tool_providers)
        initial_runtime_config = request.runtime_config
        runtime_config_issuer = "submit_run"
        runtime_config_reason = "initial runtime config"
        if initial_runtime_config is None and request.agent_definition is not None:
            initial_runtime_config = AgentRuntimeConfig.from_definition(request.agent_definition)
            runtime_config_reason = "initial agent definition"
        elif initial_runtime_config is None:
            raise ValueError("agent_definition or runtime_config is required")
        validate_runtime_config(initial_runtime_config, tool_specs)
        llm_gateway_token = self.token_manager.issue(
            kind="llm_gateway",
            audience="csp.llm-gateway",
            run_id=run_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            ttl_s=self.llm_gateway_token_ttl_s,
            metadata={"agent_config_hash": initial_runtime_config.config_hash},
        )
        web_gateway_token = ""
        if _runtime_config_uses_web(initial_runtime_config):
            if not self.web_gateway_url:
                raise ValueError("web_gateway_url is required when runtime config binds web tools")
            web_gateway_token = self.token_manager.issue(
                kind="web_gateway",
                audience="csp.web-gateway",
                run_id=run_id,
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                ttl_s=self.web_gateway_token_ttl_s,
                metadata={"agent_config_hash": initial_runtime_config.config_hash},
            )
        created_at = time.time()
        record = BackendRunRecord(
            run_id=run_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            workspace_root=workspace_root,
            run_dir=run_dir,
            status="queued",
            created_at=created_at,
            run_token_sha256=TokenManager.token_sha256(run_token),
            llm_gateway_token_sha256=TokenManager.token_sha256(llm_gateway_token),
            web_gateway_token_sha256=TokenManager.token_sha256(web_gateway_token) if web_gateway_token else "",
            runtime_config=initial_runtime_config,
            runtime_config_issuer=runtime_config_issuer,
            runtime_config_reason=runtime_config_reason,
            runtime_config_committed_at=created_at,
        )
        self._write_run_meta(record, request)
        with self._lock:
            self._records[run_id] = record
        return _PreparedRun(
            run_id=run_id,
            record=record,
            workspace_root=workspace_root,
            run_token=run_token,
            llm_gateway_token=llm_gateway_token,
            web_gateway_token=web_gateway_token,
        )

    def _submission_for(self, prepared: _PreparedRun) -> BackendRunSubmission:
        run_id = prepared.run_id
        return BackendRunSubmission(
            run_id=run_id,
            run_token=prepared.run_token,
            status="queued",
            run_dir=prepared.record.run_dir,
            status_url=f"/v1/runs/{run_id}/status",
            result_url=f"/v1/runs/{run_id}/result",
            events_url=f"/v1/runs/{run_id}/events",
            proposal_url=f"/v1/runs/{run_id}/proposal",
        )

    def _run_spec_for_request(
        self,
        run_id: str,
        request: BackendRunRequest,
        workspace_root: Path,
    ) -> AgentRunSpec:
        return AgentRunSpec(
            workspace_root=workspace_root,
            run_root=self.run_root,
            run_id=run_id,
            mode=request.mode,
            workspace_backend=request.workspace_backend,
            limits=RunLimits(
                max_steps=request.max_steps,
                max_tool_calls=request.max_tool_calls,
                max_bytes_read=request.max_bytes_read,
                max_duration_s=request.max_duration_s,
            ),
            permission_policy=request.permission_policy,
            metadata={
                **request.metadata,
                "tenant_id": request.tenant_id,
                "user_id": request.user_id,
            },
        )

    def status(self, run_id: str, token: str) -> dict[str, Any]:
        run_dir = self._authorized_run_dir(run_id, token)
        status_file = run_dir / "status.json"
        status_payload: dict[str, Any] | None = None
        if status_file.exists():
            # Resilient read: the run is concurrently flipping status.json via an atomic
            # os.replace; on Windows a plain read mid-replace hits a sharing violation. Mirror
            # the writer's retry (write_json_atomic / _atomic_replace) so the poll closes the race.
            status_payload = json.loads(read_text_resilient(status_file))
        with self._lock:
            record = self._records.get(run_id)
        if record is None:
            # Historical run (no live record, e.g. after a restart): report from status.json.
            return {
                "run_id": run_id,
                "status": (status_payload or {}).get("status", "ended"),
                "run_dir": str(run_dir),
                "status_file": status_payload,
            }
        return {
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            "user_id": record.user_id,
            "status": record.status,
            "created_at": record.created_at,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "run_dir": str(record.run_dir),
            "last_event_seq": record.last_event_seq,
            "last_event_type": record.last_event_type,
            "error": record.error,
            "error_code": record.error_code,
            # Prefer the live park-point capture; fall back to the terminal result for
            # stream-driven runs (astream_run) that record an AgentRunResult without ever
            # passing through _drive_open_session, where last_final_output stays None.
            "final_output": _json_safe(
                record.last_final_output
                if record.last_final_output is not None
                else (record.result.final_output if record.result is not None else None)
            ),
            "status_file": status_payload,
        }

    def result(self, run_id: str, token: str) -> dict[str, Any]:
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        if record.result is None:
            return {
                "run_id": record.run_id,
                "tenant_id": record.tenant_id,
                "status": record.status,
                "ready": False,
                "error": record.error,
                "error_code": record.error_code,
            }
        result = record.result
        diff_text = result.diff_path.read_text(encoding="utf-8") if result.diff_path.exists() else ""
        proposal_payload = self._read_proposal(record)
        return {
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            "status": result.status,
            "ready": True,
            "final_text": result.final_text,
            "final_output": _json_safe(result.final_output),
            "error": result.error,
            "error_code": result.error_code,
            "run_dir": str(result.run_dir),
            "manifest_path": str(result.run_dir / "manifest.json"),
            "diff_path": str(result.diff_path),
            "diff": diff_text,
            "proposal_path": str(result.proposal_path),
            "proposal": proposal_payload,
            "artifacts": [artifact.__dict__ for artifact in result.artifacts],
            "metrics": result.metrics,
        }

    def proposal(self, run_id: str, token: str) -> dict[str, Any]:
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        payload = self._read_proposal(record)
        if payload is None:
            return {
                "run_id": record.run_id,
                "tenant_id": record.tenant_id,
                "status": record.status,
                "ready": False,
                "error": record.error,
            }
        return {
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            "status": record.status,
            "ready": True,
            **payload,
        }

    def proposal_diff(self, run_id: str, token: str) -> dict[str, Any]:
        """The unified diff of the current proposal, on demand (works mid-run, not only at the
        end like ``result()``). Token-scoped so an embedder never reads the run dir off disk.
        Binary files appear as a ``<binary sha256=… size=…>`` marker line in the patch."""
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        diff_path = record.run_dir / "diff.patch"
        diff = diff_path.read_text(encoding="utf-8") if diff_path.exists() else ""
        return {"run_id": run_id, "ready": diff_path.exists(), "diff": diff}

    def cancel_run(self, run_id: str, token: str) -> dict[str, Any]:
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        with self._lock:
            if record.status in {"completed", "failed", "limited"}:
                return {
                    "run_id": record.run_id,
                    "tenant_id": record.tenant_id,
                    "status": record.status,
                    "cancel_requested": False,
                    "error": record.error,
                    "error_code": record.error_code,
                }
            record.cancellation_token.cancel()
            record.error = "run cancellation requested"
            record.error_code = "cancelled"
        # Wake a run parked on its message queue so it stops promptly. The put runs on the
        # shared loop (asyncio.Queue is not thread-safe), scheduled from this thread.
        self._call_soon(record.message_queue.put_nowait, _CLOSE_SESSION)
        return {
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            "status": record.status,
            "cancel_requested": True,
            "error": record.error,
            "error_code": record.error_code,
        }

    def interrupt_turn(self, run_id: str, token: str) -> dict[str, Any]:
        """Turn-level stop (keeps the session alive): signal the loop to halt the current
        turn at its next step boundary; the driver then parks for the next message. Unlike
        :meth:`cancel_run` this does NOT terminalize the run. A no-op on a terminal or
        not-yet-built run. The flag set is a thread-safe one-way signal (like cancel)."""
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        with self._lock:
            loop = record.loop
            terminal = record.status in {"completed", "failed", "limited"}
        requested = not terminal and loop is not None
        if requested:
            loop.interrupt_turn()
        return {
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            "status": record.status,
            "interrupt_requested": requested,
        }

    def pause_run(self, run_id: str, token: str) -> dict[str, Any]:
        """Cooperative pause (keeps the session alive): signal the loop to freeze the current
        turn at its next start-of-step boundary; the driver then parks until a resume wakes it.
        A no-op on a terminal or not-yet-built run. The flag set is a thread-safe one-way signal
        (like cancel/interrupt). Resume via the ``resume`` control command (``signal_resume``)."""
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        with self._lock:
            loop = record.loop
            terminal = record.status in {"completed", "failed", "limited"}
        requested = not terminal and loop is not None
        if requested:
            loop.pause_turn()
        return {
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            "status": record.status,
            "pause_requested": requested,
        }

    def signal_resume(self, run_id: str, token: str) -> dict[str, Any]:
        """Wake a paused *live* run: push a resume signal so the driver re-pumps the frozen turn
        with no new input. For a run not currently in memory (parked after a restart), use
        :meth:`resume_run` (checkpoint recovery) instead — the ``resume`` control command picks
        the right one. A no-op on a terminal run."""
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        with self._lock:
            terminal = record.status in {"completed", "failed", "limited"}
        if terminal:
            return {"run_id": run_id, "status": record.status, "resumed": False}
        # asyncio.Queue is not thread-safe; schedule the put on the shared loop.
        self._call_soon(record.message_queue.put_nowait, _RESUME_SESSION)
        return {"run_id": run_id, "status": record.status, "resumed": True}

    def revoke_capability(
        self,
        run_id: str,
        token: str,
        *,
        capability: str | None = None,
        lease_id: str | None = None,
        before: float | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """Revoke a capability lease on a *live* run (the operator/Daemon kill switch). Mirrors
        :meth:`pause_run`: signal the loop directly (a thread-safe vault mutation). A no-op on a
        terminal or not-in-memory run (revoking a parked-and-evicted run is not supported in v1 —
        load it first). The ``capability.denied`` audit event is emitted on the loop thread when the
        next gated call hits the revocation."""
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        with self._lock:
            loop = record.loop
            terminal = record.status in {"completed", "failed", "limited"}
        summary: dict[str, Any] = {}
        revoked = not terminal and loop is not None
        if revoked:
            summary = loop.revoke_capability(
                capability=capability, lease_id=lease_id, before=before, reason=reason
            )
        return {
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            "status": record.status,
            "revoked": revoked,
            **summary,
        }

    def dispatch(self, command: ControlCommand) -> ControlResult:
        """Reference ``ControlDispatcher``: route one ``ControlCommand`` to the in-process method
        that already backs it. The run token travels in ``command.args['token']`` (the HTTP layer
        injects the bearer token); it is consumed here and never echoed back. Auth failures
        propagate (the HTTP layer maps them to 401); a missing run is reported as ``unsupported``;
        a controlled operation error is reported as ``error`` with its code. Unknown command types
        return ``unsupported`` — the wire vocabulary stays forward-compatible."""
        args = dict(command.args)
        token = str(args.pop("token", "") or "")
        run_id = command.run_id
        ctype = command.type
        command_id = command.command_id or f"control_{uuid.uuid4().hex[:12]}"
        idempotency_key = command.command_id or command_id
        token_sha256 = TokenManager.token_sha256(token) if token else ""
        started = time.time()

        audit_authorized = False
        try:
            self._authorize_control_audit_target(run_id, token, command_type=ctype, args=args)
            audit_authorized = True
            self._emit_control_audit_event(
                run_id,
                "control.command.received",
                {
                    "command_id": command_id,
                    "command": ctype,
                    "target_run_id": run_id,
                    "actor": command.issuer,
                    "reason": command.reason,
                    "token_sha256": token_sha256,
                    "idempotency_key": idempotency_key,
                    "args_keys": sorted(key for key in command.args if key != "token"),
                },
            )
            result = self._dispatch_control_command(
                command,
                args=args,
                token=token,
                command_id=command_id,
            )
        except PermissionDenied as exc:
            if audit_authorized:
                self._emit_control_audit_event(
                    run_id,
                    "control.command.failed",
                    {
                        "command_id": command_id,
                        "command": ctype,
                        "target_run_id": run_id,
                        "actor": command.issuer,
                        "idempotency_key": idempotency_key,
                        "token_sha256": token_sha256,
                        "status": "error",
                        "error": str(exc),
                        "error_code": getattr(exc, "error_code", "permission_denied"),
                        "failure_code": getattr(exc, "error_code", "permission_denied"),
                        "duration_ms": (time.time() - started) * 1000,
                    },
                    level="warning",
                )
            raise  # auth failures map to HTTP 401 in the route layer
        except KeyError as exc:
            result = ControlResult(
                run_id=run_id, type=ctype, status="unsupported", error=str(exc), error_code="run_not_found"
            )
        except (ValueError, NativeAgentError) as exc:
            result = ControlResult(
                run_id=run_id,
                type=ctype,
                status="error",
                error=str(exc),
                error_code=getattr(exc, "error_code", "control_error"),
            )

        duration_ms = (time.time() - started) * 1000
        if result.status == "ok":
            self._emit_control_audit_event(
                run_id,
                "control.command.completed",
                {
                    "command_id": command_id,
                    "command": ctype,
                    "target_run_id": run_id,
                    "actor": command.issuer,
                    "idempotency_key": idempotency_key,
                    "token_sha256": token_sha256,
                    "status": result.status,
                    "result_code": result.error_code or result.status,
                    "state": result.state,
                    "duration_ms": duration_ms,
                },
            )
        else:
            self._emit_control_audit_event(
                run_id,
                "control.command.failed",
                {
                    "command_id": command_id,
                    "command": ctype,
                    "target_run_id": run_id,
                    "actor": command.issuer,
                    "idempotency_key": idempotency_key,
                    "token_sha256": token_sha256,
                    "status": result.status,
                    "error": result.error,
                    "error_code": result.error_code,
                    "failure_code": result.error_code,
                    "duration_ms": duration_ms,
                },
                level="warning",
            )
        return result

    def _dispatch_control_command(
        self,
        command: ControlCommand,
        *,
        args: dict[str, Any],
        token: str,
        command_id: str,
    ) -> ControlResult:
        run_id = command.run_id
        ctype = command.type

        def ok(data: dict[str, Any], *, state: str | None = None) -> ControlResult:
            return ControlResult(run_id=run_id, type=ctype, status="ok", state=state, data=dict(data))

        if ctype == "pause":
            return ok(self.pause_run(run_id, token))
        if ctype == "resume":
            with self._lock:
                live = run_id in self._records
            return ok(self.signal_resume(run_id, token) if live else self.resume_run(run_id, token))
        if ctype == "cancel":
            return ok(self.cancel_run(run_id, token))
        if ctype in {"approve", "deny"}:
            result = args.get("result") if isinstance(args.get("result"), dict) else {}
            approval_result = dict(result)
            if ctype == "approve":
                approval_result["answer"] = str(args.get("answer") or "Approve")
                approval_result["approved"] = True
            else:
                approval_result["answer"] = str(args.get("answer") or "Deny")
                approval_result["approved"] = False
                approval_result["granted"] = False
                approval_result["reason"] = command.reason or str(
                    args.get("reason") or approval_result.get("reason") or "denied"
                )
            return ok(
                self.report_task_result(
                    run_id,
                    token,
                    task_id=str(args.get("task_id") or ""),
                    result=approval_result,
                    status=str(args.get("status") or "answered"),
                )
            )
        if ctype == "interrupt":
            return ok(self.interrupt_turn(run_id, token))
        if ctype in {"inspect", "health"}:
            loop = self._authorize_active_loop(run_id, token)
            # Seed the throwaway facade with the record's authoritative FSM state (the backend
            # drives the loop directly, so the facade's own cache never tracked this run).
            session = LoopSession(loop, _state=self._record(run_id).session_state)
            if ctype == "inspect":
                inspection = session.inspect()
                return ok(inspection.to_json(), state=inspection.state.value)
            health = session.health()
            return ok(health.to_json(), state=health.state.value)
        if ctype == "status":
            return ok(self.status(run_id, token))
        if ctype == "runtime_config":
            return ok(self.runtime_config(run_id, token))
        if ctype == "replace_runtime_config":
            return ok(
                self.replace_runtime_config(
                    run_id,
                    token,
                    expected_version=int(args.get("expected_version", 0)),
                    issuer=command.issuer,
                    reason=command.reason,
                    config=AgentRuntimeConfig.from_json(args["config"]),
                )
            )
        if ctype == "send_message":
            # A control command carries its own id (command_id) — use it as the inbox dedup key
            # so a redelivered control send is processed once.
            return ok(
                self.send_message(
                    run_id,
                    token,
                    content=args.get("content") or "",
                    message_id=command_id,
                    source="control",
                )
            )
        if ctype == "create_task":
            return ok(
                self.create_task(
                    run_id,
                    token,
                    kind=str(args.get("kind") or ""),
                    request=dict(args.get("request") or {}),
                )
            )
        if ctype == "report_task_result":
            return ok(
                self.report_task_result(
                    run_id,
                    token,
                    task_id=str(args.get("task_id") or ""),
                    result=dict(args.get("result") or {}),
                    status=str(args.get("status") or "answered"),
                )
            )
        if ctype == "revoke_capability":
            before = args.get("before")
            return ok(
                self.revoke_capability(
                    run_id,
                    token,
                    capability=(str(args["capability"]) if args.get("capability") else None),
                    lease_id=(str(args["lease_id"]) if args.get("lease_id") else None),
                    before=(float(before) if before is not None else None),
                    reason=command.reason,
                )
            )
        return ControlResult(
            run_id=run_id,
            type=ctype,
            status="unsupported",
            error=f"unknown control command type: {ctype}",
            error_code="unknown_control_command",
        )

    def _emit_control_audit_event(
        self,
        run_id: str,
        event_type: str,
        data: dict[str, Any],
        *,
        level: str = "info",
    ) -> None:
        if any(sep in run_id for sep in ("/", "\\")) or ".." in run_id:
            return
        with self._lock:
            record = self._records.get(run_id)
            loop = record.loop if record is not None else None
            run_dir = record.run_dir if record is not None else self.run_root / run_id
        if record is not None:
            if loop is not None:
                loop.emit_external_event(event_type, data=data, level=level)
            return
        if not run_dir.exists():
            return
        if not _run_dir_allows_direct_audit_append(run_dir):
            return
        try:
            append_event_to_run(run_dir, event_type, data=data, level=level)
        except OSError:
            _LOGGER.debug("control audit event write skipped", exc_info=True)

    def _authorize_control_audit_target(
        self,
        run_id: str,
        token: str,
        *,
        command_type: str = "",
        args: Mapping[str, Any] | None = None,
    ) -> None:
        if command_type in {"report_task_result", "approve", "deny"}:
            try:
                self._verify_task_callback_token(run_id, token, str((args or {}).get("task_id") or ""))
                return
            except TokenError:
                pass
        claims = self._verify_run_token(run_id, token)
        self._authorize_claim_subject(run_id, claims)

    def _authorize_claim_subject(self, run_id: str, claims: Any) -> None:
        with self._lock:
            record = self._records.get(run_id)
        if record is not None and (claims.tenant_id != record.tenant_id or claims.user_id != record.user_id):
            raise PermissionDenied("token subject mismatch")

    def _verify_task_callback_token(self, run_id: str, token: str, task_id: str) -> None:
        claims = self.token_manager.verify(
            token, kind="task_callback", audience=TASK_CALLBACK_AUDIENCES, run_id=run_id
        )
        if str(claims.metadata.get("task_id") or "") != task_id:
            raise PermissionDenied("callback token does not match this task")
        self._authorize_claim_subject(run_id, claims)

    def send_message(
        self,
        run_id: str,
        token: str,
        content: str | Sequence[Any],
        *,
        message_id: str = "",
        source: str = "api",
        correlation_id: str = "",
        causation_id: str = "",
        traceparent: str = "",
        tracestate: str = "",
    ) -> dict[str, Any]:
        """Deliver a follow-up user message to a running multi-turn session. It is queued and
        consumed as the next user turn once the current turn settles.

        ``content`` is either a plain ``str`` or a sequence of content parts (``ContentPart``
        dataclasses or their JSON dicts) for a multimodal message. Multimodal parts reference
        workspace files by ``source_ref`` — the bytes are resolved at wire-build time — so the
        message itself stays small; the size limit applies to the by-reference wire form.

        The message is wrapped in an ``inbox-message.v1`` envelope (provenance + idempotency). A
        caller-supplied ``message_id`` is the dedup key: re-sending it (a retry) is processed once —
        an already-processed id short-circuits to ``status="duplicate"``. Absent an id, a uuid is
        minted (only duplicates still in flight dedup)."""
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        # Already-processed id -> idempotent no-op (the dequeue-time check below catches the rest).
        if message_id and message_id in record.seen_inbox_ids:
            return {"run_id": run_id, "status": "duplicate", "message_id": message_id}
        message = _normalize_inbound_message(content)
        # Normalize inline (data:) media to durable blobs BEFORE queueing, so the queue — and any
        # checkpoint taken while the message is still unconsumed — carries only small blob: refs,
        # never the bytes inline. The loop resolves these via the checkpoint-store blob_reader.
        if isinstance(message, list) and self.checkpoint_store is not None:
            pending: dict[str, bytes] = {}
            message = normalize_inline_media_dicts(message, pending)
            for data in pending.values():
                self.checkpoint_store.put_blob(run_id, data)
        wire_bytes = len(
            (message if isinstance(message, str) else json.dumps(message)).encode("utf-8")
        )
        if wire_bytes > self.max_message_bytes:
            raise ValueError(f"message exceeds the {self.max_message_bytes}-byte limit")
        envelope = InboxMessage(
            content=message,
            id=message_id or f"inbox_{uuid.uuid4().hex[:12]}",
            source=source,
            run_id=run_id,
            correlation_id=correlation_id,
            causation_id=causation_id,
            traceparent=traceparent,
            tracestate=tracestate,
        )
        with self._lock:
            if record.status in {"completed", "failed", "limited"}:
                raise ValueError("cannot send a message to a terminal run")
            if record.message_queue.qsize() >= self.max_message_queue_depth:
                raise ValueError("message queue is full; retry once the run drains it")
        # Enqueue on the shared loop (asyncio.Queue is not thread-safe across threads).
        self._call_soon(record.message_queue.put_nowait, envelope.to_json())
        return {"run_id": run_id, "status": "queued", "message_id": envelope.id}

    def current_runtime_config(self, run_id: str) -> AgentRuntimeConfig | None:
        record = self._record(run_id)
        with self._lock:
            return record.runtime_config

    def runtime_config(self, run_id: str, token: str) -> dict[str, Any]:
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        with self._lock:
            config = record.runtime_config
            if config is None:
                return {
                    "run_id": record.run_id,
                    "tenant_id": record.tenant_id,
                    "ready": False,
                    "config_version": 0,
                    "config_hash": "",
                    "issuer": record.runtime_config_issuer,
                    "reason": record.runtime_config_reason,
                    "committed_at": record.runtime_config_committed_at,
                }
            return {
                "run_id": record.run_id,
                "tenant_id": record.tenant_id,
                "ready": True,
                "issuer": record.runtime_config_issuer,
                "reason": record.runtime_config_reason,
                "committed_at": record.runtime_config_committed_at,
                "config": config.to_json(),
                "config_version": config.config_version,
                "config_hash": config.config_hash,
            }

    def replace_runtime_config(
        self,
        run_id: str,
        token: str,
        *,
        expected_version: int,
        issuer: str,
        reason: str,
        config: AgentRuntimeConfig,
    ) -> dict[str, Any]:
        self._authorize_run(run_id, token)
        validate_runtime_config(
            config, _backend_builtin_tool_specs(self.subagent_definitions, self.tool_providers)
        )
        record = self._record(run_id)
        with self._lock:
            if record.status in {"completed", "failed", "limited"}:
                raise ValueError("cannot update runtime config for a terminal run")
            current_version = record.runtime_config.config_version if record.runtime_config else 0
            if expected_version != current_version:
                raise ValueError(
                    f"runtime config version mismatch: expected {expected_version}, current {current_version}"
                )
            if config.config_version <= current_version:
                # Auto-bump the version, preserving every other field (incl. output_validators).
                # replace() copies all fields, so a new config field can't be silently dropped on
                # hot-swap (an enumerated rebuild here previously dropped output-validator opt-outs).
                config = replace(config, config_version=current_version + 1)
            committed_at = time.time()
            self._write_runtime_config_run_meta(
                record,
                config,
                issuer=issuer,
                reason=reason,
                committed_at=committed_at,
            )
            record.runtime_config = config
            record.runtime_config_issuer = issuer
            record.runtime_config_reason = reason
            record.runtime_config_committed_at = committed_at
            return {
                "run_id": record.run_id,
                "tenant_id": record.tenant_id,
                "ready": True,
                "issuer": issuer,
                "reason": reason,
                "committed_at": committed_at,
                "config": config.to_json(),
                "config_version": config.config_version,
                "config_hash": config.config_hash,
            }

    def report_task_result(
        self,
        run_id: str,
        token: str,
        *,
        task_id: str,
        result: dict[str, Any],
        status: str = "answered",
    ) -> dict[str, Any]:
        """Deliver a hosted-task result (e.g. a human answer, an automation
        callback) into a running run, waking it if parked. Accepts a per-task
        callback token (scoped to this run+task) or the run token (operator)."""
        loop = self._authorize_task_result(run_id, token, task_id)
        return loop.report_task_result(task_id, result, status=status)

    def create_task(
        self,
        run_id: str,
        token: str,
        *,
        kind: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Backend-initiated task creation in a running run (automation/hitl).

        Returns the task id plus a scoped callback token and URL an external system
        POSTs to when it completes the work (Trigger.dev wait-token style)."""
        loop = self._authorize_active_loop(run_id, token)
        record = self._record(run_id)
        task_id = loop.create_task(kind, request)
        callback_token = self.token_manager.issue(
            kind="task_callback",
            audience=TASK_CALLBACK_AUDIENCE,
            run_id=run_id,
            tenant_id=record.tenant_id,
            user_id=record.user_id,
            ttl_s=self.task_callback_token_ttl_s,
            metadata={"task_id": task_id},
        )
        return {
            "task_id": task_id,
            "callback_token": callback_token,
            "callback_url": f"/v1/runs/{run_id}/tasks/{task_id}/result",
        }

    def _authorize_active_loop(self, run_id: str, token: str) -> AgentLoop:
        self._authorize_run(run_id, token)
        return self._active_loop(run_id)

    def _authorize_task_result(self, run_id: str, token: str, task_id: str) -> AgentLoop:
        """A task result may be reported with a per-task callback token (scoped to
        run+task) or the run token (operator). Try the scoped token first."""
        try:
            self._verify_task_callback_token(run_id, token, task_id)
        except TokenError:
            self._authorize_run(run_id, token)
        return self._active_loop(run_id)

    def _active_loop(self, run_id: str) -> AgentLoop:
        record = self._record(run_id)
        with self._lock:
            if record.status in {"completed", "failed", "limited"}:
                raise ValueError("cannot drive tasks for a terminal run")
            loop = record.loop
        if loop is None:
            raise ValueError("run has not started")
        return loop

    def proposal_file(self, run_id: str, token: str, path: str) -> dict[str, Any]:
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        proposal = self._read_proposal(record)
        if proposal is None:
            raise ValueError("proposal snapshot is not ready")
        try:
            file_payload = read_proposal_file_payload(record.run_dir, proposal, path)
        except ProposalFileError as exc:
            if exc.reason in {"not_found", "snapshot_missing"}:
                raise KeyError(str(exc)) from exc
            if exc.reason == "escapes_run_dir":
                raise PermissionDenied(str(exc)) from exc
            raise ValueError(str(exc)) from exc
        return {
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            "status": record.status,
            **file_payload,
        }

    def export_proposal_package(self, run_id: str, token: str) -> dict[str, Any]:
        """Build the portable proposal package and return a RECEIPT — never a filesystem path.

        The tar is stored as a content-addressed blob; the receipt's ``digest`` (sha256 of the tar
        bytes) is the retrieval handle for :meth:`read_run_artifact`. This keeps the
        "embedder never reads run_dir off disk" invariant for binary artifacts too: a remote
        embedder fetches the bytes back by digest, exactly like Bazel CAS / an OCI blob."""
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        output = record.run_dir / "proposal.tar"
        payload = export_package(record.run_dir, output)
        tar_bytes = output.read_bytes()
        assert self.checkpoint_store is not None
        digest = self.checkpoint_store.put_blob(run_id, tar_bytes)
        append_event_to_run(
            record.run_dir,
            "proposal.package.exported",
            data={"package_hash": payload["package_hash"], "digest": digest, "size_bytes": len(tar_bytes)},
        )
        return {
            "package_hash": payload["package_hash"],
            "digest": digest,  # the fetch handle (sha256 of the tar bytes)
            "size_bytes": len(tar_bytes),
            "media_type": "application/x-tar",
            "name": "proposal.tar",  # advisory filename for Content-Disposition only
        }

    def read_run_artifact(
        self, run_id: str, token: str, digest: str, *, offset: int = 0, limit: int | None = None
    ) -> bytes:
        """Fetch a run artifact's bytes by its sha256 ``digest`` — the single token-scoped,
        data-returning seam for binary artifacts (the export tar today, any blob tomorrow).

        Content-addressed: the digest IS the capability (a sha256 is unguessable, so possessing one
        is proof of knowledge of the content). ``offset``/``limit`` are accepted now so a future
        streaming/range fetch is a non-breaking addition; today they slice the in-memory bytes.
        Raises ``KeyError`` (→ 404) when the digest is unknown for this run, ``ValueError`` (→ 400)
        for a malformed digest."""
        self._authorize_run(run_id, token)
        if not _ARTIFACT_DIGEST_RE.match(digest):
            raise ValueError("digest must be a 64-char sha256 hex string")
        assert self.checkpoint_store is not None
        try:
            data = self.checkpoint_store.get_blob(run_id, digest)
        except KeyError as exc:
            raise KeyError(f"artifact not found: {digest}") from exc
        if offset or limit is not None:
            data = data[offset : (None if limit is None else offset + limit)]
        return data

    def approve_proposal(
        self,
        run_id: str,
        token: str,
        *,
        approver_id: str,
        approved_paths: tuple[str, ...] = (),
        note: str = "",
    ) -> dict[str, Any]:
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        approval = create_approval(
            record.run_dir,
            approver_id=approver_id,
            approved_paths=approved_paths or None,
            note=note,
        )
        write_approval(record.run_dir / "approval.json", approval)
        append_event_to_run(
            record.run_dir,
            "proposal.approved",
            data={"approval_hash": approval["approval_hash"], "package_hash": approval["package_hash"]},
        )
        return approval

    def reject_proposal(
        self,
        run_id: str,
        token: str,
        *,
        approver_id: str,
        reason: str,
    ) -> dict[str, Any]:
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        approval = create_approval(
            record.run_dir,
            approver_id=approver_id,
            decision="rejected",
            note=reason,
        )
        write_approval(record.run_dir / "approval.json", approval)
        append_event_to_run(
            record.run_dir,
            "proposal.rejected",
            data={"approval_hash": approval["approval_hash"], "package_hash": approval["package_hash"]},
        )
        return approval

    def apply_proposal(
        self,
        run_id: str,
        token: str,
        *,
        target: Path,
        approval_path: Path | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        self._authorize_run(run_id, token)
        if not self.allowed_apply_roots:
            raise PermissionDenied("proposal apply is disabled")
        target = target.resolve()
        if not any(is_within(root, target) for root in self.allowed_apply_roots):
            raise PermissionDenied(f"apply target is outside allowed roots: {target}")
        record = self._record(run_id)
        approval = approval_path or (record.run_dir / "approval.json")
        result = apply_package(record.run_dir, approval=approval, target=target, dry_run=dry_run)
        write_apply_result(record.run_dir / "apply-result.json", result)
        event_type = "proposal.conflict" if result.status == "conflict" else "proposal.applied"
        append_event_to_run(
            record.run_dir,
            event_type,
            data={
                "status": result.status,
                "approval_hash": result.approval_hash,
                "package_hash": result.package_hash,
                "applied_paths": list(result.applied_paths),
                "conflicts": [conflict.to_json() for conflict in result.conflicts],
            },
            level="warning" if result.status == "conflict" else "info",
        )
        return result.to_json()

    def events(
        self, run_id: str, token: str, *, from_seq: int = 0, limit: int | None = None
    ) -> dict[str, Any]:
        events_path = self._authorized_run_dir(run_id, token) / "events.jsonl"
        page = _read_event_page(events_path, from_seq=from_seq, limit=limit)
        return {"run_id": run_id, **page}

    def diagnostics(self, run_id: str, token: str, *, event_limit: int = 50) -> dict[str, Any]:
        if event_limit < 1:
            raise ValueError("event_limit must be positive")
        run_dir = self._authorized_run_dir(run_id, token)
        status = self.status(run_id, token)
        status_file = status.get("status_file") if isinstance(status.get("status_file"), dict) else {}
        last_event_seq = max(
            int(status.get("last_event_seq") or 0),
            int(status_file.get("last_event_seq") or 0),
        )
        from_seq = max(0, last_event_seq - event_limit + 1) if last_event_seq else 0
        event_page = _read_event_page(run_dir / "events.jsonl", from_seq=from_seq, limit=event_limit)
        event_summaries = [_diagnostic_event_summary(event) for event in event_page["events"]]
        control_events = [
            event for event in event_summaries if str(event.get("type") or "").startswith("control.command.")
        ]
        failure = _read_optional_json(run_dir / "failure.json")
        recover_attempts = self._read_recover_attempts(run_dir)
        return {
            "run_id": run_id,
            "status": status,
            "failure": failure,
            "recovery": {
                "attempts": recover_attempts,
                "max_attempts": self.max_recover_attempts,
                "failure_marked": failure is not None,
                "unrecoverable": bool(failure and failure.get("error_code") == "unrecoverable"),
            },
            "events": {
                "from_seq": from_seq,
                "next_seq": event_page["next_seq"],
                "has_more": event_page["has_more"],
                "items": event_summaries,
            },
            "control": {"events": control_events},
            "trace_ids": _trace_ids_from_events(event_page["events"]),
        }

    def descendant_events(
        self,
        run_id: str,
        token: str,
        descendant_run_id: str,
        *,
        from_seq: int = 0,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Stream a descendant (subagent) run's events, authorized via the ancestor's run token.

        A spawned subagent is an isolated child run (id ``<parent>.sub.<task>``) under the same
        run_root but with NO backend record/token, so :meth:`events` can't reach it. The owner of
        an ancestor run reads a descendant's events.jsonl here — its tool calls + token deltas —
        for live subagent observability, without touching the filesystem itself. Authorization is
        the ancestor's token plus an id-prefix descendant check (a subagent id always extends its
        parent's with ``.sub.<task>``, at any depth)."""
        self._authorize_run(run_id, token)
        if descendant_run_id != run_id and not descendant_run_id.startswith(f"{run_id}.sub."):
            raise PermissionDenied("run is not a descendant of the authorized run")
        if any(sep in descendant_run_id for sep in ("/", "\\")) or ".." in descendant_run_id:
            raise PermissionDenied("invalid descendant run id")
        events_path = self.run_root / descendant_run_id / "events.jsonl"
        page = _read_event_page(events_path, from_seq=from_seq, limit=limit)
        return {"run_id": descendant_run_id, **page}

    def jobs(self, run_id: str, token: str) -> dict[str, Any]:
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        jobs = list_job_artifacts(record.run_dir)
        return {"run_id": run_id, "tenant_id": record.tenant_id, "jobs": jobs}

    def job_status(self, run_id: str, token: str, job_id: str) -> dict[str, Any]:
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        return {
            "run_id": run_id,
            "tenant_id": record.tenant_id,
            "job": get_job_artifact(record.run_dir, job_id),
        }

    def job_logs(
        self,
        run_id: str,
        token: str,
        job_id: str,
        *,
        stream: str = "stdout",
        tail_bytes: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        logs = read_job_log_text(
            record.run_dir,
            job_id,
            stream=stream,  # type: ignore[arg-type]
            tail_bytes=tail_bytes,
            offset=offset,
        )
        return {"run_id": run_id, "tenant_id": record.tenant_id, **logs}

    def cancel_job(self, run_id: str, token: str, job_id: str) -> dict[str, Any]:
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        payload = request_job_cancel(record.run_dir, job_id)
        return {"run_id": run_id, "tenant_id": record.tenant_id, **payload}

    def tenant_usage(self, tenant_id: str) -> dict[str, Any]:
        with self._lock:
            usage = self._usage.get(tenant_id) or TenantUsage(tenant_id)
            return usage.to_json()

    def record_event(self, run_id: str, event: AgentEvent) -> None:
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return
            record.last_event_seq = event.seq
            record.last_event_type = event.type
            if event.type == "run.started":
                record.status = "running"
                record.started_at = time.time()
            elif event.type == "run.awaiting_input":
                # Parked waiting for the next user message or a hosted-task result.
                if record.status not in {"completed", "failed", "limited"}:
                    record.status = "awaiting_input"
            elif event.type in {"run.resumed", "model.turn.started"}:
                if record.status == "awaiting_input":
                    record.status = "running"
            elif event.type == "run.finished":
                # Record terminal metadata, but DO NOT flip the gating status here. The
                # run.finished event fires inside loop.close() (on the loop's thread), while
                # record.result is only set afterward by _record_run_result on the worker
                # thread. Setting status="completed" here would let wait_for_run/result()
                # observe a terminal status before the result is recorded (result() would
                # KeyError on final_text). _record_run_result owns the terminal status so it
                # flips together with record.result, under the same lock.
                record.finished_at = time.time()
                record.error = str(event.data.get("error") or "")
                record.error_code = str(event.data.get("error_code") or "")
            elif event.type == "run.failed":
                record.status = "failed"
                record.error = str(event.data.get("error") or "")
                record.error_code = str(event.data.get("error_code") or "")

    def wait_for_run(self, run_id: str, *, timeout_s: float = 10.0) -> BackendRunState:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            status = self._record(run_id).status
            if status in {"completed", "failed", "limited"}:
                return status
            time.sleep(0.05)
        raise TimeoutError(f"run did not finish before timeout: {run_id}")

    async def _drive_session(self, run_id: str, request: BackendRunRequest, loop: AgentLoop) -> AgentRunResult:
        """Cold-start driver: open the run, take the first turn, then hand off to the
        shared open-session loop (also used by checkpoint recovery)."""
        record = self._record(run_id)
        await loop.aopen()
        first_input: str | tuple[ContentPart, ...] = request.input_parts or request.instruction
        try:
            suspension = await loop.arun_until_suspended(first_input)
        except NativeAgentError:
            # Bootstrap failed (terminal session already recorded); just finalize.
            return await loop.aclose()
        return await self._drive_open_session(record, request, loop, suspension, started=time.time(), turns=1)

    async def _drive_open_session(
        self,
        record: BackendRunRecord,
        request: BackendRunRequest,
        loop: AgentLoop,
        suspension: Suspension,
        *,
        started: float,
        turns: int,
    ) -> AgentRunResult:
        """Drive an already-open run as a multi-turn session: wait for a parked task to
        complete or await the message queue for the next user turn — until idle/lifetime/
        turn limits, cancel, or close. Shared by cold start and recovery; each park point
        durably checkpoints (loop state + backend message queue).

        Runs as a coroutine on the shared loop. The next-message wait is a pure
        ``await asyncio.Queue.get()`` (a parked session holds no thread). The hosted/in-proc
        task-wait is offloaded with asyncio.to_thread (bounded by task_wait_poll_s and
        retried) since it blocks on the job manager's cross-thread condition (P2);
        report_task_result wakes it from another thread."""
        consecutive_turn_failures = 0
        while True:
            # Keep the authoritative FSM state on the record current with each park, so a
            # concurrent control inspect/health reports the live state.
            record.session_state = state_from_suspension(suspension)
            if suspension.turn is not None:
                # Capture the settled turn's validated output so status() can surface it live.
                record.last_final_output = suspension.turn.final_output
            if suspension.reason in {"terminal", "limited"}:
                break
            if suspension.reason == "awaiting_tasks":
                self._persist_run_checkpoint(record)
                ready = await asyncio.to_thread(loop.wait_for_pending_tasks, self.task_wait_poll_s)
                if self._session_should_stop(record, started, turns):
                    break
                if ready or not loop.has_pending_tasks():
                    # A task was delivered (or none remain): resume the pump.
                    suspension = await loop.arun_until_suspended(None)
                # else: tasks still pending after the poll window -> keep waiting.
                continue
            if suspension.reason == "paused":
                # Cooperative pause: the loop froze the turn at a clean step boundary (its
                # pending_observations are kept). Park until a resume signal wakes us, then
                # re-pump the SAME turn with no new input. The loop already checkpointed the
                # pause park; persist the backend queue too.
                if self._session_should_stop(record, started, turns):
                    break
                self._persist_run_checkpoint(record)
                try:
                    # Raw get: the paused branch is the one place that must SEE _RESUME_SESSION.
                    signal = await asyncio.wait_for(record.message_queue.get(), self.idle_timeout_s)
                except asyncio.TimeoutError:
                    break
                if signal is _CLOSE_SESSION:
                    break
                if signal is not _RESUME_SESSION:
                    # A user message arrived while paused: resume the frozen turn first, then
                    # let the settled branch consume this message as the next turn.
                    record.message_queue.put_nowait(signal)
                suspension = await loop.arun_until_suspended(None)
                continue
            if suspension.reason == "turn_failed":
                # Recoverable model-turn failure: the session is alive (core kept it so). Apply
                # the retry POLICY here. Give up once too many turns fail in a row.
                consecutive_turn_failures += 1
                if consecutive_turn_failures >= self.max_consecutive_turn_failures or self._session_should_stop(
                    record, started, turns
                ):
                    loop.fail_recoverable(
                        suspension.error or "model turn failed repeatedly",
                        error_code=suspension.error_code or "model_error",
                    )
                    break
                if suspension.retryable:
                    # Transient (429 / exhausted-5xx / network): bounded auto-retry with backoff,
                    # then re-issue the SAME turn (no new user message).
                    self._persist_run_checkpoint(record)
                    await _async_sleep_before_retry(consecutive_turn_failures, self.turn_retry)
                    if self._session_should_stop(record, started, turns):
                        break
                    suspension = await loop.arun_until_suspended(None)
                    continue
                # Non-retryable but recoverable (config/auth 4xx): a blind retry would just fail
                # again. Park for the user to fix the config (model/effort) and resend. A one-shot
                # run has nobody to resend, so finalize it as failed.
                if not request.multi_turn:
                    loop.fail_recoverable(
                        suspension.error or "model turn failed",
                        error_code=suspension.error_code or "model_error",
                    )
                    break
                loop.await_user_input()
                self._persist_run_checkpoint(record)
                try:
                    message = await self._await_session_message(record)
                except asyncio.TimeoutError:
                    break
                if message is _CLOSE_SESSION:
                    break
                turns += 1
                suspension = await loop.arun_until_suspended(_queued_message_to_loop_input(message))
                continue
            if suspension.reason == "interrupted":
                # Turn-level stop (the user hit "stop"): not a failure, not terminal. Park the
                # multi-turn session for the next message; a one-shot run just closes. Does not
                # count against the turn-failure streak.
                if not request.multi_turn or self._session_should_stop(record, started, turns):
                    break
                loop.await_user_input()
                self._persist_run_checkpoint(record)
                try:
                    message = await self._await_session_message(record)
                except asyncio.TimeoutError:
                    break
                if message is _CLOSE_SESSION:
                    break
                turns += 1
                suspension = await loop.arun_until_suspended(_queued_message_to_loop_input(message))
                continue
            # settled. One-shot runs close here; multi-turn awaits the next message.
            consecutive_turn_failures = 0  # a settled turn clears the failure streak
            # Drain staged outbox sends now (the loop is still open) so a one-shot run that finishes
            # without parking still gets its side-effects dispatched; the multi-turn await below
            # also drains (a no-op then, since these are already dispatched).
            self._drain_outbox(record, loop)
            if not request.multi_turn:
                break
            if self._session_should_stop(record, started, turns):
                break
            loop.await_user_input()
            self._persist_run_checkpoint(record)
            try:
                # Pure async await — a parked multi-turn session holds no thread.
                message = await self._await_session_message(record)
            except asyncio.TimeoutError:
                break  # idle timeout
            if message is _CLOSE_SESSION:
                break
            turns += 1
            suspension = await loop.arun_until_suspended(_queued_message_to_loop_input(message))
        return await loop.aclose()

    async def _await_session_message(self, record: BackendRunRecord) -> Any:
        """Await the next queued user message, dropping stray ``_RESUME_SESSION`` sentinels (a
        resume aimed at a run that is not currently paused is a no-op). Raises
        ``asyncio.TimeoutError`` once the idle timeout elapses, mirroring the bare
        ``wait_for`` it replaces."""
        deadline = time.monotonic() + self.idle_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            message = await asyncio.wait_for(record.message_queue.get(), remaining)
            if message is _RESUME_SESSION:
                continue  # stray resume (run not paused): ignore and keep waiting
            if is_inbox_envelope(message):
                msg_id = str(message.get("id") or "")
                if msg_id and msg_id in record.seen_inbox_ids:
                    continue  # idempotent ingress: a redelivered message is processed once
                if msg_id:
                    # Mark processed; persisted at the next park checkpoint so the dedup survives a
                    # restart (the marker rides the same checkpoint as the message's effects).
                    record.seen_inbox_ids.add(msg_id)
            return message

    def _persist_run_checkpoint(self, record: BackendRunRecord) -> None:
        """Augment the loop's own park-point checkpoint with the backend-owned message
        queue, so a follow-up message that arrived but was not yet consumed survives a
        restart. No-op when the loop refuses a snapshot (a live shell job is parked-on)."""
        loop = record.loop
        if loop is None:
            return
        checkpoint = loop.snapshot()
        if checkpoint is None:
            return
        # Peek (don't drain) the residual queue; consumed messages are already reflected in
        # the loop's turn handle / pending input. Runs in the driver coroutine on the shared
        # loop, so reading the asyncio.Queue's backing deque needs no lock (single-threaded
        # loop; producers enqueue via _call_soon on this same loop).
        residual = [
            message
            for message in list(record.message_queue._queue)
            # inbox envelope dict, or legacy raw str/list parts; the close/resume sentinels (objects)
            # are dropped.
            if isinstance(message, (str, list, dict))
        ]
        checkpoint.queued_messages = residual
        checkpoint.inbox_seen_ids = sorted(record.seen_inbox_ids)
        # Overwrites the same seq the loop just committed, now with the queue included.
        assert self.checkpoint_store is not None
        self.checkpoint_store.put(checkpoint, loop.collect_checkpoint_blobs())
        self._drain_outbox(record, loop)

    def _outbox_backoff_delay(self, attempts: int) -> float:
        """Capped exponential backoff with full jitter — ``uniform(0, min(cap, base*factor**attempts))``.
        Full jitter (AWS) maximally decorrelates retries so a fleet of failed sends doesn't restorm a
        recovering target in lockstep. ``attempts`` is the number already made (>=1 here)."""
        ceiling = min(self.outbox_retry_cap_s, self.outbox_retry_base_s * (self.outbox_retry_factor ** attempts))
        return self._outbox_rng.uniform(0.0, max(0.0, ceiling))

    def _drain_outbox(self, record: BackendRunRecord, loop: AgentLoop) -> None:
        """Dispatch *due* staged outbox requests at the edge (after they are durably persisted as
        ``pending``), then persist again so a ``dispatched`` status is recorded. The send happens
        here, never in the core; a crash between the two persists redispatches on recover, made safe
        by the request's idempotency_key. A retryable failure stamps a backoff ``next_attempt_at`` so
        the request is only redispatched once its time arrives (the watchdog redrive tick wakes it,
        independent of run activity). No-op without a sender or due requests."""
        sender = record.outbox_sender
        now = time.time()
        due = loop.due_outbox(now)
        if sender is None or not due:
            return
        changed = False
        for request in due:
            # Ensure the request carries a trace before the edge sends it (requests staged via the
            # outbox tool already have one; this covers any other path). Observability only.
            if not request.traceparent:
                request.traceparent = new_traceparent()
            try:
                receipt = sender.send(request)
            except Exception as exc:  # a sender raising is a retryable transport failure
                receipt = OutboxReceipt(ok=False, error=str(exc), retryable=True)
            next_attempt_at = now + self._outbox_backoff_delay(request.attempts + 1)
            status = loop.record_outbox_result(
                request.id,
                receipt,
                max_attempts=self.outbox_max_attempts,
                next_attempt_at=next_attempt_at,
            )
            changed = True
            if request.expect_ack and status in {"dispatched", "failed"}:
                self._stage_outbox_ack(record, request, status, receipt)
        if changed:
            checkpoint = loop.snapshot()
            if checkpoint is not None:
                checkpoint.queued_messages = [
                    m for m in list(record.message_queue._queue) if isinstance(m, (str, list, dict))
                ]
                checkpoint.inbox_seen_ids = sorted(record.seen_inbox_ids)
                self.checkpoint_store.put(checkpoint, loop.collect_checkpoint_blobs())

    def _stage_outbox_ack(
        self, record: BackendRunRecord, request: Any, status: str, receipt: OutboxReceipt
    ) -> None:
        """Deliver an outbox send's receipt back to the run as an inbox message (request-reply,
        **non-park** — the agent observes it on its next activation), correlated by ``correlation_id``.
        Reuses the idempotent inbox path: a stable ack id (``ack_<request id>``) + the inbox seen-set
        make a redelivery a no-op. Dropped if the run is terminal (no consumer) or its queue is full
        (best-effort). Runs on the shared loop, so the queue put needs no cross-thread marshaling."""
        ack_id = f"ack_{request.id}"
        if record.status in {"completed", "failed", "limited"}:
            return  # terminal run — no consumer for the ack (documented limitation)
        if ack_id in record.seen_inbox_ids or record.message_queue.qsize() >= self.max_message_queue_depth:
            return
        summary = f"[outbox-ack] request {request.id} to {request.destination!r}: {status}"
        if receipt.reference:
            summary += f" (ref={receipt.reference})"
        if receipt.error:
            summary += f" (error={receipt.error})"
        envelope = InboxMessage(
            content=summary,
            id=ack_id,
            source="outbox",
            type="outbox_ack",
            run_id=record.run_id,
            correlation_id=request.correlation_id or request.id,
            causation_id=request.id,
            traceparent=request.traceparent,
            tracestate=request.tracestate,
        )
        record.message_queue.put_nowait(envelope.to_json())

    def _session_should_stop(self, record: BackendRunRecord, started: float, turns: int) -> bool:
        return (
            record.cancellation_token.requested
            or (time.time() - started) >= self.max_session_lifetime_s
            or turns >= self.max_turns
        )

    async def _run_run(
        self,
        run_id: str,
        request: BackendRunRequest,
        workspace_root: Path,
        llm_gateway_token: str,
        web_gateway_token: str,
    ) -> None:
        # Bound concurrent active runs: at capacity this awaits (without holding a thread),
        # so the run stays ``queued`` until a slot frees rather than piling work onto a
        # saturated process.
        if self._run_semaphore is not None:
            await self._run_semaphore.acquire()
        try:
            try:
                loop = self._build_loop(run_id, request, workspace_root, llm_gateway_token, web_gateway_token)
                with self._lock:
                    self._records[run_id].loop = loop
                    self._records[run_id].outbox_sender = self._outbox_sender_for(request)
                # Persist the recovery metadata before the first turn so a crash at any park
                # point can be resumed (the checkpoint itself is written by the driver).
                self._write_run_meta(self._record(run_id), request)
                result = await self._drive_session(run_id, request, loop)
                self._record_run_result(run_id, result)
            except Exception as exc:
                self._record_run_failure(run_id, exc)
        finally:
            if self._run_semaphore is not None:
                self._run_semaphore.release()

    def _capability_broker_for(self, request: BackendRunRequest) -> Any:
        """Build the run's capability broker from the factory (scoped to run identity), or None
        to leave capability gating off for this run."""
        if self.capability_broker_factory is None:
            return None
        return self.capability_broker_factory(request)

    def _outbox_sender_for(self, request: BackendRunRequest) -> Any:
        """Build the run's outbox sender from the factory (scoped to run identity), or None to leave
        staged outbox requests pending (durable, never dispatched)."""
        if self.outbox_sender_factory is None:
            return None
        return self.outbox_sender_factory(request)

    def _build_loop(
        self,
        run_id: str,
        request: BackendRunRequest,
        workspace_root: Path,
        llm_gateway_token: str,
        web_gateway_token: str,
    ) -> AgentLoop:
        """Construct the run's AgentLoop (shared by autonomous and stream-driven paths)."""
        spec = self._run_spec_for_request(run_id, request, workspace_root)
        runtime_config = self.current_runtime_config(run_id)
        adapter = self._build_model_adapter(
            spec,
            llm_gateway_token,
            runtime_config.model if runtime_config is not None else None,
            token_provider=self._llm_token_source(run_id, request, runtime_config),
        )
        return AgentLoop(
            spec=spec,
            model_adapter=adapter,
            event_sinks=(BackendRunStateSink(self, run_id), *(make() for make in self.extra_event_sink_factories)),
            permission_policy=request.permission_policy,
            cancellation_token=self._record(run_id).cancellation_token,
            shell_approval_provider=None,
            web_gateway_client=self._web_gateway_client(web_gateway_token),
            runtime_config_provider=BackendRuntimeConfigProvider(self, run_id),
            checkpoint_store=self.checkpoint_store,
            emit_output_deltas=self.emit_output_deltas,
            subagent_definitions=self.subagent_definitions,
            tool_providers=self.tool_providers,
            context_providers=self.context_providers,
            output_validators=self.output_validators,
            capability_broker=self._capability_broker_for(request),
        )

    async def astream_run(self, request: BackendRunRequest) -> AsyncIterator[dict[str, Any]]:
        """Stream-driven run: the transport-neutral programmatic seam behind the SSE endpoint.

        Drives ONE submit via ``loop.astream`` and yields wire-frame dicts: a leading
        ``{"kind":"meta",...}`` (run id/token/urls — mirrors ``BackendRunSubmission`` so the
        consumer can poll artifacts later), then ``{"kind":"event",...}`` (orchestration) and
        ``{"kind":"delta",...}`` (token deltas) per stream item, then exactly one terminal
        ``{"kind":"result",...}``. Must be driven on the shared loop (astream binds the loop it
        runs on). An in-process async consumer can ``async for`` this directly, no HTTP.

        Single-submit scope: the run is closed when the stream drains. A mid-stream external
        hosted-task park is surfaced in the result frame and then closed (HITL-over-stream is
        deferred), so this never leaves a resumable run dangling.
        """
        prepared = self._prepare_run_record(request)
        run_id = prepared.run_id
        if self._run_semaphore is not None:
            await self._run_semaphore.acquire()
        loop: AgentLoop | None = None
        closed = False
        try:
            yield {"kind": "meta", **self._submission_for(prepared).to_json()}
            loop = self._build_loop(
                run_id, request, prepared.workspace_root, prepared.llm_gateway_token, prepared.web_gateway_token
            )
            with self._lock:
                self._records[run_id].loop = loop
                self._records[run_id].outbox_sender = self._outbox_sender_for(request)
            self._write_run_meta(prepared.record, request)
            await loop.aopen()
            suspension: Suspension | None = None
            first_input: str | tuple[ContentPart, ...] = request.input_parts or request.instruction
            async with loop.astream(first_input) as stream:
                async for item in stream:
                    yield self._frame(item)
                suspension = stream.suspension
            result = await loop.aclose()
            closed = True
            self._record_run_result(run_id, result)
            frame: dict[str, Any] = {
                "kind": "result",
                "status": result.status,
                "final_text": result.final_text,
                "error": result.error,
                "error_code": result.error_code,
            }
            if suspension is not None and suspension.has_external:
                frame["awaiting_task_ids"] = list(suspension.awaiting_task_ids)
                frame["note"] = "run closed; hosted task cancelled (HITL streaming deferred)"
            yield frame
        except Exception as exc:
            if loop is not None and not closed:
                try:
                    await loop.aclose()
                except Exception:  # noqa: BLE001 - finalization best-effort; the failure is recorded below
                    pass
            self._record_run_failure(run_id, exc)
            yield {
                "kind": "result",
                "status": "failed",
                "error": str(exc),
                "error_code": getattr(exc, "error_code", "internal_error"),
            }
        finally:
            if self._run_semaphore is not None:
                self._run_semaphore.release()

    def _frame(self, item: Any) -> dict[str, Any]:
        """Wrap one astream item as a neutral wire frame (reference framing on core to_json)."""
        if isinstance(item, AgentEvent):
            return {"kind": "event", **item.to_json()}
        return {"kind": "delta", **item.to_json()}  # ModelStreamChunk

    def _record_run_result(self, run_id: str, result: AgentRunResult) -> None:
        with self._lock:
            record = self._records[run_id]
            record.result = result
            record.status = result.status
            # Reconcile the terminal BackendRunState onto the FSM (cancel -> CANCELLED).
            record.session_state = to_session_state(result.status, error_code=result.error_code)
            record.error = result.error
            record.error_code = result.error_code
            record.finished_at = time.time()
            self._usage.setdefault(record.tenant_id, TenantUsage(record.tenant_id)).add_metrics(
                result.metrics
            )

    def _record_run_failure(self, run_id: str, exc: Exception) -> None:
        # Durable failure mark FIRST — before the in-memory status flips to a terminal
        # value. A worker-level crash that bypassed the loop's own bundle would otherwise
        # leave no failure.json, and the restart scanner would treat the run as merely
        # parked and resume it into a crash loop. Writing the bundle before the terminal
        # status also closes the race where an observer sees ``failed`` but no bundle yet.
        # ``overwrite=False`` preserves the loop's richer bundle when it already wrote one.
        # There is still no auto-recovery: the bundle is purely an operator restore aid.
        self._write_failure_bundle(
            run_id,
            self.run_root / run_id,
            error=str(exc),
            error_code=getattr(exc, "error_code", "internal_error"),
            exc_type=type(exc).__name__,
            overwrite=False,
        )
        with self._lock:
            record = self._records[run_id]
            record.status = "failed"
            record.error = str(exc)
            record.error_code = getattr(exc, "error_code", "internal_error")
            record.finished_at = time.time()

    def _write_failure_bundle(
        self,
        run_id: str,
        run_dir: Path,
        *,
        error: str,
        error_code: str,
        exc_type: str,
        overwrite: bool,
    ) -> None:
        """Write ``run_dir/failure.json`` (the operator-facing failure bundle, same schema
        as the core's). ``overwrite=False`` preserves a bundle the loop already wrote
        (which carries richer, in-run context)."""
        failure_path = run_dir / "failure.json"
        if failure_path.exists() and not overwrite:
            return
        last_good_seq = 0
        if self.checkpoint_store is not None:
            try:
                stored = self.checkpoint_store.latest(run_id)
                last_good_seq = stored.seq if stored is not None else 0
            except Exception:  # pragma: no cover - last-good lookup must never mask the failure
                last_good_seq = 0
        bundle = {
            "schema_version": namespaced_id("failure.v1"),
            "run_id": run_id,
            "error": error,
            "error_code": error_code,
            "type": exc_type,
            "last_good_seq": last_good_seq,
            "restore_hint": (
                f"restore checkpoint seq {last_good_seq} for run {run_id} via CheckpointStore, "
                "then resume via recover_runs"
                if last_good_seq > 0
                else "no recoverable checkpoint; inspect run logs and run.json"
            ),
            "failed_at": time.time(),
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(failure_path, bundle)

    def _recover_attempts_path(self, run_dir: Path) -> Path:
        return run_dir / "recover_attempts.json"

    def _read_recover_attempts(self, run_dir: Path) -> int:
        try:
            payload = json.loads(self._recover_attempts_path(run_dir).read_text(encoding="utf-8"))
            return int(payload["count"])
        except (FileNotFoundError, ValueError, KeyError, OSError, TypeError):
            return 0

    def _bump_recover_attempts(self, run_dir: Path) -> int:
        count = self._read_recover_attempts(run_dir) + 1
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self._recover_attempts_path(run_dir), {"count": count})
        return count

    def _clear_recover_attempts(self, run_dir: Path) -> None:
        self._recover_attempts_path(run_dir).unlink(missing_ok=True)

    def _write_run_meta(self, record: BackendRunRecord, request: BackendRunRequest) -> None:
        """Write run.json — the durable recovery descriptor. Holds everything
        ``recover_runs`` needs to rebuild a run that was parked when the process died:
        identity, workspace, limits, policy, and the resolved runtime config (gateway
        tokens are re-issued on recovery, not stored). Runtime-config changes update this
        descriptor, so recovery uses the latest committed config instead of the run-start config."""
        config = record.runtime_config
        committed_at = record.runtime_config_committed_at or time.time()
        meta = {
            "schema_version": _RUN_META_SCHEMA_VERSION,
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            "user_id": record.user_id,
            "workspace_root": str(record.workspace_root),
            "mode": request.mode,
            "workspace_backend": request.workspace_backend,
            "multi_turn": request.multi_turn,
            # For history listing (DX-12): a created-at stamp + a short title (first instruction).
            "created_at": record.created_at,
            "title": " ".join((request.instruction or "").split())[:80],
            "limits": {
                "max_steps": request.max_steps,
                "max_tool_calls": request.max_tool_calls,
                "max_bytes_read": request.max_bytes_read,
                "max_duration_s": request.max_duration_s,
            },
            "permission_policy": request.permission_policy.to_json(),
            "runtime_config": config.to_json() if config else None,
            "runtime_config_version": config.config_version if config else 0,
            "runtime_config_hash": config.config_hash if config else "",
            "runtime_config_issuer": record.runtime_config_issuer,
            "runtime_config_reason": record.runtime_config_reason,
            "runtime_config_committed_at": committed_at,
        }
        record.run_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(record.run_dir / "run.json", meta)
        self._store_run_meta(record.run_id, meta)

    def _write_runtime_config_run_meta(
        self,
        record: BackendRunRecord,
        config: AgentRuntimeConfig,
        *,
        issuer: str,
        reason: str,
        committed_at: float,
    ) -> None:
        meta = _read_run_meta(record.run_dir)
        if meta is None:
            raise ValueError("run metadata is not ready")
        meta["runtime_config"] = config.to_json()
        meta["runtime_config_version"] = config.config_version
        meta["runtime_config_hash"] = config.config_hash
        meta["runtime_config_issuer"] = issuer
        meta["runtime_config_reason"] = reason
        meta["runtime_config_committed_at"] = committed_at
        self._store_run_meta(record.run_id, meta)
        write_json_atomic(record.run_dir / "run.json", meta)

    def _store_run_meta(self, run_id: str, meta: Mapping[str, Any]) -> None:
        if self.checkpoint_store is None:
            return
        put_metadata = getattr(self.checkpoint_store, "put_run_metadata", None)
        if callable(put_metadata):
            put_metadata(run_id, meta)

    def _read_recovery_meta(self, run_dir: Path, run_id: str) -> dict[str, Any] | None:
        meta = _read_run_meta(run_dir)
        if meta is None and self.checkpoint_store is not None:
            read_metadata = getattr(self.checkpoint_store, "run_metadata", None)
            stored = read_metadata(run_id) if callable(read_metadata) else None
            if stored is not None:
                meta = _validate_run_meta(stored)
                if meta is not None:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    write_json_atomic(run_dir / "run.json", meta)
        return meta

    def recover_runs(self) -> list[str]:
        """Scan ``run_root`` for runs left parked by a previous process and resume each
        from its checkpoint. Returns the recovered run ids. Idempotent: runs already
        tracked in-memory, terminal checkpoints, and runs missing run.json are skipped."""
        recovered: list[str] = []
        if not self.run_root.is_dir():
            return recovered
        for run_dir in sorted(path for path in self.run_root.iterdir() if path.is_dir()):
            run_id = run_dir.name
            with self._lock:
                if run_id in self._records:
                    continue
            # A failed run is never auto-resumed: its failure.json is the operator's
            # restore aid (covers the edge where a failure could not write a terminal mark).
            if (run_dir / "failure.json").exists():
                continue
            assert self.checkpoint_store is not None
            stored = self.checkpoint_store.latest(run_id)
            if stored is None or stored.checkpoint.terminal:
                continue
            if self._attempt_resume(run_dir, run_id):
                recovered.append(run_id)
        return recovered

    def _attempt_resume(self, run_dir: Path, run_id: str) -> bool:
        """Resume one run from its latest checkpoint. Returns True on success. Skips runs
        with no resumable checkpoint or missing run.json. On a resume exception, bumps the
        durable attempt counter and, once ``max_recover_attempts`` is reached, marks the run
        unrecoverable (durable failure.json) so it is never retried into a crash loop."""
        assert self.checkpoint_store is not None
        stored = self.checkpoint_store.latest(run_id)
        if stored is None or stored.checkpoint.terminal:
            return False
        meta = self._read_recovery_meta(run_dir, run_id)
        if meta is None:
            return False
        try:
            self._resume_from_checkpoint(stored, meta)
        except Exception as exc:
            attempts = self._bump_recover_attempts(run_dir)
            _LOGGER.error(
                "resume of run %s failed (attempt %d/%d): %s",
                run_id,
                attempts,
                self.max_recover_attempts,
                exc,
            )
            if attempts >= self.max_recover_attempts:
                self._write_failure_bundle(
                    run_id,
                    run_dir,
                    error=f"recovery failed after {attempts} attempts: {exc}",
                    error_code="unrecoverable",
                    exc_type=type(exc).__name__,
                    overwrite=True,
                )
                _LOGGER.error("run %s marked unrecoverable", run_id)
            return False
        self._clear_recover_attempts(run_dir)
        return True

    def resume_run(self, run_id: str, token: str) -> dict[str, Any]:
        """Resume a single parked run from its checkpoint — the token-scoped, single-run analog of
        :meth:`recover_runs` (which is a process-global, no-token operator primitive).

        This closes the read/write asymmetry: a token holder could always *read* a historical run
        (:meth:`_authorized_run_dir` needs no in-memory record), but every *write* path goes through
        :meth:`_authorize_run` → :meth:`_record`, which raises ``KeyError`` for a run not currently
        in memory. ``resume_run`` materializes that record from the durable checkpoint so a follow-up
        :meth:`send_message` works after a restart. Idempotent: a run already tracked in memory
        returns ``resumed=False`` with its current status. The run token is the capability; its
        claims are checked against the persisted ``run.json`` identity since there is no record yet.
        """
        claims = self._verify_run_token(run_id, token)
        with self._lock:
            existing = self._records.get(run_id)
        if existing is not None:
            if claims.tenant_id != existing.tenant_id or claims.user_id != existing.user_id:
                raise PermissionDenied("token subject mismatch")
            return {"run_id": run_id, "status": existing.status, "resumed": False}
        if any(sep in run_id for sep in ("/", "\\")) or ".." in run_id:
            raise PermissionDenied("invalid run id")
        run_dir = self.run_root / run_id
        meta = self._read_recovery_meta(run_dir, run_id)
        if meta is None:
            raise KeyError(f"unknown run: {run_id}")
        if claims.tenant_id != (meta.get("tenant_id") or "") or claims.user_id != (meta.get("user_id") or ""):
            raise PermissionDenied("token subject mismatch")
        if (run_dir / "failure.json").exists():
            raise ValueError("run is marked unrecoverable; inspect failure.json")
        assert self.checkpoint_store is not None
        stored = self.checkpoint_store.latest(run_id)
        if stored is None or stored.checkpoint.terminal:
            raise ValueError("run has no resumable checkpoint")
        if not self._attempt_resume(run_dir, run_id):
            # _attempt_resume bumps the durable attempt counter / writes failure.json at the cap.
            raise ValueError("resume failed; inspect run logs / failure.json")
        record = self._record(run_id)
        return {"run_id": run_id, "status": record.status, "resumed": True}

    # --- Active watchdog / lease (operational layer; the core never auto-recovers) -------

    def start_watchdog(self) -> None:
        """Begin the operational watchdog thread: heartbeat this backend's own runs and
        reclaim runs orphaned by a crashed peer (a stale lease). Opt-in and idempotent."""
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()
        thread = threading.Thread(
            target=self._watchdog_loop,
            name=f"monoid-watchdog-{self._worker_id[:8]}",
            daemon=True,
        )
        self._watchdog_thread = thread
        thread.start()

    def stop_watchdog(self) -> None:
        self._watchdog_stop.set()
        thread = self._watchdog_thread
        if thread is not None:
            thread.join(timeout=5)
        self._watchdog_thread = None

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(self.watchdog_interval_s):
            try:
                self._heartbeat_own_runs()
                self._reclaim_stale_runs()
                self._redrive_outbox()
            except Exception:  # pragma: no cover - the watchdog must never die on a tick
                _LOGGER.exception("watchdog tick failed")

    def _redrive_outbox(self) -> None:
        """Redrive due outbox requests for this worker's live runs — the operational tick that makes
        retry timing independent of run activity (a request whose backoff ``next_attempt_at`` has
        arrived is dispatched even if its run is otherwise idle). Runs on the watchdog thread but
        marshals the actual drain onto the shared loop via ``_call_soon`` (the loop and its ``_outbox``
        are single-threaded on that loop; ``_drain_outbox`` itself filters to due requests)."""
        terminal = {"completed", "failed", "limited"}
        with self._lock:
            live = [
                (record, record.loop)
                for record in self._records.values()
                if record.loop is not None
                and record.outbox_sender is not None
                and record.status not in terminal
            ]
        for record, loop in live:
            self._call_soon(self._drain_outbox, record, loop)

    def _heartbeat_own_runs(self) -> None:
        assert self.lease_store is not None
        with self._lock:
            items = list(self._records.items())
        for run_id, record in items:
            try:
                if record.status in {"completed", "failed", "limited"}:
                    # Terminal: drop the lease so no watchdog ever reclaims a finished run.
                    self.lease_store.release(run_id)
                else:
                    self.lease_store.heartbeat(run_id, self._worker_id, self.lease_ttl_s)
            except Exception:  # pragma: no cover - one bad run must not stop heartbeating others
                pass

    def _reclaim_stale_runs(self) -> list[str]:
        """Reclaim runs whose owning worker crashed (lease stale or absent). A live peer's
        run carries a fresh lease and is left untouched; the claim is a cross-process CAS so
        two watchdogs racing the same run produce exactly one winner. Candidate runs come
        from the lease store, so a shared store surfaces a peer's runs we never hosted."""
        assert self.lease_store is not None
        reclaimed: list[str] = []
        for run_id in sorted(self.lease_store.candidate_run_ids()):
            with self._lock:
                if run_id in self._records:
                    continue
            run_dir = self.run_root / run_id
            if (run_dir / "failure.json").exists():
                continue
            if not self.lease_store.is_stale(run_id):
                continue  # a live peer owns it
            if not self.lease_store.try_claim(run_id, self._worker_id, self.lease_ttl_s):
                continue  # lost the CAS to another watchdog
            if self._attempt_resume(run_dir, run_id):
                _LOGGER.info("watchdog: reclaimed orphaned run %s", run_id)
                reclaimed.append(run_id)
            elif not (run_dir / "failure.json").exists():
                # Resume failed but the attempt cap has not yet marked it unrecoverable.
                # Release our just-claimed lease so the run is retried next tick (or by a
                # peer) instead of being stranded behind a fresh lease that never resumes.
                self.lease_store.release(run_id)
        return reclaimed

    def _resume_from_checkpoint(self, stored: CheckpointRecord, meta: dict[str, Any]) -> None:
        checkpoint = stored.checkpoint
        run_id = checkpoint.run_id
        run_dir = self.run_root / run_id
        runtime_config = _runtime_config_from_meta(meta)
        limits = meta.get("limits") or {}
        request = BackendRunRequest(
            tenant_id=str(meta["tenant_id"]),
            user_id=str(meta["user_id"]),
            workspace_root=Path(meta["workspace_root"]),
            instruction="",  # the opening turn already ran; recovery resumes from the checkpoint
            mode=meta.get("mode", "propose"),
            workspace_backend=meta.get("workspace_backend", "overlay"),
            max_steps=int(limits.get("max_steps", 30)),
            max_tool_calls=int(limits.get("max_tool_calls", 100)),
            max_bytes_read=int(limits.get("max_bytes_read", 1_000_000)),
            max_duration_s=limits.get("max_duration_s", 900),
            permission_policy=PermissionPolicy.from_json(meta.get("permission_policy")),
            runtime_config=runtime_config,
            multi_turn=bool(meta.get("multi_turn", False)),
        )
        workspace_root = request.workspace_root.resolve()
        # Re-issue gateway tokens — the backend holds the signing key, so the original
        # (unstored) tokens need not survive the restart.
        llm_gateway_token = self.token_manager.issue(
            kind="llm_gateway",
            audience="csp.llm-gateway",
            run_id=run_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            ttl_s=self.llm_gateway_token_ttl_s,
            metadata={"agent_config_hash": runtime_config.config_hash},
        )
        web_gateway_token = ""
        if _runtime_config_uses_web(runtime_config) and self.web_gateway_url:
            web_gateway_token = self.token_manager.issue(
                kind="web_gateway",
                audience="csp.web-gateway",
                run_id=run_id,
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                ttl_s=self.web_gateway_token_ttl_s,
                metadata={"agent_config_hash": runtime_config.config_hash},
            )
        record = BackendRunRecord(
            run_id=run_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            workspace_root=workspace_root,
            run_dir=run_dir,
            status="awaiting_input",
            created_at=time.time(),
            run_token_sha256="",  # the client still holds its run token (verified cryptographically)
            llm_gateway_token_sha256=TokenManager.token_sha256(llm_gateway_token),
            web_gateway_token_sha256=TokenManager.token_sha256(web_gateway_token) if web_gateway_token else "",
            runtime_config=runtime_config,
            runtime_config_issuer=str(meta.get("runtime_config_issuer") or "recover_runs"),
            runtime_config_reason=str(meta.get("runtime_config_reason") or "resumed from checkpoint"),
            runtime_config_committed_at=float(meta.get("runtime_config_committed_at") or time.time()),
        )
        with self._lock:
            self._records[run_id] = record
        spec = self._run_spec_for_request(run_id, request, workspace_root)
        adapter = self._build_model_adapter(
            spec,
            llm_gateway_token,
            runtime_config.model,
            token_provider=self._llm_token_source(run_id, request, runtime_config),
        )
        loop = AgentLoop(
            spec=spec,
            model_adapter=adapter,
            event_sinks=(BackendRunStateSink(self, run_id), *(make() for make in self.extra_event_sink_factories)),
            permission_policy=request.permission_policy,
            cancellation_token=record.cancellation_token,
            shell_approval_provider=None,
            web_gateway_client=self._web_gateway_client(web_gateway_token),
            runtime_config_provider=BackendRuntimeConfigProvider(self, run_id),
            checkpoint_store=self.checkpoint_store,
            emit_output_deltas=self.emit_output_deltas,
            subagent_definitions=self.subagent_definitions,
            tool_providers=self.tool_providers,
            context_providers=self.context_providers,
            output_validators=self.output_validators,
            capability_broker=self._capability_broker_for(request),
        )
        # The base workspace is re-provisioned by the deployment (re-mount/re-clone);
        # restore re-applies the agent's delta from the checkpoint's content blobs.
        loop.restore(checkpoint, blobs=stored.blob)
        with self._lock:
            record.loop = loop
            record.outbox_sender = self._outbox_sender_for(request)
        # Restore the inbox dedup set so a message processed before the restart is not reprocessed
        # if it (or a redelivery) is queued again.
        record.seen_inbox_ids = set(checkpoint.inbox_seen_ids)
        # Re-enqueue durable pending messages on the shared loop (before the resume coroutine
        # drains them); asyncio.Queue puts must run on the loop, not this thread.
        for message in checkpoint.queued_messages:
            self._call_soon(record.message_queue.put_nowait, message)
        # Resume executes as a coroutine on the shared loop (coroutine-per-run).
        self._spawn(self._run_recovered(run_id, request, loop))

    async def _run_recovered(self, run_id: str, request: BackendRunRequest, loop: AgentLoop) -> None:
        record = self._record(run_id)
        if self._run_semaphore is not None:
            await self._run_semaphore.acquire()
        try:
            # Derive the starting park from the restored loop: tasks still pending -> a
            # hosted-task wait; otherwise a settled park awaiting the next user message.
            if loop.has_pending_tasks():
                suspension = Suspension(reason="awaiting_tasks", status="running", has_external=True)
            else:
                suspension = Suspension(reason="settled", status="completed")
            result = await self._drive_open_session(
                record, request, loop, suspension, started=time.time(), turns=1
            )
            self._record_run_result(run_id, result)
        except Exception as exc:
            self._record_run_failure(run_id, exc)
        finally:
            if self._run_semaphore is not None:
                self._run_semaphore.release()

    def _llm_token_source(
        self, run_id: str, request: BackendRunRequest, runtime_config: AgentRuntimeConfig | None
    ) -> _GatewayTokenSource:
        """A re-minting source for the run's ``llm_gateway`` token (mirrors the eager issue + the
        recovery re-issue), so a long run keeps LLM access past the token TTL without a restart."""
        return _GatewayTokenSource(
            token_manager=self.token_manager,
            kind="llm_gateway",
            audience="csp.llm-gateway",
            run_id=run_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            ttl_s=self.llm_gateway_token_ttl_s,
            metadata={"agent_config_hash": runtime_config.config_hash} if runtime_config is not None else {},
        )

    def _build_model_adapter(
        self,
        spec: AgentRunSpec,
        llm_gateway_token: str,
        model_config: ModelConfig | None,
        token_provider: Callable[[], str | None] | None = None,
    ) -> ModelAdapter:
        if self.model_adapter_factory is not None:
            # Custom factories own their credential lifecycle (they get the freshly-minted token
            # string); the refresh seam applies to the default gateway adapter only.
            return self.model_adapter_factory(spec, llm_gateway_token)
        return GatewayModelAdapter(
            model_config or ModelConfig(),
            gateway_url=self.llm_gateway_url,
            token=llm_gateway_token,
            token_provider=token_provider,
        )

    def _web_gateway_client(
        self,
        token: str,
    ) -> WebGatewayClient | None:
        if not token:
            return None
        return WebGatewayClient(self.web_gateway_url, token=token)

    def _validate_request(self, request: BackendRunRequest) -> None:
        if not request.tenant_id.strip():
            raise ValueError("tenant_id is required")
        if not request.user_id.strip():
            raise ValueError("user_id is required")
        if not request.instruction.strip() and not request.input_parts:
            raise ValueError("instruction or input_parts is required")
        if request.mode not in {"read-only", "propose", "apply"}:
            raise ValueError(f"unsupported mode: {request.mode}")
        if request.workspace_backend not in {"overlay", "staging"}:
            raise ValueError(f"unsupported workspace_backend: {request.workspace_backend}")
        if request.agent_definition is None and request.runtime_config is None:
            raise ValueError("agent_definition or runtime_config is required")

    def _check_workspace_allowed(self, workspace_root: Path) -> None:
        if not workspace_root.exists() or not workspace_root.is_dir():
            raise ValueError(f"workspace root does not exist: {workspace_root}")
        if not any(is_within(root, workspace_root) for root in self.allowed_workspace_roots):
            raise PermissionDenied(f"workspace root is outside allowed roots: {workspace_root}")

    def _authorize_run(self, run_id: str, token: str) -> None:
        claims = self._verify_run_token(run_id, token)
        record = self._record(run_id)
        if claims.tenant_id != record.tenant_id or claims.user_id != record.user_id:
            raise PermissionDenied("token subject mismatch")

    def _verify_run_token(self, run_id: str, token: str) -> Any:
        """Verify a run-access token for ``run_id`` (signature/kind/audience/run id), returning
        its claims. Does NOT require an in-memory record — the signed token is the capability."""
        try:
            return self.token_manager.verify(
                token, kind="run_access", audience=BACKEND_AUDIENCES, run_id=run_id
            )
        except TokenError as exc:
            raise PermissionDenied(str(exc)) from exc

    def _authorized_run_dir(self, run_id: str, token: str) -> Path:
        """Resolve a run's directory for a token-authorized READ — live or historical.

        A live run is checked against its in-memory record (as :meth:`_authorize_run` does). A run
        with no record (e.g. after a restart, surfaced by :meth:`list_runs`) is authorized on the
        signed run token's own claims and read straight from run_root. Rejects path separators so
        a crafted run id can't escape run_root."""
        claims = self._verify_run_token(run_id, token)
        with self._lock:
            record = self._records.get(run_id)
        if record is not None:
            if claims.tenant_id != record.tenant_id or claims.user_id != record.user_id:
                raise PermissionDenied("token subject mismatch")
            return record.run_dir
        if any(sep in run_id for sep in ("/", "\\")) or ".." in run_id:
            raise PermissionDenied("invalid run id")
        return self.run_root / run_id

    def list_runs(
        self, tenant_id: str, *, user_id: str | None = None, limit: int = 100
    ) -> dict[str, Any]:
        """List a tenant's runs from ``run_root`` (newest first) — the durable history.

        A trusted-host call (no token, like :meth:`recover_runs`): the embedder owns the run_root
        and is responsible for tenant scoping. Reads each run.json (skipping subagent child runs,
        which have none) for identity + title + created_at, takes the live status from an
        in-memory record when present else status.json, and mints a fresh read-scoped run token
        per entry so the caller can fetch events/status afterwards (mirrors submit_run returning a
        run token). ``recoverable`` flags a parked run a restart can resume via recover_runs."""
        runs: list[dict[str, Any]] = []
        if not self.run_root.is_dir():
            return {"runs": runs}
        for run_dir in self.run_root.iterdir():
            meta_path = run_dir / "run.json"
            if not run_dir.is_dir() or not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if meta.get("tenant_id") != tenant_id:
                continue
            run_user = meta.get("user_id") or ""
            if user_id is not None and run_user != user_id:
                continue
            run_id = meta.get("run_id") or run_dir.name
            with self._lock:
                record = self._records.get(run_id)
            if record is not None:
                status = record.status
            else:
                status = "ended"
                status_path = run_dir / "status.json"
                if status_path.exists():
                    try:
                        status = json.loads(read_text_resilient(status_path)).get("status", "ended")
                    except (ValueError, OSError):
                        pass
            recoverable = False
            if record is None and not (run_dir / "failure.json").exists() and self.checkpoint_store is not None:
                stored = self.checkpoint_store.latest(run_id)
                recoverable = stored is not None and not stored.checkpoint.terminal
            runs.append(
                {
                    "run_id": run_id,
                    "tenant_id": tenant_id,
                    "user_id": run_user,
                    "title": meta.get("title") or "",
                    "created_at": meta.get("created_at") or 0.0,
                    "status": status,
                    "recoverable": recoverable,
                    "read_token": self.token_manager.issue(
                        kind="run_access",
                        audience=BACKEND_AUDIENCE,
                        run_id=run_id,
                        tenant_id=tenant_id,
                        user_id=run_user,
                        ttl_s=self.run_token_ttl_s,
                    ),
                }
            )
        runs.sort(key=lambda entry: entry["created_at"], reverse=True)
        return {"runs": runs[:limit]}

    def _record(self, run_id: str) -> BackendRunRecord:
        with self._lock:
            try:
                return self._records[run_id]
            except KeyError as exc:
                raise KeyError(f"unknown run: {run_id}") from exc

    def _read_proposal(self, record: BackendRunRecord) -> dict[str, Any] | None:
        proposal_path = record.run_dir / "proposal.json"
        if not proposal_path.exists():
            return None
        payload = json.loads(proposal_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("proposal snapshot must be a JSON object")
        return payload
