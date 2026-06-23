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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from native_agent_runner.core.agents import (
    AgentDefinition,
    AgentRuntimeConfig,
    RuntimeConfigProvider,
    SubagentDefinition,
    validate_runtime_config,
)
from native_agent_runner.core.cancellation import CancellationToken
from native_agent_runner.core.events import AgentEvent
from native_agent_runner.core.packages import (
    apply_package,
    create_approval,
    export_package,
    write_apply_result,
    write_approval,
)
from native_agent_runner.core._util import write_json_atomic
from native_agent_runner.core.checkpoint import (
    CheckpointRecord,
    CheckpointStore,
    LocalFsCheckpointStore,
)
from native_agent_runner.reference.stores.lease import LeaseStore, LocalFsLeaseStore
from native_agent_runner.core.proposal_file import ProposalFileError, read_proposal_file_payload
from native_agent_runner.core.result import AgentRunResult, Suspension
from native_agent_runner.core.content import (
    ContentPart,
    content_part_from_json,
    content_part_to_json,
)
from native_agent_runner.core.spec import (
    AgentRunSpec,
    ModelConfig,
    ModelRetryConfig,
    RunLimits,
    RunMode,
    WorkspaceBackendKind,
)
from native_agent_runner.core.workspace import Workspace
from native_agent_runner.errors import NativeAgentError, PermissionDenied
from native_agent_runner.tasks import (
    get_job_artifact,
    list_job_artifacts,
    read_job_log_text,
    request_job_cancel,
)
from native_agent_runner.loop import AgentLoop
from native_agent_runner.permissions import PermissionPolicy
from native_agent_runner.providers.base import ModelAdapter
from native_agent_runner.providers.gateway import GatewayModelAdapter
from native_agent_runner.reference._shared.tokens import TokenError, TokenManager
from native_agent_runner.recorder import append_event_to_run
from native_agent_runner.tools.builtin import agent_spawn_tool, builtin_tools
from native_agent_runner.web import WebGatewayClient
from native_agent_runner.workspace.paths import is_within

BackendRunState = Literal["queued", "running", "awaiting_input", "completed", "failed", "limited"]

# Sentinel enqueued to wake/stop a session worker blocked on its message queue.
_CLOSE_SESSION = object()
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
    """Convert a dequeued backend message (text ``str`` or ``list[dict]`` of part dicts) into a
    loop ``submit`` input. ``content_part_from_json`` rebuilds the typed parts at the boundary."""
    if isinstance(message, list):
        return tuple(content_part_from_json(part) for part in message)
    return message  # str

# Durable recovery descriptor (run.json) — what recover_runs needs to rebuild a parked run.
_RUN_META_SCHEMA_VERSION = "native-agent-runner.backend-run.v1"

_LOGGER = logging.getLogger("native_agent_runner.backend")


def _read_run_meta(run_dir: Path) -> dict[str, Any] | None:
    """Read run.json if present and schema-compatible; ``None`` otherwise (never raises)."""
    try:
        payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != _RUN_META_SCHEMA_VERSION:
        return None
    return payload


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
    last_event_seq: int = 0
    last_event_type: str = ""
    cancellation_token: CancellationToken = field(default_factory=CancellationToken)
    runtime_config: AgentRuntimeConfig | None = None
    runtime_config_issuer: str = ""
    runtime_config_reason: str = ""
    loop: AgentLoop | None = None
    # Pending user messages for a multi-turn session. asyncio.Queue (not queue.Queue) so the
    # run coroutine awaits the next message WITHOUT holding a thread — a parked multi-turn
    # session is just a suspended coroutine, not a blocked worker (which would exhaust the
    # shared executor). Producers (send_message/cancel from other threads) enqueue via the
    # backend's _call_soon so the put runs on the loop. Created without a running loop (3.10+
    # binds lazily); all gets/puts happen on the shared loop.
    message_queue: asyncio.Queue[Any] = field(default_factory=asyncio.Queue, repr=False)


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
) -> tuple[Any, ...]:
    specs = list(builtin_tools(cast(Workspace, None)))
    # agent.spawn is registered dynamically by the loop bootstrap (only when the run carries
    # subagent_definitions), so config validation must know about it too when they're present —
    # otherwise a binding to agent.spawn looks like an unknown tool.
    if subagent_definitions:
        catalog = {sid: d.description for sid, d in subagent_definitions.items()}
        specs.append(agent_spawn_tool(catalog))
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
            audience="native-agent-runner.backend",
            run_id=run_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            ttl_s=self.run_token_ttl_s,
        )
        tool_specs = _backend_builtin_tool_specs(self.subagent_definitions)
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
        record = BackendRunRecord(
            run_id=run_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            workspace_root=workspace_root,
            run_dir=run_dir,
            status="queued",
            created_at=time.time(),
            run_token_sha256=TokenManager.token_sha256(run_token),
            llm_gateway_token_sha256=TokenManager.token_sha256(llm_gateway_token),
            web_gateway_token_sha256=TokenManager.token_sha256(web_gateway_token) if web_gateway_token else "",
            runtime_config=initial_runtime_config,
            runtime_config_issuer=runtime_config_issuer,
            runtime_config_reason=runtime_config_reason,
        )
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
            status_payload = json.loads(status_file.read_text(encoding="utf-8"))
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

    def send_message(
        self, run_id: str, token: str, content: str | Sequence[Any]
    ) -> dict[str, Any]:
        """Deliver a follow-up user message to a running multi-turn session. It is queued and
        consumed as the next user turn once the current turn settles.

        ``content`` is either a plain ``str`` or a sequence of content parts (``ContentPart``
        dataclasses or their JSON dicts) for a multimodal message. Multimodal parts reference
        workspace files by ``source_ref`` — the bytes are resolved at wire-build time — so the
        message itself stays small; the size limit applies to the by-reference wire form."""
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        message = _normalize_inbound_message(content)
        wire_bytes = len(
            (message if isinstance(message, str) else json.dumps(message)).encode("utf-8")
        )
        if wire_bytes > self.max_message_bytes:
            raise ValueError(f"message exceeds the {self.max_message_bytes}-byte limit")
        with self._lock:
            if record.status in {"completed", "failed", "limited"}:
                raise ValueError("cannot send a message to a terminal run")
            if record.message_queue.qsize() >= self.max_message_queue_depth:
                raise ValueError("message queue is full; retry once the run drains it")
        # Enqueue on the shared loop (asyncio.Queue is not thread-safe across threads).
        self._call_soon(record.message_queue.put_nowait, message)
        return {"run_id": run_id, "status": "queued"}

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
                }
            return {
                "run_id": record.run_id,
                "tenant_id": record.tenant_id,
                "ready": True,
                "issuer": record.runtime_config_issuer,
                "reason": record.runtime_config_reason,
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
        validate_runtime_config(config, _backend_builtin_tool_specs(self.subagent_definitions))
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
                config = AgentRuntimeConfig(
                    definition_id=config.definition_id,
                    config_version=current_version + 1,
                    model=config.model,
                    prompt=config.prompt,
                    tools=config.tools,
                    tool_search=config.tool_search,
                    metadata=config.metadata,
                )
            record.runtime_config = config
            record.runtime_config_issuer = issuer
            record.runtime_config_reason = reason
            return {
                "run_id": record.run_id,
                "tenant_id": record.tenant_id,
                "ready": True,
                "issuer": issuer,
                "reason": reason,
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
            audience="native-agent-runner.task-callback",
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
            claims = self.token_manager.verify(
                token, kind="task_callback", audience="native-agent-runner.task-callback", run_id=run_id
            )
            if str(claims.metadata.get("task_id") or "") != task_id:
                raise PermissionDenied("callback token does not match this task")
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

    def events(self, run_id: str, token: str, *, from_seq: int = 0) -> dict[str, Any]:
        events_path = self._authorized_run_dir(run_id, token) / "events.jsonl"
        events: list[dict[str, Any]] = []
        if events_path.exists():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if int(event.get("seq") or 0) >= from_seq:
                    events.append(event)
        return {"run_id": run_id, "events": events}

    def descendant_events(
        self, run_id: str, token: str, descendant_run_id: str, *, from_seq: int = 0
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
        events: list[dict[str, Any]] = []
        if events_path.exists():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if int(event.get("seq") or 0) >= from_seq:
                    events.append(event)
        return {"run_id": descendant_run_id, "events": events}

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
                    message = await asyncio.wait_for(record.message_queue.get(), self.idle_timeout_s)
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
                    message = await asyncio.wait_for(record.message_queue.get(), self.idle_timeout_s)
                except asyncio.TimeoutError:
                    break
                if message is _CLOSE_SESSION:
                    break
                turns += 1
                suspension = await loop.arun_until_suspended(_queued_message_to_loop_input(message))
                continue
            # settled. One-shot runs close here; multi-turn awaits the next message.
            consecutive_turn_failures = 0  # a settled turn clears the failure streak
            if not request.multi_turn:
                break
            if self._session_should_stop(record, started, turns):
                break
            loop.await_user_input()
            self._persist_run_checkpoint(record)
            try:
                # Pure async await — a parked multi-turn session holds no thread.
                message = await asyncio.wait_for(record.message_queue.get(), self.idle_timeout_s)
            except asyncio.TimeoutError:
                break  # idle timeout
            if message is _CLOSE_SESSION:
                break
            turns += 1
            suspension = await loop.arun_until_suspended(_queued_message_to_loop_input(message))
        return await loop.aclose()

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
            if isinstance(message, (str, list))  # str text or list[dict] parts; drop the close sentinel
        ]
        checkpoint.queued_messages = residual
        # Overwrites the same seq the loop just committed, now with the queue included.
        assert self.checkpoint_store is not None
        self.checkpoint_store.put(checkpoint, loop.collect_checkpoint_blobs())

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
            "schema_version": "native-agent-runner.failure.v1",
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
        tokens are re-issued on recovery, not stored). A mid-run runtime-config change
        is not re-persisted here, so recovery uses the config as of run start."""
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
            "created_at": time.time(),
            "title": " ".join((request.instruction or "").split())[:80],
            "limits": {
                "max_steps": request.max_steps,
                "max_tool_calls": request.max_tool_calls,
                "max_bytes_read": request.max_bytes_read,
                "max_duration_s": request.max_duration_s,
            },
            "permission_policy": request.permission_policy.to_json(),
            "runtime_config": record.runtime_config.to_json() if record.runtime_config else None,
        }
        record.run_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(record.run_dir / "run.json", meta)

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
        meta = _read_run_meta(run_dir)
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
        meta = _read_run_meta(run_dir)
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
            name=f"native-agent-watchdog-{self._worker_id[:8]}",
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
            except Exception:  # pragma: no cover - the watchdog must never die on a tick
                _LOGGER.exception("watchdog tick failed")

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
        runtime_config = AgentRuntimeConfig.from_json(meta["runtime_config"])
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
            runtime_config_issuer="recover_runs",
            runtime_config_reason="resumed from checkpoint",
        )
        with self._lock:
            self._records[run_id] = record
        spec = self._run_spec_for_request(run_id, request, workspace_root)
        adapter = self._build_model_adapter(spec, llm_gateway_token, runtime_config.model)
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
        )
        # The base workspace is re-provisioned by the deployment (re-mount/re-clone);
        # restore re-applies the agent's delta from the checkpoint's content blobs.
        loop.restore(checkpoint, blobs=stored.blob)
        with self._lock:
            record.loop = loop
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

    def _build_model_adapter(
        self,
        spec: AgentRunSpec,
        llm_gateway_token: str,
        model_config: ModelConfig | None,
    ) -> ModelAdapter:
        if self.model_adapter_factory is not None:
            return self.model_adapter_factory(spec, llm_gateway_token)
        return GatewayModelAdapter(model_config or ModelConfig(), gateway_url=self.llm_gateway_url, token=llm_gateway_token)

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
                token, kind="run_access", audience="native-agent-runner.backend", run_id=run_id
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
                        status = json.loads(status_path.read_text(encoding="utf-8")).get("status", "ended")
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
                        audience="native-agent-runner.backend",
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
