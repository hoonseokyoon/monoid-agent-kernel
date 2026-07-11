from __future__ import annotations

import asyncio
import atexit
import logging
import random
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import thread as _cf_thread
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

from monoid_agent_kernel.core.agents import (
    AgentRuntimeConfig,
    SubagentDefinition,
)
from monoid_agent_kernel.core.control import ControlCommand, ControlResult
from monoid_agent_kernel.core.durable_metadata import (
    ACCEPTED_RUN_METADATA_SCHEMA_VERSIONS,
    RUN_METADATA_SCHEMA_VERSION,
    DurableMetadataCommitter,
    read_run_metadata,
    validate_run_metadata,
)
from monoid_agent_kernel.core.capability import CapabilityBroker
from monoid_agent_kernel.core.context import ContextProvider
from monoid_agent_kernel.core.events import AgentEvent, EventSink
from monoid_agent_kernel.core.event_subscription import EventSubscription, SequenceCursor
from monoid_agent_kernel.core.event_sequencing import (
    RunEventSequencer,
)
from monoid_agent_kernel.core.outbox import OutboxSender, OutboxReceipt
from monoid_agent_kernel.core.output_validator import OutputValidator
from monoid_agent_kernel.core.lifecycle import (
    SessionState,
)
from monoid_agent_kernel.core.checkpoint import (
    CheckpointRecord,
    CheckpointStore,
    LocalFsCheckpointStore,
    RunCheckpoint,
)
from monoid_agent_kernel.reference.stores.lease import LeaseStore, LocalFsLeaseStore
from monoid_agent_kernel.core.result import AgentRunResult, Suspension
from monoid_agent_kernel.core.spec import (
    AgentRunSpec,
    ModelConfig,
    ModelRetryConfig,
)
from monoid_agent_kernel.core.workspace import Workspace
from monoid_agent_kernel.errors import NativeAgentError, PermissionDenied
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.providers.base import ModelAdapter
from monoid_agent_kernel.identifiers import (
    BACKEND_AUDIENCE,
    BACKEND_AUDIENCES,
    TASK_CALLBACK_AUDIENCE,
    TASK_CALLBACK_AUDIENCES,
)
from monoid_agent_kernel.reference._shared.tokens import TokenError, TokenManager
from monoid_agent_kernel.reference.command_inbox import (
    CommandPrincipal,
    CommandReceipt,
    CommandStore,
    InMemoryCommandStore,
    StoredCommand,
    redact_command_credential,
    sanitize_command_data,
)
from monoid_agent_kernel.reference.backend.commands import BackendCommandContext, BackendCommandService
from monoid_agent_kernel.reference.backend.jobs import JobService, JobServiceContext
from monoid_agent_kernel.reference.backend.loop_factory import (
    BackendLoopBuild,
    BackendLoopFactory,
    BackendLoopFactoryContext,
    ModelAdapterFactory,
    _GatewayTokenSource,
)
from monoid_agent_kernel.reference.backend.outbox_dispatch import (
    OutboxDispatchContext,
    OutboxDispatchService,
    OutboxRetryPolicy,
)
from monoid_agent_kernel.reference.backend.projection import (
    RunProjectionContext,
    RunProjectionService,
    _json_safe as _json_safe,
)
from monoid_agent_kernel.reference.backend.proposal import ProposalService, ProposalServiceContext
from monoid_agent_kernel.reference.backend.proposal_reader import read_proposal_snapshot
from monoid_agent_kernel.reference.backend.recovery import RecoveryContext, RecoveryService
from monoid_agent_kernel.reference.backend.runtime_config import RuntimeConfigContext, RuntimeConfigService
from monoid_agent_kernel.reference.backend.run_execution import (
    RunExecutionContext,
    RunExecutionService,
    stream_item_frame,
)
from monoid_agent_kernel.reference.backend.run_preparation import (
    RunPreparationContext,
    RunPreparationService,
    runtime_config_uses_web as _runtime_config_uses_web,
)
from monoid_agent_kernel.reference.backend.run_state import (
    RunStateMutationContext,
    RunStateMutationService,
    record_terminal as _record_terminal,
)
from monoid_agent_kernel.reference.backend.run_types import (
    BackendRunRecord,
    BackendRunRequest,
    BackendRunSubmission,
    _PreparedRun,
)
from monoid_agent_kernel.reference.backend.session import (
    BackendSessionContext,
    BackendSessionService,
    _normalize_inbound_message as _normalize_inbound_message,
)
from monoid_agent_kernel.reference.backend.session_drive import (
    SessionDriveContext,
    SessionDriveLimits,
    SessionDriveService,
    _queued_message_to_loop_input as _queued_message_to_loop_input,
)
from monoid_agent_kernel.recorder import append_event_to_run
from monoid_agent_kernel.tools.base import ToolProvider
from monoid_agent_kernel.tools.builtin import agent_spawn_tool, builtin_tools
from monoid_agent_kernel.web import WebGatewayClient

# Sentinels enqueued to wake/stop a session worker blocked on its message queue.
_CLOSE_SESSION = object()
# Wakes a paused worker: resume the SAME turn with no new input. Ignored (a no-op) by the other
# queue-waiting branches, which expect a real user message or _CLOSE_SESSION.
_RESUME_SESSION = object()

# Durable recovery descriptor (run.json) — what recover_runs needs to rebuild a parked run.
_RUN_META_SCHEMA_VERSION = RUN_METADATA_SCHEMA_VERSION
_ACCEPTED_RUN_META_SCHEMA_VERSIONS = ACCEPTED_RUN_METADATA_SCHEMA_VERSIONS
_RUN_EVENT_SEQUENCER = RunEventSequencer()

_LOGGER = logging.getLogger("monoid_agent_kernel.backend")


def _read_run_meta(run_dir: Path) -> dict[str, Any] | None:
    """Read run.json if present and schema-compatible; ``None`` otherwise (never raises)."""
    return read_run_metadata(run_dir)


def _validate_run_meta(payload: Any) -> dict[str, Any] | None:
    return validate_run_metadata(payload)


def _backend_builtin_tool_specs(
    subagent_definitions: Mapping[str, SubagentDefinition] | None = None,
    tool_providers: Sequence[ToolProvider] = (),
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
    if loop.is_running():
        async def _cancel_pending_tasks() -> None:
            pending = [
                task
                for task in asyncio.all_tasks(loop)
                if task is not asyncio.current_task(loop) and not task.done()
            ]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(_cancel_pending_tasks(), loop).result(timeout=5)
        except Exception:  # pragma: no cover - interpreter shutdown best-effort cleanup
            pass
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


@dataclass
class RunnerBackend:
    """Reference backend facade and composition root.

    Public methods stay on this object. Internal services receive explicit context
    objects, while process-level runtime ownership, loop construction, and the
    remaining product-specific Reference surfaces stay here until their own
    extraction step. See docs/RUNNER_BACKEND_RESPONSIBILITY_MAP.md for the current
    responsibility map.
    """

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
    extra_event_sink_factories: tuple[Callable[[], EventSink], ...] = ()
    # Tool/context providers attached to every run the backend builds (Skills, MCP, custom).
    # The embedder-facing seam for the loop's tool_providers/context_providers (the CLI passes
    # these to AgentLoop directly; without these fields an out-of-process embedder could not
    # attach a provider at all). INSTANCES, not factories (unlike extra_event_sink_factories):
    # a provider holds a shared, reusable resource (MCP's live httpx client + discovery cache)
    # or is immutable (SkillProvider) — both are safe to share across concurrent runs (the MCP
    # client is documented thread-safe; SkillProvider is read-only). Read at loop-build time so
    # a parked run re-attaches them on resume/restart. Their tools must also be declared to
    # config validation — see _backend_builtin_tool_specs. Empty → no providers.
    tool_providers: tuple[ToolProvider, ...] = ()
    context_providers: tuple[ContextProvider, ...] = ()
    # Output validators attached to every run the backend builds. Default-on: each runs unless a
    # run's config disables it via OutputValidatorBinding(enabled=False). Read at loop-build time
    # so a parked run re-attaches them on resume/restart, exactly like tool/context providers.
    # Empty → no validators.
    output_validators: tuple[OutputValidator, ...] = ()
    # Per-run capability broker factory: ``(request) -> CapabilityBroker | None``. Called at
    # loop-build time so a broker can be scoped to the run's identity (tenant/user/run id) — e.g.
    # a GatewayCapabilityBroker minting per-tenant tokens. None (or a None return) leaves capability
    # gating off for that run. A factory (not an instance) because a broker is typically per-run
    # identity-bound, unlike the shared tool/context providers above.
    capability_broker_factory: Callable[[BackendRunRequest], CapabilityBroker | None] | None = None
    # Per-run outbox sender (drains staged outbound sends at the edge — see core/outbox.py). A
    # factory like capability_broker_factory; None (or a None return) leaves staged requests pending
    # (durable, never dispatched). The drain performs the actual IO; the core only stages.
    outbox_sender_factory: Callable[[BackendRunRequest], OutboxSender | None] | None = None
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
    # Durable control-command transport. Use SqliteCommandStore on every instance for
    # cross-worker delivery; the in-memory default preserves single-process simplicity.
    command_store: CommandStore | None = None
    command_queue_limit: int = 100
    command_claim_ttl_s: float = 30.0
    _records: dict[str, BackendRunRecord] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _worker_id: str = field(default="", init=False, repr=False)
    _watchdog_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _watchdog_stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _run_semaphore: asyncio.BoundedSemaphore | None = field(default=None, init=False, repr=False)
    # RNG for the outbox backoff jitter — a dedicated instance so a test can seed it deterministically
    # (backend._outbox_rng.seed(...)) without perturbing global random state.
    _outbox_rng: random.Random = field(default_factory=random.Random, init=False, repr=False)
    _loop_factory: BackendLoopFactory = field(init=False, repr=False)
    _projection: RunProjectionService = field(init=False, repr=False)
    _proposal: ProposalService = field(init=False, repr=False)
    _runtime_config: RuntimeConfigService = field(init=False, repr=False)
    _jobs: JobService = field(init=False, repr=False)
    _recovery: RecoveryService = field(init=False, repr=False)
    _run_preparation: RunPreparationService = field(init=False, repr=False)
    _run_state: RunStateMutationService = field(init=False, repr=False)
    _outbox_dispatch: OutboxDispatchService = field(init=False, repr=False)
    _run_execution: RunExecutionService = field(init=False, repr=False)
    _session_boundary: BackendSessionService = field(init=False, repr=False)
    _session_drive: SessionDriveService = field(init=False, repr=False)
    _commands: BackendCommandService = field(init=False, repr=False)

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
        if self.command_store is None:
            self.command_store = InMemoryCommandStore()
        if self.command_queue_limit < 1 or self.command_claim_ttl_s <= 0:
            raise ValueError("command queue limit and claim ttl must be positive")
        self._run_preparation = self._build_run_preparation_service()
        self._run_state = self._build_run_state_mutation_service()
        self._loop_factory = self._build_loop_factory()
        self._projection = self._build_projection_service()
        self._proposal = self._build_proposal_service()
        self._runtime_config = self._build_runtime_config_service()
        self._jobs = self._build_job_service()
        self._recovery = self._build_recovery_service()
        self._outbox_dispatch = self._build_outbox_dispatch_service()
        assert self.checkpoint_store is not None
        self._session_drive = self._build_session_drive_service()
        self._run_execution = self._build_run_execution_service()
        self._session_boundary = self._build_session_boundary_service()
        self._commands = self._build_command_service()

    # --- Internal service context providers --------------------------------------------

    def _build_run_preparation_service(self) -> RunPreparationService:
        return RunPreparationService(
            RunPreparationContext(
                run_root_provider=lambda: self.run_root,
                allowed_workspace_roots_provider=lambda: self.allowed_workspace_roots,
                token_manager_provider=lambda: self.token_manager,
                run_token_ttl_s_provider=lambda: self.run_token_ttl_s,
                llm_gateway_token_ttl_s_provider=lambda: self.llm_gateway_token_ttl_s,
                web_gateway_token_ttl_s_provider=lambda: self.web_gateway_token_ttl_s,
                web_gateway_url_provider=lambda: self.web_gateway_url,
                builtin_tool_specs_provider=lambda: _backend_builtin_tool_specs(
                    self.subagent_definitions,
                    self.tool_providers,
                ),
                checkpoint_store_provider=self._checkpoint_store,
                register_record=self._register_record,
                now=time.time,
            )
        )

    def _build_run_state_mutation_service(self) -> RunStateMutationService:
        return RunStateMutationService(
            RunStateMutationContext(
                with_record_lock=self._with_record_lock,
                active_record=self._active_record,
                record=self._record,
                run_root_provider=lambda: self.run_root,
                now=time.time,
                write_failure_bundle=lambda run_id, run_dir, **kwargs: self._write_failure_bundle(
                    run_id,
                    run_dir,
                    **kwargs,
                ),
                append_event=append_event_to_run,
                event_sequencer=_RUN_EVENT_SEQUENCER,
                logger=_LOGGER,
            )
        )

    def _build_loop_factory(self) -> BackendLoopFactory:
        return BackendLoopFactory(
            BackendLoopFactoryContext(
                run_root_provider=lambda: self.run_root,
                llm_gateway_url_provider=lambda: self.llm_gateway_url,
                web_gateway_url_provider=lambda: self.web_gateway_url,
                model_adapter_factory_provider=lambda: self.model_adapter_factory,
                token_manager_provider=lambda: self.token_manager,
                llm_gateway_token_ttl_s_provider=lambda: self.llm_gateway_token_ttl_s,
                checkpoint_store_provider=lambda: self.checkpoint_store,
                emit_output_deltas_provider=lambda: self.emit_output_deltas,
                extra_event_sink_factories_provider=lambda: self.extra_event_sink_factories,
                subagent_definitions_provider=lambda: self.subagent_definitions,
                tool_providers_provider=lambda: self.tool_providers,
                context_providers_provider=lambda: self.context_providers,
                output_validators_provider=lambda: self.output_validators,
                capability_broker_factory_provider=lambda: self.capability_broker_factory,
                outbox_sender_factory_provider=lambda: self.outbox_sender_factory,
                current_runtime_config=self.current_runtime_config,
                record=self._record,
                record_event=self.record_event,
                persist_checkpoint_payload=self._persist_run_checkpoint_payload,
            )
        )

    def _build_projection_service(self) -> RunProjectionService:
        return RunProjectionService(
            RunProjectionContext(
                authorized_run_dir=self._authorized_run_dir,
                authorize_run=self._authorize_run,
                record=self._record,
                active_record=self._active_record,
                read_recover_attempts=self._read_recover_attempts,
                run_root_provider=lambda: self.run_root,
                checkpoint_store_provider=lambda: self.checkpoint_store,
                max_recover_attempts_provider=lambda: self.max_recover_attempts,
                issue_read_token=self._issue_read_token,
            )
        )

    def _build_proposal_service(self) -> ProposalService:
        return ProposalService(
            ProposalServiceContext(
                authorize_run=self._authorize_run,
                record=self._record,
                checkpoint_store_provider=lambda: self.checkpoint_store,
                emit_backend_event=self._emit_backend_event,
                allowed_apply_roots_provider=lambda: self.allowed_apply_roots,
            )
        )

    def _build_runtime_config_service(self) -> RuntimeConfigService:
        return RuntimeConfigService(
            RuntimeConfigContext(
                authorize_run=self._authorize_run,
                record=self._record,
                with_record_lock=self._with_record_lock,
                checkpoint_store_provider=lambda: self.checkpoint_store,
                builtin_tool_specs_provider=lambda: _backend_builtin_tool_specs(
                    self.subagent_definitions,
                    self.tool_providers,
                ),
                now=time.time,
            )
        )

    def _build_job_service(self) -> JobService:
        return JobService(JobServiceContext(authorize_run=self._authorize_run, record=self._record))

    def _build_recovery_service(self) -> RecoveryService:
        return RecoveryService(
            RecoveryContext(
                run_root_provider=lambda: self.run_root,
                checkpoint_store_provider=lambda: self.checkpoint_store,
                lease_store_provider=lambda: self.lease_store,
                max_recover_attempts_provider=lambda: self.max_recover_attempts,
                worker_id_provider=lambda: self._worker_id,
                lease_ttl_s_provider=lambda: self.lease_ttl_s,
                is_record_tracked=self._is_live_run,
                record=self._record,
                make_request=self._recovery_request,
                make_record=self._recovery_record,
                issue_llm_gateway_token=self._issue_recovery_llm_gateway_token,
                issue_web_gateway_token=self._issue_recovery_web_gateway_token,
                build_loop=self._build_loop_build,
                register_record=self._register_recovered_record,
                attach_loop=self._attach_loop_build,
                call_soon=lambda fn, *args: self._call_soon(fn, *args),
                spawn=self._spawn,
                drive_open_session=self._drive_open_session,
                record_run_result=self._record_run_result,
                record_run_failure=self._record_run_failure,
                acquire_run_slot=self._acquire_run_slot,
                release_run_slot=self._release_run_slot,
            )
        )

    def _build_outbox_dispatch_service(self) -> OutboxDispatchService:
        return OutboxDispatchService(
            OutboxDispatchContext(
                retry_policy_provider=lambda: OutboxRetryPolicy(
                    max_attempts=self.outbox_max_attempts,
                    base_s=self.outbox_retry_base_s,
                    factor=self.outbox_retry_factor,
                    cap_s=self.outbox_retry_cap_s,
                ),
                max_message_queue_depth_provider=lambda: self.max_message_queue_depth,
                checkpoint_store_provider=self._checkpoint_store,
                rng_provider=lambda: self._outbox_rng,
                live_outbox_runs=self._live_outbox_runs,
                call_soon=lambda fn, *args: self._call_soon(fn, *args),
                record_terminal=lambda record: _record_terminal(record),
            )
        )

    def _build_session_drive_service(self) -> SessionDriveService:
        return SessionDriveService(
            SessionDriveContext(
                limits_provider=self._session_drive_limits,
                checkpoint_store_provider=self._checkpoint_store,
                drain_outbox=self._drain_outbox,
                close_signal=_CLOSE_SESSION,
                resume_signal=_RESUME_SESSION,
            )
        )

    def _build_run_execution_service(self) -> RunExecutionService:
        return RunExecutionService(
            RunExecutionContext(
                build_loop=self._build_loop_build,
                attach_loop=self._attach_loop_build,
                record=self._record,
                drive_open_session=self._drive_open_session,
                record_run_result=self._record_run_result,
                record_run_failure=self._record_run_failure,
                acquire_run_slot=self._acquire_run_slot,
                release_run_slot=self._release_run_slot,
                submission_json=lambda prepared: self._submission_for(prepared).to_json(),
            )
        )

    def _build_session_boundary_service(self) -> BackendSessionService:
        return BackendSessionService(
            BackendSessionContext(
                authorize_run=self._authorize_run,
                verify_run_token=self._verify_run_token,
                verify_task_callback_token=self._verify_task_callback_claims,
                issue_task_callback_token=self._issue_task_callback_token,
                record=self._record,
                active_record=self._active_record,
                run_dir_for=lambda run_id: self.run_root / run_id,
                call_soon=lambda fn, *args: self._call_soon(fn, *args),
                enqueue_message_and_checkpoint=self._enqueue_message_and_checkpoint,
                persist_checkpoint_from_any_thread=self._persist_run_checkpoint_from_any_thread,
                checkpoint_store_provider=lambda: self.checkpoint_store,
                read_recovery_meta=self._read_recovery_meta,
                attempt_resume=self._attempt_resume,
                max_message_bytes_provider=lambda: self.max_message_bytes,
                max_message_queue_depth_provider=lambda: self.max_message_queue_depth,
                record_terminal=self._record_terminal_locked,
                live_loop=self._live_loop_snapshot,
                mark_cancel_requested=self._mark_cancel_requested,
                ensure_message_enqueue_allowed=self._ensure_message_enqueue_allowed,
                close_signal=_CLOSE_SESSION,
                resume_signal=_RESUME_SESSION,
            )
        )

    def _build_command_service(self) -> BackendCommandService:
        return BackendCommandService(
            BackendCommandContext(
                emit_control_audit_event=self._emit_control_audit_event,
                verify_run_token=self._verify_run_token,
                verify_task_callback_token=self._verify_task_callback_token,
                authorize_claim_subject=self._authorize_claim_subject,
                is_live_run=self._is_live_run,
                active_loop_session=self._active_loop_session,
                run_control=self._session_boundary,
                task_messages=self._session_boundary,
                capability_control=self._session_boundary,
                projection=self._projection,
                runtime_config=self._runtime_config,
            )
        )

    def _session_drive_limits(self) -> SessionDriveLimits:
        return SessionDriveLimits(
            idle_timeout_s=self.idle_timeout_s,
            max_session_lifetime_s=self.max_session_lifetime_s,
            max_turns=self.max_turns,
            task_wait_poll_s=self.task_wait_poll_s,
            max_consecutive_turn_failures=self.max_consecutive_turn_failures,
            turn_retry=self.turn_retry,
        )

    def _checkpoint_store(self) -> CheckpointStore:
        assert self.checkpoint_store is not None
        return self.checkpoint_store

    def _with_record_lock(self, fn: Callable[[], Any]) -> Any:
        with self._lock:
            return fn()

    def _active_record(self, run_id: str) -> BackendRunRecord | None:
        with self._lock:
            return self._records.get(run_id)

    def _issue_read_token(self, run_id: str, tenant_id: str, user_id: str) -> str:
        return self.token_manager.issue(
            kind="run_access",
            audience=BACKEND_AUDIENCE,
            run_id=run_id,
            tenant_id=tenant_id,
            user_id=user_id,
            ttl_s=self.run_token_ttl_s,
        )

    def _is_live_run(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._records

    def _register_record(self, record: BackendRunRecord) -> None:
        with self._lock:
            self._records[record.run_id] = record

    def _recovery_request(self, meta: Mapping[str, Any], runtime_config: AgentRuntimeConfig) -> BackendRunRequest:
        limits = meta.get("limits") or {}
        return BackendRunRequest(
            tenant_id=str(meta["tenant_id"]),
            user_id=str(meta["user_id"]),
            workspace_root=Path(meta["workspace_root"]),
            instruction="",
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

    def _recovery_record(
        self,
        run_id: str,
        request: BackendRunRequest,
        workspace_root: Path,
        llm_gateway_token: str,
        web_gateway_token: str,
        runtime_config: AgentRuntimeConfig,
        meta: Mapping[str, Any],
    ) -> BackendRunRecord:
        return BackendRunRecord(
            run_id=run_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            workspace_root=workspace_root,
            run_dir=self.run_root / run_id,
            state=SessionState.AWAITING_INPUT,
            terminal=False,
            created_at=time.time(),
            run_token_sha256="",
            llm_gateway_token_sha256=TokenManager.token_sha256(llm_gateway_token),
            web_gateway_token_sha256=TokenManager.token_sha256(web_gateway_token) if web_gateway_token else "",
            runtime_config=runtime_config,
            runtime_config_issuer=str(meta.get("runtime_config_issuer") or "recover_runs"),
            runtime_config_reason=str(meta.get("runtime_config_reason") or "resumed from checkpoint"),
            runtime_config_committed_at=float(meta.get("runtime_config_committed_at") or time.time()),
        )

    def _issue_recovery_llm_gateway_token(
        self,
        run_id: str,
        request: BackendRunRequest,
        runtime_config: AgentRuntimeConfig,
    ) -> str:
        return self.token_manager.issue(
            kind="llm_gateway",
            audience="csp.llm-gateway",
            run_id=run_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            ttl_s=self.llm_gateway_token_ttl_s,
            metadata={"agent_config_hash": runtime_config.config_hash},
        )

    def _issue_recovery_web_gateway_token(
        self,
        run_id: str,
        request: BackendRunRequest,
        runtime_config: AgentRuntimeConfig,
    ) -> str:
        if not (_runtime_config_uses_web(runtime_config) and self.web_gateway_url):
            return ""
        return self.token_manager.issue(
            kind="web_gateway",
            audience="csp.web-gateway",
            run_id=run_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            ttl_s=self.web_gateway_token_ttl_s,
            metadata={"agent_config_hash": runtime_config.config_hash},
        )

    def _register_recovered_record(self, record: BackendRunRecord) -> None:
        self._register_record(record)

    def _attach_loop_build(
        self,
        record: BackendRunRecord,
        loop_build: BackendLoopBuild,
    ) -> None:
        with self._lock:
            record.loop = loop_build.loop
            record.outbox_sender = loop_build.outbox_sender

    def _attach_recovered_loop(
        self,
        record: BackendRunRecord,
        loop_build: BackendLoopBuild,
    ) -> None:
        self._attach_loop_build(record, loop_build)

    async def _acquire_run_slot(self) -> None:
        if self._run_semaphore is not None:
            await self._run_semaphore.acquire()

    def _release_run_slot(self) -> None:
        if self._run_semaphore is not None:
            self._run_semaphore.release()

    def _active_loop_session(self, run_id: str, token: str) -> tuple[AgentLoop, SessionState]:
        loop = self._session_boundary.authorize_active_loop(run_id, token)
        record = self._record(run_id)
        return loop, record.state

    def _record_terminal_locked(self, record: BackendRunRecord) -> bool:
        with self._lock:
            return _record_terminal(record)

    def _live_loop_snapshot(self, record: BackendRunRecord) -> tuple[AgentLoop | None, bool]:
        with self._lock:
            return record.loop, _record_terminal(record)

    def _live_outbox_runs(self) -> list[tuple[BackendRunRecord, AgentLoop]]:
        with self._lock:
            return [
                (record, record.loop)
                for record in self._records.values()
                if record.loop is not None
                and record.outbox_sender is not None
                and not _record_terminal(record)
            ]

    def _mark_cancel_requested(self, record: BackendRunRecord) -> bool:
        with self._lock:
            if _record_terminal(record):
                return False
            record.cancellation_token.cancel()
            record.error = "run cancellation requested"
            record.error_code = "cancelled"
            return True

    def _ensure_message_enqueue_allowed(self, record: BackendRunRecord) -> None:
        with self._lock:
            if _record_terminal(record):
                raise ValueError("cannot send a message to a terminal run")
            if record.message_queue.qsize() >= self.max_message_queue_depth:
                raise ValueError("message queue is full; retry once the run drains it")

    def _issue_task_callback_token(self, run_id: str, tenant_id: str, user_id: str, task_id: str) -> str:
        return self.token_manager.issue(
            kind="task_callback",
            audience=TASK_CALLBACK_AUDIENCE,
            run_id=run_id,
            tenant_id=tenant_id,
            user_id=user_id,
            ttl_s=self.task_callback_token_ttl_s,
            metadata={"task_id": task_id},
        )

    def _verify_task_callback_claims(self, run_id: str, token: str, task_id: str) -> None:
        claims = self.token_manager.verify(
            token, kind="task_callback", audience=TASK_CALLBACK_AUDIENCES, run_id=run_id
        )
        if str(claims.metadata.get("task_id") or "") != task_id:
            raise PermissionDenied("callback token does not match this task")
        record = self._active_record(run_id)
        if record is not None and (claims.tenant_id != record.tenant_id or claims.user_id != record.user_id):
            raise PermissionDenied("token subject mismatch")

    # --- Shared event loop (coroutine-per-run) ------------------------------------------

    def _spawn(self, coro: Any) -> Any:
        """Schedule a coroutine on the process-shared run loop from any (sync) thread;
        returns a concurrent.futures.Future."""
        return asyncio.run_coroutine_threadsafe(coro, _get_shared_loop())

    def _call_soon(self, fn: Callable[..., Any], *args: Any) -> None:
        """Run a thread-safe callback on the process-shared run loop (fire-and-forget)."""
        _get_shared_loop().call_soon_threadsafe(fn, *args)

    def _enqueue_message_and_checkpoint(self, record: BackendRunRecord, message: Any) -> None:
        """Enqueue an inbox message on the shared loop and persist the queue snapshot before returning."""

        def _enqueue_and_persist() -> None:
            record.message_queue.put_nowait(message)
            self._persist_run_checkpoint(record)

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is _get_shared_loop():
            _enqueue_and_persist()
            return

        done: Future[None] = Future()

        def _complete() -> None:
            try:
                _enqueue_and_persist()
            except BaseException as exc:
                done.set_exception(exc)
            else:
                done.set_result(None)

        _get_shared_loop().call_soon_threadsafe(_complete)
        done.result(timeout=10.0)

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
        with self._lock:
            records = [record for record in self._records.values() if not _record_terminal(record)]
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
                if _record_terminal(record):
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
        return self._run_preparation.prepare(request)

    def _submission_for(self, prepared: _PreparedRun) -> BackendRunSubmission:
        return self._run_preparation.submission_for(prepared)

    def _run_spec_for_request(
        self,
        run_id: str,
        request: BackendRunRequest,
        workspace_root: Path,
    ) -> AgentRunSpec:
        return self._loop_factory.run_spec_for_request(run_id, request, workspace_root)

    def status(self, run_id: str, token: str) -> dict[str, Any]:
        return self._projection.status(run_id, token)

    def result(self, run_id: str, token: str) -> dict[str, Any]:
        return self._projection.result(run_id, token)

    def proposal(self, run_id: str, token: str) -> dict[str, Any]:
        return self._proposal.proposal(run_id, token)

    def proposal_diff(self, run_id: str, token: str) -> dict[str, Any]:
        """The unified diff of the current proposal, on demand (works mid-run, not only at the
        end like ``result()``). Token-scoped so an embedder never reads the run dir off disk.
        Binary files appear as a ``<binary sha256=… size=…>`` marker line in the patch."""
        return self._proposal.proposal_diff(run_id, token)

    def cancel_run(self, run_id: str, token: str) -> dict[str, Any]:
        return self._session_boundary.cancel_run(run_id, token)

    def interrupt_turn(self, run_id: str, token: str) -> dict[str, Any]:
        return self._session_boundary.interrupt_turn(run_id, token)

    def pause_run(self, run_id: str, token: str) -> dict[str, Any]:
        return self._session_boundary.pause_run(run_id, token)

    def signal_resume(self, run_id: str, token: str) -> dict[str, Any]:
        return self._session_boundary.signal_resume(run_id, token)

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
        return self._session_boundary.revoke_capability(
            run_id,
            token,
            capability=capability,
            lease_id=lease_id,
            before=before,
            reason=reason,
        )

    def dispatch(self, command: ControlCommand) -> ControlResult:
        return self._commands.dispatch(command)

    def enqueue_control(self, command: ControlCommand) -> CommandReceipt:
        """Authenticate, sanitize, and durably enqueue one idempotent command."""

        args = dict(command.args)
        token = str(args.pop("token", "") or "")
        principal = self._authorize_command_principal(command, args=args, token=token)
        with self._lock:
            locally_owned = command.run_id in self._records
        if command.type == "create_task" and not locally_owned:
            raise NativeAgentError(
                "create_task must be routed to the run owner so its callback token can be "
                "returned without durable persistence",
                error_code="command_requires_owner",
            )
        command_id = command.command_id or f"control_{uuid.uuid4().hex[:12]}"
        assert self.command_store is not None
        receipt = self.command_store.append(
            StoredCommand(
                run_id=command.run_id,
                command_id=command_id,
                type=command.type,
                args=dict(
                    redact_command_credential(sanitize_command_data(args), token)
                ),
                principal=CommandPrincipal(
                    tenant_id=principal.tenant_id,
                    user_id=principal.user_id,
                    issuer=str(redact_command_credential(command.issuer, token)),
                ),
                reason=str(redact_command_credential(command.reason, token)),
            ),
            max_pending=self.command_queue_limit,
        )
        executed = self._drain_command_inbox(command.run_id)
        current = self.command_store.receipt(command.run_id, command_id) or receipt
        local_result = executed.get(command_id)
        return (
            replace(current, transient_result=local_result.to_json())
            if local_result is not None
            else current
        )

    def _authorize_command_principal(
        self,
        command: ControlCommand,
        *,
        args: dict[str, Any],
        token: str,
    ) -> CommandPrincipal:
        try:
            claims = self._authorize_command_target(command.run_id, token)
            return CommandPrincipal(claims.tenant_id, claims.user_id, command.issuer)
        except PermissionDenied:
            if command.type not in {"approve", "deny", "report_task_result"}:
                raise
            self._commands.authorize_control_audit_target(
                command.run_id,
                token,
                command_type=command.type,
                args=args,
            )
            with self._lock:
                record = self._records.get(command.run_id)
            if record is not None:
                return CommandPrincipal(record.tenant_id, record.user_id, command.issuer)
            metadata = self._read_recovery_meta(
                self.run_root / command.run_id, command.run_id
            )
            if metadata is None:
                raise KeyError(command.run_id)
            try:
                authenticated_claims = self._verify_run_token(command.run_id, token)
            except PermissionDenied:
                # Callback credentials were already validated against their task above; decode
                # their subject here so non-owner peers can compare it with durable run metadata.
                try:
                    authenticated_claims = self.token_manager.verify(
                        token,
                        kind="task_callback",
                        audience=TASK_CALLBACK_AUDIENCES,
                        run_id=command.run_id,
                    )
                except TokenError as exc:
                    raise PermissionDenied(str(exc)) from exc
            if (
                authenticated_claims.tenant_id != str(metadata.get("tenant_id") or "")
                or authenticated_claims.user_id != str(metadata.get("user_id") or "")
            ):
                raise PermissionDenied("token subject mismatch")
            return CommandPrincipal(
                str(metadata.get("tenant_id") or ""),
                str(metadata.get("user_id") or ""),
                command.issuer,
            )

    def _authorize_command_target(self, run_id: str, token: str) -> Any:
        claims = self._verify_run_token(run_id, token)
        run_dir = self._authorized_run_dir(run_id, token)
        with self._lock:
            local_record = self._records.get(run_id)
        if local_record is None:
            metadata = self._read_recovery_meta(run_dir, run_id)
            if metadata is None:
                raise KeyError(run_id)
            if (
                str(metadata.get("tenant_id") or "") != claims.tenant_id
                or str(metadata.get("user_id") or "") != claims.user_id
            ):
                raise PermissionDenied("token subject mismatch")
        return claims

    def command_receipt(self, run_id: str, token: str, command_id: str) -> CommandReceipt:
        self._authorize_command_target(run_id, token)
        assert self.command_store is not None
        receipt = self.command_store.receipt(run_id, command_id)
        if receipt is None:
            raise KeyError(command_id)
        return receipt

    def _drain_command_inbox(self, run_id: str) -> dict[str, ControlResult]:
        """Claim and execute commands only on the instance that owns the target run."""

        with self._lock:
            if run_id not in self._records:
                return {}
        assert self.command_store is not None
        completed: dict[str, ControlResult] = {}
        while True:
            stored = self.command_store.claim(
                run_id,
                self._worker_id,
                claim_ttl_s=self.command_claim_ttl_s,
            )
            if stored is None:
                return completed
            token = self.token_manager.issue(
                kind="run_access",
                audience=BACKEND_AUDIENCE,
                run_id=run_id,
                tenant_id=stored.principal.tenant_id,
                user_id=stored.principal.user_id,
                ttl_s=60,
            )
            try:
                result = self.dispatch(stored.control_command(token=token))
            except Exception as exc:  # owner records a durable failure receipt
                result = ControlResult(
                    run_id=run_id,
                    type=stored.type,  # type: ignore[arg-type]
                    status="error",
                    error=str(exc),
                    error_code=getattr(exc, "error_code", "command_execution_error"),
                )
            self.command_store.acknowledge(run_id, stored.command_id, self._worker_id, result)
            completed[stored.command_id] = result

    def _dispatch_control_command(
        self,
        command: ControlCommand,
        *,
        args: dict[str, Any],
        token: str,
        command_id: str,
    ) -> ControlResult:
        return self._commands.dispatch_control_command(
            command,
            args=args,
            token=token,
            command_id=command_id,
        )

    def _emit_control_audit_event(
        self,
        run_id: str,
        event_type: str,
        data: dict[str, Any],
        *,
        level: str = "info",
    ) -> None:
        self._emit_backend_event(run_id, event_type, data, level=level)

    def _emit_backend_event(
        self,
        run_id: str,
        event_type: str,
        data: dict[str, Any],
        *,
        level: str = "info",
    ) -> None:
        self._run_state.emit_backend_event(run_id, event_type, data, level=level)

    def _authorize_control_audit_target(
        self,
        run_id: str,
        token: str,
        *,
        command_type: str = "",
        args: Mapping[str, Any] | None = None,
    ) -> None:
        self._commands.authorize_control_audit_target(
            run_id,
            token,
            command_type=command_type,
            args=dict(args or {}),
        )

    def _authorize_claim_subject(self, run_id: str, claims: Any) -> None:
        self._session_boundary.authorize_claim_subject(run_id, claims)

    def _verify_task_callback_token(self, run_id: str, token: str, task_id: str) -> None:
        self._session_boundary.verify_task_callback_token(run_id, token, task_id)

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
        message_type: str = "user_message",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._session_boundary.send_message(
            run_id,
            token,
            content,
            message_id=message_id,
            source=source,
            correlation_id=correlation_id,
            causation_id=causation_id,
            traceparent=traceparent,
            tracestate=tracestate,
            message_type=message_type,
            metadata=metadata,
        )

    def current_runtime_config(self, run_id: str) -> AgentRuntimeConfig | None:
        return self._runtime_config.current_runtime_config(run_id)

    def runtime_config(self, run_id: str, token: str) -> dict[str, Any]:
        return self._runtime_config.runtime_config(run_id, token)

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
        return self._runtime_config.replace_runtime_config(
            run_id,
            token,
            expected_version=expected_version,
            issuer=issuer,
            reason=reason,
            config=config,
        )

    def report_task_result(
        self,
        run_id: str,
        token: str,
        *,
        task_id: str,
        result: dict[str, Any],
        status: str = "answered",
    ) -> dict[str, Any]:
        return self._session_boundary.report_task_result(
            run_id,
            token,
            task_id=task_id,
            result=result,
            status=status,
        )

    def create_task(
        self,
        run_id: str,
        token: str,
        *,
        kind: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return self._session_boundary.create_task(run_id, token, kind=kind, request=request)

    def _authorize_active_loop(self, run_id: str, token: str) -> AgentLoop:
        return self._session_boundary.authorize_active_loop(run_id, token)

    def _authorize_task_result(self, run_id: str, token: str, task_id: str) -> AgentLoop:
        return self._session_boundary.authorize_task_result(run_id, token, task_id)

    def _active_loop(self, run_id: str) -> AgentLoop:
        return self._session_boundary.active_loop(run_id)

    def proposal_file(self, run_id: str, token: str, path: str) -> dict[str, Any]:
        return self._proposal.proposal_file(run_id, token, path)

    def export_proposal_package(self, run_id: str, token: str) -> dict[str, Any]:
        """Build the portable proposal package and return a RECEIPT — never a filesystem path.

        The tar is stored as a content-addressed blob; the receipt's ``digest`` (sha256 of the tar
        bytes) is the retrieval handle for :meth:`read_run_artifact`. This keeps the
        "embedder never reads run_dir off disk" invariant for binary artifacts too: a remote
        embedder fetches the bytes back by digest, exactly like Bazel CAS / an OCI blob."""
        return self._proposal.export_proposal_package(run_id, token)

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
        return self._proposal.read_run_artifact(run_id, token, digest, offset=offset, limit=limit)

    def approve_proposal(
        self,
        run_id: str,
        token: str,
        *,
        approver_id: str,
        approved_paths: tuple[str, ...] = (),
        note: str = "",
    ) -> dict[str, Any]:
        return self._proposal.approve_proposal(
            run_id,
            token,
            approver_id=approver_id,
            approved_paths=approved_paths,
            note=note,
        )

    def reject_proposal(
        self,
        run_id: str,
        token: str,
        *,
        approver_id: str,
        reason: str,
    ) -> dict[str, Any]:
        return self._proposal.reject_proposal(run_id, token, approver_id=approver_id, reason=reason)

    def apply_proposal(
        self,
        run_id: str,
        token: str,
        *,
        target: Path,
        approval_path: Path | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return self._proposal.apply_proposal(
            run_id,
            token,
            target=target,
            approval_path=approval_path,
            dry_run=dry_run,
        )

    def events(
        self, run_id: str, token: str, *, from_seq: int = 0, limit: int | None = None
    ) -> dict[str, Any]:
        return self._projection.events(run_id, token, from_seq=from_seq, limit=limit)

    def subscribe_events(
        self,
        run_id: str,
        token: str,
        *,
        from_seq: int = 0,
        last_event_id: str | None = None,
    ) -> EventSubscription:
        """Create a replay-safe subscription for a live or recovered authorized run."""

        cursor = SequenceCursor.resolve(from_seq=from_seq, last_event_id=last_event_id)
        return EventSubscription(
            lambda next_seq, limit: self.events(
                run_id, token, from_seq=next_seq, limit=limit
            ),
            cursor=cursor,
            read_lifecycle=lambda: self.status(run_id, token),
        )

    def diagnostics(self, run_id: str, token: str, *, event_limit: int = 50) -> dict[str, Any]:
        return self._projection.diagnostics(run_id, token, event_limit=event_limit)

    def descendant_events(
        self,
        run_id: str,
        token: str,
        descendant_run_id: str,
        *,
        from_seq: int = 0,
        limit: int | None = None,
    ) -> dict[str, Any]:
        return self._projection.descendant_events(
            run_id,
            token,
            descendant_run_id,
            from_seq=from_seq,
            limit=limit,
        )

    def descendant_status(
        self, run_id: str, token: str, descendant_run_id: str
    ) -> dict[str, Any]:
        return self._projection.descendant_status(run_id, token, descendant_run_id)

    def subscribe_descendant_events(
        self,
        run_id: str,
        token: str,
        descendant_run_id: str,
        *,
        from_seq: int = 0,
        last_event_id: str | None = None,
    ) -> EventSubscription:
        """Create a cursor-correct subscription scoped to an authorized descendant."""

        cursor = SequenceCursor.resolve(from_seq=from_seq, last_event_id=last_event_id)
        return EventSubscription(
            lambda next_seq, limit: self.descendant_events(
                run_id,
                token,
                descendant_run_id,
                from_seq=next_seq,
                limit=limit,
            ),
            cursor=cursor,
            read_lifecycle=lambda: self.descendant_status(
                run_id, token, descendant_run_id
            ),
        )

    def jobs(self, run_id: str, token: str) -> dict[str, Any]:
        return self._jobs.jobs(run_id, token)

    def job_status(self, run_id: str, token: str, job_id: str) -> dict[str, Any]:
        return self._jobs.job_status(run_id, token, job_id)

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
        return self._jobs.job_logs(
            run_id,
            token,
            job_id,
            stream=stream,
            tail_bytes=tail_bytes,
            offset=offset,
        )

    def cancel_job(self, run_id: str, token: str, job_id: str) -> dict[str, Any]:
        return self._jobs.cancel_job(run_id, token, job_id)

    def tenant_usage(self, tenant_id: str) -> dict[str, Any]:
        return self._run_state.tenant_usage(tenant_id)

    def record_event(self, run_id: str, event: AgentEvent) -> None:
        self._run_state.record_event(run_id, event)

    def wait_for_run(self, run_id: str, *, timeout_s: float = 10.0) -> SessionState:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            record = self._record(run_id)
            if _record_terminal(record):
                return record.state
            time.sleep(0.05)
        raise TimeoutError(f"run did not finish before timeout: {run_id}")

    async def _drive_session(self, run_id: str, request: BackendRunRequest, loop: AgentLoop) -> AgentRunResult:
        return await self._run_execution.drive_session(run_id, request, loop)

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
        return await self._session_drive.drive_open_session(
            record,
            request,
            loop,
            suspension,
            started=started,
            turns=turns,
        )

    async def _await_session_message(self, record: BackendRunRecord) -> Any:
        return await self._session_drive.await_session_message(record)

    def _persist_run_checkpoint(self, record: BackendRunRecord) -> None:
        self._session_drive.persist_run_checkpoint(record)

    def _persist_run_checkpoint_payload(
        self,
        record: BackendRunRecord,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
    ) -> None:
        self._session_drive.persist_run_checkpoint_payload(record, checkpoint, blobs)

    async def _persist_run_checkpoint_async(self, record: BackendRunRecord) -> None:
        await self._session_drive.persist_run_checkpoint_async(record)

    def _persist_run_checkpoint_from_any_thread(self, record: BackendRunRecord) -> None:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is _get_shared_loop():
            self._persist_run_checkpoint(record)
            return
        self._spawn(self._persist_run_checkpoint_async(record)).result(timeout=10.0)

    def _outbox_backoff_delay(self, attempts: int) -> float:
        return self._outbox_dispatch.backoff_delay(attempts)

    def _drain_outbox(self, record: BackendRunRecord, loop: AgentLoop) -> None:
        self._outbox_dispatch.drain_outbox(record, loop)

    def _stage_outbox_ack(
        self, record: BackendRunRecord, request: Any, status: str, receipt: OutboxReceipt
    ) -> None:
        self._outbox_dispatch.stage_outbox_ack(record, request, status, receipt)

    def _session_should_stop(self, record: BackendRunRecord, started: float, turns: int) -> bool:
        return self._session_drive.session_should_stop(record, started, turns)

    async def _run_run(
        self,
        run_id: str,
        request: BackendRunRequest,
        workspace_root: Path,
        llm_gateway_token: str,
        web_gateway_token: str,
    ) -> None:
        prepared = _PreparedRun(
            run_id=run_id,
            record=self._record(run_id),
            workspace_root=workspace_root,
            run_token="",
            llm_gateway_token=llm_gateway_token,
            web_gateway_token=web_gateway_token,
        )
        await self._run_execution.run_prepared(prepared, request)

    def _capability_broker_for(self, request: BackendRunRequest) -> Any:
        """Build the run's capability broker from the factory (scoped to run identity), or None
        to leave capability gating off for this run."""
        return self._loop_factory.capability_broker_for(request)

    def _outbox_sender_for(self, request: BackendRunRequest) -> Any:
        """Build the run's outbox sender from the factory (scoped to run identity), or None to leave
        staged outbox requests pending (durable, never dispatched)."""
        return self._loop_factory.outbox_sender_for(request)

    def _build_loop_build(
        self,
        run_id: str,
        request: BackendRunRequest,
        workspace_root: Path,
        llm_gateway_token: str,
        web_gateway_token: str,
    ) -> BackendLoopBuild:
        return self._loop_factory.build(
            run_id,
            request,
            workspace_root,
            llm_gateway_token,
            web_gateway_token,
        )

    def _build_loop(
        self,
        run_id: str,
        request: BackendRunRequest,
        workspace_root: Path,
        llm_gateway_token: str,
        web_gateway_token: str,
    ) -> AgentLoop:
        """Construct the run's AgentLoop (shared by autonomous and stream-driven paths)."""
        return self._loop_factory.build(
            run_id,
            request,
            workspace_root,
            llm_gateway_token,
            web_gateway_token,
            include_outbox_sender=False,
        ).loop

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
        async for frame in self._run_execution.stream_prepared(prepared, request):
            yield frame

    def _frame(self, item: Any) -> dict[str, Any]:
        """Wrap one astream item as a neutral wire frame (reference framing on core to_json)."""
        return stream_item_frame(item)

    def _record_run_result(self, run_id: str, result: AgentRunResult) -> None:
        self._run_state.record_run_result(run_id, result)

    def _record_run_failure(self, run_id: str, exc: Exception) -> None:
        self._run_state.record_run_failure(run_id, exc)

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
        self._recovery.write_failure_bundle(
            run_id,
            run_dir,
            error=error,
            error_code=error_code,
            exc_type=exc_type,
            overwrite=overwrite,
        )

    def _read_recover_attempts(self, run_dir: Path) -> int:
        return self._recovery.read_recover_attempts(run_dir)

    def _bump_recover_attempts(self, run_dir: Path) -> int:
        return self._recovery.bump_recover_attempts(run_dir)

    def _clear_recover_attempts(self, run_dir: Path) -> None:
        self._recovery.clear_recover_attempts(run_dir)

    def _write_run_meta(self, record: BackendRunRecord, request: BackendRunRequest) -> None:
        self._run_preparation.write_run_meta(record, request)

    def _write_runtime_config_run_meta(
        self,
        record: BackendRunRecord,
        config: AgentRuntimeConfig,
        *,
        issuer: str,
        reason: str,
        committed_at: float,
    ) -> None:
        self._runtime_config.write_runtime_config_run_meta(
            record,
            config,
            issuer=issuer,
            reason=reason,
            committed_at=committed_at,
        )

    def _store_run_meta(self, run_id: str, meta: Mapping[str, Any]) -> None:
        DurableMetadataCommitter(self.checkpoint_store).store_shared_metadata(run_id, meta)

    def _read_recovery_meta(self, run_dir: Path, run_id: str) -> dict[str, Any] | None:
        return self._recovery.read_recovery_meta(run_dir, run_id)

    def recover_runs(self) -> list[str]:
        """Scan ``run_root`` for runs left parked by a previous process and resume each
        from its checkpoint. Returns the recovered run ids. Idempotent: runs already
        tracked in-memory, terminal checkpoints, and runs missing run.json are skipped."""
        return self._recovery.recover_runs()

    def _attempt_resume(self, run_dir: Path, run_id: str) -> bool:
        """Resume one run from its latest checkpoint. Returns True on success. Skips runs
        with no resumable checkpoint or missing run.json. On a resume exception, bumps the
        durable attempt counter and, once ``max_recover_attempts`` is reached, marks the run
        unrecoverable (durable failure.json) so it is never retried into a crash loop."""
        return self._recovery.attempt_resume(run_dir, run_id)

    def resume_run(self, run_id: str, token: str) -> dict[str, Any]:
        return self._session_boundary.resume_run(run_id, token)

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
                self._drain_command_inboxes()
            except Exception:  # pragma: no cover - the watchdog must never die on a tick
                _LOGGER.exception("watchdog tick failed")

    def _redrive_outbox(self) -> None:
        self._outbox_dispatch.redrive_outbox()

    def _drain_command_inboxes(self) -> None:
        with self._lock:
            run_ids = tuple(self._records)
        for run_id in run_ids:
            self._drain_command_inbox(run_id)

    def _heartbeat_own_runs(self) -> None:
        assert self.lease_store is not None
        with self._lock:
            items = list(self._records.items())
        for run_id, record in items:
            try:
                if _record_terminal(record):
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
        return self._recovery.reclaim_stale_runs()

    def _resume_from_checkpoint(self, stored: CheckpointRecord, meta: dict[str, Any]) -> None:
        self._recovery.resume_from_checkpoint(stored, meta)

    async def _run_recovered(self, run_id: str, request: BackendRunRequest, loop: AgentLoop) -> None:
        await self._recovery.run_recovered(run_id, request, loop)

    def _llm_token_source(
        self, run_id: str, request: BackendRunRequest, runtime_config: AgentRuntimeConfig | None
    ) -> _GatewayTokenSource:
        """A re-minting source for the run's ``llm_gateway`` token (mirrors the eager issue + the
        recovery re-issue), so a long run keeps LLM access past the token TTL without a restart."""
        return self._loop_factory.llm_token_source(run_id, request, runtime_config)

    def _build_model_adapter(
        self,
        spec: AgentRunSpec,
        llm_gateway_token: str,
        model_config: ModelConfig | None,
        token_provider: Callable[[], str | None] | None = None,
    ) -> ModelAdapter:
        return self._loop_factory.build_model_adapter(
            spec,
            llm_gateway_token,
            model_config,
            token_provider=token_provider,
        )

    def _web_gateway_client(
        self,
        token: str,
    ) -> WebGatewayClient | None:
        return self._loop_factory.web_gateway_client(token)

    def _validate_request(self, request: BackendRunRequest) -> None:
        self._run_preparation.validate_request(request)

    def _check_workspace_allowed(self, workspace_root: Path) -> None:
        self._run_preparation.check_workspace_allowed(workspace_root)

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
        return self._projection.list_runs(tenant_id, user_id=user_id, limit=limit)

    def _record(self, run_id: str) -> BackendRunRecord:
        with self._lock:
            try:
                return self._records[run_id]
            except KeyError as exc:
                raise KeyError(f"unknown run: {run_id}") from exc

    def _read_proposal(self, record: BackendRunRecord) -> dict[str, Any] | None:
        return read_proposal_snapshot(record)
