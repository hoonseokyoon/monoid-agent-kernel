from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import Any

from native_agent_runner.core._util import sha256_bytes
from native_agent_runner.core.cancellation import CancellationToken
from native_agent_runner.core.checkpoint import (
    CheckpointStore,
    LocalFsCheckpointStore,
    RunCheckpoint,
)
from native_agent_runner.core.events import AgentEvent, EventSink
from native_agent_runner.core.content import (
    ContentPart,
    content_part_from_json,
    content_part_to_json,
    non_text_part_types,
)
from native_agent_runner.core.context import (
    ContextProvider,
    TurnContext,
    render_workspace_index_segment,
)
from native_agent_runner.core.agents import (
    AgentRuntimeConfig,
    BoundTool,
    BoundToolCatalog,
    RuntimeConfigSource,
    coerce_runtime_config_provider,
    compile_bound_tool_catalog,
    runtime_config_diff,
    transcript_config_snapshot,
    validate_runtime_config,
)
from native_agent_runner.core.manifest import build_run_manifest
from native_agent_runner.core.prompt import BASE_SYSTEM_PROMPT, compose_system_prompt
from native_agent_runner.core.result import AgentRunResult, AgentTurnResult, Suspension
from native_agent_runner.core.spec import (
    AgentRunSpec,
    ModelConfig,
    RunLimits,
    input_to_parts,
    text_from_parts,
)
from native_agent_runner.core.tool_surface import (
    DefaultToolSurfaceResolver,
    ToolAuthorization,
    ToolSearchEntry,
    ToolSurfaceResolver,
    ToolSurfaceSnapshot,
    tool_surface_manifest,
)
from native_agent_runner.core.workspace_index import build_workspace_index
from native_agent_runner.errors import (
    ModelAdapterError,
    AgentConfigError,
    NativeAgentError,
    PermissionDenied,
    RunCancelled,
    RunTimeout,
    ToolExecutionError,
    error_code_for_exception,
)
from native_agent_runner.tasks import HostedTask, TaskManager
from native_agent_runner.permissions import PermissionPolicy, matches_path_patterns
from native_agent_runner.providers.base import (
    ModelAdapter,
    ModelRequest,
    ModelTurn,
    ToolObservation,
    format_async_result_text,
)
from native_agent_runner.public_view import (
    args_preview,
    public_error_message,
    public_path,
    public_proposal_payload,
    public_result_content,
    shell_args_preview,
    web_args_preview,
)
from native_agent_runner.recorder import AgentRecorder
from native_agent_runner.shell import ShellApprovalProvider
from native_agent_runner.tool_services import CallContext, JobsService, ShellService, WebService
from native_agent_runner.tools.base import (
    DynamicToolProvider,
    ToolContext,
    ToolProvider,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from native_agent_runner.tools.builtin import builtin_tools
from native_agent_runner.web import WebGatewayClient, domain_allowed, domain_from_url
from native_agent_runner.core.workspace import Workspace
from native_agent_runner.workspace.local import default_local_workspace_factory


def _failure_result(exc: Exception, *, error_code: str | None = None) -> ToolResult:
    """Build a failed ToolResult from an exception, carrying the model-facing
    retry/category signal. Raw ``ValueError``/``TypeError`` are treated as tool
    handler errors (retryable, "tool") to match their ``tool_handler_error`` code."""
    if error_code is not None:
        code = error_code
    elif isinstance(exc, NativeAgentError):
        code = error_code_for_exception(exc)
    else:
        code = "tool_handler_error"
    retryable = getattr(exc, "retryable", code == "tool_handler_error")
    category = getattr(exc, "category", "tool" if code == "tool_handler_error" else "internal")
    return ToolResult(
        ok=False,
        error=str(exc),
        error_code=code,
        retryable=bool(retryable),
        category=str(category),
    )


@dataclass
class AgentToolContext(ToolContext):
    run_id: str
    workspace: Workspace
    recorder: AgentRecorder
    job_manager: TaskManager
    shell_service: ShellService
    web_service: WebService
    jobs_service: JobsService
    final_text: str = ""
    final_outputs: list[str] = field(default_factory=list)
    final_notes: str | None = None
    finished: bool = False
    plan: list[dict[str, Any]] = field(default_factory=list)
    permission_policy: PermissionPolicy = field(default_factory=PermissionPolicy)
    tool_search_entries: tuple[ToolSearchEntry, ...] = ()
    tool_search_max_results: int = 5
    _requested_tool_loads: list[str] = field(default_factory=list)
    _current_call: CallContext = field(default_factory=lambda: CallContext("", None, None))

    def emit_artifact(
        self, path: str, kind: str, label: str | None, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        del metadata
        data, _digest = self.workspace.read_bytes(path)
        artifact = self.recorder.emit_artifact_bytes(
            workspace_path=self.workspace.normalize(path),
            content=data,
            kind=kind,
            label=label,
        )
        self.recorder.emit(
            "artifact.emitted",
            data={"artifact_id": artifact.artifact_id, "path": artifact.path, "kind": kind},
        )
        return {
            "artifact_id": artifact.artifact_id,
            "path": artifact.path,
            "kind": artifact.kind,
            "label": artifact.label,
        }

    def list_artifacts(self) -> list[dict[str, Any]]:
        return [
            {
                "artifact_id": artifact.artifact_id,
                "path": artifact.path,
                "kind": artifact.kind,
                "label": artifact.label,
            }
            for artifact in self.recorder.artifacts
        ]

    def update_plan(self, items: list[dict[str, Any]]) -> None:
        self.plan = items
        self.recorder.emit("plan.updated", data={"items": items})

    def finish(self, summary: str, outputs: list[str], notes: str | None) -> None:
        self.final_text = summary
        self.final_outputs = list(outputs)
        self.final_notes = notes
        self.finished = True

    def execute_shell(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.shell_service.execute(args, self._current_call)

    def list_jobs(self) -> list[dict[str, Any]]:
        return self.jobs_service.list_jobs()

    def job_status(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.jobs_service.status(args)

    def job_logs(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.jobs_service.logs(args)

    def job_cancel(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.jobs_service.cancel(args)

    def job_wait(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.jobs_service.wait(args)

    def request_human_input(self, args: dict[str, Any]) -> dict[str, Any]:
        task = self.job_manager.start_task(
            "hitl",
            {
                "prompt": str(args.get("prompt") or ""),
                "choices": tuple(str(choice) for choice in (args.get("choices") or ())),
                "created_by": "model",
            },
        )
        return task.started_content(self.recorder.run_dir)

    def execute_web_search(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.web_service.search(args, self._current_call)

    def execute_web_fetch(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.web_service.fetch(args, self._current_call)

    def execute_web_context(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.web_service.context(args, self._current_call)

    def configure_tool_search(self, entries: tuple[ToolSearchEntry, ...], max_results: int) -> None:
        self.tool_search_entries = entries
        self.tool_search_max_results = max_results

    def search_tools(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "").strip().lower()
        requested_max = args.get("max_results")
        max_results = min(
            self.tool_search_max_results,
            int(requested_max) if requested_max is not None else self.tool_search_max_results,
        )
        ranked = _rank_tool_search_entries(query, self.tool_search_entries)
        results = [entry.to_json() for entry in ranked[:max_results]]
        for item in results:
            binding_id = str(item.get("binding_id") or "")
            if binding_id and binding_id not in self._requested_tool_loads:
                self._requested_tool_loads.append(binding_id)
        return {"results": results, "count": len(results)}

    def consume_tool_load_requests(self) -> tuple[str, ...]:
        requested = tuple(self._requested_tool_loads)
        self._requested_tool_loads.clear()
        return requested


def _observation_message(observation: ToolObservation) -> dict[str, Any]:
    """Provider-neutral by-value message for a tool/async observation. Preserves the
    ``is_background`` → role semantics the adapters use: a background/hosted result is a
    new user message; a tool result is a ``tool`` message keyed by ``call_id``."""
    if observation.is_background:
        return {"role": "user", "content": format_async_result_text(observation.output)}
    return {"role": "tool", "call_id": observation.call_id, "content": observation.output}


def _as_blob_reader(
    blobs: Mapping[str, bytes] | Callable[[str], bytes] | None,
) -> Callable[[str], bytes]:
    """Normalize a blob source (mapping, reader callable, or None) into a reader. A
    ``None`` source has no content — used when restoring a checkpoint with no workspace
    delta; reading any sha then raises (a delta entry without its blob is a bug)."""
    if blobs is None:
        def _empty(sha256: str) -> bytes:
            raise KeyError(sha256)

        return _empty
    if callable(blobs):
        return blobs
    return lambda sha256: blobs[sha256]


@dataclass
class RunState:
    """Mutable state threaded through a run's steps and teardown."""

    status: str = "completed"
    error: str = ""
    error_code: str = ""
    provider_error_code: str = ""
    provider_http_status: int | None = None
    final_text: str = ""
    previous_turn_handle: str | None = None
    pending_user_input: tuple[ContentPart, ...] | None = None
    pending_observations: tuple[ToolObservation, ...] = ()
    pending_binding_loads: tuple[str, ...] = ()
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    previous_surface_snapshot: ToolSurfaceSnapshot | None = None
    previous_runtime_config: AgentRuntimeConfig | None = None
    total_tool_calls: int = 0
    total_usage: dict[str, int] = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    )
    # By-value conversation log: provider-neutral user/assistant/tool messages the core
    # owns and resends each turn (vendor-independent continuation). The system prompt is
    # NOT here — it is regenerated per turn and applied via ModelRequest.system_prompt.
    messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _RunResources:
    """Objects assembled by bootstrap and reused across a run's phases."""

    workspace: Workspace
    recorder: AgentRecorder
    context: AgentToolContext
    base_tool_specs: tuple[ToolSpec, ...]
    started: float
    deadline: float | None
    static_segments: tuple[str, ...]


@dataclass
class _Session:
    """Live state for an open run, threaded across multiple submit() calls."""

    state: RunState
    res: _RunResources
    session_step: int = 0
    submit_local_step: int = 0
    terminal: bool = False
    # Monotonic checkpoint sequence for this open run; advanced once per park.
    checkpoint_seq: int = 0


@dataclass
class AgentLoop:
    spec: AgentRunSpec
    model_adapter: ModelAdapter
    _: KW_ONLY
    # Accepts a RuntimeConfigProvider, a bare AgentRuntimeConfig, or a
    # callable(run_id) -> AgentRuntimeConfig; __post_init__ coerces to a provider.
    runtime_config_provider: RuntimeConfigSource
    tool_providers: tuple[ToolProvider, ...] = ()
    dynamic_tool_providers: tuple[DynamicToolProvider, ...] = ()
    tool_surface_resolver: ToolSurfaceResolver = field(default_factory=DefaultToolSurfaceResolver)
    event_sinks: tuple[EventSink, ...] = ()
    status_file: bool = True
    permission_policy: PermissionPolicy = field(default_factory=PermissionPolicy)
    cancellation_token: CancellationToken | None = None
    shell_approval_provider: ShellApprovalProvider | None = None
    web_gateway_client: WebGatewayClient | None = None
    workspace_factory: Callable[[AgentRunSpec], Workspace] | None = None
    context_providers: tuple[ContextProvider, ...] = ()
    inject_workspace_index: bool = False
    # How checkpoints are durably stored (core defines WHAT, the store defines HOW).
    # Defaults to a local-fs store under the run root; a backend injects a durable one.
    checkpoint_store: CheckpointStore | None = None
    _bootstrap_resources: _RunResources | None = field(default=None, init=False, repr=False)
    _session: _Session | None = field(default=None, init=False, repr=False)
    _restoring: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # Coerce a bare AgentRuntimeConfig or a callable(run_id) into a provider, so callers
        # can pass any of the three forms without hand-wrapping a StaticRuntimeConfigProvider.
        self.runtime_config_provider = coerce_runtime_config_provider(self.runtime_config_provider)

    @classmethod
    def from_config(
        cls,
        spec: AgentRunSpec,
        model_adapter: ModelAdapter,
        runtime_config: RuntimeConfigSource,
        **kwargs: Any,
    ) -> AgentLoop:
        """Build a loop from a fixed config without hand-wrapping a provider.

        ``runtime_config`` may be an :class:`AgentRuntimeConfig`, a
        :class:`~native_agent_runner.RuntimeConfigProvider`, or a
        ``callable(run_id) -> AgentRuntimeConfig``. Remaining optional seams
        (``tool_providers``, ``event_sinks``, ``checkpoint_store``, …) pass through as
        keyword arguments. Collapses the full constructor to one call::

            AgentLoop.from_config(spec, adapter, config).run_once("do the thing")
        """
        return cls(spec, model_adapter, runtime_config_provider=runtime_config, **kwargs)

    def open(self) -> None:
        """Bootstrap the run and leave it idle, ready to accept submit().

        No model turn happens here. The workspace, recorder, tool registry, and
        manifest are created and ``run.started`` is emitted. A recordable bootstrap
        failure (e.g. invalid runtime config) is captured as a terminal failed
        session so close() still returns a failed result rather than raising."""
        if self._session is not None:
            raise NativeAgentError("run is already open", error_code="run_already_open")
        try:
            res = self._bootstrap()
        except Exception as exc:  # controlled recording boundary for standalone CLI
            res = self._bootstrap_resources
            if res is None:
                raise
            state = RunState()
            self._record_failure(state, res, exc)
            self._session = _Session(state=state, res=res, terminal=True)
            return
        self._session = _Session(state=RunState(), res=res)

    def submit(self, user_input: str | tuple[ContentPart, ...]) -> AgentTurnResult:
        """Run one user turn: inject ``user_input`` and step until the model settles
        (no tool calls + final text) or a per-submit limit is hit. The run stays
        open afterwards; call submit() again to continue or close() to finalize.

        Blocking wrapper over ``run_until_suspended``: when the run parks on tasks it
        waits in-process (shell monitor completes them, or an external thread reports
        a hosted-task result) and resumes, returning only once the turn settles."""
        session = self._require_open()
        suspension = self.run_until_suspended(user_input)
        while suspension.reason == "awaiting_tasks":
            self._wait_for_background_jobs(session.res.context, session.res.recorder, session.res.deadline)
            suspension = self.run_until_suspended(None)
        assert suspension.turn is not None  # non-awaiting reasons always checkpoint
        return suspension.turn

    def run_until_suspended(
        self, user_input: str | tuple[ContentPart, ...] | None = None
    ) -> Suspension:
        """Non-blocking pump. With ``user_input`` it starts a new user turn; with
        ``None`` it resumes a run parked on a task (whose result was already injected
        via report_task_result). Returns why the run suspended without blocking on
        tasks — the caller decides how to wait. Every non-``awaiting_tasks`` reason
        runs a settle checkpoint and attaches the ``AgentTurnResult`` as ``turn``."""
        session = self._require_open()
        if session.terminal:
            raise NativeAgentError(
                "run reached a terminal state and cannot accept more input",
                error_code="run_terminal",
            )
        state, res = session.state, session.res
        if user_input is not None:
            # Per-submit outcome fields describe this turn; reset before running.
            state.status = "completed"
            state.error = ""
            state.error_code = ""
            state.provider_error_code = ""
            state.provider_http_status = None
            state.final_text = ""
            # A run.finish in a prior submit must not short-circuit this one.
            res.context.finished = False
            state.pending_user_input = input_to_parts(user_input)
            self._warn_on_unforwarded_multimodal(state.pending_user_input, res.recorder)
            session.submit_local_step = 0
        try:
            suspension = self._pump_turn(state, res, session)
        except (RunCancelled, RunTimeout) as exc:
            state.status = "limited"
            state.error = str(exc)
            state.error_code = error_code_for_exception(exc)
            state.final_text = (
                "Stopped because the run was cancelled."
                if state.error_code == "cancelled"
                else "Stopped after reaching max duration."
            )
            session.terminal = True
            result = replace(
                Suspension(reason="terminal", status="limited"),
                final_text=state.final_text,
                error=state.error,
                error_code=state.error_code,
                turn=self._checkpoint_on_settle(state, res),
            )
            self._persist_checkpoint(session)
            return result
        except Exception as exc:  # controlled recording boundary for standalone CLI
            self._record_failure(state, res, exc)
            session.terminal = True
            result = replace(
                Suspension(reason="terminal", status="failed"),
                error=state.error,
                error_code=state.error_code,
                turn=self._checkpoint_on_settle(state, res),
            )
            self._persist_checkpoint(session)
            return result
        if suspension.reason == "awaiting_tasks":
            if suspension.has_external:
                # Parked on a hosted task awaiting an external report (hitl/automation).
                res.recorder.emit(
                    "run.awaiting_input",
                    data={"reason": "task", "task_ids": list(suspension.awaiting_task_ids)},
                )
            self._persist_checkpoint(session)
            return suspension
        if state.error_code == "max_tool_calls_exceeded":
            # Tool-call budget is session-cumulative; once spent the run is done.
            session.terminal = True
        result = replace(suspension, turn=self._checkpoint_on_settle(state, res))
        self._persist_checkpoint(session)
        return result

    def await_user_input(self) -> None:
        """Signal that the run is parked awaiting the next user message. A
        multi-turn driver calls this before blocking on its message channel."""
        session = self._require_open()
        session.res.recorder.emit("run.awaiting_input", data={"reason": "user"})

    def has_pending_tasks(self) -> bool:
        """Whether the run has resume-tasks still outstanding (not yet drained)."""
        session = self._require_open()
        return session.res.context.job_manager.has_resume_jobs()

    def wait_for_pending_tasks(self, timeout_s: float) -> bool:
        """Block up to ``timeout_s`` for a pending task to become ready (in-process
        completion or external report). Returns True if one is ready to drain, so
        the caller can ``run_until_suspended(None)`` to resume."""
        session = self._require_open()
        manager = session.res.context.job_manager
        deadline = time.time() + max(0.0, timeout_s)
        while manager.has_resume_jobs():
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            if manager.wait_for_reentry(min(0.25, remaining)):
                return True
        return False

    def close(self) -> AgentRunResult:
        """Finalize the run: cancel jobs, write the terminal proposal, emit
        run.finished, close the recorder, and return the cumulative result."""
        session = self._require_open()
        result = self._finalize(session.state, session.res)
        # A successfully completed run has nothing to recover: drop its checkpoints. A
        # failed/limited run KEEPS its checkpoints so the last-good one (named in
        # failure.json) is available for an operator-driven restore.
        if session.state.status == "completed":
            self._checkpoint_store().delete(self.spec.run_id)
        self._session = None
        return result

    def run_once(self, user_input: str | tuple[ContentPart, ...]) -> AgentRunResult:
        """One-shot convenience: open() + submit(user_input) + close()."""
        self.open()
        try:
            session = self._require_open()
            if not session.terminal:
                self.submit(user_input)
        finally:
            result = self.close()
        return result

    def _record_failure(self, state: RunState, res: _RunResources, exc: Exception) -> None:
        state.status = "failed"
        state.error = str(exc)
        state.error_code = error_code_for_exception(exc)
        if isinstance(exc, ModelAdapterError):
            state.provider_error_code = exc.provider_error_code
            state.provider_http_status = exc.http_status
        state.final_text = ""
        res.recorder.emit(
            "run.failed",
            data={
                "error": public_error_message(state.error),
                "error_code": state.error_code,
                "type": type(exc).__name__,
            },
            level="error",
        )
        # Failure bundle: what broke + which checkpoint to restore from. The last good
        # (non-terminal) checkpoint is the current sequence; the terminal checkpoint the
        # failure path writes next is seq+1 and is skipped by the restart scanner. No
        # auto-recovery — this is purely the operator's restore aid.
        last_good_seq = self._session.checkpoint_seq if self._session is not None else 0
        res.recorder.write_failure(
            {
                "schema_version": "native-agent-runner.failure.v1",
                "run_id": self.spec.run_id,
                "error": public_error_message(state.error),
                "error_code": state.error_code,
                "provider_error_code": state.provider_error_code,
                "type": type(exc).__name__,
                "last_good_seq": last_good_seq,
                "restore_hint": (
                    f"restore checkpoint seq {last_good_seq} for run {self.spec.run_id} "
                    "via CheckpointStore, then run_until_suspended(None)"
                )
                if last_good_seq > 0
                else "no committed checkpoint to restore from (failed before first park)",
            }
        )

    def commit_checkpoint(self) -> None:
        """Adopt the current proposed workspace state as the new diff baseline.

        Opt-in and never called automatically (at-close approval is the default).
        After this, subsequent proposals/diffs report only changes made after this
        point — the building block for incremental apply across a multi-turn run."""
        session = self._require_open()
        res = session.res
        res.workspace.snapshot_current_as_new_baseline()
        res.recorder.write_workspace_base(res.workspace.workspace_base_payload(self.spec.run_id))
        res.recorder.emit(
            "checkpoint.committed",
            data={"workspace_backend": res.workspace.backend_kind, "changed_paths": []},
        )

    def report_task_result(
        self, task_id: str, result: dict[str, Any], *, status: str = "answered"
    ) -> dict[str, Any]:
        """Complete a hosted task (e.g. a hitl request) from outside the loop —
        the backend or another thread calls this to deliver a result, waking a
        parked run. The result is injected per the task kind's ResultInjector."""
        session = self._require_open()
        return session.res.context.job_manager.report_result(task_id, result, status=status)

    # --- durable persistence (state-snapshot at park points) ---

    def snapshot(self) -> RunCheckpoint | None:
        """Capture the run's park-point state as a ``RunCheckpoint``, or ``None`` when
        a durable snapshot is unsafe right now. Pure read — never mutates state or jobs.

        Refuses (returns ``None``) while a live in-process (shell) resume-task is still
        running: its subprocess can't cross a process boundary, so the park only becomes
        durable once just hosted (hitl/automation) tasks remain. The conversation itself
        is held by the provider via ``previous_turn_handle``, so the LLM transcript is
        never serialized here."""
        session = self._require_open()
        state = session.state
        res = session.res
        manager = res.context.job_manager
        if manager.has_resume_jobs():
            hosted = set(manager.external_pending_task_ids())
            if not manager.outstanding_resume_task_ids().issubset(hosted):
                return None
        tasks_payload = manager.checkpoint_payload()
        pending_input = (
            [content_part_to_json(part) for part in state.pending_user_input]
            if state.pending_user_input is not None
            else None
        )
        return RunCheckpoint(
            run_id=self.spec.run_id,
            seq=session.checkpoint_seq,
            status=state.status,
            error=state.error,
            error_code=state.error_code,
            provider_error_code=state.provider_error_code,
            provider_http_status=state.provider_http_status,
            final_text=state.final_text,
            previous_turn_handle=state.previous_turn_handle,
            pending_user_input=pending_input,
            pending_observations=[obs.to_json() for obs in state.pending_observations],
            pending_binding_loads=list(state.pending_binding_loads),
            tool_call_counts=dict(state.tool_call_counts),
            # Latest runtime config travels in every park snapshot, so a mid-run config
            # change is re-persisted (recovery does not fall back to start-of-run config).
            previous_runtime_config=(
                state.previous_runtime_config.to_json()
                if state.previous_runtime_config is not None
                else None
            ),
            total_tool_calls=state.total_tool_calls,
            total_usage=dict(state.total_usage),
            messages=list(state.messages),
            session_step=session.session_step,
            submit_local_step=session.submit_local_step,
            terminal=session.terminal,
            hosted_tasks=tasks_payload["hosted_tasks"],
            reentry_queue=tasks_payload["reentry_queue"],
            delivered_reentry_jobs=tasks_payload["delivered_reentry_jobs"],
            workspace_delta=self._workspace_delta_entries(res.workspace),
            workspace_base=res.workspace.workspace_base_payload(self.spec.run_id),
            remaining_duration_s=(res.deadline - time.time()) if res.deadline is not None else None,
            cancellation_requested=bool(
                self.cancellation_token is not None and self.cancellation_token.requested
            ),
        )

    def _checkpoint_store(self) -> CheckpointStore:
        """The injected store, or a default local-fs store under the run root. The core
        only ever talks to this protocol — it never decides where bytes physically land."""
        if self.checkpoint_store is None:
            self.checkpoint_store = LocalFsCheckpointStore(self.spec.run_root)
        return self.checkpoint_store

    def _persist_checkpoint(self, session: _Session) -> None:
        """Best-effort durable checkpoint at a park point. No-op when ``snapshot()``
        refuses (a live shell job is parked-on) — that park is simply not durable yet.
        Advances the per-run sequence so the store commits a new last-good checkpoint."""
        checkpoint = self.snapshot()
        if checkpoint is None:
            return
        session.checkpoint_seq += 1
        checkpoint.seq = session.checkpoint_seq
        self._checkpoint_store().put(checkpoint, self.collect_checkpoint_blobs())

    @staticmethod
    def _workspace_delta_entries(workspace: Workspace) -> list[dict[str, Any]]:
        """Serialize the agent's created/modified/deleted files since the base. File
        content is NOT inline — it travels as a content-addressed blob keyed by
        ``content_sha256`` (see ``collect_checkpoint_blobs``)."""
        entries: list[dict[str, Any]] = []
        for entry in workspace.changed_entries():
            content_sha256 = sha256_bytes(entry.content) if entry.content is not None else None
            entries.append(
                {
                    "path": entry.path,
                    "kind": entry.kind,
                    "change_kind": entry.change_kind,
                    "base_sha256": entry.base_sha256,
                    "proposed_sha256": entry.proposed_sha256,
                    "content_sha256": content_sha256,
                }
            )
        return entries

    def collect_checkpoint_blobs(self) -> dict[str, bytes]:
        """Content-addressed blobs for the current park's workspace delta: the bytes of
        each created/modified file, keyed by sha256. Read at the same quiescent park as
        ``snapshot()`` so the keys match the manifest's ``content_sha256`` refs."""
        session = self._require_open()
        blobs: dict[str, bytes] = {}
        for entry in session.res.workspace.changed_entries():
            if entry.content is not None:
                blobs[sha256_bytes(entry.content)] = entry.content
        return blobs

    def restore(
        self,
        checkpoint: RunCheckpoint,
        *,
        blobs: Mapping[str, bytes] | Callable[[str], bytes] | None = None,
    ) -> None:
        """Reopen a previously-checkpointed run, rehydrating its session from
        ``checkpoint`` instead of starting fresh. Like ``open()`` but: no second
        ``run.started``/manifest, parked hosted tasks re-registered, the workspace delta
        re-applied (created/modified files restored from ``blobs``, deletions replayed),
        and any in-process shell job that died on the crash folded in as a failed
        observation so the model re-decides on the next pump.

        ``blobs`` supplies the content for the workspace delta — a mapping or a reader
        ``sha256 -> bytes`` (e.g. ``CheckpointStore.latest().blob``). The caller is
        expected to have re-provisioned the base workspace first; this only re-applies
        the agent's delta on top."""
        if self._session is not None:
            raise NativeAgentError("run is already open", error_code="run_already_open")
        self._restoring = True
        try:
            res = self._bootstrap()
        finally:
            self._restoring = False
        self._rehydrate(checkpoint, res, _as_blob_reader(blobs))

    def _rehydrate(self, cp: RunCheckpoint, res: _RunResources, blob_reader: Callable[[str], bytes]) -> None:
        # Deadline carry-over: downtime while parked does not count against
        # max_duration_s (a run parked overnight on a human should not time out). Keep
        # the elapsed-so-far consistent so _build_metrics duration stays sane.
        if cp.remaining_duration_s is not None:
            now = time.time()
            max_duration_s = self.spec.limits.max_duration_s
            started = (
                now - (max_duration_s - cp.remaining_duration_s)
                if max_duration_s is not None
                else res.started
            )
            res = replace(res, deadline=now + cp.remaining_duration_s, started=started)
        state = RunState(
            status=cp.status,
            error=cp.error,
            error_code=cp.error_code,
            provider_error_code=cp.provider_error_code,
            provider_http_status=cp.provider_http_status,
            final_text=cp.final_text,
            previous_turn_handle=cp.previous_turn_handle,
            pending_user_input=(
                tuple(content_part_from_json(part) for part in cp.pending_user_input)
                if cp.pending_user_input is not None
                else None
            ),
            pending_observations=tuple(ToolObservation.from_json(obs) for obs in cp.pending_observations),
            pending_binding_loads=tuple(cp.pending_binding_loads),
            tool_call_counts=dict(cp.tool_call_counts),
            previous_runtime_config=(
                AgentRuntimeConfig.from_json(cp.previous_runtime_config)
                if cp.previous_runtime_config is not None
                else None
            ),
            total_tool_calls=cp.total_tool_calls,
            total_usage=dict(cp.total_usage)
            or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            messages=list(cp.messages),
        )
        # Re-apply the agent's workspace delta on top of the (backend-re-provisioned)
        # base, so the restored workspace matches the checkpoint instant and
        # changed_entries() reports the same delta again.
        self._apply_workspace_delta(res.workspace, cp.workspace_delta, blob_reader, self.spec.limits)
        manager = res.context.job_manager
        manager.restore_state(
            [HostedTask.from_checkpoint(payload, res.recorder.artifacts_dir) for payload in cp.hosted_tasks],
            reentry_queue=cp.reentry_queue,
            delivered_reentry_jobs=cp.delivered_reentry_jobs,
        )
        crashed = self._crashed_shell_observations(res)
        if crashed:
            state.pending_observations = state.pending_observations + crashed
        if cp.cancellation_requested and self.cancellation_token is not None:
            self.cancellation_token.cancel()
        self._session = _Session(
            state=state,
            res=res,
            session_step=cp.session_step,
            submit_local_step=cp.submit_local_step,
            terminal=cp.terminal,
            # Continue the sequence so the next park commits cp.seq + 1.
            checkpoint_seq=cp.seq,
        )

    @staticmethod
    def _apply_workspace_delta(
        workspace: Workspace,
        entries: list[dict[str, Any]],
        blob_reader: Callable[[str], bytes],
        limits: RunLimits,
    ) -> None:
        """Replay a captured workspace delta into a freshly-bootstrapped workspace via
        its normal write surface, so the workspace tracks the same changes-vs-base. Writes
        go through ``write_bytes``/``mkdir``/``delete_path`` (not raw disk) so overlay and
        staging backends both report the delta. Deletions assume the base file was
        re-provisioned; a missing target is skipped rather than fatal. The same size caps
        as capture are enforced here as bytes are read, so a tampered/huge checkpoint cannot
        fill the disk on restore — over-cap refuses the restore (surfaced to the caller)."""
        total = 0
        for entry in entries:
            change_kind = entry.get("change_kind")
            path = entry.get("path")
            if change_kind in {"created", "modified"}:
                content_sha256 = entry.get("content_sha256")
                content = blob_reader(content_sha256) if content_sha256 else b""
                size = len(content)
                if size > limits.max_delta_file_bytes:
                    raise NativeAgentError(
                        f"workspace delta file exceeds size cap on restore: {path}",
                        error_code="workspace_delta_file_bytes_exceeded",
                    )
                total += size
                if total > limits.max_workspace_delta_bytes:
                    raise NativeAgentError(
                        "workspace delta exceeds total size cap on restore",
                        error_code="workspace_delta_bytes_exceeded",
                    )
                workspace.write_bytes(path, content, create_dirs=True)
            elif change_kind == "directory":
                workspace.mkdir(path)
            elif change_kind == "deleted":
                if workspace.exists(path):
                    workspace.delete_path(path, recursive=entry.get("kind") == "dir")

    def _crashed_shell_observations(self, res: _RunResources) -> tuple[ToolObservation, ...]:
        """A shell ``BackgroundJob`` left ``running`` in ``artifacts/jobs/*/job.json``
        means its subprocess was lost on the crash (it cannot be restored). Surface
        each as a failed background-job observation so the model re-decides; the
        original logs stay on disk untouched."""
        jobs_dir = res.recorder.artifacts_dir / "jobs"
        if not jobs_dir.is_dir():
            return ()
        observations: list[ToolObservation] = []
        for job_file in sorted(jobs_dir.glob("*/job.json")):
            try:
                payload = json.loads(job_file.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if payload.get("status") != "running":
                continue
            job_id = str(payload.get("job_id") or job_file.parent.name)
            observations.append(
                ToolObservation(
                    call_id=f"background:{job_id}",
                    tool_name="background_job",
                    output={
                        "type": "background_job_result",
                        "job_id": job_id,
                        "status": "failed",
                        "error": "process lost on restart",
                        "command_preview": str(payload.get("command_preview") or ""),
                    },
                    is_background=True,
                )
            )
        return tuple(observations)

    def create_task(self, kind: str, request: dict[str, Any]) -> str:
        """Create a task in the running run from outside the loop (backend-initiated
        automation/hitl). Returns the task id; its result is delivered later via
        report_task_result."""
        session = self._require_open()
        return session.res.context.job_manager.create_task(kind, request)

    def _require_open(self) -> _Session:
        if self._session is None:
            raise NativeAgentError("run is not open; call open() first", error_code="run_not_open")
        return self._session

    def _bootstrap(self) -> _RunResources:
        if self.permission_policy == PermissionPolicy() and self.spec.permission_policy != PermissionPolicy():
            self.permission_policy = self.spec.permission_policy
        workspace_factory = self.workspace_factory or default_local_workspace_factory
        workspace = workspace_factory(self.spec)
        recorder = AgentRecorder(
            self.spec.run_root,
            self.spec.run_id,
            extra_event_sinks=self.event_sinks,
            status_file=self.status_file,
            reopen=self._restoring,
        )
        job_manager = TaskManager(
            run_id=self.spec.run_id,
            workspace=workspace,
            recorder=recorder,
            permission_policy=self.permission_policy,
        )
        shell_service = ShellService(
            run_id=self.spec.run_id,
            workspace=workspace,
            recorder=recorder,
            job_manager=job_manager,
            permission_policy=self.permission_policy,
            approval_provider=self.shell_approval_provider,
        )
        web_service = WebService(
            recorder=recorder,
            web_gateway_client=self.web_gateway_client,
        )
        jobs_service = JobsService(job_manager=job_manager)
        context = AgentToolContext(
            self.spec.run_id,
            workspace,
            recorder,
            job_manager,
            shell_service,
            web_service,
            jobs_service,
            permission_policy=self.permission_policy,
        )
        base_registry = ToolRegistry()
        base_registry.register_many(builtin_tools(workspace))
        for provider in self.tool_providers:
            base_registry.register_many(provider.get_tools(context))

        started = time.time()
        deadline = (
            started + self.spec.limits.max_duration_s
            if self.spec.limits.max_duration_s is not None
            else None
        )
        self._bootstrap_resources = _RunResources(
            workspace=workspace,
            recorder=recorder,
            context=context,
            base_tool_specs=tuple(base_registry.specs()),
            started=started,
            deadline=deadline,
            static_segments=(),
        )
        initial_runtime_config = self._current_runtime_config(base_registry)
        initial_bound_catalog = compile_bound_tool_catalog(initial_runtime_config, base_registry)
        initial_turn = TurnContext(
            step=1,
            remaining_steps=max(0, self.spec.limits.max_steps - 1),
            remaining_tool_calls=self.spec.limits.max_tool_calls,
            deadline_s=(deadline - time.time()) if deadline is not None else None,
            plan=(),
            pending_observation_count=0,
        )
        initial_surface = self.tool_surface_resolver.resolve(
            bound_catalog=initial_bound_catalog,
            turn=initial_turn,
        )
        initial_visible_tool_specs = list(initial_surface.immediate_tools)
        workspace_index = build_workspace_index(workspace, run_id=self.spec.run_id)
        workspace_index_path = recorder.write_workspace_index(workspace_index)
        static_segments: list[str] = []
        if self.inject_workspace_index:
            index_segment = render_workspace_index_segment(workspace_index)
            if index_segment:
                static_segments.append(index_segment)
        for provider in self.context_providers:
            segment = provider.static_segment()
            if segment and segment.strip():
                static_segments.append(segment)
        # On restore (_rehydrate) the run dir already holds workspace.base.json,
        # manifest.json, and a recorded run.started. Re-writing the base would reset
        # the diff baseline; re-emitting run.started would double the lifecycle. Skip
        # all bootstrap side-effects and reuse what is already on disk.
        if not self._restoring:
            workspace_base_path = recorder.write_workspace_base(
                workspace.workspace_base_payload(self.spec.run_id)
            )
            manifest = build_run_manifest(
                self.spec,
                model_config=initial_runtime_config.model or ModelConfig(),
                tool_specs=initial_visible_tool_specs,
                permission_policy=self.permission_policy,
                tool_surface=tool_surface_manifest(
                    resolver=self.tool_surface_resolver,
                    tool_search=initial_runtime_config.tool_search,
                    dynamic_enabled=bool(self._dynamic_providers()),
                    initial_catalog_count=len(initial_bound_catalog.tools),
                ),
                agent_config={
                    "definition_id": initial_runtime_config.definition_id,
                    "config_version": initial_runtime_config.config_version,
                    "config_hash": initial_runtime_config.config_hash,
                },
                workspace_index_path=str(workspace_index_path.relative_to(recorder.run_dir).as_posix()),
                workspace_base_path=str(workspace_base_path.relative_to(recorder.run_dir).as_posix()),
            )
            recorder.write_manifest(manifest)
            recorder.emit(
                "run.started",
                data={
                    "workspace": str(self.spec.workspace_root),
                    "run_dir": str(recorder.run_dir),
                    "manifest_path": "manifest.json",
                    "mode": self.spec.mode,
                    "workspace_backend": self.spec.workspace_backend,
                    "workspace_base_path": "workspace.base.json",
                    "model_provider": (initial_runtime_config.model or ModelConfig()).provider,
                    "model": (initial_runtime_config.model or ModelConfig()).model,
                    "reasoning_effort": (initial_runtime_config.model or ModelConfig()).reasoning.effort,
                    "visible_bindings": [tool.id for tool in initial_visible_tool_specs],
                    "agent_config_hash": initial_runtime_config.config_hash,
                },
            )
        return _RunResources(
            workspace=workspace,
            recorder=recorder,
            context=context,
            base_tool_specs=tuple(base_registry.specs()),
            started=started,
            deadline=deadline,
            static_segments=tuple(static_segments),
        )

    def _warn_on_unforwarded_multimodal(
        self, parts: tuple[ContentPart, ...], recorder: AgentRecorder
    ) -> None:
        """Multimodal input is a contract-only surface for now: non-text parts are
        accepted on submit() but not yet threaded to any provider. Emit a warning
        naming the dropped part types and why, so the degradation is observable."""
        dropped = non_text_part_types(parts)
        if not dropped:
            return
        reason = "not_yet_forwarded" if getattr(self.model_adapter, "supports_multimodal", False) else "adapter_lacks_multimodal"
        recorder.emit(
            "model.input.degraded",
            data={"dropped_part_types": dropped, "reason": reason},
            level="warning",
        )

    def _dynamic_context_segment(self, res: _RunResources, turn_context: TurnContext) -> str:
        """Join each context provider's per-turn segment. Empty when no providers
        contribute, so the turn prompt stays byte-identical to the static prompt."""
        del res
        if not self.context_providers:
            return ""
        segments = []
        for provider in self.context_providers:
            segment = provider.dynamic_segment(turn_context)
            if segment and segment.strip():
                segments.append(segment.strip())
        return "\n\n".join(segments)

    def _turn_context(
        self, state: RunState, res: _RunResources, step: int, remaining_steps: int
    ) -> TurnContext:
        limits = self.spec.limits
        return TurnContext(
            step=step,
            remaining_steps=remaining_steps,
            remaining_tool_calls=max(0, limits.max_tool_calls - state.total_tool_calls),
            deadline_s=(res.deadline - time.time()) if res.deadline is not None else None,
            plan=tuple(res.context.plan),
            pending_observation_count=len(state.pending_observations),
        )

    def _registry_for_turn(
        self,
        context: AgentToolContext,
        turn: TurnContext,
        res: _RunResources,
    ) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register_many(res.base_tool_specs)
        for provider in self._dynamic_providers():
            registry.register_many(provider.get_tools_for_turn(context, turn))
        return registry

    def _dynamic_providers(self) -> tuple[DynamicToolProvider, ...]:
        providers: list[DynamicToolProvider] = list(self.dynamic_tool_providers)
        for provider in self.tool_providers:
            method = getattr(provider, "get_tools_for_turn", None)
            if callable(method):
                providers.append(provider)  # type: ignore[arg-type]
        return tuple(providers)

    def _current_runtime_config(self, registry: ToolRegistry, *, validate: bool = True) -> AgentRuntimeConfig:
        config = (
            self.runtime_config_provider.current_config(self.spec.run_id)
        )
        if config is None:
            raise AgentConfigError(
                "runtime config provider returned no config",
                error_code="agent_config_missing",
            )
        if validate:
            validate_runtime_config(config, registry)
        return config

    def _system_prompt_for_config(
        self,
        config: AgentRuntimeConfig,
        static_segments: tuple[str, ...],
    ) -> str:
        return compose_system_prompt(
            config.prompt.system_prompt_base or BASE_SYSTEM_PROMPT,
            (*config.prompt.persona_segments, *config.prompt.runtime_segments, *static_segments),
        )

    def _emit_runtime_config_if_changed(
        self,
        *,
        recorder: AgentRecorder,
        state: RunState,
        config: AgentRuntimeConfig,
        step: int,
        turn_id: str,
        parent_id: str | None,
    ) -> None:
        recorder.transcript(transcript_config_snapshot(config, step=step, turn_id=turn_id))
        previous = state.previous_runtime_config
        if previous is not None and previous.config_hash == config.config_hash:
            return
        diff = runtime_config_diff(previous, config)
        recorder.emit(
            "agent.config.updated",
            turn_id=turn_id,
            parent_id=parent_id,
            data={
                "definition_id": config.definition_id,
                "config_version": config.config_version,
                "config_hash": config.config_hash,
                "previous_config_version": None if previous is None else previous.config_version,
                "previous_config_hash": None if previous is None else previous.config_hash,
                "diff": diff,
            },
        )
        state.previous_runtime_config = config

    def _pump_turn(self, state: RunState, res: _RunResources, session: _Session) -> Suspension:
        context = res.context
        recorder = res.recorder
        deadline = res.deadline
        # The per-submit step budget continues across task-wait suspensions within one
        # submit; session_step is the global, monotonic turn counter for turn ids.
        max_steps = self.spec.limits.max_steps
        while session.submit_local_step < max_steps:
            self._check_run_boundary(deadline)
            session.submit_local_step += 1
            local_step = session.submit_local_step
            session.session_step += 1
            step = session.session_step
            background_observations = self._pop_background_observations(context, recorder, step)
            if background_observations:
                state.pending_observations = (*state.pending_observations, *background_observations)
            turn_id = f"turn_{step:04d}"
            turn_started = recorder.emit(
                "model.turn.started",
                turn_id=turn_id,
                data={"step": step, "previous_turn_handle": state.previous_turn_handle},
            )
            turn_context = self._turn_context(state, res, step, max(0, max_steps - local_step))
            turn_registry = self._registry_for_turn(context, turn_context, res)
            runtime_config = self._current_runtime_config(turn_registry)
            bound_catalog = compile_bound_tool_catalog(runtime_config, turn_registry)
            self._emit_runtime_config_if_changed(
                recorder=recorder,
                state=state,
                config=runtime_config,
                step=step,
                turn_id=turn_id,
                parent_id=turn_started.event_id,
            )
            surface_snapshot = self.tool_surface_resolver.resolve(
                bound_catalog=bound_catalog,
                turn=turn_context,
                pending_binding_loads=state.pending_binding_loads,
                previous_snapshot=state.previous_surface_snapshot,
                call_counts=state.tool_call_counts,
            )
            if not surface_snapshot.turn_id:
                surface_snapshot = replace(surface_snapshot, turn_id=turn_id)
            context.configure_tool_search(
                surface_snapshot.search_entries,
                runtime_config.tool_search.top_k,
            )
            snapshot_payload = surface_snapshot.to_transcript_json()
            snapshot_payload["step"] = step
            recorder.transcript(snapshot_payload)
            if (
                state.previous_surface_snapshot is None
                or state.previous_surface_snapshot.surface_hash != surface_snapshot.surface_hash
            ):
                recorder.emit(
                    "tool.surface.updated",
                    turn_id=turn_id,
                    parent_id=turn_started.event_id,
                    data=surface_snapshot.to_public_json(),
            )
            state.previous_surface_snapshot = surface_snapshot
            state.pending_binding_loads = ()
            dynamic_segment = self._dynamic_context_segment(res, turn_context)
            if surface_snapshot.delta_notice:
                dynamic_segment = (
                    surface_snapshot.delta_notice
                    if not dynamic_segment
                    else f"{dynamic_segment}\n\n{surface_snapshot.delta_notice}"
                )
            static_system_prompt = self._system_prompt_for_config(runtime_config, res.static_segments)
            turn_system_prompt = (
                static_system_prompt
                if not dynamic_segment
                else f"{static_system_prompt}\n\n{dynamic_segment}"
            )
            # The new user message is sent only on the first turn that consumes it
            # (the first turn of this submit); later turns of the same submit carry
            # observations against the continuation handle.
            instruction: str | None = None
            if state.pending_user_input is not None:
                instruction = text_from_parts(state.pending_user_input) or None
                state.pending_user_input = None
            # Accumulate the by-value conversation log BEFORE the call: the new user
            # message (if any) and the tool/async observations being sent this turn. The
            # assistant reply is appended after the call. The system prompt is NOT logged
            # here — it is regenerated per turn and travels via ``system_prompt``.
            if instruction is not None:
                state.messages.append({"role": "user", "content": instruction})
            for observation in state.pending_observations:
                state.messages.append(_observation_message(observation))
            # Bound the by-value conversation log: a runaway multi-turn run must settle
            # safely (status ``limited``, last-good checkpoint intact) rather than grow the
            # resent-every-turn log without limit. Checked before the call so an over-limit
            # log is never sent or re-persisted.
            log_limit_code = self._message_log_limit_exceeded(state)
            if log_limit_code is not None:
                state.status = "limited"
                state.final_text = "Stopped after reaching the conversation size limit."
                state.error_code = log_limit_code
                state.pending_observations = ()
                return Suspension(
                    reason="limited",
                    status="limited",
                    final_text=state.final_text,
                    error_code=log_limit_code,
                )
            delta_limit_code = self._workspace_delta_limit_exceeded(res.workspace)
            if delta_limit_code is not None:
                state.status = "limited"
                state.final_text = "Stopped after reaching the workspace change size limit."
                state.error_code = delta_limit_code
                state.pending_observations = ()
                return Suspension(
                    reason="limited",
                    status="limited",
                    final_text=state.final_text,
                    error_code=delta_limit_code,
                )
            request = ModelRequest(
                instruction=instruction,
                system_prompt=turn_system_prompt,
                tools=surface_snapshot.immediate_tools,
                previous_turn_handle=state.previous_turn_handle,
                observations=state.pending_observations,
                model=runtime_config.model or ModelConfig(),
                messages=tuple(state.messages),
            )
            recorder.transcript(
                {
                    "kind": "model_request",
                    "step": step,
                    "previous_turn_handle": state.previous_turn_handle,
                    "observations": [obs.__dict__ for obs in state.pending_observations],
                    "tool_surface_hash": surface_snapshot.surface_hash,
                }
            )
            try:
                turn = self.model_adapter.next_turn(request)
            except ModelAdapterError as exc:
                state.provider_error_code = exc.provider_error_code
                state.provider_http_status = exc.http_status
                recorder.transcript(
                    {
                        "kind": "model_turn",
                        "step": step,
                        "response_id": None,
                        "final_text": None,
                        "tool_calls": [],
                        "usage": {},
                        "error": str(exc),
                        "error_code": exc.error_code,
                        "provider_error_code": exc.provider_error_code,
                        "retryable": exc.retryable,
                        "http_status": exc.http_status,
                    }
                )
                raise
            except NativeAgentError:
                raise
            except Exception as exc:
                raise ModelAdapterError(str(exc)) from exc
            self._check_run_boundary(deadline)
            _accumulate_usage(state.total_usage, turn)
            state.previous_turn_handle = turn.response_id or state.previous_turn_handle
            # Append the assistant reply to the by-value log (text + any tool calls).
            state.messages.append(
                {
                    "role": "assistant",
                    "content": turn.final_text or "",
                    "tool_calls": [call.__dict__ for call in turn.tool_calls],
                }
            )
            recorder.transcript(
                {
                    "kind": "model_turn",
                    "step": step,
                    "response_id": turn.response_id,
                    "final_text": turn.final_text,
                    "tool_calls": [call.__dict__ for call in turn.tool_calls],
                    "usage": turn.usage,
                }
            )
            recorder.emit(
                "model.turn.finished",
                turn_id=turn_id,
                parent_id=turn_started.event_id,
                data={
                    "step": step,
                    "response_id": turn.response_id,
                    "tool_calls": len(turn.tool_calls),
                    "has_final": bool(turn.final_text),
                    "usage": turn.usage,
                },
            )
            recorder.emit(
                "metrics.updated",
                turn_id=turn_id,
                parent_id=turn_started.event_id,
                data={
                    "step": step,
                    "tool_calls": state.total_tool_calls,
                    "input_tokens": state.total_usage["input_tokens"],
                    "output_tokens": state.total_usage["output_tokens"],
                    "total_tokens": state.total_usage["total_tokens"],
                    "web_search_calls": context.web_service.web_search_calls,
                    "web_fetch_calls": context.web_service.web_fetch_calls,
                    "web_context_calls": context.web_service.web_context_calls,
                    "web_failed_calls": context.web_service.web_failed_calls,
                },
            )

            if not turn.tool_calls:
                if context.job_manager.has_resume_jobs():
                    # Park without blocking: clear the consumed observations and hand
                    # control back. The caller waits (in-process monitor completes, or
                    # an external reporter delivers) and resumes via run_until_suspended.
                    state.pending_observations = ()
                    external = context.job_manager.external_pending_task_ids()
                    return Suspension(
                        reason="awaiting_tasks",
                        status=state.status,  # type: ignore[arg-type]
                        awaiting_task_ids=tuple(external),
                        has_external=bool(external),
                    )
                if turn.final_text:
                    state.final_text = turn.final_text
                    # The model has consumed the pending observations and settled;
                    # the next submit must not resend them alongside a new message.
                    state.pending_observations = ()
                    return Suspension(reason="settled", status=state.status, final_text=state.final_text)  # type: ignore[arg-type]
                raise ModelAdapterError("model returned neither final text nor tool calls")

            observations: list[ToolObservation] = []
            for call in turn.tool_calls:
                self._check_run_boundary(deadline)
                state.total_tool_calls += 1
                if state.total_tool_calls > self.spec.limits.max_tool_calls:
                    state.status = "limited"
                    state.final_text = "Stopped after reaching max tool calls."
                    state.error_code = "max_tool_calls_exceeded"
                    break
                observation = self._execute_tool_call(
                    call_name=call.name,
                    call_id=call.id,
                    arguments=call.arguments,
                    bound_catalog=bound_catalog,
                    surface_snapshot=surface_snapshot,
                    call_counts=state.tool_call_counts,
                    context=context,
                    recorder=recorder,
                    turn_id=turn_id,
                    parent_id=turn_started.event_id,
                    step=step,
                )
                observations.append(observation)
                self._check_run_boundary(deadline)
            state.pending_binding_loads = _dedupe(
                (*state.pending_binding_loads, *context.consume_tool_load_requests())
            )
            state.pending_observations = tuple(observations)

            if context.finished:
                state.final_text = context.final_text
                return Suspension(reason="settled", status=state.status, final_text=state.final_text)  # type: ignore[arg-type]
            if state.status == "limited":
                return Suspension(
                    reason="limited",
                    status=state.status,  # type: ignore[arg-type]
                    final_text=state.final_text,
                    error_code=state.error_code,
                )
        state.status = "limited"
        state.final_text = "Stopped after reaching max steps."
        state.error_code = "max_steps_exceeded"
        return Suspension(
            reason="limited",
            status=state.status,  # type: ignore[arg-type]
            final_text=state.final_text,
            error_code=state.error_code,
        )

    def _message_log_limit_exceeded(self, state: RunState) -> str | None:
        """Return the limit error_code if the by-value conversation log has outgrown its
        bounds (count or approximate serialized bytes), else ``None``."""
        limits = self.spec.limits
        if len(state.messages) > limits.max_messages:
            return "message_count_exceeded"
        size = sum(len(json.dumps(message, ensure_ascii=False)) for message in state.messages)
        if size > limits.max_message_log_bytes:
            return "message_log_bytes_exceeded"
        return None

    def _workspace_delta_limit_exceeded(self, workspace: Workspace) -> str | None:
        """Return the limit error_code if the workspace delta a checkpoint would carry has
        outgrown its bounds (any single file, or the total), else ``None``. Mirrors the
        by-value message-log cap: an over-cap delta settles the run ``limited`` rather than
        being persisted into a checkpoint that would bloat the store."""
        limits = self.spec.limits
        total = 0
        for entry in workspace.changed_entries():
            if entry.content is None:
                continue
            size = len(entry.content)
            if size > limits.max_delta_file_bytes:
                return "workspace_delta_file_bytes_exceeded"
            total += size
            if total > limits.max_workspace_delta_bytes:
                return "workspace_delta_bytes_exceeded"
        return None

    def _build_metrics(self, state: RunState, res: _RunResources) -> dict[str, Any]:
        context = res.context
        model = (
            state.previous_runtime_config.model
            if state.previous_runtime_config is not None and state.previous_runtime_config.model is not None
            else ModelConfig()
        )
        metrics = {
            "status": state.status,
            "duration_s": time.time() - res.started,
            "steps_limit": self.spec.limits.max_steps,
            "tool_calls": state.total_tool_calls,
            "changed_paths": res.workspace.changed_paths(),
            "workspace_backend": self.spec.workspace_backend,
            "requested_reasoning_effort": model.reasoning.effort,
            "effective_reasoning_effort": model.reasoning.effort,
            "error_code": state.error_code,
            **context.shell_service.metrics(),
            **context.jobs_service.background_metrics(),
            **context.web_service.metrics(),
            **state.total_usage,
        }
        if state.provider_error_code:
            metrics["provider_error_code"] = state.provider_error_code
        if state.provider_http_status is not None:
            metrics["provider_http_status"] = state.provider_http_status
        if state.error:
            metrics["error"] = state.error
        return metrics

    def _finalize(self, state: RunState, res: _RunResources) -> AgentRunResult:
        context = res.context
        recorder = res.recorder
        workspace = res.workspace
        context.job_manager.cancel_all()
        diff_path = recorder.write_diff(workspace.diff_patch())
        proposal_payload = recorder.write_proposal_snapshot(workspace, diff_path)
        metrics = self._build_metrics(state, res)
        recorder.write_metrics(metrics)
        recorder.emit(
            "workspace.proposal.updated",
            data=public_proposal_payload(proposal_payload, self.permission_policy),
        )
        recorder.emit(
            "proposal.ready",
            data={
                "proposal_hash": proposal_payload.get("proposal_hash"),
                "diff_sha256": proposal_payload.get("diff_sha256"),
                "changed_paths": [
                    public_path(str(path), self.permission_policy)
                    for path in proposal_payload.get("changed_paths", [])
                ],
            },
        )
        recorder.emit(
            "run.finished",
            data={
                "status": state.status,
                "error": public_error_message(state.error),
                "error_code": state.error_code,
                "final_text": state.final_text,
                "duration_s": metrics["duration_s"],
                "diff_path": str(diff_path.relative_to(recorder.run_dir)),
                "proposal_path": "proposal.json",
                "metrics_path": "metrics.json",
            },
            level="error" if state.status == "failed" else "info",
        )
        artifacts = tuple(recorder.artifacts)
        run_dir = recorder.run_dir
        recorder.close()
        return AgentRunResult(
            run_id=self.spec.run_id,
            status=state.status,  # type: ignore[arg-type]
            final_text=state.final_text,
            run_dir=run_dir,
            diff_path=diff_path,
            proposal_path=run_dir / "proposal.json",
            artifacts=artifacts,
            final_outputs=tuple(context.final_outputs),
            final_notes=context.final_notes,
            metrics=metrics,
            error=state.error,
            error_code=state.error_code,
            final_turn_handle=state.previous_turn_handle,
        )

    def _checkpoint_on_settle(self, state: RunState, res: _RunResources) -> AgentTurnResult:
        """Preview-only checkpoint at a settle point: flush the accumulated proposal
        and metrics, emit ``turn.settled``, and keep the run open. Repeatable — it
        does not cancel jobs, emit ``proposal.ready``/``run.finished``, or close the
        recorder. Those happen once in close()/``_finalize``."""
        recorder = res.recorder
        workspace = res.workspace
        diff_path = recorder.write_diff(workspace.diff_patch())
        proposal_payload = recorder.write_proposal_snapshot(workspace, diff_path)
        metrics = self._build_metrics(state, res)
        recorder.write_metrics(metrics)
        recorder.emit(
            "workspace.proposal.updated",
            data=public_proposal_payload(proposal_payload, self.permission_policy),
        )
        public_changed = [
            public_path(str(path), self.permission_policy)
            for path in proposal_payload.get("changed_paths", [])
        ]
        recorder.emit(
            "turn.settled",
            data={
                "status": state.status,
                "final_text": state.final_text,
                "error_code": state.error_code,
                "changed_paths": public_changed,
            },
        )
        return AgentTurnResult(
            status=state.status,  # type: ignore[arg-type]
            final_text=state.final_text,
            proposal_path=recorder.run_dir / "proposal.json",
            proposal_hash=str(proposal_payload.get("proposal_hash") or ""),
            changed_paths=tuple(workspace.changed_paths()),
            turn_handle=state.previous_turn_handle,
            error=state.error,
            error_code=state.error_code,
            metrics=metrics,
        )

    def _pop_background_observations(
        self,
        context: AgentToolContext,
        recorder: AgentRecorder,
        step: int,
    ) -> tuple[ToolObservation, ...]:
        observations = context.job_manager.pop_reentry_observations()
        if not observations:
            return ()
        recorder.emit(
            "run.resumed",
            data={
                "reason": "background_job_result",
                "job_ids": [str(obs.output.get("job_id") or "") for obs in observations],
                "count": len(observations),
            },
        )
        for observation in observations:
            recorder.transcript(
                {
                    "kind": "tool_observation",
                    "step": step,
                    "call_id": observation.call_id,
                    "tool": observation.tool_name,
                    "output": observation.output,
                }
            )
            # Workspace diffs are shell-specific; gate on the shell result payload
            # so hitl/automation results don't emit phantom workspace events.
            if observation.output.get("type") == "background_job_result":
                self._emit_background_workspace_events(observation.output, context, recorder)
        return tuple(observations)

    def _wait_for_background_jobs(
        self,
        context: AgentToolContext,
        recorder: AgentRecorder,
        deadline: float | None,
    ) -> None:
        recorder.emit(
            "run.waiting",
            data={
                "reason": "waiting_for_background_jobs",
                "jobs": [
                    {
                        "job_id": job.get("job_id"),
                        "status": job.get("status"),
                        "resume_on_exit": job.get("resume_on_exit"),
                    }
                    for job in context.job_manager.list_jobs()
                    if job.get("status") == "running" and job.get("resume_on_exit")
                ],
            },
        )
        while context.job_manager.has_resume_jobs():
            self._check_run_boundary(deadline)
            wait_s = 0.25
            if deadline is not None:
                wait_s = max(0.01, min(wait_s, deadline - time.time()))
            if context.job_manager.wait_for_reentry(wait_s):
                return

    def _emit_background_workspace_events(
        self,
        payload: dict[str, Any],
        context: AgentToolContext,
        recorder: AgentRecorder,
    ) -> None:
        changed_paths = [
            public_path(str(path), self.permission_policy)
            for path in payload.get("changed_paths", [])
        ]
        if not changed_paths:
            return
        recorder.emit(
            "workspace.file.changed",
            data={
                "tool": "shell.exec",
                "job_id": payload.get("job_id"),
                "paths": changed_paths,
                "result": {
                    "status": payload.get("status"),
                    "exit_code": payload.get("exit_code"),
                    "duration_s": payload.get("duration_s"),
                    "stdout_bytes": payload.get("stdout_bytes"),
                    "stderr_bytes": payload.get("stderr_bytes"),
                },
                "mode": context.workspace.mode,
            },
        )
        self._emit_workspace_proposal(context, recorder)

    def _emit_workspace_proposal(
        self,
        context: AgentToolContext,
        recorder: AgentRecorder,
        *,
        turn_id: str | None = None,
        parent_id: str | None = None,
    ) -> None:
        diff_text = context.workspace.diff_patch()
        diff_path = recorder.write_diff(diff_text)
        recorder.emit(
            "workspace.diff.updated",
            turn_id=turn_id,
            parent_id=parent_id,
            data={
                "path": str(diff_path.relative_to(recorder.run_dir)),
                "bytes": len(diff_text.encode("utf-8")),
                "changed_paths": [public_path(path, self.permission_policy) for path in context.workspace.changed_paths()],
            },
        )
        proposal_payload = recorder.write_proposal_snapshot(context.workspace, diff_path)
        recorder.emit(
            "workspace.proposal.updated",
            turn_id=turn_id,
            parent_id=parent_id,
            data=public_proposal_payload(proposal_payload, self.permission_policy),
        )

    def _emit_tool_started(
        self,
        recorder: AgentRecorder,
        *,
        call_name: str,
        call_id: str,
        spec: ToolSpec | None,
        arguments: dict[str, Any],
        turn_id: str,
        parent_id: str | None,
    ) -> AgentEvent:
        return recorder.emit(
            "tool.call.started",
            turn_id=turn_id,
            parent_id=parent_id,
            data=_tool_start_data(call_name, call_id, spec, arguments, self.permission_policy),
        )

    def _authorize_surface_tool(
        self,
        bound_tool: BoundTool,
        snapshot: ToolSurfaceSnapshot,
        call_counts: dict[str, int],
    ) -> ToolAuthorization:
        binding_id = bound_tool.binding_id
        immediate_binding_ids = {tool.id for tool in snapshot.immediate_tools}
        authorization = snapshot.authorization_for(binding_id)
        if authorization is not None and authorization.decision == "deny":
            raise PermissionDenied(
                f"tool binding denied by config: {binding_id}",
                error_code="tool_binding_denied",
            )
        if binding_id not in immediate_binding_ids or authorization is None:
            raise PermissionDenied(
                f"tool binding is not available in this turn: {binding_id}",
                error_code="tool_not_in_surface",
            )
        if authorization.decision == "ask":
            raise PermissionDenied(
                f"tool binding requires approval: {binding_id}",
                error_code="tool_approval_required",
            )
        max_calls = authorization.quota.max_calls_per_run
        if max_calls is not None and call_counts.get(binding_id, 0) >= max_calls:
            raise PermissionDenied(
                f"tool binding quota exceeded: {binding_id}",
                error_code="tool_quota_exceeded",
            )
        return authorization

    def _check_tool_surface_scope(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        authorization: ToolAuthorization,
    ) -> None:
        scope = authorization.surface_scope
        paths = tuple(
            str(arguments[name])
            for name in spec.path_args
            if name in arguments and arguments[name] is not None
        )
        for path in paths:
            if scope.allowed_paths and not matches_path_patterns(path, scope.allowed_paths):
                raise PermissionDenied(
                    f"tool path outside allowed scope: {spec.id}",
                    error_code="tool_scope_denied",
                )
            if scope.denied_paths and matches_path_patterns(path, scope.denied_paths):
                raise PermissionDenied(
                    f"tool path denied by scope: {spec.id}",
                    error_code="tool_scope_denied",
                )
        if spec.preview_kind == "web":
            for url in _urls_from_args(arguments):
                if not url:
                    continue
                if not domain_allowed(
                    domain_from_url(url),
                    allowed_domains=scope.allowed_domains,
                    blocked_domains=scope.blocked_domains,
                ):
                    raise PermissionDenied(
                        f"tool web domain denied by scope: {spec.id}",
                        error_code="tool_scope_denied",
                    )
        if spec.preview_kind == "shell":
            command = str(arguments.get("command") or "")
            if any(command.strip().startswith(prefix) for prefix in scope.command_deny_prefixes):
                raise PermissionDenied(
                    f"tool shell command denied by scope: {spec.id}",
                    error_code="tool_scope_denied",
                )
            if scope.command_allow_prefixes and not any(
                command.strip().startswith(prefix) for prefix in scope.command_allow_prefixes
            ):
                raise PermissionDenied(
                    f"tool shell command outside allowed scope: {spec.id}",
                    error_code="tool_scope_denied",
                )

    def _invoke_handler(
        self,
        bound_tool: BoundTool,
        context: AgentToolContext,
        arguments: dict[str, Any],
        *,
        call_id: str,
        turn_id: str,
        recorder: AgentRecorder,
        started_event: AgentEvent,
        authorization: ToolAuthorization,
    ) -> ToolResult:
        spec = bound_tool.base_spec
        context._current_call = CallContext(
            tool_call_id=call_id,
            turn_id=turn_id,
            tool_event_id=started_event.event_id,
            binding_id=bound_tool.binding_id,
            tool_id=bound_tool.base_spec.id,
            model_name=bound_tool.model_name,
            authorization=authorization,
            scope=authorization.scope,
            runtime=bound_tool.runtime,
        )
        try:
            result = spec.handler(context, arguments)
        finally:
            context._current_call = CallContext("", None, None)
        if result.ok:
            self._emit_side_effect_event(spec, arguments, result, context, recorder, turn_id, started_event.event_id)
        return result

    def _finalize_tool_call(
        self,
        recorder: AgentRecorder,
        *,
        spec: ToolSpec | None,
        result: ToolResult,
        started_event: AgentEvent | None,
        call_name: str,
        call_id: str,
        step: int,
        turn_id: str,
        parent_id: str | None,
    ) -> ToolObservation:
        observation = ToolObservation(
            call_id=call_id,
            tool_name=call_name,
            output=result.to_observation(),
        )
        recorder.transcript(
            {
                "kind": "tool_observation",
                "step": step,
                "call_id": call_id,
                "tool": call_name,
                "tool_id": spec.id if spec is not None else None,
                "output": observation.output,
            }
        )
        finish_type = "tool.call.finished" if result.ok else "tool.call.failed"
        recorder.emit(
            finish_type,
            turn_id=turn_id,
            parent_id=started_event.event_id if started_event else parent_id,
            data={
                "call_id": call_id,
                "tool": call_name,
                "ok": result.ok,
                "error": public_error_message(result.error),
                "error_code": result.error_code,
            },
            level="info" if result.ok else "warning",
        )
        return observation

    def _execute_tool_call(
        self,
        *,
        call_name: str,
        call_id: str,
        arguments: dict[str, Any],
        bound_catalog: BoundToolCatalog,
        surface_snapshot: ToolSurfaceSnapshot,
        call_counts: dict[str, int],
        context: AgentToolContext,
        recorder: AgentRecorder,
        turn_id: str,
        parent_id: str | None,
        step: int,
    ) -> ToolObservation:
        spec: ToolSpec | None = None
        bound_tool: BoundTool | None = None
        result: ToolResult
        started_event: AgentEvent | None = None
        surface_decision = ""
        surface_reason = ""
        try:
            if _is_tool_search_call(call_name, bound_catalog):
                binding_id = bound_catalog.tool_search.binding_id
                spec = _surface_spec_for_binding(surface_snapshot, binding_id)
                if spec is None:
                    raise PermissionDenied(
                        f"tool binding is not available in this turn: {binding_id}",
                        error_code="tool_not_in_surface",
                    )
                started_event = self._emit_tool_started(
                    recorder,
                    call_name=call_name,
                    call_id=call_id,
                    spec=spec,
                    arguments=arguments,
                    turn_id=turn_id,
                    parent_id=parent_id,
                )
                authorization = surface_snapshot.authorization_for(binding_id)
                if authorization is None or authorization.decision == "deny":
                    raise PermissionDenied(
                        f"tool binding denied by config: {binding_id}",
                        error_code="tool_binding_denied",
                    )
                surface_decision = authorization.decision
                surface_reason = authorization.reason
                if authorization.decision == "ask":
                    raise PermissionDenied(
                        f"tool binding requires approval: {binding_id}",
                        error_code="tool_approval_required",
                    )
                ToolRegistry().validate_args(spec, arguments)
                call_counts[binding_id] = call_counts.get(binding_id, 0) + 1
                result = ToolResult(ok=True, content=context.search_tools(arguments))
            else:
                bound_tool = bound_catalog.resolve_model_call(call_name)
                if bound_tool is None:
                    raise ToolExecutionError(f"unknown tool: {call_name}", error_code="tool_unknown")
                spec = bound_tool.model_spec
                started_event = self._emit_tool_started(
                    recorder,
                    call_name=call_name,
                    call_id=call_id,
                    spec=spec,
                    arguments=arguments,
                    turn_id=turn_id,
                    parent_id=parent_id,
                )
                preview_authorization = surface_snapshot.authorization_for(bound_tool.binding_id)
                if preview_authorization is not None:
                    surface_decision = preview_authorization.decision
                    surface_reason = preview_authorization.reason
                authorization = self._authorize_surface_tool(
                    bound_tool,
                    surface_snapshot,
                    call_counts,
                )
                ToolRegistry().validate_args(spec, arguments)
                self._check_tool_surface_scope(spec, arguments, authorization)
                self._check_permissions(bound_tool.base_spec, arguments)
                call_counts[bound_tool.binding_id] = call_counts.get(bound_tool.binding_id, 0) + 1
                result = self._invoke_handler(
                    bound_tool,
                    context,
                    arguments,
                    call_id=call_id,
                    turn_id=turn_id,
                    recorder=recorder,
                    started_event=started_event,
                    authorization=authorization,
                )
        except ToolExecutionError as exc:
            if started_event is None:
                started_event = self._emit_tool_started(
                    recorder,
                    call_name=call_name,
                    call_id=call_id,
                    spec=spec,
                    arguments=arguments,
                    turn_id=turn_id,
                    parent_id=parent_id,
                )
            result = _failure_result(exc)
        except PermissionDenied as exc:
            result = _failure_result(exc)
            recorder.emit(
                "permission.denied",
                turn_id=turn_id,
                parent_id=started_event.event_id if started_event else parent_id,
                data={
                    "call_id": call_id,
                    "tool": spec.id if spec is not None else call_name,
                    "requested_tool": call_name,
                    "error": public_error_message(str(exc)),
                    "error_code": result.error_code,
                    "surface_decision": surface_decision or None,
                    "surface_reason": surface_reason or None,
                },
                level="warning",
            )
        except (NativeAgentError, ValueError, TypeError) as exc:
            result = _failure_result(exc)
            if started_event is None:
                started_event = self._emit_tool_started(
                    recorder,
                    call_name=call_name,
                    call_id=call_id,
                    spec=spec,
                    arguments=arguments,
                    turn_id=turn_id,
                    parent_id=parent_id,
                )

        return self._finalize_tool_call(
            recorder,
            spec=spec,
            result=result,
            started_event=started_event,
            call_name=call_name,
            call_id=call_id,
            step=step,
            turn_id=turn_id,
            parent_id=parent_id,
        )

    def _check_run_boundary(self, deadline: float | None) -> None:
        if self.cancellation_token is not None and self.cancellation_token.requested:
            raise RunCancelled("run cancelled")
        if deadline is not None and time.time() >= deadline:
            raise RunTimeout("run exceeded max duration")

    def _emit_side_effect_event(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
        result: ToolResult,
        context: AgentToolContext,
        recorder: AgentRecorder,
        turn_id: str,
        parent_id: str | None,
    ) -> None:
        if spec.side_effect == "read" and spec.path_args:
            recorder.emit(
                "workspace.file.read",
                turn_id=turn_id,
                parent_id=parent_id,
                data={"tool": spec.id, "paths": _public_paths_from_args(spec, arguments, self.permission_policy)},
            )
        elif spec.emits_workspace_diff:
            if (
                spec.skip_emit_if_background
                and result.content.get("job_id")
                and result.content.get("status") == "running"
            ):
                return
            if spec.changed_paths_source == "result_content":
                paths = [
                    public_path(str(path), self.permission_policy)
                    for path in result.content.get("changed_paths", [])
                ]
            else:
                paths = _public_paths_from_args(spec, arguments, self.permission_policy)
            if spec.result_payload_kind == "shell_exec":
                result_payload = _shell_result_payload(result)
            else:
                result_payload = public_result_content(result.content, self.permission_policy)
            recorder.emit(
                "workspace.file.changed",
                turn_id=turn_id,
                parent_id=parent_id,
                data={
                    "tool": spec.id,
                    "paths": paths,
                    "result": result_payload,
                    "mode": context.workspace.mode,
                },
            )
            self._emit_workspace_proposal(context, recorder, turn_id=turn_id, parent_id=parent_id)
        elif spec.side_effect == "write" and spec.path_args:
            recorder.emit(
                "workspace.file.changed",
                turn_id=turn_id,
                parent_id=parent_id,
                data={
                    "tool": spec.id,
                    "paths": _public_paths_from_args(spec, arguments, self.permission_policy),
                    "result": public_result_content(result.content, self.permission_policy),
                    "mode": context.workspace.mode,
                },
            )
            self._emit_workspace_proposal(context, recorder, turn_id=turn_id, parent_id=parent_id)

    def _check_permissions(
        self,
        spec: ToolSpec,
        arguments: dict[str, Any],
    ) -> None:
        paths = tuple(
            str(arguments[name])
            for name in spec.path_args
            if name in arguments and arguments[name] is not None
        )
        if not paths:
            return
        operation = "read" if spec.side_effect in {"read", "artifact"} else "write"
        self.permission_policy.check_paths(operation, paths)  # type: ignore[arg-type]


def _accumulate_usage(total_usage: dict[str, int], turn: ModelTurn) -> None:
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        total_usage[key] += int(turn.usage.get(key, 0))


def _dedupe(items: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def _rank_tool_search_entries(
    query: str,
    entries: tuple[ToolSearchEntry, ...],
) -> list[ToolSearchEntry]:
    terms = [term for term in query.lower().split() if term]
    if not terms:
        return list(entries)

    def score(entry: ToolSearchEntry) -> int:
        haystack = " ".join(
            [
                entry.tool_id,
                entry.exported_name,
                entry.title,
                entry.summary,
                entry.guidance.summary,
                entry.guidance.policy,
            ]
        ).lower()
        return sum(1 for term in terms if term in haystack)

    scored = [(score(entry), index, entry) for index, entry in enumerate(entries)]
    return [entry for value, _index, entry in sorted(scored, key=lambda item: (-item[0], item[1])) if value > 0]


def _is_tool_search_call(name: str, catalog: BoundToolCatalog) -> bool:
    return catalog.tool_search.enabled and name in {
        catalog.tool_search.binding_id,
        catalog.tool_search.model_name,
    }


def _surface_spec_for_binding(snapshot: ToolSurfaceSnapshot, binding_id: str) -> ToolSpec | None:
    for tool in snapshot.immediate_tools:
        if tool.id == binding_id or str(tool.annotations.get("binding_id") or "") == binding_id:
            return tool
    return None


def _urls_from_args(arguments: dict[str, Any]) -> tuple[str, ...]:
    urls: list[str] = []
    raw_url = arguments.get("url")
    if isinstance(raw_url, str):
        urls.append(raw_url)
    raw_urls = arguments.get("urls")
    if isinstance(raw_urls, list | tuple):
        urls.extend(str(item) for item in raw_urls)
    return tuple(urls)


def _shell_result_payload(result: ToolResult) -> dict[str, Any]:
    return {
        "exit_code": result.content.get("exit_code"),
        "duration_s": result.content.get("duration_s"),
        "stdout_bytes": result.content.get("stdout_bytes"),
        "stderr_bytes": result.content.get("stderr_bytes"),
    }


def _tool_start_data(
    call_name: str,
    call_id: str,
    spec: ToolSpec | None,
    arguments: dict[str, Any],
    permission_policy: PermissionPolicy,
) -> dict[str, Any]:
    preview_kind = spec.preview_kind if spec is not None else "args"
    if preview_kind == "shell":
        preview = shell_args_preview(arguments, permission_policy)
    elif preview_kind == "web":
        preview = web_args_preview(arguments, permission_policy)
    else:
        preview = args_preview(arguments, permission_policy)
    return {
        "call_id": call_id,
        "tool": call_name,
        "capability": spec.capability if spec is not None else None,
        "side_effect": spec.side_effect if spec is not None else None,
        "paths": _public_paths_from_args(spec, arguments, permission_policy) if spec is not None else [],
        "args_preview": preview,
    }


def _public_paths_from_args(
    spec: ToolSpec,
    arguments: dict[str, Any],
    permission_policy: PermissionPolicy,
) -> list[str]:
    return [
        public_path(str(arguments[name]), permission_policy)
        for name in spec.path_args
        if name in arguments and arguments[name] is not None
    ]
