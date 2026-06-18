from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import Any

from native_agent_runner.core.cancellation import CancellationToken
from native_agent_runner.core.events import AgentEvent, EventSink
from native_agent_runner.core.content import ContentPart, non_text_part_types
from native_agent_runner.core.context import (
    ContextProvider,
    TurnContext,
    render_workspace_index_segment,
)
from native_agent_runner.core.agents import (
    AgentRuntimeConfig,
    BoundTool,
    BoundToolCatalog,
    RuntimeConfigProvider,
    compile_bound_tool_catalog,
    runtime_config_diff,
    transcript_config_snapshot,
    validate_runtime_config,
)
from native_agent_runner.core.manifest import build_run_manifest
from native_agent_runner.core.prompt import BASE_SYSTEM_PROMPT, compose_system_prompt
from native_agent_runner.core.result import AgentRunResult, AgentTurnResult
from native_agent_runner.core.spec import AgentRunSpec, ModelConfig, input_to_parts, text_from_parts
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
from native_agent_runner.jobs import BackgroundJobManager
from native_agent_runner.permissions import PermissionPolicy, matches_path_patterns
from native_agent_runner.providers.base import ModelAdapter, ModelRequest, ModelTurn, ToolObservation
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
    job_manager: BackgroundJobManager
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
    terminal: bool = False


@dataclass
class AgentLoop:
    spec: AgentRunSpec
    model_adapter: ModelAdapter
    _: KW_ONLY
    runtime_config_provider: RuntimeConfigProvider
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
    _bootstrap_resources: _RunResources | None = field(default=None, init=False, repr=False)
    _session: _Session | None = field(default=None, init=False, repr=False)

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
        open afterwards; call submit() again to continue or close() to finalize."""
        session = self._require_open()
        if session.terminal:
            raise NativeAgentError(
                "run reached a terminal state and cannot accept more input",
                error_code="run_terminal",
            )
        state, res = session.state, session.res
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
        try:
            self._run_submit(state, res, session)
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
        except Exception as exc:  # controlled recording boundary for standalone CLI
            self._record_failure(state, res, exc)
            session.terminal = True
        if state.error_code == "max_tool_calls_exceeded":
            # Tool-call budget is session-cumulative; once spent the run is done.
            session.terminal = True
        return self._checkpoint_on_settle(state, res)

    def close(self) -> AgentRunResult:
        """Finalize the run: cancel jobs, write the terminal proposal, emit
        run.finished, close the recorder, and return the cumulative result."""
        session = self._require_open()
        result = self._finalize(session.state, session.res)
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
        )
        job_manager = BackgroundJobManager(
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

    def _run_submit(self, state: RunState, res: _RunResources, session: _Session) -> None:
        context = res.context
        recorder = res.recorder
        deadline = res.deadline
        # Per-submit step budget: each user turn gets a fresh max_steps. session_step
        # is the global, monotonic turn counter used for unique turn ids.
        max_steps = self.spec.limits.max_steps
        for local_step in range(1, max_steps + 1):
            self._check_run_boundary(deadline)
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
            request = ModelRequest(
                instruction=instruction,
                system_prompt=turn_system_prompt,
                tools=surface_snapshot.immediate_tools,
                previous_turn_handle=state.previous_turn_handle,
                observations=state.pending_observations,
                model=runtime_config.model or ModelConfig(),
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
                    self._wait_for_background_jobs(context, recorder, deadline)
                    state.pending_observations = ()
                    continue
                if turn.final_text:
                    state.final_text = turn.final_text
                    # The model has consumed the pending observations and settled;
                    # the next submit must not resend them alongside a new message.
                    state.pending_observations = ()
                    break
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
                break
            if state.status == "limited":
                break
        else:
            state.status = "limited"
            state.final_text = "Stopped after reaching max steps."
            state.error_code = "max_steps_exceeded"

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
