from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from native_agent_runner.core.cancellation import CancellationToken
from native_agent_runner.core.events import AgentEvent, EventSink
from native_agent_runner.core.content import MEDIA_INPUT_CAPABILITY, non_text_part_types
from native_agent_runner.core.context import (
    ContextProvider,
    TurnContext,
    render_workspace_index_segment,
)
from native_agent_runner.core.manifest import build_run_manifest
from native_agent_runner.core.prompt import BASE_SYSTEM_PROMPT, compose_system_prompt
from native_agent_runner.core.result import AgentRunResult
from native_agent_runner.core.spec import AgentRunSpec
from native_agent_runner.core.workspace_index import build_workspace_index
from native_agent_runner.errors import (
    ModelAdapterError,
    NativeAgentError,
    PermissionDenied,
    RunCancelled,
    RunTimeout,
    ToolExecutionError,
    ToolPolicyError,
    error_code_for_exception,
)
from native_agent_runner.jobs import BackgroundJobManager
from native_agent_runner.permissions import PermissionPolicy
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
from native_agent_runner.shell import (
    AutoApproveShellApprovalProvider,
    DenyShellApprovalProvider,
    ShellPolicy,
    ShellApprovalProvider,
)
from native_agent_runner.tool_services import CallContext, JobsService, ShellService, WebService
from native_agent_runner.tools.base import ToolContext, ToolProvider, ToolRegistry, ToolResult, ToolSpec
from native_agent_runner.tools.builtin import builtin_tools
from native_agent_runner.tools.policy import NormalizedToolPolicy
from native_agent_runner.web import WebGatewayClient
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

    def execute_web_search(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.web_service.search(args, self._current_call)

    def execute_web_fetch(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.web_service.fetch(args, self._current_call)

    def execute_web_context(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.web_service.context(args, self._current_call)


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
    pending_observations: tuple[ToolObservation, ...] = ()
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
    registry: ToolRegistry
    tool_policy: NormalizedToolPolicy
    visible_tool_specs: list[ToolSpec]
    capabilities: frozenset[str]
    started: float
    deadline: float | None
    system_prompt: str


@dataclass
class AgentLoop:
    spec: AgentRunSpec
    model_adapter: ModelAdapter
    tool_providers: tuple[ToolProvider, ...] = ()
    event_sinks: tuple[EventSink, ...] = ()
    status_file: bool = True
    permission_policy: PermissionPolicy = field(default_factory=PermissionPolicy)
    cancellation_token: CancellationToken | None = None
    shell_approval_provider: ShellApprovalProvider | None = None
    web_gateway_client: WebGatewayClient | None = None
    workspace_factory: Callable[[AgentRunSpec], Workspace] | None = None
    context_providers: tuple[ContextProvider, ...] = ()
    inject_workspace_index: bool = False

    def run(self) -> AgentRunResult:
        res = self._bootstrap()
        state = RunState()
        try:
            self._run_steps(state, res)
        except (RunCancelled, RunTimeout) as exc:
            state.status = "limited"
            state.error = str(exc)
            state.error_code = error_code_for_exception(exc)
            state.final_text = (
                "Stopped because the run was cancelled."
                if state.error_code == "cancelled"
                else "Stopped after reaching max duration."
            )
        except Exception as exc:  # controlled recording boundary for standalone CLI
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
        finally:
            result = self._finalize(state, res)
        return result

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
            shell_policy=self.spec.shell_policy,
            permission_policy=self.permission_policy,
        )
        shell_service = ShellService(
            run_id=self.spec.run_id,
            workspace=workspace,
            recorder=recorder,
            job_manager=job_manager,
            shell_policy=self.spec.shell_policy,
            permission_policy=self.permission_policy,
            approval_provider=_shell_approval_provider(
                self.spec.shell_policy,
                self.shell_approval_provider,
            ),
        )
        web_service = WebService(
            web_policy=self.spec.web_policy,
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
        registry = ToolRegistry()
        registry.register_many(builtin_tools(workspace))
        for provider in self.tool_providers:
            registry.register_many(provider.get_tools(context))

        capabilities = self.spec.effective_capabilities()
        try:
            tool_policy = registry.policy_view(self.spec.tool_policy, capabilities)
        except ToolPolicyError:
            recorder.close()
            raise
        visible_tool_specs = registry.visible_specs(tool_policy)
        started = time.time()
        deadline = (
            started + self.spec.limits.max_duration_s
            if self.spec.limits.max_duration_s is not None
            else None
        )
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
        system_prompt = compose_system_prompt(
            self.spec.system_prompt_base or BASE_SYSTEM_PROMPT,
            (*self.spec.persona_segments, *static_segments),
        )
        workspace_base_path = recorder.write_workspace_base(
            workspace.workspace_base_payload(self.spec.run_id)
        )
        manifest = build_run_manifest(
            self.spec,
            tool_specs=visible_tool_specs,
            permission_policy=self.permission_policy,
            tool_policy=tool_policy.to_manifest(),
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
                "model_provider": self.spec.model.provider,
                "model": self.spec.model.model,
                "reasoning_effort": self.spec.model.reasoning.effort,
                "visible_tools": [tool.id for tool in visible_tool_specs],
            },
        )
        self._warn_on_unforwarded_multimodal(recorder, capabilities)
        return _RunResources(
            workspace=workspace,
            recorder=recorder,
            context=context,
            registry=registry,
            tool_policy=tool_policy,
            visible_tool_specs=visible_tool_specs,
            capabilities=capabilities,
            started=started,
            deadline=deadline,
            system_prompt=system_prompt,
        )

    def _warn_on_unforwarded_multimodal(
        self, recorder: AgentRecorder, capabilities: frozenset[str]
    ) -> None:
        """Multimodal input is a contract-only surface for now: non-text parts are
        accepted on the spec but not yet threaded to any provider. Emit a warning
        naming the dropped part types and why, so the degradation is observable."""
        dropped = non_text_part_types(self.spec.effective_input)
        if not dropped:
            return
        if not getattr(self.model_adapter, "supports_multimodal", False):
            reason = "adapter_lacks_multimodal"
        elif MEDIA_INPUT_CAPABILITY not in capabilities:
            reason = "capability_not_granted"
        else:
            reason = "not_yet_forwarded"
        recorder.emit(
            "model.input.degraded",
            data={"dropped_part_types": dropped, "reason": reason},
            level="warning",
        )

    def _dynamic_context_segment(self, state: RunState, res: _RunResources, step: int) -> str:
        """Join each context provider's per-turn segment. Empty when no providers
        contribute, so the turn prompt stays byte-identical to the static prompt."""
        if not self.context_providers:
            return ""
        limits = self.spec.limits
        turn = TurnContext(
            step=step,
            remaining_steps=max(0, limits.max_steps - step),
            remaining_tool_calls=max(0, limits.max_tool_calls - state.total_tool_calls),
            deadline_s=(res.deadline - time.time()) if res.deadline is not None else None,
            plan=tuple(res.context.plan),
            pending_observation_count=len(state.pending_observations),
        )
        segments = []
        for provider in self.context_providers:
            segment = provider.dynamic_segment(turn)
            if segment and segment.strip():
                segments.append(segment.strip())
        return "\n\n".join(segments)

    def _run_steps(self, state: RunState, res: _RunResources) -> None:
        context = res.context
        recorder = res.recorder
        deadline = res.deadline
        for step in range(1, self.spec.limits.max_steps + 1):
            self._check_run_boundary(deadline)
            background_observations = self._pop_background_observations(context, recorder, step)
            if background_observations:
                state.pending_observations = (*state.pending_observations, *background_observations)
            turn_id = f"turn_{step:04d}"
            turn_started = recorder.emit(
                "model.turn.started",
                turn_id=turn_id,
                data={"step": step, "previous_turn_handle": state.previous_turn_handle},
            )
            dynamic_segment = self._dynamic_context_segment(state, res, step)
            turn_system_prompt = (
                res.system_prompt
                if not dynamic_segment
                else f"{res.system_prompt}\n\n{dynamic_segment}"
            )
            request = ModelRequest(
                instruction=self.spec.instruction,
                system_prompt=turn_system_prompt,
                tools=tuple(res.visible_tool_specs),
                previous_turn_handle=state.previous_turn_handle,
                observations=state.pending_observations,
            )
            recorder.transcript(
                {
                    "kind": "model_request",
                    "step": step,
                    "previous_turn_handle": state.previous_turn_handle,
                    "observations": [obs.__dict__ for obs in state.pending_observations],
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
                    registry=res.registry,
                    tool_policy=res.tool_policy,
                    context=context,
                    recorder=recorder,
                    capabilities=res.capabilities,
                    turn_id=turn_id,
                    parent_id=turn_started.event_id,
                    step=step,
                )
                observations.append(observation)
                self._check_run_boundary(deadline)
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
        metrics = {
            "status": state.status,
            "duration_s": time.time() - res.started,
            "steps_limit": self.spec.limits.max_steps,
            "tool_calls": state.total_tool_calls,
            "changed_paths": res.workspace.changed_paths(),
            "workspace_backend": self.spec.workspace_backend,
            "requested_reasoning_effort": self.spec.model.reasoning.effort,
            "effective_reasoning_effort": self.spec.model.reasoning.effort,
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
        )

    def _pop_background_observations(
        self,
        context: AgentToolContext,
        recorder: AgentRecorder,
        step: int,
    ) -> tuple[ToolObservation, ...]:
        payloads = context.job_manager.pop_reentry_observations()
        if not payloads:
            return ()
        recorder.emit(
            "run.resumed",
            data={
                "reason": "background_job_result",
                "job_ids": [str(payload.get("job_id") or "") for payload in payloads],
                "count": len(payloads),
            },
        )
        observations: list[ToolObservation] = []
        for payload in payloads:
            job_id = str(payload.get("job_id") or "")
            observation = ToolObservation(
                call_id=f"background:{job_id}",
                tool_name="background_job",
                output=payload,
                is_background=True,
            )
            recorder.transcript(
                {
                    "kind": "tool_observation",
                    "step": step,
                    "call_id": observation.call_id,
                    "tool": observation.tool_name,
                    "output": observation.output,
                }
            )
            self._emit_background_workspace_events(payload, context, recorder)
            observations.append(observation)
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

    def _authorize_tool(self, spec: ToolSpec, decision: Any, capabilities: frozenset[str]) -> None:
        if decision.decision == "deny":
            raise PermissionDenied(
                f"tool denied by policy: {spec.id}",
                error_code="tool_policy_denied",
            )
        if decision.decision == "ask":
            raise PermissionDenied(
                f"tool requires approval: {spec.id}",
                error_code="tool_approval_required",
            )
        if spec.capability not in capabilities:
            raise PermissionDenied(
                f"capability disabled: {spec.capability}",
                error_code="capability_disabled",
            )

    def _invoke_handler(
        self,
        spec: ToolSpec,
        context: AgentToolContext,
        arguments: dict[str, Any],
        *,
        call_id: str,
        turn_id: str,
        recorder: AgentRecorder,
        started_event: AgentEvent,
    ) -> ToolResult:
        context._current_call = CallContext(
            tool_call_id=call_id,
            turn_id=turn_id,
            tool_event_id=started_event.event_id,
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
        registry: ToolRegistry,
        tool_policy: NormalizedToolPolicy,
        context: AgentToolContext,
        recorder: AgentRecorder,
        capabilities: frozenset[str],
        turn_id: str,
        parent_id: str | None,
        step: int,
    ) -> ToolObservation:
        spec: ToolSpec | None = None
        result: ToolResult
        started_event: AgentEvent | None = None
        policy_decision = ""
        policy_reason = ""
        try:
            spec = registry.resolve(call_name)
            started_event = self._emit_tool_started(
                recorder,
                call_name=call_name,
                call_id=call_id,
                spec=spec,
                arguments=arguments,
                turn_id=turn_id,
                parent_id=parent_id,
            )
            decision = tool_policy.decision_for(spec.id)
            policy_decision = decision.decision
            policy_reason = decision.reason
            self._authorize_tool(spec, decision, capabilities)
            registry.validate_args(spec, arguments)
            self._check_permissions(spec, arguments, capabilities)
            result = self._invoke_handler(
                spec,
                context,
                arguments,
                call_id=call_id,
                turn_id=turn_id,
                recorder=recorder,
                started_event=started_event,
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
                    "policy_decision": policy_decision or None,
                    "policy_reason": policy_reason or None,
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
        capabilities: frozenset[str],
    ) -> None:
        self.permission_policy.check_capability(spec.capability, capabilities)
        paths = tuple(
            str(arguments[name])
            for name in spec.path_args
            if name in arguments and arguments[name] is not None
        )
        if not paths:
            return
        operation = "read" if spec.side_effect in {"read", "artifact"} else "write"
        self.permission_policy.check_paths(operation, paths)  # type: ignore[arg-type]


def _shell_approval_provider(
    policy: ShellPolicy,
    explicit: ShellApprovalProvider | None,
) -> ShellApprovalProvider | None:
    if explicit is not None:
        return explicit
    if policy.approval_mode == "auto-approve":
        return AutoApproveShellApprovalProvider(approver_id="standalone-cli")
    if policy.approval_mode == "deny":
        return DenyShellApprovalProvider()
    return None


def _accumulate_usage(total_usage: dict[str, int], turn: ModelTurn) -> None:
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        total_usage[key] += int(turn.usage.get(key, 0))


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
