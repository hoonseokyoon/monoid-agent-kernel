from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from native_agent_runner.core.cancellation import CancellationToken
from native_agent_runner.core.events import AgentEvent, EventSink
from native_agent_runner.core.manifest import build_run_manifest
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
    ShellApprovalDecision,
    ShellApprovalProvider,
    ShellApprovalRequest,
    execute_shell,
)
from native_agent_runner.tools.base import ToolContext, ToolProvider, ToolRegistry, ToolResult, ToolSpec
from native_agent_runner.tools.builtin import builtin_tools
from native_agent_runner.tools.policy import NormalizedToolPolicy
from native_agent_runner.web import (
    WebGatewayClient,
    WebPolicy,
    domain_from_url,
    public_query_preview,
    public_url_preview,
)
from native_agent_runner.core.workspace import Workspace
from native_agent_runner.workspace.local import default_local_workspace_factory


SYSTEM_PROMPT = """You are a local workspace agent.
Use only the provided tools to inspect or modify files. Do not invent files you have not read.
Respect tool errors and permissions. Finish by calling run.finish with a concise summary.
"""


@dataclass
class AgentToolContext(ToolContext):
    run_id: str
    workspace: Workspace
    recorder: AgentRecorder
    job_manager: BackgroundJobManager
    final_text: str = ""
    finished: bool = False
    plan: list[dict[str, Any]] = field(default_factory=list)
    permission_policy: PermissionPolicy = field(default_factory=PermissionPolicy)
    shell_policy: ShellPolicy = field(default_factory=ShellPolicy)
    shell_approval_provider: ShellApprovalProvider | None = None
    web_policy: WebPolicy = field(default_factory=WebPolicy)
    web_gateway_client: WebGatewayClient | None = None
    current_tool_call_id: str = ""
    current_turn_id: str | None = None
    current_tool_event_id: str | None = None
    shell_calls: int = 0
    failed_shell_calls: int = 0
    total_shell_duration_s: float = 0.0
    web_search_calls: int = 0
    web_fetch_calls: int = 0
    web_context_calls: int = 0
    web_failed_calls: int = 0
    web_result_count: int = 0
    web_bytes_returned: int = 0
    web_context_source_count: int = 0
    web_context_bytes_returned: int = 0

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
        del outputs, notes
        self.final_text = summary
        self.finished = True

    def execute_shell(self, args: dict[str, Any]) -> dict[str, Any]:
        command = str(args["command"])
        cwd = str(args.get("cwd") or ".")
        requested_timeout_s = args.get("timeout_s")
        requested_max_output_bytes = args.get("max_output_bytes")
        requested_startup_wait_s = args.get("startup_wait_s")
        timeout_s = self.shell_policy.effective_timeout(requested_timeout_s)
        max_output_bytes = self.shell_policy.effective_output_limit(requested_max_output_bytes)
        startup_wait_s = self.shell_policy.effective_startup_wait(requested_startup_wait_s)
        execution_workspace = self.shell_policy.effective_execution_workspace(self.workspace.backend_kind)
        background = bool(args.get("background", False))
        resume_on_exit = bool(args.get("resume_on_exit", True))
        env = args.get("env") or {}
        if not isinstance(env, dict):
            raise ToolExecutionError("shell env must be an object", error_code="tool_args_invalid")
        request = ShellApprovalRequest(
            run_id=self.run_id,
            tool_call_id=self.current_tool_call_id,
            command=command,
            cwd=cwd,
            requested_timeout_s=int(requested_timeout_s) if requested_timeout_s is not None else None,
            effective_timeout_s=timeout_s,
            requested_max_output_bytes=int(requested_max_output_bytes) if requested_max_output_bytes is not None else None,
            effective_max_output_bytes=max_output_bytes,
            execution_workspace=execution_workspace,
            requested_startup_wait_s=int(requested_startup_wait_s) if requested_startup_wait_s is not None else None,
            effective_startup_wait_s=startup_wait_s,
            background=background,
            resume_on_exit=resume_on_exit,
            env_keys=tuple(sorted(str(key) for key in env)),
        )
        approval_parent = self.current_tool_event_id
        self.recorder.emit(
            "tool.approval.requested",
            turn_id=self.current_turn_id,
            parent_id=approval_parent,
            data=request.to_public_json(),
        )
        provider = self.shell_approval_provider
        if provider is None:
            decision = ShellApprovalDecision(
                approved=False,
                reason="shell approval provider unavailable",
                approver_id="none",
            )
        else:
            decision = provider.approve_shell(request)
        approval_event_type = "tool.approval.approved" if decision.approved else "tool.approval.denied"
        self.recorder.emit(
            approval_event_type,
            turn_id=self.current_turn_id,
            parent_id=approval_parent,
            data={**request.to_public_json(), **decision.to_public_json()},
            level="info" if decision.approved else "warning",
        )
        if not decision.approved:
            raise ToolExecutionError(decision.reason or "shell approval denied", error_code="tool_approval_denied")

        shell_started = self.recorder.emit(
            "shell.exec.started",
            turn_id=self.current_turn_id,
            parent_id=approval_parent,
            data=request.to_public_json(),
        )
        if background:
            try:
                job = self.job_manager.start_shell_job(
                    command=command,
                    cwd=cwd,
                    timeout_s=timeout_s,
                    max_output_bytes=max_output_bytes,
                    startup_wait_s=startup_wait_s,
                    env=env,
                    requested_timeout_s=request.requested_timeout_s,
                    requested_max_output_bytes=request.requested_max_output_bytes,
                    requested_startup_wait_s=request.requested_startup_wait_s,
                    execution_workspace=execution_workspace,
                    resume_on_exit=resume_on_exit,
                )
            except Exception as exc:
                self.failed_shell_calls += 1
                self.recorder.emit(
                    "shell.exec.failed",
                    turn_id=self.current_turn_id,
                    parent_id=shell_started.event_id,
                    data={
                        **request.to_public_json(),
                        "error": _public_error_message(str(exc)),
                        "error_code": error_code_for_exception(exc),
                    },
                    level="warning",
                )
                raise
            self.shell_calls += 1
            content = job.started_content(self.recorder.run_dir)
            self.recorder.emit(
                "shell.exec.finished",
                turn_id=self.current_turn_id,
                parent_id=shell_started.event_id,
                data={
                    **request.to_public_json(),
                    "job_id": job.job_id,
                    "status": job.status,
                    "stdout_path": content["stdout_path"],
                    "stderr_path": content["stderr_path"],
                },
            )
            return content
        try:
            result = execute_shell(
                workspace=self.workspace,
                policy=self.shell_policy,
                permission_policy=self.permission_policy,
                command=command,
                cwd=cwd,
                timeout_s=timeout_s,
                max_output_bytes=max_output_bytes,
                env=env,
                requested_timeout_s=request.requested_timeout_s,
                requested_max_output_bytes=request.requested_max_output_bytes,
                execution_workspace=execution_workspace,
            )
        except Exception as exc:
            self.failed_shell_calls += 1
            self.recorder.emit(
                "shell.exec.failed",
                turn_id=self.current_turn_id,
                parent_id=shell_started.event_id,
                data={
                    **request.to_public_json(),
                    "error": _public_error_message(str(exc)),
                    "error_code": error_code_for_exception(exc),
                },
                level="warning",
            )
            raise
        self.shell_calls += 1
        self.total_shell_duration_s += result.duration_s
        if result.timed_out or result.output_truncated:
            self.failed_shell_calls += 1
        self.recorder.emit(
            "shell.exec.failed" if result.timed_out or result.output_truncated else "shell.exec.finished",
            turn_id=self.current_turn_id,
            parent_id=shell_started.event_id,
            data={
                **request.to_public_json(),
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "output_truncated": result.output_truncated,
                "duration_s": result.duration_s,
                "stdout_bytes": result.stdout_bytes,
                "stderr_bytes": result.stderr_bytes,
                "requested_timeout_s": result.requested_timeout_s,
                "effective_timeout_s": result.effective_timeout_s,
                "requested_max_output_bytes": result.requested_max_output_bytes,
                "effective_max_output_bytes": result.effective_max_output_bytes,
                "execution_workspace": result.execution_workspace,
                "changed_paths": [public_path(path, self.permission_policy) for path in result.changed_paths],
            },
            level="warning" if result.timed_out or result.output_truncated else "info",
        )
        return result.to_tool_content()

    def list_jobs(self) -> list[dict[str, Any]]:
        return self.job_manager.list_jobs()

    def job_status(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.job_manager.status(str(args["job_id"]))

    def job_logs(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.job_manager.logs(
            str(args["job_id"]),
            stream=str(args.get("stream") or "stdout"),  # type: ignore[arg-type]
            tail_bytes=args.get("tail_bytes"),
            offset=args.get("offset"),
        )

    def job_cancel(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.job_manager.cancel(str(args["job_id"]))

    def job_wait(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.job_manager.wait(str(args["job_id"]), timeout_s=args.get("timeout_s"))

    def background_metrics(self) -> dict[str, Any]:
        jobs = self.job_manager.list_jobs()
        terminal_jobs = [job for job in jobs if job.get("status") != "running"]
        failed_statuses = {"failed", "timed_out", "output_limited"}
        return {
            "background_jobs_started": len(jobs),
            "background_jobs_finished": sum(1 for job in terminal_jobs if job.get("status") == "exited"),
            "background_jobs_failed": sum(1 for job in terminal_jobs if job.get("status") in failed_statuses),
            "background_jobs_cancelled": sum(1 for job in terminal_jobs if job.get("status") == "cancelled"),
            "background_job_duration_s_total": sum(float(job.get("duration_s") or 0.0) for job in terminal_jobs),
            "background_job_bytes_stdout": sum(int(job.get("stdout_bytes") or 0) for job in jobs),
            "background_job_bytes_stderr": sum(int(job.get("stderr_bytes") or 0) for job in jobs),
        }

    def _check_web_enabled(
        self,
        *,
        feature_enabled: bool,
        calls: int,
        max_calls: int,
        disabled_message: str,
        limit_message: str,
        limit_code: str,
    ) -> None:
        if self.web_gateway_client is None or not self.web_policy.enabled:
            raise ToolExecutionError("web gateway is not configured", error_code="web_disabled")
        if not feature_enabled:
            raise ToolExecutionError(disabled_message, error_code="web_disabled")
        if calls >= max_calls:
            raise ToolExecutionError(limit_message, error_code=limit_code)

    def _run_web_call(
        self,
        prefix: str,
        *,
        event_data: dict[str, Any],
        call: Callable[[], dict[str, Any]],
        on_success: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        started = self.recorder.emit(
            f"{prefix}.started",
            turn_id=self.current_turn_id,
            parent_id=self.current_tool_event_id,
            data=event_data,
        )
        try:
            result = call()
        except Exception as exc:
            self.web_failed_calls += 1
            self.recorder.emit(
                f"{prefix}.failed",
                turn_id=self.current_turn_id,
                parent_id=started.event_id,
                data={**event_data, "error": _public_error_message(str(exc)), "error_code": error_code_for_exception(exc)},
                level="warning",
            )
            raise
        finished_extra = on_success(result)
        self.recorder.emit(
            f"{prefix}.finished",
            turn_id=self.current_turn_id,
            parent_id=started.event_id,
            data={**event_data, **finished_extra},
        )
        return result

    def execute_web_search(self, args: dict[str, Any]) -> dict[str, Any]:
        self._check_web_enabled(
            feature_enabled=self.web_policy.search_enabled,
            calls=self.web_search_calls,
            max_calls=self.web_policy.max_search_calls,
            disabled_message="web search is disabled",
            limit_message="web search call limit exceeded",
            limit_code="web_search_limit_exceeded",
        )
        query = str(args["query"])
        requested_max_results = args.get("max_results")
        effective_max_results = self.web_policy.effective_max_results(requested_max_results)
        event_data = {
            "query_preview": public_query_preview(query),
            "requested_max_results": requested_max_results,
            "effective_max_results": effective_max_results,
            "allowed_domains": list(args.get("allowed_domains") or ()),
            "blocked_domains": list(args.get("blocked_domains") or ()),
            "recency_days": args.get("recency_days"),
            "locale": args.get("locale"),
        }
        payload = {
            "protocol": "native-agent-runner.web-search.v1",
            "query": query,
            "max_results": effective_max_results,
            "allowed_domains": list(args.get("allowed_domains") or ()),
            "blocked_domains": list(args.get("blocked_domains") or ()),
            "recency_days": args.get("recency_days"),
            "locale": args.get("locale"),
        }

        def on_success(result: dict[str, Any]) -> dict[str, Any]:
            result_count = int(result.get("result_count") or len(result.get("results") or ()))
            self.web_search_calls += 1
            self.web_result_count += result_count
            return {"result_count": result_count}

        return self._run_web_call(
            "web.search",
            event_data=event_data,
            call=lambda: self.web_gateway_client.search(payload),
            on_success=on_success,
        )

    def execute_web_fetch(self, args: dict[str, Any]) -> dict[str, Any]:
        self._check_web_enabled(
            feature_enabled=self.web_policy.fetch_enabled,
            calls=self.web_fetch_calls,
            max_calls=self.web_policy.max_fetch_calls,
            disabled_message="web fetch is disabled",
            limit_message="web fetch call limit exceeded",
            limit_code="web_fetch_limit_exceeded",
        )
        url = str(args["url"])
        requested_timeout_s = args.get("timeout_s")
        requested_max_bytes = args.get("max_bytes")
        effective_timeout_s = self.web_policy.effective_timeout_s(requested_timeout_s)
        effective_max_bytes = self.web_policy.effective_max_response_bytes(requested_max_bytes)
        event_data = {
            "url_preview": public_url_preview(url),
            "domain": domain_from_url(url),
            "format": args.get("format") or "text",
            "requested_timeout_s": requested_timeout_s,
            "effective_timeout_s": effective_timeout_s,
            "requested_max_bytes": requested_max_bytes,
            "effective_max_bytes": effective_max_bytes,
        }
        payload = {
            "protocol": "native-agent-runner.web-fetch.v1",
            "url": url,
            "format": args.get("format") or "text",
            "timeout_s": effective_timeout_s,
            "max_bytes": effective_max_bytes,
        }

        def on_success(result: dict[str, Any]) -> dict[str, Any]:
            content_bytes = int(result.get("content_bytes") or len(str(result.get("content") or "").encode("utf-8")))
            self.web_fetch_calls += 1
            self.web_bytes_returned += content_bytes
            return {
                "final_domain": domain_from_url(str(result.get("final_url") or url)),
                "content_bytes": content_bytes,
                "truncated": bool(result.get("truncated", False)),
            }

        return self._run_web_call(
            "web.fetch",
            event_data=event_data,
            call=lambda: self.web_gateway_client.fetch(payload),
            on_success=on_success,
        )

    def execute_web_context(self, args: dict[str, Any]) -> dict[str, Any]:
        self._check_web_enabled(
            feature_enabled=self.web_policy.context_enabled,
            calls=self.web_context_calls,
            max_calls=self.web_policy.max_context_calls,
            disabled_message="web context is disabled",
            limit_message="web context call limit exceeded",
            limit_code="web_context_limit_exceeded",
        )
        query = str(args["query"])
        requested_max_tokens = args.get("max_tokens")
        requested_max_urls = args.get("max_urls")
        requested_max_snippets = args.get("max_snippets")
        effective_max_tokens = self.web_policy.effective_max_context_tokens(requested_max_tokens)
        effective_max_urls = self.web_policy.effective_max_context_urls(requested_max_urls)
        effective_max_snippets = self.web_policy.effective_max_context_snippets(requested_max_snippets)
        event_data = {
            "query_preview": public_query_preview(query),
            "requested_max_tokens": requested_max_tokens,
            "effective_max_tokens": effective_max_tokens,
            "requested_max_urls": requested_max_urls,
            "effective_max_urls": effective_max_urls,
            "requested_max_snippets": requested_max_snippets,
            "effective_max_snippets": effective_max_snippets,
            "allowed_domains": list(args.get("allowed_domains") or ()),
            "blocked_domains": list(args.get("blocked_domains") or ()),
            "recency_days": args.get("recency_days"),
            "locale": args.get("locale"),
        }
        payload = {
            "protocol": "native-agent-runner.web-context.v1",
            "query": query,
            "max_tokens": effective_max_tokens,
            "max_urls": effective_max_urls,
            "max_snippets": effective_max_snippets,
            "allowed_domains": list(args.get("allowed_domains") or ()),
            "blocked_domains": list(args.get("blocked_domains") or ()),
            "recency_days": args.get("recency_days"),
            "locale": args.get("locale"),
        }

        def on_success(result: dict[str, Any]) -> dict[str, Any]:
            source_count = int(result.get("source_count") or len(result.get("sources") or ()))
            context_bytes = int(result.get("context_bytes") or len(str(result.get("context") or "").encode("utf-8")))
            self.web_context_calls += 1
            self.web_context_source_count += source_count
            self.web_context_bytes_returned += context_bytes
            return {
                "source_count": source_count,
                "context_bytes": context_bytes,
                "estimated_tokens": result.get("estimated_tokens"),
            }

        return self._run_web_call(
            "web.context",
            event_data=event_data,
            call=lambda: self.web_gateway_client.context(payload),
            on_success=on_success,
        )


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

    def run(self) -> AgentRunResult:
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
        context = AgentToolContext(
            self.spec.run_id,
            workspace,
            recorder,
            job_manager,
            permission_policy=self.permission_policy,
            shell_policy=self.spec.shell_policy,
            shell_approval_provider=_shell_approval_provider(
                self.spec.shell_policy,
                self.shell_approval_provider,
            ),
            web_policy=self.spec.web_policy,
            web_gateway_client=self.web_gateway_client,
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
        status = "completed"
        error = ""
        error_code = ""
        provider_error_code = ""
        provider_http_status: int | None = None
        final_text = ""
        previous_turn_handle: str | None = None
        pending_observations: tuple[ToolObservation, ...] = ()
        total_tool_calls = 0
        total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        started = time.time()
        deadline = (
            started + self.spec.limits.max_duration_s
            if self.spec.limits.max_duration_s is not None
            else None
        )
        workspace_index_path = recorder.write_workspace_index(
            build_workspace_index(workspace, run_id=self.spec.run_id)
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

        try:
            for step in range(1, self.spec.limits.max_steps + 1):
                self._check_run_boundary(deadline)
                background_observations = self._pop_background_observations(context, recorder, step)
                if background_observations:
                    pending_observations = (*pending_observations, *background_observations)
                turn_id = f"turn_{step:04d}"
                turn_started = recorder.emit(
                    "model.turn.started",
                    turn_id=turn_id,
                    data={"step": step, "previous_turn_handle": previous_turn_handle},
                )
                request = ModelRequest(
                    instruction=self.spec.instruction,
                    system_prompt=SYSTEM_PROMPT,
                    tools=tuple(visible_tool_specs),
                    previous_turn_handle=previous_turn_handle,
                    observations=pending_observations,
                )
                recorder.transcript(
                    {
                        "kind": "model_request",
                        "step": step,
                        "previous_turn_handle": previous_turn_handle,
                        "observations": [obs.__dict__ for obs in pending_observations],
                    }
                )
                try:
                    turn = self.model_adapter.next_turn(request)
                except ModelAdapterError as exc:
                    provider_error_code = exc.provider_error_code
                    provider_http_status = exc.http_status
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
                _accumulate_usage(total_usage, turn)
                previous_turn_handle = turn.response_id or previous_turn_handle
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
                        "tool_calls": total_tool_calls,
                        "input_tokens": total_usage["input_tokens"],
                        "output_tokens": total_usage["output_tokens"],
                        "total_tokens": total_usage["total_tokens"],
                        "web_search_calls": context.web_search_calls,
                        "web_fetch_calls": context.web_fetch_calls,
                        "web_context_calls": context.web_context_calls,
                        "web_failed_calls": context.web_failed_calls,
                    },
                )

                if not turn.tool_calls:
                    if context.job_manager.has_resume_jobs():
                        self._wait_for_background_jobs(context, recorder, deadline)
                        pending_observations = ()
                        continue
                    if turn.final_text:
                        final_text = turn.final_text
                        break
                    raise ModelAdapterError("model returned neither final text nor tool calls")

                observations: list[ToolObservation] = []
                for call in turn.tool_calls:
                    self._check_run_boundary(deadline)
                    total_tool_calls += 1
                    if total_tool_calls > self.spec.limits.max_tool_calls:
                        status = "limited"
                        final_text = "Stopped after reaching max tool calls."
                        error_code = "max_tool_calls_exceeded"
                        break
                    observation = self._execute_tool_call(
                        call_name=call.name,
                        call_id=call.id,
                        arguments=call.arguments,
                        registry=registry,
                        tool_policy=tool_policy,
                        context=context,
                        recorder=recorder,
                        capabilities=capabilities,
                        turn_id=turn_id,
                        parent_id=turn_started.event_id,
                        step=step,
                    )
                    observations.append(observation)
                    self._check_run_boundary(deadline)
                pending_observations = tuple(observations)

                if context.finished:
                    final_text = context.final_text
                    break
                if status == "limited":
                    break
            else:
                status = "limited"
                final_text = "Stopped after reaching max steps."
                error_code = "max_steps_exceeded"
        except (RunCancelled, RunTimeout) as exc:
            status = "limited"
            error = str(exc)
            error_code = error_code_for_exception(exc)
            final_text = "Stopped because the run was cancelled." if error_code == "cancelled" else "Stopped after reaching max duration."
        except Exception as exc:  # controlled recording boundary for standalone CLI
            status = "failed"
            error = str(exc)
            error_code = error_code_for_exception(exc)
            if isinstance(exc, ModelAdapterError):
                provider_error_code = exc.provider_error_code
                provider_http_status = exc.http_status
            final_text = ""
            recorder.emit(
                "run.failed",
                data={
                    "error": _public_error_message(error),
                    "error_code": error_code,
                    "type": type(exc).__name__,
                },
                level="error",
            )
        finally:
            context.job_manager.cancel_all()
            diff_path = recorder.write_diff(workspace.diff_patch())
            proposal_payload = recorder.write_proposal_snapshot(workspace, diff_path)
            background_metrics = context.background_metrics()
            metrics = {
                "status": status,
                "duration_s": time.time() - started,
                "steps_limit": self.spec.limits.max_steps,
                "tool_calls": total_tool_calls,
                "changed_paths": workspace.changed_paths(),
                "workspace_backend": self.spec.workspace_backend,
                "requested_reasoning_effort": self.spec.model.reasoning.effort,
                "effective_reasoning_effort": self.spec.model.reasoning.effort,
                "error_code": error_code,
                "shell_calls": context.shell_calls,
                "failed_shell_calls": context.failed_shell_calls,
                "total_shell_duration_s": context.total_shell_duration_s,
                **background_metrics,
                "web_search_calls": context.web_search_calls,
                "web_fetch_calls": context.web_fetch_calls,
                "web_context_calls": context.web_context_calls,
                "web_failed_calls": context.web_failed_calls,
                "web_result_count": context.web_result_count,
                "web_bytes_returned": context.web_bytes_returned,
                "web_context_source_count": context.web_context_source_count,
                "web_context_bytes_returned": context.web_context_bytes_returned,
                **total_usage,
            }
            if provider_error_code:
                metrics["provider_error_code"] = provider_error_code
            if provider_http_status is not None:
                metrics["provider_http_status"] = provider_http_status
            if error:
                metrics["error"] = error
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
                    "status": status,
                    "error": _public_error_message(error),
                    "error_code": error_code,
                    "final_text": final_text,
                    "duration_s": metrics["duration_s"],
                    "diff_path": str(diff_path.relative_to(recorder.run_dir)),
                    "proposal_path": "proposal.json",
                    "metrics_path": "metrics.json",
                },
                level="error" if status == "failed" else "info",
            )
            artifacts = tuple(recorder.artifacts)
            run_dir = recorder.run_dir
            recorder.close()

        return AgentRunResult(
            run_id=self.spec.run_id,
            status=status,  # type: ignore[arg-type]
            final_text=final_text,
            run_dir=run_dir,
            diff_path=diff_path,
            proposal_path=run_dir / "proposal.json",
            artifacts=artifacts,
            metrics=metrics,
            error=error,
            error_code=error_code,
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
        if spec.id == "shell.exec" and spec.capability not in capabilities:
            raise PermissionDenied("shell is disabled", error_code="shell_disabled")
        if spec.id.startswith("web.") and spec.capability not in capabilities:
            raise PermissionDenied("web is disabled", error_code="web_disabled")

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
        context.current_tool_call_id = call_id
        context.current_turn_id = turn_id
        context.current_tool_event_id = started_event.event_id
        try:
            result = spec.handler(context, arguments)
        finally:
            context.current_tool_call_id = ""
            context.current_turn_id = None
            context.current_tool_event_id = None
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
                "error": _public_error_message(result.error),
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
            result = ToolResult(ok=False, error=str(exc), error_code=error_code_for_exception(exc))
        except PermissionDenied as exc:
            result = ToolResult(ok=False, error=str(exc), error_code=error_code_for_exception(exc))
            recorder.emit(
                "permission.denied",
                turn_id=turn_id,
                parent_id=started_event.event_id if started_event else parent_id,
                data={
                    "call_id": call_id,
                    "tool": spec.id if spec is not None else call_name,
                    "requested_tool": call_name,
                    "error": _public_error_message(str(exc)),
                    "error_code": result.error_code,
                    "policy_decision": policy_decision or None,
                    "policy_reason": policy_reason or None,
                },
                level="warning",
            )
        except (NativeAgentError, ValueError, TypeError) as exc:
            result = ToolResult(
                ok=False,
                error=str(exc),
                error_code=error_code_for_exception(exc)
                if isinstance(exc, NativeAgentError)
                else "tool_handler_error",
            )
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
        if spec.id == "shell.exec":
            if result.content.get("job_id") and result.content.get("status") == "running":
                return
            changed_paths = [
                public_path(str(path), self.permission_policy)
                for path in result.content.get("changed_paths", [])
            ]
            recorder.emit(
                "workspace.file.changed",
                turn_id=turn_id,
                parent_id=parent_id,
                data={
                    "tool": spec.id,
                    "paths": changed_paths,
                    "result": {
                        "exit_code": result.content.get("exit_code"),
                        "duration_s": result.content.get("duration_s"),
                        "stdout_bytes": result.content.get("stdout_bytes"),
                        "stderr_bytes": result.content.get("stderr_bytes"),
                    },
                    "mode": context.workspace.mode,
                },
            )
            self._emit_workspace_proposal(context, recorder, turn_id=turn_id, parent_id=parent_id)
        elif spec.side_effect == "read" and spec.path_args:
            recorder.emit(
                "workspace.file.read",
                turn_id=turn_id,
                parent_id=parent_id,
                data={"tool": spec.id, "paths": _public_paths_from_args(spec, arguments, self.permission_policy)},
            )
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


def _tool_start_data(
    call_name: str,
    call_id: str,
    spec: ToolSpec | None,
    arguments: dict[str, Any],
    permission_policy: PermissionPolicy,
) -> dict[str, Any]:
    preview = (
        shell_args_preview(arguments, permission_policy)
        if spec is not None and spec.id == "shell.exec"
        else web_args_preview(arguments, permission_policy)
        if spec is not None and spec.id.startswith("web.")
        else args_preview(arguments, permission_policy)
    )
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


def _public_error_message(error: str) -> str:
    if not error:
        return ""
    if "PRIVATE KEY" in error.upper():
        return "[redacted-sensitive-error]"
    return error
