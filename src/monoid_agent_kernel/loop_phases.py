from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Literal

from monoid_agent_kernel.core.agents import (
    AgentRuntimeConfig,
    compile_bound_tool_catalog,
)
from monoid_agent_kernel.core.context import TurnContext, render_workspace_index_segment
from monoid_agent_kernel.core.manifest import build_run_manifest
from monoid_agent_kernel.core.output_validator import (
    FinalOutputView,
    OutputRetry,
    OutputValidator,
    OutputValidatorError,
    ValidationOutcome,
)
from monoid_agent_kernel.core.result import AgentArtifact, AgentRunResult, AgentTurnResult, Suspension
from monoid_agent_kernel.core.spec import ModelConfig
from monoid_agent_kernel.core.streaming import QueueEventSink
from monoid_agent_kernel.core.tool_surface import tool_surface_manifest
from monoid_agent_kernel.core.workspace import Workspace
from monoid_agent_kernel.core.workspace_index import build_workspace_index
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.public_view import (
    public_error_message,
    public_path,
    public_proposal_payload,
)
from monoid_agent_kernel.recorder import AgentRecorder
from monoid_agent_kernel.tasks import TaskManager
from monoid_agent_kernel.tool_services import JobsService, ShellService, WebService
from monoid_agent_kernel.tools.base import ToolRegistry, ToolSpec
from monoid_agent_kernel.tools.builtin import builtin_tools
from monoid_agent_kernel.workspace.local import default_local_workspace_factory


@dataclass
class _RunResources:
    """Objects assembled by bootstrap and reused across a run's phases."""

    workspace: Workspace
    recorder: AgentRecorder
    context: Any
    base_tool_specs: tuple[ToolSpec, ...]
    started: float
    deadline: float | None
    static_segments: tuple[str, ...]


@dataclass(frozen=True)
class SettleDecision:
    """The classified outcome of a settle point."""

    kind: Literal["accept", "reprompt", "exhausted", "terminal", "defect"]
    reason: str = "settled"
    status: str = "completed"
    error_code: str = ""
    ok_values: tuple[tuple[str, Any], ...] = ()
    failures: tuple[tuple[str, str], ...] = ()
    new_history_entry: dict[str, Any] | None = None
    terminal_reason: str | None = None
    defect: tuple[str, BaseException] | None = None


def _output_repair_message(failures: list[tuple[str, str]]) -> str:
    lines = [
        "Your final response did not satisfy the required output format. "
        "Correct it and respond again:"
    ]
    for validator_id, feedback in failures:
        lines.append(f"- ({validator_id}) {feedback}" if feedback else f"- ({validator_id}) invalid output")
    return "\n".join(lines)


def _run_output_validators(
    validators: tuple[OutputValidator, ...], view: FinalOutputView
) -> tuple[list[tuple[str, str]], list[tuple[str, Any]], tuple[str, BaseException] | None]:
    failures: list[tuple[str, str]] = []
    ok_values: list[tuple[str, Any]] = []
    for validator in validators:
        try:
            outcome = validator.validate(view)
            if not isinstance(outcome, ValidationOutcome):
                raise TypeError(
                    f"validate() must return a ValidationOutcome, got {type(outcome).__name__}"
                )
        except OutputRetry as exc:
            outcome = ValidationOutcome(ok=False, feedback=exc.feedback)
        except ValueError as exc:
            outcome = ValidationOutcome(ok=False, feedback=str(exc))
        except Exception as exc:
            return failures, ok_values, (validator.id, exc)
        if outcome.ok:
            ok_values.append((validator.id, outcome.value))
        else:
            failures.append((validator.id, outcome.feedback))
    return failures, ok_values, None


def _failures_by_validator(history: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for attempt in history:
        for failure in attempt.get("failures", ()):
            vid = str(failure.get("validator_id", ""))
            counts[vid] = counts.get(vid, 0) + 1
    return counts


_OUTPUT_CONTRACT_STOPPED = "Stopped: the final response did not satisfy the output contract."


class LoopBootstrapper:
    """Build the reusable resources for an AgentLoop run."""

    def __init__(self, loop: Any) -> None:
        self._loop = loop

    def bootstrap(self) -> _RunResources:
        loop = self._loop
        if loop.permission_policy == PermissionPolicy() and loop.spec.permission_policy != PermissionPolicy():
            loop.permission_policy = loop.spec.permission_policy
        workspace_factory = loop.workspace_factory or default_local_workspace_factory
        workspace = workspace_factory(loop.spec)
        loop._stream_sink = QueueEventSink()
        recorder = AgentRecorder(
            loop.spec.run_root,
            loop.spec.run_id,
            extra_event_sinks=(*loop.event_sinks, loop._stream_sink),
            status_file=loop.status_file,
            reopen=loop._restoring,
        )
        job_manager = TaskManager(
            run_id=loop.spec.run_id,
            workspace=workspace,
            recorder=recorder,
            permission_policy=loop.permission_policy,
        )
        shell_service = ShellService(
            run_id=loop.spec.run_id,
            workspace=workspace,
            recorder=recorder,
            job_manager=job_manager,
            permission_policy=loop.permission_policy,
            approval_provider=loop.shell_approval_provider,
        )
        web_service = WebService(
            recorder=recorder,
            web_gateway_client=loop.web_gateway_client,
        )
        jobs_service = JobsService(job_manager=job_manager)
        from monoid_agent_kernel.loop import AgentToolContext

        context = AgentToolContext(
            loop.spec.run_id,
            workspace,
            recorder,
            job_manager,
            shell_service,
            web_service,
            jobs_service,
            permission_policy=loop.permission_policy,
            capability_vault=loop._capability_vault,
            outbox=loop._outbox,
        )
        base_registry = ToolRegistry()
        base_registry.register_many(builtin_tools(workspace))
        for provider in loop.tool_providers:
            base_registry.register_many(provider.get_tools(context))
        if loop.subagent_definitions:
            loop._install_subagent_capability(base_registry, context, job_manager)

        started = time.time()
        deadline = (
            started + loop.spec.limits.max_duration_s
            if loop.spec.limits.max_duration_s is not None
            else None
        )
        loop._bootstrap_resources = _RunResources(
            workspace=workspace,
            recorder=recorder,
            context=context,
            base_tool_specs=tuple(base_registry.specs()),
            started=started,
            deadline=deadline,
            static_segments=(),
        )
        initial_runtime_config = loop._current_runtime_config(base_registry)
        initial_bound_catalog = compile_bound_tool_catalog(initial_runtime_config, base_registry)
        initial_turn = TurnContext(
            step=1,
            remaining_steps=max(0, loop.spec.limits.max_steps - 1),
            remaining_tool_calls=loop.spec.limits.max_tool_calls,
            deadline_s=(deadline - time.time()) if deadline is not None else None,
            plan=(),
            pending_observation_count=0,
        )
        initial_surface = loop.tool_surface_resolver.resolve(
            bound_catalog=initial_bound_catalog,
            turn=initial_turn,
        )
        initial_visible_tool_specs = list(initial_surface.immediate_tools)
        workspace_index = build_workspace_index(workspace, run_id=loop.spec.run_id)
        workspace_index_path = recorder.write_workspace_index(workspace_index)
        static_segments: list[str] = []
        if loop.inject_workspace_index:
            index_segment = render_workspace_index_segment(workspace_index)
            if index_segment:
                static_segments.append(index_segment)
        for provider in loop.context_providers:
            segment = provider.static_segment()
            if segment and segment.strip():
                static_segments.append(segment)
        if not loop._restoring:
            workspace_base_path = recorder.write_workspace_base(
                workspace.workspace_base_payload(loop.spec.run_id)
            )
            manifest = build_run_manifest(
                loop.spec,
                model_config=initial_runtime_config.model or ModelConfig(),
                tool_specs=initial_visible_tool_specs,
                permission_policy=loop.permission_policy,
                tool_surface=tool_surface_manifest(
                    resolver=loop.tool_surface_resolver,
                    tool_search=initial_runtime_config.tool_search,
                    dynamic_enabled=bool(loop._dynamic_providers()),
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
                    "workspace": str(loop.spec.workspace_root),
                    "run_dir": str(recorder.run_dir),
                    "manifest_path": "manifest.json",
                    "mode": loop.spec.mode,
                    "workspace_backend": loop.spec.workspace_backend,
                    "workspace_base_path": "workspace.base.json",
                    "model_provider": (initial_runtime_config.model or ModelConfig()).provider,
                    "model": (initial_runtime_config.model or ModelConfig()).model,
                    "reasoning_effort": (initial_runtime_config.model or ModelConfig()).reasoning.effort,
                    "visible_bindings": [tool.id for tool in initial_visible_tool_specs],
                    "agent_config_hash": initial_runtime_config.config_hash,
                },
            )
        loop._emit_bootstrap_validator_skips(initial_runtime_config, recorder)
        return _RunResources(
            workspace=workspace,
            recorder=recorder,
            context=context,
            base_tool_specs=tuple(base_registry.specs()),
            started=started,
            deadline=deadline,
            static_segments=tuple(static_segments),
        )


class LoopSettleCoordinator:
    """Classify and apply settle points for an AgentLoop."""

    def __init__(self, loop: Any) -> None:
        self._loop = loop

    async def decide(
        self,
        state: Any,
        res: _RunResources,
        context: Any,
        turn: Any,
        runtime_config: AgentRuntimeConfig,
    ) -> SettleDecision:
        loop = self._loop
        if turn.stop_reason == "refusal":
            return SettleDecision(
                kind="terminal", reason="settled", status="failed",
                error_code="output_refused", terminal_reason="refusal",
            )
        if turn.stop_reason == "length":
            return SettleDecision(
                kind="terminal", reason="limited", status="limited",
                error_code="output_truncated", terminal_reason="truncation",
            )

        validators = loop._active_output_validators(runtime_config)
        if not validators:
            return SettleDecision(kind="accept")

        view = self.build_final_output_view(state, res, context)
        failures, ok_values, defect = await asyncio.to_thread(_run_output_validators, validators, view)
        if defect is not None:
            return SettleDecision(kind="defect", defect=defect)
        if not failures:
            return SettleDecision(kind="accept", ok_values=tuple(ok_values))

        attempt = len(state.output_failure_history) + 1
        entry = {
            "attempt": attempt,
            "failures": [{"validator_id": vid, "feedback": fb} for vid, fb in failures],
        }
        if state.output_retries >= loop.spec.limits.max_output_retries:
            return SettleDecision(
                kind="exhausted", reason="limited", status="limited",
                error_code="output_validator_unsatisfied", new_history_entry=entry,
            )
        return SettleDecision(kind="reprompt", failures=tuple(failures), new_history_entry=entry)

    def apply(
        self,
        decision: SettleDecision,
        state: Any,
        res: _RunResources,
        context: Any,
        *,
        from_finish: bool,
    ) -> Suspension | None:
        recorder = res.recorder

        if decision.kind == "defect":
            validator_id, exc = decision.defect  # type: ignore[misc]
            recorder.emit(
                "output.validator.error",
                data={"validator_id": validator_id, "error": str(exc)},
                level="error",
            )
            raise OutputValidatorError(f"output validator {validator_id!r} raised: {exc}") from exc

        if decision.new_history_entry is not None:
            state.output_failure_history.append(decision.new_history_entry)
            recorder.emit("output.validation.failed", data=decision.new_history_entry, level="warning")

        if decision.kind == "accept":
            state.output_values = dict(decision.ok_values)
            state.final_output = decision.ok_values[-1][1] if decision.ok_values else None
            for validator_id, _value in decision.ok_values:
                recorder.emit("output.validator.satisfied", data={"validator_id": validator_id})
        elif decision.kind == "terminal":
            state.status = decision.status
            state.error_code = decision.error_code
            recorder.emit("output.validation.failed", data={"reason": decision.terminal_reason}, level="warning")
        elif decision.kind == "exhausted":
            state.status = decision.status
            state.final_text = state.final_text or _OUTPUT_CONTRACT_STOPPED
            state.error_code = decision.error_code
            recorder.emit(
                "output.validator.exhausted",
                data={
                    "retries": state.output_retries,
                    "failures_by_validator": _failures_by_validator(state.output_failure_history),
                    "history": list(state.output_failure_history),
                },
                level="warning",
            )
        elif decision.kind == "reprompt":
            state.output_retries += 1

        if from_finish:
            if decision.kind in ("reprompt", "exhausted"):
                self._loop._clear_finish_metadata(context)
                if decision.kind == "reprompt":
                    state.final_text = ""
            self._loop._log_finish_observations(state)

        if decision.kind == "reprompt":
            state.pending_observations = ()
            state.messages.append({"role": "user", "content": _output_repair_message(list(decision.failures))})
            return None

        if decision.kind == "accept":
            return Suspension(reason="settled", status=state.status, final_text=state.final_text)
        return Suspension(
            reason=decision.reason,
            status=state.status,
            final_text=state.final_text,
            error_code=state.error_code,
        )

    def build_final_output_view(
        self, state: Any, res: _RunResources, context: Any
    ) -> FinalOutputView:
        workspace = res.workspace

        def _read(path: str, *, max_bytes: int | None = None) -> bytes:
            data, _digest = workspace.read_bytes(path, max_bytes=max_bytes)
            return data

        artifacts = tuple(
            AgentArtifact(
                artifact_id=getattr(a, "artifact_id", ""),
                path=getattr(a, "path", ""),
                kind=getattr(a, "kind", ""),
                label=getattr(a, "label", None),
                metadata=dict(getattr(a, "metadata", {}) or {}),
            )
            for a in res.recorder.artifacts
        )
        return FinalOutputView(
            final_text=state.final_text,
            artifacts=artifacts,
            final_outputs=(context.pending_finish.outputs if context.pending_finish else ()),
            read_bytes=_read,
        )


class LoopFinalizer:
    """Build settle checkpoints and terminal run results."""

    def __init__(self, loop: Any) -> None:
        self._loop = loop

    def build_metrics(self, state: Any, res: _RunResources) -> dict[str, Any]:
        loop = self._loop
        context = res.context
        model = (
            state.previous_runtime_config.model
            if state.previous_runtime_config is not None and state.previous_runtime_config.model is not None
            else ModelConfig()
        )
        metrics = {
            "status": state.status,
            "duration_s": time.time() - res.started,
            "steps_limit": loop.spec.limits.max_steps,
            "tool_calls": state.total_tool_calls,
            "changed_paths": res.workspace.changed_paths(),
            "workspace_backend": loop.spec.workspace_backend,
            "requested_reasoning_effort": model.reasoning.effort,
            "effective_reasoning_effort": model.reasoning.effort,
            "error_code": state.error_code,
            **context.shell_service.metrics(),
            **context.jobs_service.background_metrics(),
            **context.web_service.metrics(),
            **state.total_usage,
        }
        if context.subagent_count:
            metrics["subagent_count"] = context.subagent_count
            metrics["subagent_usage"] = dict(context.subagent_usage)
        if context.skill_activation_count:
            metrics["skill_activation_count"] = context.skill_activation_count
            metrics["skills_activated"] = list(context.skills_activated)
        if state.output_failure_history:
            metrics["output_validation"] = {
                "retries": state.output_retries,
                "failures_by_validator": _failures_by_validator(state.output_failure_history),
            }
        if state.provider_error_code:
            metrics["provider_error_code"] = state.provider_error_code
        if state.provider_http_status is not None:
            metrics["provider_http_status"] = state.provider_http_status
        if state.error:
            metrics["error"] = state.error
        return metrics

    def finalize(self, state: Any, res: _RunResources) -> AgentRunResult:
        loop = self._loop
        context = res.context
        recorder = res.recorder
        workspace = res.workspace
        context.job_manager.cancel_all()
        diff_path = recorder.write_diff(workspace.diff_patch())
        proposal_payload = recorder.write_proposal_snapshot(workspace, diff_path)
        metrics = self.build_metrics(state, res)
        recorder.write_metrics(metrics)
        recorder.emit(
            "workspace.proposal.updated",
            data=public_proposal_payload(proposal_payload, loop.permission_policy),
        )
        recorder.emit(
            "proposal.ready",
            data={
                "proposal_hash": proposal_payload.get("proposal_hash"),
                "diff_sha256": proposal_payload.get("diff_sha256"),
                "changed_paths": [
                    public_path(str(path), loop.permission_policy)
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
            run_id=loop.spec.run_id,
            status=state.status,
            final_text=state.final_text,
            run_dir=run_dir,
            diff_path=diff_path,
            proposal_path=run_dir / "proposal.json",
            artifacts=artifacts,
            final_outputs=(context.pending_finish.outputs if context.pending_finish else ()),
            final_notes=(context.pending_finish.notes if context.pending_finish else None),
            final_output=state.final_output,
            outputs=dict(state.output_values),
            metrics=metrics,
            error=state.error,
            error_code=state.error_code,
            final_turn_handle=state.previous_turn_handle,
        )

    def checkpoint_on_settle(self, state: Any, res: _RunResources) -> AgentTurnResult:
        loop = self._loop
        recorder = res.recorder
        workspace = res.workspace
        diff_path = recorder.write_diff(workspace.diff_patch())
        proposal_payload = recorder.write_proposal_snapshot(workspace, diff_path)
        metrics = self.build_metrics(state, res)
        recorder.write_metrics(metrics)
        recorder.emit(
            "workspace.proposal.updated",
            data=public_proposal_payload(proposal_payload, loop.permission_policy),
        )
        public_changed = [
            public_path(str(path), loop.permission_policy)
            for path in proposal_payload.get("changed_paths", [])
        ]
        recorder.emit(
            "turn.settled",
            data={
                "status": state.status,
                "final_text": state.final_text,
                "error_code": state.error_code,
                "changed_paths": public_changed,
                "output_validators": len(loop._active_output_validators(state.previous_runtime_config)),
                "output_retries": state.output_retries,
            },
        )
        return AgentTurnResult(
            status=state.status,
            final_text=state.final_text,
            proposal_path=recorder.run_dir / "proposal.json",
            proposal_hash=str(proposal_payload.get("proposal_hash") or ""),
            changed_paths=tuple(workspace.changed_paths()),
            turn_handle=state.previous_turn_handle,
            error=state.error,
            error_code=state.error_code,
            final_output=state.final_output,
            outputs=dict(state.output_values),
            metrics=metrics,
        )
