from __future__ import annotations

import asyncio
import base64
import fnmatch
import inspect
import json
import threading
import time
from collections.abc import Callable, Iterable, Mapping
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
from native_agent_runner.core.media import (
    MAX_FORWARDABLE_BLOCKS,
    WIRE_FORWARDABLE_PART_TYPES,
    WorkspaceMediaResolver,
    blob_shas_in_messages,
    count_tool_result_images,
    estimate_image_tokens,
    evict_tool_result_images,
    image_dimensions,
    native_image_token_cap,
    normalize_inline_media_dicts,
    normalize_inline_media_part,
    resolve_wire_messages,
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
    PromptSpec,
    RuntimeConfigSource,
    SubagentDefinition,
    ToolBinding,
    ToolSearchConfig,
    coerce_runtime_config_provider,
    collect_runtime_config_issues,
    compile_bound_tool_catalog,
    generated_tool_bindings,
    runtime_config_diff,
    transcript_config_snapshot,
    validate_runtime_config,
)
from native_agent_runner.core.manifest import build_run_manifest
from native_agent_runner.core.prompt import BASE_SYSTEM_PROMPT, compose_system_prompt
from native_agent_runner.core.result import AgentArtifact, AgentRunResult, AgentTurnResult, Suspension
from native_agent_runner.core.output_validator import (
    FinalOutputView,
    OutputRetry,
    OutputValidator,
    OutputValidatorError,
    ValidationOutcome,
)
from native_agent_runner.core.streaming import QueueEventSink, RunStream
from native_agent_runner.core.spec import (
    AgentRunSpec,
    ModelConfig,
    RunLimits,
    input_to_parts,
    text_from_parts,
    user_message_from_parts,
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
    TurnInterrupted,
    TurnPaused,
    error_code_for_exception,
)
from native_agent_runner.core.capability import (
    CapabilityBroker,
    CapabilityDenial,
    CapabilityLease,
    CapabilityPending,
    CapabilityRequest,
    CapabilityVault,
)
from native_agent_runner.core.outbox import Outbox, OutboxReceipt, OutboxRequest
from native_agent_runner.core.trace_context import new_traceparent
from native_agent_runner.tasks import (
    HostedResultInjector,
    HostedTask,
    SubagentTaskExecutor,
    TaskManager,
)
from native_agent_runner.permissions import PermissionPolicy, matches_path_patterns
from native_agent_runner.providers.base import (
    ModelAdapter,
    ModelRequest,
    ModelStreamChunk,
    ModelTurn,
    ReasoningDelta,
    TextDelta,
    ToolObservation,
    assemble_streamed_turn,
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
from native_agent_runner.tool_loader import FunctionToolProvider
from native_agent_runner.tools.builtin import agent_spawn_tool, builtin_tools
from native_agent_runner.web import WebGatewayClient, domain_allowed, domain_from_url
from native_agent_runner.core.workspace import Workspace
from native_agent_runner.workspace.local import default_local_workspace_factory


def _binding_matches(binding: ToolBinding, patterns: tuple[str, ...]) -> bool:
    """True if a tool binding matches any fnmatch pattern. Matched against the binding's
    tool id, binding id, and model name, so subagent allow/deny lists accept ids
    (``fs.read``), patterns (``mcp.*``, ``mcp.github.*``), or ``*`` for all."""
    candidates = (binding.ref.tool_id, binding.binding_id, binding.model_name or "")
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns for name in candidates if name)


def _recoverable_turn_error(exc: BaseException) -> bool:
    """Whether a model-turn exception is *recoverable* — the session should survive and the
    turn can be re-attempted (after backoff, or after the user fixes config) rather than
    terminalizing the whole run.

    Recoverable = a ``ModelAdapterError`` that is gateway-flagged ``retryable`` (transient:
    timeouts, network, 429, exhausted 5xx) OR any 4xx (config/auth/rate-limit the user can fix
    and resend against). Everything else — a generic exception, or an un-flagged 5xx — stays
    terminal, matching today's behavior.
    """
    if not isinstance(exc, ModelAdapterError):
        return False
    if exc.retryable:
        return True
    status = exc.http_status
    return status is not None and 400 <= status < 500


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


def _output_repair_message(failures: list[tuple[str, str]]) -> str:
    """The user-role message re-prompted to the model after a failed output validation. Combines
    every failing validator's feedback so the model can fix them all in one re-prompt."""
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
    """Run the active validators against the view. PURE — no recorder, no state mutation — so it
    is safe to offload to a thread (validators may block on file reads / heavy regex). Returns
    ``(failures, ok_values, defect)``: ``failures`` are ``(validator_id, feedback)`` for rejections
    (``OutputRetry`` / ``ValueError``, incl. ``JSONDecodeError`` / pydantic ``ValidationError``);
    ``ok_values`` are ``(validator_id, value)`` for passes; ``defect`` is ``(validator_id, exc)`` for
    the first validator that raised an unexpected exception (a bug the model cannot fix), else None.
    """
    failures: list[tuple[str, str]] = []
    ok_values: list[tuple[str, Any]] = []
    for validator in validators:
        try:
            outcome = validator.validate(view)
            # Guard the return shape INSIDE the try so a validator that returns None / a malformed
            # object (no ``ok``/``feedback``) is classified as a defect with the validator id, not
            # an uncaught AttributeError downstream that surfaces as a generic internal error.
            if not isinstance(outcome, ValidationOutcome):
                raise TypeError(
                    f"validate() must return a ValidationOutcome, got {type(outcome).__name__}"
                )
        except OutputRetry as exc:
            outcome = ValidationOutcome(ok=False, feedback=exc.feedback)
        except ValueError as exc:
            outcome = ValidationOutcome(ok=False, feedback=str(exc))
        except Exception as exc:  # validator defect — stop and report it to the caller (loop thread)
            return failures, ok_values, (validator.id, exc)
        if outcome.ok:
            ok_values.append((validator.id, outcome.value))
        else:
            failures.append((validator.id, outcome.feedback))
    return failures, ok_values, None


def _failures_by_validator(history: list[dict[str, Any]]) -> dict[str, int]:
    """Roll up how many attempts each validator failed across the history. An oscillating
    contradiction (validators that cannot be jointly satisfied) shows as several ids with equal,
    non-trivial counts — the signal a developer needs when a run exhausts its retries."""
    counts: dict[str, int] = {}
    for attempt in history:
        for failure in attempt.get("failures", ()):
            vid = str(failure.get("validator_id", ""))
            counts[vid] = counts.get(vid, 0) + 1
    return counts


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
    # Per-run capability leases (handles only). A tool handler reads ``capability_token`` to get
    # the access handle the gate acquired for its declared capability. None when no broker is set.
    capability_vault: CapabilityVault | None = None
    # Per-run outbox of staged external sends (handles only). A tool handler calls ``emit_outbox``
    # to durably stage a side-effect the edge drains later. None when outbox staging is unavailable.
    outbox: Outbox | None = None
    tool_search_entries: tuple[ToolSearchEntry, ...] = ()
    tool_search_max_results: int = 5
    # Depth of this run in the subagent tree (0 = top-level). Threaded into spawned
    # children as ``depth`` so the executor can enforce the nesting cap.
    subagent_depth: int = 0
    # Report-only roll-up of delegated work: how many subagents this run spawned and their
    # combined token usage. Kept SEPARATE from total_usage on purpose — total_usage also
    # tracks this run's remaining context budget, which a child's isolated tokens must not
    # inflate. Surfaced in the run metrics for cost visibility.
    subagent_count: int = 0
    subagent_usage: dict[str, int] = field(default_factory=dict)
    # Report-only roll-up of skill activations (a skill's L2 instructions being loaded via
    # the ``skill`` tool). Surfaced in run metrics for usage visibility. Skills attach via
    # the ContextProvider/ToolProvider seams, so this is the only run-state they touch.
    skill_activation_count: int = 0
    skills_activated: list[str] = field(default_factory=list)
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

    def run_script(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run a pre-resolved ``argv`` (skill.run_script) through the shell machinery —
        approval, env scrubbing, timeout, output limits, events — but WITHOUT a shell, so
        the bundled script's own args are never re-parsed by bash/powershell. ``args``
        carries ``argv`` (the real command) plus a ``command`` label for the preview."""
        argv = [str(part) for part in args.get("argv") or ()]
        rest = {key: value for key, value in args.items() if key != "argv"}
        return self.shell_service.execute(rest, self._current_call, argv_override=argv)

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

    def spawn_subagent(self, args: dict[str, Any]) -> dict[str, Any]:
        """Delegate to a child run via the ``subagent`` task kind. Foreground spawns
        block here on ``TaskManager.wait`` and return the child's final message;
        background spawns return ``started`` content and the result is injected later
        through the reentry queue (see ``SubagentTaskExecutor``)."""
        background = bool(args.get("background", False))
        call = self._current_call
        task = self.job_manager.start_task(
            "subagent",
            {
                "definition_id": str(args.get("subagent_type") or ""),
                "prompt": str(args.get("prompt") or ""),
                "depth": self.subagent_depth,
                "background": background,
                "resume_on_exit": background,
                "created_by": "model",
                # Correlation so subagent.* events nest under this spawn tool call.
                "parent_event_id": call.tool_event_id,
                "turn_id": call.turn_id,
            },
        )
        if background:
            content = task.started_content(self.recorder.run_dir)
            return {"spawned": True, "background": True, **content}
        return self.job_manager.wait(task.job_id)

    def record_skill_activation(self, name: str, *, resource_count: int = 0) -> None:
        """Observability hook called (best-effort) by the ``skill`` tool when a skill's
        instructions are loaded. Emits a ``skill.activated`` event correlated to the skill
        tool call (so an OTel sink can enrich that tool span) and bumps the run-metrics
        counter. The skill tool duck-types this method, so skills stay decoupled from the
        core contract; this is the only place run-state learns about skills."""
        call = self._current_call
        self.skill_activation_count += 1
        self.skills_activated.append(name)
        self.recorder.emit(
            "skill.activated",
            turn_id=call.turn_id,
            parent_id=call.tool_event_id,
            data={"name": name, "resource_count": int(resource_count)},
        )

    def execute_web_search(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.web_service.search(args, self._current_call, capability_token=self.capability_token("web.search"))

    def execute_web_fetch(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.web_service.fetch(args, self._current_call, capability_token=self.capability_token("web.fetch"))

    def execute_web_context(self, args: dict[str, Any]) -> dict[str, Any]:
        return self.web_service.context(args, self._current_call, capability_token=self.capability_token("web.context"))

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

    def capability_token(self, capability: str) -> str | None:
        if self.capability_vault is None:
            return None
        return self.capability_vault.token_for(capability, now=time.time())

    def emit_outbox(
        self,
        destination: str,
        payload: dict[str, Any],
        *,
        capability: str = "",
        idempotency_key: str = "",
        expect_ack: bool = False,
        reply_to: str = "",
    ) -> dict[str, Any]:
        """Stage a durable outbound side-effect. Captures the capability lease handle (never the
        secret) so the edge sender can authenticate, appends the request to the run's outbox (which
        is checkpointed), and emits ``outbox.requested``. The IO happens later, at the edge. With
        ``expect_ack`` the edge delivers the send's receipt back as an inbox message (non-park)."""
        if self.outbox is None:
            raise ToolExecutionError("outbox is not available", error_code="outbox_unavailable")
        call = self._current_call
        request = OutboxRequest(
            destination=destination,
            payload=dict(payload),
            capability=capability,
            token_ref=self.capability_token(capability) or "" if capability else "",
            run_id=self.run_id,
            idempotency_key=idempotency_key,
            expect_ack=expect_ack,
            reply_to=reply_to,
            # A fresh root trace at staging (pure, no IO): the request carries an id from birth, the
            # edge derives a child span for the actual send. Observability only — never gates anything.
            traceparent=new_traceparent(),
        )
        self.outbox.append(request)
        self.recorder.emit(
            "outbox.requested",
            turn_id=call.turn_id,
            parent_id=call.tool_event_id,
            data={
                "request_id": request.id,
                "destination": destination,
                "capability": capability,
                "traceparent": request.traceparent,
            },
        )
        return {"status": "staged", "request_id": request.id}


def _observation_message(observation: ToolObservation, media_store: dict[str, bytes]) -> dict[str, Any]:
    """Provider-neutral by-value message for a tool/async observation. Preserves the
    ``is_background`` → role semantics the adapters use: a background/hosted result is a
    new user message; a tool result is a ``tool`` message keyed by ``call_id``."""
    if observation.is_background:
        return {"role": "user", "content": format_async_result_text(observation.output)}
    message: dict[str, Any] = {
        "role": "tool",
        "call_id": observation.call_id,
        "content": observation.output,
    }
    if observation.media:
        # By reference; resolved to wire blocks at send time and delivered per provider (a follow-up
        # user message for OpenAI/gateway). Inline (data:) media a tool returned is normalized to a
        # durable blob here, symmetric with user-input media — so tool media survives restart too.
        message["media"] = normalize_inline_media_dicts(list(observation.media), media_store)
    return message


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
    # Validated value from a successful output validator (process-local; surfaced as
    # AgentRunResult.final_output, never checkpointed). Only set on a successful settle.
    final_output: Any = None
    # Validated values keyed by validator id from the last successful settle (process-local;
    # surfaced as AgentRunResult.outputs, never checkpointed). final_output is the last of these.
    output_values: dict[str, Any] = field(default_factory=dict)
    # Per-attempt rejection history for this turn-sequence (transient diagnostics, NOT checkpointed):
    # each entry {attempt, failures:[{validator_id, feedback}]}. Rolled up into output.validator.exhausted
    # + run metrics so a jointly-unsatisfiable validator set is diagnosable rather than a silent burn.
    output_failure_history: list[dict[str, Any]] = field(default_factory=list)
    # How many times an output validator has rejected the final response this turn-sequence and
    # forced a re-prompt. Checkpointed (a mid-repair restart must not re-grant the budget).
    output_retries: int = 0
    previous_turn_handle: str | None = None
    pending_user_input: tuple[ContentPart, ...] | None = None
    pending_observations: tuple[ToolObservation, ...] = ()
    pending_binding_loads: tuple[str, ...] = ()
    # Gated tool calls whose capability was escalated and is now (or will be) granted; the loop
    # auto-redispatches them at the next step boundary instead of relying on a model retry (⑤).
    # Each entry: {call_name, call_id, arguments, binding_id, task_id, capability}.
    pending_capability_replays: tuple[dict[str, Any], ...] = ()
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
    # Content-addressed bytes for inline-ingested media (``blob:<sha>`` refs in ``messages``).
    # In-memory working state, NOT serialized into the manifest — it travels as checkpoint blobs
    # (``collect_checkpoint_blobs``) and is rehydrated on restore, so an inline image survives a
    # restart and a base re-provisioning.
    media_blobs: dict[str, bytes] = field(default_factory=dict)


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
    # Opt-in token streaming for the autonomous (non-RunStream) drive: when set and the model
    # adapter supports ``astream_turn``, each text fragment is emitted as a ``model.output.delta``
    # event so an event-stream consumer (e.g. the studio app over SSE) can render tokens live.
    # Falls back to a one-shot ``next_turn`` for adapters that can't stream. Off by default.
    emit_output_deltas: bool = False
    permission_policy: PermissionPolicy = field(default_factory=PermissionPolicy)
    cancellation_token: CancellationToken | None = None
    shell_approval_provider: ShellApprovalProvider | None = None
    web_gateway_client: WebGatewayClient | None = None
    workspace_factory: Callable[[AgentRunSpec], Workspace] | None = None
    context_providers: tuple[ContextProvider, ...] = ()
    # Output validators (post-response conformance). Registered here (code) and run by default;
    # a run may disable one via an OutputValidatorBinding(enabled=False) in its runtime config. On a
    # failed validation the loop re-prompts with the validator's feedback, bounded by
    # RunLimits.max_output_retries, settling ``limited`` (output_validator_unsatisfied) on exhaustion.
    output_validators: tuple[OutputValidator, ...] = ()
    inject_workspace_index: bool = False
    # Agent-as-tool delegation: a map of subagent id -> SubagentDefinition. When non-empty
    # the bootstrap registers the ``agent.spawn`` tool and the ``subagent`` task executor; a
    # runtime config still needs an explicit binding to ``agent.spawn`` to expose the tool.
    # A child inherits the parent's tools/model/mode/limits by default (the definition can
    # narrow them); inherited by spawned children so they can delegate further (bounded by
    # RunLimits.max_subagent_depth).
    subagent_definitions: Mapping[str, SubagentDefinition] = field(default_factory=dict)
    # How checkpoints are durably stored (core defines WHAT, the store defines HOW).
    # Defaults to a local-fs store under the run root; a backend injects a durable one.
    checkpoint_store: CheckpointStore | None = None
    # Optional capability broker: when set, a bound tool that declares ``runtime.requires_lease``
    # must hold a valid lease (granted by the broker, scoped to the binding) before it runs.
    # Secrets stay in the broker; the core only gates on the lease. None = capability gating off.
    capability_broker: CapabilityBroker | None = None
    # When True (default), after a gated tool's capability is granted the loop auto-executes the
    # gated call (no model retry); see ⑤ auto-redispatch. When False, the model must retry the tool
    # (the lease is still admitted). Either way model-retry remains the fallback if replay can't run.
    capability_auto_redispatch: bool = True
    # Rotation: when > 0, a cached lease within this many seconds of expiry is proactively re-brokered
    # on use (the handle/expiry refresh under a stable contract), bounded by the lease's
    # ``max_expires_at`` ceiling. 0 (default) disables rotation — leases simply expire and re-broker.
    capability_rotate_skew_seconds: float = 0.0
    _bootstrap_resources: _RunResources | None = field(default=None, init=False, repr=False)
    _session: _Session | None = field(default=None, init=False, repr=False)
    _restoring: bool = field(default=False, init=False, repr=False)
    # Core-owned per-run event loop for sync callers. Runs continuously on a dedicated
    # daemon thread for the whole run (not just during a call), so background asyncio tasks
    # (subprocess monitors) keep progressing between turns even when a turn-by-turn driver
    # like the backend is parked between calls. The sync facade marshals coroutines onto it
    # via run_coroutine_threadsafe (see _run_sync). None until first sync use, or when an
    # async caller drives the run on its own loop. Torn down by _maybe_close_loop.
    _owned_loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    _owned_loop_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    # Dormant sink installed on the run's EventBus at bootstrap; astream activates it to tap
    # orchestration events and relay token deltas onto a stream queue. None until bootstrap.
    _stream_sink: QueueEventSink | None = field(default=None, init=False, repr=False)
    # Turn-level "stop": set from another thread via :meth:`interrupt_turn`, consumed at the
    # next step boundary (see ``_check_run_boundary``). Distinct from ``cancellation_token``
    # (which is run-level/terminal); an interrupt keeps the session alive. Cleared at the start
    # of each new user submit so a stale stop never kills the next turn.
    _interrupt_requested: bool = field(default=False, init=False, repr=False)
    # Cooperative "pause": set via :meth:`pause_turn`, consumed ONLY at the start-of-step
    # boundary (top of the pump loop) — never mid-step — so ``pending_observations`` are
    # always in a clean, resumable shape. Unlike an interrupt, a pause freezes the turn and a
    # later ``run_until_suspended(None)`` re-pump continues it. Bare one-way flag (CPython-atomic,
    # mirroring ``_interrupt_requested``). Cleared at the start of each new user submit.
    _pause_requested: bool = field(default=False, init=False, repr=False)
    # Per-run cache of granted capability leases (handles only, never secrets). Deliberately not
    # checkpointed — on restore leases are re-brokered, so a stale handle never survives on disk.
    _capability_vault: CapabilityVault = field(default_factory=CapabilityVault, init=False, repr=False)
    _outbox: Outbox = field(default_factory=Outbox, init=False, repr=False)

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

    @classmethod
    def from_tools(
        cls,
        spec: AgentRunSpec,
        model_adapter: ModelAdapter,
        tools: Iterable[ToolSpec],
        *,
        definition_id: str = "custom-agent",
        model: ModelConfig | None = None,
        prompt: PromptSpec | None = None,
        **kwargs: Any,
    ) -> AgentLoop:
        """One call to run with custom tools — no hand-wrapped provider or bindings.

        ``tools`` are ``@tool``-decorated functions or raw :class:`ToolSpec` objects. They are
        registered for the run AND exposed to the model via auto-generated :class:`ToolBinding`
        entries (binding_id/model_name derived from each tool's id). Optional seams
        (``event_sinks``, ``checkpoint_store``, extra ``tool_providers``, …) pass through::

            @tool(id="skill.word_count", side_effect="run")
            def word_count(text: str) -> dict: ...

            AgentLoop.from_tools(spec, adapter, [word_count]).run_once("count the words")
        """
        specs = tuple(tools)
        provider = FunctionToolProvider(lambda _ctx: specs)
        config = AgentRuntimeConfig(
            definition_id=definition_id,
            model=model,
            prompt=prompt or PromptSpec(),
            tools=generated_tool_bindings(specs),
        )
        existing = tuple(kwargs.pop("tool_providers", ()))
        return cls.from_config(
            spec, model_adapter, config, tool_providers=(provider, *existing), **kwargs
        )

    @staticmethod
    def validate(
        config: AgentRuntimeConfig,
        *,
        tools: Iterable[ToolSpec] = (),
        registry: ToolRegistry | None = None,
        output_validators: Iterable[OutputValidator] = (),
    ) -> list[str]:
        """Check a runtime config before a run and return all problems as readable messages
        (``[]`` == valid). Unlike the internal raising validator, this collects every issue —
        unknown tool ids, duplicate binding_ids/model_names, invalid runtime — in one pass, so a
        backend can surface them together instead of failing at bootstrap.

        Validates against the builtin tools plus any ``tools`` you'll bind (or an explicit
        ``registry``). The run ``spec`` is not needed — tool validation doesn't depend on it."""
        issues: list[str] = []
        if registry is None:
            registry = ToolRegistry()
            registry.register_many(builtin_tools(None))  # type: ignore[arg-type]
            # agent.spawn is registered only when a run supplies subagent_definitions; include it
            # here so a valid delegation config (e.g. Studio's `delegate` capability) isn't
            # false-rejected as an unknown tool.
            registry.register(agent_spawn_tool())
            # Register the caller's tools one-by-one so a bad spec (id/exported-name collision)
            # is collected rather than raised — keeping the list-returning preflight contract.
            for spec in tools:
                try:
                    registry.register(spec)
                except ValueError as exc:
                    issues.append(str(exc))
        # Output-validator bindings are opt-outs (default-on). A binding whose ``validator_id``
        # matches no registered validator is a no-op (commonly a typo) — flag it so it is not
        # silently ignored. Pass ``output_validators`` (the AgentLoop's registry) to enable this.
        registered_validator_ids = {validator.id for validator in output_validators}
        for binding in config.output_validators:
            if binding.validator_id not in registered_validator_ids:
                issues.append(
                    f"output validator binding references unknown validator_id "
                    f"{binding.validator_id!r}; no registered validator has that id (no-op)"
                )
        return issues + collect_runtime_config_issues(config, registry)

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
        a hosted-task result) and resumes, returning only once the turn settles.

        Sync facade over :meth:`asubmit`; from an async context call ``asubmit``."""
        return self._run_sync(self.asubmit(user_input))

    async def asubmit(self, user_input: str | tuple[ContentPart, ...]) -> AgentTurnResult:
        """Async form of :meth:`submit`. Awaits the model natively (or offloads a sync
        adapter to a thread) and parks on tasks without blocking the event loop."""
        session = self._require_open()
        suspension = await self.arun_until_suspended(user_input)
        while suspension.reason == "awaiting_tasks":
            await asyncio.to_thread(
                self._wait_for_background_jobs,
                session.res.context,
                session.res.recorder,
                session.res.deadline,
            )
            suspension = await self.arun_until_suspended(None)
        assert suspension.turn is not None  # non-awaiting reasons always checkpoint
        return suspension.turn

    def astream(self, user_input: str | tuple[ContentPart, ...]) -> RunStream:
        """Stream one user turn live: the async-CM analog of :meth:`asubmit`.

        Requires an open run (call :meth:`aopen`) and must be driven on the caller's running
        event loop. Yields ``AgentEvent`` (orchestration) interleaved with ``ModelStreamChunk``
        (token deltas, when the adapter exposes ``astream_turn``); read ``stream.result`` after
        the stream drains. Auto-waits in-process background jobs like ``asubmit`` and ends the
        stream when the run parks on an external hosted task (surfaced as ``stream.suspension``,
        alongside a ``run.awaiting_input`` event)::

            await loop.aopen()
            async with loop.astream("go") as stream:
                async for item in stream:
                    ...
            result = stream.result
        """
        self._require_open()
        sink = self._stream_sink
        if sink is None:  # pragma: no cover — _require_open guarantees a bootstrapped sink
            raise NativeAgentError("run is not open; call aopen() first", error_code="run_not_open")
        if self.cancellation_token is None:
            # Cooperative cancel (on early break) needs a token the boundary checks observe.
            self.cancellation_token = CancellationToken()
        token = self.cancellation_token
        return RunStream(
            sink=sink,
            drive_factory=lambda: self._astream_drive(user_input),
            request_cancel=token.cancel,
        )

    async def _astream_drive(
        self, user_input: str | tuple[ContentPart, ...]
    ) -> AgentTurnResult | Suspension:
        """``asubmit``'s body, but yields (instead of blocking) when the run parks on an
        external hosted task — the caller resumes via a fresh stream after reporting it."""
        session = self._require_open()
        suspension = await self.arun_until_suspended(user_input)
        while suspension.reason == "awaiting_tasks":
            if suspension.has_external:
                return suspension
            await asyncio.to_thread(
                self._wait_for_background_jobs,
                session.res.context,
                session.res.recorder,
                session.res.deadline,
            )
            suspension = await self.arun_until_suspended(None)
        assert suspension.turn is not None  # non-awaiting reasons always checkpoint
        return suspension.turn

    def run_until_suspended(
        self, user_input: str | tuple[ContentPart, ...] | None = None
    ) -> Suspension:
        """Non-blocking pump. With ``user_input`` it starts a new user turn; with
        ``None`` it resumes a run parked on a task (whose result was already injected
        via report_task_result). Returns why the run suspended without blocking on
        tasks — the caller decides how to wait. Every non-``awaiting_tasks`` reason
        runs a settle checkpoint and attaches the ``AgentTurnResult`` as ``turn``.

        Sync facade over :meth:`arun_until_suspended`."""
        return self._run_sync(self.arun_until_suspended(user_input))

    def interrupt_turn(self) -> None:
        """Request a turn-level stop: the running turn halts at its next step boundary and
        suspends with ``reason="interrupted"`` (the session stays alive — the next message
        continues the conversation). Thread-safe one-way signal (a bare flag set, mirroring
        ``cancellation_token.cancel()``). A no-op if no turn is in flight: the flag is cleared
        when the next submit starts, so it never kills a turn the user did not mean to stop.
        With token streaming (``emit_output_deltas`` + an ``astream_turn`` adapter) it takes
        effect mid-generation — the in-flight stream is aborted at the next token. Otherwise it
        lands at the next step boundary (a non-streamed model call finishes first)."""
        self._interrupt_requested = True

    def pause_turn(self) -> None:
        """Request a cooperative pause: the running turn freezes at the start of its next step
        and suspends with ``reason="paused"`` (the session stays alive; resume by re-pumping
        via ``run_until_suspended(None)``). Thread-safe one-way signal (a bare flag set, like
        ``interrupt_turn``). Unlike an interrupt, a pause keeps the turn's in-flight
        ``pending_observations`` so the resumed turn continues exactly where it left off, and it
        lands ONLY at a start-of-step boundary — never mid-step and never mid-generation (an
        in-flight model call always completes first). The flag is cleared when the next user
        submit starts, so a stale pause never freezes a fresh turn."""
        self._pause_requested = True

    def revoke_capability(
        self,
        *,
        capability: str | None = None,
        lease_id: str | None = None,
        before: float | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """Revoke a capability lease NOW (the operator/Daemon kill switch). Records the revocation
        in the per-run vault; the gate (``_ensure_capability_lease``) and ``token_for`` then refuse
        the handle fail-closed — a per-capability revoke is also refused re-brokering, so it cannot
        be undone by a permissive broker. Thread-safe (set mutation only, like ``pause_turn`` /
        ``interrupt_turn``); the ``capability.denied`` audit event is emitted on the loop thread at
        the gate when the next gated call hits the revocation, so this is safe to call from a
        control-plane thread. Pass ``capability="*"`` to revoke every currently-held capability.
        Returns a summary of what was revoked."""
        return self._capability_vault.revoke(
            capability=capability, lease_id=lease_id, before=before
        )

    def pending_outbox(self) -> list[OutboxRequest]:
        """Staged outbox requests awaiting (re)dispatch — the full pending set regardless of retry
        schedule. The edge drains :meth:`due_outbox`; this is for inspection/snapshot. The core never
        performs the send."""
        return self._outbox.pending()

    def due_outbox(self, now: float) -> list[OutboxRequest]:
        """Pending requests whose retry schedule (``next_attempt_at``) has arrived — the edge's
        dispatch set at time ``now``. A freshly staged request is due immediately."""
        return self._outbox.due(now)

    def record_outbox_result(
        self,
        request_id: str,
        receipt: OutboxReceipt,
        *,
        max_attempts: int = 5,
        next_attempt_at: float | None = None,
    ) -> str:
        """Record an edge sender's outcome for a staged request and emit the lifecycle event.
        Returns the new status. A retryable failure keeps the request ``pending`` (redispatched on or
        after ``next_attempt_at``, which the edge computes from its backoff policy) until
        ``max_attempts`` attempts, then dead-letters it as ``failed``; a non-retryable failure fails
        immediately. The loop never computes the schedule — it records what the edge decided. The
        ``idempotency_key`` makes the (at-least-once) redispatch safe."""
        request = self._outbox.get(request_id)
        if request is None:
            return ""
        attempts = request.attempts + 1
        recorder = self._session.res.recorder if self._session is not None else None
        if receipt.ok:
            self._outbox.mark(
                request_id, status="dispatched", attempts=attempts, reference=receipt.reference
            )
            if recorder is not None:
                recorder.emit(
                    "outbox.dispatched",
                    data={
                        "request_id": request_id,
                        "destination": request.destination,
                        "reference": receipt.reference,
                        "attempts": attempts,
                        "traceparent": request.traceparent,
                    },
                )
            return "dispatched"
        if receipt.retryable and attempts < max_attempts:
            self._outbox.mark(
                request_id,
                status="pending",
                attempts=attempts,
                next_attempt_at=next_attempt_at,
                error=receipt.error,
            )
            return "pending"
        self._outbox.mark(request_id, status="failed", attempts=attempts, error=receipt.error)
        if recorder is not None:
            recorder.emit(
                "outbox.failed",
                level="warning",
                data={
                    "request_id": request_id,
                    "destination": request.destination,
                    "reason": receipt.error,
                    "attempts": attempts,
                    "traceparent": request.traceparent,
                },
            )
        return "failed"

    async def arun_until_suspended(
        self, user_input: str | tuple[ContentPart, ...] | None = None
    ) -> Suspension:
        """Async form of :meth:`run_until_suspended` — the engine's source of truth."""
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
            # A fresh user turn gets a fresh output-validation budget and a clean result value.
            state.output_retries = 0
            state.final_output = None
            state.output_values = {}
            state.output_failure_history = []
            # A run.finish in a prior submit must not short-circuit this one OR leak its
            # outputs/notes into this turn's validator view / result (clears finished + final_text
            # + final_outputs + final_notes).
            self._clear_finish_metadata(res.context)
            # Drop a stale stop/pause so neither can immediately halt this fresh turn.
            self._interrupt_requested = False
            self._pause_requested = False
            state.pending_user_input = input_to_parts(user_input)
            self._warn_on_unforwarded_multimodal(state.pending_user_input, res.recorder)
            session.submit_local_step = 0
        try:
            suspension = await self._apump_turn(state, res, session)
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
        except ModelAdapterError as exc:
            if not _recoverable_turn_error(exc):
                # Non-recoverable model error -> terminal (same bookkeeping as the generic
                # handler below; a re-raise here would skip that handler, so inline it).
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
            # Recoverable model-turn failure: keep the session alive so the turn can be
            # re-attempted (driver decides: backoff-retry transient, or park for the user to
            # fix config + resend). The user message + observations are already committed to
            # state.messages (appended before the model call); the assistant reply was never
            # appended (success-only). The ONLY leftover to clear for an idempotent re-attempt
            # is pending_observations — otherwise a re-issue re-appends the same tool outputs.
            state.provider_error_code = exc.provider_error_code
            state.provider_http_status = exc.http_status
            res.recorder.emit(
                "turn.failed",
                data={
                    "error": public_error_message(str(exc)),
                    "error_code": exc.error_code,
                    "provider_error_code": exc.provider_error_code,
                    "http_status": exc.http_status,
                    "retryable": exc.retryable,
                },
                level="warning",
            )
            state.pending_observations = ()
            self._persist_checkpoint(session)
            return replace(
                Suspension(reason="turn_failed", status="failed"),
                error=public_error_message(str(exc)),
                error_code=exc.error_code,
                retryable=exc.retryable,
                http_status=exc.http_status,
            )
        except TurnInterrupted:
            # Turn-level stop: keep the session alive (no error, not terminal). Same idempotency
            # as turn_failed — the user message/observations are already committed; only clear
            # pending_observations so a re-issue doesn't re-append tool outputs. The driver parks
            # for the next user message. ``status`` is cosmetic here; branch on ``reason``.
            self._interrupt_requested = False
            res.recorder.emit("turn.interrupted", data={"reason": "user_stop"}, level="info")
            state.pending_observations = ()
            self._persist_checkpoint(session)
            return Suspension(reason="interrupted", status="completed")
        except TurnPaused:
            # Cooperative pause: freeze the turn at a clean start-of-step boundary and keep
            # the session alive. Unlike interrupt, pending_observations are KEPT — the resumed
            # turn (a run_until_suspended(None) re-pump) re-sends them at the next step, so the
            # pause is transparent. The park persists a checkpoint (which already serializes
            # pending_observations + the step counter), so a paused run also survives a restart.
            # ``status`` is cosmetic here; branch on ``reason``.
            self._pause_requested = False
            # Literal state names keep the engine decoupled from the FSM module (the lifecycle
            # layer sits ABOVE the loop); they match SessionState.RUNNING/PAUSED values.
            res.recorder.emit(
                "session.state.changed",
                data={"state": "paused", "from": "running", "reason": "pause_requested"},
            )
            self._persist_checkpoint(session)
            return Suspension(reason="paused", status="completed")
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

    def fail_recoverable(self, message: str, *, error_code: str = "model_error") -> None:
        """Promote a now-exhausted recoverable turn failure to a terminal run failure.

        A driver that has given up retrying a ``turn_failed`` suspension (e.g. the consecutive
        failure cap was hit) calls this to record the durable failure (``failure.json`` +
        ``run.failed``) and mark the session terminal, without having to duplicate the loop's
        terminal bookkeeping. The driver then closes the run as usual."""
        session = self._require_open()
        self._record_failure(
            session.state,
            session.res,
            ModelAdapterError(message, error_code=error_code),
            inherit_provider_detail=True,  # promotion of the prior turn.failed — keep its detail
        )
        session.terminal = True
        self._persist_checkpoint(session)

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
        # Multi-turn sync usage (open/submit*/close) ends here, in the caller thread, so the
        # owned loop is torn down now. The run_once path calls close() from within the loop;
        # there _maybe_close_loop is a no-op and run_once's finally does the teardown.
        self._maybe_close_loop()
        return result

    def run_once(self, user_input: str | tuple[ContentPart, ...]) -> AgentRunResult:
        """One-shot convenience: open() + submit(user_input) + close().

        Sync facade over :meth:`arun_once`; from an async context call ``arun_once``."""
        try:
            return self._run_sync(self.arun_once(user_input))
        finally:
            self._maybe_close_loop()

    async def arun_once(
        self,
        user_input: str | tuple[ContentPart, ...],
        *,
        seed_messages: tuple[dict[str, Any], ...] | None = None,
        seed_media_blobs: Mapping[str, bytes] | None = None,
    ) -> AgentRunResult:
        """Async form of :meth:`run_once`. ``seed_messages`` pre-loads the by-value
        conversation log before the first turn — used by a ``context: fork`` subagent to
        inherit the parent's conversation snapshot (the system prompt is regenerated from
        this run's own config, so a fork sees the history but applies its own directive).
        ``seed_media_blobs`` carries the parent's inline-media bytes so ``blob:`` refs in
        ``seed_messages`` still resolve in the child."""
        self.open()
        try:
            session = self._require_open()
            if not session.terminal:
                if seed_messages:
                    session.state.messages = [dict(message) for message in seed_messages]
                if seed_media_blobs:
                    session.state.media_blobs = dict(seed_media_blobs)
                await self.asubmit(user_input)
        finally:
            result = self.close()
        return result

    async def aopen(self) -> None:
        """Async form of :meth:`open` — offloads the (sync) bootstrap I/O to a thread so
        an event loop is not blocked during workspace/manifest setup."""
        await asyncio.to_thread(self.open)

    async def aclose(self) -> AgentRunResult:
        """Async form of :meth:`close` — offloads the (sync) finalize I/O to a thread."""
        return await asyncio.to_thread(self.close)

    def _run_sync(self, coro: Any) -> Any:
        """Drive an async core method to completion from a synchronous caller.

        Reuses ONE core-owned event loop across the run's turns (not asyncio.run per call),
        so background asyncio tasks created in one turn can span to the next — every wait
        path re-enters this loop, letting those tasks make progress and cross-thread
        ``call_soon_threadsafe`` wakeups land. Raises a clear error (rather than asyncio's
        generic one) when invoked inside a running event loop, pointing at the async API."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            coro.close()
            raise NativeAgentError(
                "sync API called inside a running event loop; await the async API "
                "(arun_once / asubmit / arun_until_suspended) instead",
                error_code="sync_in_async_loop",
            )
        loop = self._ensure_owned_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    def _ensure_owned_loop(self) -> asyncio.AbstractEventLoop:
        """Lazily start the core-owned event loop on a dedicated daemon thread and keep it
        running (run_forever) for the run's lifetime."""
        if self._owned_loop is None:
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever,
                name=f"nar-loop-{self.spec.run_id}",
                daemon=True,
            )
            thread.start()
            self._owned_loop = loop
            self._owned_loop_thread = thread
        return self._owned_loop

    def _maybe_close_loop(self) -> None:
        """Stop and tear down the core-owned loop thread, but only from outside it.
        ``run_once`` calls ``close()`` from within ``arun_once`` (i.e. on the owned loop),
        so close() must not drop the loop it is running on — the run_once facade tears it
        down afterward, from the caller thread."""
        loop = self._owned_loop
        if loop is None:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            return
        loop.call_soon_threadsafe(loop.stop)
        if self._owned_loop_thread is not None:
            self._owned_loop_thread.join(timeout=5)
        loop.close()
        self._owned_loop = None
        self._owned_loop_thread = None

    async def _acall_model(self, request: ModelRequest) -> ModelTurn:
        """Invoke the model adapter, awaiting an async adapter natively or offloading a
        sync ``next_turn`` to a thread so the event loop is never blocked on the LLM call.

        Backward compatible: an adapter exposing ``async def anext_turn`` is awaited; a
        coroutine ``next_turn`` is awaited; a plain sync ``next_turn`` runs in a thread.

        While a stream is active and the adapter supports ``astream_turn``, the streaming
        path is preferred: token chunks are relayed to the stream queue and folded into a
        ``ModelTurn`` so the rest of the turn is identical to the non-streamed path."""
        adapter = self.model_adapter
        sink = self._stream_sink
        if sink is not None and sink.active:
            astream_turn = getattr(adapter, "astream_turn", None)
            if astream_turn is not None:
                return await self._acall_model_streaming(astream_turn, request, sink)
        if self.emit_output_deltas:
            astream_turn = getattr(adapter, "astream_turn", None)
            if astream_turn is not None:
                return await self._acall_model_emitting_deltas(astream_turn, request)
        anext = getattr(adapter, "anext_turn", None)
        if anext is not None:
            return await anext(request)
        next_turn = adapter.next_turn
        if inspect.iscoroutinefunction(next_turn):
            return await next_turn(request)
        return await asyncio.to_thread(next_turn, request)

    async def _acall_model_streaming(
        self,
        astream_turn: Callable[[ModelRequest], Any],
        request: ModelRequest,
        sink: QueueEventSink,
    ) -> ModelTurn:
        """Drive an adapter's ``astream_turn``: relay each chunk to the live stream and
        accumulate them into the turn's ``ModelTurn`` (see ``assemble_streamed_turn``)."""
        chunks: list[ModelStreamChunk] = []
        async for chunk in astream_turn(request):
            sink.push_delta(chunk)
            chunks.append(chunk)
        return assemble_streamed_turn(chunks)

    async def _acall_model_emitting_deltas(
        self,
        astream_turn: Callable[[ModelRequest], Any],
        request: ModelRequest,
    ) -> ModelTurn:
        """Autonomous-drive streaming (no RunStream queue): drive ``astream_turn`` and emit each
        text fragment as a ``model.output.delta`` event, so an event-stream consumer renders
        tokens live. Tool-call/usage chunks are folded only — the assembled ``ModelTurn`` is
        identical to the one-shot path, so the rest of the turn is unchanged."""
        assert self._session is not None
        recorder = self._session.res.recorder
        chunks: list[ModelStreamChunk] = []
        agen = astream_turn(request)
        try:
            async for chunk in agen:
                chunks.append(chunk)
                if isinstance(chunk, TextDelta) and chunk.text:
                    recorder.emit("model.output.delta", data={"text": chunk.text}, level="debug")
                elif isinstance(chunk, ReasoningDelta) and chunk.text:
                    # Display-only reasoning summary (DX-13b): a separate event so a consumer
                    # renders it in a "thinking" view, distinct from the answer text.
                    recorder.emit("model.reasoning.delta", data={"text": chunk.text}, level="debug")
                # Immediate stop: when a turn interrupt arrives mid-stream, abort the in-flight
                # generation now (don't wait for the next step boundary). The text already
                # streamed stays; the except in arun_until_suspended parks the live session.
                if self._interrupt_requested:
                    raise TurnInterrupted("turn interrupted")
        finally:
            # Close the generator so the provider's stream/connection is released promptly
            # (on a normal drain this is a no-op; on the mid-stream abort it cancels the wire).
            aclose = getattr(agen, "aclose", None)
            if aclose is not None:
                await aclose()
        return assemble_streamed_turn(chunks)

    def _record_failure(
        self,
        state: RunState,
        res: _RunResources,
        exc: Exception,
        *,
        inherit_provider_detail: bool = False,
    ) -> None:
        state.status = "failed"
        state.error = str(exc)
        state.error_code = error_code_for_exception(exc)
        if inherit_provider_detail and isinstance(exc, ModelAdapterError):
            # Promotion of a recoverable turn.failed (fail_recoverable): keep the provider detail
            # that turn recorded, adopting the synthetic wrapper's fields only if it carries them.
            if exc.provider_error_code:
                state.provider_error_code = exc.provider_error_code
            if exc.http_status is not None:
                state.provider_http_status = exc.http_status
        else:
            # A fresh terminal failure reflects THIS exception — clearing any stale provider detail
            # an earlier, unrelated recoverable turn.failed may have left on the state.
            if isinstance(exc, ModelAdapterError):
                state.provider_error_code = exc.provider_error_code
                state.provider_http_status = exc.http_status
            else:
                state.provider_error_code = ""
                state.provider_http_status = None
        state.final_text = ""
        res.recorder.emit(
            "run.failed",
            data={
                "error": public_error_message(state.error),
                "error_code": state.error_code,
                "type": type(exc).__name__,
                # Provider failure detail (codes/status, never the raw body) — mirrors turn.failed
                # so the real cause (e.g. insufficient_quota / HTTP 429) reaches logs and the UI.
                "provider_error_code": state.provider_error_code,
                "http_status": state.provider_http_status,
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
            output_retries=state.output_retries,
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
            # Durable (approved) capability leases — handles only — so a restart does not re-prompt.
            capability_leases=self._capability_vault.export_durable(),
            outbox_requests=self._outbox.export(),
            pending_capability_replays=[dict(replay) for replay in state.pending_capability_replays],
            # Revocation records so a revoked capability stays dead across the restart.
            **self._capability_vault.export_revocations(),
        )

    def _checkpoint_store(self) -> CheckpointStore:
        """The injected store, or a default local-fs store under the run root. The core
        only ever talks to this protocol — it never decides where bytes physically land."""
        if self.checkpoint_store is None:
            self.checkpoint_store = LocalFsCheckpointStore(self.spec.run_root)
        return self.checkpoint_store

    def _media_blob_reader(self) -> Callable[[str], bytes] | None:
        """A ``sha -> bytes`` reader over the durable blob store, so the wire-build resolver can
        resolve a ``blob:`` ref that a peer persisted (e.g. the backend normalizing a queued inline
        message via ``put_blob``) and that is therefore not in this loop's in-memory ``media_blobs``.
        ``None`` when no store is configured (in-memory media_blobs then covers everything)."""
        store = self.checkpoint_store
        if store is None:
            return None
        run_id = self.spec.run_id
        return lambda sha: store.get_blob(run_id, sha)

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
        """Content-addressed blobs for the current park, keyed by sha256: the bytes of each
        created/modified workspace file, PLUS the inline-ingested media bytes
        (``state.media_blobs``). Read at the same quiescent park as ``snapshot()`` so the keys
        match the manifest's ``content_sha256`` / ``blob:<sha>`` refs. Both kinds share one
        content-addressed namespace (identical content dedups)."""
        session = self._require_open()
        blobs: dict[str, bytes] = dict(session.state.media_blobs)
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
            output_retries=cp.output_retries,
            total_usage=dict(cp.total_usage)
            or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            messages=list(cp.messages),
            pending_capability_replays=tuple(dict(replay) for replay in cp.pending_capability_replays),
        )
        # Reinstall durable (approved) capability leases so a human-approved capability is not
        # re-prompted after a restart. Ephemeral sync grants were never persisted; they re-broker.
        for lease_payload in cp.capability_leases:
            self._capability_vault.install(CapabilityLease.from_json(lease_payload))
        # Rehydrate staged outbox requests so a pending send survives the restart (the edge
        # re-dispatches it; the idempotency_key guards against a double-send).
        self._outbox.import_(cp.outbox_requests)
        # Restore revocation records so a capability revoked before the restart stays dead.
        self._capability_vault.import_revocations(
            lease_ids=cp.revoked_lease_ids,
            capabilities=cp.revoked_capabilities,
            before=cp.revoked_before,
        )
        # Rehydrate inline-ingested media: load every blob:<sha> referenced by the restored log
        # back into the in-memory map so wire-build can resolve it after the restart. A blob
        # missing from the store is skipped (not fatal) — resolution then surfaces it as a
        # MediaResolveError at wire-build, the same degraded path as a deleted workspace file.
        media_blobs: dict[str, bytes] = {}
        for sha in blob_shas_in_messages(tuple(state.messages)):
            try:
                media_blobs[sha] = blob_reader(sha)
            except KeyError:
                pass
        state.media_blobs = media_blobs
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

    def _install_subagent_capability(
        self,
        registry: ToolRegistry,
        context: AgentToolContext,
        job_manager: TaskManager,
    ) -> None:
        """Register the ``agent.spawn`` tool and the ``subagent`` task executor/injector
        for a run that carries ``subagent_definitions``. Called from bootstrap only when
        definitions are present; the runtime config still needs an explicit binding to
        ``agent.spawn`` for the tool to reach the model."""
        catalog = {
            sub_id: definition.description
            for sub_id, definition in self.subagent_definitions.items()
        }
        registry.register(agent_spawn_tool(catalog))
        context.subagent_depth = int(self.spec.metadata.get("subagent_depth", 0) or 0)
        job_manager.executors["subagent"] = SubagentTaskExecutor(
            run_child=self._run_subagent_child,
            definition_ids=tuple(self.subagent_definitions.keys()),
            max_depth=self.spec.limits.max_subagent_depth,
            max_subagents=self.spec.limits.max_subagents,
        )
        job_manager.injectors["subagent"] = HostedResultInjector(
            kind="subagent",
            tool_name="agent_spawn",
            result_type="subagent_result",
            as_user_message=True,
        )

    async def _run_subagent_child(self, manager: TaskManager, task: HostedTask) -> None:
        """Run one isolated child run for a ``subagent`` task and record its result on
        the task. Builds a fresh child ``AgentLoop`` (isolated overlay workspace, shared
        adapter/sinks/checkpoint store, depth+1) and stores the child's final message;
        ``SubagentTaskExecutor._arun`` then publishes it through the reentry pipe."""
        recorder = manager.recorder
        definition_id = str(task.request.get("definition_id") or "")
        depth = int(task.request.get("depth", 0) or 0)
        background = bool(task.request.get("background", False))
        parent_event_id = task.request.get("parent_event_id")
        turn_id = task.request.get("turn_id")
        definition = self.subagent_definitions[definition_id]
        parent_config = self.runtime_config_provider.current_config(self.spec.run_id)
        is_fork = definition.context == "fork"
        child_run_id = f"{self.spec.run_id}.sub.{task.job_id}"
        child_depth = depth + 1
        # At the depth cap the child must not delegate further: the resolver drops any
        # agent.spawn binding and we give it no definitions, so the tool is simply absent
        # rather than erroring at call time.
        at_max_depth = child_depth >= self.spec.limits.max_subagent_depth
        child_config = self._resolve_child_config(
            definition,
            parent_config,
            definition_id=definition_id,
            at_max_depth=at_max_depth,
            fork=is_fork,
        )
        child_definitions: Mapping[str, SubagentDefinition] = (
            {} if at_max_depth else self.subagent_definitions
        )
        # A fork inherits the parent's conversation snapshot; a fresh subagent starts empty.
        # mode/limits: a fork inherits the parent's outright; a fresh subagent may narrow them.
        seed_messages: tuple[dict[str, Any], ...] | None = None
        seed_media_blobs: Mapping[str, bytes] | None = None
        if is_fork and self._session is not None:
            seed_messages = tuple(dict(message) for message in self._session.state.messages)
            seed_media_blobs = dict(self._session.state.media_blobs)
        # Correlate the delegation on the PARENT's event stream (the child records to its
        # own run dir; stateful sinks like OTel/StatusJson are NOT shared to avoid clobber).
        started = recorder.emit(
            "subagent.started",
            turn_id=turn_id,
            parent_id=parent_event_id,
            data={
                "subagent_type": definition_id,
                "child_run_id": child_run_id,
                "depth": child_depth,
                "background": background,
            },
        )
        child_spec = AgentRunSpec(
            workspace_root=self.spec.workspace_root,
            run_root=self.spec.run_root,
            run_id=child_run_id,
            # mode/limits inherit the parent's unless the definition narrows them.
            mode=self.spec.mode if is_fork else (definition.mode or self.spec.mode),
            workspace_backend="overlay",
            limits=self.spec.limits if is_fork else (definition.limits or self.spec.limits),
            permission_policy=self.spec.permission_policy,
            metadata={
                "parent_run_id": self.spec.run_id,
                "parent_task_id": task.job_id,
                "subagent_definition_id": definition_id,
                "subagent_depth": child_depth,
            },
        )
        child = AgentLoop(
            spec=child_spec,
            model_adapter=self.model_adapter,
            runtime_config_provider=child_config,
            # Inherit the parent's tool providers so MCP/custom tools are in the child's
            # registry (the inherited bindings reference them).
            tool_providers=self.tool_providers,
            dynamic_tool_providers=self.dynamic_tool_providers,
            permission_policy=self.spec.permission_policy,
            cancellation_token=self.cancellation_token,
            shell_approval_provider=self.shell_approval_provider,
            web_gateway_client=self.web_gateway_client,
            workspace_factory=self.workspace_factory,
            checkpoint_store=self.checkpoint_store,
            subagent_definitions=child_definitions,
            status_file=False,
            # Inherit token streaming so a child's work streams into its own events.jsonl too
            # (an observer can tail run_root/<child_run_id>/events.jsonl for live subagent output).
            emit_output_deltas=self.emit_output_deltas,
        )
        result = await child.arun_once(
            task.prompt, seed_messages=seed_messages, seed_media_blobs=seed_media_blobs
        )
        usage = {
            key: result.metrics[key]
            for key in ("input_tokens", "output_tokens", "total_tokens")
            if isinstance(result.metrics, dict) and key in result.metrics
        }
        # Report-only roll-up onto the parent context (NOT total_usage; see field comment).
        if self._session is not None:
            parent_ctx = self._session.res.context
            parent_ctx.subagent_count += 1
            for key, value in usage.items():
                parent_ctx.subagent_usage[key] = parent_ctx.subagent_usage.get(key, 0) + int(value)
        task.result = {
            "type": "subagent_result",
            "task_id": task.job_id,
            "subagent_type": definition_id,
            "child_run_id": child_run_id,
            "status": result.status,
            "message": result.final_text,
            "answer": result.final_text,
            "final_text": result.final_text,
            "error": result.error,
            "usage": usage,
        }
        if result.status == "failed":
            task.status = "failed"
        recorder.emit(
            "subagent.finished" if result.status != "failed" else "subagent.failed",
            turn_id=turn_id,
            parent_id=started.event_id,
            level="error" if result.status == "failed" else "info",
            data={
                "subagent_type": definition_id,
                "child_run_id": child_run_id,
                "status": result.status,
                "usage": usage,
                "error": result.error,
                "error_code": result.error_code,
            },
        )

    def _resolve_child_config(
        self,
        definition: SubagentDefinition,
        parent_config: AgentRuntimeConfig | None,
        *,
        definition_id: str,
        at_max_depth: bool,
        fork: bool = False,
    ) -> AgentRuntimeConfig:
        """Derive a child's runtime config from the parent's, Claude-style: the child
        inherits the parent's tool bindings (so it can never exceed the parent), then the
        definition's ``tools`` allowlist and ``disallowed_tools`` denylist filter them
        (deny wins). ``model``/``tool_search`` inherit unless the definition overrides. At
        the depth cap the ``agent.spawn`` binding is dropped so the child cannot delegate.

        A ``fork`` inherits the parent FULLY — prompt, tools, model, tool_search — and the
        definition's own prompt/tools/model are ignored ("continue as me in a branch")."""
        parent_bindings: tuple[ToolBinding, ...] = parent_config.tools if parent_config else ()
        parent_model = parent_config.model if parent_config else None
        parent_search = parent_config.tool_search if parent_config else ToolSearchConfig()
        parent_prompt = parent_config.prompt if parent_config else PromptSpec()
        if fork:
            bindings = list(parent_bindings)
        elif definition.tools is None:
            bindings = list(parent_bindings)
        else:
            bindings = [b for b in parent_bindings if _binding_matches(b, definition.tools)]
        if not fork and definition.disallowed_tools:
            bindings = [b for b in bindings if not _binding_matches(b, definition.disallowed_tools)]
        if at_max_depth:
            bindings = [b for b in bindings if b.ref.tool_id != "agent.spawn"]
        return AgentRuntimeConfig(
            definition_id=definition_id,
            model=parent_model if fork else (definition.model or parent_model),
            prompt=parent_prompt if fork else definition.prompt,
            tools=tuple(bindings),
            tool_search=parent_search if fork else (definition.tool_search or parent_search),
            metadata=dict(definition.metadata),
        )

    def _bootstrap(self) -> _RunResources:
        if self.permission_policy == PermissionPolicy() and self.spec.permission_policy != PermissionPolicy():
            self.permission_policy = self.spec.permission_policy
        workspace_factory = self.workspace_factory or default_local_workspace_factory
        workspace = workspace_factory(self.spec)
        # A dormant stream sink rides on the EventBus for the whole run; astream activates it
        # per stream (no EventBus mutation, supports sequential streams). Inert otherwise.
        self._stream_sink = QueueEventSink()
        recorder = AgentRecorder(
            self.spec.run_root,
            self.spec.run_id,
            extra_event_sinks=(*self.event_sinks, self._stream_sink),
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
            capability_vault=self._capability_vault,
            outbox=self._outbox,
        )
        base_registry = ToolRegistry()
        base_registry.register_many(builtin_tools(workspace))
        for provider in self.tool_providers:
            base_registry.register_many(provider.get_tools(context))
        if self.subagent_definitions:
            self._install_subagent_capability(base_registry, context, job_manager)

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
        self._emit_bootstrap_validator_skips(initial_runtime_config, recorder)
        return _RunResources(
            workspace=workspace,
            recorder=recorder,
            context=context,
            base_tool_specs=tuple(base_registry.specs()),
            started=started,
            deadline=deadline,
            static_segments=tuple(static_segments),
        )

    def _active_output_validators(
        self, config: AgentRuntimeConfig | None
    ) -> tuple[OutputValidator, ...]:
        """Validators that run this settle: every registered validator EXCEPT those a config
        binding disables (**default on**). Resolved from the *per-turn* config so a mid-run hot-swap
        (``replace_runtime_config`` adding ``OutputValidatorBinding(enabled=False)``) takes effect.
        ``config is None`` (pre-bootstrap) → all registered. Pure — no events/state."""
        if config is None:
            return self.output_validators
        disabled_ids = {b.validator_id for b in config.output_validators if not b.enabled}
        return tuple(v for v in self.output_validators if v.id not in disabled_ids)

    def _emit_bootstrap_validator_skips(
        self, config: AgentRuntimeConfig, recorder: AgentRecorder
    ) -> None:
        """One-time discoverability at bootstrap: ``output.validator.skipped`` for each registered
        validator a config binding disables (``reason=disabled``), and for each binding referencing
        an unregistered validator (``reason=unknown_binding`` at warning level — a no-op opt-out,
        commonly a typo). The reference backend validates configs via ``validate_runtime_config``,
        not ``AgentLoop.validate``, so this bootstrap emission is the universal signal. Per-turn
        gating re-resolves via ``_active_output_validators``, so a later config change is honored."""
        registered_ids = {v.id for v in self.output_validators}
        active_ids = {v.id for v in self._active_output_validators(config)}
        for validator in self.output_validators:
            if validator.id not in active_ids:
                recorder.emit(
                    "output.validator.skipped",
                    data={"validator_id": validator.id, "reason": "disabled"},
                    level="debug",
                )
        for binding in config.output_validators:
            if binding.validator_id not in registered_ids:
                recorder.emit(
                    "output.validator.skipped",
                    data={"validator_id": binding.validator_id, "reason": "unknown_binding"},
                    level="warning",
                )

    @staticmethod
    def _clear_finish_metadata(context: AgentToolContext) -> None:
        """Reset the metadata a run.finish populated, so a REJECTED finish can't leak its
        outputs/notes into close() or re-settle the next turn on a stale flag."""
        context.finished = False
        context.final_text = ""
        context.final_outputs = []
        context.final_notes = None

    @staticmethod
    def _log_finish_observations(state: RunState) -> None:
        """Append this turn's pending tool outputs (the run.finish function_call_output, plus any
        siblings) to the by-value log and clear them — so a by-value continuation never carries a
        dangling function_call (no user/repair message can slip in ahead of the output)."""
        for observation in state.pending_observations:
            state.messages.append(_observation_message(observation, state.media_blobs))
        state.pending_observations = ()

    async def _resolve_final_output(
        self,
        state: RunState,
        res: _RunResources,
        context: AgentToolContext,
        turn: ModelTurn,
        runtime_config: AgentRuntimeConfig,
        *,
        from_finish: bool,
    ) -> Suspension | None:
        """Apply the active output validators at a settle point.

        Returns the ``Suspension`` to settle on (success / refusal / truncation / exhaustion), or
        ``None`` to continue the pump (a re-prompt has been queued). Raises ``OutputValidatorError``
        on a validator *defect* (terminalized by the loop's broad boundary). Called just before
        each ``Suspension(reason="settled")`` return; the ``run.finish`` site passes
        ``from_finish=True`` so the stale ``context.finished`` flag is cleared before a re-prompt.

        The validators are run off the event loop (``asyncio.to_thread``) so a slow/blocking
        validator never stalls the loop; all emits + state mutation stay on the loop thread.
        """
        recorder = res.recorder
        # Refusal / truncation: never validate output that is non-conforming by construction.
        if turn.stop_reason == "refusal":
            state.status = "failed"
            state.error_code = "output_refused"
            recorder.emit("output.validation.failed", data={"reason": "refusal"}, level="warning")
            return Suspension(
                reason="settled", status=state.status, final_text=state.final_text, error_code=state.error_code  # type: ignore[arg-type]
            )
        if turn.stop_reason == "length":
            state.status = "limited"
            state.error_code = "output_truncated"
            recorder.emit("output.validation.failed", data={"reason": "truncation"}, level="warning")
            return Suspension(
                reason="limited", status=state.status, final_text=state.final_text, error_code=state.error_code  # type: ignore[arg-type]
            )

        validators = self._active_output_validators(runtime_config)  # per-turn (honors hot-swap)
        if not validators:
            # No validator runs (none registered, or all disabled for this turn), but a run.finish
            # still produced a tool output that must be logged before parking — same bookkeeping as
            # a validated finish, so a multi-turn by-value continuation isn't left dangling.
            if from_finish:
                self._log_finish_observations(state)
            return Suspension(reason="settled", status=state.status, final_text=state.final_text)  # type: ignore[arg-type]

        view = self._build_final_output_view(state, res, context)
        # Offload validation — validators may block on file reads / heavy regex / I/O, and must not
        # stall the event loop (concurrent runs + background tasks share it). The helper is pure;
        # emits, state mutation, and the Suspension all happen below on the loop thread.
        failures, ok_values, defect = await asyncio.to_thread(_run_output_validators, validators, view)

        if defect is not None:
            validator_id, exc = defect
            recorder.emit(
                "output.validator.error",
                data={"validator_id": validator_id, "error": str(exc)},
                level="error",
            )
            raise OutputValidatorError(f"output validator {validator_id!r} raised: {exc}") from exc

        if not failures:
            state.output_values = dict(ok_values)  # keyed by validator id
            state.final_output = ok_values[-1][1] if ok_values else None  # back-compat: last ok wins
            if from_finish:
                # The validated run.finish is the real answer — keep its metadata, but log its
                # tool output before the run parks (multi-turn continuation needs it).
                self._log_finish_observations(state)
            for validator_id, _value in ok_values:
                recorder.emit("output.validator.satisfied", data={"validator_id": validator_id})
            return Suspension(reason="settled", status=state.status, final_text=state.final_text)  # type: ignore[arg-type]

        attempt = len(state.output_failure_history) + 1
        attempt_failures = [{"validator_id": vid, "feedback": fb} for vid, fb in failures]
        state.output_failure_history.append({"attempt": attempt, "failures": attempt_failures})
        recorder.emit(
            "output.validation.failed",
            data={"attempt": attempt, "failures": attempt_failures},
            level="warning",
        )
        if state.output_retries >= self.spec.limits.max_output_retries:
            # Exhausted: every allowed re-prompt has been issued. output_retries counts the
            # re-prompts actually made — not this terminal failed attempt (``attempt`` does that).
            state.status = "limited"
            state.final_text = state.final_text or "Stopped: the final response did not satisfy the output contract."
            state.error_code = "output_validator_unsatisfied"
            # Surface the cross-attempt failure roll-up so a jointly-unsatisfiable validator set is
            # diagnosable (oscillating ids with equal counts) rather than a silent budget burn.
            recorder.emit(
                "output.validator.exhausted",
                data={
                    "retries": state.output_retries,
                    "failures_by_validator": _failures_by_validator(state.output_failure_history),
                    "history": list(state.output_failure_history),
                },
                level="warning",
            )
            if from_finish:
                # Exhausted on a rejected run.finish: clear its metadata (else close() reports the
                # rejected outputs/notes) and log its tool output (the by-value log may continue).
                self._clear_finish_metadata(context)
                self._log_finish_observations(state)
            return Suspension(
                reason="limited", status=state.status, final_text=state.final_text, error_code=state.error_code  # type: ignore[arg-type]
            )

        state.output_retries += 1  # an actual re-prompt is about to be queued

        # Re-prompt: queue feedback and continue the pump.
        if from_finish:
            # Rejected run.finish: clear its metadata, then log its tool output BEFORE the repair
            # user message so a by-value adapter never sends a dangling function_call. (The natural
            # settle path has no pending observations: they were cleared at the call site.)
            self._clear_finish_metadata(context)
            state.final_text = ""
            self._log_finish_observations(state)
        state.pending_observations = ()
        state.messages.append({"role": "user", "content": _output_repair_message(failures)})
        return None

    def _build_final_output_view(
        self, state: RunState, res: _RunResources, context: AgentToolContext
    ) -> FinalOutputView:
        """The read-only composite (text + files) handed to a validator. ``read_bytes`` goes
        through ``workspace.read_bytes`` so it inherits the path jail + ``max_bytes_read`` cap;
        a validator may pass ``max_bytes`` to raise the cap for a legit large artifact."""
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
            )
            for a in res.recorder.artifacts
        )
        return FinalOutputView(
            final_text=state.final_text,
            artifacts=artifacts,
            final_outputs=tuple(context.final_outputs),
            read_bytes=_read,
        )

    def _warn_on_unforwarded_multimodal(
        self, parts: tuple[ContentPart, ...], recorder: AgentRecorder
    ) -> None:
        """Emit a ``model.input.degraded`` warning for any non-text part that will NOT be
        forwarded this run, so the degradation stays observable. A multimodal adapter
        forwards the wire-forwardable types (see ``WIRE_FORWARDABLE_PART_TYPES``); a
        text-only adapter forwards none."""
        dropped = non_text_part_types(parts)
        if not dropped:
            return
        if getattr(self.model_adapter, "supports_multimodal", False):
            unforwarded = [t for t in dropped if t not in WIRE_FORWARDABLE_PART_TYPES]
            reason = "type_not_forwarded"
        else:
            unforwarded = dropped
            reason = "adapter_lacks_multimodal"
        if not unforwarded:
            return
        recorder.emit(
            "model.input.degraded",
            data={"dropped_part_types": unforwarded, "reason": reason},
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

    async def _apump_turn(self, state: RunState, res: _RunResources, session: _Session) -> Suspension:
        context = res.context
        recorder = res.recorder
        deadline = res.deadline
        # Bind the run's (always-on) loop so background shell jobs schedule their asyncio
        # subprocess monitors onto it — they then progress while the run is parked between
        # turns, on the same single loop that drives the run.
        context.job_manager.bind_loop(asyncio.get_running_loop())
        # The per-submit step budget continues across task-wait suspensions within one
        # submit; session_step is the global, monotonic turn counter for turn ids.
        max_steps = self.spec.limits.max_steps
        while session.submit_local_step < max_steps:
            self._check_run_boundary(deadline)
            # Cooperative pause is checked ONLY here, at the start of a step — never inside
            # _check_run_boundary (which also runs mid-step). At this boundary the prior step's
            # tool results sit in state.pending_observations not-yet-sent, so a paused park is
            # clean and a None re-pump resumes the same turn without losing or double-sending them.
            if self._pause_requested:
                raise TurnPaused("turn paused")
            session.submit_local_step += 1
            local_step = session.submit_local_step
            session.session_step += 1
            step = session.session_step
            background_observations = self._pop_background_observations(context, recorder, step, state)
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
            # Now that the active tool set for this turn is known, expose it on the turn context so
            # a context provider's dynamic_segment can gate itself on the live config (e.g. the
            # Skills catalog tracks the skill tool binding across a hot-swap).
            turn_context = replace(
                turn_context, bound_tools=frozenset(tool.base_spec.id for tool in bound_catalog.tools)
            )
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
            # Auto-redispatch (⑤): run any gated calls whose capability was just granted, now that
            # this turn's context (catalog/surface/turn_id) exists. Each goes through the normal
            # _execute_tool_call (real permission/quota/events); the result is injected as a
            # user-message observation so the model sees the outcome without retrying. A replay that
            # can't run cleanly (no valid lease) is skipped — the model then retries (fallback).
            if state.pending_capability_replays:
                pending_replays = state.pending_capability_replays
                state.pending_capability_replays = ()
                replay_obs = tuple(
                    obs
                    for replay in pending_replays
                    if (
                        obs := self._execute_capability_replay(
                            replay,
                            bound_catalog=bound_catalog,
                            surface_snapshot=surface_snapshot,
                            call_counts=state.tool_call_counts,
                            context=context,
                            recorder=recorder,
                            turn_id=turn_id,
                            step=step,
                        )
                    )
                    is not None
                )
                if replay_obs:
                    state.pending_observations = (*state.pending_observations, *replay_obs)
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
            user_message: dict[str, Any] | None = None
            if state.pending_user_input is not None:
                # Inline ingress: any part handed in by value (a ``data:`` source_ref) is
                # persisted to the content-addressed media-blob store and rewritten to a durable
                # ``blob:<sha>`` ref BEFORE it enters the by-value log — so the log/checkpoint
                # stay by-reference (tiny, resumable) and never carry the bytes inline.
                state.pending_user_input = tuple(
                    normalize_inline_media_part(part, state.media_blobs)
                    for part in state.pending_user_input
                )
                # ``instruction`` is the text projection used only by the by-reference
                # fallback path (first turn / follow-up on a handle). ``user_message``
                # is the durable by-value log entry: a plain string for all-text input,
                # or a by-reference parts list when non-text parts are present, so an
                # image survives in the log and across checkpoint/resume.
                instruction = text_from_parts(state.pending_user_input) or None
                user_message = user_message_from_parts(state.pending_user_input)
                state.pending_user_input = None
            # Accumulate the by-value conversation log BEFORE the call: the new user
            # message (if any) and the tool/async observations being sent this turn. The
            # assistant reply is appended after the call. The system prompt is NOT logged
            # here — it is regenerated per turn and travels via ``system_prompt``.
            if user_message is not None:
                state.messages.append(user_message)
            for observation in state.pending_observations:
                state.messages.append(_observation_message(observation, state.media_blobs))
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
            # Token budget: checked before the turn against the accumulated API-reported
            # usage of prior turns, so once a cap is crossed the run settles rather than
            # starting (and paying for) another turn.
            token_limit_code = self._token_budget_exceeded(state)
            if token_limit_code is not None:
                state.status = "limited"
                state.final_text = "Stopped after reaching the token budget."
                state.error_code = token_limit_code
                state.pending_observations = ()
                return Suspension(
                    reason="limited",
                    status="limited",
                    final_text=state.final_text,
                    error_code=token_limit_code,
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
            # By-value wire copy: the durable log stays by-reference; a multimodal adapter
            # gets media resolved to wire blocks here (once per turn, not per retry). A
            # text-only adapter receives the by-reference log and projects it to text.
            wire_messages = tuple(state.messages)
            if getattr(self.model_adapter, "supports_multimodal", False):
                # Tool-result image eviction runs on the by-reference copy BEFORE resolution,
                # so dropped images are never read/encoded. Off unless a keep-N is configured.
                evicted = 0
                keep_n = self.spec.limits.keep_recent_tool_images
                if keep_n is not None:
                    before = count_tool_result_images(wire_messages)
                    wire_messages = evict_tool_result_images(wire_messages, keep_n)
                    evicted = before - count_tool_result_images(wire_messages)
                wire_messages = resolve_wire_messages(
                    wire_messages,
                    WorkspaceMediaResolver(
                        res.workspace, blobs=state.media_blobs, blob_reader=self._media_blob_reader()
                    ),
                    encoding=getattr(self.model_adapter, "wire_image_encoding", "base64"),
                )
                self._emit_media_accounting(
                    wire_messages,
                    (runtime_config.model or ModelConfig()).model,
                    recorder,
                    evicted_image_count=evicted,
                )
                # The resolved payload (inline base64) is the real size risk — the durable
                # by-reference log stays tiny. Guard it separately so an oversized media turn
                # settles ``limited`` instead of being sent.
                wire_limit_code = self._wire_bytes_exceeded(wire_messages)
                if wire_limit_code is not None:
                    state.status = "limited"
                    state.final_text = "Stopped after reaching the model request size limit."
                    state.error_code = wire_limit_code
                    state.pending_observations = ()
                    return Suspension(
                        reason="limited",
                        status="limited",
                        final_text=state.final_text,
                        error_code=wire_limit_code,
                    )
            request = ModelRequest(
                instruction=instruction,
                system_prompt=turn_system_prompt,
                tools=surface_snapshot.immediate_tools,
                previous_turn_handle=state.previous_turn_handle,
                observations=state.pending_observations,
                model=runtime_config.model or ModelConfig(),
                messages=wire_messages,
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
                turn = await self._acall_model(request)
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
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": turn.final_text or "",
                "tool_calls": [call.__dict__ for call in turn.tool_calls],
            }
            # Carry provider-native reasoning artifacts so they round-trip on the next turn
            # (DX-13a). Tagged with provider+model so replay only happens against a matching
            # adapter/model; non-reasoning adapters leave ``turn.reasoning`` empty (neutral seam).
            if turn.reasoning:
                provider_name = getattr(self.model_adapter, "provider_name", None)
                if provider_name:
                    assistant_message["reasoning"] = {
                        "provider": provider_name,
                        "model": (runtime_config.model or ModelConfig()).model,
                        "items": [dict(item) for item in turn.reasoning],
                    }
            state.messages.append(assistant_message)
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
            metrics_data: dict[str, Any] = {
                "step": step,
                "tool_calls": state.total_tool_calls,
                "input_tokens": state.total_usage["input_tokens"],
                "output_tokens": state.total_usage["output_tokens"],
                "total_tokens": state.total_usage["total_tokens"],
                "web_search_calls": context.web_service.web_search_calls,
                "web_fetch_calls": context.web_service.web_fetch_calls,
                "web_context_calls": context.web_service.web_context_calls,
                "web_failed_calls": context.web_service.web_failed_calls,
            }
            # Surface reasoning tokens (the priced, invisible "thinking" sub-count) when the
            # adapter reports them, so the studio meter can show the reasoning share (R10).
            if state.total_usage.get("reasoning_tokens"):
                metrics_data["reasoning_tokens"] = state.total_usage["reasoning_tokens"]
            recorder.emit(
                "metrics.updated",
                turn_id=turn_id,
                parent_id=turn_started.event_id,
                data=metrics_data,
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
                # Settle on final text — OR on a refusal/truncation even when the model emitted no
                # text (an OpenAI ``refusal`` content part yields stop_reason="refusal" with
                # final_text=None; a zero-token cap yields "length"). Those must reach the
                # refusal/truncation branch (output_refused / output_truncated), not the
                # "neither text nor tool calls" error below.
                if turn.final_text or turn.stop_reason in ("refusal", "length"):
                    state.final_text = turn.final_text or ""
                    # The model has consumed the pending observations and settled;
                    # the next submit must not resend them alongside a new message.
                    state.pending_observations = ()
                    settled = await self._resolve_final_output(
                        state, res, context, turn, runtime_config, from_finish=False
                    )
                    if settled is None:
                        continue  # output validation failed → repair queued, re-pump
                    return settled
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
                # Offload the (sync) tool handler — which may block on shell/web/fs — to a
                # thread so the event loop stays free. Awaited sequentially, so there is no
                # concurrent access to the shared context/state it mutates.
                observation = await asyncio.to_thread(
                    self._execute_tool_call,
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
                settled = await self._resolve_final_output(
                    state, res, context, turn, runtime_config, from_finish=True
                )
                if settled is None:
                    continue  # output validation failed → repair queued, re-pump
                return settled
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

    def _token_budget_exceeded(self, state: RunState) -> str | None:
        """Return the limit error_code if the run's accumulated API-reported usage has
        crossed a configured token budget, else ``None``. Reads ``state.total_usage`` —
        the authoritative provider actuals summed across turns, not an estimate."""
        limits = self.spec.limits
        usage = state.total_usage
        if limits.max_input_tokens is not None and usage.get("input_tokens", 0) > limits.max_input_tokens:
            return "input_tokens_exceeded"
        if limits.max_output_tokens is not None and usage.get("output_tokens", 0) > limits.max_output_tokens:
            return "output_tokens_exceeded"
        if limits.max_total_tokens is not None and usage.get("total_tokens", 0) > limits.max_total_tokens:
            return "total_tokens_exceeded"
        return None

    def _wire_bytes_exceeded(self, wire_messages: tuple[dict[str, Any], ...]) -> str | None:
        """Return ``"wire_bytes_exceeded"`` if the resolved per-turn wire payload (inline
        base64 media included) outgrows ``max_message_log_bytes``, else ``None``. Distinct
        from the durable by-reference log cap, which never carries bytes."""
        size = sum(len(json.dumps(message, ensure_ascii=False)) for message in wire_messages)
        if size > self.spec.limits.max_message_log_bytes:
            return "wire_bytes_exceeded"
        return None

    def _emit_media_accounting(
        self,
        wire_messages: tuple[dict[str, Any], ...],
        model: str | None,
        recorder: AgentRecorder,
        *,
        evicted_image_count: int = 0,
    ) -> None:
        """Emit per-turn media accounting: count resolved image blocks, estimate their input
        tokens (28×28 patch formula, clamped to the model's native cap), and warn past the
        >20-block cliff where providers enforce a stricter per-image dimension limit."""
        cap = native_image_token_cap(model)
        blocks = 0
        estimated_tokens = 0
        for message in wire_messages:
            # Resolved image blocks live in a user ``content`` list and/or a tool ``media`` list.
            parts = []
            if isinstance(message.get("content"), list):
                parts.extend(message["content"])
            if isinstance(message.get("media"), list):
                parts.extend(message["media"])
            for part in parts:
                if not (isinstance(part, dict) and part.get("type") in WIRE_FORWARDABLE_PART_TYPES):
                    continue
                blocks += 1
                source = part.get("source") or {}
                if source.get("type") != "base64":
                    continue
                try:
                    raw = base64.b64decode(source.get("data") or "")
                except (ValueError, TypeError):
                    continue
                dims = image_dimensions(raw, str(source.get("media_type") or ""))
                if dims is not None:
                    estimated_tokens += estimate_image_tokens(*dims, cap=cap)
        if blocks == 0 and evicted_image_count == 0:
            return
        if blocks > MAX_FORWARDABLE_BLOCKS:
            recorder.emit(
                "model.input.degraded",
                data={"reason": "image_block_count_cliff", "block_count": blocks},
                level="warning",
            )
        data: dict[str, Any] = {"block_count": blocks, "estimated_image_tokens": estimated_tokens}
        if evicted_image_count:
            data["evicted_image_count"] = evicted_image_count
        recorder.emit("model.input.media", data=data)

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
        if context.subagent_count:
            metrics["subagent_count"] = context.subagent_count
            metrics["subagent_usage"] = dict(context.subagent_usage)
        if context.skill_activation_count:
            metrics["skill_activation_count"] = context.skill_activation_count
            metrics["skills_activated"] = list(context.skills_activated)
        if state.output_failure_history:
            # Output-validation diagnostics surfaced in the run result: how many re-prompts were
            # spent and which validators kept failing (a contradiction shows as equal counts).
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
            final_output=state.final_output,
            outputs=dict(state.output_values),
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
                # Output-validation summary for this settle: how many validators were active and
                # how many re-prompts the run spent satisfying them (0 when none ran).
                "output_validators": len(self._active_output_validators(state.previous_runtime_config)),
                "output_retries": state.output_retries,
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
            final_output=state.final_output,
            outputs=dict(state.output_values),
            metrics=metrics,
        )

    def _pop_background_observations(
        self,
        context: AgentToolContext,
        recorder: AgentRecorder,
        step: int,
        state: RunState,
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
            # A resolved capability escalation: admit the granted lease into the vault, and (if a
            # gated call was captured + auto-redispatch is on) queue it to run at this step's top.
            elif observation.output.get("type") == "capability_grant":
                self._admit_capability_grant(observation, context, recorder, state)
        return tuple(observations)

    def _admit_capability_grant(
        self,
        observation: ToolObservation,
        context: AgentToolContext,
        recorder: AgentRecorder,
        state: RunState,
    ) -> None:
        """Store the lease from a resolved ``capability`` task in the vault (fail-closed against the
        original request scope), and queue the gated call for auto-redispatch when one was captured.
        A denial / malformed grant stores nothing — the model just sees the result observation."""
        task = context.job_manager.jobs.get(str(observation.output.get("task_id") or ""))
        request_payload = getattr(task, "request", None) if task is not None else None
        result_payload = getattr(task, "result", None) if task is not None else None
        if not isinstance(request_payload, dict) or not isinstance(result_payload, dict):
            return
        lease_payload = result_payload.get("lease")
        if not isinstance(lease_payload, dict):
            return  # denied or no lease granted
        request = CapabilityRequest(
            capability=str(request_payload.get("capability") or ""),
            scope=dict(request_payload.get("scope") or {}),
            run_id=self.spec.run_id,
            binding_id=str(request_payload.get("binding_id") or ""),
        )
        lease = CapabilityLease(
            capability=str(lease_payload.get("capability") or request.capability),
            token_ref=str(lease_payload.get("token_ref") or ""),
            expires_at=float(lease_payload.get("expires_at") or 0.0),
            scope=dict(lease_payload.get("scope") or {}),
            # Approved out-of-band → persist (handle only) so a restart does not re-prompt.
            durable=True,
        )
        try:
            self._capability_vault.admit(request, lease)
        except ValueError as exc:
            recorder.emit(
                "capability.denied",
                data={
                    "capability": request.capability,
                    "binding_id": request.binding_id,
                    "reason": str(exc),
                },
                level="warning",
            )
            return
        recorder.emit(
            "capability.granted",
            data={
                "capability": request.capability,
                "binding_id": request.binding_id,
                "lease_id": lease.lease_id,
                "expires_at": lease.expires_at,
                "scope": lease.scope,
            },
        )
        # Queue the captured gated call for auto-redispatch (drained at the next step top, where
        # turn context exists). If nothing was captured or the flag is off, the model retries.
        replay_call_name = request_payload.get("replay_call_name")
        if self.capability_auto_redispatch and replay_call_name:
            state.pending_capability_replays = (
                *state.pending_capability_replays,
                {
                    "call_name": str(replay_call_name),
                    "call_id": str(request_payload.get("replay_call_id") or ""),
                    "arguments": dict(request_payload.get("replay_arguments") or {}),
                    "binding_id": request.binding_id,
                    "capability": request.capability,
                    "task_id": str(observation.output.get("task_id") or ""),
                },
            )

    def _execute_capability_replay(
        self,
        replay: dict[str, Any],
        *,
        bound_catalog: BoundToolCatalog,
        surface_snapshot: ToolSurfaceSnapshot,
        call_counts: dict[str, int],
        context: AgentToolContext,
        recorder: AgentRecorder,
        turn_id: str,
        step: int,
    ) -> ToolObservation | None:
        """Re-execute one gated tool call after its capability was granted, returning the result as
        a user-message observation (so it never collides with the original call's pending tool
        result). Returns ``None`` to skip — falling back to model-retry — when no valid lease is
        present (the gate would otherwise re-escalate and re-park)."""
        capability = str(replay.get("capability") or "")
        if not capability or self._capability_vault.token_for(capability, now=time.time()) is None:
            return None  # lease missing/expired -> let the model retry (the granted message stands)
        observation = self._execute_tool_call(
            call_name=str(replay.get("call_name") or ""),
            call_id=str(replay.get("call_id") or ""),
            arguments=dict(replay.get("arguments") or {}),
            bound_catalog=bound_catalog,
            surface_snapshot=surface_snapshot,
            call_counts=call_counts,
            context=context,
            recorder=recorder,
            turn_id=turn_id,
            parent_id=None,
            step=step,
        )
        # Deliver as a user message (is_background) under a distinct call_id — the original call_id
        # already carries the "pending" tool result, so a second tool result there would be malformed.
        return ToolObservation(
            call_id=f"capability_replay:{replay.get('call_id') or ''}",
            tool_name=str(replay.get("call_name") or ""),
            output={
                "type": "capability_replay_result",
                "capability": capability,
                "call": str(replay.get("call_name") or ""),
                "result": observation.output,
            },
            is_background=True,
        )

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
            media=tuple(content_part_to_json(part) for part in result.media),
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
                pending = self._ensure_capability_lease(
                    bound_tool,
                    context,
                    recorder,
                    started_event,
                    turn_id,
                    call_name=call_name,
                    call_id=call_id,
                    arguments=arguments,
                )
                if pending is not None:
                    # Capability escalated: the call parks (does not execute); the model retries
                    # once the lease is granted. Not counted against the binding's call quota.
                    result = pending
                else:
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

    def _ensure_capability_lease(
        self,
        bound_tool: BoundTool,
        context: AgentToolContext,
        recorder: AgentRecorder,
        started_event: AgentEvent | None,
        turn_id: str,
        *,
        call_name: str,
        call_id: str,
        arguments: dict[str, Any],
    ) -> ToolResult | None:
        """Gate a tool call on a capability lease. Returns ``None`` to proceed (a valid lease is
        cached or was granted synchronously), or a *pending* ``ToolResult`` when the broker
        escalated the request (the run will park on a ``capability`` task and the model retries the
        tool once granted). A denial — or a scope-widening grant — raises ``PermissionDenied`` so the
        call never runs. A no-op unless a broker is configured AND the binding declares
        ``runtime.requires_lease``. Secrets never enter the core — a lease carries only a handle."""
        broker = self.capability_broker
        runtime = bound_tool.binding.runtime or {}
        if broker is None or not runtime.get("requires_lease"):
            return None
        capability = bound_tool.base_spec.capability
        if not capability:
            return None
        scope = {key: value for key, value in bound_tool.binding.scope.to_json().items() if value}
        now = time.time()
        binding_id = bound_tool.binding_id
        parent_id = started_event.event_id if started_event else None
        if self._capability_vault.is_capability_revoked(capability):
            # Hard stop: a revoked capability is refused WITHOUT re-brokering, so a permissive broker
            # cannot resurrect it. (A revoked lease_id / pre-watermark lease is filtered by get_valid
            # below and would re-broker; per-capability revocation is the authoritative kill.)
            recorder.emit(
                "capability.revoked",
                turn_id=turn_id,
                parent_id=parent_id,
                level="warning",
                data={"capability": capability, "scope": scope, "reason": "revoked"},
            )
            raise PermissionDenied(
                f"capability revoked: {capability}", error_code="capability_revoked"
            )
        cached = self._capability_vault.get_valid(capability, scope, now=now)
        if cached is not None:
            skew = self.capability_rotate_skew_seconds
            if skew > 0 and cached.can_rotate(now, skew):
                self._rotate_capability_lease(
                    cached,
                    capability,
                    scope,
                    binding_id,
                    recorder=recorder,
                    turn_id=turn_id,
                    parent_id=parent_id,
                )
            return None  # a valid, scope-covering lease is cached (refreshed if it was due)
        request = CapabilityRequest(
            capability=capability,
            scope=scope,
            run_id=self.spec.run_id,
            binding_id=binding_id,
            reason=str(runtime.get("capability_reason") or ""),
        )
        recorder.emit(
            "capability.requested",
            turn_id=turn_id,
            parent_id=parent_id,
            data={
                "capability": capability,
                "binding_id": binding_id,
                "request_id": request.request_id,
                "scope": scope,
                "reason": request.reason,
            },
        )
        grant = broker.request(request)
        if isinstance(grant, CapabilityDenial):
            recorder.emit(
                "capability.denied",
                turn_id=turn_id,
                parent_id=parent_id,
                level="warning",
                data={
                    "capability": capability,
                    "binding_id": binding_id,
                    "reason": grant.reason,
                    "retryable": grant.retryable,
                },
            )
            raise PermissionDenied(
                f"capability denied: {capability}: {grant.reason}", error_code="capability_denied"
            )
        if isinstance(grant, CapabilityPending):
            # Async approval: park the run on a capability hosted-task carrying the request AND the
            # gated call (so it can be auto-redispatched on grant — see _capability_replay_for_grant),
            # and hand the model a "pending" observation. On resolution the lease is admitted and the
            # call runs automatically (or, if auto-redispatch is off/unsafe, the model retries it).
            task_id = context.job_manager.create_task(
                "capability",
                {
                    "capability": capability,
                    "scope": scope,
                    "binding_id": binding_id,
                    "request_id": request.request_id,
                    "ttl_seconds": request.ttl_seconds,
                    "reason": request.reason,
                    "prompt": grant.prompt,
                    # The gated call, captured for auto-redispatch (durable via the hosted task).
                    "replay_call_name": call_name,
                    "replay_call_id": call_id,
                    "replay_arguments": dict(arguments),
                },
            )
            tail = (
                "Once it is granted it will run automatically; you do not need to retry."
                if self.capability_auto_redispatch
                else "Do not repeat other work for it; once it is granted, retry this tool."
            )
            return ToolResult(
                ok=True,
                content={
                    "status": "pending_capability",
                    "capability": capability,
                    "request_id": request.request_id,
                    "task_id": task_id,
                    "message": f"Access to '{capability}' is pending approval (task {task_id}). {tail}",
                },
            )
        try:
            lease = self._capability_vault.admit(request, grant)
        except ValueError as exc:
            recorder.emit(
                "capability.denied",
                turn_id=turn_id,
                parent_id=parent_id,
                level="warning",
                data={"capability": capability, "binding_id": binding_id, "reason": str(exc)},
            )
            raise PermissionDenied(
                f"capability grant rejected: {exc}", error_code="capability_scope_widened"
            ) from exc
        recorder.emit(
            "capability.granted",
            turn_id=turn_id,
            parent_id=parent_id,
            data={
                "capability": capability,
                "binding_id": binding_id,
                "lease_id": lease.lease_id,
                "expires_at": lease.expires_at,
                "scope": lease.scope,
            },
        )
        return None

    def _rotate_capability_lease(
        self,
        current: CapabilityLease,
        capability: str,
        scope: dict[str, Any],
        binding_id: str,
        *,
        recorder: AgentRecorder,
        turn_id: str,
        parent_id: str | None,
    ) -> None:
        """Proactively refresh a near-expiry lease (see ``capability_rotate_skew_seconds``). Re-brokers
        a fresh lease for the same scope and admits it, carrying over the lease's durability and its
        ``max_expires_at`` ceiling (and capping the refreshed expiry at that ceiling). A non-grant
        (deny/pending) or a scope-widening grant leaves the still-valid current lease untouched —
        rotation never disrupts an in-flight capability; the lease just expires later and re-brokers
        through the normal path."""
        broker = self.capability_broker
        if broker is None:
            return
        request = CapabilityRequest(
            capability=capability, scope=scope, run_id=self.spec.run_id, binding_id=binding_id
        )
        grant = broker.request(request)
        if not isinstance(grant, CapabilityLease):
            return  # deny/pending — keep the current valid lease, no disruption
        ceiling = current.max_expires_at
        expires_at = grant.expires_at if ceiling is None else min(grant.expires_at, ceiling)
        rotated = replace(
            grant, durable=current.durable, max_expires_at=ceiling, expires_at=expires_at
        )
        try:
            self._capability_vault.admit(request, rotated)
        except ValueError:
            return  # broker tried to widen scope — keep the current lease (fail-closed)
        recorder.emit(
            "capability.rotated",
            turn_id=turn_id,
            parent_id=parent_id,
            data={
                "capability": capability,
                "old_lease_id": current.lease_id,
                "new_lease_id": rotated.lease_id,
                "expires_at": rotated.expires_at,
            },
        )

    def _check_run_boundary(self, deadline: float | None) -> None:
        if self.cancellation_token is not None and self.cancellation_token.requested:
            raise RunCancelled("run cancelled")
        if deadline is not None and time.time() >= deadline:
            raise RunTimeout("run exceeded max duration")
        # Run-level cancel (terminal) takes precedence over a turn-level interrupt (non-terminal).
        if self._interrupt_requested:
            raise TurnInterrupted("turn interrupted")

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
    """Sum every integer usage field across turns. The core three always exist; optional
    priced sub-counts (cache_read/cache_creation/reasoning/audio) accumulate too when the
    adapter reports them, so they reach metrics and the token-budget check."""
    for key, value in turn.usage.items():
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        total_usage[key] = total_usage.get(key, 0) + value


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
