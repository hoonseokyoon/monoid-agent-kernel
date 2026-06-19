from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from native_agent_runner.core.agents import (
    AgentDefinition,
    AgentRuntimeConfig,
    RuntimeConfigProvider,
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
from native_agent_runner.core.proposal_file import ProposalFileError, read_proposal_file_payload
from native_agent_runner.core.result import AgentRunResult, Suspension
from native_agent_runner.core.spec import (
    AgentRunSpec,
    ModelConfig,
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
from native_agent_runner.tools.builtin import builtin_tools
from native_agent_runner.web import WebGatewayClient
from native_agent_runner.workspace.paths import is_within

BackendRunState = Literal["queued", "running", "awaiting_input", "completed", "failed", "limited"]

# Sentinel enqueued to wake/stop a session worker blocked on its message queue.
_CLOSE_SESSION = object()
ModelAdapterFactory = Callable[[AgentRunSpec, str], ModelAdapter]

# Durable recovery descriptor (run.json) — what recover_runs needs to rebuild a parked run.
_RUN_META_SCHEMA_VERSION = "native-agent-runner.backend-run.v1"


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
    message_queue: queue.Queue[Any] = field(default_factory=queue.Queue, repr=False)


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


def _backend_builtin_tool_specs() -> tuple[Any, ...]:
    return tuple(builtin_tools(cast(Workspace, None)))


def _runtime_config_uses_web(config: AgentRuntimeConfig) -> bool:
    return any(binding.ref.tool_id.startswith("web.") for binding in config.tools)


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
    # How checkpoints are durably stored (backend owns HOW). Defaults to a local-fs
    # store under run_root; swap for a mounted-volume path or an object-store/DB store.
    checkpoint_store: CheckpointStore | None = None
    _records: dict[str, BackendRunRecord] = field(default_factory=dict, init=False, repr=False)
    _usage: dict[str, TenantUsage] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.run_root = self.run_root.resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        roots = tuple(root.resolve() for root in self.allowed_workspace_roots)
        if not roots:
            raise ValueError("at least one allowed workspace root is required")
        self.allowed_workspace_roots = roots
        self.allowed_apply_roots = tuple(root.resolve() for root in self.allowed_apply_roots)
        if self.checkpoint_store is None:
            self.checkpoint_store = LocalFsCheckpointStore(self.run_root)

    def submit_run(self, request: BackendRunRequest) -> BackendRunSubmission:
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
        tool_specs = _backend_builtin_tool_specs()
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
        thread = threading.Thread(
            target=self._run_worker,
            args=(run_id, request, workspace_root, llm_gateway_token, web_gateway_token),
            name=f"native-agent-run-{run_id[:8]}",
            daemon=True,
        )
        thread.start()
        return BackendRunSubmission(
            run_id=run_id,
            run_token=run_token,
            status="queued",
            run_dir=run_dir,
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
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        status_file = record.run_dir / "status.json"
        status_payload: dict[str, Any] | None = None
        if status_file.exists():
            status_payload = json.loads(status_file.read_text(encoding="utf-8"))
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
        # Wake a worker blocked on its message queue so it stops promptly.
        record.message_queue.put(_CLOSE_SESSION)
        return {
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            "status": record.status,
            "cancel_requested": True,
            "error": record.error,
            "error_code": record.error_code,
        }

    def send_message(self, run_id: str, token: str, content: str) -> dict[str, Any]:
        """Deliver a follow-up user message to a running multi-turn session. It is
        queued and consumed as the next user turn once the current turn settles."""
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        with self._lock:
            if record.status in {"completed", "failed", "limited"}:
                raise ValueError("cannot send a message to a terminal run")
        record.message_queue.put(str(content))
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
        validate_runtime_config(config, _backend_builtin_tool_specs())
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
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        output = record.run_dir / "proposal.tar"
        payload = export_package(record.run_dir, output)
        append_event_to_run(
            record.run_dir,
            "proposal.package.exported",
            data={"package_hash": payload["package_hash"], "package_path": str(output)},
        )
        return payload

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
        self._authorize_run(run_id, token)
        record = self._record(run_id)
        events_path = record.run_dir / "events.jsonl"
        events: list[dict[str, Any]] = []
        if events_path.exists():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if int(event.get("seq") or 0) >= from_seq:
                    events.append(event)
        return {"run_id": run_id, "events": events}

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
                status = str(event.data.get("status") or "completed")
                if status in {"completed", "failed", "limited"}:
                    record.status = status  # type: ignore[assignment]
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

    def _drive_session(self, run_id: str, request: BackendRunRequest, loop: AgentLoop) -> AgentRunResult:
        """Cold-start driver: open the run, take the first turn, then hand off to the
        shared open-session loop (also used by checkpoint recovery)."""
        record = self._record(run_id)
        loop.open()
        try:
            suspension = loop.run_until_suspended(request.instruction)
        except NativeAgentError:
            # Bootstrap failed (terminal session already recorded); just finalize.
            return loop.close()
        return self._drive_open_session(record, request, loop, suspension, started=time.time(), turns=1)

    def _drive_open_session(
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
        complete or block on the message queue for the next user turn — until idle/
        lifetime/turn limits, cancel, or close. Shared by cold start and recovery; each
        park point durably checkpoints (loop state + backend message queue)."""
        while True:
            if suspension.reason in {"terminal", "limited"}:
                break
            if suspension.reason == "awaiting_tasks":
                self._persist_run_checkpoint(record)
                ready = loop.wait_for_pending_tasks(self.task_wait_poll_s)
                if self._session_should_stop(record, started, turns):
                    break
                if ready or not loop.has_pending_tasks():
                    # A task was delivered (or none remain): resume the pump.
                    suspension = loop.run_until_suspended(None)
                # else: tasks still pending after the poll window -> keep waiting.
                continue
            # settled. One-shot runs close here; multi-turn awaits the next message.
            if not request.multi_turn:
                break
            if self._session_should_stop(record, started, turns):
                break
            loop.await_user_input()
            self._persist_run_checkpoint(record)
            try:
                message = record.message_queue.get(timeout=self.idle_timeout_s)
            except queue.Empty:
                break  # idle timeout
            if message is _CLOSE_SESSION:
                break
            turns += 1
            suspension = loop.run_until_suspended(message)
        return loop.close()

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
        # Peek (don't drain) the residual queue; consumed messages are already reflected
        # in the loop's turn handle / pending input.
        with record.message_queue.mutex:
            residual = [
                message
                for message in list(record.message_queue.queue)
                if isinstance(message, str)
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

    def _run_worker(
        self,
        run_id: str,
        request: BackendRunRequest,
        workspace_root: Path,
        llm_gateway_token: str,
        web_gateway_token: str,
    ) -> None:
        spec = self._run_spec_for_request(run_id, request, workspace_root)
        try:
            runtime_config = self.current_runtime_config(run_id)
            adapter = self._build_model_adapter(
                spec,
                llm_gateway_token,
                runtime_config.model if runtime_config is not None else None,
            )
            loop = AgentLoop(
                spec=spec,
                model_adapter=adapter,
                event_sinks=(BackendRunStateSink(self, run_id),),
                permission_policy=request.permission_policy,
                cancellation_token=self._record(run_id).cancellation_token,
                shell_approval_provider=None,
                web_gateway_client=self._web_gateway_client(web_gateway_token),
                runtime_config_provider=BackendRuntimeConfigProvider(self, run_id),
                checkpoint_store=self.checkpoint_store,
            )
            with self._lock:
                self._records[run_id].loop = loop
            # Persist the recovery metadata before the first turn so a crash at any park
            # point can be resumed (the checkpoint itself is written by the driver).
            self._write_run_meta(self._record(run_id), request)
            result = self._drive_session(run_id, request, loop)
            self._record_run_result(run_id, result)
        except Exception as exc:
            self._record_run_failure(run_id, exc)

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
        with self._lock:
            record = self._records[run_id]
            record.status = "failed"
            record.error = str(exc)
            record.error_code = getattr(exc, "error_code", "internal_error")
            record.finished_at = time.time()

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
            meta = _read_run_meta(run_dir)
            if meta is None:
                continue
            try:
                self._resume_from_checkpoint(stored, meta)
            except Exception:
                continue
            recovered.append(run_id)
        return recovered

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
            event_sinks=(BackendRunStateSink(self, run_id),),
            permission_policy=request.permission_policy,
            cancellation_token=record.cancellation_token,
            shell_approval_provider=None,
            web_gateway_client=self._web_gateway_client(web_gateway_token),
            runtime_config_provider=BackendRuntimeConfigProvider(self, run_id),
            checkpoint_store=self.checkpoint_store,
        )
        # The base workspace is re-provisioned by the deployment (re-mount/re-clone);
        # restore re-applies the agent's delta from the checkpoint's content blobs.
        loop.restore(checkpoint, blobs=stored.blob)
        with self._lock:
            record.loop = loop
        for message in checkpoint.queued_messages:
            record.message_queue.put(message)
        thread = threading.Thread(
            target=self._run_recovered_worker,
            args=(run_id, request, loop),
            name=f"native-agent-recover-{run_id[:8]}",
            daemon=True,
        )
        thread.start()

    def _run_recovered_worker(self, run_id: str, request: BackendRunRequest, loop: AgentLoop) -> None:
        record = self._record(run_id)
        try:
            # Derive the starting park from the restored loop: tasks still pending -> a
            # hosted-task wait; otherwise a settled park awaiting the next user message.
            if loop.has_pending_tasks():
                suspension = Suspension(reason="awaiting_tasks", status="running", has_external=True)
            else:
                suspension = Suspension(reason="settled", status="completed")
            result = self._drive_open_session(
                record, request, loop, suspension, started=time.time(), turns=1
            )
            self._record_run_result(run_id, result)
        except Exception as exc:
            self._record_run_failure(run_id, exc)

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
        if not request.instruction.strip():
            raise ValueError("instruction is required")
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
        try:
            claims = self.token_manager.verify(
                token,
                kind="run_access",
                audience="native-agent-runner.backend",
                run_id=run_id,
            )
        except TokenError as exc:
            raise PermissionDenied(str(exc)) from exc
        record = self._record(run_id)
        if claims.tenant_id != record.tenant_id or claims.user_id != record.user_id:
            raise PermissionDenied("token subject mismatch")

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
