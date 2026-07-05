from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core.agents import AgentRuntimeConfig, validate_runtime_config
from monoid_agent_kernel.core.checkpoint import CheckpointStore
from monoid_agent_kernel.core.durable_metadata import (
    RUN_METADATA_SCHEMA_VERSION,
    DurableMetadataCommitter,
)
from monoid_agent_kernel.core.lifecycle import SessionState
from monoid_agent_kernel.errors import PermissionDenied
from monoid_agent_kernel.identifiers import BACKEND_AUDIENCE
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.backend.run_types import (
    BackendRunRecord,
    BackendRunRequest,
    BackendRunSubmission,
    _PreparedRun,
)
from monoid_agent_kernel.workspace.paths import is_within


def runtime_config_uses_web(config: AgentRuntimeConfig) -> bool:
    return any(binding.ref.tool_id.startswith("web.") for binding in config.tools)


@dataclass(frozen=True)
class RunPreparationContext:
    run_root_provider: Callable[[], Path]
    allowed_workspace_roots_provider: Callable[[], Sequence[Path]]
    token_manager_provider: Callable[[], TokenManager]
    run_token_ttl_s_provider: Callable[[], float]
    llm_gateway_token_ttl_s_provider: Callable[[], float]
    web_gateway_token_ttl_s_provider: Callable[[], float]
    web_gateway_url_provider: Callable[[], str]
    builtin_tool_specs_provider: Callable[[], tuple[Any, ...]]
    checkpoint_store_provider: Callable[[], CheckpointStore]
    register_record: Callable[[BackendRunRecord], None]
    now: Callable[[], float]


class RunPreparationService:
    """Reference backend run admission and durable descriptor materialization."""

    def __init__(self, context: RunPreparationContext) -> None:
        self._context = context

    def prepare(self, request: BackendRunRequest) -> _PreparedRun:
        self.validate_request(request)
        workspace_root = request.workspace_root.resolve()
        self.check_workspace_allowed(workspace_root)
        run_id = uuid.uuid4().hex
        run_dir = self._context.run_root_provider() / run_id
        token_manager = self._context.token_manager_provider()
        run_token = token_manager.issue(
            kind="run_access",
            audience=BACKEND_AUDIENCE,
            run_id=run_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            ttl_s=self._context.run_token_ttl_s_provider(),
        )
        initial_runtime_config = request.runtime_config
        runtime_config_issuer = "submit_run"
        runtime_config_reason = "initial runtime config"
        if initial_runtime_config is None and request.agent_definition is not None:
            initial_runtime_config = AgentRuntimeConfig.from_definition(request.agent_definition)
            runtime_config_reason = "initial agent definition"
        elif initial_runtime_config is None:
            raise ValueError("agent_definition or runtime_config is required")
        validate_runtime_config(initial_runtime_config, self._context.builtin_tool_specs_provider())
        llm_gateway_token = token_manager.issue(
            kind="llm_gateway",
            audience="csp.llm-gateway",
            run_id=run_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            ttl_s=self._context.llm_gateway_token_ttl_s_provider(),
            metadata={"agent_config_hash": initial_runtime_config.config_hash},
        )
        web_gateway_token = ""
        if runtime_config_uses_web(initial_runtime_config):
            if not self._context.web_gateway_url_provider():
                raise ValueError("web_gateway_url is required when runtime config binds web tools")
            web_gateway_token = token_manager.issue(
                kind="web_gateway",
                audience="csp.web-gateway",
                run_id=run_id,
                tenant_id=request.tenant_id,
                user_id=request.user_id,
                ttl_s=self._context.web_gateway_token_ttl_s_provider(),
                metadata={"agent_config_hash": initial_runtime_config.config_hash},
            )
        created_at = self._context.now()
        record = BackendRunRecord(
            run_id=run_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            workspace_root=workspace_root,
            run_dir=run_dir,
            state=SessionState.CREATED,
            terminal=False,
            created_at=created_at,
            run_token_sha256=TokenManager.token_sha256(run_token),
            llm_gateway_token_sha256=TokenManager.token_sha256(llm_gateway_token),
            web_gateway_token_sha256=TokenManager.token_sha256(web_gateway_token)
            if web_gateway_token
            else "",
            runtime_config=initial_runtime_config,
            runtime_config_issuer=runtime_config_issuer,
            runtime_config_reason=runtime_config_reason,
            runtime_config_committed_at=created_at,
        )
        self.write_run_meta(record, request)
        self._context.register_record(record)
        return _PreparedRun(
            run_id=run_id,
            record=record,
            workspace_root=workspace_root,
            run_token=run_token,
            llm_gateway_token=llm_gateway_token,
            web_gateway_token=web_gateway_token,
        )

    def submission_for(self, prepared: _PreparedRun) -> BackendRunSubmission:
        run_id = prepared.run_id
        return BackendRunSubmission(
            run_id=run_id,
            run_token=prepared.run_token,
            state=prepared.record.state,
            terminal=prepared.record.terminal,
            run_dir=prepared.record.run_dir,
            status_url=f"/v1/runs/{run_id}/status",
            result_url=f"/v1/runs/{run_id}/result",
            events_url=f"/v1/runs/{run_id}/events",
            proposal_url=f"/v1/runs/{run_id}/proposal",
        )

    def validate_request(self, request: BackendRunRequest) -> None:
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

    def check_workspace_allowed(self, workspace_root: Path) -> None:
        if not workspace_root.exists() or not workspace_root.is_dir():
            raise ValueError(f"workspace root does not exist: {workspace_root}")
        if not any(is_within(root, workspace_root) for root in self._context.allowed_workspace_roots_provider()):
            raise PermissionDenied(f"workspace root is outside allowed roots: {workspace_root}")

    def write_run_meta(self, record: BackendRunRecord, request: BackendRunRequest) -> None:
        """Write the initial run.json durable recovery descriptor."""
        config = record.runtime_config
        committed_at = record.runtime_config_committed_at or self._context.now()
        meta = {
            "schema_version": RUN_METADATA_SCHEMA_VERSION,
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            "user_id": record.user_id,
            "workspace_root": str(record.workspace_root),
            "mode": request.mode,
            "workspace_backend": request.workspace_backend,
            "multi_turn": request.multi_turn,
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
        DurableMetadataCommitter(self._context.checkpoint_store_provider()).write_initial_metadata(
            record.run_dir,
            record.run_id,
            meta,
        )
