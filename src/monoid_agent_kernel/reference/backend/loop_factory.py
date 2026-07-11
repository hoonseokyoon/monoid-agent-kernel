from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core.agents import AgentRuntimeConfig, RuntimeConfigProvider
from monoid_agent_kernel.core.capability import CapabilityBroker
from monoid_agent_kernel.core.checkpoint import CheckpointStore, RunCheckpoint
from monoid_agent_kernel.core.context import ContextProvider
from monoid_agent_kernel.core.events import AgentEvent, EventSink
from monoid_agent_kernel.core.outbox import OutboxSender
from monoid_agent_kernel.core.output_validator import OutputValidator
from monoid_agent_kernel.core.spec import AgentRunSpec, ModelConfig, RunLimits
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import AsyncModelAdapter, ModelAdapter
from monoid_agent_kernel.providers.gateway import GatewayModelAdapter
from monoid_agent_kernel.reference._shared.tokens import TokenKind, TokenManager
from monoid_agent_kernel.reference.backend.ports import MutableRunRecordPort, RunRequestPort
from monoid_agent_kernel.reference.backend.run_state import BackendRunStateSink
from monoid_agent_kernel.tools.base import ToolProvider
from monoid_agent_kernel.web import WebGatewayClient

ModelAdapterFactory = Callable[[AgentRunSpec, str], ModelAdapter | AsyncModelAdapter]


@dataclass(frozen=True)
class BackendLoopBuild:
    spec: AgentRunSpec
    model_adapter: ModelAdapter | AsyncModelAdapter
    web_gateway_client: WebGatewayClient | None
    runtime_config_provider: RuntimeConfigProvider
    capability_broker: CapabilityBroker | None
    outbox_sender: OutboxSender | None
    loop: AgentLoop


@dataclass(frozen=True)
class BackendLoopFactoryContext:
    run_root_provider: Callable[[], Path]
    llm_gateway_url_provider: Callable[[], str]
    web_gateway_url_provider: Callable[[], str | None]
    model_adapter_factory_provider: Callable[[], ModelAdapterFactory | None]
    token_manager_provider: Callable[[], TokenManager]
    llm_gateway_token_ttl_s_provider: Callable[[], int]
    checkpoint_store_provider: Callable[[], CheckpointStore | None]
    emit_output_deltas_provider: Callable[[], bool]
    extra_event_sink_factories_provider: Callable[[], tuple[Callable[[], EventSink], ...]]
    subagent_definitions_provider: Callable[[], Mapping[str, Any] | None]
    tool_providers_provider: Callable[[], tuple[ToolProvider, ...]]
    context_providers_provider: Callable[[], tuple[ContextProvider, ...]]
    output_validators_provider: Callable[[], tuple[OutputValidator, ...]]
    capability_broker_factory_provider: Callable[[], Callable[[RunRequestPort], CapabilityBroker | None] | None]
    outbox_sender_factory_provider: Callable[[], Callable[[RunRequestPort], OutboxSender | None] | None]
    current_runtime_config: Callable[[str], AgentRuntimeConfig | None]
    record: Callable[[str], MutableRunRecordPort]
    record_event: Callable[[str, AgentEvent], None]
    persist_checkpoint_payload: Callable[[MutableRunRecordPort, RunCheckpoint, Mapping[str, bytes]], None]


class BackendRuntimeConfigProvider(RuntimeConfigProvider):
    def __init__(
        self,
        current_runtime_config: Callable[[str], AgentRuntimeConfig | None],
        run_id: str,
    ) -> None:
        self._current_runtime_config = current_runtime_config
        self._run_id = run_id

    def current_config(self, run_id: str) -> AgentRuntimeConfig | None:
        del run_id
        return self._current_runtime_config(self._run_id)


@dataclass
class _GatewayTokenSource:
    """Callable gateway token source that re-mints shortly before expiry for long runs."""

    token_manager: TokenManager
    kind: TokenKind
    audience: str
    run_id: str
    tenant_id: str
    user_id: str
    ttl_s: int
    metadata: dict[str, Any] = field(default_factory=dict)
    refresh_skew_s: int = 300
    _token: str = ""
    _expires_at: float = 0.0

    def __call__(self) -> str:
        now = time.time()
        skew = min(self.refresh_skew_s, self.ttl_s // 2)
        if not self._token or now >= self._expires_at - skew:
            self._token = self.token_manager.issue(
                kind=self.kind,
                audience=self.audience,
                run_id=self.run_id,
                tenant_id=self.tenant_id,
                user_id=self.user_id,
                ttl_s=self.ttl_s,
                metadata=dict(self.metadata),
            )
            self._expires_at = now + self.ttl_s
        return self._token


class BackendLoopFactory:
    """Builds AgentLoop instances for the Reference backend facade."""

    def __init__(self, context: BackendLoopFactoryContext) -> None:
        self._context = context

    def build(
        self,
        run_id: str,
        request: RunRequestPort,
        workspace_root: Path,
        llm_gateway_token: str,
        web_gateway_token: str,
        *,
        include_outbox_sender: bool = True,
    ) -> BackendLoopBuild:
        spec = self.run_spec_for_request(run_id, request, workspace_root)
        runtime_config = self._context.current_runtime_config(run_id)
        model_adapter = self.build_model_adapter(
            spec,
            llm_gateway_token,
            runtime_config.model if runtime_config is not None else None,
            token_provider=self.llm_token_source(run_id, request, runtime_config),
        )
        web_gateway_client = self.web_gateway_client(web_gateway_token)
        runtime_config_provider = BackendRuntimeConfigProvider(
            self._context.current_runtime_config,
            run_id,
        )
        capability_broker = self.capability_broker_for(request)
        outbox_sender = self.outbox_sender_for(request) if include_outbox_sender else None
        record = self._context.record(run_id)
        loop = AgentLoop(
            spec=spec,
            model_adapter=model_adapter,
            event_sinks=(
                BackendRunStateSink(self._context.record_event, run_id),
                *(make() for make in self._context.extra_event_sink_factories_provider()),
            ),
            permission_policy=request.permission_policy,
            cancellation_token=record.cancellation_token,
            shell_approval_provider=None,
            web_gateway_client=web_gateway_client,
            runtime_config_provider=runtime_config_provider,
            checkpoint_store=self._context.checkpoint_store_provider(),
            emit_output_deltas=self._context.emit_output_deltas_provider(),
            subagent_definitions=self._context.subagent_definitions_provider(),
            tool_providers=self._context.tool_providers_provider(),
            context_providers=self._context.context_providers_provider(),
            output_validators=self._context.output_validators_provider(),
            capability_broker=capability_broker,
            checkpoint_persist_callback=lambda checkpoint, blobs: self._context.persist_checkpoint_payload(
                self._context.record(run_id),
                checkpoint,
                blobs,
            ),
        )
        return BackendLoopBuild(
            spec=spec,
            model_adapter=model_adapter,
            web_gateway_client=web_gateway_client,
            runtime_config_provider=runtime_config_provider,
            capability_broker=capability_broker,
            outbox_sender=outbox_sender,
            loop=loop,
        )

    def run_spec_for_request(
        self,
        run_id: str,
        request: RunRequestPort,
        workspace_root: Path,
    ) -> AgentRunSpec:
        return AgentRunSpec(
            workspace_root=workspace_root,
            run_root=self._context.run_root_provider(),
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

    def capability_broker_for(self, request: RunRequestPort) -> CapabilityBroker | None:
        factory = self._context.capability_broker_factory_provider()
        if factory is None:
            return None
        return factory(request)

    def outbox_sender_for(self, request: RunRequestPort) -> OutboxSender | None:
        factory = self._context.outbox_sender_factory_provider()
        if factory is None:
            return None
        return factory(request)

    def llm_token_source(
        self,
        run_id: str,
        request: RunRequestPort,
        runtime_config: AgentRuntimeConfig | None,
    ) -> _GatewayTokenSource:
        return _GatewayTokenSource(
            token_manager=self._context.token_manager_provider(),
            kind="llm_gateway",
            audience="csp.llm-gateway",
            run_id=run_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            ttl_s=self._context.llm_gateway_token_ttl_s_provider(),
            metadata={"agent_config_hash": runtime_config.config_hash} if runtime_config is not None else {},
        )

    def build_model_adapter(
        self,
        spec: AgentRunSpec,
        llm_gateway_token: str,
        model_config: ModelConfig | None,
        token_provider: Callable[[], str | None] | None = None,
    ) -> ModelAdapter:
        factory = self._context.model_adapter_factory_provider()
        if factory is not None:
            return factory(spec, llm_gateway_token)
        return GatewayModelAdapter(
            model_config or ModelConfig(),
            gateway_url=self._context.llm_gateway_url_provider(),
            token=llm_gateway_token,
            token_provider=token_provider,
        )

    def web_gateway_client(self, token: str) -> WebGatewayClient | None:
        gateway_url = self._context.web_gateway_url_provider()
        if not token or not gateway_url:
            return None
        return WebGatewayClient(gateway_url, token=token)
