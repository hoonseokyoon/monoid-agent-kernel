from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from native_agent_runner.reference._shared.tokens import TokenClaims, TokenError, TokenManager
from native_agent_runner.core.spec import ModelConfig, ReasoningConfig
from native_agent_runner.errors import ModelAdapterError, PermissionDenied
from native_agent_runner.providers.base import ModelAdapter, ModelRequest, ModelTurn, ToolObservation
from native_agent_runner.providers.openai import OpenAIModelAdapter
from native_agent_runner.tools.base import ToolResult, ToolSpec

ProviderAdapterFactory = Callable[[TokenClaims, ModelConfig], ModelAdapter]


@dataclass(frozen=True)
class LlmGatewayTurnRequest:
    protocol: str
    model: str
    system_prompt: str
    tools: tuple[ToolSpec, ...]
    reasoning: ReasoningConfig
    instruction: str = ""
    previous_turn_handle: str | None = None
    observations: tuple[ToolObservation, ...] = ()


@dataclass
class LlmGatewayTurnRecord:
    turn_handle: str
    provider_response_id: str | None
    run_id: str
    tenant_id: str
    user_id: str
    model: str
    created_at: float


@dataclass
class LlmGatewayUsage:
    tenant_id: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: dict[str, int]) -> None:
        self.calls += 1
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or 0)
        self.total_tokens += int(usage.get("total_tokens") or 0)

    def to_json(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LlmGatewayBackend:
    token_manager: TokenManager
    provider_adapter_factory: ProviderAdapterFactory | None = None
    _turns: dict[str, LlmGatewayTurnRecord] = field(default_factory=dict, init=False, repr=False)
    _usage: dict[str, LlmGatewayUsage] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def handle_turn(self, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        claims = self._authorize(token)
        request = _parse_turn_request(payload)
        self._validate_request_against_claims(request, claims)
        provider_previous_response_id = self._provider_previous_response_id(request, claims)
        adapter = self._build_adapter(claims, request)
        turn = adapter.next_turn(
            ModelRequest(
                instruction=request.instruction,
                system_prompt=request.system_prompt,
                tools=request.tools,
                previous_response_id=provider_previous_response_id,
                observations=request.observations,
            )
        )
        turn_handle = self._record_turn(claims, request, turn)
        with self._lock:
            self._usage.setdefault(claims.tenant_id, LlmGatewayUsage(claims.tenant_id)).add(turn.usage)
        return {
            "protocol": "native-agent-runner.llm-turn-result.v1",
            "turn_handle": turn_handle,
            "final_text": turn.final_text,
            "tool_calls": [
                {"call_id": call.id, "name": call.name, "arguments": call.arguments}
                for call in turn.tool_calls
            ],
            "usage": turn.usage,
        }

    def tenant_usage(self, tenant_id: str) -> dict[str, Any]:
        with self._lock:
            usage = self._usage.get(tenant_id) or LlmGatewayUsage(tenant_id)
            return usage.to_json()

    def _authorize(self, token: str) -> TokenClaims:
        try:
            return self.token_manager.verify(
                token,
                kind="llm_gateway",
                audience="csp.llm-gateway",
            )
        except TokenError as exc:
            raise PermissionDenied(str(exc)) from exc

    def _validate_request_against_claims(
        self,
        request: LlmGatewayTurnRequest,
        claims: TokenClaims,
    ) -> None:
        allowed_model = claims.metadata.get("model")
        if allowed_model is not None and request.model != allowed_model:
            raise PermissionDenied("model is not allowed by llm_gateway token")
        allowed_effort = claims.metadata.get("reasoning_effort")
        if allowed_effort is not None and request.reasoning.effort != allowed_effort:
            raise PermissionDenied("reasoning effort is not allowed by llm_gateway token")

    def _provider_previous_response_id(
        self,
        request: LlmGatewayTurnRequest,
        claims: TokenClaims,
    ) -> str | None:
        if request.previous_turn_handle is None:
            return None
        with self._lock:
            record = self._turns.get(request.previous_turn_handle)
        if record is None:
            raise ModelAdapterError("unknown previous_turn_handle")
        if record.run_id != claims.run_id or record.tenant_id != claims.tenant_id:
            raise PermissionDenied("previous_turn_handle does not belong to this run")
        return record.provider_response_id

    def _build_adapter(
        self,
        claims: TokenClaims,
        request: LlmGatewayTurnRequest,
    ) -> ModelAdapter:
        config = ModelConfig(
            provider="openai",
            model=request.model,
            reasoning=request.reasoning,
        )
        if self.provider_adapter_factory is not None:
            return self.provider_adapter_factory(claims, config)
        return OpenAIModelAdapter(config, allow_direct_provider_api=True)

    def _record_turn(
        self,
        claims: TokenClaims,
        request: LlmGatewayTurnRequest,
        turn: ModelTurn,
    ) -> str:
        turn_handle = f"turn_{uuid.uuid4().hex}"
        with self._lock:
            self._turns[turn_handle] = LlmGatewayTurnRecord(
                turn_handle=turn_handle,
                provider_response_id=turn.response_id,
                run_id=claims.run_id,
                tenant_id=claims.tenant_id,
                user_id=claims.user_id,
                model=request.model,
                created_at=time.time(),
            )
        return turn_handle


def _parse_turn_request(payload: dict[str, Any]) -> LlmGatewayTurnRequest:
    if payload.get("protocol") != "native-agent-runner.llm-turn.v1":
        raise ValueError("unsupported LLM gateway protocol")
    reasoning_raw = dict(payload.get("reasoning") or {})
    previous_turn_handle = payload.get("previous_turn_handle")
    observations = tuple(_parse_observation(item) for item in payload.get("observations") or ())
    instruction = str(payload.get("instruction") or "")
    if previous_turn_handle is None and not instruction.strip():
        raise ValueError("instruction is required for the first LLM turn")
    return LlmGatewayTurnRequest(
        protocol="native-agent-runner.llm-turn.v1",
        model=str(payload["model"]),
        system_prompt=str(payload["system_prompt"]),
        tools=tuple(_parse_tool(item) for item in payload.get("tools") or ()),
        reasoning=ReasoningConfig(
            effort=reasoning_raw.get("effort", "medium"),
            summary=reasoning_raw.get("summary", "off"),
        ),
        instruction=instruction,
        previous_turn_handle=str(previous_turn_handle) if previous_turn_handle else None,
        observations=observations,
    )


def _parse_tool(raw: dict[str, Any]) -> ToolSpec:
    def handler(_context, _args):
        return ToolResult(ok=False, error="gateway tool proxy cannot execute tools")

    tool_id = str(raw.get("id") or raw.get("name") or "")
    return ToolSpec(
        id=tool_id,
        provider_name=str(raw.get("name") or tool_id.replace(".", "_")),
        description=str(raw.get("description") or ""),
        input_schema=dict(raw.get("input_schema") or raw.get("parameters") or {}),
        capability=str(raw.get("capability") or "unknown"),
        side_effect=str(raw.get("side_effect") or "read"),  # type: ignore[arg-type]
        handler=handler,
    )


def _parse_observation(raw: dict[str, Any]) -> ToolObservation:
    return ToolObservation(
        call_id=str(raw["call_id"]),
        tool_name=str(raw.get("tool_name") or ""),
        output=dict(raw.get("output") or {}),
    )
