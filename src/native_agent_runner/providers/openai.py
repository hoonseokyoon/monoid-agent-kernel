from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from native_agent_runner.core.spec import ModelConfig
from native_agent_runner.errors import ModelAdapterError
from native_agent_runner.providers._common import build_reasoning_payload, normalize_usage
from native_agent_runner.providers.base import ModelRequest, ModelTurn, ToolCall


@dataclass
class OpenAIModelAdapter:
    """Direct OpenAI adapter for local smoke tests.

    Container and CSP-integrated runs should use GatewayModelAdapter so provider
    credentials remain inside CSP backend infrastructure.
    """

    config: ModelConfig
    api_key: str | None = None
    allow_direct_provider_api: bool = False

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        if not self.allow_direct_provider_api and os.environ.get("NAR_ALLOW_DIRECT_PROVIDER_API") != "1":
            raise ModelAdapterError(
                "direct provider API access is disabled; use GatewayModelAdapter for container runs"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelAdapterError("openai package is not installed") from exc

        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ModelAdapterError("OPENAI_API_KEY is required for OpenAIModelAdapter")

        client = OpenAI(api_key=key)
        payload = self._payload(request)
        config = request.model or self.config
        try:
            response = client.responses.create(**payload, timeout=config.timeout_s)
        except TypeError:
            response = client.responses.create(**payload)
        data = response.model_dump() if hasattr(response, "model_dump") else _coerce_response(response)
        return _parse_response(data)

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        config = request.model or self.config
        payload: dict[str, Any] = {
            "model": config.model,
            "instructions": request.system_prompt,
            "tools": [_openai_tool_schema(tool) for tool in request.tools],
        }
        reasoning_payload = build_reasoning_payload(config.reasoning)
        if reasoning_payload:
            payload["reasoning"] = reasoning_payload

        if request.previous_turn_handle:
            payload["previous_response_id"] = request.previous_turn_handle
            input_items: list[dict[str, Any]] = []
            # Third shape: a new user message on top of an existing continuation handle.
            if request.instruction:
                input_items.append({"role": "user", "content": request.instruction})
            input_items.extend(_observation_input_item(observation) for observation in request.observations)
            payload["input"] = input_items
        else:
            payload["input"] = [{"role": "user", "content": request.instruction or ""}]
        return payload


def _openai_tool_schema(tool: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.exported_name,
        "description": tool.description,
        "parameters": tool.input_schema,
    }


def _observation_input_item(observation: Any) -> dict[str, Any]:
    if observation.is_background:
        return {
            "role": "user",
            "content": (
                "Background shell job completed. Treat this as the result of the previously "
                f"started job:\n{json.dumps(observation.output, ensure_ascii=False)}"
            ),
        }
    return {
        "type": "function_call_output",
        "call_id": observation.call_id,
        "output": json.dumps(observation.output, ensure_ascii=False),
    }


def _parse_response(data: dict[str, Any]) -> ModelTurn:
    output = data.get("output") or []
    tool_calls: list[ToolCall] = []
    text_parts: list[str] = []
    for item in output:
        item_type = item.get("type")
        if item_type == "function_call":
            args_raw = item.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
            except json.JSONDecodeError as exc:
                raise ModelAdapterError(f"invalid function_call arguments for {item.get('name')}") from exc
            tool_calls.append(
                ToolCall(
                    id=str(item.get("call_id") or item.get("id") or ""),
                    name=str(item.get("name") or ""),
                    arguments=args,
                )
            )
        elif item_type == "message":
            for part in item.get("content") or []:
                if part.get("type") in {"output_text", "text"}:
                    text_parts.append(str(part.get("text") or ""))
        elif item_type in {"output_text", "text"}:
            text_parts.append(str(item.get("text") or ""))

    usage_out = normalize_usage(data.get("usage"), legacy_aliases=True)
    return ModelTurn(
        response_id=data.get("id"),
        final_text="".join(text_parts).strip() or None,
        tool_calls=tuple(tool_calls),
        usage=usage_out,
        raw=data,
    )


def _coerce_response(response: object) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    raise ModelAdapterError("unsupported OpenAI response object")
