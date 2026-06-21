from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, ClassVar

from native_agent_runner.core.spec import ModelConfig
from native_agent_runner.errors import ModelAdapterError
from native_agent_runner.providers._common import (
    build_reasoning_payload,
    normalize_usage,
)
from native_agent_runner.providers.base import (
    ModelRequest,
    ModelTurn,
    ToolCall,
    format_async_result_text,
)


@dataclass
class OpenAIModelAdapter:
    """Direct OpenAI adapter for local smoke tests.

    Container and CSP-integrated runs should use GatewayModelAdapter so provider
    credentials remain inside CSP backend infrastructure.
    """

    config: ModelConfig
    api_key: str | None = None
    allow_direct_provider_api: bool = False

    # Maps resolved base64 image blocks to Responses ``input_image`` items.
    supports_multimodal: ClassVar[bool] = True

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

        if request.messages is not None:
            # By-value: the full conversation travels as input; no server-side handle.
            input_items: list[dict[str, Any]] = []
            for message in request.messages:
                input_items.extend(_message_to_input_items(message))
            payload["input"] = input_items
        elif request.previous_turn_handle:
            payload["previous_response_id"] = request.previous_turn_handle
            input_items = []
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
        # A background/hosted task result delivered as a new user message.
        return {"role": "user", "content": format_async_result_text(observation.output)}
    return {
        "type": "function_call_output",
        "call_id": observation.call_id,
        "output": json.dumps(observation.output, ensure_ascii=False),
    }


def _user_content_items(content: list[Any]) -> list[dict[str, Any]]:
    """Map resolved by-value user parts to OpenAI Responses content items.

    ``content`` holds text part-dicts and neutral base64 media blocks (produced by the
    loop's wire-build). A base64 image becomes an ``input_image`` data-URL.
    """
    items: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            items.append({"type": "input_text", "text": str(part.get("text", ""))})
        elif part.get("type") == "image":
            source = part.get("source") or {}
            if source.get("type") == "base64":
                data_url = f"data:{source.get('media_type')};base64,{source.get('data')}"
                items.append({"type": "input_image", "image_url": data_url})
    return items


def _message_to_input_items(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate one provider-neutral by-value message into OpenAI Responses input items.
    An assistant turn with tool calls expands to an assistant text item (if any) plus a
    ``function_call`` item per call; a tool message is a ``function_call_output``."""
    role = message.get("role")
    if role == "user":
        content = message.get("content")
        if isinstance(content, list):
            # Multimodal: the loop resolved media to neutral base64 blocks. Map each part to
            # an OpenAI Responses content item (input_text / input_image data-URL).
            return [{"role": "user", "content": _user_content_items(content)}]
        return [{"role": "user", "content": content or ""}]
    if role == "tool":
        items = [
            {
                "type": "function_call_output",
                "call_id": message.get("call_id") or "",
                "output": json.dumps(message.get("content"), ensure_ascii=False),
            }
        ]
        # Images a tool returned cannot ride the tool/function output on OpenAI — deliver
        # them as a follow-up user message right after the tool result (the portable split).
        images = message.get("images")
        if isinstance(images, list):
            image_items = _user_content_items(images)
            if image_items:
                items.append({"role": "user", "content": image_items})
        return items
    if role == "assistant":
        items: list[dict[str, Any]] = []
        content = message.get("content") or ""
        if content:
            items.append({"role": "assistant", "content": content})
        for call in message.get("tool_calls") or []:
            items.append(
                {
                    "type": "function_call",
                    "call_id": call.get("id") or "",
                    "name": call.get("name") or "",
                    "arguments": json.dumps(call.get("arguments") or {}, ensure_ascii=False),
                }
            )
        return items
    return []


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
