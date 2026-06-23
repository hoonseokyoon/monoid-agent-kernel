from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
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
    ModelStreamChunk,
    ModelTurn,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    TurnComplete,
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
    # Identifies which provider's reasoning artifacts this adapter produces, so the loop tags
    # the captured reasoning block and replay only happens against a matching model.
    provider_name: ClassVar[str] = "openai"

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
            try:
                response = client.responses.create(**payload, timeout=config.timeout_s)
            except TypeError:
                response = client.responses.create(**payload)
        except ModelAdapterError:
            raise
        except Exception as exc:
            # Map provider API errors (e.g. a 400 for an unsupported reasoning effort) to a
            # classified ModelAdapterError so the gateway returns the real status (4xx, not a
            # generic 500) and the runner can treat it as recoverable. Never echo the raw body.
            raise _model_error_from_openai(exc) from exc
        data = response.model_dump() if hasattr(response, "model_dump") else _coerce_response(response)
        return _parse_response(data)

    async def astream_turn(self, request: ModelRequest) -> AsyncIterator[ModelStreamChunk]:
        """Stream a turn from the OpenAI Responses API as neutral ``ModelStreamChunk``s (text
        fragments, tool-call fragments, a terminal usage chunk). Async so the gateway's
        private-loop pump and the loop's async drive can consume it; the sync ``next_turn``
        path is unaffected. Provider errors map to a classified ``ModelAdapterError`` (no body
        leak), exactly like ``next_turn``."""
        if not self.allow_direct_provider_api and os.environ.get("NAR_ALLOW_DIRECT_PROVIDER_API") != "1":
            raise ModelAdapterError(
                "direct provider API access is disabled; use GatewayModelAdapter for container runs"
            )
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ModelAdapterError("openai package is not installed") from exc

        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ModelAdapterError("OPENAI_API_KEY is required for OpenAIModelAdapter")

        client = AsyncOpenAI(api_key=key)
        payload = self._payload(request)
        config = request.model or self.config
        try:
            try:
                stream = await client.responses.create(**payload, stream=True, timeout=config.timeout_s)
            except TypeError:
                stream = await client.responses.create(**payload, stream=True)
        except ModelAdapterError:
            raise
        except Exception as exc:
            raise _model_error_from_openai(exc) from exc

        final_data: dict[str, Any] = {}
        try:
            async for event in stream:
                etype = getattr(event, "type", "")
                if etype == "response.output_text.delta":
                    text = getattr(event, "delta", "") or ""
                    if text:
                        yield TextDelta(text)
                elif etype == "response.reasoning_summary_text.delta":
                    # Display-only reasoning summary fragment (DX-13b). Only present when the
                    # request asked for a summary (reasoning.summary != "off").
                    text = getattr(event, "delta", "") or ""
                    if text:
                        yield ReasoningDelta(text)
                elif etype == "response.output_item.added":
                    item = getattr(event, "item", None)
                    if item is not None and getattr(item, "type", "") == "function_call":
                        yield ToolCallDelta(
                            index=int(getattr(event, "output_index", 0) or 0),
                            id=str(getattr(item, "call_id", "") or getattr(item, "id", "") or "") or None,
                            name=str(getattr(item, "name", "") or "") or None,
                        )
                elif etype == "response.function_call_arguments.delta":
                    frag = getattr(event, "delta", "") or ""
                    if frag:
                        yield ToolCallDelta(
                            index=int(getattr(event, "output_index", 0) or 0),
                            arguments_fragment=frag,
                        )
                elif etype == "response.completed":
                    response = getattr(event, "response", None)
                    if response is not None and hasattr(response, "model_dump"):
                        final_data = response.model_dump()
        except ModelAdapterError:
            raise
        except Exception as exc:
            raise _model_error_from_openai(exc) from exc

        yield TurnComplete(
            response_id=final_data.get("id"),
            usage=normalize_usage(final_data.get("usage"), legacy_aliases=True),
            # encrypted_content lives only on the final response object, so reasoning items
            # are captured here (from response.completed) rather than the per-token deltas.
            reasoning=_capture_reasoning_items(final_data.get("output") or []),
        )

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        config = request.model or self.config
        payload: dict[str, Any] = {
            "model": config.model,
            "instructions": request.system_prompt,
            "tools": [_openai_tool_schema(tool) for tool in request.tools],
            # ZDR-faithful reasoning round-trip: don't persist server-side state, and ask for
            # the encrypted reasoning so it travels by-value in the message log (re-injected by
            # ``_message_to_input_items``). The engine never relies on ``previous_response_id``.
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        reasoning_payload = build_reasoning_payload(config.reasoning)
        if reasoning_payload:
            payload["reasoning"] = reasoning_payload

        if request.messages is not None:
            # By-value: the full conversation travels as input; no server-side handle. Reasoning
            # captured on assistant turns is re-injected verbatim within the active window when it
            # matches the current model (ZDR round-trip); see ``_reasoning_replay_flags``.
            input_items: list[dict[str, Any]] = []
            replay_flags = _reasoning_replay_flags(request.messages, config.model)
            for message, replay in zip(request.messages, replay_flags):
                input_items.extend(_message_to_input_items(message, replay_reasoning=replay))
            payload["input"] = input_items
        elif request.previous_turn_handle:
            # By-reference (non-ZDR): relies on server-side storage and is unsupported for
            # reasoning round-trip — the engine uses the by-value path above in production.
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


def _model_error_from_openai(exc: Exception) -> ModelAdapterError:
    """Classify an OpenAI SDK exception into a ModelAdapterError carrying the provider HTTP
    status, so downstream (gateway HTTP mapping, runner classification, core recoverability)
    can reason about it. Uses a synthetic, body-free message to avoid leaking prompt/PII."""
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    code = body.get("code") if isinstance(body, dict) else None
    if isinstance(status, int) and 400 <= status < 500:
        return ModelAdapterError(
            f"provider rejected the request (HTTP {status})",
            error_code="model_error",
            provider_error_code=str(code or ""),
            retryable=(status == 429),
            http_status=status,
        )
    if isinstance(status, int) and 500 <= status < 600:
        return ModelAdapterError(
            f"provider server error (HTTP {status})",
            error_code="model_error",
            provider_error_code=str(code or ""),
            retryable=True,
            http_status=status,
        )
    return ModelAdapterError("provider call failed", error_code="model_error")


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
    loop's wire-build). A base64 image becomes an ``input_image`` data-URL; a base64 document
    becomes an ``input_file`` with a filename.
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
        elif part.get("type") == "document":
            source = part.get("source") or {}
            if source.get("type") == "base64":
                data_url = f"data:{source.get('media_type')};base64,{source.get('data')}"
                items.append(
                    {
                        "type": "input_file",
                        "filename": str(part.get("filename") or "document.pdf"),
                        "file_data": data_url,
                    }
                )
    return items


def _message_to_input_items(
    message: dict[str, Any], *, replay_reasoning: bool = False
) -> list[dict[str, Any]]:
    """Translate one provider-neutral by-value message into OpenAI Responses input items.
    An assistant turn with tool calls expands to an assistant text item (if any) plus a
    ``function_call`` item per call; a tool message is a ``function_call_output``.

    When ``replay_reasoning`` is set and the assistant message carries a captured ``reasoning``
    block (see ``_capture_reasoning_items``), its verbatim item subsequence is emitted instead —
    preserving the reasoning→following-item adjacency OpenAI validates — and the reconstructed
    text/function_calls are suppressed to avoid duplication. Callers gate this flag per the
    active-window model-identity rule (``_reasoning_replay_flags``)."""
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
        # Media a tool returned cannot ride the tool/function output on OpenAI — deliver
        # it as a follow-up user message right after the tool result (the portable split).
        media = message.get("media")
        if isinstance(media, list):
            media_items = _user_content_items(media)
            if media_items:
                items.append({"role": "user", "content": media_items})
        return items
    if role == "assistant":
        reasoning = message.get("reasoning")
        if replay_reasoning and isinstance(reasoning, dict) and reasoning.get("items"):
            return [dict(item) for item in reasoning["items"]]
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


def _reasoning_replay_flags(
    messages: tuple[dict[str, Any], ...], current_model: str
) -> list[bool]:
    """Per-message decision of whether to replay its captured OpenAI reasoning verbatim.

    Two rules (see the DX-13a plan):
    - **Active window only**: reasoning is mandatory to round-trip only since the last ``user``
      message (the in-flight tool loop). Earlier reasoning is historical and droppable — OpenAI
      tolerates historical function_call pairs without their reasoning.
    - **All-or-nothing model identity**: ``config.model`` is re-read every step, so a hot-swap
      can land mid-loop. If any active-window reasoning block isn't ``openai`` at the current
      model, drop reasoning for the whole window so we never send a half-paired set (→ no 400).
    """
    last_user = -1
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user = index
    window_ok = True
    for message in messages[last_user + 1 :]:
        reasoning = message.get("reasoning")
        if isinstance(reasoning, dict) and reasoning.get("items"):
            if reasoning.get("provider") != "openai" or reasoning.get("model") != current_model:
                window_ok = False
                break
    return [index > last_user and window_ok for index in range(len(messages))]


def _capture_reasoning_items(output: list[Any]) -> tuple[dict[str, Any], ...]:
    """The verbatim ``reasoning``/``function_call``/``message`` output subsequence, in order.

    OpenAI pairs each reasoning item with the item that immediately follows it (a
    ``function_call`` or an assistant ``message``) and validates that adjacency on the next
    by-value request; dropping or reordering them yields a ``required following item`` 400.
    Capturing the exact subsequence verbatim — rather than reconstructing items from the parsed
    ``tool_calls``/``final_text`` — is the only construction that survives parallel/interleaved
    tool calls and reasoning→message pairings. The opaque payload (``encrypted_content`` etc.)
    is preserved; only the output-only ``status`` field is dropped, since the Responses *input*
    schema rejects it (``Unknown parameter: input[..].status``). Returns ``()`` when the turn
    carried no reasoning (non-reasoning models are untouched).
    """
    captured: list[dict[str, Any]] = []
    has_reasoning = False
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {"reasoning", "function_call", "message"}:
            captured.append({k: v for k, v in item.items() if k != "status"})
            if item_type == "reasoning":
                has_reasoning = True
    return tuple(captured) if has_reasoning else ()


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
        reasoning=_capture_reasoning_items(output),
    )


def _coerce_response(response: object) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    raise ModelAdapterError("unsupported OpenAI response object")
