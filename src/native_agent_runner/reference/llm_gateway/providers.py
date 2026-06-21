"""Reference LLM providers for the gateway — including a key-less offline provider.

The gateway's provider seam is a ``ProviderAdapterFactory`` (``LlmGatewayBackend
.provider_adapter_factory``). When none is supplied the gateway hard-defaults to
``OpenAIModelAdapter``, so a from-scratch local run needs a real OpenAI key. This module is the
LLM-side counterpart of ``reference/web_gateway/providers.py`` (``FakeWebProvider`` / ``--provider
fake``): :func:`offline_provider_factory` lets the gateway answer turns with **zero credentials**,
so the whole stack — chat, streaming, multi-turn — works offline for local dev and tests.

Reference example, not part of the supported surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from native_agent_runner.core.spec import ModelConfig
from native_agent_runner.providers.base import ModelAdapter, ModelRequest, ModelTurn
from native_agent_runner.reference._shared.tokens import TokenClaims


def _content_text(content: Any) -> str:
    """Best-effort plain text from a message ``content`` (str or content-part list)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (list, tuple)):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts).strip()
    return ""


def _latest_user_text(request: ModelRequest) -> str:
    """The newest user message — preferring the by-value log, then the instruction."""
    if request.messages:
        for message in reversed(request.messages):
            if message.get("role") == "user":
                text = _content_text(message.get("content"))
                if text:
                    return text
    return (request.instruction or "").strip()


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class EchoModelAdapter:
    """A zero-dependency offline model: it turns the latest user message into a final-text
    turn, so chat + streaming + multi-turn settle end to end with no network and no key.

    It never emits tool calls, so a tool-bound run simply gets a conversational reply rather
    than performing work — switch to a real provider to exercise the agentic tool path.
    """

    model: ModelConfig | None = None

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        user_text = _latest_user_text(request)
        if user_text:
            body = (
                "**Offline echo model** — no provider key is configured, so I can't do real "
                "reasoning or run tools yet.\n\n"
                f"You said:\n\n> {user_text}\n\n"
                "Configure a real provider (OpenAI key, or a CSP gateway) to get real answers."
            )
        else:
            body = (
                "I'm the offline echo model. Type a message and I'll repeat it back. "
                "Configure a real provider to get real answers."
            )
        in_tokens = _estimate_tokens(user_text)
        out_tokens = _estimate_tokens(body)
        return ModelTurn(
            final_text=body,
            usage={
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "total_tokens": in_tokens + out_tokens,
            },
        )


def offline_provider_factory(_claims: TokenClaims, config: ModelConfig) -> ModelAdapter:
    """A ``ProviderAdapterFactory`` for ``LlmGatewayBackend`` serving the offline echo model.

    Pass this as ``LlmGatewayBackend(provider_adapter_factory=offline_provider_factory)`` (or use
    ``native-agent llm-gateway serve --provider fake``) for a key-less gateway.
    """
    return EchoModelAdapter(model=config)
