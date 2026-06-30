"""Shared helpers for model adapters.

The OpenAI and gateway adapters build the same reasoning block and normalize
usage the same way; only their tool-schema shape genuinely differs. Keep the
common pieces here so the two adapters cannot drift.
"""

from __future__ import annotations

from typing import Any

from monoid_agent_kernel.core.spec import ReasoningConfig


def build_reasoning_payload(reasoning: ReasoningConfig) -> dict[str, Any]:
    """Reasoning block for a model request: ``{}`` when default/off, else effort/summary."""
    payload: dict[str, Any] = {}
    if reasoning.effort != "default":
        payload["effort"] = reasoning.effort
    if reasoning.summary != "off":
        payload["summary"] = reasoning.summary
    return payload


def text_from_message_content(content: Any) -> str:
    """Project a by-value message ``content`` down to plain text for text-only adapters.

    ``content`` is either a ``str`` (already text) or a list of part-dicts (the multimodal
    by-reference shape produced by ``content_part_to_json``). Non-text parts (image,
    document) are dropped here — a text-only wire keeps working even once the durable log
    carries multimodal parts. Multimodal adapters bypass this and map the parts instead.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        segments = [
            str(part.get("text", "")).strip()
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "text"
            and str(part.get("text", "")).strip()
        ]
        return "\n\n".join(segments)
    return ""


def project_message_to_text(message: dict[str, Any]) -> dict[str, Any]:
    """Return ``message`` with list ``content`` collapsed to text; pass ``str`` through.

    Used by text-only adapter send paths so a durable multimodal message (list content)
    never reaches a provider that cannot read it.
    """
    content = message.get("content")
    if isinstance(content, list):
        return {**message, "content": text_from_message_content(content)}
    return message


def _first_positive(*values: Any) -> int | None:
    """First value that coerces to a positive int, else ``None``."""
    for value in values:
        if value is None:
            continue
        try:
            n = int(value)
        except (TypeError, ValueError):
            continue
        if n > 0:
            return n
    return None


def normalize_usage(usage: dict[str, Any] | None, *, legacy_aliases: bool = False) -> dict[str, int]:
    """Coerce a provider usage dict to ``{input_tokens, output_tokens, total_tokens}`` plus
    optional priced sub-counts (``cache_read_tokens``, ``cache_creation_tokens``,
    ``reasoning_tokens``, ``audio_tokens``) included **only when present**.

    The sub-counts are folded from the various provider shapes — Anthropic's flat
    ``cache_*_input_tokens``, OpenAI's nested ``*_tokens_details``, Gemini's
    ``cachedContentTokenCount`` / ``thoughtsTokenCount``, and an already-normalized
    passthrough — so cache and reasoning tokens (priced differently) survive instead of
    being flattened away. ``legacy_aliases`` also accepts OpenAI's older ``prompt_tokens`` /
    ``completion_tokens`` names as fallbacks.
    """
    usage = usage or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if legacy_aliases:
        input_tokens = input_tokens or usage.get("prompt_tokens")
        output_tokens = output_tokens or usage.get("completion_tokens")
    normalized = {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }
    input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    audio = (int(input_details.get("audio_tokens") or 0) + int(output_details.get("audio_tokens") or 0)) or None
    details = {
        "cache_read_tokens": _first_positive(
            usage.get("cache_read_tokens"),
            usage.get("cache_read_input_tokens"),
            input_details.get("cached_tokens"),
            usage.get("cached_tokens"),
            usage.get("cachedContentTokenCount"),
        ),
        "cache_creation_tokens": _first_positive(
            usage.get("cache_creation_tokens"),
            usage.get("cache_creation_input_tokens"),
        ),
        "reasoning_tokens": _first_positive(
            usage.get("reasoning_tokens"),
            output_details.get("reasoning_tokens"),
            output_details.get("thinking_tokens"),
            usage.get("thoughtsTokenCount"),
        ),
        "audio_tokens": _first_positive(usage.get("audio_tokens"), audio),
    }
    for key, value in details.items():
        if value:
            normalized[key] = value
    return normalized
