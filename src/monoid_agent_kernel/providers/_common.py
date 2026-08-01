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


def _usage_object(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"model usage {field_name} must be an object")
    return value


def _usage_count(payload: dict[str, Any], key: str) -> int | None:
    if key not in payload:
        return None
    value = payload[key]
    if type(value) is not int or value < 0:
        raise ValueError(f"model usage {key} must be a non-negative integer")
    return value


def _first_count(*candidates: tuple[dict[str, Any], str]) -> int | None:
    for payload, key in candidates:
        value = _usage_count(payload, key)
        if value is not None:
            return value
    return None


def normalize_usage(
    usage: dict[str, Any] | None, *, legacy_aliases: bool = False
) -> dict[str, int]:
    """Validate provider usage as ``{input_tokens, output_tokens, total_tokens}`` plus
    optional priced sub-counts (``cache_read_tokens``, ``cache_creation_tokens``,
    ``reasoning_tokens``, ``audio_tokens``) included **only when present**.

    The sub-counts are folded from the various provider shapes — Anthropic's flat
    ``cache_*_input_tokens``, OpenAI's nested ``*_tokens_details``, Gemini's
    ``cachedContentTokenCount`` / ``thoughtsTokenCount``, and an already-normalized
    passthrough — so cache and reasoning tokens (priced differently) survive instead of
    being flattened away. ``legacy_aliases`` also accepts OpenAI's older ``prompt_tokens`` /
    ``completion_tokens`` names as fallbacks.
    """
    usage = _usage_object(usage, "payload")
    input_candidates = [(usage, "input_tokens")]
    output_candidates = [(usage, "output_tokens")]
    if legacy_aliases:
        input_candidates.append((usage, "prompt_tokens"))
        output_candidates.append((usage, "completion_tokens"))
    input_tokens = _first_count(*input_candidates)
    output_tokens = _first_count(*output_candidates)
    normalized_input = input_tokens if input_tokens is not None else 0
    normalized_output = output_tokens if output_tokens is not None else 0
    reported_total = _usage_count(usage, "total_tokens")
    normalized = {
        "input_tokens": normalized_input,
        "output_tokens": normalized_output,
        "total_tokens": reported_total if reported_total is not None else 0,
    }
    input_details = _usage_object(
        usage.get("input_tokens_details", usage.get("prompt_tokens_details")),
        "input_tokens_details",
    )
    output_details = _usage_object(
        usage.get("output_tokens_details", usage.get("completion_tokens_details")),
        "output_tokens_details",
    )
    input_audio = _usage_count(input_details, "audio_tokens") or 0
    output_audio = _usage_count(output_details, "audio_tokens") or 0
    audio = input_audio + output_audio
    details = {
        "cache_read_tokens": _first_count(
            (usage, "cache_read_tokens"),
            (usage, "cache_read_input_tokens"),
            (input_details, "cached_tokens"),
            (usage, "cached_tokens"),
            (usage, "cachedContentTokenCount"),
        ),
        "cache_creation_tokens": _first_count(
            (usage, "cache_creation_tokens"),
            (usage, "cache_creation_input_tokens"),
        ),
        "reasoning_tokens": _first_count(
            (usage, "reasoning_tokens"),
            (output_details, "reasoning_tokens"),
            (output_details, "thinking_tokens"),
            (usage, "thoughtsTokenCount"),
        ),
        "audio_tokens": _first_count((usage, "audio_tokens")),
    }
    if details["audio_tokens"] is None and audio:
        details["audio_tokens"] = audio
    for key, value in details.items():
        if value:
            normalized[key] = value
    return normalized
