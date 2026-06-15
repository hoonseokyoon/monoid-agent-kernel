"""Shared helpers for model adapters.

The OpenAI and gateway adapters build the same reasoning block and normalize
usage the same way; only their tool-schema shape genuinely differs. Keep the
common pieces here so the two adapters cannot drift.
"""

from __future__ import annotations

from typing import Any

from native_agent_runner.core.spec import ReasoningConfig


def build_reasoning_payload(reasoning: ReasoningConfig) -> dict[str, Any]:
    """Reasoning block for a model request: ``{}`` when default/off, else effort/summary."""
    payload: dict[str, Any] = {}
    if reasoning.effort != "default":
        payload["effort"] = reasoning.effort
    if reasoning.summary != "off":
        payload["summary"] = reasoning.summary
    return payload


def normalize_usage(usage: dict[str, Any] | None, *, legacy_aliases: bool = False) -> dict[str, int]:
    """Coerce a provider usage dict to ``{input_tokens, output_tokens, total_tokens}``.

    ``legacy_aliases`` also accepts OpenAI's older ``prompt_tokens`` /
    ``completion_tokens`` names as fallbacks.
    """
    usage = usage or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if legacy_aliases:
        input_tokens = input_tokens or usage.get("prompt_tokens")
        output_tokens = output_tokens or usage.get("completion_tokens")
    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }
