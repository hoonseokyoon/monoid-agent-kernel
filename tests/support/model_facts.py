"""Fixed v2 wire examples, independent of the new writer and reader implementations."""

from __future__ import annotations

from typing import Any


COUNTERS = (
    "audio_input_tokens", "audio_output_tokens", "audio_tokens", "cached_input_tokens",
    "cache_creation_tokens", "cache_read_tokens", "cache_write_tokens", "input_tokens",
    "output_tokens", "reasoning_tokens", "total_tokens",
)


def facts(usage: dict[str, int], *, outcome: str = "completed") -> dict[str, Any]:
    return {
        "termination": {
            "outcome": outcome, "requested_by": None,
            "ended_by": "provider_terminal" if outcome == "completed" else "provider_error",
        },
        "usage_evidence": {
            "reported": dict(usage),
            "completeness": {key: "complete" if key in usage else "unknown" for key in COUNTERS},
            "attempts": [{"index": 1, "reported": dict(usage),
                          "source": "provider" if usage else None,
                          "collection": "normal" if usage else None}],
        },
    }
