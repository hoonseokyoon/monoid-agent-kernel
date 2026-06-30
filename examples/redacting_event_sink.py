"""Reference: redact secrets from the public event stream at the integration boundary.

The kernel deliberately does not guess at secrets by argument name. It keeps
file-content fields out of public events and masks ``PermissionPolicy.redact_patterns`` paths
(see docs/CONTRACTS.md). Secret redaction beyond that is the integrating backend's job, and an
``EventSink`` is the seam to add it: wrap the sink that actually leaves your trust boundary.

This module is an example you copy and own. It re-homes the
heuristic that used to live (always-on, unconfigurable) in the core as opt-in integrator code.

Use it from the CLI:

    monoid run --workspace . --instruction "..." \
        --runtime-config-file examples/runtime-config.json \
        --llm-gateway-url http://127.0.0.1:8080/internal/llm/turns \
        --event-sink-module examples/redacting_event_sink.py:make_sink

or programmatically:

    AgentLoop(..., runtime_config_provider=provider,
        event_sinks=(RedactingEventSink(JsonlEventSink(path)),))
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from monoid_agent_kernel.contracts import AgentEvent, EventSink
from monoid_agent_kernel.recorder import StdoutJsonlSink

# Tune these to your environment; this is your policy, not the kernel's.
SENSITIVE_KEY_FRAGMENTS = (
    "token",
    "secret",
    "password",
    "credential",
    "authorization",
    "api_key",
    "apikey",
    "private_key",
)
REDACTED = "[redacted]"


def _scrub(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        return {str(k): _scrub(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(item, key) for item in value]
    lowered = key.lower()
    if any(fragment in lowered for fragment in SENSITIVE_KEY_FRAGMENTS):
        return REDACTED
    if isinstance(value, str) and "PRIVATE KEY" in value.upper():
        return REDACTED
    return value


class RedactingEventSink:
    """Wrap another EventSink and mask secret-looking values before forwarding."""

    def __init__(self, inner: EventSink) -> None:
        self._inner = inner

    def emit(self, event: AgentEvent) -> None:
        self._inner.emit(replace(event, data=_scrub(event.data)))

    def close(self) -> None:
        self._inner.close()


def make_sink() -> EventSink:
    """Factory for ``--event-sink-module``: redacted events streamed to stdout."""
    return RedactingEventSink(StdoutJsonlSink())
