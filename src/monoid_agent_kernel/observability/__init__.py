"""Observability integrations for the agent runner (opt-in, never required by the core).

The core stays zero-dependency: ``OtelEventSink`` lazily imports ``opentelemetry`` only when
instantiated, so importing this package without the ``[otel]`` extra is fine.
"""

from monoid_agent_kernel.observability.otel import OtelEventSink

__all__ = ["OtelEventSink"]
