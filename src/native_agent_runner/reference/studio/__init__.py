"""Agent Studio — a bundled reference agent app (LLM gateway + runner backend + UI/BFF).

EXAMPLE, not part of the supported public surface. It is built only against the contracts and
the other reference services, to pressure-test "can an integrator stand up a chat + agentic app
from the surface alone?" — and to surface DX gaps in the core (see ``DX_NOTES.md``).
"""

# The offline echo provider lives with the gateway it plugs into (the LLM-side counterpart of
# the WebGateway's fake provider); re-exported here for Studio callers' convenience.
from native_agent_runner.reference.llm_gateway.providers import (
    EchoModelAdapter,
    offline_provider_factory,
)
from native_agent_runner.reference.studio.server import StudioConfig, StudioServer

__all__ = [
    "EchoModelAdapter",
    "offline_provider_factory",
    "StudioConfig",
    "StudioServer",
]
