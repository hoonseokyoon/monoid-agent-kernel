"""Reference implementation of the native-agent-runner LLM gateway contract.

Provided as an example credential-boundary gateway. Real integrators are expected to build
their own against ``native_agent_runner.contracts`` and ``docs/CONTRACTS.md``. Not part of
the supported public surface.
"""

from native_agent_runner.reference.llm_gateway.service import (
    LlmGatewayBackend,
    LlmGatewayTurnRequest,
    LlmGatewayUsage,
)

__all__ = ["LlmGatewayBackend", "LlmGatewayTurnRequest", "LlmGatewayUsage"]
