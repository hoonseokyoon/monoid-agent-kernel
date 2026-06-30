"""Reference implementation of the Monoid Agent Kernel LLM gateway contract.

Provided as an example credential-boundary gateway. Real integrators are expected to build
their own against ``monoid_agent_kernel.contracts`` and ``docs/CONTRACTS.md``. Not part of
the supported public surface.
"""

from monoid_agent_kernel.reference.llm_gateway.service import (
    LlmGatewayBackend,
    LlmGatewayTurnRequest,
    LlmGatewayUsage,
)

__all__ = ["LlmGatewayBackend", "LlmGatewayTurnRequest", "LlmGatewayUsage"]
