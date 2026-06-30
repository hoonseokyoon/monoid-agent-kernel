"""Reference implementation of the Monoid Agent Kernel backend contract.

Provided as an example run-orchestration backend. Real integrators are expected to build
their own against ``monoid_agent_kernel.contracts`` and ``docs/CONTRACTS.md``. Not part of
the supported public surface.
"""

from monoid_agent_kernel.reference.backend.service import (
    BackendRunRequest,
    BackendRunSubmission,
    RunnerBackend,
)
from monoid_agent_kernel.reference._shared.tokens import TokenClaims, TokenManager

__all__ = [
    "BackendRunRequest",
    "BackendRunSubmission",
    "RunnerBackend",
    "TokenClaims",
    "TokenManager",
]
