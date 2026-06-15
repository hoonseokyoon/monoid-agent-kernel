"""Reference implementation of the native-agent-runner backend contract.

Provided as an example run-orchestration backend. Real integrators are expected to build
their own against ``native_agent_runner.contracts`` and ``docs/CONTRACTS.md``. Not part of
the supported public surface.
"""

from native_agent_runner.reference.backend.service import (
    BackendRunRequest,
    BackendRunSubmission,
    RunnerBackend,
)
from native_agent_runner.reference._shared.tokens import TokenClaims, TokenManager

__all__ = [
    "BackendRunRequest",
    "BackendRunSubmission",
    "RunnerBackend",
    "TokenClaims",
    "TokenManager",
]
