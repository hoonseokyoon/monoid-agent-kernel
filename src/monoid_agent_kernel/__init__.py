"""Monoid Agent Kernel: the stable contracts and core engine entrypoints.

The top-level package mirrors ``monoid_agent_kernel.contracts``. Helper kit,
provider, recorder, MCP, observability, and reference implementations are imported
from their explicit modules.
"""

from monoid_agent_kernel import contracts as contracts
from monoid_agent_kernel.contracts import *  # noqa: F401,F403

__all__ = [*contracts.__all__]
