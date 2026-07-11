"""Optional DBOS-backed durable control-plane Reference profile.

Importing this package does not import DBOS. Constructing :class:`DbosControlPlane`
requires the ``reference-dbos`` extra.
"""

from monoid_agent_kernel.reference.dbos.control_plane import (
    DbosControlConfig,
    DbosControlEnvelope,
    DbosControlPlane,
    DbosDependencyError,
    DbosProcessOwnershipError,
    DbosShutdownTimeout,
)

__all__ = [
    "DbosControlConfig",
    "DbosControlEnvelope",
    "DbosControlPlane",
    "DbosDependencyError",
    "DbosProcessOwnershipError",
    "DbosShutdownTimeout",
]
