"""Optional experimental DBOS activation-recovery Reference profile.

Importing this package does not import DBOS. Constructing :class:`DbosControlPlane`
or :class:`DbosRunDriver` requires the ``reference-dbos`` extra.
"""

from monoid_agent_kernel.reference.dbos.control_plane import (
    DbosControlConfig,
    DbosControlEnvelope,
    DbosControlPlane,
)
from monoid_agent_kernel.reference.dbos.runtime import (
    DbosDependencyError,
    DbosProcessOwnershipError,
    DbosShutdownTimeout,
)
from monoid_agent_kernel.reference.dbos.run_driver import (
    DBOS_RESUME_COMMAND_VERSION,
    DBOS_RUN_RECEIPT_VERSION,
    DbosResumeCommand,
    DbosRunConfig,
    DbosRunDriver,
    DbosRunReceipt,
)

__all__ = [
    "DbosControlConfig",
    "DbosControlEnvelope",
    "DbosControlPlane",
    "DbosDependencyError",
    "DbosProcessOwnershipError",
    "DbosResumeCommand",
    "DbosRunConfig",
    "DbosRunDriver",
    "DbosRunReceipt",
    "DbosShutdownTimeout",
    "DBOS_RESUME_COMMAND_VERSION",
    "DBOS_RUN_RECEIPT_VERSION",
]
