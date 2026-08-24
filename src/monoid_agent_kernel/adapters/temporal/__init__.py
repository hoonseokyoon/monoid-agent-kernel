"""Content-free Temporal records and Signal-With-Start transport.

Importing this namespace never loads ``temporalio``. Temporal workers explicitly import
``monoid_agent_kernel.adapters.temporal.workflow`` after installing the optional extra.
"""

from .dependency import TemporalDependencyMissing
from .dispatcher import (
    MAX_TEMPORAL_RPC_TIMEOUT_S,
    TemporalSignalWithStartTransport,
    temporal_dispatch_ref,
    temporal_workflow_id,
)
from .names import (
    DEFAULT_TEMPORAL_WORKFLOW_ID_PREFIX,
    TEMPORAL_COMMAND_SIGNAL,
    TEMPORAL_DRIVE_ACTIVATION_ACTIVITY,
    TEMPORAL_RUN_WORKFLOW_TYPE,
    TEMPORAL_STATUS_QUERY,
    TEMPORAL_WORKFLOW_BUILD,
)
from .records import (
    ACCEPTED_TEMPORAL_ACTIVATION_RESULT_SCHEMA_VERSIONS,
    ACCEPTED_TEMPORAL_RUN_POLICY_SCHEMA_VERSIONS,
    ACCEPTED_TEMPORAL_RUN_STATE_SCHEMA_VERSIONS,
    ACCEPTED_TEMPORAL_RUN_STATUS_SCHEMA_VERSIONS,
    MAX_ACTIVITY_ATTEMPTS,
    MAX_ACTIVITY_TIMEOUT_S,
    MAX_HISTORY_ROLLOVER_COMMANDS,
    TEMPORAL_ACTIVATION_RESULT_SCHEMA_VERSION,
    TEMPORAL_RUN_POLICY_SCHEMA_VERSION,
    TEMPORAL_RUN_STATE_SCHEMA_VERSION,
    TEMPORAL_RUN_STATUS_SCHEMA_VERSION,
    TemporalActivationResult,
    TemporalRunPhase,
    TemporalRunPolicy,
    TemporalRunState,
    TemporalRunStatus,
)

__all__ = [
    "TemporalDependencyMissing",
    "TEMPORAL_RUN_WORKFLOW_TYPE",
    "TEMPORAL_COMMAND_SIGNAL",
    "TEMPORAL_STATUS_QUERY",
    "TEMPORAL_DRIVE_ACTIVATION_ACTIVITY",
    "TEMPORAL_WORKFLOW_BUILD",
    "DEFAULT_TEMPORAL_WORKFLOW_ID_PREFIX",
    "TEMPORAL_RUN_POLICY_SCHEMA_VERSION",
    "ACCEPTED_TEMPORAL_RUN_POLICY_SCHEMA_VERSIONS",
    "TEMPORAL_RUN_STATE_SCHEMA_VERSION",
    "ACCEPTED_TEMPORAL_RUN_STATE_SCHEMA_VERSIONS",
    "TEMPORAL_ACTIVATION_RESULT_SCHEMA_VERSION",
    "ACCEPTED_TEMPORAL_ACTIVATION_RESULT_SCHEMA_VERSIONS",
    "TEMPORAL_RUN_STATUS_SCHEMA_VERSION",
    "ACCEPTED_TEMPORAL_RUN_STATUS_SCHEMA_VERSIONS",
    "MAX_ACTIVITY_TIMEOUT_S",
    "MAX_ACTIVITY_ATTEMPTS",
    "MAX_HISTORY_ROLLOVER_COMMANDS",
    "MAX_TEMPORAL_RPC_TIMEOUT_S",
    "TemporalRunPhase",
    "TemporalRunPolicy",
    "TemporalRunState",
    "TemporalActivationResult",
    "TemporalRunStatus",
    "temporal_workflow_id",
    "temporal_dispatch_ref",
    "TemporalSignalWithStartTransport",
]
