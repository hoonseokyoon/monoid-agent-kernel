"""Stable Temporal type and message names recorded in Workflow history."""

TEMPORAL_RUN_WORKFLOW_TYPE = "monoid_agent_kernel.run.v1"
TEMPORAL_COMMAND_SIGNAL = "monoid_agent_kernel.submit_command.v1"
TEMPORAL_STATUS_QUERY = "monoid_agent_kernel.status.v1"
TEMPORAL_DRIVE_ACTIVATION_ACTIVITY = "monoid_agent_kernel.drive_activation.v1"
TEMPORAL_WORKFLOW_BUILD = "monoid.temporal-run-workflow.v1"
DEFAULT_TEMPORAL_WORKFLOW_ID_PREFIX = "monoid-run-v1"

__all__ = [
    "TEMPORAL_RUN_WORKFLOW_TYPE",
    "TEMPORAL_COMMAND_SIGNAL",
    "TEMPORAL_STATUS_QUERY",
    "TEMPORAL_DRIVE_ACTIVATION_ACTIVITY",
    "TEMPORAL_WORKFLOW_BUILD",
    "DEFAULT_TEMPORAL_WORKFLOW_ID_PREFIX",
]
