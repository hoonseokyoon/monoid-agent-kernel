"""Versioned, content-free records for Temporal run orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args

from monoid_agent_kernel.core.json_ingress import is_portable_json_integer
from monoid_agent_kernel.core.safe_evidence import (
    is_safe_opaque_address,
    is_safe_opaque_id,
    is_safe_taxonomy_code,
)
from monoid_agent_kernel.core.wire_validation import (
    parse_bool,
    parse_int,
    parse_literal,
    parse_required_str,
    parse_str,
    require_list,
    require_object,
    require_only_fields,
)
from monoid_agent_kernel.hosting.admission import AdmittedCommand
from monoid_agent_kernel.identifiers import accepted_namespaced_ids, namespaced_id

from .names import TEMPORAL_WORKFLOW_BUILD


TEMPORAL_RUN_POLICY_SCHEMA_VERSION = namespaced_id("temporal-run-policy.v1")
ACCEPTED_TEMPORAL_RUN_POLICY_SCHEMA_VERSIONS = accepted_namespaced_ids("temporal-run-policy.v1")
TEMPORAL_RUN_STATE_SCHEMA_VERSION = namespaced_id("temporal-run-state.v1")
ACCEPTED_TEMPORAL_RUN_STATE_SCHEMA_VERSIONS = accepted_namespaced_ids("temporal-run-state.v1")
TEMPORAL_ACTIVATION_RESULT_SCHEMA_VERSION = namespaced_id("temporal-activation-result.v1")
ACCEPTED_TEMPORAL_ACTIVATION_RESULT_SCHEMA_VERSIONS = accepted_namespaced_ids(
    "temporal-activation-result.v1"
)
TEMPORAL_RUN_STATUS_SCHEMA_VERSION = namespaced_id("temporal-run-status.v1")
ACCEPTED_TEMPORAL_RUN_STATUS_SCHEMA_VERSIONS = accepted_namespaced_ids("temporal-run-status.v1")

MAX_ACTIVITY_TIMEOUT_S = 7 * 24 * 60 * 60
MAX_ACTIVITY_ATTEMPTS = 100
MAX_HISTORY_ROLLOVER_COMMANDS = 10_000

TemporalRunPhase = Literal["waiting", "running", "terminal"]

_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "activity_task_queue",
        "activity_start_to_close_timeout_s",
        "activity_heartbeat_timeout_s",
        "activity_max_attempts",
        "history_rollover_command_limit",
    }
)
_STATE_FIELDS = frozenset(
    {
        "schema_version",
        "workflow_build",
        "run_id",
        "policy",
        "next_command_sequence",
        "pending_commands",
        "latest_receipt_ref",
        "rollover_count",
        "duplicate_signal_count",
        "last_error_code",
    }
)
_ACTIVATION_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "command_id",
        "command_sequence",
        "command_identity_sha256",
        "receipt_ref",
        "terminal",
    }
)
_STATUS_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "phase",
        "next_command_sequence",
        "in_flight_sequence",
        "pending_count",
        "pending_head_sequence",
        "latest_receipt_ref",
        "rollover_count",
        "duplicate_signal_count",
        "last_error_code",
    }
)


def _require_integer_range(
    value: object,
    field_name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> None:
    if (
        not is_portable_json_integer(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        if maximum is None:
            raise ValueError(f"{field_name} must be a portable integer >= {minimum}")
        raise ValueError(f"{field_name} must be in the portable range [{minimum}, {maximum}]")


def _require_closed_record_fields(
    payload: dict[str, object],
    fields: frozenset[str],
    name: str,
) -> None:
    require_only_fields(payload, fields, name)
    if any(field_name not in payload for field_name in fields):
        raise ValueError(f"{name} is missing required fields")


@dataclass(frozen=True, kw_only=True)
class TemporalRunPolicy:
    """Recorded Activity and history-rollover policy for one run Workflow."""

    activity_task_queue: str
    activity_start_to_close_timeout_s: int = 3_600
    activity_heartbeat_timeout_s: int = 30
    activity_max_attempts: int = 3
    history_rollover_command_limit: int = 0
    schema_version: str = TEMPORAL_RUN_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in ACCEPTED_TEMPORAL_RUN_POLICY_SCHEMA_VERSIONS:
            raise ValueError("unsupported Temporal run policy schema")
        if not is_safe_opaque_id(self.activity_task_queue) or len(self.activity_task_queue) > 255:
            raise ValueError("Temporal Activity task queue must be a bounded opaque id")
        _require_integer_range(
            self.activity_start_to_close_timeout_s,
            "Temporal Activity start-to-close timeout",
            minimum=1,
            maximum=MAX_ACTIVITY_TIMEOUT_S,
        )
        _require_integer_range(
            self.activity_heartbeat_timeout_s,
            "Temporal Activity heartbeat timeout",
            minimum=1,
            maximum=MAX_ACTIVITY_TIMEOUT_S,
        )
        if self.activity_heartbeat_timeout_s > self.activity_start_to_close_timeout_s:
            raise ValueError("Temporal Activity heartbeat timeout exceeds start-to-close timeout")
        _require_integer_range(
            self.activity_max_attempts,
            "Temporal Activity maximum attempts",
            minimum=1,
            maximum=MAX_ACTIVITY_ATTEMPTS,
        )
        _require_integer_range(
            self.history_rollover_command_limit,
            "Temporal history rollover command limit",
            minimum=0,
            maximum=MAX_HISTORY_ROLLOVER_COMMANDS,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": TEMPORAL_RUN_POLICY_SCHEMA_VERSION,
            "activity_task_queue": self.activity_task_queue,
            "activity_start_to_close_timeout_s": self.activity_start_to_close_timeout_s,
            "activity_heartbeat_timeout_s": self.activity_heartbeat_timeout_s,
            "activity_max_attempts": self.activity_max_attempts,
            "history_rollover_command_limit": self.history_rollover_command_limit,
        }

    @classmethod
    def from_json(cls, payload: object) -> TemporalRunPolicy:
        payload = require_object(payload, "Temporal run policy")
        _require_closed_record_fields(payload, _POLICY_FIELDS, "Temporal run policy")
        return cls(
            schema_version=parse_required_str(payload, "schema_version"),
            activity_task_queue=parse_required_str(payload, "activity_task_queue"),
            activity_start_to_close_timeout_s=parse_int(
                payload, "activity_start_to_close_timeout_s"
            ),
            activity_heartbeat_timeout_s=parse_int(payload, "activity_heartbeat_timeout_s"),
            activity_max_attempts=parse_int(payload, "activity_max_attempts"),
            history_rollover_command_limit=parse_int(payload, "history_rollover_command_limit"),
        )


@dataclass(frozen=True, kw_only=True)
class TemporalRunState:
    """Complete safe-point state transferred by Continue-As-New."""

    run_id: str
    policy: TemporalRunPolicy
    next_command_sequence: int = 1
    pending_commands: tuple[AdmittedCommand, ...] = field(default_factory=tuple)
    latest_receipt_ref: str = ""
    rollover_count: int = 0
    duplicate_signal_count: int = 0
    last_error_code: str = ""
    workflow_build: str = TEMPORAL_WORKFLOW_BUILD
    schema_version: str = TEMPORAL_RUN_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in ACCEPTED_TEMPORAL_RUN_STATE_SCHEMA_VERSIONS:
            raise ValueError("unsupported Temporal run state schema")
        if self.workflow_build != TEMPORAL_WORKFLOW_BUILD:
            raise ValueError("unsupported Temporal Workflow build")
        if not is_safe_opaque_id(self.run_id):
            raise ValueError("Temporal run state run_id must be a bounded opaque id")
        if not isinstance(self.policy, TemporalRunPolicy):
            raise TypeError("Temporal run state policy must be TemporalRunPolicy")
        _require_integer_range(
            self.next_command_sequence,
            "Temporal next command sequence",
            minimum=1,
        )
        if type(self.pending_commands) is not tuple or any(
            not isinstance(command, AdmittedCommand) for command in self.pending_commands
        ):
            raise TypeError("Temporal pending commands must be a tuple of AdmittedCommand")
        sequences = tuple(command.command_sequence for command in self.pending_commands)
        if sequences != tuple(sorted(sequences)) or len(sequences) != len(set(sequences)):
            raise ValueError("Temporal pending commands must have unique ascending sequences")
        if any(
            command.run_id != self.run_id or command.command_sequence < self.next_command_sequence
            for command in self.pending_commands
        ):
            raise ValueError("Temporal pending command is outside the run sequence frontier")
        if self.latest_receipt_ref and not is_safe_opaque_address(self.latest_receipt_ref):
            raise ValueError("Temporal latest receipt ref must be an opaque address")
        if self.next_command_sequence == 1 and self.latest_receipt_ref:
            raise ValueError("Temporal initial run state cannot carry a latest receipt ref")
        if self.next_command_sequence > 1 and not self.latest_receipt_ref:
            raise ValueError("Temporal advanced run state requires a latest receipt ref")
        _require_integer_range(
            self.rollover_count,
            "Temporal rollover count",
            minimum=0,
        )
        _require_integer_range(
            self.duplicate_signal_count,
            "Temporal duplicate signal count",
            minimum=0,
        )
        if self.last_error_code and not is_safe_taxonomy_code(self.last_error_code):
            raise ValueError("Temporal run state error code must be a taxonomy code")

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": TEMPORAL_RUN_STATE_SCHEMA_VERSION,
            "workflow_build": TEMPORAL_WORKFLOW_BUILD,
            "run_id": self.run_id,
            "policy": self.policy.to_json(),
            "next_command_sequence": self.next_command_sequence,
            "pending_commands": [command.to_json() for command in self.pending_commands],
            "latest_receipt_ref": self.latest_receipt_ref,
            "rollover_count": self.rollover_count,
            "duplicate_signal_count": self.duplicate_signal_count,
            "last_error_code": self.last_error_code,
        }

    @classmethod
    def from_json(cls, payload: object) -> TemporalRunState:
        payload = require_object(payload, "Temporal run state")
        _require_closed_record_fields(payload, _STATE_FIELDS, "Temporal run state")
        pending = tuple(
            AdmittedCommand.from_json(item)
            for item in require_list(payload.get("pending_commands"), "pending_commands")
        )
        return cls(
            schema_version=parse_required_str(payload, "schema_version"),
            workflow_build=parse_required_str(payload, "workflow_build"),
            run_id=parse_required_str(payload, "run_id"),
            policy=TemporalRunPolicy.from_json(payload.get("policy")),
            next_command_sequence=parse_int(payload, "next_command_sequence"),
            pending_commands=pending,
            latest_receipt_ref=parse_str(payload, "latest_receipt_ref"),
            rollover_count=parse_int(payload, "rollover_count"),
            duplicate_signal_count=parse_int(payload, "duplicate_signal_count"),
            last_error_code=parse_str(payload, "last_error_code"),
        )


@dataclass(frozen=True, kw_only=True)
class TemporalActivationResult:
    """Content-free result returned by the versioned finite activation Activity."""

    run_id: str
    command_id: str
    command_sequence: int
    command_identity_sha256: str
    receipt_ref: str
    terminal: bool
    schema_version: str = TEMPORAL_ACTIVATION_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in ACCEPTED_TEMPORAL_ACTIVATION_RESULT_SCHEMA_VERSIONS:
            raise ValueError("unsupported Temporal activation result schema")
        if not is_safe_opaque_id(self.run_id) or not is_safe_opaque_id(self.command_id):
            raise ValueError("Temporal activation result identities must be bounded opaque ids")
        _require_integer_range(
            self.command_sequence,
            "Temporal activation result command sequence",
            minimum=1,
        )
        if (
            type(self.command_identity_sha256) is not str
            or len(self.command_identity_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.command_identity_sha256
            )
        ):
            raise ValueError("Temporal activation result command identity must be a SHA-256 digest")
        if not is_safe_opaque_address(self.receipt_ref):
            raise ValueError("Temporal activation result receipt_ref must be an opaque address")
        if type(self.terminal) is not bool:
            raise ValueError("Temporal activation result terminal must be a boolean")

    def matches(self, command: AdmittedCommand) -> bool:
        return (
            isinstance(command, AdmittedCommand)
            and self.run_id == command.run_id
            and self.command_id == command.command_id
            and self.command_sequence == command.command_sequence
            and self.command_identity_sha256 == command.identity_sha256
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": TEMPORAL_ACTIVATION_RESULT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "command_id": self.command_id,
            "command_sequence": self.command_sequence,
            "command_identity_sha256": self.command_identity_sha256,
            "receipt_ref": self.receipt_ref,
            "terminal": self.terminal,
        }

    @classmethod
    def from_command(
        cls,
        command: AdmittedCommand,
        *,
        receipt_ref: str,
        terminal: bool,
    ) -> TemporalActivationResult:
        if not isinstance(command, AdmittedCommand):
            raise TypeError("Temporal activation result requires AdmittedCommand")
        return cls(
            run_id=command.run_id,
            command_id=command.command_id,
            command_sequence=command.command_sequence,
            command_identity_sha256=command.identity_sha256,
            receipt_ref=receipt_ref,
            terminal=terminal,
        )

    @classmethod
    def from_json(cls, payload: object) -> TemporalActivationResult:
        payload = require_object(payload, "Temporal activation result")
        _require_closed_record_fields(
            payload,
            _ACTIVATION_RESULT_FIELDS,
            "Temporal activation result",
        )
        return cls(
            schema_version=parse_required_str(payload, "schema_version"),
            run_id=parse_required_str(payload, "run_id"),
            command_id=parse_required_str(payload, "command_id"),
            command_sequence=parse_int(payload, "command_sequence"),
            command_identity_sha256=parse_required_str(payload, "command_identity_sha256"),
            receipt_ref=parse_required_str(payload, "receipt_ref"),
            terminal=parse_bool(payload, "terminal"),
        )


@dataclass(frozen=True, kw_only=True)
class TemporalRunStatus:
    """Content-free Workflow Query and terminal-result projection."""

    run_id: str
    phase: TemporalRunPhase
    next_command_sequence: int
    in_flight_sequence: int
    pending_count: int
    pending_head_sequence: int
    latest_receipt_ref: str
    rollover_count: int
    duplicate_signal_count: int
    last_error_code: str = ""
    schema_version: str = TEMPORAL_RUN_STATUS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in ACCEPTED_TEMPORAL_RUN_STATUS_SCHEMA_VERSIONS:
            raise ValueError("unsupported Temporal run status schema")
        if not is_safe_opaque_id(self.run_id):
            raise ValueError("Temporal run status run_id must be a bounded opaque id")
        if type(self.phase) is not str or self.phase not in get_args(TemporalRunPhase):
            raise ValueError("Temporal run status phase is outside the portable vocabulary")
        _require_integer_range(
            self.next_command_sequence,
            "Temporal status next command sequence",
            minimum=1,
        )
        for value, name in (
            (self.in_flight_sequence, "Temporal status in-flight sequence"),
            (self.pending_count, "Temporal status pending count"),
            (self.pending_head_sequence, "Temporal status pending head sequence"),
            (self.rollover_count, "Temporal status rollover count"),
            (self.duplicate_signal_count, "Temporal status duplicate signal count"),
        ):
            _require_integer_range(value, name, minimum=0)
        if self.phase == "running" and self.in_flight_sequence < 1:
            raise ValueError("running Temporal status requires an in-flight sequence")
        if self.phase != "running" and self.in_flight_sequence != 0:
            raise ValueError("non-running Temporal status cannot carry an in-flight sequence")
        if bool(self.pending_count) != bool(self.pending_head_sequence):
            raise ValueError("Temporal status pending count and head must agree")
        if self.pending_head_sequence and self.pending_head_sequence < self.next_command_sequence:
            raise ValueError("Temporal status pending head precedes the sequence frontier")
        if self.latest_receipt_ref and not is_safe_opaque_address(self.latest_receipt_ref):
            raise ValueError("Temporal status latest receipt ref must be an opaque address")
        if self.next_command_sequence == 1 and self.latest_receipt_ref:
            raise ValueError("initial Temporal status cannot carry a latest receipt ref")
        if self.next_command_sequence > 1 and not self.latest_receipt_ref:
            raise ValueError("advanced Temporal status requires a latest receipt ref")
        if self.last_error_code and not is_safe_taxonomy_code(self.last_error_code):
            raise ValueError("Temporal run status error code must be a taxonomy code")

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": TEMPORAL_RUN_STATUS_SCHEMA_VERSION,
            "run_id": self.run_id,
            "phase": self.phase,
            "next_command_sequence": self.next_command_sequence,
            "in_flight_sequence": self.in_flight_sequence,
            "pending_count": self.pending_count,
            "pending_head_sequence": self.pending_head_sequence,
            "latest_receipt_ref": self.latest_receipt_ref,
            "rollover_count": self.rollover_count,
            "duplicate_signal_count": self.duplicate_signal_count,
            "last_error_code": self.last_error_code,
        }

    @classmethod
    def from_json(cls, payload: object) -> TemporalRunStatus:
        payload = require_object(payload, "Temporal run status")
        _require_closed_record_fields(payload, _STATUS_FIELDS, "Temporal run status")
        return cls(
            schema_version=parse_required_str(payload, "schema_version"),
            run_id=parse_required_str(payload, "run_id"),
            phase=parse_literal(payload, "phase", get_args(TemporalRunPhase)),
            next_command_sequence=parse_int(payload, "next_command_sequence"),
            in_flight_sequence=parse_int(payload, "in_flight_sequence"),
            pending_count=parse_int(payload, "pending_count"),
            pending_head_sequence=parse_int(payload, "pending_head_sequence"),
            latest_receipt_ref=parse_str(payload, "latest_receipt_ref"),
            rollover_count=parse_int(payload, "rollover_count"),
            duplicate_signal_count=parse_int(payload, "duplicate_signal_count"),
            last_error_code=parse_str(payload, "last_error_code"),
        )


__all__ = [
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
    "TemporalRunPhase",
    "TemporalRunPolicy",
    "TemporalRunState",
    "TemporalActivationResult",
    "TemporalRunStatus",
]
