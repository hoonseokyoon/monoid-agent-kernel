"""Orchestrator-neutral finite activation records and driver."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, get_args

from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.core.authority import ActivationWriteAuthority, WriteAuthorityRevoked
from monoid_agent_kernel.core.checkpoint import CheckpointRecord, RunCheckpoint
from monoid_agent_kernel.core.content import ContentPart
from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.interruption import InterruptionCause
from monoid_agent_kernel.core.json_ingress import is_portable_json_integer
from monoid_agent_kernel.core.lifecycle import SessionState, state_from_suspension
from monoid_agent_kernel.core.model_io import is_recorded_digest
from monoid_agent_kernel.core.model_invocation import DurableModelInvocation
from monoid_agent_kernel.core.outcome import (
    RetryEligibility,
    TerminalOutcome,
    TerminalOutcomeKind,
    terminal_outcome_from_suspension,
)
from monoid_agent_kernel.core.result import (
    SUSPENSION_REASONS,
    Suspension,
    suspension_checkpoint_payload,
    suspension_from_checkpoint_payload,
)
from monoid_agent_kernel.core.safe_evidence import (
    is_safe_opaque_address,
    is_safe_opaque_id,
    is_safe_taxonomy_code,
)
from monoid_agent_kernel.core.spec import input_to_parts, user_message_from_parts
from monoid_agent_kernel.core.wire_validation import (
    parse_bool,
    parse_int,
    parse_literal,
    parse_required_str,
    parse_str,
    require_object,
    require_only_fields,
)
from monoid_agent_kernel.errors import NativeAgentError
from monoid_agent_kernel.hosting.contracts import (
    CommitResult,
    FencedRunSink,
    ModelInvocationRecord,
    WriterToken,
)
from monoid_agent_kernel.hosting.execution import FencedEventSink, FencedTerminalBridge
from monoid_agent_kernel.identifiers import accepted_namespaced_ids, namespaced_id

if TYPE_CHECKING:
    from monoid_agent_kernel.core.durable_codec import DurableLoadResult
    from monoid_agent_kernel.loop import AgentLoop


ACTIVATION_COMMAND_SCHEMA_VERSION = namespaced_id("activation-command.v1")
ACCEPTED_ACTIVATION_COMMAND_SCHEMA_VERSIONS = accepted_namespaced_ids("activation-command.v1")
ACTIVATION_RECEIPT_SCHEMA_VERSION = namespaced_id("activation-receipt.v1")
ACCEPTED_ACTIVATION_RECEIPT_SCHEMA_VERSIONS = accepted_namespaced_ids("activation-receipt.v1")

ActivationCommandKind = Literal["input", "control"]
ActivationLoopFactory = Callable[["ActivationCommand", "ActivationRuntime"], "AgentLoop"]
ActivationFaultHook = Callable[[str, "ActivationCommand"], None]
ActivationInputResolver = Callable[["ActivationCommand"], "ResolvedActivationInput"]

_COMMAND_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "command_id",
        "command_sequence",
        "kind",
        "source_checkpoint_seq",
        "source_checkpoint_sha256",
        "request_digest",
        "payload_ref",
        "identity_sha256",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "command_id",
        "command_sequence",
        "command_identity_sha256",
        "checkpoint_seq",
        "checkpoint_sha256",
        "checkpoint_ref",
        "state",
        "boundary_reason",
        "terminal",
        "terminal_ref",
        "applied_input_ref",
        "event_cursor",
        "stream_cursor",
        "outcome_kind",
        "retry_eligibility",
        "error_code",
        "provider_error_code",
        "interruption_cause",
    }
)


def _require_positive_integer(value: object, field_name: str) -> None:
    if not is_portable_json_integer(value) or value < 1:
        raise ValueError(f"{field_name} must be a positive portable integer")


def _require_nonnegative_integer(value: object, field_name: str) -> None:
    if not is_portable_json_integer(value) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative portable integer")


def _require_digest(value: object, field_name: str) -> None:
    if type(value) is not str or not is_recorded_digest(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def _checkpoint_digest(checkpoint: RunCheckpoint) -> str:
    return canonical_sha256(checkpoint.to_json())


def _checkpoint_receipt_digest(checkpoint: RunCheckpoint, marker: str) -> str:
    """Hash a boundary checkpoint with only the target receipt digest field blanked."""

    payload = checkpoint.to_json()
    raw_receipts = payload.get("applied_input_receipts")
    if not isinstance(raw_receipts, Mapping):
        raise ValueError("activation checkpoint receipts are invalid")
    receipts = dict(raw_receipts)
    target = receipts.get(marker)
    if not isinstance(target, Mapping):
        raise ValueError("activation checkpoint receipt is missing")
    target = dict(target)
    target["checkpoint_sha256"] = ""
    receipts[marker] = target
    payload["applied_input_receipts"] = receipts
    return canonical_sha256(payload)


@dataclass(frozen=True, kw_only=True)
class ActivationCommand:
    """One retry-stable reference to an admitted input or control command."""

    run_id: str
    command_id: str
    command_sequence: int
    kind: ActivationCommandKind
    source_checkpoint_seq: int
    source_checkpoint_sha256: str
    request_digest: str
    payload_ref: str
    schema_version: str = ACTIVATION_COMMAND_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in ACCEPTED_ACTIVATION_COMMAND_SCHEMA_VERSIONS:
            raise ValueError("unsupported activation command schema")
        if not is_safe_opaque_id(self.run_id):
            raise ValueError("activation command run_id must be a bounded opaque id")
        if not is_safe_opaque_id(self.command_id):
            raise ValueError("activation command command_id must be a bounded opaque id")
        _require_positive_integer(self.command_sequence, "activation command sequence")
        if type(self.kind) is not str or self.kind not in get_args(ActivationCommandKind):
            raise ValueError("activation command kind is outside the portable vocabulary")
        _require_nonnegative_integer(
            self.source_checkpoint_seq,
            "activation source checkpoint sequence",
        )
        _require_digest(
            self.source_checkpoint_sha256,
            "activation source checkpoint digest",
        )
        _require_digest(self.request_digest, "activation request digest")
        if not is_safe_opaque_address(self.payload_ref):
            raise ValueError("activation payload_ref must be a bounded opaque address")

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(
            {
                "schema_version": ACTIVATION_COMMAND_SCHEMA_VERSION,
                "run_id": self.run_id,
                "command_id": self.command_id,
                "command_sequence": self.command_sequence,
                "kind": self.kind,
                "source_checkpoint_seq": self.source_checkpoint_seq,
                "source_checkpoint_sha256": self.source_checkpoint_sha256,
                "request_digest": self.request_digest,
                "payload_ref": self.payload_ref,
            }
        )

    @property
    def checkpoint_marker(self) -> str:
        return f"monoid.activation/{self.identity_sha256}"

    @property
    def applied_input_ref(self) -> str:
        return f"activation:{self.identity_sha256}"

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": ACTIVATION_COMMAND_SCHEMA_VERSION,
            "run_id": self.run_id,
            "command_id": self.command_id,
            "command_sequence": self.command_sequence,
            "kind": self.kind,
            "source_checkpoint_seq": self.source_checkpoint_seq,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "request_digest": self.request_digest,
            "payload_ref": self.payload_ref,
            "identity_sha256": self.identity_sha256,
        }

    @classmethod
    def from_json(cls, payload: object) -> ActivationCommand:
        payload = require_object(payload, "activation command")
        require_only_fields(payload, _COMMAND_FIELDS, "activation command")
        command = cls(
            schema_version=parse_required_str(payload, "schema_version"),
            run_id=parse_required_str(payload, "run_id"),
            command_id=parse_required_str(payload, "command_id"),
            command_sequence=parse_int(payload, "command_sequence"),
            kind=parse_literal(payload, "kind", get_args(ActivationCommandKind)),
            source_checkpoint_seq=parse_int(payload, "source_checkpoint_seq"),
            source_checkpoint_sha256=parse_required_str(
                payload,
                "source_checkpoint_sha256",
            ),
            request_digest=parse_required_str(payload, "request_digest"),
            payload_ref=parse_required_str(payload, "payload_ref"),
        )
        if parse_required_str(payload, "identity_sha256") != command.identity_sha256:
            raise ValueError("activation command identity digest mismatch")
        return command


@dataclass(frozen=True, kw_only=True)
class ResolvedActivationInput:
    """Private in-memory payload resolved from an admitted command's opaque reference."""

    request_digest: str
    payload_ref: str
    parts: tuple[ContentPart, ...] = field(repr=False)

    def __post_init__(self) -> None:
        _require_digest(self.request_digest, "resolved activation request digest")
        if not is_safe_opaque_address(self.payload_ref):
            raise ValueError("resolved activation payload_ref must be an opaque address")
        if type(self.parts) is not tuple:
            raise TypeError("resolved activation input parts must be a tuple")
        try:
            normalized = input_to_parts(self.parts)
            message = user_message_from_parts(normalized)
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ValueError("resolved activation input parts are invalid") from exc
        if message is None:
            raise ValueError("resolved activation input must contain user-visible content")
        object.__setattr__(self, "parts", normalized)


@dataclass(frozen=True, kw_only=True)
class ActivationReceipt:
    """Content-free canonical observation of one applied activation command."""

    run_id: str
    command_id: str
    command_sequence: int
    command_identity_sha256: str
    checkpoint_seq: int
    checkpoint_sha256: str
    checkpoint_ref: str
    state: str
    boundary_reason: str
    terminal: bool
    terminal_ref: str
    applied_input_ref: str
    event_cursor: int
    stream_cursor: int
    outcome_kind: TerminalOutcomeKind
    retry_eligibility: RetryEligibility
    error_code: str = ""
    provider_error_code: str = ""
    interruption_cause: InterruptionCause | None = None
    schema_version: str = ACTIVATION_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in ACCEPTED_ACTIVATION_RECEIPT_SCHEMA_VERSIONS:
            raise ValueError("unsupported activation receipt schema")
        if not is_safe_opaque_id(self.run_id) or not is_safe_opaque_id(self.command_id):
            raise ValueError("activation receipt identities must be bounded opaque ids")
        _require_positive_integer(self.command_sequence, "activation receipt command sequence")
        _require_digest(self.command_identity_sha256, "activation command identity")
        _require_positive_integer(self.checkpoint_seq, "activation receipt checkpoint sequence")
        _require_digest(self.checkpoint_sha256, "activation receipt checkpoint digest")
        for field_name in ("checkpoint_ref", "applied_input_ref"):
            if not is_safe_opaque_address(getattr(self, field_name)):
                raise ValueError(f"activation receipt {field_name} must be an opaque address")
        if self.terminal_ref and not is_safe_opaque_address(self.terminal_ref):
            raise ValueError("activation receipt terminal_ref must be empty or an opaque address")
        try:
            SessionState(self.state)
        except (TypeError, ValueError) as exc:
            raise ValueError("activation receipt state is outside the lifecycle vocabulary") from exc
        if type(self.boundary_reason) is not str or self.boundary_reason not in SUSPENSION_REASONS:
            raise ValueError("activation receipt boundary reason is outside the pump vocabulary")
        if type(self.terminal) is not bool:
            raise ValueError("activation receipt terminal must be a boolean")
        if self.terminal != bool(self.terminal_ref):
            raise ValueError("activation receipt terminal flag and ref must agree")
        _require_nonnegative_integer(self.event_cursor, "activation receipt event cursor")
        _require_nonnegative_integer(self.stream_cursor, "activation receipt stream cursor")
        if type(self.outcome_kind) is not str or self.outcome_kind not in get_args(
            TerminalOutcomeKind
        ):
            raise ValueError("activation receipt outcome kind is outside the portable vocabulary")
        try:
            retry = RetryEligibility(self.retry_eligibility)
        except (TypeError, ValueError) as exc:
            raise ValueError("activation receipt retry eligibility is invalid") from exc
        object.__setattr__(self, "retry_eligibility", retry)
        for field_name in ("error_code", "provider_error_code"):
            value = getattr(self, field_name)
            if type(value) is not str or (value and not is_safe_taxonomy_code(value)):
                raise ValueError(f"activation receipt {field_name} is invalid")
        if self.interruption_cause is not None:
            try:
                cause = InterruptionCause(self.interruption_cause)
            except (TypeError, ValueError) as exc:
                raise ValueError("activation receipt interruption cause is invalid") from exc
            object.__setattr__(self, "interruption_cause", cause)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": ACTIVATION_RECEIPT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "command_id": self.command_id,
            "command_sequence": self.command_sequence,
            "command_identity_sha256": self.command_identity_sha256,
            "checkpoint_seq": self.checkpoint_seq,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_ref": self.checkpoint_ref,
            "state": self.state,
            "boundary_reason": self.boundary_reason,
            "terminal": self.terminal,
            "terminal_ref": self.terminal_ref,
            "applied_input_ref": self.applied_input_ref,
            "event_cursor": self.event_cursor,
            "stream_cursor": self.stream_cursor,
            "outcome_kind": self.outcome_kind,
            "retry_eligibility": self.retry_eligibility.value,
            "error_code": self.error_code,
            "provider_error_code": self.provider_error_code,
            "interruption_cause": (
                "" if self.interruption_cause is None else self.interruption_cause.value
            ),
        }

    @classmethod
    def from_json(cls, payload: object) -> ActivationReceipt:
        payload = require_object(payload, "activation receipt")
        require_only_fields(payload, _RECEIPT_FIELDS, "activation receipt")
        raw_cause = parse_str(payload, "interruption_cause")
        return cls(
            schema_version=parse_required_str(payload, "schema_version"),
            run_id=parse_required_str(payload, "run_id"),
            command_id=parse_required_str(payload, "command_id"),
            command_sequence=parse_int(payload, "command_sequence"),
            command_identity_sha256=parse_required_str(payload, "command_identity_sha256"),
            checkpoint_seq=parse_int(payload, "checkpoint_seq"),
            checkpoint_sha256=parse_required_str(payload, "checkpoint_sha256"),
            checkpoint_ref=parse_required_str(payload, "checkpoint_ref"),
            state=parse_required_str(payload, "state"),
            boundary_reason=parse_required_str(payload, "boundary_reason"),
            terminal=parse_bool(payload, "terminal"),
            terminal_ref=parse_str(payload, "terminal_ref"),
            applied_input_ref=parse_required_str(payload, "applied_input_ref"),
            event_cursor=parse_int(payload, "event_cursor"),
            stream_cursor=parse_int(payload, "stream_cursor"),
            outcome_kind=parse_literal(payload, "outcome_kind", get_args(TerminalOutcomeKind)),
            retry_eligibility=RetryEligibility(
                parse_required_str(payload, "retry_eligibility")
            ),
            error_code=parse_str(payload, "error_code"),
            provider_error_code=parse_str(payload, "provider_error_code"),
            interruption_cause=InterruptionCause(raw_cause) if raw_cause else None,
        )

    @classmethod
    def from_checkpoint(
        cls,
        command: ActivationCommand,
        checkpoint: RunCheckpoint,
        *,
        terminal_outcome: TerminalOutcome | None = None,
    ) -> ActivationReceipt:
        marker = command.checkpoint_marker
        if marker not in checkpoint.applied_input_ids:
            raise NativeAgentError(
                "activation checkpoint has no applied command marker",
                error_code="missing_activation_marker",
            )
        raw_receipt = checkpoint.applied_input_receipts.get(marker)
        if not isinstance(raw_receipt, Mapping):
            raise NativeAgentError(
                "activation checkpoint has no identity-bound receipt",
                error_code="missing_activation_receipt",
            )
        raw_suspension = raw_receipt.get("suspension")
        if not isinstance(raw_suspension, Mapping):
            raise NativeAgentError(
                "activation receipt has no durable boundary observation",
                error_code="missing_activation_boundary",
            )
        try:
            suspension = suspension_from_checkpoint_payload(raw_suspension)
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise NativeAgentError(
                "activation receipt boundary observation is invalid",
                error_code="invalid_activation_boundary",
            ) from exc
        checkpoint_seq = raw_receipt.get("checkpoint_seq")
        checkpoint_sha256 = raw_receipt.get("checkpoint_sha256")
        event_cursor = raw_receipt.get("event_cursor", 0)
        stream_cursor = raw_receipt.get("stream_cursor", 0)
        state = state_from_suspension(suspension).value
        terminal = raw_receipt.get("terminal")
        if (
            not is_portable_json_integer(checkpoint_seq)
            or checkpoint_seq > checkpoint.seq
            or checkpoint_seq <= command.source_checkpoint_seq
            or not is_recorded_digest(checkpoint_sha256)
            or raw_receipt.get("state") != state
            or type(terminal) is not bool
            or (suspension.reason == "terminal" and not terminal)
            or (terminal and suspension.reason not in {"terminal", "limited"})
            or raw_receipt.get("command_identity_sha256") != command.identity_sha256
            or not is_portable_json_integer(event_cursor)
            or event_cursor < 0
            or not is_portable_json_integer(stream_cursor)
            or stream_cursor < 0
        ):
            raise NativeAgentError(
                "activation receipt boundary metadata is invalid",
                error_code="invalid_activation_receipt",
            )
        if checkpoint_seq == checkpoint.seq:
            if checkpoint.terminal is not terminal or checkpoint_sha256 != (
                _checkpoint_receipt_digest(checkpoint, marker)
            ):
                raise NativeAgentError(
                    "activation receipt checkpoint identity is invalid",
                    error_code="invalid_activation_receipt",
                )
        checkpoint_outcome = terminal_outcome_from_suspension(
            suspension,
            run_id=command.run_id,
            checkpoint_seq=checkpoint_seq,
        )
        expected_interruption_cause = (
            ""
            if checkpoint_outcome.interruption_cause is None
            else checkpoint_outcome.interruption_cause.value
        )
        if (
            raw_receipt.get("outcome_kind") != checkpoint_outcome.kind
            or raw_receipt.get("retry_eligibility")
            != checkpoint_outcome.retry_eligibility.value
            or raw_receipt.get("error_code") != checkpoint_outcome.error_code
            or raw_receipt.get("provider_error_code") != checkpoint_outcome.provider_error_code
            or raw_receipt.get("interruption_cause") != expected_interruption_cause
            or raw_receipt.get("terminal_outcome_ref")
            != (f"terminal:{command.run_id}" if terminal else "")
        ):
            raise NativeAgentError(
                "activation receipt outcome metadata is invalid",
                error_code="invalid_activation_receipt",
            )
        if terminal_outcome is not None:
            if not terminal or terminal_outcome.run_id != command.run_id:
                raise NativeAgentError(
                    "terminal readback does not match the activation receipt",
                    error_code="terminal_receipt_mismatch",
                )
            outcome = terminal_outcome
        else:
            outcome = checkpoint_outcome
        return cls(
            run_id=command.run_id,
            command_id=command.command_id,
            command_sequence=command.command_sequence,
            command_identity_sha256=command.identity_sha256,
            checkpoint_seq=checkpoint_seq,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_ref=f"checkpoint:{command.run_id}/{checkpoint_seq}",
            state=state,
            boundary_reason=suspension.reason,
            terminal=terminal,
            terminal_ref=f"terminal:{command.run_id}" if terminal else "",
            applied_input_ref=command.applied_input_ref,
            event_cursor=event_cursor,
            stream_cursor=stream_cursor,
            outcome_kind=outcome.kind,
            retry_eligibility=outcome.retry_eligibility,
            error_code=outcome.error_code,
            provider_error_code=outcome.provider_error_code,
            interruption_cause=outcome.interruption_cause,
        )


@dataclass(frozen=True, kw_only=True)
class ActivationRuntime:
    """Exact activation-scoped capabilities a loop factory must bind."""

    run_sink: FencedRunSink
    writer_token: WriterToken
    write_authority: ActivationWriteAuthority
    event_sink: FencedEventSink
    terminal_bridge: FencedTerminalBridge
    event_sequence_seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.writer_token, WriterToken):
            raise TypeError("activation runtime requires WriterToken")
        if not isinstance(self.write_authority, ActivationWriteAuthority):
            raise TypeError("activation runtime requires ActivationWriteAuthority")
        if self.event_sink.write_authority is not self.write_authority:
            raise ValueError("activation runtime event sink authority mismatch")
        if self.terminal_bridge.write_authority is not self.write_authority:
            raise ValueError("activation runtime terminal bridge authority mismatch")
        if self.event_sink.writer_token != self.writer_token:
            raise ValueError("activation runtime event sink token mismatch")
        if self.terminal_bridge.writer_token != self.writer_token:
            raise ValueError("activation runtime terminal bridge token mismatch")
        if self.event_sequence_seed != self.event_sink.last_sequence:
            raise ValueError("activation runtime event cursor mismatch")


@dataclass
class _ActivationRunSink:
    inner: FencedRunSink
    command: ActivationCommand
    writer_token: WriterToken
    write_authority: ActivationWriteAuthority
    event_sink: FencedEventSink

    @property
    def capabilities(self):  # noqa: ANN201 - preserves the inner adapter's exact value
        return self.inner.capabilities

    def commit_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        if writer_token != self.writer_token or checkpoint.run_id != self.command.run_id:
            self.write_authority.revoke()
            self.write_authority.assert_active()
        completed = checkpoint.last_suspension is not None
        checkpoint.active_input = {
            "input_id": self.command.checkpoint_marker,
            "source_seq": self.command.source_checkpoint_seq,
            "phase": "completed" if completed else "running",
        }
        if completed:
            try:
                suspension = suspension_from_checkpoint_payload(checkpoint.last_suspension or {})
                outcome = terminal_outcome_from_suspension(
                    suspension,
                    run_id=self.command.run_id,
                    checkpoint_seq=checkpoint.seq,
                )
            except (TypeError, ValueError, OverflowError, RecursionError) as exc:
                raise NativeAgentError(
                    "activation boundary cannot be projected safely",
                    error_code="invalid_activation_boundary",
                ) from exc
            terminal = checkpoint.terminal or suspension.reason == "terminal"
            checkpoint.applied_input_ids = sorted(
                {*checkpoint.applied_input_ids, self.command.checkpoint_marker}
            )
            receipts = {
                input_id: dict(receipt)
                for input_id, receipt in checkpoint.applied_input_receipts.items()
            }
            receipts[self.command.checkpoint_marker] = {
                "checkpoint_seq": checkpoint.seq,
                "checkpoint_sha256": "",
                "state": state_from_suspension(suspension).value,
                "terminal": terminal,
                "suspension": suspension_checkpoint_payload(suspension),
                "command_identity_sha256": self.command.identity_sha256,
                "event_cursor": self.event_sink.last_sequence,
                "stream_cursor": 0,
                "outcome_kind": outcome.kind,
                "retry_eligibility": outcome.retry_eligibility.value,
                "error_code": outcome.error_code,
                "provider_error_code": outcome.provider_error_code,
                "interruption_cause": (
                    "" if outcome.interruption_cause is None else outcome.interruption_cause.value
                ),
                "terminal_outcome_ref": f"terminal:{self.command.run_id}" if terminal else "",
            }
            checkpoint.applied_input_receipts = receipts
            receipts[self.command.checkpoint_marker]["checkpoint_sha256"] = (
                _checkpoint_receipt_digest(checkpoint, self.command.checkpoint_marker)
            )
        return self.inner.commit_checkpoint(
            checkpoint,
            blobs,
            writer_token=writer_token,
        )

    def latest_checked(self, run_id: str) -> DurableLoadResult[CheckpointRecord]:
        return self.inner.latest_checked(run_id)

    def load_invocation(
        self,
        run_id: str,
        logical_call_id: str,
    ) -> DurableLoadResult[ModelInvocationRecord]:
        return self.inner.load_invocation(run_id, logical_call_id)

    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
        stage_evidence: bool = False,
    ) -> CommitResult:
        return self.inner.commit_invocation(
            invocation,
            blobs,
            writer_token=writer_token,
            stage_evidence=stage_evidence,
        )

    def commit_model_evidence(
        self,
        invocation: DurableModelInvocation,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        return self.inner.commit_model_evidence(invocation, writer_token=writer_token)

    def append_event(self, event: AgentEvent, *, writer_token: WriterToken) -> CommitResult:
        return self.inner.append_event(event, writer_token=writer_token)

    def latest_event_sequence(self, run_id: str) -> int:
        return self.inner.latest_event_sequence(run_id)

    def settle_terminal(
        self,
        outcome: TerminalOutcome,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        return self.inner.settle_terminal(outcome, writer_token=writer_token)

    def read_terminal(self, run_id: str) -> TerminalOutcome | None:
        return self.inner.read_terminal(run_id)


@dataclass
class ActivationDriver:
    """Drive one restored run to one durable, identity-bound boundary."""

    sink: FencedRunSink
    writer_token: WriterToken
    loop_factory: ActivationLoopFactory
    input_resolver: ActivationInputResolver | None = None
    write_authority: ActivationWriteAuthority = field(default_factory=ActivationWriteAuthority)
    local_task_wait_s: float = 300.0
    fault_hook: ActivationFaultHook | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.writer_token, WriterToken):
            raise TypeError("activation driver requires WriterToken")
        if not isinstance(self.write_authority, ActivationWriteAuthority):
            raise TypeError("activation driver requires ActivationWriteAuthority")
        if not callable(self.loop_factory):
            raise TypeError("activation driver loop_factory must be callable")
        if self.input_resolver is not None and not callable(self.input_resolver):
            raise TypeError("activation driver input_resolver must be callable")
        if not math.isfinite(self.local_task_wait_s) or self.local_task_wait_s <= 0:
            raise ValueError("activation driver local_task_wait_s must be positive")

    def drive(self, command: ActivationCommand) -> ActivationReceipt:
        if not isinstance(command, ActivationCommand):
            raise TypeError("activation driver requires ActivationCommand")
        if command.run_id != self.writer_token.run_id:
            self.write_authority.revoke()
            self.write_authority.assert_active()
        stored = self._load_checkpoint(command.run_id)
        checkpoint = stored.checkpoint
        self._validate_checkpoint_run(command, checkpoint)
        if command.checkpoint_marker in checkpoint.applied_input_ids:
            return self._completed_receipt(command, checkpoint)
        continuing = self._validate_new_or_running_command(command, checkpoint)
        terminal = self.write_authority.guard_external_call(
            lambda: self.sink.read_terminal(command.run_id)
        )
        if terminal is not None:
            raise NativeAgentError(
                "terminal run cannot accept another activation",
                error_code="run_terminal",
            )

        event_sequence = self.write_authority.guard_external_call(
            lambda: self.sink.latest_event_sequence(command.run_id)
        )
        _require_nonnegative_integer(event_sequence, "authoritative event sequence")
        user_input = (
            self._resolve_input(command)
            if command.kind == "input" and not continuing
            else None
        )
        event_sink = FencedEventSink(
            self.sink,
            self.writer_token,
            self.write_authority,
            event_sequence,
        )
        run_sink = _ActivationRunSink(
            self.sink,
            command,
            self.writer_token,
            self.write_authority,
            event_sink,
        )
        terminal_bridge = FencedTerminalBridge(
            run_sink,
            self.writer_token,
            self.write_authority,
        )
        runtime = ActivationRuntime(
            run_sink=run_sink,
            writer_token=self.writer_token,
            write_authority=self.write_authority,
            event_sink=event_sink,
            terminal_bridge=terminal_bridge,
            event_sequence_seed=event_sequence,
        )
        self._run_fault_hook("before_restore", command)
        loop = self.write_authority.guard_external_call(lambda: self.loop_factory(command, runtime))
        self._validate_loop(loop, command, runtime)

        durable_boundary = False
        try:
            loop.restore(checkpoint, blobs=stored.blob)
            suspension = self._drive_to_durable_boundary(loop, user_input)
            verified = self._load_checkpoint(command.run_id).checkpoint
            receipt = ActivationReceipt.from_checkpoint(command, verified)
            expected = suspension_checkpoint_payload(suspension)
            raw_receipt = verified.applied_input_receipts.get(command.checkpoint_marker)
            if not isinstance(raw_receipt, Mapping) or raw_receipt.get("suspension") != expected:
                raise NativeAgentError(
                    "returned boundary disagrees with the canonical checkpoint",
                    error_code="activation_boundary_mismatch",
                )
            durable_boundary = True
            self._run_fault_hook("boundary_committed", command)
            if receipt.terminal:
                receipt = self._settle_checkpoint_terminal(command, verified, runtime.terminal_bridge)
                self._run_fault_hook("terminal_committed", command)
            self._run_fault_hook("before_return", command)
            return receipt
        finally:
            if durable_boundary:
                try:
                    loop.release_parked()
                except Exception:
                    try:
                        loop.discard_uncommitted()
                    except Exception:
                        pass
            else:
                try:
                    loop.discard_uncommitted()
                except Exception:
                    pass

    def _completed_receipt(
        self,
        command: ActivationCommand,
        checkpoint: RunCheckpoint,
    ) -> ActivationReceipt:
        receipt = ActivationReceipt.from_checkpoint(command, checkpoint)
        if not receipt.terminal:
            return receipt
        bridge = FencedTerminalBridge(self.sink, self.writer_token, self.write_authority)
        return self._settle_checkpoint_terminal(command, checkpoint, bridge)

    def _settle_checkpoint_terminal(
        self,
        command: ActivationCommand,
        checkpoint: RunCheckpoint,
        bridge: FencedTerminalBridge,
    ) -> ActivationReceipt:
        raw_receipt = checkpoint.applied_input_receipts.get(command.checkpoint_marker)
        raw_suspension = raw_receipt.get("suspension") if isinstance(raw_receipt, Mapping) else None
        if not isinstance(raw_suspension, Mapping):
            raise NativeAgentError(
                "terminal activation has no durable suspension observation",
                error_code="missing_activation_boundary",
            )
        suspension = suspension_from_checkpoint_payload(raw_suspension)
        proposed = terminal_outcome_from_suspension(
            suspension,
            run_id=command.run_id,
            checkpoint_seq=checkpoint.seq,
        )
        settlement = bridge.settle(proposed)
        return ActivationReceipt.from_checkpoint(
            command,
            checkpoint,
            terminal_outcome=settlement.outcome,
        )

    def _load_checkpoint(self, run_id: str) -> CheckpointRecord:
        loaded = self.write_authority.guard_external_call(lambda: self.sink.latest_checked(run_id))
        if loaded.status in {"loaded", "migrated"} and loaded.value is not None:
            return loaded.value
        if loaded.status == "missing":
            raise NativeAgentError(
                "activation source checkpoint is missing",
                error_code="checkpoint_missing",
            )
        raise NativeAgentError(
            "activation source checkpoint cannot be loaded safely",
            error_code=f"checkpoint_{loaded.status}",
        )

    @staticmethod
    def _validate_checkpoint_run(
        command: ActivationCommand,
        checkpoint: RunCheckpoint,
    ) -> None:
        if checkpoint.run_id != command.run_id:
            raise NativeAgentError(
                "activation checkpoint belongs to another run",
                error_code="checkpoint_run_mismatch",
            )

    @staticmethod
    def _validate_new_or_running_command(
        command: ActivationCommand,
        checkpoint: RunCheckpoint,
    ) -> bool:
        active = checkpoint.active_input
        continuing = False
        if active is not None:
            if not isinstance(active, Mapping):
                raise NativeAgentError(
                    "checkpoint active input metadata is invalid",
                    error_code="invalid_active_input",
                )
            active_id = active.get("input_id")
            active_phase = active.get("phase")
            active_source = active.get("source_seq")
            if active_phase == "running":
                if active_id != command.checkpoint_marker:
                    raise NativeAgentError(
                        "another input has an incomplete durable activation",
                        error_code="prior_activation_incomplete",
                    )
                if active_source != command.source_checkpoint_seq:
                    raise NativeAgentError(
                        "activation identity has a different source checkpoint",
                        error_code="activation_identity_mismatch",
                    )
                continuing = True
            elif active_phase != "completed":
                raise NativeAgentError(
                    "checkpoint active input phase is invalid",
                    error_code="invalid_active_input",
                )
        if continuing:
            return True
        if checkpoint.seq != command.source_checkpoint_seq:
            raise NativeAgentError(
                "activation source checkpoint sequence is stale",
                error_code="stale_activation_source",
            )
        if _checkpoint_digest(checkpoint) != command.source_checkpoint_sha256:
            raise NativeAgentError(
                "activation source checkpoint digest does not match",
                error_code="activation_source_mismatch",
            )
        if checkpoint.terminal:
            raise NativeAgentError(
                "terminal checkpoint cannot be activated",
                error_code="run_terminal",
            )
        return False

    def _resolve_input(self, command: ActivationCommand) -> ResolvedActivationInput:
        resolver = self.input_resolver
        if resolver is None:
            raise NativeAgentError(
                "input activation requires an opaque payload resolver",
                error_code="activation_input_resolver_missing",
            )
        try:
            resolved = self.write_authority.guard_external_call(lambda: resolver(command))
        except WriteAuthorityRevoked:
            raise
        except Exception as exc:
            raise NativeAgentError(
                "activation input payload could not be resolved",
                error_code="activation_payload_unavailable",
            ) from exc
        if not isinstance(resolved, ResolvedActivationInput):
            raise TypeError("activation input_resolver must return ResolvedActivationInput")
        if (
            resolved.request_digest != command.request_digest
            or resolved.payload_ref != command.payload_ref
        ):
            raise NativeAgentError(
                "resolved activation input does not match the admitted command",
                error_code="activation_payload_mismatch",
            )
        return resolved

    @staticmethod
    def _validate_loop(
        loop: AgentLoop,
        command: ActivationCommand,
        runtime: ActivationRuntime,
    ) -> None:
        from monoid_agent_kernel.loop import AgentLoop

        if not isinstance(loop, AgentLoop):
            raise TypeError("activation loop_factory must return AgentLoop")
        if loop.spec.run_id != command.run_id:
            raise NativeAgentError(
                "activation loop belongs to another run",
                error_code="loop_run_mismatch",
            )
        if loop.write_authority is not runtime.write_authority:
            raise RuntimeError("activation loop must bind the exact write authority")
        if loop.run_sink is not runtime.run_sink or loop.writer_token != runtime.writer_token:
            raise RuntimeError("activation loop must bind the exact run sink and writer token")
        if sum(sink is runtime.event_sink for sink in loop.authoritative_event_sinks) != 1:
            raise RuntimeError(
                "activation loop must bind the durable event sink exactly once as authoritative"
            )
        if any(sink is runtime.event_sink for sink in loop.event_sinks):
            raise RuntimeError("activation loop cannot duplicate the durable event sink")
        if loop.event_sequence_seed != runtime.event_sequence_seed:
            raise RuntimeError("activation loop must use the authoritative event sequence seed")
        if loop.checkpoint_persist_callback is not None:
            raise RuntimeError("activation loop cannot install a checkpoint callback")
        if loop.emit_output_deltas:
            raise RuntimeError("activation loop model deltas require the durable stream channel")
        if getattr(loop, "_session", None) is not None or getattr(loop, "_finalized", False):
            raise RuntimeError("activation loop_factory must return an unopened fresh loop")

    def _drive_to_durable_boundary(
        self,
        loop: AgentLoop,
        user_input: ResolvedActivationInput | None,
    ) -> Suspension:
        suspension = loop.run_until_suspended(None if user_input is None else user_input.parts)
        deadline = time.monotonic() + self.local_task_wait_s
        while suspension.reason == "awaiting_tasks" and not loop.at_quiescent_park():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("local task did not reach a durable activation boundary")
            if not loop.wait_for_pending_tasks(min(remaining, 0.25)):
                continue
            suspension = loop.run_until_suspended(None)
        return suspension

    def _run_fault_hook(self, phase: str, command: ActivationCommand) -> None:
        if self.fault_hook is not None:
            self.fault_hook(phase, command)


__all__ = [
    "ACTIVATION_COMMAND_SCHEMA_VERSION",
    "ACCEPTED_ACTIVATION_COMMAND_SCHEMA_VERSIONS",
    "ACTIVATION_RECEIPT_SCHEMA_VERSION",
    "ACCEPTED_ACTIVATION_RECEIPT_SCHEMA_VERSIONS",
    "ActivationCommandKind",
    "ActivationCommand",
    "ResolvedActivationInput",
    "ActivationReceipt",
    "ActivationRuntime",
    "ActivationLoopFactory",
    "ActivationFaultHook",
    "ActivationInputResolver",
    "ActivationDriver",
]
