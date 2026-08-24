"""Content-free command admission and finite dispatch primitives for durable hosts."""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, get_args, runtime_checkable

from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.core.json_ingress import is_portable_json_integer
from monoid_agent_kernel.core.model_io import is_recorded_digest
from monoid_agent_kernel.core.safe_evidence import (
    is_safe_opaque_address,
    is_safe_opaque_id,
    is_safe_taxonomy_code,
)
from monoid_agent_kernel.core.wire_validation import (
    parse_int,
    parse_literal,
    parse_required_str,
    parse_str,
    require_object,
    require_only_fields,
)
from monoid_agent_kernel.errors import NativeAgentError
from monoid_agent_kernel.hosting.activation import (
    ActivationCommand,
    ActivationCommandKind,
    ActivationReceipt,
)
from monoid_agent_kernel.hosting.contracts import WriterToken
from monoid_agent_kernel.identifiers import accepted_namespaced_ids, namespaced_id


ADMISSION_REQUEST_SCHEMA_VERSION = namespaced_id("admission-request.v1")
ACCEPTED_ADMISSION_REQUEST_SCHEMA_VERSIONS = accepted_namespaced_ids("admission-request.v1")
ADMITTED_COMMAND_SCHEMA_VERSION = namespaced_id("admitted-command.v1")
ACCEPTED_ADMITTED_COMMAND_SCHEMA_VERSIONS = accepted_namespaced_ids("admitted-command.v1")
ADMISSION_RECEIPT_SCHEMA_VERSION = namespaced_id("admission-receipt.v1")
ACCEPTED_ADMISSION_RECEIPT_SCHEMA_VERSIONS = accepted_namespaced_ids("admission-receipt.v1")

AdmissionState = Literal[
    "prepared",
    "dispatched",
    "activation_claimed",
    "completed",
    "run_terminal",
    "dead_letter",
]
DispatchStatus = Literal["accepted", "retry", "rejected"]

_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "command_id",
        "kind",
        "request_digest",
        "payload_ref",
        "identity_sha256",
    }
)
_COMMAND_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "command_id",
        "command_sequence",
        "kind",
        "request_digest",
        "payload_ref",
        "request_identity_sha256",
        "identity_sha256",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "command",
        "state",
        "attempt_count",
        "dispatch_ref",
        "error_code",
        "activation_command",
        "activation_receipt",
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


class AdmissionConflict(NativeAgentError):
    error_code = "admission_conflict"


class AdmissionRunUnavailable(NativeAgentError):
    error_code = "admission_run_unavailable"


class AdmissionRunTerminal(NativeAgentError):
    error_code = "admission_run_terminal"


class DispatchClaimLost(NativeAgentError):
    error_code = "dispatch_claim_lost"


class ActivationBindingConflict(NativeAgentError):
    error_code = "activation_binding_conflict"


@dataclass(frozen=True, kw_only=True)
class AdmissionRequest:
    """Caller-selected identity and opaque payload evidence for one durable command."""

    run_id: str
    command_id: str
    kind: ActivationCommandKind
    request_digest: str
    payload_ref: str
    schema_version: str = ADMISSION_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in ACCEPTED_ADMISSION_REQUEST_SCHEMA_VERSIONS:
            raise ValueError("unsupported admission request schema")
        if not is_safe_opaque_id(self.run_id) or not is_safe_opaque_id(self.command_id):
            raise ValueError("admission request identities must be bounded opaque ids")
        if type(self.kind) is not str or self.kind not in get_args(ActivationCommandKind):
            raise ValueError("admission request kind is outside the portable vocabulary")
        _require_digest(self.request_digest, "admission request digest")
        if not is_safe_opaque_address(self.payload_ref):
            raise ValueError("admission request payload_ref must be a bounded opaque address")

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(
            {
                "schema_version": ADMISSION_REQUEST_SCHEMA_VERSION,
                "run_id": self.run_id,
                "command_id": self.command_id,
                "kind": self.kind,
                "request_digest": self.request_digest,
                "payload_ref": self.payload_ref,
            }
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": ADMISSION_REQUEST_SCHEMA_VERSION,
            "run_id": self.run_id,
            "command_id": self.command_id,
            "kind": self.kind,
            "request_digest": self.request_digest,
            "payload_ref": self.payload_ref,
            "identity_sha256": self.identity_sha256,
        }

    @classmethod
    def from_json(cls, payload: object) -> AdmissionRequest:
        payload = require_object(payload, "admission request")
        require_only_fields(payload, _REQUEST_FIELDS, "admission request")
        request = cls(
            schema_version=parse_required_str(payload, "schema_version"),
            run_id=parse_required_str(payload, "run_id"),
            command_id=parse_required_str(payload, "command_id"),
            kind=parse_literal(payload, "kind", get_args(ActivationCommandKind)),
            request_digest=parse_required_str(payload, "request_digest"),
            payload_ref=parse_required_str(payload, "payload_ref"),
        )
        if parse_required_str(payload, "identity_sha256") != request.identity_sha256:
            raise ValueError("admission request identity digest mismatch")
        return request


@dataclass(frozen=True, kw_only=True)
class AdmittedCommand:
    """A PostgreSQL-ordered command reference delivered to an orchestrator at least once."""

    run_id: str
    command_id: str
    command_sequence: int
    kind: ActivationCommandKind
    request_digest: str
    payload_ref: str
    request_identity_sha256: str
    schema_version: str = ADMITTED_COMMAND_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in ACCEPTED_ADMITTED_COMMAND_SCHEMA_VERSIONS:
            raise ValueError("unsupported admitted command schema")
        if not is_safe_opaque_id(self.run_id) or not is_safe_opaque_id(self.command_id):
            raise ValueError("admitted command identities must be bounded opaque ids")
        _require_positive_integer(self.command_sequence, "admitted command sequence")
        if type(self.kind) is not str or self.kind not in get_args(ActivationCommandKind):
            raise ValueError("admitted command kind is outside the portable vocabulary")
        _require_digest(self.request_digest, "admitted command request digest")
        _require_digest(self.request_identity_sha256, "admission request identity")
        if not is_safe_opaque_address(self.payload_ref):
            raise ValueError("admitted command payload_ref must be a bounded opaque address")
        if self.request.identity_sha256 != self.request_identity_sha256:
            raise ValueError("admitted command request identity mismatch")

    @property
    def request(self) -> AdmissionRequest:
        return AdmissionRequest(
            run_id=self.run_id,
            command_id=self.command_id,
            kind=self.kind,
            request_digest=self.request_digest,
            payload_ref=self.payload_ref,
        )

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(
            {
                "schema_version": ADMITTED_COMMAND_SCHEMA_VERSION,
                "run_id": self.run_id,
                "command_id": self.command_id,
                "command_sequence": self.command_sequence,
                "kind": self.kind,
                "request_digest": self.request_digest,
                "payload_ref": self.payload_ref,
                "request_identity_sha256": self.request_identity_sha256,
            }
        )

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": ADMITTED_COMMAND_SCHEMA_VERSION,
            "run_id": self.run_id,
            "command_id": self.command_id,
            "command_sequence": self.command_sequence,
            "kind": self.kind,
            "request_digest": self.request_digest,
            "payload_ref": self.payload_ref,
            "request_identity_sha256": self.request_identity_sha256,
            "identity_sha256": self.identity_sha256,
        }

    @classmethod
    def from_request(cls, request: AdmissionRequest, command_sequence: int) -> AdmittedCommand:
        if not isinstance(request, AdmissionRequest):
            raise TypeError("admitted command requires AdmissionRequest")
        return cls(
            run_id=request.run_id,
            command_id=request.command_id,
            command_sequence=command_sequence,
            kind=request.kind,
            request_digest=request.request_digest,
            payload_ref=request.payload_ref,
            request_identity_sha256=request.identity_sha256,
        )

    @classmethod
    def from_json(cls, payload: object) -> AdmittedCommand:
        payload = require_object(payload, "admitted command")
        require_only_fields(payload, _COMMAND_FIELDS, "admitted command")
        command = cls(
            schema_version=parse_required_str(payload, "schema_version"),
            run_id=parse_required_str(payload, "run_id"),
            command_id=parse_required_str(payload, "command_id"),
            command_sequence=parse_int(payload, "command_sequence"),
            kind=parse_literal(payload, "kind", get_args(ActivationCommandKind)),
            request_digest=parse_required_str(payload, "request_digest"),
            payload_ref=parse_required_str(payload, "payload_ref"),
            request_identity_sha256=parse_required_str(
                payload, "request_identity_sha256"
            ),
        )
        if parse_required_str(payload, "identity_sha256") != command.identity_sha256:
            raise ValueError("admitted command identity digest mismatch")
        return command


def _activation_matches_admission(
    activation: ActivationCommand,
    command: AdmittedCommand,
) -> bool:
    return (
        activation.run_id == command.run_id
        and activation.command_id == command.command_id
        and activation.command_sequence == command.command_sequence
        and activation.kind == command.kind
        and activation.request_digest == command.request_digest
        and activation.payload_ref == command.payload_ref
    )


@dataclass(frozen=True, kw_only=True)
class AdmissionReceipt:
    """Current content-free projection of one admitted command and its canonical completion."""

    command: AdmittedCommand
    state: AdmissionState
    attempt_count: int = 0
    dispatch_ref: str = ""
    error_code: str = ""
    activation_command: ActivationCommand | None = None
    activation_receipt: ActivationReceipt | None = None
    schema_version: str = ADMISSION_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in ACCEPTED_ADMISSION_RECEIPT_SCHEMA_VERSIONS:
            raise ValueError("unsupported admission receipt schema")
        if not isinstance(self.command, AdmittedCommand):
            raise TypeError("admission receipt command must be AdmittedCommand")
        if type(self.state) is not str or self.state not in get_args(AdmissionState):
            raise ValueError("admission receipt state is outside the portable vocabulary")
        _require_nonnegative_integer(self.attempt_count, "admission receipt attempt count")
        if self.state in {"dispatched", "activation_claimed", "completed", "dead_letter"} and (
            self.attempt_count < 1
        ):
            raise ValueError("post-claim admission receipt requires a dispatch attempt")
        if self.dispatch_ref and not is_safe_opaque_address(self.dispatch_ref):
            raise ValueError("admission receipt dispatch_ref must be an opaque address")
        if self.error_code and not is_safe_taxonomy_code(self.error_code):
            raise ValueError("admission receipt error_code must be a taxonomy code")
        if self.activation_command is not None:
            if not isinstance(self.activation_command, ActivationCommand) or not (
                _activation_matches_admission(self.activation_command, self.command)
            ):
                raise ValueError("admission receipt activation binding is inconsistent")
        if self.activation_receipt is not None:
            activation = self.activation_command
            receipt = self.activation_receipt
            if (
                activation is None
                or not isinstance(receipt, ActivationReceipt)
                or receipt.run_id != activation.run_id
                or receipt.command_id != activation.command_id
                or receipt.command_sequence != activation.command_sequence
                or receipt.command_identity_sha256 != activation.identity_sha256
                or receipt.checkpoint_seq <= activation.source_checkpoint_seq
                or receipt.checkpoint_ref
                != f"checkpoint:{activation.run_id}/{receipt.checkpoint_seq}"
                or receipt.applied_input_ref != activation.applied_input_ref
                or receipt.terminal_ref
                != (f"terminal:{activation.run_id}" if receipt.terminal else "")
            ):
                raise ValueError("admission receipt completion is inconsistent")
        if self.state == "prepared" and (
            self.dispatch_ref or self.activation_command is not None or self.activation_receipt is not None
        ):
            raise ValueError("prepared admission receipt carries later-state evidence")
        if self.state == "prepared" and self.attempt_count == 0 and self.error_code:
            raise ValueError("unclaimed admission receipt cannot carry a dispatch error")
        if self.state == "dispatched" and (
            not self.dispatch_ref
            or self.error_code
            or self.activation_command is not None
            or self.activation_receipt is not None
        ):
            raise ValueError("dispatched admission receipt evidence is inconsistent")
        if self.state == "activation_claimed" and (
            not self.dispatch_ref
            or self.error_code
            or self.activation_command is None
            or self.activation_receipt is not None
        ):
            raise ValueError("claimed admission receipt evidence is inconsistent")
        if self.state == "completed" and (
            not self.dispatch_ref
            or self.error_code
            or self.activation_command is None
            or self.activation_receipt is None
        ):
            raise ValueError("completed admission receipt evidence is inconsistent")
        if self.state == "run_terminal" and (
            self.error_code != "run_terminal"
            or self.dispatch_ref
            or self.activation_command is not None
            or self.activation_receipt is not None
        ):
            raise ValueError("terminal-run admission receipt evidence is inconsistent")
        if self.state == "dead_letter" and (
            not self.error_code
            or self.dispatch_ref
            or self.activation_command is not None
            or self.activation_receipt is not None
        ):
            raise ValueError("dead-letter admission receipt evidence is inconsistent")

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": ADMISSION_RECEIPT_SCHEMA_VERSION,
            "command": self.command.to_json(),
            "state": self.state,
            "attempt_count": self.attempt_count,
            "dispatch_ref": self.dispatch_ref,
            "error_code": self.error_code,
            "activation_command": (
                None if self.activation_command is None else self.activation_command.to_json()
            ),
            "activation_receipt": (
                None if self.activation_receipt is None else self.activation_receipt.to_json()
            ),
        }

    @classmethod
    def from_json(cls, payload: object) -> AdmissionReceipt:
        payload = require_object(payload, "admission receipt")
        require_only_fields(payload, _RECEIPT_FIELDS, "admission receipt")
        raw_activation = payload.get("activation_command")
        raw_completion = payload.get("activation_receipt")
        return cls(
            schema_version=parse_required_str(payload, "schema_version"),
            command=AdmittedCommand.from_json(payload.get("command")),
            state=parse_literal(payload, "state", get_args(AdmissionState)),
            attempt_count=parse_int(payload, "attempt_count"),
            dispatch_ref=parse_str(payload, "dispatch_ref"),
            error_code=parse_str(payload, "error_code"),
            activation_command=(
                None if raw_activation is None else ActivationCommand.from_json(raw_activation)
            ),
            activation_receipt=(
                None if raw_completion is None else ActivationReceipt.from_json(raw_completion)
            ),
        )


@dataclass(frozen=True, kw_only=True)
class DispatchToken:
    """Exact authority for one dispatch-outbox claim, independent of run writer authority."""

    run_id: str
    command_id: str
    owner_id: str
    claim_id: str
    generation: int

    def __post_init__(self) -> None:
        for field_name in ("run_id", "command_id", "owner_id", "claim_id"):
            if not is_safe_opaque_id(getattr(self, field_name)):
                raise ValueError(f"dispatch token {field_name} must be a bounded opaque id")
        _require_positive_integer(self.generation, "dispatch token generation")


@dataclass(frozen=True, kw_only=True)
class DispatchClaim:
    token: DispatchToken
    command: AdmittedCommand
    attempt: int

    def __post_init__(self) -> None:
        if not isinstance(self.token, DispatchToken) or not isinstance(
            self.command, AdmittedCommand
        ):
            raise TypeError("dispatch claim requires DispatchToken and AdmittedCommand")
        if (
            self.token.run_id != self.command.run_id
            or self.token.command_id != self.command.command_id
        ):
            raise ValueError("dispatch claim token belongs to another command")
        _require_positive_integer(self.attempt, "dispatch claim attempt")


@dataclass(frozen=True, kw_only=True)
class DispatchResult:
    status: DispatchStatus
    dispatch_ref: str = ""
    error_code: str = ""

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in get_args(DispatchStatus):
            raise ValueError("dispatch result status is outside the portable vocabulary")
        if self.dispatch_ref and not is_safe_opaque_address(self.dispatch_ref):
            raise ValueError("dispatch result ref must be an opaque address")
        if self.error_code and not is_safe_taxonomy_code(self.error_code):
            raise ValueError("dispatch result error_code must be a taxonomy code")
        if self.status == "accepted" and (not self.dispatch_ref or self.error_code):
            raise ValueError("accepted dispatch result requires only a dispatch ref")
        if self.status in {"retry", "rejected"} and (self.dispatch_ref or not self.error_code):
            raise ValueError("failed dispatch result requires only an error code")


@runtime_checkable
class CommandTransport(Protocol):
    """Deliver one content-free admitted command to an orchestrator at least once."""

    def dispatch(self, command: AdmittedCommand) -> DispatchResult: ...


@runtime_checkable
class CommandAdmissionStore(Protocol):
    def admit(self, request: AdmissionRequest) -> AdmissionReceipt: ...

    def receipt(self, run_id: str, command_id: str) -> AdmissionReceipt | None: ...

    def bind_activation(
        self,
        command: AdmittedCommand,
        *,
        writer_token: WriterToken,
    ) -> ActivationCommand: ...


@runtime_checkable
class CommandDispatchStore(Protocol):
    def claim_dispatch(
        self,
        owner_id: str,
        claim_id: str,
        *,
        lease_s: float,
    ) -> DispatchClaim | None: ...

    def acknowledge_dispatch(
        self,
        token: DispatchToken,
        result: DispatchResult,
    ) -> AdmissionReceipt: ...

    def retry_dispatch(
        self,
        token: DispatchToken,
        *,
        error_code: str,
        delay_s: float,
    ) -> AdmissionReceipt: ...

    def reject_dispatch(
        self,
        token: DispatchToken,
        *,
        error_code: str,
    ) -> AdmissionReceipt: ...


@dataclass
class CommandOutboxDispatcher:
    """Perform one finite claim/send/settle cycle; the host owns polling and lifecycle."""

    store: CommandDispatchStore
    transport: CommandTransport
    owner_id: str
    lease_s: float = 30.0
    retry_delay_s: Callable[[int], float] = field(
        default=lambda attempt: min(60.0, 2.0 ** min(attempt, 6))
    )
    claim_id_factory: Callable[[], str] = field(
        default=lambda: f"dispatch-{uuid.uuid4().hex}"
    )

    def __post_init__(self) -> None:
        if not isinstance(self.store, CommandDispatchStore):
            raise TypeError("command dispatcher store does not satisfy CommandDispatchStore")
        if not isinstance(self.transport, CommandTransport):
            raise TypeError("command dispatcher transport does not satisfy CommandTransport")
        if not is_safe_opaque_id(self.owner_id):
            raise ValueError("command dispatcher owner_id must be a bounded opaque id")
        if not math.isfinite(self.lease_s) or self.lease_s <= 0:
            raise ValueError("command dispatcher lease_s must be positive")
        if not callable(self.retry_delay_s) or not callable(self.claim_id_factory):
            raise TypeError("command dispatcher factories must be callable")

    def dispatch_once(self) -> AdmissionReceipt | None:
        claim_id = self.claim_id_factory()
        if not is_safe_opaque_id(claim_id):
            raise ValueError("command dispatcher claim factory returned an invalid opaque id")
        claim = self.store.claim_dispatch(
            self.owner_id,
            claim_id,
            lease_s=self.lease_s,
        )
        if claim is None:
            return None
        try:
            result = self.transport.dispatch(claim.command)
        except Exception:
            result = DispatchResult(status="retry", error_code="dispatch_transport_error")
        if not isinstance(result, DispatchResult):
            result = DispatchResult(status="retry", error_code="invalid_dispatch_result")
        if result.status == "accepted":
            return self.store.acknowledge_dispatch(claim.token, result)
        if result.status == "rejected":
            return self.store.reject_dispatch(claim.token, error_code=result.error_code)
        delay_s = self.retry_delay_s(claim.attempt)
        if (
            type(delay_s) not in {int, float}
            or isinstance(delay_s, bool)
            or not math.isfinite(float(delay_s))
            or delay_s < 0
        ):
            raise ValueError("command dispatcher retry delay must be a non-negative finite number")
        return self.store.retry_dispatch(
            claim.token,
            error_code=result.error_code,
            delay_s=float(delay_s),
        )


__all__ = [
    "ADMISSION_REQUEST_SCHEMA_VERSION",
    "ACCEPTED_ADMISSION_REQUEST_SCHEMA_VERSIONS",
    "ADMITTED_COMMAND_SCHEMA_VERSION",
    "ACCEPTED_ADMITTED_COMMAND_SCHEMA_VERSIONS",
    "ADMISSION_RECEIPT_SCHEMA_VERSION",
    "ACCEPTED_ADMISSION_RECEIPT_SCHEMA_VERSIONS",
    "AdmissionState",
    "DispatchStatus",
    "AdmissionConflict",
    "AdmissionRunUnavailable",
    "AdmissionRunTerminal",
    "DispatchClaimLost",
    "ActivationBindingConflict",
    "AdmissionRequest",
    "AdmittedCommand",
    "AdmissionReceipt",
    "DispatchToken",
    "DispatchClaim",
    "DispatchResult",
    "CommandTransport",
    "CommandAdmissionStore",
    "CommandDispatchStore",
    "CommandOutboxDispatcher",
]
