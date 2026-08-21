"""Portable terminal outcome vocabulary for durable hosts and projections."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, get_args

from monoid_agent_kernel.core.json_ingress import is_portable_json_integer
from monoid_agent_kernel.core.safe_evidence import (
    is_safe_opaque_address,
    is_safe_opaque_id,
    is_safe_taxonomy_code,
)
from monoid_agent_kernel.core.wire_validation import (
    parse_int,
    parse_literal,
    parse_str,
    require_object,
    require_only_fields,
)
from monoid_agent_kernel.identifiers import accepted_namespaced_ids, namespaced_id

TERMINAL_OUTCOME_SCHEMA_VERSION = namespaced_id("terminal-outcome.v1")
ACCEPTED_TERMINAL_OUTCOME_SCHEMA_VERSIONS = accepted_namespaced_ids("terminal-outcome.v1")

TerminalOutcomeKind = Literal[
    "completed",
    "paused",
    "cancelled",
    "interrupted",
    "failed_retryable",
    "failed_config",
    "failed_terminal",
    "dispatch_unknown",
    "evidence_uncommitted",
]

_TERMINAL_OUTCOME_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "kind",
        "retry_eligibility",
        "interruption_cause",
        "checkpoint_seq",
        "final_output_ref",
        "partial_output_ref",
        "last_evidence_ref",
        "error_code",
        "provider_error_code",
        "http_status",
    }
)


class RetryEligibility(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    SAFE = "safe"
    AFTER_CONFIGURATION = "after_configuration"
    AFTER_RECONCILIATION = "after_reconciliation"
    FORBIDDEN = "forbidden"


class InterruptionCause(StrEnum):
    USER_CANCEL = "user_cancel"
    GRACEFUL_DRAIN = "graceful_drain"
    LEASE_LOST = "lease_lost"
    DEADLINE = "deadline"
    HOST_SHUTDOWN = "host_shutdown"
    PROVIDER_FAILURE = "provider_failure"
    VALIDATION_FAILURE = "validation_failure"
    UNKNOWN = "unknown"


def _require_optional_nonnegative_int(value: object, field_name: str) -> None:
    if value is None:
        return
    if not is_portable_json_integer(value) or value < 0:
        raise ValueError(f"terminal outcome {field_name} must be a non-negative integer or null")


@dataclass(frozen=True, kw_only=True)
class TerminalOutcome:
    """Content-free, provider-neutral final meaning of one run.

    Output and evidence fields are opaque references. Prompt text, model output, reasoning,
    replay payloads, and raw provider exceptions have no field in this record.
    """

    schema_version: str = TERMINAL_OUTCOME_SCHEMA_VERSION
    run_id: str
    kind: TerminalOutcomeKind
    retry_eligibility: RetryEligibility
    interruption_cause: InterruptionCause | None = None
    checkpoint_seq: int | None = None
    final_output_ref: str = ""
    partial_output_ref: str = ""
    last_evidence_ref: str = ""
    error_code: str = ""
    provider_error_code: str = ""
    http_status: int | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in ACCEPTED_TERMINAL_OUTCOME_SCHEMA_VERSIONS:
            raise ValueError("unsupported terminal outcome schema")
        if not is_safe_opaque_id(self.run_id):
            raise ValueError("terminal outcome run_id must be a bounded opaque id")
        if type(self.kind) is not str or self.kind not in get_args(TerminalOutcomeKind):
            raise ValueError("terminal outcome kind is outside the portable vocabulary")
        try:
            retry_eligibility = RetryEligibility(self.retry_eligibility)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "terminal outcome retry_eligibility is outside the portable vocabulary"
            ) from exc
        object.__setattr__(self, "retry_eligibility", retry_eligibility)
        if self.kind == "dispatch_unknown" and retry_eligibility not in {
            RetryEligibility.AFTER_RECONCILIATION,
            RetryEligibility.FORBIDDEN,
        }:
            raise ValueError(
                "terminal outcome dispatch_unknown requires reconciliation or forbids retry"
            )
        if self.interruption_cause is not None:
            try:
                interruption_cause = InterruptionCause(self.interruption_cause)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "terminal outcome interruption_cause is outside the portable vocabulary"
                ) from exc
            object.__setattr__(self, "interruption_cause", interruption_cause)
        _require_optional_nonnegative_int(self.checkpoint_seq, "checkpoint_seq")
        if self.http_status is not None and (
            type(self.http_status) is not int or not 100 <= self.http_status <= 599
        ):
            raise ValueError("terminal outcome http_status must be between 100 and 599 or null")
        for field_name in (
            "final_output_ref",
            "partial_output_ref",
            "last_evidence_ref",
        ):
            value = getattr(self, field_name)
            if type(value) is not str:
                raise ValueError(f"terminal outcome {field_name} must be a string")
            if value and not is_safe_opaque_address(value):
                raise ValueError(
                    f"terminal outcome {field_name} must be empty or a bounded opaque address"
                )
        for field_name in ("error_code", "provider_error_code"):
            value = getattr(self, field_name)
            if type(value) is not str:
                raise ValueError(f"terminal outcome {field_name} must be a string")
            if value and not is_safe_taxonomy_code(value):
                raise ValueError(
                    f"terminal outcome {field_name} must be empty or a bounded taxonomy code"
                )

    def to_json(self) -> dict[str, Any]:
        """Return the one canonical writer spelling of the outcome."""

        return {
            "schema_version": TERMINAL_OUTCOME_SCHEMA_VERSION,
            "run_id": self.run_id,
            "kind": self.kind,
            "retry_eligibility": self.retry_eligibility.value,
            "interruption_cause": (
                None if self.interruption_cause is None else self.interruption_cause.value
            ),
            "checkpoint_seq": self.checkpoint_seq,
            "final_output_ref": self.final_output_ref,
            "partial_output_ref": self.partial_output_ref,
            "last_evidence_ref": self.last_evidence_ref,
            "error_code": self.error_code,
            "provider_error_code": self.provider_error_code,
            "http_status": self.http_status,
        }

    @classmethod
    def from_json(cls, payload: object) -> TerminalOutcome:
        payload = require_object(payload, "terminal outcome")
        require_only_fields(payload, _TERMINAL_OUTCOME_FIELDS, "terminal outcome")
        schema_version = parse_str(payload, "schema_version")
        if schema_version not in ACCEPTED_TERMINAL_OUTCOME_SCHEMA_VERSIONS:
            raise ValueError("unsupported terminal outcome schema")
        raw_cause = payload.get("interruption_cause")
        cause = (
            None
            if raw_cause is None
            else InterruptionCause(parse_str(payload, "interruption_cause"))
        )
        raw_checkpoint_seq = payload.get("checkpoint_seq")
        raw_http_status = payload.get("http_status")
        return cls(
            schema_version=schema_version,
            run_id=parse_str(payload, "run_id"),
            kind=parse_literal(payload, "kind", get_args(TerminalOutcomeKind)),  # type: ignore[arg-type]
            retry_eligibility=RetryEligibility(parse_str(payload, "retry_eligibility")),
            interruption_cause=cause,
            checkpoint_seq=(
                None
                if raw_checkpoint_seq is None
                else parse_int(payload, "checkpoint_seq")
            ),
            final_output_ref=parse_str(payload, "final_output_ref"),
            partial_output_ref=parse_str(payload, "partial_output_ref"),
            last_evidence_ref=parse_str(payload, "last_evidence_ref"),
            error_code=parse_str(payload, "error_code"),
            provider_error_code=parse_str(payload, "provider_error_code"),
            http_status=None if raw_http_status is None else parse_int(payload, "http_status"),
        )


__all__ = [
    "TERMINAL_OUTCOME_SCHEMA_VERSION",
    "ACCEPTED_TERMINAL_OUTCOME_SCHEMA_VERSIONS",
    "TerminalOutcomeKind",
    "RetryEligibility",
    "InterruptionCause",
    "TerminalOutcome",
]
