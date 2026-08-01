"""Outbox request: the agent durably, idempotently emits an external side-effect (send an email,
call a webhook) — without performing the IO in the core.

The symmetric half of the inbox (``core/inbox.py``) and the outbound twin of a capability lease:
when a tool wants to send something externally it **stages** an :class:`OutboxRequest` in a per-run
:class:`Outbox` (append-only, checkpointed) rather than calling out inline. An *edge* relay (the
reference ``RunnerBackend``) drains pending requests through an :class:`OutboxSender` and marks them
dispatched — at-least-once, made effectively-once by the ``idempotency_key`` the external target
honors (the Transactional-Outbox pattern with the checkpoint as the transaction).

Security invariants (mirror the capability lease):
  - the secret never enters the core — a request carries ``token_ref`` (a capability lease handle),
    resolved to the real credential at the edge by the sender;
  - the request is gated by a capability lease before it is ever staged (the binding declares
    ``requires_lease``; the loop's gate brokers it), so egress is least-privilege and revocable.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from monoid_agent_kernel.core.json_ingress import (
    normalize_json_ingress,
    normalize_unicode_scalars,
)
from monoid_agent_kernel.core.runtime_controls import exact_runtime_bool, exact_runtime_string
from monoid_agent_kernel.core.wire_validation import (
    parse_bool,
    parse_float,
    parse_int,
    parse_literal,
    parse_str,
    require_object,
)
from monoid_agent_kernel.identifiers import accepted_namespaced_ids, namespaced_id

OUTBOX_REQUEST_VERSION = namespaced_id("outbox-request.v1")
ACCEPTED_OUTBOX_REQUEST_VERSIONS = accepted_namespaced_ids("outbox-request.v1")

OutboxStatus = Literal["pending", "dispatched", "failed"]


@dataclass
class OutboxRequest:
    """A staged outbound side-effect. ``payload`` is the destination-specific body; ``token_ref`` is
    the capability lease handle the edge sender authenticates with (never the secret). ``status``
    tracks the drain lifecycle; ``idempotency_key`` (defaults to ``id``) is what the external target
    dedupes on so a redelivery after a crash is effectively-once."""

    destination: str
    payload: dict[str, Any] = field(default_factory=dict)
    capability: str = ""
    token_ref: str = ""
    run_id: str = ""
    id: str = field(default_factory=lambda: f"outbox_{uuid.uuid4().hex[:12]}")
    idempotency_key: str = ""
    correlation_id: str = ""
    causation_id: str = ""
    # Ack-back (request-reply): when ``expect_ack`` is set, the edge delivers the send's receipt back
    # to the run as an inbox message correlated by ``correlation_id`` (non-park — the agent observes it
    # on its next activation). ``reply_to`` names the inbox to reply to; empty = the run's own inbox.
    expect_ack: bool = False
    reply_to: str = ""
    # W3C Trace Context (observability only; see core/trace_context.py). Empty until the edge stamps
    # a trace at dispatch; complements correlation/causation, never drives behavior.
    traceparent: str = ""
    tracestate: str = ""
    created_at: float = field(default_factory=time.time)
    status: OutboxStatus = "pending"
    attempts: int = 0
    # Epoch seconds the request is next eligible for dispatch. ``0.0`` = due immediately (a freshly
    # staged request). The edge stamps a future value on a retryable failure (exponential backoff +
    # jitter); the drain only dispatches requests whose time has come. Durable so the schedule
    # survives a restart instead of living in an in-process timer.
    next_attempt_at: float = 0.0
    reference: str = ""  # external id returned by the sender on success
    error: str = ""

    def __post_init__(self) -> None:
        normalize_outbox_request(self)

    def to_json(self) -> dict[str, Any]:
        normalize_outbox_request(self)
        return {
            "protocol": OUTBOX_REQUEST_VERSION,
            "id": self.id,
            "run_id": self.run_id,
            "destination": self.destination,
            "capability": self.capability,
            "payload": dict(self.payload),
            "token_ref": self.token_ref,
            # An empty idempotency key defaults to the request id — its natural dedup handle.
            "idempotency_key": self.idempotency_key or self.id,
            "correlation_id": self.correlation_id or self.id,
            "causation_id": self.causation_id,
            "expect_ack": self.expect_ack,
            "reply_to": self.reply_to,
            "traceparent": self.traceparent,
            "tracestate": self.tracestate,
            "created_at": self.created_at,
            "status": self.status,
            "attempts": self.attempts,
            "next_attempt_at": self.next_attempt_at,
            "reference": self.reference,
            "error": self.error,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> OutboxRequest:
        payload = require_object(payload, "outbox request")
        protocol = parse_str(payload, "protocol")
        if protocol and protocol not in ACCEPTED_OUTBOX_REQUEST_VERSIONS:
            raise ValueError("unsupported outbox request protocol")
        request_payload = (
            require_object(payload["payload"], "payload") if "payload" in payload else {}
        )
        kwargs: dict[str, Any] = {
            "destination": parse_str(payload, "destination"),
            "payload": dict(request_payload),
            "capability": parse_str(payload, "capability"),
            "token_ref": parse_str(payload, "token_ref"),
            "run_id": parse_str(payload, "run_id"),
            "idempotency_key": parse_str(payload, "idempotency_key"),
            "correlation_id": parse_str(payload, "correlation_id"),
            "causation_id": parse_str(payload, "causation_id"),
            "expect_ack": parse_bool(payload, "expect_ack", default=False),
            "reply_to": parse_str(payload, "reply_to"),
            "traceparent": parse_str(payload, "traceparent"),
            "tracestate": parse_str(payload, "tracestate"),
            "created_at": parse_float(payload, "created_at", default=0.0) or 0.0,
            "status": parse_literal(
                payload, "status", ("pending", "dispatched", "failed"), default="pending"
            ),
            "attempts": parse_int(payload, "attempts", default=0),
            "next_attempt_at": parse_float(payload, "next_attempt_at", default=0.0) or 0.0,
            "reference": parse_str(payload, "reference"),
            "error": parse_str(payload, "error"),
        }
        request_id = parse_str(payload, "id")
        if request_id:
            kwargs["id"] = request_id
        return cls(**kwargs)


_OUTBOX_REQUEST_STRING_FIELDS = (
    "destination",
    "capability",
    "token_ref",
    "run_id",
    "id",
    "idempotency_key",
    "correlation_id",
    "causation_id",
    "reply_to",
    "traceparent",
    "tracestate",
    "status",
    "reference",
    "error",
)


def normalize_outbox_request(value: Any) -> OutboxRequest:
    """Normalize one directly constructed request into the durable JSON domain."""

    if not isinstance(value, OutboxRequest):
        raise ValueError("outbox request must be an OutboxRequest")
    normalized_payload = normalize_json_ingress(value.payload)
    if not isinstance(normalized_payload, dict):
        raise ValueError("outbox request payload must be an object")
    value.payload = normalized_payload
    for field_name in _OUTBOX_REQUEST_STRING_FIELDS:
        field_value = exact_runtime_string(
            getattr(value, field_name),
            field_name=f"outbox request {field_name}",
        )
        setattr(value, field_name, normalize_unicode_scalars(field_value))
    value.expect_ack = exact_runtime_bool(
        value.expect_ack,
        field_name="outbox request expect_ack",
    )
    if value.status not in {"pending", "dispatched", "failed"}:
        raise ValueError("outbox request status must be one of: pending, dispatched, failed")

    # These are durable scheduler fields rather than arbitrary model data. Reject non-finite or
    # coercible controls instead of substituting a value that would alter dispatch timing.
    value.created_at = parse_float({"created_at": value.created_at}, "created_at") or 0.0
    value.attempts = parse_int({"attempts": value.attempts}, "attempts")
    value.next_attempt_at = (
        parse_float({"next_attempt_at": value.next_attempt_at}, "next_attempt_at") or 0.0
    )
    return value


@dataclass(frozen=True)
class OutboxReceipt:
    """A sender's outcome for one request. ``retryable`` distinguishes a transient failure (leave the
    request ``pending`` to redrive) from a hard one (mark ``failed`` immediately)."""

    ok: bool
    reference: str = ""
    error: str = ""
    retryable: bool = False

    def __post_init__(self) -> None:
        normalize_outbox_receipt(self)


def normalize_outbox_receipt(value: Any) -> OutboxReceipt:
    """Validate and normalize an edge sender result before it changes durable state."""

    if not isinstance(value, OutboxReceipt):
        raise ValueError("outbox sender must return OutboxReceipt")
    object.__setattr__(
        value,
        "ok",
        exact_runtime_bool(value.ok, field_name="outbox receipt ok"),
    )
    for field_name in ("reference", "error"):
        field_value = exact_runtime_string(
            getattr(value, field_name),
            field_name=f"outbox receipt {field_name}",
        )
        object.__setattr__(value, field_name, normalize_unicode_scalars(field_value))
    object.__setattr__(
        value,
        "retryable",
        exact_runtime_bool(value.retryable, field_name="outbox receipt retryable"),
    )
    return value


@runtime_checkable
class OutboxSender(Protocol):
    """The seam an integrator (an Agent Daemon / Cell edge) implements to actually perform an
    outbound send. The core only ever *stages* a request; the sender resolves ``token_ref`` to the
    real credential and delivers ``payload`` to ``destination``, returning an :class:`OutboxReceipt`.
    Transport-neutral: an in-process notifier, a webhook poster, or a queue producer all fit."""

    def send(self, request: OutboxRequest) -> OutboxReceipt: ...


@dataclass
class Outbox:
    """Per-run, append-only register of outbound requests. Holds handles (``token_ref``), never
    secrets, and is checkpointed in full (a ``pending`` request must survive a restart to be
    (re)dispatched). The engine appends + tracks status; the edge drains."""

    _requests: dict[str, OutboxRequest] = field(default_factory=dict)

    def append(self, request: OutboxRequest) -> OutboxRequest:
        self._requests[request.id] = request
        return request

    def get(self, request_id: str) -> OutboxRequest | None:
        return self._requests.get(request_id)

    def pending(self) -> list[OutboxRequest]:
        """Requests still awaiting (or eligible for re-)dispatch, oldest first. The full pending set
        (regardless of schedule) — used by the snapshot so a not-yet-due request survives a restart."""
        return [r for r in self._requests.values() if r.status == "pending"]

    def due(self, now: float) -> list[OutboxRequest]:
        """Pending requests whose ``next_attempt_at`` has arrived — the drain's dispatch predicate.
        A freshly staged request (``next_attempt_at == 0.0``) is always due."""
        return [
            r for r in self._requests.values() if r.status == "pending" and r.next_attempt_at <= now
        ]

    def mark(
        self,
        request_id: str,
        *,
        status: OutboxStatus,
        attempts: int | None = None,
        next_attempt_at: float | None = None,
        reference: str = "",
        error: str = "",
    ) -> OutboxRequest | None:
        request = self._requests.get(request_id)
        if request is None:
            return None
        request.status = status
        if attempts is not None:
            request.attempts = attempts
        if next_attempt_at is not None:
            request.next_attempt_at = next_attempt_at
        if reference:
            request.reference = reference
        request.error = error
        return request

    def export(self) -> list[dict[str, Any]]:
        """Serialize every request (all statuses) for the checkpoint."""
        return [r.to_json() for r in self._requests.values()]

    def import_(self, payloads: list[dict[str, Any]]) -> None:
        """Rehydrate requests on restore (paired with :meth:`export`)."""
        for payload in payloads:
            request = OutboxRequest.from_json(payload)
            self._requests[request.id] = request
