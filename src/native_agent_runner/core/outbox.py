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

OUTBOX_REQUEST_VERSION = "native-agent-runner.outbox-request.v1"

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

    def to_json(self) -> dict[str, Any]:
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
        kwargs: dict[str, Any] = {
            "destination": str(payload.get("destination") or ""),
            "payload": dict(payload.get("payload") or {}),
            "capability": str(payload.get("capability") or ""),
            "token_ref": str(payload.get("token_ref") or ""),
            "run_id": str(payload.get("run_id") or ""),
            "idempotency_key": str(payload.get("idempotency_key") or ""),
            "correlation_id": str(payload.get("correlation_id") or ""),
            "causation_id": str(payload.get("causation_id") or ""),
            "traceparent": str(payload.get("traceparent") or ""),
            "tracestate": str(payload.get("tracestate") or ""),
            "created_at": float(payload.get("created_at") or 0.0),
            "status": str(payload.get("status") or "pending"),  # type: ignore[arg-type]
            "attempts": int(payload.get("attempts") or 0),
            "next_attempt_at": float(payload.get("next_attempt_at") or 0.0),
            "reference": str(payload.get("reference") or ""),
            "error": str(payload.get("error") or ""),
        }
        if payload.get("id"):
            kwargs["id"] = str(payload["id"])
        return cls(**kwargs)


@dataclass(frozen=True)
class OutboxReceipt:
    """A sender's outcome for one request. ``retryable`` distinguishes a transient failure (leave the
    request ``pending`` to redrive) from a hard one (mark ``failed`` immediately)."""

    ok: bool
    reference: str = ""
    error: str = ""
    retryable: bool = False


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
        return [r for r in self._requests.values() if r.status == "pending" and r.next_attempt_at <= now]

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
