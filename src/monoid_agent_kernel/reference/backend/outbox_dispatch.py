from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from monoid_agent_kernel.core.checkpoint import CheckpointStore, RunCheckpoint
from monoid_agent_kernel.core.inbox import InboxMessage
from monoid_agent_kernel.core.outbox import OutboxReceipt, OutboxRequest
from monoid_agent_kernel.core.trace_context import new_traceparent
from monoid_agent_kernel.reference.backend.ports import MutableRunRecordPort, queued_message_snapshot


class OutboxLoopPort(Protocol):
    def due_outbox(self, now: float) -> list[OutboxRequest]: ...

    def record_outbox_result(
        self,
        request_id: str,
        receipt: OutboxReceipt,
        *,
        max_attempts: int = 5,
        next_attempt_at: float = 0.0,
    ) -> str: ...

    def snapshot(self) -> RunCheckpoint | None: ...

    def collect_checkpoint_blobs(self) -> Mapping[str, bytes]: ...


@dataclass(frozen=True)
class OutboxRetryPolicy:
    max_attempts: int
    base_s: float
    factor: float
    cap_s: float


@dataclass(frozen=True)
class OutboxDispatchContext:
    retry_policy_provider: Callable[[], OutboxRetryPolicy]
    max_message_queue_depth_provider: Callable[[], int]
    checkpoint_store_provider: Callable[[], CheckpointStore]
    rng_provider: Callable[[], random.Random]
    live_outbox_runs: Callable[[], list[tuple[MutableRunRecordPort, OutboxLoopPort]]]
    call_soon: Callable[..., None]
    record_terminal: Callable[[MutableRunRecordPort], bool]


class OutboxDispatchService:
    """Reference backend edge dispatcher for staged outbox side effects."""

    def __init__(self, context: OutboxDispatchContext) -> None:
        self._context = context

    def backoff_delay(self, attempts: int) -> float:
        """Capped exponential backoff with full jitter."""
        policy = self._context.retry_policy_provider()
        ceiling = min(policy.cap_s, policy.base_s * (policy.factor**attempts))
        return self._context.rng_provider().uniform(0.0, max(0.0, ceiling))

    def drain_outbox(self, record: MutableRunRecordPort, loop: OutboxLoopPort) -> None:
        """Dispatch due staged outbox requests and persist the resulting checkpoint state."""
        sender = record.outbox_sender
        now = time.time()
        due = loop.due_outbox(now)
        if sender is None or not due:
            return
        changed = False
        policy = self._context.retry_policy_provider()
        for request in due:
            if not request.traceparent:
                request.traceparent = new_traceparent()
            try:
                receipt = sender.send(request)
            except Exception as exc:  # a sender raising is a retryable transport failure
                receipt = OutboxReceipt(ok=False, error=str(exc), retryable=True)
            next_attempt_at = now + self.backoff_delay(request.attempts + 1)
            status = loop.record_outbox_result(
                request.id,
                receipt,
                max_attempts=policy.max_attempts,
                next_attempt_at=next_attempt_at,
            )
            changed = True
            if request.expect_ack and status in {"dispatched", "failed"}:
                self.stage_outbox_ack(record, request, status, receipt)
        if changed:
            checkpoint = loop.snapshot()
            if checkpoint is not None:
                checkpoint.queued_messages = queued_message_snapshot(record.message_queue)
                checkpoint.inbox_seen_ids = sorted(record.seen_inbox_ids)
                self._context.checkpoint_store_provider().put(checkpoint, loop.collect_checkpoint_blobs())

    def stage_outbox_ack(
        self, record: MutableRunRecordPort, request: Any, status: str, receipt: OutboxReceipt
    ) -> None:
        """Deliver an outbox send receipt back into the run inbox as a correlated message."""
        ack_id = f"ack_{request.id}"
        if self._context.record_terminal(record):
            return
        if (
            ack_id in record.seen_inbox_ids
            or record.message_queue.qsize() >= self._context.max_message_queue_depth_provider()
        ):
            return
        summary = f"[outbox-ack] request {request.id} to {request.destination!r}: {status}"
        if receipt.reference:
            summary += f" (ref={receipt.reference})"
        if receipt.error:
            summary += f" (error={receipt.error})"
        envelope = InboxMessage(
            content=summary,
            id=ack_id,
            source="outbox",
            type="outbox_ack",
            run_id=record.run_id,
            correlation_id=request.correlation_id or request.id,
            causation_id=request.id,
            traceparent=request.traceparent,
            tracestate=request.tracestate,
        )
        record.message_queue.put_nowait(envelope.to_json())

    def redrive_outbox(self) -> None:
        """Schedule due outbox drains for live runs with senders."""
        for record, loop in self._context.live_outbox_runs():
            self._context.call_soon(self.drain_outbox, record, loop)
