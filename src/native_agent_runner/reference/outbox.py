"""Reference outbox senders + a generic ``outbox.send`` tool.

These show how the ``OutboxSender`` contract is satisfied at the edge (where the IO and the secret
live) and how a tool stages a durable send. Examples, not part of the supported surface (like the
other ``reference.*`` services).

- ``RecordingOutboxSender`` — records every dispatched request and returns success; a no-IO sender
  for local dev / tests (lets a caller assert what was sent and that the lease handle rode along).
- ``OutboxToolProvider`` — yields a generic ``outbox.send`` tool (capability ``outbox.send``) whose
  handler stages the request via ``ctx.emit_outbox``; bind it with ``runtime.requires_lease`` so the
  loop's capability gate brokers a lease before the send is even staged.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from native_agent_runner.core.outbox import OutboxReceipt, OutboxRequest
from native_agent_runner.core.trace_context import child_traceparent
from native_agent_runner.tools.base import ToolContext, ToolResult, ToolSpec

OUTBOX_SEND_CAPABILITY = "outbox.send"


@dataclass
class RecordingOutboxSender:
    """Records what it was asked to send and reports success. The recorded requests expose the
    ``token_ref`` (the lease handle) so a test can assert the capability handle reached the edge —
    never a secret, which the core never held."""

    sent: list[OutboxRequest] = field(default_factory=list)
    # The child span this sender would attach to each outbound call — derived from the request's
    # traceparent so the dispatch is a child of the staged request's trace. Recorded for assertions.
    child_traceparents: list[str] = field(default_factory=list)

    def send(self, request: OutboxRequest) -> OutboxReceipt:
        self.sent.append(request)
        self.child_traceparents.append(child_traceparent(request.traceparent))
        return OutboxReceipt(ok=True, reference=f"recorded:{request.id}")


@dataclass
class FailingOutboxSender:
    """Always fails — for exercising the retry/dead-letter path. ``retryable`` controls whether the
    drain keeps the request ``pending`` (redispatch) or fails it immediately."""

    reason: str = "sender unavailable"
    retryable: bool = True

    def send(self, request: OutboxRequest) -> OutboxReceipt:
        del request
        return OutboxReceipt(ok=False, error=self.reason, retryable=self.retryable)


class OutboxToolProvider:
    """Yields a generic ``outbox.send`` tool. The handler stages a durable outbound send through the
    tool context; the loop's capability gate (when the binding declares ``requires_lease``) brokers
    an ``outbox.send`` lease first, and the staged request carries that handle for the edge sender."""

    def get_tools(self, context: ToolContext | None = None) -> Iterable[ToolSpec]:
        def handler(ctx: ToolContext, args: dict) -> ToolResult:
            result = ctx.emit_outbox(
                destination=str(args.get("destination") or ""),
                payload=dict(args.get("payload") or {}),
                capability=OUTBOX_SEND_CAPABILITY,
                idempotency_key=str(args.get("idempotency_key") or ""),
                expect_ack=bool(args.get("expect_ack", False)),
                reply_to=str(args.get("reply_to") or ""),
            )
            return ToolResult(ok=True, content=result)

        return [
            ToolSpec(
                id="outbox.send",
                description="Stage a durable outbound send (e.g. an email or webhook). The send is "
                "performed later by the edge; this returns once the request is staged.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "destination": {"type": "string"},
                        "payload": {"type": "object", "additionalProperties": True},
                        "idempotency_key": {"type": "string"},
                        "expect_ack": {"type": "boolean"},
                        "reply_to": {"type": "string"},
                    },
                    "required": ["destination"],
                    "additionalProperties": True,
                },
                capability=OUTBOX_SEND_CAPABILITY,
                side_effect="write",
                handler=handler,
            )
        ]
