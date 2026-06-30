"""Inbox message envelope (``monoid.inbox-message.v1``).

A message entering a run (a user follow-up, a control-plane send, …) is wrapped in this envelope
so it carries **provenance** (who sent it, which logical flow) and a stable **`id`** that makes
ingress idempotent: a redelivered message (a network retry) is processed once, and the envelope
survives a checkpoint/restart instead of decaying to bare content. The shape follows CloudEvents
(``id`` + ``source`` uniquely identify an event; ``type``/``time``/``subject`` context) plus the
correlation/causation pair that links a request to its eventual result across a durable boundary.

This is a transport contract owned by the *edge* (the reference ``RunnerBackend`` wraps inbound
content into it and dedups on ``id``); the engine (``AgentLoop``) never sees the envelope — it
still receives the unwrapped ``content`` via ``submit``. ``content`` is kept JSON-native (a ``str``
or a ``list`` of content-part dicts) so the envelope round-trips through the message queue and the
checkpoint with no dataclass (de)serialization, exactly like the raw form it replaces.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from monoid_agent_kernel.identifiers import accepted_namespaced_ids, namespaced_id

INBOX_PROTOCOL_VERSION = namespaced_id("inbox-message.v1")
ACCEPTED_INBOX_PROTOCOL_VERSIONS = accepted_namespaced_ids("inbox-message.v1")


@dataclass(frozen=True)
class InboxMessage:
    """One message entering a run, with provenance + an idempotency key.

    ``id`` is the dedup key — a client/control plane should supply a stable id (echoing it on a
    retry) so a redelivery is recognized; absent one, the edge mints a uuid (then only duplicates
    still in flight together dedup). ``correlation_id`` groups a whole flow (defaults to ``id`` for
    a root message); ``causation_id`` is the id of the message that directly caused this one (empty
    for a root). ``content`` is the JSON-native payload the loop ultimately submits."""

    content: str | list[dict[str, Any]]
    id: str = field(default_factory=lambda: f"inbox_{uuid.uuid4().hex[:12]}")
    source: str = "api"
    type: str = "user_message"
    run_id: str = ""
    created_at: float = field(default_factory=time.time)
    correlation_id: str = ""
    causation_id: str = ""
    # W3C Trace Context (observability only; never drives behavior). Complements correlation/causation
    # — see core/trace_context.py. Empty when the caller propagated no trace.
    traceparent: str = ""
    tracestate: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": INBOX_PROTOCOL_VERSION,
            "id": self.id,
            "source": self.source,
            "type": self.type,
            "run_id": self.run_id,
            "created_at": self.created_at,
            # An empty correlation defaults to this message's own id — it is the root of a flow.
            "correlation_id": self.correlation_id or self.id,
            "causation_id": self.causation_id,
            "traceparent": self.traceparent,
            "tracestate": self.tracestate,
            "metadata": dict(self.metadata),
            "content": self.content,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> InboxMessage:
        kwargs: dict[str, Any] = {
            "content": payload.get("content"),
            "source": str(payload.get("source") or "api"),
            "type": str(payload.get("type") or "user_message"),
            "run_id": str(payload.get("run_id") or ""),
            "created_at": float(payload.get("created_at") or 0.0),
            "correlation_id": str(payload.get("correlation_id") or ""),
            "causation_id": str(payload.get("causation_id") or ""),
            "traceparent": str(payload.get("traceparent") or ""),
            "tracestate": str(payload.get("tracestate") or ""),
            "metadata": dict(payload.get("metadata") or {}),
        }
        if payload.get("id"):
            kwargs["id"] = str(payload["id"])
        return cls(**kwargs)


def is_inbox_envelope(obj: Any) -> bool:
    """True if ``obj`` is a serialized :class:`InboxMessage` (vs. a legacy raw ``str``/``list``
    queue entry or a queue sentinel). The discriminator the queue/checkpoint use to decide whether
    to unwrap an envelope or pass content through unchanged."""
    return isinstance(obj, dict) and obj.get("protocol") in ACCEPTED_INBOX_PROTOCOL_VERSIONS
