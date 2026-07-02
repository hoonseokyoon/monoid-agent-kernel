"""Transport-neutral envelope for external agent messages.

The envelope captures the durable message-fabric meaning that sits above ``InboxMessage`` and
``OutboxRequest``: peer identity, idempotency, correlation, causation, trace context, ordered
message parts, and normalized terminal/error state. It deliberately avoids any particular A2A wire
binding. Edges can map this shape to HTTP, JSON-RPC, queues, or an in-process reference sender.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from monoid_agent_kernel.core.inbox import InboxMessage
from monoid_agent_kernel.core.outbox import OutboxRequest
from monoid_agent_kernel.core.trace_context import child_traceparent
from monoid_agent_kernel.identifiers import accepted_namespaced_ids, namespaced_id

EXTERNAL_AGENT_ENVELOPE_VERSION = namespaced_id("external-agent-envelope.v1")
ACCEPTED_EXTERNAL_AGENT_ENVELOPE_VERSIONS = accepted_namespaced_ids(
    "external-agent-envelope.v1"
)


@dataclass(frozen=True)
class ExternalAgentPart:
    """One ordered message or artifact part in an external agent envelope."""

    type: str
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    artifact_id: str = ""
    mime_type: str = ""

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type}
        if self.text:
            payload["text"] = self.text
        if self.data:
            payload["data"] = dict(self.data)
        if self.artifact_id:
            payload["artifact_id"] = self.artifact_id
        if self.mime_type:
            payload["mime_type"] = self.mime_type
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ExternalAgentPart:
        if not isinstance(payload, dict):
            raise ValueError("external agent part must be an object")
        part_type = str(payload.get("type") or "").strip()
        if not part_type:
            raise ValueError("external agent part requires type")
        data_payload = payload["data"] if "data" in payload else {}
        if not isinstance(data_payload, dict):
            raise ValueError("external agent part data must be an object")
        return cls(
            type=part_type,
            text=str(payload.get("text") or ""),
            data=dict(data_payload),
            artifact_id=str(payload.get("artifact_id") or ""),
            mime_type=str(payload.get("mime_type") or ""),
        )


@dataclass(frozen=True)
class ExternalAgentError:
    """Normalized external-agent error state."""

    code: str
    message: str = ""
    retryable: bool = False

    def to_json(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "retryable": self.retryable}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ExternalAgentError:
        if not isinstance(payload, dict):
            raise ValueError("external agent error must be an object")
        code = str(payload.get("code") or "").strip()
        if not code:
            raise ValueError("external agent error requires code")
        return cls(
            code=code,
            message=str(payload.get("message") or ""),
            retryable=bool(payload.get("retryable", False)),
        )


@dataclass(frozen=True)
class ExternalAgentResult:
    """Normalized external-agent terminal result state."""

    state: str
    terminal: bool = False
    interrupted: bool = False
    error: ExternalAgentError | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "state": self.state,
            "terminal": self.terminal,
            "interrupted": self.interrupted,
            "metadata": dict(self.metadata),
        }
        if self.error is not None:
            payload["error"] = self.error.to_json()
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ExternalAgentResult:
        if not isinstance(payload, dict):
            raise ValueError("external agent result must be an object")
        state = str(payload.get("state") or "").strip()
        if not state:
            raise ValueError("external agent result requires state")
        error_payload = payload.get("error")
        if error_payload is not None and not isinstance(error_payload, dict):
            raise ValueError("external agent result error must be an object")
        metadata_payload = payload.get("metadata") or {}
        if not isinstance(metadata_payload, dict):
            raise ValueError("external agent result metadata must be an object")
        return cls(
            state=state,
            terminal=bool(payload.get("terminal", False)),
            interrupted=bool(payload.get("interrupted", False)),
            error=(
                ExternalAgentError.from_json(error_payload)
                if isinstance(error_payload, dict)
                else None
            ),
            metadata=dict(metadata_payload),
        )


@dataclass(frozen=True)
class ExternalAgentEnvelope:
    """One external-agent message with durable routing and observability identity."""

    peer_id: str
    parts: tuple[ExternalAgentPart, ...]
    message_id: str = field(default_factory=lambda: f"ext_msg_{uuid.uuid4().hex[:12]}")
    task_id: str = ""
    request_id: str = ""
    reply_to_id: str = ""
    correlation_id: str = ""
    causation_id: str = ""
    traceparent: str = ""
    tracestate: str = ""
    capability_ref: str = ""
    result: ExternalAgentResult | None = None
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol": EXTERNAL_AGENT_ENVELOPE_VERSION,
            "peer_id": self.peer_id,
            "message_id": self.message_id,
            "task_id": self.task_id or self.correlation_id or self.message_id,
            "request_id": self.request_id,
            "reply_to_id": self.reply_to_id,
            "correlation_id": self.correlation_id or self.message_id,
            "causation_id": self.causation_id,
            "traceparent": self.traceparent,
            "tracestate": self.tracestate,
            "capability_ref": self.capability_ref,
            "parts": [part.to_json() for part in self.parts],
            "result": self.result.to_json() if self.result is not None else None,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ExternalAgentEnvelope:
        if not isinstance(payload, dict):
            raise ValueError("external agent envelope must be an object")
        protocol = payload.get("protocol")
        if protocol not in ACCEPTED_EXTERNAL_AGENT_ENVELOPE_VERSIONS:
            raise ValueError("unsupported external agent envelope protocol")
        peer_id = str(payload.get("peer_id") or "").strip()
        if not peer_id:
            raise ValueError("external agent envelope requires peer_id")
        parts_payload = payload.get("parts")
        if not isinstance(parts_payload, list) or not parts_payload:
            raise ValueError("external agent envelope requires one or more parts")
        parts = tuple(ExternalAgentPart.from_json(part) for part in parts_payload)
        message_id = str(payload.get("message_id") or "").strip()
        if not message_id:
            raise ValueError("external agent envelope requires message_id")
        result_payload = payload.get("result")
        if result_payload is not None and not isinstance(result_payload, dict):
            raise ValueError("external agent envelope result must be an object")
        metadata_payload = payload.get("metadata") or {}
        if not isinstance(metadata_payload, dict):
            raise ValueError("external agent envelope metadata must be an object")
        return cls(
            peer_id=peer_id,
            parts=parts,
            message_id=message_id,
            task_id=str(payload.get("task_id") or ""),
            request_id=str(payload.get("request_id") or ""),
            reply_to_id=str(payload.get("reply_to_id") or ""),
            correlation_id=str(payload.get("correlation_id") or ""),
            causation_id=str(payload.get("causation_id") or ""),
            traceparent=str(payload.get("traceparent") or ""),
            tracestate=str(payload.get("tracestate") or ""),
            capability_ref=str(payload.get("capability_ref") or ""),
            result=(
                ExternalAgentResult.from_json(result_payload)
                if isinstance(result_payload, dict)
                else None
            ),
            created_at=float(payload.get("created_at") or 0.0),
            metadata=dict(metadata_payload),
        )


def validate_external_agent_envelope(payload: dict[str, Any]) -> ExternalAgentEnvelope:
    """Parse and validate one serialized external-agent envelope."""

    return ExternalAgentEnvelope.from_json(payload)


def normalize_external_agent_error(
    error: str | Exception,
    *,
    code: str = "external_agent_error",
    retryable: bool = False,
) -> ExternalAgentError:
    """Return a normalized external-agent error payload."""

    return ExternalAgentError(code=code, message=str(error), retryable=retryable)


def external_agent_envelope_from_outbox_request(
    request: OutboxRequest,
    *,
    peer_id: str = "",
) -> ExternalAgentEnvelope:
    """Build an external-agent envelope from a staged outbox request."""

    payload = dict(request.payload)
    parts = _parts_from_payload(payload)
    message_id = request.idempotency_key or request.id
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    sender_peer_id = (
        peer_id
        or str(metadata.get("peer_id") or metadata.get("source_peer_id") or "").strip()
        or request.run_id
        or request.destination
    )
    return ExternalAgentEnvelope(
        peer_id=sender_peer_id,
        parts=parts,
        message_id=message_id,
        task_id=str(payload.get("task_id") or request.correlation_id or message_id),
        request_id=str(payload.get("request_id") or request.id),
        reply_to_id=str(payload.get("reply_to_id") or request.reply_to),
        correlation_id=request.correlation_id or message_id,
        causation_id=request.causation_id or request.id,
        traceparent=child_traceparent(request.traceparent),
        tracestate=request.tracestate,
        capability_ref=request.token_ref,
        metadata=dict(payload.get("metadata") or {}),
    )


def external_agent_envelope_to_inbox_message(
    envelope: ExternalAgentEnvelope,
    *,
    run_id: str,
    source: str | None = None,
) -> InboxMessage:
    """Convert an external-agent envelope into the backend inbox envelope."""

    return InboxMessage(
        content=_content_from_parts(envelope.parts),
        id=envelope.message_id,
        source=source or f"external-agent:{envelope.peer_id}",
        type="external_agent_message",
        run_id=run_id,
        created_at=envelope.created_at,
        correlation_id=envelope.correlation_id or envelope.message_id,
        causation_id=envelope.causation_id,
        traceparent=envelope.traceparent,
        tracestate=envelope.tracestate,
        metadata={
            "task_id": envelope.task_id,
            "request_id": envelope.request_id,
            "reply_to_id": envelope.reply_to_id,
            "peer_id": envelope.peer_id,
            "result": envelope.result.to_json() if envelope.result is not None else None,
            **dict(envelope.metadata),
        },
    )


def _parts_from_payload(payload: dict[str, Any]) -> tuple[ExternalAgentPart, ...]:
    parts_payload = payload.get("parts")
    if isinstance(parts_payload, list) and parts_payload:
        return tuple(ExternalAgentPart.from_json(part) for part in parts_payload)
    text = str(payload.get("text") or payload.get("message") or "")
    if text:
        return (ExternalAgentPart(type="text", text=text),)
    return (ExternalAgentPart(type="data", data={k: v for k, v in payload.items() if k != "metadata"}),)


def _content_from_parts(parts: tuple[ExternalAgentPart, ...]) -> str | list[dict[str, Any]]:
    if len(parts) == 1 and parts[0].type == "text":
        return parts[0].text
    content: list[dict[str, Any]] = []
    for part in parts:
        if part.type == "text":
            content.append({"type": "text", "text": part.text})
        elif part.type == "data":
            content.append(
                {
                    "type": "text",
                    "text": json.dumps(part.data, ensure_ascii=False, sort_keys=True),
                }
            )
        else:
            content.append(
                {
                    "type": "text",
                    "text": json.dumps(part.to_json(), ensure_ascii=False, sort_keys=True),
                }
            )
    return content
