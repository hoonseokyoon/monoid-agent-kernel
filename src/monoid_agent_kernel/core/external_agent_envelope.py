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
from monoid_agent_kernel.core.json_ingress import (
    is_finite_json_number,
    normalize_json_ingress,
    normalize_unicode_scalars,
)
from monoid_agent_kernel.core.outbox import OutboxRequest
from monoid_agent_kernel.core.trace_context import child_traceparent
from monoid_agent_kernel.core.wire_validation import (
    parse_bool,
    parse_required_str,
    parse_str,
    require_list,
    require_object,
)
from monoid_agent_kernel.identifiers import accepted_namespaced_ids, namespaced_id

EXTERNAL_AGENT_ENVELOPE_VERSION = namespaced_id("external-agent-envelope.v1")
ACCEPTED_EXTERNAL_AGENT_ENVELOPE_VERSIONS = accepted_namespaced_ids(
    "external-agent-envelope.v1"
)
RESERVED_EXTERNAL_AGENT_METADATA_KEYS = frozenset(
    {"peer_id", "task_id", "request_id", "reply_to_id", "result", "traceparent"}
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
        normalized = _normalize_external_agent_part(self)
        payload: dict[str, Any] = {"type": normalized.type}
        if normalized.text:
            payload["text"] = normalized.text
        if normalized.data:
            payload["data"] = dict(normalized.data)
        if normalized.artifact_id:
            payload["artifact_id"] = normalized.artifact_id
        if normalized.mime_type:
            payload["mime_type"] = normalized.mime_type
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ExternalAgentPart:
        payload = _normalized_object(payload, "external agent part")
        part_type = parse_required_str(payload, "type", strip=True)
        data_payload = require_object(payload["data"], "data") if "data" in payload else {}
        return cls(
            type=part_type,
            text=parse_str(payload, "text"),
            data=dict(data_payload),
            artifact_id=parse_str(payload, "artifact_id"),
            mime_type=parse_str(payload, "mime_type"),
        )


@dataclass(frozen=True)
class ExternalAgentError:
    """Normalized external-agent error state."""

    code: str
    message: str = ""
    retryable: bool = False

    def to_json(self) -> dict[str, Any]:
        normalized = _normalize_external_agent_error_value(self)
        return {
            "code": normalized.code,
            "message": normalized.message,
            "retryable": normalized.retryable,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ExternalAgentError:
        payload = _normalized_object(payload, "external agent error")
        return cls(
            code=parse_required_str(payload, "code", strip=True),
            message=parse_str(payload, "message"),
            retryable=parse_bool(payload, "retryable", default=False),
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
        normalized = _normalize_external_agent_result(self)
        payload: dict[str, Any] = {
            "state": normalized.state,
            "terminal": normalized.terminal,
            "interrupted": normalized.interrupted,
            "metadata": dict(normalized.metadata),
        }
        if normalized.error is not None:
            payload["error"] = normalized.error.to_json()
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ExternalAgentResult:
        payload = _normalized_object(payload, "external agent result")
        state = parse_required_str(payload, "state", strip=True)
        error_payload = payload.get("error")
        if error_payload is not None:
            error_payload = require_object(error_payload, "error")
        metadata_payload = require_object(payload["metadata"], "metadata") if "metadata" in payload else {}
        return cls(
            state=state,
            terminal=parse_bool(payload, "terminal", default=False),
            interrupted=parse_bool(payload, "interrupted", default=False),
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
        normalized = normalize_external_agent_envelope(self)
        return {
            "protocol": EXTERNAL_AGENT_ENVELOPE_VERSION,
            "peer_id": normalized.peer_id,
            "message_id": normalized.message_id,
            "task_id": normalized.task_id or normalized.correlation_id or normalized.message_id,
            "request_id": normalized.request_id,
            "reply_to_id": normalized.reply_to_id,
            "correlation_id": normalized.correlation_id or normalized.message_id,
            "causation_id": normalized.causation_id,
            "traceparent": normalized.traceparent,
            "tracestate": normalized.tracestate,
            "capability_ref": normalized.capability_ref,
            "parts": [part.to_json() for part in normalized.parts],
            "result": normalized.result.to_json() if normalized.result is not None else None,
            "created_at": normalized.created_at,
            "metadata": dict(normalized.metadata),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ExternalAgentEnvelope:
        payload = _normalized_object(payload, "external agent envelope")
        protocol = parse_str(payload, "protocol")
        if protocol not in ACCEPTED_EXTERNAL_AGENT_ENVELOPE_VERSIONS:
            raise ValueError("unsupported external agent envelope protocol")
        peer_id = parse_required_str(payload, "peer_id", strip=True)
        parts_payload = require_list(payload.get("parts"), "parts")
        if not parts_payload:
            raise ValueError("external agent envelope requires one or more parts")
        parts = tuple(ExternalAgentPart.from_json(part) for part in parts_payload)
        message_id = parse_required_str(payload, "message_id", strip=True)
        result_payload = payload.get("result")
        if result_payload is not None:
            result_payload = require_object(result_payload, "result")
        metadata_payload = require_object(payload["metadata"], "metadata") if "metadata" in payload else {}
        return cls(
            peer_id=peer_id,
            parts=parts,
            message_id=message_id,
            task_id=parse_str(payload, "task_id"),
            request_id=parse_str(payload, "request_id"),
            reply_to_id=parse_str(payload, "reply_to_id"),
            correlation_id=parse_str(payload, "correlation_id"),
            causation_id=parse_str(payload, "causation_id"),
            traceparent=parse_str(payload, "traceparent"),
            tracestate=parse_str(payload, "tracestate"),
            capability_ref=parse_str(payload, "capability_ref"),
            result=(
                ExternalAgentResult.from_json(result_payload)
                if isinstance(result_payload, dict)
                else None
            ),
            created_at=_finite_number(payload.get("created_at", 0.0), "created_at"),
            metadata=dict(metadata_payload),
        )


def _normalized_object(value: Any, name: str) -> dict[str, Any]:
    normalized = normalize_json_ingress(value)
    return require_object(normalized, name)


def _text(value: Any, field_name: str, *, required: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = normalize_unicode_scalars(value)
    if required and not normalized.strip():
        raise ValueError(f"{field_name} is required")
    return normalized


def _exact_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _finite_number(value: Any, field_name: str) -> float:
    if not is_finite_json_number(value):
        raise ValueError(f"{field_name} must be a finite number")
    return float(value)


def _normalize_external_agent_part(part: Any) -> ExternalAgentPart:
    if not isinstance(part, ExternalAgentPart):
        raise ValueError("external agent parts must contain ExternalAgentPart values")
    data = _normalized_object(part.data, "external agent part data")
    return ExternalAgentPart(
        type=_text(part.type, "external agent part type", required=True).strip(),
        text=_text(part.text, "external agent part text"),
        data=data,
        artifact_id=_text(part.artifact_id, "external agent part artifact_id"),
        mime_type=_text(part.mime_type, "external agent part mime_type"),
    )


def _normalize_external_agent_error_value(error: Any) -> ExternalAgentError:
    if not isinstance(error, ExternalAgentError):
        raise ValueError("external agent error must be an ExternalAgentError")
    return ExternalAgentError(
        code=_text(error.code, "external agent error code", required=True).strip(),
        message=_text(error.message, "external agent error message"),
        retryable=_exact_bool(error.retryable, "external agent error retryable"),
    )


def _normalize_external_agent_result(result: Any) -> ExternalAgentResult:
    if not isinstance(result, ExternalAgentResult):
        raise ValueError("external agent result must be an ExternalAgentResult")
    metadata = _normalized_object(result.metadata, "external agent result metadata")
    return ExternalAgentResult(
        state=_text(result.state, "external agent result state", required=True).strip(),
        terminal=_exact_bool(result.terminal, "external agent result terminal"),
        interrupted=_exact_bool(result.interrupted, "external agent result interrupted"),
        error=(
            _normalize_external_agent_error_value(result.error)
            if result.error is not None
            else None
        ),
        metadata=metadata,
    )


def normalize_external_agent_envelope(envelope: Any) -> ExternalAgentEnvelope:
    """Copy a direct Python envelope into its portable typed domain before use."""

    if not isinstance(envelope, ExternalAgentEnvelope):
        raise ValueError("external agent envelope must be an ExternalAgentEnvelope")
    if not isinstance(envelope.parts, (list, tuple)) or not envelope.parts:
        raise ValueError("external agent envelope requires one or more parts")
    metadata = _normalized_object(envelope.metadata, "external agent envelope metadata")
    return ExternalAgentEnvelope(
        peer_id=_text(envelope.peer_id, "external agent peer_id", required=True).strip(),
        parts=tuple(_normalize_external_agent_part(part) for part in envelope.parts),
        message_id=_text(
            envelope.message_id,
            "external agent message_id",
            required=True,
        ).strip(),
        task_id=_text(envelope.task_id, "external agent task_id"),
        request_id=_text(envelope.request_id, "external agent request_id"),
        reply_to_id=_text(envelope.reply_to_id, "external agent reply_to_id"),
        correlation_id=_text(envelope.correlation_id, "external agent correlation_id"),
        causation_id=_text(envelope.causation_id, "external agent causation_id"),
        traceparent=_text(envelope.traceparent, "external agent traceparent"),
        tracestate=_text(envelope.tracestate, "external agent tracestate"),
        capability_ref=_text(envelope.capability_ref, "external agent capability_ref"),
        result=(
            _normalize_external_agent_result(envelope.result)
            if envelope.result is not None
            else None
        ),
        created_at=_finite_number(envelope.created_at, "external agent created_at"),
        metadata=metadata,
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

    return _normalize_external_agent_error_value(
        ExternalAgentError(code=code, message=str(error), retryable=retryable)
    )


def external_agent_envelope_from_outbox_request(
    request: OutboxRequest,
    *,
    peer_id: str = "",
) -> ExternalAgentEnvelope:
    """Build an external-agent envelope from a staged outbox request."""

    payload = _normalized_object(request.payload, "outbox external-agent payload")
    parts = _parts_from_payload(payload)
    message_id = _text(
        request.idempotency_key or request.id,
        "outbox external-agent message_id",
        required=True,
    )
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    sender_peer_id = _text(
        peer_id or request.run_id or request.destination,
        "outbox external-agent peer_id",
        required=True,
    )
    task_id = payload.get("task_id")
    if task_id is None or task_id == "":
        task_id = request.correlation_id or message_id
    request_id = payload.get("request_id")
    if request_id is None or request_id == "":
        request_id = request.id
    reply_to_id = payload.get("reply_to_id")
    if reply_to_id is None or reply_to_id == "":
        reply_to_id = request.reply_to
    return normalize_external_agent_envelope(ExternalAgentEnvelope(
        peer_id=sender_peer_id,
        parts=parts,
        message_id=message_id,
        task_id=_text(task_id, "outbox external-agent task_id"),
        request_id=_text(request_id, "outbox external-agent request_id"),
        reply_to_id=_text(reply_to_id, "outbox external-agent reply_to_id"),
        correlation_id=request.correlation_id or message_id,
        causation_id=request.causation_id or request.id,
        traceparent=child_traceparent(request.traceparent),
        tracestate=request.tracestate,
        capability_ref=request.token_ref,
        metadata=dict(metadata),
    ))


def external_agent_envelope_to_inbox_message(
    envelope: ExternalAgentEnvelope,
    *,
    run_id: str,
    source: str | None = None,
) -> InboxMessage:
    """Convert an external-agent envelope into the backend inbox envelope."""

    envelope = normalize_external_agent_envelope(envelope)
    normalized_run_id = _text(run_id, "external agent inbox run_id", required=True)
    normalized_source = (
        _text(source, "external agent inbox source", required=True)
        if source is not None
        else f"external-agent:{envelope.peer_id}"
    )
    return InboxMessage(
        content=_content_from_parts(envelope.parts),
        id=envelope.message_id,
        source=normalized_source,
        type="external_agent_message",
        run_id=normalized_run_id,
        created_at=envelope.created_at,
        correlation_id=envelope.correlation_id or envelope.message_id,
        causation_id=envelope.causation_id,
        traceparent=envelope.traceparent,
        tracestate=envelope.tracestate,
        metadata=merge_canonical_metadata(
            envelope.metadata,
            {
                "task_id": envelope.task_id,
                "request_id": envelope.request_id,
                "reply_to_id": envelope.reply_to_id,
                "peer_id": envelope.peer_id,
                "result": envelope.result.to_json() if envelope.result is not None else None,
                "traceparent": envelope.traceparent,
            },
        ),
    )


def merge_canonical_metadata(
    user: dict[str, Any],
    canonical: dict[str, Any],
) -> dict[str, Any]:
    """Merge user metadata with canonical identity fields taking precedence."""

    user = _normalized_object(user, "external agent user metadata")
    canonical = _normalized_object(canonical, "external agent canonical metadata")
    merged = {
        str(key): value
        for key, value in dict(user).items()
        if str(key) not in RESERVED_EXTERNAL_AGENT_METADATA_KEYS
    }
    merged.update(canonical)
    return merged


def _parts_from_payload(payload: dict[str, Any]) -> tuple[ExternalAgentPart, ...]:
    parts_payload = payload.get("parts")
    if isinstance(parts_payload, list) and parts_payload:
        return tuple(ExternalAgentPart.from_json(part) for part in parts_payload)
    text = str(payload.get("text") or payload.get("message") or "")
    if text:
        return (ExternalAgentPart(type="text", text=text),)
    return (ExternalAgentPart(type="data", data={k: v for k, v in payload.items() if k != "metadata"}),)


def _content_from_parts(parts: tuple[ExternalAgentPart, ...]) -> str | list[dict[str, Any]]:
    parts = tuple(_normalize_external_agent_part(part) for part in parts)
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
                "text": json.dumps(
                    part.data,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                ),
                }
            )
        else:
            content.append(
                {
                    "type": "text",
                "text": json.dumps(
                    part.to_json(),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                ),
                }
            )
    return content
