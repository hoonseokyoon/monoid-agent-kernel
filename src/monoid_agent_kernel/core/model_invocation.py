"""Checked durable identity and dispatch state for one logical model call."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, get_args

from monoid_agent_kernel.core.durable_codec import DurableCodec, DurableLoadResult
from monoid_agent_kernel.core.json_ingress import normalize_json_ingress
from monoid_agent_kernel.core.wire_validation import parse_int, parse_literal, parse_str
from monoid_agent_kernel.identifiers import accepted_namespaced_ids, namespaced_id

MODEL_INVOCATION_SCHEMA_VERSION = namespaced_id("model-invocation.v1")
ACCEPTED_MODEL_INVOCATION_SCHEMA_VERSIONS = accepted_namespaced_ids("model-invocation.v1")

DispatchState = Literal["reserved", "dispatch_started", "settled", "unknown"]

# A receipt is normalized metadata. These names disclose model content, transport topology, or a raw
# failure rather than evidence about the call. The check walks the whole receipt so nesting a
# forbidden value under a provider-specific wrapper cannot bypass the boundary.
_FORBIDDEN_RECEIPT_KEYS = frozenset(
    {
        "content",
        "endpoint",
        "error",
        "exception",
        "exception_message",
        "instruction",
        "messages",
        "prompt",
        "raw",
        "raw_exception",
        "raw_exception_message",
        "request",
        "request_body",
        "response",
        "system_prompt",
    }
)
_FORBIDDEN_RECEIPT_KEY_PARTS = frozenset(
    {
        "authorization",
        "body",
        "content",
        "cookie",
        "endpoint",
        "exception",
        "header",
        "headers",
        "instruction",
        "message",
        "messages",
        "prompt",
        "reasoning",
        "replay",
    }
)


def _is_private_receipt_key(key: str) -> bool:
    normalized_key = key.strip().lower().replace("-", "_")
    if normalized_key in _FORBIDDEN_RECEIPT_KEYS or normalized_key.startswith("raw"):
        return True
    return bool(_FORBIDDEN_RECEIPT_KEY_PARTS.intersection(normalized_key.split("_")))


def _require_nonempty_string(value: object, field_name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"model invocation {field_name} must be a non-empty string")


def _require_string(value: object, field_name: str) -> None:
    if type(value) is not str:
        raise ValueError(f"model invocation {field_name} must be a string")


def _require_positive_int(value: object, field_name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"model invocation {field_name} must be a positive integer")


def _normalized_receipt(receipt: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if receipt is None:
        return None
    if not isinstance(receipt, Mapping):
        raise ValueError("model invocation receipt must be an object or null")
    normalized = normalize_json_ingress(
        dict(receipt),
        substitute_nonfinite=False,
    )
    if not isinstance(normalized, dict):  # pragma: no cover - dict input guarantees this shape
        raise ValueError("model invocation receipt must be an object or null")
    try:
        json.dumps(normalized, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("model invocation receipt must be portable JSON") from exc
    pending: list[object] = [normalized]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                if _is_private_receipt_key(key):
                    raise ValueError(
                        f"model invocation receipt cannot contain private field {key!r}"
                    )
                pending.append(child)
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("model invocation receipt numbers must be finite")
    return normalized


@dataclass(frozen=True, kw_only=True)
class DurableModelInvocation:
    """One revision of the durable lifecycle for a logical paid model call."""

    schema_version: str = MODEL_INVOCATION_SCHEMA_VERSION
    run_id: str
    logical_call_id: str
    revision: int
    dispatch_id: str
    dispatch_attempt: int
    idempotency_key: str
    dispatch_state: DispatchState
    request_digest: str
    digest_generation: str
    receipt: Mapping[str, Any] | None = None
    result_ref: str = ""
    failure_code: str = ""

    def __post_init__(self) -> None:
        if self.schema_version not in ACCEPTED_MODEL_INVOCATION_SCHEMA_VERSIONS:
            raise ValueError("unsupported model invocation schema")
        for field_name in (
            "run_id",
            "logical_call_id",
            "dispatch_id",
            "idempotency_key",
            "request_digest",
            "digest_generation",
        ):
            _require_nonempty_string(getattr(self, field_name), field_name)
        _require_positive_int(self.revision, "revision")
        _require_positive_int(self.dispatch_attempt, "dispatch_attempt")
        if self.dispatch_state not in get_args(DispatchState):
            raise ValueError("model invocation dispatch_state is outside the durable vocabulary")
        _require_string(self.result_ref, "result_ref")
        _require_string(self.failure_code, "failure_code")
        receipt = _normalized_receipt(self.receipt)
        object.__setattr__(self, "receipt", receipt)

        if self.dispatch_state in {"reserved", "dispatch_started"}:
            if receipt is not None or self.result_ref or self.failure_code:
                raise ValueError(
                    f"model invocation {self.dispatch_state} cannot carry a receipt, result, or failure"
                )
            return
        if self.dispatch_state == "unknown":
            if receipt is not None or self.result_ref:
                raise ValueError("unknown model invocation cannot carry a receipt or result")
            return
        if receipt is None:
            raise ValueError("settled model invocation must carry a receipt")
        if self.failure_code:
            if self.result_ref:
                raise ValueError("failed settled model invocation cannot carry a result")
        elif not self.result_ref:
            raise ValueError("successful settled model invocation must carry a result_ref")

    def to_json(self) -> dict[str, Any]:
        """Return the canonical writer shape with a detached receipt mapping."""

        return {
            "schema_version": MODEL_INVOCATION_SCHEMA_VERSION,
            "run_id": self.run_id,
            "logical_call_id": self.logical_call_id,
            "revision": self.revision,
            "dispatch_id": self.dispatch_id,
            "dispatch_attempt": self.dispatch_attempt,
            "idempotency_key": self.idempotency_key,
            "dispatch_state": self.dispatch_state,
            "request_digest": self.request_digest,
            "digest_generation": self.digest_generation,
            "receipt": _normalized_receipt(self.receipt),
            "result_ref": self.result_ref,
            "failure_code": self.failure_code,
        }

    @classmethod
    def from_json(cls, payload: object) -> DurableModelInvocation | None:
        """Compatibility wrapper over :func:`decode_model_invocation`."""

        return decode_model_invocation(payload).value


def _model_invocation_from_payload(payload: dict[str, Any]) -> DurableModelInvocation:
    raw_receipt = payload.get("receipt")
    if raw_receipt is not None and not isinstance(raw_receipt, Mapping):
        raise ValueError("model invocation receipt must be an object or null")
    return DurableModelInvocation(
        schema_version=parse_str(payload, "schema_version"),
        run_id=parse_str(payload, "run_id"),
        logical_call_id=parse_str(payload, "logical_call_id"),
        revision=parse_int(payload, "revision"),
        dispatch_id=parse_str(payload, "dispatch_id"),
        dispatch_attempt=parse_int(payload, "dispatch_attempt"),
        idempotency_key=parse_str(payload, "idempotency_key"),
        dispatch_state=parse_literal(  # type: ignore[arg-type]
            payload, "dispatch_state", get_args(DispatchState)
        ),
        request_digest=parse_str(payload, "request_digest"),
        digest_generation=parse_str(payload, "digest_generation"),
        receipt=raw_receipt,
        result_ref=parse_str(payload, "result_ref"),
        failure_code=parse_str(payload, "failure_code"),
    )


MODEL_INVOCATION_CODEC = DurableCodec[DurableModelInvocation](
    family="model-invocation",
    current_schema=MODEL_INVOCATION_SCHEMA_VERSION,
)


def decode_model_invocation(payload: object) -> DurableLoadResult[DurableModelInvocation]:
    """Decode without allowing recursive or cyclic input to escape the checked reader."""

    try:
        normalized = normalize_json_ingress(
            payload,
            substitute_nonfinite=False,
        )
        json.dumps(normalized, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return MODEL_INVOCATION_CODEC.corrupt("model invocation payload is not portable JSON")
    return MODEL_INVOCATION_CODEC.decode(normalized, _model_invocation_from_payload)


__all__ = [
    "MODEL_INVOCATION_SCHEMA_VERSION",
    "ACCEPTED_MODEL_INVOCATION_SCHEMA_VERSIONS",
    "DispatchState",
    "DurableModelInvocation",
    "MODEL_INVOCATION_CODEC",
    "decode_model_invocation",
]
