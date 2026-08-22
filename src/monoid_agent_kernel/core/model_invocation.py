"""Checked durable identity and dispatch state for one logical model call."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, get_args

from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.core.durable_codec import DurableCodec, DurableLoadResult
from monoid_agent_kernel.core.json_ingress import (
    is_finite_json_number,
    is_portable_json_integer,
    normalize_json_ingress,
)
from monoid_agent_kernel.core.model_io import is_recorded_digest, is_valid_idempotency_key
from monoid_agent_kernel.core.safe_evidence import (
    is_safe_opaque_address,
    is_safe_opaque_id,
    is_safe_taxonomy_code,
    is_safe_utc_timestamp,
)
from monoid_agent_kernel.core.wire_validation import (
    parse_int,
    parse_literal,
    parse_str,
    require_only_fields,
)
from monoid_agent_kernel.identifiers import accepted_namespaced_ids, namespaced_id

MODEL_INVOCATION_SCHEMA_VERSION = namespaced_id("model-invocation.v1")
ACCEPTED_MODEL_INVOCATION_SCHEMA_VERSIONS = accepted_namespaced_ids("model-invocation.v1")
MODEL_REQUEST_DIGEST_GENERATION = namespaced_id("model-request-digest.v1")
ACCEPTED_MODEL_REQUEST_DIGEST_GENERATIONS = accepted_namespaced_ids("model-request-digest.v1")
LOGICAL_MODEL_CALL_GENERATION = namespaced_id("logical-model-call.v1")
MODEL_DISPATCH_GENERATION = namespaced_id("model-dispatch.v1")

DispatchState = Literal["reserved", "dispatch_started", "settled", "unknown"]

_MODEL_INVOCATION_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "logical_call_id",
        "revision",
        "dispatch_id",
        "dispatch_attempt",
        "idempotency_key",
        "dispatch_state",
        "request_digest",
        "digest_generation",
        "receipt",
        "result_ref",
        "failure_code",
    }
)

# Receipt v1 is a closed evidence vocabulary. Unknown provider metadata is private by default;
# expanding the public durable record requires an explicit contract change rather than another
# spelling-specific deny rule. Keys are matched without case or separators and emitted canonically.
_RECEIPT_FIELDS = {
    "attempts": "attempts",
    "durationms": "duration_ms",
    "finishreason": "finish_reason",
    "httpstatus": "http_status",
    "latencyms": "latency_ms",
    "providererrorcode": "provider_error_code",
    "providerrequestid": "provider_request_id",
    "providerresponseid": "provider_response_id",
    "providerretried": "provider_retried",
    "requestdigest": "request_digest",
    "requestid": "request_id",
    "responseid": "response_id",
    "retryable": "retryable",
    "settledat": "settled_at",
    "startedat": "started_at",
    "stopreason": "stop_reason",
    "systemfingerprint": "system_fingerprint",
    "usage": "usage",
}
_RECEIPT_OPAQUE_ID_FIELDS = frozenset(
    {
        "provider_request_id",
        "provider_response_id",
        "request_id",
        "response_id",
        "system_fingerprint",
    }
)
_RECEIPT_CODE_FIELDS = frozenset(
    {
        "finish_reason",
        "provider_error_code",
        "stop_reason",
    }
)
_RECEIPT_TIMESTAMP_FIELDS = frozenset(
    {
        "settled_at",
        "started_at",
    }
)
_RECEIPT_BOOLEAN_FIELDS = frozenset({"provider_retried", "retryable"})
_RECEIPT_DURATION_FIELDS = frozenset({"duration_ms", "latency_ms"})
_USAGE_FIELDS = {
    "audioinputtokens": "audio_input_tokens",
    "audiooutputtokens": "audio_output_tokens",
    "cachedinputtokens": "cached_input_tokens",
    "cachereadtokens": "cache_read_tokens",
    "cachewritetokens": "cache_write_tokens",
    "inputtokens": "input_tokens",
    "outputtokens": "output_tokens",
    "reasoningtokens": "reasoning_tokens",
    "totaltokens": "total_tokens",
}

MODEL_INVOCATION_RECEIPT_FIELDS = frozenset(_RECEIPT_FIELDS.values())
"""Canonical public fields accepted by the durable model-invocation receipt."""

MODEL_INVOCATION_RECEIPT_USAGE_FIELDS = frozenset(_USAGE_FIELDS.values())
"""Canonical token counters accepted inside ``receipt.usage``."""


def logical_model_call_id(run_id: str, step_id: str) -> str:
    """Derive the content-free durable address of one AgentLoop model call."""

    if not is_safe_opaque_id(run_id):
        raise ValueError("logical model call run_id must be a bounded opaque id")
    if not is_safe_opaque_id(step_id):
        raise ValueError("logical model call step_id must be a bounded opaque id")
    digest = canonical_sha256(
        {
            "generation": LOGICAL_MODEL_CALL_GENERATION,
            "run_id": run_id,
            "step_id": step_id,
        }
    )
    return f"mcall_{digest}"


def model_dispatch_id(logical_call_id: str, dispatch_attempt: int) -> str:
    """Derive a distinct durable dispatch address for one logical-call attempt."""

    if not is_safe_opaque_id(logical_call_id):
        raise ValueError("model dispatch logical_call_id must be a bounded opaque id")
    _require_positive_int(dispatch_attempt, "dispatch attempt")
    digest = canonical_sha256(
        {
            "generation": MODEL_DISPATCH_GENERATION,
            "logical_call_id": logical_call_id,
            "dispatch_attempt": dispatch_attempt,
        }
    )
    return f"mdispatch_{digest}"


def model_invocation_receipt(receipt: Any) -> dict[str, Any]:
    """Project a settled runner receipt into the public-safe durable vocabulary.

    The invocation journal is public evidence. It keeps counts and taxonomy while omitting
    invocation context, model configuration, destinations, prompt identity, redaction details,
    and the per-attempt log. The result passes through ``_normalized_receipt`` before returning,
    so this helper cannot produce a shape the durable record later refuses.
    """

    usage: dict[str, int] = {}
    raw_usage = getattr(receipt, "usage", {})
    if isinstance(raw_usage, Mapping):
        for key, value in raw_usage.items():
            if type(key) is not str or not is_portable_json_integer(value) or value < 0:
                continue
            canonical_key = _USAGE_FIELDS.get(_collapsed_receipt_key(key))
            if canonical_key is not None:
                usage[canonical_key] = value

    projected: dict[str, Any] = {
        "attempts": getattr(receipt, "attempts", 0),
        "latency_ms": getattr(receipt, "latency_ms", 0),
        "provider_retried": getattr(receipt, "provider_retried", False),
        "retryable": getattr(receipt, "retryable", False),
        "request_digest": getattr(receipt, "request_digest", ""),
        "usage": usage,
    }
    for source, target in (
        ("stop_reason", "stop_reason"),
        ("provider_error_code", "provider_error_code"),
    ):
        value = getattr(receipt, source, "")
        if is_safe_taxonomy_code(value):
            projected[target] = value
    http_status = getattr(receipt, "http_status", None)
    if type(http_status) is int and 100 <= http_status <= 599:
        projected["http_status"] = http_status
    normalized = _normalized_receipt(projected)
    if normalized is None:  # pragma: no cover - the projection above is always a mapping
        raise AssertionError("durable model invocation receipt projection disappeared")
    return normalized


def _collapsed_receipt_key(key: str) -> str:
    return "".join(character for character in key.casefold() if character.isalnum())


def _require_string(value: object, field_name: str) -> None:
    if type(value) is not str:
        raise ValueError(f"model invocation {field_name} must be a string")


def _require_positive_int(value: object, field_name: str) -> None:
    if not is_portable_json_integer(value) or value < 1:
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
    canonical: dict[str, Any] = {}
    for key, value in normalized.items():
        canonical_key = _RECEIPT_FIELDS.get(_collapsed_receipt_key(key))
        if canonical_key is None:
            raise ValueError(
                f"model invocation receipt cannot contain private field {key!r}; "
                "the field is outside the safe evidence vocabulary"
            )
        if canonical_key in canonical:
            raise ValueError("model invocation receipt fields collide after canonicalization")
        if canonical_key == "usage":
            if not isinstance(value, dict):
                raise ValueError("model invocation receipt usage must be an object")
            usage: dict[str, int] = {}
            for usage_key, count in value.items():
                canonical_usage_key = _USAGE_FIELDS.get(_collapsed_receipt_key(usage_key))
                if canonical_usage_key is None:
                    raise ValueError(
                        "model invocation receipt usage contains a private field outside the "
                        "safe evidence vocabulary"
                    )
                if canonical_usage_key in usage:
                    raise ValueError(
                        "model invocation receipt usage fields collide after canonicalization"
                    )
                if not is_portable_json_integer(count) or count < 0:
                    raise ValueError(
                        "model invocation receipt usage values must be non-negative integers"
                    )
                usage[canonical_usage_key] = count
            canonical[canonical_key] = usage
        elif canonical_key == "request_digest":
            if type(value) is not str or not is_recorded_digest(value):
                raise ValueError(
                    "model invocation receipt request_digest must be a lowercase SHA-256 digest"
                )
            canonical[canonical_key] = value
        elif canonical_key in _RECEIPT_OPAQUE_ID_FIELDS:
            if not is_safe_opaque_id(value):
                raise ValueError(
                    f"model invocation receipt {canonical_key} must be a bounded opaque id"
                )
            canonical[canonical_key] = value
        elif canonical_key in _RECEIPT_CODE_FIELDS:
            if not is_safe_taxonomy_code(value):
                raise ValueError(f"model invocation receipt {canonical_key} must be a bounded code")
            canonical[canonical_key] = value
        elif canonical_key in _RECEIPT_TIMESTAMP_FIELDS:
            if not is_safe_utc_timestamp(value):
                raise ValueError(
                    f"model invocation receipt {canonical_key} must be a UTC RFC3339 timestamp"
                )
            canonical[canonical_key] = value
        elif canonical_key in _RECEIPT_BOOLEAN_FIELDS:
            if type(value) is not bool:
                raise ValueError(f"model invocation receipt {canonical_key} must be a boolean")
            canonical[canonical_key] = value
        elif canonical_key in _RECEIPT_DURATION_FIELDS:
            if not is_finite_json_number(value) or value < 0:
                raise ValueError(
                    f"model invocation receipt {canonical_key} must be a non-negative number"
                )
            canonical[canonical_key] = value
        elif canonical_key == "attempts":
            _require_positive_int(value, "receipt attempts")
            canonical[canonical_key] = value
        elif canonical_key == "http_status":
            if type(value) is not int or not 100 <= value <= 599:
                raise ValueError("model invocation receipt http_status must be between 100 and 599")
            canonical[canonical_key] = value
        else:  # pragma: no cover - every canonical field belongs to one group above
            raise AssertionError(f"unclassified receipt field {canonical_key}")
    return canonical


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
        ):
            if not is_safe_opaque_id(getattr(self, field_name)):
                raise ValueError(f"model invocation {field_name} must be a bounded opaque id")
        if type(self.idempotency_key) is not str or not is_valid_idempotency_key(
            self.idempotency_key
        ):
            raise ValueError("model invocation idempotency_key is outside the portable vocabulary")
        if type(self.request_digest) is not str or not is_recorded_digest(self.request_digest):
            raise ValueError("model invocation request_digest must be a lowercase SHA-256 digest")
        if (
            type(self.digest_generation) is not str
            or self.digest_generation not in ACCEPTED_MODEL_REQUEST_DIGEST_GENERATIONS
        ):
            raise ValueError("unsupported model invocation digest_generation")
        _require_positive_int(self.revision, "revision")
        _require_positive_int(self.dispatch_attempt, "dispatch_attempt")
        if type(self.dispatch_state) is not str or self.dispatch_state not in get_args(
            DispatchState
        ):
            raise ValueError("model invocation dispatch_state is outside the durable vocabulary")
        _require_string(self.result_ref, "result_ref")
        _require_string(self.failure_code, "failure_code")
        if self.result_ref and not is_safe_opaque_address(self.result_ref):
            raise ValueError(
                "model invocation result_ref must be empty or a bounded opaque address"
            )
        if self.failure_code and not is_safe_taxonomy_code(self.failure_code):
            raise ValueError("model invocation failure_code must be empty or a bounded code")
        receipt = _normalized_receipt(self.receipt)
        object.__setattr__(self, "receipt", receipt)
        if receipt is not None and receipt.get("request_digest", self.request_digest) != (
            self.request_digest
        ):
            raise ValueError(
                "model invocation receipt request_digest must match invocation request_digest"
            )

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
            "digest_generation": MODEL_REQUEST_DIGEST_GENERATION,
            "receipt": _normalized_receipt(self.receipt),
            "result_ref": self.result_ref,
            "failure_code": self.failure_code,
        }

    @classmethod
    def from_json(cls, payload: object) -> DurableModelInvocation | None:
        """Compatibility wrapper over :func:`decode_model_invocation`."""

        return decode_model_invocation(payload).value


def _model_invocation_from_payload(payload: dict[str, Any]) -> DurableModelInvocation:
    require_only_fields(payload, _MODEL_INVOCATION_FIELDS, "model invocation")
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
    "MODEL_REQUEST_DIGEST_GENERATION",
    "ACCEPTED_MODEL_REQUEST_DIGEST_GENERATIONS",
    "LOGICAL_MODEL_CALL_GENERATION",
    "MODEL_DISPATCH_GENERATION",
    "DispatchState",
    "DurableModelInvocation",
    "MODEL_INVOCATION_CODEC",
    "decode_model_invocation",
    "logical_model_call_id",
    "model_dispatch_id",
    "model_invocation_receipt",
]
