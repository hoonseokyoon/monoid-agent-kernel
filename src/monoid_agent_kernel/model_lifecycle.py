"""Internal typed boundary between model dispatch and authoritative invocation persistence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn, Protocol

from monoid_agent_kernel.core.json_ingress import (
    is_portable_json_integer,
    loads_json_ingress,
)
from monoid_agent_kernel.core.model_invocation import (
    MODEL_INVOCATION_RECEIPT_USAGE_FIELDS,
    MODEL_REQUEST_DIGEST_GENERATION,
    model_dispatch_id,
    model_invocation_receipt,
)
from monoid_agent_kernel.core.model_io import (
    MAX_MODEL_PAYLOAD_BYTES,
    ModelCallReceipt,
    is_recorded_digest,
    is_valid_idempotency_key,
)
from monoid_agent_kernel.core.model_payloads import (
    RECORDED_TURN_FIELDS,
    response_record_body,
)
from monoid_agent_kernel.core.safe_evidence import (
    is_safe_opaque_id,
    is_safe_taxonomy_code,
)
from monoid_agent_kernel.errors import (
    DurableModelCallError,
    ModelDispatchRefused,
    ModelEvidenceUncommitted,
)
from monoid_agent_kernel.providers.base import (
    ModelTurn,
    ToolCall,
    mark_provider_usage,
    normalize_model_turn,
    unportable_usage_key,
)


@dataclass(frozen=True, kw_only=True)
class ModelDispatchReservation:
    """One proposed or effective durable dispatch reservation.

    A lifecycle hook may replace only ``idempotency_key`` when it resumes an already committed
    reservation. The runner checks every other field for exact equality before adapter entry.
    """

    logical_call_id: str
    dispatch_attempt: int
    dispatch_id: str
    request_digest: str
    digest_generation: str
    idempotency_key: str

    def __post_init__(self) -> None:
        if not is_safe_opaque_id(self.logical_call_id):
            raise ValueError("model dispatch logical_call_id must be a bounded opaque id")
        if not is_portable_json_integer(self.dispatch_attempt) or self.dispatch_attempt < 1:
            raise ValueError("model dispatch attempt must be a positive portable integer")
        if not is_safe_opaque_id(self.dispatch_id):
            raise ValueError("model dispatch dispatch_id must be a bounded opaque id")
        if not is_recorded_digest(self.request_digest):
            raise ValueError("model dispatch request_digest must be a lowercase SHA-256 digest")
        if self.digest_generation != MODEL_REQUEST_DIGEST_GENERATION:
            raise ValueError("model dispatch digest_generation is unsupported")
        if not is_valid_idempotency_key(self.idempotency_key):
            raise ValueError("model dispatch idempotency_key is outside the portable vocabulary")


@dataclass(frozen=True, kw_only=True)
class ModelDispatchRecoveryQuery:
    """The canonical identity available before a durable dispatch is reserved."""

    logical_call_id: str
    request_digest: str
    digest_generation: str
    require_evidence: bool = False

    def __post_init__(self) -> None:
        if not is_safe_opaque_id(self.logical_call_id):
            raise ValueError("model recovery logical_call_id must be a bounded opaque id")
        if not is_recorded_digest(self.request_digest):
            raise ValueError("model recovery request_digest must be a lowercase SHA-256 digest")
        if self.digest_generation != MODEL_REQUEST_DIGEST_GENERATION:
            raise ValueError("model recovery digest_generation is unsupported")
        if type(self.require_evidence) is not bool:
            raise ValueError("model recovery require_evidence must be a boolean")


@dataclass(frozen=True, kw_only=True)
class RecoveredModelDispatch:
    """One authoritative settled dispatch recovered before provider entry."""

    reservation: ModelDispatchReservation
    receipt: Mapping[str, Any]
    result_blob: bytes | None = None
    failure_code: str = ""

    def __post_init__(self) -> None:
        if self.failure_code:
            if not is_safe_taxonomy_code(self.failure_code):
                raise ValueError("recovered model failure_code must be a bounded taxonomy code")
            if self.result_blob is not None:
                raise ValueError("recovered failed dispatch cannot carry a result blob")
        elif type(self.result_blob) is not bytes:
            raise ValueError("recovered successful dispatch must carry result bytes")
        object.__setattr__(self, "receipt", dict(self.receipt))


@dataclass(frozen=True, kw_only=True)
class ModelDispatchSettlement:
    """Authoritative terminal evidence for one durable dispatch attempt."""

    reservation: ModelDispatchReservation
    receipt: Mapping[str, Any]
    result_blob: bytes | None = None
    failure_code: str = ""

    def __post_init__(self) -> None:
        if self.failure_code:
            if not is_safe_taxonomy_code(self.failure_code):
                raise ValueError("model dispatch failure_code must be a bounded taxonomy code")
            if self.result_blob is not None:
                raise ValueError("failed model dispatch settlement cannot carry a result blob")
        elif type(self.result_blob) is not bytes:
            raise ValueError("successful model dispatch settlement must carry result bytes")
        object.__setattr__(self, "receipt", dict(self.receipt))


@dataclass(frozen=True, kw_only=True)
class UnknownModelDispatch:
    """Safe evidence that a started dispatch cannot be proven settled."""

    reservation: ModelDispatchReservation
    failure_code: str = "dispatch_unknown"

    def __post_init__(self) -> None:
        if not is_safe_taxonomy_code(self.failure_code):
            raise ValueError("unknown model dispatch failure_code must be a bounded taxonomy code")


class ModelCallLifecycleHook(Protocol):
    """Synchronous persistence seam for the runner's paid-call state machine."""

    def reserve(
        self, proposed: ModelDispatchReservation
    ) -> ModelDispatchReservation: ...

    def dispatch_started(self, reservation: ModelDispatchReservation) -> None: ...

    def settled(self, settlement: ModelDispatchSettlement) -> None: ...

    def unknown(self, unknown: UnknownModelDispatch) -> None: ...


class ModelCallRecoveryHook(Protocol):
    """Optional checked-load extension implemented by durable host adapters."""

    def recover(
        self, query: ModelDispatchRecoveryQuery
    ) -> RecoveredModelDispatch | None: ...


def dispatch_evidence(exc: BaseException) -> str:
    """Return explicit terminal evidence, defaulting every other shape to ambiguity."""

    return "refused" if isinstance(exc, ModelDispatchRefused) else "unknown"


def safe_failure_code(value: object, *, default: str) -> str:
    return value if is_safe_taxonomy_code(value) else default


def mark_recovered_model_usage(
    error: BaseException,
    receipt: Mapping[str, Any],
) -> None:
    """Carry already-billed public usage when recovered result verification fails."""

    raw_usage = receipt.get("usage")
    if not isinstance(raw_usage, Mapping):
        return
    usage = {
        key: value
        for key, value in raw_usage.items()
        if type(key) is str
        and key in MODEL_INVOCATION_RECEIPT_USAGE_FIELDS
        and is_portable_json_integer(value)
        and value >= 0
    }
    mark_provider_usage(error, usage)


def reserve_model_dispatch(
    hook: ModelCallLifecycleHook,
    *,
    logical_call_id: str,
    dispatch_attempt: int,
    request_digest: str,
    idempotency_key: str,
) -> ModelDispatchReservation:
    """Propose one attempt and accept only a stored-key substitution from the hook."""

    proposed = ModelDispatchReservation(
        logical_call_id=logical_call_id,
        dispatch_attempt=dispatch_attempt,
        dispatch_id=model_dispatch_id(logical_call_id, dispatch_attempt),
        request_digest=request_digest,
        digest_generation=MODEL_REQUEST_DIGEST_GENERATION,
        idempotency_key=idempotency_key,
    )
    effective = hook.reserve(proposed)
    if type(effective) is not ModelDispatchReservation:
        raise TypeError("model call lifecycle reserve must return ModelDispatchReservation")
    if (
        effective.logical_call_id != proposed.logical_call_id
        or effective.dispatch_attempt != proposed.dispatch_attempt
        or effective.dispatch_id != proposed.dispatch_id
        or effective.request_digest != proposed.request_digest
        or effective.digest_generation != proposed.digest_generation
    ):
        raise DurableModelCallError(
            "model call lifecycle reserve changed durable dispatch identity",
            error_code="durable_invocation_reservation_conflict",
        )
    return effective


def recover_model_dispatch(
    hook: ModelCallLifecycleHook,
    *,
    logical_call_id: str,
    request_digest: str,
    require_evidence: bool = False,
) -> RecoveredModelDispatch | None:
    """Ask an optional recovery hook for settled evidence and validate its identity."""

    recover = getattr(hook, "recover", None)
    if not callable(recover):
        return None
    query = ModelDispatchRecoveryQuery(
        logical_call_id=logical_call_id,
        request_digest=request_digest,
        digest_generation=MODEL_REQUEST_DIGEST_GENERATION,
        require_evidence=require_evidence,
    )
    recovered = recover(query)
    if recovered is None:
        return None
    if type(recovered) is not RecoveredModelDispatch:
        raise TypeError("model call lifecycle recover must return RecoveredModelDispatch or None")
    reservation = recovered.reservation
    if (
        reservation.logical_call_id != query.logical_call_id
        or reservation.request_digest != query.request_digest
        or reservation.digest_generation != query.digest_generation
    ):
        raise DurableModelCallError(
            "model call lifecycle recovery changed durable dispatch identity",
            error_code="durable_invocation_recovery_conflict",
        )
    return recovered


def raise_model_dispatch_unknown(
    hook: ModelCallLifecycleHook,
    reservation: ModelDispatchReservation,
    cause: Exception,
    *,
    failure_code: str = "dispatch_unknown",
    usage: Mapping[str, int] | None = None,
) -> NoReturn:
    """Commit ambiguity when possible and always stop automatic paid-call retry."""

    safe_code = safe_failure_code(failure_code, default="dispatch_unknown")
    try:
        hook.unknown(
            UnknownModelDispatch(
                reservation=reservation,
                failure_code=safe_code,
            )
        )
    except Exception as persistence_error:
        failed = DurableModelCallError(
            "model dispatch is unknown and its journal transition did not commit",
            error_code="dispatch_unknown",
        )
        if usage:
            mark_provider_usage(failed, usage)
        raise failed from persistence_error
    failed = DurableModelCallError(
        "model dispatch is unknown; automatic paid-call retry is forbidden",
        error_code="dispatch_unknown",
    )
    if usage:
        mark_provider_usage(failed, usage)
    raise failed from cause


def settle_model_dispatch(
    hook: ModelCallLifecycleHook,
    reservation: ModelDispatchReservation,
    receipt: ModelCallReceipt,
    *,
    result_blob: bytes | None = None,
    failure_code: str = "",
    stream_committed: bool | None = None,
) -> None:
    """Commit a proven terminal result or turn a failed commit into ambiguity."""

    safe_failure = safe_failure_code(failure_code, default="model_error") if failure_code else ""
    try:
        evidence = model_invocation_receipt(receipt)
        if stream_committed is not None:
            evidence["stream_committed"] = stream_committed
        hook.settled(
            ModelDispatchSettlement(
                reservation=reservation,
                receipt=evidence,
                result_blob=result_blob,
                failure_code=safe_failure,
            )
        )
    except ModelEvidenceUncommitted:
        # The invocation is already authoritative. Reclassifying a projection failure as an
        # ambiguous paid dispatch would both lose that fact and invite the wrong recovery path.
        raise
    except Exception as persistence_error:
        raise_model_dispatch_unknown(
            hook,
            reservation,
            persistence_error,
            failure_code="settlement_uncommitted",
            usage=receipt.usage,
        )


def durable_model_result_blob(turn: Any) -> bytes:
    """Encode the canonical private result body or return a typed durable refusal."""

    recorded = response_record_body(turn)
    if recorded.encoded is not None:
        return recorded.encoded
    raise DurableModelCallError(
        "durable model result could not be canonically recorded",
        error_code=(
            "durable_invocation_result_too_large"
            if recorded.unrecorded_reason == "too_large"
            else "durable_invocation_result_unencodable"
        ),
    )


def durable_model_turn(blob: bytes) -> ModelTurn:
    """Reconstruct one canonical recorded turn without trusting private durable bytes."""

    try:
        if type(blob) is not bytes or len(blob) > MAX_MODEL_PAYLOAD_BYTES:
            raise ValueError("result bytes are absent or oversized")
        body = loads_json_ingress(blob.decode("utf-8"))
        if not isinstance(body, dict) or set(body) != set(RECORDED_TURN_FIELDS):
            raise ValueError("result body fields do not match the recorded-turn contract")
        if not isinstance(body["tool_calls"], list):
            raise ValueError("result tool_calls must be a list")
        if not isinstance(body["reasoning"], list):
            raise ValueError("result reasoning must be a list")
        if not isinstance(body["usage"], Mapping):
            raise ValueError("result usage must be an object")
        calls: list[ToolCall] = []
        for call in body["tool_calls"]:
            if (
                not isinstance(call, Mapping)
                or type(call.get("id")) is not str
                or type(call.get("name")) is not str
                or not isinstance(call.get("arguments"), dict)
            ):
                raise ValueError("result tool call is malformed")
            calls.append(
                ToolCall(
                    id=call["id"],
                    name=call["name"],
                    arguments=dict(call["arguments"]),
                )
            )
        reasoning: list[dict[str, Any]] = []
        for item in body["reasoning"]:
            if not isinstance(item, Mapping):
                raise ValueError("result reasoning item is malformed")
            reasoning.append(dict(item))
        if unportable_usage_key(body["usage"]) is not None:
            raise ValueError("result usage is malformed")
        for name in ("response_id", "final_text", "stop_reason"):
            if body[name] is not None and type(body[name]) is not str:
                raise ValueError(f"result {name} is malformed")
        if type(body["provider_retried"]) is not bool:
            raise ValueError("result provider_retried is malformed")
        return normalize_model_turn(
            ModelTurn(
                response_id=body["response_id"],
                final_text=body["final_text"],
                tool_calls=tuple(calls),
                usage=dict(body["usage"]),
                raw={},
                reasoning=tuple(reasoning),
                stop_reason=body["stop_reason"],
                provider_retried=body["provider_retried"],
            )
        )
    except DurableModelCallError:
        raise
    except Exception as exc:
        raise DurableModelCallError(
            "durable model result is corrupt",
            error_code="durable_invocation_result_corrupt",
        ) from exc


__all__ = [
    "ModelDispatchReservation",
    "ModelDispatchRecoveryQuery",
    "RecoveredModelDispatch",
    "ModelDispatchSettlement",
    "UnknownModelDispatch",
    "ModelCallLifecycleHook",
    "ModelCallRecoveryHook",
]
