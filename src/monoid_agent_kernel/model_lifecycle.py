"""Internal typed boundary between model dispatch and authoritative invocation persistence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn, Protocol

from monoid_agent_kernel.core.json_ingress import is_portable_json_integer
from monoid_agent_kernel.core.model_invocation import (
    MODEL_REQUEST_DIGEST_GENERATION,
    model_dispatch_id,
    model_invocation_receipt,
)
from monoid_agent_kernel.core.model_io import (
    ModelCallReceipt,
    is_recorded_digest,
    is_valid_idempotency_key,
)
from monoid_agent_kernel.core.model_payloads import response_record_body
from monoid_agent_kernel.core.safe_evidence import (
    is_safe_opaque_id,
    is_safe_taxonomy_code,
)
from monoid_agent_kernel.errors import DurableModelCallError, ModelDispatchRefused
from monoid_agent_kernel.providers.base import mark_provider_usage


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


def dispatch_evidence(exc: BaseException) -> str:
    """Return explicit terminal evidence, defaulting every other shape to ambiguity."""

    return "refused" if isinstance(exc, ModelDispatchRefused) else "unknown"


def safe_failure_code(value: object, *, default: str) -> str:
    return value if is_safe_taxonomy_code(value) else default


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
) -> None:
    """Commit a proven terminal result or turn a failed commit into ambiguity."""

    safe_failure = safe_failure_code(failure_code, default="model_error") if failure_code else ""
    try:
        hook.settled(
            ModelDispatchSettlement(
                reservation=reservation,
                receipt=model_invocation_receipt(receipt),
                result_blob=result_blob,
                failure_code=safe_failure,
            )
        )
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


__all__ = [
    "ModelDispatchReservation",
    "ModelDispatchSettlement",
    "UnknownModelDispatch",
    "ModelCallLifecycleHook",
]
