"""Fenced host adapter for the kernel's durable model-call lifecycle."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

from monoid_agent_kernel.core.model_invocation import DurableModelInvocation
from monoid_agent_kernel.errors import DurableModelCallError, ModelEvidenceUncommitted
from monoid_agent_kernel.hosting.contracts import (
    CommitResult,
    FencedRunSink,
    ModelInvocationRecord,
    WriterToken,
)
from monoid_agent_kernel.model_lifecycle import (
    ModelDispatchRecoveryQuery,
    ModelDispatchReservation,
    ModelDispatchSettlement,
    RecoveredModelDispatch,
    UnknownModelDispatch,
    mark_recovered_model_usage,
)


@dataclass
class FencedModelCallLifecycle:
    """Translate runner transitions into checked FencedRunSink revisions."""

    sink: FencedRunSink
    writer_token: WriterToken
    evidence_policy: Literal["passive", "required", "outbox"] = "passive"
    last_invocation: DurableModelInvocation | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.evidence_policy not in {"passive", "required", "outbox"}:
            raise ValueError("model evidence policy is outside the portable vocabulary")

    def _load(self, logical_call_id: str) -> ModelInvocationRecord | None:
        try:
            loaded = self.sink.load_invocation(self.writer_token.run_id, logical_call_id)
        except Exception as exc:
            raise DurableModelCallError(
                "durable model invocation load failed",
                error_code="durable_invocation_load_failed",
            ) from exc
        if loaded.status == "missing":
            return None
        if not loaded.ok or loaded.value is None:
            error_code = (
                "durable_invocation_unsupported_version"
                if loaded.status == "unsupported_version"
                else "durable_invocation_corrupt"
            )
            raise DurableModelCallError(
                "durable model invocation head is unreadable",
                error_code=error_code,
            )
        record = loaded.value
        invocation = record.invocation
        if (
            invocation.run_id != self.writer_token.run_id
            or invocation.logical_call_id != logical_call_id
        ):
            raise DurableModelCallError(
                "durable model invocation lookup returned a different identity",
                error_code="durable_invocation_corrupt",
            )
        self.last_invocation = invocation
        return record

    def _commit(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes] | None = None,
        *,
        stage_evidence: bool = False,
    ) -> None:
        try:
            if stage_evidence:
                result = self.sink.commit_invocation(
                    invocation,
                    blobs or {},
                    writer_token=self.writer_token,
                    stage_evidence=True,
                )
            else:
                result = self.sink.commit_invocation(
                    invocation,
                    blobs or {},
                    writer_token=self.writer_token,
                )
        except Exception as exc:
            raise DurableModelCallError(
                "durable model invocation commit failed",
                error_code="durable_invocation_commit_failed",
            ) from exc
        if not isinstance(result, CommitResult):
            raise DurableModelCallError(
                "durable model invocation commit returned an invalid result",
                error_code="durable_invocation_commit_failed",
            )
        if result.status == "fenced":
            raise DurableModelCallError(
                "durable model invocation writer was fenced",
                error_code="durable_invocation_fenced",
            )
        if result.status == "conflict":
            raise DurableModelCallError(
                "durable model invocation conflicted with authoritative state",
                error_code="durable_invocation_conflict",
            )
        if result.status not in {"committed", "already_committed"}:
            raise DurableModelCallError(
                "durable model invocation commit returned an unsupported status",
                error_code="durable_invocation_commit_failed",
            )
        self.last_invocation = invocation

    def _commit_evidence(self, invocation: DurableModelInvocation) -> None:
        try:
            result = self.sink.commit_model_evidence(
                invocation,
                writer_token=self.writer_token,
            )
            if not isinstance(result, CommitResult) or result.status not in {
                "committed",
                "already_committed",
            }:
                raise RuntimeError("required evidence commit was not accepted")
        except Exception as exc:
            failed = ModelEvidenceUncommitted()
            mark_recovered_model_usage(failed, invocation.receipt or {})
            raise failed from exc

    def _commit_settled(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes] | None = None,
        *,
        require_evidence: bool = False,
    ) -> None:
        evidence_policy = "required" if require_evidence else self.evidence_policy
        self._commit(
            invocation,
            blobs,
            stage_evidence=evidence_policy == "outbox",
        )
        if evidence_policy == "required":
            self._commit_evidence(invocation)

    @staticmethod
    def _same_dispatch(
        invocation: DurableModelInvocation,
        reservation: ModelDispatchReservation,
    ) -> bool:
        return (
            invocation.logical_call_id == reservation.logical_call_id
            and invocation.dispatch_attempt == reservation.dispatch_attempt
            and invocation.dispatch_id == reservation.dispatch_id
            and invocation.request_digest == reservation.request_digest
            and invocation.digest_generation == reservation.digest_generation
            and invocation.idempotency_key == reservation.idempotency_key
        )

    def _unknown_started(self, invocation: DurableModelInvocation) -> None:
        unknown = replace(
            invocation,
            revision=invocation.revision + 1,
            dispatch_state="unknown",
            failure_code="dispatch_unknown",
        )
        self._commit(unknown)
        raise DurableModelCallError(
            "a previously started model dispatch is unknown",
            error_code="dispatch_unknown",
        )

    @staticmethod
    def _result_error(
        message: str,
        invocation: DurableModelInvocation,
    ) -> DurableModelCallError:
        error = DurableModelCallError(
            message,
            error_code="durable_invocation_result_corrupt",
        )
        mark_recovered_model_usage(error, invocation.receipt or {})
        return error

    def recover(
        self,
        query: ModelDispatchRecoveryQuery,
    ) -> RecoveredModelDispatch | None:
        record = self._load(query.logical_call_id)
        if record is None:
            return None
        invocation = record.invocation
        if (
            invocation.request_digest != query.request_digest
            or invocation.digest_generation != query.digest_generation
        ):
            raise DurableModelCallError(
                "durable model invocation request identity changed",
                error_code="durable_invocation_request_conflict",
            )
        if invocation.dispatch_state == "reserved":
            return None
        if invocation.dispatch_state == "dispatch_started":
            self._unknown_started(invocation)
        if invocation.dispatch_state == "unknown":
            raise DurableModelCallError(
                "durable model invocation remains unknown",
                error_code="dispatch_unknown",
            )
        if invocation.receipt is None:
            raise DurableModelCallError(
                "settled durable model invocation has no receipt",
                error_code="durable_invocation_corrupt",
            )
        reservation = ModelDispatchReservation(
            logical_call_id=invocation.logical_call_id,
            dispatch_attempt=invocation.dispatch_attempt,
            dispatch_id=invocation.dispatch_id,
            request_digest=invocation.request_digest,
            digest_generation=invocation.digest_generation,
            idempotency_key=invocation.idempotency_key,
        )
        if invocation.failure_code:
            # An exact idempotent mutation is the host contract's authenticated ownership check.
            # Fencing precedes idempotency, so a stale activation cannot expose a recovered
            # settlement even though the preceding checked load is intentionally token-free.
            self._commit_settled(
                invocation,
                require_evidence=query.require_evidence,
            )
            return RecoveredModelDispatch(
                reservation=reservation,
                receipt=invocation.receipt,
                failure_code=invocation.failure_code,
            )
        sha256 = invocation.result_ref.removeprefix("blob:")
        if not sha256 or invocation.result_ref != f"blob:{sha256}":
            raise self._result_error(
                "settled durable model invocation has no private result",
                invocation,
            )
        try:
            result_blob = record.blob(sha256)
        except Exception as exc:
            raise self._result_error(
                "durable model invocation result is unavailable",
                invocation,
            ) from exc
        if type(result_blob) is not bytes or hashlib.sha256(result_blob).hexdigest() != sha256:
            raise self._result_error(
                "durable model invocation result failed content verification",
                invocation,
            )
        # Replay the exact settlement through the fenced mutation path before returning any
        # private result. Supplying the original blob preserves full content identity.
        self._commit_settled(
            invocation,
            {sha256: result_blob},
            require_evidence=query.require_evidence,
        )
        return RecoveredModelDispatch(
            reservation=reservation,
            receipt=invocation.receipt,
            result_blob=result_blob,
        )

    def reserve(self, proposed: ModelDispatchReservation) -> ModelDispatchReservation:
        record = self._load(proposed.logical_call_id)
        if record is None:
            effective = proposed
            revision = 1
        else:
            invocation = record.invocation
            if invocation.dispatch_state == "reserved":
                effective = replace(proposed, idempotency_key=invocation.idempotency_key)
                if not self._same_dispatch(invocation, effective):
                    raise DurableModelCallError(
                        "stored model reservation conflicts with the current request",
                        error_code="durable_invocation_request_conflict",
                    )
                return effective
            if invocation.dispatch_state == "dispatch_started":
                self._unknown_started(invocation)
            if invocation.dispatch_state == "unknown":
                raise DurableModelCallError(
                    "durable model invocation remains unknown",
                    error_code="dispatch_unknown",
                )
            retryable_failure = (
                bool(invocation.failure_code)
                and invocation.receipt is not None
                and invocation.receipt.get("retryable") is True
            )
            if not retryable_failure:
                raise DurableModelCallError(
                    "settled model invocation cannot be dispatched again",
                    error_code="durable_invocation_already_settled",
                )
            effective = replace(proposed, idempotency_key=invocation.idempotency_key)
            if (
                effective.dispatch_attempt != invocation.dispatch_attempt + 1
                or effective.request_digest != invocation.request_digest
                or effective.digest_generation != invocation.digest_generation
            ):
                raise DurableModelCallError(
                    "model invocation retry coordinate conflicts with authoritative state",
                    error_code="durable_invocation_request_conflict",
                )
            revision = invocation.revision + 1
        self._commit(
            DurableModelInvocation(
                run_id=self.writer_token.run_id,
                logical_call_id=effective.logical_call_id,
                revision=revision,
                dispatch_id=effective.dispatch_id,
                dispatch_attempt=effective.dispatch_attempt,
                idempotency_key=effective.idempotency_key,
                dispatch_state="reserved",
                request_digest=effective.request_digest,
                digest_generation=effective.digest_generation,
            )
        )
        return effective

    def dispatch_started(self, reservation: ModelDispatchReservation) -> None:
        record = self._load(reservation.logical_call_id)
        if record is None or record.invocation.dispatch_state != "reserved":
            raise DurableModelCallError(
                "model dispatch start has no authoritative reservation",
                error_code="durable_invocation_conflict",
            )
        invocation = record.invocation
        if not self._same_dispatch(invocation, reservation):
            raise DurableModelCallError(
                "model dispatch start conflicts with its reservation",
                error_code="durable_invocation_request_conflict",
            )
        self._commit(
            replace(
                invocation,
                revision=invocation.revision + 1,
                dispatch_state="dispatch_started",
            )
        )

    def settled(self, settlement: ModelDispatchSettlement) -> None:
        reservation = settlement.reservation
        record = self._load(reservation.logical_call_id)
        if record is None or record.invocation.dispatch_state != "dispatch_started":
            raise DurableModelCallError(
                "model settlement has no authoritative started dispatch",
                error_code="durable_invocation_conflict",
            )
        invocation = record.invocation
        if not self._same_dispatch(invocation, reservation):
            raise DurableModelCallError(
                "model settlement conflicts with its started dispatch",
                error_code="durable_invocation_request_conflict",
            )
        blobs: dict[str, bytes] = {}
        result_ref = ""
        if settlement.result_blob is not None:
            sha256 = hashlib.sha256(settlement.result_blob).hexdigest()
            blobs[sha256] = settlement.result_blob
            result_ref = f"blob:{sha256}"
        self._commit_settled(
            replace(
                invocation,
                revision=invocation.revision + 1,
                dispatch_state="settled",
                receipt=dict(settlement.receipt),
                result_ref=result_ref,
                failure_code=settlement.failure_code,
            ),
            blobs,
        )

    def unknown(self, unknown: UnknownModelDispatch) -> None:
        reservation = unknown.reservation
        record = self._load(reservation.logical_call_id)
        if record is None or record.invocation.dispatch_state != "dispatch_started":
            raise DurableModelCallError(
                "unknown model dispatch has no authoritative started state",
                error_code="durable_invocation_conflict",
            )
        invocation = record.invocation
        if not self._same_dispatch(invocation, reservation):
            raise DurableModelCallError(
                "unknown model dispatch conflicts with its started state",
                error_code="durable_invocation_request_conflict",
            )
        self._commit(
            replace(
                invocation,
                revision=invocation.revision + 1,
                dispatch_state="unknown",
                failure_code=unknown.failure_code,
            )
        )


__all__ = ["FencedModelCallLifecycle"]
