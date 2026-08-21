"""Reusable implementation contracts for checkpoint stores and capability brokers."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, Protocol
from uuid import uuid4

from monoid_agent_kernel.conformance.report import (
    ConformanceRuleOutcome,
    observation,
    outcome_from_observations,
    safe_exception_summary,
)
from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.core.capability import (
    CapabilityBroker,
    CapabilityDenial,
    CapabilityLease,
    CapabilityPending,
    CapabilityRequest,
    scope_within,
)
from monoid_agent_kernel.core.checkpoint import (
    ACCEPTED_SCHEMA_VERSIONS as ACCEPTED_CHECKPOINT_SCHEMA_VERSIONS,
    SCHEMA_VERSION as CHECKPOINT_SCHEMA_VERSION,
    CheckpointStore,
    RunCheckpoint,
    checkpoint_payload_for_write,
    load_latest_checked,
)
from monoid_agent_kernel.core.events import EVENT_SCHEMA_VERSION, AgentEvent
from monoid_agent_kernel.core.inbox import InboxMessage
from monoid_agent_kernel.core.model_invocation import (
    ACCEPTED_MODEL_INVOCATION_SCHEMA_VERSIONS,
    ACCEPTED_MODEL_REQUEST_DIGEST_GENERATIONS,
    MODEL_INVOCATION_RECEIPT_FIELDS,
    MODEL_INVOCATION_RECEIPT_USAGE_FIELDS,
    MODEL_INVOCATION_SCHEMA_VERSION,
    MODEL_REQUEST_DIGEST_GENERATION,
    DurableModelInvocation,
)
from monoid_agent_kernel.core.model_io import (
    CapturePolicy,
    ModelCallCapture,
    ModelCallReceipt,
    ModelIOObserver,
    ModelIOSubscription,
    RedactionPolicy,
    Redactor,
    close_model_io_subscriptions,
    dispatch_model_call,
    redacted_or_none,
)
from monoid_agent_kernel.core.outcome import (
    ACCEPTED_TERMINAL_OUTCOME_SCHEMA_VERSIONS,
    TERMINAL_OUTCOME_SCHEMA_VERSION,
    InterruptionCause,
    RetryEligibility,
    TerminalOutcome,
)
from monoid_agent_kernel.hosting import CommitResult, FencedRunSink, WriterToken

STORE_CONTRACT_PROFILE = "checkpoint-store-contract"
BROKER_CONTRACT_PROFILE = "capability-broker-contract"
REDACTOR_CONTRACT_PROFILE = "redactor-contract"
MODEL_IO_CONTRACT_PROFILE = "model-io-observer-contract"
FENCED_RUN_SINK_CONTRACT_PROFILE = "fenced-run-sink-contract"
_CONTRACT_CHECKPOINT_BLOB = b"contract checkpoint workspace bytes\n"
_CONTRACT_CHECKPOINT_BLOB_SHA256 = hashlib.sha256(_CONTRACT_CHECKPOINT_BLOB).hexdigest()
_CONTRACT_INVOCATION_BLOB = b'{"text":"contract model result"}'
_CONTRACT_INVOCATION_BLOB_SHA256 = hashlib.sha256(_CONTRACT_INVOCATION_BLOB).hexdigest()
_CONTRACT_ALTERNATE_BLOB = b"alternate contract blob bytes\n"
_CONTRACT_ALTERNATE_BLOB_SHA256 = hashlib.sha256(_CONTRACT_ALTERNATE_BLOB).hexdigest()
_CONTRACT_UNRESOLVED_BLOB = b"contract reference recovery bytes\n"
_CONTRACT_UNRESOLVED_BLOB_SHA256 = hashlib.sha256(_CONTRACT_UNRESOLVED_BLOB).hexdigest()
_CONTRACT_MALFORMED_BLOB_SHA256 = "not-a-digest"
_CONTRACT_STALE_HANDOFF_BLOB = b"stale writer handoff-only blob bytes\n"
_CONTRACT_STALE_HANDOFF_BLOB_SHA256 = hashlib.sha256(_CONTRACT_STALE_HANDOFF_BLOB).hexdigest()
_CONTRACT_ALTERNATE_DIGEST_GENERATION = next(
    generation
    for generation in ACCEPTED_MODEL_REQUEST_DIGEST_GENERATIONS
    if generation != MODEL_REQUEST_DIGEST_GENERATION
)
_CONTRACT_ALTERNATE_INVOCATION_SCHEMA_VERSION = next(
    schema
    for schema in ACCEPTED_MODEL_INVOCATION_SCHEMA_VERSIONS
    if schema != MODEL_INVOCATION_SCHEMA_VERSION
)
_CONTRACT_ALTERNATE_CHECKPOINT_SCHEMA_VERSION = next(
    schema for schema in ACCEPTED_CHECKPOINT_SCHEMA_VERSIONS if schema != CHECKPOINT_SCHEMA_VERSION
)
_CONTRACT_ALTERNATE_TERMINAL_SCHEMA_VERSION = next(
    schema
    for schema in ACCEPTED_TERMINAL_OUTCOME_SCHEMA_VERSIONS
    if schema != TERMINAL_OUTCOME_SCHEMA_VERSION
)
_CONTRACT_INVOCATION_IDENTITY_FIELDS = frozenset(
    {
        "dispatch_id",
        "dispatch_attempt",
        "idempotency_key",
        "dispatch_state",
        "request_digest",
        "receipt",
        "result_ref",
        "failure_code",
    }
)
_CONTRACT_INVOCATION_FIXED_CANONICAL_FIELDS = frozenset({"schema_version", "digest_generation"})
_CONTRACT_RETRY_STABLE_IDENTITY_FIELDS = frozenset({"idempotency_key", "request_digest"})
_CONTRACT_QUEUED_MEDIA_CARRIERS = ("content_list", "envelope")
_CONTRACT_RECEIPT_RETRYABILITY_STATES = ("true", "false", "omitted")
_CONTRACT_CHECKPOINT_CANONICAL_ALIAS_FIELDS = frozenset(
    {
        "schema_version",
        "last_model_invocation_schema_version",
        "last_model_invocation_digest_generation",
    }
)


class CheckpointStoreFactory(Protocol):
    def __call__(self, root: Path) -> CheckpointStore: ...


class CapabilityBrokerFactory(Protocol):
    def __call__(self) -> CapabilityBroker: ...


class RedactorFactory(Protocol):
    def __call__(self) -> Redactor: ...


class ModelIOObserverFactory(Protocol):
    def __call__(self) -> ModelIOObserver: ...


class FencedRunSinkHarness(Protocol):
    """Conformance-only seam that installs exact authoritative writer coordinates."""

    @property
    def sink(self) -> FencedRunSink: ...

    def set_current_writer(self, writer_token: WriterToken) -> None:
        """Arrange test authority without constraining the host's generation allocator."""

        ...

    def reopen(self) -> FencedRunSinkHarness:
        """Open a fresh sink facade over the same durable backing store and host authority."""

        ...

    def inject_authoritative_load_fault(
        self,
        record_family: Literal["checkpoint", "invocation"],
        run_id: str,
        status: Literal["corrupt", "unsupported_version"],
        *,
        logical_call_id: str = "",
    ) -> None:
        """Replace an existing authoritative head with the requested durable read fault.

        The injected fault survives ``reopen()``. Backend harnesses use a raw storage mutation or
        an equivalent decoder hook so the sink's checked load path performs the classification.
        """

        ...

    def read_event(self, run_id: str, seq: int) -> AgentEvent | None:
        """Read one complete durable event through this facade's backing store."""

        ...

    def read_terminal(self, run_id: str) -> TerminalOutcome | None:
        """Read the complete winning terminal outcome through the durable backing store."""

        ...

    def close(self) -> None:
        """Release the sink facade and every session or client owned by this harness."""

        ...

    def race_conflicting_writes(
        self,
        mutation: str,
        writer_token: WriterToken,
        left: Callable[[FencedRunSink], CommitResult],
        right: Callable[[FencedRunSink], CommitResult],
    ) -> tuple[CommitResult, CommitResult]:
        """Coordinate competing writes at the backend's CAS read/publication gap.

        Implementations use a backend test hook or an equivalent transaction interlock. Entry-only
        synchronization is insufficient for a backend whose compare and publish steps are separate.
        The hook owns and closes any additional sink facades it opens.
        """

        ...

    def race_writer_handoff(
        self,
        mutation: str,
        stale_token: WriterToken,
        current_token: WriterToken,
        write: Callable[[FencedRunSink, WriterToken], CommitResult],
    ) -> tuple[CommitResult, CommitResult, bool]:
        """Race a real mutation against authority rotation on one backing store.

        The boolean is true exactly when rotation becomes authoritative before the stale mutation's
        publication point. Implementations coordinate the adapter and lease store with barriers or
        test hooks; they do not infer this ordering from the returned commit status.
        """

        ...


class FencedRunSinkHarnessFactory(Protocol):
    def __call__(self) -> FencedRunSinkHarness: ...


class _TrackedFencedRunSinkHarness:
    def __init__(
        self,
        inner: FencedRunSinkHarness,
        registry: _FencedHarnessRegistry,
    ) -> None:
        self._inner = inner
        self._registry = registry
        self._closed = False

    @property
    def sink(self) -> FencedRunSink:
        return self._inner.sink

    def set_current_writer(self, writer_token: WriterToken) -> None:
        self._inner.set_current_writer(writer_token)

    def reopen(self) -> FencedRunSinkHarness:
        return self._registry.track(self._inner.reopen())

    def inject_authoritative_load_fault(
        self,
        record_family: Literal["checkpoint", "invocation"],
        run_id: str,
        status: Literal["corrupt", "unsupported_version"],
        *,
        logical_call_id: str = "",
    ) -> None:
        self._inner.inject_authoritative_load_fault(
            record_family,
            run_id,
            status,
            logical_call_id=logical_call_id,
        )

    def read_event(self, run_id: str, seq: int) -> AgentEvent | None:
        return self._inner.read_event(run_id, seq)

    def read_terminal(self, run_id: str) -> TerminalOutcome | None:
        return self._inner.read_terminal(run_id)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._inner.close()

    def race_conflicting_writes(
        self,
        mutation: str,
        writer_token: WriterToken,
        left: Callable[[FencedRunSink], CommitResult],
        right: Callable[[FencedRunSink], CommitResult],
    ) -> tuple[CommitResult, CommitResult]:
        return self._inner.race_conflicting_writes(
            mutation,
            writer_token,
            left,
            right,
        )

    def race_writer_handoff(
        self,
        mutation: str,
        stale_token: WriterToken,
        current_token: WriterToken,
        write: Callable[[FencedRunSink, WriterToken], CommitResult],
    ) -> tuple[CommitResult, CommitResult, bool]:
        return self._inner.race_writer_handoff(
            mutation,
            stale_token,
            current_token,
            write,
        )


class _FencedHarnessRegistry:
    def __init__(self) -> None:
        self._opened: list[_TrackedFencedRunSinkHarness] = []

    def track(self, harness: FencedRunSinkHarness) -> _TrackedFencedRunSinkHarness:
        tracked = _TrackedFencedRunSinkHarness(harness, self)
        self._opened.append(tracked)
        return tracked

    def wrap_factory(
        self,
        factory: FencedRunSinkHarnessFactory,
    ) -> FencedRunSinkHarnessFactory:
        def open_probe() -> FencedRunSinkHarness:
            self.close_all()
            return self.track(factory())

        return open_probe

    def close_all(self) -> None:
        errors: list[BaseException] = []
        opened, self._opened = self._opened, []
        for harness in reversed(opened):
            try:
                harness.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise errors[0]


@contextmanager
def _opened_checkpoint_store(
    factory: CheckpointStoreFactory,
    root: Path,
) -> Iterator[CheckpointStore]:
    store = factory(root)
    try:
        yield store
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def run_checkpoint_store_contract(
    factory: CheckpointStoreFactory,
    root: Path,
) -> tuple[ConformanceRuleOutcome, ...]:
    """Execute backend-neutral checkpoint invariants without depending on pytest."""

    outcomes: list[ConformanceRuleOutcome] = []
    try:
        with _opened_checkpoint_store(factory, root) as store:
            missing = load_latest_checked(store, "contract_monotonic")
            store.put(RunCheckpoint(run_id="contract_monotonic", seq=2, final_text="new"))
            store.put(RunCheckpoint(run_id="contract_monotonic", seq=1, final_text="stale"))
        with _opened_checkpoint_store(factory, root) as reopened:
            latest = reopened.latest("contract_monotonic")
        outcomes.append(
            outcome_from_observations(
                "STORE-01-MONOTONIC-PUBLICATION",
                STORE_CONTRACT_PROFILE,
                (
                    observation("initial_missing", expected="missing", actual=missing.status),
                    observation(
                        "reopened_latest_sequence",
                        expected=2,
                        actual=latest.seq if latest else None,
                    ),
                    observation(
                        "reopened_latest_payload",
                        expected="new",
                        actual=latest.checkpoint.final_text if latest else None,
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("STORE-01-MONOTONIC-PUBLICATION", STORE_CONTRACT_PROFILE, exc))
    try:
        data = b"conformance-blob"
        with _opened_checkpoint_store(factory, root) as store:
            digest = store.put_blob("contract_blob", data)
        with _opened_checkpoint_store(factory, root) as reopened:
            reopened_blob = reopened.get_blob("contract_blob", digest)
        outcomes.append(
            outcome_from_observations(
                "STORE-02-CONTENT-ADDRESSED-BLOB",
                STORE_CONTRACT_PROFILE,
                (
                    observation(
                        "digest",
                        expected=hashlib.sha256(data).hexdigest(),
                        actual=digest,
                    ),
                    observation(
                        "reopened_round_trip",
                        expected=data.hex(),
                        actual=reopened_blob.hex(),
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("STORE-02-CONTENT-ADDRESSED-BLOB", STORE_CONTRACT_PROFILE, exc))
    try:
        with _opened_checkpoint_store(factory, root) as store:
            store.put(RunCheckpoint(run_id="contract_deleted", seq=1))
            store.put(RunCheckpoint(run_id="contract_isolated", seq=1))
        with _opened_checkpoint_store(factory, root) as reopened:
            reopened_before_delete = (
                reopened.latest("contract_deleted") is not None
                and reopened.latest("contract_isolated") is not None
            )
            reopened.delete("contract_deleted")
        with _opened_checkpoint_store(factory, root) as reopened_after_delete:
            deleted_missing = reopened_after_delete.latest("contract_deleted") is None
            other_present = reopened_after_delete.latest("contract_isolated") is not None
        outcomes.append(
            outcome_from_observations(
                "STORE-03-RUN-ISOLATION",
                STORE_CONTRACT_PROFILE,
                (
                    observation(
                        "runs_survive_reopen_before_delete",
                        expected=True,
                        actual=reopened_before_delete,
                    ),
                    observation(
                        "deleted_run_missing_after_reopen",
                        expected=True,
                        actual=deleted_missing,
                    ),
                    observation(
                        "other_run_present_after_reopen",
                        expected=True,
                        actual=other_present,
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("STORE-03-RUN-ISOLATION", STORE_CONTRACT_PROFILE, exc))
    return tuple(outcomes)


def _contract_event(run_id: str, *, seq: int, level: str = "info") -> AgentEvent:
    return AgentEvent(
        schema_version=EVENT_SCHEMA_VERSION,
        event_id=f"event-{seq}",
        seq=seq,
        run_id=run_id,
        timestamp="2026-08-21T00:00:00Z",
        type="checkpoint.committed",
        level=level,  # type: ignore[arg-type]
        data={"checkpoint_seq": seq},
    )


def _contract_terminal(run_id: str, *, failed: bool = False) -> TerminalOutcome:
    if failed:
        return TerminalOutcome(
            run_id=run_id,
            kind="failed_terminal",
            retry_eligibility=RetryEligibility.FORBIDDEN,
            error_code="contract_failed",
        )
    return TerminalOutcome(
        run_id=run_id,
        kind="completed",
        retry_eligibility=RetryEligibility.NOT_APPLICABLE,
        final_output_ref="blob:contract-final",
    )


def _contract_terminal_canonical_alias_status(
    factory: FencedRunSinkHarnessFactory,
    run_id: str,
    *,
    legacy_first: bool,
) -> str:
    baseline = _contract_terminal(run_id)
    legacy = replace(
        baseline,
        schema_version=_CONTRACT_ALTERNATE_TERMINAL_SCHEMA_VERSION,
    )
    first_value, retry_value = (legacy, baseline) if legacy_first else (baseline, legacy)
    harness = factory()
    token = _contract_writer(harness, run_id)
    first = harness.sink.settle_terminal(first_value, writer_token=token)
    if first.status != "committed":
        return f"setup:{first.status}"
    harness = harness.reopen()
    return harness.sink.settle_terminal(retry_value, writer_token=token).status


def _contract_checkpoint_identity_variants(
    checkpoint: RunCheckpoint,
) -> dict[str, RunCheckpoint]:
    """Vary every canonical non-key field independently for same-sequence retries."""

    payload = checkpoint.to_json()
    identity_fields = set(payload) - {"schema_version", "run_id", "seq"}
    list_of_string_fields = {
        "pending_binding_loads",
        "reentry_queue",
        "delivered_reentry_jobs",
        "revoked_lease_ids",
        "revoked_capabilities",
        "inbox_seen_ids",
        "applied_input_ids",
        "skills_activated",
    }
    special_values: dict[str, Any] = {
        "provider_http_status": 200,
        "previous_turn_handle": "turn-alternate",
        "pending_user_input": [{"kind": "contract-alternate"}],
        "previous_runtime_config": {"contract": "alternate"},
        "workspace_base": {"contract": "alternate"},
        "remaining_duration_s": 1.0,
        "queued_messages": ["contract-alternate"],
        "last_suspension": {"reason": "paused", "status": "completed"},
        "active_input": {
            "input_id": "input-alternate",
            "phase": "running",
            "source_seq": 1,
        },
        "applied_input_receipts": {"input-alternate": {"checkpoint_seq": checkpoint.seq}},
        "last_model_invocation": _contract_invocation(
            checkpoint.run_id,
            revision=1,
            dispatch_state="reserved",
        ).to_json(),
        "interruption_cause": InterruptionCause.USER_CANCEL.value,
    }
    variants: dict[str, RunCheckpoint] = {}
    for field_name in sorted(identity_fields):
        value = payload[field_name]
        if field_name in special_values:
            alternate = special_values[field_name]
        elif type(value) is bool:
            alternate = not value
        elif type(value) is int:
            alternate = value + 1
        elif type(value) is float:
            alternate = value + 1.0
        elif isinstance(value, str):
            alternate = f"{value}-alternate" if value else "contract-alternate"
        elif isinstance(value, list):
            alternate = (
                ["contract-alternate"]
                if field_name in list_of_string_fields
                else [{"contract": "alternate"}]
            )
        elif isinstance(value, dict):
            alternate = {"contract": 1}
        else:  # pragma: no cover - every current optional field is classified above
            raise AssertionError(f"unclassified checkpoint identity field: {field_name}")
        variant_payload = dict(payload)
        variant_payload[field_name] = alternate
        variant = RunCheckpoint.from_json(variant_payload)
        if variant is None:  # pragma: no cover - current-schema variants must decode
            raise AssertionError(f"invalid checkpoint identity variant: {field_name}")
        variants[field_name] = variant
    if set(variants) != identity_fields:  # pragma: no cover - loop is intentionally exhaustive
        raise AssertionError("checkpoint identity matrix is incomplete")
    return variants


def _contract_checkpoint_canonical_alias_status(
    factory: FencedRunSinkHarnessFactory,
    run_id: str,
    field_name: str,
    *,
    legacy_first: bool,
) -> str:
    """Retry one checkpoint across either direction of every accepted alias location."""

    invocation_payload = _contract_invocation(
        run_id,
        revision=1,
        dispatch_state="reserved",
    ).to_json()
    baseline = RunCheckpoint(
        run_id=run_id,
        seq=1,
        last_model_invocation=(
            invocation_payload if field_name.startswith("last_model_invocation_") else None
        ),
    )
    if field_name == "schema_version":
        legacy = replace(
            baseline,
            schema_version=_CONTRACT_ALTERNATE_CHECKPOINT_SCHEMA_VERSION,
        )
    else:
        legacy_invocation = dict(invocation_payload)
        nested_field = field_name.removeprefix("last_model_invocation_")
        alias_values = {
            "schema_version": _CONTRACT_ALTERNATE_INVOCATION_SCHEMA_VERSION,
            "digest_generation": _CONTRACT_ALTERNATE_DIGEST_GENERATION,
        }
        if set(alias_values) != _CONTRACT_INVOCATION_FIXED_CANONICAL_FIELDS:
            raise AssertionError("checkpoint nested invocation alias matrix is incomplete")
        legacy_invocation[nested_field] = alias_values[nested_field]
        legacy = replace(baseline, last_model_invocation=legacy_invocation)
    first_value, retry_value = (legacy, baseline) if legacy_first else (baseline, legacy)
    harness = factory()
    token = _contract_writer(harness, run_id)
    first = harness.sink.commit_checkpoint(first_value, {}, writer_token=token)
    if first.status != "committed":
        return f"setup:{first.status}"
    harness = harness.reopen()
    return harness.sink.commit_checkpoint(retry_value, {}, writer_token=token).status


def _contract_event_identity_variants(event: AgentEvent) -> dict[str, AgentEvent]:
    """Vary every canonical non-key event field independently."""

    variants = {
        "schema_version": replace(event, schema_version="monoid.event.v2"),
        "event_id": replace(event, event_id="event-alternate"),
        "timestamp": replace(event, timestamp="2026-08-21T00:00:01Z"),
        "type": replace(event, type="run.started"),
        "level": replace(event, level="warning"),
        "data": replace(event, data={"checkpoint_seq": event.seq, "alternate": True}),
        "turn_id": replace(event, turn_id="turn-alternate"),
        "parent_id": replace(event, parent_id="event-parent-alternate"),
    }
    identity_fields = set(event.to_json()) - {"run_id", "seq"}
    if set(variants) != identity_fields:
        raise AssertionError("event identity matrix is incomplete")
    return variants


def _contract_terminal_identity_variants(
    outcome: TerminalOutcome,
) -> dict[str, TerminalOutcome]:
    """Vary every canonical non-key terminal field independently."""

    variants = {
        "kind": replace(outcome, kind="failed_terminal"),
        "retry_eligibility": replace(
            outcome,
            retry_eligibility=RetryEligibility.FORBIDDEN,
        ),
        "interruption_cause": replace(
            outcome,
            interruption_cause=InterruptionCause.USER_CANCEL,
        ),
        "checkpoint_seq": replace(outcome, checkpoint_seq=1),
        "final_output_ref": replace(outcome, final_output_ref="blob:contract-alternate"),
        "partial_output_ref": replace(outcome, partial_output_ref="blob:contract-partial"),
        "last_evidence_ref": replace(outcome, last_evidence_ref="blob:contract-evidence"),
        "error_code": replace(outcome, error_code="contract_alternate"),
        "provider_error_code": replace(
            outcome,
            provider_error_code="provider_alternate",
        ),
        "http_status": replace(outcome, http_status=503),
    }
    identity_fields = set(outcome.to_json()) - {"schema_version", "run_id"}
    if set(variants) != identity_fields:
        raise AssertionError("terminal identity matrix is incomplete")
    return variants


def _contract_invocation(
    run_id: str,
    *,
    logical_call_id: str = "call-1",
    revision: int,
    dispatch_attempt: int = 1,
    dispatch_id: str = "dispatch-1",
    dispatch_state: str,
    idempotency_key: str | None = None,
    request_digest: str = "a" * 64,
    digest_generation: str = MODEL_REQUEST_DIGEST_GENERATION,
    retryable: bool | None = False,
    succeeded: bool = False,
) -> DurableModelInvocation:
    if idempotency_key is None:
        idempotency_key = f"contract-{hashlib.sha256(run_id.encode()).hexdigest()}"
    receipt: dict[str, Any] | None = None
    result_ref = ""
    failure_code = ""
    if dispatch_state == "settled":
        receipt = {"request_digest": request_digest}
        if retryable is not None:
            receipt["retryable"] = retryable
        if succeeded:
            result_ref = f"blob:{_CONTRACT_INVOCATION_BLOB_SHA256}"
        else:
            failure_code = "provider_refused"
    return DurableModelInvocation(
        run_id=run_id,
        logical_call_id=logical_call_id,
        revision=revision,
        dispatch_id=dispatch_id,
        dispatch_attempt=dispatch_attempt,
        idempotency_key=idempotency_key,
        dispatch_state=dispatch_state,  # type: ignore[arg-type]
        request_digest=request_digest,
        digest_generation=digest_generation,
        receipt=receipt,
        result_ref=result_ref,
        failure_code=failure_code,
    )


def _contract_queued_media_entries(
    run_id: str,
    source_ref: str,
) -> dict[str, Any]:
    media_part = {
        "type": "image",
        "source_ref": source_ref,
        "mime_type": "image/png",
    }
    entries = {
        "content_list": [dict(media_part)],
        "envelope": InboxMessage(
            content=[dict(media_part)],
            id="contract-queued-media",
            run_id=run_id,
            created_at=0.0,
        ).to_json(),
    }
    if tuple(entries) != _CONTRACT_QUEUED_MEDIA_CARRIERS:
        raise AssertionError("queued media carrier matrix is incomplete")
    return entries


def _contract_writer(
    harness: FencedRunSinkHarness,
    run_id: str,
    owner_id: str = "owner-a",
    *,
    generation: int = 1,
) -> WriterToken:
    token = WriterToken(run_id=run_id, owner_id=owner_id, generation=generation)
    harness.set_current_writer(token)
    return token


def _contract_run_id(namespace: str, suffix: str) -> str:
    return f"{namespace}-{suffix}"


def _contract_blob_hex(record: Any | None, sha256: str) -> str | None:
    if record is None:
        return None
    try:
        blob = record.blob(sha256)
    except Exception:
        return "unreadable"
    if type(blob) is not bytes:
        return "invalid-type"
    return blob.hex()


def _contract_record_digest(
    payload: dict[str, Any],
    blobs: Mapping[str, bytes] = {},
) -> str:
    return canonical_sha256(
        {
            "record": payload,
            "blobs": {
                key: hashlib.sha256(value).hexdigest() for key, value in sorted(blobs.items())
            },
        }
    )


def _contract_commit_evidence(
    result: CommitResult,
    *,
    sequence: int | None,
    content_digest: str,
    winner_digest: str = "",
) -> tuple[bool, bool, bool]:
    """Validate each optional result field when the adapter chooses to populate it."""

    sequence_ok = result.sequence is None or result.sequence == sequence
    content_ok = not result.content_digest or result.content_digest == content_digest
    winner_ok = (
        (not result.winner_digest or result.winner_digest == winner_digest)
        if winner_digest
        else not result.winner_digest
    )
    return sequence_ok, content_ok, winner_ok


def _contract_race_retry_statuses(
    left_result: CommitResult,
    right_result: CommitResult,
    reopened: FencedRunSinkHarness,
    left: Callable[[FencedRunSink], CommitResult],
    right: Callable[[FencedRunSink], CommitResult],
) -> tuple[str, str]:
    if left_result.status == "committed":
        return left(reopened.sink).status, right(reopened.sink).status
    if right_result.status == "committed":
        return right(reopened.sink).status, left(reopened.sink).status
    return "no-winner", "no-loser"


def _contract_race_write(
    sink: FencedRunSink,
    *,
    mutation: str,
    value: Any,
    blobs: Mapping[str, bytes],
    writer_token: WriterToken,
) -> CommitResult:
    if mutation == "checkpoint":
        return sink.commit_checkpoint(value, blobs, writer_token=writer_token)
    if mutation == "event":
        return sink.append_event(value, writer_token=writer_token)
    if mutation == "invocation":
        return sink.commit_invocation(value, blobs, writer_token=writer_token)
    return sink.settle_terminal(value, writer_token=writer_token)


def _contract_handoff_write(
    sink: FencedRunSink,
    writer_token: WriterToken,
    *,
    mutation: str,
    stale_token: WriterToken,
    stale_value: Any,
    stale_blobs: Mapping[str, bytes],
    current_value: Any,
    current_blobs: Mapping[str, bytes],
) -> CommitResult:
    value, blobs = (
        (stale_value, stale_blobs)
        if writer_token == stale_token
        else (current_value, current_blobs)
    )
    return _contract_race_write(
        sink,
        mutation=mutation,
        value=value,
        blobs=blobs,
        writer_token=writer_token,
    )


def _contract_authority_probe_statuses(
    sink: FencedRunSink,
    *,
    checkpoint: RunCheckpoint,
    event: AgentEvent,
    invocation: DurableModelInvocation,
    terminal: TerminalOutcome,
    writer_token: WriterToken,
) -> dict[str, str]:
    """Exercise every authoritative mutation with one deliberately invalid token."""

    return {
        "checkpoint": sink.commit_checkpoint(
            checkpoint,
            {},
            writer_token=writer_token,
        ).status,
        "event": sink.append_event(event, writer_token=writer_token).status,
        "invocation": sink.commit_invocation(
            invocation,
            {},
            writer_token=writer_token,
        ).status,
        "terminal": sink.settle_terminal(terminal, writer_token=writer_token).status,
    }


def _contract_competing_values(mutation: str, run_id: str) -> tuple[Any, Any]:
    if mutation == "checkpoint":
        workspace_delta = [
            {
                "path": "race-contract.txt",
                "kind": "file",
                "change_kind": "created",
                "content_sha256": _CONTRACT_CHECKPOINT_BLOB_SHA256,
            }
        ]
        return (
            RunCheckpoint(
                run_id=run_id,
                seq=1,
                final_text="left",
                workspace_delta=workspace_delta,
            ),
            RunCheckpoint(
                run_id=run_id,
                seq=1,
                final_text="right",
                workspace_delta=workspace_delta,
            ),
        )
    if mutation == "event":
        return _contract_event(run_id, seq=1), _contract_event(run_id, seq=1, level="warning")
    if mutation == "invocation":
        left = _contract_invocation(
            run_id,
            revision=3,
            dispatch_state="settled",
            succeeded=True,
        )
        return left, replace(
            left,
            receipt={
                **dict(left.receipt or {}),
                "provider_request_id": "provider-racer",
            },
        )
    return _contract_terminal(run_id), _contract_terminal(run_id, failed=True)


def _contract_mutation_blobs(mutation: str) -> Mapping[str, bytes]:
    if mutation == "checkpoint":
        return {_CONTRACT_CHECKPOINT_BLOB_SHA256: _CONTRACT_CHECKPOINT_BLOB}
    if mutation == "invocation":
        return {_CONTRACT_INVOCATION_BLOB_SHA256: _CONTRACT_INVOCATION_BLOB}
    return {}


def _contract_prepare_invocation_race(
    harness: FencedRunSinkHarness,
    run_id: str,
    writer_token: WriterToken,
) -> None:
    statuses = tuple(
        harness.sink.commit_invocation(
            _contract_invocation(
                run_id,
                revision=revision,
                dispatch_state=dispatch_state,
            ),
            {},
            writer_token=writer_token,
        ).status
        for revision, dispatch_state in ((1, "reserved"), (2, "dispatch_started"))
    )
    if statuses != ("committed", "committed"):
        raise AssertionError(f"invocation race setup failed: {statuses!r}")


def _contract_race_blob_hex(
    harness: FencedRunSinkHarness,
    mutation: str,
    run_id: str,
    sha256: str | None = None,
) -> str | None:
    if mutation == "checkpoint":
        return _contract_blob_hex(
            harness.sink.latest_checked(run_id).value,
            sha256 or _CONTRACT_CHECKPOINT_BLOB_SHA256,
        )
    if mutation == "invocation":
        return _contract_blob_hex(
            harness.sink.load_invocation(run_id, "call-1").value,
            sha256 or _CONTRACT_INVOCATION_BLOB_SHA256,
        )
    return None


def _contract_stale_handoff_blob_probe(
    harness: FencedRunSinkHarness,
    mutation: str,
    run_id: str,
    writer_token: WriterToken,
) -> str:
    if mutation == "checkpoint":
        checkpoint = RunCheckpoint(
            run_id=run_id,
            seq=2,
            workspace_delta=[
                {
                    "path": "stale-handoff-probe.txt",
                    "kind": "file",
                    "change_kind": "created",
                    "content_sha256": _CONTRACT_STALE_HANDOFF_BLOB_SHA256,
                }
            ],
        )
        return harness.sink.commit_checkpoint(
            checkpoint,
            {},
            writer_token=writer_token,
        ).status
    if mutation == "invocation":
        logical_call_id = "stale-handoff-probe"
        setup_statuses = tuple(
            harness.sink.commit_invocation(
                _contract_invocation(
                    run_id,
                    logical_call_id=logical_call_id,
                    revision=revision,
                    dispatch_state=dispatch_state,
                ),
                {},
                writer_token=writer_token,
            ).status
            for revision, dispatch_state in ((1, "reserved"), (2, "dispatch_started"))
        )
        if setup_statuses != ("committed", "committed"):
            return f"setup:{','.join(setup_statuses)}"
        settled = replace(
            _contract_invocation(
                run_id,
                logical_call_id=logical_call_id,
                revision=3,
                dispatch_state="settled",
                succeeded=True,
            ),
            result_ref=f"blob:{_CONTRACT_STALE_HANDOFF_BLOB_SHA256}",
        )
        return harness.sink.commit_invocation(
            settled,
            {},
            writer_token=writer_token,
        ).status
    return "not_applicable"


def _contract_mutation_payload_digest(mutation: str, value: Any) -> str:
    payload = checkpoint_payload_for_write(value) if mutation == "checkpoint" else value.to_json()
    return canonical_sha256(payload)


def _contract_race_payload_digest(
    harness: FencedRunSinkHarness,
    mutation: str,
    run_id: str,
) -> str | None:
    value: Any | None
    if mutation == "checkpoint":
        record = harness.sink.latest_checked(run_id).value
        value = record.checkpoint if record else None
    elif mutation == "event":
        value = harness.read_event(run_id, 1)
    elif mutation == "invocation":
        record = harness.sink.load_invocation(run_id, "call-1").value
        value = record.invocation if record else None
    else:
        value = harness.read_terminal(run_id)
    if value is None:
        return None
    return _contract_mutation_payload_digest(mutation, value)


def _contract_invocation_drift_status(
    factory: FencedRunSinkHarnessFactory,
    run_id: str,
    field_name: str,
    field_value: Any,
) -> str:
    harness = factory()
    token = _contract_writer(harness, run_id)
    first = harness.sink.commit_invocation(
        _contract_invocation(run_id, revision=1, dispatch_state="reserved"),
        {},
        writer_token=token,
    )
    if first.status != "committed":
        return f"setup:{first.status}"
    overrides = {field_name: field_value}
    drifted = harness.sink.commit_invocation(
        _contract_invocation(
            run_id,
            revision=2,
            dispatch_state="dispatch_started",
            **overrides,
        ),
        {},
        writer_token=token,
    )
    return drifted.status


def _contract_terminal_invocation_drift_status(
    factory: FencedRunSinkHarnessFactory,
    run_id: str,
    terminal_state: str,
    field_name: str,
    field_value: Any,
) -> str:
    harness = factory()
    token = _contract_writer(harness, run_id)
    setup_statuses = tuple(
        harness.sink.commit_invocation(
            _contract_invocation(
                run_id,
                revision=revision,
                dispatch_state=dispatch_state,
            ),
            {},
            writer_token=token,
        ).status
        for revision, dispatch_state in ((1, "reserved"), (2, "dispatch_started"))
    )
    if setup_statuses != ("committed", "committed"):
        return f"setup:{','.join(setup_statuses)}"
    return harness.sink.commit_invocation(
        _contract_invocation(
            run_id,
            revision=3,
            dispatch_state=terminal_state,
            **{field_name: field_value},
        ),
        {},
        writer_token=token,
    ).status


def _contract_authoritative_load_fault_evidence(
    factory: FencedRunSinkHarnessFactory,
    namespace: str,
    record_family: Literal["checkpoint", "invocation"],
    status: Literal["corrupt", "unsupported_version"],
) -> tuple[str, bool]:
    harness = factory()
    run_id = _contract_run_id(namespace, f"{record_family}-load-{status.replace('_', '-')}")
    token = _contract_writer(harness, run_id)
    logical_call_id = "call-1"
    if record_family == "checkpoint":
        setup = harness.sink.commit_checkpoint(
            RunCheckpoint(run_id=run_id, seq=1, final_text="load-fault-seed"),
            {},
            writer_token=token,
        )
    else:
        setup = harness.sink.commit_invocation(
            _contract_invocation(
                run_id,
                logical_call_id=logical_call_id,
                revision=1,
                dispatch_state="reserved",
            ),
            {},
            writer_token=token,
        )
    if setup.status != "committed":
        return f"setup:{setup.status}", False
    harness.inject_authoritative_load_fault(
        record_family,
        run_id,
        status,
        logical_call_id=logical_call_id,
    )
    reopened = harness.reopen()
    if record_family == "checkpoint":
        loaded = reopened.sink.latest_checked(run_id)
    else:
        loaded = reopened.sink.load_invocation(run_id, logical_call_id)
    return loaded.status, loaded.value is None


def _contract_first_invocation_state_status(
    factory: FencedRunSinkHarnessFactory,
    run_id: str,
    dispatch_state: str,
    *,
    revision: int = 1,
    dispatch_attempt: int = 1,
) -> str:
    harness = factory()
    token = _contract_writer(harness, run_id)
    return harness.sink.commit_invocation(
        _contract_invocation(
            run_id,
            revision=revision,
            dispatch_attempt=dispatch_attempt,
            dispatch_state=dispatch_state,
        ),
        {},
        writer_token=token,
    ).status


def _contract_forbidden_state_edge_status(
    factory: FencedRunSinkHarnessFactory,
    run_id: str,
    source_state: str,
    target_state: str,
) -> str:
    harness = factory()
    token = _contract_writer(harness, run_id)
    histories = {
        "reserved": ("reserved",),
        "dispatch_started": ("reserved", "dispatch_started"),
        "settled": ("reserved", "dispatch_started", "settled"),
        "unknown": ("reserved", "dispatch_started", "unknown"),
    }
    history = histories[source_state]
    setup_statuses = tuple(
        harness.sink.commit_invocation(
            _contract_invocation(
                run_id,
                revision=revision,
                dispatch_state=state,
            ),
            {},
            writer_token=token,
        ).status
        for revision, state in enumerate(history, start=1)
    )
    if any(status != "committed" for status in setup_statuses):
        return f"setup:{','.join(setup_statuses)}"
    return harness.sink.commit_invocation(
        _contract_invocation(
            run_id,
            revision=len(history) + 1,
            dispatch_state=target_state,
        ),
        {},
        writer_token=token,
    ).status


def _contract_retry_coordinate_status(
    factory: FencedRunSinkHarnessFactory,
    run_id: str,
    *,
    dispatch_attempt: int,
    dispatch_id: str,
) -> str:
    harness = factory()
    token = _contract_writer(harness, run_id)
    history = (
        _contract_invocation(run_id, revision=1, dispatch_state="reserved"),
        _contract_invocation(run_id, revision=2, dispatch_state="dispatch_started"),
        _contract_invocation(
            run_id,
            revision=3,
            dispatch_state="settled",
            retryable=True,
        ),
    )
    setup_statuses = tuple(
        harness.sink.commit_invocation(invocation, {}, writer_token=token).status
        for invocation in history
    )
    if any(status != "committed" for status in setup_statuses):
        return f"setup:{','.join(setup_statuses)}"
    return harness.sink.commit_invocation(
        _contract_invocation(
            run_id,
            revision=4,
            dispatch_attempt=dispatch_attempt,
            dispatch_id=dispatch_id,
            dispatch_state="reserved",
        ),
        {},
        writer_token=token,
    ).status


def _contract_retry_identity_drift_status(
    factory: FencedRunSinkHarnessFactory,
    run_id: str,
    field_name: str,
) -> str:
    harness = factory()
    token = _contract_writer(harness, run_id)
    history = (
        _contract_invocation(run_id, revision=1, dispatch_state="reserved"),
        _contract_invocation(run_id, revision=2, dispatch_state="dispatch_started"),
        _contract_invocation(
            run_id,
            revision=3,
            dispatch_state="settled",
            retryable=True,
        ),
    )
    setup_statuses = tuple(
        harness.sink.commit_invocation(invocation, {}, writer_token=token).status
        for invocation in history
    )
    if any(status != "committed" for status in setup_statuses):
        return f"setup:{','.join(setup_statuses)}"
    baseline = _contract_invocation(
        run_id,
        revision=4,
        dispatch_attempt=2,
        dispatch_id="dispatch-2",
        dispatch_state="reserved",
    )
    variants = {
        "idempotency_key": replace(
            baseline,
            idempotency_key="contract-retry-idempotency-drift",
        ),
        "request_digest": replace(baseline, request_digest="b" * 64),
    }
    if set(variants) != _CONTRACT_RETRY_STABLE_IDENTITY_FIELDS:
        raise AssertionError("retry stable identity matrix is incomplete")
    return harness.sink.commit_invocation(
        variants[field_name],
        {},
        writer_token=token,
    ).status


def _contract_historical_dispatch_id_reuse_status(
    factory: FencedRunSinkHarnessFactory,
    run_id: str,
) -> tuple[str, tuple[str, ...]]:
    """Reject historical ID reuse, then complete attempt three with a fresh identity."""

    harness = factory()
    token = _contract_writer(harness, run_id)
    history = (
        _contract_invocation(run_id, revision=1, dispatch_state="reserved"),
        _contract_invocation(run_id, revision=2, dispatch_state="dispatch_started"),
        _contract_invocation(
            run_id,
            revision=3,
            dispatch_state="settled",
            retryable=True,
        ),
        _contract_invocation(
            run_id,
            revision=4,
            dispatch_attempt=2,
            dispatch_id="dispatch-2",
            dispatch_state="reserved",
        ),
        _contract_invocation(
            run_id,
            revision=5,
            dispatch_attempt=2,
            dispatch_id="dispatch-2",
            dispatch_state="dispatch_started",
        ),
        _contract_invocation(
            run_id,
            revision=6,
            dispatch_attempt=2,
            dispatch_id="dispatch-2",
            dispatch_state="settled",
            retryable=True,
        ),
    )
    setup_statuses = tuple(
        harness.sink.commit_invocation(invocation, {}, writer_token=token).status
        for invocation in history
    )
    if any(status != "committed" for status in setup_statuses):
        return f"setup:{','.join(setup_statuses)}", ()
    harness = harness.reopen()
    historical_reuse = harness.sink.commit_invocation(
        _contract_invocation(
            run_id,
            revision=7,
            dispatch_attempt=3,
            dispatch_id="dispatch-1",
            dispatch_state="reserved",
        ),
        {},
        writer_token=token,
    )
    valid_third_attempt = tuple(
        harness.sink.commit_invocation(
            _contract_invocation(
                run_id,
                revision=revision,
                dispatch_attempt=3,
                dispatch_id="dispatch-3",
                dispatch_state=dispatch_state,
            ),
            {},
            writer_token=token,
        ).status
        for revision, dispatch_state in (
            (7, "reserved"),
            (8, "dispatch_started"),
            (9, "settled"),
        )
    )
    return historical_reuse.status, valid_third_attempt


def _contract_invocation_identity_status(
    factory: FencedRunSinkHarnessFactory,
    run_id: str,
    field_name: str,
) -> str:
    """Retry one invocation revision with exactly one canonical non-key field changed."""

    canonical_fields = set(
        _contract_invocation(
            run_id,
            revision=1,
            dispatch_state="reserved",
        ).to_json()
    )
    identity_fields = (
        canonical_fields
        - {
            "run_id",
            "logical_call_id",
            "revision",
        }
        - _CONTRACT_INVOCATION_FIXED_CANONICAL_FIELDS
    )
    if identity_fields != _CONTRACT_INVOCATION_IDENTITY_FIELDS:
        raise AssertionError("invocation identity matrix is incomplete")
    harness = factory()
    token = _contract_writer(harness, run_id)
    blobs: Mapping[str, bytes] = {}
    if field_name in {"receipt", "result_ref", "failure_code"}:
        setup = (
            _contract_invocation(run_id, revision=1, dispatch_state="reserved"),
            _contract_invocation(run_id, revision=2, dispatch_state="dispatch_started"),
        )
        setup_statuses = tuple(
            harness.sink.commit_invocation(
                invocation,
                {},
                writer_token=token,
            ).status
            for invocation in setup
        )
        if any(status != "committed" for status in setup_statuses):
            return f"setup:{','.join(setup_statuses)}"
        if field_name == "failure_code":
            baseline = _contract_invocation(
                run_id,
                revision=3,
                dispatch_state="settled",
            )
            variant = replace(baseline, failure_code="provider_timeout")
        else:
            baseline = _contract_invocation(
                run_id,
                revision=3,
                dispatch_state="settled",
                succeeded=True,
            )
            blobs = {
                _CONTRACT_INVOCATION_BLOB_SHA256: _CONTRACT_INVOCATION_BLOB,
            }
            if field_name == "receipt":
                variant = replace(
                    baseline,
                    receipt={
                        **dict(baseline.receipt or {}),
                        "provider_request_id": "provider-alternate",
                    },
                )
            else:
                variant = replace(
                    baseline,
                    result_ref="object:contract-alternate",
                )
    else:
        baseline = _contract_invocation(
            run_id,
            revision=1,
            dispatch_state="reserved",
        )
        variants = {
            "dispatch_id": replace(baseline, dispatch_id="dispatch-alternate"),
            "dispatch_attempt": replace(baseline, dispatch_attempt=2),
            "idempotency_key": replace(
                baseline,
                idempotency_key="contract-idempotency-alternate",
            ),
            "dispatch_state": replace(baseline, dispatch_state="dispatch_started"),
            "request_digest": replace(baseline, request_digest="b" * 64),
        }
        variant = variants[field_name]
    first = harness.sink.commit_invocation(
        baseline,
        blobs,
        writer_token=token,
    )
    if first.status != "committed":
        return f"setup:{first.status}"
    return harness.sink.commit_invocation(
        variant,
        blobs,
        writer_token=token,
    ).status


def _contract_receipt_retryability_identity_evidence(
    factory: FencedRunSinkHarnessFactory,
    run_id: str,
    *,
    winner_state: str,
    candidate_state: str,
) -> tuple[str, tuple[str, str | None]]:
    retryability_values = {
        "true": True,
        "false": False,
        "omitted": None,
    }
    if tuple(retryability_values) != _CONTRACT_RECEIPT_RETRYABILITY_STATES:
        raise AssertionError("receipt retryability state matrix is incomplete")
    harness = factory()
    token = _contract_writer(harness, run_id)
    setup = (
        _contract_invocation(run_id, revision=1, dispatch_state="reserved"),
        _contract_invocation(run_id, revision=2, dispatch_state="dispatch_started"),
    )
    setup_statuses = tuple(
        harness.sink.commit_invocation(invocation, {}, writer_token=token).status
        for invocation in setup
    )
    if any(status != "committed" for status in setup_statuses):
        return "", (f"setup:{','.join(setup_statuses)}", None)
    winner = _contract_invocation(
        run_id,
        revision=3,
        dispatch_state="settled",
        retryable=retryability_values[winner_state],
    )
    first = harness.sink.commit_invocation(winner, {}, writer_token=token)
    if first.status != "committed":
        return "", (f"setup:{first.status}", None)
    harness = harness.reopen()
    candidate = _contract_invocation(
        run_id,
        revision=3,
        dispatch_state="settled",
        retryable=retryability_values[candidate_state],
    )
    conflict = harness.sink.commit_invocation(candidate, {}, writer_token=token)
    harness = harness.reopen()
    loaded = harness.sink.load_invocation(run_id, "call-1")
    winner_digest = canonical_sha256(winner.to_json())
    loaded_digest = canonical_sha256(loaded.value.invocation.to_json()) if loaded.value else None
    return winner_digest, (conflict.status, loaded_digest)


def _contract_receipt_field_identity_evidence(
    factory: FencedRunSinkHarnessFactory,
    run_id: str,
    field_name: str,
    *,
    usage_field_name: str = "",
) -> tuple[str, tuple[str, str | None, str, str | None]]:
    receipt_values: dict[str, tuple[Any, Any]] = {
        "attempts": (1, 2),
        "duration_ms": (1.0, 2.0),
        "finish_reason": ("stop", "length"),
        "http_status": (200, 503),
        "latency_ms": (3.0, 4.0),
        "provider_error_code": ("provider_refused", "provider_timeout"),
        "provider_request_id": ("provider_request_1", "provider_request_2"),
        "provider_response_id": ("provider_response_1", "provider_response_2"),
        "provider_retried": (False, True),
        "request_digest": ("a" * 64, None),
        "request_id": ("request_1", "request_2"),
        "response_id": ("response_1", "response_2"),
        "retryable": (False, True),
        "settled_at": ("2026-08-21T00:00:00Z", "2026-08-21T00:00:01Z"),
        "started_at": ("2026-08-21T00:00:00Z", "2026-08-21T00:00:01Z"),
        "stop_reason": ("end_turn", "max_tokens"),
        "system_fingerprint": ("fingerprint_1", "fingerprint_2"),
        "usage": ({"input_tokens": 1}, {"input_tokens": 2}),
    }
    if set(receipt_values) != MODEL_INVOCATION_RECEIPT_FIELDS:
        raise AssertionError("receipt field identity matrix is incomplete")
    if usage_field_name:
        if field_name != "usage":
            raise AssertionError("nested receipt identity field must belong to usage")
        if usage_field_name not in MODEL_INVOCATION_RECEIPT_USAGE_FIELDS:
            raise AssertionError("unknown receipt usage identity field")
        winner_receipt = {"usage": {usage_field_name: 1}}
        candidate_receipt = {"usage": {usage_field_name: 2}}
    else:
        winner_value, candidate_value = receipt_values[field_name]
        winner_receipt = {field_name: winner_value}
        candidate_receipt = {} if field_name == "request_digest" else {field_name: candidate_value}
    harness = factory()
    token = _contract_writer(harness, run_id)
    setup = (
        _contract_invocation(run_id, revision=1, dispatch_state="reserved"),
        _contract_invocation(run_id, revision=2, dispatch_state="dispatch_started"),
    )
    setup_statuses = tuple(
        harness.sink.commit_invocation(invocation, {}, writer_token=token).status
        for invocation in setup
    )
    if any(status != "committed" for status in setup_statuses):
        return "", (f"setup:{','.join(setup_statuses)}", None, "not_run", None)
    winner = replace(
        _contract_invocation(
            run_id,
            revision=3,
            dispatch_state="settled",
            retryable=None,
        ),
        receipt=winner_receipt,
    )
    first = harness.sink.commit_invocation(winner, {}, writer_token=token)
    harness = harness.reopen()
    loaded_winner = harness.sink.load_invocation(run_id, "call-1")
    candidate = replace(winner, receipt=candidate_receipt)
    conflict = harness.sink.commit_invocation(candidate, {}, writer_token=token)
    harness = harness.reopen()
    loaded_after_conflict = harness.sink.load_invocation(run_id, "call-1")
    winner_digest = canonical_sha256(winner.to_json())
    return winner_digest, (
        first.status,
        (
            canonical_sha256(loaded_winner.value.invocation.to_json())
            if loaded_winner.value
            else None
        ),
        conflict.status,
        (
            canonical_sha256(loaded_after_conflict.value.invocation.to_json())
            if loaded_after_conflict.value
            else None
        ),
    )


def _contract_invocation_canonical_alias_status(
    factory: FencedRunSinkHarnessFactory,
    run_id: str,
    field_name: str,
    *,
    legacy_first: bool,
) -> str:
    """Prove same-revision aliases normalize in either retry direction."""

    harness = factory()
    token = _contract_writer(harness, run_id)
    baseline = _contract_invocation(
        run_id,
        revision=1,
        dispatch_state="reserved",
    )
    variants = {
        "schema_version": replace(
            baseline,
            schema_version=_CONTRACT_ALTERNATE_INVOCATION_SCHEMA_VERSION,
        ),
        "digest_generation": replace(
            baseline,
            digest_generation=_CONTRACT_ALTERNATE_DIGEST_GENERATION,
        ),
    }
    if set(variants) != _CONTRACT_INVOCATION_FIXED_CANONICAL_FIELDS:
        raise AssertionError("invocation canonical-tag matrix is incomplete")
    legacy = variants[field_name]
    first_value, retry_value = (legacy, baseline) if legacy_first else (baseline, legacy)
    first = harness.sink.commit_invocation(first_value, {}, writer_token=token)
    if first.status != "committed":
        return f"setup:{first.status}"
    return harness.sink.commit_invocation(
        retry_value,
        {},
        writer_token=token,
    ).status


def _contract_invocation_canonical_alias_transition_status(
    factory: FencedRunSinkHarnessFactory,
    run_id: str,
    field_name: str,
    *,
    legacy_first: bool,
) -> str:
    """Carry an accepted legacy tag across either direction of a legal transition."""

    harness = factory()
    token = _contract_writer(harness, run_id)
    reserved = _contract_invocation(
        run_id,
        revision=1,
        dispatch_state="reserved",
    )
    alias_values = {
        "schema_version": _CONTRACT_ALTERNATE_INVOCATION_SCHEMA_VERSION,
        "digest_generation": _CONTRACT_ALTERNATE_DIGEST_GENERATION,
    }
    if set(alias_values) != _CONTRACT_INVOCATION_FIXED_CANONICAL_FIELDS:
        raise AssertionError("invocation canonical-tag transition matrix is incomplete")
    if legacy_first:
        reserved = replace(reserved, **{field_name: alias_values[field_name]})
    first = harness.sink.commit_invocation(reserved, {}, writer_token=token)
    if first.status != "committed":
        return f"setup:{first.status}"
    started = _contract_invocation(
        run_id,
        revision=2,
        dispatch_state="dispatch_started",
    )
    if not legacy_first:
        started = replace(started, **{field_name: alias_values[field_name]})
    return harness.sink.commit_invocation(
        started,
        {},
        writer_token=token,
    ).status


def _contract_retry_after_terminal_invocation(
    factory: FencedRunSinkHarnessFactory,
    run_id: str,
    *,
    terminal_state: str,
    retryable: bool | None = False,
    succeeded: bool = False,
    unknown_failure_code: str = "",
) -> tuple[tuple[str, ...], str]:
    harness = factory()
    token = _contract_writer(harness, run_id)
    terminal_invocation = _contract_invocation(
        run_id,
        revision=3,
        dispatch_state=terminal_state,
        retryable=retryable,
        succeeded=succeeded,
    )
    if unknown_failure_code:
        terminal_invocation = replace(
            terminal_invocation,
            failure_code=unknown_failure_code,
        )
    history = (
        _contract_invocation(run_id, revision=1, dispatch_state="reserved"),
        _contract_invocation(run_id, revision=2, dispatch_state="dispatch_started"),
        terminal_invocation,
    )
    history_statuses = tuple(
        harness.sink.commit_invocation(
            invocation,
            (
                {_CONTRACT_INVOCATION_BLOB_SHA256: _CONTRACT_INVOCATION_BLOB}
                if invocation.result_ref
                else {}
            ),
            writer_token=token,
        ).status
        for invocation in history
    )
    retry = harness.sink.commit_invocation(
        _contract_invocation(
            run_id,
            revision=4,
            dispatch_attempt=2,
            dispatch_id="dispatch-2",
            dispatch_state="reserved",
        ),
        {},
        writer_token=token,
    )
    return history_statuses, retry.status


def _run_fenced_run_sink_contract(
    factory: FencedRunSinkHarnessFactory,
) -> tuple[ConformanceRuleOutcome, ...]:
    outcomes: list[ConformanceRuleOutcome] = []
    namespace = f"contract-{uuid4().hex}"
    try:
        harness = factory()
        capabilities = harness.sink.capabilities
        required = (
            "concurrent_writers",
            "compare_and_set",
            "lease_fencing",
            "durable_checkpoints",
            "durable_events",
            "durable_invocations",
            "terminal_first_writer_wins",
        )
        outcomes.append(
            outcome_from_observations(
                "FENCED-00-CAPABILITY-DECLARATION",
                FENCED_RUN_SINK_CONTRACT_PROFILE,
                tuple(
                    observation(
                        f"declares_{field_name}",
                        expected=True,
                        actual=getattr(capabilities, field_name),
                    )
                    for field_name in required
                ),
            )
        )
    except Exception as exc:
        outcomes.append(
            _error(
                "FENCED-00-CAPABILITY-DECLARATION",
                FENCED_RUN_SINK_CONTRACT_PROFILE,
                exc,
            )
        )

    try:
        checkpoint_canonical_alias_statuses = {
            (field_name, direction): _contract_checkpoint_canonical_alias_status(
                factory,
                _contract_run_id(
                    namespace,
                    f"checkpoint-canonical-alias-{field_name}-{direction}",
                ),
                field_name,
                legacy_first=direction == "legacy_to_current",
            )
            for field_name in sorted(_CONTRACT_CHECKPOINT_CANONICAL_ALIAS_FIELDS)
            for direction in ("current_to_legacy", "legacy_to_current")
        }
        missing_reference_harness = factory()
        missing_reference_run_id = _contract_run_id(
            namespace,
            "checkpoint-missing-reference",
        )
        missing_reference_token = _contract_writer(
            missing_reference_harness,
            missing_reference_run_id,
        )
        missing_reference_checkpoint = RunCheckpoint(
            run_id=missing_reference_run_id,
            seq=1,
            workspace_delta=[
                {
                    "path": "missing-reference.txt",
                    "kind": "file",
                    "change_kind": "created",
                    "content_sha256": _CONTRACT_UNRESOLVED_BLOB_SHA256,
                }
            ],
        )
        missing_reference = missing_reference_harness.sink.commit_checkpoint(
            missing_reference_checkpoint,
            {},
            writer_token=missing_reference_token,
        )
        missing_reference_harness = missing_reference_harness.reopen()
        missing_reference_load = missing_reference_harness.sink.latest_checked(
            missing_reference_run_id
        )
        missing_reference_recovery = missing_reference_harness.sink.commit_checkpoint(
            missing_reference_checkpoint,
            {_CONTRACT_UNRESOLVED_BLOB_SHA256: _CONTRACT_UNRESOLVED_BLOB},
            writer_token=missing_reference_token,
        )
        missing_reference_harness = missing_reference_harness.reopen()
        missing_reference_recovery_load = missing_reference_harness.sink.latest_checked(
            missing_reference_run_id
        )
        missing_reference_recovery_bytes = _contract_blob_hex(
            missing_reference_recovery_load.value,
            _CONTRACT_UNRESOLVED_BLOB_SHA256,
        )
        missing_media_harness = factory()
        missing_media_run_id = _contract_run_id(
            namespace,
            "checkpoint-missing-media-reference",
        )
        missing_media_token = _contract_writer(
            missing_media_harness,
            missing_media_run_id,
        )
        missing_media_checkpoint = RunCheckpoint(
            run_id=missing_media_run_id,
            seq=1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source_ref": f"blob:{_CONTRACT_UNRESOLVED_BLOB_SHA256}",
                            "mime_type": "image/png",
                        }
                    ],
                }
            ],
        )
        missing_media = missing_media_harness.sink.commit_checkpoint(
            missing_media_checkpoint,
            {},
            writer_token=missing_media_token,
        )
        missing_media_harness = missing_media_harness.reopen()
        missing_media_load = missing_media_harness.sink.latest_checked(missing_media_run_id)
        missing_media_recovery = missing_media_harness.sink.commit_checkpoint(
            missing_media_checkpoint,
            {_CONTRACT_UNRESOLVED_BLOB_SHA256: _CONTRACT_UNRESOLVED_BLOB},
            writer_token=missing_media_token,
        )
        missing_media_harness = missing_media_harness.reopen()
        missing_media_recovery_load = missing_media_harness.sink.latest_checked(
            missing_media_run_id
        )
        missing_media_recovery_bytes = _contract_blob_hex(
            missing_media_recovery_load.value,
            _CONTRACT_UNRESOLVED_BLOB_SHA256,
        )
        missing_queued_media_evidence: dict[
            str,
            tuple[str, str, str, str | None],
        ] = {}
        for carrier in _CONTRACT_QUEUED_MEDIA_CARRIERS:
            queued_harness = factory()
            queued_run_id = _contract_run_id(
                namespace,
                f"checkpoint-missing-queued-media-{carrier.replace('_', '-')}",
            )
            queued_entry = _contract_queued_media_entries(
                queued_run_id,
                f"blob:{_CONTRACT_UNRESOLVED_BLOB_SHA256}",
            )[carrier]
            queued_token = _contract_writer(queued_harness, queued_run_id)
            queued_checkpoint = RunCheckpoint(
                run_id=queued_run_id,
                seq=1,
                queued_messages=[queued_entry],
            )
            rejected = queued_harness.sink.commit_checkpoint(
                queued_checkpoint,
                {},
                writer_token=queued_token,
            )
            queued_harness = queued_harness.reopen()
            rejected_load = queued_harness.sink.latest_checked(queued_run_id)
            recovered = queued_harness.sink.commit_checkpoint(
                queued_checkpoint,
                {_CONTRACT_UNRESOLVED_BLOB_SHA256: _CONTRACT_UNRESOLVED_BLOB},
                writer_token=queued_token,
            )
            queued_harness = queued_harness.reopen()
            recovered_load = queued_harness.sink.latest_checked(queued_run_id)
            missing_queued_media_evidence[carrier] = (
                rejected.status,
                rejected_load.status,
                recovered.status,
                _contract_blob_hex(
                    recovered_load.value,
                    _CONTRACT_UNRESOLVED_BLOB_SHA256,
                ),
            )
        malformed_workspace_harness = factory()
        malformed_workspace_run_id = _contract_run_id(
            namespace,
            "checkpoint-malformed-workspace-reference",
        )
        malformed_workspace_token = _contract_writer(
            malformed_workspace_harness,
            malformed_workspace_run_id,
        )
        malformed_workspace_reference = malformed_workspace_harness.sink.commit_checkpoint(
            RunCheckpoint(
                run_id=malformed_workspace_run_id,
                seq=1,
                workspace_delta=[
                    {
                        "path": "malformed-workspace-reference.txt",
                        "kind": "file",
                        "change_kind": "created",
                        "content_sha256": _CONTRACT_MALFORMED_BLOB_SHA256,
                    }
                ],
            ),
            {},
            writer_token=malformed_workspace_token,
        )
        malformed_workspace_harness = malformed_workspace_harness.reopen()
        malformed_workspace_load = malformed_workspace_harness.sink.latest_checked(
            malformed_workspace_run_id
        )
        malformed_media_harness = factory()
        malformed_media_run_id = _contract_run_id(
            namespace,
            "checkpoint-malformed-media-reference",
        )
        malformed_media_token = _contract_writer(
            malformed_media_harness,
            malformed_media_run_id,
        )
        malformed_media_reference = malformed_media_harness.sink.commit_checkpoint(
            RunCheckpoint(
                run_id=malformed_media_run_id,
                seq=1,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source_ref": (f"blob:{_CONTRACT_MALFORMED_BLOB_SHA256}"),
                                "mime_type": "image/png",
                            }
                        ],
                    }
                ],
            ),
            {},
            writer_token=malformed_media_token,
        )
        malformed_media_harness = malformed_media_harness.reopen()
        malformed_media_load = malformed_media_harness.sink.latest_checked(malformed_media_run_id)
        malformed_queued_media_evidence: dict[str, tuple[str, str]] = {}
        for carrier in _CONTRACT_QUEUED_MEDIA_CARRIERS:
            queued_harness = factory()
            queued_run_id = _contract_run_id(
                namespace,
                f"checkpoint-malformed-queued-media-{carrier.replace('_', '-')}",
            )
            queued_entry = _contract_queued_media_entries(
                queued_run_id,
                f"blob:{_CONTRACT_MALFORMED_BLOB_SHA256}",
            )[carrier]
            queued_token = _contract_writer(queued_harness, queued_run_id)
            malformed = queued_harness.sink.commit_checkpoint(
                RunCheckpoint(
                    run_id=queued_run_id,
                    seq=1,
                    queued_messages=[queued_entry],
                ),
                {},
                writer_token=queued_token,
            )
            queued_harness = queued_harness.reopen()
            malformed_load = queued_harness.sink.latest_checked(queued_run_id)
            malformed_queued_media_evidence[carrier] = (
                malformed.status,
                malformed_load.status,
            )
        backing_harness = factory()
        backing_run_id = _contract_run_id(
            namespace,
            "checkpoint-authoritative-backing-reference",
        )
        backing_token = _contract_writer(backing_harness, backing_run_id)
        backing_seed_checkpoint = RunCheckpoint(
            run_id=backing_run_id,
            seq=1,
            workspace_delta=[
                {
                    "path": "backing-seed.txt",
                    "kind": "file",
                    "change_kind": "created",
                    "content_sha256": _CONTRACT_CHECKPOINT_BLOB_SHA256,
                }
            ],
        )
        backing_seed = backing_harness.sink.commit_checkpoint(
            backing_seed_checkpoint,
            {_CONTRACT_CHECKPOINT_BLOB_SHA256: _CONTRACT_CHECKPOINT_BLOB},
            writer_token=backing_token,
        )
        backing_harness = backing_harness.reopen()
        backing_checkpoint = RunCheckpoint(
            run_id=backing_run_id,
            seq=2,
            messages=[
                {
                    "role": "tool",
                    "content": "media from authoritative backing",
                    "media": [
                        {
                            "type": "image",
                            "source_ref": f"blob:{_CONTRACT_CHECKPOINT_BLOB_SHA256}",
                            "mime_type": "image/png",
                        }
                    ],
                }
            ],
            queued_messages=list(
                _contract_queued_media_entries(
                    backing_run_id,
                    f"blob:{_CONTRACT_CHECKPOINT_BLOB_SHA256}",
                ).values()
            ),
            workspace_delta=[
                {
                    "path": "backing-reuse.txt",
                    "kind": "file",
                    "change_kind": "created",
                    "content_sha256": _CONTRACT_CHECKPOINT_BLOB_SHA256,
                }
            ],
        )
        backing_reference = backing_harness.sink.commit_checkpoint(
            backing_checkpoint,
            {},
            writer_token=backing_token,
        )
        backing_harness = backing_harness.reopen()
        backing_load = backing_harness.sink.latest_checked(backing_run_id)
        backing_reference_bytes = _contract_blob_hex(
            backing_load.value,
            _CONTRACT_CHECKPOINT_BLOB_SHA256,
        )
        uppercase_key_harness = factory()
        uppercase_key_run_id = _contract_run_id(
            namespace,
            "checkpoint-uppercase-blob-key",
        )
        uppercase_key_token = _contract_writer(
            uppercase_key_harness,
            uppercase_key_run_id,
        )
        uppercase_key_checkpoint = RunCheckpoint(
            run_id=uppercase_key_run_id,
            seq=1,
            final_text="uppercase blob key must be rejected",
        )
        uppercase_key = uppercase_key_harness.sink.commit_checkpoint(
            uppercase_key_checkpoint,
            {_CONTRACT_CHECKPOINT_BLOB_SHA256.upper(): _CONTRACT_CHECKPOINT_BLOB},
            writer_token=uppercase_key_token,
        )
        uppercase_key_harness = uppercase_key_harness.reopen()
        uppercase_key_load = uppercase_key_harness.sink.latest_checked(uppercase_key_run_id)
        uppercase_reference_checkpoint = RunCheckpoint(
            run_id=uppercase_key_run_id,
            seq=1,
            workspace_delta=[
                {
                    "path": "uppercase-key-recovery.txt",
                    "kind": "file",
                    "change_kind": "created",
                    "content_sha256": _CONTRACT_CHECKPOINT_BLOB_SHA256,
                }
            ],
        )
        uppercase_key_blob_leak = uppercase_key_harness.sink.commit_checkpoint(
            uppercase_reference_checkpoint,
            {},
            writer_token=uppercase_key_token,
        )
        uppercase_key_recovery = uppercase_key_harness.sink.commit_checkpoint(
            uppercase_reference_checkpoint,
            {_CONTRACT_CHECKPOINT_BLOB_SHA256: _CONTRACT_CHECKPOINT_BLOB},
            writer_token=uppercase_key_token,
        )
        uppercase_key_harness = uppercase_key_harness.reopen()
        uppercase_key_recovery_load = uppercase_key_harness.sink.latest_checked(
            uppercase_key_run_id
        )
        uppercase_key_recovery_bytes = _contract_blob_hex(
            uppercase_key_recovery_load.value,
            _CONTRACT_CHECKPOINT_BLOB_SHA256,
        )
        malformed_harness = factory()
        malformed_run_id = _contract_run_id(namespace, "checkpoint-malformed-fresh-blob")
        malformed_token = _contract_writer(malformed_harness, malformed_run_id)
        malformed_seed_checkpoint = RunCheckpoint(
            run_id=malformed_run_id,
            seq=1,
            workspace_delta=[
                {
                    "path": "seed-contract.txt",
                    "kind": "file",
                    "change_kind": "created",
                    "content_sha256": _CONTRACT_CHECKPOINT_BLOB_SHA256,
                }
            ],
        )
        malformed_seed = malformed_harness.sink.commit_checkpoint(
            malformed_seed_checkpoint,
            {_CONTRACT_CHECKPOINT_BLOB_SHA256: _CONTRACT_CHECKPOINT_BLOB},
            writer_token=malformed_token,
        )
        malformed_checkpoint = RunCheckpoint(
            run_id=malformed_run_id,
            seq=2,
            workspace_delta=[
                {
                    "path": "malformed-contract.txt",
                    "kind": "file",
                    "change_kind": "created",
                    "content_sha256": _CONTRACT_CHECKPOINT_BLOB_SHA256,
                }
            ],
        )
        malformed_fresh_blob = malformed_harness.sink.commit_checkpoint(
            malformed_checkpoint,
            {_CONTRACT_CHECKPOINT_BLOB_SHA256: _CONTRACT_ALTERNATE_BLOB},
            writer_token=malformed_token,
        )
        malformed_harness = malformed_harness.reopen()
        malformed_after_rejection = malformed_harness.sink.latest_checked(malformed_run_id)
        malformed_preserved_bytes = _contract_blob_hex(
            malformed_after_rejection.value,
            _CONTRACT_CHECKPOINT_BLOB_SHA256,
        )
        malformed_recovery = malformed_harness.sink.commit_checkpoint(
            malformed_checkpoint,
            {_CONTRACT_CHECKPOINT_BLOB_SHA256: _CONTRACT_CHECKPOINT_BLOB},
            writer_token=malformed_token,
        )
        malformed_harness = malformed_harness.reopen()
        malformed_recovery_load = malformed_harness.sink.latest_checked(malformed_run_id)
        malformed_recovery_bytes = _contract_blob_hex(
            malformed_recovery_load.value,
            _CONTRACT_CHECKPOINT_BLOB_SHA256,
        )

        harness = factory()
        run_id = _contract_run_id(namespace, "checkpoint")
        token = _contract_writer(harness, run_id)
        missing = harness.sink.latest_checked(run_id)
        checkpoint = RunCheckpoint(
            run_id=run_id,
            seq=1,
            final_text="winner",
            workspace_delta=[
                {
                    "path": "contract.txt",
                    "kind": "file",
                    "change_kind": "created",
                    "content_sha256": _CONTRACT_CHECKPOINT_BLOB_SHA256,
                }
            ],
        )
        checkpoint_blobs = {
            _CONTRACT_CHECKPOINT_BLOB_SHA256: _CONTRACT_CHECKPOINT_BLOB,
        }
        first = harness.sink.commit_checkpoint(
            checkpoint,
            checkpoint_blobs,
            writer_token=token,
        )
        harness = harness.reopen()
        repeated = harness.sink.commit_checkpoint(
            checkpoint,
            checkpoint_blobs,
            writer_token=token,
        )
        blob_key_conflict = harness.sink.commit_checkpoint(
            checkpoint,
            {
                **checkpoint_blobs,
                _CONTRACT_ALTERNATE_BLOB_SHA256: _CONTRACT_ALTERNATE_BLOB,
            },
            writer_token=token,
        )
        harness = harness.reopen()
        blob_key_conflict_reference = harness.sink.commit_checkpoint(
            RunCheckpoint(
                run_id=run_id,
                seq=2,
                final_text="conflicting-blob-reference",
                workspace_delta=[
                    {
                        "path": "conflicting-blob-reference.txt",
                        "kind": "file",
                        "change_kind": "created",
                        "content_sha256": _CONTRACT_ALTERNATE_BLOB_SHA256,
                    }
                ],
            ),
            {},
            writer_token=token,
        )
        blob_key_conflict_reference_load = harness.sink.latest_checked(run_id)
        blob_bytes_conflict = harness.sink.commit_checkpoint(
            checkpoint,
            {_CONTRACT_CHECKPOINT_BLOB_SHA256: _CONTRACT_ALTERNATE_BLOB},
            writer_token=token,
        )
        conflicting_checkpoint = RunCheckpoint(
            run_id=run_id,
            seq=1,
            final_text="challenger",
        )
        conflict = harness.sink.commit_checkpoint(
            conflicting_checkpoint,
            {},
            writer_token=token,
        )
        checkpoint_identity_statuses = {
            field_name: harness.sink.commit_checkpoint(
                variant,
                checkpoint_blobs,
                writer_token=token,
            ).status
            for field_name, variant in _contract_checkpoint_identity_variants(checkpoint).items()
        }
        loaded = harness.sink.latest_checked(run_id)
        referenced_blob_bytes = _contract_blob_hex(
            loaded.value,
            _CONTRACT_CHECKPOINT_BLOB_SHA256,
        )
        checkpoint_digest = _contract_record_digest(
            checkpoint_payload_for_write(checkpoint),
            checkpoint_blobs,
        )
        checkpoint_commit_evidence = {
            "committed": _contract_commit_evidence(
                first,
                sequence=checkpoint.seq,
                content_digest=checkpoint_digest,
            ),
            "already_committed": _contract_commit_evidence(
                repeated,
                sequence=checkpoint.seq,
                content_digest=checkpoint_digest,
            ),
            "conflict": _contract_commit_evidence(
                conflict,
                sequence=conflicting_checkpoint.seq,
                content_digest=_contract_record_digest(
                    checkpoint_payload_for_write(conflicting_checkpoint)
                ),
                winner_digest=checkpoint_digest,
            ),
        }

        monotonic_harness = factory()
        monotonic_run_id = _contract_run_id(namespace, "checkpoint-monotonic")
        monotonic_token = _contract_writer(monotonic_harness, monotonic_run_id)
        newer = monotonic_harness.sink.commit_checkpoint(
            RunCheckpoint(run_id=monotonic_run_id, seq=2, final_text="newer"),
            {},
            writer_token=monotonic_token,
        )
        delayed_checkpoint = RunCheckpoint(
            run_id=monotonic_run_id,
            seq=1,
            final_text="delayed",
            workspace_delta=[
                {
                    "path": "delayed-checkpoint.txt",
                    "kind": "file",
                    "change_kind": "created",
                    "content_sha256": _CONTRACT_ALTERNATE_BLOB_SHA256,
                }
            ],
        )
        delayed_blobs = {
            _CONTRACT_ALTERNATE_BLOB_SHA256: _CONTRACT_ALTERNATE_BLOB,
        }
        delayed = monotonic_harness.sink.commit_checkpoint(
            delayed_checkpoint,
            delayed_blobs,
            writer_token=monotonic_token,
        )
        monotonic_harness = monotonic_harness.reopen()
        delayed_retry = monotonic_harness.sink.commit_checkpoint(
            delayed_checkpoint,
            delayed_blobs,
            writer_token=monotonic_token,
        )
        head_after_delayed = monotonic_harness.sink.latest_checked(monotonic_run_id)
        delayed_blob_reference = monotonic_harness.sink.commit_checkpoint(
            RunCheckpoint(
                run_id=monotonic_run_id,
                seq=3,
                final_text="delayed-blob-reference",
                workspace_delta=[
                    {
                        "path": "reused-delayed-checkpoint.txt",
                        "kind": "file",
                        "change_kind": "created",
                        "content_sha256": _CONTRACT_ALTERNATE_BLOB_SHA256,
                    }
                ],
            ),
            {},
            writer_token=monotonic_token,
        )
        monotonic_harness = monotonic_harness.reopen()
        delayed_blob_reference_load = monotonic_harness.sink.latest_checked(monotonic_run_id)
        delayed_blob_reference_bytes = _contract_blob_hex(
            delayed_blob_reference_load.value,
            _CONTRACT_ALTERNATE_BLOB_SHA256,
        )
        checkpoint_load_fault_evidence = {
            status: _contract_authoritative_load_fault_evidence(
                factory,
                namespace,
                "checkpoint",
                status,
            )
            for status in ("corrupt", "unsupported_version")
        }
        outcomes.append(
            outcome_from_observations(
                "FENCED-01-CHECKPOINT-CONTENT-IDENTITY",
                FENCED_RUN_SINK_CONTRACT_PROFILE,
                (
                    observation("initial_load", expected="missing", actual=missing.status),
                    observation("first_status", expected="committed", actual=first.status),
                    observation(
                        "repeat_status", expected="already_committed", actual=repeated.status
                    ),
                    observation(
                        "missing_reference_status",
                        expected="conflict",
                        actual=missing_reference.status,
                    ),
                    observation(
                        "missing_reference_not_published",
                        expected="missing",
                        actual=missing_reference_load.status,
                    ),
                    observation(
                        "missing_reference_recovery",
                        expected="committed",
                        actual=missing_reference_recovery.status,
                    ),
                    observation(
                        "missing_reference_recovery_bytes",
                        expected=_CONTRACT_UNRESOLVED_BLOB.hex(),
                        actual=missing_reference_recovery_bytes,
                    ),
                    observation(
                        "missing_media_reference_status",
                        expected="conflict",
                        actual=missing_media.status,
                    ),
                    observation(
                        "missing_media_reference_not_published",
                        expected="missing",
                        actual=missing_media_load.status,
                    ),
                    observation(
                        "missing_media_reference_recovery",
                        expected="committed",
                        actual=missing_media_recovery.status,
                    ),
                    observation(
                        "missing_media_reference_recovery_bytes",
                        expected=_CONTRACT_UNRESOLVED_BLOB.hex(),
                        actual=missing_media_recovery_bytes,
                    ),
                    *(
                        observation(
                            f"missing_queued_media_{carrier}_reference_status",
                            expected="conflict",
                            actual=evidence[0],
                        )
                        for carrier, evidence in missing_queued_media_evidence.items()
                    ),
                    *(
                        observation(
                            f"missing_queued_media_{carrier}_not_published",
                            expected="missing",
                            actual=evidence[1],
                        )
                        for carrier, evidence in missing_queued_media_evidence.items()
                    ),
                    *(
                        observation(
                            f"missing_queued_media_{carrier}_recovery",
                            expected="committed",
                            actual=evidence[2],
                        )
                        for carrier, evidence in missing_queued_media_evidence.items()
                    ),
                    *(
                        observation(
                            f"missing_queued_media_{carrier}_recovery_bytes",
                            expected=_CONTRACT_UNRESOLVED_BLOB.hex(),
                            actual=evidence[3],
                        )
                        for carrier, evidence in missing_queued_media_evidence.items()
                    ),
                    observation(
                        "malformed_workspace_reference_status",
                        expected="conflict",
                        actual=malformed_workspace_reference.status,
                    ),
                    observation(
                        "malformed_workspace_reference_not_published",
                        expected="missing",
                        actual=malformed_workspace_load.status,
                    ),
                    observation(
                        "malformed_media_reference_status",
                        expected="conflict",
                        actual=malformed_media_reference.status,
                    ),
                    observation(
                        "malformed_media_reference_not_published",
                        expected="missing",
                        actual=malformed_media_load.status,
                    ),
                    *(
                        observation(
                            f"malformed_queued_media_{carrier}_reference_status",
                            expected="conflict",
                            actual=evidence[0],
                        )
                        for carrier, evidence in malformed_queued_media_evidence.items()
                    ),
                    *(
                        observation(
                            f"malformed_queued_media_{carrier}_not_published",
                            expected="missing",
                            actual=evidence[1],
                        )
                        for carrier, evidence in malformed_queued_media_evidence.items()
                    ),
                    observation(
                        "authoritative_backing_reference_statuses",
                        expected=("committed", "committed"),
                        actual=(backing_seed.status, backing_reference.status),
                    ),
                    observation(
                        "authoritative_backing_reference_head",
                        expected=2,
                        actual=backing_load.sequence,
                    ),
                    observation(
                        "authoritative_backing_reference_bytes",
                        expected=_CONTRACT_CHECKPOINT_BLOB.hex(),
                        actual=backing_reference_bytes,
                    ),
                    observation(
                        "uppercase_blob_key_status",
                        expected="conflict",
                        actual=uppercase_key.status,
                    ),
                    observation(
                        "uppercase_blob_key_not_published",
                        expected="missing",
                        actual=uppercase_key_load.status,
                    ),
                    observation(
                        "uppercase_blob_key_bytes_not_published",
                        expected="conflict",
                        actual=uppercase_key_blob_leak.status,
                    ),
                    observation(
                        "uppercase_blob_key_recovery",
                        expected="committed",
                        actual=uppercase_key_recovery.status,
                    ),
                    observation(
                        "uppercase_blob_key_recovery_bytes",
                        expected=_CONTRACT_CHECKPOINT_BLOB.hex(),
                        actual=uppercase_key_recovery_bytes,
                    ),
                    observation(
                        "blob_key_conflict",
                        expected="conflict",
                        actual=blob_key_conflict.status,
                    ),
                    observation(
                        "blob_key_conflict_not_published",
                        expected=("conflict", 1),
                        actual=(
                            blob_key_conflict_reference.status,
                            blob_key_conflict_reference_load.sequence,
                        ),
                    ),
                    observation(
                        "blob_bytes_conflict",
                        expected="conflict",
                        actual=blob_bytes_conflict.status,
                    ),
                    observation(
                        "malformed_fresh_blob_status",
                        expected="conflict",
                        actual=malformed_fresh_blob.status,
                    ),
                    observation(
                        "malformed_fresh_blob_seed",
                        expected="committed",
                        actual=malformed_seed.status,
                    ),
                    observation(
                        "malformed_fresh_blob_preserves_existing_bytes",
                        expected=_CONTRACT_CHECKPOINT_BLOB.hex(),
                        actual=malformed_preserved_bytes,
                    ),
                    observation(
                        "malformed_fresh_blob_head_not_published",
                        expected=malformed_seed_checkpoint.seq,
                        actual=malformed_after_rejection.sequence,
                    ),
                    observation(
                        "malformed_fresh_blob_recovery",
                        expected="committed",
                        actual=malformed_recovery.status,
                    ),
                    observation(
                        "malformed_fresh_blob_recovery_bytes",
                        expected=_CONTRACT_CHECKPOINT_BLOB.hex(),
                        actual=malformed_recovery_bytes,
                    ),
                    observation("conflict_status", expected="conflict", actual=conflict.status),
                    *(
                        observation(
                            f"checkpoint_{status}_evidence",
                            expected=(True, True, True),
                            actual=evidence,
                        )
                        for status, evidence in checkpoint_commit_evidence.items()
                    ),
                    *(
                        observation(
                            f"checkpoint_canonical_alias_{field_name}_{direction}",
                            expected="already_committed",
                            actual=status,
                        )
                        for (field_name, direction), status in (
                            checkpoint_canonical_alias_statuses.items()
                        )
                    ),
                    *(
                        observation(
                            f"checkpoint_identity_{field_name}",
                            expected="conflict",
                            actual=status,
                        )
                        for field_name, status in checkpoint_identity_statuses.items()
                    ),
                    observation(
                        "latest_load",
                        expected="loaded",
                        actual=loaded.status,
                    ),
                    observation(
                        "latest_sequence",
                        expected=1,
                        actual=loaded.sequence,
                    ),
                    observation(
                        "latest_record_sequence",
                        expected=1,
                        actual=(loaded.value.seq if loaded.value else None),
                    ),
                    observation(
                        "latest_run_binding",
                        expected=run_id,
                        actual=(loaded.value.checkpoint.run_id if loaded.value else None),
                    ),
                    observation(
                        "winner_remains_latest",
                        expected="winner",
                        actual=(loaded.value.checkpoint.final_text if loaded.value else None),
                    ),
                    observation(
                        "checkpoint_manifest_digest",
                        expected=canonical_sha256(checkpoint.to_json()),
                        actual=(
                            canonical_sha256(loaded.value.checkpoint.to_json())
                            if loaded.value
                            else None
                        ),
                    ),
                    observation(
                        "referenced_blob_round_trip",
                        expected=_CONTRACT_CHECKPOINT_BLOB.hex(),
                        actual=referenced_blob_bytes,
                    ),
                    observation(
                        "newer_checkpoint",
                        expected="committed",
                        actual=newer.status,
                    ),
                    observation(
                        "delayed_checkpoint",
                        expected="committed",
                        actual=delayed.status,
                    ),
                    observation(
                        "delayed_checkpoint_retry",
                        expected="already_committed",
                        actual=delayed_retry.status,
                    ),
                    observation(
                        "head_after_delayed_sequence",
                        expected=2,
                        actual=head_after_delayed.sequence,
                    ),
                    observation(
                        "head_after_delayed_text",
                        expected="newer",
                        actual=(
                            head_after_delayed.value.checkpoint.final_text
                            if head_after_delayed.value
                            else None
                        ),
                    ),
                    observation(
                        "delayed_blob_reference_status",
                        expected="committed",
                        actual=delayed_blob_reference.status,
                    ),
                    observation(
                        "delayed_blob_reference_head",
                        expected=3,
                        actual=delayed_blob_reference_load.sequence,
                    ),
                    observation(
                        "delayed_blob_reference_bytes",
                        expected=_CONTRACT_ALTERNATE_BLOB.hex(),
                        actual=delayed_blob_reference_bytes,
                    ),
                    *(
                        observation(
                            f"authoritative_load_{status}",
                            expected=(status, True),
                            actual=actual,
                        )
                        for status, actual in checkpoint_load_fault_evidence.items()
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(
            _error(
                "FENCED-01-CHECKPOINT-CONTENT-IDENTITY",
                FENCED_RUN_SINK_CONTRACT_PROFILE,
                exc,
            )
        )

    try:
        harness = factory()
        run_id = _contract_run_id(namespace, "fence-first")
        stale = _contract_writer(harness, run_id)
        checkpoint = RunCheckpoint(run_id=run_id, seq=1, final_text="winner")
        event = _contract_event(run_id, seq=1)
        invocation = _contract_invocation(run_id, revision=1, dispatch_state="reserved")
        terminal = _contract_terminal(run_id)
        initial_checkpoint = harness.sink.commit_checkpoint(checkpoint, {}, writer_token=stale)
        initial_event = harness.sink.append_event(event, writer_token=stale)
        initial_invocation = harness.sink.commit_invocation(
            invocation,
            {},
            writer_token=stale,
        )
        initial_terminal = harness.sink.settle_terminal(terminal, writer_token=stale)
        current = _contract_writer(harness, run_id, "owner-b", generation=2)
        existing_authority_probe_statuses = {
            "stale_generation_current_owner": _contract_authority_probe_statuses(
                harness.sink,
                checkpoint=checkpoint,
                event=event,
                invocation=invocation,
                terminal=terminal,
                writer_token=WriterToken(
                    run_id=run_id,
                    owner_id=current.owner_id,
                    generation=stale.generation,
                ),
            ),
            "wrong_owner_current_generation": _contract_authority_probe_statuses(
                harness.sink,
                checkpoint=checkpoint,
                event=event,
                invocation=invocation,
                terminal=terminal,
                writer_token=WriterToken(
                    run_id=run_id,
                    owner_id=stale.owner_id,
                    generation=current.generation,
                ),
            ),
        }
        stale_checkpoint = harness.sink.commit_checkpoint(checkpoint, {}, writer_token=stale)
        stale_event = harness.sink.append_event(event, writer_token=stale)
        stale_invocation = harness.sink.commit_invocation(
            invocation,
            {},
            writer_token=stale,
        )
        stale_conflicting_statuses = {
            "checkpoint": harness.sink.commit_checkpoint(
                replace(checkpoint, final_text="stale-challenger"),
                {},
                writer_token=stale,
            ).status,
            "event": harness.sink.append_event(
                replace(event, level="warning"),
                writer_token=stale,
            ).status,
            "invocation": harness.sink.commit_invocation(
                replace(invocation, dispatch_id="dispatch-stale"),
                {},
                writer_token=stale,
            ).status,
            "terminal": harness.sink.settle_terminal(
                _contract_terminal(run_id, failed=True),
                writer_token=stale,
            ).status,
        }
        stale_malformed_checkpoint = harness.sink.commit_checkpoint(
            checkpoint,
            {_CONTRACT_CHECKPOINT_BLOB_SHA256: _CONTRACT_ALTERNATE_BLOB},
            writer_token=stale,
        )
        stale_malformed_invocation = harness.sink.commit_invocation(
            invocation,
            {_CONTRACT_INVOCATION_BLOB_SHA256: _CONTRACT_ALTERNATE_BLOB},
            writer_token=stale,
        )
        partial_authority_malformed_statuses = {}
        for authority_case, invalid_token in {
            "stale_generation_current_owner": WriterToken(
                run_id=run_id,
                owner_id=current.owner_id,
                generation=stale.generation,
            ),
            "wrong_owner_current_generation": WriterToken(
                run_id=run_id,
                owner_id=stale.owner_id,
                generation=current.generation,
            ),
        }.items():
            partial_authority_malformed_statuses[authority_case] = {
                "checkpoint": harness.sink.commit_checkpoint(
                    checkpoint,
                    {_CONTRACT_CHECKPOINT_BLOB_SHA256: _CONTRACT_ALTERNATE_BLOB},
                    writer_token=invalid_token,
                ).status,
                "invocation": harness.sink.commit_invocation(
                    invocation,
                    {_CONTRACT_INVOCATION_BLOB_SHA256: _CONTRACT_ALTERNATE_BLOB},
                    writer_token=invalid_token,
                ).status,
            }
        stale_terminal = harness.sink.settle_terminal(terminal, writer_token=stale)
        current_checkpoint = harness.sink.commit_checkpoint(
            RunCheckpoint(run_id=run_id, seq=2, final_text="current"),
            {},
            writer_token=current,
        )
        current_event = harness.sink.append_event(
            _contract_event(run_id, seq=2), writer_token=current
        )
        current_invocation = harness.sink.commit_invocation(
            _contract_invocation(run_id, revision=2, dispatch_state="dispatch_started"),
            {},
            writer_token=current,
        )
        current_terminal = harness.sink.settle_terminal(terminal, writer_token=current)
        harness = harness.reopen()
        reopened_initial_event = harness.read_event(run_id, 1)
        reopened_current_event = harness.read_event(run_id, 2)
        same_run_event_payloads = (
            (
                canonical_sha256(reopened_initial_event.to_json())
                if reopened_initial_event
                else None
            ),
            (
                canonical_sha256(reopened_current_event.to_json())
                if reopened_current_event
                else None
            ),
        )

        fresh_authority_probe_statuses = {}
        fresh_authority_malformed_statuses = {}
        fresh_authority_blob_visibility = {}
        fresh_authority_blob_visibility_setup = {}
        fresh_authority_recovery_statuses = {}
        for authority_case in (
            "stale_owner_and_generation",
            "stale_generation_current_owner",
            "wrong_owner_current_generation",
        ):
            fresh_harness = factory()
            fresh_run_id = _contract_run_id(namespace, f"fresh-{authority_case}")
            fresh_stale = _contract_writer(fresh_harness, fresh_run_id)
            fresh_current = _contract_writer(
                fresh_harness,
                fresh_run_id,
                "owner-b",
                generation=2,
            )
            owner_id, generation = {
                "stale_owner_and_generation": (
                    fresh_stale.owner_id,
                    fresh_stale.generation,
                ),
                "stale_generation_current_owner": (
                    fresh_current.owner_id,
                    fresh_stale.generation,
                ),
                "wrong_owner_current_generation": (
                    fresh_stale.owner_id,
                    fresh_current.generation,
                ),
            }[authority_case]
            invalid_token = WriterToken(
                run_id=fresh_run_id,
                owner_id=owner_id,
                generation=generation,
            )
            fresh_records = {
                "checkpoint": RunCheckpoint(run_id=fresh_run_id, seq=1),
                "event": _contract_event(fresh_run_id, seq=1),
                "invocation": _contract_invocation(
                    fresh_run_id,
                    revision=1,
                    dispatch_state="reserved",
                ),
                "terminal": _contract_terminal(fresh_run_id),
            }
            stale_only_blobs = {
                mutation: f"{authority_case}:{mutation}:stale-only\n".encode()
                for mutation in ("checkpoint", "invocation")
            }
            stale_only_digests = {
                mutation: hashlib.sha256(blob).hexdigest()
                for mutation, blob in stale_only_blobs.items()
            }
            malformed_blob_maps = {
                "checkpoint": {
                    _CONTRACT_CHECKPOINT_BLOB_SHA256: _CONTRACT_ALTERNATE_BLOB,
                    stale_only_digests["checkpoint"]: stale_only_blobs["checkpoint"],
                },
                "invocation": {
                    _CONTRACT_INVOCATION_BLOB_SHA256: _CONTRACT_ALTERNATE_BLOB,
                    stale_only_digests["invocation"]: stale_only_blobs["invocation"],
                },
            }
            fresh_authority_malformed_statuses[authority_case] = {
                "checkpoint": fresh_harness.sink.commit_checkpoint(
                    fresh_records["checkpoint"],
                    malformed_blob_maps["checkpoint"],
                    writer_token=invalid_token,
                ).status,
                "invocation": fresh_harness.sink.commit_invocation(
                    fresh_records["invocation"],
                    malformed_blob_maps["invocation"],
                    writer_token=invalid_token,
                ).status,
            }
            fresh_authority_probe_statuses[authority_case] = _contract_authority_probe_statuses(
                fresh_harness.sink,
                **fresh_records,
                writer_token=invalid_token,
            )
            fresh_harness = fresh_harness.reopen()
            checkpoint_blob_visibility = fresh_harness.sink.commit_checkpoint(
                RunCheckpoint(
                    run_id=fresh_run_id,
                    seq=2,
                    workspace_delta=[
                        {
                            "path": "stale-only-checkpoint-blob.txt",
                            "kind": "file",
                            "change_kind": "created",
                            "content_sha256": stale_only_digests["checkpoint"],
                        }
                    ],
                ),
                {},
                writer_token=fresh_current,
            )
            checkpoint_blob_visibility_load = fresh_harness.sink.latest_checked(fresh_run_id)
            visibility_call_id = "call-stale-only-blob"
            invocation_blob_visibility_setup = tuple(
                fresh_harness.sink.commit_invocation(
                    _contract_invocation(
                        fresh_run_id,
                        logical_call_id=visibility_call_id,
                        revision=revision,
                        dispatch_state=dispatch_state,
                    ),
                    {},
                    writer_token=fresh_current,
                ).status
                for revision, dispatch_state in ((1, "reserved"), (2, "dispatch_started"))
            )
            invocation_blob_visibility = fresh_harness.sink.commit_invocation(
                replace(
                    _contract_invocation(
                        fresh_run_id,
                        logical_call_id=visibility_call_id,
                        revision=3,
                        dispatch_state="settled",
                        succeeded=True,
                    ),
                    result_ref=f"blob:{stale_only_digests['invocation']}",
                ),
                {},
                writer_token=fresh_current,
            )
            invocation_blob_visibility_load = fresh_harness.sink.load_invocation(
                fresh_run_id,
                visibility_call_id,
            )
            fresh_authority_blob_visibility[authority_case] = {
                "checkpoint": (
                    checkpoint_blob_visibility.status,
                    checkpoint_blob_visibility_load.status,
                ),
                "invocation": (
                    invocation_blob_visibility.status,
                    invocation_blob_visibility_load.sequence,
                ),
            }
            fresh_authority_blob_visibility_setup[authority_case] = invocation_blob_visibility_setup
            fresh_authority_recovery_statuses[authority_case] = _contract_authority_probe_statuses(
                fresh_harness.sink,
                **fresh_records,
                writer_token=fresh_current,
            )
        handoff_observations = []
        for handoff_kind, current_owner in (
            ("lease_renewal", "owner-a"),
            ("owner_reassignment", "owner-b"),
        ):
            for mutation in ("checkpoint", "event", "invocation", "terminal"):
                handoff_harness = factory()
                handoff_run_id = _contract_run_id(
                    namespace,
                    f"handoff-{handoff_kind}-{mutation}",
                )
                handoff_stale = _contract_writer(handoff_harness, handoff_run_id)
                handoff_current = WriterToken(
                    run_id=handoff_run_id,
                    owner_id=current_owner,
                    generation=2,
                )
                if mutation == "invocation":
                    _contract_prepare_invocation_race(
                        handoff_harness,
                        handoff_run_id,
                        handoff_stale,
                    )
                current_value, competing_value = _contract_competing_values(
                    mutation,
                    handoff_run_id,
                )
                current_blobs = _contract_mutation_blobs(mutation)
                if mutation == "checkpoint":
                    stale_value = replace(
                        current_value,
                        workspace_delta=[
                            {
                                "path": "stale-handoff.txt",
                                "kind": "file",
                                "change_kind": "created",
                                "content_sha256": _CONTRACT_STALE_HANDOFF_BLOB_SHA256,
                            }
                        ],
                    )
                    stale_blobs = {
                        _CONTRACT_STALE_HANDOFF_BLOB_SHA256: _CONTRACT_STALE_HANDOFF_BLOB,
                    }
                elif mutation == "invocation":
                    stale_value = replace(
                        current_value,
                        result_ref=f"blob:{_CONTRACT_STALE_HANDOFF_BLOB_SHA256}",
                    )
                    stale_blobs = {
                        _CONTRACT_STALE_HANDOFF_BLOB_SHA256: _CONTRACT_STALE_HANDOFF_BLOB,
                    }
                else:
                    stale_value = competing_value
                    stale_blobs = current_blobs
                handoff_write = partial(
                    _contract_handoff_write,
                    mutation=mutation,
                    stale_token=handoff_stale,
                    stale_value=stale_value,
                    stale_blobs=stale_blobs,
                    current_value=current_value,
                    current_blobs=current_blobs,
                )
                stale_result, current_result, rotation_first = handoff_harness.race_writer_handoff(
                    mutation,
                    handoff_stale,
                    handoff_current,
                    handoff_write,
                )
                expected_statuses = (
                    ("fenced", "committed")
                    if rotation_first
                    else (
                        "committed",
                        "conflict",
                    )
                )
                handoff_observations.append(
                    observation(
                        f"handoff_{handoff_kind}_{mutation}_linearization",
                        expected=expected_statuses,
                        actual=(stale_result.status, current_result.status),
                    )
                )
                handoff_read_harness = handoff_harness.reopen()
                winner_value = current_value if rotation_first else stale_value
                winner_payload_digest = _contract_race_payload_digest(
                    handoff_read_harness,
                    mutation,
                    handoff_run_id,
                )
                handoff_observations.append(
                    observation(
                        f"handoff_{handoff_kind}_{mutation}_winner_payload",
                        expected=_contract_mutation_payload_digest(
                            mutation,
                            winner_value,
                        ),
                        actual=winner_payload_digest,
                    )
                )
                if mutation in {"checkpoint", "invocation"}:
                    winner_blob = (
                        _CONTRACT_CHECKPOINT_BLOB
                        if rotation_first and mutation == "checkpoint"
                        else _CONTRACT_INVOCATION_BLOB
                        if rotation_first
                        else _CONTRACT_STALE_HANDOFF_BLOB
                    )
                    winner_blob_sha256 = (
                        _CONTRACT_CHECKPOINT_BLOB_SHA256
                        if rotation_first and mutation == "checkpoint"
                        else _CONTRACT_INVOCATION_BLOB_SHA256
                        if rotation_first
                        else _CONTRACT_STALE_HANDOFF_BLOB_SHA256
                    )
                    handoff_observations.append(
                        observation(
                            f"handoff_{handoff_kind}_{mutation}_blob_bytes",
                            expected=winner_blob.hex(),
                            actual=_contract_race_blob_hex(
                                handoff_read_harness,
                                mutation,
                                handoff_run_id,
                                winner_blob_sha256,
                            ),
                        )
                    )
                    stale_blob_probe = _contract_stale_handoff_blob_probe(
                        handoff_harness.reopen(),
                        mutation,
                        handoff_run_id,
                        handoff_current,
                    )
                    handoff_observations.append(
                        observation(
                            f"handoff_{handoff_kind}_{mutation}_stale_blob_visibility",
                            expected="conflict" if rotation_first else "committed",
                            actual=stale_blob_probe,
                        )
                    )
        outcomes.append(
            outcome_from_observations(
                "FENCED-02-FENCE-PRECEDES-IDEMPOTENCY",
                FENCED_RUN_SINK_CONTRACT_PROFILE,
                (
                    observation(
                        "initial_checkpoint", expected="committed", actual=initial_checkpoint.status
                    ),
                    observation("initial_event", expected="committed", actual=initial_event.status),
                    observation(
                        "initial_invocation",
                        expected="committed",
                        actual=initial_invocation.status,
                    ),
                    observation(
                        "initial_terminal", expected="committed", actual=initial_terminal.status
                    ),
                    observation(
                        "stale_checkpoint", expected="fenced", actual=stale_checkpoint.status
                    ),
                    observation("stale_event", expected="fenced", actual=stale_event.status),
                    observation(
                        "stale_invocation", expected="fenced", actual=stale_invocation.status
                    ),
                    *(
                        observation(
                            f"stale_conflicting_{mutation}",
                            expected="fenced",
                            actual=status,
                        )
                        for mutation, status in stale_conflicting_statuses.items()
                    ),
                    observation(
                        "stale_malformed_checkpoint",
                        expected="fenced",
                        actual=stale_malformed_checkpoint.status,
                    ),
                    observation(
                        "stale_malformed_invocation",
                        expected="fenced",
                        actual=stale_malformed_invocation.status,
                    ),
                    *(
                        observation(
                            f"malformed_{authority_case}_{mutation}",
                            expected="fenced",
                            actual=status,
                        )
                        for authority_case, mutation_statuses in (
                            partial_authority_malformed_statuses.items()
                        )
                        for mutation, status in mutation_statuses.items()
                    ),
                    observation("stale_terminal", expected="fenced", actual=stale_terminal.status),
                    *(
                        observation(
                            f"existing_{authority_case}_{mutation}",
                            expected="fenced",
                            actual=status,
                        )
                        for authority_case, mutation_statuses in (
                            existing_authority_probe_statuses.items()
                        )
                        for mutation, status in mutation_statuses.items()
                    ),
                    observation(
                        "new_generation_writes",
                        expected="committed",
                        actual=current_checkpoint.status,
                    ),
                    observation(
                        "new_generation_appends_event",
                        expected="committed",
                        actual=current_event.status,
                    ),
                    observation(
                        "new_generation_commits_invocation",
                        expected="committed",
                        actual=current_invocation.status,
                    ),
                    observation(
                        "new_generation_reads_terminal_winner",
                        expected="already_committed",
                        actual=current_terminal.status,
                    ),
                    observation(
                        "same_run_event_sequence_payloads",
                        expected=(
                            canonical_sha256(event.to_json()),
                            canonical_sha256(_contract_event(run_id, seq=2).to_json()),
                        ),
                        actual=same_run_event_payloads,
                    ),
                    *(
                        observation(
                            f"fresh_{authority_case}_{mutation}",
                            expected="fenced",
                            actual=status,
                        )
                        for authority_case, mutation_statuses in (
                            fresh_authority_probe_statuses.items()
                        )
                        for mutation, status in mutation_statuses.items()
                    ),
                    *(
                        observation(
                            f"fresh_malformed_{authority_case}_{mutation}",
                            expected="fenced",
                            actual=status,
                        )
                        for authority_case, mutation_statuses in (
                            fresh_authority_malformed_statuses.items()
                        )
                        for mutation, status in mutation_statuses.items()
                    ),
                    *(
                        observation(
                            f"fresh_malformed_{authority_case}_{mutation}_blob_visibility",
                            expected={
                                "checkpoint": ("conflict", "missing"),
                                "invocation": ("conflict", 2),
                            }[mutation],
                            actual=actual,
                        )
                        for authority_case, mutation_visibility in (
                            fresh_authority_blob_visibility.items()
                        )
                        for mutation, actual in mutation_visibility.items()
                    ),
                    *(
                        observation(
                            f"fresh_malformed_{authority_case}_invocation_blob_visibility_setup",
                            expected=("committed", "committed"),
                            actual=actual,
                        )
                        for authority_case, actual in (
                            fresh_authority_blob_visibility_setup.items()
                        )
                    ),
                    *(
                        observation(
                            f"fresh_{authority_case}_{mutation}_recovery",
                            expected="committed",
                            actual=status,
                        )
                        for authority_case, mutation_statuses in (
                            fresh_authority_recovery_statuses.items()
                        )
                        for mutation, status in mutation_statuses.items()
                    ),
                    *handoff_observations,
                ),
            )
        )
    except Exception as exc:
        outcomes.append(
            _error(
                "FENCED-02-FENCE-PRECEDES-IDEMPOTENCY",
                FENCED_RUN_SINK_CONTRACT_PROFILE,
                exc,
            )
        )

    try:
        terminal_canonical_alias_statuses = {
            direction: _contract_terminal_canonical_alias_status(
                factory,
                _contract_run_id(namespace, f"terminal-canonical-alias-{direction}"),
                legacy_first=direction == "legacy_to_current",
            )
            for direction in ("current_to_legacy", "legacy_to_current")
        }
        harness = factory()
        run_id = _contract_run_id(namespace, "terminal")
        token = _contract_writer(harness, run_id)
        event = _contract_event(run_id, seq=1)
        first_event = harness.sink.append_event(event, writer_token=token)
        terminal = _contract_terminal(run_id)
        first_terminal = harness.sink.settle_terminal(terminal, writer_token=token)
        harness = harness.reopen()
        reopened_event = harness.read_event(run_id, event.seq)
        reopened_terminal = harness.read_terminal(run_id)
        repeated_event = harness.sink.append_event(event, writer_token=token)
        conflicting_event = _contract_event(run_id, seq=1, level="warning")
        conflict_event = harness.sink.append_event(
            conflicting_event,
            writer_token=token,
        )
        event_identity_statuses = {
            field_name: harness.sink.append_event(
                variant,
                writer_token=token,
            ).status
            for field_name, variant in _contract_event_identity_variants(event).items()
        }
        repeated_terminal = harness.sink.settle_terminal(terminal, writer_token=token)
        conflicting_terminal = _contract_terminal(run_id, failed=True)
        conflict_terminal = harness.sink.settle_terminal(
            conflicting_terminal,
            writer_token=token,
        )
        terminal_identity_statuses = {
            field_name: harness.sink.settle_terminal(
                variant,
                writer_token=token,
            ).status
            for field_name, variant in _contract_terminal_identity_variants(terminal).items()
        }
        event_digest = _contract_record_digest(event.to_json())
        event_commit_evidence = {
            "committed": _contract_commit_evidence(
                first_event,
                sequence=event.seq,
                content_digest=event_digest,
            ),
            "already_committed": _contract_commit_evidence(
                repeated_event,
                sequence=event.seq,
                content_digest=event_digest,
            ),
            "conflict": _contract_commit_evidence(
                conflict_event,
                sequence=conflicting_event.seq,
                content_digest=_contract_record_digest(conflicting_event.to_json()),
                winner_digest=event_digest,
            ),
        }
        terminal_digest = _contract_record_digest(terminal.to_json())
        terminal_commit_evidence = {
            "committed": _contract_commit_evidence(
                first_terminal,
                sequence=None,
                content_digest=terminal_digest,
            ),
            "already_committed": _contract_commit_evidence(
                repeated_terminal,
                sequence=None,
                content_digest=terminal_digest,
            ),
            "conflict": _contract_commit_evidence(
                conflict_terminal,
                sequence=None,
                content_digest=_contract_record_digest(conflicting_terminal.to_json()),
                winner_digest=terminal_digest,
            ),
        }

        race_observations = []
        for mutation in ("checkpoint", "event", "invocation", "terminal"):
            race_harness = factory()
            race_run_id = _contract_run_id(namespace, f"race-{mutation}")
            race_token = _contract_writer(race_harness, race_run_id)
            if mutation == "invocation":
                _contract_prepare_invocation_race(
                    race_harness,
                    race_run_id,
                    race_token,
                )
            left_value, right_value = _contract_competing_values(mutation, race_run_id)
            race_blobs = _contract_mutation_blobs(mutation)
            left_write = partial(
                _contract_race_write,
                mutation=mutation,
                value=left_value,
                blobs=race_blobs,
                writer_token=race_token,
            )
            right_write = partial(
                _contract_race_write,
                mutation=mutation,
                value=right_value,
                blobs=race_blobs,
                writer_token=race_token,
            )
            left_result, right_result = race_harness.race_conflicting_writes(
                mutation,
                race_token,
                left_write,
                right_write,
            )
            winning_value = (
                left_value
                if left_result.status == "committed"
                else right_value
                if right_result.status == "committed"
                else None
            )
            reopened_race_harness = race_harness.reopen()
            retry_winner, retry_loser = _contract_race_retry_statuses(
                left_result,
                right_result,
                reopened_race_harness,
                left_write,
                right_write,
            )
            race_observations.extend(
                (
                    observation(
                        f"{mutation}_race_statuses",
                        expected=("committed", "conflict"),
                        actual=tuple(sorted((left_result.status, right_result.status))),
                    ),
                    observation(
                        f"{mutation}_race_winner_after_reopen",
                        expected="already_committed",
                        actual=retry_winner,
                    ),
                    observation(
                        f"{mutation}_race_loser_after_reopen",
                        expected="conflict",
                        actual=retry_loser,
                    ),
                    observation(
                        f"{mutation}_race_winner_payload",
                        expected=(
                            _contract_mutation_payload_digest(mutation, winning_value)
                            if winning_value is not None
                            else None
                        ),
                        actual=_contract_race_payload_digest(
                            reopened_race_harness,
                            mutation,
                            race_run_id,
                        ),
                    ),
                )
            )
            if mutation in {"checkpoint", "invocation"}:
                race_observations.append(
                    observation(
                        f"{mutation}_race_blob_bytes",
                        expected=(
                            _CONTRACT_CHECKPOINT_BLOB.hex()
                            if mutation == "checkpoint"
                            else _CONTRACT_INVOCATION_BLOB.hex()
                        ),
                        actual=_contract_race_blob_hex(
                            reopened_race_harness,
                            mutation,
                            race_run_id,
                        ),
                    )
                )
        outcomes.append(
            outcome_from_observations(
                "FENCED-03-EVENT-AND-TERMINAL-WINNERS",
                FENCED_RUN_SINK_CONTRACT_PROFILE,
                (
                    observation("event_first", expected="committed", actual=first_event.status),
                    observation(
                        "event_repeat", expected="already_committed", actual=repeated_event.status
                    ),
                    observation(
                        "event_conflict", expected="conflict", actual=conflict_event.status
                    ),
                    *(
                        observation(
                            f"event_{status}_evidence",
                            expected=(True, True, True),
                            actual=evidence,
                        )
                        for status, evidence in event_commit_evidence.items()
                    ),
                    observation(
                        "event_reopened_payload_digest",
                        expected=canonical_sha256(event.to_json()),
                        actual=(
                            canonical_sha256(reopened_event.to_json())
                            if reopened_event is not None
                            else None
                        ),
                    ),
                    *(
                        observation(
                            f"event_identity_{field_name}",
                            expected="conflict",
                            actual=status,
                        )
                        for field_name, status in event_identity_statuses.items()
                    ),
                    observation(
                        "terminal_first", expected="committed", actual=first_terminal.status
                    ),
                    observation(
                        "terminal_repeat",
                        expected="already_committed",
                        actual=repeated_terminal.status,
                    ),
                    observation(
                        "terminal_reopened_payload_digest",
                        expected=canonical_sha256(terminal.to_json()),
                        actual=(
                            canonical_sha256(reopened_terminal.to_json())
                            if reopened_terminal is not None
                            else None
                        ),
                    ),
                    *(
                        observation(
                            f"terminal_canonical_alias_{direction}",
                            expected="already_committed",
                            actual=status,
                        )
                        for direction, status in terminal_canonical_alias_statuses.items()
                    ),
                    observation(
                        "terminal_conflict", expected="conflict", actual=conflict_terminal.status
                    ),
                    *(
                        observation(
                            f"terminal_{status}_evidence",
                            expected=(True, True, True),
                            actual=evidence,
                        )
                        for status, evidence in terminal_commit_evidence.items()
                    ),
                    *(
                        observation(
                            f"terminal_identity_{field_name}",
                            expected="conflict",
                            actual=status,
                        )
                        for field_name, status in terminal_identity_statuses.items()
                    ),
                    *race_observations,
                ),
            )
        )
    except Exception as exc:
        outcomes.append(
            _error(
                "FENCED-03-EVENT-AND-TERMINAL-WINNERS",
                FENCED_RUN_SINK_CONTRACT_PROFILE,
                exc,
            )
        )

    try:
        invocation_identity_statuses = {
            field_name: _contract_invocation_identity_status(
                factory,
                _contract_run_id(namespace, f"invocation-identity-{field_name}"),
                field_name,
            )
            for field_name in sorted(_CONTRACT_INVOCATION_IDENTITY_FIELDS)
        }
        receipt_retryability_identity_results = {
            (winner_state, candidate_state): (
                _contract_receipt_retryability_identity_evidence(
                    factory,
                    _contract_run_id(
                        namespace,
                        (f"invocation-receipt-retryability-{winner_state}-to-{candidate_state}"),
                    ),
                    winner_state=winner_state,
                    candidate_state=candidate_state,
                )
            )
            for winner_state in _CONTRACT_RECEIPT_RETRYABILITY_STATES
            for candidate_state in _CONTRACT_RECEIPT_RETRYABILITY_STATES
            if winner_state != candidate_state
        }
        receipt_field_identity_results = {
            field_name: _contract_receipt_field_identity_evidence(
                factory,
                _contract_run_id(
                    namespace,
                    f"invocation-receipt-field-{field_name.replace('_', '-')}",
                ),
                field_name,
            )
            for field_name in sorted(MODEL_INVOCATION_RECEIPT_FIELDS)
        }
        receipt_usage_identity_results = {
            field_name: _contract_receipt_field_identity_evidence(
                factory,
                _contract_run_id(
                    namespace,
                    f"invocation-receipt-usage-{field_name.replace('_', '-')}",
                ),
                "usage",
                usage_field_name=field_name,
            )
            for field_name in sorted(MODEL_INVOCATION_RECEIPT_USAGE_FIELDS)
        }
        invocation_canonical_alias_statuses = {
            (field_name, direction): _contract_invocation_canonical_alias_status(
                factory,
                _contract_run_id(
                    namespace,
                    f"invocation-canonical-alias-{field_name}-{direction}",
                ),
                field_name,
                legacy_first=direction == "legacy_to_current",
            )
            for field_name in sorted(_CONTRACT_INVOCATION_FIXED_CANONICAL_FIELDS)
            for direction in ("current_to_legacy", "legacy_to_current")
        }
        invocation_canonical_alias_transition_statuses = {
            field_name: _contract_invocation_canonical_alias_transition_status(
                factory,
                _contract_run_id(
                    namespace,
                    f"invocation-canonical-alias-transition-{field_name}",
                ),
                field_name,
                legacy_first=False,
            )
            for field_name in sorted(_CONTRACT_INVOCATION_FIXED_CANONICAL_FIELDS)
        }
        invocation_canonical_alias_recovery_statuses = {
            field_name: _contract_invocation_canonical_alias_transition_status(
                factory,
                _contract_run_id(
                    namespace,
                    f"invocation-canonical-alias-recovery-{field_name}",
                ),
                field_name,
                legacy_first=True,
            )
            for field_name in sorted(_CONTRACT_INVOCATION_FIXED_CANONICAL_FIELDS)
        }
        missing_reference_harness = factory()
        missing_reference_run_id = _contract_run_id(
            namespace,
            "invocation-missing-reference",
        )
        missing_reference_token = _contract_writer(
            missing_reference_harness,
            missing_reference_run_id,
        )
        missing_reference_setup_statuses = tuple(
            missing_reference_harness.sink.commit_invocation(
                _contract_invocation(
                    missing_reference_run_id,
                    revision=revision,
                    dispatch_state=dispatch_state,
                ),
                {},
                writer_token=missing_reference_token,
            ).status
            for revision, dispatch_state in ((1, "reserved"), (2, "dispatch_started"))
        )
        missing_reference_invocation = replace(
            _contract_invocation(
                missing_reference_run_id,
                revision=3,
                dispatch_state="settled",
                succeeded=True,
            ),
            result_ref=f"blob:{_CONTRACT_UNRESOLVED_BLOB_SHA256}",
        )
        missing_reference = missing_reference_harness.sink.commit_invocation(
            missing_reference_invocation,
            {},
            writer_token=missing_reference_token,
        )
        missing_reference_harness = missing_reference_harness.reopen()
        missing_reference_load = missing_reference_harness.sink.load_invocation(
            missing_reference_run_id,
            "call-1",
        )
        missing_reference_recovery = missing_reference_harness.sink.commit_invocation(
            missing_reference_invocation,
            {_CONTRACT_UNRESOLVED_BLOB_SHA256: _CONTRACT_UNRESOLVED_BLOB},
            writer_token=missing_reference_token,
        )
        missing_reference_harness = missing_reference_harness.reopen()
        missing_reference_recovery_load = missing_reference_harness.sink.load_invocation(
            missing_reference_run_id,
            "call-1",
        )
        missing_reference_recovery_bytes = _contract_blob_hex(
            missing_reference_recovery_load.value,
            _CONTRACT_UNRESOLVED_BLOB_SHA256,
        )
        malformed_reference_harness = factory()
        malformed_reference_run_id = _contract_run_id(
            namespace,
            "invocation-malformed-reference",
        )
        malformed_reference_token = _contract_writer(
            malformed_reference_harness,
            malformed_reference_run_id,
        )
        malformed_reference_setup = tuple(
            malformed_reference_harness.sink.commit_invocation(
                _contract_invocation(
                    malformed_reference_run_id,
                    revision=revision,
                    dispatch_state=dispatch_state,
                ),
                {},
                writer_token=malformed_reference_token,
            ).status
            for revision, dispatch_state in ((1, "reserved"), (2, "dispatch_started"))
        )
        malformed_reference_invocation = replace(
            _contract_invocation(
                malformed_reference_run_id,
                revision=3,
                dispatch_state="settled",
                succeeded=True,
            ),
            result_ref=f"blob:{_CONTRACT_MALFORMED_BLOB_SHA256}",
        )
        malformed_reference = malformed_reference_harness.sink.commit_invocation(
            malformed_reference_invocation,
            {},
            writer_token=malformed_reference_token,
        )
        malformed_reference_harness = malformed_reference_harness.reopen()
        malformed_reference_load = malformed_reference_harness.sink.load_invocation(
            malformed_reference_run_id,
            "call-1",
        )
        external_reference_harness = factory()
        external_reference_run_id = _contract_run_id(
            namespace,
            "invocation-external-reference",
        )
        external_reference_token = _contract_writer(
            external_reference_harness,
            external_reference_run_id,
        )
        external_reference_setup = tuple(
            external_reference_harness.sink.commit_invocation(
                _contract_invocation(
                    external_reference_run_id,
                    revision=revision,
                    dispatch_state=dispatch_state,
                ),
                {},
                writer_token=external_reference_token,
            ).status
            for revision, dispatch_state in ((1, "reserved"), (2, "dispatch_started"))
        )
        external_reference_invocation = replace(
            _contract_invocation(
                external_reference_run_id,
                revision=3,
                dispatch_state="settled",
                succeeded=True,
            ),
            result_ref="object:contract-result",
        )
        external_reference = external_reference_harness.sink.commit_invocation(
            external_reference_invocation,
            {},
            writer_token=external_reference_token,
        )
        external_reference_harness = external_reference_harness.reopen()
        external_reference_load = external_reference_harness.sink.load_invocation(
            external_reference_run_id,
            "call-1",
        )
        backing_harness = factory()
        backing_run_id = _contract_run_id(
            namespace,
            "invocation-authoritative-backing-reference",
        )
        backing_token = _contract_writer(backing_harness, backing_run_id)
        backing_seed_call_id = "backing-seed-call"
        backing_seed_statuses = tuple(
            backing_harness.sink.commit_invocation(
                _contract_invocation(
                    backing_run_id,
                    logical_call_id=backing_seed_call_id,
                    revision=revision,
                    dispatch_state=dispatch_state,
                    succeeded=revision == 3,
                ),
                (
                    {_CONTRACT_INVOCATION_BLOB_SHA256: _CONTRACT_INVOCATION_BLOB}
                    if revision == 3
                    else {}
                ),
                writer_token=backing_token,
            ).status
            for revision, dispatch_state in (
                (1, "reserved"),
                (2, "dispatch_started"),
                (3, "settled"),
            )
        )
        backing_harness = backing_harness.reopen()
        backing_call_id = "backing-reference-call"
        backing_setup_statuses = tuple(
            backing_harness.sink.commit_invocation(
                _contract_invocation(
                    backing_run_id,
                    logical_call_id=backing_call_id,
                    revision=revision,
                    dispatch_state=dispatch_state,
                ),
                {},
                writer_token=backing_token,
            ).status
            for revision, dispatch_state in ((1, "reserved"), (2, "dispatch_started"))
        )
        backing_invocation = _contract_invocation(
            backing_run_id,
            logical_call_id=backing_call_id,
            revision=3,
            dispatch_state="settled",
            succeeded=True,
        )
        backing_reference = backing_harness.sink.commit_invocation(
            backing_invocation,
            {},
            writer_token=backing_token,
        )
        backing_harness = backing_harness.reopen()
        backing_load = backing_harness.sink.load_invocation(
            backing_run_id,
            backing_call_id,
        )
        backing_reference_bytes = _contract_blob_hex(
            backing_load.value,
            _CONTRACT_INVOCATION_BLOB_SHA256,
        )
        uppercase_key_harness = factory()
        uppercase_key_run_id = _contract_run_id(
            namespace,
            "invocation-uppercase-blob-key",
        )
        uppercase_key_token = _contract_writer(
            uppercase_key_harness,
            uppercase_key_run_id,
        )
        uppercase_key_invocation = _contract_invocation(
            uppercase_key_run_id,
            revision=1,
            dispatch_state="reserved",
        )
        uppercase_key = uppercase_key_harness.sink.commit_invocation(
            uppercase_key_invocation,
            {_CONTRACT_INVOCATION_BLOB_SHA256.upper(): _CONTRACT_INVOCATION_BLOB},
            writer_token=uppercase_key_token,
        )
        uppercase_key_harness = uppercase_key_harness.reopen()
        uppercase_key_load = uppercase_key_harness.sink.load_invocation(
            uppercase_key_run_id,
            "call-1",
        )
        uppercase_key_setup_statuses = tuple(
            uppercase_key_harness.sink.commit_invocation(
                _contract_invocation(
                    uppercase_key_run_id,
                    revision=revision,
                    dispatch_state=dispatch_state,
                ),
                {},
                writer_token=uppercase_key_token,
            ).status
            for revision, dispatch_state in ((1, "reserved"), (2, "dispatch_started"))
        )
        uppercase_key_settled_invocation = _contract_invocation(
            uppercase_key_run_id,
            revision=3,
            dispatch_state="settled",
            succeeded=True,
        )
        uppercase_key_blob_leak = uppercase_key_harness.sink.commit_invocation(
            uppercase_key_settled_invocation,
            {},
            writer_token=uppercase_key_token,
        )
        uppercase_key_harness = uppercase_key_harness.reopen()
        uppercase_key_blob_leak_load = uppercase_key_harness.sink.load_invocation(
            uppercase_key_run_id,
            "call-1",
        )
        uppercase_key_recovery = uppercase_key_harness.sink.commit_invocation(
            uppercase_key_settled_invocation,
            {_CONTRACT_INVOCATION_BLOB_SHA256: _CONTRACT_INVOCATION_BLOB},
            writer_token=uppercase_key_token,
        )
        uppercase_key_harness = uppercase_key_harness.reopen()
        uppercase_key_recovery_load = uppercase_key_harness.sink.load_invocation(
            uppercase_key_run_id,
            "call-1",
        )
        uppercase_key_recovery_bytes = _contract_blob_hex(
            uppercase_key_recovery_load.value,
            _CONTRACT_INVOCATION_BLOB_SHA256,
        )
        malformed_harness = factory()
        malformed_run_id = _contract_run_id(namespace, "invocation-malformed-fresh-blob")
        malformed_token = _contract_writer(malformed_harness, malformed_run_id)
        malformed_seed_call_id = "blob-seed-call"
        malformed_seed_statuses = tuple(
            malformed_harness.sink.commit_invocation(
                _contract_invocation(
                    malformed_run_id,
                    logical_call_id=malformed_seed_call_id,
                    revision=revision,
                    dispatch_state=dispatch_state,
                    succeeded=revision == 3,
                ),
                (
                    {_CONTRACT_INVOCATION_BLOB_SHA256: _CONTRACT_INVOCATION_BLOB}
                    if revision == 3
                    else {}
                ),
                writer_token=malformed_token,
            ).status
            for revision, dispatch_state in (
                (1, "reserved"),
                (2, "dispatch_started"),
                (3, "settled"),
            )
        )
        malformed_setup_statuses = tuple(
            malformed_harness.sink.commit_invocation(
                _contract_invocation(
                    malformed_run_id,
                    revision=revision,
                    dispatch_state=dispatch_state,
                ),
                {},
                writer_token=malformed_token,
            ).status
            for revision, dispatch_state in ((1, "reserved"), (2, "dispatch_started"))
        )
        malformed_fresh_blob = malformed_harness.sink.commit_invocation(
            _contract_invocation(
                malformed_run_id,
                revision=3,
                dispatch_state="settled",
                succeeded=True,
            ),
            {_CONTRACT_INVOCATION_BLOB_SHA256: _CONTRACT_ALTERNATE_BLOB},
            writer_token=malformed_token,
        )
        malformed_harness = malformed_harness.reopen()
        malformed_seed_load = malformed_harness.sink.load_invocation(
            malformed_run_id,
            malformed_seed_call_id,
        )
        malformed_preserved_bytes = _contract_blob_hex(
            malformed_seed_load.value,
            _CONTRACT_INVOCATION_BLOB_SHA256,
        )
        malformed_fresh_load = malformed_harness.sink.load_invocation(
            malformed_run_id,
            "call-1",
        )
        malformed_recovery_invocation = _contract_invocation(
            malformed_run_id,
            revision=3,
            dispatch_state="settled",
            succeeded=True,
        )
        malformed_recovery = malformed_harness.sink.commit_invocation(
            malformed_recovery_invocation,
            {_CONTRACT_INVOCATION_BLOB_SHA256: _CONTRACT_INVOCATION_BLOB},
            writer_token=malformed_token,
        )
        malformed_harness = malformed_harness.reopen()
        malformed_recovery_load = malformed_harness.sink.load_invocation(
            malformed_run_id,
            "call-1",
        )
        malformed_recovery_bytes = _contract_blob_hex(
            malformed_recovery_load.value,
            _CONTRACT_INVOCATION_BLOB_SHA256,
        )

        harness = factory()
        run_id = _contract_run_id(namespace, "invocation")
        token = _contract_writer(harness, run_id)
        missing = harness.sink.load_invocation(run_id, "call-1")
        reserved_invocation = _contract_invocation(
            run_id,
            revision=1,
            dispatch_state="reserved",
        )
        reserved = harness.sink.commit_invocation(
            reserved_invocation,
            {},
            writer_token=token,
        )
        repeated_reserved = harness.sink.commit_invocation(
            reserved_invocation,
            {},
            writer_token=token,
        )
        conflicting_reserved_invocation = _contract_invocation(
            run_id,
            revision=1,
            dispatch_state="reserved",
            dispatch_id="dispatch-conflict",
        )
        conflicting_reserved = harness.sink.commit_invocation(
            conflicting_reserved_invocation,
            {},
            writer_token=token,
        )
        reserved_digest = _contract_record_digest(reserved_invocation.to_json())
        invocation_commit_evidence = {
            "committed": _contract_commit_evidence(
                reserved,
                sequence=reserved_invocation.revision,
                content_digest=reserved_digest,
            ),
            "already_committed": _contract_commit_evidence(
                repeated_reserved,
                sequence=reserved_invocation.revision,
                content_digest=reserved_digest,
            ),
            "conflict": _contract_commit_evidence(
                conflicting_reserved,
                sequence=conflicting_reserved_invocation.revision,
                content_digest=_contract_record_digest(conflicting_reserved_invocation.to_json()),
                winner_digest=reserved_digest,
            ),
        }
        started_invocation = _contract_invocation(
            run_id,
            revision=2,
            dispatch_state="dispatch_started",
        )
        started = harness.sink.commit_invocation(
            started_invocation,
            {},
            writer_token=token,
        )
        settled_failure_invocation = _contract_invocation(
            run_id,
            revision=3,
            dispatch_state="settled",
            retryable=True,
        )
        settled_failure = harness.sink.commit_invocation(
            settled_failure_invocation,
            {},
            writer_token=token,
        )
        next_attempt_invocation = _contract_invocation(
            run_id,
            revision=4,
            dispatch_attempt=2,
            dispatch_id="dispatch-2",
            dispatch_state="reserved",
        )
        next_attempt = harness.sink.commit_invocation(
            next_attempt_invocation,
            {},
            writer_token=token,
        )
        harness = harness.reopen()
        old_revision_retries: dict[int, tuple[str, int | None]] = {}
        for old_invocation in (
            reserved_invocation,
            started_invocation,
            settled_failure_invocation,
        ):
            retry = harness.sink.commit_invocation(
                old_invocation,
                {},
                writer_token=token,
            )
            reloaded = harness.sink.load_invocation(run_id, "call-1")
            old_revision_retries[old_invocation.revision] = (
                retry.status,
                reloaded.sequence,
            )
        historical_conflict_cases = {
            "revision_1": replace(
                reserved_invocation,
                dispatch_id="dispatch-historical-conflict-1",
            ),
            "revision_2": replace(
                started_invocation,
                dispatch_id="dispatch-historical-conflict-2",
            ),
            "revision_3": replace(
                settled_failure_invocation,
                dispatch_id="dispatch-historical-conflict-3",
            ),
            "revision_3_receipt": replace(
                settled_failure_invocation,
                receipt={
                    **dict(settled_failure_invocation.receipt or {}),
                    "provider_request_id": "historical-provider-request",
                },
            ),
            "revision_3_retryable_false": replace(
                settled_failure_invocation,
                receipt={
                    **dict(settled_failure_invocation.receipt or {}),
                    "retryable": False,
                },
            ),
            "revision_3_retryable_omitted": replace(
                settled_failure_invocation,
                receipt={
                    key: value
                    for key, value in dict(settled_failure_invocation.receipt or {}).items()
                    if key != "retryable"
                },
            ),
            "revision_3_failure_code": replace(
                settled_failure_invocation,
                failure_code="historical_provider_refused",
            ),
        }
        old_revision_conflicts: dict[str, tuple[str, int | None]] = {}
        for label, conflicting_invocation in historical_conflict_cases.items():
            conflicting_retry = harness.sink.commit_invocation(
                conflicting_invocation,
                {},
                writer_token=token,
            )
            reloaded_after_conflict = harness.sink.load_invocation(run_id, "call-1")
            old_revision_conflicts[label] = (
                conflicting_retry.status,
                reloaded_after_conflict.sequence,
            )
        loaded = harness.sink.load_invocation(run_id, "call-1")

        result_call_id = "call-result"
        result_setup_statuses = tuple(
            harness.sink.commit_invocation(
                _contract_invocation(
                    run_id,
                    logical_call_id=result_call_id,
                    revision=revision,
                    dispatch_state=dispatch_state,
                ),
                {},
                writer_token=token,
            ).status
            for revision, dispatch_state in ((1, "reserved"), (2, "dispatch_started"))
        )
        settled_result_invocation = _contract_invocation(
            run_id,
            logical_call_id=result_call_id,
            revision=3,
            dispatch_state="settled",
            succeeded=True,
        )
        settled_result = harness.sink.commit_invocation(
            settled_result_invocation,
            {_CONTRACT_INVOCATION_BLOB_SHA256: _CONTRACT_INVOCATION_BLOB},
            writer_token=token,
        )
        harness = harness.reopen()
        repeated_result = harness.sink.commit_invocation(
            settled_result_invocation,
            {_CONTRACT_INVOCATION_BLOB_SHA256: _CONTRACT_INVOCATION_BLOB},
            writer_token=token,
        )
        result_blob_key_conflict = harness.sink.commit_invocation(
            settled_result_invocation,
            {
                _CONTRACT_INVOCATION_BLOB_SHA256: _CONTRACT_INVOCATION_BLOB,
                _CONTRACT_ALTERNATE_BLOB_SHA256: _CONTRACT_ALTERNATE_BLOB,
            },
            writer_token=token,
        )
        harness = harness.reopen()
        conflicting_blob_call_id = "call-conflicting-blob-reference"
        conflicting_blob_reference_setup = tuple(
            harness.sink.commit_invocation(
                _contract_invocation(
                    run_id,
                    logical_call_id=conflicting_blob_call_id,
                    revision=revision,
                    dispatch_state=dispatch_state,
                ),
                {},
                writer_token=token,
            ).status
            for revision, dispatch_state in ((1, "reserved"), (2, "dispatch_started"))
        )
        conflicting_blob_reference = harness.sink.commit_invocation(
            replace(
                _contract_invocation(
                    run_id,
                    logical_call_id=conflicting_blob_call_id,
                    revision=3,
                    dispatch_state="settled",
                    succeeded=True,
                ),
                result_ref=f"blob:{_CONTRACT_ALTERNATE_BLOB_SHA256}",
            ),
            {},
            writer_token=token,
        )
        conflicting_blob_reference_load = harness.sink.load_invocation(
            run_id,
            conflicting_blob_call_id,
        )
        result_blob_bytes_conflict = harness.sink.commit_invocation(
            settled_result_invocation,
            {_CONTRACT_INVOCATION_BLOB_SHA256: _CONTRACT_ALTERNATE_BLOB},
            writer_token=token,
        )
        loaded_result = harness.sink.load_invocation(run_id, result_call_id)
        result_blob_bytes = _contract_blob_hex(
            loaded_result.value,
            _CONTRACT_INVOCATION_BLOB_SHA256,
        )
        reloaded_primary_after_result = harness.sink.load_invocation(run_id, "call-1")
        missing_logical_call = harness.sink.load_invocation(
            run_id,
            "call-missing",
        )
        invocation_load_fault_evidence = {
            status: _contract_authoritative_load_fault_evidence(
                factory,
                namespace,
                "invocation",
                status,
            )
            for status in ("corrupt", "unsupported_version")
        }
        outcomes.append(
            outcome_from_observations(
                "FENCED-04-INVOCATION-LIFECYCLE",
                FENCED_RUN_SINK_CONTRACT_PROFILE,
                (
                    observation("initial_load", expected="missing", actual=missing.status),
                    observation("reserve", expected="committed", actual=reserved.status),
                    observation(
                        "reserve_repeat",
                        expected="already_committed",
                        actual=repeated_reserved.status,
                    ),
                    observation(
                        "reserve_conflict",
                        expected="conflict",
                        actual=conflicting_reserved.status,
                    ),
                    *(
                        observation(
                            f"invocation_{status}_evidence",
                            expected=(True, True, True),
                            actual=evidence,
                        )
                        for status, evidence in invocation_commit_evidence.items()
                    ),
                    *(
                        observation(
                            f"invocation_identity_{field_name}",
                            expected="conflict",
                            actual=status,
                        )
                        for field_name, status in invocation_identity_statuses.items()
                    ),
                    *(
                        observation(
                            (f"receipt_retryability_identity_{winner_state}_to_{candidate_state}"),
                            expected=("conflict", evidence[0]),
                            actual=evidence[1],
                        )
                        for (winner_state, candidate_state), evidence in (
                            receipt_retryability_identity_results.items()
                        )
                    ),
                    *(
                        observation(
                            f"receipt_field_identity_{field_name}",
                            expected=(
                                "committed",
                                evidence[0],
                                "conflict",
                                evidence[0],
                            ),
                            actual=evidence[1],
                        )
                        for field_name, evidence in receipt_field_identity_results.items()
                    ),
                    *(
                        observation(
                            f"receipt_usage_identity_{field_name}",
                            expected=(
                                "committed",
                                evidence[0],
                                "conflict",
                                evidence[0],
                            ),
                            actual=evidence[1],
                        )
                        for field_name, evidence in receipt_usage_identity_results.items()
                    ),
                    *(
                        observation(
                            f"invocation_canonical_alias_{field_name}_{direction}",
                            expected="already_committed",
                            actual=status,
                        )
                        for (field_name, direction), status in (
                            invocation_canonical_alias_statuses.items()
                        )
                    ),
                    *(
                        observation(
                            f"invocation_canonical_alias_transition_{field_name}",
                            expected="committed",
                            actual=status,
                        )
                        for field_name, status in (
                            invocation_canonical_alias_transition_statuses.items()
                        )
                    ),
                    *(
                        observation(
                            f"invocation_canonical_alias_recovery_{field_name}",
                            expected="committed",
                            actual=status,
                        )
                        for field_name, status in (
                            invocation_canonical_alias_recovery_statuses.items()
                        )
                    ),
                    observation("start", expected="committed", actual=started.status),
                    observation(
                        "settled_failure", expected="committed", actual=settled_failure.status
                    ),
                    observation("proven_retry", expected="committed", actual=next_attempt.status),
                    *(
                        observation(
                            f"old_revision_{revision}_retry_and_head",
                            expected=("already_committed", 4),
                            actual=status_and_head,
                        )
                        for revision, status_and_head in old_revision_retries.items()
                    ),
                    *(
                        observation(
                            f"old_{label}_conflict_and_head",
                            expected=("conflict", 4),
                            actual=status_and_head,
                        )
                        for label, status_and_head in old_revision_conflicts.items()
                    ),
                    observation("latest_load", expected="loaded", actual=loaded.status),
                    observation("latest_revision", expected=4, actual=loaded.sequence),
                    observation(
                        "latest_record_revision",
                        expected=4,
                        actual=(loaded.value.revision if loaded.value else None),
                    ),
                    observation(
                        "latest_run_binding",
                        expected=run_id,
                        actual=(loaded.value.invocation.run_id if loaded.value else None),
                    ),
                    observation(
                        "latest_call_binding",
                        expected="call-1",
                        actual=(loaded.value.invocation.logical_call_id if loaded.value else None),
                    ),
                    observation(
                        "latest_invocation_digest",
                        expected=canonical_sha256(next_attempt_invocation.to_json()),
                        actual=(
                            canonical_sha256(loaded.value.invocation.to_json())
                            if loaded.value
                            else None
                        ),
                    ),
                    observation(
                        "result_setup",
                        expected=("committed", "committed"),
                        actual=result_setup_statuses,
                    ),
                    observation(
                        "settled_result", expected="committed", actual=settled_result.status
                    ),
                    observation(
                        "settled_result_repeat",
                        expected="already_committed",
                        actual=repeated_result.status,
                    ),
                    observation(
                        "result_blob_key_conflict",
                        expected="conflict",
                        actual=result_blob_key_conflict.status,
                    ),
                    observation(
                        "result_blob_key_conflict_reference_setup",
                        expected=("committed", "committed"),
                        actual=conflicting_blob_reference_setup,
                    ),
                    observation(
                        "result_blob_key_conflict_not_published",
                        expected=("conflict", 2),
                        actual=(
                            conflicting_blob_reference.status,
                            conflicting_blob_reference_load.sequence,
                        ),
                    ),
                    observation(
                        "result_blob_bytes_conflict",
                        expected="conflict",
                        actual=result_blob_bytes_conflict.status,
                    ),
                    observation(
                        "missing_reference_setup",
                        expected=("committed", "committed"),
                        actual=missing_reference_setup_statuses,
                    ),
                    observation(
                        "missing_reference_status",
                        expected="conflict",
                        actual=missing_reference.status,
                    ),
                    observation(
                        "missing_reference_head_not_published",
                        expected=2,
                        actual=missing_reference_load.sequence,
                    ),
                    observation(
                        "missing_reference_recovery",
                        expected="committed",
                        actual=missing_reference_recovery.status,
                    ),
                    observation(
                        "missing_reference_recovery_head",
                        expected=3,
                        actual=missing_reference_recovery_load.sequence,
                    ),
                    observation(
                        "missing_reference_recovery_bytes",
                        expected=_CONTRACT_UNRESOLVED_BLOB.hex(),
                        actual=missing_reference_recovery_bytes,
                    ),
                    observation(
                        "malformed_reference_setup",
                        expected=("committed", "committed"),
                        actual=malformed_reference_setup,
                    ),
                    observation(
                        "malformed_reference_status",
                        expected="conflict",
                        actual=malformed_reference.status,
                    ),
                    observation(
                        "malformed_reference_head_not_published",
                        expected=2,
                        actual=malformed_reference_load.sequence,
                    ),
                    observation(
                        "external_reference_setup",
                        expected=("committed", "committed"),
                        actual=external_reference_setup,
                    ),
                    observation(
                        "external_reference_status",
                        expected="committed",
                        actual=external_reference.status,
                    ),
                    observation(
                        "external_reference_head",
                        expected=3,
                        actual=external_reference_load.sequence,
                    ),
                    observation(
                        "external_reference_result_ref",
                        expected="object:contract-result",
                        actual=(
                            external_reference_load.value.invocation.result_ref
                            if external_reference_load.value
                            else None
                        ),
                    ),
                    observation(
                        "external_reference_payload",
                        expected=canonical_sha256(external_reference_invocation.to_json()),
                        actual=(
                            canonical_sha256(external_reference_load.value.invocation.to_json())
                            if external_reference_load.value
                            else None
                        ),
                    ),
                    observation(
                        "authoritative_backing_reference_setup",
                        expected=(
                            ("committed", "committed", "committed"),
                            ("committed", "committed"),
                        ),
                        actual=(backing_seed_statuses, backing_setup_statuses),
                    ),
                    observation(
                        "authoritative_backing_reference_status",
                        expected="committed",
                        actual=backing_reference.status,
                    ),
                    observation(
                        "authoritative_backing_reference_head",
                        expected=3,
                        actual=backing_load.sequence,
                    ),
                    observation(
                        "authoritative_backing_reference_bytes",
                        expected=_CONTRACT_INVOCATION_BLOB.hex(),
                        actual=backing_reference_bytes,
                    ),
                    observation(
                        "uppercase_blob_key_status",
                        expected="conflict",
                        actual=uppercase_key.status,
                    ),
                    observation(
                        "uppercase_blob_key_not_published",
                        expected="missing",
                        actual=uppercase_key_load.status,
                    ),
                    observation(
                        "uppercase_blob_key_setup",
                        expected=("committed", "committed"),
                        actual=uppercase_key_setup_statuses,
                    ),
                    observation(
                        "uppercase_blob_key_bytes_not_published",
                        expected=("conflict", 2),
                        actual=(
                            uppercase_key_blob_leak.status,
                            uppercase_key_blob_leak_load.sequence,
                        ),
                    ),
                    observation(
                        "uppercase_blob_key_recovery",
                        expected=("committed", 3),
                        actual=(
                            uppercase_key_recovery.status,
                            uppercase_key_recovery_load.sequence,
                        ),
                    ),
                    observation(
                        "uppercase_blob_key_recovery_bytes",
                        expected=_CONTRACT_INVOCATION_BLOB.hex(),
                        actual=uppercase_key_recovery_bytes,
                    ),
                    observation(
                        "malformed_fresh_blob_setup",
                        expected=("committed", "committed"),
                        actual=malformed_setup_statuses,
                    ),
                    observation(
                        "malformed_fresh_blob_seed",
                        expected=("committed", "committed", "committed"),
                        actual=malformed_seed_statuses,
                    ),
                    observation(
                        "malformed_fresh_blob_status",
                        expected="conflict",
                        actual=malformed_fresh_blob.status,
                    ),
                    observation(
                        "malformed_fresh_blob_head_not_published",
                        expected=2,
                        actual=malformed_fresh_load.sequence,
                    ),
                    observation(
                        "malformed_fresh_blob_preserves_existing_bytes",
                        expected=_CONTRACT_INVOCATION_BLOB.hex(),
                        actual=malformed_preserved_bytes,
                    ),
                    observation(
                        "malformed_fresh_blob_recovery",
                        expected="committed",
                        actual=malformed_recovery.status,
                    ),
                    observation(
                        "malformed_fresh_blob_recovery_head",
                        expected=3,
                        actual=malformed_recovery_load.sequence,
                    ),
                    observation(
                        "malformed_fresh_blob_recovery_bytes",
                        expected=_CONTRACT_INVOCATION_BLOB.hex(),
                        actual=malformed_recovery_bytes,
                    ),
                    observation("result_load", expected="loaded", actual=loaded_result.status),
                    observation(
                        "result_ref",
                        expected=f"blob:{_CONTRACT_INVOCATION_BLOB_SHA256}",
                        actual=(
                            loaded_result.value.invocation.result_ref
                            if loaded_result.value
                            else None
                        ),
                    ),
                    observation(
                        "result_invocation_digest",
                        expected=canonical_sha256(settled_result_invocation.to_json()),
                        actual=(
                            canonical_sha256(loaded_result.value.invocation.to_json())
                            if loaded_result.value
                            else None
                        ),
                    ),
                    observation(
                        "result_blob_round_trip",
                        expected=_CONTRACT_INVOCATION_BLOB.hex(),
                        actual=result_blob_bytes,
                    ),
                    observation(
                        "primary_call_after_result_head",
                        expected=4,
                        actual=reloaded_primary_after_result.sequence,
                    ),
                    observation(
                        "primary_call_after_result_binding",
                        expected="call-1",
                        actual=(
                            reloaded_primary_after_result.value.invocation.logical_call_id
                            if reloaded_primary_after_result.value
                            else None
                        ),
                    ),
                    observation(
                        "primary_call_after_result_payload",
                        expected=canonical_sha256(next_attempt_invocation.to_json()),
                        actual=(
                            canonical_sha256(
                                reloaded_primary_after_result.value.invocation.to_json()
                            )
                            if reloaded_primary_after_result.value
                            else None
                        ),
                    ),
                    observation(
                        "missing_logical_call_status",
                        expected="missing",
                        actual=missing_logical_call.status,
                    ),
                    observation(
                        "missing_logical_call_value_absent",
                        expected=True,
                        actual=missing_logical_call.value is None,
                    ),
                    *(
                        observation(
                            f"authoritative_load_{status}",
                            expected=(status, True),
                            actual=actual,
                        )
                        for status, actual in invocation_load_fault_evidence.items()
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(
            _error(
                "FENCED-04-INVOCATION-LIFECYCLE",
                FENCED_RUN_SINK_CONTRACT_PROFILE,
                exc,
            )
        )

    try:
        harness = factory()
        run_id = _contract_run_id(namespace, "invocation-refusals")
        token = _contract_writer(harness, run_id)
        harness.sink.commit_invocation(
            _contract_invocation(run_id, revision=1, dispatch_state="reserved"),
            {},
            writer_token=token,
        )
        gap = harness.sink.commit_invocation(
            _contract_invocation(run_id, revision=3, dispatch_state="dispatch_started"),
            {},
            writer_token=token,
        )
        initial_refusal_statuses = {
            state: _contract_first_invocation_state_status(
                factory,
                _contract_run_id(namespace, f"invocation-first-{state.replace('_', '-')}"),
                state,
            )
            for state in ("dispatch_started", "settled", "unknown")
        }
        invalid_initial_coordinates = {
            "revision_2": (2, 1),
            "attempt_2": (1, 2),
            "revision_2_attempt_2": (2, 2),
        }
        initial_coordinate_statuses = {
            label: _contract_first_invocation_state_status(
                factory,
                _contract_run_id(namespace, f"invocation-first-coordinate-{label}"),
                "reserved",
                revision=revision,
                dispatch_attempt=dispatch_attempt,
            )
            for label, (revision, dispatch_attempt) in invalid_initial_coordinates.items()
        }
        forbidden_edges = (
            ("reserved", "reserved"),
            ("reserved", "settled"),
            ("reserved", "unknown"),
            ("dispatch_started", "reserved"),
            ("dispatch_started", "dispatch_started"),
            ("settled", "reserved"),
            ("settled", "dispatch_started"),
            ("settled", "settled"),
            ("settled", "unknown"),
            ("unknown", "reserved"),
            ("unknown", "dispatch_started"),
            ("unknown", "settled"),
            ("unknown", "unknown"),
        )
        forbidden_edge_statuses = {
            (source, target): _contract_forbidden_state_edge_status(
                factory,
                _contract_run_id(
                    namespace,
                    f"invocation-edge-{source.replace('_', '-')}-{target.replace('_', '-')}",
                ),
                source,
                target,
            )
            for source, target in forbidden_edges
        }
        invalid_retry_coordinates = {
            "same_attempt": (1, "dispatch-2"),
            "same_dispatch_id": (2, "dispatch-1"),
            "same_attempt_and_dispatch_id": (1, "dispatch-1"),
            "skipped_attempt": (3, "dispatch-3"),
        }
        invalid_retry_coordinate_statuses = {
            label: _contract_retry_coordinate_status(
                factory,
                _contract_run_id(namespace, f"invocation-retry-coordinate-{label}"),
                dispatch_attempt=attempt,
                dispatch_id=dispatch_id,
            )
            for label, (attempt, dispatch_id) in invalid_retry_coordinates.items()
        }
        retry_identity_drift_statuses = {
            field_name: _contract_retry_identity_drift_status(
                factory,
                _contract_run_id(
                    namespace,
                    f"invocation-retry-identity-{field_name.replace('_', '-')}",
                ),
                field_name,
            )
            for field_name in sorted(_CONTRACT_RETRY_STABLE_IDENTITY_FIELDS)
        }
        (
            historical_dispatch_id_status,
            valid_third_attempt_statuses,
        ) = _contract_historical_dispatch_id_reuse_status(
            factory,
            _contract_run_id(namespace, "invocation-retry-historical-dispatch-id"),
        )
        identity_drift_statuses = {
            "idempotency_key": _contract_invocation_drift_status(
                factory,
                _contract_run_id(namespace, "invocation-drift-idempotency-key"),
                "idempotency_key",
                "contract-idempotency-key-drift",
            ),
            "request_digest": _contract_invocation_drift_status(
                factory,
                _contract_run_id(namespace, "invocation-drift-request-digest"),
                "request_digest",
                "b" * 64,
            ),
            "dispatch_id": _contract_invocation_drift_status(
                factory,
                _contract_run_id(namespace, "invocation-drift-dispatch-id"),
                "dispatch_id",
                "dispatch-drift",
            ),
            "dispatch_attempt": _contract_invocation_drift_status(
                factory,
                _contract_run_id(namespace, "invocation-drift-dispatch-attempt"),
                "dispatch_attempt",
                2,
            ),
        }
        terminal_identity_drift_values = {
            "idempotency_key": "contract-terminal-idempotency-drift",
            "request_digest": "b" * 64,
            "dispatch_id": "dispatch-terminal-drift",
            "dispatch_attempt": 2,
        }
        terminal_identity_drift_statuses = {
            (terminal_state, field_name): _contract_terminal_invocation_drift_status(
                factory,
                _contract_run_id(
                    namespace,
                    (
                        f"invocation-terminal-drift-{terminal_state.replace('_', '-')}"
                        f"-{field_name.replace('_', '-')}"
                    ),
                ),
                terminal_state,
                field_name,
                field_value,
            )
            for terminal_state in ("settled", "unknown")
            for field_name, field_value in terminal_identity_drift_values.items()
        }
        terminal_retry_cases = {
            "unknown": {
                "terminal_state": "unknown",
            },
            "unknown_with_failure_code": {
                "terminal_state": "unknown",
                "unknown_failure_code": "transport_uncertain",
            },
            "success": {
                "terminal_state": "settled",
                "succeeded": True,
            },
            "retryable_tagged_success": {
                "terminal_state": "settled",
                "retryable": True,
                "succeeded": True,
            },
            "nonretry_failure": {
                "terminal_state": "settled",
            },
            "omitted_retryable_failure": {
                "terminal_state": "settled",
                "retryable": None,
            },
            "retryable_failure": {
                "terminal_state": "settled",
                "retryable": True,
            },
        }
        terminal_retry_expectations = {
            "unknown": "conflict",
            "unknown_with_failure_code": "conflict",
            "success": "conflict",
            "retryable_tagged_success": "conflict",
            "nonretry_failure": "conflict",
            "omitted_retryable_failure": "conflict",
            "retryable_failure": "committed",
        }
        if terminal_retry_cases.keys() != terminal_retry_expectations.keys():
            raise AssertionError("terminal retry policy matrix is incomplete")
        terminal_retry_results = {
            label: _contract_retry_after_terminal_invocation(
                factory,
                _contract_run_id(namespace, f"invocation-after-{label.replace('_', '-')}"),
                **case,
            )
            for label, case in terminal_retry_cases.items()
        }
        outcomes.append(
            outcome_from_observations(
                "FENCED-05-INVOCATION-REFUSES-ILLEGAL-TRANSITIONS",
                FENCED_RUN_SINK_CONTRACT_PROFILE,
                (
                    *(
                        observation(
                            f"first_state_{state}",
                            expected="conflict",
                            actual=status,
                        )
                        for state, status in initial_refusal_statuses.items()
                    ),
                    *(
                        observation(
                            f"first_coordinate_{label}",
                            expected="conflict",
                            actual=status,
                        )
                        for label, status in initial_coordinate_statuses.items()
                    ),
                    observation("revision_gap", expected="conflict", actual=gap.status),
                    *(
                        observation(
                            f"state_edge_{source}_to_{target}",
                            expected="conflict",
                            actual=status,
                        )
                        for (source, target), status in forbidden_edge_statuses.items()
                    ),
                    *(
                        observation(
                            f"retry_coordinate_{label}",
                            expected="conflict",
                            actual=status,
                        )
                        for label, status in invalid_retry_coordinate_statuses.items()
                    ),
                    *(
                        observation(
                            f"retry_identity_drift_{field_name}",
                            expected="conflict",
                            actual=status,
                        )
                        for field_name, status in retry_identity_drift_statuses.items()
                    ),
                    observation(
                        "retry_coordinate_historical_dispatch_id",
                        expected="conflict",
                        actual=historical_dispatch_id_status,
                    ),
                    observation(
                        "valid_third_attempt_lifecycle",
                        expected=("committed", "committed", "committed"),
                        actual=valid_third_attempt_statuses,
                    ),
                    *(
                        observation(
                            f"identity_drift_{field_name}",
                            expected="conflict",
                            actual=status,
                        )
                        for field_name, status in identity_drift_statuses.items()
                    ),
                    *(
                        observation(
                            f"terminal_identity_drift_{terminal_state}_{field_name}",
                            expected="conflict",
                            actual=status,
                        )
                        for (terminal_state, field_name), status in (
                            terminal_identity_drift_statuses.items()
                        )
                    ),
                    *(
                        observation(
                            f"{label}_history",
                            expected=("committed", "committed", "committed"),
                            actual=history_and_retry[0],
                        )
                        for label, history_and_retry in terminal_retry_results.items()
                    ),
                    *(
                        observation(
                            f"retry_after_{label}",
                            expected=terminal_retry_expectations[label],
                            actual=history_and_retry[1],
                        )
                        for label, history_and_retry in terminal_retry_results.items()
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(
            _error(
                "FENCED-05-INVOCATION-REFUSES-ILLEGAL-TRANSITIONS",
                FENCED_RUN_SINK_CONTRACT_PROFILE,
                exc,
            )
        )

    try:
        harness = factory()
        run_a_id = _contract_run_id(namespace, "run-a")
        run_b_id = _contract_run_id(namespace, "run-b")
        run_a_token = _contract_writer(harness, run_a_id)
        run_b_token = _contract_writer(harness, run_b_id)
        run_a_records = (
            RunCheckpoint(run_id=run_a_id, seq=1, final_text="run-a-checkpoint"),
            _contract_event(run_a_id, seq=1),
            _contract_invocation(
                run_a_id,
                revision=1,
                dispatch_id="dispatch-run-a",
                dispatch_state="reserved",
                request_digest="a" * 64,
            ),
            _contract_terminal(run_a_id),
        )
        run_b_records = (
            RunCheckpoint(run_id=run_b_id, seq=1, final_text="run-b-checkpoint"),
            _contract_event(run_b_id, seq=1),
            _contract_invocation(
                run_b_id,
                revision=1,
                dispatch_id="dispatch-run-b",
                dispatch_state="reserved",
                request_digest="b" * 64,
            ),
            _contract_terminal(run_b_id),
        )
        checkpoint, event, invocation, terminal = run_b_records
        swapped_checkpoint = harness.sink.commit_checkpoint(
            checkpoint,
            {},
            writer_token=run_a_token,
        )
        swapped_event = harness.sink.append_event(event, writer_token=run_a_token)
        swapped_invocation = harness.sink.commit_invocation(
            invocation,
            {},
            writer_token=run_a_token,
        )
        swapped_blob_checkpoint_record = RunCheckpoint(
            run_id=run_b_id,
            seq=2,
            workspace_delta=[
                {
                    "path": "cross-run-only-checkpoint-blob.txt",
                    "kind": "file",
                    "change_kind": "created",
                    "content_sha256": _CONTRACT_STALE_HANDOFF_BLOB_SHA256,
                }
            ],
        )
        swapped_blob_checkpoint = harness.sink.commit_checkpoint(
            swapped_blob_checkpoint_record,
            {_CONTRACT_STALE_HANDOFF_BLOB_SHA256: _CONTRACT_STALE_HANDOFF_BLOB},
            writer_token=run_a_token,
        )
        swapped_blob_invocation_record = _contract_invocation(
            run_b_id,
            logical_call_id="cross-run-blob-call",
            revision=1,
            dispatch_state="reserved",
        )
        swapped_blob_invocation = harness.sink.commit_invocation(
            swapped_blob_invocation_record,
            {_CONTRACT_ALTERNATE_BLOB_SHA256: _CONTRACT_ALTERNATE_BLOB},
            writer_token=run_a_token,
        )
        swapped_malformed_checkpoint = harness.sink.commit_checkpoint(
            checkpoint,
            {_CONTRACT_CHECKPOINT_BLOB_SHA256: _CONTRACT_ALTERNATE_BLOB},
            writer_token=run_a_token,
        )
        swapped_malformed_invocation = harness.sink.commit_invocation(
            invocation,
            {_CONTRACT_INVOCATION_BLOB_SHA256: _CONTRACT_ALTERNATE_BLOB},
            writer_token=run_a_token,
        )
        swapped_terminal = harness.sink.settle_terminal(
            terminal,
            writer_token=run_a_token,
        )
        run_a_checkpoint, run_a_event, run_a_invocation, run_a_terminal = run_a_records
        authorized_a = (
            harness.sink.commit_checkpoint(
                run_a_checkpoint,
                {},
                writer_token=run_a_token,
            ),
            harness.sink.append_event(run_a_event, writer_token=run_a_token),
            harness.sink.commit_invocation(
                run_a_invocation,
                {},
                writer_token=run_a_token,
            ),
            harness.sink.settle_terminal(run_a_terminal, writer_token=run_a_token),
        )
        authorized_b = (
            harness.sink.commit_checkpoint(
                checkpoint,
                {},
                writer_token=run_b_token,
            ),
            harness.sink.append_event(event, writer_token=run_b_token),
            harness.sink.commit_invocation(
                invocation,
                {},
                writer_token=run_b_token,
            ),
            harness.sink.settle_terminal(terminal, writer_token=run_b_token),
        )
        harness = harness.reopen()
        repeated_a = (
            harness.sink.commit_checkpoint(
                run_a_checkpoint,
                {},
                writer_token=run_a_token,
            ),
            harness.sink.append_event(run_a_event, writer_token=run_a_token),
            harness.sink.commit_invocation(
                run_a_invocation,
                {},
                writer_token=run_a_token,
            ),
            harness.sink.settle_terminal(run_a_terminal, writer_token=run_a_token),
        )
        repeated_b = (
            harness.sink.commit_checkpoint(
                checkpoint,
                {},
                writer_token=run_b_token,
            ),
            harness.sink.append_event(event, writer_token=run_b_token),
            harness.sink.commit_invocation(
                invocation,
                {},
                writer_token=run_b_token,
            ),
            harness.sink.settle_terminal(terminal, writer_token=run_b_token),
        )
        loaded_a_checkpoint = harness.sink.latest_checked(run_a_id)
        loaded_b_checkpoint = harness.sink.latest_checked(run_b_id)
        loaded_a_invocation = harness.sink.load_invocation(run_a_id, "call-1")
        loaded_b_invocation = harness.sink.load_invocation(run_b_id, "call-1")
        loaded_a_event = harness.read_event(run_a_id, 1)
        loaded_b_event = harness.read_event(run_b_id, 1)
        loaded_a_terminal = harness.read_terminal(run_a_id)
        loaded_b_terminal = harness.read_terminal(run_b_id)
        loaded_checkpoint_payloads = (
            (
                canonical_sha256(loaded_a_checkpoint.value.checkpoint.to_json())
                if loaded_a_checkpoint.value
                else None
            ),
            (
                canonical_sha256(loaded_b_checkpoint.value.checkpoint.to_json())
                if loaded_b_checkpoint.value
                else None
            ),
        )
        loaded_invocation_payloads = (
            (
                canonical_sha256(loaded_a_invocation.value.invocation.to_json())
                if loaded_a_invocation.value
                else None
            ),
            (
                canonical_sha256(loaded_b_invocation.value.invocation.to_json())
                if loaded_b_invocation.value
                else None
            ),
        )
        loaded_event_payloads = (
            canonical_sha256(loaded_a_event.to_json()) if loaded_a_event else None,
            canonical_sha256(loaded_b_event.to_json()) if loaded_b_event else None,
        )
        loaded_terminal_payloads = (
            canonical_sha256(loaded_a_terminal.to_json()) if loaded_a_terminal else None,
            canonical_sha256(loaded_b_terminal.to_json()) if loaded_b_terminal else None,
        )
        cross_run_checkpoint_blob_leak = harness.sink.commit_checkpoint(
            swapped_blob_checkpoint_record,
            {},
            writer_token=run_b_token,
        )
        cross_run_invocation_blob_setup = tuple(
            harness.sink.commit_invocation(
                _contract_invocation(
                    run_b_id,
                    logical_call_id="cross-run-blob-call",
                    revision=revision,
                    dispatch_state=dispatch_state,
                ),
                {},
                writer_token=run_b_token,
            ).status
            for revision, dispatch_state in ((1, "reserved"), (2, "dispatch_started"))
        )
        cross_run_invocation_blob_leak = harness.sink.commit_invocation(
            replace(
                _contract_invocation(
                    run_b_id,
                    logical_call_id="cross-run-blob-call",
                    revision=3,
                    dispatch_state="settled",
                    succeeded=True,
                ),
                result_ref=f"blob:{_CONTRACT_ALTERNATE_BLOB_SHA256}",
            ),
            {},
            writer_token=run_b_token,
        )

        cross_blob_harness = factory()
        cross_blob_run_a_id = _contract_run_id(namespace, "blob-run-a")
        cross_blob_run_b_id = _contract_run_id(namespace, "blob-run-b")
        cross_blob_run_a_token = _contract_writer(
            cross_blob_harness,
            cross_blob_run_a_id,
        )
        cross_blob_run_b_token = _contract_writer(
            cross_blob_harness,
            cross_blob_run_b_id,
        )
        cross_blob_seed = cross_blob_harness.sink.commit_checkpoint(
            RunCheckpoint(
                run_id=cross_blob_run_a_id,
                seq=1,
                workspace_delta=[
                    {
                        "path": "run-private-blob.txt",
                        "kind": "file",
                        "change_kind": "created",
                        "content_sha256": _CONTRACT_CHECKPOINT_BLOB_SHA256,
                    }
                ],
            ),
            {_CONTRACT_CHECKPOINT_BLOB_SHA256: _CONTRACT_CHECKPOINT_BLOB},
            writer_token=cross_blob_run_a_token,
        )
        cross_blob_checkpoint = cross_blob_harness.sink.commit_checkpoint(
            RunCheckpoint(
                run_id=cross_blob_run_b_id,
                seq=1,
                workspace_delta=[
                    {
                        "path": "foreign-run-reference.txt",
                        "kind": "file",
                        "change_kind": "created",
                        "content_sha256": _CONTRACT_CHECKPOINT_BLOB_SHA256,
                    }
                ],
            ),
            {},
            writer_token=cross_blob_run_b_token,
        )
        cross_blob_invocation_setup = tuple(
            cross_blob_harness.sink.commit_invocation(
                _contract_invocation(
                    cross_blob_run_b_id,
                    revision=revision,
                    dispatch_state=dispatch_state,
                ),
                {},
                writer_token=cross_blob_run_b_token,
            ).status
            for revision, dispatch_state in ((1, "reserved"), (2, "dispatch_started"))
        )
        cross_blob_invocation = cross_blob_harness.sink.commit_invocation(
            replace(
                _contract_invocation(
                    cross_blob_run_b_id,
                    revision=3,
                    dispatch_state="settled",
                    succeeded=True,
                ),
                result_ref=f"blob:{_CONTRACT_CHECKPOINT_BLOB_SHA256}",
            ),
            {},
            writer_token=cross_blob_run_b_token,
        )
        outcomes.append(
            outcome_from_observations(
                "FENCED-06-WRITER-TOKEN-RUN-BINDING",
                FENCED_RUN_SINK_CONTRACT_PROFILE,
                (
                    observation(
                        "cross_run_checkpoint",
                        expected="fenced",
                        actual=swapped_checkpoint.status,
                    ),
                    observation("cross_run_event", expected="fenced", actual=swapped_event.status),
                    observation(
                        "cross_run_invocation",
                        expected="fenced",
                        actual=swapped_invocation.status,
                    ),
                    observation(
                        "cross_run_blob_checkpoint",
                        expected="fenced",
                        actual=swapped_blob_checkpoint.status,
                    ),
                    observation(
                        "cross_run_blob_invocation",
                        expected="fenced",
                        actual=swapped_blob_invocation.status,
                    ),
                    observation(
                        "cross_run_checkpoint_blob_not_published",
                        expected="conflict",
                        actual=cross_run_checkpoint_blob_leak.status,
                    ),
                    observation(
                        "cross_run_invocation_blob_setup",
                        expected=("committed", "committed"),
                        actual=cross_run_invocation_blob_setup,
                    ),
                    observation(
                        "cross_run_invocation_blob_not_published",
                        expected="conflict",
                        actual=cross_run_invocation_blob_leak.status,
                    ),
                    observation(
                        "cross_run_malformed_checkpoint",
                        expected="fenced",
                        actual=swapped_malformed_checkpoint.status,
                    ),
                    observation(
                        "cross_run_malformed_invocation",
                        expected="fenced",
                        actual=swapped_malformed_invocation.status,
                    ),
                    observation(
                        "cross_run_terminal",
                        expected="fenced",
                        actual=swapped_terminal.status,
                    ),
                    *(
                        observation(
                            f"run_a_bound_{mutation}",
                            expected="committed",
                            actual=result.status,
                        )
                        for mutation, result in zip(
                            ("checkpoint", "event", "invocation", "terminal"),
                            authorized_a,
                            strict=True,
                        )
                    ),
                    *(
                        observation(
                            f"run_b_bound_{mutation}",
                            expected="committed",
                            actual=result.status,
                        )
                        for mutation, result in zip(
                            ("checkpoint", "event", "invocation", "terminal"),
                            authorized_b,
                            strict=True,
                        )
                    ),
                    observation(
                        "run_a_idempotent_retries",
                        expected=("already_committed",) * 4,
                        actual=tuple(result.status for result in repeated_a),
                    ),
                    observation(
                        "run_b_idempotent_retries",
                        expected=("already_committed",) * 4,
                        actual=tuple(result.status for result in repeated_b),
                    ),
                    observation(
                        "run_a_loaded_bindings",
                        expected=(run_a_id, run_a_id),
                        actual=(
                            (
                                loaded_a_checkpoint.value.checkpoint.run_id
                                if loaded_a_checkpoint.value
                                else None
                            ),
                            (
                                loaded_a_invocation.value.invocation.run_id
                                if loaded_a_invocation.value
                                else None
                            ),
                        ),
                    ),
                    observation(
                        "run_b_loaded_bindings",
                        expected=(run_b_id, run_b_id),
                        actual=(
                            (
                                loaded_b_checkpoint.value.checkpoint.run_id
                                if loaded_b_checkpoint.value
                                else None
                            ),
                            (
                                loaded_b_invocation.value.invocation.run_id
                                if loaded_b_invocation.value
                                else None
                            ),
                        ),
                    ),
                    observation(
                        "run_a_checkpoint_payload",
                        expected=canonical_sha256(run_a_checkpoint.to_json()),
                        actual=loaded_checkpoint_payloads[0],
                    ),
                    observation(
                        "run_b_checkpoint_payload",
                        expected=canonical_sha256(checkpoint.to_json()),
                        actual=loaded_checkpoint_payloads[1],
                    ),
                    observation(
                        "run_a_invocation_payload",
                        expected=canonical_sha256(run_a_invocation.to_json()),
                        actual=loaded_invocation_payloads[0],
                    ),
                    observation(
                        "run_b_invocation_payload",
                        expected=canonical_sha256(invocation.to_json()),
                        actual=loaded_invocation_payloads[1],
                    ),
                    observation(
                        "run_a_event_payload",
                        expected=canonical_sha256(run_a_event.to_json()),
                        actual=loaded_event_payloads[0],
                    ),
                    observation(
                        "run_b_event_payload",
                        expected=canonical_sha256(event.to_json()),
                        actual=loaded_event_payloads[1],
                    ),
                    observation(
                        "run_a_terminal_payload",
                        expected=canonical_sha256(run_a_terminal.to_json()),
                        actual=loaded_terminal_payloads[0],
                    ),
                    observation(
                        "run_b_terminal_payload",
                        expected=canonical_sha256(terminal.to_json()),
                        actual=loaded_terminal_payloads[1],
                    ),
                    observation(
                        "cross_run_blob_seed",
                        expected="committed",
                        actual=cross_blob_seed.status,
                    ),
                    observation(
                        "cross_run_blob_checkpoint_reference",
                        expected="conflict",
                        actual=cross_blob_checkpoint.status,
                    ),
                    observation(
                        "cross_run_blob_invocation_setup",
                        expected=("committed", "committed"),
                        actual=cross_blob_invocation_setup,
                    ),
                    observation(
                        "cross_run_blob_invocation_reference",
                        expected="conflict",
                        actual=cross_blob_invocation.status,
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(
            _error(
                "FENCED-06-WRITER-TOKEN-RUN-BINDING",
                FENCED_RUN_SINK_CONTRACT_PROFILE,
                exc,
            )
        )
    return tuple(outcomes)


def run_fenced_run_sink_contract(
    factory: FencedRunSinkHarnessFactory,
) -> tuple[ConformanceRuleOutcome, ...]:
    """Execute backend-neutral fencing, conflict, terminal, and invocation invariants."""

    registry = _FencedHarnessRegistry()
    try:
        return _run_fenced_run_sink_contract(registry.wrap_factory(factory))
    finally:
        registry.close_all()


def run_capability_broker_contract(
    factory: CapabilityBrokerFactory,
) -> tuple[ConformanceRuleOutcome, ...]:
    """Execute the broker outcome and least-privilege contract without pytest."""

    request = CapabilityRequest(
        capability="web.search",
        scope={"allowed_domains": ["a.example"]},
        run_id="contract_run",
        ttl_seconds=300,
    )
    try:
        outcome = factory().request(request)
    except Exception as exc:
        return (_error("BROKER-01-OUTCOME", BROKER_CONTRACT_PROFILE, exc),)
    valid_outcome = isinstance(outcome, (CapabilityLease, CapabilityDenial, CapabilityPending))
    if isinstance(outcome, CapabilityPending):
        named_capability: object = outcome.request.capability
    elif isinstance(outcome, (CapabilityLease, CapabilityDenial)):
        named_capability = outcome.capability
    else:
        named_capability = None
    outcomes = [
        outcome_from_observations(
            "BROKER-01-OUTCOME",
            BROKER_CONTRACT_PROFILE,
            (
                observation(
                    "grant_union",
                    expected=True,
                    actual=valid_outcome,
                ),
                observation(
                    "capability_identity",
                    expected=request.capability,
                    actual=named_capability,
                ),
            ),
        )
    ]
    if not valid_outcome:
        outcomes.append(
            ConformanceRuleOutcome(
                rule_id="BROKER-02-LEASE-LEAST-PRIVILEGE",
                profile_id=BROKER_CONTRACT_PROFILE,
                status="skipped",
                error="broker returned an invalid outcome",
            )
        )
        return tuple(outcomes)
    if isinstance(outcome, CapabilityLease):
        now = time.time()
        outcomes.append(
            outcome_from_observations(
                "BROKER-02-LEASE-LEAST-PRIVILEGE",
                BROKER_CONTRACT_PROFILE,
                (
                    observation(
                        "scope_narrowing",
                        expected=True,
                        actual=scope_within(outcome.scope, request.scope),
                    ),
                    observation("future_expiry", expected=True, actual=outcome.expires_at > now),
                    observation("token_handle", expected=True, actual=bool(outcome.token_ref)),
                ),
            )
        )
    else:
        outcomes.append(
            ConformanceRuleOutcome(
                rule_id="BROKER-02-LEASE-LEAST-PRIVILEGE",
                profile_id=BROKER_CONTRACT_PROFILE,
                status="skipped",
                error="broker policy did not grant this request",
            )
        )
    return tuple(outcomes)


def run_redactor_contract(factory: RedactorFactory) -> tuple[ConformanceRuleOutcome, ...]:
    """Execute the obligations a `Redactor` owes the model-I/O capture pipeline.

    A redactor is the one place an integrator can silently turn "redacted" into "disclosed", so the
    rules check the two properties the pipeline actually depends on — a stable result, and no
    survival of a value the policy named a secret — plus the caller-side guarantee that a redactor
    which fails produces nothing rather than raw content.
    """

    outcomes: list[ConformanceRuleOutcome] = []
    policy = RedactionPolicy(patterns=(r"sk-[A-Za-z0-9]+",), literals=("hunter2",))
    payload = {
        "api_key": "sk-live-must-not-survive",
        "prompt": "the key is sk-abc123 and the password is hunter2",
        "nested": {"Authorization": "Bearer must-not-survive", "count": 7},
        "items": ["sk-xyz789", 3, None],
    }

    try:
        # One instance, called twice -- not two instances called once. A ``CapturePolicy`` holds its
        # redactor for the life of the policy, so per-instance state is exactly the nondeterminism
        # production would hit, and constructing a second instance would hide it behind a fresh one.
        redactor = factory()
        first = redactor.redact(payload, policy=policy)
        second = redactor.redact(payload, policy=policy)
        outcomes.append(
            outcome_from_observations(
                "REDACTOR-01-DETERMINISTIC",
                REDACTOR_CONTRACT_PROFILE,
                (
                    # Canonical JSON, so key order cannot make two equal results look different.
                    observation(
                        "repeated_redaction_is_identical",
                        expected=canonical_sha256({"value": _jsonish(first)}),
                        actual=canonical_sha256({"value": _jsonish(second)}),
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("REDACTOR-01-DETERMINISTIC", REDACTOR_CONTRACT_PROFILE, exc))

    try:
        redacted = factory().redact(payload, policy=policy)
        rendered = json.dumps(_jsonish(redacted))
        outcomes.append(
            outcome_from_observations(
                "REDACTOR-02-NO-DEFAULT-SECRET-LEAK",
                REDACTOR_CONTRACT_PROFILE,
                (
                    # A secret-named key must not survive at any depth, in a mapping or inside a list.
                    observation(
                        "top_level_secret_key",
                        expected=False,
                        actual="sk-live-must-not-survive" in rendered,
                    ),
                    observation(
                        "nested_secret_key",
                        expected=False,
                        actual="Bearer must-not-survive" in rendered,
                    ),
                    # Non-secret data must survive, or "redact everything" would pass every rule.
                    observation("non_secret_value_survives", expected=True, actual="7" in rendered),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(
            _error("REDACTOR-02-NO-DEFAULT-SECRET-LEAK", REDACTOR_CONTRACT_PROFILE, exc)
        )

    try:
        raised = redacted_or_none(payload, policy=policy, redactor=_FailingRedactor())
        survived = redacted_or_none(payload, policy=policy, redactor=factory())
        outcomes.append(
            outcome_from_observations(
                "REDACTOR-03-FAILURE-IS-CONTAINED",
                REDACTOR_CONTRACT_PROFILE,
                (
                    # A raising redactor yields nothing. Falling back to the raw value would turn a
                    # redaction failure into a disclosure -- the opposite of what was asked for.
                    observation("failure_yields_nothing", expected=True, actual=raised is None),
                    observation("failure_does_not_propagate", expected=True, actual=True),
                    # ``None`` has to mean failure, not "redacted to empty", or the caller cannot
                    # tell a downgrade from empty content.
                    observation(
                        "success_is_distinguishable", expected=True, actual=survived is not None
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("REDACTOR-03-FAILURE-IS-CONTAINED", REDACTOR_CONTRACT_PROFILE, exc))

    try:
        mapping_result = factory().redact(payload, policy=policy)
        text_result = factory().redact("a sk-abc123 line", policy=policy)
        outcomes.append(
            outcome_from_observations(
                "REDACTOR-04-PRESERVES-THE-VALUE-SHAPE",
                REDACTOR_CONTRACT_PROFILE,
                (
                    # "Mask the whole payload" is a tempting one-liner that satisfies every leak rule
                    # and then hands the pipeline a scalar where it needs fields. The pipeline itself
                    # fails closed on this, but a redactor that trips it silently loses its consumer's
                    # content, so the contract names it rather than leaving it to be discovered.
                    observation(
                        "mapping_stays_a_mapping",
                        expected=True,
                        actual=isinstance(mapping_result, Mapping),
                    ),
                    observation(
                        "text_stays_text", expected=True, actual=isinstance(text_result, str)
                    ),
                    observation(
                        "mapping_keys_are_preserved",
                        expected=sorted(payload),
                        actual=sorted(mapping_result)
                        if isinstance(mapping_result, Mapping)
                        else None,
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(
            _error("REDACTOR-04-PRESERVES-THE-VALUE-SHAPE", REDACTOR_CONTRACT_PROFILE, exc)
        )

    try:
        redacted = factory().redact(payload, policy=policy)
        free_text = factory().redact("a sk-abc123 line with hunter2 in it", policy=policy)
        rendered = json.dumps(_jsonish({"payload": redacted, "text": free_text}))
        outcomes.append(
            outcome_from_observations(
                "REDACTOR-05-NO-POLICY-TEXT-LEAK",
                REDACTOR_CONTRACT_PROFILE,
                (
                    # The other axes of a RedactionPolicy. Key names are no help in a paragraph, and
                    # model *output* is all paragraph, so a redactor that masks only secret-named keys
                    # protects the request side and leaks the response side entirely.
                    observation(
                        "pattern_in_a_mapping_value", expected=False, actual="sk-abc123" in rendered
                    ),
                    observation(
                        "literal_in_a_mapping_value", expected=False, actual="hunter2" in rendered
                    ),
                    # Inside a list, where a recursive implementation can easily stop descending.
                    observation(
                        "pattern_inside_a_list", expected=False, actual="sk-xyz789" in rendered
                    ),
                    # And on a bare string, the shape a final_text capture actually has.
                    observation(
                        "pattern_in_free_text", expected=False, actual="sk-abc123" in str(free_text)
                    ),
                    observation(
                        "literal_in_free_text", expected=False, actual="hunter2" in str(free_text)
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("REDACTOR-05-NO-POLICY-TEXT-LEAK", REDACTOR_CONTRACT_PROFILE, exc))

    return tuple(outcomes)


def run_model_io_observer_contract(
    factory: ModelIOObserverFactory,
) -> tuple[ConformanceRuleOutcome, ...]:
    """Execute the guarantees the capture pipeline gives a `ModelIOObserver`.

    The rules are pipeline guarantees rather than observer obligations, so each one drives the
    factory's observer through `dispatch_model_call` for tolerance and puts a recording observer of
    our own alongside it to witness what the capture actually held. An opaque implementation cannot
    report what it received, and asking it to would make the suite test the reporting rather than the
    contract.
    """

    outcomes: list[ConformanceRuleOutcome] = []
    content = {"final_text": "settled output", "api_key": "sk-must-not-survive"}
    receipt = ModelCallReceipt()
    # Every observer this suite constructs, so a factory returning an exporter that owns a file, thread
    # or network client has all of them released -- not just the first rule's. A public conformance
    # suite gets run repeatedly, which is exactly where a per-run leak accumulates.
    subjects: list[ModelIOObserver] = []

    def new_subject() -> ModelIOObserver:
        subject = factory()
        subjects.append(subject)
        return subject

    try:
        witness = _RecordingObserver()
        subject = new_subject()
        returned = dispatch_model_call(
            receipt=receipt,
            content=content,
            subscriptions=(
                ModelIOSubscription(subject, CapturePolicy(mode="full")),
                ModelIOSubscription(witness, CapturePolicy(mode="full")),
            ),
        )
        outcomes.append(
            outcome_from_observations(
                "MODELIO-01-PARTIAL-IMPLEMENTATION-LEGAL",
                MODEL_IO_CONTRACT_PROFILE,
                (
                    # An observer that declares only ``on_model_call`` is a complete implementation.
                    observation(
                        "declares_on_model_call",
                        expected=True,
                        actual=callable(getattr(subject, "on_model_call", None)),
                    ),
                    observation("close_is_optional", expected=True, actual=True),
                    observation(
                        "delivery_reached_a_peer_observer", expected=1, actual=len(witness.captures)
                    ),
                    observation("receipt_returned", expected=True, actual=returned is not None),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(
            _error("MODELIO-01-PARTIAL-IMPLEMENTATION-LEGAL", MODEL_IO_CONTRACT_PROFILE, exc)
        )

    try:
        witness = _RecordingObserver()
        returned = dispatch_model_call(
            receipt=receipt,
            content=content,
            subscriptions=(
                ModelIOSubscription(_RaisingObserver(), CapturePolicy(mode="full")),
                ModelIOSubscription(new_subject(), CapturePolicy(mode="full")),
                ModelIOSubscription(witness, CapturePolicy(mode="full")),
            ),
        )
        outcomes.append(
            outcome_from_observations(
                "MODELIO-02-OBSERVER-FAILURE-CONTAINED",
                MODEL_IO_CONTRACT_PROFILE,
                (
                    # The call already happened and the provider has already been paid; a broken
                    # exporter does not get to undo that, nor to starve the observers behind it.
                    observation("dispatch_did_not_raise", expected=True, actual=True),
                    observation(
                        "later_observers_still_ran", expected=1, actual=len(witness.captures)
                    ),
                    observation(
                        "receipt_still_returned", expected=True, actual=returned is not None
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(
            _error("MODELIO-02-OBSERVER-FAILURE-CONTAINED", MODEL_IO_CONTRACT_PROFILE, exc)
        )

    try:
        witness = _RecordingObserver()
        dispatch_model_call(
            receipt=receipt,
            content=content,
            subscriptions=(
                ModelIOSubscription(new_subject(), CapturePolicy(mode="none")),
                ModelIOSubscription(witness, CapturePolicy(mode="none")),
            ),
        )
        captured = witness.captures[0]
        rendered = json.dumps(
            _jsonish({"content": captured.content, "digests": dict(captured.digests)})
        )
        outcomes.append(
            outcome_from_observations(
                "MODELIO-03-NONE-POLICY-RECEIVES-NO-CONTENT",
                MODEL_IO_CONTRACT_PROFILE,
                (
                    observation("mode", expected="none", actual=captured.mode),
                    observation("content_absent", expected=True, actual=captured.content is None),
                    # Not even a digest: ``none`` means the consumer learns nothing about the content,
                    # and a digest of a short prompt is a guessable one.
                    observation("digests_absent", expected=0, actual=len(captured.digests)),
                    observation("lengths_absent", expected=0, actual=len(captured.lengths)),
                    observation(
                        "nothing_leaked", expected=False, actual="sk-must-not-survive" in rendered
                    ),
                    # The receipt still arrives: it is metadata only, so it is safe at every mode.
                    observation(
                        "receipt_still_delivered",
                        expected=True,
                        actual=captured.receipt is not None,
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(
            _error("MODELIO-03-NONE-POLICY-RECEIVES-NO-CONTENT", MODEL_IO_CONTRACT_PROFILE, exc)
        )

    # ``close_model_io_subscriptions`` de-duplicates by identity and swallows failures, so a factory
    # returning one shared instance is closed once and a raising ``close`` cannot lose the outcomes.
    close_model_io_subscriptions(
        tuple(ModelIOSubscription(subject, CapturePolicy()) for subject in subjects)
    )
    return tuple(outcomes)


class _RecordingObserver:
    """Witnesses what a capture held, for rules an opaque implementation cannot report on."""

    def __init__(self) -> None:
        self.captures: list[ModelCallCapture] = []

    def on_model_call(self, capture: ModelCallCapture) -> None:
        self.captures.append(capture)


class _RaisingObserver:
    """An observer that always fails, for the containment rule."""

    def on_model_call(self, capture: ModelCallCapture) -> None:
        del capture
        raise RuntimeError("exporter unavailable")


class _FailingRedactor:
    """A redactor that always fails, for the fail-closed rule."""

    def redact(self, value: Any, *, policy: RedactionPolicy) -> Any:
        raise RuntimeError("redactor unavailable")


def _jsonish(value: Any) -> Any:
    """Coerce a redacted payload to JSON-safe types so it can be digested and searched."""
    if isinstance(value, Mapping):
        return {str(key): _jsonish(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonish(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _error(rule_id: str, profile_id: str, exc: Exception) -> ConformanceRuleOutcome:
    return ConformanceRuleOutcome(
        rule_id=rule_id,
        profile_id=profile_id,
        status="error",
        error=safe_exception_summary(exc),
    )
