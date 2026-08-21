"""Reusable implementation contracts for checkpoint stores and capability brokers."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from pathlib import Path
from threading import Barrier
from typing import Any, Callable, Protocol
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
from monoid_agent_kernel.core.checkpoint import CheckpointStore, RunCheckpoint, load_latest_checked
from monoid_agent_kernel.core.events import EVENT_SCHEMA_VERSION, AgentEvent
from monoid_agent_kernel.core.model_invocation import (
    ACCEPTED_MODEL_INVOCATION_SCHEMA_VERSIONS,
    ACCEPTED_MODEL_REQUEST_DIGEST_GENERATIONS,
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
_CONTRACT_INVOCATION_FIXED_CANONICAL_FIELDS = frozenset(
    {"schema_version", "digest_generation"}
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
        "applied_input_receipts": {
            "input-alternate": {"checkpoint_seq": checkpoint.seq}
        },
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


def _contract_event_identity_variants(event: AgentEvent) -> dict[str, AgentEvent]:
    """Vary every canonical non-key event field independently."""

    variants = {
        "event_id": replace(event, event_id="event-alternate"),
        "timestamp": replace(event, timestamp="2026-08-21T00:00:01Z"),
        "type": replace(event, type="run.started"),
        "level": replace(event, level="warning"),
        "data": replace(event, data={"checkpoint_seq": event.seq, "alternate": True}),
        "turn_id": replace(event, turn_id="turn-alternate"),
        "parent_id": replace(event, parent_id="event-parent-alternate"),
    }
    identity_fields = set(event.to_json()) - {"schema_version", "run_id", "seq"}
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
    retryable: bool = False,
    succeeded: bool = False,
) -> DurableModelInvocation:
    if idempotency_key is None:
        idempotency_key = f"contract-{hashlib.sha256(run_id.encode()).hexdigest()}"
    receipt = None
    result_ref = ""
    failure_code = ""
    if dispatch_state == "settled":
        receipt = {"request_digest": request_digest, "retryable": retryable}
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


def _contract_race(
    left: Callable[[], CommitResult],
    right: Callable[[], CommitResult],
) -> tuple[CommitResult, CommitResult]:
    barrier = Barrier(3)

    def invoke(operation: Callable[[], CommitResult]) -> CommitResult:
        barrier.wait(timeout=10)
        return operation()

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="fenced-contract") as executor:
        left_future = executor.submit(invoke, left)
        right_future = executor.submit(invoke, right)
        barrier.wait(timeout=10)
        return left_future.result(timeout=10), right_future.result(timeout=10)


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
    writer_token: WriterToken,
) -> CommitResult:
    if mutation == "checkpoint":
        return sink.commit_checkpoint(value, {}, writer_token=writer_token)
    if mutation == "event":
        return sink.append_event(value, writer_token=writer_token)
    if mutation == "invocation":
        return sink.commit_invocation(value, {}, writer_token=writer_token)
    return sink.settle_terminal(value, writer_token=writer_token)


def _contract_handoff_write(
    sink: FencedRunSink,
    writer_token: WriterToken,
    *,
    mutation: str,
    value: Any,
) -> CommitResult:
    return _contract_race_write(
        sink,
        mutation=mutation,
        value=value,
        writer_token=writer_token,
    )


def _contract_competing_values(mutation: str, run_id: str) -> tuple[Any, Any]:
    if mutation == "checkpoint":
        return (
            RunCheckpoint(run_id=run_id, seq=1, final_text="left"),
            RunCheckpoint(run_id=run_id, seq=1, final_text="right"),
        )
    if mutation == "event":
        return _contract_event(run_id, seq=1), _contract_event(run_id, seq=1, level="warning")
    if mutation == "invocation":
        return (
            _contract_invocation(run_id, revision=1, dispatch_state="reserved"),
            _contract_invocation(
                run_id,
                revision=1,
                dispatch_state="reserved",
                dispatch_id="dispatch-racer",
            ),
        )
    return _contract_terminal(run_id), _contract_terminal(run_id, failed=True)


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
    identity_fields = canonical_fields - {
        "run_id",
        "logical_call_id",
        "revision",
    } - _CONTRACT_INVOCATION_FIXED_CANONICAL_FIELDS
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


def _contract_invocation_canonical_alias_status(
    factory: FencedRunSinkHarnessFactory,
    run_id: str,
    field_name: str,
) -> str:
    """Prove accepted legacy tags normalize to the same current canonical record."""

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
    first = harness.sink.commit_invocation(baseline, {}, writer_token=token)
    if first.status != "committed":
        return f"setup:{first.status}"
    return harness.sink.commit_invocation(
        variants[field_name],
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
    retryable: bool = False,
    succeeded: bool = False,
) -> tuple[tuple[str, ...], str]:
    harness = factory()
    token = _contract_writer(harness, run_id)
    history = (
        _contract_invocation(run_id, revision=1, dispatch_state="reserved"),
        _contract_invocation(run_id, revision=2, dispatch_state="dispatch_started"),
        _contract_invocation(
            run_id,
            revision=3,
            dispatch_state=terminal_state,
            retryable=retryable,
            succeeded=succeeded,
        ),
    )
    history_statuses = tuple(
        harness.sink.commit_invocation(invocation, {}, writer_token=token).status
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


def run_fenced_run_sink_contract(
    factory: FencedRunSinkHarnessFactory,
) -> tuple[ConformanceRuleOutcome, ...]:
    """Execute backend-neutral fencing, conflict, terminal, and invocation invariants."""

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
        blob_bytes_conflict = harness.sink.commit_checkpoint(
            checkpoint,
            {_CONTRACT_CHECKPOINT_BLOB_SHA256: _CONTRACT_ALTERNATE_BLOB},
            writer_token=token,
        )
        conflict = harness.sink.commit_checkpoint(
            RunCheckpoint(run_id=run_id, seq=1, final_text="challenger"),
            {},
            writer_token=token,
        )
        checkpoint_identity_statuses = {
            field_name: harness.sink.commit_checkpoint(
                variant,
                checkpoint_blobs,
                writer_token=token,
            ).status
            for field_name, variant in _contract_checkpoint_identity_variants(
                checkpoint
            ).items()
        }
        loaded = harness.sink.latest_checked(run_id)

        monotonic_harness = factory()
        monotonic_run_id = _contract_run_id(namespace, "checkpoint-monotonic")
        monotonic_token = _contract_writer(monotonic_harness, monotonic_run_id)
        newer = monotonic_harness.sink.commit_checkpoint(
            RunCheckpoint(run_id=monotonic_run_id, seq=2, final_text="newer"),
            {},
            writer_token=monotonic_token,
        )
        monotonic_harness.sink.commit_checkpoint(
            RunCheckpoint(run_id=monotonic_run_id, seq=1, final_text="delayed"),
            {},
            writer_token=monotonic_token,
        )
        monotonic_harness = monotonic_harness.reopen()
        head_after_delayed = monotonic_harness.sink.latest_checked(monotonic_run_id)
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
                        "blob_key_conflict",
                        expected="conflict",
                        actual=blob_key_conflict.status,
                    ),
                    observation(
                        "blob_bytes_conflict",
                        expected="conflict",
                        actual=blob_bytes_conflict.status,
                    ),
                    observation("conflict_status", expected="conflict", actual=conflict.status),
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
                        actual=_contract_blob_hex(
                            loaded.value,
                            _CONTRACT_CHECKPOINT_BLOB_SHA256,
                        ),
                    ),
                    observation(
                        "newer_checkpoint",
                        expected="committed",
                        actual=newer.status,
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
        stale_checkpoint = harness.sink.commit_checkpoint(
            checkpoint, {}, writer_token=stale
        )
        stale_event = harness.sink.append_event(event, writer_token=stale)
        stale_invocation = harness.sink.commit_invocation(
            invocation,
            {},
            writer_token=stale,
        )
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

        fresh_harness = factory()
        fresh_run_id = _contract_run_id(namespace, "stale-new")
        fresh_stale = _contract_writer(fresh_harness, fresh_run_id)
        fresh_current = _contract_writer(
            fresh_harness,
            fresh_run_id,
            "owner-b",
            generation=2,
        )
        fresh_checkpoint = RunCheckpoint(run_id=fresh_run_id, seq=1)
        fresh_event = _contract_event(fresh_run_id, seq=1)
        fresh_invocation = _contract_invocation(
            fresh_run_id,
            revision=1,
            dispatch_state="reserved",
        )
        fresh_terminal = _contract_terminal(fresh_run_id)
        stale_new_checkpoint = fresh_harness.sink.commit_checkpoint(
            fresh_checkpoint,
            {},
            writer_token=fresh_stale,
        )
        stale_new_event = fresh_harness.sink.append_event(
            fresh_event,
            writer_token=fresh_stale,
        )
        stale_new_invocation = fresh_harness.sink.commit_invocation(
            fresh_invocation,
            {},
            writer_token=fresh_stale,
        )
        stale_new_terminal = fresh_harness.sink.settle_terminal(
            fresh_terminal,
            writer_token=fresh_stale,
        )
        current_after_stale_checkpoint = fresh_harness.sink.commit_checkpoint(
            fresh_checkpoint,
            {},
            writer_token=fresh_current,
        )
        current_after_stale_event = fresh_harness.sink.append_event(
            fresh_event,
            writer_token=fresh_current,
        )
        current_after_stale_invocation = fresh_harness.sink.commit_invocation(
            fresh_invocation,
            {},
            writer_token=fresh_current,
        )
        current_after_stale_terminal = fresh_harness.sink.settle_terminal(
            fresh_terminal,
            writer_token=fresh_current,
        )
        handoff_observations = []
        for mutation in ("checkpoint", "event", "invocation", "terminal"):
            handoff_harness = factory()
            handoff_run_id = _contract_run_id(namespace, f"handoff-{mutation}")
            handoff_stale = _contract_writer(handoff_harness, handoff_run_id)
            handoff_current = WriterToken(
                run_id=handoff_run_id,
                owner_id="owner-b",
                generation=2,
            )
            handoff_value, _ = _contract_competing_values(mutation, handoff_run_id)
            handoff_write = partial(
                _contract_handoff_write,
                mutation=mutation,
                value=handoff_value,
            )
            stale_result, current_result, rotation_first = (
                handoff_harness.race_writer_handoff(
                    mutation,
                    handoff_stale,
                    handoff_current,
                    handoff_write,
                )
            )
            expected_statuses = (
                ("fenced", "committed")
                if rotation_first
                else ("committed", "already_committed")
            )
            handoff_observations.append(
                observation(
                    f"handoff_{mutation}_linearization",
                    expected=expected_statuses,
                    actual=(stale_result.status, current_result.status),
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
                    observation(
                        "initial_event", expected="committed", actual=initial_event.status
                    ),
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
                    observation(
                        "stale_terminal", expected="fenced", actual=stale_terminal.status
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
                        "stale_new_checkpoint",
                        expected="fenced",
                        actual=stale_new_checkpoint.status,
                    ),
                    observation(
                        "stale_new_event",
                        expected="fenced",
                        actual=stale_new_event.status,
                    ),
                    observation(
                        "stale_new_invocation",
                        expected="fenced",
                        actual=stale_new_invocation.status,
                    ),
                    observation(
                        "stale_new_terminal",
                        expected="fenced",
                        actual=stale_new_terminal.status,
                    ),
                    observation(
                        "current_after_stale_checkpoint",
                        expected="committed",
                        actual=current_after_stale_checkpoint.status,
                    ),
                    observation(
                        "current_after_stale_event",
                        expected="committed",
                        actual=current_after_stale_event.status,
                    ),
                    observation(
                        "current_after_stale_invocation",
                        expected="committed",
                        actual=current_after_stale_invocation.status,
                    ),
                    observation(
                        "current_after_stale_terminal",
                        expected="committed",
                        actual=current_after_stale_terminal.status,
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
        harness = factory()
        run_id = _contract_run_id(namespace, "terminal")
        token = _contract_writer(harness, run_id)
        event = _contract_event(run_id, seq=1)
        first_event = harness.sink.append_event(event, writer_token=token)
        terminal = _contract_terminal(run_id)
        first_terminal = harness.sink.settle_terminal(terminal, writer_token=token)
        harness = harness.reopen()
        repeated_event = harness.sink.append_event(event, writer_token=token)
        conflict_event = harness.sink.append_event(
            _contract_event(run_id, seq=1, level="warning"), writer_token=token
        )
        event_identity_statuses = {
            field_name: harness.sink.append_event(
                variant,
                writer_token=token,
            ).status
            for field_name, variant in _contract_event_identity_variants(event).items()
        }
        repeated_terminal = harness.sink.settle_terminal(terminal, writer_token=token)
        conflict_terminal = harness.sink.settle_terminal(
            _contract_terminal(run_id, failed=True), writer_token=token
        )
        terminal_identity_statuses = {
            field_name: harness.sink.settle_terminal(
                variant,
                writer_token=token,
            ).status
            for field_name, variant in _contract_terminal_identity_variants(terminal).items()
        }

        race_observations = []
        for mutation in ("checkpoint", "event", "invocation", "terminal"):
            race_harness = factory()
            race_run_id = _contract_run_id(namespace, f"race-{mutation}")
            race_token = _contract_writer(race_harness, race_run_id)
            left_value, right_value = _contract_competing_values(mutation, race_run_id)
            left_write = partial(
                _contract_race_write,
                mutation=mutation,
                value=left_value,
                writer_token=race_token,
            )
            right_write = partial(
                _contract_race_write,
                mutation=mutation,
                value=right_value,
                writer_token=race_token,
            )
            left_harness = race_harness.reopen()
            right_harness = race_harness.reopen()
            left_result, right_result = _contract_race(
                partial(left_write, left_harness.sink),
                partial(right_write, right_harness.sink),
            )
            retry_winner, retry_loser = _contract_race_retry_statuses(
                left_result,
                right_result,
                race_harness.reopen(),
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
                        "terminal_conflict", expected="conflict", actual=conflict_terminal.status
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
        invocation_canonical_alias_statuses = {
            field_name: _contract_invocation_canonical_alias_status(
                factory,
                _contract_run_id(namespace, f"invocation-canonical-alias-{field_name}"),
                field_name,
            )
            for field_name in sorted(_CONTRACT_INVOCATION_FIXED_CANONICAL_FIELDS)
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
        conflicting_reserved = harness.sink.commit_invocation(
            _contract_invocation(
                run_id,
                revision=1,
                dispatch_state="reserved",
                dispatch_id="dispatch-conflict",
            ),
            {},
            writer_token=token,
        )
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
        result_blob_bytes_conflict = harness.sink.commit_invocation(
            settled_result_invocation,
            {_CONTRACT_INVOCATION_BLOB_SHA256: _CONTRACT_ALTERNATE_BLOB},
            writer_token=token,
        )
        loaded_result = harness.sink.load_invocation(run_id, result_call_id)
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
                            f"invocation_identity_{field_name}",
                            expected="conflict",
                            actual=status,
                        )
                        for field_name, status in invocation_identity_statuses.items()
                    ),
                    *(
                        observation(
                            f"invocation_canonical_alias_{field_name}",
                            expected="already_committed",
                            actual=status,
                        )
                        for field_name, status in invocation_canonical_alias_statuses.items()
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
                    observation(
                        "proven_retry", expected="committed", actual=next_attempt.status
                    ),
                    *(
                        observation(
                            f"old_revision_{revision}_retry_and_head",
                            expected=("already_committed", 4),
                            actual=status_and_head,
                        )
                        for revision, status_and_head in old_revision_retries.items()
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
                        actual=(
                            loaded.value.invocation.logical_call_id if loaded.value else None
                        ),
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
                        "result_blob_bytes_conflict",
                        expected="conflict",
                        actual=result_blob_bytes_conflict.status,
                    ),
                    observation(
                        "result_load", expected="loaded", actual=loaded_result.status
                    ),
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
                        actual=_contract_blob_hex(
                            loaded_result.value,
                            _CONTRACT_INVOCATION_BLOB_SHA256,
                        ),
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
                _contract_run_id(namespace, f"invocation-first-{state.replace('_', '-') }"),
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
        unknown_history, after_unknown = _contract_retry_after_terminal_invocation(
            factory,
            _contract_run_id(namespace, "invocation-after-unknown"),
            terminal_state="unknown",
        )
        success_history, after_success = _contract_retry_after_terminal_invocation(
            factory,
            _contract_run_id(namespace, "invocation-after-success"),
            terminal_state="settled",
            succeeded=True,
        )
        nonretry_history, after_nonretry_failure = _contract_retry_after_terminal_invocation(
            factory,
            _contract_run_id(namespace, "invocation-after-nonretry-failure"),
            terminal_state="settled",
        )
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
                            f"identity_drift_{field_name}",
                            expected="conflict",
                            actual=status,
                        )
                        for field_name, status in identity_drift_statuses.items()
                    ),
                    observation(
                        "unknown_history",
                        expected=("committed", "committed", "committed"),
                        actual=unknown_history,
                    ),
                    observation(
                        "retry_after_unknown", expected="conflict", actual=after_unknown
                    ),
                    observation(
                        "success_history",
                        expected=("committed", "committed", "committed"),
                        actual=success_history,
                    ),
                    observation(
                        "retry_after_success", expected="conflict", actual=after_success
                    ),
                    observation(
                        "nonretry_failure_history",
                        expected=("committed", "committed", "committed"),
                        actual=nonretry_history,
                    ),
                    observation(
                        "retry_after_nonretry_failure",
                        expected="conflict",
                        actual=after_nonretry_failure,
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
        checkpoint = RunCheckpoint(run_id=run_b_id, seq=1)
        event = _contract_event(run_b_id, seq=1)
        invocation = _contract_invocation(
            run_b_id, revision=1, dispatch_state="reserved"
        )
        terminal = _contract_terminal(run_b_id)
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
        swapped_terminal = harness.sink.settle_terminal(
            terminal,
            writer_token=run_a_token,
        )
        authorized_checkpoint = harness.sink.commit_checkpoint(
            checkpoint,
            {},
            writer_token=run_b_token,
        )
        authorized_event = harness.sink.append_event(event, writer_token=run_b_token)
        authorized_invocation = harness.sink.commit_invocation(
            invocation,
            {},
            writer_token=run_b_token,
        )
        authorized_terminal = harness.sink.settle_terminal(
            terminal,
            writer_token=run_b_token,
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
                    observation(
                        "cross_run_event", expected="fenced", actual=swapped_event.status
                    ),
                    observation(
                        "cross_run_invocation",
                        expected="fenced",
                        actual=swapped_invocation.status,
                    ),
                    observation(
                        "cross_run_terminal",
                        expected="fenced",
                        actual=swapped_terminal.status,
                    ),
                    observation(
                        "run_bound_checkpoint",
                        expected="committed",
                        actual=authorized_checkpoint.status,
                    ),
                    observation(
                        "run_bound_event",
                        expected="committed",
                        actual=authorized_event.status,
                    ),
                    observation(
                        "run_bound_invocation",
                        expected="committed",
                        actual=authorized_invocation.status,
                    ),
                    observation(
                        "run_bound_terminal",
                        expected="committed",
                        actual=authorized_terminal.status,
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
                    observation("top_level_secret_key", expected=False, actual="sk-live-must-not-survive" in rendered),
                    observation("nested_secret_key", expected=False, actual="Bearer must-not-survive" in rendered),
                    # Non-secret data must survive, or "redact everything" would pass every rule.
                    observation("non_secret_value_survives", expected=True, actual="7" in rendered),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("REDACTOR-02-NO-DEFAULT-SECRET-LEAK", REDACTOR_CONTRACT_PROFILE, exc))

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
                    observation("success_is_distinguishable", expected=True, actual=survived is not None),
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
                    observation("mapping_stays_a_mapping", expected=True, actual=isinstance(mapping_result, Mapping)),
                    observation("text_stays_text", expected=True, actual=isinstance(text_result, str)),
                    observation(
                        "mapping_keys_are_preserved",
                        expected=sorted(payload),
                        actual=sorted(mapping_result) if isinstance(mapping_result, Mapping) else None,
                    ),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("REDACTOR-04-PRESERVES-THE-VALUE-SHAPE", REDACTOR_CONTRACT_PROFILE, exc))

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
                    observation("pattern_in_a_mapping_value", expected=False, actual="sk-abc123" in rendered),
                    observation("literal_in_a_mapping_value", expected=False, actual="hunter2" in rendered),
                    # Inside a list, where a recursive implementation can easily stop descending.
                    observation("pattern_inside_a_list", expected=False, actual="sk-xyz789" in rendered),
                    # And on a bare string, the shape a final_text capture actually has.
                    observation("pattern_in_free_text", expected=False, actual="sk-abc123" in str(free_text)),
                    observation("literal_in_free_text", expected=False, actual="hunter2" in str(free_text)),
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
                    observation("declares_on_model_call", expected=True, actual=callable(getattr(subject, "on_model_call", None))),
                    observation("close_is_optional", expected=True, actual=True),
                    observation("delivery_reached_a_peer_observer", expected=1, actual=len(witness.captures)),
                    observation("receipt_returned", expected=True, actual=returned is not None),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("MODELIO-01-PARTIAL-IMPLEMENTATION-LEGAL", MODEL_IO_CONTRACT_PROFILE, exc))

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
                    observation("later_observers_still_ran", expected=1, actual=len(witness.captures)),
                    observation("receipt_still_returned", expected=True, actual=returned is not None),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("MODELIO-02-OBSERVER-FAILURE-CONTAINED", MODEL_IO_CONTRACT_PROFILE, exc))

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
        rendered = json.dumps(_jsonish({"content": captured.content, "digests": dict(captured.digests)}))
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
                    observation("nothing_leaked", expected=False, actual="sk-must-not-survive" in rendered),
                    # The receipt still arrives: it is metadata only, so it is safe at every mode.
                    observation("receipt_still_delivered", expected=True, actual=captured.receipt is not None),
                ),
            )
        )
    except Exception as exc:
        outcomes.append(_error("MODELIO-03-NONE-POLICY-RECEIVES-NO-CONTENT", MODEL_IO_CONTRACT_PROFILE, exc))

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
