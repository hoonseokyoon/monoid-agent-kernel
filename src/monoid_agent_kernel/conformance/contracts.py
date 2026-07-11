"""Reusable implementation contracts for checkpoint stores and capability brokers."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from monoid_agent_kernel.conformance.report import (
    ConformanceRuleOutcome,
    observation,
    outcome_from_observations,
    safe_exception_summary,
)
from monoid_agent_kernel.core.capability import (
    CapabilityBroker,
    CapabilityDenial,
    CapabilityLease,
    CapabilityPending,
    CapabilityRequest,
    scope_within,
)
from monoid_agent_kernel.core.checkpoint import CheckpointStore, RunCheckpoint, load_latest_checked

STORE_CONTRACT_PROFILE = "checkpoint-store-contract"
BROKER_CONTRACT_PROFILE = "capability-broker-contract"


class CheckpointStoreFactory(Protocol):
    def __call__(self, root: Path) -> CheckpointStore: ...


class CapabilityBrokerFactory(Protocol):
    def __call__(self) -> CapabilityBroker: ...


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


def _error(rule_id: str, profile_id: str, exc: Exception) -> ConformanceRuleOutcome:
    return ConformanceRuleOutcome(
        rule_id=rule_id,
        profile_id=profile_id,
        status="error",
        error=safe_exception_summary(exc),
    )
