from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from monoid_agent_kernel.conformance.contracts import (
    run_capability_broker_contract,
    run_checkpoint_store_contract,
    run_model_io_observer_contract,
    run_redactor_contract,
)
from monoid_agent_kernel.core.capability import AutoGrantBroker, CapabilityBroker
from monoid_agent_kernel.core.checkpoint import (
    CheckpointRecord,
    CheckpointStore,
    LocalFsCheckpointStore,
)
from monoid_agent_kernel.core.model_io import DefaultRedactor
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.capability import (
    DenyAllBroker,
    GatewayCapabilityBroker,
    HumanEscalationBroker,
)
from monoid_agent_kernel.reference.stores.sqlite import SqliteCheckpointStore


STORE_FACTORIES: tuple[Callable[[Path], CheckpointStore], ...] = (
    LocalFsCheckpointStore,
    lambda root: SqliteCheckpointStore(root / "contract.db"),
)
BROKER_FACTORIES: tuple[Callable[[], CapabilityBroker], ...] = (
    AutoGrantBroker,
    DenyAllBroker,
    HumanEscalationBroker,
    lambda: GatewayCapabilityBroker(
        token_manager=TokenManager.from_secret("x" * 32), tenant_id="tenant", user_id="user"
    ),
)


@pytest.mark.parametrize("factory", STORE_FACTORIES, ids=("localfs", "sqlite"))
def test_reusable_checkpoint_store_contract(
    factory: Callable[[Path], CheckpointStore], tmp_path: Path
) -> None:
    outcomes = run_checkpoint_store_contract(factory, tmp_path)

    assert [outcome.rule_id for outcome in outcomes] == [
        "STORE-01-MONOTONIC-PUBLICATION",
        "STORE-02-CONTENT-ADDRESSED-BLOB",
        "STORE-03-RUN-ISOLATION",
    ]
    assert all(outcome.status == "passed" for outcome in outcomes)


def test_checkpoint_store_contract_rejects_a_process_local_store(tmp_path: Path) -> None:
    class VolatileStore:
        def __init__(self) -> None:
            self.records: dict[str, CheckpointRecord] = {}
            self.blobs: dict[tuple[str, str], bytes] = {}

        def put(self, checkpoint, blobs=None) -> None:  # noqa: ANN001
            current = self.records.get(checkpoint.run_id)
            if current is None or checkpoint.seq > current.seq:
                self.records[checkpoint.run_id] = CheckpointRecord(
                    seq=checkpoint.seq,
                    checkpoint=checkpoint,
                )
            for digest, data in (blobs or {}).items():
                self.blobs[(checkpoint.run_id, digest)] = data

        def latest(self, run_id: str) -> CheckpointRecord | None:
            return self.records.get(run_id)

        def delete(self, run_id: str) -> None:
            self.records.pop(run_id, None)

        def put_blob(self, run_id: str, data: bytes) -> str:
            digest = hashlib.sha256(data).hexdigest()
            self.blobs[(run_id, digest)] = data
            return digest

        def get_blob(self, run_id: str, digest: str) -> bytes:
            return self.blobs[(run_id, digest)]

    outcomes = run_checkpoint_store_contract(lambda _root: VolatileStore(), tmp_path)

    assert [outcome.status for outcome in outcomes] == ["failed", "error", "failed"]


@pytest.mark.parametrize(
    "factory", BROKER_FACTORIES, ids=("auto-grant", "deny-all", "pending", "gateway")
)
def test_reusable_capability_broker_contract(factory: Callable[[], CapabilityBroker]) -> None:
    outcomes = run_capability_broker_contract(factory)

    assert outcomes[0].status == "passed"
    assert outcomes[1].status in {"passed", "skipped"}


def test_broker_contract_reports_invalid_result_without_raising() -> None:
    class InvalidBroker:
        def request(self, request: object) -> None:
            del request
            return None

    outcomes = run_capability_broker_contract(InvalidBroker)  # type: ignore[arg-type]

    assert [outcome.status for outcome in outcomes] == ["failed", "skipped"]
    assert outcomes[0].observations[0].observation_id == "grant_union"
    assert outcomes[0].observations[0].actual is False
    assert outcomes[1].error == "broker returned an invalid outcome"


def test_contract_suites_are_reachable_through_the_public_conformance_package() -> None:
    """An external implementer imports `monoid_agent_kernel.conformance`, not the private module
    path, and depends on the rule ids being stable and ordered. Assert the surface they actually use,
    including the rule-id sequence a report is keyed by."""
    import monoid_agent_kernel.conformance as conformance

    assert conformance.run_redactor_contract is run_redactor_contract
    assert conformance.run_model_io_observer_contract is run_model_io_observer_contract
    assert {"RedactorFactory", "ModelIOObserverFactory"} <= set(conformance.__all__)

    outcomes = conformance.run_redactor_contract(DefaultRedactor)

    assert [outcome.rule_id for outcome in outcomes] == [
        "REDACTOR-01-DETERMINISTIC",
        "REDACTOR-02-NO-DEFAULT-SECRET-LEAK",
        "REDACTOR-03-FAILURE-IS-CONTAINED",
    ]
    assert all(outcome.profile_id == "redactor-contract" for outcome in outcomes)
    assert all(outcome.status == "passed" for outcome in outcomes)

    class Observer:
        def on_model_call(self, capture: object) -> None:
            del capture

    observer_outcomes = conformance.run_model_io_observer_contract(Observer)

    assert [outcome.rule_id for outcome in observer_outcomes] == [
        "MODELIO-01-PARTIAL-IMPLEMENTATION-LEGAL",
        "MODELIO-02-OBSERVER-FAILURE-CONTAINED",
        "MODELIO-03-NONE-POLICY-RECEIVES-NO-CONTENT",
    ]
    assert all(outcome.profile_id == "model-io-observer-contract" for outcome in observer_outcomes)
    assert all(outcome.status == "passed" for outcome in observer_outcomes)


def test_redactor_contract_redacts_exception_details() -> None:
    secret = "redactor-secret-must-not-enter-report"

    class FailingRedactor:
        def redact(self, value: object, *, policy: object) -> object:
            del value, policy
            raise RuntimeError(secret)

    outcomes = run_redactor_contract(FailingRedactor)  # type: ignore[arg-type]

    assert outcomes[0].status == "error"
    assert outcomes[0].error == "RuntimeError: details redacted"
    assert all(secret not in str(outcome.to_json()) for outcome in outcomes)


def test_broker_contract_redacts_exception_details() -> None:
    secret = "broker-secret-must-not-enter-report"

    class FailingBroker:
        def request(self, request: object) -> None:
            del request
            raise RuntimeError(secret)

    outcomes = run_capability_broker_contract(FailingBroker)  # type: ignore[arg-type]

    assert outcomes[0].status == "error"
    assert outcomes[0].error == "RuntimeError: details redacted"
    assert secret not in str(outcomes[0].to_json())
