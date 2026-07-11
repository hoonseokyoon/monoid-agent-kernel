from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from monoid_agent_kernel.conformance.contracts import (
    run_capability_broker_contract,
    run_checkpoint_store_contract,
)
from monoid_agent_kernel.core.capability import AutoGrantBroker, CapabilityBroker
from monoid_agent_kernel.core.checkpoint import CheckpointStore, LocalFsCheckpointStore
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


@pytest.mark.parametrize(
    "factory", BROKER_FACTORIES, ids=("auto-grant", "deny-all", "pending", "gateway")
)
def test_reusable_capability_broker_contract(factory: Callable[[], CapabilityBroker]) -> None:
    outcomes = run_capability_broker_contract(factory)

    assert outcomes[0].status == "passed"
    assert outcomes[1].status in {"passed", "skipped"}
