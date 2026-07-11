from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from monoid_agent_kernel.core.control import ControlResult
from monoid_agent_kernel.reference.command_inbox import (
    CommandPrincipal,
    CommandQueueFull,
    CommandStore,
    InMemoryCommandStore,
    SqliteCommandStore,
    StoredCommand,
)


def _memory(root: Path) -> CommandStore:
    del root
    return InMemoryCommandStore()


def _sqlite(root: Path) -> CommandStore:
    return SqliteCommandStore(root / "commands.db")


@pytest.fixture(params=(_memory, _sqlite), ids=("memory", "sqlite"))
def store(request: pytest.FixtureRequest, tmp_path: Path) -> CommandStore:
    factory: Callable[[Path], CommandStore] = request.param
    return factory(tmp_path)


def _command(command_id: str, *, created_at: float = 1.0) -> StoredCommand:
    return StoredCommand(
        run_id="run_1",
        command_id=command_id,
        type="status",
        args={},
        principal=CommandPrincipal("tenant", "user", "operator"),
        created_at=created_at,
    )


def test_append_is_idempotent_and_claims_in_order(store: CommandStore) -> None:
    first = store.append(_command("cmd_1", created_at=1.0), max_pending=10)
    duplicate = store.append(_command("cmd_1", created_at=99.0), max_pending=10)
    store.append(_command("cmd_2", created_at=2.0), max_pending=10)

    assert duplicate.command_id == first.command_id
    persisted = store.read_command("run_1", "cmd_1")
    assert persisted is not None and persisted.created_at == 1.0
    claimed = store.claim("run_1", "worker", claim_ttl_s=30)
    assert claimed is not None and claimed.command_id == "cmd_1"
    result = ControlResult(run_id="run_1", type="status", status="ok", data={"state": "running"})
    receipt = store.acknowledge("run_1", "cmd_1", "worker", result)
    assert receipt.status == "completed"
    assert receipt.result is not None and receipt.result["data"]["state"] == "running"
    assert store.claim("run_1", "worker", claim_ttl_s=30).command_id == "cmd_2"  # type: ignore[union-attr]


def test_stale_claim_is_recoverable_and_wrong_worker_cannot_ack(store: CommandStore) -> None:
    store.append(_command("cmd_stale"), max_pending=10)
    assert store.claim("run_1", "crashed", claim_ttl_s=30) is not None
    reclaimed = store.claim("run_1", "recovery", claim_ttl_s=-1)
    assert reclaimed is not None and reclaimed.claimed_by == "recovery"
    with pytest.raises(RuntimeError, match="not claimed"):
        store.acknowledge(
            "run_1",
            "cmd_stale",
            "crashed",
            ControlResult(run_id="run_1", type="status", status="ok"),
        )


def test_stale_claim_is_not_reclaimed_by_same_worker(store: CommandStore) -> None:
    store.append(_command("cmd_in_flight"), max_pending=10)
    assert store.claim("run_1", "worker", claim_ttl_s=30) is not None

    assert store.claim("run_1", "worker", claim_ttl_s=-1) is None
    recovered = store.claim("run_1", "replacement", claim_ttl_s=-1)
    assert recovered is not None and recovered.claimed_by == "replacement"


def test_queue_limit_counts_only_unacknowledged_commands(store: CommandStore) -> None:
    store.append(_command("cmd_1"), max_pending=1)
    with pytest.raises(CommandQueueFull):
        store.append(_command("cmd_2"), max_pending=1)
    claimed = store.claim("run_1", "worker", claim_ttl_s=30)
    assert claimed is not None
    store.acknowledge(
        "run_1",
        "cmd_1",
        "worker",
        ControlResult(run_id="run_1", type="status", status="ok"),
    )
    assert store.append(_command("cmd_2"), max_pending=1).status == "pending"


def test_persisted_command_redacts_credential_shaped_fields(store: CommandStore) -> None:
    command = StoredCommand(
        run_id="run_1",
        command_id="cmd_secret",
        type="status",
        args={
            "access_token": "bearer-secret",
            "accessToken": "camel-bearer-secret",
            "apiKey": "camel-api-key",
            "client_secret": "oauth-client-secret",
            "secret_key": "signing-secret",
            "nested": {
                "password": "do-not-store",
                "safe": "visible",
                "token_ref": "capability-handle",
                "callbackToken": "camel-callback-secret",
            },
        },
        principal=CommandPrincipal("tenant", "user", "operator"),
    )
    store.append(command, max_pending=10)
    claimed = store.claim("run_1", "worker", claim_ttl_s=30)

    assert claimed is not None
    assert claimed.args["access_token"] == "[redacted]"
    assert claimed.args["accessToken"] == "[redacted]"
    assert claimed.args["apiKey"] == "[redacted]"
    assert claimed.args["client_secret"] == "[redacted]"
    assert claimed.args["secret_key"] == "[redacted]"
    assert claimed.args["nested"] == {
        "password": "[redacted]",
        "safe": "visible",
        "token_ref": "capability-handle",
        "callbackToken": "[redacted]",
    }
    receipt = store.acknowledge(
        "run_1",
        "cmd_secret",
        "worker",
        ControlResult(
            run_id="run_1",
            type="status",
            status="ok",
            data={
                "refreshToken": "result-secret",
                "api_secret": "service-secret",
                "safe": "visible",
            },
        ),
    )
    assert receipt.result is not None
    assert receipt.result["data"] == {
        "refreshToken": "[redacted]",
        "api_secret": "[redacted]",
        "safe": "visible",
    }


def test_sqlite_cross_instance_claim_has_exactly_one_winner(tmp_path: Path) -> None:
    db = tmp_path / "shared.db"
    first = SqliteCommandStore(db)
    second = SqliteCommandStore(db)
    first.append(_command("cmd_race"), max_pending=10)
    barrier = Barrier(2)

    def claim(item: tuple[str, SqliteCommandStore]) -> tuple[str, StoredCommand | None]:
        worker, candidate = item
        barrier.wait()
        return worker, candidate.claim("run_1", worker, claim_ttl_s=30)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, (("worker_a", first), ("worker_b", second))))

    winners = [(worker, command) for worker, command in results if command is not None]
    assert len(winners) == 1
    assert winners[0][1].claimed_by == winners[0][0]  # type: ignore[union-attr]
