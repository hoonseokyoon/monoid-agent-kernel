from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from monoid_agent_kernel.core.control import ControlResult
from monoid_agent_kernel.reference.command_inbox import (
    CommandConflict,
    CommandPrincipal,
    CommandQueueFull,
    CommandStore,
    InMemoryCommandStore,
    SqliteCommandStore,
    StoredCommand,
    redact_command_credential,
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


def test_bearer_redaction_covers_nested_keys_and_values() -> None:
    credential = "signed-bearer-value"
    redacted = redact_command_credential(
        {
            credential: "marker",
            "nested": {f"prefix-{credential}": credential},
            "tuple": ("safe", credential),
        },
        credential,
    )

    assert credential not in str(redacted)
    assert redacted == {
        "[redacted]": "marker",
        "nested": {"prefix-[redacted]": "[redacted]"},
        "tuple": ["safe", "[redacted]"],
    }


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


def test_duplicate_id_rejects_a_different_authenticated_command(store: CommandStore) -> None:
    store.append(_command("cmd_duplicate"), max_pending=10)
    different = StoredCommand(
        run_id="run_1",
        command_id="cmd_duplicate",
        type="report_task_result",
        args={"task_id": "task_other"},
        principal=CommandPrincipal("tenant", "user", "callback-worker"),
    )

    with pytest.raises(CommandConflict, match="already belongs"):
        store.append(different, max_pending=10)


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


def test_claim_does_not_skip_in_flight_head_command(store: CommandStore) -> None:
    store.append(_command("cmd_first", created_at=2.0), max_pending=10)
    store.append(_command("cmd_second", created_at=1.0), max_pending=10)
    claimed = store.claim("run_1", "worker_a", claim_ttl_s=30)
    assert claimed is not None and claimed.command_id == "cmd_first"

    assert store.claim("run_1", "worker_a", claim_ttl_s=30) is None
    assert store.claim("run_1", "worker_b", claim_ttl_s=30) is None
    recovered = store.claim("run_1", "worker_b", claim_ttl_s=-1)
    assert recovered is not None and recovered.command_id == "cmd_first"


def test_claim_command_can_reserve_recovery_out_of_order(store: CommandStore) -> None:
    store.append(_command("cmd_older"), max_pending=10)
    store.append(_command("cmd_resume"), max_pending=10)

    reserved = store.claim_command(
        "run_1", "cmd_resume", "recovery-worker", claim_ttl_s=30
    )
    assert reserved is not None and reserved.command_id == "cmd_resume"
    store.acknowledge(
        "run_1",
        "cmd_resume",
        "recovery-worker",
        ControlResult(run_id="run_1", type="status", status="ok"),
    )
    head = store.claim("run_1", "ordinary-worker", claim_ttl_s=30)
    assert head is not None and head.command_id == "cmd_older"


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


def test_resume_recovery_reservation_bypasses_a_full_queue(store: CommandStore) -> None:
    store.append(_command("cmd_head"), max_pending=1)
    resume = StoredCommand(
        run_id="run_1",
        command_id="cmd_resume",
        type="resume",
        args={},
        principal=CommandPrincipal("tenant", "user", "operator"),
    )

    reserved = store.append(resume, max_pending=1, recovery_reservation=True)

    assert reserved.status == "pending"
    with pytest.raises(CommandQueueFull):
        store.append(_command("cmd_still_full"), max_pending=1)
    claimed = store.claim_command(
        "run_1", "cmd_resume", "recovery-worker", claim_ttl_s=30
    )
    assert claimed is not None and claimed.type == "resume"


def test_recovery_reservation_rejects_non_resume_commands(store: CommandStore) -> None:
    with pytest.raises(ValueError, match="restricted to resume"):
        store.append(_command("cmd_status"), max_pending=1, recovery_reservation=True)


def test_immediate_command_requires_an_empty_lane(store: CommandStore) -> None:
    store.append(_command("cmd_head"), max_pending=10)

    with pytest.raises(CommandQueueFull, match="lane is busy"):
        store.append(_command("cmd_secret"), max_pending=10, require_empty=True)


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
