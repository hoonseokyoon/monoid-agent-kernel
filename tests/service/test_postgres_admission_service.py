from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.serial, pytest.mark.service]


def _selected() -> bool:
    return os.environ.get("MONOID_SERVICE_PROFILE") in {"postgres", "objectstore", "combined"}


if not _selected():
    pytest.skip("PostgreSQL service profile is not selected", allow_module_level=True)

from psycopg import sql  # noqa: E402

from monoid_agent_kernel.adapters.postgres import (  # noqa: E402
    PostgresCommandAdmissionStore,
    PostgresConfig,
    PostgresDatabase,
    PostgresFencedRunSink,
    PostgresMigrations,
    PostgresWriterAuthorityStore,
)
from monoid_agent_kernel.core._util import canonical_sha256  # noqa: E402
from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore  # noqa: E402
from monoid_agent_kernel.core.outcome import (  # noqa: E402
    RetryEligibility,
    TerminalOutcome,
)
from monoid_agent_kernel.core.spec import AgentRunSpec  # noqa: E402
from monoid_agent_kernel.errors import ModelAdapterError  # noqa: E402
from monoid_agent_kernel.hosting import (  # noqa: E402
    ActivationBindingConflict,
    ActivationCommand,
    ActivationDriver,
    ActivationRuntime,
    AdmittedCommand,
    AdmissionConflict,
    AdmissionRequest,
    AdmissionRunTerminal,
    CommandOutboxDispatcher,
    DispatchClaimLost,
    DispatchResult,
    WriterToken,
)
from monoid_agent_kernel.loop import AgentLoop  # noqa: E402
from monoid_agent_kernel.providers.base import ModelTurn  # noqa: E402
from support.runtime import runtime_config, runtime_provider  # noqa: E402


_POSTGRES_TARGETS = [
    ("MONOID_POSTGRES16_DSN", 16, {"postgres", "objectstore", "combined"}),
    ("MONOID_POSTGRES18_DSN", 18, {"combined"}),
]


class _RetryableAdapter:
    def next_turn(self, request: Any) -> ModelTurn:
        del request
        raise ModelAdapterError(
            "private provider failure",
            error_code="provider_unavailable",
            retryable=True,
        )


@dataclass
class _CountingAdapter:
    calls: int = 0

    def next_turn(self, request: Any) -> ModelTurn:
        del request
        self.calls += 1
        return ModelTurn(final_text="private completion")


class _ForbiddenAdapter:
    def next_turn(self, request: Any) -> ModelTurn:
        del request
        raise AssertionError("duplicate activation must not call the provider")


@pytest.fixture(
    params=[
        pytest.param(_POSTGRES_TARGETS[0], id="postgres16"),
        pytest.param(_POSTGRES_TARGETS[1], id="postgres18"),
    ]
)
def postgres_target(request: pytest.FixtureRequest) -> tuple[str, int]:
    dsn_variable, expected_major, profiles = request.param
    if os.environ["MONOID_SERVICE_PROFILE"] not in profiles:
        pytest.skip(f"PostgreSQL {expected_major} is outside the selected profile")
    dsn = os.environ.get(dsn_variable)
    if not dsn:
        pytest.fail(f"{dsn_variable} is required for the selected service profile")
    return dsn, expected_major


@dataclass
class _Harness:
    database: PostgresDatabase
    authority: PostgresWriterAuthorityStore
    sink: PostgresFencedRunSink
    admission: PostgresCommandAdmissionStore

    def seed(self, tmp_path: Path, run_id: str) -> tuple[WriterToken, AgentRunSpec]:
        spec = AgentRunSpec(
            run_id=run_id,
            workspace_root=tmp_path / f"workspace-{run_id}",
            run_root=tmp_path / f"source-{run_id}",
        )
        spec.workspace_root.mkdir(parents=True)
        source_store = LocalFsCheckpointStore(spec.run_root)
        source = AgentLoop(
            spec=spec,
            model_adapter=_RetryableAdapter(),
            runtime_config_provider=runtime_provider(runtime_config("fs.write")),
            checkpoint_store=source_store,
        )
        source.open()
        assert source.run_until_suspended("resume me").reason == "turn_failed"
        source.release_parked()
        record = source_store.latest(run_id)
        assert record is not None

        token = self.authority.claim(
            run_id,
            "activation-worker",
            timedelta(minutes=5),
        ).writer_token
        assert self.sink.commit_checkpoint(record.checkpoint, {}, writer_token=token).status in {
            "committed",
            "already_committed",
        }
        return token, spec


@pytest.fixture
def harness(postgres_target: tuple[str, int]) -> Iterator[_Harness]:
    dsn, _ = postgres_target
    schema = f"monoid_pr08_{uuid.uuid4().hex}"
    database = PostgresDatabase(
        PostgresConfig(
            dsn=dsn,
            schema=schema,
            min_pool_size=1,
            max_pool_size=12,
            pool_timeout_s=10,
            application_name="monoid-pr08-service-test",
        )
    )
    database.open()
    PostgresMigrations(database).apply()
    authority = PostgresWriterAuthorityStore(database)
    sink = PostgresFencedRunSink(database)
    admission = PostgresCommandAdmissionStore(database)
    authority.check_ready()
    sink.check_ready()
    admission.check_ready()
    try:
        yield _Harness(database, authority, sink, admission)
    finally:
        try:
            with database.connection() as connection:
                with connection.transaction():
                    with connection.cursor() as cursor:
                        cursor.execute(
                            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                                sql.Identifier(schema)
                            )
                        )
        finally:
            database.close()


def _request(run_id: str, command_id: str, *, kind: str = "control") -> AdmissionRequest:
    digest = canonical_sha256({"run_id": run_id, "command_id": command_id, "kind": kind})
    return AdmissionRequest(
        run_id=run_id,
        command_id=command_id,
        kind=kind,  # type: ignore[arg-type]
        request_digest=digest,
        payload_ref=f"blob:{digest}",
    )


def _loop_factory(
    tmp_path: Path,
    spec: AgentRunSpec,
    adapter: object,
):
    def build(command: ActivationCommand, runtime: ActivationRuntime) -> AgentLoop:
        return AgentLoop(
            spec=AgentRunSpec(
                run_id=command.run_id,
                workspace_root=spec.workspace_root,
                run_root=tmp_path / f"replacement-{command.command_id}",
            ),
            model_adapter=adapter,
            runtime_config_provider=runtime_provider(runtime_config("fs.write")),
            run_sink=runtime.run_sink,
            writer_token=runtime.writer_token,
            write_authority=runtime.write_authority,
            authoritative_event_sinks=(runtime.event_sink,),
            event_sequence_seed=runtime.event_sequence_seed,
            status_file=False,
        )

    return build


def _accept_all(command: AdmittedCommand) -> DispatchResult:
    return DispatchResult(
        status="accepted",
        dispatch_ref=f"temporal:{command.run_id}/{command.command_id}",
    )


@dataclass
class _Transport:
    callback: Any

    def dispatch(self, command: AdmittedCommand) -> DispatchResult:
        return self.callback(command)


def _dispatcher(
    harness: _Harness,
    callback: Any,
    claim_ids: Iterator[str],
) -> CommandOutboxDispatcher:
    return CommandOutboxDispatcher(
        store=harness.admission,
        transport=_Transport(callback),
        owner_id="dispatcher-1",
        retry_delay_s=lambda attempt: 0.0,
        claim_id_factory=lambda: next(claim_ids),
    )


def test_postgres_admission_is_atomic_idempotent_and_terminal_aware(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    token, _ = harness.seed(tmp_path, "admission-atomic")
    request = _request(token.run_id, "command-1")

    first = harness.admission.admit(request)
    duplicate = harness.admission.admit(request)

    assert duplicate == first
    assert first.state == "prepared"
    assert first.command.command_sequence == 1
    with pytest.raises(AdmissionConflict):
        harness.admission.admit(replace(request, request_digest="0" * 64))

    with harness.database.transaction() as connection:
        with harness.database.cursor(connection) as cursor:
            for table in ("activation_admission_record", "activation_dispatch_outbox"):
                cursor.execute(
                    sql.SQL("SELECT count(*) FROM {} WHERE run_id = %s").format(
                        sql.Identifier(harness.database.config.schema, table)
                    ),
                    (request.run_id,),
                )
                assert cursor.fetchone() == (1,)

    assert (
        harness.sink.settle_terminal(
            TerminalOutcome(
                run_id=request.run_id,
                kind="cancelled",
                retry_eligibility=RetryEligibility.FORBIDDEN,
                error_code="cancelled",
            ),
            writer_token=token,
        ).status
        == "committed"
    )
    assert harness.admission.admit(request) == first
    with pytest.raises(AdmissionRunTerminal):
        harness.admission.admit(_request(request.run_id, "command-2"))


def test_concurrent_admission_converges_and_assigns_one_sequence_per_run(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    token, _ = harness.seed(tmp_path, "admission-race")
    shared = _request(token.run_id, "shared-command")
    barrier = threading.Barrier(8)

    def admit_shared(_: int):
        barrier.wait(timeout=5)
        return harness.admission.admit(shared)

    with ThreadPoolExecutor(max_workers=8) as executor:
        shared_receipts = list(executor.map(admit_shared, range(8)))

    assert shared_receipts == [shared_receipts[0]] * 8
    assert shared_receipts[0].command.command_sequence == 1

    unique_barrier = threading.Barrier(8)

    def admit_unique(index: int):
        unique_barrier.wait(timeout=5)
        return harness.admission.admit(_request(token.run_id, f"command-{index}"))

    with ThreadPoolExecutor(max_workers=8) as executor:
        unique_receipts = list(executor.map(admit_unique, range(8)))

    assert {receipt.command.command_sequence for receipt in unique_receipts} == set(range(2, 10))


def test_postgres_dispatch_claims_preserve_order_expiry_and_fencing(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    token, _ = harness.seed(tmp_path, "dispatch-order")
    first = harness.admission.admit(_request(token.run_id, "command-1")).command
    second = harness.admission.admit(_request(token.run_id, "command-2")).command

    claim = harness.admission.claim_dispatch("owner-a", "claim-1", lease_s=0.05)
    assert claim is not None and claim.command == first and claim.attempt == 1
    assert harness.admission.claim_dispatch("owner-b", "claim-blocked", lease_s=1) is None
    time.sleep(0.08)
    replacement = harness.admission.claim_dispatch("owner-b", "claim-2", lease_s=1)
    assert replacement is not None
    assert replacement.command == first
    assert replacement.attempt == 2
    assert replacement.token.generation == claim.token.generation + 1
    with pytest.raises(DispatchClaimLost):
        harness.admission.acknowledge_dispatch(
            claim.token,
            DispatchResult(status="accepted", dispatch_ref="temporal:stale"),
        )
    harness.admission.acknowledge_dispatch(
        replacement.token,
        DispatchResult(status="accepted", dispatch_ref="temporal:first"),
    )
    next_claim = harness.admission.claim_dispatch("owner-a", "claim-3", lease_s=1)
    assert next_claim is not None and next_claim.command == second
    retry = harness.admission.retry_dispatch(
        next_claim.token,
        error_code="transport_busy",
        delay_s=0,
    )
    assert retry.state == "prepared" and retry.error_code == "transport_busy"
    assert (
        harness.admission.retry_dispatch(
            next_claim.token,
            error_code="transport_busy",
            delay_s=0,
        )
        == retry
    )
    with pytest.raises(DispatchClaimLost):
        harness.admission.retry_dispatch(
            next_claim.token,
            error_code="transport_busy",
            delay_s=1,
        )
    final_claim = harness.admission.claim_dispatch("owner-a", "claim-4", lease_s=1)
    assert final_claim is not None and final_claim.attempt == 2
    dead = harness.admission.reject_dispatch(
        final_claim.token,
        error_code="unsupported_command",
    )
    assert dead.state == "dead_letter"


@pytest.mark.parametrize("settlement", ("acknowledge", "retry", "reject"))
def test_dispatch_settlement_rechecks_expiry_after_row_lock_wait(
    harness: _Harness,
    tmp_path: Path,
    settlement: str,
) -> None:
    token, _ = harness.seed(tmp_path, f"settlement-lock-{settlement}")
    harness.admission.admit(_request(token.run_id, "command-1"))
    claim = harness.admission.claim_dispatch("owner-a", "claim-1", lease_s=0.3)
    assert claim is not None

    def settle() -> None:
        if settlement == "acknowledge":
            harness.admission.acknowledge_dispatch(
                claim.token,
                DispatchResult(status="accepted", dispatch_ref="temporal:lock-wait"),
            )
        elif settlement == "retry":
            harness.admission.retry_dispatch(
                claim.token,
                error_code="transport_busy",
                delay_s=0,
            )
        else:
            harness.admission.reject_dispatch(
                claim.token,
                error_code="transport_rejected",
            )

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        with harness.database.connection() as blocking_connection:
            with blocking_connection.transaction():
                with blocking_connection.cursor() as cursor:
                    cursor.execute(
                        sql.SQL(
                            "SELECT 1 FROM {} WHERE run_id = %s AND command_id = %s FOR UPDATE"
                        ).format(
                            sql.Identifier(
                                harness.database.config.schema,
                                "activation_dispatch_outbox",
                            )
                        ),
                        (token.run_id, "command-1"),
                    )
                    assert cursor.fetchone() == (1,)
                    future = executor.submit(settle)
                    time.sleep(0.05)
                    assert not future.done()
                    time.sleep(0.3)
        with pytest.raises(DispatchClaimLost):
            future.result(timeout=5)
    finally:
        executor.shutdown(wait=True)

    receipt = harness.admission.receipt(token.run_id, "command-1")
    assert receipt is not None and receipt.state == "prepared"
    replacement = harness.admission.claim_dispatch("owner-b", "claim-2", lease_s=1)
    assert replacement is not None and replacement.attempt == 2


def test_postgres_claim_race_has_one_winner(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    token, _ = harness.seed(tmp_path, "dispatch-race")
    harness.admission.admit(_request(token.run_id, "command-1"))
    barrier = threading.Barrier(8)

    def claim(index: int) -> AdmittedCommand | None:
        barrier.wait(timeout=5)
        result = harness.admission.claim_dispatch(
            f"owner-{index}",
            f"claim-{index}",
            lease_s=2,
        )
        return None if result is None else result.command

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(claim, range(8)))

    assert [result for result in results if result is not None] == [
        harness.admission.receipt(token.run_id, "command-1").command  # type: ignore[union-attr]
    ]


def test_concurrent_reuse_of_one_claim_id_returns_one_exact_claim(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    first_token, _ = harness.seed(tmp_path, "claim-id-race-a")
    second_token, _ = harness.seed(tmp_path, "claim-id-race-b")
    harness.admission.admit(_request(first_token.run_id, "command-1"))
    harness.admission.admit(_request(second_token.run_id, "command-1"))
    barrier = threading.Barrier(2)

    def claim(_: int):
        barrier.wait(timeout=5)
        return harness.admission.claim_dispatch("owner-a", "shared-claim", lease_s=2)

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(claim, range(2)))

    assert claims[0] is not None
    assert claims == [claims[0], claims[0]]


def test_duplicate_transport_delivery_applies_one_activation(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    token, spec = harness.seed(tmp_path, "dispatch-duplicate-apply")
    admitted = harness.admission.admit(_request(token.run_id, "command-1")).command
    pending: dict[str, AdmittedCommand] = {}
    deliveries: list[str] = []

    def accept_with_first_response_loss(command: AdmittedCommand) -> DispatchResult:
        deliveries.append(command.identity_sha256)
        pending.setdefault(command.identity_sha256, command)
        if len(deliveries) == 1:
            raise TimeoutError("private acceptance response loss")
        return _accept_all(command)

    claim_ids = iter(("claim-1", "claim-2"))
    dispatcher = _dispatcher(harness, accept_with_first_response_loss, claim_ids)
    assert dispatcher.dispatch_once().state == "prepared"  # type: ignore[union-attr]
    assert dispatcher.dispatch_once().state == "dispatched"  # type: ignore[union-attr]
    assert deliveries == [admitted.identity_sha256, admitted.identity_sha256]
    assert list(pending.values()) == [admitted]

    activation = harness.admission.bind_activation(admitted, writer_token=token)
    adapter = _CountingAdapter()
    first = ActivationDriver(
        sink=harness.sink,
        writer_token=token,
        loop_factory=_loop_factory(tmp_path, spec, adapter),
    ).drive(activation)
    duplicate = ActivationDriver(
        sink=harness.sink,
        writer_token=token,
        loop_factory=lambda command, runtime: (_ for _ in ()).throw(
            AssertionError((command, runtime))
        ),
    ).drive(activation)

    assert duplicate == first
    assert adapter.calls == 1
    receipt = harness.admission.receipt(token.run_id, admitted.command_id)
    assert receipt is not None
    assert receipt.state == "completed"
    assert receipt.activation_receipt == first
    assert "private completion" not in str(receipt.to_json())


def test_queued_commands_bind_checkpoint_only_when_activated(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    token, spec = harness.seed(tmp_path, "deferred-source-binding")
    first = harness.admission.admit(_request(token.run_id, "command-1")).command
    second = harness.admission.admit(_request(token.run_id, "command-2")).command
    dispatcher = _dispatcher(harness, _accept_all, iter(("claim-1", "claim-2")))
    assert dispatcher.dispatch_once() is not None
    assert dispatcher.dispatch_once() is not None

    first_activation = harness.admission.bind_activation(first, writer_token=token)
    first_receipt = ActivationDriver(
        sink=harness.sink,
        writer_token=token,
        loop_factory=_loop_factory(tmp_path, spec, _CountingAdapter()),
    ).drive(first_activation)
    second_activation = harness.admission.bind_activation(second, writer_token=token)

    assert second_activation.source_checkpoint_seq == first_receipt.checkpoint_seq
    assert second_activation.source_checkpoint_seq > first_activation.source_checkpoint_seq
    assert harness.admission.bind_activation(second, writer_token=token) == second_activation

    assert harness.authority.release(token).status == "released"
    replacement = harness.authority.claim(
        token.run_id,
        "replacement-worker",
        timedelta(minutes=5),
    ).writer_token
    with pytest.raises(ActivationBindingConflict):
        harness.admission.bind_activation(second, writer_token=token)
    assert harness.admission.bind_activation(second, writer_token=replacement) == second_activation
