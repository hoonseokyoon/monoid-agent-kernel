from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.serial, pytest.mark.service, pytest.mark.slow]

if os.environ.get("MONOID_SERVICE_PROFILE") != "combined":
    pytest.skip("combined PostgreSQL and Temporal profile is not selected", allow_module_level=True)

from psycopg import sql  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import Worker  # noqa: E402

from monoid_agent_kernel.adapters.postgres import (  # noqa: E402
    PostgresCommandAdmissionStore,
    PostgresConfig,
    PostgresDatabase,
    PostgresFencedRunSink,
    PostgresMigrations,
    PostgresWriterAuthorityStore,
)
from monoid_agent_kernel.adapters.temporal import (  # noqa: E402
    TEMPORAL_STATUS_QUERY,
    TemporalRunPolicy,
    TemporalRunStatus,
    TemporalSignalWithStartTransport,
    temporal_workflow_id,
)
from monoid_agent_kernel.adapters.temporal.activity import (  # noqa: E402
    TemporalActivationActivity,
    TemporalActivityPolicy,
)
from monoid_agent_kernel.adapters.temporal.worker import TemporalWorkerGroup  # noqa: E402
from monoid_agent_kernel.adapters.temporal.workflow import TemporalRunWorkflow  # noqa: E402
from monoid_agent_kernel.core._util import canonical_sha256  # noqa: E402
from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore  # noqa: E402
from monoid_agent_kernel.core.interruption import InterruptionCause  # noqa: E402
from monoid_agent_kernel.core.spec import AgentRunSpec  # noqa: E402
from monoid_agent_kernel.errors import ModelAdapterError  # noqa: E402
from monoid_agent_kernel.hosting import (  # noqa: E402
    ActivationCommand,
    ActivationRuntime,
    AdmissionRequest,
    WriterLease,
)
from monoid_agent_kernel.loop import AgentLoop  # noqa: E402
from monoid_agent_kernel.providers.base import ModelTurn  # noqa: E402
from support.runtime import runtime_config, runtime_provider  # noqa: E402


class _RetryableAdapter:
    def next_turn(self, request: Any) -> ModelTurn:
        del request
        raise ModelAdapterError(
            "private provider failure",
            error_code="provider_unavailable",
            retryable=True,
        )


@dataclass
class _SlowCountingAdapter:
    delay_s: float
    calls: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def next_turn(self, request: Any) -> ModelTurn:
        del request
        with self._lock:
            self.calls += 1
        time.sleep(self.delay_s)
        return ModelTurn(final_text="private production activity completion")


@dataclass
class _FileCountingAdapter:
    counter_path: Path

    def next_turn(self, request: Any) -> ModelTurn:
        del request
        current = int(self.counter_path.read_text("utf-8"))
        self.counter_path.write_text(str(current + 1), encoding="utf-8")
        return ModelTurn(final_text="private replacement model result")


@dataclass
class _BlockingAdapter:
    started: threading.Event
    release: threading.Event
    calls: int = 0

    def next_turn(self, request: Any) -> ModelTurn:
        del request
        self.calls += 1
        self.started.set()
        assert self.release.wait(30)
        return ModelTurn(final_text="private result released after worker drain")


@dataclass
class _AmbiguousFirstClaim:
    inner: PostgresWriterAuthorityStore
    first_claim_delay_s: float = 1.5
    claim_calls: int = 0
    read_calls: int = 0
    owner_ids: list[str] = field(default_factory=list)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def claim(self, run_id: str, owner_id: str, ttl: timedelta) -> WriterLease:
        self.claim_calls += 1
        self.owner_ids.append(owner_id)
        if self.claim_calls == 1:
            time.sleep(self.first_claim_delay_s)
        lease = self.inner.claim(run_id, owner_id, ttl)
        if self.claim_calls <= 2:
            raise ConnectionError("private committed PostgreSQL claim response was lost")
        return lease

    def read(self, run_id: str):  # noqa: ANN201 - exact store result passthrough
        self.read_calls += 1
        if self.read_calls == 1:
            raise ConnectionError("private PostgreSQL reconciliation response was lost")
        return self.inner.read(run_id)


@dataclass
class _Harness:
    database: PostgresDatabase
    authority: PostgresWriterAuthorityStore
    sink: PostgresFencedRunSink
    admission: PostgresCommandAdmissionStore

    def seed(self, tmp_path: Path, run_id: str) -> tuple[WriterLease, AgentRunSpec]:
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
        assert source.run_until_suspended("resume through Temporal").reason == "turn_failed"
        source.release_parked()
        record = source_store.latest(run_id)
        assert record is not None
        lease = self.authority.claim(run_id, "seed-writer", timedelta(minutes=5))
        assert self.sink.commit_checkpoint(
            record.checkpoint,
            {},
            writer_token=lease.writer_token,
        ).status in {"committed", "already_committed"}
        return lease, spec

@pytest.fixture
def harness() -> Iterator[_Harness]:
    dsn = os.environ.get("MONOID_POSTGRES16_DSN")
    if not dsn:
        pytest.fail("MONOID_POSTGRES16_DSN is required for the combined profile")
    schema = f"monoid_pr10_{uuid.uuid4().hex}"
    database = PostgresDatabase(
        PostgresConfig(
            dsn=dsn,
            schema=schema,
            min_pool_size=1,
            max_pool_size=16,
            pool_timeout_s=10,
            application_name="monoid-pr10-combined-test",
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


def _prepare_temporal_cli() -> tuple[str, str]:
    cli_version = os.environ.get("MONOID_TEMPORAL_CLI_VERSION")
    if not cli_version:
        pytest.fail("MONOID_TEMPORAL_CLI_VERSION is required for the combined profile")
    root = Path(__file__).resolve().parents[2]
    cache_dir = Path(os.environ.get("MONOID_TEMPORAL_CLI_CACHE", root / ".tmp/temporal-cli-cache"))
    prepared = subprocess.run(
        [
            sys.executable,
            str(root / "tools/v023_ci.py"),
            "prepare-temporal-cli",
            "--cache-dir",
            str(cache_dir),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr or prepared.stdout
    artifact = json.loads(prepared.stdout)
    return str(artifact["executable"]), str(artifact["embedded_server"])


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_marker(process: subprocess.Popen[str], marker_path: Path) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if marker_path.is_file():
            return
        return_code = process.poll()
        if return_code is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"crash worker exited before its settled-invocation marker ({return_code}): "
                f"{stderr or stdout}"
            )
        time.sleep(0.05)
    raise AssertionError("crash worker did not settle a model invocation before the deadline")


def _request(run_id: str) -> AdmissionRequest:
    digest = canonical_sha256({"run_id": run_id, "command_id": "command-1"})
    return AdmissionRequest(
        run_id=run_id,
        command_id="command-1",
        kind="control",
        request_digest=digest,
        payload_ref=f"blob:{digest}",
    )


def _loop_factory(tmp_path: Path, spec: AgentRunSpec, adapter: object):
    def build(command: ActivationCommand, runtime: ActivationRuntime) -> AgentLoop:
        return AgentLoop(
            spec=AgentRunSpec(
                run_id=command.run_id,
                workspace_root=spec.workspace_root,
                run_root=tmp_path / "replacement-runs",
            ),
            model_adapter=adapter,
            runtime_config_provider=runtime_provider(runtime_config("fs.write")),
            run_sink=runtime.run_sink,
            writer_token=runtime.writer_token,
            write_authority=runtime.write_authority,
            cancellation_token=runtime.cancellation_token,
            authoritative_event_sinks=(runtime.event_sink,),
            event_sequence_seed=runtime.event_sequence_seed,
            async_model_cancel_grace_s=0.1,
            status_file=False,
        )

    return build


async def _wait_for_completion(
    environment: WorkflowEnvironment,
    harness: _Harness,
    run_id: str,
) -> tuple[TemporalRunStatus, Any]:
    handle = environment.client.get_workflow_handle(
        temporal_workflow_id(run_id),
        result_type=dict,
    )

    async def wait() -> tuple[TemporalRunStatus, Any]:
        while True:
            receipt = await asyncio.to_thread(harness.admission.receipt, run_id, "command-1")
            status = TemporalRunStatus.from_json(
                await handle.query(TEMPORAL_STATUS_QUERY, result_type=dict)
            )
            if receipt is not None and receipt.state == "completed" and status.next_command_sequence == 2:
                return status, receipt
            await asyncio.sleep(0.05)

    return await asyncio.wait_for(wait(), timeout=30)


def test_temporal_activity_drives_actual_postgres_boundary_and_releases_lease(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    executable, expected_server = _prepare_temporal_cli()
    run_id = f"combined-{uuid.uuid4().hex}"
    seed_lease, spec = harness.seed(tmp_path, run_id)
    admitted = harness.admission.admit(_request(run_id)).command
    dispatch_claim = harness.admission.claim_dispatch(
        "combined-dispatcher",
        "combined-claim",
        lease_s=30,
    )
    assert dispatch_claim is not None and dispatch_claim.command == admitted
    assert harness.authority.release(seed_lease.writer_token).status == "released"
    adapter = _SlowCountingAdapter(delay_s=0.7)
    response_loss_authority = _AmbiguousFirstClaim(harness.authority)

    async def run() -> tuple[TemporalRunStatus, Any]:
        async with await WorkflowEnvironment.start_local(
            dev_server_existing_path=executable,
            dev_server_log_level="warn",
        ) as environment:
            workflow_queue = f"workflow-{uuid.uuid4().hex}"
            activity_queue = f"activity-{uuid.uuid4().hex}"
            transport = TemporalSignalWithStartTransport(
                client=environment.client,
                event_loop=asyncio.get_running_loop(),
                workflow_task_queue=workflow_queue,
                run_policy=TemporalRunPolicy(
                    activity_task_queue=activity_queue,
                    activity_start_to_close_timeout_s=30,
                    activity_heartbeat_timeout_s=1,
                    activity_max_attempts=3,
                ),
            )
            accepted = await transport.dispatch_async(admitted)
            assert accepted.status == "accepted"
            await asyncio.to_thread(
                harness.admission.acknowledge_dispatch,
                dispatch_claim.token,
                accepted,
            )
            activation = TemporalActivationActivity(
                authority_store=response_loss_authority,
                admission_store=harness.admission,
                run_sink=harness.sink,
                loop_factory=_loop_factory(tmp_path, spec, adapter),
                policy=TemporalActivityPolicy(
                    writer_lease_ttl_s=2,
                    writer_lease_renew_interval_s=0.25,
                    heartbeat_interval_s=0.1,
                    supervisor_join_timeout_s=2,
                    local_task_wait_s=5,
                ),
            )
            async with TemporalWorkerGroup(
                client=environment.client,
                workflow_task_queue=workflow_queue,
                activity_task_queue=activity_queue,
                activation_activity=activation,
                max_concurrent_activities=2,
                graceful_shutdown_timeout_s=5,
            ):
                status, receipt = await _wait_for_completion(environment, harness, run_id)
                handle = environment.client.get_workflow_handle(temporal_workflow_id(run_id))
                await handle.cancel()
                return status, receipt

    status, receipt = asyncio.run(run())

    assert status.phase == "waiting"
    assert receipt.activation_receipt is not None
    assert receipt.activation_command is not None
    assert (
        receipt.activation_receipt.command_identity_sha256
        == receipt.activation_command.identity_sha256
    )
    assert adapter.calls == 1
    assert response_loss_authority.claim_calls == 4
    assert response_loss_authority.read_calls == 1
    assert response_loss_authority.owner_ids[0] == response_loss_authority.owner_ids[1]
    assert len(set(response_loss_authority.owner_ids)) == 3
    authority = harness.authority.read(run_id)
    assert authority is not None and authority.revoked is True and authority.active is False
    assert expected_server
    assert "private production activity completion" not in str(receipt.to_json())


def test_worker_kill_takes_over_expired_generation_without_repeating_paid_call(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    executable, _ = _prepare_temporal_cli()
    run_id = f"takeover-{uuid.uuid4().hex}"
    seed_lease, spec = harness.seed(tmp_path, run_id)
    admitted = harness.admission.admit(_request(run_id)).command
    dispatch_claim = harness.admission.claim_dispatch(
        "takeover-dispatcher",
        "takeover-claim",
        lease_s=30,
    )
    assert dispatch_claim is not None and dispatch_claim.command == admitted
    assert harness.authority.release(seed_lease.writer_token).status == "released"
    marker_path = tmp_path / "settled-invocation.marker"
    counter_path = tmp_path / "provider-calls.txt"
    counter_path.write_text("0", encoding="utf-8")
    port = _free_tcp_port()
    temporal_target = f"127.0.0.1:{port}"
    root = Path(__file__).resolve().parents[2]
    process: subprocess.Popen[str] | None = None

    async def run() -> tuple[TemporalRunStatus, Any]:
        nonlocal process
        async with await WorkflowEnvironment.start_local(
            port=port,
            dev_server_existing_path=executable,
            dev_server_log_level="warn",
        ) as environment:
            workflow_queue = f"workflow-{uuid.uuid4().hex}"
            activity_queue = f"activity-{uuid.uuid4().hex}"
            transport = TemporalSignalWithStartTransport(
                client=environment.client,
                event_loop=asyncio.get_running_loop(),
                workflow_task_queue=workflow_queue,
                run_policy=TemporalRunPolicy(
                    activity_task_queue=activity_queue,
                    activity_start_to_close_timeout_s=60,
                    activity_heartbeat_timeout_s=1,
                    activity_max_attempts=10,
                ),
            )
            async with Worker(
                environment.client,
                task_queue=workflow_queue,
                workflows=[TemporalRunWorkflow],
            ):
                accepted = await transport.dispatch_async(admitted)
                assert accepted.status == "accepted"
                await asyncio.to_thread(
                    harness.admission.acknowledge_dispatch,
                    dispatch_claim.token,
                    accepted,
                )
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(root / "tests/support/temporal_activity_worker.py"),
                        "--temporal-target",
                        temporal_target,
                        "--postgres-dsn",
                        harness.database.config.dsn,
                        "--postgres-schema",
                        harness.database.config.schema,
                        "--activity-task-queue",
                        activity_queue,
                        "--workspace-root",
                        str(spec.workspace_root),
                        "--run-root",
                        str(tmp_path / "crashed-worker-runs"),
                        "--marker-path",
                        str(marker_path),
                        "--counter-path",
                        str(counter_path),
                    ],
                    cwd=root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                await asyncio.to_thread(_wait_for_marker, process, marker_path)
                process.terminate()
                await asyncio.to_thread(process.wait, 10)
                assert process.returncode is not None
                await asyncio.sleep(2.5)
                expired = await asyncio.to_thread(harness.authority.read, run_id)
                assert expired is not None and expired.active is False

                replacement = TemporalActivationActivity(
                    authority_store=harness.authority,
                    admission_store=harness.admission,
                    run_sink=harness.sink,
                    loop_factory=_loop_factory(
                        tmp_path,
                        spec,
                        _FileCountingAdapter(counter_path),
                    ),
                    policy=TemporalActivityPolicy(
                        writer_lease_ttl_s=2,
                        writer_lease_renew_interval_s=0.4,
                        heartbeat_interval_s=0.2,
                        supervisor_join_timeout_s=2,
                        local_task_wait_s=5,
                    ),
                )
                with ThreadPoolExecutor(max_workers=2) as executor:
                    async with Worker(
                        environment.client,
                        task_queue=activity_queue,
                        activities=[replacement.run],
                        activity_executor=executor,
                        max_concurrent_activities=2,
                        max_heartbeat_throttle_interval=timedelta(seconds=0.2),
                        default_heartbeat_throttle_interval=timedelta(seconds=0.2),
                    ):
                        status, receipt = await _wait_for_completion(
                            environment,
                            harness,
                            run_id,
                        )
                handle = environment.client.get_workflow_handle(temporal_workflow_id(run_id))
                await handle.cancel()
                return status, receipt

    try:
        status, receipt = asyncio.run(run())
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)

    assert status.next_command_sequence == 2
    assert receipt.activation_receipt is not None
    assert counter_path.read_text("utf-8") == "1"
    final_authority = harness.authority.read(run_id)
    assert final_authority is not None
    assert final_authority.writer_token.generation >= 3
    assert final_authority.revoked is True and final_authority.active is False


def test_worker_shutdown_during_paid_call_commits_reconciliation_receipt(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    executable, _ = _prepare_temporal_cli()
    run_id = f"drain-{uuid.uuid4().hex}"
    seed_lease, spec = harness.seed(tmp_path, run_id)
    admitted = harness.admission.admit(_request(run_id)).command
    dispatch_claim = harness.admission.claim_dispatch(
        "drain-dispatcher",
        "drain-claim",
        lease_s=30,
    )
    assert dispatch_claim is not None and dispatch_claim.command == admitted
    assert harness.authority.release(seed_lease.writer_token).status == "released"
    started = threading.Event()
    release = threading.Event()
    adapter = _BlockingAdapter(started=started, release=release)

    async def run() -> tuple[TemporalRunStatus, Any]:
        async with await WorkflowEnvironment.start_local(
            dev_server_existing_path=executable,
            dev_server_log_level="warn",
        ) as environment:
            workflow_queue = f"workflow-{uuid.uuid4().hex}"
            activity_queue = f"activity-{uuid.uuid4().hex}"
            transport = TemporalSignalWithStartTransport(
                client=environment.client,
                event_loop=asyncio.get_running_loop(),
                workflow_task_queue=workflow_queue,
                run_policy=TemporalRunPolicy(
                    activity_task_queue=activity_queue,
                    activity_start_to_close_timeout_s=30,
                    activity_heartbeat_timeout_s=1,
                    activity_max_attempts=5,
                ),
            )
            activation = TemporalActivationActivity(
                authority_store=harness.authority,
                admission_store=harness.admission,
                run_sink=harness.sink,
                loop_factory=_loop_factory(tmp_path, spec, adapter),
                policy=TemporalActivityPolicy(
                    writer_lease_ttl_s=4,
                    writer_lease_renew_interval_s=0.4,
                    heartbeat_interval_s=0.2,
                    supervisor_join_timeout_s=2,
                    local_task_wait_s=5,
                ),
            )
            group = TemporalWorkerGroup(
                client=environment.client,
                workflow_task_queue=workflow_queue,
                activity_task_queue=activity_queue,
                activation_activity=activation,
                max_concurrent_activities=2,
                graceful_shutdown_timeout_s=5,
            )
            await group.__aenter__()
            try:
                accepted = await transport.dispatch_async(admitted)
                assert accepted.status == "accepted"
                await asyncio.to_thread(
                    harness.admission.acknowledge_dispatch,
                    dispatch_claim.token,
                    accepted,
                )
                assert await asyncio.to_thread(started.wait, 10)
            finally:
                await group.__aexit__(None, None, None)
                release.set()

            async with Worker(
                environment.client,
                task_queue=workflow_queue,
                workflows=[TemporalRunWorkflow],
            ):
                status, receipt = await _wait_for_completion(environment, harness, run_id)
            handle = environment.client.get_workflow_handle(temporal_workflow_id(run_id))
            await handle.cancel()
            return status, receipt

    try:
        status, receipt = asyncio.run(run())
    finally:
        release.set()

    assert status.next_command_sequence == 2
    assert receipt.activation_receipt is not None
    assert receipt.activation_receipt.boundary_reason == "terminal"
    assert receipt.activation_receipt.error_code == "dispatch_unknown"
    assert receipt.activation_receipt.retry_eligibility.value == "after_reconciliation"
    assert receipt.activation_receipt.interruption_cause is None
    assert receipt.activation_receipt.terminal is True
    assert adapter.calls == 1
    authority = harness.authority.read(run_id)
    assert authority is not None and authority.revoked is True and authority.active is False


def test_worker_shutdown_before_provider_entry_commits_graceful_drain_receipt(
    harness: _Harness,
    tmp_path: Path,
) -> None:
    executable, _ = _prepare_temporal_cli()
    run_id = f"safe-drain-{uuid.uuid4().hex}"
    seed_lease, spec = harness.seed(tmp_path, run_id)
    admitted = harness.admission.admit(_request(run_id)).command
    dispatch_claim = harness.admission.claim_dispatch(
        "safe-drain-dispatcher",
        "safe-drain-claim",
        lease_s=30,
    )
    assert dispatch_claim is not None and dispatch_claim.command == admitted
    assert harness.authority.release(seed_lease.writer_token).status == "released"
    entered_factory = threading.Event()
    adapter = _SlowCountingAdapter(delay_s=0)
    base_factory = _loop_factory(tmp_path, spec, adapter)

    def drain_aware_factory(
        command: ActivationCommand,
        runtime: ActivationRuntime,
    ) -> AgentLoop:
        entered_factory.set()
        deadline = time.monotonic() + 10
        while not runtime.cancellation_token.requested and time.monotonic() < deadline:
            time.sleep(0.01)
        assert runtime.cancellation_token.cause is InterruptionCause.GRACEFUL_DRAIN
        return base_factory(command, runtime)

    async def run() -> tuple[TemporalRunStatus, Any]:
        async with await WorkflowEnvironment.start_local(
            dev_server_existing_path=executable,
            dev_server_log_level="warn",
        ) as environment:
            workflow_queue = f"workflow-{uuid.uuid4().hex}"
            activity_queue = f"activity-{uuid.uuid4().hex}"
            transport = TemporalSignalWithStartTransport(
                client=environment.client,
                event_loop=asyncio.get_running_loop(),
                workflow_task_queue=workflow_queue,
                run_policy=TemporalRunPolicy(
                    activity_task_queue=activity_queue,
                    activity_start_to_close_timeout_s=30,
                    activity_heartbeat_timeout_s=1,
                    activity_max_attempts=5,
                ),
            )
            activation = TemporalActivationActivity(
                authority_store=harness.authority,
                admission_store=harness.admission,
                run_sink=harness.sink,
                loop_factory=drain_aware_factory,
                policy=TemporalActivityPolicy(
                    writer_lease_ttl_s=4,
                    writer_lease_renew_interval_s=0.4,
                    heartbeat_interval_s=0.2,
                    supervisor_join_timeout_s=2,
                    local_task_wait_s=5,
                ),
            )
            group = TemporalWorkerGroup(
                client=environment.client,
                workflow_task_queue=workflow_queue,
                activity_task_queue=activity_queue,
                activation_activity=activation,
                max_concurrent_activities=2,
                graceful_shutdown_timeout_s=5,
            )
            await group.__aenter__()
            try:
                accepted = await transport.dispatch_async(admitted)
                assert accepted.status == "accepted"
                await asyncio.to_thread(
                    harness.admission.acknowledge_dispatch,
                    dispatch_claim.token,
                    accepted,
                )
                assert await asyncio.to_thread(entered_factory.wait, 10)
            finally:
                await group.__aexit__(None, None, None)

            async with Worker(
                environment.client,
                task_queue=workflow_queue,
                workflows=[TemporalRunWorkflow],
            ):
                status, receipt = await _wait_for_completion(environment, harness, run_id)
            handle = environment.client.get_workflow_handle(temporal_workflow_id(run_id))
            await handle.cancel()
            return status, receipt

    status, receipt = asyncio.run(run())

    assert status.next_command_sequence == 2
    assert receipt.activation_receipt is not None
    assert receipt.activation_receipt.boundary_reason == "interrupted"
    assert receipt.activation_receipt.interruption_cause is InterruptionCause.GRACEFUL_DRAIN
    assert receipt.activation_receipt.terminal is False
    assert adapter.calls == 0
    authority = harness.authority.read(run_id)
    assert authority is not None and authority.revoked is True and authority.active is False
