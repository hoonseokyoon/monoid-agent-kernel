from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterator

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.serial, pytest.mark.service]


def _selected() -> bool:
    return os.environ.get("MONOID_SERVICE_PROFILE") in {"objectstore", "combined"}


if not _selected():
    pytest.skip("ObjectStore service profile is not selected", allow_module_level=True)

import boto3  # noqa: E402
from botocore import config as botocore_config  # noqa: E402
from psycopg import sql  # noqa: E402

from monoid_agent_kernel.adapters.object_store import (  # noqa: E402
    S3ContentAddressedBlobStore,
    S3ObjectStoreConfig,
    S3ObjectStoreAdmin,
)
from monoid_agent_kernel.adapters.postgres import (  # noqa: E402
    PostgresConfig,
    PostgresDatabase,
    PostgresDurableStreamCorrupt,
    PostgresFencedRunSink,
    PostgresMigrations,
    PostgresObjectGarbageCollector,
    PostgresObjectStoreDurableStreamStore,
    PostgresOperations,
    PostgresWriterAuthorityStore,
)
from monoid_agent_kernel.core.authority import ActivationWriteAuthority  # noqa: E402
from monoid_agent_kernel.conformance import run_durable_stream_store_contract  # noqa: E402
from monoid_agent_kernel.core.model_invocation import logical_model_call_id  # noqa: E402
from monoid_agent_kernel.core.model_stream import (  # noqa: E402
    ModelStreamContext,
    ModelStreamDelta,
    ModelStreamOutcome,
)
from monoid_agent_kernel.core.outcome import RetryEligibility, TerminalOutcome  # noqa: E402
from monoid_agent_kernel.hosting import (  # noqa: E402
    DurableModelStreamObserver,
    DurableStreamIdentity,
    WriterToken,
    durable_model_stream_id,
)


_POSTGRES_TARGETS = [
    ("MONOID_POSTGRES16_DSN", 16, {"objectstore", "combined"}),
    ("MONOID_POSTGRES18_DSN", 18, {"combined"}),
]


def _client() -> object:
    endpoint = os.environ.get("MONOID_MINIO_ENDPOINT")
    if not endpoint:
        pytest.fail("MONOID_MINIO_ENDPOINT is required for durable stream service tests")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("MONOID_MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("MONOID_MINIO_SECRET_KEY"),
        region_name="us-east-1",
        config=botocore_config.Config(
            signature_version="s3v4",
            connect_timeout=10,
            read_timeout=30,
            retries={"max_attempts": 5, "mode": "standard"},
            s3={"addressing_style": "path"},
        ),
    )


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
    streams: PostgresObjectStoreDurableStreamStore

    def claim(self, run_id: str, owner_id: str = "stream-worker") -> WriterToken:
        return self.authority.claim(run_id, owner_id, timedelta(minutes=5)).writer_token


class _ContractHarness:
    def __init__(self, harness: _Harness, run_id: str) -> None:
        self.store = harness.streams
        self._authority = harness.authority
        self.writer_token = harness.claim(run_id, "contract-worker")

    def replace_writer(self) -> WriterToken:
        assert self._authority.release(self.writer_token).status == "released"
        self.writer_token = self._authority.claim(
            self.writer_token.run_id,
            "contract-replacement",
            timedelta(minutes=5),
        ).writer_token
        return self.writer_token


class _BlockingGetStore:
    def __init__(self, delegate: object) -> None:
        self.delegate = delegate
        self.entered = threading.Event()
        self.release = threading.Event()

    def put_if_absent(self, sha256: str, data: bytes) -> object:
        return self.delegate.put_if_absent(sha256, data)  # type: ignore[attr-defined,no-any-return]

    def stat(self, sha256: str) -> object:
        return self.delegate.stat(sha256)  # type: ignore[attr-defined,no-any-return]

    def get_checked(self, sha256: str) -> bytes:
        self.entered.set()
        assert self.release.wait(10)
        return self.delegate.get_checked(sha256)  # type: ignore[attr-defined,no-any-return]


class _CountingPutStore:
    def __init__(self, delegate: object) -> None:
        self.delegate = delegate
        self.puts = 0

    def put_if_absent(self, sha256: str, data: bytes) -> object:
        self.puts += 1
        return self.delegate.put_if_absent(sha256, data)  # type: ignore[attr-defined,no-any-return]

    def stat(self, sha256: str) -> object:
        return self.delegate.stat(sha256)  # type: ignore[attr-defined,no-any-return]

    def get_checked(self, sha256: str) -> bytes:
        return self.delegate.get_checked(sha256)  # type: ignore[attr-defined,no-any-return]


class _BlockingPutStore:
    def __init__(self, delegate: object) -> None:
        self.delegate = delegate
        self.entered = threading.Event()
        self.release = threading.Event()
        self.puts = 0

    def put_if_absent(self, sha256: str, data: bytes) -> object:
        self.puts += 1
        self.entered.set()
        assert self.release.wait(10)
        return self.delegate.put_if_absent(sha256, data)  # type: ignore[attr-defined,no-any-return]

    def stat(self, sha256: str) -> object:
        return self.delegate.stat(sha256)  # type: ignore[attr-defined,no-any-return]

    def get_checked(self, sha256: str) -> bytes:
        return self.delegate.get_checked(sha256)  # type: ignore[attr-defined,no-any-return]


class _BlockingStatStore:
    def __init__(self, delegate: object) -> None:
        self.delegate = delegate
        self.captured = threading.Event()
        self.release = threading.Event()
        self._block_once = True

    def put_if_absent(self, sha256: str, data: bytes) -> object:
        return self.delegate.put_if_absent(sha256, data)  # type: ignore[attr-defined,no-any-return]

    def stat(self, sha256: str) -> object:
        value = self.delegate.stat(sha256)  # type: ignore[attr-defined]
        if self._block_once:
            self._block_once = False
            self.captured.set()
            assert self.release.wait(10)
        return value

    def get_checked(self, sha256: str) -> bytes:
        return self.delegate.get_checked(sha256)  # type: ignore[attr-defined,no-any-return]


@pytest.fixture
def harness(postgres_target: tuple[str, int]) -> Iterator[_Harness]:
    dsn, expected_major = postgres_target
    schema = f"monoid_pr11_{uuid.uuid4().hex}"
    client = _client()
    bucket = f"monoid-v023-stream-{uuid.uuid4().hex}"
    client.create_bucket(Bucket=bucket)  # type: ignore[attr-defined]
    client.put_bucket_versioning(  # type: ignore[attr-defined]
        Bucket=bucket,
        VersioningConfiguration={"Status": "Enabled"},
    )
    database = PostgresDatabase(
        PostgresConfig(
            dsn=dsn,
            schema=schema,
            min_pool_size=1,
            max_pool_size=12,
            pool_timeout_s=10,
            application_name="monoid-pr11-stream-service-test",
        )
    )
    database.open()
    assert database.health().server_major == expected_major
    PostgresMigrations(database).apply()
    object_store = S3ContentAddressedBlobStore(
        S3ObjectStoreConfig(
            bucket=bucket,
            prefix="durable-stream",
            endpoint_url=os.environ["MONOID_MINIO_ENDPOINT"],
            addressing_style="path",
            max_object_bytes=8 * 1024 * 1024,
        ),
        client=client,
    )
    authority = PostgresWriterAuthorityStore(database)
    sink = PostgresFencedRunSink(database)
    streams = PostgresObjectStoreDurableStreamStore(database, object_store)
    authority.check_ready()
    sink.check_ready()
    streams.check_ready()
    try:
        yield _Harness(database, authority, sink, streams)
    finally:
        try:
            with database.transaction() as connection:
                with database.cursor(connection) as cursor:
                    cursor.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
        finally:
            database.close()
        objects = client.list_objects_v2(Bucket=bucket).get("Contents", [])  # type: ignore[attr-defined]
        for value in objects:
            client.delete_object(Bucket=bucket, Key=value["Key"])  # type: ignore[attr-defined]
        versions = client.list_object_versions(Bucket=bucket)  # type: ignore[attr-defined]
        for value in (*versions.get("Versions", []), *versions.get("DeleteMarkers", [])):
            client.delete_object(  # type: ignore[attr-defined]
                Bucket=bucket,
                Key=value["Key"],
                VersionId=value["VersionId"],
            )
        client.delete_bucket(Bucket=bucket)  # type: ignore[attr-defined]


def _identity(run_id: str, *, suffix: str = "main", channel: str = "output") -> DurableStreamIdentity:
    return DurableStreamIdentity(
        run_id=run_id,
        stream_id=f"stream-{suffix}",
        logical_call_id=f"logical-call-{suffix}",
        channel=channel,
    )


def test_actual_postgres_objectstore_passes_reusable_stream_contract(
    harness: _Harness,
) -> None:
    outcomes = run_durable_stream_store_contract(
        lambda run_id: _ContractHarness(harness, run_id)
    )

    assert all(outcome.passed for outcome in outcomes), [outcome.to_json() for outcome in outcomes]


def test_actual_operations_snapshot_and_s3_doctor_are_public_and_content_free(
    harness: _Harness,
) -> None:
    private_content = b"private-operations-qualification-payload"
    run_id = f"run-private-operations-{uuid.uuid4().hex}"
    token = harness.claim(run_id, "operations-worker")
    identity = _identity(run_id, suffix="operations")
    assert harness.streams.open(identity, writer_token=token).status == "opened"
    assert harness.streams.append(
        identity,
        generation=1,
        start_offset=0,
        data=private_content,
        writer_token=token,
    ).status == "committed"
    with harness.database.transaction() as connection:
        with harness.database.cursor(connection) as cursor:
            cursor.execute(
                sql.SQL(
                    "UPDATE {}.{} SET opened_at = pg_catalog.clock_timestamp() - "
                    "interval '1 day' WHERE run_id = %s AND stream_id = %s AND channel = %s"
                ).format(
                    sql.Identifier(harness.database.config.schema),
                    sql.Identifier("durable_stream_head"),
                ),
                (identity.run_id, identity.stream_id, identity.channel),
            )
    assert harness.streams.seal(
        identity,
        generation=1,
        final_size_bytes=len(private_content),
        final_sha256=hashlib.sha256(private_content).hexdigest(),
        writer_token=token,
    ).status == "sealed"
    assert harness.streams.reset(
        identity,
        expected_generation=1,
        reset_id="operations-current-generation",
        writer_token=token,
    ).status == "reset"

    operations = PostgresOperations(harness.database)
    assert operations.check_ready().current is True
    snapshot = operations.snapshot()
    by_identity = {
        (metric.name, metric.attributes): metric.value for metric in snapshot.metrics
    }

    assert by_identity[
        ("monoid.postgres.authority.count", (("state", "active"),))
    ] >= 1
    assert by_identity[("monoid.postgres.stream.head.count", (("state", "open"),))] >= 1
    assert by_identity[("monoid.postgres.stream.chunk.count", ())] >= 1
    assert by_identity[("monoid.postgres.stream.chunk.bytes", ())] >= len(private_content)
    assert by_identity[("monoid.postgres.stream.oldest_open_age", ())] < 60
    assert by_identity[("monoid.postgres.object.association.count", ())] >= 1

    snapshot_json = json.dumps(snapshot.to_json(), sort_keys=True)
    assert private_content.decode() not in snapshot_json
    assert run_id not in snapshot_json
    assert harness.database.config.schema not in snapshot_json
    assert harness.database.config.dsn not in snapshot_json

    delegate = harness.streams.object_store
    admin = S3ObjectStoreAdmin(delegate.config, client=_client())  # type: ignore[attr-defined]
    doctor = admin.doctor()
    public_report = repr(doctor)
    assert doctor.ok is True
    assert doctor.reachable is True
    assert doctor.versioning_enabled is True
    assert delegate.config.bucket not in public_report  # type: ignore[attr-defined]
    assert os.environ["MONOID_MINIO_ENDPOINT"] not in public_report


def test_actual_postgres_objectstore_stream_reconnect_reset_seal_and_terminal_race(
    harness: _Harness,
) -> None:
    run_id = f"run-stream-{uuid.uuid4().hex}"
    token = harness.claim(run_id)
    identity = _identity(run_id)

    opened = harness.streams.open(identity, writer_token=token)
    assert opened.status == "opened"
    first = harness.streams.append(
        identity,
        generation=1,
        start_offset=0,
        data=b"hello ",
        writer_token=token,
    )
    assert first.status == "committed"
    assert harness.streams.append(
        identity,
        generation=1,
        start_offset=0,
        data=b"hello ",
        writer_token=token,
    ).status == "already_committed"
    assert harness.streams.append(
        identity,
        generation=1,
        start_offset=0,
        data=b"wrong",
        writer_token=token,
    ).status == "conflict"
    assert harness.streams.append(
        identity,
        generation=1,
        start_offset=99,
        data=b"gap",
        writer_token=token,
    ).status == "gap"
    second = harness.streams.append(
        identity,
        generation=1,
        start_offset=6,
        data="세계".encode(),
        writer_token=token,
    )
    assert second.status == "committed"

    first_page = harness.streams.read_after(identity, generation=1, cursor=0, limit=1)
    assert first_page.status == "ok"
    assert [chunk.data for chunk in first_page.chunks] == [b"hello "]
    second_page = harness.streams.read_after(
        identity,
        generation=1,
        cursor=first_page.next_cursor,
        limit=10,
    )
    assert second_page.status == "ok"
    assert b"".join(chunk.data for chunk in first_page.chunks + second_page.chunks).decode() == (
        "hello 세계"
    )
    assert harness.streams.read_after(identity, generation=1, cursor=1).status == "gap"

    complete = "hello 세계".encode()
    assert harness.streams.seal(
        identity,
        generation=1,
        final_size_bytes=len(complete),
        final_sha256="0" * 64,
        writer_token=token,
    ).status == "conflict"
    sealed = harness.streams.seal(
        identity,
        generation=1,
        final_size_bytes=len(complete),
        final_sha256=hashlib.sha256(complete).hexdigest(),
        writer_token=token,
    )
    assert sealed.status == "sealed"
    assert harness.streams.seal(
        identity,
        generation=1,
        final_size_bytes=len(complete),
        final_sha256=hashlib.sha256(complete).hexdigest(),
        writer_token=token,
    ).status == "already_sealed"
    assert harness.streams.append(
        identity,
        generation=1,
        start_offset=len(complete),
        data=b"late",
        writer_token=token,
    ).status == "sealed"

    reset = harness.streams.reset(
        identity,
        expected_generation=1,
        reset_id="retry-attempt-2",
        writer_token=token,
    )
    assert reset.status == "reset"
    assert reset.applied_generation == 2
    assert harness.streams.reset(
        identity,
        expected_generation=1,
        reset_id="retry-attempt-2",
        writer_token=token,
    ).status == "already_reset"
    old_read = harness.streams.read_after(identity, generation=1, cursor=len(complete))
    assert old_read.status == "reset"
    assert old_read.head is not None and old_read.head.generation == 2
    assert harness.streams.append(
        identity,
        generation=1,
        start_offset=len(complete),
        data=b"stale",
        writer_token=token,
    ).status == "old_generation"
    assert harness.streams.append(
        identity,
        generation=2,
        start_offset=0,
        data=b"replacement",
        writer_token=token,
    ).status == "committed"

    terminal_run = f"run-terminal-stream-{uuid.uuid4().hex}"
    terminal_token = harness.claim(terminal_run)
    terminal_identity = _identity(terminal_run, suffix="terminal")
    assert harness.streams.open(terminal_identity, writer_token=terminal_token).status == "opened"
    assert harness.streams.append(
        terminal_identity,
        generation=1,
        start_offset=0,
        data=b"before-terminal",
        writer_token=terminal_token,
    ).status == "committed"
    terminal = TerminalOutcome(
        run_id=terminal_run,
        kind="completed",
        retry_eligibility=RetryEligibility.NOT_APPLICABLE,
    )
    assert harness.sink.settle_terminal(terminal, writer_token=terminal_token).status == "committed"
    assert harness.streams.append(
        terminal_identity,
        generation=1,
        start_offset=0,
        data=b"before-terminal",
        writer_token=terminal_token,
    ).status == "already_committed"
    assert harness.streams.append(
        terminal_identity,
        generation=1,
        start_offset=len(b"before-terminal"),
        data=b"late",
        writer_token=terminal_token,
    ).status == "run_terminal"
    assert harness.streams.open(
        _identity(terminal_run, suffix="after-terminal"),
        writer_token=terminal_token,
    ).status == "run_terminal"


def test_actual_stream_store_fences_takeover_and_model_observer_persists(
    harness: _Harness,
) -> None:
    takeover_run = f"run-stream-takeover-{uuid.uuid4().hex}"
    stale = harness.claim(takeover_run, "worker-old")
    identity = _identity(takeover_run, suffix="takeover")
    assert harness.streams.open(identity, writer_token=stale).status == "opened"
    assert harness.authority.release(stale).status == "released"
    current = harness.claim(takeover_run, "worker-new")
    assert current.generation == stale.generation + 1
    assert harness.streams.append(
        identity,
        generation=1,
        start_offset=0,
        data=b"stale",
        writer_token=stale,
    ).status == "fenced"
    reopened = PostgresObjectStoreDurableStreamStore(
        harness.database,
        harness.streams.object_store,
    )
    reopened.check_ready()
    assert reopened.open(identity, writer_token=current).status == "already_open"
    assert reopened.append(
        identity,
        generation=1,
        start_offset=0,
        data=b"current",
        writer_token=current,
    ).status == "committed"

    observer_run = f"run-stream-observer-{uuid.uuid4().hex}"
    observer_token = harness.claim(observer_run, "observer-worker")
    write_authority = ActivationWriteAuthority()
    observer = DurableModelStreamObserver(
        harness.streams,
        writer_token=observer_token,
        write_authority=write_authority,
        chunk_bytes=8,
        flush_interval_s=0.02,
        max_buffer_bytes=64,
    )
    context = ModelStreamContext(
        run_id=observer_run,
        root_run_id=observer_run,
        turn_id="turn-observer",
        stream_id="stream-observer",
        step=1,
        provider="fixture",
        model="fixture-model",
        started_at="2026-08-25T00:00:00Z",
    )
    writer = observer.open(context)
    writer.push(ModelStreamDelta(channel="reasoning", text="why "))
    writer.push(ModelStreamDelta(channel="output", text="durable "))
    writer.push(ModelStreamDelta(channel="output", text="answer"))
    writer.close(ModelStreamOutcome(status="completed", final_text="durable answer"))

    output_identity = DurableStreamIdentity(
        run_id=observer_run,
        stream_id=durable_model_stream_id(observer_run, context.turn_id),
        logical_call_id=logical_model_call_id(observer_run, context.turn_id),
        channel="output",
    )
    output = harness.streams.read_after(output_identity, generation=1, cursor=0)
    assert output.status == "ok"
    assert b"".join(chunk.data for chunk in output.chunks) == b"durable answer"
    assert output.head is not None and output.head.state == "sealed"


def test_actual_model_observer_prepares_both_lanes_before_recovery_seals(
    harness: _Harness,
) -> None:
    run_id = f"run-stream-prepared-recovery-{uuid.uuid4().hex}"
    stale = harness.claim(run_id, "prepared-worker")
    context = ModelStreamContext(
        run_id=run_id,
        root_run_id=run_id,
        turn_id="turn-prepared-recovery",
        stream_id="execution-local-prepared-recovery",
        step=1,
        provider="fixture",
        model="fixture-model",
        started_at="2026-08-25T00:00:00Z",
    )
    writer = DurableModelStreamObserver(
        harness.streams,
        writer_token=stale,
        write_authority=ActivationWriteAuthority(),
        chunk_bytes=1024,
        flush_interval_s=10,
        max_buffer_bytes=4096,
    ).open(context)
    writer.begin_dispatch()  # type: ignore[attr-defined]
    writer.push(ModelStreamDelta(channel="reasoning", text="complete reasoning"))
    writer.push(ModelStreamDelta(channel="output", text="complete answer"))

    writer.prepare_settlement()  # type: ignore[attr-defined]

    output = DurableStreamIdentity(
        run_id=run_id,
        stream_id=durable_model_stream_id(run_id, context.turn_id),
        logical_call_id=logical_model_call_id(run_id, context.turn_id),
        channel="output",
    )
    reasoning = DurableStreamIdentity(
        run_id=run_id,
        stream_id=output.stream_id,
        logical_call_id=output.logical_call_id,
        channel="reasoning",
    )
    expected = ((output, b"complete answer"), (reasoning, b"complete reasoning"))
    for identity, data in expected:
        prepared = harness.streams.read_after(identity, generation=1, cursor=0)
        assert b"".join(chunk.data for chunk in prepared.chunks) == data
        assert prepared.head is not None and prepared.head.state == "open"

    # Model invocation settlement can commit here while the stream generation remains open.
    # A new owner must observe all prepared bytes and seal them without a replacement reset.
    writer.abort()  # type: ignore[attr-defined]
    assert harness.authority.release(stale).status == "released"
    current = harness.claim(run_id, "prepared-recovery-worker")
    recovered = DurableModelStreamObserver(
        harness.streams,
        writer_token=current,
        write_authority=ActivationWriteAuthority(),
        chunk_bytes=8,
        flush_interval_s=10,
    ).open(context)
    recovered.close(ModelStreamOutcome(status="completed", final_text="complete answer"))

    for identity, data in expected:
        sealed = harness.streams.read_after(identity, generation=1, cursor=0)
        assert b"".join(chunk.data for chunk in sealed.chunks) == data
        assert sealed.head is not None and sealed.head.state == "sealed"


def test_model_observer_resets_abandoned_open_lanes_after_takeover(
    harness: _Harness,
) -> None:
    run_id = f"run-stream-abandoned-{uuid.uuid4().hex}"
    stale = harness.claim(run_id, "abandoned-worker")
    context = ModelStreamContext(
        run_id=run_id,
        root_run_id=run_id,
        turn_id="turn-abandoned",
        stream_id="execution-local-abandoned",
        step=1,
        provider="fixture",
        model="fixture-model",
        started_at="2026-08-25T00:00:00Z",
    )
    output = DurableStreamIdentity(
        run_id=run_id,
        stream_id=durable_model_stream_id(run_id, context.turn_id),
        logical_call_id=logical_model_call_id(run_id, context.turn_id),
        channel="output",
    )
    reasoning = DurableStreamIdentity(
        run_id=run_id,
        stream_id=output.stream_id,
        logical_call_id=output.logical_call_id,
        channel="reasoning",
    )
    for identity, data in ((output, b"stale output"), (reasoning, b"stale reasoning")):
        assert harness.streams.open(identity, writer_token=stale).status == "opened"
        assert harness.streams.append(
            identity,
            generation=1,
            start_offset=0,
            data=data,
            writer_token=stale,
        ).status == "committed"

    assert harness.authority.release(stale).status == "released"
    current = harness.claim(run_id, "replacement-worker")
    replacement = DurableModelStreamObserver(
        harness.streams,
        writer_token=current,
        write_authority=ActivationWriteAuthority(),
        chunk_bytes=16,
        flush_interval_s=10,
    ).open(context)
    replacement.push(ModelStreamDelta(channel="output", text="fresh"))
    replacement.close(ModelStreamOutcome(status="completed", final_text="fresh"))

    output_read = harness.streams.read_after(output, generation=2, cursor=0)
    reasoning_read = harness.streams.read_after(reasoning, generation=2, cursor=0)
    assert b"".join(chunk.data for chunk in output_read.chunks) == b"fresh"
    assert output_read.head is not None and output_read.head.state == "sealed"
    assert reasoning_read.chunks == ()
    assert reasoning_read.head is not None and reasoning_read.head.state == "sealed"
    assert harness.streams.read_after(output, generation=1, cursor=0).status == "reset"
    assert harness.streams.read_after(reasoning, generation=1, cursor=0).status == "reset"

    recovered_run = f"run-stream-recovered-output-{uuid.uuid4().hex}"
    recovered_stale = harness.claim(recovered_run, "recovered-output-worker")
    recovered_context = ModelStreamContext(
        run_id=recovered_run,
        root_run_id=recovered_run,
        turn_id="turn-recovered-output",
        stream_id="execution-local-recovered-output",
        step=1,
        provider="fixture",
        model="fixture-model",
        started_at="2026-08-25T00:00:00Z",
    )
    recovered_output = DurableStreamIdentity(
        run_id=recovered_run,
        stream_id=durable_model_stream_id(recovered_run, recovered_context.turn_id),
        logical_call_id=logical_model_call_id(
            recovered_run,
            recovered_context.turn_id,
        ),
        channel="output",
    )
    assert harness.streams.open(
        recovered_output,
        writer_token=recovered_stale,
    ).status == "opened"
    assert harness.streams.append(
        recovered_output,
        generation=1,
        start_offset=0,
        data=b"truncated ",
        writer_token=recovered_stale,
    ).status == "committed"
    assert harness.authority.release(recovered_stale).status == "released"
    recovered_current = harness.claim(recovered_run, "recovered-output-replacement")
    recovered_writer = DurableModelStreamObserver(
        harness.streams,
        writer_token=recovered_current,
        write_authority=ActivationWriteAuthority(),
        chunk_bytes=8,
        flush_interval_s=10,
    ).open(recovered_context)
    recovered_writer.close(
        ModelStreamOutcome(
            status="completed",
            final_text="authoritative 세계 output",
        )
    )
    recovered_read = harness.streams.read_after(
        recovered_output,
        generation=2,
        cursor=0,
    )
    assert b"".join(chunk.data for chunk in recovered_read.chunks).decode() == (
        "authoritative 세계 output"
    )
    assert recovered_read.head is not None and recovered_read.head.state == "sealed"

    empty_run = f"run-stream-empty-dispatch-{uuid.uuid4().hex}"
    empty_stale = harness.claim(empty_run, "empty-dispatch-worker")
    empty_context = ModelStreamContext(
        run_id=empty_run,
        root_run_id=empty_run,
        turn_id="turn-empty-dispatch",
        stream_id="execution-local-empty-dispatch",
        step=1,
        provider="fixture",
        model="fixture-model",
        started_at="2026-08-25T00:00:00Z",
    )
    empty_output = DurableStreamIdentity(
        run_id=empty_run,
        stream_id=durable_model_stream_id(empty_run, empty_context.turn_id),
        logical_call_id=logical_model_call_id(empty_run, empty_context.turn_id),
        channel="output",
    )
    empty_reasoning = DurableStreamIdentity(
        run_id=empty_run,
        stream_id=empty_output.stream_id,
        logical_call_id=empty_output.logical_call_id,
        channel="reasoning",
    )
    for identity, data in (
        (empty_output, b"stale output"),
        (empty_reasoning, b"stale reasoning"),
    ):
        assert harness.streams.open(identity, writer_token=empty_stale).status == "opened"
        assert harness.streams.append(
            identity,
            generation=1,
            start_offset=0,
            data=data,
            writer_token=empty_stale,
        ).status == "committed"

    assert harness.authority.release(empty_stale).status == "released"
    empty_current = harness.claim(empty_run, "empty-dispatch-replacement")
    empty_writer = DurableModelStreamObserver(
        harness.streams,
        writer_token=empty_current,
        write_authority=ActivationWriteAuthority(),
        chunk_bytes=8,
        flush_interval_s=10,
    ).open(empty_context)
    empty_writer.begin_dispatch()
    empty_writer.close(ModelStreamOutcome(status="completed", final_text=None))

    for identity in (empty_output, empty_reasoning):
        current_read = harness.streams.read_after(identity, generation=2, cursor=0)
        assert current_read.chunks == ()
        assert current_read.head is not None and current_read.head.state == "sealed"
        assert harness.streams.read_after(identity, generation=1, cursor=0).status == "reset"

    settled_run = f"run-stream-settled-recovery-{uuid.uuid4().hex}"
    settled_stale = harness.claim(settled_run, "settled-recovery-worker")
    settled_context = ModelStreamContext(
        run_id=settled_run,
        root_run_id=settled_run,
        turn_id="turn-settled-recovery",
        stream_id="execution-local-settled-recovery",
        step=1,
        provider="fixture",
        model="fixture-model",
        started_at="2026-08-25T00:00:00Z",
    )
    settled_output = DurableStreamIdentity(
        run_id=settled_run,
        stream_id=durable_model_stream_id(settled_run, settled_context.turn_id),
        logical_call_id=logical_model_call_id(settled_run, settled_context.turn_id),
        channel="output",
    )
    settled_reasoning = DurableStreamIdentity(
        run_id=settled_run,
        stream_id=settled_output.stream_id,
        logical_call_id=settled_output.logical_call_id,
        channel="reasoning",
    )
    for identity, data in (
        (settled_output, b"settled answer"),
        (settled_reasoning, b"settled thought"),
    ):
        assert harness.streams.open(identity, writer_token=settled_stale).status == "opened"
        assert harness.streams.append(
            identity,
            generation=1,
            start_offset=0,
            data=data,
            writer_token=settled_stale,
        ).status == "committed"
        assert harness.streams.seal(
            identity,
            generation=1,
            final_size_bytes=len(data),
            final_sha256=hashlib.sha256(data).hexdigest(),
            writer_token=settled_stale,
        ).status == "sealed"

    assert harness.authority.release(settled_stale).status == "released"
    settled_current = harness.claim(settled_run, "settled-recovery-replacement")
    settled_writer = DurableModelStreamObserver(
        harness.streams,
        writer_token=settled_current,
        write_authority=ActivationWriteAuthority(),
        chunk_bytes=8,
        flush_interval_s=10,
    ).open(settled_context)
    settled_writer.close(
        ModelStreamOutcome(status="completed", final_text="settled answer")
    )

    for identity, data in (
        (settled_output, b"settled answer"),
        (settled_reasoning, b"settled thought"),
    ):
        settled_read = harness.streams.read_after(identity, generation=1, cursor=0)
        assert b"".join(chunk.data for chunk in settled_read.chunks) == data
        assert settled_read.head is not None and settled_read.head.state == "sealed"

    failed_run = f"run-stream-failed-recovery-{uuid.uuid4().hex}"
    failed_stale = harness.claim(failed_run, "failed-recovery-worker")
    failed_context = ModelStreamContext(
        run_id=failed_run,
        root_run_id=failed_run,
        turn_id="turn-failed-recovery",
        stream_id="execution-local-failed-recovery",
        step=1,
        provider="fixture",
        model="fixture-model",
        started_at="2026-08-25T00:00:00Z",
    )
    failed_output = DurableStreamIdentity(
        run_id=failed_run,
        stream_id=durable_model_stream_id(failed_run, failed_context.turn_id),
        logical_call_id=logical_model_call_id(failed_run, failed_context.turn_id),
        channel="output",
    )
    failed_reasoning = DurableStreamIdentity(
        run_id=failed_run,
        stream_id=failed_output.stream_id,
        logical_call_id=failed_output.logical_call_id,
        channel="reasoning",
    )
    failed_prior = (
        (failed_output, b"partial output"),
        (failed_reasoning, b"partial reasoning"),
    )
    for identity, data in failed_prior:
        assert harness.streams.open(identity, writer_token=failed_stale).status == "opened"
        assert harness.streams.append(
            identity,
            generation=1,
            start_offset=0,
            data=data,
            writer_token=failed_stale,
        ).status == "committed"

    assert harness.authority.release(failed_stale).status == "released"
    failed_current = harness.claim(failed_run, "failed-recovery-replacement")
    failed_writer = DurableModelStreamObserver(
        harness.streams,
        writer_token=failed_current,
        write_authority=ActivationWriteAuthority(),
        chunk_bytes=8,
        flush_interval_s=10,
    ).open(failed_context)
    failed_writer.close(ModelStreamOutcome(status="failed", error_code="provider_error"))

    for identity, data in failed_prior:
        failed_read = harness.streams.read_after(identity, generation=1, cursor=0)
        assert b"".join(chunk.data for chunk in failed_read.chunks) == data
        assert failed_read.head is not None and failed_read.head.state == "sealed"


def test_rejected_appends_do_not_upload_unassociated_bytes(harness: _Harness) -> None:
    counting = _CountingPutStore(harness.streams.object_store)
    streams = PostgresObjectStoreDurableStreamStore(
        harness.database,
        counting,  # type: ignore[arg-type]
    )
    streams.check_ready()
    run_id = f"run-stream-precheck-{uuid.uuid4().hex}"
    token = harness.claim(run_id, "precheck-worker")
    identity = _identity(run_id, suffix="precheck")
    assert streams.open(identity, writer_token=token).status == "opened"

    assert streams.append(
        identity,
        generation=1,
        start_offset=99,
        data=b"gap",
        writer_token=token,
    ).status == "gap"
    assert counting.puts == 0

    data = b"accepted"
    assert streams.append(
        identity,
        generation=1,
        start_offset=0,
        data=data,
        writer_token=token,
    ).status == "committed"
    assert counting.puts == 1
    assert streams.seal(
        identity,
        generation=1,
        final_size_bytes=len(data),
        final_sha256=hashlib.sha256(data).hexdigest(),
        writer_token=token,
    ).status == "sealed"
    assert streams.append(
        identity,
        generation=1,
        start_offset=len(data),
        data=b"after-seal",
        writer_token=token,
    ).status == "sealed"
    assert counting.puts == 1

    assert streams.reset(
        identity,
        expected_generation=1,
        reset_id="precheck-reset",
        writer_token=token,
    ).status == "reset"
    assert streams.append(
        identity,
        generation=1,
        start_offset=len(data),
        data=b"old-generation",
        writer_token=token,
    ).status == "old_generation"
    assert counting.puts == 1

    terminal_run = f"run-stream-precheck-terminal-{uuid.uuid4().hex}"
    terminal_token = harness.claim(terminal_run, "terminal-precheck-worker")
    terminal_identity = _identity(terminal_run, suffix="terminal-precheck")
    assert streams.open(terminal_identity, writer_token=terminal_token).status == "opened"
    assert harness.sink.settle_terminal(
        TerminalOutcome(
            run_id=terminal_run,
            kind="completed",
            retry_eligibility=RetryEligibility.NOT_APPLICABLE,
        ),
        writer_token=terminal_token,
    ).status == "committed"
    assert streams.append(
        terminal_identity,
        generation=1,
        start_offset=0,
        data=b"after-terminal",
        writer_token=terminal_token,
    ).status == "run_terminal"
    assert counting.puts == 1

    stale_run = f"run-stream-precheck-stale-{uuid.uuid4().hex}"
    stale = harness.claim(stale_run, "stale-precheck-worker")
    stale_identity = _identity(stale_run, suffix="stale-precheck")
    assert streams.open(stale_identity, writer_token=stale).status == "opened"
    assert harness.authority.release(stale).status == "released"
    harness.claim(stale_run, "current-precheck-worker")
    assert streams.append(
        stale_identity,
        generation=1,
        start_offset=0,
        data=b"stale-writer",
        writer_token=stale,
    ).status == "fenced"
    assert counting.puts == 1


def test_terminal_settlement_linearizes_after_inflight_chunk_upload(
    harness: _Harness,
) -> None:
    blocking = _BlockingPutStore(harness.streams.object_store)
    streams = PostgresObjectStoreDurableStreamStore(
        harness.database,
        blocking,  # type: ignore[arg-type]
    )
    streams.check_ready()
    run_id = f"run-stream-upload-fence-{uuid.uuid4().hex}"
    token = harness.claim(run_id, "upload-fence-worker")
    identity = _identity(run_id, suffix="upload-fence")
    assert streams.open(identity, writer_token=token).status == "opened"
    terminal = TerminalOutcome(
        run_id=run_id,
        kind="completed",
        retry_eligibility=RetryEligibility.NOT_APPLICABLE,
    )
    terminal_entered = threading.Event()

    def settle_terminal() -> object:
        terminal_entered.set()
        return harness.sink.settle_terminal(terminal, writer_token=token)

    with ThreadPoolExecutor(max_workers=2) as executor:
        append_future = executor.submit(
            lambda: streams.append(
                identity,
                generation=1,
                start_offset=0,
                data=b"inflight",
                writer_token=token,
            )
        )
        assert blocking.entered.wait(10)
        terminal_future = executor.submit(settle_terminal)
        assert terminal_entered.wait(10)
        try:
            with pytest.raises(TimeoutError):
                terminal_future.result(timeout=0.2)
        finally:
            blocking.release.set()
        assert append_future.result(timeout=20).status == "committed"
        assert terminal_future.result(timeout=20).status == "committed"

    assert blocking.puts == 1
    assert streams.append(
        identity,
        generation=1,
        start_offset=len(b"inflight"),
        data=b"late",
        writer_token=token,
    ).status == "run_terminal"
    assert blocking.puts == 1


def test_gc_delete_between_put_check_and_digest_lock_cannot_publish_chunk(
    harness: _Harness,
) -> None:
    data = b"digest-lock-race"
    sha256 = hashlib.sha256(data).hexdigest()
    delegate = harness.streams.object_store
    delegate.put_if_absent(sha256, data)  # type: ignore[attr-defined]
    admin = S3ObjectStoreAdmin(delegate.config, client=_client())  # type: ignore[attr-defined]
    collector = PostgresObjectGarbageCollector(harness.database, admin)
    collector.check_ready()
    plan = collector.plan(grace_period=timedelta(0))
    assert [candidate.sha256 for candidate in plan.candidates] == [sha256]

    blocking = _BlockingStatStore(delegate)
    streams = PostgresObjectStoreDurableStreamStore(
        harness.database,
        blocking,  # type: ignore[arg-type]
    )
    streams.check_ready()
    run_id = f"run-stream-gc-race-{uuid.uuid4().hex}"
    token = harness.claim(run_id, "gc-race-worker")
    identity = _identity(run_id, suffix="gc-race")
    assert streams.open(identity, writer_token=token).status == "opened"

    with ThreadPoolExecutor(max_workers=1) as executor:
        append_future = executor.submit(
            lambda: streams.append(
                identity,
                generation=1,
                start_offset=0,
                data=data,
                writer_token=token,
            )
        )
        assert blocking.captured.wait(10)
        try:
            receipts = collector.apply(plan)
        finally:
            blocking.release.set()
        assert [receipt.status for receipt in receipts] == ["deleted"]
        with pytest.raises(PostgresDurableStreamCorrupt, match="digest-locked association"):
            append_future.result(timeout=20)

    assert delegate.stat(sha256) is None  # type: ignore[attr-defined]
    replay = streams.read_after(identity, generation=1, cursor=0)
    assert replay.status == "ok"
    assert replay.head is not None and replay.head.cursor_bytes == 0
    assert replay.chunks == ()


def test_actual_stream_terminal_reset_races_and_slow_seal_do_not_starve_renewal(
    harness: _Harness,
) -> None:
    terminal_run = f"run-stream-terminal-race-{uuid.uuid4().hex}"
    terminal_token = harness.claim(terminal_run, "terminal-racer")
    terminal_identity = _identity(terminal_run, suffix="terminal-race")
    assert harness.streams.open(
        terminal_identity,
        writer_token=terminal_token,
    ).status == "opened"
    barrier = threading.Barrier(2)

    def append_against_terminal() -> str:
        barrier.wait(timeout=10)
        return harness.streams.append(
            terminal_identity,
            generation=1,
            start_offset=0,
            data=b"racing-delta",
            writer_token=terminal_token,
        ).status

    def settle_terminal() -> str:
        barrier.wait(timeout=10)
        return harness.sink.settle_terminal(
            TerminalOutcome(
                run_id=terminal_run,
                kind="completed",
                retry_eligibility=RetryEligibility.NOT_APPLICABLE,
            ),
            writer_token=terminal_token,
        ).status

    with ThreadPoolExecutor(max_workers=2) as executor:
        append_future = executor.submit(append_against_terminal)
        terminal_future = executor.submit(settle_terminal)
        append_status = append_future.result(timeout=20)
        terminal_status = terminal_future.result(timeout=20)
    assert terminal_status == "committed"
    assert append_status in {"committed", "run_terminal"}
    assert harness.streams.append(
        terminal_identity,
        generation=1,
        start_offset=(len(b"racing-delta") if append_status == "committed" else 0),
        data=b"always-late",
        writer_token=terminal_token,
    ).status == "run_terminal"

    reset_run = f"run-stream-reset-race-{uuid.uuid4().hex}"
    reset_token = harness.claim(reset_run, "reset-racer")
    reset_identity = _identity(reset_run, suffix="reset-race")
    assert harness.streams.open(reset_identity, writer_token=reset_token).status == "opened"
    reset_barrier = threading.Barrier(2)

    def append_against_reset() -> str:
        reset_barrier.wait(timeout=10)
        return harness.streams.append(
            reset_identity,
            generation=1,
            start_offset=0,
            data=b"generation-one",
            writer_token=reset_token,
        ).status

    def reset_generation() -> str:
        reset_barrier.wait(timeout=10)
        return harness.streams.reset(
            reset_identity,
            expected_generation=1,
            reset_id="concurrent-reset",
            writer_token=reset_token,
        ).status

    with ThreadPoolExecutor(max_workers=2) as executor:
        append_future = executor.submit(append_against_reset)
        reset_future = executor.submit(reset_generation)
        append_status = append_future.result(timeout=20)
        reset_status = reset_future.result(timeout=20)
    assert reset_status == "reset"
    assert append_status in {"committed", "old_generation"}
    assert harness.streams.read_after(reset_identity, generation=1, cursor=0).status == "reset"

    receipt_run = f"run-stream-reset-receipt-{uuid.uuid4().hex}"
    receipt_token = harness.claim(receipt_run, "receipt-racer")
    receipt_identity = _identity(receipt_run, suffix="reset-receipt")
    assert harness.streams.open(receipt_identity, writer_token=receipt_token).status == "opened"
    receipt_barrier = threading.Barrier(2)

    def same_reset() -> str:
        receipt_barrier.wait(timeout=10)
        return harness.streams.reset(
            receipt_identity,
            expected_generation=1,
            reset_id="same-reset-request",
            writer_token=receipt_token,
        ).status

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_reset = executor.submit(same_reset)
        second_reset = executor.submit(same_reset)
        statuses = sorted(
            (first_reset.result(timeout=20), second_reset.result(timeout=20))
        )
    assert statuses == ["already_reset", "reset"]

    seal_run = f"run-stream-slow-seal-{uuid.uuid4().hex}"
    seal_token = harness.claim(seal_run, "seal-worker")
    blocking_store = _BlockingGetStore(harness.streams.object_store)
    sealing_streams = PostgresObjectStoreDurableStreamStore(
        harness.database,
        blocking_store,  # type: ignore[arg-type]
    )
    sealing_streams.check_ready()
    seal_identity = _identity(seal_run, suffix="slow-seal")
    assert sealing_streams.open(seal_identity, writer_token=seal_token).status == "opened"
    seal_bytes = b"checked outside authority lock"
    assert sealing_streams.append(
        seal_identity,
        generation=1,
        start_offset=0,
        data=seal_bytes,
        writer_token=seal_token,
    ).status == "committed"

    with ThreadPoolExecutor(max_workers=2) as executor:
        seal_future = executor.submit(
            sealing_streams.seal,
            seal_identity,
            generation=1,
            final_size_bytes=len(seal_bytes),
            final_sha256=hashlib.sha256(seal_bytes).hexdigest(),
            writer_token=seal_token,
        )
        assert blocking_store.entered.wait(10)
        renew_future = executor.submit(
            harness.authority.renew,
            seal_token,
            timedelta(minutes=5),
        )
        assert renew_future.result(timeout=2).status == "renewed"
        blocking_store.release.set()
        assert seal_future.result(timeout=20).status == "sealed"
