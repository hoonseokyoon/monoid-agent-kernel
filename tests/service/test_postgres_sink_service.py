from __future__ import annotations

import hashlib
import math
import os
import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.serial, pytest.mark.service]


def _selected() -> bool:
    return os.environ.get("MONOID_SERVICE_PROFILE") in {"postgres", "objectstore", "combined"}


if not _selected():
    pytest.skip("PostgreSQL service profile is not selected", allow_module_level=True)

import psycopg  # noqa: E402
from support.runtime import runtime_config, runtime_provider  # noqa: E402
from psycopg import sql  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402
from psycopg.types.json import Json  # noqa: E402
from psycopg_pool import ConnectionPool  # noqa: E402

from monoid_agent_kernel.adapters.postgres import (  # noqa: E402
    PostgresBlobCorrupt,
    PostgresConfig,
    PostgresDatabase,
    PostgresFencedRunSink,
    PostgresMigrations,
    PostgresObjectGarbageCollector,
    PostgresObjectStoreFencedRunSink,
    PostgresSchemaIncompatible,
    PostgresWriterAuthorityStore,
)
from monoid_agent_kernel.core._util import canonical_sha256  # noqa: E402
from monoid_agent_kernel.core.checkpoint import (  # noqa: E402
    LocalFsCheckpointStore,
    RunCheckpoint,
)
from monoid_agent_kernel.core.events import AgentEvent, make_agent_event  # noqa: E402
from monoid_agent_kernel.core.model_invocation import (  # noqa: E402
    MODEL_REQUEST_DIGEST_GENERATION,
    DurableModelInvocation,
)
from monoid_agent_kernel.core.outcome import (  # noqa: E402
    RetryEligibility,
    TerminalOutcome,
)
from monoid_agent_kernel.core.spec import AgentRunSpec  # noqa: E402
from monoid_agent_kernel.conformance import run_fenced_run_sink_contract  # noqa: E402
from monoid_agent_kernel.errors import ModelAdapterError  # noqa: E402
from monoid_agent_kernel.hosting import (  # noqa: E402
    ActivationCommand,
    ActivationDriver,
    ActivationRuntime,
    BlobNotFound,
    BlobTooLarge,
    CommitResult,
    WriterToken,
)
from monoid_agent_kernel.hosting.model_calls import FencedModelCallLifecycle  # noqa: E402
from monoid_agent_kernel.loop import AgentLoop  # noqa: E402
from monoid_agent_kernel.model_lifecycle import ModelDispatchReservation  # noqa: E402
from monoid_agent_kernel.providers.base import ModelTurn  # noqa: E402


_POSTGRES_TARGETS = [
    (
        "MONOID_POSTGRES16_DSN",
        16,
        {"postgres", "objectstore", "combined"},
    ),
    (
        "MONOID_POSTGRES18_DSN",
        18,
        {"combined"},
    ),
]


class _RetryableActivationAdapter:
    def next_turn(self, request: Any) -> ModelTurn:
        del request
        raise ModelAdapterError(
            "private provider failure",
            error_code="provider_unavailable",
            retryable=True,
        )


@dataclass
class _CountingActivationAdapter:
    calls: int = 0

    def next_turn(self, request: Any) -> ModelTurn:
        del request
        self.calls += 1
        return ModelTurn(final_text="ok")


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
class _SinkHarness:
    database: PostgresDatabase
    authority: PostgresWriterAuthorityStore
    sink: PostgresFencedRunSink

    def claim(self, run_id: str, owner_id: str = "worker-a") -> WriterToken:
        return self.authority.claim(
            run_id,
            owner_id,
            timedelta(seconds=30),
        ).writer_token

    def set_current_writer(self, writer_token: WriterToken) -> None:
        """Install an exact contract token while retaining PostgreSQL as authority."""

        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                cursor.execute(
                    sql.SQL(
                        "WITH sampled AS (SELECT pg_catalog.clock_timestamp() AS db_now) "
                        "INSERT INTO {} "
                        "(run_id, owner_id, generation, leased_until, revoked, updated_at) "
                        "SELECT %s, %s, %s, db_now + interval '5 minutes', false, db_now "
                        "FROM sampled ON CONFLICT (run_id) DO UPDATE SET "
                        "owner_id = EXCLUDED.owner_id, generation = EXCLUDED.generation, "
                        "leased_until = EXCLUDED.leased_until, revoked = false, "
                        "updated_at = EXCLUDED.updated_at"
                    ).format(sql.Identifier(self.database.config.schema, "run_authority")),
                    (
                        writer_token.run_id,
                        writer_token.owner_id,
                        writer_token.generation,
                    ),
                )

    def reopen(self) -> _SinkHarness:
        sink = PostgresFencedRunSink(self.database)
        sink.check_ready()
        authority = PostgresWriterAuthorityStore(self.database)
        authority.check_ready()
        return _SinkHarness(database=self.database, authority=authority, sink=sink)

    def inject_authoritative_load_fault(
        self,
        record_family: Literal["checkpoint", "invocation"],
        run_id: str,
        status: Literal["corrupt", "unsupported_version"],
        *,
        logical_call_id: str = "",
    ) -> None:
        if record_family == "checkpoint":
            table = "checkpoint_record"
            identity_sql = "run_id = %s AND sequence = (SELECT sequence FROM {} WHERE run_id = %s)"
            head_table = "checkpoint_head"
            parameters = (run_id, run_id)
            future_schema = "monoid.checkpoint.v999"
        elif record_family == "invocation":
            table = "invocation_record"
            identity_sql = (
                "run_id = %s AND logical_call_id = %s AND revision = "
                "(SELECT revision FROM {} WHERE run_id = %s AND logical_call_id = %s)"
            )
            head_table = "invocation_head"
            parameters = (run_id, logical_call_id, run_id, logical_call_id)
            future_schema = "monoid.model-invocation.v999"
        else:  # pragma: no cover - conformance protocol constrains this value
            raise ValueError("unknown PostgreSQL fault family")
        rendered_identity = sql.SQL(identity_sql).format(
            sql.Identifier(self.database.config.schema, head_table)
        )
        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                if status == "corrupt":
                    cursor.execute(
                        sql.SQL("UPDATE {} SET content_digest = %s WHERE ").format(
                            sql.Identifier(self.database.config.schema, table)
                        )
                        + rendered_identity,
                        ("0" * 64, *parameters),
                    )
                    if cursor.rowcount != 1:
                        raise ValueError("load fault requires an authoritative record")
                    return
                cursor.execute(
                    sql.SQL("SELECT payload, submitted_blobs FROM {} WHERE ").format(
                        sql.Identifier(self.database.config.schema, table)
                    )
                    + rendered_identity,
                    parameters,
                )
                row = cursor.fetchone()
                if row is None:
                    raise ValueError("load fault requires an authoritative record")
                payload = {**dict(row[0]), "schema_version": future_schema}
                content_digest = canonical_sha256({"record": payload, "blobs": row[1]})
                cursor.execute(
                    sql.SQL(
                        "UPDATE {} SET payload = %s, schema_version = %s, "
                        "content_digest = %s WHERE "
                    ).format(sql.Identifier(self.database.config.schema, table))
                    + rendered_identity,
                    (Json(payload), future_schema, content_digest, *parameters),
                )

    def read_event(self, run_id: str, seq: int) -> AgentEvent | None:
        return self.sink.read_event(run_id, seq)

    def read_terminal(self, run_id: str) -> TerminalOutcome | None:
        return self.sink.read_terminal(run_id)

    def close(self) -> None:
        """The pytest fixture owns the shared database and pool lifecycle."""

    def race_conflicting_writes(
        self,
        mutation: str,
        writer_token: WriterToken,
        left: Callable[[PostgresFencedRunSink], CommitResult],
        right: Callable[[PostgresFencedRunSink], CommitResult],
    ) -> tuple[CommitResult, CommitResult]:
        del mutation, writer_token
        barrier = threading.Barrier(3)
        left_sink = self.reopen().sink
        right_sink = self.reopen().sink

        def invoke(
            operation: Callable[[PostgresFencedRunSink], CommitResult],
            sink: PostgresFencedRunSink,
        ) -> CommitResult:
            barrier.wait(timeout=10)
            return operation(sink)

        with ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(invoke, left, left_sink)
            right_future = executor.submit(invoke, right, right_sink)
            barrier.wait(timeout=10)
            return left_future.result(timeout=20), right_future.result(timeout=20)

    def race_writer_handoff(
        self,
        mutation: str,
        stale_token: WriterToken,
        current_token: WriterToken,
        write: Callable[[PostgresFencedRunSink, WriterToken], CommitResult],
    ) -> tuple[CommitResult, CommitResult, bool]:
        del mutation
        schema = self.database.config.schema
        stale_sink = self.reopen().sink
        with ThreadPoolExecutor(max_workers=1) as executor:
            with self.database.connection() as blocker:
                with blocker.transaction():
                    with blocker.cursor() as cursor:
                        cursor.execute(
                            sql.SQL("SELECT run_id FROM {} WHERE run_id = %s FOR UPDATE").format(
                                sql.Identifier(schema, "run_authority")
                            ),
                            (stale_token.run_id,),
                        )
                        future = executor.submit(write, stale_sink, stale_token)
                        _wait_for_lock(
                            self.database,
                            self.database.config.application_name,
                        )
                        cursor.execute(
                            sql.SQL(
                                "UPDATE {} SET owner_id = %s, generation = %s, "
                                "leased_until = pg_catalog.clock_timestamp() + "
                                "interval '5 minutes', revoked = false, "
                                "updated_at = pg_catalog.clock_timestamp() WHERE run_id = %s"
                            ).format(sql.Identifier(schema, "run_authority")),
                            (
                                current_token.owner_id,
                                current_token.generation,
                                current_token.run_id,
                            ),
                        )
                stale_result = future.result(timeout=20)
        current_result = write(self.reopen().sink, current_token)
        return stale_result, current_result, True


@dataclass
class _ExternalSinkHarness(_SinkHarness):
    object_store: object
    object_admin: object

    def reopen(self) -> _ExternalSinkHarness:
        sink = PostgresObjectStoreFencedRunSink(self.database, self.object_store)  # type: ignore[arg-type]
        sink.check_ready()
        authority = PostgresWriterAuthorityStore(self.database)
        authority.check_ready()
        return _ExternalSinkHarness(
            database=self.database,
            authority=authority,
            sink=sink,
            object_store=self.object_store,
            object_admin=self.object_admin,
        )


@pytest.fixture
def sink_harness(postgres_target: tuple[str, int]) -> Iterator[_SinkHarness]:
    dsn, _ = postgres_target
    schema = f"monoid_pr04_{uuid.uuid4().hex}"
    database = PostgresDatabase(
        PostgresConfig(
            dsn=dsn,
            schema=schema,
            min_pool_size=1,
            max_pool_size=8,
            pool_timeout_s=10,
            application_name="monoid-pr04-service-test",
            max_bytea_blob_bytes=64,
        )
    )
    database.open()
    PostgresMigrations(database).apply()
    authority = PostgresWriterAuthorityStore(database)
    authority.check_ready()
    sink = PostgresFencedRunSink(database)
    sink.check_ready()
    try:
        yield _SinkHarness(database=database, authority=authority, sink=sink)
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


@pytest.fixture
def activation_sink(sink_harness: _SinkHarness) -> Iterator[PostgresFencedRunSink]:
    config = replace(
        sink_harness.database.config,
        application_name="monoid-pr07-activation-test",
        max_bytea_blob_bytes=1024 * 1024,
    )
    with PostgresDatabase(config) as database:
        sink = PostgresFencedRunSink(database)
        sink.check_ready()
        yield sink


@pytest.fixture
def external_sink_harness(postgres_target: tuple[str, int]) -> Iterator[_ExternalSinkHarness]:
    if os.environ["MONOID_SERVICE_PROFILE"] not in {"objectstore", "combined"}:
        pytest.skip("external object association requires the object-store profile")
    import boto3
    from botocore import config as botocore_config

    from monoid_agent_kernel.adapters.object_store import (
        S3ContentAddressedBlobStore,
        S3ObjectStoreAdmin,
        S3ObjectStoreConfig,
    )

    endpoint = os.environ.get("MONOID_MINIO_ENDPOINT")
    if not endpoint:
        pytest.fail("MONOID_MINIO_ENDPOINT is required for the selected service profile")
    client = boto3.client(
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
    bucket = f"monoid-v023-association-{uuid.uuid4().hex}"
    client.create_bucket(Bucket=bucket)
    client.put_bucket_versioning(
        Bucket=bucket,
        VersioningConfiguration={"Status": "Enabled"},
    )
    object_config = S3ObjectStoreConfig(
            bucket=bucket,
            prefix="external-sink",
            endpoint_url=endpoint,
            addressing_style="path",
            multipart_threshold_bytes=6 * 1024 * 1024,
            multipart_part_bytes=5 * 1024 * 1024,
            max_object_bytes=32 * 1024 * 1024,
        )
    object_store = S3ContentAddressedBlobStore(
        object_config,
        client=client,
    )
    object_admin = S3ObjectStoreAdmin(object_config, client=client)
    dsn, _ = postgres_target
    schema = f"monoid_pr06_{uuid.uuid4().hex}"
    database = PostgresDatabase(
        PostgresConfig(
            dsn=dsn,
            schema=schema,
            min_pool_size=1,
            max_pool_size=8,
            pool_timeout_s=10,
            application_name="monoid-pr06-service-test",
        )
    )
    database.open()
    PostgresMigrations(database).apply()
    authority = PostgresWriterAuthorityStore(database)
    authority.check_ready()
    sink = PostgresObjectStoreFencedRunSink(database, object_store)
    sink.check_ready()
    harness = _ExternalSinkHarness(
        database=database,
        authority=authority,
        sink=sink,
        object_store=object_store,
        object_admin=object_admin,
    )
    try:
        yield harness
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
            uploads = client.list_multipart_uploads(Bucket=bucket).get("Uploads", [])
            for upload in uploads:
                client.abort_multipart_upload(
                    Bucket=bucket,
                    Key=upload["Key"],
                    UploadId=upload["UploadId"],
                )
            objects = client.list_objects_v2(Bucket=bucket).get("Contents", [])
            for value in objects:
                client.delete_object(Bucket=bucket, Key=value["Key"])
            versions = client.list_object_versions(Bucket=bucket)
            for value in (*versions.get("Versions", []), *versions.get("DeleteMarkers", [])):
                client.delete_object(
                    Bucket=bucket,
                    Key=value["Key"],
                    VersionId=value["VersionId"],
                )
            client.delete_bucket(Bucket=bucket)


def _blob(data: bytes) -> tuple[str, dict[str, bytes]]:
    digest = hashlib.sha256(data).hexdigest()
    return digest, {digest: data}


def _checkpoint_with_blob(run_id: str, sequence: int, sha256: str, text: str) -> RunCheckpoint:
    return RunCheckpoint(
        run_id=run_id,
        seq=sequence,
        final_text=text,
        workspace_delta=[
            {
                "path": f"checkpoint-{sequence}.txt",
                "kind": "file",
                "change_kind": "created",
                "content_sha256": sha256,
            }
        ],
    )


def _invocation(
    run_id: str,
    revision: int,
    state: str,
    *,
    logical_call_id: str = "call-1",
    attempt: int = 1,
    dispatch_id: str = "dispatch-1",
    retryable: bool = False,
    succeeded_blob: str = "",
    request_digest: str = "a" * 64,
    duration_ms: float | None = None,
    evidence_policy: str = "passive",
) -> DurableModelInvocation:
    receipt = None
    result_ref = ""
    failure_code = ""
    if state == "settled":
        receipt = {"request_digest": request_digest, "retryable": retryable}
        if duration_ms is not None:
            receipt["duration_ms"] = duration_ms
        if succeeded_blob:
            result_ref = f"blob:{succeeded_blob}"
        else:
            failure_code = "provider_refused"
    return DurableModelInvocation(
        run_id=run_id,
        logical_call_id=logical_call_id,
        revision=revision,
        dispatch_id=dispatch_id,
        dispatch_attempt=attempt,
        idempotency_key=f"key-{hashlib.sha256(run_id.encode()).hexdigest()}",
        dispatch_state=state,  # type: ignore[arg-type]
        request_digest=request_digest,
        digest_generation=MODEL_REQUEST_DIGEST_GENERATION,
        evidence_policy=evidence_policy,  # type: ignore[arg-type]
        receipt=receipt,
        result_ref=result_ref,
        failure_code=failure_code,
    )


def _table_count(harness: _SinkHarness, table: str, run_id: str) -> int:
    with harness.database.transaction() as connection:
        with harness.database.cursor(connection) as cursor:
            cursor.execute(
                sql.SQL("SELECT count(*) FROM {} WHERE run_id = %s").format(
                    sql.Identifier(harness.database.config.schema, table)
                ),
                (run_id,),
            )
            return int(cursor.fetchone()[0])


def _rewrite_payload_without_digest(
    harness: _SinkHarness,
    table: str,
    identity_sql: str,
    identity_parameters: tuple[object, ...],
    mutate: Callable[[dict[str, object]], dict[str, object]],
) -> None:
    with harness.database.transaction() as connection:
        with harness.database.cursor(connection) as cursor:
            cursor.execute(
                sql.SQL(f"SELECT payload FROM {{}} WHERE {identity_sql}").format(
                    sql.Identifier(harness.database.config.schema, table)
                ),
                identity_parameters,
            )
            changed = mutate(dict(cursor.fetchone()[0]))
            cursor.execute(
                sql.SQL(f"UPDATE {{}} SET payload = %s WHERE {identity_sql}").format(
                    sql.Identifier(harness.database.config.schema, table)
                ),
                (Json(changed), *identity_parameters),
            )


def test_global_blob_rows_are_acquired_in_digest_order(
    sink_harness: _SinkHarness,
) -> None:
    class RecordingCursor:
        def __init__(self) -> None:
            self.inserted_digests: list[str] = []

        def execute(self, _query: object, parameters: tuple[object, ...]) -> None:
            if len(parameters) == 3:
                self.inserted_digests.append(str(parameters[0]))

        def fetchone(self) -> tuple[str]:
            return (self.inserted_digests[-1],)

    left_sha, left = _blob(b"left-lock")
    right_sha, right = _blob(b"right-lock")
    by_digest = {left_sha: left[left_sha], right_sha: right[right_sha]}
    reversed_input = dict(reversed(sorted(by_digest.items())))
    cursor = RecordingCursor()

    sink_harness.sink._persist_blobs(cursor, "run-lock-order", reversed_input)

    assert list(reversed_input) == sorted((left_sha, right_sha), reverse=True)
    assert cursor.inserted_digests == sorted((left_sha, right_sha))


def test_checkpoint_identity_blobs_monotonic_head_and_restart(
    sink_harness: _SinkHarness,
) -> None:
    run_id = "run-checkpoint"
    token = sink_harness.claim(run_id)
    sha256, blobs = _blob(b"checkpoint-private")
    first = _checkpoint_with_blob(run_id, 1, sha256, "winner")

    assert sink_harness.sink.latest_checked(run_id).status == "missing"
    committed = sink_harness.sink.commit_checkpoint(first, blobs, writer_token=token)
    assert committed.status == "committed"
    assert committed.sequence == 1
    assert sink_harness.sink.commit_checkpoint(first, blobs, writer_token=token).status == (
        "already_committed"
    )
    assert (
        sink_harness.sink.commit_checkpoint(
            replace(first, final_text="loser"),
            blobs,
            writer_token=token,
        ).status
        == "conflict"
    )

    loaded = sink_harness.sink.latest_checked(run_id)
    assert loaded.status == "loaded"
    assert loaded.sequence == 1
    assert loaded.value is not None
    assert loaded.value.checkpoint.final_text == "winner"
    assert loaded.value.blob(sha256) == b"checkpoint-private"

    assert (
        sink_harness.sink.commit_checkpoint(
            RunCheckpoint(run_id=run_id, seq=2, final_text="newer"),
            {},
            writer_token=token,
        ).status
        == "committed"
    )
    assert (
        sink_harness.sink.commit_checkpoint(
            RunCheckpoint(run_id=run_id, seq=0, final_text="delayed"),
            {},
            writer_token=token,
        ).status
        == "committed"
    )
    assert sink_harness.sink.latest_checked(run_id).value.checkpoint.final_text == "newer"  # type: ignore[union-attr]
    assert (
        sink_harness.sink.commit_checkpoint(
            _checkpoint_with_blob(run_id, 3, sha256, "same-run-reuse"),
            {},
            writer_token=token,
        ).status
        == "committed"
    )

    reopened_database = PostgresDatabase(sink_harness.database.config)
    reopened_database.open()
    try:
        reopened = PostgresFencedRunSink(reopened_database)
        reopened.check_ready()
        restarted = reopened.latest_checked(run_id)
        assert restarted.status == "loaded"
        assert restarted.sequence == 3
        assert restarted.value is not None
        assert restarted.value.blob(sha256) == b"checkpoint-private"
    finally:
        reopened_database.close()


def test_payload_json_preserves_negative_zero_and_checkpoint_nul(
    sink_harness: _SinkHarness,
) -> None:
    run_id = "run-json-identity"
    token = sink_harness.claim(run_id)
    checkpoint = RunCheckpoint(
        run_id=run_id,
        seq=1,
        final_text="before\x00after",
        remaining_duration_s=-0.0,
    )

    assert sink_harness.sink.commit_checkpoint(checkpoint, {}, writer_token=token).status == (
        "committed"
    )
    loaded = sink_harness.sink.latest_checked(run_id)
    assert loaded.status == "loaded"
    assert loaded.value is not None
    assert loaded.value.checkpoint.final_text == "before\x00after"
    remaining = loaded.value.checkpoint.remaining_duration_s
    assert remaining == 0.0
    assert math.copysign(1.0, remaining) == -1.0


def test_checkpoint_sequence_respects_postgres_bigint_boundaries(
    sink_harness: _SinkHarness,
) -> None:
    run_id = "run-bigint-bound"
    token = sink_harness.claim(run_id)

    maximum = sink_harness.sink.commit_checkpoint(
        RunCheckpoint(run_id=run_id, seq=(1 << 63) - 1),
        {},
        writer_token=token,
    )

    result = sink_harness.sink.commit_checkpoint(
        RunCheckpoint(run_id=run_id, seq=10**30),
        {},
        writer_token=token,
    )

    assert maximum.status == "committed"
    assert result.status == "conflict"
    assert result.sequence == 10**30
    assert _table_count(sink_harness, "checkpoint_record", run_id) == 1


@pytest.mark.parametrize("bad_sequence", [-1, "invalid", True])
def test_invalid_checkpoint_sequence_returns_conflict_without_evidence(
    sink_harness: _SinkHarness,
    bad_sequence: object,
) -> None:
    run_id = f"run-invalid-sequence-{type(bad_sequence).__name__}"
    token = sink_harness.claim(run_id)

    result = sink_harness.sink.commit_checkpoint(
        RunCheckpoint(run_id=run_id, seq=bad_sequence),  # type: ignore[arg-type]
        {},
        writer_token=token,
    )

    assert result.status == "conflict"
    assert result.sequence is None
    assert _table_count(sink_harness, "checkpoint_record", run_id) == 0


def test_invocation_revision_outside_postgres_bigint_is_conflict(
    sink_harness: _SinkHarness,
) -> None:
    run_id = "run-invocation-bigint-bound"
    token = sink_harness.claim(run_id)

    result = sink_harness.sink.commit_invocation(
        _invocation(run_id, 10**30, "reserved"),
        {},
        writer_token=token,
    )

    assert result.status == "conflict"
    assert result.sequence == 10**30
    assert _table_count(sink_harness, "invocation_record", run_id) == 0


def test_deleting_run_authority_cascades_records_and_heads(
    sink_harness: _SinkHarness,
) -> None:
    run_id = "run-authority-cascade"
    token = sink_harness.claim(run_id)
    assert (
        sink_harness.sink.commit_checkpoint(
            RunCheckpoint(run_id=run_id, seq=1),
            {},
            writer_token=token,
        ).status
        == "committed"
    )
    assert (
        sink_harness.sink.commit_invocation(
            _invocation(run_id, 1, "reserved"),
            {},
            writer_token=token,
        ).status
        == "committed"
    )

    with sink_harness.database.transaction() as connection:
        with sink_harness.database.cursor(connection) as cursor:
            cursor.execute(
                sql.SQL("DELETE FROM {} WHERE run_id = %s").format(
                    sql.Identifier(sink_harness.database.config.schema, "run_authority")
                ),
                (run_id,),
            )

    for table in (
        "run_authority",
        "checkpoint_record",
        "checkpoint_head",
        "invocation_record",
        "invocation_head",
    ):
        assert _table_count(sink_harness, table, run_id) == 0


def test_deleting_run_authority_cascades_complete_sink_records(
    sink_harness: _SinkHarness,
) -> None:
    run_id = "run-complete-sink-cascade"
    token = sink_harness.claim(run_id)
    event = make_agent_event(run_id=run_id, seq=1, event_type="run.finished")
    terminal = TerminalOutcome(
        run_id=run_id,
        kind="completed",
        retry_eligibility=RetryEligibility.NOT_APPLICABLE,
    )
    assert sink_harness.sink.append_event(event, writer_token=token).status == "committed"
    assert sink_harness.sink.settle_terminal(terminal, writer_token=token).status == "committed"

    required = tuple(
        _invocation(
            run_id,
            revision,
            state,
            logical_call_id="call-required-cascade",
            dispatch_id="dispatch-required-cascade",
            evidence_policy="required",
        )
        for revision, state in ((1, "reserved"), (2, "dispatch_started"), (3, "settled"))
    )
    for item in required:
        assert sink_harness.sink.commit_invocation(item, {}, writer_token=token).status == (
            "committed"
        )
    assert (
        sink_harness.sink.commit_model_evidence(required[-1], writer_token=token).status
        == "committed"
    )

    outbox = tuple(
        _invocation(
            run_id,
            revision,
            state,
            logical_call_id="call-outbox-cascade",
            dispatch_id="dispatch-outbox-cascade",
            evidence_policy="outbox",
        )
        for revision, state in ((1, "reserved"), (2, "dispatch_started"), (3, "settled"))
    )
    for item in outbox[:-1]:
        assert sink_harness.sink.commit_invocation(item, {}, writer_token=token).status == (
            "committed"
        )
    assert (
        sink_harness.sink.commit_invocation(
            outbox[-1],
            {},
            writer_token=token,
            stage_evidence=True,
        ).status
        == "committed"
    )

    with sink_harness.database.transaction() as connection:
        with sink_harness.database.cursor(connection) as cursor:
            cursor.execute(
                sql.SQL("DELETE FROM {} WHERE run_id = %s").format(
                    sql.Identifier(sink_harness.database.config.schema, "run_authority")
                ),
                (run_id,),
            )

    for table in (
        "run_authority",
        "event_record",
        "event_head",
        "terminal_record",
        "invocation_record",
        "invocation_head",
        "model_evidence_record",
        "model_evidence_outbox",
    ):
        assert _table_count(sink_harness, table, run_id) == 0


def test_corrupt_records_cannot_acknowledge_idempotent_retries(
    sink_harness: _SinkHarness,
) -> None:
    run_id = "run-corrupt-idempotence"
    token = sink_harness.claim(run_id)
    checkpoint = RunCheckpoint(run_id=run_id, seq=1, final_text="original")
    invocation = _invocation(run_id, 1, "reserved")
    assert (
        sink_harness.sink.commit_checkpoint(checkpoint, {}, writer_token=token).status
        == "committed"
    )
    assert (
        sink_harness.sink.commit_invocation(invocation, {}, writer_token=token).status
        == "committed"
    )

    _rewrite_payload_without_digest(
        sink_harness,
        "checkpoint_record",
        "run_id = %s AND sequence = %s",
        (run_id, 1),
        lambda payload: {**payload, "final_text": "corrupt"},
    )
    _rewrite_payload_without_digest(
        sink_harness,
        "invocation_record",
        "run_id = %s AND logical_call_id = %s AND revision = %s",
        (run_id, "call-1", 1),
        lambda payload: {**payload, "dispatch_state": "dispatch_started"},
    )

    assert (
        sink_harness.sink.commit_checkpoint(checkpoint, {}, writer_token=token).status == "conflict"
    )
    assert (
        sink_harness.sink.commit_invocation(invocation, {}, writer_token=token).status == "conflict"
    )
    assert sink_harness.sink.latest_checked(run_id).status == "corrupt"
    assert sink_harness.sink.load_invocation(run_id, "call-1").status == "corrupt"


def test_corrupt_event_and_terminal_cannot_acknowledge_retries(
    sink_harness: _SinkHarness,
) -> None:
    run_id = "run-corrupt-event-terminal"
    token = sink_harness.claim(run_id)
    event = make_agent_event(run_id=run_id, seq=1, event_type="run.finished")
    terminal = TerminalOutcome(
        run_id=run_id,
        kind="completed",
        retry_eligibility=RetryEligibility.NOT_APPLICABLE,
    )
    assert sink_harness.sink.append_event(event, writer_token=token).status == "committed"
    assert sink_harness.sink.settle_terminal(terminal, writer_token=token).status == "committed"

    _rewrite_payload_without_digest(
        sink_harness,
        "event_record",
        "run_id = %s AND sequence = %s",
        (run_id, 1),
        lambda payload: {**payload, "level": "warning"},
    )
    _rewrite_payload_without_digest(
        sink_harness,
        "terminal_record",
        "run_id = %s",
        (run_id,),
        lambda payload: {**payload, "error_code": "corrupt"},
    )

    assert sink_harness.sink.append_event(event, writer_token=token).status == "conflict"
    assert sink_harness.sink.settle_terminal(terminal, writer_token=token).status == "conflict"
    with pytest.raises(RuntimeError, match="event record is corrupt"):
        sink_harness.sink.read_event(run_id, 1)
    with pytest.raises(RuntimeError, match="terminal record is corrupt"):
        sink_harness.sink.read_terminal(run_id)


def test_blob_bounds_fencing_and_cross_run_authorization(
    sink_harness: _SinkHarness,
) -> None:
    run_a = "run-blob-a"
    run_b = "run-blob-b"
    token_a = sink_harness.claim(run_a)
    token_b = sink_harness.claim(run_b)
    sha256, blobs = _blob(b"run-a-private")

    assert (
        sink_harness.sink.commit_checkpoint(
            _checkpoint_with_blob(run_a, 1, sha256, "seed"),
            blobs,
            writer_token=token_a,
        ).status
        == "committed"
    )
    assert (
        sink_harness.sink.commit_checkpoint(
            _checkpoint_with_blob(run_b, 1, sha256, "foreign"),
            {},
            writer_token=token_b,
        ).status
        == "conflict"
    )
    assert (
        sink_harness.sink.commit_checkpoint(
            _checkpoint_with_blob(run_b, 1, sha256, "authorized-dedup"),
            blobs,
            writer_token=token_b,
        ).status
        == "committed"
    )

    over_limit = b"x" * 65
    over_sha = hashlib.sha256(over_limit).hexdigest()
    assert (
        sink_harness.sink.commit_checkpoint(
            RunCheckpoint(run_id=run_b, seq=2, final_text="oversized"),
            {over_sha: over_limit},
            writer_token=token_b,
        ).status
        == "conflict"
    )
    assert (
        sink_harness.sink.commit_checkpoint(
            RunCheckpoint(run_id=run_b, seq=2, final_text="uppercase"),
            {sha256.upper(): blobs[sha256]},
            writer_token=token_b,
        ).status
        == "conflict"
    )

    assert sink_harness.authority.release(token_b).status == "released"
    current_b = sink_harness.claim(run_b, "worker-b")
    stale_sha, stale_blobs = _blob(b"stale-private")
    assert (
        sink_harness.sink.commit_checkpoint(
            _checkpoint_with_blob(run_b, 2, stale_sha, "stale"),
            stale_blobs,
            writer_token=token_b,
        ).status
        == "fenced"
    )
    assert (
        sink_harness.sink.commit_checkpoint(
            _checkpoint_with_blob(run_b, 2, stale_sha, "stale-reference"),
            {},
            writer_token=current_b,
        ).status
        == "conflict"
    )
    assert (
        sink_harness.sink.commit_checkpoint(
            RunCheckpoint(run_id=run_b, seq=2),
            {},
            writer_token=token_a,
        ).status
        == "fenced"
    )
    assert _table_count(sink_harness, "run_blob", run_b) == 1


def test_existing_bytea_digest_with_different_bytes_fails_closed(
    sink_harness: _SinkHarness,
) -> None:
    run_id = "run-blob-corrupt"
    token = sink_harness.claim(run_id)
    sha256, blobs = _blob(b"same-length-a")
    assert (
        sink_harness.sink.commit_checkpoint(
            _checkpoint_with_blob(run_id, 1, sha256, "seed"),
            blobs,
            writer_token=token,
        ).status
        == "committed"
    )
    with sink_harness.database.transaction() as connection:
        with sink_harness.database.cursor(connection) as cursor:
            cursor.execute(
                sql.SQL("UPDATE {} SET content = %s WHERE sha256 = %s").format(
                    sql.Identifier(sink_harness.database.config.schema, "bytea_blob")
                ),
                (b"same-length-b", sha256),
            )

    assert (
        sink_harness.sink.commit_checkpoint(
            RunCheckpoint(run_id=run_id, seq=2),
            blobs,
            writer_token=token,
        ).status
        == "conflict"
    )
    loaded = sink_harness.sink.latest_checked(run_id)
    assert loaded.sequence == 1
    assert loaded.value is not None
    with pytest.raises(PostgresBlobCorrupt, match="content-address"):
        loaded.value.blob(sha256)


def test_invocation_lifecycle_result_blob_and_retry_rules(
    sink_harness: _SinkHarness,
) -> None:
    run_id = "run-invocation"
    token = sink_harness.claim(run_id)
    result_sha, result_blobs = _blob(b"model-result")

    assert sink_harness.sink.load_invocation(run_id, "call-1").status == "missing"
    assert (
        sink_harness.sink.commit_invocation(
            _invocation(run_id, 1, "settled"),
            {},
            writer_token=token,
        ).status
        == "conflict"
    )
    reserved = _invocation(run_id, 1, "reserved")
    assert sink_harness.sink.commit_invocation(reserved, {}, writer_token=token).status == (
        "committed"
    )
    assert sink_harness.sink.commit_invocation(reserved, {}, writer_token=token).status == (
        "already_committed"
    )
    assert (
        sink_harness.sink.commit_invocation(
            replace(reserved, request_digest="b" * 64),
            {},
            writer_token=token,
        ).status
        == "conflict"
    )
    assert (
        sink_harness.sink.commit_invocation(
            _invocation(run_id, 2, "dispatch_started"),
            {},
            writer_token=token,
        ).status
        == "committed"
    )
    settled = _invocation(
        run_id,
        3,
        "settled",
        succeeded_blob=result_sha,
        duration_ms=-0.0,
    )
    assert sink_harness.sink.commit_invocation(settled, {}, writer_token=token).status == (
        "conflict"
    )
    assert (
        sink_harness.sink.commit_invocation(
            settled,
            result_blobs,
            writer_token=token,
        ).status
        == "committed"
    )
    loaded = sink_harness.sink.load_invocation(run_id, "call-1")
    assert loaded.status == "loaded"
    assert loaded.sequence == 3
    assert loaded.value is not None
    assert loaded.value.invocation == settled
    duration = loaded.value.invocation.receipt["duration_ms"]  # type: ignore[index]
    assert duration == 0.0
    assert math.copysign(1.0, duration) == -1.0
    assert loaded.value.blob(result_sha) == b"model-result"
    assert (
        sink_harness.sink.commit_invocation(
            _invocation(run_id, 4, "reserved", attempt=2, dispatch_id="dispatch-2"),
            {},
            writer_token=token,
        ).status
        == "conflict"
    )

    retry_call = "retry-call"
    history = (
        _invocation(run_id, 1, "reserved", logical_call_id=retry_call),
        _invocation(run_id, 2, "dispatch_started", logical_call_id=retry_call),
        _invocation(
            run_id,
            3,
            "settled",
            logical_call_id=retry_call,
            retryable=True,
        ),
    )
    assert tuple(
        sink_harness.sink.commit_invocation(item, {}, writer_token=token).status for item in history
    ) == ("committed", "committed", "committed")
    assert (
        sink_harness.sink.commit_invocation(
            history[-1],
            {},
            writer_token=token,
            stage_evidence=True,
        ).status
        == "conflict"
    )
    retry = _invocation(
        run_id,
        4,
        "reserved",
        logical_call_id=retry_call,
        attempt=2,
        dispatch_id="dispatch-2",
    )
    assert sink_harness.sink.commit_invocation(retry, {}, writer_token=token).status == (
        "committed"
    )
    second_attempt_tail = (
        _invocation(
            run_id,
            5,
            "dispatch_started",
            logical_call_id=retry_call,
            attempt=2,
            dispatch_id="dispatch-2",
        ),
        _invocation(
            run_id,
            6,
            "settled",
            logical_call_id=retry_call,
            attempt=2,
            dispatch_id="dispatch-2",
            retryable=True,
        ),
    )
    assert tuple(
        sink_harness.sink.commit_invocation(item, {}, writer_token=token).status
        for item in second_attempt_tail
    ) == ("committed", "committed")
    with sink_harness.database.transaction() as connection:
        with sink_harness.database.cursor(connection) as cursor:
            cursor.execute(
                sql.SQL(
                    "UPDATE {} SET dispatch_id = %s WHERE run_id = %s AND logical_call_id = %s "
                    "AND revision <= %s"
                ).format(sql.Identifier(sink_harness.database.config.schema, "invocation_record")),
                ("corrupt-typed-dispatch", run_id, retry_call, 3),
            )
    assert (
        sink_harness.sink.commit_invocation(
            _invocation(
                run_id,
                7,
                "reserved",
                logical_call_id=retry_call,
                attempt=3,
                dispatch_id="dispatch-1",
            ),
            {},
            writer_token=token,
        ).status
        == "conflict"
    )
    with sink_harness.database.transaction() as connection:
        with sink_harness.database.cursor(connection) as cursor:
            cursor.execute(
                sql.SQL(
                    "UPDATE {} SET dispatch_id = %s WHERE run_id = %s AND logical_call_id = %s "
                    "AND revision = %s"
                ).format(sql.Identifier(sink_harness.database.config.schema, "invocation_record")),
                ("corrupt-head-dispatch", run_id, retry_call, 6),
            )
    assert sink_harness.sink.load_invocation(run_id, retry_call).status == "corrupt"
    assert (
        sink_harness.sink.commit_invocation(
            _invocation(run_id, 1, "reserved", logical_call_id="evidence-call"),
            {},
            writer_token=token,
            stage_evidence=True,
        ).status
        == "conflict"
    )


def test_invocation_transition_rejects_a_corrupt_prior_digest(
    sink_harness: _SinkHarness,
) -> None:
    run_id = "run-corrupt-transition"
    token = sink_harness.claim(run_id)
    reserved = _invocation(run_id, 1, "reserved")
    assert sink_harness.sink.commit_invocation(reserved, {}, writer_token=token).status == (
        "committed"
    )

    with sink_harness.database.transaction() as connection:
        with sink_harness.database.cursor(connection) as cursor:
            cursor.execute(
                sql.SQL(
                    "SELECT payload FROM {} WHERE run_id = %s AND logical_call_id = %s "
                    "AND revision = %s"
                ).format(sql.Identifier(sink_harness.database.config.schema, "invocation_record")),
                (run_id, "call-1", 1),
            )
            changed = {**cursor.fetchone()[0], "dispatch_state": "dispatch_started"}
            cursor.execute(
                sql.SQL(
                    "UPDATE {} SET payload = %s WHERE run_id = %s AND logical_call_id = %s "
                    "AND revision = %s"
                ).format(sql.Identifier(sink_harness.database.config.schema, "invocation_record")),
                (Json(changed), run_id, "call-1", 1),
            )

    result = sink_harness.sink.commit_invocation(
        _invocation(run_id, 2, "settled"),
        {},
        writer_token=token,
    )
    assert result.status == "conflict"
    assert _table_count(sink_harness, "invocation_record", run_id) == 1
    assert sink_harness.sink.load_invocation(run_id, "call-1").status == "corrupt"


@pytest.mark.parametrize("evidence_policy", ["required", "outbox"])
def test_invocation_evidence_policies_are_persisted_from_reservation(
    sink_harness: _SinkHarness,
    evidence_policy: str,
) -> None:
    run_id = f"run-evidence-{evidence_policy}-reservation"
    token = sink_harness.claim(run_id)

    result = sink_harness.sink.commit_invocation(
        _invocation(run_id, 1, "reserved", evidence_policy=evidence_policy),
        {},
        writer_token=token,
    )

    assert result.status == "committed"
    loaded = sink_harness.sink.load_invocation(run_id, "call-1")
    assert loaded.status == "loaded"
    assert loaded.value is not None
    assert loaded.value.invocation.evidence_policy == evidence_policy


def test_required_evidence_lifecycle_reserves_before_dispatch(
    sink_harness: _SinkHarness,
) -> None:
    run_id = "run-required-lifecycle"
    token = sink_harness.claim(run_id)
    lifecycle = FencedModelCallLifecycle(
        sink=sink_harness.sink,  # type: ignore[arg-type]
        writer_token=token,
        evidence_policy="required",
    )
    reservation = ModelDispatchReservation(
        logical_call_id="call-required",
        dispatch_attempt=1,
        dispatch_id="dispatch-required",
        request_digest="a" * 64,
        digest_generation=MODEL_REQUEST_DIGEST_GENERATION,
        idempotency_key=f"key-{hashlib.sha256(run_id.encode()).hexdigest()}",
    )

    lifecycle.reserve(reservation)

    assert _table_count(sink_harness, "invocation_record", run_id) == 1
    loaded = sink_harness.sink.load_invocation(run_id, "call-required")
    assert loaded.value is not None
    assert loaded.value.invocation.evidence_policy == "required"


def test_event_and_terminal_records_are_immutable_and_restart_safe(
    sink_harness: _SinkHarness,
) -> None:
    run_id = "run-event-terminal"
    token = sink_harness.claim(run_id)
    event = make_agent_event(
        run_id=run_id,
        seq=1,
        event_type="checkpoint.committed",
        data={"checkpoint_seq": 1, "text": "before\x00after", "offset": -0.0},
    )
    terminal = TerminalOutcome(
        run_id=run_id,
        kind="completed",
        retry_eligibility=RetryEligibility.NOT_APPLICABLE,
        checkpoint_seq=1,
        final_output_ref="blob:final-output",
    )

    event_commit = sink_harness.sink.append_event(event, writer_token=token)
    terminal_commit = sink_harness.sink.settle_terminal(terminal, writer_token=token)
    late_event = make_agent_event(
        run_id=run_id,
        seq=2,
        event_type="run.finished",
    )

    assert event_commit.status == "committed"
    assert terminal_commit.status == "committed"
    assert sink_harness.sink.latest_event_sequence(run_id) == 1
    reopened = sink_harness.reopen()
    assert reopened.read_event(run_id, 1) == event
    assert reopened.read_terminal(run_id) == terminal
    assert reopened.sink.append_event(event, writer_token=token).status == "already_committed"
    assert reopened.sink.settle_terminal(terminal, writer_token=token).status == (
        "already_committed"
    )
    assert reopened.sink.append_event(late_event, writer_token=token).status == "conflict"
    assert reopened.read_event(run_id, 2) is None
    assert reopened.sink.latest_event_sequence(run_id) == 1
    assert (
        reopened.sink.append_event(replace(event, level="warning"), writer_token=token).status
        == "conflict"
    )
    assert (
        reopened.sink.settle_terminal(
            replace(
                terminal,
                kind="failed_terminal",
                retry_eligibility=RetryEligibility.FORBIDDEN,
            ),
            writer_token=token,
        ).status
        == "conflict"
    )
    assert sink_harness.sink.capabilities.durable_events is True
    assert sink_harness.sink.capabilities.terminal_first_writer_wins is True


def test_activation_process_replacement_converges_on_postgres_receipt(
    sink_harness: _SinkHarness,
    activation_sink: PostgresFencedRunSink,
    tmp_path: Path,
) -> None:
    run_id = "run-neutral-activation"
    spec = AgentRunSpec(
        run_id=run_id,
        workspace_root=tmp_path / "workspace",
        run_root=tmp_path / "source-runs",
    )
    spec.workspace_root.mkdir(parents=True)
    source_store = LocalFsCheckpointStore(spec.run_root)
    source = AgentLoop(
        spec=spec,
        model_adapter=_RetryableActivationAdapter(),
        runtime_config_provider=runtime_provider(runtime_config("fs.write")),
        checkpoint_store=source_store,
    )
    source.open()
    assert source.run_until_suspended("resume on PostgreSQL").reason == "turn_failed"
    source.release_parked()
    source_record = source_store.latest(run_id)
    assert source_record is not None

    token = sink_harness.claim(run_id)
    assert (
        activation_sink.commit_checkpoint(
            source_record.checkpoint,
            {},
            writer_token=token,
        ).status
        == "committed"
    )
    request_digest = canonical_sha256({"run_id": run_id, "command_id": "input-1"})
    command = ActivationCommand(
        run_id=run_id,
        command_id="input-1",
        command_sequence=1,
        kind="control",
        source_checkpoint_seq=source_record.seq,
        source_checkpoint_sha256=canonical_sha256(source_record.checkpoint.to_json()),
        request_digest=request_digest,
        payload_ref=f"blob:{request_digest}",
    )
    adapter = _CountingActivationAdapter()
    factory_calls = 0

    def loop_factory(command: ActivationCommand, runtime: ActivationRuntime) -> AgentLoop:
        nonlocal factory_calls
        factory_calls += 1
        return AgentLoop(
            spec=AgentRunSpec(
                run_id=command.run_id,
                workspace_root=spec.workspace_root,
                run_root=tmp_path / "postgres-activation-runs",
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

    first = ActivationDriver(
        sink=activation_sink,
        writer_token=token,
        loop_factory=loop_factory,
    ).drive(command)

    assert (first.boundary_reason, first.error_code, first.provider_error_code) == (
        "settled",
        "",
        "",
    )
    assert first.event_cursor == activation_sink.latest_event_sequence(run_id)
    assert adapter.calls == 1
    assert factory_calls == 1
    assert sink_harness.authority.release(token).status == "released"
    replacement_token = sink_harness.claim(run_id, "worker-b")

    def forbidden_factory(command: ActivationCommand, runtime: ActivationRuntime) -> AgentLoop:
        del command, runtime
        raise AssertionError("canonical receipt must bypass loop construction")

    replacement = ActivationDriver(
        sink=activation_sink,
        writer_token=replacement_token,
        loop_factory=forbidden_factory,
    ).drive(command)

    assert replacement == first
    assert adapter.calls == 1
    assert factory_calls == 1


def test_required_evidence_binds_exact_current_settled_invocation(
    sink_harness: _SinkHarness,
) -> None:
    run_id = "run-required-evidence"
    token = sink_harness.claim(run_id)
    history = (
        _invocation(run_id, 1, "reserved", evidence_policy="required"),
        _invocation(run_id, 2, "dispatch_started", evidence_policy="required"),
        _invocation(
            run_id,
            3,
            "settled",
            evidence_policy="required",
            duration_ms=1,
        ),
    )
    assert tuple(
        sink_harness.sink.commit_invocation(item, {}, writer_token=token).status for item in history
    ) == ("committed", "committed", "committed")

    mismatched = replace(
        history[-1],
        receipt={**dict(history[-1].receipt or {}), "duration_ms": 1.0},
    )
    assert mismatched == history[-1]
    assert canonical_sha256(mismatched.to_json()) != canonical_sha256(history[-1].to_json())
    assert (
        sink_harness.sink.commit_model_evidence(mismatched, writer_token=token).status == "conflict"
    )
    assert _table_count(sink_harness, "model_evidence_record", run_id) == 0

    committed = sink_harness.sink.commit_model_evidence(history[-1], writer_token=token)
    repeated = sink_harness.sink.commit_model_evidence(history[-1], writer_token=token)

    assert committed.status == "committed"
    assert repeated.status == "already_committed"
    assert _table_count(sink_harness, "model_evidence_record", run_id) == 1

    sink_harness.authority.release(token)
    current = sink_harness.claim(run_id, "worker-b")
    assert (
        sink_harness.sink.commit_model_evidence(history[-1], writer_token=token).status == "fenced"
    )
    assert (
        sink_harness.sink.commit_model_evidence(history[-1], writer_token=current).status
        == "already_committed"
    )


def test_outbox_settlement_and_staging_are_one_transaction(
    sink_harness: _SinkHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "run-outbox-atomic"
    token = sink_harness.claim(run_id)
    history = (
        _invocation(run_id, 1, "reserved", evidence_policy="outbox"),
        _invocation(run_id, 2, "dispatch_started", evidence_policy="outbox"),
    )
    assert (
        sink_harness.sink.commit_invocation(
            history[0],
            {},
            writer_token=token,
            stage_evidence=True,
        ).status
        == "conflict"
    )
    assert tuple(
        sink_harness.sink.commit_invocation(item, {}, writer_token=token).status for item in history
    ) == ("committed", "committed")
    settled = _invocation(run_id, 3, "settled", evidence_policy="outbox")

    assert sink_harness.sink.commit_invocation(settled, {}, writer_token=token).status == "conflict"
    committed = sink_harness.sink.commit_invocation(
        settled,
        {},
        writer_token=token,
        stage_evidence=True,
    )
    assert committed.status == "committed"
    assert _table_count(sink_harness, "invocation_record", run_id) == 3
    assert _table_count(sink_harness, "model_evidence_outbox", run_id) == 1
    assert (
        sink_harness.sink.commit_invocation(
            settled,
            {},
            writer_token=token,
            stage_evidence=True,
        ).status
        == "already_committed"
    )
    with sink_harness.database.transaction() as connection:
        with sink_harness.database.cursor(connection) as cursor:
            cursor.execute(
                sql.SQL(
                    "DELETE FROM {} WHERE run_id = %s AND logical_call_id = %s AND revision = %s"
                ).format(
                    sql.Identifier(
                        sink_harness.database.config.schema,
                        "model_evidence_outbox",
                    )
                ),
                (run_id, "call-1", 3),
            )
    assert (
        sink_harness.sink.commit_invocation(
            settled,
            {},
            writer_token=token,
            stage_evidence=True,
        ).status
        == "conflict"
    )
    assert sink_harness.sink.load_invocation(run_id, "call-1").sequence == 3

    rollback_run = "run-outbox-rollback"
    rollback_token = sink_harness.claim(rollback_run)
    for invocation in (
        _invocation(rollback_run, 1, "reserved", evidence_policy="outbox"),
        _invocation(rollback_run, 2, "dispatch_started", evidence_policy="outbox"),
    ):
        assert (
            sink_harness.sink.commit_invocation(
                invocation,
                {},
                writer_token=rollback_token,
            ).status
            == "committed"
        )

    def reject_stage(
        _cursor: object,
        invocation: object,
        *,
        outbox: bool,
    ) -> tuple[CommitResult, str, dict[str, object] | None]:
        del outbox
        return (
            CommitResult(
                status="conflict",
                sequence=getattr(invocation, "revision", None),
            ),
            "",
            None,
        )

    monkeypatch.setattr(sink_harness.sink, "_commit_evidence_locked", reject_stage)
    rejected = sink_harness.sink.commit_invocation(
        _invocation(rollback_run, 3, "settled", evidence_policy="outbox"),
        {},
        writer_token=rollback_token,
        stage_evidence=True,
    )
    assert rejected.status == "conflict"
    assert sink_harness.sink.load_invocation(rollback_run, "call-1").sequence == 2
    assert _table_count(sink_harness, "invocation_record", rollback_run) == 2
    assert _table_count(sink_harness, "model_evidence_outbox", rollback_run) == 0
    assert sink_harness.sink.capabilities.transactional_outbox is True


def test_actual_postgres_satisfies_full_fenced_run_sink_contract(
    sink_harness: _SinkHarness,
) -> None:
    outcomes = run_fenced_run_sink_contract(sink_harness.reopen)
    failures = {
        outcome.rule_id: tuple(
            (item.observation_id, item.expected, item.actual)
            for item in outcome.observations
            if not item.passed
        )
        for outcome in outcomes
        if not outcome.passed
    }

    assert tuple(outcome.rule_id for outcome in outcomes) == (
        "FENCED-00-CAPABILITY-DECLARATION",
        "FENCED-01-CHECKPOINT-CONTENT-IDENTITY",
        "FENCED-02-FENCE-PRECEDES-IDEMPOTENCY",
        "FENCED-03-EVENT-AND-TERMINAL-WINNERS",
        "FENCED-04-INVOCATION-LIFECYCLE",
        "FENCED-05-INVOCATION-REFUSES-ILLEGAL-TRANSITIONS",
        "FENCED-06-WRITER-TOKEN-RUN-BINDING",
    )
    assert failures == {}


def test_competing_checkpoint_and_invocation_coordinates_have_one_winner(
    sink_harness: _SinkHarness,
) -> None:
    checkpoint_run = "run-checkpoint-race"
    checkpoint_token = sink_harness.claim(checkpoint_run)
    barrier = threading.Barrier(3)

    def commit_checkpoint(text: str) -> str:
        barrier.wait(timeout=5)
        return sink_harness.sink.commit_checkpoint(
            RunCheckpoint(run_id=checkpoint_run, seq=1, final_text=text),
            {},
            writer_token=checkpoint_token,
        ).status

    with ThreadPoolExecutor(max_workers=2) as executor:
        left = executor.submit(commit_checkpoint, "left")
        right = executor.submit(commit_checkpoint, "right")
        barrier.wait(timeout=5)
        checkpoint_statuses = (left.result(timeout=5), right.result(timeout=5))
    assert sorted(checkpoint_statuses) == ["committed", "conflict"]

    invocation_run = "run-invocation-race"
    invocation_token = sink_harness.claim(invocation_run)
    assert (
        sink_harness.sink.commit_invocation(
            _invocation(invocation_run, 1, "reserved"),
            {},
            writer_token=invocation_token,
        ).status
        == "committed"
    )
    assert (
        sink_harness.sink.commit_invocation(
            _invocation(invocation_run, 2, "dispatch_started"),
            {},
            writer_token=invocation_token,
        ).status
        == "committed"
    )
    result_sha, result_blobs = _blob(b"race-result")
    baseline = _invocation(invocation_run, 3, "settled", succeeded_blob=result_sha)
    competing = replace(
        baseline,
        receipt={**dict(baseline.receipt or {}), "provider_request_id": "provider-racer"},
    )
    barrier = threading.Barrier(3)

    def commit_invocation(value: DurableModelInvocation) -> str:
        barrier.wait(timeout=5)
        return sink_harness.sink.commit_invocation(
            value,
            result_blobs,
            writer_token=invocation_token,
        ).status

    with ThreadPoolExecutor(max_workers=2) as executor:
        left = executor.submit(commit_invocation, baseline)
        right = executor.submit(commit_invocation, competing)
        barrier.wait(timeout=5)
        invocation_statuses = (left.result(timeout=5), right.result(timeout=5))
    assert sorted(invocation_statuses) == ["committed", "conflict"]


def _wait_for_lock(database: PostgresDatabase, application_name: str) -> None:
    deadline = time.monotonic() + 5
    while True:
        with database.connection() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT EXISTS (SELECT 1 FROM pg_stat_activity "
                        "WHERE application_name = %s AND wait_event_type = 'Lock')",
                        (application_name,),
                    )
                    if bool(cursor.fetchone()[0]):
                        return
        if time.monotonic() >= deadline:
            pytest.fail("PostgreSQL sink did not block on the authority row")
        time.sleep(0.01)


def test_authority_rotation_linearizes_before_stale_blob_publication(
    sink_harness: _SinkHarness,
) -> None:
    run_id = "run-handoff"
    stale = sink_harness.claim(run_id)
    stale_sha, stale_blobs = _blob(b"stale-handoff")
    stale_checkpoint = _checkpoint_with_blob(run_id, 1, stale_sha, "stale")
    schema = sink_harness.database.config.schema

    with ThreadPoolExecutor(max_workers=1) as executor:
        with sink_harness.database.connection() as blocker:
            with blocker.transaction():
                with blocker.cursor() as cursor:
                    cursor.execute(
                        sql.SQL("SELECT run_id FROM {} WHERE run_id = %s FOR UPDATE").format(
                            sql.Identifier(schema, "run_authority")
                        ),
                        (run_id,),
                    )
                    future = executor.submit(
                        sink_harness.sink.commit_checkpoint,
                        stale_checkpoint,
                        stale_blobs,
                        writer_token=stale,
                    )
                    _wait_for_lock(
                        sink_harness.database,
                        sink_harness.database.config.application_name,
                    )
                    cursor.execute(
                        sql.SQL(
                            "UPDATE {} SET owner_id = %s, generation = generation + 1, "
                            "leased_until = pg_catalog.clock_timestamp() + interval '30 seconds', "
                            "revoked = false, updated_at = pg_catalog.clock_timestamp() "
                            "WHERE run_id = %s RETURNING generation"
                        ).format(sql.Identifier(schema, "run_authority")),
                        ("worker-b", run_id),
                    )
                    generation = int(cursor.fetchone()[0])
        stale_result = future.result(timeout=5)

    current = WriterToken(run_id=run_id, owner_id="worker-b", generation=generation)
    assert stale_result.status == "fenced"
    assert (
        sink_harness.sink.commit_checkpoint(
            stale_checkpoint,
            {},
            writer_token=current,
        ).status
        == "conflict"
    )
    assert (
        sink_harness.sink.commit_checkpoint(
            RunCheckpoint(run_id=run_id, seq=1, final_text="current"),
            {},
            writer_token=current,
        ).status
        == "committed"
    )
    assert _table_count(sink_harness, "run_blob", run_id) == 0


def test_all_sink_mutations_use_canonical_readback_after_ambiguous_commit(
    sink_harness: _SinkHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = "run-ambiguous"
    token = sink_harness.claim(run_id)
    original_transaction = sink_harness.database.transaction

    def install_one_ambiguous_commit() -> None:
        raised = False

        @contextmanager
        def transaction() -> Iterator[object]:
            nonlocal raised
            with original_transaction() as connection:
                yield connection
            if not raised:
                raised = True
                raise psycopg.OperationalError("injected response loss after commit")

        monkeypatch.setattr(sink_harness.database, "transaction", transaction)

    install_one_ambiguous_commit()
    checkpoint = RunCheckpoint(run_id=run_id, seq=1, final_text="ambiguous")
    result = sink_harness.sink.commit_checkpoint(checkpoint, {}, writer_token=token)
    assert result.status == "already_committed"
    assert sink_harness.sink.latest_checked(run_id).value.checkpoint == checkpoint  # type: ignore[union-attr]

    monkeypatch.setattr(sink_harness.database, "transaction", original_transaction)
    install_one_ambiguous_commit()
    invocation = _invocation(run_id, 1, "reserved")
    result = sink_harness.sink.commit_invocation(invocation, {}, writer_token=token)
    assert result.status == "already_committed"
    assert sink_harness.sink.load_invocation(run_id, "call-1").value.invocation == invocation  # type: ignore[union-attr]

    monkeypatch.setattr(sink_harness.database, "transaction", original_transaction)
    install_one_ambiguous_commit()
    event = make_agent_event(
        run_id=run_id,
        seq=1,
        event_type="checkpoint.committed",
    )
    assert sink_harness.sink.append_event(event, writer_token=token).status == ("already_committed")
    assert sink_harness.sink.read_event(run_id, 1) == event

    monkeypatch.setattr(sink_harness.database, "transaction", original_transaction)
    install_one_ambiguous_commit()
    terminal = TerminalOutcome(
        run_id=run_id,
        kind="completed",
        retry_eligibility=RetryEligibility.NOT_APPLICABLE,
    )
    assert sink_harness.sink.settle_terminal(terminal, writer_token=token).status == (
        "already_committed"
    )
    assert sink_harness.sink.read_terminal(run_id) == terminal

    monkeypatch.setattr(sink_harness.database, "transaction", original_transaction)
    evidence_run = "run-ambiguous-evidence"
    evidence_token = sink_harness.claim(evidence_run)
    required_history = (
        _invocation(evidence_run, 1, "reserved", evidence_policy="required"),
        _invocation(evidence_run, 2, "dispatch_started", evidence_policy="required"),
        _invocation(
            evidence_run,
            3,
            "settled",
            evidence_policy="required",
            duration_ms=1,
        ),
    )
    for item in required_history:
        assert (
            sink_harness.sink.commit_invocation(item, {}, writer_token=evidence_token).status
            == "committed"
        )
    equal_but_canonically_distinct = replace(
        required_history[-1],
        receipt={**dict(required_history[-1].receipt or {}), "duration_ms": 1.0},
    )
    assert equal_but_canonically_distinct == required_history[-1]
    install_one_ambiguous_commit()
    assert (
        sink_harness.sink.commit_model_evidence(
            equal_but_canonically_distinct,
            writer_token=evidence_token,
        ).status
        == "conflict"
    )
    assert _table_count(sink_harness, "model_evidence_record", evidence_run) == 0

    monkeypatch.setattr(sink_harness.database, "transaction", original_transaction)
    install_one_ambiguous_commit()
    assert (
        sink_harness.sink.commit_model_evidence(
            required_history[-1], writer_token=evidence_token
        ).status
        == "already_committed"
    )

    monkeypatch.setattr(sink_harness.database, "transaction", original_transaction)
    outbox_run = "run-ambiguous-outbox"
    outbox_token = sink_harness.claim(outbox_run)
    for revision, state in ((1, "reserved"), (2, "dispatch_started")):
        assert (
            sink_harness.sink.commit_invocation(
                _invocation(outbox_run, revision, state, evidence_policy="outbox"),
                {},
                writer_token=outbox_token,
            ).status
            == "committed"
        )
    install_one_ambiguous_commit()
    assert (
        sink_harness.sink.commit_invocation(
            _invocation(outbox_run, 3, "settled", evidence_policy="outbox"),
            {},
            writer_token=outbox_token,
            stage_evidence=True,
        ).status
        == "already_committed"
    )
    assert _table_count(sink_harness, "model_evidence_outbox", outbox_run) == 1


def _rewrite_record(
    harness: _SinkHarness,
    table: str,
    identity_sql: str,
    identity_parameters: tuple[object, ...],
    mutate: Callable[[dict[str, object]], dict[str, object]],
) -> None:
    with harness.database.transaction() as connection:
        with harness.database.cursor(connection) as cursor:
            cursor.execute(
                sql.SQL(f"SELECT payload, submitted_blobs FROM {{}} WHERE {identity_sql}").format(
                    sql.Identifier(harness.database.config.schema, table)
                ),
                identity_parameters,
            )
            payload, projection = cursor.fetchone()
            changed = mutate(dict(payload))
            digest = canonical_sha256({"record": changed, "blobs": projection})
            cursor.execute(
                sql.SQL(
                    f"UPDATE {{}} SET payload = %s, schema_version = %s, "
                    f"content_digest = %s WHERE {identity_sql}"
                ).format(sql.Identifier(harness.database.config.schema, table)),
                (Json(changed), changed["schema_version"], digest, *identity_parameters),
            )


def test_checked_readers_classify_future_and_corrupt_payloads(
    sink_harness: _SinkHarness,
) -> None:
    future_run = "run-future-records"
    token = sink_harness.claim(future_run)
    assert (
        sink_harness.sink.commit_checkpoint(
            RunCheckpoint(run_id=future_run, seq=1),
            {},
            writer_token=token,
        ).status
        == "committed"
    )
    assert (
        sink_harness.sink.commit_invocation(
            _invocation(future_run, 1, "reserved"),
            {},
            writer_token=token,
        ).status
        == "committed"
    )
    _rewrite_record(
        sink_harness,
        "checkpoint_record",
        "run_id = %s AND sequence = %s",
        (future_run, 1),
        lambda payload: {**payload, "schema_version": "monoid.checkpoint.v999"},
    )
    _rewrite_record(
        sink_harness,
        "invocation_record",
        "run_id = %s AND logical_call_id = %s AND revision = %s",
        (future_run, "call-1", 1),
        lambda payload: {**payload, "schema_version": "monoid.model-invocation.v999"},
    )
    assert sink_harness.sink.latest_checked(future_run).status == "unsupported_version"
    assert sink_harness.sink.load_invocation(future_run, "call-1").status == ("unsupported_version")

    corrupt_run = "run-corrupt-records"
    corrupt_token = sink_harness.claim(corrupt_run)
    assert (
        sink_harness.sink.commit_checkpoint(
            RunCheckpoint(run_id=corrupt_run, seq=1),
            {},
            writer_token=corrupt_token,
        ).status
        == "committed"
    )
    assert (
        sink_harness.sink.commit_invocation(
            _invocation(corrupt_run, 1, "reserved"),
            {},
            writer_token=corrupt_token,
        ).status
        == "committed"
    )
    _rewrite_record(
        sink_harness,
        "checkpoint_record",
        "run_id = %s AND sequence = %s",
        (corrupt_run, 1),
        lambda payload: {**payload, "status": {"invalid": "shape"}},
    )
    _rewrite_record(
        sink_harness,
        "invocation_record",
        "run_id = %s AND logical_call_id = %s AND revision = %s",
        (corrupt_run, "call-1", 1),
        lambda payload: {**payload, "unexpected_private_field": "corrupt"},
    )
    assert sink_harness.sink.latest_checked(corrupt_run).status == "corrupt"
    assert sink_harness.sink.load_invocation(corrupt_run, "call-1").status == "corrupt"


def test_sink_readiness_requires_reader_and_writer_compatibility(
    sink_harness: _SinkHarness,
) -> None:
    run_id = "run-reader-incompatible"
    token = sink_harness.claim(run_id)
    sha256, blobs = _blob(b"lazy-before-forward-migration")
    assert (
        sink_harness.sink.commit_checkpoint(
            _checkpoint_with_blob(run_id, 1, sha256, "loaded-before-forward"),
            blobs,
            writer_token=token,
        ).status
        == "committed"
    )
    loaded = sink_harness.sink.latest_checked(run_id)
    assert loaded.value is not None

    with sink_harness.database.transaction() as connection:
        with sink_harness.database.cursor(connection) as cursor:
            cursor.execute(
                sql.SQL(
                    "INSERT INTO {} "
                    "(migration_id, ordinal, checksum_sha256, reader_floor, writer_floor) "
                    "VALUES (%s, %s, %s, %s, %s)"
                ).format(
                    sql.Identifier(
                        sink_harness.database.config.schema,
                        "monoid_schema_migrations",
                    )
                ),
                ("0006_reader_incompatible", 6, "f" * 64, 6, 5),
            )

    sink = sink_harness.sink
    with pytest.raises(PostgresSchemaIncompatible, match="reader and writer"):
        sink.check_ready()
    with pytest.raises(PostgresSchemaIncompatible, match="successful check_ready"):
        sink.latest_checked(run_id)
    with pytest.raises(PostgresSchemaIncompatible, match="successful check_ready"):
        loaded.value.blob(sha256)


def test_sink_uses_positional_rows_with_a_caller_dict_pool(
    sink_harness: _SinkHarness,
) -> None:
    caller_pool = ConnectionPool(
        sink_harness.database.config.dsn,
        kwargs={"row_factory": dict_row},
        min_size=1,
        max_size=2,
        open=False,
    )
    caller_pool.open(wait=True, timeout=10)
    database = PostgresDatabase(sink_harness.database.config, pool=caller_pool)
    database.open()
    try:
        authority = PostgresWriterAuthorityStore(database)
        authority.check_ready()
        sink = PostgresFencedRunSink(database)
        sink.check_ready()
        token = authority.claim("run-dict-sink", "worker-a", timedelta(seconds=30)).writer_token
        checkpoint = RunCheckpoint(run_id="run-dict-sink", seq=1, final_text="dict-safe")
        assert sink.commit_checkpoint(checkpoint, {}, writer_token=token).status == "committed"
        assert sink.latest_checked("run-dict-sink").value.checkpoint == checkpoint  # type: ignore[union-attr]
        with database.connection() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1 AS value")
                    assert cursor.fetchone() == {"value": 1}
    finally:
        database.close()
        caller_pool.close()


def test_sink_requires_explicit_migration_readiness(postgres_target: tuple[str, int]) -> None:
    dsn, _ = postgres_target
    database = PostgresDatabase(
        PostgresConfig(
            dsn=dsn,
            schema=f"monoid_pr04_unready_{uuid.uuid4().hex}",
            min_pool_size=0,
            max_pool_size=2,
        )
    )
    database.open()
    try:
        sink = PostgresFencedRunSink(database)
        with pytest.raises(PostgresSchemaIncompatible, match="check_ready"):
            sink.latest_checked("run-unready")
        with pytest.raises(PostgresSchemaIncompatible, match="not current"):
            sink.check_ready()
    finally:
        database.close()


def test_external_object_profile_passes_full_fenced_sink_contract(
    external_sink_harness: _ExternalSinkHarness,
) -> None:
    outcomes = run_fenced_run_sink_contract(external_sink_harness.reopen)

    failed = [outcome.to_json() for outcome in outcomes if not outcome.passed]
    assert not failed, failed


def test_external_sink_preserves_historical_bytea_records_and_references(
    external_sink_harness: _ExternalSinkHarness,
) -> None:
    base_sink = PostgresFencedRunSink(external_sink_harness.database)
    base_sink.check_ready()
    checkpoint_run = "run-bytea-before-external-checkpoint"
    checkpoint_token = external_sink_harness.claim(checkpoint_run)
    checkpoint_sha, checkpoint_blobs = _blob(b"historical bytea checkpoint")
    assert base_sink.commit_checkpoint(
        _checkpoint_with_blob(checkpoint_run, 1, checkpoint_sha, "before-external"),
        checkpoint_blobs,
        writer_token=checkpoint_token,
    ).status == "committed"

    loaded_historical = external_sink_harness.sink.latest_checked(checkpoint_run)
    assert loaded_historical.value is not None
    assert loaded_historical.value.blob(checkpoint_sha) == checkpoint_blobs[checkpoint_sha]
    assert external_sink_harness.sink.commit_checkpoint(
        _checkpoint_with_blob(checkpoint_run, 2, checkpoint_sha, "after-external"),
        {},
        writer_token=checkpoint_token,
    ).status == "committed"
    loaded_continuation = external_sink_harness.sink.latest_checked(checkpoint_run)
    assert loaded_continuation.value is not None
    assert loaded_continuation.value.blob(checkpoint_sha) == checkpoint_blobs[checkpoint_sha]

    invocation_run = "run-bytea-before-external-invocation"
    invocation_token = external_sink_harness.claim(invocation_run)
    result_sha, result_blobs = _blob(b"historical bytea invocation")
    invocation_history = (
        _invocation(invocation_run, 1, "reserved"),
        _invocation(invocation_run, 2, "dispatch_started"),
        _invocation(invocation_run, 3, "settled", succeeded_blob=result_sha),
    )
    for invocation in invocation_history[:-1]:
        assert base_sink.commit_invocation(
            invocation,
            {},
            writer_token=invocation_token,
        ).status == "committed"
    assert base_sink.commit_invocation(
        invocation_history[-1],
        result_blobs,
        writer_token=invocation_token,
    ).status == "committed"
    loaded_invocation = external_sink_harness.sink.load_invocation(invocation_run, "call-1")
    assert loaded_invocation.value is not None
    assert loaded_invocation.value.blob(result_sha) == result_blobs[result_sha]

    assert _table_count(external_sink_harness, "run_object_blob", checkpoint_run) == 0
    assert _table_count(external_sink_harness, "run_object_blob", invocation_run) == 0


def test_external_sink_uses_inline_backing_when_object_limit_is_lower(
    external_sink_harness: _ExternalSinkHarness,
) -> None:
    class LowerLimitStore:
        def __init__(self, delegate: object, max_bytes: int) -> None:
            self.delegate = delegate
            self.max_bytes = max_bytes

        def put_if_absent(self, sha256: str, data: bytes) -> object:
            if len(data) > self.max_bytes:
                raise BlobTooLarge("injected object-store size boundary")
            return self.delegate.put_if_absent(sha256, data)  # type: ignore[attr-defined]

        def stat(self, sha256: str) -> object:
            return self.delegate.stat(sha256)  # type: ignore[attr-defined]

        def get_checked(self, sha256: str) -> bytes:
            return self.delegate.get_checked(sha256)  # type: ignore[attr-defined,no-any-return]

    limited_store = LowerLimitStore(external_sink_harness.object_store, max_bytes=8)
    sink = PostgresObjectStoreFencedRunSink(
        external_sink_harness.database,
        limited_store,  # type: ignore[arg-type]
    )
    sink.check_ready()

    checkpoint_run = "run-object-limit-inline-checkpoint"
    checkpoint_token = external_sink_harness.claim(checkpoint_run)
    checkpoint_sha, checkpoint_blobs = _blob(b"inline checkpoint after object size limit")
    assert sink.commit_checkpoint(
        _checkpoint_with_blob(checkpoint_run, 1, checkpoint_sha, "inline-fallback"),
        checkpoint_blobs,
        writer_token=checkpoint_token,
    ).status == "committed"
    loaded_checkpoint = sink.latest_checked(checkpoint_run)
    assert loaded_checkpoint.value is not None
    assert loaded_checkpoint.value.blob(checkpoint_sha) == checkpoint_blobs[checkpoint_sha]

    invocation_run = "run-object-limit-inline-invocation"
    invocation_token = external_sink_harness.claim(invocation_run)
    result_sha, result_blobs = _blob(b"inline invocation after object size limit")
    invocation_history = (
        _invocation(invocation_run, 1, "reserved"),
        _invocation(invocation_run, 2, "dispatch_started"),
        _invocation(invocation_run, 3, "settled", succeeded_blob=result_sha),
    )
    for invocation in invocation_history[:-1]:
        assert sink.commit_invocation(
            invocation,
            {},
            writer_token=invocation_token,
        ).status == "committed"
    assert sink.commit_invocation(
        invocation_history[-1],
        result_blobs,
        writer_token=invocation_token,
    ).status == "committed"
    loaded_invocation = sink.load_invocation(invocation_run, "call-1")
    assert loaded_invocation.value is not None
    assert loaded_invocation.value.blob(result_sha) == result_blobs[result_sha]

    assert _table_count(external_sink_harness, "run_blob", checkpoint_run) == 1
    assert _table_count(external_sink_harness, "run_blob", invocation_run) == 1
    assert _table_count(external_sink_harness, "run_object_blob", checkpoint_run) == 0
    assert _table_count(external_sink_harness, "run_object_blob", invocation_run) == 0
    assert external_sink_harness.object_store.stat(checkpoint_sha) is None  # type: ignore[attr-defined]
    assert external_sink_harness.object_store.stat(result_sha) is None  # type: ignore[attr-defined]

    rejected_run = "run-object-and-inline-limits"
    rejected_token = external_sink_harness.claim(rejected_run)
    rejected_data = b"x" * (external_sink_harness.database.config.max_bytea_blob_bytes + 1)
    rejected_sha, rejected_blobs = _blob(rejected_data)
    assert sink.commit_checkpoint(
        _checkpoint_with_blob(rejected_run, 1, rejected_sha, "outside-both-limits"),
        rejected_blobs,
        writer_token=rejected_token,
    ).status == "conflict"
    assert _table_count(external_sink_harness, "checkpoint_record", rejected_run) == 0
    assert _table_count(external_sink_harness, "run_blob", rejected_run) == 0
    assert _table_count(external_sink_harness, "run_object_blob", rejected_run) == 0


def test_external_object_association_is_run_scoped_and_survives_reopen(
    external_sink_harness: _ExternalSinkHarness,
) -> None:
    run_a = "run-object-associated-a"
    run_b = "run-object-associated-b"
    run_c = "run-object-associated-c"
    token_a = external_sink_harness.claim(run_a)
    token_b = external_sink_harness.claim(run_b)
    token_c = external_sink_harness.claim(run_c)
    sha256, blobs = _blob(b"external private checkpoint bytes")
    checkpoint_a = _checkpoint_with_blob(run_a, 1, sha256, "associated")

    assert external_sink_harness.sink.commit_checkpoint(
        checkpoint_a,
        blobs,
        writer_token=token_a,
    ).status == "committed"
    reopened = external_sink_harness.reopen()
    loaded = reopened.sink.latest_checked(run_a)
    assert loaded.value is not None
    assert loaded.value.blob(sha256) == blobs[sha256]

    checkpoint_b = _checkpoint_with_blob(run_b, 1, sha256, "global-dedup")
    assert reopened.sink.commit_checkpoint(checkpoint_b, blobs, writer_token=token_b).status == (
        "committed"
    )
    class CountingStore:
        def __init__(self, delegate: object) -> None:
            self.delegate = delegate
            self.stat_calls = 0
            self.get_calls = 0

        def put_if_absent(self, digest: str, data: bytes) -> object:
            return self.delegate.put_if_absent(digest, data)  # type: ignore[attr-defined]

        def stat(self, digest: str) -> object:
            self.stat_calls += 1
            return self.delegate.stat(digest)  # type: ignore[attr-defined]

        def get_checked(self, digest: str) -> bytes:
            self.get_calls += 1
            return self.delegate.get_checked(digest)  # type: ignore[attr-defined,no-any-return]

    counted_store = CountingStore(external_sink_harness.object_store)
    counted_sink = PostgresObjectStoreFencedRunSink(
        external_sink_harness.database,
        counted_store,  # type: ignore[arg-type]
    )
    counted_sink.check_ready()
    checkpoint_c = _checkpoint_with_blob(run_c, 1, sha256, "cross-run")
    assert counted_sink.commit_checkpoint(checkpoint_c, {}, writer_token=token_c).status == (
        "conflict"
    )
    with pytest.raises(KeyError):
        counted_sink._read_blob(run_c, sha256)
    assert counted_store.stat_calls == counted_store.get_calls == 0

    with external_sink_harness.database.transaction() as connection:
        with external_sink_harness.database.cursor(connection) as cursor:
            cursor.execute(
                sql.SQL("SELECT run_id FROM {} WHERE sha256 = %s ORDER BY run_id").format(
                    sql.Identifier(
                        external_sink_harness.database.config.schema,
                        "run_object_blob",
                    )
                ),
                (sha256,),
            )
            assert [str(row[0]) for row in cursor] == [run_a, run_b]
            cursor.execute(
                sql.SQL("SELECT count(*) FROM {} WHERE sha256 = %s").format(
                    sql.Identifier(
                        external_sink_harness.database.config.schema,
                        "object_blob",
                    )
                ),
                (sha256,),
            )
            assert int(cursor.fetchone()[0]) == 1


def test_associated_external_object_missing_is_a_typed_restore_failure(
    external_sink_harness: _ExternalSinkHarness,
) -> None:
    run_id = "run-object-missing"
    token = external_sink_harness.claim(run_id)
    sha256, blobs = _blob(b"associated object later removed")
    assert external_sink_harness.sink.commit_checkpoint(
        _checkpoint_with_blob(run_id, 1, sha256, "missing"),
        blobs,
        writer_token=token,
    ).status == "committed"
    inventory = external_sink_harness.object_admin.inventory_page(limit=1000)  # type: ignore[attr-defined]
    entry = next(value for value in inventory.entries if value.sha256 == sha256)
    assert external_sink_harness.object_admin.delete_if_match(  # type: ignore[attr-defined]
        sha256,
        entry.delete_token,
    ).status == "deleted"

    loaded = external_sink_harness.sink.latest_checked(run_id)
    assert loaded.value is not None
    with pytest.raises(BlobNotFound, match="missing"):
        loaded.value.blob(sha256)


def test_external_object_upload_followed_by_postgres_rollback_is_not_canonical(
    external_sink_harness: _ExternalSinkHarness,
) -> None:
    class InjectedRollback(Exception):
        pass

    class RollbackSink(PostgresObjectStoreFencedRunSink):
        def _persist_blobs(
            self,
            cursor: object,
            run_id: str,
            blobs: Mapping[str, bytes],
        ) -> None:
            super()._persist_blobs(cursor, run_id, blobs)
            raise InjectedRollback

    run_id = "run-object-postgres-rollback"
    token = external_sink_harness.claim(run_id)
    sha256, blobs = _blob(b"object survives PostgreSQL rollback")
    sink = RollbackSink(
        external_sink_harness.database,
        external_sink_harness.object_store,  # type: ignore[arg-type]
    )
    sink.check_ready()

    with pytest.raises(InjectedRollback):
        sink.commit_checkpoint(
            _checkpoint_with_blob(run_id, 1, sha256, "rollback"),
            blobs,
            writer_token=token,
        )

    assert external_sink_harness.object_store.stat(sha256) is not None  # type: ignore[attr-defined]
    assert _table_count(external_sink_harness, "checkpoint_record", run_id) == 0
    assert _table_count(external_sink_harness, "run_object_blob", run_id) == 0
    with external_sink_harness.database.transaction() as connection:
        with external_sink_harness.database.cursor(connection) as cursor:
            cursor.execute(
                sql.SQL("SELECT count(*) FROM {} WHERE sha256 = %s").format(
                    sql.Identifier(
                        external_sink_harness.database.config.schema,
                        "object_blob",
                    )
                ),
                (sha256,),
            )
            assert int(cursor.fetchone()[0]) == 0


def test_object_first_fence_failure_leaves_only_a_collectable_orphan(
    external_sink_harness: _ExternalSinkHarness,
) -> None:
    run_id = "run-object-first-orphan"
    stale = external_sink_harness.claim(run_id, "worker-a")
    assert external_sink_harness.authority.release(stale).status == "released"
    current = external_sink_harness.claim(run_id, "worker-b")
    sha256, blobs = _blob(b"uploaded before stale writer fence")

    result = external_sink_harness.sink.commit_checkpoint(
        _checkpoint_with_blob(run_id, 1, sha256, "must-not-commit"),
        blobs,
        writer_token=stale,
    )

    assert result.status == "fenced"
    assert external_sink_harness.object_store.stat(sha256) is not None  # type: ignore[attr-defined]
    assert _table_count(external_sink_harness, "checkpoint_record", run_id) == 0
    assert _table_count(external_sink_harness, "run_object_blob", run_id) == 0
    collector = PostgresObjectGarbageCollector(
        external_sink_harness.database,
        external_sink_harness.object_admin,  # type: ignore[arg-type]
    )
    collector.check_ready()
    plan = collector.plan(grace_period=timedelta(0))
    candidate = next(value for value in plan.candidates if value.sha256 == sha256)
    assert candidate.generation == 0

    with pytest.raises(ValueError, match="plan identity"):
        collector.apply(replace(plan, next_token="tampered"))

    receipts = collector.apply(plan)

    receipt = next(value for value in receipts if value.sha256 == sha256)
    assert receipt.status == "deleted"
    assert receipt.candidate_generation == receipt.observed_generation == 0
    assert external_sink_harness.object_store.stat(sha256) is None  # type: ignore[attr-defined]
    assert external_sink_harness.sink.commit_checkpoint(
        _checkpoint_with_blob(run_id, 1, sha256, "revived"),
        blobs,
        writer_token=current,
    ).status == "committed"
    with external_sink_harness.database.transaction() as connection:
        with external_sink_harness.database.cursor(connection) as cursor:
            cursor.execute(
                sql.SQL("SELECT generation, state FROM {} WHERE sha256 = %s").format(
                    sql.Identifier(
                        external_sink_harness.database.config.schema,
                        "object_blob",
                    )
                ),
                (sha256,),
            )
            assert cursor.fetchone() == (2, "available")
    assert collector.apply(plan) == receipts
    assert external_sink_harness.sink._read_blob(run_id, sha256) == blobs[sha256]


def test_gc_grace_and_stale_generation_plan_never_delete_a_recreated_object(
    external_sink_harness: _ExternalSinkHarness,
) -> None:
    sha256, blobs = _blob(b"stale GC generation must not delete recreation")
    external_sink_harness.object_store.put_if_absent(sha256, blobs[sha256])  # type: ignore[attr-defined]
    collector = PostgresObjectGarbageCollector(
        external_sink_harness.database,
        external_sink_harness.object_admin,  # type: ignore[arg-type]
    )
    collector.check_ready()

    grace_plan = collector.plan(grace_period=timedelta(days=1))
    assert all(candidate.sha256 != sha256 for candidate in grace_plan.candidates)
    assert external_sink_harness.object_store.stat(sha256) is not None  # type: ignore[attr-defined]

    stale_plan = collector.plan(grace_period=timedelta(0))
    deleting_plan = collector.plan(grace_period=timedelta(0))
    assert stale_plan.plan_id != deleting_plan.plan_id
    assert collector.apply(deleting_plan)[0].status == "deleted"
    external_sink_harness.object_store.put_if_absent(sha256, blobs[sha256])  # type: ignore[attr-defined]

    stale_receipt = collector.apply(stale_plan)[0]

    assert stale_receipt.status == "skipped_generation"
    assert stale_receipt.candidate_generation == 0
    assert stale_receipt.observed_generation == 1
    assert external_sink_harness.object_store.get_checked(sha256) == blobs[sha256]  # type: ignore[attr-defined]


def test_gc_and_association_digest_lock_race_has_only_two_safe_outcomes(
    external_sink_harness: _ExternalSinkHarness,
) -> None:
    run_id = "run-object-gc-race"
    token = external_sink_harness.claim(run_id)
    sha256, blobs = _blob(b"association versus garbage collection")
    external_sink_harness.object_store.put_if_absent(sha256, blobs[sha256])  # type: ignore[attr-defined]
    collector = PostgresObjectGarbageCollector(
        external_sink_harness.database,
        external_sink_harness.object_admin,  # type: ignore[arg-type]
    )
    collector.check_ready()
    plan = collector.plan(grace_period=timedelta(0))
    assert any(candidate.sha256 == sha256 for candidate in plan.candidates)
    barrier = threading.Barrier(3)

    def associate() -> CommitResult:
        barrier.wait(timeout=10)
        return external_sink_harness.reopen().sink.commit_checkpoint(
            _checkpoint_with_blob(run_id, 1, sha256, "race"),
            blobs,
            writer_token=token,
        )

    def collect() -> tuple[object, ...]:
        barrier.wait(timeout=10)
        return collector.apply(plan)

    with ThreadPoolExecutor(max_workers=2) as executor:
        association_future = executor.submit(associate)
        collection_future = executor.submit(collect)
        barrier.wait(timeout=10)
        association = association_future.result(timeout=30)
        receipts = collection_future.result(timeout=30)

    receipt = next(value for value in receipts if value.sha256 == sha256)
    assert (association.status, receipt.status) in {
        ("committed", "skipped_associated"),
        ("committed", "deleted"),
        ("committed", "already_missing"),
        ("conflict", "deleted"),
        ("conflict", "already_missing"),
    }
    if association.status == "committed":
        assert external_sink_harness.sink._read_blob(run_id, sha256) == blobs[sha256]
    else:
        with pytest.raises(KeyError):
            external_sink_harness.sink._read_blob(run_id, sha256)
