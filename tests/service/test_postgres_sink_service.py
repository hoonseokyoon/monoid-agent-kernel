from __future__ import annotations

import hashlib
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Callable, Iterator

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.serial, pytest.mark.service]


def _selected() -> bool:
    return os.environ.get("MONOID_SERVICE_PROFILE") in {"postgres", "objectstore", "combined"}


if not _selected():
    pytest.skip("PostgreSQL service profile is not selected", allow_module_level=True)

import psycopg  # noqa: E402
from psycopg import sql  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402
from psycopg.types.json import Jsonb  # noqa: E402
from psycopg_pool import ConnectionPool  # noqa: E402

from monoid_agent_kernel.adapters.postgres import (  # noqa: E402
    PostgresBlobCorrupt,
    PostgresConfig,
    PostgresDatabase,
    PostgresFencedRunSink,
    PostgresMigrations,
    PostgresSchemaIncompatible,
    PostgresWriterAuthorityStore,
)
from monoid_agent_kernel.core._util import canonical_sha256  # noqa: E402
from monoid_agent_kernel.core.checkpoint import RunCheckpoint  # noqa: E402
from monoid_agent_kernel.core.model_invocation import (  # noqa: E402
    MODEL_REQUEST_DIGEST_GENERATION,
    DurableModelInvocation,
)
from monoid_agent_kernel.hosting import WriterToken  # noqa: E402


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


@pytest.fixture
def sink_harness(postgres_target: tuple[str, int]) -> Iterator[_SinkHarness]:
    dsn, _ = postgres_target
    schema = f"monoid_pr03_{uuid.uuid4().hex}"
    database = PostgresDatabase(
        PostgresConfig(
            dsn=dsn,
            schema=schema,
            min_pool_size=1,
            max_pool_size=8,
            pool_timeout_s=10,
            application_name="monoid-pr03-service-test",
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
) -> DurableModelInvocation:
    receipt = None
    result_ref = ""
    failure_code = ""
    if state == "settled":
        receipt = {"request_digest": request_digest, "retryable": retryable}
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
    settled = _invocation(run_id, 3, "settled", succeeded_blob=result_sha)
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
    assert (
        sink_harness.sink.commit_invocation(
            replace(retry, revision=5, dispatch_id="dispatch-1"),
            {},
            writer_token=token,
        ).status
        == "conflict"
    )
    assert (
        sink_harness.sink.commit_invocation(
            _invocation(run_id, 1, "reserved", logical_call_id="evidence-call"),
            {},
            writer_token=token,
            stage_evidence=True,
        ).status
        == "conflict"
    )


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


def test_ambiguous_checkpoint_and_invocation_commit_use_canonical_readback(
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
                (Jsonb(changed), changed["schema_version"], digest, *identity_parameters),
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
            schema=f"monoid_pr03_unready_{uuid.uuid4().hex}",
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
