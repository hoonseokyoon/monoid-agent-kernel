from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Iterator

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.serial, pytest.mark.service]


def _selected() -> bool:
    return os.environ.get("MONOID_SERVICE_PROFILE") in {"postgres", "objectstore", "combined"}


if not _selected():
    pytest.skip("PostgreSQL service profile is not selected", allow_module_level=True)

import psycopg  # noqa: E402
from psycopg import IsolationLevel, sql  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402
from psycopg_pool import ConnectionPool  # noqa: E402

from monoid_agent_kernel.adapters.postgres import (  # noqa: E402
    PostgresConfig,
    PostgresDatabase,
    PostgresMigrationDrift,
    PostgresMigrations,
    PostgresOperations,
    PostgresSchemaIncompatible,
    PostgresWriterAuthorityStore,
    MigrationStatus,
)
from monoid_agent_kernel.adapters.postgres.authority import (  # noqa: E402
    _ELAPSED_TTL_INTERVAL,
    _ttl_microseconds,
)
from monoid_agent_kernel.hosting import WriterLease, WriterLeaseUnavailable, WriterToken  # noqa: E402


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


@pytest.fixture
def postgres_database(postgres_target: tuple[str, int]) -> Iterator[PostgresDatabase]:
    dsn, _ = postgres_target
    schema = f"monoid_pr02_{uuid.uuid4().hex}"
    database = PostgresDatabase(
        PostgresConfig(
            dsn=dsn,
            schema=schema,
            min_pool_size=1,
            max_pool_size=6,
            pool_timeout_s=10,
            application_name="monoid-pr02-service-test",
        )
    )
    database.open()
    try:
        yield database
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


def _wait_until_application_blocks_on_lock(
    database: PostgresDatabase,
    application_name: str,
) -> None:
    deadline = time.monotonic() + 5
    while True:
        with database.connection() as observer_connection:
            with observer_connection.transaction():
                with observer_connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT EXISTS ("
                        "SELECT 1 FROM pg_stat_activity "
                        "WHERE application_name = %s AND wait_event_type = 'Lock'"
                        ")",
                        (application_name,),
                    )
                    blocked = bool(cursor.fetchone()[0])
        if blocked:
            return
        if time.monotonic() >= deadline:
            pytest.fail("PostgreSQL application did not block on the expected lock")
        time.sleep(0.01)


@pytest.mark.parametrize(
    ("dsn_variable", "expected_major", "profiles"),
    _POSTGRES_TARGETS,
)
def test_pinned_postgresql_service_is_reachable(
    dsn_variable: str,
    expected_major: int,
    profiles: set[str],
) -> None:
    if os.environ["MONOID_SERVICE_PROFILE"] not in profiles:
        pytest.skip(f"PostgreSQL {expected_major} is outside the selected profile")
    dsn = os.environ.get(dsn_variable)
    if not dsn:
        pytest.fail(f"{dsn_variable} is required for the selected service profile")

    with psycopg.connect(dsn, connect_timeout=10) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_setting('server_version_num')")
            version_number = int(cursor.fetchone()[0])
            cursor.execute("SELECT 1")
            assert cursor.fetchone() == (1,)

    assert version_number // 10000 == expected_major


def test_writer_ttl_is_an_elapsed_duration_across_dst(
    postgres_database: PostgresDatabase,
) -> None:
    base = datetime(2026, 3, 7, 17, tzinfo=UTC)
    ttl = timedelta(hours=24)
    query = (
        "SELECT extract(epoch FROM ((%s::timestamptz + "
        + _ELAPSED_TTL_INTERVAL
        + ") - %s::timestamptz)), "
        "extract(epoch FROM ((%s::timestamptz + %s) - %s::timestamptz))"
    )
    with postgres_database.connection() as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL TIME ZONE 'America/New_York'")
                cursor.execute(
                    query,
                    (
                        base,
                        _ttl_microseconds(ttl),
                        base,
                        base,
                        ttl,
                        base,
                    ),
                )
                elapsed_seconds, calendar_seconds = cursor.fetchone()

    assert elapsed_seconds == 86_400
    assert calendar_seconds == 82_800


def test_explicit_migration_lifecycle_and_doctor(
    postgres_database: PostgresDatabase,
) -> None:
    migrations = PostgresMigrations(postgres_database)
    store = PostgresWriterAuthorityStore(postgres_database)

    before = migrations.status()
    assert before.schema_exists is False
    assert before.current_version == 0
    assert tuple(item.migration_id for item in migrations.plan().pending) == (
        "0001_authority",
        "0002_checkpoint_invocation",
        "0003_event_terminal_evidence_outbox",
        "0004_object_association_gc",
        "0005_activation_admission_dispatch",
        "0006_durable_stream",
    )
    assert migrations.doctor().ok is False
    with pytest.raises(PostgresSchemaIncompatible, match="check_ready"):
        store.read("run-unready")
    with pytest.raises(PostgresSchemaIncompatible, match="not current"):
        migrations.require_reader_compatible()
    with pytest.raises(PostgresSchemaIncompatible, match="not current"):
        store.check_ready()

    first = migrations.apply()
    assert tuple(item.migration_id for item in first.applied) == (
        "0001_authority",
        "0002_checkpoint_invocation",
        "0003_event_terminal_evidence_outbox",
        "0004_object_association_gc",
        "0005_activation_admission_dispatch",
        "0006_durable_stream",
    )
    assert first.status.current is True
    assert first.status.current_version == 6
    assert first.status.schema == postgres_database.config.schema

    repeated = migrations.apply()
    assert repeated.applied == ()
    assert repeated.status == migrations.status()
    assert migrations.plan().pending == ()
    assert migrations.doctor().ok is True
    assert store.check_ready().current is True
    with postgres_database.connection() as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_schemas(false)")
                search_path = tuple(cursor.fetchone()[0])
    assert postgres_database.config.schema not in search_path


def test_migration_checksum_drift_fails_closed(postgres_database: PostgresDatabase) -> None:
    migrations = PostgresMigrations(postgres_database)
    migrations.apply()
    with postgres_database.connection() as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL("UPDATE {}.{} SET checksum_sha256 = %s WHERE migration_id = %s").format(
                        sql.Identifier(postgres_database.config.schema),
                        sql.Identifier("monoid_schema_migrations"),
                    ),
                    ("0" * 64, "0001_authority"),
                )

    with pytest.raises(PostgresMigrationDrift, match="checksum drift"):
        migrations.status()
    report = migrations.doctor()
    assert report.ok is False
    assert report.migration_status is None
    assert any("PostgresMigrationDrift" in error for error in report.errors)


def test_operations_snapshot_is_read_only_and_aggregate_only(
    postgres_database: PostgresDatabase,
) -> None:
    operations = PostgresOperations(postgres_database)
    with pytest.raises(PostgresSchemaIncompatible, match="check_ready"):
        operations.snapshot()

    PostgresMigrations(postgres_database).apply()
    assert operations.check_ready().current is True
    snapshot = operations.snapshot()

    assert snapshot.source == "postgres"
    assert snapshot.collected_at.tzinfo is not None
    assert snapshot.metrics
    assert snapshot.metrics == tuple(
        sorted(snapshot.metrics, key=lambda metric: (metric.name, metric.attributes))
    )
    assert {metric.name for metric in snapshot.metrics} >= {
        "monoid.postgres.authority.count",
        "monoid.postgres.invocation.count",
        "monoid.postgres.object.count",
        "monoid.postgres.outbox.oldest_age",
        "monoid.postgres.schema.version",
        "monoid.postgres.stream.chunk.bytes",
    }
    public = repr(snapshot.to_json())
    assert postgres_database.config.schema not in public
    assert postgres_database.config.dsn not in public

    with postgres_database.read_snapshot() as (connection, snapshot_boundary):
        assert snapshot_boundary.tzinfo is not None
        with postgres_database.cursor(connection) as cursor:
            cursor.execute(
                "SELECT pg_catalog.current_setting('transaction_isolation'), "
                "pg_catalog.current_setting('transaction_read_only')"
            )
            assert cursor.fetchone() == ("repeatable read", "on")

    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        with postgres_database.transaction(read_only=True) as connection:
            with postgres_database.cursor(connection) as cursor:
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {}.{} "
                        "(run_id, owner_id, generation, leased_until, revoked, updated_at) "
                        "VALUES ('readonly-probe', 'readonly-probe', 1, "
                        "pg_catalog.clock_timestamp() + interval '1 minute', false, "
                        "pg_catalog.clock_timestamp())"
                    ).format(
                        sql.Identifier(postgres_database.config.schema),
                        sql.Identifier("run_authority"),
                    )
                )
    with pytest.raises(ValueError, match="isolation_level"):
        with postgres_database.transaction(isolation_level="serializable"):  # type: ignore[arg-type]
            pass


def test_operations_outbox_lag_uses_claimable_rows_only(
    postgres_database: PostgresDatabase,
) -> None:
    PostgresMigrations(postgres_database).apply()
    sampled_at = datetime.now(UTC)
    activation_rows = (
        (
            "activation-ready",
            "pending",
            2,
            sampled_at - timedelta(minutes=1),
            None,
            sampled_at - timedelta(minutes=5),
        ),
        (
            "activation-delayed",
            "pending",
            9,
            sampled_at + timedelta(days=1),
            None,
            sampled_at - timedelta(days=1),
        ),
        (
            "activation-leased",
            "leased",
            8,
            sampled_at - timedelta(days=1),
            sampled_at + timedelta(days=1),
            sampled_at - timedelta(days=1),
        ),
    )
    evidence_rows = (
        (
            "evidence-ready",
            3,
            sampled_at - timedelta(minutes=1),
            None,
            None,
            sampled_at - timedelta(minutes=5),
        ),
        (
            "evidence-delayed",
            10,
            sampled_at + timedelta(days=1),
            None,
            None,
            sampled_at - timedelta(days=1),
        ),
        (
            "evidence-leased",
            11,
            sampled_at - timedelta(days=1),
            "evidence-owner",
            sampled_at + timedelta(days=1),
            sampled_at - timedelta(days=1),
        ),
    )
    schema = sql.Identifier(postgres_database.config.schema)
    with postgres_database.transaction() as connection:
        with postgres_database.cursor(connection) as cursor:
            for index, (
                run_id,
                state,
                attempts,
                available_at,
                leased_until,
                created_at,
            ) in enumerate(
                activation_rows,
                start=1,
            ):
                command_id = f"command-{index}"
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {}.{} "
                        "(run_id, owner_id, generation, leased_until, revoked, updated_at) "
                        "VALUES (%s, %s, 1, %s, false, %s)"
                    ).format(schema, sql.Identifier("run_authority")),
                    (run_id, f"authority-{index}", sampled_at + timedelta(days=2), sampled_at),
                )
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {}.{} "
                        "(run_id, command_id, command_sequence, command_kind, request_digest, "
                        "payload_ref, request_identity_sha256, admitted_identity_sha256, "
                        "admitted_content_digest, admitted_payload, created_at, updated_at) "
                        "VALUES (%s, %s, 1, 'input', %s, %s, %s, %s, %s, "
                        "'{{}}'::json, %s, %s)"
                    ).format(schema, sql.Identifier("activation_admission_record")),
                    (
                        run_id,
                        command_id,
                        f"{index:064x}",
                        f"blob:payload-{index}",
                        f"{index + 10:064x}",
                        f"{index + 20:064x}",
                        f"{index + 30:064x}",
                        created_at,
                        created_at,
                    ),
                )
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {}.{} "
                        "(run_id, command_id, delivery_state, attempt_count, available_at, "
                        "claim_owner, claim_id, claim_generation, leased_until, created_at, "
                        "updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                    ).format(schema, sql.Identifier("activation_dispatch_outbox")),
                    (
                        run_id,
                        command_id,
                        state,
                        attempts,
                        available_at,
                        f"dispatch-owner-{index}",
                        f"dispatch-claim-{index}",
                        attempts,
                        leased_until,
                        created_at,
                        created_at,
                    ),
                )

            for index, (
                run_id,
                attempts,
                available_at,
                lease_owner,
                leased_until,
                created_at,
            ) in enumerate(evidence_rows, start=101):
                logical_call_id = f"logical-{index}"
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {}.{} "
                        "(run_id, owner_id, generation, leased_until, revoked, updated_at) "
                        "VALUES (%s, %s, 1, %s, false, %s)"
                    ).format(schema, sql.Identifier("run_authority")),
                    (run_id, f"authority-{index}", sampled_at + timedelta(days=2), sampled_at),
                )
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {}.{} "
                        "(run_id, logical_call_id, revision, schema_version, dispatch_id, "
                        "dispatch_attempt, dispatch_state, idempotency_key, request_digest, "
                        "digest_generation, evidence_policy, result_ref, failure_code, "
                        "content_digest, payload, submitted_blobs, committed_at) "
                        "VALUES (%s, %s, 1, 'test-invocation.v1', %s, 1, 'settled', %s, %s, "
                        "'test-digest.v1', 'outbox', 'object:test', '', %s, '{{}}'::json, "
                        "'{{}}'::jsonb, %s)"
                    ).format(schema, sql.Identifier("invocation_record")),
                    (
                        run_id,
                        logical_call_id,
                        f"dispatch-{index}",
                        f"idempotency-{index}",
                        f"{index:064x}",
                        f"{index + 30:064x}",
                        created_at,
                    ),
                )
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {}.{} "
                        "(run_id, logical_call_id, revision, schema_version, evidence_policy, "
                        "content_digest, payload, attempt_count, available_at, lease_owner, "
                        "leased_until, created_at, updated_at) "
                        "VALUES (%s, %s, 1, 'test-evidence.v1', 'outbox', %s, '{{}}'::json, "
                        "%s, %s, %s, %s, %s, %s)"
                    ).format(schema, sql.Identifier("model_evidence_outbox")),
                    (
                        run_id,
                        logical_call_id,
                        f"{index + 40:064x}",
                        attempts,
                        available_at,
                        lease_owner,
                        leased_until,
                        created_at,
                        created_at,
                    ),
                )

    operations = PostgresOperations(postgres_database)
    operations.check_ready()
    metrics = {
        (metric.name, metric.attributes): metric.value for metric in operations.snapshot().metrics
    }
    for queue, expected_attempts in (("activation", 2), ("model_evidence", 3)):
        attributes = (("queue", queue),)
        assert metrics[("monoid.postgres.outbox.max_attempts", attributes)] == expected_attempts
        oldest_age = metrics[("monoid.postgres.outbox.oldest_age", attributes)]
        assert 240 <= oldest_age < 600


def test_operations_snapshot_keeps_schema_and_aggregates_point_in_time(
    postgres_database: PostgresDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migrations = PostgresMigrations(postgres_database)
    migrations.apply()
    operations = PostgresOperations(postgres_database)
    operations.check_ready()
    inspection_started = threading.Event()
    mutation_committed = threading.Event()
    mutation_observed_at: list[datetime] = []
    original_inspect = PostgresMigrations._inspect

    def inspect_then_pause(
        migration_store: PostgresMigrations,
        connection: object,
    ) -> MigrationStatus:
        status = original_inspect(migration_store, connection)
        if not inspection_started.is_set():
            inspection_started.set()
            assert mutation_committed.wait(5)
        return status

    monkeypatch.setattr(PostgresMigrations, "_inspect", inspect_then_pause)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(operations.snapshot)
        assert inspection_started.wait(5)
        try:
            with postgres_database.transaction() as connection:
                with postgres_database.cursor(connection) as cursor:
                    cursor.execute(
                        sql.SQL(
                            "INSERT INTO {}.{} "
                            "(migration_id, ordinal, checksum_sha256, reader_floor, "
                            "writer_floor) VALUES (%s, %s, %s, %s, %s)"
                        ).format(
                            sql.Identifier(postgres_database.config.schema),
                            sql.Identifier("monoid_schema_migrations"),
                        ),
                        ("0007_snapshot_marker", 7, "f" * 64, 1, 1),
                    )
                    cursor.execute(
                        sql.SQL(
                            "INSERT INTO {}.{} "
                            "(run_id, owner_id, generation, leased_until, revoked, updated_at) "
                            "VALUES (%s, %s, 1, pg_catalog.clock_timestamp() + "
                            "interval '5 minutes', false, pg_catalog.clock_timestamp())"
                        ).format(
                            sql.Identifier(postgres_database.config.schema),
                            sql.Identifier("run_authority"),
                        ),
                        ("snapshot-new-run", "snapshot-new-owner"),
                    )
                    cursor.execute("SELECT pg_catalog.clock_timestamp()")
                    mutation_observed_at.append(cursor.fetchone()[0])
        finally:
            mutation_committed.set()
        snapshot = future.result(timeout=5)

    metrics = {(metric.name, metric.attributes): metric.value for metric in snapshot.metrics}
    assert snapshot.collected_at < mutation_observed_at[0]
    assert metrics[("monoid.postgres.schema.version", ())] == 6
    assert metrics[("monoid.postgres.authority.count", (("state", "total"),))] == 0
    assert migrations.status().current_version == 7


def test_forward_schema_uses_declared_reader_and_writer_floors(
    postgres_database: PostgresDatabase,
) -> None:
    migrations = PostgresMigrations(postgres_database)
    migrations.apply()
    operations = PostgresOperations(postgres_database)
    assert operations.check_ready().current_version == 6
    with postgres_database.connection() as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {}.{} "
                        "(migration_id, ordinal, checksum_sha256, reader_floor, writer_floor) "
                        "VALUES (%s, %s, %s, %s, %s)"
                    ).format(
                        sql.Identifier(postgres_database.config.schema),
                        sql.Identifier("monoid_schema_migrations"),
                    ),
                    ("0007_forward_marker", 7, "f" * 64, 1, 1),
                )

    compatible = migrations.status()
    assert compatible.current_version == 7
    assert compatible.pending == ()
    assert compatible.reader_compatible is True
    assert compatible.writer_compatible is True
    version_metric = next(
        metric
        for metric in operations.snapshot().metrics
        if metric.name == "monoid.postgres.schema.version"
    )
    assert version_metric.value == 7

    with postgres_database.connection() as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        "UPDATE {}.{} SET reader_floor = 7, writer_floor = 7 "
                        "WHERE migration_id = %s"
                    ).format(
                        sql.Identifier(postgres_database.config.schema),
                        sql.Identifier("monoid_schema_migrations"),
                    ),
                    ("0007_forward_marker",),
                )

    incompatible = migrations.status()
    assert incompatible.reader_compatible is False
    assert incompatible.writer_compatible is False
    with pytest.raises(PostgresSchemaIncompatible, match="not current"):
        migrations.require_reader_compatible()
    with pytest.raises(PostgresSchemaIncompatible, match="not current"):
        migrations.require_writer_compatible()
    with pytest.raises(PostgresSchemaIncompatible, match="not current"):
        operations.snapshot()
    with pytest.raises(PostgresSchemaIncompatible, match="check_ready"):
        operations.snapshot()


def test_migration_advisory_lock_serializes_independent_runners(
    postgres_database: PostgresDatabase,
) -> None:
    peer_databases = [
        PostgresDatabase(
            PostgresConfig(
                dsn=postgres_database.config.dsn,
                schema=postgres_database.config.schema,
                min_pool_size=1,
                max_pool_size=2,
                application_name=f"monoid-pr02-migrator-{index}",
            )
        )
        for index in range(2)
    ]
    for database in peer_databases:
        database.open()
    barrier = threading.Barrier(2)

    def migrate(index: int) -> tuple[str, ...]:
        barrier.wait(timeout=5)
        result = PostgresMigrations(peer_databases[index]).apply()
        return tuple(item.migration_id for item in result.applied)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            applied = tuple(executor.map(migrate, range(2)))
    finally:
        for database in peer_databases:
            database.close()

    assert sorted(applied, key=len) == [
        (),
        (
            "0001_authority",
            "0002_checkpoint_invocation",
            "0003_event_terminal_evidence_outbox",
            "0004_object_association_gc",
            "0005_activation_admission_dispatch",
            "0006_durable_stream",
        ),
    ]
    assert PostgresMigrations(postgres_database).status().current is True


def test_migration_path_resists_pooled_temp_table_shadowing(
    postgres_database: PostgresDatabase,
) -> None:
    caller_pool = ConnectionPool(
        postgres_database.config.dsn,
        min_size=1,
        max_size=1,
        open=False,
    )
    caller_pool.open(wait=True, timeout=10)
    database = PostgresDatabase(
        PostgresConfig(
            dsn=postgres_database.config.dsn,
            schema=postgres_database.config.schema,
            min_pool_size=1,
            max_pool_size=1,
            application_name="monoid-pr02-temp-shadow",
        ),
        pool=caller_pool,
    )
    database.open()
    try:
        with database.connection() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        CREATE TEMP TABLE monoid_schema_migrations (
                            migration_id text,
                            ordinal smallint,
                            checksum_sha256 character(64),
                            reader_floor smallint,
                            writer_floor smallint,
                            applied_at timestamp with time zone
                        )
                        """
                    )
                    cursor.execute(
                        """
                        CREATE TEMP TABLE run_authority (
                            leased_until timestamp with time zone,
                            revoked boolean
                        )
                        """
                    )

        result = PostgresMigrations(database).apply()
        assert result.status.current is True
        assert tuple(item.migration_id for item in result.applied) == (
            "0001_authority",
            "0002_checkpoint_invocation",
            "0003_event_terminal_evidence_outbox",
            "0004_object_association_gc",
            "0005_activation_admission_dispatch",
            "0006_durable_stream",
        )

        with database.connection() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute("SELECT count(*) FROM pg_temp.monoid_schema_migrations")
                    assert cursor.fetchone() == (0,)
                    cursor.execute(
                        """
                        SELECT EXISTS (
                            SELECT 1
                            FROM pg_class AS relation
                            JOIN pg_namespace AS namespace
                              ON namespace.oid = relation.relnamespace
                            WHERE namespace.nspname = %s
                              AND relation.relname = 'run_authority_expiry_idx'
                              AND relation.relkind = 'i'
                        ), EXISTS (
                            SELECT 1
                            FROM pg_class AS relation
                            WHERE relation.relnamespace = pg_my_temp_schema()
                              AND relation.relname = 'run_authority_expiry_idx'
                              AND relation.relkind = 'i'
                        )
                        """,
                        (database.config.schema,),
                    )
                    assert cursor.fetchone() == (True, False)
    finally:
        database.close()
        caller_pool.close()


def test_migration_schema_cannot_shadow_pg_catalog_builtins(
    postgres_database: PostgresDatabase,
) -> None:
    schema = postgres_database.config.schema
    with postgres_database.connection() as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
                cursor.execute(
                    sql.SQL(
                        "CREATE FUNCTION {}.{}() RETURNS pg_catalog.timestamptz "
                        "LANGUAGE SQL IMMUTABLE "
                        "AS 'SELECT ''-infinity''::pg_catalog.timestamptz'"
                    ).format(
                        sql.Identifier(schema),
                        sql.Identifier("clock_timestamp"),
                    )
                )
                cursor.execute(
                    sql.SQL(
                        "CREATE FUNCTION {}.{}(pg_catalog.text) RETURNS pg_catalog.int4 "
                        "LANGUAGE SQL IMMUTABLE AS 'SELECT 0::pg_catalog.int4'"
                    ).format(
                        sql.Identifier(schema),
                        sql.Identifier("octet_length"),
                    )
                )

    result = PostgresMigrations(postgres_database).apply()
    assert result.status.current is True
    store = PostgresWriterAuthorityStore(postgres_database)
    store.check_ready()
    lease = store.claim("run-shadow-builtins", "worker-a", timedelta(seconds=10))
    assert lease.observed_at.year >= 2020

    class RollBackDefaultProbe(Exception):
        pass

    try:
        with postgres_database.connection() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        sql.SQL(
                            "INSERT INTO {}.{} "
                            "(migration_id, ordinal, checksum_sha256, reader_floor, writer_floor) "
                            "VALUES (%s, %s, %s, %s, %s) RETURNING applied_at"
                        ).format(
                            sql.Identifier(schema),
                            sql.Identifier("monoid_schema_migrations"),
                        ),
                        ("9999_default_probe", 7, "f" * 64, 1, 1),
                    )
                    assert cursor.fetchone()[0].year >= 2020
                raise RollBackDefaultProbe
    except RollBackDefaultProbe:
        pass


def test_caller_pool_row_factory_does_not_change_adapter_rows(
    postgres_database: PostgresDatabase,
) -> None:
    caller_pool = ConnectionPool(
        postgres_database.config.dsn,
        kwargs={"row_factory": dict_row},
        min_size=1,
        max_size=1,
        open=False,
    )
    caller_pool.open(wait=True, timeout=10)
    database = PostgresDatabase(
        PostgresConfig(
            dsn=postgres_database.config.dsn,
            schema=postgres_database.config.schema,
            min_pool_size=1,
            max_pool_size=1,
            application_name="monoid-pr02-dict-row",
        ),
        pool=caller_pool,
    )
    try:
        database.open()
        assert PostgresMigrations(database).apply().status.current is True
        store = PostgresWriterAuthorityStore(database)
        store.check_ready()
        lease = store.claim("run-dict-row", "worker-a", timedelta(seconds=10))
        assert store.read("run-dict-row").writer_token == lease.writer_token  # type: ignore[union-attr]

        with database.connection() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1 AS value")
                    assert cursor.fetchone() == {"value": 1}
    finally:
        database.close()
        caller_pool.close()


@pytest.mark.parametrize(
    "isolation_level",
    [IsolationLevel.REPEATABLE_READ, IsolationLevel.SERIALIZABLE],
)
def test_migration_lock_wait_enforces_read_committed_on_caller_pools(
    postgres_database: PostgresDatabase,
    isolation_level: IsolationLevel,
) -> None:
    caller_pools: list[ConnectionPool[psycopg.Connection[tuple[object, ...]]]] = []
    peer_databases: list[PostgresDatabase] = []

    def configure(connection: psycopg.Connection[tuple[object, ...]]) -> None:
        connection.isolation_level = isolation_level

    for index in range(2):
        application_name = f"monoid-pr02-migration-{isolation_level.name.lower()}-{index}"
        caller_pool = ConnectionPool(
            postgres_database.config.dsn,
            kwargs={"application_name": application_name},
            min_size=1,
            max_size=1,
            open=False,
            configure=configure,
        )
        caller_pool.open(wait=True, timeout=10)
        database = PostgresDatabase(
            PostgresConfig(
                dsn=postgres_database.config.dsn,
                schema=postgres_database.config.schema,
                min_pool_size=1,
                max_pool_size=1,
                application_name=application_name,
            ),
            pool=caller_pool,
        )
        database.open()
        caller_pools.append(caller_pool)
        peer_databases.append(database)

    barrier = threading.Barrier(2)

    def migrate(index: int) -> tuple[str, ...]:
        barrier.wait(timeout=5)
        result = PostgresMigrations(peer_databases[index]).apply()
        return tuple(item.migration_id for item in result.applied)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            with postgres_database.connection() as blocking_connection:
                with blocking_connection.transaction():
                    with blocking_connection.cursor() as cursor:
                        cursor.execute(
                            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                            (f"monoid-agent-kernel:migrations:{postgres_database.config.schema}",),
                        )
                    futures = tuple(executor.submit(migrate, index) for index in range(2))
                    for database in peer_databases:
                        _wait_until_application_blocks_on_lock(
                            postgres_database,
                            database.config.application_name,
                        )
            applied = tuple(future.result(timeout=10) for future in futures)
    finally:
        for database in peer_databases:
            database.close()
        for caller_pool in caller_pools:
            caller_pool.close()

    assert sorted(applied, key=len) == [
        (),
        (
            "0001_authority",
            "0002_checkpoint_invocation",
            "0003_event_terminal_evidence_outbox",
            "0004_object_association_gc",
            "0005_activation_admission_dispatch",
            "0006_durable_stream",
        ),
    ]
    assert PostgresMigrations(postgres_database).status().current is True


def test_db_clock_writer_claim_renew_release_and_handoff(
    postgres_database: PostgresDatabase,
) -> None:
    PostgresMigrations(postgres_database).apply()
    store = PostgresWriterAuthorityStore(postgres_database)
    store.check_ready()
    ttl = timedelta(seconds=3)

    first = store.claim("run-authority", "worker-a", ttl)
    assert first.writer_token.generation == 1
    assert first.leased_until > first.observed_at

    repeated = store.claim("run-authority", "worker-a", timedelta(seconds=30))
    assert repeated.writer_token == first.writer_token
    assert repeated.leased_until == first.leased_until
    with pytest.raises(WriterLeaseUnavailable) as unavailable:
        store.claim("run-authority", "worker-b", ttl)
    assert unavailable.value.authority.writer_token == first.writer_token

    renewed = store.renew(first.writer_token, timedelta(seconds=10))
    assert renewed.status == "renewed"
    assert renewed.lease is not None
    assert renewed.lease.writer_token == first.writer_token
    assert renewed.lease.leased_until > first.leased_until

    wrong_owner = WriterToken(
        run_id=first.writer_token.run_id,
        owner_id="worker-b",
        generation=first.writer_token.generation,
    )
    assert store.renew(wrong_owner, ttl).status == "fenced"

    released = store.release(first.writer_token)
    assert released.status == "released"
    assert released.authority is not None and released.authority.revoked
    assert store.release(first.writer_token).status == "already_released"

    replacement = store.claim("run-authority", "worker-b", ttl)
    assert replacement.writer_token.generation == 2
    assert store.renew(first.writer_token, ttl).status == "fenced"
    assert store.release(first.writer_token).status == "fenced"
    assert store.read("run-authority").writer_token == replacement.writer_token  # type: ignore[union-attr]


def test_expired_token_cannot_renew_and_handoff_uses_database_clock(
    postgres_database: PostgresDatabase,
) -> None:
    PostgresMigrations(postgres_database).apply()
    store = PostgresWriterAuthorityStore(postgres_database)
    store.check_ready()
    expired = store.claim("run-expiry", "worker-a", timedelta(milliseconds=60))

    deadline = time.monotonic() + 5
    while True:
        observed = store.read("run-expiry")
        assert observed is not None
        if not observed.active:
            break
        if time.monotonic() >= deadline:
            pytest.fail("database clock did not observe lease expiry")
        time.sleep(0.01)

    assert store.renew(expired.writer_token, timedelta(seconds=1)).status == "fenced"
    assert store.release(expired.writer_token).status == "fenced"
    replacement = store.claim("run-expiry", "worker-b", timedelta(seconds=1))
    assert replacement.writer_token.generation == 2
    assert replacement.observed_at >= observed.observed_at


def test_concurrent_expiry_handoff_has_one_generation_winner_across_pools(
    postgres_database: PostgresDatabase,
) -> None:
    PostgresMigrations(postgres_database).apply()
    first_store = PostgresWriterAuthorityStore(postgres_database)
    first_store.check_ready()
    first_store.claim("run-race", "worker-a", timedelta(milliseconds=60))

    deadline = time.monotonic() + 5
    while first_store.read("run-race").active:  # type: ignore[union-attr]
        if time.monotonic() >= deadline:
            pytest.fail("database clock did not observe lease expiry")
        time.sleep(0.01)

    peer_databases = [
        PostgresDatabase(
            PostgresConfig(
                dsn=postgres_database.config.dsn,
                schema=postgres_database.config.schema,
                min_pool_size=1,
                max_pool_size=2,
                application_name=f"monoid-pr02-racer-{index}",
            )
        )
        for index in range(2)
    ]
    for database in peer_databases:
        database.open()
    barrier = threading.Barrier(2)

    def race(index: int) -> WriterLease | WriterLeaseUnavailable:
        barrier.wait(timeout=5)
        try:
            store = PostgresWriterAuthorityStore(peer_databases[index])
            store.check_ready()
            return store.claim(
                "run-race",
                f"worker-{index + 1}",
                timedelta(seconds=2),
            )
        except WriterLeaseUnavailable as exc:
            return exc

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = tuple(executor.map(race, range(2)))
    finally:
        for database in peer_databases:
            database.close()

    winners = [outcome for outcome in outcomes if isinstance(outcome, WriterLease)]
    losers = [outcome for outcome in outcomes if isinstance(outcome, WriterLeaseUnavailable)]
    assert len(winners) == len(losers) == 1
    assert winners[0].writer_token.generation == 2
    assert losers[0].authority.writer_token == winners[0].writer_token
    current = first_store.read("run-race")
    assert current is not None and current.writer_token == winners[0].writer_token


def test_lock_wait_samples_database_clock_after_the_row_lock(
    postgres_database: PostgresDatabase,
) -> None:
    PostgresMigrations(postgres_database).apply()
    first_store = PostgresWriterAuthorityStore(postgres_database)
    first_store.check_ready()
    first = first_store.claim("run-lock-clock", "worker-a", timedelta(milliseconds=700))

    waiter_database = PostgresDatabase(
        PostgresConfig(
            dsn=postgres_database.config.dsn,
            schema=postgres_database.config.schema,
            min_pool_size=1,
            max_pool_size=2,
            application_name="monoid-pr02-clock-waiter",
        )
    )
    waiter_database.open()
    waiter_store = PostgresWriterAuthorityStore(waiter_database)
    waiter_store.check_ready()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with postgres_database.connection() as blocking_connection:
                with blocking_connection.transaction():
                    with blocking_connection.cursor() as cursor:
                        cursor.execute(
                            sql.SQL("SELECT run_id FROM {}.{} WHERE run_id = %s FOR UPDATE").format(
                                sql.Identifier(postgres_database.config.schema),
                                sql.Identifier("run_authority"),
                            ),
                            (first.writer_token.run_id,),
                        )
                        assert cursor.fetchone() == (first.writer_token.run_id,)
                    future = executor.submit(
                        waiter_store.claim,
                        first.writer_token.run_id,
                        "worker-b",
                        timedelta(seconds=2),
                    )
                    _wait_until_application_blocks_on_lock(
                        postgres_database,
                        waiter_database.config.application_name,
                    )
                    time.sleep(0.85)
            replacement = future.result(timeout=5)
    finally:
        waiter_database.close()

    assert replacement.writer_token.generation == 2
    assert replacement.writer_token.owner_id == "worker-b"
    assert replacement.observed_at > first.leased_until


def test_read_observes_a_prior_inflight_authority_mutation(
    postgres_database: PostgresDatabase,
) -> None:
    PostgresMigrations(postgres_database).apply()
    first_store = PostgresWriterAuthorityStore(postgres_database)
    first_store.check_ready()
    first = first_store.claim("run-read-lock", "worker-a", timedelta(seconds=10))

    reader_database = PostgresDatabase(
        PostgresConfig(
            dsn=postgres_database.config.dsn,
            schema=postgres_database.config.schema,
            min_pool_size=1,
            max_pool_size=2,
            application_name="monoid-pr02-authority-reader",
        )
    )
    reader_database.open()
    reader_store = PostgresWriterAuthorityStore(reader_database)
    reader_store.check_ready()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with postgres_database.connection() as blocking_connection:
                with blocking_connection.transaction():
                    with blocking_connection.cursor() as cursor:
                        cursor.execute(
                            sql.SQL(
                                "WITH sampled AS (SELECT clock_timestamp() AS db_now) "
                                "UPDATE {}.{} AS authority SET "
                                "leased_until = sampled.db_now, revoked = true, "
                                "updated_at = sampled.db_now FROM sampled "
                                "WHERE authority.run_id = %s"
                            ).format(
                                sql.Identifier(postgres_database.config.schema),
                                sql.Identifier("run_authority"),
                            ),
                            (first.writer_token.run_id,),
                        )
                        assert cursor.rowcount == 1
                    future = executor.submit(reader_store.read, first.writer_token.run_id)
                    _wait_until_application_blocks_on_lock(
                        postgres_database,
                        reader_database.config.application_name,
                    )
            observed = future.result(timeout=5)
    finally:
        reader_database.close()

    assert observed is not None
    assert observed.writer_token == first.writer_token
    assert observed.revoked is True
    assert observed.observed_at >= observed.leased_until


def test_fresh_claim_samples_database_clock_after_unique_conflict_rollback(
    postgres_database: PostgresDatabase,
) -> None:
    PostgresMigrations(postgres_database).apply()
    waiter_database = PostgresDatabase(
        PostgresConfig(
            dsn=postgres_database.config.dsn,
            schema=postgres_database.config.schema,
            min_pool_size=1,
            max_pool_size=2,
            application_name="monoid-pr02-insert-waiter",
        )
    )
    waiter_database.open()
    waiter_store = PostgresWriterAuthorityStore(waiter_database)
    waiter_store.check_ready()

    class RollBackConflict(Exception):
        pass

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                with postgres_database.connection() as blocking_connection:
                    with blocking_connection.transaction():
                        with blocking_connection.cursor() as cursor:
                            cursor.execute(
                                sql.SQL(
                                    "WITH sampled AS (SELECT clock_timestamp() AS db_now) "
                                    "INSERT INTO {}.{} "
                                    "(run_id, owner_id, generation, leased_until, revoked, "
                                    "updated_at) "
                                    "SELECT %s, %s, 1, db_now + interval '10 seconds', false, "
                                    "db_now FROM sampled RETURNING updated_at"
                                ).format(
                                    sql.Identifier(postgres_database.config.schema),
                                    sql.Identifier("run_authority"),
                                ),
                                ("run-insert-wait", "rolled-back-owner"),
                            )
                            blocker_time = cursor.fetchone()[0]
                        future = executor.submit(
                            waiter_store.claim,
                            "run-insert-wait",
                            "worker-b",
                            timedelta(seconds=2),
                        )
                        _wait_until_application_blocks_on_lock(
                            postgres_database,
                            waiter_database.config.application_name,
                        )
                        time.sleep(0.85)
                        raise RollBackConflict
            except RollBackConflict:
                pass
            lease = future.result(timeout=5)
    finally:
        waiter_database.close()

    assert lease.writer_token.generation == 1
    assert lease.writer_token.owner_id == "worker-b"
    assert lease.observed_at >= blocker_time + timedelta(milliseconds=700)


@pytest.mark.parametrize(
    "isolation_level",
    [IsolationLevel.REPEATABLE_READ, IsolationLevel.SERIALIZABLE],
)
def test_claim_reconciliation_enforces_read_committed_on_caller_pool(
    postgres_database: PostgresDatabase,
    isolation_level: IsolationLevel,
) -> None:
    PostgresMigrations(postgres_database).apply()

    def configure(connection: psycopg.Connection[tuple[object, ...]]) -> None:
        connection.isolation_level = isolation_level

    application_name = f"monoid-pr02-{isolation_level.name.lower()}-waiter"
    caller_pool = ConnectionPool(
        postgres_database.config.dsn,
        kwargs={"application_name": application_name},
        min_size=1,
        max_size=2,
        open=False,
        configure=configure,
    )
    caller_pool.open(wait=True, timeout=10)
    waiter_database = PostgresDatabase(
        PostgresConfig(
            dsn=postgres_database.config.dsn,
            schema=postgres_database.config.schema,
            min_pool_size=1,
            max_pool_size=2,
            application_name=application_name,
        ),
        pool=caller_pool,
    )
    waiter_database.open()
    waiter_store = PostgresWriterAuthorityStore(waiter_database)
    waiter_store.check_ready()

    with waiter_database.connection() as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute("SHOW transaction_isolation")
                assert cursor.fetchone()[0] == isolation_level.name.lower().replace("_", " ")

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with postgres_database.connection() as blocking_connection:
                with blocking_connection.transaction():
                    with blocking_connection.cursor() as cursor:
                        cursor.execute(
                            sql.SQL(
                                "WITH sampled AS (SELECT clock_timestamp() AS db_now) "
                                "INSERT INTO {}.{} "
                                "(run_id, owner_id, generation, leased_until, revoked, "
                                "updated_at) "
                                "SELECT %s, %s, 1, db_now + interval '10 seconds', false, "
                                "db_now FROM sampled"
                            ).format(
                                sql.Identifier(postgres_database.config.schema),
                                sql.Identifier("run_authority"),
                            ),
                            ("run-isolation", "committed-owner"),
                        )

                    def claim() -> WriterLeaseUnavailable:
                        try:
                            waiter_store.claim(
                                "run-isolation",
                                "waiting-owner",
                                timedelta(seconds=2),
                            )
                        except WriterLeaseUnavailable as exc:
                            return exc
                        raise AssertionError("conflicting claimant unexpectedly acquired the lease")

                    future = executor.submit(claim)
                    _wait_until_application_blocks_on_lock(
                        postgres_database,
                        waiter_database.config.application_name,
                    )
            unavailable = future.result(timeout=5)
    finally:
        waiter_database.close()
        caller_pool.close()

    assert unavailable.authority.writer_token == WriterToken(
        run_id="run-isolation",
        owner_id="committed-owner",
        generation=1,
    )
