from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Iterator

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.serial, pytest.mark.service]


def _selected() -> bool:
    return os.environ.get("MONOID_SERVICE_PROFILE") in {"postgres", "objectstore", "combined"}


if not _selected():
    pytest.skip("PostgreSQL service profile is not selected", allow_module_level=True)

import psycopg  # noqa: E402
from psycopg import sql  # noqa: E402

from monoid_agent_kernel.adapters.postgres import (  # noqa: E402
    PostgresConfig,
    PostgresDatabase,
    PostgresMigrationDrift,
    PostgresMigrations,
    PostgresSchemaIncompatible,
    PostgresWriterAuthorityStore,
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


def test_explicit_migration_lifecycle_and_doctor(
    postgres_database: PostgresDatabase,
) -> None:
    migrations = PostgresMigrations(postgres_database)
    store = PostgresWriterAuthorityStore(postgres_database)

    before = migrations.status()
    assert before.schema_exists is False
    assert before.current_version == 0
    assert tuple(item.migration_id for item in migrations.plan().pending) == ("0001_authority",)
    assert migrations.doctor().ok is False
    with pytest.raises(PostgresSchemaIncompatible, match="check_ready"):
        store.read("run-unready")
    with pytest.raises(PostgresSchemaIncompatible, match="not current"):
        migrations.require_reader_compatible()
    with pytest.raises(PostgresSchemaIncompatible, match="not current"):
        store.check_ready()

    first = migrations.apply()
    assert tuple(item.migration_id for item in first.applied) == ("0001_authority",)
    assert first.status.current is True
    assert first.status.current_version == 1
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
                    sql.SQL(
                        "UPDATE {}.{} SET checksum_sha256 = %s WHERE migration_id = %s"
                    ).format(
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


def test_forward_schema_uses_declared_reader_and_writer_floors(
    postgres_database: PostgresDatabase,
) -> None:
    migrations = PostgresMigrations(postgres_database)
    migrations.apply()
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
                    ("0002_forward_marker", 2, "f" * 64, 1, 1),
                )

    compatible = migrations.status()
    assert compatible.current_version == 2
    assert compatible.pending == ()
    assert compatible.reader_compatible is True
    assert compatible.writer_compatible is True

    with postgres_database.connection() as connection:
        with connection.transaction():
            with connection.cursor() as cursor:
                cursor.execute(
                    sql.SQL(
                        "UPDATE {}.{} SET reader_floor = 2, writer_floor = 2 "
                        "WHERE migration_id = %s"
                    ).format(
                        sql.Identifier(postgres_database.config.schema),
                        sql.Identifier("monoid_schema_migrations"),
                    ),
                    ("0002_forward_marker",),
                )

    incompatible = migrations.status()
    assert incompatible.reader_compatible is False
    assert incompatible.writer_compatible is False
    with pytest.raises(PostgresSchemaIncompatible, match="not current"):
        migrations.require_reader_compatible()
    with pytest.raises(PostgresSchemaIncompatible, match="not current"):
        migrations.require_writer_compatible()


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

    assert sorted(applied, key=len) == [(), ("0001_authority",)]
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
                    deadline = time.monotonic() + 5
                    while True:
                        with postgres_database.connection() as observer_connection:
                            with observer_connection.transaction():
                                with observer_connection.cursor() as cursor:
                                    cursor.execute(
                                        "SELECT EXISTS ("
                                        "SELECT 1 FROM pg_stat_activity "
                                        "WHERE application_name = %s AND wait_event_type = 'Lock'"
                                        ")",
                                        (waiter_database.config.application_name,),
                                    )
                                    blocked = bool(cursor.fetchone()[0])
                        if blocked:
                            break
                        if time.monotonic() >= deadline:
                            pytest.fail("claimant did not block on the authority row lock")
                        time.sleep(0.01)
                    time.sleep(0.85)
            replacement = future.result(timeout=5)
    finally:
        waiter_database.close()

    assert replacement.writer_token.generation == 2
    assert replacement.writer_token.owner_id == "worker-b"
    assert replacement.observed_at > first.leased_until
