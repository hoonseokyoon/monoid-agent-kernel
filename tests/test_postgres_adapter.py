from __future__ import annotations

import hashlib
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.resources import files
from typing import Iterator

import pytest

from monoid_agent_kernel.adapters.postgres import (
    PostgresConfig,
    PostgresDatabase,
    PostgresDatabaseClosed,
    bundled_migrations,
)
from monoid_agent_kernel.adapters.postgres.admission import _duration_microseconds


class _FakeCursor:
    def __init__(self, row: tuple[object, ...]) -> None:
        self.row = row
        self.executed: list[tuple[object, object]] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, statement: object, parameters: object = None) -> None:
        self.executed.append((statement, parameters))

    def fetchone(self) -> tuple[object, ...]:
        if self.executed and "statement_timestamp()" in str(self.executed[-1][0]):
            return (datetime(2026, 8, 24, tzinfo=UTC), "30000ms", "300000ms")
        return self.row


class _FakeConnection:
    def __init__(self) -> None:
        self.cursor_value = _FakeCursor((160015, datetime(2026, 8, 24, tzinfo=UTC)))
        self.transactions = 0
        self.row_factories: list[object | None] = []

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.transactions += 1
        yield

    def cursor(self, *, row_factory: object | None = None) -> _FakeCursor:
        self.row_factories.append(row_factory)
        return self.cursor_value


class _FakePool:
    def __init__(self) -> None:
        self.connection_value = _FakeConnection()
        self.open_calls: list[tuple[bool, float]] = []
        self.close_calls = 0
        self.borrow_timeouts: list[float] = []

    def open(self, *, wait: bool, timeout: float) -> None:
        self.open_calls.append((wait, timeout))

    def close(self) -> None:
        self.close_calls += 1

    @contextmanager
    def connection(self, *, timeout: float) -> Iterator[_FakeConnection]:
        self.borrow_timeouts.append(timeout)
        yield self.connection_value


def test_admission_durations_are_time_only_microseconds() -> None:
    assert _duration_microseconds(0) == 0
    assert _duration_microseconds(0.0000001) == 1
    assert _duration_microseconds(86_400) == 86_400_000_000


def test_postgres_config_hides_dsn_and_validates_schema_and_pool() -> None:
    config = PostgresConfig(dsn="postgresql://user:secret@db/service")
    assert "secret" not in repr(config)
    assert config.schema == "monoid_kernel"

    for schema in (
        "",
        "quoted.schema",
        "schema-name",
        "1schema",
        "x" * 64,
        "pg_temp",
        "PG_CATALOG",
        "pg_toast_42",
        "information_schema",
        "INFORMATION_SCHEMA",
    ):
        with pytest.raises(ValueError, match="schema"):
            PostgresConfig(dsn="postgresql://db/service", schema=schema)
    assert PostgresConfig(dsn="postgresql://db/service", schema="public").schema == "public"
    with pytest.raises(ValueError, match="max_pool_size"):
        PostgresConfig(dsn="postgresql://db/service", min_pool_size=2, max_pool_size=1)
    for field_name in ("lock_timeout_s", "statement_timeout_s"):
        for invalid_timeout in (True, 0, float("nan"), 86_401):
            with pytest.raises(ValueError, match=field_name):
                PostgresConfig(
                    dsn="postgresql://db/service",
                    **{field_name: invalid_timeout},
                )
    assert PostgresConfig(dsn="postgresql://db/service").max_bytea_blob_bytes == 8 * 1024 * 1024
    for invalid_limit in (True, 0, -1, 1 << 30):
        with pytest.raises(ValueError, match="max_bytea_blob_bytes"):
            PostgresConfig(
                dsn="postgresql://db/service",
                max_bytea_blob_bytes=invalid_limit,
            )


def test_external_pool_is_health_checked_but_never_opened_or_closed() -> None:
    pool = _FakePool()
    database = PostgresDatabase(
        PostgresConfig(dsn="postgresql://unused/service", pool_timeout_s=7),
        pool=pool,
    )

    health = database.open()
    assert health.server_major == 16
    assert pool.open_calls == []
    assert pool.borrow_timeouts == [7.0]
    database.close()
    assert pool.close_calls == 0


def test_owned_pool_lifecycle_is_explicit_and_reopenable(monkeypatch: pytest.MonkeyPatch) -> None:
    pools: list[_FakePool] = []
    database = PostgresDatabase(PostgresConfig(dsn="postgresql://unused/service"))

    def new_pool() -> _FakePool:
        pool = _FakePool()
        pools.append(pool)
        return pool

    monkeypatch.setattr(database, "_new_pool", new_pool)

    with pytest.raises(PostgresDatabaseClosed):
        database.health()
    database.open()
    database.close()
    database.open()
    database.close()

    assert len(pools) == 2
    assert all(pool.open_calls == [(True, 30.0)] for pool in pools)
    assert all(pool.close_calls == 1 for pool in pools)


def test_adapter_construction_has_no_connection_or_migration_side_effect() -> None:
    pool = _FakePool()
    database = PostgresDatabase(PostgresConfig(dsn="postgresql://unused/service"), pool=pool)

    assert database.opened is False
    assert pool.open_calls == []
    assert pool.borrow_timeouts == []


def test_adapter_transaction_pins_isolation_and_trusted_search_path() -> None:
    pool = _FakePool()
    database = PostgresDatabase(PostgresConfig(dsn="postgresql://unused/service"), pool=pool)
    database.open()
    pool.connection_value.cursor_value.executed.clear()
    pool.connection_value.row_factories.clear()

    with database.transaction():
        pass

    assert pool.connection_value.cursor_value.executed == [
        ("SET TRANSACTION ISOLATION LEVEL READ COMMITTED", None),
        ("SET LOCAL search_path TO pg_catalog, pg_temp", None),
        (
            "SELECT pg_catalog.statement_timestamp(), "
            "pg_catalog.set_config('lock_timeout', %s, true), "
            "pg_catalog.set_config('statement_timeout', %s, true)",
            ("30000ms", "300000ms"),
        ),
    ]
    assert len(pool.connection_value.row_factories) == 1
    assert getattr(pool.connection_value.row_factories[0], "__name__", "") == "_tuple_row"
    database.close()


def test_read_snapshot_returns_the_first_setup_statement_boundary() -> None:
    pool = _FakePool()
    database = PostgresDatabase(PostgresConfig(dsn="postgresql://unused/service"), pool=pool)
    database.open()
    pool.connection_value.cursor_value.executed.clear()

    with database.read_snapshot() as (connection, snapshot_boundary):
        assert connection is pool.connection_value
        assert snapshot_boundary == datetime(2026, 8, 24, tzinfo=UTC)

    assert pool.connection_value.cursor_value.executed[0] == (
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY",
        None,
    )
    database.close()


def test_bundled_migration_metadata_hashes_exact_wheel_resource() -> None:
    migrations = bundled_migrations()
    assert tuple(item.migration_id for item in migrations) == (
        "0001_authority",
        "0002_checkpoint_invocation",
        "0003_event_terminal_evidence_outbox",
        "0004_object_association_gc",
        "0005_activation_admission_dispatch",
        "0006_durable_stream",
    )
    root = files("monoid_agent_kernel.adapters.postgres").joinpath("sql")
    for migration in migrations:
        raw = root.joinpath(f"{migration.migration_id}.sql").read_bytes()
        assert migration.checksum_sha256 == hashlib.sha256(raw).hexdigest()
        assert migration.reader_floor == migration.writer_floor == 1
