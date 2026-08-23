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
        return self.row


class _FakeConnection:
    def __init__(self) -> None:
        self.cursor_value = _FakeCursor((160015, datetime(2026, 8, 24, tzinfo=UTC)))
        self.transactions = 0

    @contextmanager
    def transaction(self) -> Iterator[None]:
        self.transactions += 1
        yield

    def cursor(self) -> _FakeCursor:
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


def test_postgres_config_hides_dsn_and_validates_schema_and_pool() -> None:
    config = PostgresConfig(dsn="postgresql://user:secret@db/service")
    assert "secret" not in repr(config)
    assert config.schema == "monoid_kernel"

    for schema in ("", "quoted.schema", "schema-name", "1schema", "x" * 64):
        with pytest.raises(ValueError, match="schema"):
            PostgresConfig(dsn="postgresql://db/service", schema=schema)
    with pytest.raises(ValueError, match="max_pool_size"):
        PostgresConfig(dsn="postgresql://db/service", min_pool_size=2, max_pool_size=1)


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

    with database.transaction():
        pass

    assert pool.connection_value.cursor_value.executed == [
        ("SET TRANSACTION ISOLATION LEVEL READ COMMITTED", None),
        ("SET LOCAL search_path TO pg_catalog, pg_temp", None),
    ]
    database.close()


def test_bundled_migration_metadata_hashes_exact_wheel_resource() -> None:
    migrations = bundled_migrations()
    assert tuple(item.migration_id for item in migrations) == ("0001_authority",)
    raw = (
        files("monoid_agent_kernel.adapters.postgres")
        .joinpath("sql", "0001_authority.sql")
        .read_bytes()
    )
    assert migrations[0].checksum_sha256 == hashlib.sha256(raw).hexdigest()
    assert migrations[0].reader_floor == migrations[0].writer_floor == 1
