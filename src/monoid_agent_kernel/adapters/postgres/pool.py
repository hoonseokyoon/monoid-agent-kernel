"""Synchronous PostgreSQL pool lifecycle with a lazy psycopg boundary."""

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator, Literal

from monoid_agent_kernel.adapters.postgres.config import PostgresConfig


class PostgresDependencyMissing(RuntimeError):
    """The PostgreSQL optional extra is unavailable."""


class PostgresDatabaseClosed(RuntimeError):
    """An operation attempted to borrow from a database that was not opened."""


class UnsupportedPostgresVersion(RuntimeError):
    """The connected server is older than the supported production floor."""


def _tuple_row(_cursor: object) -> type[tuple]:
    """Return positional rows without importing the optional psycopg package."""

    return tuple


def _timeout_milliseconds(seconds: float) -> str:
    return f"{max(1, math.ceil(float(seconds) * 1_000))}ms"


@dataclass(frozen=True, kw_only=True)
class PostgresHealth:
    server_version_num: int
    database_time: datetime

    @property
    def server_major(self) -> int:
        return self.server_version_num // 10000


class PostgresDatabase:
    """One adapter database and either an owned or caller-owned compatible sync pool.

    Construction never connects or migrates. ``open`` creates and health-checks an owned pool. A
    caller-provided pool must already be open; the wrapper borrows it for health checks and never
    opens or closes it.
    """

    def __init__(self, config: PostgresConfig, *, pool: object | None = None) -> None:
        if not isinstance(config, PostgresConfig):
            raise TypeError("PostgresDatabase config must be PostgresConfig")
        if pool is not None and not callable(getattr(pool, "connection", None)):
            raise TypeError("caller-provided PostgreSQL pool must expose connection()")
        self.config = config
        self._pool: object | None = pool
        self._owns_pool = pool is None
        self._opened = False
        self._lifecycle_lock = threading.RLock()

    @property
    def owns_pool(self) -> bool:
        return self._owns_pool

    @property
    def opened(self) -> bool:
        with self._lifecycle_lock:
            return self._opened

    def _new_pool(self) -> object:
        try:
            from psycopg.pq import TransactionStatus
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # pragma: no cover - exercised in isolated import tests
            raise PostgresDependencyMissing(
                "install monoid-agent-kernel[postgres] to use the PostgreSQL adapter"
            ) from exc

        def reset(connection: Any) -> None:
            if connection.info.transaction_status != TransactionStatus.IDLE:
                connection.rollback()

        return ConnectionPool(
            self.config.dsn,
            kwargs={
                "connect_timeout": self.config.connect_timeout_s,
                "application_name": self.config.application_name,
            },
            min_size=self.config.min_pool_size,
            max_size=self.config.max_pool_size,
            open=False,
            check=ConnectionPool.check_connection,
            reset=reset,
            timeout=float(self.config.pool_timeout_s),
            name=f"monoid-{self.config.schema}",
        )

    def open(self) -> PostgresHealth:
        """Open an owned pool or attach to an already-open caller pool, then check the server."""

        with self._lifecycle_lock:
            if self._opened:
                return self.health()
            if self._owns_pool:
                pool = self._new_pool()
                self._pool = pool
                try:
                    pool.open(wait=True, timeout=float(self.config.pool_timeout_s))  # type: ignore[attr-defined]
                except BaseException:
                    try:
                        pool.close()  # type: ignore[attr-defined]
                    finally:
                        self._pool = None
                    raise
            self._opened = True
            try:
                return self.health()
            except BaseException:
                self._opened = False
                if self._owns_pool and self._pool is not None:
                    try:
                        self._pool.close()  # type: ignore[attr-defined]
                    finally:
                        self._pool = None
                raise

    def close(self) -> None:
        """Detach and close only an adapter-owned pool."""

        with self._lifecycle_lock:
            if not self._opened:
                return
            self._opened = False
            if self._owns_pool and self._pool is not None:
                try:
                    self._pool.close()  # type: ignore[attr-defined]
                finally:
                    self._pool = None

    @contextmanager
    def connection(self) -> Iterator[Any]:
        """Borrow one connection from the opened pool."""

        with self._lifecycle_lock:
            if not self._opened or self._pool is None:
                raise PostgresDatabaseClosed("PostgreSQL database must be opened before use")
            pool = self._pool
        with pool.connection(timeout=float(self.config.pool_timeout_s)) as connection:  # type: ignore[attr-defined]
            yield connection

    @contextmanager
    def cursor(self, connection: Any) -> Iterator[Any]:
        """Create an adapter cursor whose rows are positional regardless of caller pool policy."""

        with connection.cursor(row_factory=_tuple_row) as cursor:
            yield cursor

    @contextmanager
    def _configured_transaction(
        self,
        *,
        read_only: bool,
        isolation_level: Literal["read_committed", "repeatable_read"],
    ) -> Iterator[tuple[Any, datetime]]:
        """Start one configured transaction and return its setup-statement boundary."""

        if type(read_only) is not bool:
            raise TypeError("PostgreSQL transaction read_only must be a boolean")
        if type(isolation_level) is not str:
            raise TypeError("PostgreSQL transaction isolation_level must be a string")
        isolation_sql = {
            "read_committed": "READ COMMITTED",
            "repeatable_read": "REPEATABLE READ",
        }.get(isolation_level)
        if isolation_sql is None:
            raise ValueError("PostgreSQL transaction isolation_level is unsupported")

        with self.connection() as connection:
            with connection.transaction():
                with self.cursor(connection) as cursor:
                    # This is deliberately the first statement. READ COMMITTED mutation paths may
                    # wait on row, unique-index, or advisory locks and then need a new statement
                    # snapshot. Aggregate operations select REPEATABLE READ explicitly.
                    cursor.execute(
                        f"SET TRANSACTION ISOLATION LEVEL {isolation_sql}"
                        + (", READ ONLY" if read_only else "")
                    )
                    # Caller-provided pooled sessions may carry an untrusted search_path. Adapter
                    # relations are schema-qualified; built-ins resolve from pg_catalog, and temp
                    # objects remain reachable only after it. SET LOCAL restores caller state.
                    cursor.execute("SET LOCAL search_path TO pg_catalog, pg_temp")
                    cursor.execute(
                        "SELECT pg_catalog.statement_timestamp(), "
                        "pg_catalog.set_config('lock_timeout', %s, true), "
                        "pg_catalog.set_config('statement_timeout', %s, true)",
                        (
                            _timeout_milliseconds(self.config.lock_timeout_s),
                            _timeout_milliseconds(self.config.statement_timeout_s),
                        ),
                    )
                    setup = cursor.fetchone()
                    if setup is None or not isinstance(setup[0], datetime):
                        raise RuntimeError(
                            "PostgreSQL transaction setup returned no snapshot boundary"
                        )
                    setup_boundary = setup[0]
                yield connection, setup_boundary

    @contextmanager
    def transaction(
        self,
        *,
        read_only: bool = False,
        isolation_level: Literal["read_committed", "repeatable_read"] = "read_committed",
    ) -> Iterator[Any]:
        """Start a bounded adapter transaction with a trusted local search path."""

        with self._configured_transaction(
            read_only=read_only,
            isolation_level=isolation_level,
        ) as (connection, _setup_boundary):
            yield connection

    @contextmanager
    def read_snapshot(self) -> Iterator[tuple[Any, datetime]]:
        """Yield one repeatable read-only connection and its first-statement boundary."""

        with self._configured_transaction(
            read_only=True,
            isolation_level="repeatable_read",
        ) as snapshot:
            yield snapshot

    def health(self) -> PostgresHealth:
        """Verify connectivity, supported major version, and database-clock availability."""

        with self.transaction(read_only=True) as connection:
            with self.cursor(connection) as cursor:
                cursor.execute(
                    "SELECT pg_catalog.current_setting('server_version_num')::pg_catalog.int4, "
                    "pg_catalog.clock_timestamp()"
                )
                row = cursor.fetchone()
        if row is None:  # pragma: no cover - PostgreSQL SELECT always returns one row
            raise RuntimeError("PostgreSQL health query returned no row")
        health = PostgresHealth(server_version_num=int(row[0]), database_time=row[1])
        if health.server_major < 16:
            raise UnsupportedPostgresVersion(
                f"PostgreSQL {health.server_major} is below the supported major 16"
            )
        return health

    def __enter__(self) -> PostgresDatabase:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = [
    "PostgresDependencyMissing",
    "PostgresDatabaseClosed",
    "UnsupportedPostgresVersion",
    "PostgresHealth",
    "PostgresDatabase",
]
