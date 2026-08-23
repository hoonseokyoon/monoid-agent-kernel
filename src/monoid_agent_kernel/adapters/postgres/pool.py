"""Synchronous PostgreSQL pool lifecycle with a lazy psycopg boundary."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator

from monoid_agent_kernel.adapters.postgres.config import PostgresConfig


class PostgresDependencyMissing(RuntimeError):
    """The PostgreSQL optional extra is unavailable."""


class PostgresDatabaseClosed(RuntimeError):
    """An operation attempted to borrow from a database that was not opened."""


class UnsupportedPostgresVersion(RuntimeError):
    """The connected server is older than the supported production floor."""


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
    def transaction(self) -> Iterator[Any]:
        """Start the adapter's READ COMMITTED transaction with a trusted local search path."""

        with self.connection() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    # This is deliberately the first statement. Adapter linearization may wait on
                    # row, unique-index, or advisory locks and then needs a new statement snapshot.
                    cursor.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                    # Caller-provided pooled sessions may carry an untrusted search_path. Adapter
                    # relations are schema-qualified; built-ins resolve from pg_catalog, and temp
                    # objects remain reachable only after it. SET LOCAL restores caller state.
                    cursor.execute("SET LOCAL search_path TO pg_catalog, pg_temp")
                yield connection

    def health(self) -> PostgresHealth:
        """Verify connectivity, supported major version, and database-clock availability."""

        with self.transaction() as connection:
            with connection.cursor() as cursor:
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
