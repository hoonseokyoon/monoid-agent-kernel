"""PostgreSQL DB-clock writer generations and exact-token leases."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from monoid_agent_kernel.adapters.postgres.migrations import (
    MigrationStatus,
    PostgresMigrations,
    PostgresSchemaIncompatible,
)
from monoid_agent_kernel.adapters.postgres.pool import PostgresDatabase
from monoid_agent_kernel.core.safe_evidence import is_safe_opaque_id
from monoid_agent_kernel.hosting import (
    ReleaseResult,
    RenewResult,
    WriterAuthority,
    WriterLease,
    WriterLeaseUnavailable,
    WriterToken,
)


_AUTHORITY_TABLE = "run_authority"
_ELAPSED_TTL_INTERVAL = (
    "(%s::pg_catalog.text || ' microseconds')::pg_catalog.interval"
)


def _ttl_microseconds(ttl: object) -> int:
    if type(ttl) is not timedelta or ttl <= timedelta(0):
        raise ValueError("writer lease ttl must be a positive timedelta")
    # psycopg preserves ``timedelta.days`` in PostgreSQL's interval day field. Adding that field
    # to timestamptz follows calendar days in the session timezone and can be 23 or 25 hours at a
    # DST boundary. Encode the complete duration as time-only microseconds instead.
    return ((ttl.days * 86_400) + ttl.seconds) * 1_000_000 + ttl.microseconds


def _authority_from_row(row: Any) -> WriterAuthority:
    return WriterAuthority(
        writer_token=WriterToken(
            run_id=str(row[0]),
            owner_id=str(row[1]),
            generation=int(row[2]),
        ),
        leased_until=row[3],
        revoked=bool(row[4]),
        observed_at=row[5],
    )


class PostgresWriterAuthorityStore:
    """Canonical run writer authority stored under a PostgreSQL row lock."""

    def __init__(self, database: PostgresDatabase) -> None:
        if not isinstance(database, PostgresDatabase):
            raise TypeError("PostgresWriterAuthorityStore database must be PostgresDatabase")
        self.database = database
        self._ready = False

    def _qualified_table(self) -> object:
        from psycopg import sql

        return sql.Identifier(self.database.config.schema, _AUTHORITY_TABLE)

    def check_ready(self) -> MigrationStatus:
        """Fail before serving mutations unless the installed schema is current for this writer."""

        self._ready = False
        status = PostgresMigrations(self.database).require_writer_compatible()
        self._ready = True
        return status

    def _require_ready(self) -> None:
        if not self._ready:
            raise PostgresSchemaIncompatible(
                "PostgreSQL writer authority store requires a successful check_ready()"
            )

    def _read_locked(self, cursor: object, run_id: str) -> WriterAuthority | None:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "SELECT run_id, owner_id, generation, leased_until, revoked "
                "FROM {} WHERE run_id = %s FOR UPDATE"
            ).format(self._qualified_table()),
            (run_id,),
        )
        row = cursor.fetchone()  # type: ignore[attr-defined]
        if row is None:
            return None
        # PostgreSQL may evaluate SELECT target-list expressions before FOR UPDATE finishes
        # waiting. Sample in a second statement so expiry is judged strictly after this
        # transaction owns the row lock.
        cursor.execute("SELECT pg_catalog.clock_timestamp()")  # type: ignore[attr-defined]
        clock_row = cursor.fetchone()  # type: ignore[attr-defined]
        if clock_row is None:  # pragma: no cover - PostgreSQL SELECT always returns one row
            raise RuntimeError("PostgreSQL writer authority clock query returned no row")
        return _authority_from_row((*row, clock_row[0]))

    def _insert_first(
        self,
        cursor: object,
        run_id: str,
        owner_id: str,
        ttl_microseconds: int,
    ) -> WriterLease | None:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "WITH sampled AS (SELECT pg_catalog.clock_timestamp() AS db_now) "
                "INSERT INTO {} "
                "(run_id, owner_id, generation, leased_until, revoked, updated_at) "
                "SELECT %s, %s, 1, db_now, true, db_now FROM sampled "
                "ON CONFLICT (run_id) DO NOTHING "
                "RETURNING run_id"
            ).format(self._qualified_table()),
            (run_id, owner_id),
        )
        row = cursor.fetchone()  # type: ignore[attr-defined]
        if row is None:
            return None
        # A unique-index conflict can wait for another uncommitted insertion and then succeed if
        # that transaction rolls back. The first statement writes only an invisible revoked
        # placeholder; start the active TTL from a fresh clock sampled after the wait completed.
        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "WITH sampled AS (SELECT pg_catalog.clock_timestamp() AS db_now) "
                "UPDATE {} AS authority SET "
                "leased_until = sampled.db_now + "
                + _ELAPSED_TTL_INTERVAL
                + ", revoked = false, "
                "updated_at = sampled.db_now FROM sampled "
                "WHERE authority.run_id = %s AND authority.owner_id = %s "
                "AND authority.generation = 1 AND authority.revoked "
                "RETURNING authority.run_id, authority.owner_id, authority.generation, "
                "authority.leased_until, authority.revoked, sampled.db_now"
            ).format(self._qualified_table()),
            (ttl_microseconds, run_id, owner_id),
        )
        activated_row = cursor.fetchone()  # type: ignore[attr-defined]
        if activated_row is None:  # pragma: no cover - this transaction owns its inserted row
            raise RuntimeError("PostgreSQL writer claim could not activate its inserted row")
        authority = _authority_from_row(activated_row)
        return WriterLease(
            writer_token=authority.writer_token,
            leased_until=authority.leased_until,
            observed_at=authority.observed_at,
        )

    def _take_over(
        self,
        cursor: object,
        run_id: str,
        owner_id: str,
        ttl_microseconds: int,
    ) -> WriterLease:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "WITH sampled AS (SELECT pg_catalog.clock_timestamp() AS db_now) "
                "UPDATE {} AS authority SET "
                "owner_id = %s, generation = authority.generation + 1, "
                "leased_until = sampled.db_now + "
                + _ELAPSED_TTL_INTERVAL
                + ", revoked = false, "
                "updated_at = sampled.db_now "
                "FROM sampled WHERE authority.run_id = %s "
                "RETURNING authority.run_id, authority.owner_id, authority.generation, "
                "authority.leased_until, authority.revoked, sampled.db_now"
            ).format(self._qualified_table()),
            (owner_id, ttl_microseconds, run_id),
        )
        row = cursor.fetchone()  # type: ignore[attr-defined]
        if row is None:  # pragma: no cover - caller holds the existing row lock
            raise RuntimeError("PostgreSQL writer authority disappeared during takeover")
        authority = _authority_from_row(row)
        return WriterLease(
            writer_token=authority.writer_token,
            leased_until=authority.leased_until,
            observed_at=authority.observed_at,
        )

    def claim(self, run_id: str, owner_id: str, ttl: timedelta) -> WriterLease:
        self._require_ready()
        if not is_safe_opaque_id(run_id):
            raise ValueError("writer claim run_id must be a bounded opaque id")
        if not is_safe_opaque_id(owner_id):
            raise ValueError("writer claim owner_id must be a bounded opaque id")
        ttl_microseconds = _ttl_microseconds(ttl)

        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                current = self._read_locked(cursor, run_id)
                if current is None:
                    inserted = self._insert_first(cursor, run_id, owner_id, ttl_microseconds)
                    if inserted is not None:
                        return inserted
                    # A concurrent absent-row insert committed while ON CONFLICT waited. This
                    # new statement observes and locks that winner under READ COMMITTED.
                    current = self._read_locked(cursor, run_id)
                    if current is None:  # pragma: no cover - conflicting row must now exist
                        raise RuntimeError("PostgreSQL writer claim conflict lost its winner")
                if current.active:
                    if current.writer_token.owner_id == owner_id:
                        return WriterLease(
                            writer_token=current.writer_token,
                            leased_until=current.leased_until,
                            observed_at=current.observed_at,
                        )
                    raise WriterLeaseUnavailable(current)
                return self._take_over(cursor, run_id, owner_id, ttl_microseconds)

    def renew(self, writer_token: WriterToken, ttl: timedelta) -> RenewResult:
        self._require_ready()
        if not isinstance(writer_token, WriterToken):
            raise TypeError("writer renew requires WriterToken")
        ttl_microseconds = _ttl_microseconds(ttl)
        from psycopg import sql

        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                current = self._read_locked(cursor, writer_token.run_id)
                if current is None or current.writer_token != writer_token or not current.active:
                    return RenewResult(status="fenced", authority=current)
                cursor.execute(
                    sql.SQL(
                        "WITH sampled AS (SELECT pg_catalog.clock_timestamp() AS db_now) "
                        "UPDATE {} AS authority SET "
                        "leased_until = sampled.db_now + "
                        + _ELAPSED_TTL_INTERVAL
                        + ", updated_at = sampled.db_now "
                        "FROM sampled WHERE authority.run_id = %s "
                        "AND authority.owner_id = %s AND authority.generation = %s "
                        "AND NOT authority.revoked AND authority.leased_until > sampled.db_now "
                        "RETURNING authority.run_id, authority.owner_id, "
                        "authority.generation, authority.leased_until, authority.revoked, "
                        "sampled.db_now"
                    ).format(self._qualified_table()),
                    (
                        ttl_microseconds,
                        writer_token.run_id,
                        writer_token.owner_id,
                        writer_token.generation,
                    ),
                )
                row = cursor.fetchone()
                if row is None:
                    # The database clock crossed expiry between observation and UPDATE.
                    current = self._read_locked(cursor, writer_token.run_id)
                    return RenewResult(status="fenced", authority=current)
                authority = _authority_from_row(row)
                return RenewResult(
                    status="renewed",
                    lease=WriterLease(
                        writer_token=authority.writer_token,
                        leased_until=authority.leased_until,
                        observed_at=authority.observed_at,
                    ),
                )

    def release(self, writer_token: WriterToken) -> ReleaseResult:
        self._require_ready()
        if not isinstance(writer_token, WriterToken):
            raise TypeError("writer release requires WriterToken")
        from psycopg import sql

        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                current = self._read_locked(cursor, writer_token.run_id)
                if current is None or current.writer_token != writer_token:
                    return ReleaseResult(status="fenced", authority=current)
                if current.revoked:
                    return ReleaseResult(status="already_released", authority=current)
                if not current.active:
                    return ReleaseResult(status="fenced", authority=current)
                cursor.execute(
                    sql.SQL(
                        "WITH sampled AS (SELECT pg_catalog.clock_timestamp() AS db_now) "
                        "UPDATE {} AS authority SET "
                        "leased_until = sampled.db_now, revoked = true, "
                        "updated_at = sampled.db_now "
                        "FROM sampled WHERE authority.run_id = %s "
                        "AND authority.owner_id = %s AND authority.generation = %s "
                        "AND NOT authority.revoked AND authority.leased_until > sampled.db_now "
                        "RETURNING authority.run_id, authority.owner_id, "
                        "authority.generation, authority.leased_until, authority.revoked, "
                        "sampled.db_now"
                    ).format(self._qualified_table()),
                    (
                        writer_token.run_id,
                        writer_token.owner_id,
                        writer_token.generation,
                    ),
                )
                row = cursor.fetchone()
                if row is None:
                    current = self._read_locked(cursor, writer_token.run_id)
                    return ReleaseResult(status="fenced", authority=current)
                return ReleaseResult(status="released", authority=_authority_from_row(row))

    def read(self, run_id: str) -> WriterAuthority | None:
        self._require_ready()
        if not is_safe_opaque_id(run_id):
            raise ValueError("writer authority run_id must be a bounded opaque id")

        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                # Keep the returned row version and observed_at on one linearized boundary.
                # `_read_locked` acquires the row before sampling the database clock.
                return self._read_locked(cursor, run_id)


__all__ = ["PostgresWriterAuthorityStore"]
