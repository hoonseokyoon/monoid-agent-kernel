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


def _validate_ttl(ttl: object) -> None:
    if type(ttl) is not timedelta or ttl <= timedelta(0):
        raise ValueError("writer lease ttl must be a positive timedelta")


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
                "SELECT run_id, owner_id, generation, leased_until, revoked, "
                "clock_timestamp() FROM {} WHERE run_id = %s FOR UPDATE"
            ).format(self._qualified_table()),
            (run_id,),
        )
        row = cursor.fetchone()  # type: ignore[attr-defined]
        return None if row is None else _authority_from_row(row)

    def _insert_first(
        self,
        cursor: object,
        run_id: str,
        owner_id: str,
        ttl: timedelta,
    ) -> WriterLease | None:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "WITH sampled AS (SELECT clock_timestamp() AS db_now) "
                "INSERT INTO {} "
                "(run_id, owner_id, generation, leased_until, revoked, updated_at) "
                "SELECT %s, %s, 1, db_now + %s, false, db_now FROM sampled "
                "ON CONFLICT (run_id) DO NOTHING "
                "RETURNING run_id, owner_id, generation, leased_until, revoked, updated_at"
            ).format(self._qualified_table()),
            (run_id, owner_id, ttl),
        )
        row = cursor.fetchone()  # type: ignore[attr-defined]
        if row is None:
            return None
        authority = _authority_from_row(row)
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
        ttl: timedelta,
    ) -> WriterLease:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "WITH sampled AS (SELECT clock_timestamp() AS db_now) "
                "UPDATE {} AS authority SET "
                "owner_id = %s, generation = authority.generation + 1, "
                "leased_until = sampled.db_now + %s, revoked = false, "
                "updated_at = sampled.db_now "
                "FROM sampled WHERE authority.run_id = %s "
                "RETURNING authority.run_id, authority.owner_id, authority.generation, "
                "authority.leased_until, authority.revoked, sampled.db_now"
            ).format(self._qualified_table()),
            (owner_id, ttl, run_id),
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
        _validate_ttl(ttl)

        with self.database.connection() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    current = self._read_locked(cursor, run_id)
                    if current is None:
                        inserted = self._insert_first(cursor, run_id, owner_id, ttl)
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
                    return self._take_over(cursor, run_id, owner_id, ttl)

    def renew(self, writer_token: WriterToken, ttl: timedelta) -> RenewResult:
        self._require_ready()
        if not isinstance(writer_token, WriterToken):
            raise TypeError("writer renew requires WriterToken")
        _validate_ttl(ttl)
        from psycopg import sql

        with self.database.connection() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    current = self._read_locked(cursor, writer_token.run_id)
                    if (
                        current is None
                        or current.writer_token != writer_token
                        or not current.active
                    ):
                        return RenewResult(status="fenced", authority=current)
                    cursor.execute(
                        sql.SQL(
                            "WITH sampled AS (SELECT clock_timestamp() AS db_now) "
                            "UPDATE {} AS authority SET "
                            "leased_until = sampled.db_now + %s, updated_at = sampled.db_now "
                            "FROM sampled WHERE authority.run_id = %s "
                            "AND authority.owner_id = %s AND authority.generation = %s "
                            "AND NOT authority.revoked AND authority.leased_until > sampled.db_now "
                            "RETURNING authority.run_id, authority.owner_id, "
                            "authority.generation, authority.leased_until, authority.revoked, "
                            "sampled.db_now"
                        ).format(self._qualified_table()),
                        (
                            ttl,
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

        with self.database.connection() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    current = self._read_locked(cursor, writer_token.run_id)
                    if current is None or current.writer_token != writer_token:
                        return ReleaseResult(status="fenced", authority=current)
                    if current.revoked:
                        return ReleaseResult(status="already_released", authority=current)
                    if not current.active:
                        return ReleaseResult(status="fenced", authority=current)
                    cursor.execute(
                        sql.SQL(
                            "WITH sampled AS (SELECT clock_timestamp() AS db_now) "
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
        from psycopg import sql

        with self.database.connection() as connection:
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        sql.SQL(
                            "SELECT run_id, owner_id, generation, leased_until, revoked, "
                            "clock_timestamp() FROM {} WHERE run_id = %s"
                        ).format(self._qualified_table()),
                        (run_id,),
                    )
                    row = cursor.fetchone()
        return None if row is None else _authority_from_row(row)


__all__ = ["PostgresWriterAuthorityStore"]
