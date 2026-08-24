"""Explicit ordered PostgreSQL migrations and compatibility inspection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from importlib.resources import files

from monoid_agent_kernel.adapters.postgres.pool import (
    PostgresDatabase,
    PostgresHealth,
)
from monoid_agent_kernel.core.json_ingress import portable_type_name


SCHEMA_VERSION = 6
_METADATA_TABLE = "monoid_schema_migrations"
_MIGRATION_LOCK_NAMESPACE = "monoid-agent-kernel:migrations"
_SCHEMA_TOKEN = "__MONOID_SCHEMA__"


class PostgresMigrationError(RuntimeError):
    """Base error for explicit schema management."""


class PostgresMigrationDrift(PostgresMigrationError):
    """Installed migration identity or content differs from the bundled history."""


class PostgresSchemaIncompatible(PostgresMigrationError):
    """The installed schema is outside this adapter's reader or writer window."""


@dataclass(frozen=True, kw_only=True)
class MigrationInfo:
    migration_id: str
    ordinal: int
    checksum_sha256: str
    reader_floor: int
    writer_floor: int


@dataclass(frozen=True, kw_only=True)
class InstalledMigration(MigrationInfo):
    applied_at: datetime


@dataclass(frozen=True, kw_only=True)
class MigrationStatus:
    schema: str
    schema_exists: bool
    metadata_exists: bool
    installed: tuple[InstalledMigration, ...]
    pending: tuple[MigrationInfo, ...]
    reader_compatible: bool
    writer_compatible: bool

    @property
    def current_version(self) -> int:
        return self.installed[-1].ordinal if self.installed else 0

    @property
    def latest_version(self) -> int:
        return len(self.installed) + len(self.pending)

    @property
    def current(self) -> bool:
        return not self.pending and self.reader_compatible and self.writer_compatible


@dataclass(frozen=True, kw_only=True)
class MigrationPlan:
    schema: str
    installed_version: int
    pending: tuple[MigrationInfo, ...]


@dataclass(frozen=True, kw_only=True)
class MigrationApplyResult:
    applied: tuple[MigrationInfo, ...]
    status: MigrationStatus


@dataclass(frozen=True, kw_only=True)
class PostgresDoctorReport:
    ok: bool
    health: PostgresHealth | None
    migration_status: MigrationStatus | None
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class _MigrationSource:
    info: MigrationInfo
    sql_text: str


@lru_cache(maxsize=1)
def _sources() -> tuple[_MigrationSource, ...]:
    definitions = (
        ("0001_authority", 1, 1, 1),
        ("0002_checkpoint_invocation", 2, 1, 1),
        ("0003_event_terminal_evidence_outbox", 3, 1, 1),
        ("0004_object_association_gc", 4, 1, 1),
        ("0005_activation_admission_dispatch", 5, 1, 1),
        ("0006_durable_stream", 6, 1, 1),
    )
    root = files("monoid_agent_kernel.adapters.postgres").joinpath("sql")
    loaded: list[_MigrationSource] = []
    for migration_id, ordinal, reader_floor, writer_floor in definitions:
        raw = root.joinpath(f"{migration_id}.sql").read_bytes()
        loaded.append(
            _MigrationSource(
                info=MigrationInfo(
                    migration_id=migration_id,
                    ordinal=ordinal,
                    checksum_sha256=hashlib.sha256(raw).hexdigest(),
                    reader_floor=reader_floor,
                    writer_floor=writer_floor,
                ),
                sql_text=raw.decode("utf-8"),
            )
        )
    return tuple(loaded)


def bundled_migrations() -> tuple[MigrationInfo, ...]:
    """Return immutable public metadata for the exact SQL resources in this wheel."""

    return tuple(source.info for source in _sources())


def _render_migration(sql_text: str, schema: str) -> object:
    """Render only the reserved schema token as a quoted PostgreSQL identifier."""

    from psycopg import sql

    parts = sql_text.split(_SCHEMA_TOKEN)
    if len(parts) == 1:
        raise PostgresMigrationError("PostgreSQL migration resource is missing its schema token")
    rendered: list[object] = [sql.SQL(parts[0])]
    for part in parts[1:]:
        rendered.extend((sql.Identifier(schema), sql.SQL(part)))
    return sql.Composed(rendered)


class PostgresMigrations:
    """Read-only schema inspection plus explicitly invoked migration application."""

    def __init__(self, database: PostgresDatabase) -> None:
        if not isinstance(database, PostgresDatabase):
            raise TypeError("PostgresMigrations database must be PostgresDatabase")
        self.database = database

    @staticmethod
    def _schema_exists(cursor: object, schema: str) -> bool:
        cursor.execute(  # type: ignore[attr-defined]
            "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname = %s)",
            (schema,),
        )
        row = cursor.fetchone()  # type: ignore[attr-defined]
        return bool(row[0])

    @staticmethod
    def _table_exists(cursor: object, schema: str, table: str) -> bool:
        cursor.execute(  # type: ignore[attr-defined]
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_catalog.pg_class AS relation
                JOIN pg_catalog.pg_namespace AS namespace
                  ON namespace.oid = relation.relnamespace
                WHERE namespace.nspname = %s
                  AND relation.relname = %s
                  AND relation.relkind IN ('r', 'p')
            )
            """,
            (schema, table),
        )
        row = cursor.fetchone()  # type: ignore[attr-defined]
        return bool(row[0])

    def _inspect(self, connection: object) -> MigrationStatus:
        from psycopg import sql

        schema = self.database.config.schema
        sources = _sources()
        with self.database.cursor(connection) as cursor:
            schema_exists = self._schema_exists(cursor, schema)
            if not schema_exists:
                return MigrationStatus(
                    schema=schema,
                    schema_exists=False,
                    metadata_exists=False,
                    installed=(),
                    pending=tuple(source.info for source in sources),
                    reader_compatible=True,
                    writer_compatible=True,
                )
            metadata_exists = self._table_exists(cursor, schema, _METADATA_TABLE)
            if not metadata_exists:
                return MigrationStatus(
                    schema=schema,
                    schema_exists=True,
                    metadata_exists=False,
                    installed=(),
                    pending=tuple(source.info for source in sources),
                    reader_compatible=True,
                    writer_compatible=True,
                )
            cursor.execute(
                sql.SQL(
                    "SELECT migration_id, ordinal, checksum_sha256, reader_floor, "
                    "writer_floor, applied_at FROM {}.{} ORDER BY ordinal"
                ).format(sql.Identifier(schema), sql.Identifier(_METADATA_TABLE))
            )
            rows = tuple(cursor.fetchall())

        installed = tuple(
            InstalledMigration(
                migration_id=str(row[0]),
                ordinal=int(row[1]),
                checksum_sha256=str(row[2]),
                reader_floor=int(row[3]),
                writer_floor=int(row[4]),
                applied_at=row[5],
            )
            for row in rows
        )
        known_count = min(len(installed), len(sources))
        for index, item in enumerate(installed[:known_count]):
            expected = sources[index].info
            actual_identity = (item.migration_id, item.ordinal)
            expected_identity = (expected.migration_id, expected.ordinal)
            if actual_identity != expected_identity:
                raise PostgresMigrationDrift(
                    "installed PostgreSQL migrations are not the bundled ordered prefix"
                )
            if item.checksum_sha256 != expected.checksum_sha256:
                raise PostgresMigrationDrift(
                    f"installed PostgreSQL migration checksum drift: {item.migration_id}"
                )
            if (item.reader_floor, item.writer_floor) != (
                expected.reader_floor,
                expected.writer_floor,
            ):
                raise PostgresMigrationDrift(
                    f"installed PostgreSQL migration compatibility drift: {item.migration_id}"
                )
        for index, item in enumerate(installed[known_count:], start=known_count + 1):
            if item.ordinal != index:
                raise PostgresMigrationDrift(
                    "installed PostgreSQL migrations are not a contiguous ordered history"
                )

        reader_floor = max((item.reader_floor for item in installed), default=1)
        writer_floor = max((item.writer_floor for item in installed), default=1)
        return MigrationStatus(
            schema=schema,
            schema_exists=True,
            metadata_exists=True,
            installed=installed,
            pending=tuple(source.info for source in sources[len(installed) :]),
            reader_compatible=reader_floor <= SCHEMA_VERSION,
            writer_compatible=writer_floor <= SCHEMA_VERSION,
        )

    def status(self) -> MigrationStatus:
        with self.database.transaction() as connection:
            return self._inspect(connection)

    def plan(self) -> MigrationPlan:
        status = self.status()
        return MigrationPlan(
            schema=status.schema,
            installed_version=status.current_version,
            pending=status.pending,
        )

    def require_reader_compatible(self) -> MigrationStatus:
        status = self.status()
        if not status.reader_compatible or status.pending:
            raise PostgresSchemaIncompatible(
                "PostgreSQL schema is not current for this adapter reader"
            )
        return status

    def require_writer_compatible(self) -> MigrationStatus:
        status = self.status()
        if not status.writer_compatible or status.pending:
            raise PostgresSchemaIncompatible(
                "PostgreSQL schema is not current for adapter mutations"
            )
        return status

    def apply(self) -> MigrationApplyResult:
        from psycopg import sql

        schema = self.database.config.schema
        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                cursor.execute(
                    "SELECT pg_catalog.pg_advisory_xact_lock("
                    "pg_catalog.hashtextextended(%s, 0))",
                    (f"{_MIGRATION_LOCK_NAMESPACE}:{schema}",),
                )
            before = self._inspect(connection)
            pending_ids = {item.migration_id for item in before.pending}
            pending_sources = tuple(
                source for source in _sources() if source.info.migration_id in pending_ids
            )
            if pending_sources:
                with self.database.cursor(connection) as cursor:
                    cursor.execute(
                        sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
                    )
                    for source in pending_sources:
                        cursor.execute(_render_migration(source.sql_text, schema))
                        cursor.execute(
                            sql.SQL(
                                "INSERT INTO {}.{} "
                                "(migration_id, ordinal, checksum_sha256, reader_floor, "
                                "writer_floor, applied_at) "
                                "VALUES (%s, %s, %s, %s, %s, "
                                "pg_catalog.clock_timestamp())"
                            ).format(
                                sql.Identifier(schema),
                                sql.Identifier(_METADATA_TABLE),
                            ),
                            (
                                source.info.migration_id,
                                source.info.ordinal,
                                source.info.checksum_sha256,
                                source.info.reader_floor,
                                source.info.writer_floor,
                            ),
                        )
            after = self._inspect(connection)
            if not after.current:
                # Preserve migration atomicity: a failed postcondition must roll back the same
                # transaction instead of committing a partially applied schema.
                raise PostgresSchemaIncompatible(
                    "PostgreSQL migrations completed without a current compatible schema"
                )
        return MigrationApplyResult(
            applied=tuple(source.info for source in pending_sources),
            status=after,
        )

    def doctor(self) -> PostgresDoctorReport:
        errors: list[str] = []
        health: PostgresHealth | None = None
        status: MigrationStatus | None = None
        try:
            health = self.database.health()
        except Exception as exc:  # noqa: BLE001 - doctor converts diagnostics into a report
            errors.append(f"health: {portable_type_name(exc)}")
        if health is not None:
            try:
                status = self.status()
                if status.pending:
                    errors.append(
                        "migration: pending "
                        + ", ".join(item.migration_id for item in status.pending)
                    )
                if not status.reader_compatible:
                    errors.append("migration: adapter reader is incompatible")
                if not status.writer_compatible:
                    errors.append("migration: adapter writer is incompatible")
            except Exception as exc:  # noqa: BLE001 - doctor converts diagnostics into a report
                errors.append(f"migration: {portable_type_name(exc)}")
        return PostgresDoctorReport(
            ok=not errors,
            health=health,
            migration_status=status,
            errors=tuple(errors),
        )


__all__ = [
    "SCHEMA_VERSION",
    "PostgresMigrationError",
    "PostgresMigrationDrift",
    "PostgresSchemaIncompatible",
    "MigrationInfo",
    "InstalledMigration",
    "MigrationStatus",
    "MigrationPlan",
    "MigrationApplyResult",
    "PostgresDoctorReport",
    "bundled_migrations",
    "PostgresMigrations",
]
