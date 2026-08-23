"""PostgreSQL production adapter with an optional, lazy psycopg boundary."""

from .authority import PostgresWriterAuthorityStore
from .config import PostgresConfig
from .migrations import (
    SCHEMA_VERSION,
    InstalledMigration,
    MigrationApplyResult,
    MigrationInfo,
    MigrationPlan,
    MigrationStatus,
    PostgresDoctorReport,
    PostgresMigrationDrift,
    PostgresMigrationError,
    PostgresMigrations,
    PostgresSchemaIncompatible,
    bundled_migrations,
)
from .pool import (
    PostgresDatabase,
    PostgresDatabaseClosed,
    PostgresDependencyMissing,
    PostgresHealth,
    UnsupportedPostgresVersion,
)
from .sink import PostgresBlobCorrupt, PostgresFencedRunSink

__all__ = [
    "PostgresConfig",
    "PostgresDependencyMissing",
    "PostgresDatabaseClosed",
    "UnsupportedPostgresVersion",
    "PostgresHealth",
    "PostgresDatabase",
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
    "PostgresWriterAuthorityStore",
    "PostgresBlobCorrupt",
    "PostgresFencedRunSink",
]
