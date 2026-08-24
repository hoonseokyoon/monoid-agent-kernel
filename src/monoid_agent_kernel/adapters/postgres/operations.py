"""Read-only aggregate operations snapshot for the PostgreSQL production adapter."""

from __future__ import annotations

from typing import Any

from monoid_agent_kernel.adapters.postgres.migrations import (
    MigrationStatus,
    PostgresMigrations,
    PostgresSchemaIncompatible,
)
from monoid_agent_kernel.adapters.postgres.pool import PostgresDatabase
from monoid_agent_kernel.hosting.operations import OperationalMetric, OperationalSnapshot


def _count(value: object) -> int:
    result = int(value or 0)
    if result < 0:  # pragma: no cover - aggregate counts and constrained columns are non-negative
        raise RuntimeError("PostgreSQL operations query returned a negative aggregate")
    return result


def _seconds(value: object) -> float:
    result = max(0.0, float(value or 0.0))
    return result


def _attributes(**values: str) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(values.items()))


class PostgresOperations:
    """Explicit readiness plus bounded aggregate collection with no run identifiers or payloads."""

    def __init__(self, database: PostgresDatabase) -> None:
        if not isinstance(database, PostgresDatabase):
            raise TypeError("PostgresOperations database must be PostgresDatabase")
        self.database = database
        self._ready = False

    def check_ready(self) -> MigrationStatus:
        self._ready = False
        status = PostgresMigrations(self.database).require_reader_compatible()
        self._ready = True
        return status

    def _require_ready(self) -> None:
        if not self._ready:
            raise PostgresSchemaIncompatible(
                "PostgreSQL operations require a successful check_ready()"
            )

    def _table(self, name: str) -> object:
        from psycopg import sql

        return sql.Identifier(self.database.config.schema, name)

    @staticmethod
    def _one(cursor: Any, query: object, parameters: tuple[object, ...] = ()) -> tuple[Any, ...]:
        cursor.execute(query, parameters)
        row = cursor.fetchone()
        if row is None:  # pragma: no cover - aggregate queries always return one row
            raise RuntimeError("PostgreSQL operations aggregate query returned no row")
        return tuple(row)

    def snapshot(self) -> OperationalSnapshot:
        """Collect one READ ONLY database-clock snapshot of fixed-cardinality aggregates."""

        self._require_ready()
        from psycopg import sql

        with self.database.read_snapshot() as (connection, collected_at):
            try:
                status = PostgresMigrations(self.database)._require_reader_compatible(connection)
                schema_version = status.current_version
                with self.database.cursor(connection) as cursor:
                    authority = self._one(
                        cursor,
                        sql.SQL(
                            "SELECT count(*), "
                            "count(*) FILTER (WHERE NOT revoked AND leased_until > %s), "
                            "count(*) FILTER (WHERE NOT revoked AND leased_until <= %s), "
                            "count(*) FILTER (WHERE revoked), "
                            "extract(epoch FROM (min(leased_until) FILTER "
                            "(WHERE NOT revoked AND leased_until > %s) - %s)) "
                            "FROM {}"
                        ).format(self._table("run_authority")),
                        (collected_at, collected_at, collected_at, collected_at),
                    )
                    activation_outbox = self._one(
                        cursor,
                        sql.SQL(
                            "WITH dispatch_rows AS ("
                            "SELECT dispatch.*, (((dispatch.delivery_state = 'pending' "
                            "AND dispatch.available_at <= %s) OR "
                            "(dispatch.delivery_state = 'leased' "
                            "AND dispatch.leased_until <= %s)) "
                            "AND NOT EXISTS (SELECT 1 FROM {} AS terminal "
                            "WHERE terminal.run_id = dispatch.run_id) "
                            "AND NOT EXISTS (SELECT 1 FROM {} AS prior_dispatch "
                            "JOIN {} AS prior_admission "
                            "ON prior_admission.run_id = prior_dispatch.run_id "
                            "AND prior_admission.command_id = prior_dispatch.command_id "
                            "WHERE prior_admission.run_id = admission.run_id "
                            "AND prior_admission.command_sequence < admission.command_sequence "
                            "AND prior_dispatch.delivery_state <> 'delivered')) AS actionable "
                            "FROM {} AS dispatch JOIN {} AS admission "
                            "ON admission.run_id = dispatch.run_id "
                            "AND admission.command_id = dispatch.command_id) "
                            "SELECT "
                            "count(*) FILTER (WHERE delivery_state = 'pending'), "
                            "count(*) FILTER (WHERE delivery_state = 'leased'), "
                            "count(*) FILTER (WHERE delivery_state = 'delivered'), "
                            "count(*) FILTER (WHERE delivery_state = 'run_terminal'), "
                            "count(*) FILTER (WHERE delivery_state = 'dead_letter'), "
                            "coalesce(max(attempt_count) FILTER (WHERE actionable), 0), "
                            "extract(epoch FROM (%s - min(created_at) "
                            "FILTER (WHERE actionable))) FROM dispatch_rows"
                        ).format(
                            self._table("terminal_record"),
                            self._table("activation_dispatch_outbox"),
                            self._table("activation_admission_record"),
                            self._table("activation_dispatch_outbox"),
                            self._table("activation_admission_record"),
                        ),
                        (collected_at, collected_at, collected_at),
                    )
                    evidence_outbox = self._one(
                        cursor,
                        sql.SQL(
                            "SELECT "
                            "count(*) FILTER (WHERE delivery_state = 'pending' AND "
                            "(lease_owner IS NULL OR leased_until <= %s)), "
                            "count(*) FILTER (WHERE delivery_state = 'pending' "
                            "AND lease_owner IS NOT NULL AND leased_until > %s), "
                            "count(*) FILTER (WHERE delivery_state = 'delivered'), "
                            "count(*) FILTER (WHERE delivery_state = 'dead_letter'), "
                            "coalesce(max(attempt_count) FILTER (WHERE "
                            "delivery_state = 'pending' AND available_at <= %s AND "
                            "(lease_owner IS NULL OR leased_until <= %s)), 0), "
                            "extract(epoch FROM (%s - min(created_at) FILTER "
                            "(WHERE delivery_state = 'pending' AND available_at <= %s AND "
                            "(lease_owner IS NULL OR leased_until <= %s)))) "
                            "FROM {}"
                        ).format(self._table("model_evidence_outbox")),
                        (
                            collected_at,
                            collected_at,
                            collected_at,
                            collected_at,
                            collected_at,
                            collected_at,
                            collected_at,
                        ),
                    )
                    invocations = self._one(
                        cursor,
                        sql.SQL(
                            "SELECT "
                            "count(*) FILTER (WHERE record.dispatch_state = 'reserved'), "
                            "count(*) FILTER (WHERE record.dispatch_state = 'dispatch_started'), "
                            "count(*) FILTER (WHERE record.dispatch_state = 'settled'), "
                            "count(*) FILTER (WHERE record.dispatch_state = 'unknown') "
                            "FROM {} AS head JOIN {} AS record "
                            "ON record.run_id = head.run_id "
                            "AND record.logical_call_id = head.logical_call_id "
                            "AND record.revision = head.revision"
                        ).format(
                            self._table("invocation_head"),
                            self._table("invocation_record"),
                        ),
                    )
                    stream_heads = self._one(
                        cursor,
                        sql.SQL(
                            "SELECT "
                            "count(*) FILTER (WHERE head.state = 'open'), "
                            "count(*) FILTER (WHERE head.state = 'sealed'), "
                            "coalesce(sum(head.cursor_bytes), 0), "
                            "coalesce(sum(head.cursor_bytes) FILTER "
                            "(WHERE head.state = 'open'), 0), "
                            "extract(epoch FROM (%s - min(coalesce(reset.recorded_at, "
                            "head.opened_at)) FILTER (WHERE head.state = 'open'))) "
                            "FROM {} AS head LEFT JOIN {} AS reset "
                            "ON reset.run_id = head.run_id "
                            "AND reset.stream_id = head.stream_id "
                            "AND reset.channel = head.channel "
                            "AND reset.generation = head.generation"
                        ).format(
                            self._table("durable_stream_head"),
                            self._table("durable_stream_reset_receipt"),
                        ),
                        (collected_at,),
                    )
                    stream_chunks = self._one(
                        cursor,
                        sql.SQL(
                            "SELECT count(*), coalesce(sum(end_offset - start_offset), 0) FROM {}"
                        ).format(self._table("durable_stream_chunk")),
                    )
                    objects = self._one(
                        cursor,
                        sql.SQL(
                            "SELECT "
                            "count(*) FILTER (WHERE blob.state = 'available'), "
                            "count(*) FILTER (WHERE blob.state = 'deleted'), "
                            "coalesce(sum(blob.size_bytes) FILTER "
                            "(WHERE blob.state = 'available'), 0), "
                            "(SELECT count(*) FROM {}), "
                            "count(*) FILTER (WHERE blob.state = 'available' AND NOT EXISTS "
                            "(SELECT 1 FROM {} AS association "
                            "WHERE association.sha256 = blob.sha256)) "
                            "FROM {} AS blob"
                        ).format(
                            self._table("run_object_blob"),
                            self._table("run_object_blob"),
                            self._table("object_blob"),
                        ),
                    )
                    gc_receipts = self._one(
                        cursor,
                        sql.SQL(
                            "SELECT "
                            "count(*) FILTER (WHERE status IN ('deleted', 'already_missing')), "
                            "count(*) FILTER (WHERE status IN "
                            "('skipped_associated', 'skipped_generation')), "
                            "count(*) FILTER (WHERE status = 'precondition_failed') "
                            "FROM {}"
                        ).format(self._table("object_gc_receipt")),
                    )
            except Exception:
                self._ready = False
                raise

        metrics = [
            OperationalMetric(name="monoid.postgres.schema.version", value=schema_version),
            *(
                OperationalMetric(
                    name="monoid.postgres.authority.count",
                    value=_count(value),
                    attributes=_attributes(state=state),
                )
                for state, value in zip(
                    ("total", "active", "expired", "revoked"),
                    authority[:4],
                    strict=True,
                )
            ),
            OperationalMetric(
                name="monoid.postgres.authority.seconds_to_next_expiry",
                value=_seconds(authority[4]),
                unit="s",
            ),
        ]
        for queue, row, states in (
            (
                "activation",
                activation_outbox,
                ("pending", "leased", "delivered", "run_terminal", "dead_letter"),
            ),
            (
                "model_evidence",
                evidence_outbox,
                ("pending", "leased", "delivered", "dead_letter"),
            ),
        ):
            for state, value in zip(states, row[: len(states)], strict=True):
                metrics.append(
                    OperationalMetric(
                        name="monoid.postgres.outbox.count",
                        value=_count(value),
                        attributes=_attributes(queue=queue, state=state),
                    )
                )
            metrics.extend(
                (
                    OperationalMetric(
                        name="monoid.postgres.outbox.max_attempts",
                        value=_count(row[len(states)]),
                        attributes=_attributes(queue=queue),
                    ),
                    OperationalMetric(
                        name="monoid.postgres.outbox.oldest_age",
                        value=_seconds(row[len(states) + 1]),
                        unit="s",
                        attributes=_attributes(queue=queue),
                    ),
                )
            )
        for state, value in zip(
            ("reserved", "dispatch_started", "settled", "unknown"),
            invocations,
            strict=True,
        ):
            metrics.append(
                OperationalMetric(
                    name="monoid.postgres.invocation.count",
                    value=_count(value),
                    attributes=_attributes(state=state),
                )
            )
        metrics.extend(
            (
                OperationalMetric(
                    name="monoid.postgres.stream.head.count",
                    value=_count(stream_heads[0]),
                    attributes=_attributes(state="open"),
                ),
                OperationalMetric(
                    name="monoid.postgres.stream.head.count",
                    value=_count(stream_heads[1]),
                    attributes=_attributes(state="sealed"),
                ),
                OperationalMetric(
                    name="monoid.postgres.stream.current_bytes",
                    value=_count(stream_heads[2]),
                    unit="By",
                    attributes=_attributes(state="all"),
                ),
                OperationalMetric(
                    name="monoid.postgres.stream.current_bytes",
                    value=_count(stream_heads[3]),
                    unit="By",
                    attributes=_attributes(state="open"),
                ),
                OperationalMetric(
                    name="monoid.postgres.stream.oldest_open_age",
                    value=_seconds(stream_heads[4]),
                    unit="s",
                ),
                OperationalMetric(
                    name="monoid.postgres.stream.chunk.count",
                    value=_count(stream_chunks[0]),
                ),
                OperationalMetric(
                    name="monoid.postgres.stream.chunk.bytes",
                    value=_count(stream_chunks[1]),
                    unit="By",
                ),
            )
        )
        for state, value in zip(("available", "deleted"), objects[:2], strict=True):
            metrics.append(
                OperationalMetric(
                    name="monoid.postgres.object.count",
                    value=_count(value),
                    attributes=_attributes(state=state),
                )
            )
        metrics.extend(
            (
                OperationalMetric(
                    name="monoid.postgres.object.bytes",
                    value=_count(objects[2]),
                    unit="By",
                    attributes=_attributes(state="available"),
                ),
                OperationalMetric(
                    name="monoid.postgres.object.association.count",
                    value=_count(objects[3]),
                ),
                OperationalMetric(
                    name="monoid.postgres.object.orphan_metadata.count",
                    value=_count(objects[4]),
                ),
            )
        )
        for state, value in zip(
            ("deleted", "skipped", "precondition_failed"),
            gc_receipts,
            strict=True,
        ):
            metrics.append(
                OperationalMetric(
                    name="monoid.postgres.object_gc.receipt.count",
                    value=_count(value),
                    attributes=_attributes(state=state),
                )
            )
        return OperationalSnapshot(
            source="postgres",
            collected_at=collected_at,
            metrics=tuple(sorted(metrics, key=lambda metric: (metric.name, metric.attributes))),
        )


__all__ = ["PostgresOperations"]
