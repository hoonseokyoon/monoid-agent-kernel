"""PostgreSQL-fenced authoritative run journal."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from typing import Any, Callable, get_args

from monoid_agent_kernel.adapters.postgres.authority import PostgresWriterAuthorityStore
from monoid_agent_kernel.adapters.postgres.migrations import (
    MigrationStatus,
    PostgresMigrations,
    PostgresSchemaIncompatible,
)
from monoid_agent_kernel.adapters.postgres.pool import PostgresDatabase
from monoid_agent_kernel.core._storage_capabilities import StorageCapabilities
from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.core.checkpoint import (
    CHECKPOINT_CODEC,
    CheckpointRecord,
    RunCheckpoint,
    bind_checkpoint_record_result,
    checkpoint_blob_references,
    checkpoint_payload_for_write,
    decode_checkpoint,
)
from monoid_agent_kernel.core.durable_codec import DurableLoadResult
from monoid_agent_kernel.core.events import (
    EVENT_SCHEMA_VERSION,
    AgentEvent,
    AgentEventLevel,
    AgentEventType,
)
from monoid_agent_kernel.core.json_ingress import (
    is_portable_json_integer,
    normalize_json_ingress,
)
from monoid_agent_kernel.core.model_invocation import (
    MODEL_INVOCATION_CODEC,
    DurableModelInvocation,
    decode_model_invocation,
)
from monoid_agent_kernel.core.model_io import is_recorded_digest
from monoid_agent_kernel.core.outcome import TerminalOutcome
from monoid_agent_kernel.core.safe_evidence import is_safe_opaque_id, is_safe_utc_timestamp
from monoid_agent_kernel.hosting import CommitResult, ModelInvocationRecord, WriterToken


_CAPABILITIES = StorageCapabilities(
    concurrent_writers=True,
    compare_and_set=True,
    lease_fencing=True,
    durable_checkpoints=True,
    durable_events=True,
    durable_invocations=True,
    terminal_first_writer_wins=True,
    transactional_outbox=True,
)
_STABLE_INVOCATION_FIELDS = (
    "logical_call_id",
    "idempotency_key",
    "request_digest",
    "digest_generation",
    "evidence_policy",
)
_POSTGRES_BIGINT_MAX = (1 << 63) - 1
_CHECKPOINT_TYPED_FIELDS = ("run_id", "seq", "schema_version")
_INVOCATION_TYPED_FIELDS = (
    "run_id",
    "logical_call_id",
    "revision",
    "schema_version",
    "dispatch_id",
    "dispatch_attempt",
    "dispatch_state",
    "idempotency_key",
    "request_digest",
    "digest_generation",
    "evidence_policy",
    "result_ref",
    "failure_code",
)
_EVENT_TYPED_FIELDS = (
    "run_id",
    "seq",
    "event_id",
    "schema_version",
    "type",
    "level",
)
_TERMINAL_TYPED_FIELDS = (
    "run_id",
    "schema_version",
    "kind",
    "retry_eligibility",
    "checkpoint_seq",
    "error_code",
)
_EVIDENCE_TYPED_FIELDS = (
    "run_id",
    "logical_call_id",
    "revision",
    "schema_version",
    "evidence_policy",
)


class PostgresBlobCorrupt(RuntimeError):
    """A run-associated bytea blob no longer matches its immutable content address."""


class _BlobRaceConflict(Exception):
    """Roll back blobs inserted before a concurrent global digest collision was observed."""


class _EvidenceStageConflict(Exception):
    """Roll back an invocation whose required outbox entry could not be staged."""


def _blob_projection(blobs: Mapping[str, bytes]) -> dict[str, str]:
    return {key: hashlib.sha256(value).hexdigest() for key, value in sorted(blobs.items())}


def _record_digest(payload: dict[str, Any], blobs: Mapping[str, bytes]) -> str:
    return canonical_sha256({"record": payload, "blobs": _blob_projection(blobs)})


def _record_only_digest(payload: dict[str, Any]) -> str:
    return _record_digest(payload, {})


def _stored_projection_is_valid(value: object) -> bool:
    return isinstance(value, dict) and all(
        type(key) is str and is_recorded_digest(key) and type(digest) is str and digest == key
        for key, digest in value.items()
    )


def _checked_stored_digest(
    payload: object,
    submitted_blobs: object,
    content_digest: object,
) -> str | None:
    if (
        not isinstance(payload, dict)
        or not _stored_projection_is_valid(submitted_blobs)
        or type(content_digest) is not str
        or not is_recorded_digest(content_digest)
    ):
        return None
    calculated = canonical_sha256({"record": payload, "blobs": submitted_blobs})
    return content_digest if calculated == content_digest else None


def _payload_matches_typed_values(
    payload: object,
    fields: tuple[str, ...],
    typed_values: tuple[object, ...],
) -> bool:
    return (
        isinstance(payload, dict) and tuple(payload.get(field) for field in fields) == typed_values
    )


def _is_ambiguous_database_error(exc: Exception) -> bool:
    try:
        from psycopg import InterfaceError, OperationalError
    except ImportError:  # pragma: no cover - a database operation already requires psycopg
        return False
    return isinstance(exc, (InterfaceError, OperationalError))


class PostgresFencedRunSink:
    """Run-fenced durable journal backed by PostgreSQL and bounded bytea blobs.

    Checkpoints, invocations, events, terminal outcomes, required evidence, and transactional
    evidence outbox entries share one run-bound authority and canonical readback model.
    """

    def __init__(self, database: PostgresDatabase) -> None:
        if not isinstance(database, PostgresDatabase):
            raise TypeError("PostgresFencedRunSink database must be PostgresDatabase")
        self.database = database
        self._authority = PostgresWriterAuthorityStore(database)
        self._ready = False

    @property
    def capabilities(self) -> StorageCapabilities:
        return _CAPABILITIES

    def check_ready(self) -> MigrationStatus:
        """Fail before reads or mutations unless all required migrations are installed."""

        self._ready = False
        status = PostgresMigrations(self.database).require_writer_compatible()
        if not status.reader_compatible:
            raise PostgresSchemaIncompatible(
                "PostgreSQL fenced run sink requires reader and writer compatibility"
            )
        self._ready = True
        return status

    def _require_ready(self) -> None:
        if not self._ready:
            raise PostgresSchemaIncompatible(
                "PostgreSQL fenced run sink requires a successful check_ready()"
            )

    def _table(self, name: str) -> object:
        from psycopg import sql

        return sql.Identifier(self.database.config.schema, name)

    def _current_writer_locked(self, cursor: object, writer_token: WriterToken) -> bool:
        current = self._authority._read_locked(cursor, writer_token.run_id)
        return current is not None and current.writer_token == writer_token and current.active

    def _validated_blobs(self, blobs: object) -> dict[str, bytes] | None:
        if not isinstance(blobs, Mapping):
            return None
        try:
            copied = dict(blobs.items())
        except Exception:
            return None
        for key, value in copied.items():
            if (
                type(key) is not str
                or not is_recorded_digest(key)
                or type(value) is not bytes
                or len(value) > self.database.config.max_bytea_blob_bytes
                or hashlib.sha256(value).hexdigest() != key
            ):
                return None
        return copied

    def _submitted_blobs_preserve_backing(
        self,
        cursor: object,
        blobs: Mapping[str, bytes],
    ) -> bool:
        from psycopg import sql

        for sha256 in sorted(blobs):
            value = blobs[sha256]
            cursor.execute(  # type: ignore[attr-defined]
                sql.SQL("SELECT size_bytes, content FROM {} WHERE sha256 = %s").format(
                    self._table("bytea_blob")
                ),
                (sha256,),
            )
            row = cursor.fetchone()  # type: ignore[attr-defined]
            if row is not None and (int(row[0]) != len(value) or bytes(row[1]) != value):
                return False
        return True

    def _associated_blob_is_valid(self, cursor: object, run_id: str, sha256: str) -> bool:
        if not is_recorded_digest(sha256):
            return False
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "SELECT blob.size_bytes, blob.content FROM {} AS association "
                "JOIN {} AS blob ON blob.sha256 = association.sha256 "
                "WHERE association.run_id = %s AND association.sha256 = %s"
            ).format(self._table("run_blob"), self._table("bytea_blob")),
            (run_id, sha256),
        )
        row = cursor.fetchone()  # type: ignore[attr-defined]
        if row is None:
            return False
        value = bytes(row[1])
        return int(row[0]) == len(value) and hashlib.sha256(value).hexdigest() == sha256

    def _references_resolve(
        self,
        cursor: object,
        run_id: str,
        references: set[str],
        blobs: Mapping[str, bytes],
    ) -> bool:
        return all(
            sha256 in blobs or self._associated_blob_is_valid(cursor, run_id, sha256)
            for sha256 in references
        )

    def _persist_blobs(
        self,
        cursor: object,
        run_id: str,
        blobs: Mapping[str, bytes],
    ) -> None:
        from psycopg import sql

        for sha256 in sorted(blobs):
            value = blobs[sha256]
            cursor.execute(  # type: ignore[attr-defined]
                sql.SQL(
                    "INSERT INTO {} (sha256, size_bytes, content) VALUES (%s, %s, %s) "
                    "ON CONFLICT (sha256) DO NOTHING RETURNING sha256"
                ).format(self._table("bytea_blob")),
                (sha256, len(value), value),
            )
            inserted = cursor.fetchone()  # type: ignore[attr-defined]
            if inserted is None:
                cursor.execute(  # type: ignore[attr-defined]
                    sql.SQL("SELECT size_bytes, content FROM {} WHERE sha256 = %s").format(
                        self._table("bytea_blob")
                    ),
                    (sha256,),
                )
                existing = cursor.fetchone()  # type: ignore[attr-defined]
                if existing is None or (
                    int(existing[0]) != len(value) or bytes(existing[1]) != value
                ):
                    raise _BlobRaceConflict
            cursor.execute(  # type: ignore[attr-defined]
                sql.SQL(
                    "INSERT INTO {} (run_id, sha256) VALUES (%s, %s) "
                    "ON CONFLICT (run_id, sha256) DO NOTHING"
                ).format(self._table("run_blob")),
                (run_id, sha256),
            )

    @staticmethod
    def _stored_result(
        row: object,
        content_digest: str,
        *,
        sequence: int | None,
        typed_fields: tuple[str, ...],
        expected_typed_values: tuple[object, ...],
    ) -> CommitResult | None:
        if row is None:
            return None
        raw_winner_digest = row[0]  # type: ignore[index]
        winner_digest = (
            raw_winner_digest
            if type(raw_winner_digest) is str and is_recorded_digest(raw_winner_digest)
            else ""
        )
        stored_payload = row[1]  # type: ignore[index]
        stored_typed_values = tuple(row[3:])  # type: ignore[index]
        if (
            _checked_stored_digest(stored_payload, row[2], row[0]) is None  # type: ignore[index]
            or stored_typed_values != expected_typed_values
            or not _payload_matches_typed_values(
                stored_payload,
                typed_fields,
                stored_typed_values,
            )
        ):
            return CommitResult(
                status="conflict",
                sequence=sequence,
                content_digest=content_digest,
                winner_digest=winner_digest,
            )
        if winner_digest == content_digest:
            return CommitResult(
                status="already_committed",
                sequence=sequence,
                content_digest=content_digest,
            )
        return CommitResult(
            status="conflict",
            sequence=sequence,
            content_digest=content_digest,
            winner_digest=winner_digest,
        )

    def _checkpoint_coordinate(
        self,
        cursor: object,
        run_id: str,
        sequence: int,
        schema_version: str,
        content_digest: str,
    ) -> CommitResult | None:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "SELECT content_digest, payload, submitted_blobs, run_id, sequence, "
                "schema_version FROM {} WHERE run_id = %s AND sequence = %s"
            ).format(self._table("checkpoint_record")),
            (run_id, sequence),
        )
        return self._stored_result(
            cursor.fetchone(),  # type: ignore[attr-defined]
            content_digest,
            sequence=sequence,
            typed_fields=_CHECKPOINT_TYPED_FIELDS,
            expected_typed_values=(run_id, sequence, schema_version),
        )

    def _commit_checkpoint_locked(
        self,
        cursor: object,
        checkpoint: object,
        blobs: object,
    ) -> tuple[CommitResult, str]:
        if not isinstance(checkpoint, RunCheckpoint):
            return CommitResult(status="conflict"), ""
        if type(checkpoint.seq) is not int or checkpoint.seq < 0:
            return CommitResult(status="conflict"), ""
        if checkpoint.seq > _POSTGRES_BIGINT_MAX:
            return CommitResult(status="conflict", sequence=checkpoint.seq), ""
        submitted = self._validated_blobs(blobs)
        if submitted is None:
            return CommitResult(status="conflict", sequence=checkpoint.seq), ""
        try:
            payload = checkpoint_payload_for_write(checkpoint)
            references = checkpoint_blob_references(checkpoint)
        except (TypeError, ValueError, OverflowError, RecursionError):
            return CommitResult(status="conflict", sequence=checkpoint.seq), ""
        if not self._submitted_blobs_preserve_backing(cursor, submitted):
            return CommitResult(status="conflict", sequence=checkpoint.seq), ""
        if not self._references_resolve(cursor, checkpoint.run_id, references, submitted):
            return CommitResult(status="conflict", sequence=checkpoint.seq), ""
        content_digest = _record_digest(payload, submitted)
        stored = self._checkpoint_coordinate(
            cursor,
            checkpoint.run_id,
            checkpoint.seq,
            str(payload["schema_version"]),
            content_digest,
        )
        if stored is not None:
            return stored, content_digest

        self._persist_blobs(cursor, checkpoint.run_id, submitted)
        from psycopg import sql
        from psycopg.types.json import Json, Jsonb

        projection = _blob_projection(submitted)
        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "INSERT INTO {} (run_id, sequence, schema_version, content_digest, payload, "
                "submitted_blobs) VALUES (%s, %s, %s, %s, %s, %s)"
            ).format(self._table("checkpoint_record")),
            (
                checkpoint.run_id,
                checkpoint.seq,
                payload["schema_version"],
                content_digest,
                Json(payload),
                Jsonb(projection),
            ),
        )
        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "INSERT INTO {} AS head (run_id, sequence) VALUES (%s, %s) "
                "ON CONFLICT (run_id) DO UPDATE SET sequence = EXCLUDED.sequence, "
                "updated_at = pg_catalog.clock_timestamp() "
                "WHERE EXCLUDED.sequence > head.sequence"
            ).format(self._table("checkpoint_head")),
            (checkpoint.run_id, checkpoint.seq),
        )
        return (
            CommitResult(
                status="committed",
                sequence=checkpoint.seq,
                content_digest=content_digest,
            ),
            content_digest,
        )

    def _reconcile_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        writer_token: WriterToken,
        content_digest: str,
    ) -> CommitResult | None:
        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                if not self._current_writer_locked(cursor, writer_token):
                    return CommitResult(status="fenced")
                return self._checkpoint_coordinate(
                    cursor,
                    checkpoint.run_id,
                    checkpoint.seq,
                    str(checkpoint_payload_for_write(checkpoint)["schema_version"]),
                    content_digest,
                )

    def commit_checkpoint(
        self,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        self._require_ready()
        if not isinstance(writer_token, WriterToken):
            raise TypeError("PostgreSQL checkpoint commit requires WriterToken")
        if getattr(checkpoint, "run_id", None) != writer_token.run_id:
            return CommitResult(status="fenced")
        result: CommitResult | None = None
        content_digest = ""
        try:
            with self.database.transaction() as connection:
                with self.database.cursor(connection) as cursor:
                    if not self._current_writer_locked(cursor, writer_token):
                        return CommitResult(status="fenced")
                    result, content_digest = self._commit_checkpoint_locked(
                        cursor,
                        checkpoint,
                        blobs,
                    )
            if result is None:  # pragma: no cover - transaction body always assigns a result
                raise RuntimeError("PostgreSQL checkpoint transaction returned no result")
            return result
        except _BlobRaceConflict:
            return CommitResult(
                status="conflict",
                sequence=getattr(checkpoint, "seq", None),
                content_digest=content_digest,
            )
        except Exception as exc:
            if not content_digest or not _is_ambiguous_database_error(exc):
                raise
            try:
                reconciled = self._reconcile_checkpoint(
                    checkpoint,
                    writer_token,
                    content_digest,
                )
            except Exception:
                raise exc
            if reconciled is None:
                raise
            return reconciled

    @staticmethod
    def _event_payload(event: object) -> dict[str, Any] | None:
        if not isinstance(event, AgentEvent):
            return None
        if (
            event.schema_version != EVENT_SCHEMA_VERSION
            or not is_safe_opaque_id(event.event_id)
            or not is_safe_opaque_id(event.run_id)
            or not is_portable_json_integer(event.seq)
            or not 1 <= event.seq <= _POSTGRES_BIGINT_MAX
            or not is_safe_utc_timestamp(event.timestamp)
            or type(event.type) is not str
            or event.type not in get_args(AgentEventType)
            or type(event.level) is not str
            or event.level not in get_args(AgentEventLevel)
            or (event.turn_id is not None and not is_safe_opaque_id(event.turn_id))
            or (event.parent_id is not None and not is_safe_opaque_id(event.parent_id))
            or not isinstance(event.data, dict)
        ):
            return None
        try:
            payload = normalize_json_ingress(
                event.to_json(),
                substitute_nonfinite=False,
            )
            digest = canonical_sha256(payload)
        except (TypeError, ValueError, OverflowError, RecursionError):
            return None
        if not isinstance(payload, dict) or not is_recorded_digest(digest):
            return None
        return payload

    def _event_coordinate(
        self,
        cursor: object,
        event: AgentEvent,
        payload: dict[str, Any],
        content_digest: str,
    ) -> CommitResult | None:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "SELECT content_digest, payload, '{{}}'::jsonb, run_id, sequence, event_id, "
                "schema_version, event_type, event_level, event_timestamp "
                "FROM {} WHERE run_id = %s AND sequence = %s"
            ).format(self._table("event_record")),
            (event.run_id, event.seq),
        )
        row = cursor.fetchone()  # type: ignore[attr-defined]
        if row is None:
            return None
        try:
            timestamp_matches = row[9] == datetime.fromisoformat(
                payload["timestamp"][:-1] + "+00:00"
            )
        except (TypeError, ValueError):
            timestamp_matches = False
        if not timestamp_matches:
            return CommitResult(
                status="conflict",
                sequence=event.seq,
                content_digest=content_digest,
                winner_digest=(
                    row[0] if type(row[0]) is str and is_recorded_digest(row[0]) else ""
                ),
            )
        return self._stored_result(
            row[:9],
            content_digest,
            sequence=event.seq,
            typed_fields=_EVENT_TYPED_FIELDS,
            expected_typed_values=tuple(payload[field] for field in _EVENT_TYPED_FIELDS),
        )

    def _event_id_winner(
        self,
        cursor: object,
        event: AgentEvent,
        content_digest: str,
    ) -> CommitResult | None:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL("SELECT content_digest FROM {} WHERE run_id = %s AND event_id = %s").format(
                self._table("event_record")
            ),
            (event.run_id, event.event_id),
        )
        row = cursor.fetchone()  # type: ignore[attr-defined]
        if row is None:
            return None
        winner_digest = row[0] if type(row[0]) is str and is_recorded_digest(row[0]) else ""
        return CommitResult(
            status="conflict",
            sequence=event.seq,
            content_digest=content_digest,
            winner_digest=winner_digest,
        )

    def _commit_event_locked(
        self,
        cursor: object,
        event: object,
    ) -> tuple[CommitResult, str, dict[str, Any] | None]:
        sequence = getattr(event, "seq", None)
        evidence_sequence = (
            sequence if is_portable_json_integer(sequence) and sequence >= 0 else None
        )
        payload = self._event_payload(event)
        if payload is None or not isinstance(event, AgentEvent):
            return CommitResult(status="conflict", sequence=evidence_sequence), "", None
        content_digest = _record_only_digest(payload)
        stored = self._event_coordinate(cursor, event, payload, content_digest)
        if stored is not None:
            return stored, content_digest, payload
        duplicate_id = self._event_id_winner(cursor, event, content_digest)
        if duplicate_id is not None:
            return duplicate_id, content_digest, payload

        from psycopg import sql
        from psycopg.types.json import Json

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "INSERT INTO {} (run_id, sequence, event_id, schema_version, event_timestamp, "
                "event_type, event_level, content_digest, payload) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
            ).format(self._table("event_record")),
            (
                event.run_id,
                event.seq,
                event.event_id,
                payload["schema_version"],
                event.timestamp,
                event.type,
                event.level,
                content_digest,
                Json(payload),
            ),
        )
        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "INSERT INTO {} AS head (run_id, sequence) VALUES (%s, %s) "
                "ON CONFLICT (run_id) DO UPDATE SET sequence = EXCLUDED.sequence, "
                "updated_at = pg_catalog.clock_timestamp() "
                "WHERE EXCLUDED.sequence > head.sequence"
            ).format(self._table("event_head")),
            (event.run_id, event.seq),
        )
        return (
            CommitResult(
                status="committed",
                sequence=event.seq,
                content_digest=content_digest,
            ),
            content_digest,
            payload,
        )

    def _reconcile_event(
        self,
        event: AgentEvent,
        writer_token: WriterToken,
        payload: dict[str, Any],
        content_digest: str,
    ) -> CommitResult | None:
        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                if not self._current_writer_locked(cursor, writer_token):
                    return CommitResult(status="fenced")
                return self._event_coordinate(cursor, event, payload, content_digest)

    def append_event(
        self,
        event: AgentEvent,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        self._require_ready()
        if not isinstance(writer_token, WriterToken):
            raise TypeError("PostgreSQL event append requires WriterToken")
        if getattr(event, "run_id", None) != writer_token.run_id:
            return CommitResult(status="fenced")
        result: CommitResult | None = None
        content_digest = ""
        payload: dict[str, Any] | None = None
        try:
            with self.database.transaction() as connection:
                with self.database.cursor(connection) as cursor:
                    if not self._current_writer_locked(cursor, writer_token):
                        return CommitResult(status="fenced")
                    result, content_digest, payload = self._commit_event_locked(cursor, event)
            if result is None:  # pragma: no cover - transaction body always assigns a result
                raise RuntimeError("PostgreSQL event transaction returned no result")
            return result
        except Exception as exc:
            if not content_digest or payload is None or not _is_ambiguous_database_error(exc):
                raise
            try:
                reconciled = self._reconcile_event(
                    event,
                    writer_token,
                    payload,
                    content_digest,
                )
            except Exception:
                raise exc
            if reconciled is None:
                raise
            return reconciled

    def _terminal_coordinate(
        self,
        cursor: object,
        outcome: TerminalOutcome,
        payload: dict[str, Any],
        content_digest: str,
    ) -> CommitResult | None:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "SELECT content_digest, payload, '{{}}'::jsonb, run_id, schema_version, "
                "outcome_kind, retry_eligibility, checkpoint_sequence, error_code "
                "FROM {} WHERE run_id = %s"
            ).format(self._table("terminal_record")),
            (outcome.run_id,),
        )
        return self._stored_result(
            cursor.fetchone(),  # type: ignore[attr-defined]
            content_digest,
            sequence=None,
            typed_fields=_TERMINAL_TYPED_FIELDS,
            expected_typed_values=tuple(payload[field] for field in _TERMINAL_TYPED_FIELDS),
        )

    def _commit_terminal_locked(
        self,
        cursor: object,
        outcome: object,
    ) -> tuple[CommitResult, str, dict[str, Any] | None]:
        if not isinstance(outcome, TerminalOutcome):
            return CommitResult(status="conflict"), "", None
        try:
            payload = TerminalOutcome.from_json(outcome.to_json()).to_json()
        except (TypeError, ValueError, OverflowError, RecursionError):
            return CommitResult(status="conflict"), "", None
        checkpoint_sequence = payload["checkpoint_seq"]
        if checkpoint_sequence is not None and checkpoint_sequence > _POSTGRES_BIGINT_MAX:
            return CommitResult(status="conflict"), "", None
        content_digest = _record_only_digest(payload)
        stored = self._terminal_coordinate(cursor, outcome, payload, content_digest)
        if stored is not None:
            return stored, content_digest, payload

        from psycopg import sql
        from psycopg.types.json import Json

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "INSERT INTO {} (run_id, schema_version, outcome_kind, retry_eligibility, "
                "checkpoint_sequence, error_code, content_digest, payload) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            ).format(self._table("terminal_record")),
            (
                outcome.run_id,
                payload["schema_version"],
                payload["kind"],
                payload["retry_eligibility"],
                checkpoint_sequence,
                payload["error_code"],
                content_digest,
                Json(payload),
            ),
        )
        return (
            CommitResult(status="committed", content_digest=content_digest),
            content_digest,
            payload,
        )

    def _reconcile_terminal(
        self,
        outcome: TerminalOutcome,
        writer_token: WriterToken,
        payload: dict[str, Any],
        content_digest: str,
    ) -> CommitResult | None:
        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                if not self._current_writer_locked(cursor, writer_token):
                    return CommitResult(status="fenced")
                return self._terminal_coordinate(cursor, outcome, payload, content_digest)

    def settle_terminal(
        self,
        outcome: TerminalOutcome,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        self._require_ready()
        if not isinstance(writer_token, WriterToken):
            raise TypeError("PostgreSQL terminal settlement requires WriterToken")
        if getattr(outcome, "run_id", None) != writer_token.run_id:
            return CommitResult(status="fenced")
        result: CommitResult | None = None
        content_digest = ""
        payload: dict[str, Any] | None = None
        try:
            with self.database.transaction() as connection:
                with self.database.cursor(connection) as cursor:
                    if not self._current_writer_locked(cursor, writer_token):
                        return CommitResult(status="fenced")
                    result, content_digest, payload = self._commit_terminal_locked(cursor, outcome)
            if result is None:  # pragma: no cover - transaction body always assigns a result
                raise RuntimeError("PostgreSQL terminal transaction returned no result")
            return result
        except Exception as exc:
            if not content_digest or payload is None or not _is_ambiguous_database_error(exc):
                raise
            try:
                reconciled = self._reconcile_terminal(
                    outcome,
                    writer_token,
                    payload,
                    content_digest,
                )
            except Exception:
                raise exc
            if reconciled is None:
                raise
            return reconciled

    def _invocation_coordinate(
        self,
        cursor: object,
        invocation: DurableModelInvocation,
        payload: dict[str, Any],
        content_digest: str,
    ) -> CommitResult | None:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "SELECT content_digest, payload, submitted_blobs, run_id, logical_call_id, "
                "revision, schema_version, dispatch_id, dispatch_attempt, dispatch_state, "
                "idempotency_key, request_digest, digest_generation, evidence_policy, result_ref, "
                "failure_code FROM {} WHERE run_id = %s AND logical_call_id = %s AND revision = %s"
            ).format(self._table("invocation_record")),
            (invocation.run_id, invocation.logical_call_id, invocation.revision),
        )
        return self._stored_result(
            cursor.fetchone(),  # type: ignore[attr-defined]
            content_digest,
            sequence=invocation.revision,
            typed_fields=_INVOCATION_TYPED_FIELDS,
            expected_typed_values=tuple(payload[field] for field in _INVOCATION_TYPED_FIELDS),
        )

    def _invocation_head(self, cursor: object, run_id: str, logical_call_id: str) -> object:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "SELECT record.revision, record.content_digest, record.payload, "
                "record.submitted_blobs, record.run_id, record.logical_call_id, record.revision, "
                "record.schema_version, record.dispatch_id, record.dispatch_attempt, "
                "record.dispatch_state, record.idempotency_key, record.request_digest, "
                "record.digest_generation, record.evidence_policy, record.result_ref, "
                "record.failure_code "
                "FROM {} AS head LEFT JOIN {} AS record "
                "ON record.run_id = head.run_id "
                "AND record.logical_call_id = head.logical_call_id "
                "AND record.revision = head.revision "
                "WHERE head.run_id = %s AND head.logical_call_id = %s"
            ).format(self._table("invocation_head"), self._table("invocation_record")),
            (run_id, logical_call_id),
        )
        return cursor.fetchone()  # type: ignore[attr-defined]

    def _checked_invocation_head(
        self,
        cursor: object,
        run_id: str,
        logical_call_id: str,
    ) -> tuple[DurableModelInvocation, str] | None:
        row = self._invocation_head(cursor, run_id, logical_call_id)
        if row is None or row[0] is None or row[1] is None or not isinstance(row[2], dict):
            return None
        revision = int(row[0])
        stored_typed_values = tuple(row[4:])
        content_digest = _checked_stored_digest(row[2], row[3], row[1])
        if (
            content_digest is None
            or stored_typed_values[:3] != (run_id, logical_call_id, revision)
            or not _payload_matches_typed_values(
                row[2],
                _INVOCATION_TYPED_FIELDS,
                stored_typed_values,
            )
        ):
            return None
        decoded = decode_model_invocation(row[2])
        if not decoded.ok or decoded.value is None:
            return None
        invocation = decoded.value
        if (
            invocation.run_id != run_id
            or invocation.logical_call_id != logical_call_id
            or invocation.revision != revision
        ):
            return None
        return invocation, _record_only_digest(row[2])

    def _evidence_coordinate(
        self,
        cursor: object,
        invocation: DurableModelInvocation,
        payload: dict[str, Any],
        content_digest: str,
        *,
        outbox: bool,
    ) -> CommitResult | None:
        from psycopg import sql

        table = "model_evidence_outbox" if outbox else "model_evidence_record"
        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "SELECT content_digest, payload, '{{}}'::jsonb, run_id, logical_call_id, "
                "revision, schema_version, evidence_policy FROM {} "
                "WHERE run_id = %s AND logical_call_id = %s AND revision = %s"
            ).format(self._table(table)),
            (invocation.run_id, invocation.logical_call_id, invocation.revision),
        )
        return self._stored_result(
            cursor.fetchone(),  # type: ignore[attr-defined]
            content_digest,
            sequence=invocation.revision,
            typed_fields=_EVIDENCE_TYPED_FIELDS,
            expected_typed_values=tuple(payload[field] for field in _EVIDENCE_TYPED_FIELDS),
        )

    def _commit_evidence_locked(
        self,
        cursor: object,
        invocation: object,
        *,
        outbox: bool,
    ) -> tuple[CommitResult, str, dict[str, Any] | None]:
        revision = getattr(invocation, "revision", None)
        sequence = revision if is_portable_json_integer(revision) and revision >= 0 else None
        if (
            not isinstance(invocation, DurableModelInvocation)
            or invocation.dispatch_state != "settled"
            or invocation.receipt is None
            or (outbox and invocation.evidence_policy != "outbox")
        ):
            return CommitResult(status="conflict", sequence=sequence), "", None
        try:
            payload = invocation.to_json()
            content_digest = _record_only_digest(payload)
        except (TypeError, ValueError, OverflowError, RecursionError):
            return CommitResult(status="conflict", sequence=sequence), "", None
        authoritative = self._checked_invocation_head(
            cursor,
            invocation.run_id,
            invocation.logical_call_id,
        )
        if authoritative is None or authoritative[1] != content_digest:
            return (
                CommitResult(
                    status="conflict",
                    sequence=invocation.revision,
                    content_digest=content_digest,
                ),
                content_digest,
                payload,
            )
        stored = self._evidence_coordinate(
            cursor,
            invocation,
            payload,
            content_digest,
            outbox=outbox,
        )
        if stored is not None:
            return stored, content_digest, payload

        from psycopg import sql
        from psycopg.types.json import Json

        table = "model_evidence_outbox" if outbox else "model_evidence_record"
        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "INSERT INTO {} (run_id, logical_call_id, revision, schema_version, "
                "evidence_policy, content_digest, payload) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)"
            ).format(self._table(table)),
            (
                invocation.run_id,
                invocation.logical_call_id,
                invocation.revision,
                payload["schema_version"],
                invocation.evidence_policy,
                content_digest,
                Json(payload),
            ),
        )
        return (
            CommitResult(
                status="committed",
                sequence=invocation.revision,
                content_digest=content_digest,
            ),
            content_digest,
            payload,
        )

    def _invocation_transition_winner(
        self,
        cursor: object,
        invocation: DurableModelInvocation,
    ) -> str | None:
        head_row = self._invocation_head(cursor, invocation.run_id, invocation.logical_call_id)
        if head_row is None:
            if (
                invocation.revision == 1
                and invocation.dispatch_attempt == 1
                and invocation.dispatch_state == "reserved"
            ):
                return None
            return ""
        if head_row[0] is None or head_row[1] is None or not isinstance(head_row[2], dict):
            return ""
        previous_digest = str(head_row[1])
        stored_typed_values = tuple(head_row[4:])
        if (
            _checked_stored_digest(head_row[2], head_row[3], head_row[1]) is None
            or int(head_row[0]) != stored_typed_values[2]
            or not _payload_matches_typed_values(
                head_row[2],
                _INVOCATION_TYPED_FIELDS,
                stored_typed_values,
            )
        ):
            return previous_digest
        decoded = decode_model_invocation(head_row[2])
        if not decoded.ok or decoded.value is None:
            return previous_digest
        previous = decoded.value
        if (
            previous.run_id != invocation.run_id
            or previous.logical_call_id != invocation.logical_call_id
            or previous.revision != int(head_row[0])
        ):
            return previous_digest
        if invocation.revision != previous.revision + 1:
            return previous_digest
        invocation_payload = invocation.to_json()
        previous_payload = previous.to_json()
        if any(
            invocation_payload[field_name] != previous_payload[field_name]
            for field_name in _STABLE_INVOCATION_FIELDS
        ):
            return previous_digest
        if previous.dispatch_state in {"settled", "unknown"}:
            retryable_failure = (
                previous.dispatch_state == "settled"
                and bool(previous.failure_code)
                and previous.receipt is not None
                and previous.receipt.get("retryable") is True
            )
            if not retryable_failure:
                return previous_digest
            if not (
                invocation.dispatch_state == "reserved"
                and invocation.dispatch_attempt == previous.dispatch_attempt + 1
                and not self._dispatch_id_was_used(cursor, invocation)
            ):
                return previous_digest
            return None
        if invocation.dispatch_attempt != previous.dispatch_attempt:
            return previous_digest
        if invocation.dispatch_id != previous.dispatch_id:
            return previous_digest
        allowed_next = {
            "reserved": frozenset({"dispatch_started"}),
            "dispatch_started": frozenset({"settled", "unknown"}),
        }
        if invocation.dispatch_state not in allowed_next[previous.dispatch_state]:
            return previous_digest
        return None

    def _dispatch_id_was_used(
        self,
        cursor: object,
        invocation: DurableModelInvocation,
    ) -> bool:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "SELECT content_digest, payload, submitted_blobs, run_id, logical_call_id, "
                "revision, schema_version, dispatch_id, dispatch_attempt, dispatch_state, "
                "idempotency_key, request_digest, digest_generation, evidence_policy, result_ref, "
                "failure_code FROM {} WHERE run_id = %s AND logical_call_id = %s"
            ).format(self._table("invocation_record")),
            (invocation.run_id, invocation.logical_call_id),
        )
        for row in cursor:  # type: ignore[operator]
            stored_typed_values = tuple(row[3:])
            if _checked_stored_digest(
                row[1], row[2], row[0]
            ) is None or not _payload_matches_typed_values(
                row[1],
                _INVOCATION_TYPED_FIELDS,
                stored_typed_values,
            ):
                return True
            decoded = decode_model_invocation(row[1])
            if not decoded.ok or decoded.value is None:
                return True
            if decoded.value.dispatch_id == invocation.dispatch_id:
                return True
        return False

    def _commit_invocation_locked(
        self,
        cursor: object,
        invocation: object,
        blobs: object,
        *,
        stage_evidence: object,
    ) -> tuple[CommitResult, str]:
        if not isinstance(invocation, DurableModelInvocation):
            return CommitResult(status="conflict"), ""
        if not 1 <= invocation.revision <= _POSTGRES_BIGINT_MAX:
            return CommitResult(status="conflict", sequence=invocation.revision), ""
        if not 1 <= invocation.dispatch_attempt <= _POSTGRES_BIGINT_MAX:
            return CommitResult(status="conflict", sequence=invocation.revision), ""
        if (
            type(stage_evidence) is not bool
            or (
                stage_evidence
                and not (
                    invocation.evidence_policy == "outbox"
                    and invocation.dispatch_state == "settled"
                    and invocation.receipt is not None
                )
            )
            or (
                not stage_evidence
                and invocation.evidence_policy == "outbox"
                and invocation.dispatch_state == "settled"
            )
        ):
            return CommitResult(status="conflict", sequence=invocation.revision), ""
        submitted = self._validated_blobs(blobs)
        if submitted is None:
            return CommitResult(status="conflict", sequence=invocation.revision), ""
        try:
            payload = invocation.to_json()
        except (TypeError, ValueError, OverflowError, RecursionError):
            return CommitResult(status="conflict", sequence=invocation.revision), ""
        references = (
            {invocation.result_ref.removeprefix("blob:")}
            if invocation.result_ref.startswith("blob:")
            else set()
        )
        if not self._submitted_blobs_preserve_backing(cursor, submitted):
            return CommitResult(status="conflict", sequence=invocation.revision), ""
        if not self._references_resolve(cursor, invocation.run_id, references, submitted):
            return CommitResult(status="conflict", sequence=invocation.revision), ""
        content_digest = _record_digest(payload, submitted)
        stored = self._invocation_coordinate(cursor, invocation, payload, content_digest)
        if stored is not None:
            if stage_evidence and stored.status == "already_committed":
                evidence_digest = _record_only_digest(payload)
                staged = self._evidence_coordinate(
                    cursor,
                    invocation,
                    payload,
                    evidence_digest,
                    outbox=True,
                )
                if staged is None or staged.status != "already_committed":
                    return (
                        CommitResult(
                            status="conflict",
                            sequence=invocation.revision,
                            content_digest=content_digest,
                        ),
                        content_digest,
                    )
            return stored, content_digest
        transition_winner = self._invocation_transition_winner(cursor, invocation)
        if transition_winner is not None:
            return (
                CommitResult(
                    status="conflict",
                    sequence=invocation.revision,
                    content_digest=content_digest,
                    winner_digest=transition_winner,
                ),
                content_digest,
            )

        self._persist_blobs(cursor, invocation.run_id, submitted)
        from psycopg import sql
        from psycopg.types.json import Json, Jsonb

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "INSERT INTO {} (run_id, logical_call_id, revision, schema_version, dispatch_id, "
                "dispatch_attempt, dispatch_state, idempotency_key, request_digest, "
                "digest_generation, evidence_policy, result_ref, failure_code, content_digest, "
                "payload, submitted_blobs) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "%s, %s, %s, %s, %s, %s)"
            ).format(self._table("invocation_record")),
            (
                invocation.run_id,
                invocation.logical_call_id,
                invocation.revision,
                payload["schema_version"],
                invocation.dispatch_id,
                invocation.dispatch_attempt,
                invocation.dispatch_state,
                invocation.idempotency_key,
                invocation.request_digest,
                payload["digest_generation"],
                invocation.evidence_policy,
                invocation.result_ref,
                invocation.failure_code,
                content_digest,
                Json(payload),
                Jsonb(_blob_projection(submitted)),
            ),
        )
        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "INSERT INTO {} (run_id, logical_call_id, revision) VALUES (%s, %s, %s) "
                "ON CONFLICT (run_id, logical_call_id) DO UPDATE SET revision = EXCLUDED.revision, "
                "updated_at = pg_catalog.clock_timestamp()"
            ).format(self._table("invocation_head")),
            (invocation.run_id, invocation.logical_call_id, invocation.revision),
        )
        if stage_evidence:
            staged, _, _ = self._commit_evidence_locked(
                cursor,
                invocation,
                outbox=True,
            )
            if staged.status != "committed":
                raise _EvidenceStageConflict
        return (
            CommitResult(
                status="committed",
                sequence=invocation.revision,
                content_digest=content_digest,
            ),
            content_digest,
        )

    def _reconcile_invocation(
        self,
        invocation: DurableModelInvocation,
        writer_token: WriterToken,
        content_digest: str,
        *,
        stage_evidence: bool,
    ) -> CommitResult | None:
        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                if not self._current_writer_locked(cursor, writer_token):
                    return CommitResult(status="fenced")
                result = self._invocation_coordinate(
                    cursor,
                    invocation,
                    invocation.to_json(),
                    content_digest,
                )
                if (
                    not stage_evidence
                    or result is None
                    or result.status not in {"committed", "already_committed"}
                ):
                    return result
                payload = invocation.to_json()
                staged = self._evidence_coordinate(
                    cursor,
                    invocation,
                    payload,
                    _record_only_digest(payload),
                    outbox=True,
                )
                if staged is not None and staged.status == "already_committed":
                    return result
                return CommitResult(
                    status="conflict",
                    sequence=invocation.revision,
                    content_digest=content_digest,
                )

    def commit_invocation(
        self,
        invocation: DurableModelInvocation,
        blobs: Mapping[str, bytes],
        *,
        writer_token: WriterToken,
        stage_evidence: bool = False,
    ) -> CommitResult:
        self._require_ready()
        if not isinstance(writer_token, WriterToken):
            raise TypeError("PostgreSQL invocation commit requires WriterToken")
        if getattr(invocation, "run_id", None) != writer_token.run_id:
            return CommitResult(status="fenced")
        result: CommitResult | None = None
        content_digest = ""
        try:
            with self.database.transaction() as connection:
                with self.database.cursor(connection) as cursor:
                    if not self._current_writer_locked(cursor, writer_token):
                        return CommitResult(status="fenced")
                    result, content_digest = self._commit_invocation_locked(
                        cursor,
                        invocation,
                        blobs,
                        stage_evidence=stage_evidence,
                    )
            if result is None:  # pragma: no cover - transaction body always assigns a result
                raise RuntimeError("PostgreSQL invocation transaction returned no result")
            return result
        except (_BlobRaceConflict, _EvidenceStageConflict):
            return CommitResult(
                status="conflict",
                sequence=getattr(invocation, "revision", None),
                content_digest=content_digest,
            )
        except Exception as exc:
            if not content_digest or not _is_ambiguous_database_error(exc):
                raise
            try:
                reconciled = self._reconcile_invocation(
                    invocation,
                    writer_token,
                    content_digest,
                    stage_evidence=stage_evidence,
                )
            except Exception:
                raise exc
            if reconciled is None:
                raise
            return reconciled

    def _reconcile_evidence(
        self,
        invocation: DurableModelInvocation,
        writer_token: WriterToken,
        payload: dict[str, Any],
        content_digest: str,
    ) -> CommitResult | None:
        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                if not self._current_writer_locked(cursor, writer_token):
                    return CommitResult(status="fenced", sequence=invocation.revision)
                authoritative = self._checked_invocation_head(
                    cursor,
                    invocation.run_id,
                    invocation.logical_call_id,
                )
                if authoritative is None or authoritative[1] != content_digest:
                    return CommitResult(
                        status="conflict",
                        sequence=invocation.revision,
                        content_digest=content_digest,
                    )
                return self._evidence_coordinate(
                    cursor,
                    invocation,
                    payload,
                    content_digest,
                    outbox=False,
                )

    def commit_model_evidence(
        self,
        invocation: DurableModelInvocation,
        *,
        writer_token: WriterToken,
    ) -> CommitResult:
        self._require_ready()
        if not isinstance(writer_token, WriterToken):
            raise TypeError("PostgreSQL model evidence commit requires WriterToken")
        if getattr(invocation, "run_id", None) != writer_token.run_id:
            return CommitResult(
                status="fenced",
                sequence=(
                    invocation.revision if isinstance(invocation, DurableModelInvocation) else None
                ),
            )
        result: CommitResult | None = None
        content_digest = ""
        payload: dict[str, Any] | None = None
        try:
            with self.database.transaction() as connection:
                with self.database.cursor(connection) as cursor:
                    if not self._current_writer_locked(cursor, writer_token):
                        return CommitResult(
                            status="fenced",
                            sequence=(
                                invocation.revision
                                if isinstance(invocation, DurableModelInvocation)
                                else None
                            ),
                        )
                    result, content_digest, payload = self._commit_evidence_locked(
                        cursor,
                        invocation,
                        outbox=False,
                    )
            if result is None:  # pragma: no cover - transaction body always assigns a result
                raise RuntimeError("PostgreSQL model evidence transaction returned no result")
            return result
        except Exception as exc:
            if not content_digest or payload is None or not _is_ambiguous_database_error(exc):
                raise
            try:
                reconciled = self._reconcile_evidence(
                    invocation,
                    writer_token,
                    payload,
                    content_digest,
                )
            except Exception:
                raise exc
            if reconciled is None:
                raise
            return reconciled

    def _blob_reader(self, run_id: str) -> Callable[[str], bytes]:
        return lambda sha256: self._read_blob(run_id, sha256)

    def _read_blob(self, run_id: str, sha256: str) -> bytes:
        self._require_ready()
        if not is_safe_opaque_id(run_id) or not is_recorded_digest(sha256):
            raise KeyError(sha256)
        from psycopg import sql

        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                cursor.execute(
                    sql.SQL(
                        "SELECT blob.size_bytes, blob.content FROM {} AS association "
                        "JOIN {} AS blob ON blob.sha256 = association.sha256 "
                        "WHERE association.run_id = %s AND association.sha256 = %s"
                    ).format(self._table("run_blob"), self._table("bytea_blob")),
                    (run_id, sha256),
                )
                row = cursor.fetchone()
        if row is None:
            raise KeyError(sha256)
        value = bytes(row[1])
        if int(row[0]) != len(value) or hashlib.sha256(value).hexdigest() != sha256:
            raise PostgresBlobCorrupt("PostgreSQL bytea blob failed content-address verification")
        return value

    def latest_checked(self, run_id: str) -> DurableLoadResult[CheckpointRecord]:
        self._require_ready()
        if not is_safe_opaque_id(run_id):
            raise ValueError("PostgreSQL checkpoint run_id must be a bounded opaque id")
        from psycopg import sql

        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                cursor.execute(
                    sql.SQL(
                        "SELECT head.sequence, record.content_digest, record.payload, "
                        "record.submitted_blobs, record.run_id, record.sequence, "
                        "record.schema_version FROM {} AS head LEFT JOIN {} AS record "
                        "ON record.run_id = head.run_id AND record.sequence = head.sequence "
                        "WHERE head.run_id = %s"
                    ).format(self._table("checkpoint_head"), self._table("checkpoint_record")),
                    (run_id,),
                )
                row = cursor.fetchone()
        if row is None:
            return CHECKPOINT_CODEC.missing().map(
                lambda checkpoint: CheckpointRecord(seq=checkpoint.seq, checkpoint=checkpoint)
            )
        sequence = int(row[0])
        stored_typed_values = tuple(row[4:])
        if (
            row[1] is None
            or _checked_stored_digest(row[2], row[3], row[1]) is None
            or stored_typed_values[:2] != (run_id, sequence)
            or not _payload_matches_typed_values(
                row[2],
                _CHECKPOINT_TYPED_FIELDS,
                stored_typed_values,
            )
        ):
            return CHECKPOINT_CODEC.corrupt(
                "PostgreSQL checkpoint head or content digest is inconsistent",
                sequence=sequence,
            ).map(lambda checkpoint: CheckpointRecord(seq=checkpoint.seq, checkpoint=checkpoint))
        decoded = replace(decode_checkpoint(row[2]), sequence=sequence)
        return bind_checkpoint_record_result(
            decoded.map(
                lambda checkpoint: CheckpointRecord(
                    seq=sequence,
                    checkpoint=checkpoint,
                    _blob_reader=self._blob_reader(run_id),
                )
            ),
            run_id,
        )

    def load_invocation(
        self,
        run_id: str,
        logical_call_id: str,
    ) -> DurableLoadResult[ModelInvocationRecord]:
        self._require_ready()
        if not is_safe_opaque_id(run_id):
            raise ValueError("PostgreSQL invocation run_id must be a bounded opaque id")
        if not is_safe_opaque_id(logical_call_id):
            raise ValueError("PostgreSQL logical_call_id must be a bounded opaque id")
        from psycopg import sql

        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                cursor.execute(
                    sql.SQL(
                        "SELECT head.revision, record.content_digest, record.payload, "
                        "record.submitted_blobs, record.run_id, record.logical_call_id, "
                        "record.revision, record.schema_version, record.dispatch_id, "
                        "record.dispatch_attempt, record.dispatch_state, record.idempotency_key, "
                        "record.request_digest, record.digest_generation, record.evidence_policy, "
                        "record.result_ref, record.failure_code "
                        "FROM {} AS head LEFT JOIN {} AS record "
                        "ON record.run_id = head.run_id "
                        "AND record.logical_call_id = head.logical_call_id "
                        "AND record.revision = head.revision "
                        "WHERE head.run_id = %s AND head.logical_call_id = %s"
                    ).format(self._table("invocation_head"), self._table("invocation_record")),
                    (run_id, logical_call_id),
                )
                row = cursor.fetchone()
        if row is None:
            return MODEL_INVOCATION_CODEC.missing()
        revision = int(row[0])
        stored_typed_values = tuple(row[4:])
        if (
            row[1] is None
            or _checked_stored_digest(row[2], row[3], row[1]) is None
            or stored_typed_values[:3] != (run_id, logical_call_id, revision)
            or not _payload_matches_typed_values(
                row[2],
                _INVOCATION_TYPED_FIELDS,
                stored_typed_values,
            )
        ):
            return MODEL_INVOCATION_CODEC.corrupt(
                "PostgreSQL invocation head or content digest is inconsistent",
                sequence=revision,
            )
        decoded = replace(decode_model_invocation(row[2]), sequence=revision)
        if decoded.ok and decoded.value is not None:
            invocation = decoded.value
            if (
                invocation.run_id != run_id
                or invocation.logical_call_id != logical_call_id
                or invocation.revision != revision
            ):
                return MODEL_INVOCATION_CODEC.corrupt(
                    "PostgreSQL invocation payload does not match its authoritative coordinate",
                    observed_schema=invocation.schema_version,
                    sequence=revision,
                )
        return decoded.map(
            lambda invocation: ModelInvocationRecord(
                revision=revision,
                invocation=invocation,
                _blob_reader=self._blob_reader(run_id),
            )
        )

    def read_event(self, run_id: str, sequence: int) -> AgentEvent | None:
        """Read and verify one complete event by its authoritative coordinate."""

        self._require_ready()
        if not is_safe_opaque_id(run_id):
            raise ValueError("PostgreSQL event run_id must be a bounded opaque id")
        if (
            not is_portable_json_integer(sequence)
            or sequence < 1
            or sequence > _POSTGRES_BIGINT_MAX
        ):
            raise ValueError("PostgreSQL event sequence must fit a positive bigint")
        from psycopg import sql

        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                cursor.execute(
                    sql.SQL(
                        "SELECT content_digest, payload, run_id, sequence, event_id, "
                        "schema_version, event_type, event_level, event_timestamp "
                        "FROM {} WHERE run_id = %s AND sequence = %s"
                    ).format(self._table("event_record")),
                    (run_id, sequence),
                )
                row = cursor.fetchone()
        if row is None:
            return None
        payload = row[1]
        try:
            event = AgentEvent(**payload)
        except (TypeError, ValueError):
            raise RuntimeError("PostgreSQL event payload is corrupt") from None
        expected_typed_values = tuple(payload[field] for field in _EVENT_TYPED_FIELDS)
        stored_typed_values = tuple(row[2:8])
        try:
            timestamp_matches = row[8] == datetime.fromisoformat(
                payload["timestamp"][:-1] + "+00:00"
            )
        except (TypeError, ValueError):
            timestamp_matches = False
        if (
            _checked_stored_digest(payload, {}, row[0]) is None
            or stored_typed_values != expected_typed_values
            or not timestamp_matches
            or self._event_payload(event) != payload
        ):
            raise RuntimeError("PostgreSQL event record is corrupt")
        return event

    def read_terminal(self, run_id: str) -> TerminalOutcome | None:
        """Read and verify the first-writer terminal winner for a run."""

        self._require_ready()
        if not is_safe_opaque_id(run_id):
            raise ValueError("PostgreSQL terminal run_id must be a bounded opaque id")
        from psycopg import sql

        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                cursor.execute(
                    sql.SQL(
                        "SELECT content_digest, payload, '{{}}'::jsonb, run_id, schema_version, "
                        "outcome_kind, retry_eligibility, checkpoint_sequence, error_code "
                        "FROM {} WHERE run_id = %s"
                    ).format(self._table("terminal_record")),
                    (run_id,),
                )
                row = cursor.fetchone()
        if row is None:
            return None
        try:
            outcome = TerminalOutcome.from_json(row[1])
        except (TypeError, ValueError, OverflowError, RecursionError):
            raise RuntimeError("PostgreSQL terminal payload is corrupt") from None
        payload = outcome.to_json()
        result = self._stored_result(
            row,
            _record_only_digest(payload),
            sequence=None,
            typed_fields=_TERMINAL_TYPED_FIELDS,
            expected_typed_values=tuple(payload[field] for field in _TERMINAL_TYPED_FIELDS),
        )
        if result is None or result.status != "already_committed":
            raise RuntimeError("PostgreSQL terminal record is corrupt")
        return outcome


__all__ = ["PostgresBlobCorrupt", "PostgresFencedRunSink"]
