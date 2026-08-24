"""PostgreSQL metadata and ObjectStore bytes for fenced durable streams."""

from __future__ import annotations

import hashlib

from monoid_agent_kernel.adapters.postgres.authority import PostgresWriterAuthorityStore
from monoid_agent_kernel.adapters.postgres.migrations import (
    MigrationStatus,
    PostgresMigrations,
    PostgresSchemaIncompatible,
)
from monoid_agent_kernel.adapters.postgres.pool import PostgresDatabase
from monoid_agent_kernel.core.json_ingress import is_portable_json_integer
from monoid_agent_kernel.core.safe_evidence import is_safe_opaque_id
from monoid_agent_kernel.hosting import (
    BlobCorrupt,
    BlobNotFound,
    BlobStat,
    ContentAddressedBlobStore,
    DurableStreamAppendResult,
    DurableStreamChunk,
    DurableStreamHead,
    DurableStreamIdentity,
    DurableStreamOpenResult,
    DurableStreamReadChunk,
    DurableStreamReadResult,
    DurableStreamResetResult,
    DurableStreamSealResult,
    MAX_STREAM_CHUNK_BYTES,
    MAX_STREAM_READ_CHUNKS,
    WriterToken,
)


_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class PostgresDurableStreamCorrupt(BlobCorrupt):
    """Stream metadata and immutable physical bytes no longer agree."""


def _supports_store(value: object) -> bool:
    return all(
        callable(getattr(value, method, None))
        for method in ("put_if_absent", "stat", "get_checked")
    )


def _portable_positive(value: object, field_name: str) -> int:
    if not is_portable_json_integer(value) or value < 1:  # type: ignore[operator]
        raise ValueError(f"{field_name} must be a positive portable integer")
    return value  # type: ignore[return-value]


def _portable_non_negative(value: object, field_name: str) -> int:
    if not is_portable_json_integer(value) or value < 0:  # type: ignore[operator]
        raise ValueError(f"{field_name} must be a non-negative portable integer")
    return value  # type: ignore[return-value]


def _head_from_row(row: object) -> DurableStreamHead:
    values = tuple(row)  # type: ignore[arg-type]
    return DurableStreamHead(
        identity=DurableStreamIdentity(
            run_id=str(values[0]),
            stream_id=str(values[1]),
            channel=str(values[2]),
            logical_call_id=str(values[3]),
        ),
        generation=int(values[4]),
        cursor_bytes=int(values[5]),
        next_chunk_sequence=int(values[6]),
        state=str(values[7]),  # type: ignore[arg-type]
        final_sha256=str(values[8]),
    )


def _identity_matches(stored: DurableStreamIdentity, submitted: DurableStreamIdentity) -> bool:
    return stored == submitted


class PostgresObjectStoreDurableStreamStore:
    """Fenced stream head/chunk journal with immutable ObjectStore batch bytes."""

    def __init__(
        self,
        database: PostgresDatabase,
        object_store: ContentAddressedBlobStore,
    ) -> None:
        if not isinstance(database, PostgresDatabase):
            raise TypeError("PostgreSQL durable stream database must be PostgresDatabase")
        if not _supports_store(object_store):
            raise TypeError("durable stream object_store must satisfy ContentAddressedBlobStore")
        self.database = database
        self.object_store = object_store
        self._authority = PostgresWriterAuthorityStore(database)
        self._ready = False

    def _table(self, name: str) -> object:
        from psycopg import sql

        return sql.Identifier(self.database.config.schema, name)

    def check_ready(self) -> MigrationStatus:
        self._ready = False
        status = PostgresMigrations(self.database).require_writer_compatible()
        if not status.reader_compatible:
            raise PostgresSchemaIncompatible(
                "PostgreSQL durable stream store requires reader and writer compatibility"
            )
        self._ready = True
        return status

    def _require_ready(self) -> None:
        if not self._ready:
            raise PostgresSchemaIncompatible(
                "PostgreSQL durable stream store requires a successful check_ready()"
            )

    @staticmethod
    def _validate_identity_and_token(
        identity: object,
        writer_token: object,
    ) -> tuple[DurableStreamIdentity, WriterToken] | None:
        if not isinstance(identity, DurableStreamIdentity):
            raise TypeError("durable stream mutation requires DurableStreamIdentity")
        if not isinstance(writer_token, WriterToken):
            raise TypeError("durable stream mutation requires WriterToken")
        if identity.run_id != writer_token.run_id:
            return None
        return identity, writer_token

    def _current_writer_locked(self, cursor: object, writer_token: WriterToken) -> bool:
        current = self._authority._read_locked(cursor, writer_token.run_id)
        return current is not None and current.writer_token == writer_token and current.active

    def _head_locked(
        self,
        cursor: object,
        identity: DurableStreamIdentity,
        *,
        shared: bool = False,
    ) -> DurableStreamHead | None:
        from psycopg import sql

        lock = sql.SQL("FOR SHARE") if shared else sql.SQL("FOR UPDATE")
        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "SELECT run_id, stream_id, channel, logical_call_id, generation, "
                "cursor_bytes, next_chunk_sequence, state, final_sha256 FROM {} "
                "WHERE run_id = %s AND stream_id = %s AND channel = %s "
            ).format(self._table("durable_stream_head"))
            + lock,
            (identity.run_id, identity.stream_id, identity.channel),
        )
        row = cursor.fetchone()  # type: ignore[attr-defined]
        return _head_from_row(row) if row is not None else None

    def _run_is_terminal(self, cursor: object, run_id: str) -> bool:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL("SELECT 1 FROM {} WHERE run_id = %s").format(
                self._table("terminal_record")
            ),
            (run_id,),
        )
        return cursor.fetchone() is not None  # type: ignore[attr-defined]

    @staticmethod
    def _stat_matches(first: BlobStat | None, second: BlobStat) -> bool:
        return first is not None and (
            first.sha256,
            first.size_bytes,
            first.locator,
        ) == (second.sha256, second.size_bytes, second.locator)

    def _put_checked(self, data: bytes) -> BlobStat:
        sha256 = hashlib.sha256(data).hexdigest()
        result = self.object_store.put_if_absent(sha256, data)
        stat = result.stat
        if (
            stat.sha256 != sha256
            or stat.size_bytes != len(data)
            or not self._stat_matches(self.object_store.stat(sha256), stat)
        ):
            raise PostgresDurableStreamCorrupt(
                "ObjectStore stream put result disagrees with checked caller bytes"
            )
        return stat

    def _record_object(
        self,
        cursor: object,
        run_id: str,
        stat: BlobStat,
    ) -> None:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            "SELECT pg_catalog.pg_advisory_xact_lock("
            "pg_catalog.hashtextextended(%s, 0))",
            (f"monoid-agent-kernel:object:{self.database.config.schema}:{stat.sha256}",),
        )
        # GC uses the same digest lock. The stat captured before this transaction can become stale
        # while GC deletes an unassociated object, so physical storage is authoritative only after
        # this lock has been acquired and before metadata is made available/associated.
        if not self._stat_matches(self.object_store.stat(stat.sha256), stat):
            raise PostgresDurableStreamCorrupt(
                "ObjectStore stream object changed before digest-locked association"
            )
        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "SELECT size_bytes, locator, generation, state FROM {} WHERE sha256 = %s"
            ).format(self._table("object_blob")),
            (stat.sha256,),
        )
        row = cursor.fetchone()  # type: ignore[attr-defined]
        if row is None:
            cursor.execute(  # type: ignore[attr-defined]
                sql.SQL(
                    "INSERT INTO {} (sha256, size_bytes, locator, generation, state) "
                    "VALUES (%s, %s, %s, 1, 'available')"
                ).format(self._table("object_blob")),
                (stat.sha256, stat.size_bytes, stat.locator),
            )
        elif str(row[3]) == "deleted":
            cursor.execute(  # type: ignore[attr-defined]
                sql.SQL(
                    "UPDATE {} SET size_bytes = %s, locator = %s, generation = generation + 1, "
                    "state = 'available', verified_at = pg_catalog.clock_timestamp(), "
                    "deleted_at = NULL WHERE sha256 = %s"
                ).format(self._table("object_blob")),
                (stat.size_bytes, stat.locator, stat.sha256),
            )
        elif int(row[0]) == stat.size_bytes and str(row[1]) == stat.locator:
            cursor.execute(  # type: ignore[attr-defined]
                sql.SQL(
                    "UPDATE {} SET verified_at = pg_catalog.clock_timestamp() WHERE sha256 = %s"
                ).format(self._table("object_blob")),
                (stat.sha256,),
            )
        else:
            raise PostgresDurableStreamCorrupt(
                "PostgreSQL object metadata conflicts with stream chunk storage"
            )
        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "INSERT INTO {} (run_id, sha256) VALUES (%s, %s) "
                "ON CONFLICT (run_id, sha256) DO NOTHING"
            ).format(self._table("run_object_blob")),
            (run_id, stat.sha256),
        )

    def open(
        self,
        identity: DurableStreamIdentity,
        *,
        writer_token: WriterToken,
    ) -> DurableStreamOpenResult:
        self._require_ready()
        checked = self._validate_identity_and_token(identity, writer_token)
        if checked is None:
            return DurableStreamOpenResult(status="fenced")
        identity, writer_token = checked
        from psycopg import sql

        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                if not self._current_writer_locked(cursor, writer_token):
                    return DurableStreamOpenResult(status="fenced")
                current = self._head_locked(cursor, identity)
                if current is not None:
                    if not _identity_matches(current.identity, identity):
                        return DurableStreamOpenResult(status="conflict", head=current)
                    return DurableStreamOpenResult(
                        status="already_open" if current.state == "open" else "sealed",
                        head=current,
                    )
                if self._run_is_terminal(cursor, identity.run_id):
                    return DurableStreamOpenResult(status="run_terminal")
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {} "
                        "(run_id, stream_id, channel, logical_call_id, generation, "
                        "cursor_bytes, next_chunk_sequence, state) "
                        "VALUES (%s, %s, %s, %s, 1, 0, 1, 'open')"
                    ).format(self._table("durable_stream_head")),
                    (
                        identity.run_id,
                        identity.stream_id,
                        identity.channel,
                        identity.logical_call_id,
                    ),
                )
                return DurableStreamOpenResult(
                    status="opened",
                    head=DurableStreamHead(
                        identity=identity,
                        generation=1,
                        cursor_bytes=0,
                        next_chunk_sequence=1,
                        state="open",
                    ),
                )

    def reset(
        self,
        identity: DurableStreamIdentity,
        *,
        expected_generation: int,
        reset_id: str,
        writer_token: WriterToken,
    ) -> DurableStreamResetResult:
        self._require_ready()
        expected_generation = _portable_positive(
            expected_generation,
            "durable stream expected_generation",
        )
        if not is_safe_opaque_id(reset_id):
            raise ValueError("durable stream reset_id must be a bounded opaque id")
        checked = self._validate_identity_and_token(identity, writer_token)
        if checked is None:
            return DurableStreamResetResult(status="fenced")
        identity, writer_token = checked
        from psycopg import sql

        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                if not self._current_writer_locked(cursor, writer_token):
                    return DurableStreamResetResult(status="fenced")
                current = self._head_locked(cursor, identity)
                if current is None:
                    return DurableStreamResetResult(status="conflict")
                if not _identity_matches(current.identity, identity):
                    return DurableStreamResetResult(status="conflict", head=current)
                cursor.execute(
                    sql.SQL(
                        "SELECT generation FROM {} WHERE run_id = %s AND stream_id = %s "
                        "AND channel = %s AND reset_id = %s"
                    ).format(self._table("durable_stream_reset_receipt")),
                    (identity.run_id, identity.stream_id, identity.channel, reset_id),
                )
                receipt = cursor.fetchone()
                if receipt is not None:
                    return DurableStreamResetResult(
                        status="already_reset",
                        head=current,
                        applied_generation=int(receipt[0]),
                    )
                if current.generation != expected_generation:
                    return DurableStreamResetResult(status="old_generation", head=current)
                if self._run_is_terminal(cursor, identity.run_id):
                    return DurableStreamResetResult(status="run_terminal", head=current)
                generation = _portable_positive(
                    current.generation + 1,
                    "durable stream reset generation",
                )
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {} "
                        "(run_id, stream_id, channel, reset_id, generation) "
                        "VALUES (%s, %s, %s, %s, %s)"
                    ).format(self._table("durable_stream_reset_receipt")),
                    (
                        identity.run_id,
                        identity.stream_id,
                        identity.channel,
                        reset_id,
                        generation,
                    ),
                )
                cursor.execute(
                    sql.SQL(
                        "UPDATE {} SET generation = %s, cursor_bytes = 0, "
                        "next_chunk_sequence = 1, state = 'open', final_sha256 = '', "
                        "sealed_at = NULL, updated_at = pg_catalog.clock_timestamp() "
                        "WHERE run_id = %s AND stream_id = %s AND channel = %s"
                    ).format(self._table("durable_stream_head")),
                    (generation, identity.run_id, identity.stream_id, identity.channel),
                )
                head = DurableStreamHead(
                    identity=identity,
                    generation=generation,
                    cursor_bytes=0,
                    next_chunk_sequence=1,
                    state="open",
                )
                return DurableStreamResetResult(
                    status="reset",
                    head=head,
                    applied_generation=generation,
                )

    def _chunk_at(
        self,
        cursor: object,
        identity: DurableStreamIdentity,
        generation: int,
        start_offset: int,
    ) -> DurableStreamChunk | None:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "SELECT chunk.chunk_sequence, chunk.start_offset, chunk.end_offset, "
                "chunk.sha256, blob.size_bytes, blob.locator, blob.state "
                "FROM {} AS chunk JOIN {} AS blob ON blob.sha256 = chunk.sha256 "
                "WHERE chunk.run_id = %s AND chunk.stream_id = %s AND chunk.channel = %s "
                "AND chunk.generation = %s AND chunk.start_offset = %s"
            ).format(
                self._table("durable_stream_chunk"),
                self._table("object_blob"),
            ),
            (
                identity.run_id,
                identity.stream_id,
                identity.channel,
                generation,
                start_offset,
            ),
        )
        row = cursor.fetchone()  # type: ignore[attr-defined]
        if row is None:
            return None
        chunk = DurableStreamChunk(
            identity=identity,
            generation=generation,
            sequence=int(row[0]),
            start_offset=int(row[1]),
            end_offset=int(row[2]),
            sha256=str(row[3]),
            locator=str(row[5]),
        )
        if str(row[6]) != "available" or int(row[4]) != chunk.size_bytes:
            raise PostgresDurableStreamCorrupt(
                "stream chunk references unavailable or inconsistent object metadata"
            )
        return chunk

    def append(
        self,
        identity: DurableStreamIdentity,
        *,
        generation: int,
        start_offset: int,
        data: bytes,
        writer_token: WriterToken,
    ) -> DurableStreamAppendResult:
        self._require_ready()
        generation = _portable_positive(generation, "durable stream append generation")
        start_offset = _portable_non_negative(
            start_offset,
            "durable stream append start_offset",
        )
        if type(data) is not bytes or not data or len(data) > MAX_STREAM_CHUNK_BYTES:
            raise ValueError(
                "durable stream append data must be non-empty bytes within MAX_STREAM_CHUNK_BYTES"
            )
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("durable stream append data must be complete UTF-8") from exc
        end_offset = _portable_positive(
            start_offset + len(data),
            "durable stream append end_offset",
        )
        checked = self._validate_identity_and_token(identity, writer_token)
        if checked is None:
            return DurableStreamAppendResult(status="fenced")
        identity, writer_token = checked
        sha256 = hashlib.sha256(data).hexdigest()
        from psycopg import sql

        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                if not self._current_writer_locked(cursor, writer_token):
                    return DurableStreamAppendResult(status="fenced")
                current = self._head_locked(cursor, identity)
                if current is None:
                    return DurableStreamAppendResult(status="conflict")
                if not _identity_matches(current.identity, identity):
                    return DurableStreamAppendResult(status="conflict", head=current)
                if generation != current.generation:
                    return DurableStreamAppendResult(status="old_generation", head=current)
                existing = self._chunk_at(cursor, identity, generation, start_offset)
                if existing is not None:
                    if existing.end_offset != end_offset or existing.sha256 != sha256:
                        return DurableStreamAppendResult(status="conflict", head=current)
                    expected = BlobStat(
                        sha256=existing.sha256,
                        size_bytes=existing.size_bytes,
                        locator=existing.locator,
                    )
                    if not self._stat_matches(self.object_store.stat(existing.sha256), expected):
                        raise PostgresDurableStreamCorrupt(
                            "idempotent stream chunk no longer has its checked object"
                        )
                    return DurableStreamAppendResult(
                        status="already_committed",
                        head=current,
                        chunk=existing,
                    )
                if current.state == "sealed":
                    return DurableStreamAppendResult(status="sealed", head=current)
                if self._run_is_terminal(cursor, identity.run_id):
                    return DurableStreamAppendResult(status="run_terminal", head=current)
                if start_offset != current.cursor_bytes:
                    return DurableStreamAppendResult(
                        status="gap" if start_offset > current.cursor_bytes else "conflict",
                        head=current,
                    )
                # One append is a bounded (<= MAX_STREAM_CHUNK_BYTES) mutation. Keep the run
                # authority and stream head locked through its physical put so terminal, reset,
                # and takeover linearize strictly before or after the object+metadata commit.
                # Seal releases these locks before its unbounded multi-chunk checked-read pass.
                stat = self._put_checked(data)
                self._record_object(cursor, identity.run_id, stat)
                sequence = current.next_chunk_sequence
                cursor.execute(
                    sql.SQL(
                        "INSERT INTO {} "
                        "(run_id, stream_id, channel, generation, chunk_sequence, "
                        "start_offset, end_offset, sha256) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                    ).format(self._table("durable_stream_chunk")),
                    (
                        identity.run_id,
                        identity.stream_id,
                        identity.channel,
                        generation,
                        sequence,
                        start_offset,
                        end_offset,
                        stat.sha256,
                    ),
                )
                cursor.execute(
                    sql.SQL(
                        "UPDATE {} SET cursor_bytes = %s, next_chunk_sequence = %s, "
                        "updated_at = pg_catalog.clock_timestamp() "
                        "WHERE run_id = %s AND stream_id = %s AND channel = %s"
                    ).format(self._table("durable_stream_head")),
                    (
                        end_offset,
                        sequence + 1,
                        identity.run_id,
                        identity.stream_id,
                        identity.channel,
                    ),
                )
                head = DurableStreamHead(
                    identity=identity,
                    generation=generation,
                    cursor_bytes=end_offset,
                    next_chunk_sequence=sequence + 1,
                    state="open",
                )
                chunk = DurableStreamChunk(
                    identity=identity,
                    generation=generation,
                    sequence=sequence,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    sha256=stat.sha256,
                    locator=stat.locator,
                )
                return DurableStreamAppendResult(status="committed", head=head, chunk=chunk)

    def _calculate_generation_digest(
        self,
        identity: DurableStreamIdentity,
        generation: int,
        *,
        expected_cursor: int,
        expected_next_sequence: int,
    ) -> str:
        from psycopg import sql

        expected_offset = 0
        expected_sequence = 1
        digest = hashlib.sha256()
        while expected_offset < expected_cursor:
            with self.database.transaction() as connection:
                with self.database.cursor(connection) as cursor:
                    cursor.execute(  # type: ignore[attr-defined]
                        sql.SQL(
                            "SELECT chunk.chunk_sequence, chunk.start_offset, chunk.end_offset, "
                            "chunk.sha256, blob.size_bytes, blob.locator, blob.state "
                            "FROM {} AS chunk JOIN {} AS blob ON blob.sha256 = chunk.sha256 "
                            "WHERE chunk.run_id = %s AND chunk.stream_id = %s "
                            "AND chunk.channel = %s AND chunk.generation = %s "
                            "AND chunk.chunk_sequence >= %s "
                            "ORDER BY chunk.chunk_sequence LIMIT 100"
                        ).format(
                            self._table("durable_stream_chunk"),
                            self._table("object_blob"),
                        ),
                        (
                            identity.run_id,
                            identity.stream_id,
                            identity.channel,
                            generation,
                            expected_sequence,
                        ),
                    )
                    rows = tuple(cursor.fetchall())  # type: ignore[attr-defined]
            if not rows:
                raise PostgresDurableStreamCorrupt(
                    "durable stream head cursor has no complete chunk history"
                )
            for row in rows:
                chunk = DurableStreamChunk(
                    identity=identity,
                    generation=generation,
                    sequence=int(row[0]),
                    start_offset=int(row[1]),
                    end_offset=int(row[2]),
                    sha256=str(row[3]),
                    locator=str(row[5]),
                )
                if (
                    chunk.sequence != expected_sequence
                    or chunk.start_offset != expected_offset
                    or chunk.end_offset > expected_cursor
                    or str(row[6]) != "available"
                    or int(row[4]) != chunk.size_bytes
                ):
                    raise PostgresDurableStreamCorrupt(
                        "durable stream chunk metadata is not contiguous and available"
                    )
                digest.update(self._checked_chunk_bytes(chunk))
                expected_sequence += 1
                expected_offset = chunk.end_offset
                if expected_offset == expected_cursor:
                    break
        if expected_sequence != expected_next_sequence:
            raise PostgresDurableStreamCorrupt(
                "durable stream head sequence disagrees with committed chunks"
            )
        return digest.hexdigest() if expected_cursor else _EMPTY_SHA256

    def _checked_chunk_bytes(self, chunk: DurableStreamChunk) -> bytes:
        current = self.object_store.stat(chunk.sha256)
        expected = BlobStat(
            sha256=chunk.sha256,
            size_bytes=chunk.size_bytes,
            locator=chunk.locator,
        )
        if current is None:
            raise BlobNotFound("durable stream chunk object is missing")
        if not self._stat_matches(current, expected):
            raise PostgresDurableStreamCorrupt(
                "durable stream chunk metadata disagrees with ObjectStore"
            )
        data = self.object_store.get_checked(chunk.sha256)
        if len(data) != chunk.size_bytes or hashlib.sha256(data).hexdigest() != chunk.sha256:
            raise PostgresDurableStreamCorrupt(
                "durable stream checked bytes disagree with chunk metadata"
            )
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PostgresDurableStreamCorrupt(
                "durable stream chunk bytes are not complete UTF-8"
            ) from exc
        return data

    def seal(
        self,
        identity: DurableStreamIdentity,
        *,
        generation: int,
        final_size_bytes: int,
        final_sha256: str,
        writer_token: WriterToken,
    ) -> DurableStreamSealResult:
        self._require_ready()
        generation = _portable_positive(generation, "durable stream seal generation")
        final_size_bytes = _portable_non_negative(
            final_size_bytes,
            "durable stream final_size_bytes",
        )
        if (
            type(final_sha256) is not str
            or len(final_sha256) != 64
            or any(character not in "0123456789abcdef" for character in final_sha256)
        ):
            raise ValueError("durable stream final_sha256 must be lowercase SHA-256")
        checked = self._validate_identity_and_token(identity, writer_token)
        if checked is None:
            return DurableStreamSealResult(status="fenced")
        identity, writer_token = checked
        from psycopg import sql

        # Validate authority and capture a bounded metadata snapshot first. Physical ObjectStore
        # reads happen after releasing both the authority and stream-head locks, so a large stream
        # seal cannot starve lease renewal. The final transaction validates every captured head
        # coordinate again before publication.
        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                if not self._current_writer_locked(cursor, writer_token):
                    return DurableStreamSealResult(status="fenced")
                snapshot = self._head_locked(cursor, identity)
                if snapshot is None:
                    return DurableStreamSealResult(status="conflict")
                if not _identity_matches(snapshot.identity, identity):
                    return DurableStreamSealResult(status="conflict", head=snapshot)
                if generation != snapshot.generation:
                    return DurableStreamSealResult(status="old_generation", head=snapshot)
                if snapshot.state == "sealed":
                    return DurableStreamSealResult(
                        status=(
                            "already_sealed"
                            if snapshot.cursor_bytes == final_size_bytes
                            and snapshot.final_sha256 == final_sha256
                            else "conflict"
                        ),
                        head=snapshot,
                    )
                if self._run_is_terminal(cursor, identity.run_id):
                    return DurableStreamSealResult(status="run_terminal", head=snapshot)
                if snapshot.cursor_bytes != final_size_bytes:
                    return DurableStreamSealResult(status="gap", head=snapshot)
        calculated = self._calculate_generation_digest(
            identity,
            generation,
            expected_cursor=snapshot.cursor_bytes,
            expected_next_sequence=snapshot.next_chunk_sequence,
        )
        if calculated != final_sha256:
            return DurableStreamSealResult(status="conflict", head=snapshot)

        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                if not self._current_writer_locked(cursor, writer_token):
                    return DurableStreamSealResult(status="fenced")
                current = self._head_locked(cursor, identity)
                if current is None:
                    return DurableStreamSealResult(status="conflict")
                if not _identity_matches(current.identity, identity):
                    return DurableStreamSealResult(status="conflict", head=current)
                if generation != current.generation:
                    return DurableStreamSealResult(status="old_generation", head=current)
                if current.state == "sealed":
                    return DurableStreamSealResult(
                        status=(
                            "already_sealed"
                            if current.cursor_bytes == final_size_bytes
                            and current.final_sha256 == final_sha256
                            else "conflict"
                        ),
                        head=current,
                    )
                if self._run_is_terminal(cursor, identity.run_id):
                    return DurableStreamSealResult(status="run_terminal", head=current)
                if current.cursor_bytes != final_size_bytes:
                    return DurableStreamSealResult(status="gap", head=current)
                if (
                    current.cursor_bytes != snapshot.cursor_bytes
                    or current.next_chunk_sequence != snapshot.next_chunk_sequence
                ):
                    return DurableStreamSealResult(status="gap", head=current)
                cursor.execute(
                    sql.SQL(
                        "UPDATE {} SET state = 'sealed', final_sha256 = %s, "
                        "sealed_at = pg_catalog.clock_timestamp(), "
                        "updated_at = pg_catalog.clock_timestamp() "
                        "WHERE run_id = %s AND stream_id = %s AND channel = %s"
                    ).format(self._table("durable_stream_head")),
                    (
                        final_sha256,
                        identity.run_id,
                        identity.stream_id,
                        identity.channel,
                    ),
                )
                head = DurableStreamHead(
                    identity=identity,
                    generation=generation,
                    cursor_bytes=current.cursor_bytes,
                    next_chunk_sequence=current.next_chunk_sequence,
                    state="sealed",
                    final_sha256=final_sha256,
                )
                return DurableStreamSealResult(status="sealed", head=head)

    def read_after(
        self,
        identity: DurableStreamIdentity,
        *,
        generation: int,
        cursor: int,
        limit: int = 100,
    ) -> DurableStreamReadResult:
        self._require_ready()
        if not isinstance(identity, DurableStreamIdentity):
            raise TypeError("durable stream read requires DurableStreamIdentity")
        generation = _portable_positive(generation, "durable stream read generation")
        cursor = _portable_non_negative(cursor, "durable stream read cursor")
        if type(limit) is not int or not (1 <= limit <= MAX_STREAM_READ_CHUNKS):
            raise ValueError(
                "durable stream read limit must be between 1 and MAX_STREAM_READ_CHUNKS"
            )
        from psycopg import sql

        metadata: tuple[DurableStreamChunk, ...] = ()
        with self.database.transaction() as connection:
            with self.database.cursor(connection) as db_cursor:
                head = self._head_locked(db_cursor, identity, shared=True)
                if head is None or not _identity_matches(head.identity, identity):
                    return DurableStreamReadResult(
                        status="not_found",
                        requested_generation=generation,
                        requested_cursor=cursor,
                    )
                if generation < head.generation:
                    return DurableStreamReadResult(
                        status="reset",
                        requested_generation=generation,
                        requested_cursor=cursor,
                        head=head,
                    )
                if generation > head.generation or cursor > head.cursor_bytes:
                    return DurableStreamReadResult(
                        status="gap",
                        requested_generation=generation,
                        requested_cursor=cursor,
                        head=head,
                    )
                if cursor != 0:
                    db_cursor.execute(
                        sql.SQL(
                            "SELECT 1 FROM {} WHERE run_id = %s AND stream_id = %s "
                            "AND channel = %s AND generation = %s AND end_offset = %s"
                        ).format(self._table("durable_stream_chunk")),
                        (
                            identity.run_id,
                            identity.stream_id,
                            identity.channel,
                            generation,
                            cursor,
                        ),
                    )
                    if db_cursor.fetchone() is None:
                        return DurableStreamReadResult(
                            status="gap",
                            requested_generation=generation,
                            requested_cursor=cursor,
                            head=head,
                        )
                db_cursor.execute(
                    sql.SQL(
                        "SELECT chunk.chunk_sequence, chunk.start_offset, chunk.end_offset, "
                        "chunk.sha256, blob.size_bytes, blob.locator, blob.state "
                        "FROM {} AS chunk JOIN {} AS blob ON blob.sha256 = chunk.sha256 "
                        "WHERE chunk.run_id = %s AND chunk.stream_id = %s "
                        "AND chunk.channel = %s AND chunk.generation = %s "
                        "AND chunk.start_offset >= %s ORDER BY chunk.start_offset LIMIT %s"
                    ).format(
                        self._table("durable_stream_chunk"),
                        self._table("object_blob"),
                    ),
                    (
                        identity.run_id,
                        identity.stream_id,
                        identity.channel,
                        generation,
                        cursor,
                        limit,
                    ),
                )
                values: list[DurableStreamChunk] = []
                expected_offset = cursor
                for row in db_cursor.fetchall():
                    chunk = DurableStreamChunk(
                        identity=identity,
                        generation=generation,
                        sequence=int(row[0]),
                        start_offset=int(row[1]),
                        end_offset=int(row[2]),
                        sha256=str(row[3]),
                        locator=str(row[5]),
                    )
                    if (
                        chunk.start_offset != expected_offset
                        or int(row[4]) != chunk.size_bytes
                        or str(row[6]) != "available"
                    ):
                        raise PostgresDurableStreamCorrupt(
                            "durable stream replay metadata contains a gap"
                        )
                    values.append(chunk)
                    expected_offset = chunk.end_offset
                if cursor < head.cursor_bytes and not values:
                    raise PostgresDurableStreamCorrupt(
                        "durable stream replay cursor has no following chunk"
                    )
                if expected_offset < head.cursor_bytes and len(values) < limit:
                    raise PostgresDurableStreamCorrupt(
                        "durable stream replay ended before the canonical head"
                    )
                metadata = tuple(values)
        chunks = tuple(
            DurableStreamReadChunk(chunk=chunk, data=self._checked_chunk_bytes(chunk))
            for chunk in metadata
        )
        return DurableStreamReadResult(
            status="ok",
            requested_generation=generation,
            requested_cursor=cursor,
            head=head,
            chunks=chunks,
        )


__all__ = ["PostgresDurableStreamCorrupt", "PostgresObjectStoreDurableStreamStore"]
