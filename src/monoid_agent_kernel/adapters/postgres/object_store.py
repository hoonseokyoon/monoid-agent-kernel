"""PostgreSQL run association for external content-addressed objects and safe GC."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import datetime, timedelta

from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.core.checkpoint import RunCheckpoint
from monoid_agent_kernel.core.model_invocation import DurableModelInvocation
from monoid_agent_kernel.core.model_io import is_recorded_digest
from monoid_agent_kernel.hosting import CommitResult, WriterToken
from monoid_agent_kernel.hosting.blobs import (
    BlobCorrupt,
    BlobNotFound,
    BlobStat,
    BlobTooLarge,
    ContentAddressedBlobStore,
)
from monoid_agent_kernel.hosting.object_store_admin import (
    ObjectGcCandidate,
    ObjectGcPlan,
    ObjectGcReceipt,
    ObjectStoreAdmin,
)

from .migrations import MigrationStatus, PostgresMigrations, PostgresSchemaIncompatible
from .pool import PostgresDatabase
from .sink import PostgresFencedRunSink


class PostgresObjectAssociationCorrupt(BlobCorrupt):
    """PostgreSQL external-object metadata disagrees with checked physical storage."""


class _PreparedObjectBlobs(dict[str, bytes]):
    def __init__(self, values: Mapping[str, bytes], stats: Mapping[str, BlobStat]) -> None:
        super().__init__(values)
        self.stats = dict(stats)


class _PreparedInlineBlobs(dict[str, bytes]):
    """Validated fallback bytes selected for the PostgreSQL bytea backend."""


def _supports(value: object, names: tuple[str, ...]) -> bool:
    return all(callable(getattr(value, name, None)) for name in names)


def _validate_external_blobs(blobs: object) -> dict[str, bytes] | None:
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
            or hashlib.sha256(value).hexdigest() != key
        ):
            return None
    return copied


def _lock_digest(cursor: object, schema: str, sha256: str) -> None:
    cursor.execute(  # type: ignore[attr-defined]
        "SELECT pg_catalog.pg_advisory_xact_lock("
        "pg_catalog.hashtextextended(%s, 0))",
        (f"monoid-agent-kernel:object:{schema}:{sha256}",),
    )


def _gc_plan_payload(
    *,
    created_at: datetime,
    grace_before: datetime,
    candidates: tuple[ObjectGcCandidate, ...],
    next_token: str | None,
) -> dict[str, object]:
    return {
        "created_at": created_at.isoformat(),
        "grace_before": grace_before.isoformat(),
        "next_token": next_token,
        "candidates": [
            {
                "sha256": candidate.sha256,
                "size_bytes": candidate.size_bytes,
                "locator": candidate.locator,
                "last_modified": candidate.last_modified.isoformat(),
                "delete_token_sha256": hashlib.sha256(
                    candidate.delete_token.encode("ascii")
                ).hexdigest(),
                "generation": candidate.generation,
            }
            for candidate in candidates
        ],
    }


class PostgresObjectStoreFencedRunSink(PostgresFencedRunSink):
    """Full fenced run sink with object-first bytes and PostgreSQL run authorization."""

    def __init__(
        self,
        database: PostgresDatabase,
        object_store: ContentAddressedBlobStore,
    ) -> None:
        super().__init__(database)
        if not _supports(object_store, ("put_if_absent", "stat", "get_checked")):
            raise TypeError("external fenced sink object_store must satisfy ContentAddressedBlobStore")
        self.object_store = object_store

    def _prepare_blobs(
        self,
        blobs: object,
    ) -> _PreparedObjectBlobs | _PreparedInlineBlobs | None:
        checked = _validate_external_blobs(blobs)
        if checked is None:
            return None
        stats: dict[str, BlobStat] = {}
        try:
            for sha256 in sorted(checked):
                result = self.object_store.put_if_absent(sha256, checked[sha256])
                stat = result.stat
                if (
                    stat.sha256 != sha256
                    or stat.size_bytes != len(checked[sha256])
                    or not stat.locator
                ):
                    raise BlobCorrupt("object-store put result disagrees with caller bytes")
                stats[sha256] = stat
        except BlobTooLarge:
            inline = super()._validated_blobs(checked)
            return _PreparedInlineBlobs(inline) if inline is not None else None
        return _PreparedObjectBlobs(checked, stats)

    def _validated_blobs(self, blobs: object) -> dict[str, bytes] | None:
        if not isinstance(blobs, (_PreparedObjectBlobs, _PreparedInlineBlobs)):
            return None
        return blobs

    def _submitted_blobs_preserve_backing(
        self,
        cursor: object,
        blobs: Mapping[str, bytes],
    ) -> bool:
        if isinstance(blobs, _PreparedInlineBlobs):
            return super()._submitted_blobs_preserve_backing(cursor, blobs)
        return isinstance(blobs, _PreparedObjectBlobs) and set(blobs) == set(blobs.stats)

    def _object_metadata(
        self,
        cursor: object,
        sha256: str,
    ) -> tuple[int, str, int, str] | None:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "SELECT size_bytes, locator, generation, state FROM {} WHERE sha256 = %s"
            ).format(self._table("object_blob")),
            (sha256,),
        )
        row = cursor.fetchone()  # type: ignore[attr-defined]
        if row is None:
            return None
        return int(row[0]), str(row[1]), int(row[2]), str(row[3])

    @staticmethod
    def _stat_matches(stat: BlobStat | None, expected: BlobStat) -> bool:
        return stat is not None and (
            stat.sha256,
            stat.size_bytes,
            stat.locator,
        ) == (
            expected.sha256,
            expected.size_bytes,
            expected.locator,
        )

    def _associated_blob_is_valid(self, cursor: object, run_id: str, sha256: str) -> bool:
        if not is_recorded_digest(sha256):
            return False
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "SELECT blob.size_bytes, blob.locator, blob.state FROM {} AS association "
                "JOIN {} AS blob ON blob.sha256 = association.sha256 "
                "WHERE association.run_id = %s AND association.sha256 = %s"
            ).format(self._table("run_object_blob"), self._table("object_blob")),
            (run_id, sha256),
        )
        row = cursor.fetchone()  # type: ignore[attr-defined]
        if row is None or str(row[2]) != "available":
            return False
        expected = BlobStat(
            sha256=sha256,
            size_bytes=int(row[0]),
            locator=str(row[1]),
        )
        return self._stat_matches(self.object_store.stat(sha256), expected)

    def _has_object_association(self, cursor: object, run_id: str, sha256: str) -> bool:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL("SELECT 1 FROM {} WHERE run_id = %s AND sha256 = %s").format(
                self._table("run_object_blob")
            ),
            (run_id, sha256),
        )
        return cursor.fetchone() is not None  # type: ignore[attr-defined]

    def _reference_is_valid(self, cursor: object, run_id: str, sha256: str) -> bool:
        if self._has_object_association(cursor, run_id, sha256):
            return self._associated_blob_is_valid(cursor, run_id, sha256)
        return super()._associated_blob_is_valid(cursor, run_id, sha256)

    def _references_resolve(
        self,
        cursor: object,
        run_id: str,
        references: set[str],
        blobs: Mapping[str, bytes],
    ) -> bool:
        if isinstance(blobs, _PreparedInlineBlobs):
            return all(
                sha256 in blobs or self._reference_is_valid(cursor, run_id, sha256)
                for sha256 in references
            )
        if not isinstance(blobs, _PreparedObjectBlobs):
            return False
        for sha256 in sorted(references | set(blobs)):
            _lock_digest(cursor, self.database.config.schema, sha256)
        for sha256 in sorted(blobs):
            prepared = blobs.stats[sha256]
            if not self._stat_matches(self.object_store.stat(sha256), prepared):
                return False
            metadata = self._object_metadata(cursor, sha256)
            if metadata is not None and metadata[3] == "available" and (
                metadata[0] != prepared.size_bytes or metadata[1] != prepared.locator
            ):
                return False
        return all(
            sha256 in blobs or self._reference_is_valid(cursor, run_id, sha256)
            for sha256 in references
        )

    def _persist_blobs(
        self,
        cursor: object,
        run_id: str,
        blobs: Mapping[str, bytes],
    ) -> None:
        if isinstance(blobs, _PreparedInlineBlobs):
            super()._persist_blobs(cursor, run_id, blobs)
            return
        if not isinstance(blobs, _PreparedObjectBlobs):
            raise RuntimeError("external object metadata requires prepared blob stats")
        from psycopg import sql

        for sha256 in sorted(blobs):
            stat = blobs.stats[sha256]
            metadata = self._object_metadata(cursor, sha256)
            if metadata is None:
                cursor.execute(  # type: ignore[attr-defined]
                    sql.SQL(
                        "INSERT INTO {} (sha256, size_bytes, locator, generation, state) "
                        "VALUES (%s, %s, %s, 1, 'available')"
                    ).format(self._table("object_blob")),
                    (sha256, stat.size_bytes, stat.locator),
                )
            elif metadata[3] == "deleted":
                cursor.execute(  # type: ignore[attr-defined]
                    sql.SQL(
                        "UPDATE {} SET size_bytes = %s, locator = %s, "
                        "generation = generation + 1, state = 'available', "
                        "verified_at = pg_catalog.clock_timestamp(), deleted_at = NULL "
                        "WHERE sha256 = %s"
                    ).format(self._table("object_blob")),
                    (stat.size_bytes, stat.locator, sha256),
                )
            elif metadata[0] == stat.size_bytes and metadata[1] == stat.locator:
                cursor.execute(  # type: ignore[attr-defined]
                    sql.SQL(
                        "UPDATE {} SET verified_at = pg_catalog.clock_timestamp() "
                        "WHERE sha256 = %s"
                    ).format(self._table("object_blob")),
                    (sha256,),
                )
            else:
                raise PostgresObjectAssociationCorrupt(
                    "PostgreSQL object metadata conflicts with checked physical storage"
                )
            cursor.execute(  # type: ignore[attr-defined]
                sql.SQL(
                    "INSERT INTO {} (run_id, sha256) VALUES (%s, %s) "
                    "ON CONFLICT (run_id, sha256) DO NOTHING"
                ).format(self._table("run_object_blob")),
                (run_id, sha256),
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
        prepared = self._prepare_blobs(blobs)
        if prepared is None:
            return super().commit_checkpoint(checkpoint, blobs, writer_token=writer_token)
        return super().commit_checkpoint(checkpoint, prepared, writer_token=writer_token)

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
            return CommitResult(
                status="fenced",
                sequence=(
                    invocation.revision
                    if isinstance(invocation, DurableModelInvocation)
                    else None
                ),
            )
        prepared = self._prepare_blobs(blobs)
        if prepared is None:
            return super().commit_invocation(
                invocation,
                blobs,
                writer_token=writer_token,
                stage_evidence=stage_evidence,
            )
        return super().commit_invocation(
            invocation,
            prepared,
            writer_token=writer_token,
            stage_evidence=stage_evidence,
        )

    def _read_blob(self, run_id: str, sha256: str) -> bytes:
        self._require_ready()
        from monoid_agent_kernel.core.safe_evidence import is_safe_opaque_id

        if not is_safe_opaque_id(run_id) or not is_recorded_digest(sha256):
            raise KeyError(sha256)
        from psycopg import sql

        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                cursor.execute(
                    sql.SQL(
                        "SELECT blob.size_bytes, blob.locator, blob.state FROM {} AS association "
                        "JOIN {} AS blob ON blob.sha256 = association.sha256 "
                        "WHERE association.run_id = %s AND association.sha256 = %s"
                    ).format(self._table("run_object_blob"), self._table("object_blob")),
                    (run_id, sha256),
                )
                row = cursor.fetchone()
        if row is None:
            return super()._read_blob(run_id, sha256)
        if str(row[2]) != "available":
            raise PostgresObjectAssociationCorrupt(
                "run association references an unavailable physical object"
            )
        expected = BlobStat(sha256=sha256, size_bytes=int(row[0]), locator=str(row[1]))
        current = self.object_store.stat(sha256)
        if current is None:
            raise BlobNotFound("run-associated physical object is missing")
        if not self._stat_matches(current, expected):
            raise PostgresObjectAssociationCorrupt(
                "PostgreSQL object metadata disagrees with physical storage"
            )
        data = self.object_store.get_checked(sha256)
        if len(data) != expected.size_bytes:
            raise PostgresObjectAssociationCorrupt(
                "checked physical bytes disagree with PostgreSQL object metadata"
            )
        return data


class PostgresObjectGarbageCollector:
    """Bounded dry-run planning and digest-locked explicit orphan deletion."""

    def __init__(self, database: PostgresDatabase, object_admin: ObjectStoreAdmin) -> None:
        if not isinstance(database, PostgresDatabase):
            raise TypeError("PostgreSQL object GC database must be PostgresDatabase")
        if not _supports(
            object_admin,
            (
                "inventory_page",
                "delete_if_match",
                "incomplete_multipart_page",
                "abort_incomplete_multipart",
            ),
        ):
            raise TypeError("PostgreSQL object GC admin must satisfy ObjectStoreAdmin")
        self.database = database
        self.object_admin = object_admin
        self._ready = False

    def _table(self, name: str) -> object:
        from psycopg import sql

        return sql.Identifier(self.database.config.schema, name)

    def check_ready(self) -> MigrationStatus:
        self._ready = False
        status = PostgresMigrations(self.database).require_writer_compatible()
        if not status.reader_compatible:
            raise PostgresSchemaIncompatible(
                "PostgreSQL object GC requires reader and writer compatibility"
            )
        self._ready = True
        return status

    def _require_ready(self) -> None:
        if not self._ready:
            raise PostgresSchemaIncompatible(
                "PostgreSQL object GC requires a successful check_ready()"
            )

    def _database_now(self) -> datetime:
        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                cursor.execute("SELECT pg_catalog.clock_timestamp()")
                return cursor.fetchone()[0]

    def plan(
        self,
        *,
        grace_period: timedelta,
        continuation_token: str | None = None,
        limit: int = 1000,
    ) -> ObjectGcPlan:
        self._require_ready()
        if not isinstance(grace_period, timedelta) or not (
            timedelta(0) <= grace_period <= timedelta(days=3650)
        ):
            raise ValueError("object GC grace_period must be between zero and 3650 days")
        created_at = self._database_now()
        grace_before = created_at - grace_period
        page = self.object_admin.inventory_page(
            continuation_token=continuation_token,
            limit=limit,
        )
        candidates: list[ObjectGcCandidate] = []
        from psycopg import sql

        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                for entry in sorted(page.entries, key=lambda value: value.sha256):
                    if entry.last_modified > grace_before:
                        continue
                    cursor.execute(
                        sql.SQL(
                            "SELECT blob.size_bytes, blob.locator, blob.generation, blob.state, "
                            "EXISTS (SELECT 1 FROM {} AS association "
                            "WHERE association.sha256 = blob.sha256) "
                            "FROM {} AS blob WHERE blob.sha256 = %s"
                        ).format(self._table("run_object_blob"), self._table("object_blob")),
                        (entry.sha256,),
                    )
                    row = cursor.fetchone()
                    if row is not None:
                        if bool(row[4]):
                            continue
                        if str(row[3]) == "available" and (
                            int(row[0]) != entry.size_bytes or str(row[1]) != entry.locator
                        ):
                            raise PostgresObjectAssociationCorrupt(
                                "GC inventory disagrees with available PostgreSQL object metadata"
                            )
                        generation = int(row[2])
                    else:
                        generation = 0
                    candidates.append(
                        ObjectGcCandidate(
                            sha256=entry.sha256,
                            size_bytes=entry.size_bytes,
                            locator=entry.locator,
                            last_modified=entry.last_modified,
                            delete_token=entry.delete_token,
                            generation=generation,
                        )
                    )
        frozen_candidates = tuple(candidates)
        identity = _gc_plan_payload(
            created_at=created_at,
            grace_before=grace_before,
            candidates=frozen_candidates,
            next_token=page.next_token,
        )
        return ObjectGcPlan(
            plan_id=canonical_sha256(identity),
            created_at=created_at,
            grace_before=grace_before,
            candidates=frozen_candidates,
            next_token=page.next_token,
        )

    def _stored_receipt(
        self,
        cursor: object,
        plan_id: str,
        sha256: str,
    ) -> ObjectGcReceipt | None:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "SELECT candidate_generation, observed_generation, status, recorded_at "
                "FROM {} WHERE plan_id = %s AND sha256 = %s"
            ).format(self._table("object_gc_receipt")),
            (plan_id, sha256),
        )
        row = cursor.fetchone()  # type: ignore[attr-defined]
        if row is None:
            return None
        return ObjectGcReceipt(
            plan_id=plan_id,
            sha256=sha256,
            candidate_generation=int(row[0]),
            observed_generation=int(row[1]),
            status=str(row[2]),  # type: ignore[arg-type]
            recorded_at=row[3],
        )

    def _record_receipt(
        self,
        cursor: object,
        plan_id: str,
        candidate: ObjectGcCandidate,
        *,
        observed_generation: int,
        status: str,
    ) -> ObjectGcReceipt:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "INSERT INTO {} (plan_id, sha256, candidate_generation, observed_generation, "
                "status, delete_token_sha256) VALUES (%s, %s, %s, %s, %s, %s) "
                "RETURNING recorded_at"
            ).format(self._table("object_gc_receipt")),
            (
                plan_id,
                candidate.sha256,
                candidate.generation,
                observed_generation,
                status,
                hashlib.sha256(candidate.delete_token.encode("ascii")).hexdigest(),
            ),
        )
        recorded_at = cursor.fetchone()[0]  # type: ignore[attr-defined]
        return ObjectGcReceipt(
            plan_id=plan_id,
            sha256=candidate.sha256,
            candidate_generation=candidate.generation,
            observed_generation=observed_generation,
            status=status,  # type: ignore[arg-type]
            recorded_at=recorded_at,
        )

    def _apply_candidate(
        self,
        plan: ObjectGcPlan,
        candidate: ObjectGcCandidate,
    ) -> ObjectGcReceipt:
        from psycopg import sql

        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                _lock_digest(cursor, self.database.config.schema, candidate.sha256)
                stored = self._stored_receipt(cursor, plan.plan_id, candidate.sha256)
                if stored is not None:
                    return stored
                cursor.execute(
                    sql.SQL(
                        "SELECT generation, state FROM {} WHERE sha256 = %s"
                    ).format(self._table("object_blob")),
                    (candidate.sha256,),
                )
                metadata = cursor.fetchone()
                observed_generation = int(metadata[0]) if metadata is not None else 0
                cursor.execute(
                    sql.SQL(
                        "SELECT EXISTS (SELECT 1 FROM {} WHERE sha256 = %s)"
                    ).format(self._table("run_object_blob")),
                    (candidate.sha256,),
                )
                if bool(cursor.fetchone()[0]):
                    return self._record_receipt(
                        cursor,
                        plan.plan_id,
                        candidate,
                        observed_generation=observed_generation,
                        status="skipped_associated",
                    )
                if observed_generation != candidate.generation:
                    return self._record_receipt(
                        cursor,
                        plan.plan_id,
                        candidate,
                        observed_generation=observed_generation,
                        status="skipped_generation",
                    )
                deleted = self.object_admin.delete_if_match(
                    candidate.sha256,
                    candidate.delete_token,
                )
                if deleted.status in {"deleted", "already_missing"}:
                    if metadata is None:
                        cursor.execute(
                            sql.SQL(
                                "INSERT INTO {} (sha256, size_bytes, locator, generation, state, "
                                "deleted_at) VALUES (%s, %s, %s, 1, 'deleted', "
                                "pg_catalog.clock_timestamp())"
                            ).format(self._table("object_blob")),
                            (candidate.sha256, candidate.size_bytes, candidate.locator),
                        )
                    else:
                        cursor.execute(
                            sql.SQL(
                                "UPDATE {} SET generation = generation + 1, state = 'deleted', "
                                "deleted_at = pg_catalog.clock_timestamp() WHERE sha256 = %s"
                            ).format(self._table("object_blob")),
                            (candidate.sha256,),
                        )
                return self._record_receipt(
                    cursor,
                    plan.plan_id,
                    candidate,
                    observed_generation=observed_generation,
                    status=deleted.status,
                )

    def apply(self, plan: ObjectGcPlan) -> tuple[ObjectGcReceipt, ...]:
        self._require_ready()
        if not isinstance(plan, ObjectGcPlan):
            raise TypeError("object GC apply requires ObjectGcPlan")
        if (
            tuple(sorted(plan.candidates, key=lambda candidate: candidate.sha256))
            != plan.candidates
            or len({candidate.sha256 for candidate in plan.candidates})
            != len(plan.candidates)
            or any(candidate.last_modified > plan.grace_before for candidate in plan.candidates)
            or canonical_sha256(
                _gc_plan_payload(
                    created_at=plan.created_at,
                    grace_before=plan.grace_before,
                    candidates=plan.candidates,
                    next_token=plan.next_token,
                )
            )
            != plan.plan_id
        ):
            raise ValueError("object GC plan identity or candidate ordering is invalid")
        return tuple(self._apply_candidate(plan, candidate) for candidate in plan.candidates)


__all__ = [
    "PostgresObjectAssociationCorrupt",
    "PostgresObjectStoreFencedRunSink",
    "PostgresObjectGarbageCollector",
]
