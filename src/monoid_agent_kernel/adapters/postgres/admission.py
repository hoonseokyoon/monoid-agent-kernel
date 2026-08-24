"""PostgreSQL command admission, ordered dispatch outbox, and activation binding."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from monoid_agent_kernel.adapters.postgres.authority import (
    _ELAPSED_TTL_INTERVAL,
    PostgresWriterAuthorityStore,
)
from monoid_agent_kernel.adapters.postgres.migrations import (
    MigrationStatus,
    PostgresMigrations,
    PostgresSchemaIncompatible,
)
from monoid_agent_kernel.adapters.postgres.pool import PostgresDatabase
from monoid_agent_kernel.adapters.postgres.sink import PostgresFencedRunSink
from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.core.checkpoint import RunCheckpoint, decode_checkpoint
from monoid_agent_kernel.core.safe_evidence import (
    is_safe_opaque_id,
    is_safe_taxonomy_code,
)
from monoid_agent_kernel.errors import NativeAgentError
from monoid_agent_kernel.hosting.activation import ActivationCommand, ActivationReceipt
from monoid_agent_kernel.hosting.admission import (
    ActivationBindingConflict,
    AdmittedCommand,
    AdmissionConflict,
    AdmissionReceipt,
    AdmissionRequest,
    AdmissionRunTerminal,
    AdmissionRunUnavailable,
    DispatchClaim,
    DispatchClaimLost,
    DispatchResult,
    DispatchToken,
)
from monoid_agent_kernel.hosting.contracts import WriterToken


_POSTGRES_BIGINT_MAX = (1 << 63) - 1
_DELIVERY_STATES = frozenset({"pending", "leased", "delivered", "dead_letter"})


class PostgresAdmissionCorrupt(NativeAgentError):
    error_code = "admission_corrupt"


def _is_ambiguous_database_error(exc: Exception) -> bool:
    try:
        from psycopg import InterfaceError, OperationalError
    except ImportError:  # pragma: no cover - database operations already require psycopg
        return False
    return isinstance(exc, (InterfaceError, OperationalError))


def _is_unique_violation(exc: Exception) -> bool:
    try:
        from psycopg.errors import UniqueViolation
    except ImportError:  # pragma: no cover - database operations already require psycopg
        return False
    return isinstance(exc, UniqueViolation)


def _content_digest(payload: dict[str, object]) -> str:
    return canonical_sha256(payload)


def _duration_microseconds(seconds: float) -> int:
    """Encode an elapsed duration without PostgreSQL calendar-day normalization."""

    return math.ceil(seconds * 1_000_000)


@dataclass(frozen=True)
class _StoredAdmission:
    command: AdmittedCommand
    activation: ActivationCommand | None
    delivery_state: Literal["pending", "leased", "delivered", "dead_letter"]
    attempt_count: int
    dispatch_ref: str
    last_error_code: str
    claim_owner: str
    claim_id: str
    claim_generation: int
    lease_active: bool
    retry_delay_microseconds: int


class PostgresCommandAdmissionStore:
    """Production admission store and dispatch outbox over one PostgreSQL schema."""

    def __init__(self, database: PostgresDatabase) -> None:
        if not isinstance(database, PostgresDatabase):
            raise TypeError("PostgresCommandAdmissionStore database must be PostgresDatabase")
        self.database = database
        self._authority = PostgresWriterAuthorityStore(database)
        self._sink = PostgresFencedRunSink(database)
        self._ready = False

    def check_ready(self) -> MigrationStatus:
        self._ready = False
        status = PostgresMigrations(self.database).require_writer_compatible()
        if not status.reader_compatible:
            raise PostgresSchemaIncompatible(
                "PostgreSQL command admission requires reader and writer compatibility"
            )
        self._sink.check_ready()
        self._ready = True
        return status

    def _require_ready(self) -> None:
        if not self._ready:
            raise PostgresSchemaIncompatible(
                "PostgreSQL command admission requires a successful check_ready()"
            )

    def _table(self, name: str) -> object:
        from psycopg import sql

        return sql.Identifier(self.database.config.schema, name)

    def _lock_run(self, cursor: object, run_id: str) -> bool:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL("SELECT run_id FROM {} WHERE run_id = %s FOR SHARE").format(
                self._table("run_authority")
            ),
            (run_id,),
        )
        return cursor.fetchone() is not None  # type: ignore[attr-defined]

    def _terminal_exists(self, cursor: object, run_id: str) -> bool:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL("SELECT 1 FROM {} WHERE run_id = %s").format(
                self._table("terminal_record")
            ),
            (run_id,),
        )
        return cursor.fetchone() is not None  # type: ignore[attr-defined]

    def _row(
        self,
        cursor: object,
        run_id: str,
        command_id: str,
        *,
        for_update: bool = False,
    ) -> tuple[object, ...] | None:
        from psycopg import sql

        suffix = sql.SQL(" FOR UPDATE OF dispatch") if for_update else sql.SQL("")
        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "SELECT admission.run_id, admission.command_id, admission.command_sequence, "
                "admission.command_kind, admission.request_digest, admission.payload_ref, "
                "admission.request_identity_sha256, admission.admitted_identity_sha256, "
                "admission.admitted_content_digest, admission.admitted_payload, "
                "admission.activation_identity_sha256, admission.activation_content_digest, "
                "admission.activation_payload, dispatch.delivery_state, dispatch.attempt_count, "
                "dispatch.dispatch_ref, dispatch.last_error_code, "
                "COALESCE(dispatch.claim_owner, ''), COALESCE(dispatch.claim_id, ''), "
                "dispatch.claim_generation, dispatch.leased_until, "
                "dispatch.retry_delay_microseconds "
                "FROM {} AS admission JOIN {} AS dispatch "
                "ON dispatch.run_id = admission.run_id "
                "AND dispatch.command_id = admission.command_id "
                "WHERE admission.run_id = %s AND admission.command_id = %s"
            ).format(
                self._table("activation_admission_record"),
                self._table("activation_dispatch_outbox"),
            )
            + suffix,
            (run_id, command_id),
        )
        row = cursor.fetchone()  # type: ignore[attr-defined]
        if row is None:
            return None
        # PostgreSQL may evaluate a SELECT target-list clock expression before FOR UPDATE finishes
        # waiting. Sample the DB clock in a second statement, after this transaction owns the row
        # lock, so a settlement cannot use authority that expired during lock contention.
        cursor.execute("SELECT pg_catalog.clock_timestamp()")  # type: ignore[attr-defined]
        clock_row = cursor.fetchone()  # type: ignore[attr-defined]
        if clock_row is None:  # pragma: no cover - PostgreSQL SELECT always returns one row
            raise RuntimeError("PostgreSQL dispatch clock query returned no row")
        leased_until = row[20]
        try:
            lease_active = leased_until is not None and leased_until > clock_row[0]
        except TypeError as exc:
            raise PostgresAdmissionCorrupt("dispatch lease timestamp is invalid") from exc
        return (*tuple(row[:20]), lease_active, row[21])

    @staticmethod
    def _decode_row(row: tuple[object, ...]) -> _StoredAdmission:
        admitted_payload = row[9]
        if not isinstance(admitted_payload, dict):
            raise PostgresAdmissionCorrupt("admitted command payload is not an object")
        try:
            command = AdmittedCommand.from_json(admitted_payload)
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise PostgresAdmissionCorrupt("admitted command payload is invalid") from exc
        typed = (
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            row[5],
            row[6],
            row[7],
        )
        expected = (
            command.run_id,
            command.command_id,
            command.command_sequence,
            command.kind,
            command.request_digest,
            command.payload_ref,
            command.request_identity_sha256,
            command.identity_sha256,
        )
        if typed != expected or row[8] != _content_digest(command.to_json()):
            raise PostgresAdmissionCorrupt("admitted command typed identity is inconsistent")

        activation_payload = row[12]
        activation: ActivationCommand | None = None
        if activation_payload is not None:
            if not isinstance(activation_payload, dict):
                raise PostgresAdmissionCorrupt("activation binding payload is not an object")
            try:
                activation = ActivationCommand.from_json(activation_payload)
            except (TypeError, ValueError, OverflowError, RecursionError) as exc:
                raise PostgresAdmissionCorrupt("activation binding payload is invalid") from exc
            if (
                row[10] != activation.identity_sha256
                or row[11] != _content_digest(activation.to_json())
                or activation.run_id != command.run_id
                or activation.command_id != command.command_id
                or activation.command_sequence != command.command_sequence
                or activation.kind != command.kind
                or activation.request_digest != command.request_digest
                or activation.payload_ref != command.payload_ref
            ):
                raise PostgresAdmissionCorrupt("activation binding identity is inconsistent")
        elif row[10] is not None or row[11] is not None:
            raise PostgresAdmissionCorrupt("activation binding fields are partially populated")

        delivery_state = row[13]
        attempt_count = row[14]
        claim_generation = row[19]
        if (
            type(delivery_state) is not str
            or delivery_state not in _DELIVERY_STATES
            or type(attempt_count) is not int
            or attempt_count < 0
            or type(claim_generation) is not int
            or claim_generation < 0
            or type(row[15]) is not str
            or type(row[16]) is not str
            or type(row[17]) is not str
            or type(row[18]) is not str
            or type(row[20]) is not bool
            or type(row[21]) is not int
            or row[21] < 0
        ):
            raise PostgresAdmissionCorrupt("dispatch outbox metadata is invalid")
        return _StoredAdmission(
            command=command,
            activation=activation,
            delivery_state=delivery_state,  # type: ignore[arg-type]
            attempt_count=attempt_count,
            dispatch_ref=row[15],
            last_error_code=row[16],
            claim_owner=row[17],
            claim_id=row[18],
            claim_generation=claim_generation,
            lease_active=row[20],
            retry_delay_microseconds=row[21],
        )

    def _stored(
        self,
        cursor: object,
        run_id: str,
        command_id: str,
        *,
        for_update: bool = False,
    ) -> _StoredAdmission | None:
        row = self._row(cursor, run_id, command_id, for_update=for_update)
        return None if row is None else self._decode_row(row)

    def _read_stored(self, run_id: str, command_id: str) -> _StoredAdmission | None:
        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                return self._stored(cursor, run_id, command_id)

    def _completion(
        self,
        activation: ActivationCommand,
    ) -> ActivationReceipt | None:
        loaded = self._sink.latest_checked(activation.run_id)
        if loaded.status == "missing":
            return None
        if loaded.status not in {"loaded", "migrated"} or loaded.value is None:
            raise PostgresAdmissionCorrupt(
                "activation completion checkpoint cannot be loaded safely"
            )
        checkpoint = loaded.value.checkpoint
        if activation.checkpoint_marker not in checkpoint.applied_input_ids:
            return None
        try:
            projected = ActivationReceipt.from_checkpoint(activation, checkpoint)
        except NativeAgentError as exc:
            raise PostgresAdmissionCorrupt("activation completion receipt is invalid") from exc
        if not projected.terminal:
            return projected
        terminal = self._sink.read_terminal(activation.run_id)
        if terminal is None:
            return None
        try:
            return ActivationReceipt.from_checkpoint(
                activation,
                checkpoint,
                terminal_outcome=terminal,
            )
        except NativeAgentError as exc:
            raise PostgresAdmissionCorrupt(
                "activation terminal completion receipt is invalid"
            ) from exc

    def _receipt_from_stored(self, stored: _StoredAdmission) -> AdmissionReceipt:
        if stored.delivery_state == "dead_letter":
            return AdmissionReceipt(
                command=stored.command,
                state="dead_letter",
                attempt_count=stored.attempt_count,
                error_code=stored.last_error_code,
            )
        if stored.delivery_state != "delivered":
            if stored.activation is not None:
                raise PostgresAdmissionCorrupt(
                    "activation was bound before orchestrator delivery"
                )
            return AdmissionReceipt(
                command=stored.command,
                state="prepared",
                attempt_count=stored.attempt_count,
                error_code=stored.last_error_code,
            )
        if stored.activation is None:
            return AdmissionReceipt(
                command=stored.command,
                state="dispatched",
                attempt_count=stored.attempt_count,
                dispatch_ref=stored.dispatch_ref,
            )
        completion = self._completion(stored.activation)
        return AdmissionReceipt(
            command=stored.command,
            state="completed" if completion is not None else "activation_claimed",
            attempt_count=stored.attempt_count,
            dispatch_ref=stored.dispatch_ref,
            activation_command=stored.activation,
            activation_receipt=completion,
        )

    def receipt(self, run_id: str, command_id: str) -> AdmissionReceipt | None:
        self._require_ready()
        if not is_safe_opaque_id(run_id) or not is_safe_opaque_id(command_id):
            raise ValueError("admission receipt target must use bounded opaque ids")
        stored = self._read_stored(run_id, command_id)
        return None if stored is None else self._receipt_from_stored(stored)

    def _require_receipt(self, run_id: str, command_id: str) -> AdmissionReceipt:
        receipt = self.receipt(run_id, command_id)
        if receipt is None:
            raise PostgresAdmissionCorrupt("committed admission row is missing")
        return receipt

    def _lock_admission_head(self, cursor: object, run_id: str) -> int:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "INSERT INTO {} AS head (run_id, sequence) VALUES (%s, 0) "
                "ON CONFLICT (run_id) DO NOTHING"
            ).format(self._table("activation_admission_head")),
            (run_id,),
        )
        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL("SELECT sequence FROM {} WHERE run_id = %s FOR UPDATE").format(
                self._table("activation_admission_head")
            ),
            (run_id,),
        )
        row = cursor.fetchone()  # type: ignore[attr-defined]
        if row is None or type(row[0]) is not int or not 0 <= row[0] <= _POSTGRES_BIGINT_MAX:
            raise PostgresAdmissionCorrupt("admission sequence head is missing")
        return row[0]

    def _insert_admission(
        self,
        cursor: object,
        request: AdmissionRequest,
        current_sequence: int,
    ) -> AdmittedCommand:
        from psycopg import sql
        from psycopg.types.json import Json

        sequence = current_sequence + 1
        if sequence > _POSTGRES_BIGINT_MAX:
            raise OverflowError("admission command sequence exhausted PostgreSQL bigint")
        command = AdmittedCommand.from_request(request, sequence)
        payload = command.to_json()
        digest = _content_digest(payload)
        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "INSERT INTO {} (run_id, command_id, command_sequence, command_kind, "
                "request_digest, payload_ref, request_identity_sha256, "
                "admitted_identity_sha256, admitted_content_digest, admitted_payload) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            ).format(self._table("activation_admission_record")),
            (
                command.run_id,
                command.command_id,
                command.command_sequence,
                command.kind,
                command.request_digest,
                command.payload_ref,
                command.request_identity_sha256,
                command.identity_sha256,
                digest,
                Json(payload),
            ),
        )
        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL("INSERT INTO {} (run_id, command_id) VALUES (%s, %s)").format(
                self._table("activation_dispatch_outbox")
            ),
            (command.run_id, command.command_id),
        )
        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "UPDATE {} SET sequence = %s, updated_at = pg_catalog.clock_timestamp() "
                "WHERE run_id = %s"
            ).format(self._table("activation_admission_head")),
            (sequence, command.run_id),
        )
        return command

    def admit(self, request: AdmissionRequest) -> AdmissionReceipt:
        self._require_ready()
        if not isinstance(request, AdmissionRequest):
            raise TypeError("PostgreSQL admission requires AdmissionRequest")
        try:
            with self.database.transaction() as connection:
                with self.database.cursor(connection) as cursor:
                    if not self._lock_run(cursor, request.run_id):
                        raise AdmissionRunUnavailable("admission run does not exist")
                    current_sequence = self._lock_admission_head(cursor, request.run_id)
                    existing = self._stored(cursor, request.run_id, request.command_id)
                    if existing is not None:
                        if existing.command.request_identity_sha256 != request.identity_sha256:
                            raise AdmissionConflict(
                                "command ID already belongs to a different admission request"
                            )
                    else:
                        if self._terminal_exists(cursor, request.run_id):
                            raise AdmissionRunTerminal(
                                "terminal run cannot admit another command"
                            )
                        self._insert_admission(cursor, request, current_sequence)
            return self._require_receipt(request.run_id, request.command_id)
        except (AdmissionConflict, AdmissionRunTerminal, AdmissionRunUnavailable):
            raise
        except Exception as exc:
            if not _is_ambiguous_database_error(exc):
                raise
            try:
                stored = self._read_stored(request.run_id, request.command_id)
            except Exception:
                raise exc
            if stored is None:
                raise exc
            if stored.command.request_identity_sha256 != request.identity_sha256:
                raise AdmissionConflict(
                    "command ID already belongs to a different admission request"
                ) from None
            return self._receipt_from_stored(stored)

    def _claim_by_id(self, claim_id: str) -> _StoredAdmission | None:
        from psycopg import sql

        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                cursor.execute(
                    sql.SQL(
                        "SELECT run_id, command_id FROM {} WHERE claim_id = %s"
                    ).format(self._table("activation_dispatch_outbox")),
                    (claim_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                return self._stored(cursor, str(row[0]), str(row[1]))

    @staticmethod
    def _dispatch_claim(stored: _StoredAdmission, owner_id: str, claim_id: str) -> DispatchClaim:
        if (
            stored.delivery_state != "leased"
            or not stored.lease_active
            or stored.claim_owner != owner_id
            or stored.claim_id != claim_id
            or stored.claim_generation < 1
            or stored.attempt_count < 1
        ):
            raise DispatchClaimLost("dispatch claim is no longer active")
        return DispatchClaim(
            token=DispatchToken(
                run_id=stored.command.run_id,
                command_id=stored.command.command_id,
                owner_id=owner_id,
                claim_id=claim_id,
                generation=stored.claim_generation,
            ),
            command=stored.command,
            attempt=stored.attempt_count,
        )

    def claim_dispatch(
        self,
        owner_id: str,
        claim_id: str,
        *,
        lease_s: float,
    ) -> DispatchClaim | None:
        self._require_ready()
        if not is_safe_opaque_id(owner_id) or not is_safe_opaque_id(claim_id):
            raise ValueError("dispatch owner and claim IDs must be bounded opaque ids")
        if (
            type(lease_s) not in {int, float}
            or isinstance(lease_s, bool)
            or not math.isfinite(float(lease_s))
            or not 0 < float(lease_s) <= 86_400
        ):
            raise ValueError("dispatch lease_s must be in the range (0, 86400]")
        from psycopg import sql

        try:
            with self.database.transaction() as connection:
                with self.database.cursor(connection) as cursor:
                    cursor.execute(
                        sql.SQL(
                            "SELECT run_id, command_id FROM {} WHERE claim_id = %s"
                        ).format(self._table("activation_dispatch_outbox")),
                        (claim_id,),
                    )
                    existing_key = cursor.fetchone()
                    if existing_key is not None:
                        existing = self._stored(
                            cursor,
                            str(existing_key[0]),
                            str(existing_key[1]),
                        )
                        assert existing is not None
                        if existing.claim_owner != owner_id:
                            raise AdmissionConflict("dispatch claim ID belongs to another owner")
                        return self._dispatch_claim(existing, owner_id, claim_id)
                    cursor.execute(
                        sql.SQL(
                            "WITH candidate AS ("
                            "SELECT dispatch.run_id, dispatch.command_id "
                            "FROM {} AS dispatch JOIN {} AS admission "
                            "ON admission.run_id = dispatch.run_id "
                            "AND admission.command_id = dispatch.command_id "
                            "WHERE ((dispatch.delivery_state = 'pending' "
                            "AND dispatch.available_at <= pg_catalog.clock_timestamp()) "
                            "OR (dispatch.delivery_state = 'leased' "
                            "AND dispatch.leased_until <= pg_catalog.clock_timestamp())) "
                            "AND NOT EXISTS (SELECT 1 FROM {} AS prior_dispatch "
                            "JOIN {} AS prior_admission "
                            "ON prior_admission.run_id = prior_dispatch.run_id "
                            "AND prior_admission.command_id = prior_dispatch.command_id "
                            "WHERE prior_admission.run_id = admission.run_id "
                            "AND prior_admission.command_sequence < admission.command_sequence "
                            "AND prior_dispatch.delivery_state <> 'delivered') "
                            "ORDER BY dispatch.available_at, dispatch.created_at, "
                            "dispatch.run_id, admission.command_sequence "
                            "FOR UPDATE OF dispatch SKIP LOCKED LIMIT 1) "
                            "UPDATE {} AS dispatch SET delivery_state = 'leased', "
                            "attempt_count = dispatch.attempt_count + 1, claim_owner = %s, "
                            "claim_id = %s, claim_generation = dispatch.claim_generation + 1, "
                            "leased_until = pg_catalog.clock_timestamp() + "
                            + _ELAPSED_TTL_INTERVAL
                            + ", "
                            "last_error_code = '', retry_delay_microseconds = 0, "
                            "updated_at = pg_catalog.clock_timestamp() "
                            "FROM candidate WHERE dispatch.run_id = candidate.run_id "
                            "AND dispatch.command_id = candidate.command_id "
                            "RETURNING dispatch.run_id, dispatch.command_id"
                        ).format(
                            self._table("activation_dispatch_outbox"),
                            self._table("activation_admission_record"),
                            self._table("activation_dispatch_outbox"),
                            self._table("activation_admission_record"),
                            self._table("activation_dispatch_outbox"),
                        ),
                        (owner_id, claim_id, _duration_microseconds(float(lease_s))),
                    )
                    selected = cursor.fetchone()
                    if selected is None:
                        return None
                    stored = self._stored(cursor, str(selected[0]), str(selected[1]))
                    assert stored is not None
                    return self._dispatch_claim(stored, owner_id, claim_id)
        except (AdmissionConflict, DispatchClaimLost):
            raise
        except Exception as exc:
            if not (_is_ambiguous_database_error(exc) or _is_unique_violation(exc)):
                raise
            try:
                stored = self._claim_by_id(claim_id)
            except Exception:
                raise exc
            if stored is None:
                raise exc
            if stored.claim_owner != owner_id:
                raise AdmissionConflict("dispatch claim ID belongs to another owner") from None
            return self._dispatch_claim(stored, owner_id, claim_id)

    @staticmethod
    def _token_matches(stored: _StoredAdmission, token: DispatchToken) -> bool:
        return (
            stored.command.run_id == token.run_id
            and stored.command.command_id == token.command_id
            and stored.claim_owner == token.owner_id
            and stored.claim_id == token.claim_id
            and stored.claim_generation == token.generation
        )

    def _require_active_claim(
        self,
        stored: _StoredAdmission | None,
        token: DispatchToken,
    ) -> _StoredAdmission:
        if (
            stored is None
            or not self._token_matches(stored, token)
            or stored.delivery_state != "leased"
            or not stored.lease_active
        ):
            raise DispatchClaimLost("dispatch mutation requires the current active claim")
        return stored

    def acknowledge_dispatch(
        self,
        token: DispatchToken,
        result: DispatchResult,
    ) -> AdmissionReceipt:
        self._require_ready()
        if not isinstance(token, DispatchToken) or not isinstance(result, DispatchResult):
            raise TypeError("dispatch acknowledgement requires DispatchToken and DispatchResult")
        if result.status != "accepted":
            raise ValueError("dispatch acknowledgement requires an accepted result")
        from psycopg import sql

        try:
            with self.database.transaction() as connection:
                with self.database.cursor(connection) as cursor:
                    stored = self._stored(cursor, token.run_id, token.command_id, for_update=True)
                    if (
                        stored is not None
                        and stored.delivery_state == "delivered"
                        and self._token_matches(stored, token)
                        and stored.dispatch_ref == result.dispatch_ref
                    ):
                        pass
                    else:
                        self._require_active_claim(stored, token)
                        cursor.execute(
                            sql.SQL(
                                "UPDATE {} SET delivery_state = 'delivered', leased_until = NULL, "
                                "dispatch_ref = %s, delivered_at = pg_catalog.clock_timestamp(), "
                                "last_error_code = '', retry_delay_microseconds = 0, "
                                "updated_at = pg_catalog.clock_timestamp() "
                                "WHERE run_id = %s AND command_id = %s"
                            ).format(self._table("activation_dispatch_outbox")),
                            (result.dispatch_ref, token.run_id, token.command_id),
                        )
            return self._require_receipt(token.run_id, token.command_id)
        except DispatchClaimLost:
            raise
        except Exception as exc:
            if not _is_ambiguous_database_error(exc):
                raise
            try:
                stored = self._read_stored(token.run_id, token.command_id)
            except Exception:
                raise exc
            if (
                stored is None
                or stored.delivery_state != "delivered"
                or not self._token_matches(stored, token)
                or stored.dispatch_ref != result.dispatch_ref
            ):
                raise exc
            return self._receipt_from_stored(stored)

    def _settle_failed_dispatch(
        self,
        token: DispatchToken,
        *,
        error_code: str,
        delivery_state: Literal["pending", "dead_letter"],
        delay_s: float = 0.0,
    ) -> AdmissionReceipt:
        self._require_ready()
        if not isinstance(token, DispatchToken):
            raise TypeError("dispatch settlement requires DispatchToken")
        if not is_safe_taxonomy_code(error_code):
            raise ValueError("dispatch settlement error_code must be a taxonomy code")
        if (
            type(delay_s) not in {int, float}
            or isinstance(delay_s, bool)
            or not math.isfinite(float(delay_s))
            or not 0 <= float(delay_s) <= 86_400
        ):
            raise ValueError("dispatch retry delay must be in the range [0, 86400]")
        from psycopg import sql
        delay_microseconds = _duration_microseconds(float(delay_s))

        try:
            with self.database.transaction() as connection:
                with self.database.cursor(connection) as cursor:
                    stored = self._stored(cursor, token.run_id, token.command_id, for_update=True)
                    if (
                        stored is not None
                        and stored.delivery_state == delivery_state
                        and self._token_matches(stored, token)
                        and stored.last_error_code == error_code
                        and stored.retry_delay_microseconds == delay_microseconds
                    ):
                        pass
                    else:
                        self._require_active_claim(stored, token)
                        cursor.execute(
                            sql.SQL(
                                "UPDATE {} SET delivery_state = %s, leased_until = NULL, "
                                "available_at = pg_catalog.clock_timestamp() + "
                                + _ELAPSED_TTL_INTERVAL
                                + ", "
                                "dispatch_ref = '', delivered_at = NULL, last_error_code = %s, "
                                "retry_delay_microseconds = %s, "
                                "updated_at = pg_catalog.clock_timestamp() "
                                "WHERE run_id = %s AND command_id = %s"
                            ).format(self._table("activation_dispatch_outbox")),
                            (
                                delivery_state,
                                delay_microseconds,
                                error_code,
                                delay_microseconds,
                                token.run_id,
                                token.command_id,
                            ),
                        )
            return self._require_receipt(token.run_id, token.command_id)
        except DispatchClaimLost:
            raise
        except Exception as exc:
            if not _is_ambiguous_database_error(exc):
                raise
            try:
                stored = self._read_stored(token.run_id, token.command_id)
            except Exception:
                raise exc
            if (
                stored is None
                or stored.delivery_state != delivery_state
                or not self._token_matches(stored, token)
                or stored.last_error_code != error_code
                or stored.retry_delay_microseconds != delay_microseconds
            ):
                raise exc
            return self._receipt_from_stored(stored)

    def retry_dispatch(
        self,
        token: DispatchToken,
        *,
        error_code: str,
        delay_s: float,
    ) -> AdmissionReceipt:
        return self._settle_failed_dispatch(
            token,
            error_code=error_code,
            delivery_state="pending",
            delay_s=delay_s,
        )

    def reject_dispatch(
        self,
        token: DispatchToken,
        *,
        error_code: str,
    ) -> AdmissionReceipt:
        return self._settle_failed_dispatch(
            token,
            error_code=error_code,
            delivery_state="dead_letter",
        )

    def _checked_checkpoint_locked(self, cursor: object, run_id: str) -> RunCheckpoint:
        from psycopg import sql

        cursor.execute(  # type: ignore[attr-defined]
            sql.SQL(
                "SELECT head.sequence, record.content_digest, record.payload, "
                "record.submitted_blobs, record.run_id, record.sequence, record.schema_version "
                "FROM {} AS head JOIN {} AS record ON record.run_id = head.run_id "
                "AND record.sequence = head.sequence WHERE head.run_id = %s FOR SHARE OF head"
            ).format(
                self._table("checkpoint_head"),
                self._table("checkpoint_record"),
            ),
            (run_id,),
        )
        row = cursor.fetchone()  # type: ignore[attr-defined]
        if row is None or not isinstance(row[2], dict) or not isinstance(row[3], dict):
            raise AdmissionRunUnavailable("activation source checkpoint is unavailable")
        payload = row[2]
        digest = canonical_sha256({"record": payload, "blobs": row[3]})
        if (
            row[1] != digest
            or (row[4], row[5], row[6])
            != (payload.get("run_id"), payload.get("seq"), payload.get("schema_version"))
            or row[0] != row[5]
        ):
            raise PostgresAdmissionCorrupt("activation source checkpoint identity is invalid")
        decoded = decode_checkpoint(payload)
        if decoded.status not in {"loaded", "migrated"} or decoded.value is None:
            raise PostgresAdmissionCorrupt(
                "activation source checkpoint cannot be decoded safely"
            )
        return decoded.value

    def _read_bound_activation(
        self,
        command: AdmittedCommand,
        writer_token: WriterToken,
    ) -> ActivationCommand | None:
        with self.database.transaction() as connection:
            with self.database.cursor(connection) as cursor:
                current = self._authority._read_locked(cursor, writer_token.run_id)
                if (
                    current is None
                    or current.writer_token != writer_token
                    or not current.active
                ):
                    raise ActivationBindingConflict("activation binding writer was fenced")
                stored = self._stored(cursor, command.run_id, command.command_id)
                if stored is None or stored.command != command:
                    raise AdmissionConflict("activation binding admission identity changed")
                return stored.activation

    def bind_activation(
        self,
        command: AdmittedCommand,
        *,
        writer_token: WriterToken,
    ) -> ActivationCommand:
        self._require_ready()
        if not isinstance(command, AdmittedCommand) or not isinstance(writer_token, WriterToken):
            raise TypeError("activation binding requires AdmittedCommand and WriterToken")
        if command.run_id != writer_token.run_id:
            raise ActivationBindingConflict("activation binding writer belongs to another run")
        from psycopg import sql
        from psycopg.types.json import Json

        try:
            with self.database.transaction() as connection:
                with self.database.cursor(connection) as cursor:
                    current = self._authority._read_locked(cursor, writer_token.run_id)
                    if (
                        current is None
                        or current.writer_token != writer_token
                        or not current.active
                    ):
                        raise ActivationBindingConflict("activation binding writer was fenced")
                    stored = self._stored(
                        cursor,
                        command.run_id,
                        command.command_id,
                        for_update=True,
                    )
                    if stored is None or stored.command != command:
                        raise AdmissionConflict("activation binding admission identity changed")
                    if stored.delivery_state != "delivered":
                        raise ActivationBindingConflict(
                            "activation binding requires orchestrator delivery"
                        )
                    if stored.activation is not None:
                        return stored.activation
                    if self._terminal_exists(cursor, command.run_id):
                        raise AdmissionRunTerminal("terminal run cannot bind another activation")
                    checkpoint = self._checked_checkpoint_locked(cursor, command.run_id)
                    if checkpoint.terminal:
                        raise AdmissionRunTerminal(
                            "terminal checkpoint cannot bind another activation"
                        )
                    activation = ActivationCommand(
                        run_id=command.run_id,
                        command_id=command.command_id,
                        command_sequence=command.command_sequence,
                        kind=command.kind,
                        source_checkpoint_seq=checkpoint.seq,
                        source_checkpoint_sha256=canonical_sha256(checkpoint.to_json()),
                        request_digest=command.request_digest,
                        payload_ref=command.payload_ref,
                    )
                    payload = activation.to_json()
                    cursor.execute(
                        sql.SQL(
                            "UPDATE {} SET activation_identity_sha256 = %s, "
                            "activation_content_digest = %s, activation_payload = %s, "
                            "updated_at = pg_catalog.clock_timestamp() "
                            "WHERE run_id = %s AND command_id = %s "
                            "AND activation_payload IS NULL"
                        ).format(self._table("activation_admission_record")),
                        (
                            activation.identity_sha256,
                            _content_digest(payload),
                            Json(payload),
                            command.run_id,
                            command.command_id,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise ActivationBindingConflict(
                            "activation binding lost its immutable coordinate"
                        )
                    return activation
        except (
            ActivationBindingConflict,
            AdmissionConflict,
            AdmissionRunTerminal,
            AdmissionRunUnavailable,
            PostgresAdmissionCorrupt,
        ):
            raise
        except Exception as exc:
            if not _is_ambiguous_database_error(exc):
                raise
            try:
                activation = self._read_bound_activation(command, writer_token)
            except Exception:
                raise exc
            if activation is None:
                raise exc
            return activation


__all__ = ["PostgresAdmissionCorrupt", "PostgresCommandAdmissionStore"]
