"""Durable, multi-instance command inbox implementations for the Reference backend."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, get_args

from monoid_agent_kernel.core.agents import AgentRuntimeConfig
from monoid_agent_kernel.core.control import (
    ACCEPTED_CONTROL_PROTOCOL_VERSIONS,
    ControlCommand,
    ControlCommandType,
    ControlResult,
    ControlResultStatus,
)
from monoid_agent_kernel.core.json_ingress import (
    is_finite_json_number,
    loads_json_ingress,
    normalize_json_ingress,
    normalize_unicode_scalars,
)
from monoid_agent_kernel.identifiers import namespaced_id
from monoid_agent_kernel.reference._shared.control_transport import (
    COMMAND_RECEIPT_VERSION as COMMAND_RECEIPT_VERSION,
    CommandConflict,
    CommandPrincipal,
    CommandQueueFull,
    CommandReceipt,
    CommandStatus,
    redact_command_credential as redact_command_credential,
    sanitize_command_args,
    sanitize_command_data,
)

COMMAND_ENVELOPE_VERSION = namespaced_id("command-inbox.v1")
_DURABLE_RUNTIME_CONFIG_KEY = "_monoid_durable_runtime_config"
_DURABLE_RUNTIME_CONFIG_MARKER = object()
_COMMAND_STATUSES = frozenset({"pending", "claimed", "completed", "failed"})
_COMMAND_TYPES = frozenset(get_args(ControlCommandType))
_RESULT_STATUSES = frozenset(get_args(ControlResultStatus))


@dataclass(frozen=True)
class StoredCommand:
    run_id: str
    command_id: str
    type: str
    args: dict[str, Any]
    principal: CommandPrincipal
    token_sha256: str = ""
    reason: str = ""
    created_at: float = field(default_factory=time.time)
    status: CommandStatus = "pending"
    claimed_by: str = ""
    claimed_at: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": COMMAND_ENVELOPE_VERSION,
            "run_id": self.run_id,
            "command_id": self.command_id,
            "type": self.type,
            "args": sanitize_command_args(self.type, self.args),
            "principal": self.principal.to_json(),
            "token_sha256": self.token_sha256,
            "reason": self.reason,
            "created_at": self.created_at,
            "status": self.status,
            "claimed_by": self.claimed_by,
            "claimed_at": self.claimed_at,
        }

    def control_command(
        self, *, token: str, transient_args: dict[str, Any] | None = None
    ) -> ControlCommand:
        args = dict(self.args if transient_args is None else transient_args)
        if transient_args is None and self.type == "replace_runtime_config":
            config_payload = args.get("config")
            if isinstance(config_payload, dict):
                # A v0.19 worker could have queued this v1 command with literal leading bangs in
                # tool scopes. Fresh commands are parsed strictly before enqueue; only the retained
                # copy crosses the compatibility decoder here.
                args["config"] = AgentRuntimeConfig.from_durable_json(config_payload).to_json()
                args[_DURABLE_RUNTIME_CONFIG_KEY] = _DURABLE_RUNTIME_CONFIG_MARKER
        return ControlCommand(
            type=self.type,  # type: ignore[arg-type]
            run_id=self.run_id,
            args={**args, "token": token},
            issuer=self.principal.actor,
            reason=self.reason,
            command_id=self.command_id,
        )


class CommandStore(Protocol):
    def append(
        self, command: StoredCommand, *, max_pending: int, require_empty: bool = False
    ) -> CommandReceipt: ...

    def read_command(self, run_id: str, command_id: str) -> StoredCommand | None: ...

    def claim(self, run_id: str, worker_id: str, *, claim_ttl_s: float) -> StoredCommand | None: ...

    def acknowledge(
        self, run_id: str, command_id: str, worker_id: str, result: ControlResult
    ) -> CommandReceipt: ...

    def receipt(self, run_id: str, command_id: str) -> CommandReceipt | None: ...


def _required_text(value: Any, field_name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{field_name} must be a string")
    return normalize_unicode_scalars(value)


def _required_object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _finite_number(value: Any, field_name: str) -> float:
    if not is_finite_json_number(value):
        raise ValueError(f"{field_name} must be a finite number")
    return float(value)


def _positive_integer(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _exact_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _normalize_stored_command(command: StoredCommand) -> StoredCommand:
    """Canonicalize a submitted command before lookup or persistence."""

    if not isinstance(command, StoredCommand):
        raise ValueError("command must be a StoredCommand")
    if not isinstance(command.args, dict):
        raise ValueError("command args must be an object")
    if not isinstance(command.principal, CommandPrincipal):
        raise ValueError("command principal must be a CommandPrincipal")

    command_type = _required_text(command.type, "command type")
    if command_type not in _COMMAND_TYPES:
        raise ValueError("command type is invalid")
    status = _required_text(command.status, "command status")
    if status not in _COMMAND_STATUSES:
        raise ValueError("command status is invalid")
    args = normalize_json_ingress(sanitize_command_args(command_type, command.args))
    if not isinstance(args, dict):  # sanitize_command_args guarantees this today
        raise ValueError("command args must be an object")
    return replace(
        command,
        run_id=_required_text(command.run_id, "command run_id"),
        command_id=_required_text(command.command_id, "command command_id"),
        type=command_type,
        args=args,
        principal=CommandPrincipal(
            tenant_id=_required_text(command.principal.tenant_id, "command principal tenant_id"),
            user_id=_required_text(command.principal.user_id, "command principal user_id"),
            issuer=_required_text(command.principal.issuer, "command principal issuer"),
        ),
        token_sha256=_required_text(command.token_sha256, "command token_sha256"),
        reason=_required_text(command.reason, "command reason"),
        created_at=_finite_number(command.created_at, "command created_at"),
        status=status,  # type: ignore[arg-type]
        claimed_by=_required_text(command.claimed_by, "command claimed_by"),
        claimed_at=_finite_number(command.claimed_at, "command claimed_at"),
    )


def _normalize_result_payload(result: ControlResult) -> dict[str, Any]:
    if not isinstance(result, ControlResult):
        raise ValueError("command result must be a ControlResult")
    result_type = _required_text(result.type, "command result type")
    if result_type not in _COMMAND_TYPES:
        raise ValueError("command result type is invalid")
    result_status = _required_text(result.status, "command result status")
    if result_status not in _RESULT_STATUSES:
        raise ValueError("command result status is invalid")
    if result.state is not None:
        _required_text(result.state, "command result state")
    if not isinstance(result.data, dict):
        raise ValueError("command result data must be an object")
    _required_text(result.run_id, "command result run_id")
    _required_text(result.error, "command result error")
    _required_text(result.error_code, "command result error_code")
    payload = normalize_json_ingress(sanitize_command_data(result.to_json()))
    if not isinstance(payload, dict):  # ControlResult.to_json guarantees this today
        raise ValueError("command result must be an object")
    return payload


def _normalize_retained_result_payload(value: Any) -> dict[str, Any]:
    """Validate a decoded durable result with the same rules as a fresh acknowledgement."""

    payload = _required_object(value, "command result")
    protocol = _required_text(payload.get("protocol", ""), "command result protocol")
    if protocol and protocol not in ACCEPTED_CONTROL_PROTOCOL_VERSIONS:
        raise ValueError("command result protocol is invalid")
    state = payload.get("state")
    if state is not None:
        state = _required_text(state, "command result state")
    return _normalize_result_payload(
        ControlResult(
            run_id=_required_text(payload.get("run_id"), "command result run_id"),
            type=_required_text(payload.get("type"), "command result type"),  # type: ignore[arg-type]
            status=_required_text(  # type: ignore[arg-type]
                payload.get("status"), "command result status"
            ),
            state=state,
            data=_required_object(payload.get("data", {}), "command result data"),
            error=_required_text(payload.get("error", ""), "command result error"),
            error_code=_required_text(payload.get("error_code", ""), "command result error_code"),
        )
    )


def _same_command_identity(existing: StoredCommand, submitted: StoredCommand) -> bool:
    return (
        existing.type == submitted.type
        and existing.args == submitted.args
        and existing.principal == submitted.principal
        and existing.reason == submitted.reason
    )


def _raise_duplicate_conflict(command_id: str) -> None:
    raise CommandConflict(f"command_id {command_id!r} already belongs to a different command")


class InMemoryCommandStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._commands: dict[tuple[str, str], StoredCommand] = {}
        self._results: dict[tuple[str, str], dict[str, Any]] = {}

    def append(
        self, command: StoredCommand, *, max_pending: int, require_empty: bool = False
    ) -> CommandReceipt:
        command = _normalize_stored_command(command)
        max_pending = _positive_integer(max_pending, "max_pending")
        require_empty = _exact_bool(require_empty, "require_empty")
        key = (command.run_id, command.command_id)
        with self._lock:
            existing = self._commands.get(key)
            if existing is not None:
                if not _same_command_identity(existing, command):
                    _raise_duplicate_conflict(command.command_id)
                return self._receipt(existing)
            pending = sum(
                item.run_id == command.run_id and item.status in {"pending", "claimed"}
                for item in self._commands.values()
            )
            if require_empty and pending:
                raise CommandQueueFull(
                    f"command lane is busy for immediate command {command.command_id}"
                )
            if pending >= max_pending:
                raise CommandQueueFull(f"command queue is full for run {command.run_id}")
            persisted = command
            self._commands[key] = persisted
            return self._receipt(persisted)

    def read_command(self, run_id: str, command_id: str) -> StoredCommand | None:
        run_id = _required_text(run_id, "run_id")
        command_id = _required_text(command_id, "command_id")
        with self._lock:
            return self._commands.get((run_id, command_id))

    def claim(self, run_id: str, worker_id: str, *, claim_ttl_s: float) -> StoredCommand | None:
        run_id = _required_text(run_id, "run_id")
        worker_id = _required_text(worker_id, "worker_id")
        claim_ttl_s = _finite_number(claim_ttl_s, "claim_ttl_s")
        now = time.time()
        with self._lock:
            selected = next(
                (
                    item
                    for item in self._commands.values()
                    if item.run_id == run_id and item.status in {"pending", "claimed"}
                ),
                None,
            )
            if selected is None:
                return None
            if selected.status == "claimed" and (
                selected.claimed_by == worker_id or now - selected.claimed_at <= claim_ttl_s
            ):
                return None
            claimed = StoredCommand(
                **{
                    **selected.__dict__,
                    "status": "claimed",
                    "claimed_by": worker_id,
                    "claimed_at": now,
                }
            )
            self._commands[(run_id, selected.command_id)] = claimed
            return claimed

    def acknowledge(
        self, run_id: str, command_id: str, worker_id: str, result: ControlResult
    ) -> CommandReceipt:
        run_id = _required_text(run_id, "run_id")
        command_id = _required_text(command_id, "command_id")
        worker_id = _required_text(worker_id, "worker_id")
        result_payload = _normalize_result_payload(result)
        key = (run_id, command_id)
        with self._lock:
            command = self._commands[key]
            if command.status != "claimed" or command.claimed_by != worker_id:
                raise RuntimeError("command is not claimed by this worker")
            status: CommandStatus = "completed" if result.status == "ok" else "failed"
            acknowledged = StoredCommand(**{**command.__dict__, "status": status})
            self._commands[key] = acknowledged
            self._results[key] = result_payload
            return self._receipt(acknowledged)

    def receipt(self, run_id: str, command_id: str) -> CommandReceipt | None:
        run_id = _required_text(run_id, "run_id")
        command_id = _required_text(command_id, "command_id")
        with self._lock:
            command = self._commands.get((run_id, command_id))
            return self._receipt(command) if command is not None else None

    def _receipt(self, command: StoredCommand) -> CommandReceipt:
        return CommandReceipt(
            run_id=command.run_id,
            command_id=command.command_id,
            status=command.status,
            result=self._results.get((command.run_id, command.command_id)),
            created_at=command.created_at,
            updated_at=command.claimed_at or command.created_at,
        )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS command_inbox (
    ordinal INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    command_id TEXT NOT NULL,
    command_type TEXT NOT NULL,
    args TEXT NOT NULL,
    principal TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at REAL NOT NULL,
    status TEXT NOT NULL,
    claimed_by TEXT NOT NULL DEFAULT '',
    claimed_at REAL NOT NULL DEFAULT 0,
    result TEXT,
    updated_at REAL NOT NULL,
    schema_version TEXT NOT NULL,
    token_sha256 TEXT NOT NULL DEFAULT '',
    UNIQUE(run_id, command_id)
);
CREATE INDEX IF NOT EXISTS command_inbox_pending
ON command_inbox(run_id, status, ordinal);
"""


class SqliteCommandStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(command_inbox)")}
            if "schema_version" not in columns:
                conn.execute(
                    "ALTER TABLE command_inbox ADD COLUMN schema_version TEXT NOT NULL "
                    f"DEFAULT '{COMMAND_ENVELOPE_VERSION}'"
                )
            if "token_sha256" not in columns:
                conn.execute(
                    "ALTER TABLE command_inbox ADD COLUMN token_sha256 TEXT NOT NULL DEFAULT ''"
                )
            conn.commit()

    def append(
        self, command: StoredCommand, *, max_pending: int, require_empty: bool = False
    ) -> CommandReceipt:
        command = _normalize_stored_command(command)
        max_pending = _positive_integer(max_pending, "max_pending")
        require_empty = _exact_bool(require_empty, "require_empty")
        with self._lock, closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._row(conn, command.run_id, command.command_id)
            if row is not None:
                if not _same_command_identity(self._command_from_row(row), command):
                    conn.rollback()
                    _raise_duplicate_conflict(command.command_id)
                conn.commit()
                return self._receipt_from_row(row)
            pending = conn.execute(
                "SELECT COUNT(*) FROM command_inbox WHERE run_id=? AND status IN ('pending','claimed')",
                (command.run_id,),
            ).fetchone()[0]
            if require_empty and int(pending):
                conn.rollback()
                raise CommandQueueFull(
                    f"command lane is busy for immediate command {command.command_id}"
                )
            if int(pending) >= max_pending:
                conn.rollback()
                raise CommandQueueFull(f"command queue is full for run {command.run_id}")
            conn.execute(
                "INSERT INTO command_inbox(run_id, command_id, command_type, args, principal, reason, "
                "created_at, status, updated_at, schema_version, token_sha256) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)",
                (
                    command.run_id,
                    command.command_id,
                    command.type,
                    json.dumps(
                        command.args,
                        sort_keys=True,
                        allow_nan=False,
                    ),
                    json.dumps(command.principal.to_json(), sort_keys=True, allow_nan=False),
                    command.reason,
                    command.created_at,
                    command.created_at,
                    COMMAND_ENVELOPE_VERSION,
                    command.token_sha256,
                ),
            )
            row = self._row(conn, command.run_id, command.command_id)
            conn.commit()
            assert row is not None
            return self._receipt_from_row(row)

    def read_command(self, run_id: str, command_id: str) -> StoredCommand | None:
        run_id = _required_text(run_id, "run_id")
        command_id = _required_text(command_id, "command_id")
        with closing(self._connect()) as conn:
            row = self._row(conn, run_id, command_id)
        return self._command_from_row(row) if row is not None else None

    def claim(self, run_id: str, worker_id: str, *, claim_ttl_s: float) -> StoredCommand | None:
        run_id = _required_text(run_id, "run_id")
        worker_id = _required_text(worker_id, "worker_id")
        claim_ttl_s = _finite_number(claim_ttl_s, "claim_ttl_s")
        now = time.time()
        with self._lock, closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM command_inbox WHERE run_id=? AND status IN ('pending','claimed') "
                "ORDER BY ordinal LIMIT 1",
                (run_id,),
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            if row["status"] == "claimed" and (
                row["claimed_by"] == worker_id or now - float(row["claimed_at"]) <= claim_ttl_s
            ):
                conn.commit()
                return None
            conn.execute(
                "UPDATE command_inbox SET status='claimed', claimed_by=?, claimed_at=?, updated_at=? "
                "WHERE ordinal=?",
                (worker_id, now, now, row[0]),
            )
            claimed = conn.execute(
                "SELECT * FROM command_inbox WHERE ordinal=?", (row[0],)
            ).fetchone()
            conn.commit()
            assert claimed is not None
            return self._command_from_row(claimed)

    def acknowledge(
        self, run_id: str, command_id: str, worker_id: str, result: ControlResult
    ) -> CommandReceipt:
        run_id = _required_text(run_id, "run_id")
        command_id = _required_text(command_id, "command_id")
        worker_id = _required_text(worker_id, "worker_id")
        result_payload = _normalize_result_payload(result)
        status = "completed" if result.status == "ok" else "failed"
        now = time.time()
        with self._lock, closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = conn.execute(
                "UPDATE command_inbox SET status=?, result=?, updated_at=? "
                "WHERE run_id=? AND command_id=? AND status='claimed' AND claimed_by=?",
                (
                    status,
                    json.dumps(
                        result_payload,
                        sort_keys=True,
                        allow_nan=False,
                    ),
                    now,
                    run_id,
                    command_id,
                    worker_id,
                ),
            ).rowcount
            if changed != 1:
                conn.rollback()
                raise RuntimeError("command is not claimed by this worker")
            row = self._row(conn, run_id, command_id)
            conn.commit()
            assert row is not None
            return self._receipt_from_row(row)

    def receipt(self, run_id: str, command_id: str) -> CommandReceipt | None:
        run_id = _required_text(run_id, "run_id")
        command_id = _required_text(command_id, "command_id")
        with closing(self._connect()) as conn:
            row = self._row(conn, run_id, command_id)
        return self._receipt_from_row(row) if row is not None else None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @staticmethod
    def _row(conn: sqlite3.Connection, run_id: str, command_id: str) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM command_inbox WHERE run_id=? AND command_id=?",
            (run_id, command_id),
        ).fetchone()

    @staticmethod
    def _command_from_row(row: sqlite3.Row) -> StoredCommand:
        schema_version = _required_text(row["schema_version"], "command schema_version")
        if schema_version != COMMAND_ENVELOPE_VERSION:
            raise ValueError(f"unsupported command inbox schema: {row['schema_version']}")
        args = _required_object(loads_json_ingress(row["args"]), "command args")
        principal = _required_object(loads_json_ingress(row["principal"]), "command principal")
        return _normalize_stored_command(
            StoredCommand(
                run_id=row["run_id"],
                command_id=row["command_id"],
                type=row["command_type"],
                args=args,
                principal=CommandPrincipal(
                    tenant_id=_required_text(
                        principal.get("tenant_id"), "command principal tenant_id"
                    ),
                    user_id=_required_text(principal.get("user_id"), "command principal user_id"),
                    issuer=_required_text(principal.get("issuer", ""), "command principal issuer"),
                ),
                token_sha256=row["token_sha256"],
                reason=row["reason"],
                created_at=row["created_at"],
                status=row["status"],
                claimed_by=row["claimed_by"],
                claimed_at=row["claimed_at"],
            )
        )

    @staticmethod
    def _receipt_from_row(row: sqlite3.Row) -> CommandReceipt:
        status = _required_text(row["status"], "command status")
        if status not in _COMMAND_STATUSES:
            raise ValueError("command status is invalid")
        result_value = row["result"]
        return CommandReceipt(
            run_id=_required_text(row["run_id"], "command run_id"),
            command_id=_required_text(row["command_id"], "command command_id"),
            status=status,  # type: ignore[arg-type]
            result=(
                _normalize_retained_result_payload(loads_json_ingress(result_value))
                if result_value is not None
                else None
            ),
            created_at=_finite_number(row["created_at"], "command created_at"),
            updated_at=_finite_number(row["updated_at"], "command updated_at"),
        )
