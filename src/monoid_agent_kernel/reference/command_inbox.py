"""Durable, multi-instance command inbox implementations for the Reference backend."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from monoid_agent_kernel.core.control import ControlCommand, ControlResult
from monoid_agent_kernel.errors import NativeAgentError
from monoid_agent_kernel.identifiers import namespaced_id

COMMAND_ENVELOPE_VERSION = namespaced_id("command-inbox.v1")
COMMAND_RECEIPT_VERSION = namespaced_id("command-receipt.v1")
CommandStatus = Literal["pending", "claimed", "completed", "failed"]

_SENSITIVE_KEY_PARTS = ("authorization", "credential", "password", "secret", "token", "api_key")


class CommandQueueFull(NativeAgentError):
    error_code = "command_queue_full"


@dataclass(frozen=True)
class CommandPrincipal:
    tenant_id: str
    user_id: str
    issuer: str = ""

    @property
    def actor(self) -> str:
        authenticated = f"{self.tenant_id}/{self.user_id}"
        return f"{authenticated} ({self.issuer})" if self.issuer else authenticated

    def to_json(self) -> dict[str, str]:
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "issuer": self.issuer,
        }


@dataclass(frozen=True)
class StoredCommand:
    run_id: str
    command_id: str
    type: str
    args: dict[str, Any]
    principal: CommandPrincipal
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
            "args": sanitize_command_data(self.args),
            "principal": self.principal.to_json(),
            "reason": self.reason,
            "created_at": self.created_at,
            "status": self.status,
            "claimed_by": self.claimed_by,
            "claimed_at": self.claimed_at,
        }

    def control_command(self, *, token: str) -> ControlCommand:
        return ControlCommand(
            type=self.type,  # type: ignore[arg-type]
            run_id=self.run_id,
            args={**self.args, "token": token},
            issuer=self.principal.actor,
            reason=self.reason,
            command_id=self.command_id,
        )


@dataclass(frozen=True)
class CommandReceipt:
    run_id: str
    command_id: str
    status: CommandStatus
    result: dict[str, Any] | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": COMMAND_RECEIPT_VERSION,
            "run_id": self.run_id,
            "command_id": self.command_id,
            "status": self.status,
            "result": dict(self.result) if self.result is not None else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class CommandStore(Protocol):
    def append(self, command: StoredCommand, *, max_pending: int) -> CommandReceipt: ...

    def claim(self, run_id: str, worker_id: str, *, claim_ttl_s: float) -> StoredCommand | None: ...

    def acknowledge(
        self, run_id: str, command_id: str, worker_id: str, result: ControlResult
    ) -> CommandReceipt: ...

    def receipt(self, run_id: str, command_id: str) -> CommandReceipt | None: ...


def sanitize_command_data(value: Any, *, key: str = "") -> Any:
    """Return JSON-safe persisted data with credential-shaped fields redacted."""

    lowered = key.lower()
    if key and any(part in lowered for part in _SENSITIVE_KEY_PARTS):
        return "[redacted]"
    if isinstance(value, dict):
        return {
            str(item_key): sanitize_command_data(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_command_data(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def redact_command_credential(value: Any, credential: str) -> Any:
    """Remove the authenticated bearer value if a caller repeated it in payload text."""

    if not credential:
        return value
    if isinstance(value, dict):
        return {
            str(item_key): redact_command_credential(item, credential)
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_command_credential(item, credential) for item in value]
    if isinstance(value, str):
        return value.replace(credential, "[redacted]")
    return value


class InMemoryCommandStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._commands: dict[tuple[str, str], StoredCommand] = {}
        self._results: dict[tuple[str, str], dict[str, Any]] = {}

    def append(self, command: StoredCommand, *, max_pending: int) -> CommandReceipt:
        key = (command.run_id, command.command_id)
        with self._lock:
            existing = self._commands.get(key)
            if existing is not None:
                return self._receipt(existing)
            pending = sum(
                item.run_id == command.run_id and item.status in {"pending", "claimed"}
                for item in self._commands.values()
            )
            if pending >= max_pending:
                raise CommandQueueFull(f"command queue is full for run {command.run_id}")
            persisted = StoredCommand(
                **{**command.__dict__, "args": dict(sanitize_command_data(command.args))}
            )
            self._commands[key] = persisted
            return self._receipt(persisted)

    def claim(self, run_id: str, worker_id: str, *, claim_ttl_s: float) -> StoredCommand | None:
        now = time.time()
        with self._lock:
            eligible = [
                item
                for item in self._commands.values()
                if item.run_id == run_id
                and (
                    item.status == "pending"
                    or (item.status == "claimed" and now - item.claimed_at > claim_ttl_s)
                )
            ]
            if not eligible:
                return None
            selected = min(eligible, key=lambda item: (item.created_at, item.command_id))
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
        key = (run_id, command_id)
        with self._lock:
            command = self._commands[key]
            if command.status != "claimed" or command.claimed_by != worker_id:
                raise RuntimeError("command is not claimed by this worker")
            status: CommandStatus = "completed" if result.status == "ok" else "failed"
            acknowledged = StoredCommand(**{**command.__dict__, "status": status})
            self._commands[key] = acknowledged
            self._results[key] = sanitize_command_data(result.to_json())
            return self._receipt(acknowledged)

    def receipt(self, run_id: str, command_id: str) -> CommandReceipt | None:
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
    UNIQUE(run_id, command_id)
);
CREATE INDEX IF NOT EXISTS command_inbox_pending
ON command_inbox(run_id, status, ordinal);
"""


class SqliteCommandStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(command_inbox)")}
            if "schema_version" not in columns:
                conn.execute(
                    "ALTER TABLE command_inbox ADD COLUMN schema_version TEXT NOT NULL "
                    f"DEFAULT '{COMMAND_ENVELOPE_VERSION}'"
                )

    def append(self, command: StoredCommand, *, max_pending: int) -> CommandReceipt:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._row(conn, command.run_id, command.command_id)
            if row is not None:
                conn.commit()
                return self._receipt_from_row(row)
            pending = conn.execute(
                "SELECT COUNT(*) FROM command_inbox WHERE run_id=? AND status IN ('pending','claimed')",
                (command.run_id,),
            ).fetchone()[0]
            if int(pending) >= max_pending:
                conn.rollback()
                raise CommandQueueFull(f"command queue is full for run {command.run_id}")
            conn.execute(
                "INSERT INTO command_inbox(run_id, command_id, command_type, args, principal, reason, "
                "created_at, status, updated_at, schema_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
                (
                    command.run_id,
                    command.command_id,
                    command.type,
                    json.dumps(sanitize_command_data(command.args), sort_keys=True),
                    json.dumps(command.principal.to_json(), sort_keys=True),
                    command.reason,
                    command.created_at,
                    command.created_at,
                    COMMAND_ENVELOPE_VERSION,
                ),
            )
            row = self._row(conn, command.run_id, command.command_id)
            conn.commit()
            assert row is not None
            return self._receipt_from_row(row)

    def claim(self, run_id: str, worker_id: str, *, claim_ttl_s: float) -> StoredCommand | None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM command_inbox WHERE run_id=? AND "
                "(status='pending' OR (status='claimed' AND claimed_at<?)) "
                "ORDER BY ordinal LIMIT 1",
                (run_id, now - claim_ttl_s),
            ).fetchone()
            if row is None:
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
        status = "completed" if result.status == "ok" else "failed"
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = conn.execute(
                "UPDATE command_inbox SET status=?, result=?, updated_at=? "
                "WHERE run_id=? AND command_id=? AND status='claimed' AND claimed_by=?",
                (
                    status,
                    json.dumps(sanitize_command_data(result.to_json()), sort_keys=True),
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
        with self._connect() as conn:
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
        if str(row[13]) != COMMAND_ENVELOPE_VERSION:
            raise ValueError(f"unsupported command inbox schema: {row[13]}")
        principal = json.loads(row[5])
        return StoredCommand(
            run_id=str(row[1]),
            command_id=str(row[2]),
            type=str(row[3]),
            args=dict(json.loads(row[4])),
            principal=CommandPrincipal(
                tenant_id=str(principal["tenant_id"]),
                user_id=str(principal["user_id"]),
                issuer=str(principal.get("issuer") or ""),
            ),
            reason=str(row[6]),
            created_at=float(row[7]),
            status=str(row[8]),  # type: ignore[arg-type]
            claimed_by=str(row[9]),
            claimed_at=float(row[10]),
        )

    @staticmethod
    def _receipt_from_row(row: sqlite3.Row) -> CommandReceipt:
        return CommandReceipt(
            run_id=str(row[1]),
            command_id=str(row[2]),
            status=str(row[8]),  # type: ignore[arg-type]
            result=dict(json.loads(row[11])) if row[11] else None,
            created_at=float(row[7]),
            updated_at=float(row[12]),
        )
