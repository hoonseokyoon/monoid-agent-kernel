"""Durable, multi-instance command inbox implementations for the Reference backend."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from monoid_agent_kernel.core.control import ControlCommand, ControlResult
from monoid_agent_kernel.errors import NativeAgentError
from monoid_agent_kernel.identifiers import namespaced_id

COMMAND_ENVELOPE_VERSION = namespaced_id("command-inbox.v1")
COMMAND_RECEIPT_VERSION = namespaced_id("command-receipt.v1")
CommandStatus = Literal["pending", "claimed", "completed", "failed"]

_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "access_token",
        "authorization",
        "bearer_token",
        "callback_token",
        "credential",
        "password",
        "refresh_token",
        "secret",
        "token",
    }
)
_SENSITIVE_COMPACT_KEYS = frozenset(key.replace("_", "") for key in _SENSITIVE_KEYS)


class CommandQueueFull(NativeAgentError):
    error_code = "command_queue_full"


class CommandConflict(NativeAgentError):
    error_code = "command_id_conflict"


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
            "args": sanitize_command_data(self.args),
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
        return ControlCommand(
            type=self.type,  # type: ignore[arg-type]
            run_id=self.run_id,
            args={**(self.args if transient_args is None else transient_args), "token": token},
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
    transient_result: dict[str, Any] | None = field(default=None, repr=False, compare=False)

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
    def append(
        self, command: StoredCommand, *, max_pending: int, require_empty: bool = False
    ) -> CommandReceipt: ...

    def read_command(self, run_id: str, command_id: str) -> StoredCommand | None: ...

    def claim(self, run_id: str, worker_id: str, *, claim_ttl_s: float) -> StoredCommand | None: ...

    def claim_command(
        self,
        run_id: str,
        command_id: str,
        worker_id: str,
        *,
        claim_ttl_s: float,
    ) -> StoredCommand | None: ...

    def acknowledge(
        self, run_id: str, command_id: str, worker_id: str, result: ControlResult
    ) -> CommandReceipt: ...

    def receipt(self, run_id: str, command_id: str) -> CommandReceipt | None: ...


def sanitize_command_data(value: Any, *, key: str = "") -> Any:
    """Return JSON-safe persisted data with credential-shaped fields redacted."""

    lowered = key.lower()
    compact = "".join(character for character in lowered if character.isalnum())
    sensitive_suffix = compact.endswith(("password", "secret", "secretkey"))
    if key and (compact in _SENSITIVE_COMPACT_KEYS or sensitive_suffix):
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
            str(item_key).replace(credential, "[redacted]"): redact_command_credential(
                item, credential
            )
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_command_credential(item, credential) for item in value]
    if isinstance(value, str):
        return value.replace(credential, "[redacted]")
    return value


def _same_command_identity(existing: StoredCommand, submitted: StoredCommand) -> bool:
    return (
        existing.type == submitted.type
        and existing.args == sanitize_command_data(submitted.args)
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
            persisted = StoredCommand(
                **{**command.__dict__, "args": dict(sanitize_command_data(command.args))}
            )
            self._commands[key] = persisted
            return self._receipt(persisted)

    def read_command(self, run_id: str, command_id: str) -> StoredCommand | None:
        with self._lock:
            return self._commands.get((run_id, command_id))

    def claim(self, run_id: str, worker_id: str, *, claim_ttl_s: float) -> StoredCommand | None:
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
                selected.claimed_by == worker_id
                or now - selected.claimed_at <= claim_ttl_s
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

    def claim_command(
        self,
        run_id: str,
        command_id: str,
        worker_id: str,
        *,
        claim_ttl_s: float,
    ) -> StoredCommand | None:
        now = time.time()
        key = (run_id, command_id)
        with self._lock:
            selected = self._commands.get(key)
            if selected is None or selected.status not in {"pending", "claimed"}:
                return None
            if selected.status == "claimed" and (
                selected.claimed_by == worker_id
                or now - selected.claimed_at <= claim_ttl_s
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
            self._commands[key] = claimed
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
                    json.dumps(sanitize_command_data(command.args), sort_keys=True),
                    json.dumps(command.principal.to_json(), sort_keys=True),
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
        with closing(self._connect()) as conn:
            row = self._row(conn, run_id, command_id)
        return self._command_from_row(row) if row is not None else None

    def claim(self, run_id: str, worker_id: str, *, claim_ttl_s: float) -> StoredCommand | None:
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
                row["claimed_by"] == worker_id
                or now - float(row["claimed_at"]) <= claim_ttl_s
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

    def claim_command(
        self,
        run_id: str,
        command_id: str,
        worker_id: str,
        *,
        claim_ttl_s: float,
    ) -> StoredCommand | None:
        now = time.time()
        with self._lock, closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._row(conn, run_id, command_id)
            if row is None or row["status"] not in {"pending", "claimed"}:
                conn.commit()
                return None
            if row["status"] == "claimed" and (
                row["claimed_by"] == worker_id
                or now - float(row["claimed_at"]) <= claim_ttl_s
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
        status = "completed" if result.status == "ok" else "failed"
        now = time.time()
        with self._lock, closing(self._connect()) as conn:
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
        if str(row["schema_version"]) != COMMAND_ENVELOPE_VERSION:
            raise ValueError(f"unsupported command inbox schema: {row['schema_version']}")
        principal = json.loads(row["principal"])
        return StoredCommand(
            run_id=str(row["run_id"]),
            command_id=str(row["command_id"]),
            type=str(row["command_type"]),
            args=dict(json.loads(row["args"])),
            principal=CommandPrincipal(
                tenant_id=str(principal["tenant_id"]),
                user_id=str(principal["user_id"]),
                issuer=str(principal.get("issuer") or ""),
            ),
            token_sha256=str(row["token_sha256"]),
            reason=str(row["reason"]),
            created_at=float(row["created_at"]),
            status=str(row["status"]),  # type: ignore[arg-type]
            claimed_by=str(row["claimed_by"]),
            claimed_at=float(row["claimed_at"]),
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
