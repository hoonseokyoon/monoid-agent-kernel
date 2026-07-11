"""Process-owned DBOS runtime primitives shared by optional Reference components."""

from __future__ import annotations

import importlib
import threading
from typing import Any


class DbosDependencyError(RuntimeError):
    """Raised when the explicitly selected DBOS profile is unavailable."""


class DbosProcessOwnershipError(RuntimeError):
    """Raised when another Reference component owns the process-global DBOS runtime."""


class DbosShutdownTimeout(RuntimeError):
    """Raised when active DBOS work outlives the configured shutdown grace."""


_PROCESS_OWNER_LOCK = threading.Lock()
_PROCESS_OWNER_TOKEN: object | None = None


def claim_process_owner() -> object:
    """Claim the single process-global DBOS registry before registering workflows."""

    global _PROCESS_OWNER_TOKEN
    with _PROCESS_OWNER_LOCK:
        if _PROCESS_OWNER_TOKEN is not None:
            raise DbosProcessOwnershipError(
                "another DBOS Reference component already owns the process-global runtime"
            )
        token = object()
        _PROCESS_OWNER_TOKEN = token
        return token


def release_process_owner(token: object) -> None:
    """Release a matching process owner token."""

    global _PROCESS_OWNER_TOKEN
    with _PROCESS_OWNER_LOCK:
        if _PROCESS_OWNER_TOKEN is token:
            _PROCESS_OWNER_TOKEN = None


def create_owned_runtime(dbos_module: Any, config: Any) -> Any:
    """Create DBOS after verifying no external runtime already populated its registry."""

    implementation = importlib.import_module("dbos._dbos")
    if getattr(implementation, "_dbos_global_instance", None) is not None:
        raise DbosProcessOwnershipError(
            "an existing DBOS runtime is active; runtime injection is required for shared hosting"
        )
    return dbos_module.DBOS(
        config={
            "name": config.name,
            "system_database_url": config.system_database_url,
            "application_version": config.application_version,
            "executor_id": config.executor_id,
            "run_admin_server": False,
        }
    )


def load_dbos() -> Any:
    """Load the optional DBOS dependency only after the profile is selected."""

    try:
        return importlib.import_module("dbos")
    except ImportError as exc:
        raise DbosDependencyError(
            "DBOS Reference runtime requires the 'reference-dbos' extra"
        ) from exc
