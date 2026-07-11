"""Reference durable stores for the Monoid backend.

The core defines the seams (``CheckpointStore``; the backend defines ``LeaseStore``); these
are integrator implementations of the HOW. ``SqliteCheckpointStore``/``SqliteLeaseStore``
are zero-dependency (stdlib ``sqlite3``) durable backends — a DB transaction gives atomic
last-good commits and CAS leases for free, and a single shared db can host both, which is
what lets a different worker reclaim a run across process/host boundaries.
"""

from __future__ import annotations

from monoid_agent_kernel.reference.stores.lease import LeaseStore, LocalFsLeaseStore
from monoid_agent_kernel.reference.stores.sqlite import SqliteCheckpointStore, SqliteLeaseStore
from monoid_agent_kernel.reference.command_inbox import (
    CommandConflict,
    CommandStore,
    InMemoryCommandStore,
    SqliteCommandStore,
)

__all__ = [
    "LeaseStore",
    "LocalFsLeaseStore",
    "SqliteCheckpointStore",
    "SqliteLeaseStore",
    "CommandStore",
    "CommandConflict",
    "InMemoryCommandStore",
    "SqliteCommandStore",
]
