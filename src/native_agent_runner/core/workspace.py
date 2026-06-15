"""Workspace abstraction for the agent engine.

Defines the value types the engine exchanges with a workspace implementation.
These live in ``core`` (not ``workspace/local.py``) so the engine can depend on
them without importing the concrete ``LocalWorkspaceBackend``; the local backend
imports them back from here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FileEntry:
    path: str
    kind: str
    size: int = 0


@dataclass(frozen=True)
class ChangedEntry:
    path: str
    kind: str
    size: int = 0
    sha256: str | None = None
    content: bytes | None = None
    base_sha256: str | None = None
    proposed_sha256: str | None = None
    change_kind: str = "modified"
