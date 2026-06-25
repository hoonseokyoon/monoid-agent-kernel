"""Workspace abstraction for the agent engine.

Defines the value types the engine exchanges with a workspace implementation.
These live in ``core`` (not ``workspace/local.py``) so the engine can depend on
them without importing the concrete ``LocalWorkspaceBackend``; the local backend
imports them back from here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from native_agent_runner.core.spec import AgentRunSpec, RunMode, WorkspaceBackendKind


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


class Workspace(Protocol):
    """The workspace surface the agent engine depends on.

    ``LocalWorkspaceBackend`` is the reference implementation; integrators may
    supply their own via ``AgentLoop.workspace_factory``. This Protocol is for
    static typing only (not ``@runtime_checkable``) — the engine never branches
    on the concrete type.
    """

    root: Path
    mode: RunMode
    backend_kind: WorkspaceBackendKind
    max_bytes_read: int

    def normalize(self, path: str | None) -> str:
        ...

    def resolve_existing_or_parent(
        self, path: str | None, *, for_write: bool = False
    ) -> tuple[str, Path]:
        ...

    def path_kind(self, path: str | None) -> str | None:
        ...

    def exists(self, path: str | None) -> bool:
        ...

    def read_bytes(self, path: str | None, *, max_bytes: int | None = None) -> tuple[bytes, str]:
        ...

    def write_bytes(
        self,
        path: str | None,
        data: bytes,
        *,
        create_dirs: bool = False,
        expected_sha256: str | None = None,
    ) -> str:
        ...

    def mkdir(self, path: str | None) -> str:
        ...

    def copy_path(
        self,
        source_path: str | None,
        destination_path: str | None,
        *,
        overwrite: bool = False,
        create_dirs: bool = False,
        recursive: bool = False,
        max_entries: int = 1000,
        max_bytes: int = 50_000_000,
    ) -> dict[str, int | str]:
        ...

    def move_path(
        self,
        source_path: str | None,
        destination_path: str | None,
        *,
        overwrite: bool = False,
        create_dirs: bool = False,
        recursive: bool = False,
        max_entries: int = 1000,
        max_bytes: int = 50_000_000,
    ) -> dict[str, int | str]:
        ...

    def delete_path(
        self,
        path: str | None,
        *,
        recursive: bool = False,
        max_entries: int = 1000,
        max_bytes: int = 50_000_000,
    ) -> dict[str, int | str]:
        ...

    def list_entries(
        self, path: str | None = ".", *, recursive: bool = False, max_entries: int = 200
    ) -> list[FileEntry]:
        ...

    def glob(self, pattern: str, *, root: str | None = ".", max_matches: int = 200) -> list[str]:
        ...

    def text_files(
        self, root: str | None = ".", *, file_glob: str | None = None, max_files: int = 500
    ) -> Iterable[str]:
        ...

    def diff_patch(self) -> str:
        ...

    def changed_paths(self) -> list[str]:
        ...

    def changed_entries(self) -> list[ChangedEntry]:
        ...

    def snapshot_current_as_new_baseline(self) -> None:
        """Adopt the current proposed state as the new diff baseline.

        After this call, ``diff_patch()`` / ``changed_entries()`` report only the
        changes made *after* this point. Used by ``AgentLoop.commit_checkpoint()``
        to support incremental apply across a multi-turn run. Mutating workspaces
        only; a read-only workspace raises.
        """
        ...

    def workspace_base_payload(self, run_id: str) -> dict[str, Any]:
        ...


WorkspaceFactory = Callable[[AgentRunSpec], Workspace]
"""Builds the run's :class:`Workspace` from its :class:`AgentRunSpec`.

This is the type of ``AgentLoop.workspace_factory``. The default is
``default_local_workspace_factory`` (``workspace/local.py``), which returns the
local-filesystem backend; pass your own to back the engine with a different
workspace implementation.
"""
