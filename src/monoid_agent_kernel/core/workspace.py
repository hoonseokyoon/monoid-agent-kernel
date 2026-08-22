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

from monoid_agent_kernel.core.authority import ActivationWriteAuthority
from monoid_agent_kernel.core.spec import AgentRunSpec, RunMode, WorkspaceBackendKind


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
        overwrite: bool = True,
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
        directory_mode: str = "merge",
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
        directory_mode: str = "merge",
    ) -> dict[str, int | str]:
        ...

    def stat_path(self, path: str | None) -> dict[str, Any]:
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


@dataclass(frozen=True)
class AuthorityBoundWorkspace:
    """Expose a workspace only through one activation's revocable authority.

    The raw backend remains private so built-in tools and retained tool contexts cannot bypass
    the same fence used by the recorder and loop. Reads fail once authority is gone; mutations
    are linearized with ``revoke()`` so no new mutation can start after revocation returns.
    """

    _backend: Workspace
    _write_authority: ActivationWriteAuthority

    @property
    def mode(self) -> RunMode:
        self._write_authority.assert_active()
        return self._backend.mode

    @property
    def backend_kind(self) -> WorkspaceBackendKind:
        self._write_authority.assert_active()
        return self._backend.backend_kind

    @property
    def max_bytes_read(self) -> int:
        self._write_authority.assert_active()
        return self._backend.max_bytes_read

    def normalize(self, path: str | None) -> str:
        self._write_authority.assert_active()
        return self._backend.normalize(path)

    def path_kind(self, path: str | None) -> str | None:
        self._write_authority.assert_active()
        return self._backend.path_kind(path)

    def exists(self, path: str | None) -> bool:
        self._write_authority.assert_active()
        return self._backend.exists(path)

    def read_bytes(
        self, path: str | None, *, max_bytes: int | None = None
    ) -> tuple[bytes, str]:
        self._write_authority.assert_active()
        return self._backend.read_bytes(path, max_bytes=max_bytes)

    def write_bytes(
        self,
        path: str | None,
        data: bytes,
        *,
        create_dirs: bool = False,
        expected_sha256: str | None = None,
        overwrite: bool = True,
    ) -> str:
        return self._write_authority.guard_local_mutation(
            lambda: self._backend.write_bytes(
                path,
                data,
                create_dirs=create_dirs,
                expected_sha256=expected_sha256,
                overwrite=overwrite,
            )
        )

    def mkdir(self, path: str | None) -> str:
        return self._write_authority.guard_local_mutation(lambda: self._backend.mkdir(path))

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
        directory_mode: str = "merge",
    ) -> dict[str, int | str]:
        return self._write_authority.guard_local_mutation(
            lambda: self._backend.copy_path(
                source_path,
                destination_path,
                overwrite=overwrite,
                create_dirs=create_dirs,
                recursive=recursive,
                max_entries=max_entries,
                max_bytes=max_bytes,
                directory_mode=directory_mode,
            )
        )

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
        directory_mode: str = "merge",
    ) -> dict[str, int | str]:
        return self._write_authority.guard_local_mutation(
            lambda: self._backend.move_path(
                source_path,
                destination_path,
                overwrite=overwrite,
                create_dirs=create_dirs,
                recursive=recursive,
                max_entries=max_entries,
                max_bytes=max_bytes,
                directory_mode=directory_mode,
            )
        )

    def stat_path(self, path: str | None) -> dict[str, Any]:
        self._write_authority.assert_active()
        return self._backend.stat_path(path)

    def delete_path(
        self,
        path: str | None,
        *,
        recursive: bool = False,
        max_entries: int = 1000,
        max_bytes: int = 50_000_000,
    ) -> dict[str, int | str]:
        return self._write_authority.guard_local_mutation(
            lambda: self._backend.delete_path(
                path,
                recursive=recursive,
                max_entries=max_entries,
                max_bytes=max_bytes,
            )
        )

    def list_entries(
        self, path: str | None = ".", *, recursive: bool = False, max_entries: int = 200
    ) -> list[FileEntry]:
        self._write_authority.assert_active()
        return self._backend.list_entries(path, recursive=recursive, max_entries=max_entries)

    def glob(
        self, pattern: str, *, root: str | None = ".", max_matches: int = 200
    ) -> list[str]:
        self._write_authority.assert_active()
        return self._backend.glob(pattern, root=root, max_matches=max_matches)

    def text_files(
        self, root: str | None = ".", *, file_glob: str | None = None, max_files: int = 500
    ) -> Iterable[str]:
        self._write_authority.assert_active()
        # Materialize while authority is known active. Returning the backend's lazy iterator would
        # let an abandoned handler continue filesystem reads after this method returned.
        return tuple(self._backend.text_files(root, file_glob=file_glob, max_files=max_files))

    def diff_patch(self) -> str:
        self._write_authority.assert_active()
        return self._backend.diff_patch()

    def changed_paths(self) -> list[str]:
        self._write_authority.assert_active()
        return self._backend.changed_paths()

    def changed_entries(self) -> list[ChangedEntry]:
        self._write_authority.assert_active()
        return self._backend.changed_entries()

    def snapshot_current_as_new_baseline(self) -> None:
        self._write_authority.guard_local_mutation(
            self._backend.snapshot_current_as_new_baseline
        )

    def workspace_base_payload(self, run_id: str) -> dict[str, Any]:
        self._write_authority.assert_active()
        return self._backend.workspace_base_payload(run_id)


def _workspace_root_path(workspace: Workspace) -> Path:
    """Return a native root only to trusted kernel adapters.

    The authority-bound proxy has no public ``root`` or path-resolving API because a retained
    ``Path`` would remain writable after revocation. Direct shell and workspace-index adapters use
    this private boundary and still receive an authority check before the path leaves the proxy.
    """

    if isinstance(workspace, AuthorityBoundWorkspace):
        workspace._write_authority.assert_active()
        return workspace._backend.root
    return workspace.root


WorkspaceFactory = Callable[[AgentRunSpec], Workspace]
"""Builds the run's :class:`Workspace` from its :class:`AgentRunSpec`.

This is the type of ``AgentLoop.workspace_factory``. The default is
``default_local_workspace_factory`` (``workspace/local.py``), which returns the
local-filesystem backend; pass your own to back the engine with a different
workspace implementation.
"""
