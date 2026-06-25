"""Contract every ``Workspace`` backend must honor, regardless of how it stores bytes.

The engine talks to a ``Workspace`` (core/workspace.py) only through this surface; ``test_workspace_
protocol.py`` checks the members *exist*, this suite checks they *behave*. Invariants: writes
round-trip with their sha256, the proposed state is observable (exists/path_kind/list/glob),
least-surprise guards hold (optimistic ``expected_sha256``, ``max_bytes_read``, no absolute/parent
paths), the changed-entry delta reflects edits, and re-baselining collapses the delta. Parametrized
over a workspace factory — a new backend (git-worktree / object-store) is verified by adding one
``pytest.param``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from native_agent_runner.core._util import sha256_bytes
from native_agent_runner.core.workspace import ChangedEntry, FileEntry, Workspace
from native_agent_runner.errors import WorkspaceError
from native_agent_runner.workspace.local import LocalWorkspaceBackend

WorkspaceFactory = Callable[[Path], Workspace]

WORKSPACE_FACTORIES = [
    pytest.param(
        lambda root: LocalWorkspaceBackend(root, mode="propose", backend_kind="overlay"),
        id="local_overlay",
    ),
    pytest.param(
        lambda root: LocalWorkspaceBackend(root, mode="propose", backend_kind="staging"),
        id="local_staging",
    ),
]


@pytest.fixture(params=WORKSPACE_FACTORIES)
def workspace(request: pytest.FixtureRequest, tmp_path: Path) -> Workspace:
    factory: WorkspaceFactory = request.param
    root = tmp_path / "ws"
    root.mkdir()
    return factory(root)


def test_write_read_round_trip_with_hash(workspace: Workspace) -> None:
    data = b"hello workspace"
    write_sha = workspace.write_bytes("notes.md", data)
    read_data, read_sha = workspace.read_bytes("notes.md")
    assert read_data == data
    assert write_sha == read_sha == sha256_bytes(data)


def test_exists_and_path_kind_reflect_proposed_state(workspace: Workspace) -> None:
    assert workspace.exists("a.txt") is False
    assert workspace.path_kind("a.txt") is None
    workspace.write_bytes("a.txt", b"x")
    assert workspace.exists("a.txt") is True
    assert workspace.path_kind("a.txt") == "file"
    workspace.mkdir("sub")
    assert workspace.exists("sub") is True
    assert workspace.path_kind("sub") == "dir"


def test_expected_sha256_is_an_optimistic_guard(workspace: Workspace) -> None:
    # A brand-new path: the precondition is the sha of empty content.
    workspace.write_bytes("c.txt", b"v1", expected_sha256=sha256_bytes(b""))
    # A stale precondition is rejected; the current sha is accepted.
    with pytest.raises(WorkspaceError):
        workspace.write_bytes("c.txt", b"v2", expected_sha256="deadbeef")
    workspace.write_bytes("c.txt", b"v2", expected_sha256=sha256_bytes(b"v1"))
    assert workspace.read_bytes("c.txt")[0] == b"v2"


def test_read_bytes_enforces_max_bytes(workspace: Workspace) -> None:
    workspace.write_bytes("big.bin", b"0123456789")
    with pytest.raises(WorkspaceError):
        workspace.read_bytes("big.bin", max_bytes=1)


def test_normalize_and_path_escape_guards(workspace: Workspace) -> None:
    assert workspace.normalize(".") == "."
    assert workspace.normalize("a/./b") == "a/b"
    with pytest.raises(WorkspaceError):
        workspace.normalize("../escape")
    with pytest.raises(WorkspaceError):
        workspace.normalize("/abs/path")


def test_changed_entries_report_a_created_file(workspace: Workspace) -> None:
    workspace.write_bytes("new.txt", b"fresh")
    assert "new.txt" in workspace.changed_paths()
    entries = {entry.path: entry for entry in workspace.changed_entries()}
    assert "new.txt" in entries
    entry = entries["new.txt"]
    assert isinstance(entry, ChangedEntry)
    assert entry.change_kind == "created"
    assert entry.content == b"fresh"
    assert (entry.proposed_sha256 or entry.sha256) == sha256_bytes(b"fresh")


def test_delete_removes_a_file(workspace: Workspace) -> None:
    workspace.write_bytes("gone.txt", b"bye")
    assert workspace.exists("gone.txt") is True
    workspace.delete_path("gone.txt")
    assert workspace.exists("gone.txt") is False


def test_copy_and_move(workspace: Workspace) -> None:
    workspace.write_bytes("src.txt", b"payload")
    workspace.copy_path("src.txt", "copy.txt")
    assert workspace.exists("src.txt") and workspace.exists("copy.txt")
    assert workspace.read_bytes("copy.txt")[0] == b"payload"

    workspace.move_path("src.txt", "moved.txt")
    assert workspace.exists("src.txt") is False
    assert workspace.exists("moved.txt") is True
    assert workspace.read_bytes("moved.txt")[0] == b"payload"


def test_glob_and_list_entries(workspace: Workspace) -> None:
    workspace.write_bytes("a.txt", b"a")
    workspace.write_bytes("b.txt", b"b")
    workspace.write_bytes("c.md", b"c")
    assert set(workspace.glob("*.txt")) == {"a.txt", "b.txt"}
    listed = workspace.list_entries(".")
    assert all(isinstance(entry, FileEntry) for entry in listed)
    assert {"a.txt", "b.txt", "c.md"} <= {entry.path for entry in listed}


def test_rebaseline_collapses_the_delta(workspace: Workspace) -> None:
    workspace.write_bytes("x.txt", b"one")
    assert workspace.changed_entries()  # there is a pending change
    workspace.snapshot_current_as_new_baseline()
    assert workspace.changed_entries() == []  # the proposed state is now the baseline


def test_workspace_base_payload_shape(workspace: Workspace) -> None:
    payload = workspace.workspace_base_payload("run_1")
    assert payload["run_id"] == "run_1"
    assert "schema_version" in payload
    assert isinstance(payload["entries"], list)
