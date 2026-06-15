from pathlib import Path

from native_agent_runner.core.workspace import Workspace
from native_agent_runner.workspace.local import LocalWorkspaceBackend

# The members the engine relies on. Kept explicit so removing or renaming a
# surface member trips this test rather than silently breaking integrators.
WORKSPACE_MEMBERS = (
    # attributes
    "root",
    "mode",
    "backend_kind",
    "max_bytes_read",
    # methods
    "normalize",
    "resolve_existing_or_parent",
    "path_kind",
    "exists",
    "read_bytes",
    "write_bytes",
    "mkdir",
    "copy_path",
    "move_path",
    "delete_path",
    "list_entries",
    "glob",
    "text_files",
    "diff_patch",
    "changed_paths",
    "changed_entries",
    "workspace_base_payload",
)


def test_local_backend_satisfies_workspace_surface(tmp_path: Path) -> None:
    backend = LocalWorkspaceBackend(tmp_path)
    for name in WORKSPACE_MEMBERS:
        assert hasattr(backend, name), f"LocalWorkspaceBackend is missing {name!r}"


def test_local_backend_is_usable_as_workspace(tmp_path: Path) -> None:
    # Static type compatibility: the concrete backend is accepted where a
    # Workspace is expected.
    workspace: Workspace = LocalWorkspaceBackend(tmp_path)
    assert workspace.normalize(".") == "."
