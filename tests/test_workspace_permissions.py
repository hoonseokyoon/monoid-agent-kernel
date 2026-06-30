from __future__ import annotations

from pathlib import Path

import pytest

from monoid_agent_kernel.errors import PermissionDenied, WorkspaceError
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.workspace.local import LocalWorkspaceBackend


def test_normalizes_and_blocks_parent_escape(tmp_path: Path) -> None:
    workspace = LocalWorkspaceBackend(tmp_path)

    assert workspace.normalize("a\\b.txt") == "a/b.txt"
    with pytest.raises(WorkspaceError):
        workspace.normalize("../secret.txt")


def test_blocks_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")

    workspace = LocalWorkspaceBackend(root)
    with pytest.raises(WorkspaceError):
        workspace.resolve_existing_or_parent("link/file.txt", for_write=True)


def test_permission_policy_default_allows_secret_looking_paths() -> None:
    policy = PermissionPolicy()

    policy.check_paths("read", (".env",))
    policy.check_paths("write", ("nested/token.txt",))
    policy.check_paths("write", ("secret_santa.md",))


def test_permission_policy_denies_explicit_path_patterns() -> None:
    policy = PermissionPolicy(deny_patterns=(".env", "nested/*.txt"))

    with pytest.raises(PermissionDenied):
        policy.check_paths("read", (".env",))
    with pytest.raises(PermissionDenied):
        policy.check_paths("write", ("nested/token.txt",))
    policy.check_paths("write", ("secret_santa.md",))


def test_propose_overlay_does_not_modify_base(tmp_path: Path) -> None:
    notes = tmp_path / "notes.md"
    notes.write_text("old\n", encoding="utf-8")
    workspace = LocalWorkspaceBackend(tmp_path, mode="propose")

    workspace.write_bytes("notes.md", b"new\n")

    assert notes.read_text(encoding="utf-8") == "old\n"
    assert workspace.read_bytes("notes.md")[0] == b"new\n"
    assert "-old" in workspace.diff_patch()
    assert "+new" in workspace.diff_patch()


def test_workspace_base_snapshot_includes_secret_looking_paths_by_default(tmp_path: Path) -> None:
    tmp_path.joinpath("notes.md").write_text("notes\n", encoding="utf-8")
    tmp_path.joinpath(".env").write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")
    tmp_path.joinpath("tokenizer.json").write_text("{}", encoding="utf-8")

    workspace = LocalWorkspaceBackend(tmp_path, mode="propose")
    payload = workspace.workspace_base_payload("run_1")

    assert payload["schema_version"] == "monoid.workspace-base.v1"
    assert payload["workspace_backend"] == "overlay"
    assert any(entry["path"] == "notes.md" and entry["sha256"] for entry in payload["entries"])
    assert any(entry["path"] == ".env" and entry["sha256"] for entry in payload["entries"])
    assert any(entry["path"] == "tokenizer.json" and entry["sha256"] for entry in payload["entries"])
    serialized = str(payload)
    assert ".env" in serialized
    assert "tokenizer.json" in serialized
    assert payload["excluded"] == []


def test_staging_backend_writes_directly_and_diffs_from_base(tmp_path: Path) -> None:
    notes = tmp_path / "notes.md"
    notes.write_text("old\n", encoding="utf-8")
    workspace = LocalWorkspaceBackend(tmp_path, mode="propose", backend_kind="staging")

    workspace.write_bytes("notes.md", b"new\n")
    workspace.write_bytes("created.txt", b"created\n", create_dirs=False)
    workspace.delete_path("created.txt")

    assert notes.read_text(encoding="utf-8") == "new\n"
    assert not tmp_path.joinpath("created.txt").exists()
    assert "-old" in workspace.diff_patch()
    assert "+new" in workspace.diff_patch()
    entries = {entry.path: entry for entry in workspace.changed_entries()}
    assert set(entries) == {"notes.md"}
    assert entries["notes.md"].change_kind == "modified"
    assert entries["notes.md"].base_sha256
    assert entries["notes.md"].proposed_sha256


def test_apply_mode_writes_base(tmp_path: Path) -> None:
    notes = tmp_path / "notes.md"
    notes.write_text("old\n", encoding="utf-8")
    workspace = LocalWorkspaceBackend(tmp_path, mode="apply")

    workspace.write_bytes("notes.md", b"new\n")

    assert notes.read_text(encoding="utf-8") == "new\n"
