from __future__ import annotations

import os
from pathlib import Path

import pytest

from monoid_agent_kernel.errors import ToolExecutionError, WorkspaceError
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.recorder import AgentRecorder
from monoid_agent_kernel.tools.base import ToolRegistry
from monoid_agent_kernel.tools.builtin import builtin_tools
from monoid_agent_kernel.tools.defaults import default_tool_bindings
from monoid_agent_kernel.workspace.local import LocalWorkspaceBackend, sha256_bytes


class DummyContext:
    def emit_artifact(self, path: str, kind: str, label: str | None, metadata: dict[str, object]) -> dict[str, object]:
        return {"path": path, "kind": kind, "label": label, "metadata": metadata}

    def list_artifacts(self) -> list[dict[str, object]]:
        return []

    def update_plan(self, items: list[dict[str, object]]) -> None:
        self.items = items

    def finish(self, summary: str, outputs: list[str], notes: str | None) -> None:
        self.summary = summary
        self.outputs = outputs
        self.notes = notes


class FilteringContext(DummyContext):
    def __init__(self, policy: PermissionPolicy) -> None:
        self.policy = policy

    def path_allowed(self, path: str, operation: str = "read") -> bool:
        try:
            op = operation if operation in {"read", "write", "artifact", "run"} else "read"
            self.policy.check_paths(op, (path,))  # type: ignore[arg-type]
        except Exception:
            return False
        return True


def _registry(workspace: LocalWorkspaceBackend) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_many(builtin_tools(workspace))
    return registry


def test_read_search_and_patch_tools(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("alpha\nbeta\n", encoding="utf-8")
    workspace = LocalWorkspaceBackend(tmp_path, mode="propose")
    registry = _registry(workspace)
    context = DummyContext()

    read = registry.resolve("fs.read")
    result = read.handler(context, {"path": "notes.md"})
    digest = result.content["sha256"]
    assert result.ok
    assert "alpha" in result.content["content"]

    search = registry.resolve("text.search")
    matches = search.handler(context, {"pattern": "beta", "root": "."})
    assert matches.content["count"] == 1

    patch = registry.resolve("fs.patch")
    patched = patch.handler(
        context,
        {
            "path": "notes.md",
            "expected_sha256": digest,
            "replacements": [{"old": "beta", "new": "gamma"}],
        },
    )
    assert patched.ok
    assert tmp_path.joinpath("notes.md").read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert b"gamma" in workspace.read_bytes("notes.md")[0]


def test_patch_hash_mismatch_fails(tmp_path: Path) -> None:
    tmp_path.joinpath("notes.md").write_text("alpha\n", encoding="utf-8")
    workspace = LocalWorkspaceBackend(tmp_path, mode="propose")
    patch = _registry(workspace).resolve("fs.patch")

    with pytest.raises(WorkspaceError):
        patch.handler(
            DummyContext(),
            {
                "path": "notes.md",
                "expected_sha256": sha256_bytes(b"other"),
                "replacements": [{"old": "alpha", "new": "beta"}],
            },
        )


def test_search_includes_secret_looking_files_by_default(tmp_path: Path) -> None:
    tmp_path.joinpath(".env").write_text("SECRET_TOKEN=abc\n", encoding="utf-8")
    tmp_path.joinpath("notes.md").write_text("SECRET_TOKEN is not here\n", encoding="utf-8")
    workspace = LocalWorkspaceBackend(tmp_path, mode="propose")
    search = _registry(workspace).resolve("text.search")

    matches = search.handler(DummyContext(), {"pattern": "SECRET_TOKEN", "root": "."})

    assert matches.content["count"] == 2
    assert {match["path"] for match in matches.content["matches"]} == {".env", "notes.md"}


def test_read_tools_filter_denied_descendant_paths(tmp_path: Path) -> None:
    tmp_path.joinpath(".env").write_text("SECRET_TOKEN=abc\n", encoding="utf-8")
    tmp_path.joinpath("notes.md").write_text("SECRET_TOKEN appears here\n", encoding="utf-8")
    workspace = LocalWorkspaceBackend(tmp_path, mode="propose")
    registry = _registry(workspace)
    context = FilteringContext(PermissionPolicy(deny_patterns=(".env",)))

    search = registry.resolve("text.search").handler(context, {"pattern": "SECRET_TOKEN", "root": "."})
    listed = registry.resolve("fs.list").handler(context, {"path": ".", "recursive": True})
    globbed = registry.resolve("fs.glob").handler(context, {"pattern": "*", "root": "."})
    tree = registry.resolve("fs.tree").handler(context, {"path": "."})

    assert {match["path"] for match in search.content["matches"]} == {"notes.md"}
    assert ".env" not in {entry["path"] for entry in listed.content["entries"]}
    assert ".env" not in globbed.content["matches"]
    assert ".env" not in tree.content["tree"]
    assert search.content["skipped_reasons"]["permission_denied"] == 1


def test_read_range_stat_write_and_patch_options(tmp_path: Path) -> None:
    tmp_path.joinpath("notes.md").write_bytes(b"alpha\nbeta\n")
    workspace = LocalWorkspaceBackend(tmp_path, mode="propose", max_bytes_read=1)
    registry = _registry(workspace)
    context = DummyContext()

    stat = registry.resolve("fs.stat").handler(context, {"path": "notes.md"})
    ranged = registry.resolve("fs.read").handler(
        context,
        {"path": "notes.md", "start_line": 2, "end_line": 2, "max_bytes": 100},
    )

    assert stat.content["size"] == len("alpha\nbeta\n".encode("utf-8"))
    assert ranged.content["content"] == "beta\n"
    assert ranged.content["total_lines"] == 2

    with pytest.raises(WorkspaceError, match="destination already exists"):
        registry.resolve("fs.write").handler(
            context,
            {"path": "notes.md", "content": "replacement", "if_exists": "fail", "max_bytes": 100},
        )
    workspace.max_bytes_read = 100
    with pytest.raises(WorkspaceError, match="must not be empty"):
        registry.resolve("fs.patch").handler(
            context,
            {"path": "notes.md", "replacements": [{"old": "", "new": "x"}]},
        )


def test_directory_replace_mode_and_symlink_safety(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source.joinpath("fresh.txt").write_text("fresh\n", encoding="utf-8")
    dest = tmp_path / "dest"
    dest.mkdir()
    dest.joinpath("stale.txt").write_text("stale\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside.joinpath("target.txt").write_text("outside\n", encoding="utf-8")
    try:
        os.symlink(outside / "target.txt", source / "link.txt")
    except OSError:
        pytest.skip("symlink creation is not available in this environment")

    workspace = LocalWorkspaceBackend(tmp_path, mode="propose")
    registry = _registry(workspace)
    context = DummyContext()

    entries = registry.resolve("fs.list").handler(context, {"path": "source", "recursive": True})
    assert any(entry["path"] == "source/link.txt" and entry["kind"] == "symlink" for entry in entries.content["entries"])

    top_target = tmp_path / "top-target.txt"
    top_target.write_text("inside\n", encoding="utf-8")
    top_link = tmp_path / "top-link.txt"
    os.symlink(top_target, top_link)

    with pytest.raises(WorkspaceError, match="symlink file operations are not supported"):
        registry.resolve("fs.copy").handler(
            context,
            {"source_path": "top-link.txt", "destination_path": "top-copy.txt"},
        )
    with pytest.raises(WorkspaceError, match="symlink file operations are not supported"):
        registry.resolve("fs.move").handler(
            context,
            {"source_path": "top-link.txt", "destination_path": "top-moved.txt"},
        )
    with pytest.raises(WorkspaceError, match="symlink file operations are not supported"):
        registry.resolve("fs.delete").handler(context, {"path": "top-link.txt"})
    assert top_target.exists()
    assert top_link.is_symlink()

    alias_target = tmp_path / "alias-target"
    alias_target.mkdir()
    alias_target.joinpath("file.txt").write_text("target\n", encoding="utf-8")
    alias_dir = tmp_path / "alias-dir"
    try:
        os.symlink(alias_target, alias_dir, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is not available in this environment")

    with pytest.raises(WorkspaceError, match="symlink file operations are not supported"):
        registry.resolve("fs.copy").handler(
            context,
            {"source_path": "alias-dir/file.txt", "destination_path": "alias-copy.txt"},
        )
    with pytest.raises(WorkspaceError, match="symlink file operations are not supported"):
        registry.resolve("fs.move").handler(
            context,
            {"source_path": "alias-dir/file.txt", "destination_path": "alias-move.txt"},
        )
    with pytest.raises(WorkspaceError, match="symlink file operations are not supported"):
        registry.resolve("fs.delete").handler(context, {"path": "alias-dir/file.txt"})
    with pytest.raises(WorkspaceError, match="symlink file operations are not supported"):
        registry.resolve("fs.copy").handler(
            context,
            {"source_path": "source/fresh.txt", "destination_path": "alias-dir/copied.txt"},
        )
    with pytest.raises(WorkspaceError, match="symlink file operations are not supported"):
        workspace.write_bytes("alias-dir/new.txt", b"new\n", create_dirs=True)
    with pytest.raises(WorkspaceError, match="symlink file operations are not supported"):
        workspace.mkdir("alias-dir/new-dir")
    assert alias_target.joinpath("file.txt").read_text(encoding="utf-8") == "target\n"
    assert not alias_target.joinpath("copied.txt").exists()

    with pytest.raises(WorkspaceError, match="symlink file operations are not supported"):
        registry.resolve("fs.copy").handler(
            context,
            {"source_path": "source", "destination_path": "copy", "recursive": True},
        )

    (source / "link.txt").unlink()
    copied = registry.resolve("fs.copy").handler(
        context,
        {
            "source_path": "source",
            "destination_path": "dest",
            "recursive": True,
            "overwrite": True,
            "directory_mode": "replace",
        },
    )
    assert copied.ok
    assert workspace.exists("dest/fresh.txt")
    assert not workspace.exists("dest/stale.txt")


def test_tool_registry_rejects_duplicate_export_names(tmp_path: Path) -> None:
    workspace = LocalWorkspaceBackend(tmp_path)
    registry = ToolRegistry()
    first = builtin_tools(workspace)[0]
    registry.register(first)

    with pytest.raises(ValueError):
        registry.register(first)


def test_default_tool_binding_presets_have_expected_policy() -> None:
    read = default_tool_bindings("read")
    write = default_tool_bindings("write")
    shell = default_tool_bindings("shell")

    assert {"fs.read", "fs.list", "fs.tree", "fs.stat", "fs.glob", "text.search", "fs.read_media"} == {
        binding.ref.tool_id for binding in read
    }
    write_authorizations = {binding.ref.tool_id: binding.authorization for binding in write}
    write_approvals = {binding.ref.tool_id: binding.requires_approval for binding in write}
    assert write_authorizations["fs.write"] == "allow"
    assert write_authorizations["fs.patch"] == "allow"
    assert write_authorizations["fs.mkdir"] == "allow"
    assert write_authorizations["fs.copy"] == "ask"
    assert write_authorizations["fs.move"] == "ask"
    assert write_authorizations["fs.delete"] == "ask"
    assert write_approvals["fs.copy"] is True
    assert write_approvals["fs.move"] is True
    assert write_approvals["fs.delete"] is True
    assert {"shell.exec", "job.status", "job.logs", "job.cancel", "job.wait"} == {
        binding.ref.tool_id for binding in shell
    }
    assert all(binding.title and binding.summary for binding in (*read, *write, *shell))


def test_artifact_recorder_preserves_metadata(tmp_path: Path) -> None:
    recorder = AgentRecorder(tmp_path / "runs", "run_artifacts", status_file=False)
    try:
        artifact = recorder.emit_artifact_bytes(
            workspace_path="chart.svg",
            content=b"<svg />",
            kind="image/svg+xml",
            label="Chart",
            metadata={"source": "unit"},
        )
    finally:
        recorder.close()

    assert artifact.metadata == {"source": "unit"}


def test_file_copy_move_delete_tools_in_propose_mode(tmp_path: Path) -> None:
    tmp_path.joinpath("source.txt").write_bytes(b"alpha")
    workspace = LocalWorkspaceBackend(tmp_path, mode="propose")
    registry = _registry(workspace)
    context = DummyContext()

    copied = registry.resolve("fs.copy").handler(
        context,
        {"source_path": "source.txt", "destination_path": "copy.txt"},
    )
    moved = registry.resolve("fs.move").handler(
        context,
        {"source_path": "copy.txt", "destination_path": "moved.txt"},
    )
    deleted = registry.resolve("fs.delete").handler(context, {"path": "source.txt"})

    assert copied.ok and moved.ok and deleted.ok
    assert tmp_path.joinpath("source.txt").read_bytes() == b"alpha"
    assert not tmp_path.joinpath("copy.txt").exists()
    assert not tmp_path.joinpath("moved.txt").exists()
    assert workspace.exists("source.txt") is False
    assert workspace.exists("copy.txt") is False
    assert workspace.read_bytes("moved.txt")[0] == b"alpha"
    changes = {entry.path: entry.change_kind for entry in workspace.changed_entries()}
    assert changes == {"moved.txt": "created", "source.txt": "deleted"}


def test_file_copy_move_delete_tools_in_apply_mode(tmp_path: Path) -> None:
    tmp_path.joinpath("source.txt").write_bytes(b"alpha")
    workspace = LocalWorkspaceBackend(tmp_path, mode="apply")
    registry = _registry(workspace)
    context = DummyContext()

    registry.resolve("fs.copy").handler(context, {"source_path": "source.txt", "destination_path": "copy.txt"})
    registry.resolve("fs.move").handler(context, {"source_path": "copy.txt", "destination_path": "moved.txt"})
    registry.resolve("fs.delete").handler(context, {"path": "source.txt"})

    assert not tmp_path.joinpath("source.txt").exists()
    assert not tmp_path.joinpath("copy.txt").exists()
    assert tmp_path.joinpath("moved.txt").read_bytes() == b"alpha"


def test_file_operations_collision_recursive_bounds_and_secret_looking_paths(tmp_path: Path) -> None:
    tmp_path.joinpath("source.txt").write_bytes(b"alpha")
    tmp_path.joinpath("target.txt").write_bytes(b"target")
    tree = tmp_path / "tree"
    tree.mkdir()
    tree.joinpath("a.txt").write_bytes(b"a")
    tree.joinpath("sub").mkdir()
    tree.joinpath("sub", "b.txt").write_bytes(b"b")
    workspace = LocalWorkspaceBackend(tmp_path, mode="propose")
    registry = _registry(workspace)
    context = DummyContext()

    with pytest.raises(WorkspaceError, match="destination already exists"):
        registry.resolve("fs.copy").handler(
            context,
            {"source_path": "source.txt", "destination_path": "target.txt"},
        )
    with pytest.raises(WorkspaceError, match="directory requires recursive=true"):
        registry.resolve("fs.copy").handler(
            context,
            {"source_path": "tree", "destination_path": "tree-copy"},
        )
    with pytest.raises(WorkspaceError, match="max entries"):
        registry.resolve("fs.delete").handler(
            context,
            {"path": "tree", "recursive": True, "max_entries": 1},
        )
    tree.joinpath(".env").write_text("SECRET=x\n", encoding="utf-8")
    deleted = registry.resolve("fs.delete").handler(context, {"path": "tree", "recursive": True})
    assert deleted.ok
    assert workspace.exists("tree/.env") is False


def test_file_operation_net_zero_create_delete_and_create_move(tmp_path: Path) -> None:
    workspace = LocalWorkspaceBackend(tmp_path, mode="propose")

    workspace.write_bytes("temp.txt", b"temp")
    workspace.delete_path("temp.txt")
    assert workspace.changed_entries() == []

    workspace.write_bytes("draft.txt", b"draft")
    workspace.move_path("draft.txt", "final.txt")
    changes = workspace.changed_entries()
    assert [(entry.path, entry.change_kind) for entry in changes] == [("final.txt", "created")]
    assert changes[0].content == b"draft"


def test_registry_lookup_by_core_id_and_exported_name(tmp_path: Path) -> None:
    registry = _registry(LocalWorkspaceBackend(tmp_path))

    assert registry.resolve("fs.read").id == "fs.read"
    assert registry.resolve("fs_read").id == "fs.read"
    with pytest.raises(ToolExecutionError):
        registry.resolve("fs.missing")
