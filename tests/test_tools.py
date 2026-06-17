from __future__ import annotations

from pathlib import Path

import pytest

from native_agent_runner.errors import ToolExecutionError, WorkspaceError
from native_agent_runner.tools.base import ToolRegistry
from native_agent_runner.tools.builtin import builtin_tools
from native_agent_runner.workspace.local import LocalWorkspaceBackend, sha256_bytes


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


def test_tool_registry_rejects_duplicate_export_names(tmp_path: Path) -> None:
    workspace = LocalWorkspaceBackend(tmp_path)
    registry = ToolRegistry()
    first = builtin_tools(workspace)[0]
    registry.register(first)

    with pytest.raises(ValueError):
        registry.register(first)


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
