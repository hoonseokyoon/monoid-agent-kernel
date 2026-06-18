from __future__ import annotations

import json
import shutil
import tarfile
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from click.testing import CliRunner

from conftest import runtime_config, runtime_provider

from native_agent_runner.reference.backend.http import create_backend_server
from native_agent_runner.reference.backend.service import BackendRunRequest, RunnerBackend
from native_agent_runner.reference._shared.tokens import TokenManager
from native_agent_runner.cli import main
from native_agent_runner.core.packages import (
    apply_package,
    create_approval,
    export_package,
    import_package,
    verify_package,
    write_approval,
)
from native_agent_runner.core.schemas import validate_run_dir
from native_agent_runner.core.spec import AgentRunSpec
from native_agent_runner.errors import PermissionDenied, WorkspaceError
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelTurn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call


def _provider(*tool_ids: str):
    return runtime_provider(runtime_config(*(tool_ids or ("fs.write", "fs.delete", "run.finish"))))


def _run_with_created_and_modified_files(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_bytes(b"alpha\n")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call(
                        "fs_write",
                        {"path": "notes.md", "content": "beta\n", "create_dirs": False},
                        "call_notes",
                    ),
                    fake_tool_call(
                        "fs_write",
                        {"path": "SUMMARY.md", "content": "summary\n", "create_dirs": False},
                        "call_summary",
                    ),
                ),
            ),
            ModelTurn(final_text="done"),
        ]
    )
    result = AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace,
            run_root=tmp_path / "runs",
        ),
        model_adapter=adapter,
        runtime_config_provider=_provider("fs.write", "run.finish"),
    ).run_once("Prepare package.")
    return workspace, result.run_dir


def test_package_export_verify_stable_and_tamper_detection(tmp_path: Path) -> None:
    _workspace, run_dir = _run_with_created_and_modified_files(tmp_path)
    first_tar = tmp_path / "proposal-1.tar"
    second_tar = tmp_path / "proposal-2.tar"

    first = export_package(run_dir, first_tar)
    second = export_package(run_dir, second_tar)

    assert first["package_hash"] == second["package_hash"]
    assert first_tar.read_bytes() == second_tar.read_bytes()
    assert verify_package(run_dir).ok is True
    tar_verification = verify_package(first_tar)
    assert tar_verification.ok is True
    assert tar_verification.package["package_hash"] == first["package_hash"]
    assert run_dir.joinpath("proposal.package.json").exists()
    assert validate_run_dir(run_dir) == []

    run_dir.joinpath("proposal", "files", "SUMMARY.md").write_text("tampered\n", encoding="utf-8")
    tampered = verify_package(run_dir)
    assert tampered.ok is False
    assert any("hash mismatch" in issue for issue in tampered.issues)


def test_package_import_roundtrip_and_path_traversal_rejection(tmp_path: Path) -> None:
    _workspace, run_dir = _run_with_created_and_modified_files(tmp_path)
    package_path = tmp_path / "proposal.tar"
    exported = export_package(run_dir, package_path)
    imported_dir = tmp_path / "imported"

    imported = import_package(package_path, imported_dir)

    assert imported["package_hash"] == exported["package_hash"]
    assert verify_package(imported_dir).ok is True
    assert imported_dir.joinpath("proposal.package.json").exists()
    assert imported_dir.joinpath("workspace.index.json").exists()

    malicious = tmp_path / "malicious.tar"
    with tarfile.open(malicious, "w") as archive:
        info = tarfile.TarInfo("../evil.txt")
        data = b"evil"
        info.size = len(data)
        archive.addfile(info, fileobj=_BytesReader(data))

    try:
        import_package(malicious, tmp_path / "bad-import")
    except WorkspaceError as exc:
        assert "parent traversal" in str(exc)
    else:
        raise AssertionError("malicious package should be rejected")


def test_approval_and_reference_apply_full_partial_dry_run_and_conflict(tmp_path: Path) -> None:
    _workspace, run_dir = _run_with_created_and_modified_files(tmp_path)
    export_package(run_dir, tmp_path / "proposal.tar")
    full = create_approval(
        run_dir,
        approver_id="user_a",
        approved_at="2026-06-14T00:00:00Z",
    )
    full_again = create_approval(
        run_dir,
        approver_id="user_a",
        approved_at="2026-06-14T00:00:00Z",
    )
    assert full["approval_hash"] == full_again["approval_hash"]
    approval_path = write_approval(run_dir / "approval.json", full)

    target = tmp_path / "target"
    target.mkdir()
    target.joinpath("notes.md").write_bytes(b"alpha\n")
    dry_run = apply_package(run_dir, approval=approval_path, target=target, dry_run=True)
    assert dry_run.status == "dry_run"
    assert not target.joinpath("SUMMARY.md").exists()
    assert target.joinpath("notes.md").read_bytes() == b"alpha\n"

    applied = apply_package(run_dir, approval=approval_path, target=target)
    assert applied.status == "applied"
    assert target.joinpath("notes.md").read_bytes() == b"beta\n"
    assert target.joinpath("SUMMARY.md").read_text(encoding="utf-8") == "summary\n"

    partial = create_approval(
        run_dir,
        approver_id="user_a",
        approved_paths=("SUMMARY.md",),
        approved_at="2026-06-14T00:00:00Z",
    )
    partial_target = tmp_path / "partial-target"
    partial_target.mkdir()
    partial_target.joinpath("notes.md").write_bytes(b"alpha\n")
    partial_result = apply_package(run_dir, approval=partial, target=partial_target)
    assert partial_result.status == "applied"
    assert partial_target.joinpath("notes.md").read_bytes() == b"alpha\n"
    assert partial_target.joinpath("SUMMARY.md").read_text(encoding="utf-8") == "summary\n"
    assert "notes.md" in partial_result.skipped_paths

    conflict_target = tmp_path / "conflict-target"
    conflict_target.mkdir()
    conflict_target.joinpath("notes.md").write_bytes(b"changed\n")
    conflict = apply_package(run_dir, approval=full, target=conflict_target)
    assert conflict.status == "conflict"
    assert conflict.conflicts[0].path == "notes.md"
    assert not conflict_target.joinpath("SUMMARY.md").exists()


def test_package_apply_deleted_file_and_directory_entries(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("old.txt").write_bytes(b"old\n")
    doomed = workspace / "doomed"
    doomed.mkdir()
    doomed.joinpath("a.txt").write_bytes(b"a\n")
    doomed.joinpath("sub").mkdir()
    doomed.joinpath("sub", "b.txt").write_bytes(b"b\n")
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="turn_1",
                tool_calls=(
                    fake_tool_call("fs_delete", {"path": "old.txt"}, "delete_file"),
                    fake_tool_call("fs_delete", {"path": "doomed", "recursive": True}, "delete_dir"),
                ),
            ),
            ModelTurn(final_text="done"),
        ]
    )

    result = AgentLoop(
        spec=AgentRunSpec(
            workspace_root=workspace,
            run_root=tmp_path / "runs",
        ),
        model_adapter=adapter,
        runtime_config_provider=_provider("fs.delete", "run.finish"),
    ).run_once("Delete old files.")

    assert result.status == "completed"
    assert workspace.joinpath("old.txt").exists()
    assert workspace.joinpath("doomed", "a.txt").exists()
    proposal = json.loads(result.proposal_path.read_text(encoding="utf-8"))
    changes = {file["path"]: (file["kind"], file["change_kind"]) for file in proposal["files"]}
    assert changes["old.txt"] == ("missing", "deleted")
    assert changes["doomed"] == ("dir", "deleted")
    assert changes["doomed/a.txt"] == ("missing", "deleted")
    assert changes["doomed/sub"] == ("dir", "deleted")
    assert verify_package(result.run_dir).ok is True

    approval = create_approval(result.run_dir, approver_id="user_a", approved_at="2026-06-14T00:00:00Z")
    target = tmp_path / "target"
    shutil.copytree(workspace, target)
    applied = apply_package(result.run_dir, approval=approval, target=target)
    assert applied.status == "applied"
    assert not target.joinpath("old.txt").exists()
    assert not target.joinpath("doomed").exists()

    partial = create_approval(
        result.run_dir,
        approver_id="user_a",
        approved_paths=("doomed",),
        approved_at="2026-06-14T00:00:00Z",
    )
    partial_target = tmp_path / "partial-target"
    shutil.copytree(workspace, partial_target)
    conflict = apply_package(result.run_dir, approval=partial, target=partial_target)
    assert conflict.status == "conflict"
    assert conflict.conflicts[0].path == "doomed"
    assert "unapproved path" in conflict.conflicts[0].reason

    mismatch_target = tmp_path / "mismatch-target"
    shutil.copytree(workspace, mismatch_target)
    mismatch_target.joinpath("old.txt").write_bytes(b"changed\n")
    mismatch = apply_package(result.run_dir, approval=approval, target=mismatch_target)
    assert mismatch.status == "conflict"
    assert any(conflict.path == "old.txt" for conflict in mismatch.conflicts)


def test_cli_package_workflow(tmp_path: Path) -> None:
    _workspace, run_dir = _run_with_created_and_modified_files(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    target.joinpath("notes.md").write_bytes(b"alpha\n")
    runner = CliRunner()
    package_path = tmp_path / "proposal.tar"
    approval_path = tmp_path / "approval.json"

    exported = runner.invoke(main, ["package", "export", str(run_dir), "--output", str(package_path), "--json"])
    assert exported.exit_code == 0, exported.output
    verified = runner.invoke(main, ["package", "verify", str(package_path), "--json"])
    assert verified.exit_code == 0, verified.output
    assert json.loads(verified.stdout)["ok"] is True
    inspected = runner.invoke(main, ["package", "inspect", str(package_path), "--json"])
    assert inspected.exit_code == 0, inspected.output
    assert "SUMMARY.md" in json.dumps(json.loads(inspected.stdout))
    imported_dir = tmp_path / "imported"
    imported = runner.invoke(
        main,
        ["package", "import", str(package_path), "--output", str(imported_dir), "--json"],
    )
    assert imported.exit_code == 0, imported.output
    assert json.loads(imported.stdout)["package_hash"] == json.loads(exported.stdout)["package_hash"]
    assert imported_dir.joinpath("proposal.package.json").exists()
    approved = runner.invoke(
        main,
        [
            "package",
            "approve",
            str(run_dir),
            "--approver",
            "user_a",
            "--output",
            str(approval_path),
            "--json",
        ],
    )
    assert approved.exit_code == 0, approved.output
    dry_run = runner.invoke(
        main,
        [
            "package",
            "apply",
            str(run_dir),
            "--approval",
            str(approval_path),
            "--target",
            str(target),
            "--dry-run",
            "--json",
        ],
    )
    assert dry_run.exit_code == 0, dry_run.output
    assert json.loads(dry_run.stdout)["status"] == "dry_run"
    applied = runner.invoke(
        main,
        [
            "package",
            "apply",
            str(run_dir),
            "--approval",
            str(approval_path),
            "--target",
            str(target),
            "--json",
        ],
    )
    assert applied.exit_code == 0, applied.output
    assert json.loads(applied.stdout)["status"] == "applied"
    assert target.joinpath("SUMMARY.md").read_text(encoding="utf-8") == "summary\n"
    assert validate_run_dir(run_dir) == []


def test_backend_package_endpoints_auth_disabled_apply_and_allowed_apply(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_bytes(b"alpha\n")
    target = tmp_path / "target"
    target.mkdir()
    target.joinpath("notes.md").write_bytes(b"alpha\n")
    token_manager = TokenManager.from_secret("z" * 32)

    def factory(_spec, _llm_gateway_token):
        return FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="turn_1",
                    tool_calls=(
                        fake_tool_call(
                            "fs_write",
                            {"path": "SUMMARY.md", "content": "backend summary\n", "create_dirs": False},
                            "call_write",
                        ),
                    ),
                ),
                ModelTurn(final_text="done"),
            ]
        )

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        allowed_apply_roots=(target,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
    )
    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        created = _json_post(
            f"{base_url}/v1/runs",
            {
                "tenant_id": "tenant_a",
                "user_id": "user_a",
                "workspace_root": str(workspace),
                "instruction": "Create summary.",
                "runtime_config": runtime_config("fs.write", "run.finish").to_json(),
            },
            token="admin",
        )
        run_id = created["run_id"]
        run_token = created["run_token"]
        assert backend.wait_for_run(run_id, timeout_s=5) == "completed"

        with pytest.raises(HTTPError) as exc_info:
            _json_post(f"{base_url}/v1/runs/{run_id}/proposal/export", {})
        assert exc_info.value.code == 401

        exported = _json_post(f"{base_url}/v1/runs/{run_id}/proposal/export", {}, token=run_token)
        assert exported["package_hash"]
        approval = _json_post(
            f"{base_url}/v1/runs/{run_id}/proposal/approve",
            {"approver_id": "user_a"},
            token=run_token,
        )
        assert approval["decision"] == "approved"
        applied = _json_post(
            f"{base_url}/v1/runs/{run_id}/proposal/apply",
            {"target": str(target)},
            token=run_token,
        )
        assert applied["status"] == "applied"
        assert target.joinpath("SUMMARY.md").read_text(encoding="utf-8") == "backend summary\n"
        events = _json_get(f"{base_url}/v1/runs/{run_id}/events", token=run_token)["events"]
        assert "proposal.package.exported" in [event["type"] for event in events]
        assert "proposal.approved" in [event["type"] for event in events]
        assert "proposal.applied" in [event["type"] for event in events]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_backend_apply_endpoint_disabled_without_apply_roots(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_bytes(b"alpha\n")
    token_manager = TokenManager.from_secret("q" * 32)
    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=lambda _spec, _token: FakeModelAdapter(turns=[ModelTurn(final_text="done")]),
    )
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="Finish.",
            runtime_config=runtime_config("run.finish"),
        )
    )
    backend.wait_for_run(submission.run_id, timeout_s=5)
    backend.export_proposal_package(submission.run_id, submission.run_token)
    approval = backend.approve_proposal(submission.run_id, submission.run_token, approver_id="user_a")
    assert approval["decision"] == "approved"
    with pytest.raises(PermissionDenied, match="proposal apply is disabled"):
        backend.apply_proposal(
            submission.run_id,
            submission.run_token,
            target=tmp_path / "target",
        )


def test_backend_package_apply_endpoint_handles_deletion_package(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("old.txt").write_bytes(b"old\n")
    target = tmp_path / "target"
    shutil.copytree(workspace, target)
    token_manager = TokenManager.from_secret("d" * 32)

    def factory(_spec, _llm_gateway_token):
        return FakeModelAdapter(
            turns=[
                ModelTurn(
                    response_id="turn_1",
                    tool_calls=(fake_tool_call("fs_delete", {"path": "old.txt"}, "delete_old"),),
                ),
                ModelTurn(final_text="done"),
            ]
        )

    backend = RunnerBackend(
        run_root=tmp_path / "runs",
        token_manager=token_manager,
        allowed_workspace_roots=(workspace,),
        allowed_apply_roots=(target,),
        llm_gateway_url="http://llm-gateway.internal/v1/turns",
        model_adapter_factory=factory,
    )
    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        created = _json_post(
            f"{base_url}/v1/runs",
            {
                "tenant_id": "tenant_a",
                "user_id": "user_a",
                "workspace_root": str(workspace),
                "instruction": "Delete old file.",
                "runtime_config": runtime_config("fs.delete", "run.finish").to_json(),
            },
            token="admin",
        )
        run_id = created["run_id"]
        run_token = created["run_token"]
        assert backend.wait_for_run(run_id, timeout_s=5) == "completed"

        _json_post(f"{base_url}/v1/runs/{run_id}/proposal/export", {}, token=run_token)
        _json_post(
            f"{base_url}/v1/runs/{run_id}/proposal/approve",
            {"approver_id": "user_a"},
            token=run_token,
        )
        applied = _json_post(
            f"{base_url}/v1/runs/{run_id}/proposal/apply",
            {"target": str(target)},
            token=run_token,
        )
        assert applied["status"] == "applied"
        assert not target.joinpath("old.txt").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _json_post(url: str, payload: dict, *, token: str | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, data=body, headers=headers, method="POST")
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _json_get(url: str, *, token: str) -> dict:
    request = Request(url, headers={"Authorization": f"Bearer {token}"}, method="GET")
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


class _BytesReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._data) - self._offset
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk
