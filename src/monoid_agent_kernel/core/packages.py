from __future__ import annotations

import base64
import hashlib
import json
import shutil
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from monoid_agent_kernel.core._util import canonical_sha256, utc_timestamp, write_json_atomic
from monoid_agent_kernel.errors import PermissionDenied, WorkspaceError
from monoid_agent_kernel.workspace.paths import is_within, normalize_workspace_path

PACKAGE_SCHEMA_VERSION = "native-agent-runner.proposal-package.v1"
APPROVAL_SCHEMA_VERSION = "native-agent-runner.approval.v1"
APPLY_RESULT_SCHEMA_VERSION = "native-agent-runner.apply-result.v1"

PackageSourceKind = Literal["run_dir", "tar"]
ApprovalDecision = Literal["approved", "rejected"]
ApplyStatus = Literal["dry_run", "applied", "conflict", "rejected"]


@dataclass(frozen=True)
class PackageVerification:
    ok: bool
    issues: tuple[str, ...]
    package: dict[str, Any]
    root: Path
    source_kind: PackageSourceKind


@dataclass(frozen=True)
class ApplyConflict:
    path: str
    reason: str
    expected_sha256: str | None = None
    current_sha256: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "reason": self.reason,
            "expected_sha256": self.expected_sha256,
            "current_sha256": self.current_sha256,
        }


@dataclass(frozen=True)
class ApplyResult:
    status: ApplyStatus
    applied_paths: tuple[str, ...] = ()
    skipped_paths: tuple[str, ...] = ()
    conflicts: tuple[ApplyConflict, ...] = ()
    approval_hash: str = ""
    package_hash: str = ""
    apply_hash: str = ""

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": APPLY_RESULT_SCHEMA_VERSION,
            "status": self.status,
            "applied_paths": list(self.applied_paths),
            "skipped_paths": list(self.skipped_paths),
            "conflicts": [conflict.to_json() for conflict in self.conflicts],
            "approval_hash": self.approval_hash,
            "package_hash": self.package_hash,
        }
        payload["apply_hash"] = self.apply_hash or canonical_sha256(payload, drop=("apply_hash",))
        return payload


@dataclass(frozen=True)
class _PackageSource:
    root: Path
    kind: PackageSourceKind


def export_package(run_dir: Path, output: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    package = build_package_manifest(run_dir)
    write_json_atomic(run_dir / "proposal.package.json", package)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w") as archive:
        _add_deterministic_file(archive, run_dir / "proposal.package.json", "proposal.package.json")
        for rel in _package_paths(package):
            _add_deterministic_file(archive, run_dir / rel, rel)
    return {**package, "package_path": str(output)}


def build_package_manifest(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest = _read_json(run_dir / "manifest.json")
    proposal = _read_json(run_dir / "proposal.json")
    required_paths = ["manifest.json", "proposal.json", "diff.patch"]
    workspace_index_path = manifest.get("workspace_index_path")
    if isinstance(workspace_index_path, str) and workspace_index_path:
        required_paths.append(_safe_package_path(workspace_index_path))
    workspace_base_path = manifest.get("workspace_base_path")
    if isinstance(workspace_base_path, str) and workspace_base_path:
        required_paths.append(_safe_package_path(workspace_base_path))
    snapshot_paths: list[str] = []
    workspace_paths: dict[str, str] = {}
    for file_info in proposal.get("files") or []:
        if not isinstance(file_info, dict):
            continue
        snapshot_path = file_info.get("snapshot_path")
        if isinstance(snapshot_path, str) and snapshot_path:
            rel = _safe_package_path(snapshot_path)
            snapshot_paths.append(rel)
            workspace_paths[rel] = str(file_info.get("path") or "")
    file_entries = [
        _package_file_entry(run_dir, rel, role=_role_for_path(rel), workspace_path=workspace_paths.get(rel))
        for rel in sorted(set(required_paths + snapshot_paths))
    ]
    payload: dict[str, Any] = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "run_id": str(proposal.get("run_id") or manifest.get("run_id") or run_dir.name),
        "created_at": str(manifest.get("created_at") or utc_timestamp()),
        "proposal_hash": str(proposal["proposal_hash"]),
        "diff_sha256": str(proposal["diff_sha256"]),
        "files": file_entries,
    }
    payload["package_hash"] = canonical_sha256(payload, drop=("package_hash",))
    return payload


def verify_package(source: Path) -> PackageVerification:
    with _materialize_source(source) as package_source:
        return _verify_materialized_source(package_source)


def inspect_package(source: Path) -> dict[str, Any]:
    with _materialize_source(source) as package_source:
        verification = _verify_materialized_source(package_source)
        proposal: dict[str, Any] = {}
        proposal_path = verification.root / "proposal.json"
        if proposal_path.exists():
            proposal = _read_json(proposal_path)
        return {
            "ok": verification.ok,
            "issues": list(verification.issues),
            "source_kind": verification.source_kind,
            "package": verification.package,
            "proposal": {
                "proposal_hash": proposal.get("proposal_hash"),
                "changed_paths": proposal.get("changed_paths", []),
                "files": proposal.get("files", []),
            },
        }


def import_package(source: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise WorkspaceError(f"import output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix=f"{output.name}.tmp-", dir=output.parent))
    try:
        if source.is_dir():
            verification = verify_package(source)
            if not verification.ok:
                raise WorkspaceError(f"package verification failed: {'; '.join(verification.issues)}")
            _copy_package_source(verification.root, verification.package, tmp_root)
        else:
            _extract_tar_safely(source, tmp_root)
        imported = verify_package(tmp_root)
        if not imported.ok:
            raise WorkspaceError(f"imported package verification failed: {'; '.join(imported.issues)}")
        tmp_root.replace(output)
        return {
            "ok": True,
            "output": str(output),
            "package_hash": imported.package.get("package_hash", ""),
            "source_kind": imported.source_kind,
        }
    except Exception:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        raise


def create_approval(
    source: Path,
    *,
    approver_id: str,
    decision: ApprovalDecision = "approved",
    approved_paths: tuple[str, ...] | None = None,
    rejected_paths: tuple[str, ...] | None = None,
    note: str = "",
    approved_at: str | None = None,
) -> dict[str, Any]:
    with _materialize_source(source) as package_source:
        verification = _verify_materialized_source(package_source)
        if not verification.ok:
            raise WorkspaceError(f"package verification failed: {'; '.join(verification.issues)}")
        proposal = _read_json(verification.root / "proposal.json")
        changed_paths = tuple(str(path) for path in proposal.get("changed_paths") or ())
        changed_set = set(changed_paths)
        approved = tuple(_normalize_approval_path(path) for path in (approved_paths or ()))
        rejected = tuple(_normalize_approval_path(path) for path in (rejected_paths or ()))
        if decision == "approved" and not approved:
            approved = changed_paths
        if decision == "rejected" and not rejected:
            rejected = changed_paths
        unknown = (set(approved) | set(rejected)) - changed_set
        if unknown:
            raise WorkspaceError(f"approval references unknown paths: {', '.join(sorted(unknown))}")
        overlap = set(approved) & set(rejected)
        if overlap:
            raise WorkspaceError(f"approval paths overlap rejected paths: {', '.join(sorted(overlap))}")
        payload: dict[str, Any] = {
            "schema_version": APPROVAL_SCHEMA_VERSION,
            "approval_id": _approval_id(
                verification.package["package_hash"],
                decision,
                approver_id,
                approved,
                rejected,
            ),
            "decision": decision,
            "package_hash": verification.package["package_hash"],
            "proposal_hash": verification.package["proposal_hash"],
            "approved_paths": sorted(approved),
            "rejected_paths": sorted(rejected),
            "approver_id": approver_id,
            "approved_at": approved_at or utc_timestamp(),
            "note": note,
        }
        payload["approval_hash"] = canonical_sha256(payload, drop=("approval_hash",))
        return payload


def write_approval(path: Path, approval: dict[str, Any]) -> Path:
    write_json_atomic(path, approval)
    return path


def apply_package(
    source: Path,
    *,
    approval: dict[str, Any] | Path,
    target: Path,
    dry_run: bool = False,
) -> ApplyResult:
    with _materialize_source(source) as package_source:
        verification = _verify_materialized_source(package_source)
        if not verification.ok:
            raise WorkspaceError(f"package verification failed: {'; '.join(verification.issues)}")
        approval_payload = _read_json(approval) if isinstance(approval, Path) else dict(approval)
        _verify_approval_for_package(approval_payload, verification.package)
        if approval_payload["decision"] == "rejected":
            return _final_apply_result(
                status="rejected",
                skipped_paths=tuple(_proposal_changed_paths(verification.root)),
                approval_hash=str(approval_payload["approval_hash"]),
                package_hash=str(verification.package["package_hash"]),
            )

        target = target.resolve()
        target.mkdir(parents=True, exist_ok=True)
        proposal = _read_json(verification.root / "proposal.json")
        approved = tuple(str(path) for path in approval_payload.get("approved_paths") or ())
        approved_set = set(approved or _proposal_changed_paths(verification.root))
        approved_files: list[dict[str, Any]] = []
        skipped: list[str] = []
        for file_info in proposal.get("files") or []:
            if not isinstance(file_info, dict):
                continue
            rel = _normalize_approval_path(str(file_info.get("path") or ""))
            if rel not in approved_set:
                skipped.append(rel)
                continue
            approved_files.append(file_info)

        would_apply: list[str] = []
        conflicts: list[ApplyConflict] = []
        approved_deleted_paths = _approved_deleted_paths(approved_files)
        ordered_files = _ordered_apply_files(approved_files)
        for file_info in ordered_files:
            conflict = _apply_one(
                verification.root,
                target,
                file_info,
                dry_run=True,
                approved_deleted_paths=approved_deleted_paths,
            )
            if conflict is not None:
                conflicts.append(conflict)
            else:
                would_apply.append(str(file_info.get("path") or ""))
        status: ApplyStatus
        if conflicts:
            status = "conflict"
            applied: list[str] = []
        elif dry_run:
            status = "dry_run"
            applied = would_apply
        else:
            status = "applied"
            applied = []
            for file_info in ordered_files:
                conflict = _apply_one(
                    verification.root,
                    target,
                    file_info,
                    dry_run=False,
                    approved_deleted_paths=approved_deleted_paths,
                )
                if conflict is not None:
                    raise WorkspaceError(f"unexpected apply conflict after preflight: {conflict.path}")
                applied.append(str(file_info.get("path") or ""))
        return _final_apply_result(
            status=status,
            applied_paths=tuple(applied),
            skipped_paths=tuple(skipped),
            conflicts=tuple(conflicts),
            approval_hash=str(approval_payload["approval_hash"]),
            package_hash=str(verification.package["package_hash"]),
        )


def write_apply_result(path: Path, result: ApplyResult) -> Path:
    write_json_atomic(path, result.to_json())
    return path


@contextmanager
def _materialize_source(source: Path) -> Iterator[_PackageSource]:
    source = source.resolve()
    if source.is_dir():
        yield _PackageSource(source, "run_dir")
        return
    with tempfile.TemporaryDirectory(prefix="native-agent-package-") as tmp:
        root = Path(tmp)
        _extract_tar_safely(source, root)
        yield _PackageSource(root, "tar")


def _package_manifest(root: Path) -> dict[str, Any]:
    path = root / "proposal.package.json"
    if path.exists():
        payload = _read_json(path)
        if not isinstance(payload, dict):
            raise WorkspaceError("proposal.package.json must contain an object")
        return payload
    return build_package_manifest(root)


def _verify_materialized_source(package_source: _PackageSource) -> PackageVerification:
    issues: list[str] = []
    root = package_source.root
    try:
        package = _package_manifest(root)
    except Exception as exc:
        return PackageVerification(False, (str(exc),), {}, root, package_source.kind)
    _verify_package_payload(root, package, issues)
    return PackageVerification(not issues, tuple(issues), package, root, package_source.kind)


def _verify_package_payload(root: Path, package: dict[str, Any], issues: list[str]) -> None:
    if package.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        issues.append("unsupported package schema")
    expected_hash = canonical_sha256(package, drop=("package_hash",))
    if package.get("package_hash") != expected_hash:
        issues.append("package_hash mismatch")
    seen: set[str] = set()
    for file_info in package.get("files") or []:
        if not isinstance(file_info, dict):
            issues.append("package file entry must be an object")
            continue
        rel = str(file_info.get("path") or "")
        try:
            safe_rel = _safe_package_path(rel)
        except WorkspaceError as exc:
            issues.append(str(exc))
            continue
        if safe_rel in seen:
            issues.append(f"duplicate package path: {safe_rel}")
        seen.add(safe_rel)
        path = (root / safe_rel).resolve()
        if not is_within(root.resolve(), path):
            issues.append(f"package path escapes root: {safe_rel}")
            continue
        if not path.is_file():
            issues.append(f"package file missing: {safe_rel}")
            continue
        actual = _sha256_file(path)
        if file_info.get("sha256") != actual:
            issues.append(f"file hash mismatch: {safe_rel}")
    proposal_path = root / "proposal.json"
    if proposal_path.exists():
        proposal = _read_json(proposal_path)
        if package.get("proposal_hash") != proposal.get("proposal_hash"):
            issues.append("proposal_hash mismatch")
        diff_path = root / "diff.patch"
        if diff_path.exists() and package.get("diff_sha256") != _sha256_file(diff_path):
            issues.append("diff_sha256 mismatch")
        for index, file_info in enumerate(proposal.get("files") or []):
            if not isinstance(file_info, dict):
                continue
            snapshot_path = file_info.get("snapshot_path")
            if not isinstance(snapshot_path, str):
                continue
            snapshot = root / _safe_package_path(snapshot_path)
            if not snapshot.is_file():
                issues.append(f"proposal snapshot missing: {snapshot_path}")
                continue
            if file_info.get("snapshot_sha256") != _sha256_file(snapshot):
                issues.append(f"proposal snapshot hash mismatch: {snapshot_path}")
    else:
        issues.append("proposal.json missing")


def _apply_one(
    root: Path,
    target_root: Path,
    file_info: dict[str, Any],
    *,
    dry_run: bool,
    approved_deleted_paths: set[str],
) -> ApplyConflict | None:
    rel = _normalize_approval_path(str(file_info.get("path") or ""))
    change_kind = str(file_info.get("change_kind") or "")
    target_path = _resolve_target_path(target_root, rel)
    current = _target_hash(target_path)
    base_sha = file_info.get("base_sha256")
    if change_kind == "created":
        if target_path.exists():
            return ApplyConflict(rel, "target already exists", None, current)
        if not dry_run:
            _write_snapshot(root, file_info, target_path)
        return None
    if change_kind == "modified":
        if current != base_sha:
            return ApplyConflict(rel, "base hash mismatch", str(base_sha), current)
        if not dry_run:
            _write_snapshot(root, file_info, target_path)
        return None
    if change_kind == "directory":
        if target_path.exists() and not target_path.is_dir():
            return ApplyConflict(rel, "target exists and is not a directory", None, current)
        if not dry_run:
            target_path.mkdir(parents=True, exist_ok=True)
        return None
    if change_kind == "deleted":
        if target_path.exists() and target_path.is_dir():
            conflict = _dir_delete_conflict(target_root, rel, target_path, approved_deleted_paths)
            if conflict is not None:
                return conflict
            if not dry_run:
                target_path.rmdir()
            return None
        if current != base_sha:
            return ApplyConflict(rel, "base hash mismatch", str(base_sha), current)
        if not dry_run and target_path.exists():
            target_path.unlink()
        return None
    return ApplyConflict(rel, f"unsupported change kind: {change_kind}", None, current)


def _approved_deleted_paths(files: list[dict[str, Any]]) -> set[str]:
    return {
        _normalize_approval_path(str(file_info.get("path") or ""))
        for file_info in files
        if str(file_info.get("change_kind") or "") == "deleted"
    }


def _ordered_apply_files(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(file_info: dict[str, Any]) -> tuple[int, int, str]:
        rel = _normalize_approval_path(str(file_info.get("path") or ""))
        change_kind = str(file_info.get("change_kind") or "")
        kind = str(file_info.get("kind") or "")
        if change_kind == "deleted" and kind == "dir":
            return (2, -rel.count("/"), rel)
        if change_kind == "deleted":
            return (1, -rel.count("/"), rel)
        return (0, rel.count("/"), rel)

    return sorted(files, key=key)


def _dir_delete_conflict(
    target_root: Path,
    rel: str,
    target_path: Path,
    approved_deleted_paths: set[str],
) -> ApplyConflict | None:
    for child in sorted(target_path.rglob("*"), key=lambda item: item.as_posix()):
        resolved = child.resolve()
        if not is_within(target_root.resolve(), resolved):
            return ApplyConflict(rel, "directory contains path that escapes target root", None, None)
        child_rel = resolved.relative_to(target_root.resolve()).as_posix()
        if child_rel not in approved_deleted_paths:
            return ApplyConflict(rel, f"directory contains unapproved path: {child_rel}", None, None)
    return None


def _write_snapshot(root: Path, file_info: dict[str, Any], target_path: Path) -> None:
    snapshot_path = file_info.get("snapshot_path")
    if not isinstance(snapshot_path, str) or not snapshot_path:
        raise WorkspaceError(f"snapshot missing for {file_info.get('path')}")
    snapshot = (root / _safe_package_path(snapshot_path)).resolve()
    if not is_within(root.resolve(), snapshot):
        raise PermissionDenied("snapshot path escapes package root")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(snapshot, target_path)


def _verify_approval_for_package(approval: dict[str, Any], package: dict[str, Any]) -> None:
    if approval.get("schema_version") != APPROVAL_SCHEMA_VERSION:
        raise WorkspaceError("unsupported approval schema")
    expected_hash = canonical_sha256(approval, drop=("approval_hash",))
    if approval.get("approval_hash") != expected_hash:
        raise WorkspaceError("approval_hash mismatch")
    if approval.get("package_hash") != package.get("package_hash"):
        raise WorkspaceError("approval package_hash mismatch")
    if approval.get("proposal_hash") != package.get("proposal_hash"):
        raise WorkspaceError("approval proposal_hash mismatch")


def _final_apply_result(
    *,
    status: ApplyStatus,
    applied_paths: tuple[str, ...] = (),
    skipped_paths: tuple[str, ...] = (),
    conflicts: tuple[ApplyConflict, ...] = (),
    approval_hash: str,
    package_hash: str,
) -> ApplyResult:
    base = ApplyResult(
        status=status,
        applied_paths=tuple(sorted(applied_paths)),
        skipped_paths=tuple(sorted(skipped_paths)),
        conflicts=conflicts,
        approval_hash=approval_hash,
        package_hash=package_hash,
    )
    payload = base.to_json()
    return ApplyResult(
        status=status,
        applied_paths=base.applied_paths,
        skipped_paths=base.skipped_paths,
        conflicts=conflicts,
        approval_hash=approval_hash,
        package_hash=package_hash,
        apply_hash=str(payload["apply_hash"]),
    )


def _package_paths(package: dict[str, Any]) -> list[str]:
    return sorted(str(file_info["path"]) for file_info in package.get("files") or [])


def _package_file_entry(
    root: Path,
    rel: str,
    *,
    role: str,
    workspace_path: str | None = None,
) -> dict[str, Any]:
    safe_rel = _safe_package_path(rel)
    path = (root / safe_rel).resolve()
    if not is_within(root.resolve(), path):
        raise PermissionDenied(f"package path escapes run dir: {safe_rel}")
    if not path.is_file():
        raise WorkspaceError(f"package file missing: {safe_rel}")
    entry: dict[str, Any] = {
        "path": safe_rel,
        "role": role,
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if workspace_path is not None:
        entry["workspace_path"] = workspace_path
    return entry


def _role_for_path(rel: str) -> str:
    if rel == "manifest.json":
        return "manifest"
    if rel == "workspace.index.json":
        return "workspace_index"
    if rel == "workspace.base.json":
        return "workspace_base"
    if rel == "proposal.json":
        return "proposal"
    if rel == "diff.patch":
        return "diff"
    if rel.startswith("proposal/files/"):
        return "snapshot"
    return "other"


def _add_deterministic_file(archive: tarfile.TarFile, path: Path, rel: str) -> None:
    data = path.read_bytes()
    info = tarfile.TarInfo(rel)
    info.size = len(data)
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    archive.addfile(info, fileobj=_BytesReader(data))


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


def _extract_tar_safely(source: Path, root: Path) -> None:
    if not source.is_file():
        raise WorkspaceError(f"package source does not exist: {source}")
    seen: set[str] = set()
    with tarfile.open(source, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            rel = _safe_package_path(member.name)
            if rel in seen:
                raise WorkspaceError(f"duplicate package path: {rel}")
            seen.add(rel)
            target = (root / rel).resolve()
            if not is_within(root.resolve(), target):
                raise PermissionDenied(f"tar member escapes package root: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            handle = archive.extractfile(member)
            if handle is None:
                continue
            target.write_bytes(handle.read())


def _copy_package_source(source_root: Path, package: dict[str, Any], output_root: Path) -> None:
    write_json_atomic(output_root / "proposal.package.json", package)
    for rel in _package_paths(package):
        safe_rel = _safe_package_path(rel)
        source = (source_root / safe_rel).resolve()
        target = (output_root / safe_rel).resolve()
        if not is_within(source_root.resolve(), source):
            raise PermissionDenied(f"package path escapes source root: {safe_rel}")
        if not is_within(output_root.resolve(), target):
            raise PermissionDenied(f"package path escapes output root: {safe_rel}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def _read_json(path_or_payload: Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_payload, dict):
        return dict(path_or_payload)
    payload = json.loads(path_or_payload.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise WorkspaceError(f"{path_or_payload.name} must contain an object")
    return payload


def _safe_package_path(raw: str) -> str:
    rel = normalize_workspace_path(raw)
    if rel == ".":
        raise WorkspaceError("package path cannot be workspace root")
    return rel


def _normalize_approval_path(raw: str) -> str:
    rel = normalize_workspace_path(raw)
    if rel == ".":
        raise WorkspaceError("approval path cannot be workspace root")
    return rel


def _resolve_target_path(root: Path, rel: str) -> Path:
    safe_rel = _normalize_approval_path(rel)
    root = root.resolve()
    candidate = root / Path(safe_rel)
    if candidate.exists():
        resolved = candidate.resolve()
    else:
        parent = candidate.parent
        while not parent.exists() and parent != root.parent:
            parent = parent.parent
        if not parent.exists():
            raise WorkspaceError(f"no existing parent for target path: {safe_rel}")
        resolved_parent = parent.resolve()
        if not is_within(root, resolved_parent):
            raise PermissionDenied(f"target parent escapes root: {safe_rel}")
        resolved = resolved_parent / candidate.relative_to(parent)
    if not is_within(root, resolved):
        raise PermissionDenied(f"target path escapes root: {safe_rel}")
    return resolved


def _target_hash(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return _sha256_file(path)


def _proposal_changed_paths(root: Path) -> tuple[str, ...]:
    proposal = _read_json(root / "proposal.json")
    return tuple(str(path) for path in proposal.get("changed_paths") or ())


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _approval_id(
    package_hash: str,
    decision: str,
    approver_id: str,
    approved: tuple[str, ...],
    rejected: tuple[str, ...],
) -> str:
    data = json.dumps(
        {
            "package_hash": package_hash,
            "decision": decision,
            "approver_id": approver_id,
            "approved": sorted(approved),
            "rejected": sorted(rejected),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "approval_" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()[:12]).decode("ascii").rstrip("=")
