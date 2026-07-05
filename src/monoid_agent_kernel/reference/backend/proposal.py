from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core.checkpoint import CheckpointStore
from monoid_agent_kernel.core.packages import (
    apply_package,
    create_approval,
    export_package,
    write_apply_result,
    write_approval,
)
from monoid_agent_kernel.core.proposal_file import ProposalFileError, read_proposal_file_payload
from monoid_agent_kernel.errors import PermissionDenied
from monoid_agent_kernel.reference.backend.ports import RunRecordPort
from monoid_agent_kernel.reference.backend.run_state import (
    record_lifecycle_payload as _record_lifecycle_payload,
)
from monoid_agent_kernel.workspace.paths import is_within

_ARTIFACT_DIGEST_RE = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True)
class ProposalServiceContext:
    authorize_run: Callable[[str, str], None]
    record: Callable[[str], RunRecordPort]
    read_proposal: Callable[[RunRecordPort], dict[str, Any] | None]
    checkpoint_store_provider: Callable[[], CheckpointStore | None]
    emit_backend_event: Callable[..., None]
    allowed_apply_roots_provider: Callable[[], tuple[Path, ...]]


class ProposalService:
    """Reference proposal package, approval, artifact, and apply operations."""

    def __init__(self, context: ProposalServiceContext) -> None:
        self._context = context

    def proposal(self, run_id: str, token: str) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        record = self._context.record(run_id)
        payload = self._context.read_proposal(record)
        if payload is None:
            return {
                "run_id": record.run_id,
                "tenant_id": record.tenant_id,
                **_record_lifecycle_payload(record),
                "ready": False,
                "error": record.error,
            }
        return {
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            **_record_lifecycle_payload(record),
            "ready": True,
            **payload,
        }

    def proposal_diff(self, run_id: str, token: str) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        record = self._context.record(run_id)
        diff_path = record.run_dir / "diff.patch"
        diff = diff_path.read_text(encoding="utf-8") if diff_path.exists() else ""
        return {"run_id": run_id, "ready": diff_path.exists(), "diff": diff}

    def proposal_file(self, run_id: str, token: str, path: str) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        record = self._context.record(run_id)
        proposal = self._context.read_proposal(record)
        if proposal is None:
            raise ValueError("proposal snapshot is not ready")
        try:
            file_payload = read_proposal_file_payload(record.run_dir, proposal, path)
        except ProposalFileError as exc:
            if exc.reason in {"not_found", "snapshot_missing"}:
                raise KeyError(str(exc)) from exc
            if exc.reason == "escapes_run_dir":
                raise PermissionDenied(str(exc)) from exc
            raise ValueError(str(exc)) from exc
        return {
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            **_record_lifecycle_payload(record),
            **file_payload,
        }

    def export_proposal_package(self, run_id: str, token: str) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        record = self._context.record(run_id)
        output = record.run_dir / "proposal.tar"
        payload = export_package(record.run_dir, output)
        tar_bytes = output.read_bytes()
        checkpoint_store = self._checkpoint_store()
        digest = checkpoint_store.put_blob(run_id, tar_bytes)
        self._context.emit_backend_event(
            run_id,
            "proposal.package.exported",
            data={"package_hash": payload["package_hash"], "digest": digest, "size_bytes": len(tar_bytes)},
        )
        return {
            "package_hash": payload["package_hash"],
            "digest": digest,
            "size_bytes": len(tar_bytes),
            "media_type": "application/x-tar",
            "name": "proposal.tar",
        }

    def read_run_artifact(
        self,
        run_id: str,
        token: str,
        digest: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> bytes:
        self._context.authorize_run(run_id, token)
        if not _ARTIFACT_DIGEST_RE.match(digest):
            raise ValueError("digest must be a 64-char sha256 hex string")
        try:
            data = self._checkpoint_store().get_blob(run_id, digest)
        except KeyError as exc:
            raise KeyError(f"artifact not found: {digest}") from exc
        if offset or limit is not None:
            data = data[offset : (None if limit is None else offset + limit)]
        return data

    def approve_proposal(
        self,
        run_id: str,
        token: str,
        *,
        approver_id: str,
        approved_paths: tuple[str, ...] = (),
        note: str = "",
    ) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        record = self._context.record(run_id)
        approval = create_approval(
            record.run_dir,
            approver_id=approver_id,
            approved_paths=approved_paths or None,
            note=note,
        )
        write_approval(record.run_dir / "approval.json", approval)
        self._context.emit_backend_event(
            run_id,
            "proposal.approved",
            data={"approval_hash": approval["approval_hash"], "package_hash": approval["package_hash"]},
        )
        return approval

    def reject_proposal(
        self,
        run_id: str,
        token: str,
        *,
        approver_id: str,
        reason: str,
    ) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        record = self._context.record(run_id)
        approval = create_approval(
            record.run_dir,
            approver_id=approver_id,
            decision="rejected",
            note=reason,
        )
        write_approval(record.run_dir / "approval.json", approval)
        self._context.emit_backend_event(
            run_id,
            "proposal.rejected",
            data={"approval_hash": approval["approval_hash"], "package_hash": approval["package_hash"]},
        )
        return approval

    def apply_proposal(
        self,
        run_id: str,
        token: str,
        *,
        target: Path,
        approval_path: Path | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        allowed_apply_roots = self._context.allowed_apply_roots_provider()
        if not allowed_apply_roots:
            raise PermissionDenied("proposal apply is disabled")
        target = target.resolve()
        if not any(is_within(root, target) for root in allowed_apply_roots):
            raise PermissionDenied(f"apply target is outside allowed roots: {target}")
        record = self._context.record(run_id)
        approval = approval_path or (record.run_dir / "approval.json")
        result = apply_package(record.run_dir, approval=approval, target=target, dry_run=dry_run)
        write_apply_result(record.run_dir / "apply-result.json", result)
        event_type = "proposal.conflict" if result.status == "conflict" else "proposal.applied"
        self._context.emit_backend_event(
            run_id,
            event_type,
            data={
                "status": result.status,
                "approval_hash": result.approval_hash,
                "package_hash": result.package_hash,
                "applied_paths": list(result.applied_paths),
                "conflicts": [conflict.to_json() for conflict in result.conflicts],
            },
            level="warning" if result.status == "conflict" else "info",
        )
        return result.to_json()

    def _checkpoint_store(self) -> CheckpointStore:
        checkpoint_store = self._context.checkpoint_store_provider()
        assert checkpoint_store is not None
        return checkpoint_store
