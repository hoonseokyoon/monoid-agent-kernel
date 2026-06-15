"""Resolve a single proposal file's content payload from a run's snapshots.

This domain logic was duplicated between the CLI (`package show-file`) and the
reference backend (`BackendService.proposal_file`). It lives in core so both
callers depend on one implementation; each maps `ProposalFileError.reason` to its
own error type and decorates the returned payload as needed.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

from native_agent_runner.workspace.paths import is_within, normalize_workspace_path

# Stable reason codes callers map to their own error types.
PROPOSAL_FILE_REASONS = (
    "invalid_files",
    "not_found",
    "not_a_file",
    "escapes_run_dir",
    "snapshot_missing",
)


class ProposalFileError(Exception):
    """A proposal file payload could not be resolved.

    `reason` is one of ``PROPOSAL_FILE_REASONS``; callers translate it to their
    own error type (CLI: ``ClickException``; backend: ``KeyError``/``ValueError``/
    ``PermissionDenied``).
    """

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


def read_proposal_file_payload(
    run_dir: Path,
    proposal: dict[str, Any],
    file_path: str,
) -> dict[str, Any]:
    """Return ``{path, kind, size, sha256, encoding, content}`` for one proposal file.

    `proposal` is the parsed ``proposal.json`` payload; `run_dir` is the run
    directory holding the snapshots. Raises ``ProposalFileError`` on any failure.
    """
    rel = normalize_workspace_path(file_path)
    files = proposal.get("files")
    if not isinstance(files, list):
        raise ProposalFileError("proposal has no files array", reason="invalid_files")
    file_info = next(
        (item for item in files if isinstance(item, dict) and item.get("path") == rel),
        None,
    )
    if file_info is None:
        raise ProposalFileError(f"proposal file not found: {rel}", reason="not_found")
    snapshot_path = file_info.get("snapshot_path")
    if not isinstance(snapshot_path, str) or not snapshot_path:
        raise ProposalFileError(f"proposal path is not a file: {rel}", reason="not_a_file")
    abs_path = (run_dir / snapshot_path).resolve()
    if not is_within(run_dir.resolve(), abs_path):
        raise ProposalFileError(
            "proposal snapshot path escapes run directory", reason="escapes_run_dir"
        )
    if not abs_path.exists() or not abs_path.is_file():
        raise ProposalFileError(f"proposal snapshot file not found: {rel}", reason="snapshot_missing")
    data = abs_path.read_bytes()
    payload: dict[str, Any] = {
        "path": rel,
        "kind": "file",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    try:
        payload["encoding"] = "utf-8"
        payload["content"] = data.decode("utf-8")
    except UnicodeDecodeError:
        payload["encoding"] = "base64"
        payload["content"] = base64.b64encode(data).decode("ascii")
    return payload
