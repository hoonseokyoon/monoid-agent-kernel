from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from native_agent_runner.tasks import list_job_artifacts
from native_agent_runner.permissions import PermissionPolicy
from native_agent_runner.public_view import public_path


def project_run_status(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    status_payload = _read_json_if_exists(run_dir / "status.json")
    metrics = _read_json_if_exists(run_dir / "metrics.json")
    proposal = _read_json_if_exists(run_dir / "proposal.json")
    manifest = _read_json_if_exists(run_dir / "manifest.json")
    package = _read_json_if_exists(run_dir / "proposal.package.json")
    approval = _read_json_if_exists(run_dir / "approval.json")
    apply_result = _read_json_if_exists(run_dir / "apply-result.json")
    permission_policy = PermissionPolicy.from_json(manifest.get("permission_policy"))
    jobs = _public_jobs(list_job_artifacts(run_dir), permission_policy)

    projection: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_id": _first_string(
            status_payload.get("run_id"),
            metrics.get("run_id"),
            proposal.get("run_id"),
            run_dir.name,
        ),
        "status": status_payload.get("status") or metrics.get("status") or "unknown",
        "error_code": status_payload.get("error_code") or metrics.get("error_code") or "",
        "workspace_backend": (
            status_payload.get("workspace_backend")
            or metrics.get("workspace_backend")
            or manifest.get("workspace_backend")
            or ""
        ),
        "waiting_for_background_jobs": bool(status_payload.get("waiting_for_background_jobs", False)),
        "jobs": jobs,
        "running_jobs": [job for job in jobs if job.get("status") == "running"],
        "completed_jobs": [job for job in jobs if job.get("status") != "running"],
        "current_step": status_payload.get("current_step"),
        "current_tool": status_payload.get("current_tool"),
        "agent_config": status_payload.get("agent_config") or manifest.get("agent_config") or {},
        "changed_paths": _public_paths(proposal.get("changed_paths") or [], permission_policy),
        "proposal_hash": proposal.get("proposal_hash"),
        "diff_sha256": proposal.get("diff_sha256"),
        "package_hash": package.get("package_hash"),
        "approval_status": approval.get("decision") or "",
        "approval_hash": approval.get("approval_hash"),
        "apply_status": apply_result.get("status") or "",
        "apply_hash": apply_result.get("apply_hash"),
        "last_event_seq": int(status_payload.get("last_event_seq") or 0),
        "last_event_type": status_payload.get("last_event_type") or "",
    }
    _apply_event_projection(run_dir / "events.jsonl", projection, permission_policy)
    return projection


def _apply_event_projection(
    events_path: Path,
    projection: dict[str, Any],
    permission_policy: PermissionPolicy,
) -> None:
    if not events_path.exists():
        return
    for event in _iter_events(events_path):
        event_type = str(event.get("type") or "")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        seq = int(event.get("seq") or 0)
        if seq >= int(projection.get("last_event_seq") or 0):
            projection["last_event_seq"] = seq
            projection["last_event_type"] = event_type
        if event_type == "run.started":
            projection["status"] = "running"
            projection["workspace_backend"] = data.get("workspace_backend") or projection.get("workspace_backend", "")
        elif event_type == "run.finished":
            projection["status"] = data.get("status") or projection["status"]
            projection["error_code"] = data.get("error_code") or projection["error_code"]
        elif event_type == "run.failed":
            projection["status"] = "failed"
            projection["error_code"] = data.get("error_code") or projection["error_code"]
        elif event_type == "run.waiting":
            projection["status"] = "waiting_for_background_jobs"
            projection["waiting_for_background_jobs"] = True
        elif event_type == "run.resumed":
            projection["status"] = "running"
            projection["waiting_for_background_jobs"] = False
        elif event_type == "agent.config.updated":
            projection["agent_config"] = {
                "definition_id": data.get("definition_id"),
                "config_version": data.get("config_version"),
                "config_hash": data.get("config_hash"),
            }
        elif event_type == "model.turn.started":
            projection["current_step"] = data.get("step")
        elif event_type == "tool.call.started":
            projection["current_tool"] = data.get("tool")
        elif event_type in {"tool.call.finished", "tool.call.failed"}:
            projection["current_tool"] = None
        elif event_type == "workspace.proposal.updated":
            projection["changed_paths"] = _public_paths(
                data.get("changed_paths") or projection["changed_paths"],
                permission_policy,
            )
            projection["proposal_hash"] = data.get("proposal_hash") or projection["proposal_hash"]
            projection["diff_sha256"] = data.get("diff_sha256") or projection["diff_sha256"]
        elif event_type == "proposal.package.exported":
            projection["package_hash"] = data.get("package_hash") or projection["package_hash"]
        elif event_type == "proposal.approved":
            projection["approval_status"] = "approved"
            projection["approval_hash"] = data.get("approval_hash") or projection["approval_hash"]
        elif event_type == "proposal.rejected":
            projection["approval_status"] = "rejected"
            projection["approval_hash"] = data.get("approval_hash") or projection["approval_hash"]
        elif event_type in {"proposal.applied", "proposal.conflict"}:
            projection["apply_status"] = data.get("status") or (
                "conflict" if event_type == "proposal.conflict" else "applied"
            )


def _iter_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _public_paths(paths: object, permission_policy: PermissionPolicy) -> list[str]:
    if not isinstance(paths, list):
        return []
    return [public_path(str(path), permission_policy) for path in paths]


def _public_jobs(jobs: list[dict[str, Any]], permission_policy: PermissionPolicy) -> list[dict[str, Any]]:
    public: list[dict[str, Any]] = []
    for job in jobs:
        payload = {
            key: value
            for key, value in job.items()
            if key not in {"command"}
        }
        payload["changed_paths"] = _public_paths(payload.get("changed_paths") or [], permission_policy)
        public.append(payload)
    return public


def _first_string(*values: object) -> str:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return ""
