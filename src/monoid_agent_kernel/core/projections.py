from __future__ import annotations

from pathlib import Path
from typing import Any

from monoid_agent_kernel.core._event_log import read_committed_event_payloads
from monoid_agent_kernel.core.json_ingress import loads_json_ingress
from monoid_agent_kernel.core.lifecycle import (
    SessionState,
    session_state_from_run_status,
    session_state_value,
)
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.public_view import public_path
from monoid_agent_kernel.tasks import public_job_artifacts, run_permission_policy


def project_run_status(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    status_payload = _read_json_if_exists(run_dir / "status.json")
    metrics = _read_json_if_exists(run_dir / "metrics.json")
    proposal = _read_json_if_exists(run_dir / "proposal.json")
    manifest = _read_json_if_exists(run_dir / "manifest.json")
    package = _read_json_if_exists(run_dir / "proposal.package.json")
    approval = _read_json_if_exists(run_dir / "approval.json")
    apply_result = _read_json_if_exists(run_dir / "apply-result.json")
    permission_policy = run_permission_policy(run_dir)
    # Already projected by the reader. There used to be a `_public_jobs` here that applied its own
    # partial version of the same rules -- it dropped `command` and redacted `changed_paths` but
    # left `cwd` exact -- and a second pass now would double-truncate what the reader already cut.
    jobs = public_job_artifacts(run_dir)
    state = _payload_state(status_payload, metrics)
    terminal = _payload_terminal(status_payload, state)

    projection: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_id": _first_string(
            status_payload.get("run_id"),
            metrics.get("run_id"),
            proposal.get("run_id"),
            run_dir.name,
        ),
        "state": state,
        "terminal": terminal,
        "error_code": status_payload.get("error_code") or metrics.get("error_code") or "",
        "workspace_backend": (
            status_payload.get("workspace_backend")
            or metrics.get("workspace_backend")
            or manifest.get("workspace_backend")
            or ""
        ),
        "waiting_for_background_jobs": _payload_bool(
            status_payload,
            "waiting_for_background_jobs",
            default=False,
        ),
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
        "last_event_seq": _payload_nonnegative_int(
            status_payload,
            "last_event_seq",
            default=0,
        ),
        "last_event_type": status_payload.get("last_event_type") or "",
        # Empty on a clean read. A degraded result combines the run's durable snapshots with the
        # valid event prefix, so none of its state fields can be trusted as current without first
        # checking this member.
        "event_log_error": "",
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
    read = read_committed_event_payloads(events_path)
    projection["event_log_error"] = read.corruption
    for event in read.payloads:
        event_type = _optional_text(event.get("type"), "event type") or ""
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        seq = _payload_nonnegative_int(event, "seq", default=0)
        if seq >= projection["last_event_seq"]:
            projection["last_event_seq"] = seq
            projection["last_event_type"] = event_type
        if event_type == "run.started":
            projection["state"] = session_state_value(SessionState.RUNNING)
            projection["terminal"] = False
            projection["workspace_backend"] = data.get("workspace_backend") or projection.get(
                "workspace_backend", ""
            )
        elif event_type == "run.finished":
            projection["state"] = session_state_value(
                session_state_from_run_status(
                    _optional_text(data.get("status"), "run.finished status")
                    or projection["state"],
                    error_code=(
                        _optional_text(data.get("error_code"), "run.finished error_code")
                        or projection["error_code"]
                        or ""
                    ),
                    terminal=True,
                )
            )
            projection["terminal"] = True
            projection["error_code"] = data.get("error_code") or projection["error_code"]
        elif event_type == "run.failed":
            projection["state"] = session_state_value(SessionState.FAILED)
            projection["terminal"] = True
            projection["error_code"] = data.get("error_code") or projection["error_code"]
        elif event_type == "run.waiting":
            projection["state"] = session_state_value(SessionState.AWAITING_TASKS)
            projection["terminal"] = False
            projection["waiting_for_background_jobs"] = True
        elif event_type == "run.resumed":
            projection["state"] = session_state_value(SessionState.RUNNING)
            projection["terminal"] = False
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


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = loads_json_ingress(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _payload_state(status_payload: dict[str, Any], metrics: dict[str, Any]) -> str:
    raw = _first_present_text(
        (status_payload, "state"),
        (status_payload, "status"),
        (metrics, "state"),
        (metrics, "status"),
        field_name="run state",
    )
    if not raw:
        return session_state_value(SessionState.CREATED)
    error_code = (
        _first_present_text(
            (status_payload, "error_code"),
            (metrics, "error_code"),
            field_name="run error_code",
        )
        or ""
    )
    state = session_state_from_run_status(
        raw,
        error_code=error_code,
        terminal=_payload_bool(status_payload, "terminal", default=False),
    )
    return session_state_value(state)


def _payload_terminal(status_payload: dict[str, Any], state: str) -> bool:
    if "terminal" in status_payload:
        return _payload_bool(status_payload, "terminal", default=False)
    raw_state = (
        _first_present_text(
            (status_payload, "state"),
            (status_payload, "status"),
            field_name="run state",
        )
        or state
    )
    return raw_state in {"completed", "failed", "limited", "cancelled"} or state in {
        SessionState.CANCELLED.value,
        SessionState.FAILED.value,
        SessionState.COMPLETED.value,
    }


def _public_paths(paths: object, permission_policy: PermissionPolicy) -> list[str]:
    if not isinstance(paths, list):
        return []
    return [public_path(str(path), permission_policy) for path in paths]


def _first_string(*values: object) -> str:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return ""


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError(f"{field_name} must be a string")
    return value


def _first_present_text(
    *candidates: tuple[dict[str, Any], str],
    field_name: str,
) -> str | None:
    for payload, key in candidates:
        if key not in payload or payload[key] is None:
            continue
        value = _optional_text(payload[key], field_name)
        if value:
            return value
    return None


def _payload_bool(payload: dict[str, Any], key: str, *, default: bool) -> bool:
    if key not in payload:
        return default
    value = payload[key]
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value


def _payload_nonnegative_int(payload: dict[str, Any], key: str, *, default: int) -> int:
    if key not in payload:
        return default
    value = payload[key]
    if type(value) is not int or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value
