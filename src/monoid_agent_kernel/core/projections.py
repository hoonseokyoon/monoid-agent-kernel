from __future__ import annotations

from pathlib import Path
from typing import Any

from monoid_agent_kernel.core._event_log import read_committed_event_payloads
from monoid_agent_kernel.core.json_ingress import loads_json_ingress
from monoid_agent_kernel.core.lifecycle import (
    SessionState,
    lifecycle_from_status_artifact,
    session_state_from_run_status,
    session_state_value,
)
from monoid_agent_kernel.core.outcome import InterruptionCause
from monoid_agent_kernel.permissions import PermissionPolicy
from monoid_agent_kernel.public_view import public_path
from monoid_agent_kernel.tasks import public_job_artifacts, run_permission_policy


# The non-terminal parks this projection can be sitting in when a model turn starts. Named
# once so the offline projection and ``recorder.py:StatusJsonSink`` clear the same set: the sink
# used to clear only ``AWAITING_INPUT``, which left a task-parked run reading as parked after the
# turn that unparked it had already begun. PAUSED joined when the pause became visible on the
# status surfaces: a resumed pump unparks it exactly like the other two.
_PARKED_STATES = frozenset(
    {
        session_state_value(SessionState.AWAITING_INPUT),
        session_state_value(SessionState.AWAITING_TASKS),
        session_state_value(SessionState.PAUSED),
    }
)

# The failure-classification facts ``turn.failed`` carries beside ``error``/``error_code``
# (minus the metering-only ``provider_usage``). One rule for their whole life on this
# projection: the park assigns them, the unpark clears them, and a non-failed terminal heals
# them — the same three moments ``recorder.py:StatusJsonSink`` binds.
_FAILURE_CLASSIFICATION_DEFAULTS: dict[str, Any] = {
    "provider_error_code": "",
    "http_status": None,
    "retryable": False,
    "config_recoverable": False,
    "provider_retried": False,
}


#: Marker keys a failure-quarantine writer stamps on the terminal ``status.json`` statement
#: it mints beside ``failure.json`` — one per quarantine lane, honest about who wrote it:
#: recovery's give-up sites, and the backend's ``record_run_failure``. Declared in core so
#: the two readers of the one bit ("this terminal statement is a quarantine, not a close")
#: — this module's replay override and the backend's closed-run guard — cannot drift from
#: the writer (``reference/backend/run_state.py:write_failure_status_artifact``), which
#: validates its marker against this same tuple.
FAILURE_QUARANTINE_MARKERS: tuple[str, ...] = (
    "given_up_by_recovery",
    "recorded_by_run_failure",
)


def status_artifact_failure_quarantined(payload: Any) -> bool:
    """Whether a durable status payload carries a failure-quarantine marker.

    Guarded ``is True`` per key: a hand-edited truthy string must not activate the override.
    A later genuine recovery rewrites the artifact without the marker, so the answer dies
    with the quarantine."""
    if not isinstance(payload, dict):
        return False
    return any(payload.get(key) is True for key in FAILURE_QUARANTINE_MARKERS)


def status_artifact_records_close(payload: Any) -> bool:
    """Whether a durable status payload records a CLOSE — a terminal outcome the run's own
    close path wrote — as opposed to a failure-quarantine statement or a live park.

    The one reader behind the recovery closed-run guard AND ``list_runs``' ``recoverable``
    fact: a run that closed limited keeps a non-terminal park checkpoint by design, so this
    artifact fact is the only durable marker that the run already ended — recovery must not
    re-drive it, and the projection must not advertise it resumable. A quarantine statement
    answers False on purpose: while its ``failure.json`` stands every resume path refuses the
    dir on the bundle, and once an operator lifts the quarantine (the restore-hint flow) this
    guard must not keep refusing the resume the hint prescribes. Unreadable or malformed
    payloads answer False — a best-effort projection must not block a genuine recovery."""
    if not isinstance(payload, dict):
        return False
    if status_artifact_failure_quarantined(payload):
        return False
    try:
        _state, terminal = lifecycle_from_status_artifact(payload)
    except ValueError:
        return False
    return terminal


def _event_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    return value if isinstance(value, str) else ""


def _event_flag(data: dict[str, Any], key: str) -> bool:
    """A boolean fact off durable event data, or ``False`` — a truthy string on a corrupt or
    hand-written log must not become a claim that a failure is retryable."""
    value = data.get(key)
    return value if type(value) is bool else False


def _event_http_status(data: dict[str, Any]) -> int | None:
    value = data.get("http_status")
    return value if type(value) is int else None


def _event_interruption_cause(data: dict[str, Any]) -> str | None:
    value = data.get("interruption_cause")
    if not isinstance(value, str) or not value:
        return None
    try:
        return InterruptionCause(value).value
    except ValueError:
        return None


def _assign_failure_classification(projection: dict[str, Any], data: dict[str, Any]) -> None:
    projection["provider_error_code"] = _event_text(data, "provider_error_code")
    projection["http_status"] = _event_http_status(data)
    projection["retryable"] = _event_flag(data, "retryable")
    projection["config_recoverable"] = _event_flag(data, "config_recoverable")
    projection["provider_retried"] = _event_flag(data, "provider_retried")


def _clear_failure_classification(projection: dict[str, Any]) -> None:
    projection.update(_FAILURE_CLASSIFICATION_DEFAULTS)


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
        # Declared here rather than created by the one branch that writes it, so the projection's
        # shape does not depend on which events a run happened to emit. Every source of this
        # string is already filtered through ``public_view.py:public_error_message`` by its
        # writer (status.json, metrics.json, and the ``turn.failed`` event data below).
        "error": status_payload.get("error") or metrics.get("error") or "",
        # The classification beside the error, declared for the same reason and seeded from
        # status.json (the one artifact that spells these keys this way; metrics.json spells
        # the status ``provider_http_status``). Guarded reads: a corrupt artifact must not
        # turn a string into a retryable claim.
        "provider_error_code": _event_text(status_payload, "provider_error_code"),
        "http_status": _event_http_status(status_payload),
        "retryable": _event_flag(status_payload, "retryable"),
        "config_recoverable": _event_flag(status_payload, "config_recoverable"),
        "provider_retried": _event_flag(status_payload, "provider_retried"),
        "interruption_cause": (
            _event_interruption_cause(status_payload)
            or _event_interruption_cause(metrics)
        ),
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
    if status_artifact_failure_quarantined(status_payload):
        # A failure quarantine ends a run WITHOUT a live recorder — recovery's give-up, and
        # the backend's ``record_run_failure`` alike — so no terminal event ever reaches
        # events.jsonl: the log honestly ends at the park the run died in, and the replay
        # above just resurrected that park. The quarantine's terminal statement exists only
        # in the status artifact (written beside failure.json), so it is re-applied over the
        # replayed park here: without this, the offline reader answered a healthy
        # ``awaiting_input`` for a quarantined run while ``status()``/``list_runs`` (which
        # read the artifact) answered ``failed``. One marker helper for every lane, guarded
        # against hand-edited truthy strings; a later genuine recovery rewrites the artifact
        # without the marker, so the override dies with the quarantine.
        projection["state"] = _payload_state(status_payload, metrics)
        projection["terminal"] = _payload_terminal(status_payload, projection["state"])
        projection["error"] = status_payload.get("error") or ""
        projection["error_code"] = status_payload.get("error_code") or ""
        projection["provider_error_code"] = _event_text(status_payload, "provider_error_code")
        projection["http_status"] = _event_http_status(status_payload)
        projection["retryable"] = _event_flag(status_payload, "retryable")
        projection["config_recoverable"] = _event_flag(status_payload, "config_recoverable")
        projection["provider_retried"] = _event_flag(status_payload, "provider_retried")
        projection["interruption_cause"] = None
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
            # Terminal branches ASSIGN, never or-fallback. The or kept a dead turn's
            # ``error_code`` on a cleanly completed run: ``turn.failed -> (recovery) ->
            # run.finished{status:"completed", error_code:""}`` read the empty string as
            # "keep the stale value", and ``error`` was never touched at all — the same
            # sequence the sink twin already healed.
            projection["state"] = session_state_value(
                session_state_from_run_status(
                    _optional_text(data.get("status"), "run.finished status")
                    or projection["state"],
                    error_code=_optional_text(data.get("error_code"), "run.finished error_code")
                    or "",
                    terminal=True,
                )
            )
            projection["terminal"] = True
            projection["error"] = _event_text(data, "error")
            projection["error_code"] = _event_text(data, "error_code")
            projection["interruption_cause"] = _event_interruption_cause(data)
            # ...and the classification heals with it, except on a failed terminal, where the
            # ``run.failed`` one event earlier owns it and a clear here would undo that record.
            if projection["state"] != session_state_value(SessionState.FAILED):
                _clear_failure_classification(projection)
        elif event_type == "run.failed":
            projection["state"] = session_state_value(SessionState.FAILED)
            projection["terminal"] = True
            projection["error"] = _event_text(data, "error")
            projection["error_code"] = _event_text(data, "error_code")
            # Exactly what the terminal event carries: the classification, minus
            # ``provider_retried`` — a per-call fact the terminal vocabulary deliberately
            # drops, so it is cleared rather than carried over from the park.
            _assign_failure_classification(projection, data)
            projection["interruption_cause"] = None
        elif event_type == "run.waiting":
            projection["state"] = session_state_value(SessionState.AWAITING_TASKS)
            projection["terminal"] = False
            projection["waiting_for_background_jobs"] = True
        elif event_type == "run.awaiting_input":
            # The other park on this same stream. Handling only ``run.waiting`` here left a run
            # parked for a hosted task or for user input reading as *running* to `monoid status`,
            # while ``recorder.py:StatusJsonSink`` — the other consumer of the same events —
            # handled both. Both parks now, on both readers.
            projection["state"] = session_state_value(SessionState.AWAITING_INPUT)
            projection["terminal"] = False
        elif event_type == "run.resumed":
            projection["state"] = session_state_value(SessionState.RUNNING)
            projection["terminal"] = False
            projection["waiting_for_background_jobs"] = False
        elif event_type == "turn.failed":
            # A recoverable model-turn park. The event carries the whole classification and this
            # projection used to show two of its seven facts, so an offline `monoid status` on a
            # parked run could not separate an ``insufficient_quota`` (fix config) from a
            # ``rate_limit`` (wait). Assigned, not or-ed: the newest park owns the answer.
            # State is left alone on purpose: ``turn.failed`` is not terminal and the park that
            # follows it (``run.awaiting_input``) names the state.
            projection["error"] = _event_text(data, "error")
            projection["error_code"] = _event_text(data, "error_code")
            _assign_failure_classification(projection, data)
        elif event_type == "turn.paused":
            # The newest turn park is authoritative. A pause has no interruption cause, so it
            # clears the cause projected from an older interrupted park.
            projection["interruption_cause"] = None
        elif event_type == "turn.settled":
            projection["interruption_cause"] = _event_interruption_cause(data)
        elif event_type == "turn.interrupted":
            projection["interruption_cause"] = _event_interruption_cause(data)
        elif event_type == "session.state.changed":
            # The pause park's session-lane projection — the sink twin binds the same event.
            # Only the state this reader can prove: the pause is today's sole emitter.
            if data.get("state") == session_state_value(SessionState.PAUSED):
                projection["state"] = session_state_value(SessionState.PAUSED)
                projection["terminal"] = False
        elif event_type == "agent.config.updated":
            projection["agent_config"] = {
                "definition_id": data.get("definition_id"),
                "config_version": data.get("config_version"),
                "config_hash": data.get("config_hash"),
            }
        elif event_type == "model.turn.started":
            projection["current_step"] = data.get("step")
            # A model turn starting means the run is no longer parked, whichever park it was in.
            # ``run.resumed`` clears the job wait explicitly, but nothing emits it after a
            # user-input park, so without this a resumed session read as parked forever.
            if projection["state"] in _PARKED_STATES:
                projection["state"] = session_state_value(SessionState.RUNNING)
                projection["terminal"] = False
                projection["waiting_for_background_jobs"] = False
            # The unpark clear, outside the parked-state guard on purpose: a retried turn
            # never passes through a parked state (the driver re-pumps straight from
            # ``turn_failed``), and the dead turn's error must not ride beside
            # state="running". While parked the failure remains — the model turn *starting*
            # is what supersedes it. Same rule, same moment, on the sink twin.
            projection["error"] = ""
            projection["error_code"] = ""
            _clear_failure_classification(projection)
            projection["interruption_cause"] = None
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
