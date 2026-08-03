from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from monoid_agent_kernel.core._util import read_text_resilient
from monoid_agent_kernel.core._event_log import (
    EventLogCorruption,
    inspect_event_log_tail,
    read_committed_event_payloads,
)
from monoid_agent_kernel.core.checkpoint import CheckpointStore
from monoid_agent_kernel.core.durable_metadata import DurableMetadataCommitter
from monoid_agent_kernel.core.event_sequencing import (
    RunEventSequencer,
    diagnostic_event_summary,
)
from monoid_agent_kernel.core.json_ingress import (
    loads_json_ingress,
    normalize_json_ingress,
    normalize_unicode_scalars,
)
from monoid_agent_kernel.core.lifecycle import (
    lifecycle_from_status_artifact,
    session_state_value,
)
from monoid_agent_kernel.core.subagent_runtime import (
    subagent_diagnostics_from_events,
    validate_descendant_run_id,
)
from monoid_agent_kernel.core.trace_context import trace_id_of
from monoid_agent_kernel.errors import PermissionDenied
from monoid_agent_kernel.public_view import public_error_message
from monoid_agent_kernel.reference.backend.content_hydration import hydrate_settled_text
from monoid_agent_kernel.reference.backend.ports import RunRecordPort
from monoid_agent_kernel.reference.backend.proposal_reader import read_proposal_snapshot
from monoid_agent_kernel.reference.backend.run_state import (
    record_lifecycle_payload as _record_lifecycle_payload,
)

_RUN_EVENT_SEQUENCER = RunEventSequencer()


class EventPageReader(Protocol):
    def __call__(
        self,
        events_path: Path,
        *,
        from_seq: int,
        limit: int | None,
    ) -> dict[str, Any]: ...


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = loads_json_ingress(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _diagnostic_event_summary(event: dict[str, Any]) -> dict[str, Any]:
    return diagnostic_event_summary(event)


def _trace_ids_from_events(events: list[dict[str, Any]]) -> list[str]:
    trace_ids: set[str] = set()
    for event in events:
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        trace_id = trace_id_of(str(data.get("traceparent") or ""))
        if trace_id:
            trace_ids.add(trace_id)
    return sorted(trace_ids)


def _json_safe(value: Any) -> Any:
    """Render a value safe for JSON wire projection at any nesting depth."""
    try:
        return loads_json_ingress(
            json.dumps(
                normalize_json_ingress(value),
                default=repr,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except RecursionError:
        return "<value exceeds JSON nesting limit>"
    except json.JSONDecodeError as exc:
        if exc.msg == "JSON nesting exceeds the parser limit":
            return "<value exceeds JSON nesting limit>"
        return normalize_unicode_scalars(repr(value))
    except (TypeError, ValueError):
        return normalize_unicode_scalars(repr(value))


def _status_payload_lifecycle(
    status_payload: Mapping[str, Any] | None,
    run_dir: Path,
) -> dict[str, Any]:
    state, terminal = lifecycle_from_status_artifact(
        status_payload,
        failure_present=(run_dir / "failure.json").exists(),
    )
    return {"state": session_state_value(state), "terminal": terminal}


def _status_last_event_seq(status_payload: Mapping[str, Any] | None) -> int:
    raw = (status_payload or {}).get("last_event_seq", 0)
    return raw if type(raw) is int and raw >= 0 else 0


def _status_payload_classification(status_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """The error + failure classification a durable ``status.json`` carries, guarded.

    The record-is-None branch of ``status()`` serves this so a post-restart operator gets the
    same answer the live record gave — one vocabulary, both branches. Guarded reads for the
    same reason ``run_state.py`` guards its event twins: this is disk data, and a truthy
    string must not become a claim that a failure is retryable.
    """

    payload = status_payload or {}

    def _text(name: str) -> str:
        value = payload.get(name)
        return value if isinstance(value, str) else ""

    def _flag(name: str) -> bool:
        value = payload.get(name)
        return value if type(value) is bool else False

    http_status = payload.get("http_status")
    return {
        "error": _text("error"),
        "error_code": _text("error_code"),
        "retryable": _flag("retryable"),
        "http_status": http_status if type(http_status) is int else None,
        "config_recoverable": _flag("config_recoverable"),
        "provider_error_code": _text("provider_error_code"),
        "provider_retried": _flag("provider_retried"),
    }


def _committed_event_highwatermark(run_dir: Path, advertised: int) -> int:
    """Reconcile a best-effort status watermark with the authoritative committed JSONL tail."""

    events_path = run_dir / "events.jsonl"
    try:
        committed = inspect_event_log_tail(events_path).last_seq
    except (EventLogCorruption, OSError):
        # Tail corruption is exceptional. Preserve the last fully decoded prefix so a committed
        # start before the damaged record is still classified as recovery replay.
        readable = read_committed_event_payloads(events_path).payloads
        committed = int(readable[-1]["seq"]) if readable else 0
    return max(advertised, committed)


@dataclass(frozen=True)
class RunProjectionContext:
    authorized_run_dir: Callable[[str, str], Path]
    authorize_run: Callable[[str, str], None]
    record: Callable[[str], RunRecordPort]
    active_record: Callable[[str], RunRecordPort | None]
    read_recover_attempts: Callable[[Path], int]
    run_root_provider: Callable[[], Path]
    checkpoint_store_provider: Callable[[], CheckpointStore | None]
    max_recover_attempts_provider: Callable[[], int]
    issue_read_token: Callable[[str, str, str], str]
    read_event_page: EventPageReader


class RunProjectionService:
    """Read-only run projections for the RunnerBackend facade."""

    def __init__(self, context: RunProjectionContext) -> None:
        self._context = context

    def status(self, run_id: str, token: str) -> dict[str, Any]:
        run_dir = self._context.authorized_run_dir(run_id, token)
        status_file = run_dir / "status.json"
        status_payload: dict[str, Any] | None = None
        if status_file.exists():
            # Resilient read: the run may be concurrently flipping status.json via atomic replace.
            status_payload = loads_json_ingress(read_text_resilient(status_file))
        record = self._context.active_record(run_id)
        if record is None:
            last_event_seq = _committed_event_highwatermark(
                run_dir,
                _status_last_event_seq(status_payload),
            )
            return {
                "run_id": run_id,
                **_status_payload_lifecycle(status_payload, run_dir),
                # status.json carries the classification a parked failure wrote, so the
                # operator who restarts the backend gets the same top-level answer the live
                # record served — not a payload with no error slot at all.
                **_status_payload_classification(status_payload),
                "last_event_seq": last_event_seq,
                "run_dir": str(run_dir),
                "status_file": status_payload,
            }
        return {
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            "user_id": record.user_id,
            **_record_lifecycle_payload(record),
            "created_at": record.created_at,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "run_dir": str(record.run_dir),
            "last_event_seq": _committed_event_highwatermark(
                record.run_dir,
                record.last_event_seq,
            ),
            "last_event_type": record.last_event_type,
            "error": record.error,
            "error_code": record.error_code,
            # What the park knew — the whole classification, one vocabulary. The driver
            # records it and does not act on it, so this surface is the only place it
            # becomes useful to anyone.
            "retryable": record.retryable,
            "http_status": record.http_status,
            "config_recoverable": record.config_recoverable,
            "provider_error_code": record.provider_error_code,
            "provider_retried": record.provider_retried,
            "final_output": _json_safe(
                record.last_final_output
                if record.last_final_output is not None
                else (record.result.final_output if record.result is not None else None)
            ),
            "status_file": status_payload,
        }

    def result(self, run_id: str, token: str) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        record = self._context.record(run_id)
        if record.result is None:
            return {
                "run_id": record.run_id,
                "tenant_id": record.tenant_id,
                **_record_lifecycle_payload(record),
                "ready": False,
                "error": record.error,
                "error_code": record.error_code,
                "retryable": record.retryable,
                "http_status": record.http_status,
                "config_recoverable": record.config_recoverable,
                "provider_error_code": record.provider_error_code,
                "provider_retried": record.provider_retried,
            }
        result = record.result
        diff_text = (
            result.diff_path.read_text(encoding="utf-8") if result.diff_path.exists() else ""
        )
        proposal_payload = read_proposal_snapshot(record)
        return {
            "run_id": record.run_id,
            "tenant_id": record.tenant_id,
            **_record_lifecycle_payload(record),
            "status": result.status,
            "ready": True,
            "final_text": result.final_text,
            "final_output": _json_safe(result.final_output),
            # ``AgentRunResult.error`` is deliberately raw: the embedding application is inside
            # the trust boundary. This response is not — the route hands this dict to
            # ``_write_json`` unchanged — so the filter lands here, on the one error expression
            # that did not already come pre-filtered from its writer (``record.error`` on the
            # not-ready branch above, and ``metrics["error"]`` below, both go through
            # ``public_error_message`` in ``run_state.py``/``loop_phases.py``). The in-process
            # result object keeps the whole message.
            "error": public_error_message(result.error),
            "error_code": result.error_code,
            "retryable": record.retryable,
            "http_status": record.http_status,
            "config_recoverable": record.config_recoverable,
            "provider_error_code": record.provider_error_code,
            "provider_retried": record.provider_retried,
            "run_dir": str(result.run_dir),
            "manifest_path": str(result.run_dir / "manifest.json"),
            "diff_path": str(result.diff_path),
            "diff": diff_text,
            "proposal_path": str(result.proposal_path),
            "proposal": proposal_payload,
            "artifacts": [artifact.__dict__ for artifact in result.artifacts],
            "metrics": result.metrics,
        }

    def events(
        self, run_id: str, token: str, *, from_seq: int = 0, limit: int | None = None
    ) -> dict[str, Any]:
        run_dir = self._context.authorized_run_dir(run_id, token)
        page = self._context.read_event_page(
            run_dir / "events.jsonl", from_seq=from_seq, limit=limit
        )
        # One seam for every consumer that reads events through the backend: the REST and SSE
        # twins on reference/backend/http.py, the Studio BFF's read_page, and `monoid studio
        # accept`. Hydrating at those call sites instead would leave each new transport to
        # remember.
        hydrate_settled_text(page.get("events"), run_dir)
        return {"run_id": run_id, **page}

    def descendant_events(
        self,
        run_id: str,
        token: str,
        descendant_run_id: str,
        *,
        from_seq: int = 0,
        limit: int | None = None,
    ) -> dict[str, Any]:
        self._context.authorize_run(run_id, token)
        try:
            validate_descendant_run_id(run_id, descendant_run_id)
        except ValueError as exc:
            raise PermissionDenied(str(exc)) from exc
        run_dir = self._context.run_root_provider() / descendant_run_id
        page = self._context.read_event_page(
            run_dir / "events.jsonl", from_seq=from_seq, limit=limit
        )
        # The descendant feed hydrates from the *child's* run dir, and it has to happen here:
        # Studio cannot reach a child run dir at all (`_token_for` raises for an id it never
        # issued a token for), which is why this endpoint exists in the first place.
        hydrate_settled_text(page.get("events"), run_dir)
        return {"run_id": descendant_run_id, **page}

    def descendant_status(self, run_id: str, token: str, descendant_run_id: str) -> dict[str, Any]:
        """Read descendant lifecycle after authorizing its root ancestor."""

        self._context.authorize_run(run_id, token)
        try:
            validate_descendant_run_id(run_id, descendant_run_id)
        except ValueError as exc:
            raise PermissionDenied(str(exc)) from exc
        run_dir = self._context.run_root_provider() / descendant_run_id
        status_payload = _read_optional_json(run_dir / "status.json")
        record = self._context.active_record(descendant_run_id)
        if record is not None:
            return {
                "run_id": descendant_run_id,
                **_record_lifecycle_payload(record),
                "last_event_seq": record.last_event_seq,
                "last_event_type": record.last_event_type,
                "status_file": status_payload,
            }
        return {
            "run_id": descendant_run_id,
            **_status_payload_lifecycle(status_payload, run_dir),
            "status_file": status_payload,
        }

    def diagnostics(self, run_id: str, token: str, *, event_limit: int = 50) -> dict[str, Any]:
        if event_limit < 1:
            raise ValueError("event_limit must be positive")
        run_dir = self._context.authorized_run_dir(run_id, token)
        status = self.status(run_id, token)
        status_file = (
            status.get("status_file") if isinstance(status.get("status_file"), dict) else {}
        )
        from_seq = _RUN_EVENT_SEQUENCER.diagnostics_from_seq(
            status,
            status_file,
            event_limit=event_limit,
        )
        event_page = self._context.read_event_page(
            run_dir / "events.jsonl",
            from_seq=from_seq,
            limit=event_limit,
        )
        event_summaries = [_diagnostic_event_summary(event) for event in event_page["events"]]
        control_events = [
            event
            for event in event_summaries
            if str(event.get("type") or "").startswith("control.command.")
        ]
        failure = _read_optional_json(run_dir / "failure.json")
        recover_attempts = self._context.read_recover_attempts(run_dir)
        return {
            "run_id": run_id,
            "status": status,
            "failure": failure,
            "recovery": {
                "attempts": recover_attempts,
                "max_attempts": self._context.max_recover_attempts_provider(),
                "failure_marked": failure is not None,
                "unrecoverable": bool(failure and failure.get("error_code") == "unrecoverable"),
            },
            "events": {
                "from_seq": from_seq,
                "next_seq": event_page["next_seq"],
                "has_more": event_page["has_more"],
                "items": event_summaries,
            },
            "control": {"events": control_events},
            "subagents": subagent_diagnostics_from_events(event_page["events"]),
            "trace_ids": _trace_ids_from_events(event_page["events"]),
        }

    def list_runs(
        self, tenant_id: str, *, user_id: str | None = None, limit: int = 100
    ) -> dict[str, Any]:
        runs: list[dict[str, Any]] = []
        run_root = self._context.run_root_provider()
        checkpoint_store = self._context.checkpoint_store_provider()
        if not run_root.is_dir():
            return {"runs": runs}
        metadata_committer = DurableMetadataCommitter(checkpoint_store)
        for run_dir in run_root.iterdir():
            if not run_dir.is_dir():
                continue
            meta = metadata_committer.read_recovery_metadata(run_dir, run_dir.name)
            if meta is None:
                continue
            if meta.get("tenant_id") != tenant_id:
                continue
            run_user = meta.get("user_id") or ""
            if user_id is not None and run_user != user_id:
                continue
            run_id = meta.get("run_id") or run_dir.name
            record = self._context.active_record(run_id)
            if record is not None:
                lifecycle = {
                    **_record_lifecycle_payload(record),
                    "last_event_seq": _committed_event_highwatermark(
                        run_dir,
                        record.last_event_seq,
                    ),
                }
            else:
                status_payload: dict[str, Any] | None = None
                status_path = run_dir / "status.json"
                if status_path.exists():
                    try:
                        payload = loads_json_ingress(read_text_resilient(status_path))
                        status_payload = payload if isinstance(payload, dict) else None
                    except (ValueError, OSError):
                        status_payload = None
                lifecycle = {
                    **_status_payload_lifecycle(status_payload, run_dir),
                    "last_event_seq": _committed_event_highwatermark(
                        run_dir,
                        _status_last_event_seq(status_payload),
                    ),
                }
            recoverable = False
            if (
                record is None
                and not (run_dir / "failure.json").exists()
                and checkpoint_store is not None
            ):
                stored = checkpoint_store.latest(run_id)
                recoverable = stored is not None and not stored.checkpoint.terminal
            runs.append(
                {
                    "run_id": run_id,
                    "tenant_id": tenant_id,
                    "user_id": run_user,
                    "title": meta.get("title") or "",
                    "created_at": meta.get("created_at") or 0.0,
                    **lifecycle,
                    "recoverable": recoverable,
                    "read_token": self._context.issue_read_token(run_id, tenant_id, run_user),
                }
            )
        runs.sort(key=lambda entry: entry["created_at"], reverse=True)
        return {"runs": runs[:limit]}
