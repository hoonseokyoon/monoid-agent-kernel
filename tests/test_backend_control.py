"""Control protocol: RunnerBackend.dispatch routing + the POST /v1/runs/{id}/control route."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest
from support.http import http_json, serving
from support.runtime import runtime_config, tool_binding
from support.waiting import eventually

from monoid_agent_kernel.core._util import write_json_atomic
from monoid_agent_kernel.core.checkpoint import RunCheckpoint
from monoid_agent_kernel.core.capability import AutoGrantBroker
from monoid_agent_kernel.core.control import ControlCommand
from monoid_agent_kernel.core.events import make_agent_event
from monoid_agent_kernel.core.inbox import InboxMessage
from monoid_agent_kernel.core.lifecycle import SessionState
from monoid_agent_kernel.core.outcome import InterruptionCause
from monoid_agent_kernel.core.projections import project_run_status
from monoid_agent_kernel.core.result import AgentRunResult, Suspension
from monoid_agent_kernel.core.tool_surface import ToolScope
from monoid_agent_kernel.errors import ModelAdapterError, NativeAgentError, PermissionDenied
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.recorder import AgentRecorder
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.backend.http import create_backend_server
from monoid_agent_kernel.reference.backend.service import (
    BackendRunRecord,
    BackendRunRequest,
    RunnerBackend,
    _CLOSE_SESSION,
    _RESUME_SESSION,
    _RUN_META_SCHEMA_VERSION,
)
from monoid_agent_kernel.tools.base import ToolContext, ToolResult, ToolSpec


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("notes.md").write_text("notes\n", encoding="utf-8")
    return workspace


def _config() -> Any:
    return runtime_config("fs.read", "fs.write", "run.finish")


def _backend(backend_factory: Any, workspace: Path, turns: list[ModelTurn]) -> RunnerBackend:
    backend = backend_factory.create(workspace=workspace, turns=turns)
    backend.idle_timeout_s = 10.0
    return backend


def _parked_multi_turn_run(backend: RunnerBackend, workspace: Path) -> tuple[str, str]:
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hello",
            runtime_config=_config(),
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert eventually(lambda: backend._record(run_id).state is SessionState.AWAITING_INPUT)
    return run_id, token


def _dispatch(backend: RunnerBackend, run_id: str, token: str, ctype: str, **args: Any) -> Any:
    return backend.dispatch(
        ControlCommand(type=ctype, run_id=run_id, args={"token": token, **args})
    )  # type: ignore[arg-type]


def _events(backend: RunnerBackend, run_id: str) -> list[dict[str, Any]]:
    events_path = backend._record(run_id).run_dir / "events.jsonl"
    return [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]


def _backend_record(run_id: str, run_dir: Path, workspace: Path) -> BackendRunRecord:
    return BackendRunRecord(
        run_id=run_id,
        tenant_id="tenant_a",
        user_id="user_a",
        workspace_root=workspace,
        run_dir=run_dir,
        state=SessionState.CREATED,
        terminal=False,
        created_at=0.0,
        run_token_sha256="run-token",
        llm_gateway_token_sha256="llm-token",
    )


def test_control_command_from_json_rejects_present_wrong_type_args() -> None:
    with pytest.raises(ValueError):
        ControlCommand.from_json({"type": "status", "run_id": "run_1", "args": []})


def test_control_command_from_json_accepts_legacy_protocol_id() -> None:
    command = ControlCommand.from_json(
        {
            "protocol": "native-agent-runner.control-command.v1",
            "type": "status",
            "run_id": "run_1",
            "args": {},
        }
    )

    assert command.type == "status"
    assert command.run_id == "run_1"


@pytest.mark.parametrize("event_type", ["run.resumed", "model.turn.started"])
def test_task_resume_events_promote_awaiting_tasks_to_running(
    tmp_path: Path,
    backend_factory: Any,
    event_type: str,
) -> None:
    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_task_resume"
    record = _backend_record(run_id, tmp_path / "runs" / run_id, workspace)
    record.state = SessionState.AWAITING_TASKS
    with backend._lock:
        backend._records[run_id] = record

    backend.record_event(run_id, make_agent_event(run_id=run_id, seq=1, event_type=event_type))

    assert record.state is SessionState.RUNNING
    assert record.terminal is False
    record.state = SessionState.CANCELLED
    record.terminal = True


def test_record_event_captures_the_whole_classification_a_turn_failed_carries(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    """The record captured error/error_code and dropped the five facts beside them.

    The state stays untouched — session_drive owns this record's lifecycle — but the
    classification must reach the record, or GET /status answers with half the taxonomy the
    event beside it carries.
    """
    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_turn_failed_classified"
    record = _backend_record(run_id, tmp_path / "runs" / run_id, workspace)
    record.state = SessionState.RUNNING
    with backend._lock:
        backend._records[run_id] = record

    backend.record_event(
        run_id,
        make_agent_event(
            run_id=run_id,
            seq=1,
            event_type="turn.failed",
            data={
                "error": "model rejected the key",
                "error_code": "model_error",
                "provider_error_code": "insufficient_quota",
                "http_status": 422,
                "retryable": False,
                "config_recoverable": True,
                "provider_retried": True,
            },
        ),
    )

    assert record.error == "model rejected the key"
    assert record.error_code == "model_error"
    assert record.provider_error_code == "insufficient_quota"
    assert record.http_status == 422
    assert record.retryable is False
    assert record.config_recoverable is True
    assert record.provider_retried is True
    # The state-untouched rule stays: the park that follows names the state.
    assert record.state is SessionState.RUNNING

    # The guarded-reader rule stays too: a truthy string must not become a claim.
    backend.record_event(
        run_id,
        make_agent_event(
            run_id=run_id,
            seq=2,
            event_type="turn.failed",
            data={
                "error": "boom",
                "error_code": "model_error",
                "retryable": "yes, definitely",
                "http_status": "422",
            },
        ),
    )
    assert record.retryable is False
    assert record.http_status is None

    record.state = SessionState.CANCELLED
    record.terminal = True


def test_turn_settled_clears_a_backend_record_interruption_cause(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_settled_cause_clear"
    record = _backend_record(run_id, tmp_path / "runs" / run_id, workspace)
    record.interruption_cause = InterruptionCause.USER_CANCEL
    with backend._lock:
        backend._records[run_id] = record

    backend.record_event(
        run_id,
        make_agent_event(
            run_id=run_id,
            seq=1,
            event_type="turn.settled",
            data={"status": "completed", "interruption_cause": ""},
        ),
    )

    assert record.interruption_cause is None
    record.state = SessionState.CANCELLED
    record.terminal = True


def test_record_event_captures_the_whole_classification_a_run_failed_carries(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    """The run.failed twin of the turn.failed capture: a fresh terminal has no park to promote.

    A non-recoverable model failure on the stream lane (or a first-turn failure anywhere)
    reaches its terminal without ever parking, so the driver's park promotion never runs and
    ``record_run_result``'s FAILED heal keeps whatever the record has — defaults. The event
    beside the record carries the whole classification; copying only the error pair left live
    ``status()``/``result()`` omitting the provider classification status.json carries.
    """
    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_run_failed_classified"
    record = _backend_record(run_id, tmp_path / "runs" / run_id, workspace)
    record.state = SessionState.RUNNING
    with backend._lock:
        backend._records[run_id] = record

    backend.record_event(
        run_id,
        make_agent_event(
            run_id=run_id,
            seq=1,
            event_type="run.failed",
            data={
                "error": "provider rejected the request (HTTP 422)",
                "error_code": "model_error",
                "type": "ModelAdapterError",
                "provider_error_code": "insufficient_quota",
                "http_status": 422,
                "retryable": False,
                "config_recoverable": True,
            },
        ),
    )

    assert record.state is SessionState.FAILED
    assert record.terminal is True
    assert record.error_code == "model_error"
    assert record.provider_error_code == "insufficient_quota"
    assert record.http_status == 422
    assert record.retryable is False
    assert record.config_recoverable is True
    # The terminal vocabulary drops the per-call fact (run.failed does not even carry it).
    assert record.provider_retried is False


def test_a_model_turn_starting_unparks_a_paused_record_and_clears_the_stale_failure(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    """PAUSED joins the park-clear set, and the unpark clears the dead turn's answer.

    The record's clear set named two of the three non-terminal parks, so a resumed pause
    served state="paused" through the whole resumed turn — and a retried turn kept the
    previous failure's error while running.
    """
    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_paused_unpark"
    record = _backend_record(run_id, tmp_path / "runs" / run_id, workspace)
    record.state = SessionState.PAUSED
    record.error = "model rejected the key"
    record.error_code = "model_error"
    record.provider_error_code = "insufficient_quota"
    record.http_status = 422
    record.retryable = True
    record.config_recoverable = True
    record.provider_retried = True
    with backend._lock:
        backend._records[run_id] = record

    backend.record_event(
        run_id,
        make_agent_event(run_id=run_id, seq=1, event_type="model.turn.started"),
    )

    assert record.state is SessionState.RUNNING
    assert record.terminal is False
    assert record.error == ""
    assert record.error_code == ""
    assert record.provider_error_code == ""
    assert record.http_status is None
    assert record.retryable is False
    assert record.config_recoverable is False
    assert record.provider_retried is False

    record.state = SessionState.CANCELLED
    record.terminal = True


def test_run_finished_event_defers_terminal_until_result_is_recorded(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_finish_deferred"
    run_dir = tmp_path / "runs" / run_id
    record = _backend_record(run_id, run_dir, workspace)
    record.state = SessionState.RUNNING
    with backend._lock:
        backend._records[run_id] = record

    backend.record_event(
        run_id,
        make_agent_event(
            run_id=run_id,
            seq=1,
            event_type="run.finished",
            data={"error": "", "error_code": ""},
        ),
    )

    assert record.finished_at is not None
    assert record.terminal is False
    assert record.result is None
    assert backend.tenant_usage("tenant_a")["runs"] == 0

    result = AgentRunResult(
        run_id=run_id,
        status="completed",
        final_text="done",
        run_dir=run_dir,
        diff_path=run_dir / "diff.patch",
        proposal_path=run_dir / "proposal.json",
        metrics={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    )
    backend._record_run_result(run_id, result)

    assert record.result is result
    assert record.state is SessionState.COMPLETED
    assert record.terminal is True
    usage = backend.tenant_usage("tenant_a")
    assert usage["runs"] == 1
    assert usage["total_tokens"] == 5


def test_record_run_failure_writes_bundle_before_terminal_flip(
    tmp_path: Path,
    backend_factory: Any,
    monkeypatch: Any,
) -> None:
    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_failure_order"
    run_dir = tmp_path / "runs" / run_id
    record = _backend_record(run_id, run_dir, workspace)
    record.state = SessionState.RUNNING
    with backend._lock:
        backend._records[run_id] = record

    def _raise_before_terminal(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("bundle write failed")

    monkeypatch.setattr(backend, "_write_failure_bundle", _raise_before_terminal)

    with pytest.raises(RuntimeError, match="bundle write failed"):
        backend._record_run_failure(run_id, RuntimeError("worker boom"))

    assert record.state is SessionState.RUNNING
    assert record.terminal is False
    assert record.error == ""
    assert not (run_dir / "failure.json").exists()
    with backend._lock:
        backend._records.pop(run_id, None)


def test_the_backend_failure_bundle_states_the_classification_the_exception_carried(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    """The reference backend writes the same ``monoid.failure.v1`` artifact the core does, and
    its copy is the one a worker crash leaves behind -- the case where the bundle is the only
    record there is. An operator restoring from it must be able to tell "resend after fixing the
    config" from "this will fail again the same way"."""

    from monoid_agent_kernel.errors import ModelAdapterError

    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_failure_classification"
    run_dir = backend.run_root / run_id
    record = _backend_record(run_id, run_dir, workspace)
    record.state = SessionState.RUNNING
    with backend._lock:
        backend._records[run_id] = record

    backend._record_run_failure(
        run_id,
        ModelAdapterError(
            "the gateway sent no generation_applied echo",
            provider_error_code="gateway_generation_not_applied",
            retryable=False,
            config_recoverable=True,
        ),
    )

    bundle = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    assert bundle["config_recoverable"] is True
    assert bundle["retryable"] is False


def test_a_run_that_dies_of_an_exception_meters_what_it_had_already_spent(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    """A run that dies of a driver exception after N billed turns left the tenant ledger
    reporting zero for every one of them -- not even the run count -- while
    ``record_run_result`` beside it fed the same ledger. It never produces an
    ``AgentRunResult``, so what it spent lives in the last committed checkpoint."""

    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_failure_metered"
    record = _backend_record(run_id, backend.run_root / run_id, workspace)
    record.state = SessionState.RUNNING
    with backend._lock:
        backend._records[run_id] = record
    backend.checkpoint_store.put(
        RunCheckpoint(
            run_id=run_id,
            seq=1,
            terminal=False,
            total_usage={
                "input_tokens": 40,
                "output_tokens": 20,
                "total_tokens": 60,
                "reasoning_tokens": 5,
            },
        )
    )

    backend._record_run_failure(run_id, RuntimeError("worker boom"))

    usage = backend.tenant_usage("tenant_a")
    assert usage["runs"] == 1
    assert usage["total_tokens"] == 60
    assert usage["reasoning_tokens"] == 5


def test_a_failure_with_no_checkpoint_meters_from_the_status_projection(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    """The fallback: a run that died before its first park has no checkpoint, and the operator
    status file on disk holds the last ``metrics.updated`` payload. With neither, the run is
    still counted -- which is more than the ledger used to say."""

    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_failure_status_fallback"
    run_dir = backend.run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir.joinpath("status.json").write_text(
        json.dumps({"run_id": run_id, "metrics": {"total_tokens": 25, "input_tokens": 25}}),
        encoding="utf-8",
    )
    record = _backend_record(run_id, run_dir, workspace)
    record.state = SessionState.RUNNING
    with backend._lock:
        backend._records[run_id] = record

    backend._record_run_failure(run_id, RuntimeError("worker boom"))

    usage = backend.tenant_usage("tenant_a")
    assert usage["runs"] == 1
    assert usage["total_tokens"] == 25


def test_a_metered_failure_that_is_recovered_and_completes_is_not_billed_twice(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    """Both terminal paths report CUMULATIVE totals, from different sources. Without a per-run
    high-water mark, a run metered on failure and then recovered to completion would have every
    pre-crash token counted a second time -- and the run counted as two runs."""

    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_failure_then_recovered"
    run_dir = backend.run_root / run_id
    record = _backend_record(run_id, run_dir, workspace)
    record.state = SessionState.RUNNING
    with backend._lock:
        backend._records[run_id] = record
    backend.checkpoint_store.put(
        RunCheckpoint(
            run_id=run_id,
            seq=1,
            terminal=False,
            total_usage={"input_tokens": 40, "output_tokens": 20, "total_tokens": 60},
        )
    )

    backend._record_run_failure(run_id, RuntimeError("worker boom"))
    backend._record_run_result(
        run_id,
        AgentRunResult(
            run_id=run_id,
            status="completed",
            final_text="recovered and finished",
            run_dir=run_dir,
            diff_path=run_dir / "diff.patch",
            proposal_path=run_dir / "proposal.json",
            # The cumulative total of the whole run: the 60 already metered, plus 30 more.
            metrics={"input_tokens": 60, "output_tokens": 30, "total_tokens": 90},
        ),
    )

    usage = backend.tenant_usage("tenant_a")
    assert usage["runs"] == 1
    assert usage["total_tokens"] == 90
    assert usage["input_tokens"] == 60
    assert usage["output_tokens"] == 30


def test_a_corrupt_status_metric_is_dropped_not_raised(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    """The status.json fallback used to pass values through untouched, so one corrupt
    metric turned failure-recording into an escaping ValueError — after ``runs`` was
    already incremented, and past ``run_execution``'s failure paths, eating the streaming
    client's terminal frame. Unreadable read-keys are dropped per key; the readable ones
    still reach the ledger, and ``record_run_result``'s strictness for kernel-written
    values is untouched."""

    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_corrupt_status_metrics"
    run_dir = backend.run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir.joinpath("status.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "metrics": {
                    "input_tokens": 12.5,  # not a count
                    "output_tokens": -3,  # not a count either
                    "total_tokens": 25,
                    "status": "failed",  # never was a count
                },
            }
        ),
        encoding="utf-8",
    )
    record = _backend_record(run_id, run_dir, workspace)
    record.state = SessionState.RUNNING
    with backend._lock:
        backend._records[run_id] = record

    backend._record_run_failure(run_id, RuntimeError("worker boom"))  # must not raise

    usage = backend.tenant_usage("tenant_a")
    assert usage["runs"] == 1
    assert usage["total_tokens"] == 25
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0


def test_failure_metering_takes_the_fresher_reading_per_key(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    """Checkpoints commit at parks; status.json updates per billed event. A run that parks
    at T1, bills more turns, then dies mid-turn has status.json ahead of its checkpoint —
    and the all-or-nothing fallback (checkpoint wins if present) meant T2-T1 never reached
    the ledger. Per-key max over the validated readings is strictly closer to "billed once
    per token", and the high-water delta semantics are unchanged."""

    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_stale_checkpoint_fresh_status"
    run_dir = backend.run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    backend.checkpoint_store.put(
        RunCheckpoint(
            run_id=run_id,
            seq=1,
            terminal=False,
            total_usage={"input_tokens": 200, "output_tokens": 100, "total_tokens": 300},
        )
    )
    run_dir.joinpath("status.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "metrics": {"input_tokens": 300, "output_tokens": 150, "total_tokens": 450},
            }
        ),
        encoding="utf-8",
    )
    record = _backend_record(run_id, run_dir, workspace)
    record.state = SessionState.RUNNING
    with backend._lock:
        backend._records[run_id] = record

    backend._record_run_failure(run_id, RuntimeError("died mid-turn after the park"))

    usage = backend.tenant_usage("tenant_a")
    assert usage["runs"] == 1
    assert usage["input_tokens"] == 300
    assert usage["output_tokens"] == 150
    assert usage["total_tokens"] == 450


def test_record_run_failure_writes_the_terminal_status_artifact(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    """The THIRD failure.json writer makes the same terminal statement the give-up sites make.

    ``record_run_failure`` wrote failure.json and flipped the in-memory record FAILED but never
    touched ``status.json`` — so after a restart every status surface served the run's old park
    (``awaiting_input, terminal=false``) forever, while ``recover_runs`` skipped the dir on
    failure.json: byte-for-byte the symptom the recovery give-up sites already fixed. The
    statement now goes through the ONE shared writer, with this lane's own honest marker and
    the failure's own error_code (not ``unrecoverable``)."""

    from monoid_agent_kernel.core.schemas import STATUS_SCHEMA, _validate_json_file

    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_failure_status_artifact"
    run_dir = backend.run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # The park-shaped artifact a crashed driver leaves behind, with identity + metrics riding.
    write_json_atomic(
        run_dir / "status.json",
        {
            "run_id": run_id,
            "state": "awaiting_input",
            "terminal": False,
            "last_event_seq": 7,
            "last_event_type": "run.awaiting_input",
            "updated_at": "2026-08-03T00:00:00Z",
            "metrics": {"input_tokens": 25, "total_tokens": 25},
            "provider_retried": True,
        },
    )
    record = _backend_record(run_id, run_dir, workspace)
    record.state = SessionState.AWAITING_INPUT
    # What a prior turn.failed park recorded — the four facts a FAILED terminal keeps.
    record.retryable = True
    record.http_status = 429
    record.config_recoverable = True
    record.provider_error_code = "rate_limit"
    record.provider_retried = True
    with backend._lock:
        backend._records[run_id] = record

    backend._record_run_failure(run_id, RuntimeError("worker boom"))

    artifact = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert (artifact["state"], artifact["terminal"]) == ("failed", True)
    assert "worker boom" in artifact["error"]
    # The failure's own error code — not the recovery lane's "unrecoverable".
    assert artifact["error_code"] == "internal_error"
    assert artifact["error_type"] == "RuntimeError"
    # This lane's own honest marker — recovery did not give this run up; its driver died.
    assert artifact["recorded_by_run_failure"] is True
    assert "given_up_by_recovery" not in artifact
    # The terminal vocabulary drops the per-call fact, exactly as run.failed does.
    assert "provider_retried" not in artifact
    # Merged over the prior payload: identity and metrics survive.
    assert artifact["metrics"] == {"input_tokens": 25, "total_tokens": 25}
    assert artifact["last_event_seq"] == 7
    issues: list = []
    _validate_json_file(run_dir / "status.json", STATUS_SCHEMA, issues)
    assert issues == [], issues
    # The offline projection honors the quarantine marker over the stale (park-ending) log.
    projection = project_run_status(run_dir)
    assert (projection["state"], projection["terminal"]) == ("failed", True)
    # And the record states what the artifact states: the EXCEPTION's own classification,
    # through the same guarded reads — not the last park's. The run died of this driver
    # exception, and a bare RuntimeError claims nothing, so the park's stale 429/rate_limit
    # facts must not survive on the live record while status.json beside it says otherwise.
    # (Pin flipped from "the four park facts stay": that kept live status()/result()
    # disagreeing with the just-written artifact until the record was released.)
    assert record.state is SessionState.FAILED
    assert record.terminal is True
    assert record.provider_retried is False
    assert record.retryable is False
    assert record.http_status is None
    assert record.config_recoverable is False
    assert record.provider_error_code == ""


def test_record_run_failure_copies_a_classified_exception_onto_the_record(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    """A classified driver death answers the same on the live record and the artifact.

    ``record_run_failure`` wrote the exception's provider code, HTTP status and recovery flags
    into status.json but mutated only the error pair on the live record — so ``status()`` /
    ``result()``, which prefer the active record, served default or stale classification while
    the durable artifact beside them carried the truth."""

    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_failure_classified_record"
    run_dir = backend.run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    record = _backend_record(run_id, run_dir, workspace)
    record.state = SessionState.RUNNING
    with backend._lock:
        backend._records[run_id] = record

    backend._record_run_failure(
        run_id,
        ModelAdapterError(
            "provider rejected the request (HTTP 422)",
            error_code="model_error",
            provider_error_code="insufficient_quota",
            retryable=False,
            config_recoverable=True,
            http_status=422,
            provider_retried=True,
        ),
    )

    artifact = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    # One answer on both surfaces, from the same guarded reads.
    for surface_value, artifact_key in (
        (record.retryable, "retryable"),
        (record.config_recoverable, "config_recoverable"),
        (record.http_status, "http_status"),
        (record.provider_error_code, "provider_error_code"),
    ):
        assert surface_value == artifact[artifact_key]
    assert record.config_recoverable is True
    assert record.http_status == 422
    assert record.provider_error_code == "insufficient_quota"
    assert record.retryable is False
    # The terminal vocabulary still drops the per-call fact on both.
    assert record.provider_retried is False
    assert "provider_retried" not in artifact


def _giveup_recovery_meta(run_id: str, workspace: Path) -> dict[str, Any]:
    config = _config()
    return {
        "schema_version": _RUN_META_SCHEMA_VERSION,
        "run_id": run_id,
        "tenant_id": "tenant_a",
        "user_id": "user_a",
        "workspace_root": str(workspace),
        "runtime_config": config.to_json(),
        "runtime_config_hash": config.config_hash,
    }


def test_recovery_giveup_after_max_attempts_meters_the_checkpointed_spend(
    tmp_path: Path,
    backend_factory: Any,
    monkeypatch: Any,
) -> None:
    """The resume-failed-max-attempts give-up wrote failure.json and stopped: a run that
    crashed after N billed turns and can never be resumed was never counted and its
    checkpointed spend never reached any ledger — the exact class the failure path closes.
    The give-up has no live record, so it goes through the record-free metering seam."""

    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    backend.max_recover_attempts = 1
    run_id = "run_unrecoverable_spend"
    run_dir = backend.run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    backend.checkpoint_store.put(
        RunCheckpoint(
            run_id=run_id,
            seq=3,
            terminal=False,
            total_usage={"input_tokens": 40, "output_tokens": 20, "total_tokens": 60},
        )
    )
    write_json_atomic(run_dir / "run.json", _giveup_recovery_meta(run_id, workspace))

    def _boom(stored: Any, meta: Any) -> None:
        del stored, meta
        raise RuntimeError("resume boom")

    monkeypatch.setattr(backend._recovery, "resume_from_checkpoint", _boom)

    assert backend.recover_runs() == []
    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["error_code"] == "unrecoverable"

    usage = backend.tenant_usage("tenant_a")
    assert usage["runs"] == 1
    assert usage["total_tokens"] == 60
    assert usage["input_tokens"] == 40


def test_corrupt_durable_state_giveup_meters_from_the_status_projection(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    """The other give-up: corrupt durable state quarantines the run without a resume
    attempt. The checkpoint is unreadable by construction, so the spend comes from the
    status projection — same source hierarchy as the failure path."""

    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_corrupt_state_spend"
    run_dir = backend.run_root / run_id
    backend.checkpoint_store.put(RunCheckpoint(run_id=run_id, seq=1, terminal=False))
    manifest = run_dir / "checkpoints" / "1" / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = "monoid.checkpoint.v99"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    write_json_atomic(run_dir / "run.json", _giveup_recovery_meta(run_id, workspace))
    write_json_atomic(
        run_dir / "status.json",
        {"run_id": run_id, "metrics": {"input_tokens": 25, "total_tokens": 25}},
    )

    assert backend.recover_runs() == []
    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["error_code"] == "checkpoint_unsupported_version"

    usage = backend.tenant_usage("tenant_a")
    assert usage["runs"] == 1
    assert usage["total_tokens"] == 25


def test_cancelling_a_parked_run_records_cancelled_and_keeps_checkpoints(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    """Cancelling a PARKED run acked ``cancel_requested: true`` and then recorded a clean
    COMPLETED — the per-submit reset state was "completed", so ``run.finished`` said so,
    ``record_run_result`` overwrote error_code to "", and close() DELETED the checkpoints.
    The same cancel mid-turn correctly landed CANCELLED. The close boundary now promotes an
    acknowledged cancel through the same vocabulary the mid-run handler uses."""

    workspace = _workspace(tmp_path)
    backend = _backend(backend_factory, workspace, [ModelTurn(response_id="r1", final_text="hi")])
    run_id, token = _parked_multi_turn_run(backend, workspace)

    ack = backend.cancel_run(run_id, token)
    assert ack["cancel_requested"] is True
    assert backend.wait_for_run(run_id, timeout_s=20) is SessionState.CANCELLED

    record = backend._record(run_id)
    assert record.error_code == "cancelled"
    assert record.interruption_cause is InterruptionCause.USER_CANCEL
    result = record.result
    assert result is not None
    assert (result.status, result.error_code) == ("limited", "cancelled")
    assert result.interruption_cause is InterruptionCause.USER_CANCEL
    # A cancelled run keeps its checkpoints — only a clean completion has nothing to restore.
    stored = backend.checkpoint_store.latest(run_id)
    assert stored is not None
    assert stored.checkpoint.terminal is True
    assert stored.checkpoint.cancellation_requested is True
    assert stored.checkpoint.interruption_cause == InterruptionCause.USER_CANCEL.value
    finished = [e for e in _events(backend, run_id) if e["type"] == "run.finished"]
    assert finished and finished[-1]["data"]["status"] == "limited"
    assert finished[-1]["data"]["error_code"] == "cancelled"
    assert finished[-1]["data"]["interruption_cause"] == "user_cancel"


def test_cancel_of_a_parked_run_is_durable_at_the_ack(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    """The park checkpoint predates the cancel (``cancellation_requested=False``), and
    cancel_run only cancelled the in-memory token — so a crash between the ack and the
    terminal record restored the run UNcancelled, despite what the operator was told.
    A cancel of a quiescent (parked) run now commits a checkpoint carrying the flag before
    the ack returns; the restore path already honors it (test_cancellation.py)."""

    workspace = _workspace(tmp_path)
    backend = backend_factory.create(
        workspace=workspace,
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("hitl_request", {"prompt": "Pick"}, "c1"),),
            ),
            ModelTurn(final_text="never reached"),
        ],
    )
    # Wide poll so the parked drive cannot wake and terminalize between the ack and our read.
    backend.task_wait_poll_s = 2.0
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="ask the human",
            runtime_config=runtime_config("hitl.request"),
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert eventually(lambda: backend._record(run_id).state is SessionState.AWAITING_TASKS)
    stored = backend.checkpoint_store.latest(run_id)
    assert stored is not None and stored.checkpoint.cancellation_requested is False

    ack = backend.cancel_run(run_id, token)
    assert ack["cancel_requested"] is True

    # Durable BEFORE the drive wakes: the ack checkpoint is a park artifact, not terminal.
    stored = backend.checkpoint_store.latest(run_id)
    assert stored is not None
    assert stored.checkpoint.cancellation_requested is True
    assert stored.checkpoint.terminal is False

    assert backend.wait_for_run(run_id, timeout_s=20) is SessionState.CANCELLED


class _ErrorScriptAdapter:
    """Drives a script of turns/exceptions: a ModelTurn is returned, an exception raised."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests: list[Any] = []

    def next_turn(self, request: Any) -> Any:
        self.requests.append(request)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _classified_error() -> ModelAdapterError:
    return ModelAdapterError(
        "quota exhausted",
        http_status=422,
        retryable=False,
        config_recoverable=True,
        provider_error_code="insufficient_quota",
        provider_retried=True,
    )


def _error_script_backend(
    backend_factory: Any, workspace: Path, script: list[Any]
) -> RunnerBackend:
    def factory(spec: Any, llm_gateway_token: str) -> _ErrorScriptAdapter:
        del spec, llm_gateway_token
        return _ErrorScriptAdapter(script)

    backend = backend_factory.create(workspace=workspace, model_adapter_factory=factory)
    backend.idle_timeout_s = 30.0
    return backend


_TERMINAL_CLASSIFICATION_KEYS = (
    "retryable",
    "http_status",
    "config_recoverable",
    "provider_error_code",
    "provider_retried",
)

_EMPTY_TERMINAL_CLASSIFICATION = {
    "retryable": False,
    "http_status": None,
    "config_recoverable": False,
    "provider_error_code": "",
    "provider_retried": False,
}


def _assert_one_terminal_answer(
    backend: RunnerBackend,
    run_id: str,
    token: str,
    *,
    state: str,
    error_code: str,
    classification: dict[str, Any],
) -> None:
    """The cell that caught the terminal-heal twin-miss: the live record, the durable
    status.json, and the offline projection must serve ONE answer at a terminal — the same
    state/terminal pair, the same error_code, the same five classification facts."""

    live = backend.status(run_id, token)
    run_dir = backend._record(run_id).run_dir
    status_payload = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    projection = project_run_status(run_dir)

    views = {
        "live record": {
            "state": live["state"],
            "terminal": live["terminal"],
            "error_code": live["error_code"],
            **{key: live.get(key) for key in _TERMINAL_CLASSIFICATION_KEYS},
        },
        "status.json": {
            "state": status_payload.get("state"),
            "terminal": status_payload.get("terminal"),
            "error_code": status_payload.get("error_code", ""),
            # Absent keys carry their reader defaults, matching _status_payload_classification.
            "retryable": status_payload.get("retryable", False),
            "http_status": status_payload.get("http_status"),
            "config_recoverable": status_payload.get("config_recoverable", False),
            "provider_error_code": status_payload.get("provider_error_code", ""),
            "provider_retried": status_payload.get("provider_retried", False),
        },
        "offline projection": {
            "state": projection["state"],
            "terminal": projection["terminal"],
            "error_code": projection["error_code"],
            **{key: projection.get(key) for key in _TERMINAL_CLASSIFICATION_KEYS},
        },
    }
    expected = {"state": state, "terminal": True, "error_code": error_code, **classification}
    for reader, view in views.items():
        assert view == expected, {"reader": reader, "view": view, "expected": expected}


def _close_session(backend: RunnerBackend, run_id: str) -> None:
    """Deterministic idle-close: the same signal the idle timeout's park wait resolves to."""
    backend._call_soon(backend._record(run_id).message_queue.put_nowait, _CLOSE_SESSION)


def test_cancelled_close_boundary_terminal_serves_one_classification(
    tmp_path: Path, backend_factory: Any
) -> None:
    """Empirically traced: cancel of a turn_failed park healed status.json but the live
    record kept the dead turn's five facts beside error_code="cancelled" — the two branches
    of the same status() endpoint disagreed across a restart. The record-side terminal heal
    in record_run_result closes the third consumer of the one rule."""

    workspace = _workspace(tmp_path)
    backend = _error_script_backend(backend_factory, workspace, [_classified_error()])
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hello",
            runtime_config=_config(),
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert eventually(lambda: backend._record(run_id).state is SessionState.AWAITING_INPUT)
    # The park carried the classification onto the record (the park promotion's job).
    assert backend._record(run_id).config_recoverable is True

    backend.cancel_run(run_id, token)
    assert backend.wait_for_run(run_id, timeout_s=20) is SessionState.CANCELLED

    _assert_one_terminal_answer(
        backend,
        run_id,
        token,
        state="cancelled",
        error_code="cancelled",
        classification=_EMPTY_TERMINAL_CLASSIFICATION,
    )


def test_failed_close_boundary_terminal_serves_one_classification(
    tmp_path: Path, backend_factory: Any
) -> None:
    """The FAILED cell: the give-up promotion keeps the four what-it-died-of facts on every
    reader and drops the per-call provider_retried on every reader — the record used to
    keep provider_retried=True while the terminal vocabulary dropped it everywhere else."""

    workspace = _workspace(tmp_path)
    backend = _error_script_backend(backend_factory, workspace, [_classified_error()])
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hello",
            runtime_config=_config(),
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert eventually(lambda: backend._record(run_id).state is SessionState.AWAITING_INPUT)
    assert backend._record(run_id).provider_retried is True

    _close_session(backend, run_id)  # closing on the unrecovered park IS the give-up
    assert backend.wait_for_run(run_id, timeout_s=20) is SessionState.FAILED

    _assert_one_terminal_answer(
        backend,
        run_id,
        token,
        state="failed",
        error_code="model_error",
        classification={
            "retryable": False,
            "http_status": 422,
            "config_recoverable": True,
            "provider_error_code": "insufficient_quota",
            "provider_retried": False,
        },
    )


def test_completed_close_boundary_terminal_serves_one_classification(
    tmp_path: Path, backend_factory: Any
) -> None:
    """The COMPLETED cell: a run that failed a turn, recovered on a resend, settled, and
    closed clean must read clean on all three readers."""

    workspace = _workspace(tmp_path)
    backend = _error_script_backend(
        backend_factory,
        workspace,
        [_classified_error(), ModelTurn(response_id="r2", final_text="fixed")],
    )
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hello",
            runtime_config=_config(),
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert eventually(lambda: backend._record(run_id).state is SessionState.AWAITING_INPUT)

    backend.send_message(run_id, token, "resend after config fix")
    assert eventually(
        lambda: backend._record(run_id).state is SessionState.AWAITING_INPUT
        and backend._record(run_id).error_code == ""
    )
    _close_session(backend, run_id)
    assert backend.wait_for_run(run_id, timeout_s=20) is SessionState.COMPLETED

    _assert_one_terminal_answer(
        backend,
        run_id,
        token,
        state="completed",
        error_code="",
        classification=_EMPTY_TERMINAL_CLASSIFICATION,
    )


def test_limited_close_boundary_terminal_serves_one_classification(
    tmp_path: Path, backend_factory: Any
) -> None:
    """The LIMITED cell — a paused mid-turn run whose session is closed (pause_run + idle
    timeout, the finding-2 trace, driven end-to-end through the backend): the run must not
    finalize a clean COMPLETED with deleted checkpoints, and all three readers must agree
    on the closed_unsettled outcome."""

    workspace = _workspace(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def factory(spec: Any, llm_gateway_token: str) -> FakeModelAdapter:
        del spec, llm_gateway_token
        return FakeModelAdapter(
            turns=[
                ModelTurn(response_id="r1", tool_calls=(fake_tool_call("test_gate", {}, "c1"),)),
                ModelTurn(response_id="r2", final_text="never reached"),
            ]
        )

    backend = backend_factory.create(
        workspace=workspace,
        model_adapter_factory=factory,
        tool_providers=(_GateToolProvider(entered, release),),
    )
    backend.idle_timeout_s = 30.0
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="go",
            runtime_config=runtime_config("test.gate"),
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert entered.wait(timeout=10)
    assert backend.pause_run(run_id, token)["pause_requested"] is True
    release.set()
    assert eventually(lambda: backend._record(run_id).state is SessionState.PAUSED)

    _close_session(backend, run_id)  # the user never resumes; the session idles out
    assert backend.wait_for_run(run_id, timeout_s=20) is SessionState.LIMITED

    _assert_one_terminal_answer(
        backend,
        run_id,
        token,
        state="limited",
        error_code="closed_unsettled",
        classification=_EMPTY_TERMINAL_CLASSIFICATION,
    )
    # ...and the checkpoints holding the frozen turn survive the close.
    stored = backend.checkpoint_store.latest(run_id)
    assert stored is not None
    assert stored.checkpoint.terminal is True


def test_cancel_ack_does_not_clobber_a_committed_park_when_the_drive_is_mid_pump(
    tmp_path: Path, backend_factory: Any
) -> None:
    """TOCTOU half (a): the old cancel path read record.state on the HTTP thread and
    persisted later on the shared loop — a park state read just before the drive resumed
    let a mid-turn snapshot land at the SAME seq as the committed park checkpoint,
    replacing its content (LocalFsCheckpointStore.put overwrites same-seq). The check and
    the snapshot now run as one callable on the drive's own loop, gated on the loop-level
    quiescence marker (``at_quiescent_park``), so a mid-pump cancel skips the park
    re-commit entirely (the pump's own boundary check owns durability there — the
    documented honest window)."""

    workspace = _workspace(tmp_path)
    backend = _backend(backend_factory, workspace, [ModelTurn(response_id="r1", final_text="hi")])
    run_id, token = _parked_multi_turn_run(backend, workspace)
    record = backend._record(run_id)
    stored = backend.checkpoint_store.latest(run_id)
    assert stored is not None and stored.checkpoint.last_suspension is not None
    park_seq = stored.seq

    # The exact marker the pump clears synchronously at entry, before its first await —
    # from any shared-loop callable's view this IS "a pump owns the state now".
    record.loop._session.last_suspension = None

    ack = backend.cancel_run(run_id, token)
    assert ack["cancel_requested"] is True

    # The committed park checkpoint's content is untouched (no same-seq overwrite): the
    # ack is honestly non-durable in this window rather than durably corrupting the park.
    manifest = json.loads(
        (record.run_dir / "checkpoints" / str(park_seq) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["last_suspension"] is not None
    assert manifest["cancellation_requested"] is False
    assert backend.wait_for_run(run_id, timeout_s=20) is SessionState.CANCELLED


def test_cancel_ack_racing_a_close_is_still_a_successful_cancel(
    tmp_path: Path, backend_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TOCTOU half (b): an idle-timeout close landing between the ack and the persist made
    ``snapshot()`` raise run_not_open — and cancel_run 500'd AFTER acknowledging the
    cancel. A run that just ended is exactly what the caller asked for: the persist's
    run_not_open/run_terminal is swallowed (debug-logged) and the ack stands."""

    def _closed(self: RunnerBackend, record: Any) -> None:
        del self, record
        raise NativeAgentError("run is not open; call open() first", error_code="run_not_open")

    monkeypatch.setattr(RunnerBackend, "_persist_run_checkpoint_from_any_thread", _closed)
    workspace = _workspace(tmp_path)
    backend = _backend(backend_factory, workspace, [ModelTurn(response_id="r1", final_text="hi")])
    run_id, token = _parked_multi_turn_run(backend, workspace)

    ack = backend.cancel_run(run_id, token)

    assert ack["cancel_requested"] is True
    assert backend.wait_for_run(run_id, timeout_s=20) is SessionState.CANCELLED


def test_the_backend_failure_bundle_claims_no_classification_it_was_not_given(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    """The other half: a plain worker exception carries no provider verdict, and the guarded
    read must answer ``False`` rather than promoting a truthy attribute into a claim."""

    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_failure_unclassified"
    run_dir = backend.run_root / run_id
    record = _backend_record(run_id, run_dir, workspace)
    record.state = SessionState.RUNNING
    with backend._lock:
        backend._records[run_id] = record

    boom = RuntimeError("worker boom")
    boom.retryable = "yes, definitely"  # type: ignore[attr-defined]
    backend._record_run_failure(run_id, boom)

    bundle = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    assert bundle["retryable"] is False
    assert bundle["config_recoverable"] is False


def test_usage_comes_from_terminal_result_not_metric_events(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_usage_terminal"
    run_dir = tmp_path / "runs" / run_id
    record = _backend_record(run_id, run_dir, workspace)
    record.state = SessionState.RUNNING
    with backend._lock:
        backend._records[run_id] = record

    backend.record_event(
        run_id,
        make_agent_event(
            run_id=run_id,
            seq=1,
            event_type="metrics.updated",
            data={"metrics": {"total_tokens": 999}},
        ),
    )
    assert backend.tenant_usage("tenant_a")["total_tokens"] == 0

    backend._record_run_result(
        run_id,
        AgentRunResult(
            run_id=run_id,
            status="limited",
            final_text="",
            run_dir=run_dir,
            diff_path=run_dir / "diff.patch",
            proposal_path=run_dir / "proposal.json",
            metrics={"input_tokens": 4, "output_tokens": 6, "total_tokens": 10},
            error="limited",
            error_code="cancelled",
        ),
    )

    usage = backend.tenant_usage("tenant_a")
    assert usage["runs"] == 1
    assert usage["total_tokens"] == 10


def test_tenant_usage_is_process_local_after_backend_restart(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    workspace = _workspace(tmp_path)
    run_root = tmp_path / "runs"
    backend1 = backend_factory.create(
        run_root=run_root,
        workspace=workspace,
        turns=[ModelTurn(response_id="r1", final_text="done", usage={"total_tokens": 7})],
    )
    submission = backend1.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="finish",
            runtime_config=_config(),
        )
    )
    assert backend1.wait_for_run(submission.run_id, timeout_s=20) is SessionState.COMPLETED
    assert backend1.tenant_usage("tenant_a")["runs"] == 1

    backend2 = backend_factory.create(run_root=run_root, workspace=workspace, turns=[])

    assert backend2.tenant_usage("tenant_a")["runs"] == 0
    assert backend2.tenant_usage("tenant_a")["total_tokens"] == 0


def test_limited_suspension_marks_record_terminal_before_close(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_limited"
    run_dir = tmp_path / "runs" / run_id
    record = _backend_record(run_id, run_dir, workspace)
    with backend._lock:
        backend._records[run_id] = record
    request = BackendRunRequest(
        tenant_id="tenant_a",
        user_id="user_a",
        workspace_root=workspace,
        instruction="limited",
        runtime_config=_config(),
        multi_turn=True,
    )

    class _ClosingLoop:
        terminal_seen: bool | None = None

        async def aclose(self) -> AgentRunResult:
            self.terminal_seen = record.terminal
            return AgentRunResult(
                run_id=run_id,
                status="limited",
                final_text="",
                run_dir=run_dir,
                diff_path=run_dir / "diff.patch",
                proposal_path=run_dir / "proposal.json",
            )

    loop = _ClosingLoop()

    result = asyncio.run(
        backend._drive_open_session(  # noqa: SLF001 - lifecycle regression around the driver boundary
            record,
            request,
            loop,  # type: ignore[arg-type]
            Suspension(reason="limited", status="limited"),
            started=0.0,
            turns=1,
        )
    )

    assert result.status == "limited"
    assert loop.terminal_seen is True
    assert record.state is SessionState.LIMITED
    assert record.terminal is True


def test_session_message_wait_ignores_stray_resume(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    backend.idle_timeout_s = 1.0
    record = _backend_record("run_resume", tmp_path / "runs" / "run_resume", workspace)
    record.message_queue.put_nowait(_RESUME_SESSION)
    record.message_queue.put_nowait("next")

    message = asyncio.run(backend._await_session_message(record))  # noqa: SLF001 - driver boundary regression

    assert message == "next"


def test_session_message_wait_skips_duplicate_inbox_envelope(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    backend.idle_timeout_s = 1.0
    record = _backend_record("run_inbox", tmp_path / "runs" / "run_inbox", workspace)
    record.seen_inbox_ids.add("msg_1")
    record.message_queue.put_nowait(InboxMessage(content="duplicate", id="msg_1").to_json())
    record.message_queue.put_nowait(InboxMessage(content="fresh", id="msg_2").to_json())

    message = asyncio.run(backend._await_session_message(record))  # noqa: SLF001 - driver boundary regression

    assert message["id"] == "msg_2"
    assert record.seen_inbox_ids == {"msg_1", "msg_2"}


def test_paused_session_requeues_user_message_before_resuming_frozen_turn(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    backend.idle_timeout_s = 1.0
    run_id = "run_paused"
    run_dir = tmp_path / "runs" / run_id
    record = _backend_record(run_id, run_dir, workspace)
    request = BackendRunRequest(
        tenant_id="tenant_a",
        user_id="user_a",
        workspace_root=workspace,
        instruction="paused",
        runtime_config=_config(),
        multi_turn=True,
    )

    class _PausedLoop:
        inputs: list[Any]

        def __init__(self) -> None:
            self.inputs = []

        def snapshot(self) -> None:
            return None

        async def arun_until_suspended(self, value: Any) -> Suspension:
            self.inputs.append(value)
            return Suspension(reason="terminal", status="completed")

        async def aclose(self) -> AgentRunResult:
            return AgentRunResult(
                run_id=run_id,
                status="completed",
                final_text="",
                run_dir=run_dir,
                diff_path=run_dir / "diff.patch",
                proposal_path=run_dir / "proposal.json",
            )

    loop = _PausedLoop()
    record.loop = loop  # type: ignore[assignment]
    record.message_queue.put_nowait("queued while paused")
    with backend._lock:
        backend._records[run_id] = record

    asyncio.run(
        backend._drive_open_session(  # noqa: SLF001 - driver boundary regression
            record,
            request,
            loop,  # type: ignore[arg-type]
            Suspension(reason="paused", status="running"),
            started=time.time(),
            turns=1,
        )
    )

    assert loop.inputs == [None]
    assert record.message_queue.get_nowait() == "queued while paused"


def test_checkpoint_persist_carries_pending_messages_and_seen_inbox_ids(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_checkpoint"
    record = _backend_record(run_id, tmp_path / "runs" / run_id, workspace)
    record.seen_inbox_ids.update({"msg_1", "msg_2"})
    envelope = InboxMessage(content="queued", id="msg_3").to_json()
    record.message_queue.put_nowait("plain")
    record.message_queue.put_nowait(_RESUME_SESSION)
    record.message_queue.put_nowait(envelope)

    class _SnapshotLoop:
        def snapshot(self) -> RunCheckpoint:
            return RunCheckpoint(run_id=run_id, seq=1)

        def collect_checkpoint_blobs(self) -> dict[str, bytes]:
            return {}

        def due_outbox(self, now: float) -> list[Any]:
            del now
            return []

    record.loop = _SnapshotLoop()  # type: ignore[assignment]

    backend._persist_run_checkpoint(record)  # noqa: SLF001 - driver boundary regression

    assert backend.checkpoint_store is not None
    stored = backend.checkpoint_store.latest(run_id)
    assert stored is not None
    assert stored.checkpoint.queued_messages == ["plain", envelope]
    assert stored.checkpoint.inbox_seen_ids == ["msg_1", "msg_2"]


def test_send_message_enqueue_persists_queue_snapshot_before_return(
    tmp_path: Path,
    backend_factory: Any,
) -> None:
    workspace = _workspace(tmp_path)
    backend = backend_factory.create(workspace=workspace, turns=[])
    run_id = "run_enqueue_checkpoint"
    record = _backend_record(run_id, tmp_path / "runs" / run_id, workspace)
    envelope = InboxMessage(content="queued before task result", id="msg_review").to_json()

    class _SnapshotLoop:
        def snapshot(self) -> RunCheckpoint:
            return RunCheckpoint(run_id=run_id, seq=1)

        def collect_checkpoint_blobs(self) -> dict[str, bytes]:
            return {}

        def due_outbox(self, now: float) -> list[Any]:
            del now
            return []

    record.loop = _SnapshotLoop()  # type: ignore[assignment]

    backend._enqueue_message_and_checkpoint(record, envelope)  # noqa: SLF001 - enqueue durability regression

    assert backend.checkpoint_store is not None
    stored = backend.checkpoint_store.latest(run_id)
    assert stored is not None
    assert stored.checkpoint.queued_messages == [envelope]


class _UnopenedLoop:
    def __init__(self) -> None:
        self.calls = 0

    def emit_external_event(
        self,
        event_type: str,
        *,
        data: dict[str, Any] | None = None,
        level: str = "info",
        turn_id: str | None = None,
    ) -> bool:
        del event_type, data, level, turn_id
        self.calls += 1
        return False


def test_dispatch_inspect_and_health_report_live_state(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(
        backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")]
    )
    run_id, token = _parked_multi_turn_run(backend, workspace)

    inspect = _dispatch(backend, run_id, token, "inspect")
    assert inspect.status == "ok"
    assert inspect.state == "awaiting_input"
    assert inspect.data["state"] == "awaiting_input"
    assert inspect.data["run_id"] == run_id
    assert inspect.data["terminal"] is False

    health = _dispatch(backend, run_id, token, "health")
    assert health.status == "ok"
    assert health.state == "awaiting_input"
    assert health.data["alive"] is True
    assert health.data["can_accept_input"] is True

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_dispatch_emits_control_audit_events_without_token_leak(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(
        backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")]
    )
    run_id, token = _parked_multi_turn_run(backend, workspace)

    status = backend.dispatch(
        ControlCommand(
            type="status",
            run_id=run_id,
            args={"token": token},
            issuer="operator_a",
            reason="check run",
            command_id="cmd_status",
        )
    )
    assert status.status == "ok"

    bad_replace = backend.dispatch(
        ControlCommand(
            type="replace_runtime_config",
            run_id=run_id,
            args={"token": token, "expected_version": 99, "config": _config().to_json()},
            issuer="operator_a",
            reason="bad version",
            command_id="cmd_bad_replace",
        )
    )
    assert bad_replace.status == "error"

    with pytest.raises(PermissionDenied):
        backend.dispatch(
            ControlCommand(
                type="inspect",
                run_id=run_id,
                args={"token": "bad-token"},
                issuer="operator_b",
                reason="bad auth",
                command_id="cmd_bad_auth",
            )
        )

    events = [
        event for event in _events(backend, run_id) if event["type"].startswith("control.command.")
    ]
    by_id = {(event["type"], event["data"]["command_id"]): event["data"] for event in events}

    received = by_id[("control.command.received", "cmd_status")]
    assert received["command"] == "status"
    assert received["actor"] == "operator_a"
    assert received["reason"] == "check run"
    assert received["token_sha256"] == TokenManager.token_sha256(token)
    assert received["idempotency_key"] == "cmd_status"
    assert received["args_keys"] == []
    completed = by_id[("control.command.completed", "cmd_status")]
    assert completed["status"] == "ok"
    assert completed["idempotency_key"] == "cmd_status"
    assert completed["result_code"] == "ok"
    assert completed["token_sha256"] == TokenManager.token_sha256(token)

    failed = by_id[("control.command.failed", "cmd_bad_replace")]
    assert failed["command"] == "replace_runtime_config"
    assert failed["status"] == "error"
    assert failed["error_code"] == "control_error"
    assert failed["failure_code"] == "control_error"
    assert failed["idempotency_key"] == "cmd_bad_replace"

    assert all(event["data"]["command_id"] != "cmd_bad_auth" for event in events)

    serialized_events = "\n".join(json.dumps(event, sort_keys=True) for event in events)
    assert token not in serialized_events
    assert "bad-token" not in serialized_events

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_dispatch_control_audit_uses_live_recorder_sequence(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(
        backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")]
    )
    run_id, token = _parked_multi_turn_run(backend, workspace)

    status = _dispatch(backend, run_id, token, "status")
    assert status.status == "ok"
    record = backend._record(run_id)
    assert record.loop is not None
    assert record.loop.emit_external_event(
        "control.test.after_audit",
        data={"ok": True},
        turn_id="turn_external",
    )

    events = _events(backend, run_id)
    seqs = [event["seq"] for event in events]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))
    completed_seq = max(
        event["seq"] for event in events if event["type"] == "control.command.completed"
    )
    after_event = next(
        event for event in events if event["type"] == "control.test.after_audit"
    )
    assert after_event["seq"] > completed_seq
    assert after_event["turn_id"] == "turn_external"

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_dispatch_skips_run_audit_before_loop_owns_sequence(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(
        backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")]
    )
    run_id, token = _parked_multi_turn_run(backend, workspace)
    record = backend._record(run_id)
    loop = record.loop
    assert loop is not None
    before = _events(backend, run_id)

    record.loop = None
    try:
        status = _dispatch(backend, run_id, token, "status")
    finally:
        record.loop = loop

    assert status.status == "ok"
    assert _events(backend, run_id) == before

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_dispatch_appends_queued_run_audit_before_recorder_starts(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(backend_factory, workspace, [ModelTurn(response_id="r1", final_text="done")])
    prepared = backend._prepare_run_record(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hello",
            runtime_config=_config(),
        )
    )

    result = _dispatch(backend, prepared.run_id, prepared.run_token, "status")

    assert result.status == "ok"
    events = _events(backend, prepared.run_id)
    assert [event["type"] for event in events] == [
        "control.command.received",
        "control.command.completed",
    ]
    recorder = AgentRecorder(backend.run_root, prepared.run_id)
    try:
        assert recorder.emit("run.started", data={"mode": "propose"}).seq == 3
    finally:
        recorder.close()
        backend.cancel_run(prepared.run_id, prepared.run_token)
        backend._records.pop(prepared.run_id, None)


def test_control_audit_skips_direct_append_when_loop_is_not_open(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(
        backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")]
    )
    run_id, token = _parked_multi_turn_run(backend, workspace)
    record = backend._record(run_id)
    loop = record.loop
    unopened = _UnopenedLoop()
    before = _events(backend, run_id)

    record.loop = unopened  # type: ignore[assignment]
    try:
        backend._emit_control_audit_event(
            run_id,
            "control.command.received",
            {"command_id": "cmd_starting", "command": "status"},
        )
    finally:
        record.loop = loop

    assert unopened.calls == 1
    assert _events(backend, run_id) == before

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_dispatch_appends_terminal_run_audit_after_recorder_closes(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(backend_factory, workspace, [ModelTurn(response_id="r1", final_text="done")])
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hello",
            runtime_config=_config(),
        )
    )
    assert backend.wait_for_run(submission.run_id, timeout_s=20) == "completed"
    before = _events(backend, submission.run_id)

    result = _dispatch(backend, submission.run_id, submission.run_token, "status")

    assert result.status == "ok"
    after = _events(backend, submission.run_id)
    appended = after[len(before) :]
    assert [event["type"] for event in appended] == [
        "control.command.received",
        "control.command.completed",
    ]
    seqs = [event["seq"] for event in after]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))


def test_control_audit_skips_recordless_nonterminal_run(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(backend_factory, workspace, [ModelTurn(response_id="r1", final_text="done")])
    run_id = "run_remote_live"
    run_dir = backend.run_root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps({"run_id": run_id, "status": "running", "last_event_seq": 1}),
        encoding="utf-8",
    )
    original_events = json.dumps({"seq": 1, "type": "run.started"}) + "\n"
    (run_dir / "events.jsonl").write_text(original_events, encoding="utf-8")

    backend._emit_control_audit_event(
        run_id,
        "control.command.received",
        {"command_id": "cmd_remote", "command": "status"},
    )

    assert (run_dir / "events.jsonl").read_text(encoding="utf-8") == original_events


def test_dispatch_routes_existing_ops_and_unknown(tmp_path: Path, backend_factory: Any) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(
        backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")]
    )
    run_id, token = _parked_multi_turn_run(backend, workspace)

    assert _dispatch(backend, run_id, token, "status").status == "ok"
    assert _dispatch(backend, run_id, token, "runtime_config").status == "ok"

    # Pause/resume acks (the deep freeze/continue is covered at the loop level).
    pause = _dispatch(backend, run_id, token, "pause")
    assert pause.status == "ok"
    assert pause.data["pause_requested"] is True
    resume = _dispatch(backend, run_id, token, "resume")
    assert resume.status == "ok"
    assert resume.data["resumed"] is True

    # Direct callers stay forward-compatible. Text still crosses the same Unicode-scalar
    # normalization boundary as known command types before the dispatcher sees it.
    unknown = _dispatch(backend, run_id, token, f"frob{chr(0xD800)}nicate")
    assert unknown.status == "unsupported"
    assert unknown.type == "frob\ufffdnicate"
    assert unknown.error_code == "unknown_control_command"

    class StringSubclass(str):
        pass

    for invalid_type in (True, StringSubclass("status")):
        with pytest.raises(ValueError, match="control command type must be a string"):
            backend.dispatch(
                ControlCommand(  # type: ignore[arg-type]
                    type=invalid_type,
                    run_id=run_id,
                    args={"token": token},
                )
            )

    with pytest.raises(ValueError, match="expected_version must be an integer"):
        backend.dispatch(
            ControlCommand(
                type="replace_runtime_config",
                run_id=run_id,
                args={"token": token, "expected_version": True, "config": _config().to_json()},
            )
        )

    cancel = _dispatch(backend, run_id, token, "cancel")
    assert cancel.status == "ok"
    assert backend.wait_for_run(run_id, timeout_s=20) in {
        "completed",
        "failed",
        "limited",
        "cancelled",
    }


def test_dispatch_inspect_on_terminal_run_is_error(tmp_path: Path, backend_factory: Any) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(backend_factory, workspace, [ModelTurn(response_id="r1", final_text="done")])
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="hello",
            runtime_config=_config(),
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert backend.wait_for_run(run_id, timeout_s=20) == "completed"

    # inspect/health need a live loop; on a terminal run they report a controlled error.
    result = _dispatch(backend, run_id, token, "inspect")
    assert result.status == "error"
    # status still works on a terminal run (it reads the record).
    assert _dispatch(backend, run_id, token, "status").status == "ok"


def test_dispatch_bad_token_raises_permission_denied(tmp_path: Path, backend_factory: Any) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(
        backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")]
    )
    run_id, token = _parked_multi_turn_run(backend, workspace)
    with pytest.raises(PermissionDenied):
        backend.dispatch(ControlCommand(type="inspect", run_id=run_id, args={"token": "bad"}))
    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_http_control_route_keeps_strict_decoder_and_dispatches_inspect(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(
        backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")]
    )
    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    with serving(server) as base_url:
        created = http_json(
            f"{base_url}/v1/runs",
            {
                "tenant_id": "tenant_a",
                "user_id": "user_a",
                "workspace_root": str(workspace),
                "instruction": "hello",
                "runtime_config": _config().to_json(),
                "multi_turn": True,
            },
            token="admin",
        )
        run_id, run_token = created["run_id"], created["run_token"]
        assert eventually(lambda: backend._record(run_id).state is SessionState.AWAITING_INPUT)

        for malformed, expected_error in (
            ({"type": "frobnicate"}, "type must be one of"),
            ({"type": "status", "issuer": 7}, "issuer must be a string"),
        ):
            with pytest.raises(HTTPError) as caught:
                http_json(
                    f"{base_url}/v1/runs/{run_id}/control",
                    malformed,
                    token=run_token,
                )
            assert caught.value.code == 400
            error = json.loads(caught.value.read().decode("utf-8"))["error"]
            assert expected_error in error

        result = http_json(
            f"{base_url}/v1/runs/{run_id}/control",
            {"type": "inspect"},
            token=run_token,
        )
        assert result["status"] == "ok"
        assert result["state"] == "awaiting_input"
        assert result["protocol"] == "monoid.control-command.v1"

        backend.cancel_run(run_id, run_token)
        backend.wait_for_run(run_id, timeout_s=20)


def test_capability_task_kind_creates_and_resolves(tmp_path: Path, backend_factory: Any) -> None:
    # Step 5: a scoped-capability request rides the hosted-task seam. The Daemon creates a
    # capability park and resolves it via report_task_result (both reachable through dispatch).
    workspace = _workspace(tmp_path)
    backend = _backend(
        backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")]
    )
    run_id, token = _parked_multi_turn_run(backend, workspace)

    created = backend.create_task(
        run_id,
        token,
        kind="capability",
        request={"capability": "web.search", "scope": {"allowed_domains": ["example.edu"]}},
    )
    assert "task_id" in created and "callback_token" in created

    resolved = backend.report_task_result(
        run_id,
        token,
        task_id=created["task_id"],
        result={"granted": True, "token_ref": "secret-ref://lease-1"},
    )
    assert resolved.get("delivered") is True

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_dispatch_report_task_result_accepts_callback_token(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(
        backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")]
    )
    run_id, token = _parked_multi_turn_run(backend, workspace)

    task = backend.create_task(
        run_id,
        token,
        kind="hitl",
        request={"prompt": "Continue?", "choices": ("Yes", "No")},
    )
    result = backend.dispatch(
        ControlCommand(
            type="report_task_result",
            run_id=run_id,
            args={
                "token": task["callback_token"],
                "task_id": task["task_id"],
                "result": {"answer": "Yes"},
            },
            issuer="callback_worker",
            command_id="cmd_callback_result",
        )
    )

    assert result.status == "ok"
    assert result.data["delivered"] is True
    events = [
        event for event in _events(backend, run_id) if event["type"].startswith("control.command.")
    ]
    by_id = {(event["type"], event["data"]["command_id"]): event["data"] for event in events}
    assert (
        by_id[("control.command.received", "cmd_callback_result")]["command"]
        == "report_task_result"
    )
    assert by_id[("control.command.completed", "cmd_callback_result")]["result_code"] == "ok"

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_dispatch_approve_accepts_callback_token(tmp_path: Path, backend_factory: Any) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(
        backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")]
    )
    run_id, token = _parked_multi_turn_run(backend, workspace)

    task = backend.create_task(
        run_id,
        token,
        kind="hitl",
        request={"prompt": "Approve callback?", "choices": ("Approve", "Deny")},
    )
    approved = backend.dispatch(
        ControlCommand(
            type="approve",
            run_id=run_id,
            args={"token": task["callback_token"], "task_id": task["task_id"]},
            issuer="callback_worker",
            command_id="cmd_callback_approve",
        )
    )

    assert approved.status == "ok"
    assert approved.data["delivered"] is True

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_dispatch_deny_overwrites_conflicting_result_fields(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(
        backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")]
    )
    run_id, token = _parked_multi_turn_run(backend, workspace)

    task = backend.create_task(
        run_id,
        token,
        kind="hitl",
        request={"prompt": "Approve this?", "choices": ("Approve", "Deny")},
    )
    denied = backend.dispatch(
        ControlCommand(
            type="deny",
            run_id=run_id,
            args={
                "token": token,
                "task_id": task["task_id"],
                "result": {
                    "answer": "Approve",
                    "approved": True,
                    "granted": True,
                    "lease": {"capability": "web.search", "token_ref": "secret-ref://lease-1"},
                    "token_ref": "secret-ref://lease-1",
                },
            },
            issuer="operator_a",
            reason="policy denied",
            command_id="cmd_conflicting_deny",
        )
    )

    assert denied.status == "ok"
    job = json.loads(
        (
            backend._record(run_id).run_dir / "artifacts" / "tasks" / task["task_id"] / "task.json"
        ).read_text(encoding="utf-8")
    )
    assert job["result"]["answer"] == "Deny"
    assert job["result"]["approved"] is False
    assert job["result"]["granted"] is False
    assert job["result"]["reason"] == "policy denied"
    assert "lease" not in job["result"]
    assert "token_ref" not in job["result"]

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_dispatch_approve_and_deny_are_audited_task_decisions(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(
        backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")]
    )
    run_id, token = _parked_multi_turn_run(backend, workspace)

    approve_task = backend.create_task(
        run_id,
        token,
        kind="hitl",
        request={"prompt": "Approve this action?", "choices": ("Approve", "Deny")},
    )
    approved = backend.dispatch(
        ControlCommand(
            type="approve",
            run_id=run_id,
            args={"token": token, "task_id": approve_task["task_id"]},
            issuer="operator_a",
            reason="approved by reviewer",
            command_id="cmd_approve",
        )
    )
    assert approved.status == "ok"
    assert approved.data["delivered"] is True

    deny_task = backend.create_task(
        run_id,
        token,
        kind="hitl",
        request={"prompt": "Approve this second action?", "choices": ("Approve", "Deny")},
    )
    denied = backend.dispatch(
        ControlCommand(
            type="deny",
            run_id=run_id,
            args={"token": token, "task_id": deny_task["task_id"]},
            issuer="operator_a",
            reason="policy denied",
            command_id="cmd_deny",
        )
    )
    assert denied.status == "ok"
    assert denied.data["delivered"] is True

    events = [
        event for event in _events(backend, run_id) if event["type"].startswith("control.command.")
    ]
    by_id = {(event["type"], event["data"]["command_id"]): event["data"] for event in events}
    assert by_id[("control.command.received", "cmd_approve")]["command"] == "approve"
    assert by_id[("control.command.completed", "cmd_approve")]["result_code"] == "ok"
    assert by_id[("control.command.received", "cmd_deny")]["command"] == "deny"
    assert by_id[("control.command.completed", "cmd_deny")]["idempotency_key"] == "cmd_deny"

    tasks_dir = backend._record(run_id).run_dir / "artifacts" / "tasks"
    approved_job = json.loads(
        (tasks_dir / approve_task["task_id"] / "task.json").read_text(encoding="utf-8")
    )
    denied_job = json.loads(
        (tasks_dir / deny_task["task_id"] / "task.json").read_text(encoding="utf-8")
    )
    assert approved_job["result"]["approved"] is True
    assert denied_job["result"]["approved"] is False
    assert denied_job["result"]["granted"] is False

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


class _GateToolProvider:
    """Yields one tool whose handler blocks until released — lets a test hold a run mid-turn
    so it can request a pause deterministically."""

    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        self._entered = entered
        self._release = release

    def get_tools(self, context: ToolContext | None = None) -> list[ToolSpec]:
        def handler(ctx: ToolContext, args: dict) -> ToolResult:
            self._entered.set()
            self._release.wait(timeout=10)
            return ToolResult(ok=True, content={"gated": True})

        return [
            ToolSpec(
                id="test.gate",
                description="block until released",
                input_schema={"type": "object", "properties": {}, "additionalProperties": True},
                capability="test.gate",
                side_effect="read",
                handler=handler,
            )
        ]


class _CapCountingProvider:
    """A capability-gated tool that counts executions — for the revoke end-to-end test."""

    def __init__(self) -> None:
        self.calls = 0

    def get_tools(self, context: ToolContext | None = None) -> list[ToolSpec]:
        provider = self

        def handler(ctx: ToolContext, args: dict) -> ToolResult:
            provider.calls += 1
            return ToolResult(ok=True, content={"ran": True})

        return [
            ToolSpec(
                id="ext.fetch",
                description="external fetch needing web.search capability",
                input_schema={"type": "object", "properties": {}, "additionalProperties": True},
                capability="web.search",
                side_effect="read",
                handler=handler,
            )
        ]


def test_dispatch_revoke_capability_blocks_subsequent_call(
    tmp_path: Path, backend_factory: Any
) -> None:
    # End-to-end operator kill switch: a gated tool runs on a granted lease, the Daemon dispatches
    # revoke_capability, and the next gated call is refused — through the Control protocol.
    workspace = _workspace(tmp_path)
    provider = _CapCountingProvider()
    turns = [
        ModelTurn(response_id="r1", tool_calls=(fake_tool_call("ext_fetch", {}, "c1"),)),
        ModelTurn(response_id="r2", final_text="first"),
        ModelTurn(response_id="r3", tool_calls=(fake_tool_call("ext_fetch", {}, "c2"),)),
        ModelTurn(response_id="r4", final_text="second"),
    ]

    def factory(spec: Any, llm_gateway_token: str) -> FakeModelAdapter:
        del spec, llm_gateway_token
        return FakeModelAdapter(turns=list(turns))

    backend = backend_factory.create(
        workspace=workspace,
        model_adapter_factory=factory,
        tool_providers=(provider,),
        capability_broker_factory=lambda req: AutoGrantBroker(),
    )
    backend.idle_timeout_s = 10.0
    binding = tool_binding(
        "ext.fetch", runtime={"requires_lease": True}, scope=ToolScope(allowed_domains=("a.edu",))
    )
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="go",
            runtime_config=runtime_config(bindings=(binding,)),
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token
    assert eventually(lambda: backend._record(run_id).state is SessionState.AWAITING_INPUT)
    assert provider.calls == 1  # the tool ran on the granted lease

    revoke = _dispatch(backend, run_id, token, "revoke_capability", capability="web.search")
    assert revoke.status == "ok"
    assert revoke.data["revoked"] is True
    assert revoke.data["capabilities"] == ["web.search"]

    # A follow-up message re-issues the gated call; revocation refuses it (no re-broker).
    backend.send_message(run_id, token, content="again")
    assert eventually(lambda: backend._record(run_id).state is SessionState.AWAITING_INPUT)
    assert provider.calls == 1  # still 1 — the gated tool stayed blocked after revocation

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_driver_pauses_mid_turn_then_resumes_to_settle(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    entered, release = threading.Event(), threading.Event()
    turns = [
        ModelTurn(response_id="r1", tool_calls=(fake_tool_call("test_gate", {}, "c1"),)),
        ModelTurn(response_id="r2", final_text="done"),
    ]

    def factory(spec: Any, llm_gateway_token: str) -> FakeModelAdapter:
        del spec, llm_gateway_token
        return FakeModelAdapter(turns=list(turns))

    backend = backend_factory.create(
        workspace=workspace,
        model_adapter_factory=factory,
        tool_providers=(_GateToolProvider(entered, release),),
    )
    backend.idle_timeout_s = 10.0
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="go",
            runtime_config=runtime_config("test.gate"),
            multi_turn=True,
        )
    )
    run_id, token = submission.run_id, submission.run_token

    # The gate tool is executing -> the run is mid-turn. Request a pause, then release it.
    assert entered.wait(timeout=10)
    assert backend.pause_run(run_id, token)["pause_requested"] is True
    release.set()

    # The loop hits the next step boundary, raises TurnPaused; the driver parks the run PAUSED.
    assert eventually(lambda: backend._record(run_id).state is SessionState.PAUSED)
    inspect = _dispatch(backend, run_id, token, "inspect")
    assert inspect.state == "paused"

    # Resume re-pumps the SAME turn (the gate observation is re-sent) to settle.
    assert _dispatch(backend, run_id, token, "resume").data["resumed"] is True
    assert eventually(lambda: backend._record(run_id).state is SessionState.AWAITING_INPUT)

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)
