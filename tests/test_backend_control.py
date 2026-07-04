"""Control protocol: RunnerBackend.dispatch routing + the POST /v1/runs/{id}/control route."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from support.http import http_json, serving
from support.runtime import runtime_config, tool_binding
from support.waiting import eventually

from monoid_agent_kernel.core.checkpoint import RunCheckpoint
from monoid_agent_kernel.core.capability import AutoGrantBroker
from monoid_agent_kernel.core.control import ControlCommand
from monoid_agent_kernel.core.events import make_agent_event
from monoid_agent_kernel.core.inbox import InboxMessage
from monoid_agent_kernel.core.lifecycle import SessionState
from monoid_agent_kernel.core.result import AgentRunResult, Suspension
from monoid_agent_kernel.core.tool_surface import ToolScope
from monoid_agent_kernel.errors import PermissionDenied
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call
from monoid_agent_kernel.recorder import AgentRecorder
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.backend.http import create_backend_server
from monoid_agent_kernel.reference.backend.service import (
    BackendRunRecord,
    BackendRunRequest,
    RunnerBackend,
    _RESUME_SESSION,
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
    return backend.dispatch(ControlCommand(type=ctype, run_id=run_id, args={"token": token, **args}))  # type: ignore[arg-type]


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


class _UnopenedLoop:
    def __init__(self) -> None:
        self.calls = 0

    def emit_external_event(
        self,
        event_type: str,
        *,
        data: dict[str, Any] | None = None,
        level: str = "info",
    ) -> bool:
        del event_type, data, level
        self.calls += 1
        return False


def test_dispatch_inspect_and_health_report_live_state(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")])
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
    backend = _backend(backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")])
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

    events = [event for event in _events(backend, run_id) if event["type"].startswith("control.command.")]
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
    backend = _backend(backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")])
    run_id, token = _parked_multi_turn_run(backend, workspace)

    status = _dispatch(backend, run_id, token, "status")
    assert status.status == "ok"
    record = backend._record(run_id)
    assert record.loop is not None
    assert record.loop.emit_external_event("control.test.after_audit", data={"ok": True})

    events = _events(backend, run_id)
    seqs = [event["seq"] for event in events]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))
    completed_seq = max(event["seq"] for event in events if event["type"] == "control.command.completed")
    after_seq = next(event["seq"] for event in events if event["type"] == "control.test.after_audit")
    assert after_seq > completed_seq

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_dispatch_skips_run_audit_before_loop_owns_sequence(
    tmp_path: Path, backend_factory: Any
) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")])
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
    backend = _backend(backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")])
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
    backend = _backend(backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")])
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

    # Unknown command type stays forward-compatible: unsupported, not a crash.
    unknown = _dispatch(backend, run_id, token, "frobnicate")
    assert unknown.status == "unsupported"
    assert unknown.error_code == "unknown_control_command"

    cancel = _dispatch(backend, run_id, token, "cancel")
    assert cancel.status == "ok"
    assert backend.wait_for_run(run_id, timeout_s=20) in {"completed", "failed", "limited", "cancelled"}


def test_dispatch_inspect_on_terminal_run_is_error(
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
    run_id, token = submission.run_id, submission.run_token
    assert backend.wait_for_run(run_id, timeout_s=20) == "completed"

    # inspect/health need a live loop; on a terminal run they report a controlled error.
    result = _dispatch(backend, run_id, token, "inspect")
    assert result.status == "error"
    # status still works on a terminal run (it reads the record).
    assert _dispatch(backend, run_id, token, "status").status == "ok"


def test_dispatch_bad_token_raises_permission_denied(tmp_path: Path, backend_factory: Any) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")])
    run_id, token = _parked_multi_turn_run(backend, workspace)
    with pytest.raises(PermissionDenied):
        backend.dispatch(ControlCommand(type="inspect", run_id=run_id, args={"token": "bad"}))
    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_http_control_route_dispatches_inspect(tmp_path: Path, backend_factory: Any) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")])
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


def test_capability_task_kind_creates_and_resolves(
    tmp_path: Path, backend_factory: Any
) -> None:
    # Step 5: a scoped-capability request rides the hosted-task seam. The Daemon creates a
    # capability park and resolves it via report_task_result (both reachable through dispatch).
    workspace = _workspace(tmp_path)
    backend = _backend(backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")])
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
    backend = _backend(backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")])
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
    events = [event for event in _events(backend, run_id) if event["type"].startswith("control.command.")]
    by_id = {(event["type"], event["data"]["command_id"]): event["data"] for event in events}
    assert by_id[("control.command.received", "cmd_callback_result")]["command"] == "report_task_result"
    assert by_id[("control.command.completed", "cmd_callback_result")]["result_code"] == "ok"

    backend.cancel_run(run_id, token)
    backend.wait_for_run(run_id, timeout_s=20)


def test_dispatch_approve_accepts_callback_token(tmp_path: Path, backend_factory: Any) -> None:
    workspace = _workspace(tmp_path)
    backend = _backend(backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")])
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
    backend = _backend(backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")])
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
            backend._record(run_id).run_dir
            / "artifacts"
            / "tasks"
            / task["task_id"]
            / "task.json"
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
    backend = _backend(backend_factory, workspace, [ModelTurn(response_id="r1", final_text="first")])
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

    events = [event for event in _events(backend, run_id) if event["type"].startswith("control.command.")]
    by_id = {(event["type"], event["data"]["command_id"]): event["data"] for event in events}
    assert by_id[("control.command.received", "cmd_approve")]["command"] == "approve"
    assert by_id[("control.command.completed", "cmd_approve")]["result_code"] == "ok"
    assert by_id[("control.command.received", "cmd_deny")]["command"] == "deny"
    assert by_id[("control.command.completed", "cmd_deny")]["idempotency_key"] == "cmd_deny"

    tasks_dir = backend._record(run_id).run_dir / "artifacts" / "tasks"
    approved_job = json.loads((tasks_dir / approve_task["task_id"] / "task.json").read_text(encoding="utf-8"))
    denied_job = json.loads((tasks_dir / deny_task["task_id"] / "task.json").read_text(encoding="utf-8"))
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
