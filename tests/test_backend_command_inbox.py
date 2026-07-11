from __future__ import annotations

import sqlite3
import threading
import time
from urllib.error import HTTPError
from pathlib import Path
from typing import Any

import pytest

from support.http import http_json, serving
from support.runtime import runtime_config
from support.waiting import eventually

from monoid_agent_kernel.reference.backend.http import create_backend_server
from monoid_agent_kernel.reference.backend.service import BackendRunRequest
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.command_inbox import (
    CommandPrincipal,
    InMemoryCommandStore,
    SqliteCommandStore,
    StoredCommand,
)
from monoid_agent_kernel.reference.stores.sqlite import SqliteCheckpointStore, SqliteLeaseStore
from monoid_agent_kernel.errors import NativeAgentError, PermissionDenied
from monoid_agent_kernel.identifiers import BACKEND_AUDIENCE, TASK_CALLBACK_AUDIENCE
from monoid_agent_kernel.core.control import ControlCommand, ControlResult
from monoid_agent_kernel.providers.base import ModelTurn


def test_cross_worker_http_command_is_drained_by_owner_with_durable_receipt(
    backend_factory: Any, tmp_path: Path
) -> None:
    workspace = backend_factory.workspace()
    run_root = tmp_path / "runs"
    db = tmp_path / "shared.db"
    token_manager = backend_factory.token_manager()
    owner = backend_factory.create(
        run_root=run_root,
        workspace=workspace,
        token_manager=token_manager,
        checkpoint_store=SqliteCheckpointStore(db),
        lease_store=SqliteLeaseStore(db),
        command_store=SqliteCommandStore(db),
    )
    peer = backend_factory.create(
        run_root=run_root,
        workspace=workspace,
        token_manager=token_manager,
        checkpoint_store=SqliteCheckpointStore(db),
        lease_store=SqliteLeaseStore(db),
        command_store=SqliteCommandStore(db),
    )
    owner.watchdog_interval_s = 0.05
    submission = owner.submit_run(
        BackendRunRequest(
            tenant_id="tenant_a",
            user_id="user_a",
            workspace_root=workspace,
            instruction="wait",
            runtime_config=runtime_config("run.finish"),
            multi_turn=True,
        )
    )
    assert eventually(
        lambda: owner._record(submission.run_id).state.value == "awaiting_input",
        timeout_s=10,
    )
    owner.start_watchdog()
    wrong_subject = token_manager.issue(
        kind="run_access",
        audience=BACKEND_AUDIENCE,
        run_id=submission.run_id,
        tenant_id="other_tenant",
        user_id="other_user",
        ttl_s=60,
    )
    with pytest.raises(PermissionDenied, match="subject mismatch"):
        peer.enqueue_control(
            ControlCommand(
                type="status",
                run_id=submission.run_id,
                args={"token": wrong_subject},
                command_id="cmd_wrong_subject",
            )
        )
    with pytest.raises(PermissionDenied, match="subject mismatch"):
        peer.enqueue_control(
            ControlCommand(
                type="approve",
                run_id=submission.run_id,
                args={"token": wrong_subject, "task_id": "task_unknown"},
                command_id="cmd_wrong_callback_subject",
            )
        )
    wrong_callback_subject = token_manager.issue(
        kind="task_callback",
        audience=TASK_CALLBACK_AUDIENCE,
        run_id=submission.run_id,
        tenant_id="other_tenant",
        user_id="other_user",
        ttl_s=60,
        metadata={"task_id": "task_unknown"},
    )
    with pytest.raises(PermissionDenied, match="subject mismatch"):
        peer.enqueue_control(
            ControlCommand(
                type="approve",
                run_id=submission.run_id,
                args={"token": wrong_callback_subject, "task_id": "task_unknown"},
                command_id="cmd_wrong_callback_credential_subject",
            )
        )
    with pytest.raises(NativeAgentError, match="command_id must not contain"):
        peer.enqueue_control(
            ControlCommand(
                type="status",
                run_id=submission.run_id,
                args={"token": submission.run_token},
                command_id=f"cmd_{submission.run_token}",
            )
        )
    valid_callback = token_manager.issue(
        kind="task_callback",
        audience=TASK_CALLBACK_AUDIENCE,
        run_id=submission.run_id,
        tenant_id="tenant_a",
        user_id="user_a",
        ttl_s=60,
        metadata={"task_id": "task_unknown"},
    )
    other_callback = token_manager.issue(
        kind="task_callback",
        audience=TASK_CALLBACK_AUDIENCE,
        run_id=submission.run_id,
        tenant_id="tenant_a",
        user_id="user_a",
        ttl_s=60,
        metadata={"task_id": "task_other"},
    )
    server = create_backend_server(peer, host="127.0.0.1", port=0, admin_token="admin")
    try:
        with serving(server) as base_url:
            with pytest.raises(HTTPError) as exc_info:
                http_json(
                    f"{base_url}/v1/runs/{submission.run_id}/control",
                    {
                        "type": "create_task",
                        "command_id": "cmd_remote_create_task",
                        "args": {
                            "kind": "automation",
                            "request": {"description": "external work"},
                        },
                    },
                    token=submission.run_token,
                )
            assert exc_info.value.code == 400

            callback_queued = http_json(
                f"{base_url}/v1/runs/{submission.run_id}/control",
                {
                    "type": "report_task_result",
                    "command_id": "cmd_remote_callback",
                    "args": {"task_id": "task_unknown", "result": {"answer": "done"}},
                },
                token=valid_callback,
            )
            assert callback_queued["status"] in {"pending", "claimed"}
            callback_receipt_url = (
                f"{base_url}/v1/runs/{submission.run_id}/control/cmd_remote_callback"
            )
            assert eventually(
                lambda: http_json(callback_receipt_url, token=valid_callback)["status"]
                in {"completed", "failed"},
                timeout_s=10,
            )
            with pytest.raises(HTTPError) as duplicate_error:
                http_json(
                    f"{base_url}/v1/runs/{submission.run_id}/control",
                    {
                        "type": "report_task_result",
                        "command_id": "cmd_remote_callback",
                        "args": {"task_id": "task_other", "result": {"answer": "different"}},
                    },
                    token=other_callback,
                )
            assert duplicate_error.value.code == 400

            queued = http_json(
                f"{base_url}/v1/runs/{submission.run_id}/control",
                {
                    "type": "status",
                    "command_id": "cmd_cross_worker",
                    "issuer": "operator-name",
                    "reason": f"requested with {submission.run_token}",
                    "args": {"access_token": "must-not-persist"},
                },
                token=submission.run_token,
            )
            assert queued["status"] in {"pending", "claimed"}

            receipt_url = f"{base_url}/v1/runs/{submission.run_id}/control/cmd_cross_worker"
            assert eventually(
                lambda: http_json(receipt_url, token=submission.run_token)["status"] == "completed",
                timeout_s=10,
            )
            completed = http_json(receipt_url, token=submission.run_token)
            assert completed["result"]["status"] == "ok"
            assert completed["result"]["data"]["state"] == "awaiting_input"

            duplicate = http_json(
                f"{base_url}/v1/runs/{submission.run_id}/control",
                    {
                        "type": "status",
                        "command_id": "cmd_cross_worker",
                        "issuer": "operator-name",
                        "reason": f"requested with {submission.run_token}",
                        "args": {"access_token": "must-not-persist"},
                    },
                token=submission.run_token,
            )
            assert duplicate["status"] == "ok"

            cancel = http_json(
                f"{base_url}/v1/runs/{submission.run_id}/control",
                {
                    "type": "cancel",
                    "command_id": "cmd_cross_cancel",
                    "issuer": "operator-name",
                },
                token=submission.run_token,
            )
            assert cancel.get("command_id") == "cmd_cross_cancel" or cancel["status"] == "ok"
            assert eventually(
                lambda: owner.status(submission.run_id, submission.run_token)["terminal"] is True,
                timeout_s=10,
            )
    finally:
        owner.stop_watchdog()

    events = owner.events(submission.run_id, submission.run_token)["events"]
    received = [
        event
        for event in events
        if event["type"] == "control.command.received"
        and event["data"]["command_id"] == "cmd_cross_worker"
    ]
    assert len(received) == 1
    assert received[0]["data"]["actor"] == "tenant_a/user_a (operator-name)"
    assert received[0]["data"]["token_sha256"] == TokenManager.token_sha256(
        submission.run_token
    )

    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT args, principal, result, token_sha256 FROM command_inbox WHERE command_id=?",
            ("cmd_cross_worker",),
        ).fetchone()
    assert row is not None
    persisted = " ".join(str(value) for value in row if value is not None)
    assert "must-not-persist" not in persisted
    assert submission.run_token not in persisted
    assert '"tenant_id": "tenant_a"' in row[1]
    assert '"user_id": "user_a"' in row[1]
    assert row[3] == TokenManager.token_sha256(submission.run_token)

    owner._heartbeat_own_runs()
    with pytest.raises(NativeAgentError, match="no live owner"):
        peer.enqueue_control(
            ControlCommand(
                type="status",
                run_id=submission.run_id,
                args={"token": submission.run_token},
                command_id="cmd_ownerless",
            )
        )


def test_local_command_returns_transient_callback_and_callback_token_can_enqueue(
    backend_factory: Any,
) -> None:
    workspace = backend_factory.workspace()
    backend = backend_factory.create(workspace=workspace)
    submission = backend.submit_run(
        BackendRunRequest(
            tenant_id="tenant",
            user_id="user",
            workspace_root=workspace,
            instruction="wait",
            runtime_config=runtime_config("run.finish"),
            multi_turn=True,
        )
    )
    assert eventually(
        lambda: backend._record(submission.run_id).state.value == "awaiting_input",
        timeout_s=10,
    )

    assert backend.command_store is not None
    original_append = backend.command_store.append
    competing_drains: list[threading.Thread] = []
    immediate_requirements: list[bool] = []

    def append_with_watchdog_race(
        command: Any, *, max_pending: int, require_empty: bool = False
    ) -> Any:
        receipt = original_append(
            command, max_pending=max_pending, require_empty=require_empty
        )
        if command.type == "create_task":
            immediate_requirements.append(require_empty)
            started = threading.Event()

            def compete() -> None:
                started.set()
                backend._drain_command_inbox(command.run_id)

            thread = threading.Thread(target=compete, daemon=True)
            competing_drains.append(thread)
            thread.start()
            assert started.wait(timeout=1)
            time.sleep(0.05)
        return receipt

    backend.command_store.append = append_with_watchdog_race  # type: ignore[method-assign]
    dispatched_args: dict[str, dict[str, Any]] = {}
    original_dispatch = backend._commands.dispatch

    def capture_dispatch(command: ControlCommand, **kwargs: Any) -> Any:
        dispatched_args[command.command_id] = command.args
        return original_dispatch(command, **kwargs)

    backend._commands.dispatch = capture_dispatch  # type: ignore[method-assign]

    server = create_backend_server(backend, host="127.0.0.1", port=0, admin_token="admin")
    with serving(server) as base_url:
        created = http_json(
            f"{base_url}/v1/runs/{submission.run_id}/control",
            {
                "type": "create_task",
                "command_id": "cmd_create_task",
                "args": {
                    "kind": "automation",
                    "request": {"description": "external work"},
                },
                "issuer": "operator",
            },
            token=submission.run_token,
        )
        callback_token = created["data"]["callback_token"]
        task_id = created["data"]["task_id"]
        assert callback_token and callback_token != "[redacted]"
        assert immediate_requirements == [True]
        for thread in competing_drains:
            thread.join(timeout=2)
            assert not thread.is_alive()

        reported = http_json(
            f"{base_url}/v1/runs/{submission.run_id}/control",
            {
                "type": "report_task_result",
                "command_id": "cmd_callback_report",
                "args": {
                    "task_id": task_id,
                    "result": {
                        "answer": "done",
                        "token_ref": "capability-handle",
                        "password": "legitimate-task-data",
                        "secret_key": "also-task-data",
                    },
                },
                "issuer": "callback-worker",
            },
            token=callback_token,
        )
        assert reported["status"] == "ok"
        assert dispatched_args["cmd_callback_report"]["result"] == {
            "answer": "done",
            "token_ref": "capability-handle",
            "password": "legitimate-task-data",
            "secret_key": "also-task-data",
        }

    persisted = backend.command_receipt(submission.run_id, submission.run_token, "cmd_create_task")
    assert persisted.result is not None
    assert persisted.result["data"]["callback_token"] == "[redacted]"
    report_receipt = backend.command_receipt(
        submission.run_id, submission.run_token, "cmd_callback_report"
    )
    assert report_receipt.result is not None
    assert callback_token not in str(report_receipt.to_json())

    backend.cancel_run(submission.run_id, submission.run_token)


def test_http_resume_recovers_ownerless_run_with_full_command_inbox(
    backend_factory: Any, tmp_path: Path
) -> None:
    workspace = backend_factory.workspace()
    run_root = tmp_path / "runs"
    token_manager = backend_factory.token_manager()
    first = backend_factory.create(
        run_root=run_root,
        workspace=workspace,
        token_manager=token_manager,
        turns=[ModelTurn(response_id="parked", final_text="parked")],
    )
    submission = first.submit_run(
        BackendRunRequest(
            tenant_id="tenant",
            user_id="user",
            workspace_root=workspace,
            instruction="park",
            runtime_config=runtime_config("run.finish"),
            multi_turn=True,
        )
    )
    assert eventually(
        lambda: first._record(submission.run_id).state.value == "awaiting_input",
        timeout_s=10,
    )

    restarted = backend_factory.create(
        run_root=run_root,
        workspace=workspace,
        token_manager=token_manager,
        turns=[ModelTurn(response_id="resumed", final_text="resumed")],
        command_queue_limit=1,
    )
    assert restarted.command_store is not None
    original_claim_command = restarted.command_store.claim_command
    observed_claim_ttls: list[float] = []

    def claim_command_with_ttl(
        run_id: str,
        command_id: str,
        worker_id: str,
        *,
        claim_ttl_s: float,
    ) -> Any:
        observed_claim_ttls.append(claim_ttl_s)
        return original_claim_command(
            run_id,
            command_id,
            worker_id,
            claim_ttl_s=claim_ttl_s,
        )

    restarted.command_store.claim_command = claim_command_with_ttl  # type: ignore[method-assign]
    restarted.command_store.append(
        StoredCommand(
            run_id=submission.run_id,
            command_id="cmd_blocked_head",
            type="status",
            args={},
            principal=CommandPrincipal("tenant", "user", "queued-before-restart"),
        ),
        max_pending=1,
    )
    server = create_backend_server(restarted, host="127.0.0.1", port=0, admin_token="admin")
    with serving(server) as base_url:
        resume_payload = {
            "type": "resume",
            "command_id": "cmd_ownerless_resume",
            "issuer": f"restart-worker-{submission.run_token}",
            "reason": f"recover with {submission.run_token}",
        }
        resumed = http_json(
            f"{base_url}/v1/runs/{submission.run_id}/control",
            resume_payload,
            token=submission.run_token,
        )
        repeated = http_json(
            f"{base_url}/v1/runs/{submission.run_id}/control",
            resume_payload,
            token=submission.run_token,
        )

    assert resumed["status"] == "ok"
    assert resumed["data"]["resumed"] is True
    assert repeated == resumed
    assert observed_claim_ttls == [restarted.command_claim_ttl_s]
    assert submission.run_id in restarted._records
    assert restarted.lease_store is not None
    assert restarted.lease_store.owner(submission.run_id) == restarted._worker_id
    durable_receipt = restarted.command_store.receipt(
        submission.run_id, "cmd_ownerless_resume"
    )
    assert durable_receipt is not None
    assert durable_receipt.status == "completed"
    blocked_receipt = restarted.command_store.receipt(
        submission.run_id, "cmd_blocked_head"
    )
    assert blocked_receipt is not None
    assert blocked_receipt.status == "completed"
    audits = [
        event
        for event in restarted.events(submission.run_id, submission.run_token)["events"]
        if event["type"].startswith("control.command.")
        and event["data"].get("command_id") == "cmd_ownerless_resume"
    ]
    assert audits
    assert submission.run_token not in str(audits)
    assert all(event["data"]["actor"].startswith("tenant/user") for event in audits)
    assert all(
        event["data"]["token_sha256"] == TokenManager.token_sha256(submission.run_token)
        for event in audits
    )

    restarted.cancel_run(submission.run_id, submission.run_token)
    first.cancel_run(submission.run_id, submission.run_token)


@pytest.mark.parametrize("terminal_after_ack", (False, True))
def test_completed_ownerless_resume_preserves_receipt_and_rehydrates_only_if_resumable(
    backend_factory: Any, tmp_path: Path, terminal_after_ack: bool
) -> None:
    workspace = backend_factory.workspace()
    run_root = tmp_path / "runs"
    token_manager = backend_factory.token_manager()
    first = backend_factory.create(
        run_root=run_root,
        workspace=workspace,
        token_manager=token_manager,
        turns=[ModelTurn(response_id="parked", final_text="parked")],
    )
    submission = first.submit_run(
        BackendRunRequest(
            tenant_id="tenant",
            user_id="user",
            workspace_root=workspace,
            instruction="park",
            runtime_config=runtime_config("run.finish"),
            multi_turn=True,
        )
    )
    assert eventually(
        lambda: first._record(submission.run_id).state.value == "awaiting_input",
        timeout_s=10,
    )

    command_store = InMemoryCommandStore()
    stored = StoredCommand(
        run_id=submission.run_id,
        command_id="cmd_completed_resume",
        type="resume",
        args={},
        principal=CommandPrincipal("tenant", "user", "previous-worker"),
        token_sha256=TokenManager.token_sha256(submission.run_token),
        reason="recover",
    )
    command_store.append(stored, max_pending=1, recovery_reservation=True)
    assert command_store.claim_command(
        submission.run_id,
        stored.command_id,
        "crashed-worker",
        claim_ttl_s=30,
    ) is not None
    previous_result = ControlResult(
        run_id=submission.run_id,
        type="resume",
        status="ok",
        data={"run_id": submission.run_id, "resumed": True},
    )
    command_store.acknowledge(
        submission.run_id,
        stored.command_id,
        "crashed-worker",
        previous_result,
    )
    if terminal_after_ack:
        first.cancel_run(submission.run_id, submission.run_token)
        assert eventually(
            lambda: first.status(submission.run_id, submission.run_token)["terminal"] is True,
            timeout_s=10,
        )

    restarted = backend_factory.create(
        run_root=run_root,
        workspace=workspace,
        token_manager=token_manager,
        turns=[ModelTurn(response_id="resumed", final_text="resumed")],
        command_store=command_store,
    )
    receipt = restarted.enqueue_control(
        ControlCommand(
            type="resume",
            run_id=submission.run_id,
            args={"token": submission.run_token},
            issuer="previous-worker",
            reason="recover",
            command_id=stored.command_id,
        )
    )

    assert receipt.status == "completed"
    assert receipt.transient_result == previous_result.to_json()
    assert restarted.lease_store is not None
    if terminal_after_ack:
        assert submission.run_id not in restarted._records
        assert restarted.lease_store.owner(submission.run_id) is None
    else:
        assert submission.run_id in restarted._records
        assert restarted.lease_store.owner(submission.run_id) == restarted._worker_id
    audits = [
        event
        for event in restarted.events(submission.run_id, submission.run_token)["events"]
        if event["type"].startswith("control.command.")
        and event["data"].get("command_id") == stored.command_id
    ]
    assert audits == []

    if not terminal_after_ack:
        restarted.cancel_run(submission.run_id, submission.run_token)
        first.cancel_run(submission.run_id, submission.run_token)


def test_ownerless_resume_keeps_lease_when_receipt_acknowledgement_fails(
    backend_factory: Any, tmp_path: Path
) -> None:
    workspace = backend_factory.workspace()
    run_root = tmp_path / "runs"
    token_manager = backend_factory.token_manager()
    first = backend_factory.create(
        run_root=run_root,
        workspace=workspace,
        token_manager=token_manager,
        turns=[ModelTurn(response_id="parked", final_text="parked")],
    )
    submission = first.submit_run(
        BackendRunRequest(
            tenant_id="tenant",
            user_id="user",
            workspace_root=workspace,
            instruction="park",
            runtime_config=runtime_config("run.finish"),
            multi_turn=True,
        )
    )
    assert eventually(
        lambda: first._record(submission.run_id).state.value == "awaiting_input",
        timeout_s=10,
    )

    restarted = backend_factory.create(
        run_root=run_root,
        workspace=workspace,
        token_manager=token_manager,
        turns=[ModelTurn(response_id="resumed", final_text="resumed")],
    )
    assert restarted.command_store is not None
    original_acknowledge = restarted.command_store.acknowledge
    failures_remaining = 1

    def fail_resume_acknowledgement(
        run_id: str,
        command_id: str,
        worker_id: str,
        result: ControlResult,
    ) -> Any:
        nonlocal failures_remaining
        if command_id == "cmd_ack_failure" and failures_remaining:
            failures_remaining -= 1
            raise RuntimeError("simulated durable acknowledgement failure")
        return original_acknowledge(run_id, command_id, worker_id, result)

    restarted.command_store.acknowledge = fail_resume_acknowledgement  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="simulated durable acknowledgement failure"):
        restarted.enqueue_control(
            ControlCommand(
                type="resume",
                run_id=submission.run_id,
                args={"token": submission.run_token},
                command_id="cmd_ack_failure",
            )
        )

    assert submission.run_id in restarted._records
    assert restarted.lease_store is not None
    assert restarted.lease_store.owner(submission.run_id) == restarted._worker_id
    repaired = restarted.enqueue_control(
        ControlCommand(
            type="resume",
            run_id=submission.run_id,
            args={"token": submission.run_token},
            command_id="cmd_ack_failure",
        )
    )
    assert repaired.status == "completed"
    audits = [
        event["type"]
        for event in restarted.events(submission.run_id, submission.run_token)["events"]
        if event["type"].startswith("control.command.")
        and event["data"].get("command_id") == "cmd_ack_failure"
    ]
    assert audits == ["control.command.completed"]

    restarted.command_store.acknowledge = original_acknowledge  # type: ignore[method-assign]
    restarted.cancel_run(submission.run_id, submission.run_token)
    first.cancel_run(submission.run_id, submission.run_token)
