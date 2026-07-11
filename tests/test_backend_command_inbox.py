from __future__ import annotations

import sqlite3
from urllib.error import HTTPError
from pathlib import Path
from typing import Any

import pytest

from support.http import http_json, serving
from support.runtime import runtime_config
from support.waiting import eventually

from monoid_agent_kernel.reference.backend.http import create_backend_server
from monoid_agent_kernel.reference.backend.service import BackendRunRequest
from monoid_agent_kernel.reference.command_inbox import SqliteCommandStore
from monoid_agent_kernel.reference.stores.sqlite import SqliteCheckpointStore, SqliteLeaseStore
from monoid_agent_kernel.errors import PermissionDenied
from monoid_agent_kernel.identifiers import BACKEND_AUDIENCE
from monoid_agent_kernel.core.control import ControlCommand


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

    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT args, principal, result FROM command_inbox WHERE command_id=?",
            ("cmd_cross_worker",),
        ).fetchone()
    assert row is not None
    persisted = " ".join(str(value) for value in row if value is not None)
    assert "must-not-persist" not in persisted
    assert submission.run_token not in persisted
    assert '"tenant_id": "tenant_a"' in row[1]
    assert '"user_id": "user_a"' in row[1]


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
                    },
                },
                "issuer": "callback-worker",
            },
            token=callback_token,
        )
        assert reported["status"] == "ok"

    persisted = backend.command_receipt(submission.run_id, submission.run_token, "cmd_create_task")
    assert persisted.result is not None
    assert persisted.result["data"]["callback_token"] == "[redacted]"
    report_receipt = backend.command_receipt(
        submission.run_id, submission.run_token, "cmd_callback_report"
    )
    assert report_receipt.result is not None
    assert callback_token not in str(report_receipt.to_json())

    backend.cancel_run(submission.run_id, submission.run_token)
