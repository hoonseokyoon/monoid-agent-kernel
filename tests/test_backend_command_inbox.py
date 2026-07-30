from __future__ import annotations

import json
import sqlite3
import threading
import time
from urllib.error import HTTPError
from pathlib import Path
from typing import Any

import pytest

from support.http import http_json, serving
from support.runtime import runtime_config, tool_binding
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
from monoid_agent_kernel.core.control import ControlCommand
from monoid_agent_kernel.core.tool_surface import ToolScope
from monoid_agent_kernel.permissions import matches_path_patterns


def test_fresh_queued_runtime_config_rejects_a_bare_negation(
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
    config = runtime_config(
        bindings=(
            tool_binding("fs.read", scope=ToolScope(allowed_paths=("!literal",))),
        ),
        version=2,
    ).to_json()
    config["tools"][0]["scope"].pop("path_pattern_encoding")
    config["tools"][0]["scope"]["allowed_paths"] = ["!ambiguous"]

    with pytest.raises(ValueError, match="negated path patterns"):
        backend.enqueue_control(
            ControlCommand(
                type="replace_runtime_config",
                run_id=submission.run_id,
                command_id="cmd_ambiguous_scope",
                args={
                    "token": submission.run_token,
                    "expected_version": 1,
                    "config": config,
                },
            )
        )

    assert backend.command_store is not None
    assert backend.command_store.read_command(submission.run_id, "cmd_ambiguous_scope") is None
    backend.cancel_run(submission.run_id, submission.run_token)
    backend.wait_for_run(submission.run_id, timeout_s=20)


def test_retained_backslash_bang_scope_stays_inert_through_inbox_dispatch(
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

    legacy_config = runtime_config(
        bindings=(
            tool_binding(
                "fs.read",
                scope=ToolScope(allowed_paths=("placeholder",)),
            ),
        ),
        version=2,
    ).to_json()
    legacy_scope = legacy_config["tools"][0]["scope"]
    legacy_scope["allowed_paths"] = [r"\!inert", "secret//file"]
    legacy_scope.pop("path_pattern_encoding", None)
    stored = StoredCommand(
        run_id=submission.run_id,
        command_id="cmd_legacy_backslash_scope",
        type="replace_runtime_config",
        args={
            "expected_version": 1,
            "config": legacy_config,
        },
        principal=CommandPrincipal("tenant", "user", "operator"),
    )
    assert backend.command_store is not None
    backend.command_store.append(stored, max_pending=10)

    completed = backend._drain_command_inbox(submission.run_id)

    assert completed[stored.command_id].status == "ok", completed[stored.command_id].error
    effective = backend.current_runtime_config(submission.run_id)
    assert effective is not None
    restored_scope = effective.tools[0].scope
    assert restored_scope.allowed_paths == (r"\!inert", "secret//file")
    assert matches_path_patterns("!inert", restored_scope.allowed_paths) is False
    assert matches_path_patterns("secret/file", restored_scope.allowed_paths) is True

    events = backend.events(submission.run_id, submission.run_token)["events"]
    received = next(
        event["data"]
        for event in events
        if event["type"] == "control.command.received"
        and event["data"]["command_id"] == stored.command_id
    )
    assert received["args_keys"] == ["config", "expected_version"]
    assert all("durable_runtime_config" not in key for key in received["args_keys"])
    persisted = backend.command_store.read_command(submission.run_id, stored.command_id)
    assert persisted is not None
    assert "_monoid_durable_runtime_config" not in json.dumps(persisted.to_json())

    backend.cancel_run(submission.run_id, submission.run_token)
    backend.wait_for_run(submission.run_id, timeout_s=20)


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


@pytest.mark.parametrize("store_kind", ("memory", "sqlite"))
def test_enqueue_control_redacts_bearer_reintroduced_by_json_coercion(
    backend_factory: Any,
    tmp_path: Path,
    store_kind: str,
) -> None:
    workspace = backend_factory.workspace(f"workspace-{store_kind}")
    db = tmp_path / "commands.db"
    command_store = (
        InMemoryCommandStore() if store_kind == "memory" else SqliteCommandStore(db)
    )
    token_manager = backend_factory.token_manager()
    backend = backend_factory.create(
        workspace=workspace,
        token_manager=token_manager,
        command_store=command_store,
    )
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

    captured: list[ControlCommand] = []
    original_dispatch = backend._commands.dispatch

    def capture_dispatch(command: ControlCommand, **kwargs: Any) -> Any:
        captured.append(command)
        return original_dispatch(command, **kwargs)

    backend._commands.dispatch = capture_dispatch  # type: ignore[method-assign]

    class OpaqueValue:
        def __init__(self, bearer: str) -> None:
            self.bearer = bearer

        def __repr__(self) -> str:
            return f"OpaqueValue({self.bearer})"

    def command_for(token: str) -> tuple[ControlCommand, OpaqueValue]:
        opaque = OpaqueValue(token)
        return (
            ControlCommand(
                type="status",
                run_id=submission.run_id,
                command_id="cmd_repr_bearer",
                args={
                    "token": token,
                    "bytes": token.encode(),
                    "opaque": opaque,
                    "nested": [token.encode(), opaque],
                    "plain": f"prefix-{token}",
                },
            ),
            opaque,
        )

    first_command, first_opaque = command_for(submission.run_token)
    first_receipt = backend.enqueue_control(first_command)
    assert first_receipt.status == "completed"
    assert len(captured) == 1
    assert captured[0].args["bytes"] == submission.run_token.encode()
    assert captured[0].args["opaque"] is first_opaque
    assert captured[0].args["nested"][1] is first_opaque

    persisted = command_store.read_command(submission.run_id, "cmd_repr_bearer")
    assert persisted is not None
    assert persisted.args == {
        "bytes": "b'[redacted]'",
        "opaque": "OpaqueValue([redacted])",
        "nested": ["b'[redacted]'", "OpaqueValue([redacted])"],
        "plain": "prefix-[redacted]",
    }
    assert submission.run_token not in str(persisted.args)

    rotated_token = token_manager.issue(
        kind="run_access",
        audience=BACKEND_AUDIENCE,
        run_id=submission.run_id,
        tenant_id="tenant",
        user_id="user",
        ttl_s=60,
    )
    assert rotated_token != submission.run_token
    rotated_command, _ = command_for(rotated_token)
    duplicate = backend.enqueue_control(rotated_command)
    assert duplicate.status == "completed"
    assert duplicate.result == first_receipt.result
    assert len(captured) == 1

    if store_kind == "sqlite":
        with sqlite3.connect(db) as connection:
            raw_args = connection.execute(
                "SELECT args FROM command_inbox WHERE run_id=? AND command_id=?",
                (submission.run_id, "cmd_repr_bearer"),
            ).fetchone()
        assert raw_args is not None
        assert submission.run_token not in raw_args[0]
        assert rotated_token not in raw_args[0]

    backend.cancel_run(submission.run_id, submission.run_token)


def test_direct_control_ingress_is_canonical_before_sqlite_identity(
    backend_factory: Any,
    tmp_path: Path,
) -> None:
    workspace = backend_factory.workspace("command-json-ingress")
    command_store = SqliteCommandStore(tmp_path / "commands.db")
    backend = backend_factory.create(workspace=workspace, command_store=command_store)
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

    lone_surrogate = chr(0xD800)

    def command() -> ControlCommand:
        return ControlCommand(
            type="status",
            run_id=submission.run_id,
            command_id=f"cmd_{lone_surrogate}",
            issuer=f"operator_{lone_surrogate}",
            reason=f"check_{lone_surrogate}",
            args={
                "token": submission.run_token,
                "payload": {
                    "score": float("nan"),
                    "text": f"bad{lone_surrogate}text",
                },
            },
        )

    first = backend.enqueue_control(command())
    duplicate = backend.enqueue_control(command())

    assert first.status == "completed"
    assert duplicate.result == first.result
    assert first.command_id == "cmd_\ufffd"
    persisted = command_store.read_command(submission.run_id, f"cmd_{lone_surrogate}")
    assert persisted is not None
    assert persisted.command_id == "cmd_\ufffd"
    assert persisted.principal.issuer == "operator_\ufffd"
    assert persisted.reason == "check_\ufffd"
    assert persisted.args["payload"] == {
        "score": None,
        "text": "bad\ufffdtext",
    }

    invalid_config = runtime_config("run.finish", version=2).to_json()
    with pytest.raises(ValueError, match="expected_version must be an integer"):
        backend.enqueue_control(
            ControlCommand(
                type="replace_runtime_config",
                run_id=submission.run_id,
                command_id="cmd_bad_expected_version",
                args={
                    "token": submission.run_token,
                    "expected_version": float("nan"),
                    "config": invalid_config,
                },
            )
        )
    assert (
        command_store.read_command(submission.run_id, "cmd_bad_expected_version")
        is None
    )

    with pytest.raises(ValueError, match="before must be a finite number"):
        backend.enqueue_control(
            ControlCommand(
                type="revoke_capability",
                run_id=submission.run_id,
                command_id="cmd_bad_revoke_before",
                args={"token": submission.run_token, "before": float("inf")},
            )
        )
    assert command_store.read_command(submission.run_id, "cmd_bad_revoke_before") is None

    backend.cancel_run(submission.run_id, submission.run_token)
    backend.wait_for_run(submission.run_id, timeout_s=20)
