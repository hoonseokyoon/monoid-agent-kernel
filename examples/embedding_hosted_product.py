"""Offline golden path for a hosted, multi-tenant Monoid control plane.

Run from a checkout with::

    python examples/embedding_hosted_product.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from monoid_agent_kernel import AgentRuntimeConfig  # noqa: E402
from monoid_agent_kernel.core.control import ControlCommand  # noqa: E402
from monoid_agent_kernel.providers.base import ModelTurn  # noqa: E402
from monoid_agent_kernel.providers.fake import FakeModelAdapter  # noqa: E402
from monoid_agent_kernel.reference._shared.tokens import TokenManager  # noqa: E402
from monoid_agent_kernel.reference.backend.service import (  # noqa: E402
    BackendRunRequest,
    RunnerBackend,
)
from monoid_agent_kernel.reference.command_inbox import SqliteCommandStore  # noqa: E402
from monoid_agent_kernel.reference.stores.sqlite import (  # noqa: E402
    SqliteCheckpointStore,
    SqliteLeaseStore,
)


def _adapter_factory(_spec: Any, _gateway_token: str) -> FakeModelAdapter:
    return FakeModelAdapter(
        turns=[
            ModelTurn(response_id="hosted-ready", final_text="Ready for product input."),
            ModelTurn(response_id="hosted-followup", final_text="Processed product input."),
        ]
    )


def _backend(
    *, root: Path, database: Path, token_manager: TokenManager, workspaces: Path
) -> RunnerBackend:
    return RunnerBackend(
        run_root=root / "runs",
        token_manager=token_manager,
        allowed_workspace_roots=(workspaces,),
        llm_gateway_url="http://offline.invalid/internal/llm/turns",
        model_adapter_factory=_adapter_factory,
        checkpoint_store=SqliteCheckpointStore(database),
        lease_store=SqliteLeaseStore(database),
        command_store=SqliteCommandStore(database),
        watchdog_interval_s=0.02,
    )


def _wait_receipt(
    backend: RunnerBackend,
    run_id: str,
    token: str,
    command_id: str,
    *,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        receipt = backend.command_receipt(run_id, token, command_id).to_json()
        if receipt["status"] in {"completed", "failed"}:
            return receipt
        time.sleep(0.02)
    raise TimeoutError(f"command receipt did not settle: {command_id}")


def _wait_state(
    backend: RunnerBackend,
    run_id: str,
    token: str,
    expected: str,
    *,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = backend.status(run_id, token)
        if status["state"] == expected:
            return status
        time.sleep(0.02)
    raise TimeoutError(f"run did not reach {expected}: {run_id}")


def _scan_for_credentials(root: Path, credentials: tuple[str, ...]) -> bool:
    needles = tuple(item.encode("utf-8") for item in credentials if item)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if any(needle in data for needle in needles):
            return True
    return False


def run_hosted_product(root: Path) -> dict[str, Any]:
    """Exercise shared ownership, durable commands, receipts, and cursor streaming."""

    workspaces = root / "workspaces"
    tenant_a_workspace = workspaces / "tenant-a"
    tenant_b_workspace = workspaces / "tenant-b"
    tenant_a_workspace.mkdir(parents=True, exist_ok=True)
    tenant_b_workspace.mkdir(parents=True, exist_ok=True)
    database = root / "control-plane.db"
    token_manager = TokenManager.from_secret("offline-hosted-example-secret-32b")
    owner = _backend(
        root=root, database=database, token_manager=token_manager, workspaces=workspaces
    )
    peer = _backend(
        root=root, database=database, token_manager=token_manager, workspaces=workspaces
    )
    submissions = []
    callback_token = ""
    approval_callback_token = ""
    try:
        for tenant, user, workspace in (
            ("tenant_a", "user_a", tenant_a_workspace),
            ("tenant_b", "user_b", tenant_b_workspace),
        ):
            submissions.append(
                owner.submit_run(
                    BackendRunRequest(
                        tenant_id=tenant,
                        user_id=user,
                        workspace_root=workspace,
                        instruction="Initialize the hosted session.",
                        runtime_config=AgentRuntimeConfig(
                            definition_id=f"hosted-{tenant}"
                        ),
                        multi_turn=True,
                    )
                )
            )
        for submission in submissions:
            _wait_state(
                owner,
                submission.run_id,
                submission.run_token,
                "awaiting_input",
            )
        owner.start_watchdog()

        primary = submissions[0]
        subscription = owner.subscribe_events(
            primary.run_id, primary.run_token, from_seq=1
        )
        first_page = subscription.poll(limit=500)
        replay_free_page = subscription.poll(limit=500)

        peer.enqueue_control(
            ControlCommand(
                type="status",
                run_id=primary.run_id,
                args={"token": primary.run_token},
                issuer="hosted-api",
                reason="offline golden path",
                command_id="hosted-status-1",
            )
        )
        status_receipt = _wait_receipt(
            peer, primary.run_id, primary.run_token, "hosted-status-1"
        )

        created = owner.enqueue_control(
            ControlCommand(
                type="create_task",
                run_id=primary.run_id,
                args={
                    "token": primary.run_token,
                    "kind": "automation",
                    "request": {"description": "Verify the external callback path."},
                },
                issuer="hosted-api",
                command_id="hosted-task-1",
            )
        )
        created_result = created.transient_result or {}
        task_data = dict(created_result.get("data") or {})
        callback_token = str(task_data["callback_token"])
        task_id = str(task_data["task_id"])
        peer.enqueue_control(
            ControlCommand(
                type="report_task_result",
                run_id=primary.run_id,
                args={
                    "token": callback_token,
                    "task_id": task_id,
                    "result": {"verified": True, "token_ref": "vault://hosted/example"},
                },
                issuer="external-worker",
                command_id="hosted-task-result-1",
            )
        )
        task_receipt = _wait_receipt(
            peer, primary.run_id, callback_token, "hosted-task-result-1"
        )

        approval_created = owner.enqueue_control(
            ControlCommand(
                type="create_task",
                run_id=primary.run_id,
                args={
                    "token": primary.run_token,
                    "kind": "hitl",
                    "request": {
                        "prompt": "Approve the offline deployment?",
                        "choices": ["Approve", "Deny"],
                    },
                },
                issuer="hosted-api",
                command_id="hosted-approval-task-1",
            )
        )
        approval_data = dict((approval_created.transient_result or {}).get("data") or {})
        approval_callback_token = str(approval_data["callback_token"])
        approval_task_id = str(approval_data["task_id"])
        peer.enqueue_control(
            ControlCommand(
                type="approve",
                run_id=primary.run_id,
                args={"token": approval_callback_token, "task_id": approval_task_id},
                issuer="product-approver",
                reason="offline policy approved",
                command_id="hosted-approval-decision-1",
            )
        )
        approval_receipt = _wait_receipt(
            peer,
            primary.run_id,
            approval_callback_token,
            "hosted-approval-decision-1",
        )

        for submission in submissions:
            owner.cancel_run(submission.run_id, submission.run_token)
        owner.drain(timeout_s=5)
        tenant_usage = {
            tenant: owner.tenant_usage(tenant)
            for tenant in ("tenant_a", "tenant_b")
        }
    finally:
        for submission in submissions:
            try:
                owner.cancel_run(submission.run_id, submission.run_token)
            except Exception:
                pass
        owner.drain(timeout_s=5)
        owner.stop_watchdog()
        owner.shutdown(drain=False)
        peer.shutdown(drain=False)

    historical = _backend(
        root=root, database=database, token_manager=token_manager, workspaces=workspaces
    )
    try:
        historical_events = historical.subscribe_events(
            submissions[0].run_id, submissions[0].run_token, from_seq=1
        ).poll(limit=500)["events"]
    finally:
        historical.shutdown(drain=False)

    return {
        "status": "ok",
        "tenants": ["tenant_a", "tenant_b"],
        "run_ids": [submission.run_id for submission in submissions],
        "initial_event_count": len(first_page["events"]),
        "replay_count": len(replay_free_page["events"]),
        "status_receipt": status_receipt["status"],
        "task_receipt": task_receipt["status"],
        "approval_receipt": approval_receipt["status"],
        "historical_event_count": len(historical_events),
        "tenant_usage": tenant_usage,
        "credential_leak_detected": _scan_for_credentials(
            root,
            (
                submissions[0].run_token,
                submissions[1].run_token,
                callback_token,
                approval_callback_token,
            ),
        ),
        "network_required": False,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="monoid-hosted-embedding-") as tmp:
        print(json.dumps(run_hosted_product(Path(tmp)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
