"""Offline hosted golden path using the Reference durable-inbox assembly.

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

from monoid_agent_kernel.contracts import (  # noqa: E402
    AgentRuntimeConfig,
    ControlCommand,
    ModelTurn,
)
from monoid_agent_kernel.errors import PermissionDenied  # noqa: E402
from monoid_agent_kernel.providers import FakeModelAdapter  # noqa: E402
from monoid_agent_kernel.reference.backend import (  # noqa: E402
    BackendRunRequest,
    RunnerBackend,
    TokenManager,
)
from monoid_agent_kernel.reference.stores import (  # noqa: E402
    SqliteCheckpointStore,
    SqliteCommandStore,
    SqliteLeaseStore,
)


def _offline_adapter() -> FakeModelAdapter:
    return FakeModelAdapter(
        turns=[
            ModelTurn(response_id="hosted-ready", final_text="Ready for product input."),
            ModelTurn(response_id="hosted-followup", final_text="Processed product input."),
        ]
    )


def _reference_inbox_backend(
    *,
    root: Path,
    database: Path,
    token_manager: TokenManager,
    workspaces: Path,
    observed_gateway_tokens: list[str],
) -> RunnerBackend:
    def adapter_factory(_spec: Any, gateway_token: str) -> FakeModelAdapter:
        observed_gateway_tokens.append(gateway_token)
        return _offline_adapter()

    return RunnerBackend(
        run_root=root / "runs",
        token_manager=token_manager,
        allowed_workspace_roots=(workspaces,),
        llm_gateway_url="http://offline.invalid/internal/llm/turns",
        model_adapter_factory=adapter_factory,
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
        except OSError as exc:
            raise RuntimeError(f"durable credential scan could not read {path}") from exc
        if any(needle in data for needle in needles):
            return True
    return False


def run_hosted_product(root: Path) -> dict[str, Any]:
    """Exercise one Reference inbox assembly across two backend instances."""

    workspaces = root / "workspaces"
    tenant_a_workspace = workspaces / "tenant-a"
    tenant_b_workspace = workspaces / "tenant-b"
    tenant_a_workspace.mkdir(parents=True, exist_ok=True)
    tenant_b_workspace.mkdir(parents=True, exist_ok=True)
    database = root / "control-plane.db"
    signing_secret = "offline-hosted-example-secret-32b"
    token_manager = TokenManager.from_secret(signing_secret)
    observed_gateway_tokens: list[str] = []
    owner = _reference_inbox_backend(
        root=root,
        database=database,
        token_manager=token_manager,
        workspaces=workspaces,
        observed_gateway_tokens=observed_gateway_tokens,
    )
    peer = _reference_inbox_backend(
        root=root,
        database=database,
        token_manager=token_manager,
        workspaces=workspaces,
        observed_gateway_tokens=observed_gateway_tokens,
    )
    checked_store = SqliteCheckpointStore(database)
    submissions = []
    checked_loads: dict[str, dict[str, str]] = {}
    callback_token = ""
    approval_callback_token = ""
    cross_tenant_access_denied = False
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
            checkpoint_load = checked_store.latest_checked(submission.run_id)
            metadata_load = checked_store.run_metadata_checked(submission.run_id)
            if not checkpoint_load.ok or not metadata_load.ok:
                raise RuntimeError(
                    "hosted durable state could not be loaded: "
                    f"checkpoint={checkpoint_load.status}, metadata={metadata_load.status}"
                )
            checked_loads[submission.run_id] = {
                "checkpoint": checkpoint_load.status,
                "run_metadata": metadata_load.status,
            }
        owner.start_watchdog()

        primary = submissions[0]
        try:
            peer.enqueue_control(
                ControlCommand(
                    type="status",
                    run_id=primary.run_id,
                    args={"token": submissions[1].run_token},
                    issuer="hosted-api",
                    reason="prove cross-tenant denial",
                    command_id="hosted-cross-tenant-status",
                )
            )
        except PermissionDenied:
            cross_tenant_access_denied = True
        if not cross_tenant_access_denied:
            raise RuntimeError("cross-tenant command unexpectedly passed authorization")

        subscription = owner.subscribe_events(
            primary.run_id, primary.run_token, from_seq=1
        )
        first_page = subscription.poll(limit=500)
        replay_free_page = subscription.poll(limit=500)

        status_command = ControlCommand(
            type="status",
            run_id=primary.run_id,
            args={"token": primary.run_token},
            issuer="hosted-api",
            reason="offline golden path",
            command_id="hosted-status-1",
        )
        peer.enqueue_control(status_command)
        status_receipt = _wait_receipt(
            peer, primary.run_id, primary.run_token, "hosted-status-1"
        )
        duplicate_status_receipt = peer.enqueue_control(status_command).to_json()

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

    historical = _reference_inbox_backend(
        root=root,
        database=database,
        token_manager=token_manager,
        workspaces=workspaces,
        observed_gateway_tokens=observed_gateway_tokens,
    )
    try:
        historical_events = historical.subscribe_events(
            submissions[0].run_id, submissions[0].run_token, from_seq=1
        ).poll(limit=500)["events"]
    finally:
        historical.shutdown(drain=False)

    return {
        "status": "ok",
        "runtime_profile": "reference-inbox",
        "tenants": ["tenant_a", "tenant_b"],
        "run_ids": [submission.run_id for submission in submissions],
        "checked_loads": checked_loads,
        "initial_event_count": len(first_page["events"]),
        "replay_count": len(replay_free_page["events"]),
        "status_receipt": status_receipt["status"],
        "status_result": status_receipt["result"],
        "status_duplicate_matches": duplicate_status_receipt == status_receipt,
        "task_receipt": task_receipt["status"],
        "task_result": task_receipt["result"],
        "approval_receipt": approval_receipt["status"],
        "approval_result": approval_receipt["result"],
        "historical_event_count": len(historical_events),
        "tenant_usage": tenant_usage,
        "cross_tenant_access_denied": cross_tenant_access_denied,
        "observed_gateway_token_count": len(observed_gateway_tokens),
        "credential_leak_detected": _scan_for_credentials(
            root,
            (
                submissions[0].run_token,
                submissions[1].run_token,
                callback_token,
                approval_callback_token,
                signing_secret,
                *observed_gateway_tokens,
            ),
        ),
        "network_required": False,
    }


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="monoid-hosted-embedding-") as tmp:
        print(json.dumps(run_hosted_product(Path(tmp)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
