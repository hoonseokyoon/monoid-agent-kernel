from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.serial, pytest.mark.service, pytest.mark.slow]

if os.environ.get("MONOID_SERVICE_PROFILE") not in {"temporal", "combined"}:
    pytest.skip("Temporal service profile is not selected", allow_module_level=True)

from temporalio import activity  # noqa: E402
from temporalio.api.workflowservice.v1 import GetSystemInfoRequest  # noqa: E402
from temporalio.client import WorkflowHistory  # noqa: E402
from temporalio.exceptions import ApplicationError, WorkflowAlreadyStartedError  # noqa: E402
from temporalio.service import RPCError, RPCStatusCode  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402
from temporalio.worker import Replayer, Worker  # noqa: E402

from monoid_agent_kernel.adapters.temporal import (  # noqa: E402
    TEMPORAL_DRIVE_ACTIVATION_ACTIVITY,
    TEMPORAL_STATUS_QUERY,
    TemporalActivationResult,
    TemporalRunPolicy,
    TemporalRunStatus,
    TemporalSignalWithStartTransport,
    temporal_workflow_id,
)
from monoid_agent_kernel.adapters.temporal.workflow import (  # noqa: E402
    TemporalRunWorkflow,
)
from monoid_agent_kernel.hosting import (  # noqa: E402
    AdmissionRequest,
    AdmittedCommand,
)


_ACTIVITY_ORDER: dict[str, list[int]] = defaultdict(list)
_ACTIVITY_STARTED: dict[tuple[str, int], asyncio.Event] = {}
_ACTIVITY_RELEASE: dict[tuple[str, int], asyncio.Event] = {}
_ACTIVITY_RETRYABLE_FAILURES: dict[tuple[str, int], int] = {}


class _RaisingTemporalClient:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def start_workflow(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise self.error


@activity.defn(name=TEMPORAL_DRIVE_ACTIVATION_ACTIVITY)
async def _drive_test_activation(payload: dict[str, Any]) -> dict[str, Any]:
    command = AdmittedCommand.from_json(payload)
    coordinate = (command.run_id, command.command_sequence)
    _ACTIVITY_ORDER[command.run_id].append(command.command_sequence)
    failures_remaining = _ACTIVITY_RETRYABLE_FAILURES.get(coordinate, 0)
    if failures_remaining:
        _ACTIVITY_RETRYABLE_FAILURES[coordinate] = failures_remaining - 1
        raise ApplicationError(
            "retryable qualification failure",
            type="monoid.qualification_retryable",
            non_retryable=False,
        )
    started = _ACTIVITY_STARTED.get(coordinate)
    if started is not None:
        started.set()
    release = _ACTIVITY_RELEASE.get(coordinate)
    if release is not None:
        await release.wait()
    return TemporalActivationResult.from_command(
        command,
        receipt_ref=f"checkpoint:{command.run_id}/{command.command_sequence + 1}",
        terminal=command.command_id.endswith("-terminal"),
    ).to_json()


def _admitted_command(
    run_id: str,
    sequence: int,
    *,
    terminal: bool = False,
) -> AdmittedCommand:
    return AdmittedCommand.from_request(
        AdmissionRequest(
            run_id=run_id,
            command_id=(f"command-{sequence}-terminal" if terminal else f"command-{sequence}"),
            kind="control" if terminal else "input",
            request_digest=f"{sequence:064x}",
            payload_ref=f"object:private/{run_id}/{sequence}",
        ),
        sequence,
    )


def _prepare_temporal_cli() -> tuple[str, str]:
    cli_version = os.environ.get("MONOID_TEMPORAL_CLI_VERSION")
    if not cli_version:
        pytest.fail("MONOID_TEMPORAL_CLI_VERSION is required for the selected service profile")
    assert cli_version.startswith("v")

    root = Path(__file__).resolve().parents[2]
    cache_dir = Path(os.environ.get("MONOID_TEMPORAL_CLI_CACHE", root / ".tmp/temporal-cli-cache"))
    prepared = subprocess.run(
        [
            sys.executable,
            str(root / "tools/v023_ci.py"),
            "prepare-temporal-cli",
            "--cache-dir",
            str(cache_dir),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stderr or prepared.stdout
    artifact = json.loads(prepared.stdout)
    assert artifact["version"] == cli_version
    return str(artifact["executable"]), str(artifact["embedded_server"])


def _decoded_history_payloads(history: WorkflowHistory) -> list[object]:
    decoded: list[object] = []
    pending: list[object] = [json.loads(history.to_json())]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            metadata = value.get("metadata")
            data = value.get("data")
            if (
                isinstance(metadata, dict)
                and metadata.get("encoding") == "anNvbi9wbGFpbg=="
                and isinstance(data, str)
            ):
                decoded.append(json.loads(base64.b64decode(data)))
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return decoded


def _assert_history_payload_privacy(history: WorkflowHistory) -> None:
    forbidden_keys = {
        "prompt",
        "response",
        "reasoning",
        "workspace_bytes",
        "model_result",
        "raw_checkpoint",
        "credential",
        "provider_exception_text",
    }
    for payload in _decoded_history_payloads(history):
        pending = [payload]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                assert forbidden_keys.isdisjoint(value)
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
            elif isinstance(value, str):
                assert "raw-private-model-response" not in value


async def _wait_for_activity_count(run_id: str, count: int) -> None:
    async def wait() -> None:
        while len(_ACTIVITY_ORDER[run_id]) < count:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), timeout=10)


def test_pinned_temporal_local_server_executes_workflow() -> None:
    executable, expected_server_version = _prepare_temporal_cli()

    async def run() -> tuple[TemporalRunStatus, str, WorkflowHistory]:
        run_id = f"smoke-{uuid.uuid4().hex}"
        async with await WorkflowEnvironment.start_local(
            dev_server_existing_path=executable,
            dev_server_log_level="warn",
        ) as environment:
            system_info = await environment.client.service_client.workflow_service.get_system_info(
                GetSystemInfoRequest()
            )
            task_queue = f"monoid-v023-{uuid.uuid4()}"
            transport = TemporalSignalWithStartTransport(
                client=environment.client,
                event_loop=asyncio.get_running_loop(),
                workflow_task_queue=task_queue,
                run_policy=TemporalRunPolicy(activity_task_queue=task_queue),
            )
            accepted = await transport.dispatch_async(_admitted_command(run_id, 1, terminal=True))
            assert accepted.status == "accepted"
            handle = environment.client.get_workflow_handle(
                temporal_workflow_id(run_id),
                result_type=dict,
            )
            async with Worker(
                environment.client,
                task_queue=task_queue,
                workflows=[TemporalRunWorkflow],
                activities=[_drive_test_activation],
            ):
                result = TemporalRunStatus.from_json(await handle.result())
                history = await handle.fetch_history()
                await Replayer(workflows=[TemporalRunWorkflow]).replay_workflow(history)
                return result, system_info.server_version, history

    result, server_version, history = asyncio.run(run())
    assert result.phase == "terminal"
    assert result.next_command_sequence == 2
    assert server_version == expected_server_version
    _assert_history_payload_privacy(history)


def test_temporal_signal_with_start_orders_and_deduplicates_on_actual_server() -> None:
    executable, _ = _prepare_temporal_cli()

    async def run() -> None:
        run_id = f"run-{uuid.uuid4().hex}"
        _ACTIVITY_ORDER.pop(run_id, None)
        async with await WorkflowEnvironment.start_local(
            dev_server_existing_path=executable,
            dev_server_log_level="warn",
        ) as environment:
            task_queue = f"monoid-v023-{uuid.uuid4().hex}"
            policy = TemporalRunPolicy(activity_task_queue=task_queue)
            transport = TemporalSignalWithStartTransport(
                client=environment.client,
                event_loop=asyncio.get_running_loop(),
                workflow_task_queue=task_queue,
                run_policy=policy,
            )
            command_1 = _admitted_command(run_id, 1)
            command_2 = _admitted_command(run_id, 2)
            command_3 = _admitted_command(run_id, 3, terminal=True)
            workflow_id = temporal_workflow_id(run_id)

            async with Worker(
                environment.client,
                task_queue=task_queue,
                workflows=[TemporalRunWorkflow],
                activities=[_drive_test_activation],
            ):
                accepted_2 = await asyncio.to_thread(transport.dispatch, command_2)
                duplicate_2 = await asyncio.to_thread(transport.dispatch, command_2)
                handle = environment.client.get_workflow_handle(workflow_id, result_type=dict)
                pending = TemporalRunStatus.from_json(
                    await handle.query(TEMPORAL_STATUS_QUERY, result_type=dict)
                )
                assert pending.pending_count == 1
                assert pending.pending_head_sequence == 2

                accepted_1 = await asyncio.to_thread(transport.dispatch, command_1)
                duplicate_1 = await asyncio.to_thread(transport.dispatch, command_1)
                accepted_3 = await asyncio.to_thread(transport.dispatch, command_3)
                raw_result = await handle.result()
                terminal = TemporalRunStatus.from_json(raw_result)

                assert {
                    accepted_1.dispatch_ref,
                    duplicate_1.dispatch_ref,
                    accepted_2.dispatch_ref,
                    duplicate_2.dispatch_ref,
                    accepted_3.dispatch_ref,
                } == {f"temporal:{workflow_id}"}
                assert _ACTIVITY_ORDER[run_id] == [1, 2, 3]
                assert terminal.phase == "terminal"
                assert terminal.next_command_sequence == 4
                assert terminal.duplicate_signal_count >= 1

                history = await handle.fetch_history()
                await Replayer(workflows=[TemporalRunWorkflow]).replay_workflow(history)
                history_json = history.to_json()
                assert "raw-private-model-response" not in history_json
                _assert_history_payload_privacy(history)

                closed = await asyncio.to_thread(
                    transport.dispatch,
                    _admitted_command(run_id, 4),
                )
                assert closed.status == "rejected"
                assert closed.error_code == "temporal_workflow_closed"

    asyncio.run(run())


def test_temporal_activity_retry_exhaustion_redrives_without_closing_workflow() -> None:
    executable, _ = _prepare_temporal_cli()

    async def run() -> None:
        run_id = f"retry-exhaustion-{uuid.uuid4().hex}"
        coordinate = (run_id, 1)
        _ACTIVITY_ORDER.pop(run_id, None)
        _ACTIVITY_RETRYABLE_FAILURES[coordinate] = 2
        try:
            async with await WorkflowEnvironment.start_local(
                dev_server_existing_path=executable,
                dev_server_log_level="warn",
            ) as environment:
                task_queue = f"monoid-v023-{uuid.uuid4().hex}"
                policy = TemporalRunPolicy(
                    activity_task_queue=task_queue,
                    activity_max_attempts=2,
                )
                transport = TemporalSignalWithStartTransport(
                    client=environment.client,
                    event_loop=asyncio.get_running_loop(),
                    workflow_task_queue=task_queue,
                    run_policy=policy,
                )
                command = _admitted_command(run_id, 1, terminal=True)

                accepted = await transport.dispatch_async(command)
                assert accepted.status == "accepted"
                handle = environment.client.get_workflow_handle(
                    temporal_workflow_id(run_id),
                    result_type=dict,
                )
                async with Worker(
                    environment.client,
                    task_queue=task_queue,
                    workflows=[TemporalRunWorkflow],
                    activities=[_drive_test_activation],
                ):
                    terminal = TemporalRunStatus.from_json(await handle.result())
                    history = await handle.fetch_history()
                    await Replayer(workflows=[TemporalRunWorkflow]).replay_workflow(history)

                assert _ACTIVITY_ORDER[run_id] == [1, 1, 1]
                assert terminal.phase == "terminal"
                assert terminal.next_command_sequence == 2
                assert terminal.last_error_code == "temporal_activity_retry_exhausted"
                _assert_history_payload_privacy(history)
        finally:
            _ACTIVITY_RETRYABLE_FAILURES.pop(coordinate, None)
            _ACTIVITY_ORDER.pop(run_id, None)

    asyncio.run(run())


def test_temporal_transport_classifies_sdk_failures_without_private_text() -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()
        cases = (
            (
                asyncio.TimeoutError("raw-private-timeout"),
                "retry",
                "temporal_acceptance_unknown",
            ),
            (
                RPCError(
                    "raw-private-unavailable",
                    RPCStatusCode.UNAVAILABLE,
                    b"",
                ),
                "retry",
                "temporal_rpc_retryable",
            ),
            (
                RPCError(
                    "raw-private-invalid",
                    RPCStatusCode.INVALID_ARGUMENT,
                    b"",
                ),
                "rejected",
                "temporal_request_rejected",
            ),
            (
                WorkflowAlreadyStartedError(
                    "workflow-1",
                    "workflow-v1",
                    run_id="run-1",
                ),
                "rejected",
                "temporal_workflow_closed",
            ),
        )
        for error, expected_status, expected_code in cases:
            transport = TemporalSignalWithStartTransport(
                client=_RaisingTemporalClient(error),
                event_loop=loop,
                workflow_task_queue="workflow-v1",
                run_policy=TemporalRunPolicy(activity_task_queue="activity-v1"),
            )
            result = await transport.dispatch_async(_admitted_command("run-1", 1))
            assert result.status == expected_status
            assert result.error_code == expected_code
            assert "raw-private" not in repr(result)

    asyncio.run(run())


def test_temporal_continue_as_new_transfers_pending_command_on_actual_server() -> None:
    executable, _ = _prepare_temporal_cli()

    async def run() -> None:
        run_id = f"run-{uuid.uuid4().hex}"
        coordinate = (run_id, 1)
        _ACTIVITY_ORDER.pop(run_id, None)
        _ACTIVITY_STARTED[coordinate] = asyncio.Event()
        _ACTIVITY_RELEASE[coordinate] = asyncio.Event()
        try:
            async with await WorkflowEnvironment.start_local(
                dev_server_existing_path=executable,
                dev_server_log_level="warn",
            ) as environment:
                task_queue = f"monoid-v023-{uuid.uuid4().hex}"
                transport = TemporalSignalWithStartTransport(
                    client=environment.client,
                    event_loop=asyncio.get_running_loop(),
                    workflow_task_queue=task_queue,
                    run_policy=TemporalRunPolicy(
                        activity_task_queue=task_queue,
                        history_rollover_command_limit=1,
                    ),
                )
                workflow_id = temporal_workflow_id(run_id)
                async with Worker(
                    environment.client,
                    task_queue=task_queue,
                    workflows=[TemporalRunWorkflow],
                    activities=[_drive_test_activation],
                ):
                    await asyncio.to_thread(
                        transport.dispatch,
                        _admitted_command(run_id, 1),
                    )
                    handle = environment.client.get_workflow_handle(
                        workflow_id,
                        result_type=dict,
                    )
                    first_description = await handle.describe()
                    await asyncio.wait_for(_ACTIVITY_STARTED[coordinate].wait(), timeout=10)
                    await asyncio.to_thread(
                        transport.dispatch,
                        _admitted_command(run_id, 2),
                    )
                    _ACTIVITY_RELEASE[coordinate].set()
                    await _wait_for_activity_count(run_id, 2)

                    async def wait_for_rollover() -> TemporalRunStatus:
                        while True:
                            status = TemporalRunStatus.from_json(
                                await handle.query(TEMPORAL_STATUS_QUERY, result_type=dict)
                            )
                            if status.rollover_count >= 2 and status.next_command_sequence == 3:
                                return status
                            await asyncio.sleep(0.01)

                    rolled = await asyncio.wait_for(wait_for_rollover(), timeout=10)
                    current_description = await handle.describe()
                    assert rolled.pending_count == 0
                    assert current_description.run_id != first_description.run_id

                    await asyncio.to_thread(
                        transport.dispatch,
                        _admitted_command(run_id, 3, terminal=True),
                    )
                    terminal = TemporalRunStatus.from_json(await handle.result())
                    assert terminal.phase == "terminal"
                    assert terminal.rollover_count >= 2
                    assert _ACTIVITY_ORDER[run_id] == [1, 2, 3]

                    first_history = await environment.client.get_workflow_handle(
                        workflow_id,
                        run_id=first_description.run_id,
                    ).fetch_history()
                    await Replayer(workflows=[TemporalRunWorkflow]).replay_workflow(first_history)
        finally:
            _ACTIVITY_STARTED.pop(coordinate, None)
            _ACTIVITY_RELEASE.pop(coordinate, None)

    asyncio.run(run())


def test_checked_in_temporal_v1_history_replays() -> None:
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "temporal_replay_v1"
        / "run-workflow-v1.json"
    )
    assert fixture.is_file()
    history = WorkflowHistory.from_json("monoid-temporal-replay-v1", fixture.read_text("utf-8"))

    asyncio.run(Replayer(workflows=[TemporalRunWorkflow]).replay_workflow(history))
    _assert_history_payload_privacy(history)
