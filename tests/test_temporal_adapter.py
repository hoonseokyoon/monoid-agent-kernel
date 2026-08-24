from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import pytest

from monoid_agent_kernel.adapters.temporal import (
    ACCEPTED_TEMPORAL_RUN_POLICY_SCHEMA_VERSIONS,
    DEFAULT_TEMPORAL_WORKFLOW_ID_PREFIX,
    MAX_ACTIVITY_ATTEMPTS,
    MAX_ACTIVITY_TIMEOUT_S,
    MAX_HISTORY_ROLLOVER_COMMANDS,
    MAX_TEMPORAL_RPC_TIMEOUT_S,
    TemporalActivationResult,
    TemporalRunPolicy,
    TemporalRunState,
    TemporalRunStatus,
    TemporalSignalWithStartTransport,
    temporal_dispatch_ref,
    temporal_workflow_id,
)
from monoid_agent_kernel.core.safe_evidence import is_safe_opaque_address, is_safe_opaque_id
from monoid_agent_kernel.hosting import (
    AdmissionRequest,
    AdmittedCommand,
    CommandTransport,
)


pytestmark = pytest.mark.unit


def _command(
    sequence: int = 1,
    *,
    run_id: str = "run-temporal-1",
    command_id: str | None = None,
) -> AdmittedCommand:
    return AdmittedCommand.from_request(
        AdmissionRequest(
            run_id=run_id,
            command_id=command_id or f"command-{sequence}",
            kind="input",
            request_digest=f"{sequence:064x}",
            payload_ref=f"object:private/{sequence}",
        ),
        sequence,
    )


def _policy(**changes: object) -> TemporalRunPolicy:
    values: dict[str, object] = {"activity_task_queue": "activity-v1"}
    values.update(changes)
    return TemporalRunPolicy(**values)  # type: ignore[arg-type]


class _FakeClient:
    async def start_workflow(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class _SecretReprClient(_FakeClient):
    def __repr__(self) -> str:
        return "TemporalClient(api_key=raw-private-credential)"


def test_temporal_records_round_trip_closed_json_schemas() -> None:
    first = _command(1)
    second = _command(2)
    policy = _policy(history_rollover_command_limit=2)
    state = TemporalRunState(
        run_id=first.run_id,
        policy=policy,
        next_command_sequence=2,
        pending_commands=(second,),
        latest_receipt_ref="checkpoint:run-temporal-1/2",
        rollover_count=1,
        duplicate_signal_count=3,
        last_error_code="temporal_invalid_command",
    )
    result = TemporalActivationResult.from_command(
        second,
        receipt_ref="checkpoint:run-temporal-1/3",
        terminal=False,
    )
    status = TemporalRunStatus(
        run_id=first.run_id,
        phase="waiting",
        next_command_sequence=2,
        in_flight_sequence=0,
        pending_count=1,
        pending_head_sequence=2,
        latest_receipt_ref="checkpoint:run-temporal-1/2",
        rollover_count=1,
        duplicate_signal_count=3,
        last_error_code="temporal_invalid_command",
    )

    assert TemporalRunPolicy.from_json(policy.to_json()) == policy
    assert TemporalRunState.from_json(state.to_json()) == state
    assert TemporalActivationResult.from_json(result.to_json()) == result
    assert TemporalRunStatus.from_json(status.to_json()) == status
    assert result.matches(second)
    assert not result.matches(first)
    json.dumps(
        {
            "state": state.to_json(),
            "result": result.to_json(),
            "status": status.to_json(),
        },
        allow_nan=False,
    )


def test_temporal_record_readers_accept_legacy_namespace_and_write_current() -> None:
    payload = _policy().to_json()
    payload["schema_version"] = ACCEPTED_TEMPORAL_RUN_POLICY_SCHEMA_VERSIONS[1]

    restored = TemporalRunPolicy.from_json(payload)

    assert restored.to_json()["schema_version"] == ACCEPTED_TEMPORAL_RUN_POLICY_SCHEMA_VERSIONS[0]


@pytest.mark.parametrize(
    "changes",
    (
        {"activity_task_queue": "bad queue"},
        {"activity_task_queue": "x" * 256},
        {"activity_start_to_close_timeout_s": 0},
        {"activity_start_to_close_timeout_s": MAX_ACTIVITY_TIMEOUT_S + 1},
        {"activity_heartbeat_timeout_s": 3_601},
        {"activity_max_attempts": 0},
        {"activity_max_attempts": MAX_ACTIVITY_ATTEMPTS + 1},
        {"history_rollover_command_limit": -1},
        {"history_rollover_command_limit": MAX_HISTORY_ROLLOVER_COMMANDS + 1},
        {"activity_max_attempts": True},
    ),
)
def test_temporal_policy_rejects_nonportable_controls(changes: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _policy(**changes)


def test_temporal_state_rejects_sequence_and_identity_inconsistency() -> None:
    with pytest.raises(ValueError, match="ascending"):
        TemporalRunState(
            run_id="run-temporal-1",
            policy=_policy(),
            pending_commands=(_command(2), _command(1)),
        )
    with pytest.raises(ValueError, match="frontier"):
        TemporalRunState(
            run_id="run-temporal-1",
            policy=_policy(),
            next_command_sequence=2,
            pending_commands=(_command(1),),
            latest_receipt_ref="checkpoint:run-temporal-1/2",
        )
    with pytest.raises(ValueError, match="another run|outside the run"):
        TemporalRunState(
            run_id="run-temporal-1",
            policy=_policy(),
            pending_commands=(_command(run_id="run-temporal-2"),),
        )
    with pytest.raises(ValueError, match="latest receipt"):
        TemporalRunState(
            run_id="run-temporal-1",
            policy=_policy(),
            next_command_sequence=2,
        )


def test_temporal_records_reject_unknown_private_fields_without_echoing_them() -> None:
    private_value = "raw-private-model-response"
    payload = TemporalRunState(
        run_id="run-temporal-1",
        policy=_policy(),
    ).to_json()
    payload["raw_model_response"] = private_value

    with pytest.raises(ValueError) as caught:
        TemporalRunState.from_json(payload)

    assert private_value not in str(caught.value)
    assert private_value not in repr(caught.value)


@pytest.mark.parametrize(
    ("factory", "field_name"),
    (
        (lambda: _policy().to_json(), "history_rollover_command_limit"),
        (
            lambda: TemporalRunState(
                run_id="run-temporal-1",
                policy=_policy(),
            ).to_json(),
            "duplicate_signal_count",
        ),
        (
            lambda: TemporalActivationResult.from_command(
                _command(),
                receipt_ref="checkpoint:run-temporal-1/2",
                terminal=False,
            ).to_json(),
            "terminal",
        ),
        (
            lambda: TemporalRunStatus(
                run_id="run-temporal-1",
                phase="waiting",
                next_command_sequence=1,
                in_flight_sequence=0,
                pending_count=0,
                pending_head_sequence=0,
                latest_receipt_ref="",
                rollover_count=0,
                duplicate_signal_count=0,
            ).to_json(),
            "last_error_code",
        ),
    ),
)
def test_temporal_record_readers_require_every_v1_field(
    factory: Callable[[], dict[str, object]],
    field_name: str,
) -> None:
    payload = factory()
    del payload[field_name]
    readers = {
        "history_rollover_command_limit": TemporalRunPolicy.from_json,
        "duplicate_signal_count": TemporalRunState.from_json,
        "terminal": TemporalActivationResult.from_json,
        "last_error_code": TemporalRunStatus.from_json,
    }

    with pytest.raises(ValueError, match="missing required"):
        readers[field_name](payload)


def test_temporal_workflow_identity_is_deterministic_bounded_and_content_free() -> None:
    run_id = "customer-neutral-run-1"
    first = temporal_workflow_id(run_id)
    second = temporal_workflow_id(run_id)
    other = temporal_workflow_id("customer-neutral-run-2")
    dispatch_ref = temporal_dispatch_ref(first)

    assert first == second
    assert first != other
    assert first.startswith(f"{DEFAULT_TEMPORAL_WORKFLOW_ID_PREFIX}-")
    assert run_id not in first
    assert is_safe_opaque_id(first)
    assert is_safe_opaque_address(dispatch_ref)
    assert len(dispatch_ref) <= 256


@pytest.mark.parametrize(
    ("prefix", "timeout"),
    (
        ("bad prefix", 30.0),
        ("x" * 183, 30.0),
        ("x" * 256, 30.0),
        ("valid", 0.0),
        ("valid", MAX_TEMPORAL_RPC_TIMEOUT_S + 1),
        ("valid", float("nan")),
        ("valid", True),
    ),
)
def test_temporal_transport_rejects_invalid_host_controls(
    prefix: str,
    timeout: object,
) -> None:
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises((TypeError, ValueError)):
            TemporalSignalWithStartTransport(
                client=_FakeClient(),
                event_loop=loop,
                workflow_task_queue="workflow-v1",
                run_policy=_policy(),
                workflow_id_prefix=prefix,
                rpc_timeout_s=timeout,  # type: ignore[arg-type]
            )
    finally:
        loop.close()


def test_temporal_transport_rejects_oversized_workflow_task_queue() -> None:
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(ValueError, match="task queue"):
            TemporalSignalWithStartTransport(
                client=_FakeClient(),
                event_loop=loop,
                workflow_task_queue="x" * 256,
                run_policy=_policy(),
            )
    finally:
        loop.close()


def test_temporal_sync_transport_is_structural_and_retries_when_owner_loop_is_down() -> None:
    loop = asyncio.new_event_loop()
    try:
        transport = TemporalSignalWithStartTransport(
            client=_FakeClient(),
            event_loop=loop,
            workflow_task_queue="workflow-v1",
            run_policy=_policy(),
        )

        assert isinstance(transport, CommandTransport)
        result = transport.dispatch(_command())
        assert result.status == "retry"
        assert result.error_code == "temporal_client_loop_unavailable"
    finally:
        loop.close()


def test_temporal_transport_repr_excludes_client_and_loop_state() -> None:
    loop = asyncio.new_event_loop()
    try:
        transport = TemporalSignalWithStartTransport(
            client=_SecretReprClient(),
            event_loop=loop,
            workflow_task_queue="workflow-v1",
            run_policy=_policy(),
        )

        rendered = repr(transport)
        assert "raw-private-credential" not in rendered
        assert repr(loop) not in rendered
    finally:
        loop.close()


def test_temporal_sync_transport_rejects_owner_loop_deadlock() -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()
        transport = TemporalSignalWithStartTransport(
            client=_FakeClient(),
            event_loop=loop,
            workflow_task_queue="workflow-v1",
            run_policy=_policy(),
        )
        with pytest.raises(RuntimeError, match="outside"):
            transport.dispatch(_command())

    asyncio.run(run())
