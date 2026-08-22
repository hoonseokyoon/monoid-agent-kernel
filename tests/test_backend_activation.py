from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from monoid_agent_kernel.core.authority import (
    ActivationWriteAuthority,
    WriteAuthorityRevoked,
)
from monoid_agent_kernel.core.outcome import InterruptionCause
from monoid_agent_kernel.core.result import Suspension
from monoid_agent_kernel.errors import NativeAgentError
from monoid_agent_kernel.reference.backend.activation import is_activation_lease_loss
from monoid_agent_kernel.reference.backend.run_execution import (
    RunExecutionContext,
    RunExecutionService,
)
from monoid_agent_kernel.reference.backend.recovery import RecoveryContext, RecoveryService


class _DiscardableLoop:
    def __init__(self, tmp_path: Path) -> None:
        self.discard_calls = 0
        self.close_calls = 0
        self.spec = SimpleNamespace(run_root=tmp_path, run_id="run_stale")

    async def aopen(self) -> None:
        return None

    async def aclose(self) -> None:
        self.close_calls += 1
        raise AssertionError("a stale activation must not close")

    def discard_uncommitted(self) -> None:
        self.discard_calls += 1

    def has_pending_tasks(self) -> bool:
        return False


def _execution_service(
    loop: _DiscardableLoop,
    *,
    failures: list[Exception],
    results: list[Any],
    released: list[bool],
    unregistered: list[Any],
) -> RunExecutionService:
    async def acquire() -> None:
        return None

    return RunExecutionService(
        RunExecutionContext(
            build_loop=lambda *args: SimpleNamespace(loop=loop),
            attach_loop=lambda *args: None,
            unregister_record=unregistered.append,
            record=lambda run_id: SimpleNamespace(
                run_id=run_id,
                write_authority=ActivationWriteAuthority(),
            ),
            drive_open_session=lambda *args, **kwargs: None,  # type: ignore[arg-type]
            record_run_result=lambda run_id, result: results.append((run_id, result)),
            record_run_failure=lambda run_id, exc: failures.append(exc),
            acquire_run_slot=acquire,
            release_run_slot=lambda: released.append(True),
            submission_json=lambda prepared: {"run_id": prepared.run_id},
        )
    )


def _prepared(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        run_id="run_stale",
        record=SimpleNamespace(
            run_id="run_stale",
            write_authority=ActivationWriteAuthority(),
        ),
        workspace_root=tmp_path,
        llm_gateway_token="llm",
        web_gateway_token="web",
    )


def test_core_lease_fence_errors_share_the_host_activation_disposition() -> None:
    assert is_activation_lease_loss(
        WriteAuthorityRevoked()
    )
    assert not is_activation_lease_loss(
        NativeAgentError("ordinary failure", error_code="internal_error")
    )


def test_autonomous_execution_discards_lease_loss_without_recording_failure(
    tmp_path: Path,
) -> None:
    loop = _DiscardableLoop(tmp_path)
    failures: list[Exception] = []
    results: list[Any] = []
    released: list[bool] = []
    unregistered: list[Any] = []
    service = _execution_service(
        loop,
        failures=failures,
        results=results,
        released=released,
        unregistered=unregistered,
    )

    async def stale_drive(*args: Any, **kwargs: Any) -> None:
        raise WriteAuthorityRevoked()

    service.drive_session = stale_drive  # type: ignore[method-assign]

    prepared = _prepared(tmp_path)
    asyncio.run(service.run_prepared(prepared, SimpleNamespace()))

    assert loop.discard_calls == 1
    assert loop.close_calls == 0
    assert failures == []
    assert results == []
    assert released == [True]
    assert unregistered == [prepared.record]


def test_stream_execution_discards_lease_loss_without_a_terminal_frame(
    tmp_path: Path,
) -> None:
    loop = _DiscardableLoop(tmp_path)
    failures: list[Exception] = []
    results: list[Any] = []
    released: list[bool] = []
    unregistered: list[Any] = []
    service = _execution_service(
        loop,
        failures=failures,
        results=results,
        released=released,
        unregistered=unregistered,
    )

    class _LeaseLostStream:
        suspension = Suspension(
            reason="interrupted",
            status="completed",
            error_code="lease_lost",
            interruption_cause=InterruptionCause.LEASE_LOST,
        )

        async def __aenter__(self) -> _LeaseLostStream:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        def __aiter__(self) -> _LeaseLostStream:
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

    loop.astream = lambda user_input: _LeaseLostStream()  # type: ignore[attr-defined]

    prepared = _prepared(tmp_path)
    prepared.record.write_authority.revoke()

    async def collect() -> list[dict[str, Any]]:
        return [
            frame
            async for frame in service.stream_prepared(
                prepared,
                SimpleNamespace(input_parts=(), instruction="go"),
            )
        ]

    frames = asyncio.run(collect())

    assert frames == [{"kind": "meta", "run_id": "run_stale"}]
    assert loop.discard_calls == 1
    assert loop.close_calls == 0
    assert failures == []
    assert results == []
    assert released == [True]
    assert unregistered == [prepared.record]


def test_recovered_execution_discards_lease_loss_without_recording_failure(
    tmp_path: Path,
) -> None:
    loop = _DiscardableLoop(tmp_path)
    failures: list[Exception] = []
    results: list[Any] = []
    released: list[bool] = []
    unregistered: list[Any] = []
    recovered_record = SimpleNamespace(
        run_id="run_stale",
        write_authority=ActivationWriteAuthority(),
    )

    async def acquire() -> None:
        return None

    async def stale_session(*args: Any, **kwargs: Any) -> None:
        raise WriteAuthorityRevoked()

    service = RecoveryService(
        RecoveryContext(
            run_root_provider=lambda: tmp_path,
            checkpoint_store_provider=lambda: None,
            lease_store_provider=lambda: None,
            max_recover_attempts_provider=lambda: 3,
            worker_id_provider=lambda: "worker-1",
            lease_ttl_s_provider=lambda: 30.0,
            is_record_tracked=lambda run_id: False,
            record=lambda run_id: recovered_record,
            make_request=lambda *args: SimpleNamespace(),
            make_record=lambda *args: SimpleNamespace(),
            issue_llm_gateway_token=lambda *args: "llm",
            issue_web_gateway_token=lambda *args: "web",
            build_loop=lambda *args: SimpleNamespace(loop=loop),
            register_record=lambda record: True,
            unregister_record=unregistered.append,
            attach_loop=lambda *args: None,
            call_soon=lambda *args: None,
            spawn=lambda awaitable: None,
            drive_open_session=stale_session,  # type: ignore[arg-type]
            record_run_result=lambda run_id, result: results.append((run_id, result)),
            record_run_failure=lambda run_id, exc: failures.append(exc),
            meter_abandoned_run=lambda run_id, tenant_id: None,
            acquire_run_slot=acquire,
            release_run_slot=lambda: released.append(True),
        )
    )

    asyncio.run(
        service.run_recovered(
            "run_stale",
            SimpleNamespace(),
            loop,  # type: ignore[arg-type]
            Suspension(reason="settled", status="completed"),
        )
    )

    assert loop.discard_calls == 1
    assert loop.close_calls == 0
    assert failures == []
    assert results == []
    assert released == [True]
    assert unregistered == [recovered_record]
