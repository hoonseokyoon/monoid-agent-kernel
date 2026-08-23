from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from support.backend_factory import ManagedBackendFactory
from support.runtime import runtime_config

from monoid_agent_kernel.core.authority import (
    ActivationWriteAuthority,
    WriteAuthorityRevoked,
)
from monoid_agent_kernel.core.lifecycle import SessionState
from monoid_agent_kernel.core.model_io import (
    CapturePolicy,
    ModelCallCapture,
    ModelIOSubscription,
)
from monoid_agent_kernel.core.result import AgentRunResult, Suspension
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.reference.backend.recovery import RecoveryService
from monoid_agent_kernel.reference.backend.service import BackendRunRequest


class RecordingObserver:
    def __init__(self) -> None:
        self.captures: list[ModelCallCapture] = []
        self.close_count = 0

    def on_model_call(self, capture: ModelCallCapture) -> None:
        self.captures.append(capture)

    def close(self) -> None:
        self.close_count += 1


def test_backend_materializes_and_owns_one_subscription_per_run(
    tmp_path: Path,
    backend_factory: ManagedBackendFactory,
) -> None:
    workspace = backend_factory.workspace()
    observers: list[RecordingObserver] = []

    def make_subscription() -> ModelIOSubscription:
        observer = RecordingObserver()
        observers.append(observer)
        return ModelIOSubscription(observer, CapturePolicy(mode="digest"))

    backend = backend_factory.create(
        workspace=workspace,
        turns=[ModelTurn(response_id="r1", final_text="done")],
        model_io_subscription_factories=(make_subscription,),
    )

    submissions = [
        backend.submit_run(
            BackendRunRequest(
                tenant_id="tenant-a",
                user_id="user-a",
                workspace_root=workspace,
                instruction=f"run {index}",
                runtime_config=runtime_config(),
            )
        )
        for index in range(2)
    ]

    for submission in submissions:
        assert backend.wait_for_run(submission.run_id, timeout_s=10) is SessionState.COMPLETED
    assert len(observers) == 2
    assert observers[0] is not observers[1]
    assert [len(observer.captures) for observer in observers] == [1, 1]
    assert {observer.captures[0].receipt.context.run_id for observer in observers} == {
        submission.run_id for submission in submissions
    }
    assert [observer.close_count for observer in observers] == [1, 1]


def test_backend_closes_partial_subscriptions_when_a_later_factory_fails(
    tmp_path: Path,
    backend_factory: ManagedBackendFactory,
) -> None:
    workspace = backend_factory.workspace()
    first = RecordingObserver()

    def fail() -> ModelIOSubscription:
        raise RuntimeError("observer unavailable")

    backend = backend_factory.create(
        workspace=workspace,
        model_io_subscription_factories=(
            lambda: ModelIOSubscription(first, CapturePolicy(mode="digest")),
            fail,
        ),
    )
    request = BackendRunRequest(
        tenant_id="tenant-a",
        user_id="user-a",
        workspace_root=workspace,
        instruction="never starts",
        runtime_config=runtime_config(),
    )

    # Build directly so the assertion targets factory ownership instead of the async admission
    # wrapper's terminal failure projection.
    prepared: Any = backend._run_preparation.prepare(request)
    try:
        try:
            backend._build_loop_build(
                prepared.run_id,
                request,
                prepared.workspace_root,
                prepared.llm_gateway_token,
                prepared.web_gateway_token,
            )
        except RuntimeError as exc:
            assert str(exc) == "observer unavailable"
        else:  # pragma: no cover - the failing factory must abort construction
            raise AssertionError("model-I/O factory failure was ignored")
    finally:
        backend._unregister_recovered_record(prepared.record)

    assert first.captures == []
    assert first.close_count == 1


def test_backend_rejects_invalid_subscription_and_closes_prior_observers(
    tmp_path: Path,
    backend_factory: ManagedBackendFactory,
) -> None:
    workspace = backend_factory.workspace()
    first = RecordingObserver()
    backend = backend_factory.create(
        workspace=workspace,
        model_io_subscription_factories=(
            lambda: ModelIOSubscription(first, CapturePolicy(mode="digest")),
            lambda: object(),
        ),
    )
    request = BackendRunRequest(
        tenant_id="tenant-a",
        user_id="user-a",
        workspace_root=workspace,
        instruction="never starts",
        runtime_config=runtime_config(),
    )
    prepared: Any = backend._run_preparation.prepare(request)
    try:
        try:
            backend._build_loop_build(
                prepared.run_id,
                request,
                prepared.workspace_root,
                prepared.llm_gateway_token,
                prepared.web_gateway_token,
            )
        except TypeError as exc:
            assert str(exc) == "model-I/O subscription factory must return ModelIOSubscription"
        else:  # pragma: no cover - the invalid factory result must abort construction
            raise AssertionError("invalid model-I/O subscription was accepted")
    finally:
        backend._unregister_recovered_record(prepared.record)

    assert first.close_count == 1


def test_backend_closes_subscriptions_when_downstream_loop_composition_fails(
    tmp_path: Path,
    backend_factory: ManagedBackendFactory,
) -> None:
    workspace = backend_factory.workspace()
    observer = RecordingObserver()

    def fail_event_sink() -> Any:
        raise RuntimeError("event sink unavailable")

    backend = backend_factory.create(
        workspace=workspace,
        model_io_subscription_factories=(
            lambda: ModelIOSubscription(observer, CapturePolicy(mode="digest")),
        ),
        extra_event_sink_factories=(fail_event_sink,),
    )
    request = BackendRunRequest(
        tenant_id="tenant-a",
        user_id="user-a",
        workspace_root=workspace,
        instruction="never starts",
        runtime_config=runtime_config(),
    )
    prepared: Any = backend._run_preparation.prepare(request)
    try:
        try:
            backend._build_loop_build(
                prepared.run_id,
                request,
                prepared.workspace_root,
                prepared.llm_gateway_token,
                prepared.web_gateway_token,
            )
        except RuntimeError as exc:
            assert str(exc) == "event sink unavailable"
        else:  # pragma: no cover - the downstream factory must abort construction
            raise AssertionError("event-sink factory failure was ignored")
    finally:
        backend._unregister_recovered_record(prepared.record)

    assert observer.close_count == 1


def test_cancelled_autonomous_execution_discards_run_owned_subscription(
    tmp_path: Path,
    backend_factory: ManagedBackendFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = backend_factory.workspace()
    observer = RecordingObserver()
    backend = backend_factory.create(
        workspace=workspace,
        model_io_subscription_factories=(
            lambda: ModelIOSubscription(observer, CapturePolicy(mode="digest")),
        ),
    )
    request = BackendRunRequest(
        tenant_id="tenant-a",
        user_id="user-a",
        workspace_root=workspace,
        instruction="cancel after composition",
        runtime_config=runtime_config(),
    )
    prepared: Any = backend._run_preparation.prepare(request)
    entered = asyncio.Event()

    async def block_drive(*_args: Any, **_kwargs: Any) -> Any:
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(backend._run_execution, "drive_session", block_drive)

    async def exercise() -> None:
        task = asyncio.create_task(backend._run_execution.run_prepared(prepared, request))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(exercise())
    finally:
        backend._unregister_recovered_record(prepared.record)

    assert observer.close_count == 1


def test_recovered_execution_failure_discards_activation_before_recording_failure(
    tmp_path: Path,
    backend_factory: ManagedBackendFactory,
) -> None:
    workspace = backend_factory.workspace()
    backend = backend_factory.create(workspace=workspace)
    observer = RecordingObserver()
    subscription = ModelIOSubscription(observer, CapturePolicy(mode="digest"))
    failures: list[Exception] = []
    release_count = 0

    class RecoveredLoop:
        def has_pending_tasks(self) -> bool:
            return False

        def discard_uncommitted(self) -> None:
            from monoid_agent_kernel.core.model_io import close_model_io_subscriptions

            close_model_io_subscriptions((subscription,))

    async def fail_drive(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("recovered execution failed")

    async def acquire_slot() -> None:
        return None

    def release_slot() -> None:
        nonlocal release_count
        release_count += 1

    context = replace(
        backend._recovery._context,
        record=lambda _run_id: SimpleNamespace(
            write_authority=ActivationWriteAuthority(),
        ),
        drive_open_session=fail_drive,
        record_run_failure=lambda _record, exc: failures.append(exc),
        acquire_run_slot=acquire_slot,
        release_run_slot=release_slot,
    )
    service = RecoveryService(context)
    request = BackendRunRequest(
        tenant_id="tenant-a",
        user_id="user-a",
        workspace_root=workspace,
        instruction="resume",
        runtime_config=runtime_config(),
    )

    asyncio.run(
        service.run_recovered(
            "run-recovered",
            request,
            RecoveredLoop(),  # type: ignore[arg-type]
            suspension=Suspension(reason="settled", status="completed"),
        )
    )

    assert [str(exc) for exc in failures] == ["recovered execution failed"]
    assert observer.close_count == 1
    assert release_count == 1


def test_recovered_activation_cancelled_while_waiting_for_slot_is_discarded(
    tmp_path: Path,
    backend_factory: ManagedBackendFactory,
) -> None:
    workspace = backend_factory.workspace()
    backend = backend_factory.create(workspace=workspace)
    observer = RecordingObserver()
    subscription = ModelIOSubscription(observer, CapturePolicy(mode="digest"))
    release_count = 0
    failures: list[Exception] = []

    class RecoveredLoop:
        def has_pending_tasks(self) -> bool:
            return False

        def discard_uncommitted(self) -> None:
            from monoid_agent_kernel.core.model_io import close_model_io_subscriptions

            close_model_io_subscriptions((subscription,))

    def release_slot() -> None:
        nonlocal release_count
        release_count += 1

    async def exercise() -> None:
        entered = asyncio.Event()

        async def block_acquire() -> None:
            entered.set()
            await asyncio.Event().wait()

        context = replace(
            backend._recovery._context,
            acquire_run_slot=block_acquire,
            release_run_slot=release_slot,
            record_run_failure=lambda _record, exc: failures.append(exc),
        )
        service = RecoveryService(context)
        request = BackendRunRequest(
            tenant_id="tenant-a",
            user_id="user-a",
            workspace_root=workspace,
            instruction="resume",
            runtime_config=runtime_config(),
        )
        task = asyncio.create_task(
            service.run_recovered(
                "run-waiting-slot",
                request,
                RecoveredLoop(),  # type: ignore[arg-type]
                suspension=Suspension(reason="settled", status="completed"),
            )
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    assert observer.close_count == 1
    assert release_count == 0
    assert failures == []


def test_recovered_record_registration_is_an_atomic_first_writer_claim(
    tmp_path: Path,
    backend_factory: ManagedBackendFactory,
) -> None:
    workspace = backend_factory.workspace()
    backend = backend_factory.create(workspace=workspace)
    request = BackendRunRequest(
        tenant_id="tenant-a",
        user_id="user-a",
        workspace_root=workspace,
        instruction="prepare contenders",
        runtime_config=runtime_config(),
    )
    prepared: Any = backend._run_preparation.prepare(request)
    seed = prepared.record
    backend._unregister_recovered_record(seed)
    first = replace(seed, write_authority=ActivationWriteAuthority())
    contender = replace(seed, write_authority=ActivationWriteAuthority())

    try:
        assert backend._register_recovered_record(first) is True
        assert backend._register_recovered_record(contender) is False
        assert backend._record(first.run_id) is first
    finally:
        backend._unregister_recovered_record(first)


def test_stale_record_unregistration_preserves_a_replacement_owner(
    tmp_path: Path,
    backend_factory: ManagedBackendFactory,
) -> None:
    workspace = backend_factory.workspace()
    backend = backend_factory.create(workspace=workspace)
    request = BackendRunRequest(
        tenant_id="tenant-a",
        user_id="user-a",
        workspace_root=workspace,
        instruction="replace owner",
        runtime_config=runtime_config(),
    )
    prepared: Any = backend._run_preparation.prepare(request)
    stale = prepared.record
    replacement = replace(stale, write_authority=ActivationWriteAuthority())
    backend._unregister_recovered_record(stale)
    assert backend._register_recovered_record(replacement) is True

    try:
        backend._unregister_recovered_record(stale)
        assert backend._record(stale.run_id) is replacement
    finally:
        backend._unregister_recovered_record(replacement)


def test_activation_settlement_cannot_mutate_the_replacement_record(
    tmp_path: Path,
    backend_factory: ManagedBackendFactory,
) -> None:
    workspace = backend_factory.workspace()
    backend = backend_factory.create(workspace=workspace)
    request = BackendRunRequest(
        tenant_id="tenant-a",
        user_id="user-a",
        workspace_root=workspace,
        instruction="replace before stale settlement",
        runtime_config=runtime_config(),
    )
    prepared: Any = backend._run_preparation.prepare(request)
    stale = prepared.record
    replacement = replace(stale, write_authority=ActivationWriteAuthority())
    backend._unregister_recovered_record(stale)
    assert backend._register_recovered_record(replacement) is True

    try:
        with pytest.raises(WriteAuthorityRevoked):
            backend._record_activation_result(
                stale,
                AgentRunResult(
                    run_id=stale.run_id,
                    status="completed",
                    final_text="stale",
                    run_dir=stale.run_dir,
                    diff_path=stale.run_dir / "diff.patch",
                    proposal_path=stale.run_dir / "proposal.json",
                ),
            )
        assert backend._record(stale.run_id) is replacement
        assert replacement.terminal is False
        assert replacement.result is None
    finally:
        backend._unregister_recovered_record(replacement)


def test_watchdog_cas_loser_keeps_lease_for_locally_tracked_winner(
    tmp_path: Path,
    backend_factory: ManagedBackendFactory,
) -> None:
    workspace = backend_factory.workspace()
    backend = backend_factory.create(workspace=workspace)
    run_id = "run-local-winner"
    (tmp_path / run_id).mkdir()
    tracked_run_ids: set[str] = set()

    class ClaimedLeaseStore:
        def __init__(self) -> None:
            self.released: list[str] = []

        def candidate_run_ids(self) -> list[str]:
            return [run_id]

        def is_stale(self, candidate_run_id: str) -> bool:
            assert candidate_run_id == run_id
            return True

        def try_claim(self, candidate_run_id: str, worker_id: str, ttl_s: float) -> bool:
            assert candidate_run_id == run_id
            assert worker_id
            assert ttl_s > 0
            return True

        def release(self, candidate_run_id: str) -> None:
            self.released.append(candidate_run_id)

    class LosingWatchdogRecovery(RecoveryService):
        def attempt_resume(self, run_dir: Path, candidate_run_id: str) -> bool:
            assert run_dir == tmp_path / run_id
            tracked_run_ids.add(candidate_run_id)
            return False

    lease_store = ClaimedLeaseStore()
    context = replace(
        backend._recovery._context,
        lease_store_provider=lambda: lease_store,
        run_root_provider=lambda: tmp_path,
        is_record_tracked=lambda candidate_run_id: candidate_run_id in tracked_run_ids,
    )

    assert LosingWatchdogRecovery(context).reclaim_stale_runs() == []
    assert lease_store.released == []
