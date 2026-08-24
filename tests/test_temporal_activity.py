from __future__ import annotations

import asyncio
import hashlib
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest


pytest.importorskip("temporalio")

from temporalio.exceptions import ApplicationError  # noqa: E402
from temporalio.testing import ActivityEnvironment  # noqa: E402

import monoid_agent_kernel.adapters.temporal.activity as activity_module  # noqa: E402
import monoid_agent_kernel.adapters.temporal.worker as worker_module  # noqa: E402
from monoid_agent_kernel.adapters.temporal.activity import (  # noqa: E402
    MAX_TEMPORAL_ACTIVITY_LOCAL_DURATION_S,
    TemporalActivationActivity,
    TemporalActivityPolicy,
)
from monoid_agent_kernel.adapters.temporal.records import TemporalActivationResult  # noqa: E402
from monoid_agent_kernel.core.authority import WriteAuthorityRevoked  # noqa: E402
from monoid_agent_kernel.core.interruption import InterruptionCause  # noqa: E402
from monoid_agent_kernel.hosting import (  # noqa: E402
    ActivationBindingConflict,
    ActivationBindingWriterFenced,
    ActivationCommand,
    ActivationLoopConfigurationError,
    AdmissionRequest,
    AdmittedCommand,
    ReleaseResult,
    RenewResult,
    WriterAuthority,
    WriterLease,
    WriterLeaseUnavailable,
    WriterToken,
)


pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 24, tzinfo=UTC)


def _command() -> AdmittedCommand:
    return AdmittedCommand.from_request(
        AdmissionRequest(
            run_id="temporal-activity-run",
            command_id="command-1",
            kind="control",
            request_digest="1" * 64,
            payload_ref="object:private/command-1",
        ),
        1,
    )


@dataclass
class _AuthorityStore:
    renew_fenced_after: int = 0
    release_status: str = "released"
    lease_interval_s: float = 30.0

    def __post_init__(self) -> None:
        self.claimed_owner = ""
        self.renew_count = 0
        self.renew_threads: list[int] = []
        self.release_count = 0
        self.renewed = threading.Event()
        self.released = threading.Event()

    def _lease(self, token: WriterToken) -> WriterLease:
        return WriterLease(
            writer_token=token,
            observed_at=_NOW,
            leased_until=_NOW + timedelta(seconds=self.lease_interval_s),
        )

    def _authority(self, token: WriterToken, *, revoked: bool) -> WriterAuthority:
        return WriterAuthority(
            writer_token=token,
            observed_at=_NOW,
            leased_until=_NOW + timedelta(seconds=30),
            revoked=revoked,
        )

    def claim(self, run_id: str, owner_id: str, ttl: timedelta) -> WriterLease:
        assert ttl > timedelta(0)
        self.claimed_owner = owner_id
        return self._lease(WriterToken(run_id=run_id, owner_id=owner_id, generation=1))

    def renew(self, writer_token: WriterToken, ttl: timedelta) -> RenewResult:
        assert ttl > timedelta(0)
        self.renew_count += 1
        self.renew_threads.append(threading.get_ident())
        self.renewed.set()
        if self.renew_fenced_after and self.renew_count >= self.renew_fenced_after:
            return RenewResult(
                status="fenced",
                authority=self._authority(writer_token, revoked=False),
            )
        return RenewResult(status="renewed", lease=self._lease(writer_token))

    def release(self, writer_token: WriterToken) -> ReleaseResult:
        self.release_count += 1
        self.released.set()
        if self.release_status == "fenced":
            return ReleaseResult(
                status="fenced",
                authority=self._authority(writer_token, revoked=False),
            )
        return ReleaseResult(
            status=self.release_status,  # type: ignore[arg-type]
            authority=self._authority(writer_token, revoked=True),
        )

    def read(self, run_id: str) -> None:
        del run_id
        return None


class _AdmissionStore:
    def __init__(self) -> None:
        self.bound_tokens: list[WriterToken] = []

    def bind_activation(
        self,
        command: AdmittedCommand,
        *,
        writer_token: WriterToken,
    ) -> ActivationCommand:
        self.bound_tokens.append(writer_token)
        return ActivationCommand(
            run_id=command.run_id,
            command_id=command.command_id,
            command_sequence=command.command_sequence,
            kind=command.kind,
            source_checkpoint_seq=1,
            source_checkpoint_sha256="2" * 64,
            request_digest=command.request_digest,
            payload_ref=command.payload_ref,
        )


class _DriverDouble:
    behavior: Any = None
    constructed: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.constructed.append(kwargs)

    def drive(self, activation: object) -> object:
        del activation
        behavior = type(self).behavior
        if behavior is not None:
            return behavior(self.kwargs)
        return SimpleNamespace(
            checkpoint_ref="checkpoint:temporal-activity-run/2",
            terminal=False,
        )


@pytest.fixture(autouse=True)
def _install_driver_double(monkeypatch: pytest.MonkeyPatch) -> None:
    _DriverDouble.behavior = None
    _DriverDouble.constructed.clear()
    monkeypatch.setattr(activity_module, "ActivationDriver", _DriverDouble)


def _activity(
    store: _AuthorityStore,
    *,
    admission_store: _AdmissionStore | None = None,
    policy: TemporalActivityPolicy | None = None,
) -> TemporalActivationActivity:
    return TemporalActivationActivity(
        authority_store=store,
        admission_store=_AdmissionStore() if admission_store is None else admission_store,
        run_sink=object(),  # type: ignore[arg-type]
        loop_factory=lambda command, runtime: (command, runtime),  # type: ignore[arg-type,return-value]
        policy=policy or TemporalActivityPolicy(
            writer_lease_ttl_s=2,
            writer_lease_renew_interval_s=0.02,
            heartbeat_interval_s=0.01,
            authority_call_timeout_s=1,
            driver_call_timeout_s=1,
            supervisor_join_timeout_s=0.1,
            local_task_wait_s=1,
        ),
    )


def test_temporal_activity_policy_has_fail_safe_bounds() -> None:
    assert TemporalActivityPolicy().writer_lease_ttl == timedelta(seconds=30)
    assert TemporalActivityPolicy().driver_call_timeout_s == 3_300
    for changes in (
        {"writer_lease_ttl_s": 0.5},
        {"writer_lease_ttl_s": 10, "writer_lease_renew_interval_s": 6},
        {"heartbeat_interval_s": 0},
        {"authority_call_timeout_s": 0},
        {"driver_call_timeout_s": 0},
        {"supervisor_join_timeout_s": float("nan")},
        {"local_task_wait_s": MAX_TEMPORAL_ACTIVITY_LOCAL_DURATION_S + 1},
        {"writer_lease_ttl_s": True},
        {"worker_shutdown_cause": InterruptionCause.USER_CANCEL},
    ):
        with pytest.raises(ValueError):
            TemporalActivityPolicy(**changes)  # type: ignore[arg-type]


def test_activity_phase_timeout_reserves_cleanup_inside_temporal_deadline() -> None:
    timeout = activity_module._activity_phase_timeout(
        SimpleNamespace(start_to_close_timeout=timedelta(seconds=10)),
        attempt_started_monotonic=time.monotonic() - 3,
        configured_timeout_s=20,
        cleanup_timeout_s=2,
        phase_name="driver",
    )

    assert 4.9 < timeout < 5.1
    assert (
        activity_module._activity_phase_timeout(
            SimpleNamespace(start_to_close_timeout=None),
            attempt_started_monotonic=time.monotonic(),
            configured_timeout_s=20,
            cleanup_timeout_s=2,
            phase_name="bootstrap",
        )
        == 20
    )
    with pytest.raises(ValueError, match="must exceed the cleanup reserve"):
        activity_module._activity_phase_timeout(
            SimpleNamespace(start_to_close_timeout=timedelta(seconds=1)),
            attempt_started_monotonic=time.monotonic(),
            configured_timeout_s=20,
            cleanup_timeout_s=1,
            phase_name="bootstrap",
        )


def test_impossible_activity_budget_is_a_non_retryable_configuration_conflict() -> None:
    policy = TemporalActivityPolicy(
        writer_lease_ttl_s=2,
        writer_lease_renew_interval_s=0.02,
        heartbeat_interval_s=0.01,
        authority_call_timeout_s=1,
        driver_call_timeout_s=1,
        supervisor_join_timeout_s=1,
        local_task_wait_s=1,
    )
    store = _AuthorityStore()

    with pytest.raises(ApplicationError) as raised:
        ActivityEnvironment().run(
            _activity(store, policy=policy).run,
            _command().to_json(),
        )

    assert raised.value.type == "monoid.activation_config_conflict"
    assert raised.value.non_retryable is True
    assert store.claimed_owner == ""
    assert store.release_count == 0
    assert _DriverDouble.constructed == []


def test_temporal_activity_repr_excludes_host_dependencies() -> None:
    private_text = "raw-private-dsn-or-credential"

    class Secret:
        def __repr__(self) -> str:
            return private_text

        def __call__(self, *args: object) -> object:
            return args

    configured = TemporalActivationActivity(
        authority_store=Secret(),  # type: ignore[arg-type]
        admission_store=Secret(),  # type: ignore[arg-type]
        run_sink=Secret(),  # type: ignore[arg-type]
        loop_factory=Secret(),  # type: ignore[arg-type]
        input_resolver=Secret(),  # type: ignore[arg-type]
    )

    assert private_text not in repr(configured)


def test_threaded_activity_renews_heartbeats_and_releases_content_free_owner() -> None:
    store = _AuthorityStore()
    environment = ActivityEnvironment()
    environment.info = environment.info.__class__(
        **{
            **environment.info.__dict__,
            "task_token": b"raw-temporal-task-token",
        }
    )
    heartbeat_threads: list[int] = []
    heartbeat_details: list[tuple[object, ...]] = []

    def heartbeat(*details: object) -> None:
        heartbeat_threads.append(threading.get_ident())
        heartbeat_details.append(details)

    environment.on_heartbeat = heartbeat

    def wait_for_supervisor(kwargs: dict[str, Any]) -> object:
        assert store.renewed.wait(1)
        deadline = time.monotonic() + 1
        while store.renew_count < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert kwargs["cancellation_token"].requested is False
        return SimpleNamespace(
            checkpoint_ref="checkpoint:temporal-activity-run/2",
            terminal=False,
        )

    _DriverDouble.behavior = wait_for_supervisor
    raw = environment.run(_activity(store).run, _command().to_json())
    result = TemporalActivationResult.from_json(raw)

    assert result.matches(_command())
    assert store.renew_count >= 2
    assert store.release_count == 1
    assert len(set(store.renew_threads)) >= 2
    assert len(heartbeat_threads) >= 2
    assert heartbeat_details and all(details == () for details in heartbeat_details)
    expected_owner = "temporal-activity-" + hashlib.sha256(
        b"raw-temporal-task-token"
    ).hexdigest()
    assert store.claimed_owner == expected_owner
    assert b"raw-temporal-task-token".decode() not in store.claimed_owner


def test_bootstrap_preserves_the_activity_context_for_every_store_call() -> None:
    tenant_context: ContextVar[str] = ContextVar("test_temporal_bootstrap_tenant")
    observed: list[tuple[str, str | None]] = []

    class ContextAuthorityStore(_AuthorityStore):
        def claim(self, run_id: str, owner_id: str, ttl: timedelta) -> WriterLease:
            observed.append(("claim", tenant_context.get(None)))
            return super().claim(run_id, owner_id, ttl)

        def renew(self, writer_token: WriterToken, ttl: timedelta) -> RenewResult:
            observed.append(("renew", tenant_context.get(None)))
            return super().renew(writer_token, ttl)

        def release(self, writer_token: WriterToken) -> ReleaseResult:
            observed.append(("release", tenant_context.get(None)))
            return super().release(writer_token)

    class ContextAdmissionStore(_AdmissionStore):
        def bind_activation(
            self,
            command: AdmittedCommand,
            *,
            writer_token: WriterToken,
        ) -> ActivationCommand:
            observed.append(("bind", tenant_context.get(None)))
            return super().bind_activation(command, writer_token=writer_token)

    token = tenant_context.set("tenant-context-a")
    try:
        result = ActivityEnvironment().run(
            _activity(
                ContextAuthorityStore(),
                admission_store=ContextAdmissionStore(),
            ).run,
            _command().to_json(),
        )
    finally:
        tenant_context.reset(token)

    assert result["run_id"] == "temporal-activity-run"
    assert observed[:3] == [
        ("claim", "tenant-context-a"),
        ("renew", "tenant-context-a"),
        ("bind", "tenant-context-a"),
    ]
    assert ("release", "tenant-context-a") in observed


def test_control_heartbeat_is_capped_to_the_server_activity_timeout() -> None:
    heartbeat_times: list[float] = []
    environment = ActivityEnvironment()
    environment.info = environment.info.__class__(
        **{
            **environment.info.__dict__,
            "heartbeat_timeout": timedelta(seconds=0.1),
            "start_to_close_timeout": timedelta(seconds=2),
        }
    )
    def observe_heartbeat(*details: object) -> None:
        assert details == ()
        heartbeat_times.append(time.monotonic())

    environment.on_heartbeat = observe_heartbeat

    def wait_through_heartbeats(kwargs: dict[str, Any]) -> object:
        del kwargs
        time.sleep(0.22)
        return SimpleNamespace(
            checkpoint_ref="checkpoint:temporal-activity-run/2",
            terminal=False,
        )

    _DriverDouble.behavior = wait_through_heartbeats
    policy = TemporalActivityPolicy(
        writer_lease_ttl_s=2,
        writer_lease_renew_interval_s=0.4,
        heartbeat_interval_s=1,
        authority_call_timeout_s=0.5,
        driver_call_timeout_s=1,
        supervisor_join_timeout_s=0.1,
        local_task_wait_s=1,
    )

    result = environment.run(
        _activity(_AuthorityStore(), policy=policy).run,
        _command().to_json(),
    )

    assert result["run_id"] == "temporal-activity-run"
    assert len(heartbeat_times) >= 4


def test_activity_reconciles_a_lost_claim_response_with_the_same_owner() -> None:
    class LoseFirstClaimResponse(_AuthorityStore):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.claim_calls = 0
            self.owners: list[str] = []

        def claim(self, run_id: str, owner_id: str, ttl: timedelta) -> WriterLease:
            self.claim_calls += 1
            self.owners.append(owner_id)
            lease = super().claim(run_id, owner_id, ttl)
            if self.claim_calls == 1:
                raise ConnectionError("private committed claim response was lost")
            return lease

    store = LoseFirstClaimResponse()
    raw = ActivityEnvironment().run(_activity(store).run, _command().to_json())

    assert TemporalActivationResult.from_json(raw).matches(_command())
    assert store.claim_calls == 2
    assert len(set(store.owners)) == 1
    assert store.release_count == 1


def test_activity_heartbeats_while_the_writer_claim_is_blocked() -> None:
    periodic_heartbeat = threading.Event()
    heartbeat_count = 0

    class BlockingClaimStore(_AuthorityStore):
        def claim(self, run_id: str, owner_id: str, ttl: timedelta) -> WriterLease:
            assert periodic_heartbeat.wait(1)
            return super().claim(run_id, owner_id, ttl)

    def observe_heartbeat(*details: object) -> None:
        nonlocal heartbeat_count
        assert details == ()
        heartbeat_count += 1
        if heartbeat_count >= 2:
            periodic_heartbeat.set()

    environment = ActivityEnvironment()
    environment.on_heartbeat = observe_heartbeat
    raw = environment.run(_activity(BlockingClaimStore()).run, _command().to_json())

    assert TemporalActivationResult.from_json(raw).matches(_command())
    assert heartbeat_count >= 2


def test_heartbeat_failure_during_claim_prevents_activation_drive() -> None:
    heartbeat_failed = threading.Event()
    heartbeat_count = 0

    class ClaimAfterHeartbeatFailure(_AuthorityStore):
        def claim(self, run_id: str, owner_id: str, ttl: timedelta) -> WriterLease:
            assert heartbeat_failed.wait(1)
            time.sleep(0.05)
            return super().claim(run_id, owner_id, ttl)

    def fail_periodic_heartbeat(*details: object) -> None:
        nonlocal heartbeat_count
        assert details == ()
        heartbeat_count += 1
        if heartbeat_count >= 2:
            heartbeat_failed.set()
            raise ConnectionError("private heartbeat transport failure")

    store = ClaimAfterHeartbeatFailure()
    environment = ActivityEnvironment()
    environment.on_heartbeat = fail_periodic_heartbeat
    with pytest.raises(ApplicationError) as raised:
        environment.run(_activity(store).run, _command().to_json())

    assert raised.value.type == "monoid.activation_lease_lost"
    assert raised.value.non_retryable is False
    assert _DriverDouble.constructed == []
    assert store.released.wait(1)
    assert store.release_count == 1


def test_writer_claim_timeout_returns_the_activity_executor_and_releases_late_authority() -> None:
    claim_started = threading.Event()
    allow_claim = threading.Event()

    class StuckClaimStore(_AuthorityStore):
        def claim(self, run_id: str, owner_id: str, ttl: timedelta) -> WriterLease:
            claim_started.set()
            assert allow_claim.wait(1)
            return super().claim(run_id, owner_id, ttl)

    policy = TemporalActivityPolicy(
        writer_lease_ttl_s=2,
        writer_lease_renew_interval_s=0.02,
        heartbeat_interval_s=0.01,
        authority_call_timeout_s=0.05,
        supervisor_join_timeout_s=0.1,
        local_task_wait_s=1,
    )
    store = StuckClaimStore()
    claim_timer = threading.Timer(0.4, allow_claim.set)
    claim_timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(ApplicationError) as raised:
            ActivityEnvironment().run(
                _activity(store, policy=policy).run,
                _command().to_json(),
            )
        elapsed = time.monotonic() - started
        assert raised.value.type == "monoid.activation_lease_lost"
        assert raised.value.non_retryable is False
        assert claim_started.is_set()
        assert _DriverDouble.constructed == []
        assert elapsed < 0.25
    finally:
        allow_claim.set()
        claim_timer.cancel()
        claim_timer.join()
        assert store.released.wait(1)
    assert store.release_count == 1


def test_bootstrap_timeout_is_capped_by_the_temporal_attempt_deadline() -> None:
    claim_started = threading.Event()
    allow_claim = threading.Event()

    class StuckClaimStore(_AuthorityStore):
        def claim(self, run_id: str, owner_id: str, ttl: timedelta) -> WriterLease:
            claim_started.set()
            assert allow_claim.wait(1)
            return super().claim(run_id, owner_id, ttl)

    policy = TemporalActivityPolicy(
        writer_lease_ttl_s=2,
        writer_lease_renew_interval_s=0.02,
        heartbeat_interval_s=0.01,
        authority_call_timeout_s=1,
        driver_call_timeout_s=1,
        supervisor_join_timeout_s=0.05,
        local_task_wait_s=1,
    )
    store = StuckClaimStore()
    environment = ActivityEnvironment()
    environment.info = environment.info.__class__(
        **{
            **environment.info.__dict__,
            "start_to_close_timeout": timedelta(seconds=0.25),
        }
    )
    claim_timer = threading.Timer(0.5, allow_claim.set)
    claim_timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(ApplicationError) as raised:
            environment.run(
                _activity(store, policy=policy).run,
                _command().to_json(),
            )
        elapsed = time.monotonic() - started
        assert raised.value.type == "monoid.activation_lease_lost"
        assert raised.value.non_retryable is False
        assert claim_started.is_set()
        assert _DriverDouble.constructed == []
        assert elapsed < 0.35
    finally:
        allow_claim.set()
        claim_timer.cancel()
        claim_timer.join()
        assert store.released.wait(1)
    assert store.release_count == 1


def test_activation_binding_timeout_returns_the_activity_executor_and_releases_authority() -> None:
    binding_started = threading.Event()
    allow_binding = threading.Event()

    class StuckAdmissionStore(_AdmissionStore):
        def bind_activation(
            self,
            command: AdmittedCommand,
            *,
            writer_token: WriterToken,
        ) -> ActivationCommand:
            binding_started.set()
            assert allow_binding.wait(1)
            return super().bind_activation(command, writer_token=writer_token)

    policy = TemporalActivityPolicy(
        writer_lease_ttl_s=2,
        writer_lease_renew_interval_s=0.02,
        heartbeat_interval_s=0.01,
        authority_call_timeout_s=0.05,
        supervisor_join_timeout_s=0.1,
        local_task_wait_s=1,
    )
    store = _AuthorityStore()
    binding_timer = threading.Timer(0.4, allow_binding.set)
    binding_timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(ApplicationError) as raised:
            ActivityEnvironment().run(
                _activity(
                    store,
                    admission_store=StuckAdmissionStore(),
                    policy=policy,
                ).run,
                _command().to_json(),
            )
        elapsed = time.monotonic() - started
        assert raised.value.type == "monoid.activation_lease_lost"
        assert raised.value.non_retryable is False
        assert binding_started.is_set()
        assert _DriverDouble.constructed == []
        assert elapsed < 0.25
    finally:
        allow_binding.set()
        binding_timer.cancel()
        binding_timer.join()
        assert store.released.wait(1)
    assert store.release_count == 1


def test_driver_timeout_returns_the_activity_executor_and_releases_late_authority() -> None:
    driver_started = threading.Event()
    allow_driver = threading.Event()

    def block_driver(kwargs: dict[str, Any]) -> object:
        driver_started.set()
        assert allow_driver.wait(1)
        assert kwargs["write_authority"].revoked is True
        return SimpleNamespace(
            checkpoint_ref="checkpoint:temporal-activity-run/2",
            terminal=False,
        )

    _DriverDouble.behavior = block_driver
    policy = TemporalActivityPolicy(
        writer_lease_ttl_s=2,
        writer_lease_renew_interval_s=0.02,
        heartbeat_interval_s=0.01,
        authority_call_timeout_s=1,
        driver_call_timeout_s=0.05,
        supervisor_join_timeout_s=0.1,
        local_task_wait_s=1,
    )
    store = _AuthorityStore()
    driver_timer = threading.Timer(0.4, allow_driver.set)
    driver_timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(ApplicationError) as raised:
            ActivityEnvironment().run(
                _activity(store, policy=policy).run,
                _command().to_json(),
            )
        elapsed = time.monotonic() - started
        assert raised.value.type == "monoid.activation_lease_lost"
        assert raised.value.non_retryable is False
        assert driver_started.is_set()
        assert elapsed < 0.3
    finally:
        allow_driver.set()
        driver_timer.cancel()
        driver_timer.join()
        assert store.released.wait(1)
    assert store.release_count == 1


def test_slow_activation_binding_renews_from_the_remaining_lease_deadline() -> None:
    class SlowAdmissionStore(_AdmissionStore):
        def bind_activation(
            self,
            command: AdmittedCommand,
            *,
            writer_token: WriterToken,
        ) -> ActivationCommand:
            time.sleep(0.07)
            return super().bind_activation(command, writer_token=writer_token)

    policy = TemporalActivityPolicy(
        writer_lease_ttl_s=2,
        writer_lease_renew_interval_s=0.04,
        heartbeat_interval_s=0.01,
        authority_call_timeout_s=1,
        supervisor_join_timeout_s=0.2,
        local_task_wait_s=1,
    )
    store = _AuthorityStore(lease_interval_s=0.1)

    def wait_for_immediate_renewal(kwargs: dict[str, Any]) -> object:
        del kwargs
        deadline = time.monotonic() + 0.2
        while store.renew_count < 2 and time.monotonic() < deadline:
            time.sleep(0.002)
        assert store.renew_count >= 2
        return SimpleNamespace(
            checkpoint_ref="checkpoint:temporal-activity-run/2",
            terminal=False,
        )

    _DriverDouble.behavior = wait_for_immediate_renewal
    raw = ActivityEnvironment().run(
        _activity(
            store,
            admission_store=SlowAdmissionStore(),
            policy=policy,
        ).run,
        _command().to_json(),
    )

    assert TemporalActivationResult.from_json(raw).matches(_command())
    assert store.renew_count >= 2
    assert store.release_count == 1


def test_activity_heartbeats_while_writer_renewal_is_blocked() -> None:
    renewal_blocked = threading.Event()
    heartbeat_during_renewal = threading.Event()

    class BlockingRenewalStore(_AuthorityStore):
        def renew(self, writer_token: WriterToken, ttl: timedelta) -> RenewResult:
            if self.renew_count == 1:
                renewal_blocked.set()
                assert heartbeat_during_renewal.wait(1)
            return super().renew(writer_token, ttl)

    def observe_heartbeat(*details: object) -> None:
        assert details == ()
        if renewal_blocked.is_set():
            heartbeat_during_renewal.set()

    store = BlockingRenewalStore()
    environment = ActivityEnvironment()
    environment.on_heartbeat = observe_heartbeat

    def wait_for_periodic_renewal(kwargs: dict[str, Any]) -> object:
        del kwargs
        deadline = time.monotonic() + 1
        while store.renew_count < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert store.renew_count >= 2
        return SimpleNamespace(
            checkpoint_ref="checkpoint:temporal-activity-run/2",
            terminal=False,
        )

    _DriverDouble.behavior = wait_for_periodic_renewal
    raw = environment.run(_activity(store).run, _command().to_json())

    assert TemporalActivationResult.from_json(raw).matches(_command())
    assert heartbeat_during_renewal.is_set()
    assert store.release_count == 1


def test_blocked_renewal_revokes_authority_at_the_local_lease_deadline() -> None:
    renewal_blocked = threading.Event()
    allow_renewal_return = threading.Event()
    external_call_started = threading.Event()

    class DeadlineStore(_AuthorityStore):
        def renew(self, writer_token: WriterToken, ttl: timedelta) -> RenewResult:
            if self.renew_count == 1:
                renewal_blocked.set()
                assert allow_renewal_return.wait(1)
            return super().renew(writer_token, ttl)

    store = DeadlineStore(lease_interval_s=0.08)

    def refuse_external_call_after_deadline(kwargs: dict[str, Any]) -> object:
        authority = kwargs["write_authority"]
        assert renewal_blocked.wait(1)
        deadline = time.monotonic() + 1
        while not authority.revoked and time.monotonic() < deadline:
            time.sleep(0.005)
        try:
            with pytest.raises(WriteAuthorityRevoked):
                authority.guard_external_call(external_call_started.set)
        finally:
            allow_renewal_return.set()
        return SimpleNamespace(
            checkpoint_ref="checkpoint:temporal-activity-run/2",
            terminal=False,
        )

    _DriverDouble.behavior = refuse_external_call_after_deadline
    with pytest.raises(ApplicationError) as raised:
        ActivityEnvironment().run(_activity(store).run, _command().to_json())

    assert raised.value.type == "monoid.activation_lease_lost"
    assert raised.value.non_retryable is False
    assert external_call_started.is_set() is False
    assert store.release_count == 1


def test_activity_heartbeats_while_cleanup_joins_a_blocked_renewal() -> None:
    renewal_blocked = threading.Event()
    drive_finished = threading.Event()
    heartbeat_after_drive = threading.Event()

    class CleanupRenewalStore(_AuthorityStore):
        def renew(self, writer_token: WriterToken, ttl: timedelta) -> RenewResult:
            if self.renew_count == 1:
                renewal_blocked.set()
                assert heartbeat_after_drive.wait(1)
            return super().renew(writer_token, ttl)

    def observe_heartbeat(*details: object) -> None:
        assert details == ()
        if renewal_blocked.is_set() and drive_finished.is_set():
            heartbeat_after_drive.set()

    def finish_while_renewal_is_blocked(kwargs: dict[str, Any]) -> object:
        del kwargs
        assert renewal_blocked.wait(1)
        drive_finished.set()
        return SimpleNamespace(
            checkpoint_ref="checkpoint:temporal-activity-run/2",
            terminal=False,
        )

    store = CleanupRenewalStore()
    environment = ActivityEnvironment()
    environment.on_heartbeat = observe_heartbeat
    _DriverDouble.behavior = finish_while_renewal_is_blocked

    raw = environment.run(_activity(store).run, _command().to_json())

    assert TemporalActivationResult.from_json(raw).matches(_command())
    assert heartbeat_after_drive.is_set()
    assert store.release_count == 1


def test_activity_heartbeats_while_writer_release_is_blocked() -> None:
    release_blocked = threading.Event()
    heartbeat_during_release = threading.Event()

    class BlockingReleaseStore(_AuthorityStore):
        def release(self, writer_token: WriterToken) -> ReleaseResult:
            release_blocked.set()
            assert heartbeat_during_release.wait(1)
            return super().release(writer_token)

    def observe_heartbeat(*details: object) -> None:
        assert details == ()
        if release_blocked.is_set():
            heartbeat_during_release.set()

    store = BlockingReleaseStore()
    environment = ActivityEnvironment()
    environment.on_heartbeat = observe_heartbeat
    raw = environment.run(_activity(store).run, _command().to_json())

    assert TemporalActivationResult.from_json(raw).matches(_command())
    assert heartbeat_during_release.is_set()
    assert store.release_count == 1


def test_writer_cleanup_timeout_is_bounded_and_retryable() -> None:
    release_started = threading.Event()
    allow_release = threading.Event()
    release_completed = threading.Event()

    class StuckReleaseStore(_AuthorityStore):
        def release(self, writer_token: WriterToken) -> ReleaseResult:
            release_started.set()
            assert allow_release.wait(1)
            result = super().release(writer_token)
            release_completed.set()
            return result

    policy = TemporalActivityPolicy(
        writer_lease_ttl_s=2,
        writer_lease_renew_interval_s=0.02,
        heartbeat_interval_s=0.01,
        supervisor_join_timeout_s=0.05,
        local_task_wait_s=1,
    )
    release_timer = threading.Timer(0.4, allow_release.set)
    release_timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(ApplicationError) as raised:
            ActivityEnvironment().run(
                _activity(StuckReleaseStore(), policy=policy).run,
                _command().to_json(),
            )
        elapsed = time.monotonic() - started
        assert raised.value.type == "monoid.activation_lease_lost"
        assert raised.value.non_retryable is False
        assert release_started.is_set()
        assert elapsed < 0.25
    finally:
        allow_release.set()
        release_timer.cancel()
        release_timer.join()
        assert release_completed.wait(1)


def test_worker_shutdown_policy_can_report_host_shutdown() -> None:
    store = _AuthorityStore()
    environment = ActivityEnvironment()
    entered = threading.Event()
    outcome: list[object] = []
    policy = TemporalActivityPolicy(
        writer_lease_ttl_s=2,
        writer_lease_renew_interval_s=0.02,
        heartbeat_interval_s=0.01,
        supervisor_join_timeout_s=0.2,
        local_task_wait_s=1,
        worker_shutdown_cause=InterruptionCause.HOST_SHUTDOWN,
    )

    def wait_for_shutdown(kwargs: dict[str, Any]) -> object:
        entered.set()
        token = kwargs["cancellation_token"]
        deadline = time.monotonic() + 2
        while not token.requested and time.monotonic() < deadline:
            time.sleep(0.005)
        assert token.cause is InterruptionCause.HOST_SHUTDOWN
        return SimpleNamespace(
            checkpoint_ref="checkpoint:temporal-activity-run/2",
            terminal=False,
        )

    _DriverDouble.behavior = wait_for_shutdown
    thread = threading.Thread(
        target=lambda: outcome.append(
            environment.run(_activity(store, policy=policy).run, _command().to_json())
        )
    )
    thread.start()
    assert entered.wait(1)
    environment.worker_shutdown()
    thread.join(3)

    assert not thread.is_alive()
    assert TemporalActivationResult.from_json(outcome[0]).matches(_command())


@pytest.mark.parametrize(
    ("notify", "expected_cause"),
    (
        ("cancel", InterruptionCause.USER_CANCEL),
        ("worker_shutdown", InterruptionCause.GRACEFUL_DRAIN),
    ),
)
def test_activity_control_requests_share_the_driver_cancellation_token(
    notify: str,
    expected_cause: InterruptionCause,
) -> None:
    store = _AuthorityStore()
    environment = ActivityEnvironment()
    entered = threading.Event()
    outcome: list[object] = []

    def wait_for_control(kwargs: dict[str, Any]) -> object:
        entered.set()
        token = kwargs["cancellation_token"]
        deadline = time.monotonic() + 2
        while not token.requested and time.monotonic() < deadline:
            time.sleep(0.005)
        assert token.cause is expected_cause
        return SimpleNamespace(
            checkpoint_ref="checkpoint:temporal-activity-run/2",
            terminal=False,
        )

    _DriverDouble.behavior = wait_for_control
    thread = threading.Thread(
        target=lambda: outcome.append(environment.run(_activity(store).run, _command().to_json()))
    )
    thread.start()
    assert entered.wait(1)
    getattr(environment, notify)()
    thread.join(3)

    assert not thread.is_alive()
    assert TemporalActivationResult.from_json(outcome[0]).matches(_command())
    assert store.release_count == 1


def test_renew_fence_revokes_activity_and_returns_retryable_public_error() -> None:
    store = _AuthorityStore(renew_fenced_after=2)
    environment = ActivityEnvironment()

    def wait_for_fence(kwargs: dict[str, Any]) -> object:
        authority = kwargs["write_authority"]
        deadline = time.monotonic() + 2
        while not authority.revoked and time.monotonic() < deadline:
            time.sleep(0.005)
        authority.assert_active()
        raise AssertionError("unreachable")

    _DriverDouble.behavior = wait_for_fence
    with pytest.raises(ApplicationError) as raised:
        environment.run(_activity(store).run, _command().to_json())

    assert raised.value.type == "monoid.activation_lease_lost"
    assert raised.value.non_retryable is False
    assert store.release_count == 1


def test_fenced_release_prevents_successful_activity_result() -> None:
    store = _AuthorityStore(release_status="fenced")

    with pytest.raises(ApplicationError) as raised:
        ActivityEnvironment().run(_activity(store).run, _command().to_json())

    assert raised.value.type == "monoid.activation_lease_lost"
    assert raised.value.non_retryable is False


def test_competing_writer_lease_returns_retryable_public_error() -> None:
    class BusyAuthorityStore(_AuthorityStore):
        def claim(self, run_id: str, owner_id: str, ttl: timedelta) -> WriterLease:
            del owner_id, ttl
            competing = WriterToken(run_id=run_id, owner_id="competing-owner", generation=7)
            raise WriterLeaseUnavailable(self._authority(competing, revoked=False))

    store = BusyAuthorityStore()
    with pytest.raises(ApplicationError) as raised:
        ActivityEnvironment().run(_activity(store).run, _command().to_json())

    assert raised.value.type == "monoid.activation_lease_unavailable"
    assert raised.value.non_retryable is False
    assert raised.value.next_retry_delay == timedelta(seconds=30)
    assert "competing-owner" not in str(raised.value)
    assert store.release_count == 0


def test_activation_binding_writer_fence_returns_retryable_lease_loss() -> None:
    class FencedAdmissionStore(_AdmissionStore):
        def bind_activation(
            self,
            command: AdmittedCommand,
            *,
            writer_token: WriterToken,
        ) -> object:
            del command, writer_token
            raise ActivationBindingWriterFenced("private fenced writer detail")

    store = _AuthorityStore()
    with pytest.raises(ApplicationError) as raised:
        ActivityEnvironment().run(
            _activity(store, admission_store=FencedAdmissionStore()).run,
            _command().to_json(),
        )

    assert raised.value.type == "monoid.activation_lease_lost"
    assert raised.value.non_retryable is False
    assert "private fenced writer detail" not in str(raised.value)
    assert store.released.wait(1)
    assert store.release_count == 1


def test_durable_activation_binding_conflict_remains_nonretryable() -> None:
    class ConflictingAdmissionStore(_AdmissionStore):
        def bind_activation(
            self,
            command: AdmittedCommand,
            *,
            writer_token: WriterToken,
        ) -> object:
            del command, writer_token
            raise ActivationBindingConflict("private durable identity conflict")

    store = _AuthorityStore()
    with pytest.raises(ApplicationError) as raised:
        ActivityEnvironment().run(
            _activity(
                store,
                admission_store=ConflictingAdmissionStore(),
            ).run,
            _command().to_json(),
        )

    assert raised.value.type == "monoid.activation_config_conflict"
    assert raised.value.non_retryable is True
    assert "private durable identity conflict" not in str(raised.value)
    assert store.released.wait(1)
    assert store.release_count == 1


def test_loop_wiring_failure_is_a_nonretryable_configuration_error() -> None:
    private_text = "private loop wiring detail"

    def fail(kwargs: dict[str, Any]) -> object:
        del kwargs
        raise ActivationLoopConfigurationError(private_text)

    store = _AuthorityStore()
    _DriverDouble.behavior = fail
    with pytest.raises(ApplicationError) as raised:
        ActivityEnvironment().run(_activity(store).run, _command().to_json())

    assert raised.value.type == "monoid.activation_config_conflict"
    assert raised.value.non_retryable is True
    assert private_text not in str(raised.value)
    assert store.release_count == 1


def test_activity_error_taxonomy_never_serializes_private_exception_text() -> None:
    store = _AuthorityStore()
    private_text = "raw-private-provider-or-database-failure"

    def fail(kwargs: dict[str, Any]) -> object:
        del kwargs
        raise RuntimeError(private_text)

    _DriverDouble.behavior = fail
    with pytest.raises(ApplicationError) as raised:
        ActivityEnvironment().run(_activity(store).run, _command().to_json())

    assert raised.value.type == "monoid.activation_transient"
    assert private_text not in str(raised.value)
    assert private_text not in repr(raised.value)


def test_invalid_activity_payload_is_nonretryable_and_does_not_echo_private_fields() -> None:
    private_text = "raw-private-model-response"
    payload = {**_command().to_json(), "model_response": private_text}

    with pytest.raises(ApplicationError) as raised:
        ActivityEnvironment().run(_activity(_AuthorityStore()).run, payload)

    assert raised.value.type == "monoid.activation_corrupt"
    assert raised.value.non_retryable is True
    assert private_text not in str(raised.value)


def test_worker_group_starts_activity_before_workflow_and_drains_in_reverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle: list[str] = []
    created: list[dict[str, Any]] = []

    class FakeWorker:
        def __init__(self, client: object, **kwargs: Any) -> None:
            del client
            self.kind = "workflow" if kwargs.get("workflows") else "activity"
            self.kwargs = kwargs
            created.append(kwargs)

        async def __aenter__(self) -> object:
            lifecycle.append(f"enter-{self.kind}")
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            del exc_info
            lifecycle.append(f"exit-{self.kind}")

    monkeypatch.setattr(worker_module, "Worker", FakeWorker)
    group = worker_module.TemporalWorkerGroup(
        client=object(),
        workflow_task_queue="workflow-v1",
        activity_task_queue="activity-v1",
        activation_activity=_activity(_AuthorityStore()),
        max_concurrent_activities=3,
        graceful_shutdown_timeout_s=17,
    )

    async def run() -> None:
        async with group:
            assert lifecycle == ["enter-activity", "enter-workflow"]

    asyncio.run(run())

    assert lifecycle == [
        "enter-activity",
        "enter-workflow",
        "exit-workflow",
        "exit-activity",
    ]
    activity_options = next(options for options in created if options.get("activities"))
    assert activity_options["max_concurrent_activities"] == 3
    assert activity_options["graceful_shutdown_timeout"] == timedelta(seconds=17)
    assert group._activity_executor._shutdown is True


def test_worker_group_rejects_a_shutdown_window_shorter_than_supervisor_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_module, "Worker", lambda *args, **kwargs: (args, kwargs))

    with pytest.raises(ValueError, match="cover Activity local timeout bounds"):
        worker_module.TemporalWorkerGroup(
            client=object(),
            workflow_task_queue="workflow-v1",
            activity_task_queue="activity-v1",
            activation_activity=_activity(_AuthorityStore()),
            graceful_shutdown_timeout_s=0.5,
        )


def test_owned_executor_cleanup_does_not_join_a_stuck_activity_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeWorker:
        def __init__(self, client: object, **kwargs: Any) -> None:
            del client, kwargs

        async def __aenter__(self) -> object:
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            del exc_info

    monkeypatch.setattr(worker_module, "Worker", FakeWorker)
    group = worker_module.TemporalWorkerGroup(
        client=object(),
        workflow_task_queue="workflow-v1",
        activity_task_queue="activity-v1",
        activation_activity=_activity(_AuthorityStore()),
        max_concurrent_activities=1,
        graceful_shutdown_timeout_s=2,
    )
    started = threading.Event()
    release = threading.Event()
    future: Any = None

    def block() -> None:
        started.set()
        assert release.wait(2)

    async def run() -> float:
        nonlocal future
        async with group:
            future = group._activity_executor.submit(block)
            assert started.wait(1)
            exit_started = time.monotonic()
        return time.monotonic() - exit_started

    release_timer = threading.Timer(0.5, release.set)
    release_timer.start()
    try:
        exit_elapsed = asyncio.run(run())
        assert exit_elapsed < 0.25
        assert group._activity_executor._shutdown is True
    finally:
        release.set()
        release_timer.cancel()
        release_timer.join()
        if future is not None:
            future.result(timeout=1)


def test_worker_group_preserves_a_user_owned_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeWorker:
        def __init__(self, client: object, **kwargs: Any) -> None:
            del client, kwargs

        async def __aenter__(self) -> object:
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            del exc_info

    monkeypatch.setattr(worker_module, "Worker", FakeWorker)
    executor = worker_module.ThreadPoolExecutor(max_workers=2)
    try:
        group = worker_module.TemporalWorkerGroup(
            client=object(),
            workflow_task_queue="workflow-v1",
            activity_task_queue="activity-v1",
            activation_activity=_activity(_AuthorityStore()),
            max_concurrent_activities=2,
            graceful_shutdown_timeout_s=2,
            activity_executor=executor,
        )

        async def run() -> None:
            async with group:
                pass

        asyncio.run(run())
        assert executor.submit(lambda: "owned-by-host").result() == "owned-by-host"
    finally:
        executor.shutdown(wait=True)


def test_worker_group_rejects_an_undersized_external_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_module, "Worker", lambda *args, **kwargs: (args, kwargs))
    executor = worker_module.ThreadPoolExecutor(max_workers=1)
    try:
        with pytest.raises(ValueError, match="capacity"):
            worker_module.TemporalWorkerGroup(
                client=object(),
                workflow_task_queue="workflow-v1",
                activity_task_queue="activity-v1",
                activation_activity=_activity(_AuthorityStore()),
                max_concurrent_activities=2,
                graceful_shutdown_timeout_s=2,
                activity_executor=executor,
            )
        assert executor.submit(lambda: "still-host-owned").result() == "still-host-owned"
    finally:
        executor.shutdown(wait=True)


def test_worker_group_rejects_a_shutdown_external_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_module, "Worker", lambda *args, **kwargs: (args, kwargs))
    executor = worker_module.ThreadPoolExecutor(max_workers=1)
    executor.shutdown(wait=True)

    with pytest.raises(ValueError, match="must be active"):
        worker_module.TemporalWorkerGroup(
            client=object(),
            workflow_task_queue="workflow-v1",
            activity_task_queue="activity-v1",
            activation_activity=_activity(_AuthorityStore()),
            max_concurrent_activities=1,
            graceful_shutdown_timeout_s=2,
            activity_executor=executor,
        )


def test_worker_group_releases_an_owned_executor_after_constructor_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executors: list[Any] = []

    def fail_on_workflow(client: object, **kwargs: Any) -> object:
        del client
        if kwargs.get("activities"):
            executors.append(kwargs["activity_executor"])
            return object()
        raise RuntimeError("workflow worker construction failed")

    monkeypatch.setattr(worker_module, "Worker", fail_on_workflow)
    with pytest.raises(RuntimeError, match="construction failed"):
        worker_module.TemporalWorkerGroup(
            client=object(),
            workflow_task_queue="workflow-v1",
            activity_task_queue="activity-v1",
            activation_activity=_activity(_AuthorityStore()),
            graceful_shutdown_timeout_s=2,
        )

    assert len(executors) == 1
    assert executors[0]._shutdown is True
