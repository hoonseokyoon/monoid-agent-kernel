"""Finite threaded Temporal Activity backed by neutral Monoid hosting contracts.

Import this module only in environments with the ``temporal`` optional dependency installed.
"""

from __future__ import annotations

import contextvars
import hashlib
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from monoid_agent_kernel.core.authority import ActivationWriteAuthority, WriteAuthorityRevoked
from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.interruption import InterruptionCause
from monoid_agent_kernel.hosting.activation import (
    ActivationDriver,
    ActivationInputResolver,
    ActivationLoopFactory,
)
from monoid_agent_kernel.hosting.admission import (
    ActivationBindingConflict,
    ActivationBindingWriterFenced,
    AdmittedCommand,
    CommandAdmissionStore,
)
from monoid_agent_kernel.hosting.authority import (
    ReleaseResult,
    WriterAuthorityStore,
    WriterLease,
    WriterLeaseUnavailable,
    claim_writer_lease,
    renew_writer_lease,
)
from monoid_agent_kernel.hosting.contracts import FencedRunSink

from .dependency import TemporalDependencyMissing
from .names import TEMPORAL_DRIVE_ACTIVATION_ACTIVITY
from .records import TemporalActivationResult

try:
    from temporalio import activity
    from temporalio.exceptions import ApplicationError
except ImportError as exc:  # pragma: no cover - exercised by isolated import tests
    raise TemporalDependencyMissing(
        "install monoid-agent-kernel[temporal] to use the Temporal Activity"
    ) from exc


MAX_TEMPORAL_ACTIVITY_LOCAL_DURATION_S = 7 * 24 * 60 * 60

_CORRUPT_ERROR_CODES = frozenset(
    {
        "admission_corrupt",
        "admission_conflict",
        "checkpoint_corrupt",
        "checkpoint_missing",
        "checkpoint_run_mismatch",
        "invalid_activation_boundary",
        "invalid_activation_receipt",
        "invalid_active_input",
        "missing_activation_boundary",
        "missing_activation_marker",
        "missing_activation_receipt",
        "terminal_receipt_mismatch",
    }
)
_UNSUPPORTED_ERROR_CODES = frozenset({"checkpoint_unsupported_version"})
_CONFIG_CONFLICT_ERROR_CODES = frozenset(
    {
        "activation_binding_conflict",
        "activation_boundary_mismatch",
        "activation_identity_mismatch",
        "activation_input_resolver_missing",
        "activation_loop_config_conflict",
        "activation_payload_mismatch",
        "activation_source_mismatch",
        "admission_run_terminal",
        "admission_run_unavailable",
        "loop_run_mismatch",
        "prior_activation_incomplete",
        "run_terminal",
        "stale_activation_source",
    }
)


def _require_duration(value: object, field_name: str, *, minimum: float = 0.001) -> float:
    if (
        type(value) not in {int, float}
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not minimum <= float(value) <= MAX_TEMPORAL_ACTIVITY_LOCAL_DURATION_S
    ):
        raise ValueError(
            f"{field_name} must be finite and in "
            f"[{minimum}, {MAX_TEMPORAL_ACTIVITY_LOCAL_DURATION_S}]"
        )
    return float(value)


@dataclass(frozen=True, kw_only=True)
class TemporalActivityPolicy:
    """Process-local lease, heartbeat, and shutdown policy for activation Activities."""

    writer_lease_ttl_s: float = 30.0
    writer_lease_renew_interval_s: float = 10.0
    heartbeat_interval_s: float = 5.0
    supervisor_join_timeout_s: float = 30.0
    local_task_wait_s: float = 300.0
    worker_shutdown_cause: InterruptionCause = InterruptionCause.GRACEFUL_DRAIN

    def __post_init__(self) -> None:
        ttl = _require_duration(self.writer_lease_ttl_s, "writer lease ttl", minimum=1.0)
        renew = _require_duration(
            self.writer_lease_renew_interval_s,
            "writer lease renew interval",
        )
        _require_duration(self.heartbeat_interval_s, "Temporal heartbeat interval")
        _require_duration(self.supervisor_join_timeout_s, "lease supervisor join timeout")
        _require_duration(self.local_task_wait_s, "activation local task wait")
        if renew * 2 > ttl:
            raise ValueError("writer lease ttl must cover at least two renew intervals")
        if self.worker_shutdown_cause not in {
            InterruptionCause.GRACEFUL_DRAIN,
            InterruptionCause.HOST_SHUTDOWN,
        }:
            raise ValueError(
                "worker shutdown cause must be graceful_drain or host_shutdown"
            )

    @property
    def writer_lease_ttl(self) -> timedelta:
        return timedelta(seconds=float(self.writer_lease_ttl_s))


def _activity_owner_id(task_token: bytes) -> str:
    if type(task_token) is not bytes or not task_token:
        raise ValueError("Temporal Activity task token must be non-empty bytes")
    return f"temporal-activity-{hashlib.sha256(task_token).hexdigest()}"


class _SupervisorUnhealthy(RuntimeError):
    pass


class _TemporalLeaseSupervisor:
    def __init__(
        self,
        *,
        store: WriterAuthorityStore,
        policy: TemporalActivityPolicy,
        write_authority: ActivationWriteAuthority,
        cancellation_token: CancellationToken,
    ) -> None:
        self._store = store
        self._lease: WriterLease | None = None
        self._lease_lock = threading.Lock()
        self._policy = policy
        self._write_authority = write_authority
        self._cancellation_token = cancellation_token
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._failure = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("lease supervisor can only be started once")
        activity_context = contextvars.copy_context()
        self._thread = threading.Thread(
            target=activity_context.run,
            args=(self._run,),
            name="monoid-temporal-lease-supervisor",
            daemon=True,
        )
        self._thread.start()

    def install_lease(self, lease: WriterLease) -> None:
        if not isinstance(lease, WriterLease):
            raise TypeError("lease supervisor requires WriterLease")
        self.assert_healthy()
        with self._lease_lock:
            if self._lease is not None:
                raise RuntimeError("lease supervisor authority can only be installed once")
            self._lease = lease
        self._wake.set()

    def stop_and_join(self) -> bool:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(float(self._policy.supervisor_join_timeout_s))
        if thread.is_alive():
            self._fail()
            return False
        return True

    def assert_healthy(self) -> None:
        if self._failure.is_set():
            raise _SupervisorUnhealthy("Temporal activation lease supervisor is unhealthy")

    def _fail(self) -> None:
        self._failure.set()
        self._write_authority.revoke()
        self._cancellation_token._cancel_for_authority_loss()

    def _observe_control(self) -> None:
        if activity.is_cancelled():
            self._cancellation_token.cancel(InterruptionCause.USER_CANCEL)
        if activity.is_worker_shutdown():
            self._cancellation_token.cancel(self._policy.worker_shutdown_cause)

    def _run(self) -> None:
        heartbeat_interval = float(self._policy.heartbeat_interval_s)
        renew_interval = float(self._policy.writer_lease_renew_interval_s)
        now = time.monotonic()
        next_heartbeat = now + heartbeat_interval
        next_renew: float | None = None
        try:
            while True:
                self._observe_control()
                if self._stop.is_set():
                    return
                now = time.monotonic()
                if now >= next_heartbeat:
                    activity.heartbeat()
                    next_heartbeat = now + heartbeat_interval
                with self._lease_lock:
                    lease = self._lease
                if lease is not None and next_renew is None:
                    next_renew = now + renew_interval
                if lease is not None and next_renew is not None and now >= next_renew:
                    renewed = renew_writer_lease(
                        self._store,
                        lease.writer_token,
                        self._policy.writer_lease_ttl,
                        write_authority=self._write_authority,
                    )
                    if renewed.status != "renewed" or renewed.lease is None:
                        self._fail()
                        return
                    with self._lease_lock:
                        self._lease = renewed.lease
                    next_renew = now + renew_interval
                deadlines = [next_heartbeat]
                if next_renew is not None:
                    deadlines.append(next_renew)
                wait_s = max(0.0, min(deadlines) - time.monotonic())
                self._wake.wait(wait_s)
                self._wake.clear()
        except BaseException:  # the public Activity exposes only a stable failure taxonomy
            self._fail()


def _application_error(exc: Exception) -> ApplicationError:
    if isinstance(exc, WriterLeaseUnavailable):
        retry_delay_s = min(
            MAX_TEMPORAL_ACTIVITY_LOCAL_DURATION_S,
            max(
                0.001,
                (exc.authority.leased_until - exc.authority.observed_at).total_seconds(),
            ),
        )
        return ApplicationError(
            "Temporal activation writer lease is temporarily unavailable",
            type="monoid.activation_lease_unavailable",
            next_retry_delay=timedelta(seconds=retry_delay_s),
        )
    if isinstance(
        exc,
        (WriteAuthorityRevoked, _SupervisorUnhealthy, ActivationBindingWriterFenced),
    ):
        return ApplicationError(
            "Temporal activation lost writer authority",
            type="monoid.activation_lease_lost",
        )
    if isinstance(exc, ActivationBindingConflict):
        return ApplicationError(
            "Temporal activation conflicts with the durable command binding",
            type="monoid.activation_config_conflict",
            non_retryable=True,
        )
    code = getattr(exc, "error_code", "")
    if code in _CORRUPT_ERROR_CODES:
        return ApplicationError(
            "Temporal activation durable state is corrupt",
            type="monoid.activation_corrupt",
            non_retryable=True,
        )
    if code in _UNSUPPORTED_ERROR_CODES:
        return ApplicationError(
            "Temporal activation durable state version is unsupported",
            type="monoid.activation_unsupported",
            non_retryable=True,
        )
    if code in _CONFIG_CONFLICT_ERROR_CODES or isinstance(exc, (TypeError, ValueError)):
        return ApplicationError(
            "Temporal activation configuration or identity conflicts with durable state",
            type="monoid.activation_config_conflict",
            non_retryable=True,
        )
    return ApplicationError(
        "Temporal activation encountered a transient infrastructure failure",
        type="monoid.activation_transient",
    )


@dataclass(kw_only=True)
class TemporalActivationActivity:
    """Claim one writer generation and drive one admitted command to a durable boundary."""

    authority_store: WriterAuthorityStore = field(repr=False)
    admission_store: CommandAdmissionStore = field(repr=False)
    run_sink: FencedRunSink = field(repr=False)
    loop_factory: ActivationLoopFactory = field(repr=False)
    input_resolver: ActivationInputResolver | None = field(default=None, repr=False)
    policy: TemporalActivityPolicy = field(default_factory=TemporalActivityPolicy)

    def __post_init__(self) -> None:
        if not callable(self.loop_factory):
            raise TypeError("Temporal activation loop_factory must be callable")
        if self.input_resolver is not None and not callable(self.input_resolver):
            raise TypeError("Temporal activation input_resolver must be callable")
        if not isinstance(self.policy, TemporalActivityPolicy):
            raise TypeError("Temporal activation policy must be TemporalActivityPolicy")

    @activity.defn(
        name=TEMPORAL_DRIVE_ACTIVATION_ACTIVITY,
        no_thread_cancel_exception=True,
    )
    def run(self, payload: dict[str, Any]) -> dict[str, object]:
        try:
            command = AdmittedCommand.from_json(payload)
        except (TypeError, ValueError, OverflowError, RecursionError):
            raise ApplicationError(
                "Temporal activation command payload is invalid",
                type="monoid.activation_corrupt",
                non_retryable=True,
            ) from None

        write_authority = ActivationWriteAuthority()
        cancellation_token = CancellationToken()
        supervisor = _TemporalLeaseSupervisor(
            store=self.authority_store,
            policy=self.policy,
            write_authority=write_authority,
            cancellation_token=cancellation_token,
        )
        lease: WriterLease | None = None
        result: TemporalActivationResult | None = None
        primary_error: Exception | None = None
        release_error: Exception | None = None
        supervisor_stopped = True
        try:
            info = activity.info()
            owner_id = _activity_owner_id(info.task_token)
            activity.heartbeat()
            supervisor.start()
            lease = claim_writer_lease(
                self.authority_store,
                command.run_id,
                owner_id,
                self.policy.writer_lease_ttl,
            )
            supervisor.assert_healthy()
            write_authority.assert_active()
            renewed = renew_writer_lease(
                self.authority_store,
                lease.writer_token,
                self.policy.writer_lease_ttl,
                write_authority=write_authority,
            )
            if renewed.status != "renewed" or renewed.lease is None:
                raise WriteAuthorityRevoked()
            lease = renewed.lease
            supervisor.assert_healthy()
            write_authority.assert_active()
            supervisor.install_lease(lease)
            supervisor.assert_healthy()
            write_authority.assert_active()
            if activity.is_cancelled():
                cancellation_token.cancel(InterruptionCause.USER_CANCEL)
            if activity.is_worker_shutdown():
                cancellation_token.cancel(self.policy.worker_shutdown_cause)
            activation = self.admission_store.bind_activation(
                command,
                writer_token=lease.writer_token,
            )
            receipt = ActivationDriver(
                sink=self.run_sink,
                writer_token=lease.writer_token,
                loop_factory=self.loop_factory,
                input_resolver=self.input_resolver,
                write_authority=write_authority,
                cancellation_token=cancellation_token,
                local_task_wait_s=float(self.policy.local_task_wait_s),
            ).drive(activation)
            supervisor.assert_healthy()
            write_authority.assert_active()
            result = TemporalActivationResult.from_command(
                command,
                receipt_ref=receipt.checkpoint_ref,
                terminal=receipt.terminal,
            )
        except Exception as exc:
            primary_error = exc
        finally:
            supervisor_stopped = supervisor.stop_and_join()
            try:
                supervisor.assert_healthy()
            except Exception as exc:
                if primary_error is None:
                    primary_error = exc
            if not supervisor_stopped:
                release_error = _SupervisorUnhealthy(
                    "Temporal activation lease supervisor did not stop"
                )
            elif lease is not None:
                try:
                    released = self.authority_store.release(lease.writer_token)
                    if not isinstance(released, ReleaseResult):
                        raise TypeError("writer authority store returned an invalid release result")
                    if released.status not in {"released", "already_released"}:
                        raise WriteAuthorityRevoked()
                except Exception as exc:
                    release_error = exc
            write_authority.revoke()

        if primary_error is not None:
            raise _application_error(primary_error) from None
        if release_error is not None:
            raise _application_error(release_error) from None
        if result is None:  # pragma: no cover - guarded by the branches above
            raise ApplicationError(
                "Temporal activation produced no public result",
                type="monoid.activation_transient",
            )
        return result.to_json()


__all__ = [
    "MAX_TEMPORAL_ACTIVITY_LOCAL_DURATION_S",
    "TemporalActivityPolicy",
    "TemporalActivationActivity",
]
