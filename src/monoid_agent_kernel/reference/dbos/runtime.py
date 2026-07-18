"""Process-owned DBOS runtime primitives shared by optional Reference components."""

from __future__ import annotations

import importlib
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any
from typing import Literal

from monoid_agent_kernel.reference.dbos._compat_226 import (
    _construct_owned_runtime_226,
    _DbosConstructionUncertain,
    _DbosOwnershipConflict,
    _OwnedDbosRuntime226,
)

DBOS_REFERENCE_WORKFLOW_NAMESPACE = "monoid.reference"
DbosHostState = Literal[
    "registering",
    "launching",
    "running",
    "closing",
    "closed",
    "fenced",
]

_SAFE_OPERATION_TOKEN = re.compile(r"[a-z][a-z0-9]{0,63}").fullmatch

__all__ = [
    "DbosDependencyError",
    "DbosProcessOwnershipError",
    "DbosShutdownTimeout",
]


class DbosDependencyError(RuntimeError):
    """Raised when the explicitly selected DBOS profile is unavailable."""


class DbosProcessOwnershipError(RuntimeError):
    """Raised when another Reference component owns the process-global DBOS runtime."""


class DbosShutdownTimeout(RuntimeError):
    """Raised when DBOS stopped state cannot be proven within the shutdown grace."""


@dataclass(frozen=True)
class DbosHostConfig:
    """Private process-wide DBOS identity for the optional Reference host."""

    system_database_url: str
    name: str = "monoid-reference-dbos"
    application_version: str = "monoid-reference-dbos-v1"
    executor_id: str = "stable-local-slot"
    shutdown_grace_s: int = 30

    def __post_init__(self) -> None:
        for value, label in (
            (self.system_database_url, "system_database_url"),
            (self.name, "name"),
            (self.application_version, "application_version"),
            (self.executor_id, "executor_id"),
        ):
            if type(value) is not str or not value:
                raise ValueError(f"DBOS host {label} is required")
        _require_shutdown_grace(self.shutdown_grace_s)

    @property
    def workflow_namespace(self) -> str:
        return DBOS_REFERENCE_WORKFLOW_NAMESPACE

    def workflow_name(self, surface: str, operation: str) -> str:
        """Return a deterministic v1 workflow-registry name."""

        if (
            type(surface) is not str
            or _SAFE_OPERATION_TOKEN(surface) is None
            or type(operation) is not str
            or _SAFE_OPERATION_TOKEN(operation) is None
        ):
            raise ValueError("DBOS workflow surface and operation must be safe tokens")
        return f"{self.workflow_namespace}.dbos-{surface}-{operation}.v1"


@dataclass(frozen=True, kw_only=True)
class _DbosHostParticipant:
    """Private lifecycle hooks registered by one host-owned Reference component.

    Hooks run on host lifecycle threads. Admission, activity, and close hooks must return
    promptly; a missed close deadline fences ownership for process termination.
    """

    participant_id: str
    queue_name: str
    host_config: DbosHostConfig
    register_workflows: Callable[[Any], None]
    preflight: Callable[[], None]
    register_queue: Callable[[Any], None]
    stop_admission: Callable[[], None]
    active_count: Callable[[], int]
    mark_closed: Callable[[], None]

    def __post_init__(self) -> None:
        if type(self.participant_id) is not str or not self.participant_id:
            raise ValueError("DBOS host participant_id is required")
        if type(self.queue_name) is not str or not self.queue_name:
            raise ValueError("DBOS host participant queue_name is required")
        if not isinstance(self.host_config, DbosHostConfig):
            raise TypeError("DBOS host participant requires DbosHostConfig")
        for callback in (
            self.register_workflows,
            self.preflight,
            self.register_queue,
            self.stop_admission,
            self.active_count,
            self.mark_closed,
        ):
            if not callable(callback):
                raise TypeError("DBOS host participant hooks must be callable")


class DbosRuntimeHost:
    """Private owner for one DBOS runtime, listener set, launch, and shutdown."""

    def __init__(self, config: DbosHostConfig) -> None:
        if not isinstance(config, DbosHostConfig):
            raise TypeError("DBOS runtime host requires DbosHostConfig")
        self.config = config
        self._state_lock = threading.Lock()
        self._state_condition = threading.Condition(self._state_lock)
        self._state: DbosHostState = "registering"
        self._admission_open = False
        self._close_requested = False
        self._lifecycle_thread_id: int | None = None
        self._shutdown_owner_id: int | None = None
        self._shutdown_deadline: float | None = None
        self._shutdown_grace_s: int | None = None
        self._shutdown_error: DbosShutdownTimeout | None = None
        self._shutdown_thread: threading.Thread | None = None
        self._shutdown_watchdog: threading.Thread | None = None
        self._fenced_cleanup_thread: threading.Thread | None = None
        self._participants: dict[str, _DbosHostParticipant] = {}
        self._owner_token = claim_process_owner()
        try:
            dbos_module = load_dbos()
        except BaseException:
            release_process_owner(self._owner_token)
            raise
        try:
            self._runtime: _OwnedDbosRuntime226 = _construct_owned_runtime_226(dbos_module, config)
        except _DbosOwnershipConflict as exc:
            release_process_owner(self._owner_token)
            raise DbosProcessOwnershipError(str(exc)) from None
        except _DbosConstructionUncertain:
            raise DbosProcessOwnershipError(
                "DBOS host construction is uncertain; terminate the process"
            ) from None
        except BaseException:
            raise DbosProcessOwnershipError(
                "DBOS host construction is uncertain; terminate the process"
            ) from None

    @property
    def state(self) -> DbosHostState:
        with self._state_lock:
            return self._state

    @property
    def accepting(self) -> bool:
        with self._state_lock:
            return self._admission_open and self._state == "running"

    def workflow_name(self, surface: str, operation: str) -> str:
        return self.config.workflow_name(surface, operation)

    def _register_participant(self, participant: _DbosHostParticipant) -> None:
        if not isinstance(participant, _DbosHostParticipant):
            raise TypeError("DBOS runtime host participant must be typed")
        with self._state_lock:
            if self._state != "registering":
                raise RuntimeError("DBOS participants must register before host launch")
            if participant.host_config != self.config:
                raise ValueError("DBOS host participant process identity does not match")
            if participant.participant_id in self._participants:
                raise ValueError("DBOS host participant ids must be unique")
            if any(
                existing.queue_name == participant.queue_name
                for existing in self._participants.values()
            ):
                raise ValueError("DBOS host participant queues must be unique")
            self._participants[participant.participant_id] = participant

    def launch(self) -> None:
        with self._state_lock:
            if self._state == "running":
                return
            if self._state != "registering":
                raise RuntimeError(f"DBOS runtime host cannot launch from {self._state}")
            if not self._participants:
                raise RuntimeError("DBOS runtime host requires at least one participant")
            self._state = "launching"
            self._lifecycle_thread_id = threading.get_ident()
            participants = self._ordered_participants()

        try:
            self._require_launching_owner()
            for participant in participants:
                participant.register_workflows(self._runtime)
                self._require_launching_owner()
            for participant in participants:
                participant.preflight()
                self._require_launching_owner()
            self._runtime.listen_queues(
                tuple(sorted(participant.queue_name for participant in participants))
            )
            self._require_launching_owner()
        except DbosProcessOwnershipError:
            self._fence("DBOS host launch ownership changed; terminate the process")
            raise
        except BaseException:
            self._rollback_launch(participants)
            raise

        try:
            self._require_launching_owner()
            self._runtime.launch()
            self._require_launching_owner()
            for participant in participants:
                participant.register_queue(self._runtime)
                self._require_launching_owner()
        except BaseException:
            self._fence_live_launch(participants)
            raise DbosProcessOwnershipError(
                "DBOS host launch is uncertain after runtime activation; terminate the process"
            ) from None

        with self._state_condition:
            if self._state == "launching":
                self._admission_open = not self._close_requested
                self._state = "running"
                self._lifecycle_thread_id = None
                self._state_condition.notify_all()
                return
        self._fence_live_launch(participants)
        raise DbosProcessOwnershipError(
            "DBOS host launch ownership was fenced; terminate the process"
        )

    def close(self, *, timeout_s: int | None = None) -> None:
        caller_id = threading.get_ident()
        with self._state_condition:
            try:
                if self._state == "closed":
                    return
                grace_s = self.config.shutdown_grace_s if timeout_s is None else timeout_s
                _require_shutdown_grace(grace_s)
                caller_deadline = time.monotonic() + grace_s
                if self._state == "launching" and self._lifecycle_thread_id == caller_id:
                    raise RuntimeError("DBOS runtime host cannot close from a launch callback")
                if self._state == "closing" and self._lifecycle_thread_id == caller_id:
                    raise RuntimeError("DBOS runtime host cannot close from a shutdown callback")
                if self._state == "fenced":
                    raise self._shutdown_error or DbosShutdownTimeout(
                        "DBOS runtime host is fenced; terminate the process"
                    )

                if self._shutdown_owner_id is None:
                    self._claim_shutdown_authority_locked(
                        caller_id=caller_id,
                        grace_s=grace_s,
                        deadline=caller_deadline,
                    )
                if self._state == "launching":
                    self._close_requested = True
                    while self._state == "launching":
                        self._wait_for_state_change(caller_deadline)
                    if self._state == "closed":
                        return
                if self._state == "fenced":
                    raise self._shutdown_error or DbosShutdownTimeout(
                        "DBOS runtime host is fenced; terminate the process"
                    )
                if self._state != "closing":
                    self._start_shutdown_locked()

                while self._state == "closing":
                    self._wait_for_state_change(caller_deadline)
                if self._state == "closed":
                    return
                raise self._shutdown_error or DbosShutdownTimeout(
                    "DBOS runtime host is fenced; terminate the process"
                )
            except BaseException as exc:
                if (
                    self._shutdown_owner_id == caller_id
                    and self._state in {"registering", "launching", "running", "closing"}
                ):
                    self._fence_locked(
                        "DBOS lifecycle owner exceeded the shutdown grace or was interrupted; "
                        "terminate the process"
                    )
                    if isinstance(exc, DbosShutdownTimeout):
                        raise self._shutdown_error or AssertionError("unreachable") from None
                raise

    def __enter__(self) -> DbosRuntimeHost:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    def _close_worker(
        self,
        participants: tuple[_DbosHostParticipant, ...],
        grace_s: int,
        deadline: float,
    ) -> None:
        with self._state_condition:
            if self._state not in {"closing", "fenced"}:
                return
            self._lifecycle_thread_id = threading.get_ident()

        try:
            stopped = _stop_participants(participants)
            self._runtime.destroy(
                workflow_completion_timeout_sec=_dbos_workflow_grace(grace_s),
                deadline=deadline,
            )
            _require_participants_drained(participants, deadline=deadline)
            if not stopped:
                raise DbosShutdownTimeout(
                    "DBOS participant admission shutdown is uncertain; terminate the process"
                )
            for participant in participants:
                participant.mark_closed()
            self._runtime.require_released()
        except BaseException:
            self._fence("DBOS runtime host shutdown is uncertain; terminate the process")
            return

        with self._state_condition:
            if self._state != "closing":
                self._lifecycle_thread_id = None
                self._state_condition.notify_all()
                return
            if time.monotonic() > deadline:
                self._fence_locked(
                    "DBOS lifecycle owner exceeded the shutdown grace; terminate the process"
                )
                return
            release_process_owner(self._owner_token)
            self._state = "closed"
            self._lifecycle_thread_id = None
            self._shutdown_owner_id = None
            self._shutdown_deadline = None
            self._shutdown_grace_s = None
            self._state_condition.notify_all()

    def _ordered_participants(self) -> tuple[_DbosHostParticipant, ...]:
        return tuple(self._participants[key] for key in sorted(self._participants))

    def _require_launching_owner(self) -> None:
        with self._state_lock:
            if self._state != "launching":
                raise DbosProcessOwnershipError(
                    "DBOS host launch ownership was fenced; terminate the process"
                )
        try:
            self._runtime.require_owned()
        except _DbosOwnershipConflict:
            raise DbosProcessOwnershipError(
                "DBOS process-global runtime ownership changed; terminate the process"
            ) from None

    def _claim_shutdown_authority_locked(
        self,
        *,
        caller_id: int,
        grace_s: int,
        deadline: float,
    ) -> None:
        self._shutdown_owner_id = caller_id
        self._shutdown_deadline = deadline
        self._shutdown_grace_s = grace_s
        watchdog = threading.Thread(
            target=self._shutdown_watchdog_worker,
            args=(deadline,),
            name="monoid-dbos-host-deadline",
            daemon=True,
        )
        self._shutdown_watchdog = watchdog
        try:
            watchdog.start()
        except BaseException:
            self._fence_locked("DBOS shutdown watchdog did not start; terminate the process")
            raise self._shutdown_error or AssertionError("unreachable") from None

    def _start_shutdown_locked(self) -> None:
        if self._state not in {"registering", "running"}:
            raise RuntimeError(f"DBOS runtime host cannot close from {self._state}")
        grace_s = self._shutdown_grace_s
        deadline = self._shutdown_deadline
        if grace_s is None or deadline is None:
            self._fence_locked("DBOS shutdown authority is incomplete; terminate the process")
            raise self._shutdown_error or AssertionError("unreachable")
        self._admission_open = False
        self._state = "closing"
        participants = self._ordered_participants()
        worker = threading.Thread(
            target=self._close_worker,
            args=(participants, grace_s, deadline),
            name="monoid-dbos-host-shutdown",
            daemon=True,
        )
        self._shutdown_thread = worker
        try:
            worker.start()
        except BaseException:
            self._fence_locked("DBOS shutdown worker did not start; terminate the process")
            raise self._shutdown_error or AssertionError("unreachable") from None

    def _shutdown_watchdog_worker(self, deadline: float) -> None:
        with self._state_condition:
            while self._shutdown_owner_id is not None and self._state not in {
                "closed",
                "fenced",
            }:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._fence_locked(
                        "DBOS lifecycle owner exceeded the shutdown grace; "
                        "terminate the process"
                    )
                    return
                self._state_condition.wait(timeout=remaining)

    def _wait_for_state_change(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DbosShutdownTimeout(
                "DBOS close caller exceeded its wait; the authoritative shutdown continues"
            )
        self._state_condition.wait(timeout=remaining)

    def _rollback_launch(
        self,
        participants: tuple[_DbosHostParticipant, ...],
    ) -> None:
        stopped = _stop_participants(participants)
        deadline = time.monotonic() + self.config.shutdown_grace_s
        try:
            self._require_launching_owner()
            self._runtime.destroy(
                workflow_completion_timeout_sec=0,
                deadline=deadline,
            )
            _require_participants_drained(participants, deadline=deadline)
            if not stopped:
                raise DbosShutdownTimeout("DBOS participant launch rollback is uncertain")
            for participant in participants:
                participant.mark_closed()
                self._runtime.require_released()
            self._runtime.require_released()
        except BaseException:
            self._fence("DBOS host launch rollback is uncertain; terminate the process")
            raise DbosProcessOwnershipError(
                "DBOS host launch rollback is uncertain; terminate the process"
            ) from None
        with self._state_condition:
            if self._state != "launching":
                raise DbosProcessOwnershipError(
                    "DBOS host launch rollback is uncertain; terminate the process"
                )
            release_process_owner(self._owner_token)
            self._state = "closed"
            self._lifecycle_thread_id = None
            self._state_condition.notify_all()

    def _fence(self, message: str) -> None:
        with self._state_condition:
            self._fence_locked(message)

    def _fence_locked(self, message: str) -> None:
        self._admission_open = False
        self._state = "fenced"
        self._lifecycle_thread_id = None
        if self._shutdown_error is None:
            self._shutdown_error = DbosShutdownTimeout(message)
        self._state_condition.notify_all()

    def _fence_live_launch(
        self,
        participants: tuple[_DbosHostParticipant, ...],
    ) -> None:
        self._fence("DBOS live launch cleanup is uncertain; terminate the process")
        worker = threading.Thread(
            target=self._cleanup_fenced_launch,
            args=(participants,),
            name="monoid-dbos-host-fenced-cleanup",
            daemon=True,
        )
        self._fenced_cleanup_thread = worker
        try:
            worker.start()
        except BaseException:
            return

    def _cleanup_fenced_launch(
        self,
        participants: tuple[_DbosHostParticipant, ...],
    ) -> None:
        _stop_participants(participants)
        try:
            if self._runtime.owns_globals():
                self._runtime.destroy(
                    workflow_completion_timeout_sec=0,
                    deadline=time.monotonic() + self.config.shutdown_grace_s,
                )
        except BaseException:
            return


_PROCESS_OWNER_LOCK = threading.Lock()
_PROCESS_OWNER_TOKEN: object | None = None


def claim_process_owner() -> object:
    """Claim the single process-global DBOS registry before registering workflows."""

    global _PROCESS_OWNER_TOKEN
    with _PROCESS_OWNER_LOCK:
        if _PROCESS_OWNER_TOKEN is not None:
            raise DbosProcessOwnershipError(
                "another DBOS Reference component already owns the process-global runtime"
            )
        token = object()
        _PROCESS_OWNER_TOKEN = token
        return token


def release_process_owner(token: object) -> None:
    """Release a matching process owner token."""

    global _PROCESS_OWNER_TOKEN
    with _PROCESS_OWNER_LOCK:
        if _PROCESS_OWNER_TOKEN is token:
            _PROCESS_OWNER_TOKEN = None


def _require_shutdown_grace(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("DBOS shutdown timeout must be at least two whole seconds")


def _dbos_workflow_grace(shutdown_grace_s: int) -> int:
    """Reserve the final two seconds for DBOS stopped-state proof and host cleanup."""

    return max(0, shutdown_grace_s - 2)


def _stop_participants(participants: tuple[_DbosHostParticipant, ...]) -> bool:
    stopped = True
    for participant in reversed(participants):
        try:
            participant.stop_admission()
        except BaseException:
            stopped = False
    return stopped


def _require_participants_drained(
    participants: tuple[_DbosHostParticipant, ...],
    *,
    deadline: float,
) -> None:
    while True:
        active = False
        for participant in participants:
            try:
                count = participant.active_count()
            except BaseException:
                raise DbosShutdownTimeout(
                    "DBOS participant active-work state is unavailable; terminate the process"
                ) from None
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise DbosShutdownTimeout(
                    "DBOS participant active-work state is invalid; terminate the process"
                )
            active = active or count > 0
        if not active:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DbosShutdownTimeout(
                "DBOS participant work survived the shutdown grace; terminate the process"
            )
        time.sleep(min(0.01, remaining))


def create_owned_runtime(dbos_module: Any, config: Any) -> Any:
    """Construct the legacy standalone runtime after a pre-mutation ownership check."""

    implementation = importlib.import_module("dbos._dbos")
    if (
        getattr(implementation, "_dbos_global_instance", None) is not None
        or getattr(implementation, "_dbos_global_registry", None) is not None
    ):
        raise DbosProcessOwnershipError(
            "an existing DBOS runtime or registry is active; shared host ownership is required"
        )
    return dbos_module.DBOS(
        config={
            "name": config.name,
            "system_database_url": config.system_database_url,
            "application_version": config.application_version,
            "executor_id": config.executor_id,
            "run_admin_server": False,
        }
    )


def load_dbos() -> Any:
    """Load the optional DBOS dependency only after the profile is selected."""

    try:
        return importlib.import_module("dbos")
    except ImportError as exc:
        raise DbosDependencyError(
            "DBOS Reference runtime requires the 'reference-dbos' extra"
        ) from exc
