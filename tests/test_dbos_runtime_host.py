from __future__ import annotations

import importlib
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from monoid_agent_kernel.reference.dbos import (
    DbosProcessOwnershipError,
    DbosShutdownTimeout,
)
from monoid_agent_kernel.reference.dbos import runtime as runtime_module
from monoid_agent_kernel.reference.dbos._compat_226 import (
    _DbosCleanupUncertain,
    _DbosOwnershipConflict,
)
from monoid_agent_kernel.reference.dbos.runtime import (
    DBOS_REFERENCE_WORKFLOW_NAMESPACE,
    DbosHostConfig,
    DbosRuntimeHost,
    _DbosHostParticipant,
)
from monoid_agent_kernel.reference.dbos.control_plane import (
    DBOS_CONTROL_STEP_NAME,
    DBOS_CONTROL_WORKFLOW_NAME,
)
from monoid_agent_kernel.reference.dbos.run_driver import (
    DBOS_RUN_STEP_NAME,
    DBOS_RUN_WORKFLOW_NAME,
)

pytestmark = pytest.mark.serial


class _FakeRuntime:
    def __init__(self, calls: list[object]) -> None:
        self.calls = calls
        self.launch_error: BaseException | None = None
        self.destroy_error: BaseException | None = None
        self.release_error: BaseException | None = None
        self.owned = True
        self.released = False

    def require_owned(self) -> None:
        if not self.owned:
            raise _DbosOwnershipConflict("ownership changed")

    def owns_globals(self) -> bool:
        return self.owned

    def require_released(self) -> None:
        if self.release_error is not None:
            raise self.release_error
        if not self.released:
            raise _DbosCleanupUncertain("globals remain")

    def listen_queues(self, queues: tuple[str, ...]) -> None:
        self.calls.append(("runtime:listen", queues))

    def launch(self) -> None:
        self.calls.append("runtime:launch")
        if self.launch_error is not None:
            raise self.launch_error

    def destroy(
        self,
        *,
        workflow_completion_timeout_sec: int,
        deadline: float,
    ) -> None:
        del deadline
        self.calls.append(
            (
                "runtime:destroy",
                workflow_completion_timeout_sec,
            )
        )
        if self.destroy_error is not None:
            raise self.destroy_error
        self.owned = False
        self.released = True


class _FakeDbosClass:
    destroy_calls: list[tuple[bool, int]] = []

    @classmethod
    def destroy(
        cls,
        *,
        destroy_registry: bool,
        workflow_completion_timeout_sec: int,
    ) -> None:
        cls.destroy_calls.append((destroy_registry, workflow_completion_timeout_sec))


@pytest.fixture(autouse=True)
def _reset_process_owner() -> None:
    with runtime_module._PROCESS_OWNER_LOCK:
        runtime_module._PROCESS_OWNER_TOKEN = None
    _FakeDbosClass.destroy_calls = []
    yield
    with runtime_module._PROCESS_OWNER_LOCK:
        runtime_module._PROCESS_OWNER_TOKEN = None


def _config() -> DbosHostConfig:
    return DbosHostConfig(
        system_database_url="sqlite:///dbos.sqlite",
        application_version="host-test-v1",
        executor_id="stable-test-slot",
        shutdown_grace_s=3,
    )


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    fake: _FakeRuntime,
) -> SimpleNamespace:
    module = SimpleNamespace(DBOS=_FakeDbosClass)
    monkeypatch.setattr(runtime_module, "load_dbos", lambda: module)

    first = True

    def _construct(dbos_module: Any, config: DbosHostConfig) -> _FakeRuntime:
        nonlocal first
        del dbos_module, config
        if first:
            first = False
            return fake
        return _FakeRuntime(fake.calls)

    monkeypatch.setattr(
        runtime_module,
        "_construct_owned_runtime_226",
        _construct,
    )
    return module


def _participant(
    participant_id: str,
    queue_name: str,
    calls: list[object],
    *,
    active: int | Callable[[], int] = 0,
    fail_at: str = "",
    failure: BaseException | None = None,
    host_config: DbosHostConfig | None = None,
) -> _DbosHostParticipant:
    def _call(stage: str) -> None:
        calls.append(f"{participant_id}:{stage}")
        if fail_at == stage:
            raise failure or RuntimeError(f"{stage} failed")

    def _register(runtime: Any) -> None:
        assert isinstance(runtime, _FakeRuntime)
        _call("register")

    def _active() -> int:
        _call("active")
        return active() if callable(active) else active

    return _DbosHostParticipant(
        participant_id=participant_id,
        queue_name=queue_name,
        host_config=host_config or _config(),
        register_workflows=lambda runtime: _call("workflows"),
        preflight=lambda: _call("preflight"),
        register_queue=_register,
        stop_admission=lambda: _call("stop"),
        active_count=_active,
        mark_closed=lambda: _call("closed"),
    )


def test_host_models_import_without_loading_optional_dbos() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    code = """
import sys
from monoid_agent_kernel.reference.dbos.runtime import DbosHostConfig, DbosRuntimeHost
assert DbosHostConfig and DbosRuntimeHost
assert 'dbos' not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_host_config_owns_stable_workflow_namespace() -> None:
    config = _config()

    assert config.workflow_namespace == DBOS_REFERENCE_WORKFLOW_NAMESPACE
    assert config.workflow_name("control", "workflow") == DBOS_CONTROL_WORKFLOW_NAME
    assert config.workflow_name("control", "step") == DBOS_CONTROL_STEP_NAME
    assert config.workflow_name("run", "workflow") == DBOS_RUN_WORKFLOW_NAME
    assert config.workflow_name("run", "step") == DBOS_RUN_STEP_NAME


@pytest.mark.parametrize(
    ("surface", "operation"),
    [("control-plane", "step"), ("control", "dispatch-step"), ("Control", "step")],
)
def test_host_config_rejects_ambiguous_workflow_tokens(
    surface: str,
    operation: str,
) -> None:
    with pytest.raises(ValueError, match="safe tokens"):
        _config().workflow_name(surface, operation)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("system_database_url", ""),
        ("name", ""),
        ("application_version", ""),
        ("executor_id", ""),
        ("shutdown_grace_s", 0),
        ("shutdown_grace_s", 1),
        ("shutdown_grace_s", True),
    ],
)
def test_host_config_rejects_invalid_process_identity(field: str, value: object) -> None:
    values: dict[str, object] = {
        "system_database_url": "sqlite:///dbos.sqlite",
        "name": "host-test",
        "application_version": "host-test-v1",
        "executor_id": "stable-test-slot",
        "shutdown_grace_s": 3,
    }
    values[field] = value

    with pytest.raises(ValueError):
        DbosHostConfig(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("environment_name", ["DBOS__CLOUD", "DBOS__CONDUCTOR_KEY"])
def test_host_rejects_cloud_or_conductor_mode_before_constructing_dbos(
    monkeypatch: pytest.MonkeyPatch,
    environment_name: str,
) -> None:
    module = SimpleNamespace(DBOS=_FakeDbosClass)
    monkeypatch.setattr(runtime_module, "load_dbos", lambda: module)
    monkeypatch.setenv(environment_name, "true" if environment_name == "DBOS__CLOUD" else "key")

    with pytest.raises(DbosProcessOwnershipError, match="self-hosted mode"):
        DbosRuntimeHost(_config())

    with runtime_module._PROCESS_OWNER_LOCK:
        assert runtime_module._PROCESS_OWNER_TOKEN is None


def test_host_launches_and_closes_all_participants_once_in_deterministic_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    host._register_participant(_participant("run", "queue-run", calls))
    host._register_participant(_participant("control", "queue-control", calls))

    host.launch()
    host.launch()

    assert host.state == "running"
    assert host.accepting is True
    assert calls == [
        "control:workflows",
        "run:workflows",
        "control:preflight",
        "run:preflight",
        ("runtime:listen", ("queue-control", "queue-run")),
        "runtime:launch",
        "control:register",
        "run:register",
    ]

    host.close()
    host.close()

    assert host.state == "closed"
    assert host.accepting is False
    assert calls[8:] == [
        "run:stop",
        "control:stop",
        ("runtime:destroy", 1),
        "control:active",
        "run:active",
        "control:closed",
        "run:closed",
    ]


def test_host_rejects_duplicate_and_late_participant_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    host._register_participant(_participant("control", "queue-control", calls))

    with pytest.raises(ValueError, match="ids must be unique"):
        host._register_participant(_participant("control", "queue-other", calls))
    with pytest.raises(ValueError, match="queues must be unique"):
        host._register_participant(_participant("run", "queue-control", calls))

    host.launch()
    with pytest.raises(RuntimeError, match="before host launch"):
        host._register_participant(_participant("run", "queue-run", calls))
    host.close()


def test_host_rejects_participant_from_a_different_process_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    mismatched = DbosHostConfig(
        system_database_url="sqlite:///dbos.sqlite",
        application_version="host-test-v1",
        executor_id="different-stable-slot",
        shutdown_grace_s=3,
    )

    with pytest.raises(ValueError, match="process identity does not match"):
        host._register_participant(
            _participant("control", "queue-control", calls, host_config=mismatched)
        )

    host.close()


def test_host_opens_global_admission_only_after_every_queue_is_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    observed: list[bool] = []

    host._register_participant(
        _DbosHostParticipant(
            participant_id="control",
            queue_name="queue-control",
            host_config=_config(),
            register_workflows=lambda runtime: None,
            preflight=lambda: None,
            register_queue=lambda runtime: observed.append(host.accepting),
            stop_admission=lambda: None,
            active_count=lambda: 0,
            mark_closed=lambda: None,
        )
    )

    host.launch()

    assert observed == [False]
    assert host.accepting is True
    host.close()


def test_one_host_owns_the_process_until_verified_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    first = DbosRuntimeHost(_config())

    with pytest.raises(DbosProcessOwnershipError):
        DbosRuntimeHost(_config())

    first.close()
    replacement = DbosRuntimeHost(_config())
    replacement.close()


@pytest.mark.parametrize("stage", ["workflows", "preflight"])
def test_prelaunch_failure_rolls_back_registry_and_releases_process_owner(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    host._register_participant(_participant("control", "queue-control", calls, fail_at=stage))

    with pytest.raises(RuntimeError, match=rf"{stage} failed"):
        host.launch()

    assert host.state == "closed"
    assert ("runtime:destroy", 0) in calls
    replacement = DbosRuntimeHost(_config())
    replacement.close()


def test_failure_after_runtime_activation_fences_process_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    host._register_participant(_participant("control", "queue-control", calls, fail_at="register"))

    with pytest.raises(DbosProcessOwnershipError, match="runtime activation"):
        host.launch()

    assert host.state == "fenced"
    assert host.accepting is False
    assert host._fenced_cleanup_thread is not None
    host._fenced_cleanup_thread.join(timeout=3)
    assert host._fenced_cleanup_thread.is_alive() is False
    assert ("runtime:destroy", 0) in calls
    with pytest.raises(DbosProcessOwnershipError):
        DbosRuntimeHost(_config())


def test_live_launch_failure_fences_before_best_effort_cleanup_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    fake.launch_error = RuntimeError("launch failed")
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    stop_entered = threading.Event()
    release_stop = threading.Event()
    participant = _participant("control", "queue-control", calls)

    def _stop() -> None:
        stop_entered.set()
        assert release_stop.wait(timeout=3)

    host._register_participant(
        _DbosHostParticipant(
            participant_id=participant.participant_id,
            queue_name=participant.queue_name,
            host_config=participant.host_config,
            register_workflows=participant.register_workflows,
            preflight=participant.preflight,
            register_queue=participant.register_queue,
            stop_admission=_stop,
            active_count=participant.active_count,
            mark_closed=participant.mark_closed,
        )
    )
    try:
        with pytest.raises(DbosProcessOwnershipError, match="runtime activation"):
            host.launch()

        assert host.state == "fenced"
        assert host.accepting is False
        assert stop_entered.wait(timeout=1)
        assert host._fenced_cleanup_thread is not None
        assert host._fenced_cleanup_thread.is_alive()
    finally:
        release_stop.set()
        assert host._fenced_cleanup_thread is not None
        host._fenced_cleanup_thread.join(timeout=3)


@pytest.mark.parametrize("failure", ["destroy", "participant"])
def test_uncertain_launch_rollback_fences_process_ownership(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    if failure == "destroy":
        fake.destroy_error = RuntimeError("private DBOS cleanup detail")
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    host._register_participant(
        _participant(
            "control",
            "queue-control",
            calls,
            active=(1 if failure == "participant" else 0),
            fail_at="preflight",
        )
    )

    with pytest.raises(DbosProcessOwnershipError, match="rollback is uncertain") as raised:
        host.launch()

    assert "private DBOS" not in str(raised.value)
    assert host.state == "fenced"
    with pytest.raises(DbosProcessOwnershipError):
        DbosRuntimeHost(_config())


def test_prelaunch_base_exception_rolls_back_and_preserves_the_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    host._register_participant(
        _participant(
            "control",
            "queue-control",
            calls,
            fail_at="preflight",
            failure=KeyboardInterrupt(),
        )
    )

    with pytest.raises(KeyboardInterrupt):
        host.launch()

    assert host.state == "closed"
    replacement = DbosRuntimeHost(_config())
    replacement.close()


def test_live_base_exception_is_replaced_with_a_static_ownership_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    fake.launch_error = KeyboardInterrupt()
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    host._register_participant(_participant("control", "queue-control", calls))

    with pytest.raises(DbosProcessOwnershipError, match="runtime activation"):
        host.launch()

    assert host.state == "fenced"
    with pytest.raises(DbosProcessOwnershipError):
        DbosRuntimeHost(_config())


def test_close_racing_launch_waits_and_completes_one_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    preflight_entered = threading.Event()
    release_preflight = threading.Event()
    failures: list[BaseException] = []

    def _preflight() -> None:
        preflight_entered.set()
        assert release_preflight.wait(timeout=3)

    host._register_participant(
        _DbosHostParticipant(
            participant_id="control",
            queue_name="queue-control",
            host_config=_config(),
            register_workflows=lambda runtime: None,
            preflight=_preflight,
            register_queue=lambda runtime: calls.append("control:register"),
            stop_admission=lambda: calls.append("control:stop"),
            active_count=lambda: 0,
            mark_closed=lambda: calls.append("control:closed"),
        )
    )

    def _launch() -> None:
        try:
            host.launch()
        except BaseException as exc:
            failures.append(exc)

    def _close() -> None:
        try:
            host.close()
        except BaseException as exc:
            failures.append(exc)

    launch_thread = threading.Thread(target=_launch)
    close_thread = threading.Thread(target=_close)
    launch_thread.start()
    assert preflight_entered.wait(timeout=3)
    close_thread.start()
    deadline = time.monotonic() + 3
    while not host._close_requested and time.monotonic() < deadline:
        time.sleep(0.01)
    assert host._close_requested is True
    assert close_thread.is_alive()

    release_preflight.set()
    launch_thread.join(timeout=3)
    close_thread.join(timeout=3)

    assert launch_thread.is_alive() is False
    assert close_thread.is_alive() is False
    assert failures == []
    assert host.state == "closed"
    assert host.accepting is False
    assert calls.count("runtime:launch") == 1
    assert calls.count(("runtime:destroy", 1)) == 1


def test_close_fences_when_launch_does_not_yield_within_shutdown_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    preflight_entered = threading.Event()
    release_preflight = threading.Event()
    launch_failures: list[BaseException] = []

    def _preflight() -> None:
        preflight_entered.set()
        release_preflight.wait(timeout=3)

    participant = _participant("control", "queue-control", calls)
    host._register_participant(
        _DbosHostParticipant(
            participant_id=participant.participant_id,
            queue_name=participant.queue_name,
            host_config=participant.host_config,
            register_workflows=participant.register_workflows,
            preflight=_preflight,
            register_queue=participant.register_queue,
            stop_admission=participant.stop_admission,
            active_count=participant.active_count,
            mark_closed=participant.mark_closed,
        )
    )

    def _launch() -> None:
        try:
            host.launch()
        except BaseException as exc:
            launch_failures.append(exc)

    launch_thread = threading.Thread(target=_launch)
    launch_thread.start()
    assert preflight_entered.wait(timeout=3)

    with pytest.raises(DbosShutdownTimeout, match="lifecycle owner exceeded"):
        host.close(timeout_s=2)

    assert host.state == "fenced"
    assert host.accepting is False
    release_preflight.set()
    launch_thread.join(timeout=3)
    assert launch_thread.is_alive() is False
    assert len(launch_failures) == 1
    assert isinstance(launch_failures[0], DbosProcessOwnershipError)
    assert "runtime:launch" not in calls


def test_concurrent_close_waits_for_the_single_shutdown_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    destroy_entered = threading.Event()
    release_destroy = threading.Event()
    original_destroy = fake.destroy
    failures: list[BaseException] = []

    def _destroy(*, workflow_completion_timeout_sec: int, deadline: float) -> None:
        destroy_entered.set()
        assert release_destroy.wait(timeout=3)
        original_destroy(
            workflow_completion_timeout_sec=workflow_completion_timeout_sec,
            deadline=deadline,
        )

    fake.destroy = _destroy  # type: ignore[method-assign]

    def _close() -> None:
        try:
            host.close()
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(target=_close)
    second = threading.Thread(target=_close)
    first.start()
    assert destroy_entered.wait(timeout=3)
    second.start()
    assert second.is_alive()
    release_destroy.set()
    first.join(timeout=3)
    second.join(timeout=3)

    assert failures == []
    assert host.state == "closed"
    assert calls.count(("runtime:destroy", 1)) == 1


def test_joining_close_times_out_without_revoking_shutdown_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    destroy_entered = threading.Event()
    release_destroy = threading.Event()
    first_failures: list[BaseException] = []
    original_destroy = fake.destroy

    def _destroy(*, workflow_completion_timeout_sec: int, deadline: float) -> None:
        destroy_entered.set()
        release_destroy.wait(timeout=3)
        original_destroy(
            workflow_completion_timeout_sec=workflow_completion_timeout_sec,
            deadline=deadline,
        )

    fake.destroy = _destroy  # type: ignore[method-assign]

    def _first_close() -> None:
        try:
            host.close()
        except BaseException as exc:
            first_failures.append(exc)

    first = threading.Thread(target=_first_close)
    first.start()
    assert destroy_entered.wait(timeout=3)

    with pytest.raises(DbosShutdownTimeout, match="close caller exceeded its wait"):
        host.close(timeout_s=2)

    assert host.state == "closing"
    assert first.is_alive()
    release_destroy.set()
    first.join(timeout=3)
    assert first.is_alive() is False
    assert first_failures == []
    assert host.state == "closed"
    assert calls.count(("runtime:destroy", 1)) == 1


def test_close_interruption_while_launching_fences_the_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    preflight_entered = threading.Event()
    release_preflight = threading.Event()
    launch_failures: list[BaseException] = []

    def _preflight() -> None:
        preflight_entered.set()
        assert release_preflight.wait(timeout=3)

    participant = _participant("control", "queue-control", calls)
    host._register_participant(
        _DbosHostParticipant(
            participant_id=participant.participant_id,
            queue_name=participant.queue_name,
            host_config=participant.host_config,
            register_workflows=participant.register_workflows,
            preflight=_preflight,
            register_queue=participant.register_queue,
            stop_admission=participant.stop_admission,
            active_count=participant.active_count,
            mark_closed=participant.mark_closed,
        )
    )

    def _launch() -> None:
        try:
            host.launch()
        except BaseException as exc:
            launch_failures.append(exc)

    def _interrupt(deadline: float) -> None:
        del deadline
        raise KeyboardInterrupt()

    launch_thread = threading.Thread(target=_launch)
    launch_thread.start()
    assert preflight_entered.wait(timeout=3)
    monkeypatch.setattr(host, "_wait_for_state_change", _interrupt)

    with pytest.raises(KeyboardInterrupt):
        host.close()

    assert host.state == "fenced"
    assert host.accepting is False
    release_preflight.set()
    launch_thread.join(timeout=3)
    assert launch_thread.is_alive() is False
    assert len(launch_failures) == 1
    assert isinstance(launch_failures[0], DbosProcessOwnershipError)


def test_close_interruption_before_launch_fences_the_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())

    def _interrupt() -> None:
        raise KeyboardInterrupt()

    monkeypatch.setattr(host, "_start_shutdown_locked", _interrupt)

    with pytest.raises(KeyboardInterrupt):
        host.close()

    assert host.state == "fenced"
    assert host.accepting is False
    with pytest.raises(DbosProcessOwnershipError):
        DbosRuntimeHost(_config())


def test_close_interruption_while_shutdown_waits_fences_the_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    destroy_entered = threading.Event()
    release_destroy = threading.Event()
    original_destroy = fake.destroy

    def _destroy(*, workflow_completion_timeout_sec: int, deadline: float) -> None:
        destroy_entered.set()
        release_destroy.wait(timeout=3)
        original_destroy(
            workflow_completion_timeout_sec=workflow_completion_timeout_sec,
            deadline=deadline,
        )

    def _interrupt(deadline: float) -> None:
        del deadline
        assert host._state_condition.wait_for(destroy_entered.is_set, timeout=1)
        raise KeyboardInterrupt()

    fake.destroy = _destroy  # type: ignore[method-assign]
    host._register_participant(_participant("control", "queue-control", calls))
    host.launch()
    monkeypatch.setattr(host, "_wait_for_state_change", _interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            host.close()

        assert host.state == "fenced"
        assert host.accepting is False
    finally:
        release_destroy.set()
        assert host._shutdown_thread is not None
        host._shutdown_thread.join(timeout=3)


def test_initiating_close_is_bounded_when_runtime_destroy_never_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    destroy_entered = threading.Event()
    release_destroy = threading.Event()
    original_destroy = fake.destroy

    def _destroy(*, workflow_completion_timeout_sec: int, deadline: float) -> None:
        destroy_entered.set()
        release_destroy.wait(timeout=3)
        original_destroy(
            workflow_completion_timeout_sec=workflow_completion_timeout_sec,
            deadline=deadline,
        )

    fake.destroy = _destroy  # type: ignore[method-assign]

    def _wait_without_a_caller_timer(deadline: float) -> None:
        del deadline
        host._state_condition.wait()

    monkeypatch.setattr(host, "_wait_for_state_change", _wait_without_a_caller_timer)
    started = time.monotonic()
    try:
        with pytest.raises(DbosShutdownTimeout, match="lifecycle owner exceeded"):
            host.close(timeout_s=2)

        assert time.monotonic() - started < 3
        assert destroy_entered.is_set()
        assert host.state == "fenced"
    finally:
        release_destroy.set()
        assert host._shutdown_thread is not None
        host._shutdown_thread.join(timeout=3)


def test_cleanup_probe_cannot_block_the_shutdown_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    probe_entered = threading.Event()
    release_probe = threading.Event()
    original_require_released = fake.require_released

    def _require_released() -> None:
        probe_entered.set()
        release_probe.wait(timeout=3)
        original_require_released()

    fake.require_released = _require_released  # type: ignore[method-assign]
    host._register_participant(_participant("control", "queue-control", calls))
    host.launch()
    started = time.monotonic()
    try:
        with pytest.raises(DbosShutdownTimeout, match="lifecycle owner exceeded"):
            host.close(timeout_s=2)

        assert time.monotonic() - started < 3
        assert probe_entered.is_set()
        assert host.state == "fenced"
    finally:
        release_probe.set()
        assert host._shutdown_thread is not None
        host._shutdown_thread.join(timeout=3)


def test_minimum_shutdown_grace_reserves_time_for_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    host._register_participant(_participant("control", "queue-control", calls))
    host.launch()

    with pytest.raises(ValueError, match="at least two"):
        host.close(timeout_s=1)
    assert host.state == "running"

    host.close(timeout_s=2)

    assert host.state == "closed"
    assert calls.count(("runtime:destroy", 0)) == 1


def test_participant_work_may_drain_within_the_shared_shutdown_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    samples = 0

    def _active() -> int:
        nonlocal samples
        samples += 1
        return 1 if samples < 3 else 0

    host = DbosRuntimeHost(_config())
    host._register_participant(_participant("control", "queue-control", calls, active=_active))
    host.launch()

    host.close(timeout_s=2)

    assert samples == 3
    assert host.state == "closed"


@pytest.mark.parametrize("failure", ["participant", "destroy", "stop", "released"])
def test_uncertain_shutdown_stops_admission_and_fences_process_ownership(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    if failure == "destroy":
        fake.destroy_error = RuntimeError("private DBOS cleanup detail")
    if failure == "released":
        fake.release_error = _DbosCleanupUncertain("foreign DBOS detail")
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    host._register_participant(
        _participant(
            "control",
            "queue-control",
            calls,
            active=(1 if failure == "participant" else 0),
            fail_at=("stop" if failure == "stop" else ""),
        )
    )
    host.launch()

    with pytest.raises(DbosShutdownTimeout) as raised:
        host.close(timeout_s=2)

    assert "private DBOS" not in str(raised.value)
    assert "foreign DBOS" not in str(raised.value)
    assert host.state == "fenced"
    assert host.accepting is False
    assert ("runtime:destroy", 0) in calls
    with pytest.raises(DbosProcessOwnershipError):
        DbosRuntimeHost(_config())


def test_external_runtime_conflict_releases_only_the_reference_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    module = _install_fake_runtime(monkeypatch, fake)
    attempts = 0

    def _create(dbos_module: Any, config: DbosHostConfig) -> _FakeRuntime:
        nonlocal attempts
        del dbos_module, config
        attempts += 1
        if attempts == 1:
            raise _DbosOwnershipConflict("external runtime")
        return _FakeRuntime(calls)

    monkeypatch.setattr(runtime_module, "_construct_owned_runtime_226", _create)

    with pytest.raises(DbosProcessOwnershipError, match="external runtime"):
        DbosRuntimeHost(_config())

    assert module.DBOS.destroy_calls == []
    replacement = DbosRuntimeHost(_config())
    replacement.close()


def test_launch_fences_before_touching_a_replaced_process_global_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    host._register_participant(_participant("control", "queue-control", calls))
    fake.owned = False

    with pytest.raises(DbosProcessOwnershipError, match="ownership changed"):
        host.launch()

    assert host.state == "fenced"
    assert calls == []


def test_constructor_failure_fences_without_destroying_an_unknown_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    attempts = 0

    def _create(dbos_module: Any, config: DbosHostConfig) -> _FakeRuntime:
        nonlocal attempts
        del dbos_module, config
        attempts += 1
        if attempts == 1:
            raise RuntimeError("construction failed")
        return fake

    monkeypatch.setattr(runtime_module, "_construct_owned_runtime_226", _create)

    with pytest.raises(DbosProcessOwnershipError, match="construction is uncertain"):
        DbosRuntimeHost(_config())

    assert _FakeDbosClass.destroy_calls == []
    with pytest.raises(DbosProcessOwnershipError):
        DbosRuntimeHost(_config())


_REAL_DBOS_SUBPROCESS_CASE = "MONOID_TEST_REAL_DBOS_HOST_CASE"


def _delegate_real_dbos_case(test_name: str, case: str) -> bool:
    if os.environ.get(_REAL_DBOS_SUBPROCESS_CASE) == case:
        return False
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env[_REAL_DBOS_SUBPROCESS_CASE] = case
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"tests/test_dbos_runtime_host.py::{test_name}",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return True


def test_real_dbos_host_launches_queue_work_and_resets_global_state(
    tmp_path: Path,
) -> None:
    dbos = pytest.importorskip("dbos")
    if _delegate_real_dbos_case(
        "test_real_dbos_host_launches_queue_work_and_resets_global_state",
        "launch-close",
    ):
        return
    implementation = importlib.import_module("dbos._dbos")
    dbos.DBOS.destroy(destroy_registry=True, workflow_completion_timeout_sec=0)
    host: DbosRuntimeHost | None = None
    closed = threading.Event()
    try:
        config = DbosHostConfig(
            system_database_url=f"sqlite:///{tmp_path / 'host.sqlite'}",
            name="monoid-reference-host-smoke",
            application_version="monoid-reference-host-smoke-v1",
            executor_id="stable-host-smoke-slot",
            shutdown_grace_s=30,
        )
        host = DbosRuntimeHost(config)
        surface: dict[str, Any] = {}

        def _register_workflows(runtime: Any) -> None:
            @runtime.step(name=host.workflow_name("host", "step"), retries_allowed=False)
            def add_one(value: int) -> int:
                return value + 1

            @runtime.workflow(name=host.workflow_name("host", "workflow"))
            def host_workflow(value: int) -> int:
                return add_one(value)

            surface["runtime"] = runtime
            surface["workflow"] = host_workflow

        queue_name = "monoid-reference-host-smoke-v1"
        host._register_participant(
            _DbosHostParticipant(
                participant_id="host-smoke",
                queue_name=queue_name,
                host_config=config,
                register_workflows=_register_workflows,
                preflight=lambda: None,
                register_queue=lambda registrar: registrar.register_queue(
                    queue_name,
                    worker_concurrency=1,
                    concurrency=1,
                    polling_interval_sec=0.01,
                    on_conflict="always_update",
                ),
                stop_admission=lambda: None,
                active_count=lambda: 0,
                mark_closed=closed.set,
            )
        )

        host.launch()
        handle = surface["runtime"].enqueue_workflow(queue_name, surface["workflow"], 41)

        assert handle.get_result() == 42
        host.close()
        assert host.state == "closed"
        assert closed.is_set()
        assert getattr(implementation, "_dbos_global_instance", None) is None
        assert getattr(implementation, "_dbos_global_registry", None) is None
    finally:
        if host is not None and host.state not in {"closed", "fenced"}:
            host.close()
        dbos.DBOS.destroy(destroy_registry=True, workflow_completion_timeout_sec=0)


def test_real_dbos_idle_host_closes_with_the_minimum_grace(tmp_path: Path) -> None:
    dbos = pytest.importorskip("dbos")
    if _delegate_real_dbos_case(
        "test_real_dbos_idle_host_closes_with_the_minimum_grace",
        "minimum-close",
    ):
        return
    implementation = importlib.import_module("dbos._dbos")
    dbos.DBOS.destroy(destroy_registry=True, workflow_completion_timeout_sec=0)
    host: DbosRuntimeHost | None = None
    closed = threading.Event()
    try:
        config = DbosHostConfig(
            system_database_url=f"sqlite:///{tmp_path / 'minimum-close.sqlite'}",
            name="monoid-host-min-close",
            application_version="monoid-host-min-close-v1",
            executor_id="stable-min-close-slot",
            shutdown_grace_s=2,
        )
        host = DbosRuntimeHost(config)
        host._register_participant(
            _DbosHostParticipant(
                participant_id="minimum-close",
                queue_name="monoid-reference-host-minimum-close-v1",
                host_config=config,
                register_workflows=lambda runtime: None,
                preflight=lambda: None,
                register_queue=lambda runtime: None,
                stop_admission=lambda: None,
                active_count=lambda: 0,
                mark_closed=closed.set,
            )
        )

        host.launch()
        host.close()

        assert host.state == "closed"
        assert closed.is_set()
        assert getattr(implementation, "_dbos_global_instance", None) is None
        assert getattr(implementation, "_dbos_global_registry", None) is None
    finally:
        if host is not None and host.state not in {"closed", "fenced"}:
            host.close()
        dbos.DBOS.destroy(destroy_registry=True, workflow_completion_timeout_sec=0)


def test_real_dbos_external_registry_is_rejected_without_erasure() -> None:
    dbos = pytest.importorskip("dbos")
    if _delegate_real_dbos_case(
        "test_real_dbos_external_registry_is_rejected_without_erasure",
        "external-registry",
    ):
        return
    implementation = importlib.import_module("dbos._dbos")
    dbos.DBOS.destroy(destroy_registry=True, workflow_completion_timeout_sec=0)
    try:

        @dbos.DBOS.workflow(name="external.registry.workflow.v1")
        def external_workflow() -> str:
            return "external"

        del external_workflow
        external_registry = getattr(implementation, "_dbos_global_registry", None)
        assert external_registry is not None

        with pytest.raises(DbosProcessOwnershipError, match="runtime or registry is active"):
            DbosRuntimeHost(_config())

        assert getattr(implementation, "_dbos_global_instance", None) is None
        assert getattr(implementation, "_dbos_global_registry", None) is external_registry
    finally:
        dbos.DBOS.destroy(destroy_registry=True, workflow_completion_timeout_sec=0)


def test_real_dbos_replacement_is_not_launched_or_destroyed_by_the_host(
    tmp_path: Path,
) -> None:
    dbos = pytest.importorskip("dbos")
    if _delegate_real_dbos_case(
        "test_real_dbos_replacement_is_not_launched_or_destroyed_by_the_host",
        "replacement",
    ):
        return
    implementation = importlib.import_module("dbos._dbos")
    dbos.DBOS.destroy(destroy_registry=True, workflow_completion_timeout_sec=0)
    try:
        config = DbosHostConfig(
            system_database_url=f"sqlite:///{tmp_path / 'host.sqlite'}",
            application_version="host-replacement-test-v1",
            executor_id="stable-host-slot",
        )
        host = DbosRuntimeHost(config)
        dbos.DBOS.destroy(destroy_registry=True, workflow_completion_timeout_sec=0)
        foreign = dbos.DBOS(
            config={
                "name": "foreign-dbos-runtime",
                "system_database_url": f"sqlite:///{tmp_path / 'foreign.sqlite'}",
                "application_version": "foreign-runtime-v1",
                "executor_id": "foreign-slot",
                "run_admin_server": False,
            }
        )
        foreign_registry = getattr(implementation, "_dbos_global_registry", None)
        host._register_participant(_participant("control", "queue-control", [], host_config=config))

        with pytest.raises(DbosProcessOwnershipError, match="ownership changed"):
            host.launch()

        assert host.state == "fenced"
        assert foreign._launched is False
        assert foreign._listening_queues is None
        assert getattr(implementation, "_dbos_global_instance", None) is foreign
        assert getattr(implementation, "_dbos_global_registry", None) is foreign_registry
    finally:
        dbos.DBOS.destroy(destroy_registry=True, workflow_completion_timeout_sec=0)
