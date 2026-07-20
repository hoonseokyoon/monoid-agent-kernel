from __future__ import annotations

import asyncio
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

from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore
from monoid_agent_kernel.core.control import ControlCommand, ControlResult
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
    _require_shared_host_config,
)
from monoid_agent_kernel.reference.dbos.control_plane import (
    DBOS_CONTROL_STEP_NAME,
    DBOS_CONTROL_WORKFLOW_NAME,
    DbosControlConfig,
    DbosControlEnvelope,
    _register_hosted_control_plane,
)
from monoid_agent_kernel.reference.dbos.run_driver import (
    DBOS_RUN_STEP_NAME,
    DBOS_RUN_WORKFLOW_NAME,
    DbosResumeCommand,
    DbosRunConfig,
    _register_hosted_run_driver,
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

    def preflight_launch(self) -> None:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if running_loop is not None:
            raise _DbosOwnershipConflict(
                "DBOS Reference launch requires a dedicated synchronous lifecycle thread"
            )

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
        admission_count=lambda: 0,
        active_count=_active,
        mark_closed=lambda: _call("closed"),
    )


def test_host_models_import_without_loading_optional_dbos() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    code = """
import sys
from monoid_agent_kernel.reference.dbos.control_plane import DbosControlConfig
from monoid_agent_kernel.reference.dbos.runtime import DbosHostConfig, DbosRuntimeHost
from monoid_agent_kernel.reference.dbos.run_driver import DbosRunConfig
control = DbosControlConfig(system_database_url='sqlite:///dbos.sqlite', executor_id='slot')
run = DbosRunConfig(system_database_url='sqlite:///dbos.sqlite', executor_id='slot')
assert control._host_config() == run._host_config() == DbosHostConfig(
    system_database_url='sqlite:///dbos.sqlite', executor_id='slot'
)
assert DbosRuntimeHost
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


def _aligned_surface_configs() -> tuple[DbosControlConfig, DbosRunConfig]:
    control = DbosControlConfig(
        system_database_url="sqlite:///shared.sqlite",
        name="shared-host",
        application_version="shared-v2",
        executor_id="shared-slot",
        queue_name="control-local-queue",
        polling_interval_s=0.01,
        shutdown_grace_s=3,
    )
    run = DbosRunConfig(
        system_database_url="sqlite:///shared.sqlite",
        name="shared-host",
        application_version="shared-v2",
        executor_id="shared-slot",
        queue_name="run-local-queue",
        polling_interval_s=0.2,
        checkpoint_retry_interval_s=0.3,
        shutdown_grace_s=3,
        local_task_wait_s=17,
    )
    return control, run


def test_surface_configs_project_one_explicit_shared_host_configuration() -> None:
    control, run = _aligned_surface_configs()

    shared = _require_shared_host_config(control._host_config(), run._host_config())

    assert shared == DbosHostConfig(
        system_database_url="sqlite:///shared.sqlite",
        name="shared-host",
        application_version="shared-v2",
        executor_id="shared-slot",
        shutdown_grace_s=3,
    )


def test_shared_host_config_requires_typed_participants() -> None:
    with pytest.raises(ValueError, match="at least one participant"):
        _require_shared_host_config()
    with pytest.raises(TypeError, match="typed host configurations"):
        _require_shared_host_config(_config(), object())  # type: ignore[arg-type]


@pytest.mark.parametrize("surface", ["control", "run"])
@pytest.mark.parametrize(
    ("field", "different"),
    [
        ("system_database_url", "sqlite:///different.sqlite"),
        ("name", "different-host"),
        ("application_version", "different-v2"),
        ("executor_id", "different-slot"),
        ("shutdown_grace_s", 4),
    ],
)
def test_shared_host_config_rejects_each_surface_process_mismatch(
    surface: str,
    field: str,
    different: object,
) -> None:
    control, run = _aligned_surface_configs()
    values: dict[str, object] = {
        "system_database_url": "sqlite:///shared.sqlite",
        "name": "shared-host",
        "application_version": "shared-v2",
        "executor_id": "shared-slot",
        "shutdown_grace_s": 3,
    }
    values[field] = different
    if surface == "control":
        mismatched = DbosControlConfig(**values)._host_config()  # type: ignore[arg-type]
        matching = run._host_config()
    else:
        mismatched = DbosRunConfig(**values)._host_config()  # type: ignore[arg-type]
        matching = control._host_config()

    with pytest.raises(ValueError, match="host configurations do not match"):
        _require_shared_host_config(matching, mismatched)


def test_shared_host_config_preserves_recovery_defaults_as_explicit_mismatch() -> None:
    database_url = "sqlite:///defaults.sqlite"
    host = DbosHostConfig(system_database_url=database_url)
    run = DbosRunConfig(system_database_url=database_url)
    control = DbosControlConfig(system_database_url=database_url)

    assert run._host_config() == host
    with pytest.raises(ValueError, match="host configurations do not match"):
        _require_shared_host_config(host, control._host_config())


@pytest.mark.parametrize("surface", ["control", "run"])
def test_host_projection_tightens_identity_types_without_changing_surface_validation(
    surface: str,
) -> None:
    config_type = DbosControlConfig if surface == "control" else DbosRunConfig
    config = config_type(  # type: ignore[call-arg]
        system_database_url="sqlite:///types.sqlite",
        name=object(),
    )

    with pytest.raises(ValueError, match="host name is required"):
        config._host_config()


@pytest.mark.parametrize("surface", ["control", "run"])
def test_host_projection_preserves_standalone_one_second_grace(surface: str) -> None:
    config_type = DbosControlConfig if surface == "control" else DbosRunConfig
    config = config_type(  # type: ignore[call-arg]
        system_database_url="sqlite:///grace.sqlite",
        shutdown_grace_s=1,
    )

    assert config.shutdown_grace_s == 1
    with pytest.raises(ValueError, match="at least two whole seconds"):
        config._host_config()


def test_shared_host_config_mismatch_does_not_disclose_database_urls() -> None:
    first = DbosHostConfig(system_database_url="postgresql://secret-a@host/database")
    second = DbosHostConfig(system_database_url="postgresql://secret-b@host/database")

    with pytest.raises(ValueError, match="host configurations do not match") as raised:
        _require_shared_host_config(first, second)

    assert "secret-a" not in str(raised.value)
    assert "secret-b" not in str(raised.value)


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
        ("runtime:destroy", 0),
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
            admission_count=lambda: 0,
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
            admission_count=participant.admission_count,
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


def test_async_launch_rejection_rolls_back_and_releases_process_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    host = DbosRuntimeHost(_config())
    host._register_participant(_participant("control", "queue-control", calls))

    async def _attempt_launch() -> None:
        with pytest.raises(DbosProcessOwnershipError, match="synchronous lifecycle thread"):
            host.launch()

    asyncio.run(_attempt_launch())

    assert host.state == "closed"
    assert host.accepting is False
    assert "runtime:launch" not in calls
    assert ("runtime:destroy", 0) in calls
    replacement = DbosRuntimeHost(_config())
    replacement.close()


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
            admission_count=lambda: 0,
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
    assert calls.count(("runtime:destroy", 0)) == 1


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
            admission_count=participant.admission_count,
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
    assert calls.count(("runtime:destroy", 0)) == 1


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
    assert calls.count(("runtime:destroy", 0)) == 1


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
            admission_count=participant.admission_count,
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


def test_participant_stop_time_reduces_the_dbos_workflow_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    config = DbosHostConfig(
        system_database_url="sqlite:///dbos.sqlite",
        application_version="host-stop-budget-v1",
        executor_id="stable-stop-budget-slot",
        shutdown_grace_s=4,
    )
    participant = _participant(
        "control",
        "queue-control",
        calls,
        host_config=config,
    )

    def _slow_stop() -> None:
        time.sleep(1.1)

    host = DbosRuntimeHost(config)
    host._register_participant(
        _DbosHostParticipant(
            participant_id=participant.participant_id,
            queue_name=participant.queue_name,
            host_config=participant.host_config,
            register_workflows=participant.register_workflows,
            preflight=participant.preflight,
            register_queue=participant.register_queue,
            stop_admission=_slow_stop,
            admission_count=participant.admission_count,
            active_count=participant.active_count,
            mark_closed=participant.mark_closed,
        )
    )
    host.launch()

    host.close()

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


def test_admitted_facade_operations_drain_before_runtime_destroy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    fake = _FakeRuntime(calls)
    _install_fake_runtime(monkeypatch, fake)
    samples = 0
    participant = _participant("control", "queue-control", calls)

    def _admissions() -> int:
        nonlocal samples
        samples += 1
        return 1 if samples < 3 else 0

    original_destroy = fake.destroy

    def _destroy(*, workflow_completion_timeout_sec: int, deadline: float) -> None:
        assert samples == 3
        original_destroy(
            workflow_completion_timeout_sec=workflow_completion_timeout_sec,
            deadline=deadline,
        )

    fake.destroy = _destroy  # type: ignore[method-assign]
    host = DbosRuntimeHost(_config())
    host._register_participant(
        _DbosHostParticipant(
            participant_id=participant.participant_id,
            queue_name=participant.queue_name,
            host_config=participant.host_config,
            register_workflows=participant.register_workflows,
            preflight=participant.preflight,
            register_queue=participant.register_queue,
            stop_admission=participant.stop_admission,
            admission_count=_admissions,
            active_count=participant.active_count,
            mark_closed=participant.mark_closed,
        )
    )
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
                admission_count=lambda: 0,
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


@pytest.mark.slow
def test_real_dbos_host_composes_control_and_run_under_one_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dbos = pytest.importorskip("dbos")
    if _delegate_real_dbos_case(
        "test_real_dbos_host_composes_control_and_run_under_one_runtime",
        "control-run-composition",
    ):
        return
    implementation = importlib.import_module("dbos._dbos")
    dbos.DBOS.destroy(destroy_registry=True, workflow_completion_timeout_sec=0)
    host: DbosRuntimeHost | None = None
    replacement: DbosRuntimeHost | None = None
    constructed_runtimes: list[Any] = []
    original_construct = runtime_module._construct_owned_runtime_226

    def _construct(dbos_module: Any, config: DbosHostConfig) -> Any:
        runtime = original_construct(dbos_module, config)
        constructed_runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(runtime_module, "_construct_owned_runtime_226", _construct)
    try:
        database_url = f"sqlite:///{tmp_path / 'composition.sqlite'}"
        control_config = DbosControlConfig(
            system_database_url=database_url,
            name="monoid-host-composition",
            application_version="monoid-host-composition-v1",
            executor_id="stable-host-composition-slot",
            queue_name="composition-control",
            polling_interval_s=0.01,
            shutdown_grace_s=5,
        )
        run_config = DbosRunConfig(
            system_database_url=database_url,
            name="monoid-host-composition",
            application_version="monoid-host-composition-v1",
            executor_id="stable-host-composition-slot",
            queue_name="composition-run",
            polling_interval_s=0.01,
            checkpoint_retry_interval_s=0.01,
            shutdown_grace_s=5,
            local_task_wait_s=5,
        )
        shared_config = _require_shared_host_config(
            control_config._host_config(),
            run_config._host_config(),
        )
        host = DbosRuntimeHost(shared_config)
        observed_control: list[str] = []

        def _dispatch(envelope: DbosControlEnvelope) -> ControlResult:
            observed_control.append(envelope.command_id)
            return ControlResult(
                run_id=envelope.run_id,
                type=envelope.type,
                status="ok",
                state="running",
                data={"runtime_owner": "shared-host"},
            )

        def _unexpected_loop(command: DbosResumeCommand) -> Any:
            del command
            raise AssertionError("missing-checkpoint recovery must not construct an AgentLoop")

        store = LocalFsCheckpointStore(tmp_path / "runs")
        driver = _register_hosted_run_driver(
            host,
            run_config,
            store,
            _unexpected_loop,
        )
        plane = _register_hosted_control_plane(host, control_config, _dispatch)
        owned_runtime = host._runtime
        assert constructed_runtimes == [owned_runtime]
        assert set(host._participants) == {"control", "run"}

        lifecycle_calls: list[tuple[str, object]] = []
        original_listen = owned_runtime.listen_queues
        original_launch = owned_runtime.launch
        original_register_queue = owned_runtime.register_queue
        original_destroy = owned_runtime.destroy

        def _listen(queues: tuple[str, ...]) -> None:
            lifecycle_calls.append(("listen", tuple(queues)))
            original_listen(queues)

        def _launch() -> None:
            lifecycle_calls.append(("launch", None))
            original_launch()

        def _register_queue(name: str, **kwargs: Any) -> Any:
            lifecycle_calls.append(("queue", name))
            return original_register_queue(name, **kwargs)

        def _destroy(*, workflow_completion_timeout_sec: int, deadline: float) -> None:
            lifecycle_calls.append(("destroy", workflow_completion_timeout_sec))
            original_destroy(
                workflow_completion_timeout_sec=workflow_completion_timeout_sec,
                deadline=deadline,
            )

        monkeypatch.setattr(owned_runtime, "listen_queues", _listen)
        monkeypatch.setattr(owned_runtime, "launch", _launch)
        monkeypatch.setattr(owned_runtime, "register_queue", _register_queue)
        monkeypatch.setattr(owned_runtime, "destroy", _destroy)

        host.launch()
        host.launch()
        expected_queues = tuple(
            sorted((plane.registered_queue_name, driver.registered_queue_name))
        )
        assert lifecycle_calls == [
            ("listen", expected_queues),
            ("launch", None),
            ("queue", plane.registered_queue_name),
            ("queue", driver.registered_queue_name),
        ]
        assert plane._runtime is driver._runtime is owned_runtime
        assert getattr(implementation, "_dbos_global_instance", None) is owned_runtime._runtime
        assert plane._workflow is not None
        assert driver._workflow is not None
        assert host.state == "running"
        assert host.accepting is plane._accepting is driver._accepting is True
        with pytest.raises(DbosProcessOwnershipError, match="already owns"):
            DbosRuntimeHost(shared_config)

        run_id = "shared/run"
        command_id = "shared/command"
        control_command = ControlCommand(
            type="status",
            run_id=run_id,
            command_id=command_id,
        )
        run_command = DbosResumeCommand(run_id, command_id, 1)
        control_workflow_id = plane.workflow_id(run_id, command_id)
        run_workflow_id = driver.workflow_id(run_id, command_id)
        assert control_workflow_id == (
            "monoid/run/shared%2Frun/control/shared%2Fcommand"
        )
        assert run_workflow_id == (
            "monoid/run/shared%2Frun/resume/shared%2Fcommand"
        )
        initial_control = plane.enqueue_control(
            control_command,
            tenant_id="tenant",
            user_id="user",
        )
        initial_run = driver.enqueue_resume(run_command)
        completed_control = plane.wait_for_receipt(
            run_id,
            control_command.command_id,
            timeout_s=30,
        )
        completed_run = driver.wait_for_receipt(run_command, timeout_s=30)

        assert initial_control.status in {"pending", "completed"}
        assert initial_run.status in {"pending", "failed"}
        assert completed_control.status == "completed"
        assert completed_control.result is not None
        assert completed_control.result["data"] == {"runtime_owner": "shared-host"}
        assert completed_run.status == "failed"
        assert completed_run.error_code == "checkpoint_missing"
        assert observed_control == [control_command.command_id]
        assert store.latest(run_id) is None

        host.close()
        host.close()
        assert [name for name, _ in lifecycle_calls] == [
            "listen",
            "launch",
            "queue",
            "queue",
            "destroy",
        ]
        assert host.state == "closed"
        assert plane._closed is driver._closed is True
        assert constructed_runtimes == [owned_runtime]
        assert getattr(implementation, "_dbos_global_instance", None) is None
        assert getattr(implementation, "_dbos_global_registry", None) is None

        replacement = DbosRuntimeHost(shared_config)
        assert constructed_runtimes == [owned_runtime, replacement._runtime]
        assert replacement._runtime is not owned_runtime
        replacement.close()
        assert replacement.state == "closed"
        assert getattr(implementation, "_dbos_global_instance", None) is None
        assert getattr(implementation, "_dbos_global_registry", None) is None
    finally:
        try:
            for candidate in (replacement, host):
                if candidate is None or candidate.state == "closed":
                    continue
                if candidate.state == "fenced":
                    cleanup = candidate._fenced_cleanup_thread
                    if cleanup is not None:
                        cleanup.join(timeout=6)
                    continue
                candidate.close()
        finally:
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
                admission_count=lambda: 0,
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


def test_real_dbos_async_launch_rejection_releases_ownership(tmp_path: Path) -> None:
    dbos = pytest.importorskip("dbos")
    if _delegate_real_dbos_case(
        "test_real_dbos_async_launch_rejection_releases_ownership",
        "async-launch",
    ):
        return
    implementation = importlib.import_module("dbos._dbos")
    dbos.DBOS.destroy(destroy_registry=True, workflow_completion_timeout_sec=0)
    host: DbosRuntimeHost | None = None
    replacement: DbosRuntimeHost | None = None
    try:
        config = DbosHostConfig(
            system_database_url=f"sqlite:///{tmp_path / 'async-launch.sqlite'}",
            name="monoid-host-async-launch",
            application_version="monoid-host-async-launch-v1",
            executor_id="stable-async-launch-slot",
        )
        host = DbosRuntimeHost(config)
        host._register_participant(
            _DbosHostParticipant(
                participant_id="async-launch",
                queue_name="monoid-reference-host-async-launch-v1",
                host_config=config,
                register_workflows=lambda runtime: None,
                preflight=lambda: None,
                register_queue=lambda runtime: None,
                stop_admission=lambda: None,
                admission_count=lambda: 0,
                active_count=lambda: 0,
                mark_closed=lambda: None,
            )
        )

        async def _attempt_launch() -> None:
            with pytest.raises(DbosProcessOwnershipError, match="synchronous lifecycle thread"):
                host.launch()

        asyncio.run(_attempt_launch())

        assert host.state == "closed"
        assert getattr(implementation, "_dbos_global_instance", None) is None
        assert getattr(implementation, "_dbos_global_registry", None) is None
        replacement = DbosRuntimeHost(config)
        replacement.close()
        assert replacement.state == "closed"
    finally:
        if replacement is not None and replacement.state not in {"closed", "fenced"}:
            replacement.close()
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
