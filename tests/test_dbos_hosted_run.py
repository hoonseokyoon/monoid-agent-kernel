from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore
from monoid_agent_kernel.reference.dbos import run_driver as run_module
from monoid_agent_kernel.reference.dbos import runtime as runtime_module
from monoid_agent_kernel.reference.dbos._compat_226 import _DbosOwnershipConflict
from monoid_agent_kernel.reference.dbos.run_driver import (
    DBOS_RUN_STEP_NAME,
    DBOS_RUN_WORKFLOW_NAME,
    DbosResumeCommand,
    DbosRunConfig,
    DbosRunReceipt,
    _register_hosted_run_driver,
)
from monoid_agent_kernel.reference.dbos.runtime import (
    DbosHostConfig,
    DbosProcessOwnershipError,
    DbosRuntimeHost,
)

pytestmark = pytest.mark.serial


class _FakeQueue:
    concurrency = 1
    worker_concurrency = 1
    partition_queue = True
    priority_enabled = False


class _FakeHandle:
    def __init__(self, runtime: _FakeOwnedRuntime, workflow_id: str) -> None:
        self._runtime = runtime
        self._workflow_id = workflow_id

    def get_status(self) -> Any:
        self._runtime.status_entered.set()
        if not self._runtime.release_status.wait(timeout=3):
            raise TimeoutError("status was not released")
        if self._runtime.status_conflict:
            raise _DbosOwnershipConflict("private status replacement detail")
        return self._runtime.statuses[self._workflow_id]


class _FakeClient:
    def __init__(self, runtime: _FakeOwnedRuntime) -> None:
        self._runtime = runtime

    def register_queue(self, queue_name: str, **kwargs: Any) -> _FakeQueue:
        self._runtime.calls.append(("client:queue", queue_name, dict(kwargs)))
        return _FakeQueue()

    def destroy(self) -> None:
        self._runtime.calls.append("client:destroy")


class _FakeOwnedRuntime:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.statuses: dict[str, Any] = {}
        self.owned = True
        self.released = False
        self.enqueue_conflict = False
        self.enqueue_error: BaseException | None = None
        self.retrieve_conflict = False
        self.status_conflict = False
        self.fail_workflow_registration = False
        self.fail_queue_registration = False
        self.launch_hook: Any | None = None
        self.queue_register_entered = threading.Event()
        self.release_queue_register = threading.Event()
        self.release_queue_register.set()
        self.enqueue_entered = threading.Event()
        self.release_enqueue = threading.Event()
        self.release_enqueue.set()
        self.retrieve_entered = threading.Event()
        self.release_retrieve = threading.Event()
        self.release_retrieve.set()
        self.status_entered = threading.Event()
        self.release_status = threading.Event()
        self.release_status.set()
        self.destroy_entered = threading.Event()

    def require_owned(self) -> None:
        if not self.owned:
            raise _DbosOwnershipConflict("private ownership detail")

    def owns_globals(self) -> bool:
        return self.owned

    def require_released(self) -> None:
        if not self.released:
            raise RuntimeError("fake runtime was not released")

    def preflight_launch(self) -> None:
        self.calls.append("runtime:preflight")

    def step(self, *, name: str, retries_allowed: bool) -> Any:
        self.calls.append(("runtime:step", name, retries_allowed))
        if self.fail_workflow_registration:
            raise RuntimeError("workflow registration failed")

        def _decorate(function: Any) -> Any:
            return function

        return _decorate

    def workflow(self, *, name: str) -> Any:
        self.calls.append(("runtime:workflow", name))

        def _decorate(function: Any) -> Any:
            return function

        return _decorate

    def listen_queues(self, queue_names: tuple[str, ...]) -> None:
        self.calls.append(("runtime:listen", queue_names))

    def launch(self) -> None:
        self.calls.append("runtime:launch")
        if self.launch_hook is not None:
            self.launch_hook()

    def register_queue(self, queue_name: str, **kwargs: Any) -> _FakeQueue:
        self.calls.append(("runtime:queue", queue_name, dict(kwargs)))
        self.queue_register_entered.set()
        if not self.release_queue_register.wait(timeout=3):
            raise TimeoutError("queue registration was not released")
        if self.fail_queue_registration:
            raise RuntimeError("queue registration failed")
        return _FakeQueue()

    def enqueue_workflow_with_identity(
        self,
        queue_name: str,
        workflow: Any,
        payload: dict[str, Any],
        *,
        workflow_id: str,
        queue_partition_key: str,
    ) -> _FakeHandle:
        del workflow
        self.calls.append(
            (
                "runtime:enqueue",
                queue_name,
                workflow_id,
                queue_partition_key,
                dict(payload),
            )
        )
        self.enqueue_entered.set()
        if not self.release_enqueue.wait(timeout=3):
            raise TimeoutError("enqueue was not released")
        if self.enqueue_error is not None:
            raise self.enqueue_error
        if self.enqueue_conflict:
            raise _DbosOwnershipConflict("private enqueue replacement detail")
        self.statuses.setdefault(
            workflow_id,
            SimpleNamespace(
                status="PENDING",
                input={"args": [dict(payload)]},
                output=None,
            ),
        )
        return _FakeHandle(self, workflow_id)

    def retrieve_workflow(self, workflow_id: str) -> _FakeHandle:
        self.calls.append(("runtime:retrieve", workflow_id))
        self.retrieve_entered.set()
        if not self.release_retrieve.wait(timeout=3):
            raise TimeoutError("retrieve was not released")
        if self.retrieve_conflict:
            raise _DbosOwnershipConflict("private retrieve replacement detail")
        return _FakeHandle(self, workflow_id)

    def destroy(
        self,
        *,
        workflow_completion_timeout_sec: int,
        deadline: float,
    ) -> None:
        del deadline
        self.calls.append(("runtime:destroy", workflow_completion_timeout_sec))
        self.destroy_entered.set()
        self.owned = False
        self.released = True


@pytest.fixture(autouse=True)
def _reset_process_owner() -> None:
    with runtime_module._PROCESS_OWNER_LOCK:
        runtime_module._PROCESS_OWNER_TOKEN = None
    yield
    with runtime_module._PROCESS_OWNER_LOCK:
        runtime_module._PROCESS_OWNER_TOKEN = None


def _config() -> DbosRunConfig:
    return DbosRunConfig(
        system_database_url="sqlite:///hosted-run.sqlite",
        name="hosted-run",
        application_version="hosted-run-v1",
        executor_id="stable-hosted-slot",
        polling_interval_s=0.01,
        checkpoint_retry_interval_s=0.01,
        shutdown_grace_s=3,
        local_task_wait_s=3,
    )


def _host_config() -> DbosHostConfig:
    return _config()._host_config()


def _unused_loop_factory(command: DbosResumeCommand) -> Any:
    del command
    raise AssertionError("loop factory should not run in this test")


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    runtime: _FakeOwnedRuntime,
) -> None:
    class _FakeDBAPIError(Exception):
        pass

    sqlalchemy_module = ModuleType("sqlalchemy")
    sqlalchemy_exc_module = ModuleType("sqlalchemy.exc")
    setattr(sqlalchemy_exc_module, "DBAPIError", _FakeDBAPIError)
    setattr(sqlalchemy_module, "exc", sqlalchemy_exc_module)
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqlalchemy_module)
    monkeypatch.setitem(sys.modules, "sqlalchemy.exc", sqlalchemy_exc_module)
    module = SimpleNamespace(
        DBOSClient=lambda *, system_database_url: _FakeClient(runtime),
    )
    monkeypatch.setattr(runtime_module, "load_dbos", lambda: module)
    monkeypatch.setattr(
        runtime_module,
        "_construct_owned_runtime_226",
        lambda dbos_module, config: runtime,
    )
    monkeypatch.setattr(run_module, "load_dbos", lambda: module)


def _register_driver(
    host: DbosRuntimeHost,
    *,
    loop_factory: Any = _unused_loop_factory,
) -> Any:
    return _register_hosted_run_driver(host, _config(), object(), loop_factory)


def _command(command_id: str = "resume") -> DbosResumeCommand:
    return DbosResumeCommand("run/hosted", command_id, 1)


def test_hosted_run_defers_registration_and_uses_owned_workflow_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeOwnedRuntime()
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    driver = _register_driver(host)

    assert runtime.calls == []
    with pytest.raises(RuntimeError, match="host owns"):
        driver.launch()
    with pytest.raises(RuntimeError, match="host owns"):
        driver.close()

    host.launch()
    command = _command("command/hosted")
    admitted = driver.enqueue_resume(command)
    observed = driver.run_receipt(command)

    expected_workflow_id = driver.workflow_id(command.run_id, command.command_id)
    assert admitted.status == observed.status == "pending"
    assert ("runtime:step", DBOS_RUN_STEP_NAME, False) in runtime.calls
    assert ("runtime:workflow", DBOS_RUN_WORKFLOW_NAME) in runtime.calls
    assert ("runtime:listen", (driver.registered_queue_name,)) in runtime.calls
    enqueue_call = next(call for call in runtime.calls if call[0] == "runtime:enqueue")
    assert enqueue_call[2:4] == (expected_workflow_id, command.run_id)

    host.close()

    assert host.state == "closed"
    assert driver._closed is True
    assert runtime.released is True


def test_hosted_run_rejects_config_duplicate_and_late_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeOwnedRuntime()
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    mismatched = DbosRunConfig(
        system_database_url=_config().system_database_url,
        name=_config().name,
        application_version=_config().application_version,
        executor_id="different-slot",
        shutdown_grace_s=3,
    )

    with pytest.raises(ValueError, match="host configurations do not match"):
        _register_hosted_run_driver(host, mismatched, object(), _unused_loop_factory)
    assert host._participants == {}

    _register_driver(host)
    with pytest.raises(ValueError, match="ids must be unique"):
        _register_driver(host)
    host.launch()
    with pytest.raises(RuntimeError, match="before host launch"):
        _register_driver(host)
    host.close()


def test_hosted_run_opens_admission_only_after_host_launch_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeOwnedRuntime()
    runtime.release_queue_register.clear()
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    driver = _register_driver(host)
    launch_errors: list[BaseException] = []

    def _launch() -> None:
        try:
            host.launch()
        except BaseException as exc:
            launch_errors.append(exc)

    launch_thread = threading.Thread(target=_launch)
    launch_thread.start()
    assert runtime.queue_register_entered.wait(timeout=1)
    assert host.state == "launching"
    with pytest.raises(RuntimeError, match="not accepting"):
        driver.enqueue_resume(_command("early"))

    runtime.release_queue_register.set()
    launch_thread.join(timeout=3)
    assert not launch_thread.is_alive()
    assert launch_errors == []
    assert host.accepting is True
    driver.enqueue_resume(_command("ready"))
    host.close()


def test_hosted_run_startup_cleanup_failure_stays_closed_but_receipts_remain_readable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeOwnedRuntime()
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    driver = _register_driver(host)
    runtime.launch_hook = driver._mark_cleanup_failure

    host.launch()

    command = _command("recovered")
    workflow_id = driver.workflow_id(command.run_id, command.command_id)
    runtime.statuses[workflow_id] = SimpleNamespace(
        status="PENDING",
        input={"args": [command.to_json()]},
        output=None,
    )
    with pytest.raises(RuntimeError, match="not accepting"):
        driver.enqueue_resume(_command("new"))
    assert driver.run_receipt(command).status == "pending"
    assert host.state == "running"
    assert host.accepting is True

    host.close()


def test_hosted_run_close_cannot_overtake_an_admitted_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeOwnedRuntime()
    runtime.release_enqueue.clear()
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    driver = _register_driver(host)
    host.launch()
    enqueue_errors: list[BaseException] = []
    close_errors: list[BaseException] = []

    def _enqueue() -> None:
        try:
            driver.enqueue_resume(_command("first"))
        except BaseException as exc:
            enqueue_errors.append(exc)

    def _close() -> None:
        try:
            host.close()
        except BaseException as exc:
            close_errors.append(exc)

    enqueue_thread = threading.Thread(target=_enqueue)
    enqueue_thread.start()
    assert runtime.enqueue_entered.wait(timeout=1)
    close_thread = threading.Thread(target=_close)
    close_thread.start()
    deadline = time.monotonic() + 1
    while host.accepting and time.monotonic() < deadline:
        time.sleep(0.001)
    assert host.accepting is False
    assert runtime.destroy_entered.is_set() is False

    runtime.release_enqueue.set()
    enqueue_thread.join(timeout=3)
    close_thread.join(timeout=3)

    assert enqueue_errors == []
    assert close_errors == []
    assert runtime.destroy_entered.is_set()
    assert host.state == "closed"
    with pytest.raises(RuntimeError, match="not accepting"):
        driver.enqueue_resume(_command("late"))


@pytest.mark.parametrize("operation", ["retrieve", "status"])
def test_hosted_run_drains_admitted_receipt_reads_before_destroy(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    runtime = _FakeOwnedRuntime()
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    driver = _register_driver(host)
    host.launch()
    command = _command("receipt")
    driver.enqueue_resume(command)
    runtime.retrieve_entered.clear()
    runtime.status_entered.clear()
    if operation == "retrieve":
        runtime.release_retrieve.clear()
    else:
        runtime.release_status.clear()
    receipt_errors: list[BaseException] = []
    close_errors: list[BaseException] = []

    def _read_receipt() -> None:
        try:
            driver.run_receipt(command)
        except BaseException as exc:
            receipt_errors.append(exc)

    def _close() -> None:
        try:
            host.close()
        except BaseException as exc:
            close_errors.append(exc)

    receipt_thread = threading.Thread(target=_read_receipt)
    receipt_thread.start()
    entered = runtime.retrieve_entered if operation == "retrieve" else runtime.status_entered
    assert entered.wait(timeout=1)
    close_thread = threading.Thread(target=_close)
    close_thread.start()
    deadline = time.monotonic() + 1
    while host.accepting and time.monotonic() < deadline:
        time.sleep(0.001)
    assert host.accepting is False
    assert runtime.destroy_entered.is_set() is False
    retrieve_calls = sum(
        isinstance(call, tuple) and call[0] == "runtime:retrieve" for call in runtime.calls
    )
    with pytest.raises(RuntimeError, match="not accepting"):
        driver.run_receipt(command)
    assert (
        sum(isinstance(call, tuple) and call[0] == "runtime:retrieve" for call in runtime.calls)
        == retrieve_calls
    )

    runtime.release_retrieve.set()
    runtime.release_status.set()
    receipt_thread.join(timeout=3)
    close_thread.join(timeout=3)

    assert receipt_errors == []
    assert close_errors == []
    assert host.state == "closed"


def test_hosted_run_nonownership_enqueue_error_releases_admission_without_fencing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeOwnedRuntime()
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    driver = _register_driver(host)
    host.launch()
    runtime.enqueue_error = RuntimeError("no ambient DBOS context")

    with pytest.raises(RuntimeError, match="no ambient DBOS context"):
        driver.enqueue_resume(_command("ambient"))

    assert driver._hosted_admission_count() == 0
    assert host.state == "running"
    assert host.accepting is True
    runtime.enqueue_error = None
    assert driver.enqueue_resume(_command("after-error")).status == "pending"
    host.close()


def test_hosted_run_drive_activity_drains_under_host_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeOwnedRuntime()
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    driver = _register_driver(host)
    host.launch()
    started = threading.Event()
    release = threading.Event()
    command = _command("drive")

    def _drive_one(observed: DbosResumeCommand) -> DbosRunReceipt:
        assert observed == command
        started.set()
        assert release.wait(timeout=3)
        return DbosRunReceipt(
            run_id=observed.run_id,
            command_id=observed.command_id,
            status="failed",
            error="test completion",
            error_code="test_completion",
        )

    monkeypatch.setattr(driver, "_drive_one", _drive_one)
    workflow = driver._workflow
    assert callable(workflow)
    workflow_errors: list[BaseException] = []

    def _run_workflow() -> None:
        try:
            workflow(command.to_json())
        except BaseException as exc:
            workflow_errors.append(exc)

    workflow_thread = threading.Thread(target=_run_workflow)
    workflow_thread.start()
    assert started.wait(timeout=1)
    timer = threading.Timer(0.1, release.set)
    timer.start()
    host.close()
    timer.join(timeout=1)
    workflow_thread.join(timeout=3)

    assert workflow_errors == []
    assert host.state == "closed"


def test_drive_receipt_read_is_rejected_after_host_starts_closing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeOwnedRuntime()
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    driver = _register_driver(host)
    host.launch()
    command = _command("drive-close")
    drive_started = threading.Event()
    attempt_receipt = threading.Event()
    receipt_rejected = threading.Event()

    def _drive_one(observed: DbosResumeCommand) -> DbosRunReceipt:
        drive_started.set()
        assert attempt_receipt.wait(timeout=3)
        try:
            driver.run_receipt(observed)
        except RuntimeError as exc:
            assert "not accepting" in str(exc)
            receipt_rejected.set()
        return DbosRunReceipt(
            run_id=observed.run_id,
            command_id=observed.command_id,
            status="failed",
            error="test completion",
            error_code="test_completion",
        )

    monkeypatch.setattr(driver, "_drive_one", _drive_one)
    workflow = driver._workflow
    assert callable(workflow)
    workflow_thread = threading.Thread(target=workflow, args=(command.to_json(),))
    workflow_thread.start()
    assert drive_started.wait(timeout=1)
    close_errors: list[BaseException] = []

    def _close() -> None:
        try:
            host.close()
        except BaseException as exc:
            close_errors.append(exc)

    close_thread = threading.Thread(target=_close)
    close_thread.start()
    deadline = time.monotonic() + 1
    while not driver._closing and time.monotonic() < deadline:
        time.sleep(0.001)
    assert driver._closing is True
    attempt_receipt.set()
    assert receipt_rejected.wait(timeout=1)
    workflow_thread.join(timeout=3)
    close_thread.join(timeout=3)

    assert close_errors == []
    assert host.state == "closed"
    assert not any(
        isinstance(call, tuple) and call[0] == "runtime:retrieve" for call in runtime.calls
    )


@pytest.mark.parametrize("operation", ["enqueue", "retrieve", "status"])
def test_hosted_run_ownership_conflicts_fence_the_host_without_private_details(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    runtime = _FakeOwnedRuntime()
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    driver = _register_driver(host)
    host.launch()
    command = _command("owned")
    if operation == "enqueue":
        runtime.enqueue_conflict = True
    else:
        driver.enqueue_resume(command)
        if operation == "retrieve":
            runtime.retrieve_conflict = True
        else:
            runtime.status_conflict = True

    with pytest.raises(DbosProcessOwnershipError, match="ownership changed") as raised:
        if operation == "enqueue":
            driver.enqueue_resume(command)
        else:
            driver.run_receipt(command)

    assert "private" not in str(raised.value)
    assert host.state == "fenced"
    assert host.accepting is False


def test_hosted_run_postlaunch_queue_failure_fences_and_cleans_the_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeOwnedRuntime()
    runtime.fail_queue_registration = True
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    _register_driver(host)

    with pytest.raises(DbosProcessOwnershipError, match="uncertain after runtime activation"):
        host.launch()

    cleanup = host._fenced_cleanup_thread
    assert cleanup is not None
    cleanup.join(timeout=3)
    assert not cleanup.is_alive()
    assert host.state == "fenced"
    assert host.accepting is False
    assert runtime.released is True


def test_hosted_run_prelaunch_registration_failure_rolls_back_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeOwnedRuntime()
    runtime.fail_workflow_registration = True
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    driver = _register_driver(host)

    with pytest.raises(RuntimeError, match="workflow registration failed"):
        host.launch()

    assert host.state == "closed"
    assert driver._closed is True
    assert runtime.released is True
    replacement = DbosRuntimeHost(_host_config())
    replacement.close()


@pytest.mark.slow
def test_real_hosted_run_executes_and_releases_process_globals(tmp_path: Path) -> None:
    dbos = pytest.importorskip("dbos")
    dbos.DBOS.destroy(destroy_registry=True, workflow_completion_timeout_sec=0)
    host: DbosRuntimeHost | None = None
    replacement: DbosRuntimeHost | None = None
    try:
        config = DbosRunConfig(
            system_database_url=f"sqlite:///{tmp_path / 'hosted-run.sqlite'}",
            name="monoid-hosted-run-smoke",
            application_version="monoid-hosted-run-smoke-v1",
            executor_id="stable-hosted-run-smoke-slot",
            polling_interval_s=0.01,
            checkpoint_retry_interval_s=0.01,
            shutdown_grace_s=3,
            local_task_wait_s=3,
        )
        host = DbosRuntimeHost(config._host_config())
        driver = _register_hosted_run_driver(
            host,
            config,
            LocalFsCheckpointStore(tmp_path / "runs"),
            _unused_loop_factory,
        )
        host.launch()
        command = DbosResumeCommand("missing-run", "missing-checkpoint", 1)
        initial = driver.enqueue_resume(command)
        assert initial.status in {"pending", "failed"}
        completed = driver.wait_for_receipt(command, timeout_s=30)
        assert completed.status == "failed"
        assert completed.error_code == "checkpoint_missing"
        host.close()
        assert host.state == "closed"
        assert driver._closed is True
        replacement = DbosRuntimeHost(config._host_config())
        replacement.close()
        assert replacement.state == "closed"
    finally:
        try:
            for candidate in (replacement, host):
                if candidate is None or candidate.state == "closed":
                    continue
                if candidate.state == "fenced":
                    cleanup = candidate._fenced_cleanup_thread
                    if cleanup is not None:
                        cleanup.join(timeout=4)
                    continue
                candidate.close()
        finally:
            dbos.DBOS.destroy(destroy_registry=True, workflow_completion_timeout_sec=0)
