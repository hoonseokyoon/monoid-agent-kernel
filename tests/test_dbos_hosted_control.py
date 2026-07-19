from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from monoid_agent_kernel.core.control import ControlCommand, ControlResult
from monoid_agent_kernel.reference.dbos import control_plane as control_module
from monoid_agent_kernel.reference.dbos import runtime as runtime_module
from monoid_agent_kernel.reference.dbos._compat_226 import _DbosOwnershipConflict
from monoid_agent_kernel.reference.dbos.control_plane import (
    DBOS_CONTROL_STEP_NAME,
    DBOS_CONTROL_WORKFLOW_NAME,
    DbosControlConfig,
    DbosControlEnvelope,
    _register_hosted_control_plane,
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
        self.retrieve_conflict = False
        self.status_conflict = False
        self.fail_workflow_registration = False
        self.fail_queue_registration = False
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
        if self.enqueue_conflict:
            raise _DbosOwnershipConflict("private enqueue replacement detail")
        self.statuses.setdefault(
            workflow_id,
            SimpleNamespace(
                status="PENDING",
                input={"args": [dict(payload)]},
                output=None,
                created_at=1_000,
                updated_at=1_000,
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


def _config() -> DbosControlConfig:
    return DbosControlConfig(
        system_database_url="sqlite:///hosted-control.sqlite",
        name="hosted-control",
        application_version="hosted-control-v1",
        executor_id="stable-hosted-slot",
        polling_interval_s=0.01,
        shutdown_grace_s=3,
    )


def _host_config() -> DbosHostConfig:
    return _config()._host_config()


def _ok(envelope: DbosControlEnvelope) -> ControlResult:
    return ControlResult(
        run_id=envelope.run_id,
        type=envelope.type,
        status="ok",
        state="running",
    )


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    runtime: _FakeOwnedRuntime,
) -> None:
    module = SimpleNamespace(
        DBOSClient=lambda *, system_database_url: _FakeClient(runtime),
    )
    monkeypatch.setattr(runtime_module, "load_dbos", lambda: module)
    monkeypatch.setattr(
        runtime_module,
        "_construct_owned_runtime_226",
        lambda dbos_module, config: runtime,
    )
    monkeypatch.setattr(control_module, "load_dbos", lambda: module)


def test_hosted_control_defers_registration_and_uses_owned_workflow_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeOwnedRuntime()
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    plane = _register_hosted_control_plane(host, _config(), _ok)

    assert runtime.calls == []
    with pytest.raises(RuntimeError, match="host owns"):
        plane.launch()
    with pytest.raises(RuntimeError, match="host owns"):
        plane.close()

    host.launch()
    command = ControlCommand(
        type="status",
        run_id="run/hosted",
        command_id="command/hosted",
    )
    admitted = plane.enqueue_control(command, tenant_id="tenant", user_id="user")
    observed = plane.command_receipt("run/hosted", "command/hosted")

    expected_workflow_id = plane.workflow_id("run/hosted", "command/hosted")
    assert admitted.status == observed.status == "pending"
    assert ("runtime:step", DBOS_CONTROL_STEP_NAME, False) in runtime.calls
    assert ("runtime:workflow", DBOS_CONTROL_WORKFLOW_NAME) in runtime.calls
    assert (
        "runtime:listen",
        (plane.registered_queue_name,),
    ) in runtime.calls
    enqueue_call = next(call for call in runtime.calls if call[0] == "runtime:enqueue")
    assert enqueue_call[2:4] == (expected_workflow_id, "run/hosted")

    host.close()

    assert host.state == "closed"
    assert plane._closed is True
    assert runtime.released is True


def test_hosted_control_rejects_config_duplicate_and_late_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeOwnedRuntime()
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    mismatched = DbosControlConfig(
        system_database_url=_config().system_database_url,
        name=_config().name,
        application_version=_config().application_version,
        executor_id="different-slot",
        shutdown_grace_s=3,
    )

    with pytest.raises(ValueError, match="host configurations do not match"):
        _register_hosted_control_plane(host, mismatched, _ok)
    assert host._participants == {}

    _register_hosted_control_plane(host, _config(), _ok)
    with pytest.raises(ValueError, match="ids must be unique"):
        _register_hosted_control_plane(host, _config(), _ok)
    host.launch()
    with pytest.raises(RuntimeError, match="before host launch"):
        _register_hosted_control_plane(host, _config(), _ok)
    host.close()


def test_hosted_control_opens_admission_only_after_host_launch_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeOwnedRuntime()
    runtime.release_queue_register.clear()
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    plane = _register_hosted_control_plane(host, _config(), _ok)
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
        plane.enqueue_control(
            ControlCommand(type="status", run_id="run", command_id="early"),
            tenant_id="tenant",
            user_id="user",
        )

    runtime.release_queue_register.set()
    launch_thread.join(timeout=3)
    assert not launch_thread.is_alive()
    assert launch_errors == []
    assert host.accepting is True
    plane.enqueue_control(
        ControlCommand(type="status", run_id="run", command_id="ready"),
        tenant_id="tenant",
        user_id="user",
    )
    host.close()


def test_hosted_control_close_cannot_overtake_an_admitted_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeOwnedRuntime()
    runtime.release_enqueue.clear()
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    plane = _register_hosted_control_plane(host, _config(), _ok)
    host.launch()
    first_errors: list[BaseException] = []
    close_errors: list[BaseException] = []

    def _enqueue_first() -> None:
        try:
            plane.enqueue_control(
                ControlCommand(type="status", run_id="run", command_id="first"),
                tenant_id="tenant",
                user_id="user",
            )
        except BaseException as exc:
            first_errors.append(exc)

    def _close() -> None:
        try:
            host.close()
        except BaseException as exc:
            close_errors.append(exc)

    enqueue_thread = threading.Thread(target=_enqueue_first)
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

    assert not enqueue_thread.is_alive()
    assert not close_thread.is_alive()
    assert first_errors == []
    assert close_errors == []
    assert runtime.destroy_entered.is_set()
    assert host.state == "closed"
    with pytest.raises(RuntimeError, match="not accepting"):
        plane.enqueue_control(
            ControlCommand(type="status", run_id="run", command_id="late"),
            tenant_id="tenant",
            user_id="user",
        )


@pytest.mark.parametrize("operation", ["retrieve", "status"])
def test_hosted_control_drains_admitted_receipt_reads_before_destroy(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    runtime = _FakeOwnedRuntime()
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    plane = _register_hosted_control_plane(host, _config(), _ok)
    host.launch()
    plane.enqueue_control(
        ControlCommand(type="status", run_id="run", command_id="receipt"),
        tenant_id="tenant",
        user_id="user",
    )
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
            plane.command_receipt("run", "receipt")
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
        plane.command_receipt("run", "receipt")
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


def test_hosted_control_dispatch_activity_drains_under_host_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeOwnedRuntime()
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    started = threading.Event()
    release = threading.Event()

    def _dispatch(envelope: DbosControlEnvelope) -> ControlResult:
        started.set()
        assert release.wait(timeout=3)
        return _ok(envelope)

    plane = _register_hosted_control_plane(host, _config(), _dispatch)
    host.launch()
    workflow = plane._workflow
    assert callable(workflow)
    payload = DbosControlEnvelope.from_control_command(
        ControlCommand(type="status", run_id="run", command_id="dispatch"),
        tenant_id="tenant",
        user_id="user",
    ).to_json()
    workflow_errors: list[BaseException] = []

    def _run_workflow() -> None:
        try:
            workflow(payload)
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


def test_dispatcher_receipt_read_is_rejected_after_host_starts_closing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeOwnedRuntime()
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    dispatch_started = threading.Event()
    attempt_receipt = threading.Event()
    receipt_rejected = threading.Event()
    plane: Any = None

    def _dispatch(envelope: DbosControlEnvelope) -> ControlResult:
        dispatch_started.set()
        assert attempt_receipt.wait(timeout=3)
        try:
            plane.command_receipt(envelope.run_id, envelope.command_id)
        except RuntimeError as exc:
            assert "not accepting" in str(exc)
            receipt_rejected.set()
        return _ok(envelope)

    plane = _register_hosted_control_plane(host, _config(), _dispatch)
    host.launch()
    workflow = plane._workflow
    assert callable(workflow)
    payload = DbosControlEnvelope.from_control_command(
        ControlCommand(type="status", run_id="run", command_id="dispatch-close"),
        tenant_id="tenant",
        user_id="user",
    ).to_json()
    workflow_thread = threading.Thread(target=workflow, args=(payload,))
    workflow_thread.start()
    assert dispatch_started.wait(timeout=1)
    close_errors: list[BaseException] = []

    def _close() -> None:
        try:
            host.close()
        except BaseException as exc:
            close_errors.append(exc)

    close_thread = threading.Thread(target=_close)
    close_thread.start()
    deadline = time.monotonic() + 1
    while not plane._closing and time.monotonic() < deadline:
        time.sleep(0.001)
    assert plane._closing is True
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
def test_hosted_control_ownership_conflicts_fence_the_host_without_private_details(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    runtime = _FakeOwnedRuntime()
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    plane = _register_hosted_control_plane(host, _config(), _ok)
    host.launch()
    command = ControlCommand(type="status", run_id="run", command_id="owned")
    if operation == "enqueue":
        runtime.enqueue_conflict = True
    else:
        plane.enqueue_control(command, tenant_id="tenant", user_id="user")
        if operation == "retrieve":
            runtime.retrieve_conflict = True
        else:
            runtime.status_conflict = True

    with pytest.raises(DbosProcessOwnershipError, match="ownership changed") as raised:
        if operation == "enqueue":
            plane.enqueue_control(command, tenant_id="tenant", user_id="user")
        else:
            plane.command_receipt("run", "owned")

    assert "private" not in str(raised.value)
    assert host.state == "fenced"
    assert host.accepting is False


def test_hosted_control_prelaunch_registration_failure_rolls_back_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeOwnedRuntime()
    runtime.fail_workflow_registration = True
    _install_fake_runtime(monkeypatch, runtime)
    host = DbosRuntimeHost(_host_config())
    plane = _register_hosted_control_plane(host, _config(), _ok)

    with pytest.raises(RuntimeError, match="workflow registration failed"):
        host.launch()

    assert host.state == "closed"
    assert plane._closed is True
    assert runtime.released is True
    replacement = DbosRuntimeHost(_host_config())
    replacement.close()
