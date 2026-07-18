from __future__ import annotations

import asyncio
import importlib
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from monoid_agent_kernel.reference.dbos import _compat_226 as compat_module
from monoid_agent_kernel.reference.dbos._compat_226 import (
    _construct_owned_runtime_226,
    _DbosCleanupUncertain,
    _DbosConstructionUncertain,
    _DbosOwnershipConflict,
)

pytestmark = pytest.mark.serial

dbos = pytest.importorskip("dbos")
implementation = importlib.import_module("dbos._dbos")
real_dbos_class = dbos.DBOS


@pytest.fixture(autouse=True)
def _clean_dbos_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DBOS__CLOUD",
        "DBOS__CONDUCTOR_KEY",
        "DBOS__CONDUCTOR_URL",
        "DBOS__CONDUCTOR_APP_NAME",
    ):
        monkeypatch.delenv(name, raising=False)
    real_dbos_class.destroy(destroy_registry=True, workflow_completion_timeout_sec=0)
    yield
    real_dbos_class.destroy(destroy_registry=True, workflow_completion_timeout_sec=0)


def _config(tmp_path: Path, *, stem: str = "owned") -> SimpleNamespace:
    return SimpleNamespace(
        name=f"monoid-reference-{stem}",
        system_database_url=f"sqlite:///{tmp_path / f'{stem}.sqlite'}",
        application_version=f"monoid-reference-{stem}-v1",
        executor_id=f"stable-{stem}-slot",
    )


def _raw_config(tmp_path: Path, *, stem: str) -> dict[str, Any]:
    config = _config(tmp_path, stem=stem)
    return {
        "name": config.name,
        "system_database_url": config.system_database_url,
        "application_version": config.application_version,
        "executor_id": config.executor_id,
        "run_admin_server": False,
    }


def test_owned_adapter_runs_work_without_classmethod_dispatch(tmp_path: Path) -> None:
    owned = _construct_owned_runtime_226(dbos, _config(tmp_path))

    @owned.step(name="monoid.reference.adapter-step.v1", retries_allowed=False)
    def add_one(value: int) -> int:
        return value + 1

    @owned.workflow(name="monoid.reference.adapter-workflow.v1")
    def workflow(value: int) -> int:
        return add_one(value)

    queue_name = "monoid-reference-owned-v1"
    owned.listen_queues((queue_name,))
    owned.launch()
    queue = owned.register_queue(
        queue_name,
        worker_concurrency=1,
        concurrency=1,
        polling_interval_sec=0.01,
        on_conflict="always_update",
    )
    handle = owned.enqueue_workflow(queue_name, workflow, 41)
    retrieved = owned.retrieve_queue(queue_name)

    assert queue.name == queue_name
    assert queue._client_system_database is owned._runtime._sys_db
    assert retrieved._client_system_database is owned._runtime._sys_db
    assert retrieved.concurrency == 1
    assert handle.get_result() == 42
    owned.destroy(workflow_completion_timeout_sec=3, deadline=time.monotonic() + 3)
    assert getattr(implementation, "_dbos_global_instance", None) is None
    assert getattr(implementation, "_dbos_global_registry", None) is None


@pytest.mark.parametrize("decorator_name", ["step", "workflow"])
def test_owned_adapter_guards_decorator_application_after_replacement(
    tmp_path: Path,
    decorator_name: str,
) -> None:
    owned = _construct_owned_runtime_226(dbos, _config(tmp_path))
    original_names = set(owned._registry.workflow_info_map)
    decorator = getattr(owned, decorator_name)(name=f"monoid.reference.guarded-{decorator_name}.v1")
    implementation._dbos_global_instance = None
    implementation._dbos_global_registry = None
    foreign = real_dbos_class(config=_raw_config(tmp_path, stem=f"g{decorator_name}"))

    with pytest.raises(_DbosOwnershipConflict, match="ownership changed"):

        @decorator
        def guarded() -> str:
            return "guarded"

    assert set(owned._registry.workflow_info_map) == original_names
    assert getattr(implementation, "_dbos_global_instance", None) is foreign


def test_fresh_registry_probe_rejects_same_config_conductor_state(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runtime = real_dbos_class(
        config=_raw_config(tmp_path, stem="owned"),
        conductor_key="external-key",
    )

    with pytest.raises(_DbosConstructionUncertain, match="process identity changed"):
        compat_module._capture_fresh_registry(implementation, runtime, config)


@pytest.mark.parametrize("environment_name", ["DBOS__CLOUD", "DBOS__CONDUCTOR_KEY"])
def test_owned_adapter_rejects_cloud_and_conductor_before_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    environment_name: str,
) -> None:
    monkeypatch.setenv(environment_name, "true" if environment_name == "DBOS__CLOUD" else "key")

    with pytest.raises(_DbosOwnershipConflict, match="self-hosted mode"):
        _construct_owned_runtime_226(dbos, _config(tmp_path))

    assert getattr(implementation, "_dbos_global_instance", None) is None
    assert getattr(implementation, "_dbos_global_registry", None) is None


def test_owned_adapter_rejects_a_preexisting_registry_without_erasure(tmp_path: Path) -> None:
    @real_dbos_class.workflow(name="external.preexisting.workflow.v1")
    def external_workflow() -> str:
        return "external"

    del external_workflow
    external_registry = getattr(implementation, "_dbos_global_registry", None)

    with pytest.raises(_DbosOwnershipConflict, match="runtime or registry is active"):
        _construct_owned_runtime_226(dbos, _config(tmp_path))

    assert getattr(implementation, "_dbos_global_registry", None) is external_registry
    assert getattr(implementation, "_dbos_global_instance", None) is None


def test_owned_adapter_fences_a_same_config_runtime_created_during_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expected = _config(tmp_path)
    original_factory = compat_module._owned_runtime_type
    foreign_holder: dict[str, Any] = {}

    def _race_constructor(dbos_class: type[Any], marker: object) -> type[Any]:
        foreign_holder["runtime"] = dbos_class(config=_raw_config(tmp_path, stem="owned"))
        return original_factory(dbos_class, marker)

    monkeypatch.setattr(compat_module, "_owned_runtime_type", _race_constructor)

    with pytest.raises(_DbosConstructionUncertain, match="ownership is uncertain"):
        _construct_owned_runtime_226(dbos, expected)

    foreign = getattr(implementation, "_dbos_global_instance", None)
    assert foreign is foreign_holder["runtime"]
    assert foreign._config["name"] == expected.name
    assert foreign._config["system_database_url"] == expected.system_database_url


def test_owned_adapter_normalizes_postconstruction_probe_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fail_probe(implementation: Any, runtime: Any, config: Any) -> Any:
        del implementation, runtime, config
        raise KeyboardInterrupt

    monkeypatch.setattr(compat_module, "_capture_fresh_registry", _fail_probe)

    with pytest.raises(_DbosConstructionUncertain, match="validation failed"):
        _construct_owned_runtime_226(dbos, _config(tmp_path))

    assert getattr(implementation, "_dbos_global_instance", None) is not None
    assert getattr(implementation, "_dbos_global_registry", None) is not None


def test_owned_adapter_does_not_launch_a_replacement_created_during_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    owned = _construct_owned_runtime_226(dbos, _config(tmp_path))
    launch_entered = threading.Event()
    release_launch = threading.Event()
    failures: list[BaseException] = []

    def _blocked_launch() -> None:
        launch_entered.set()
        release_launch.wait(timeout=3)

    monkeypatch.setattr(owned._runtime, "_launch", _blocked_launch)

    def _launch() -> None:
        try:
            owned.launch()
        except BaseException as exc:
            failures.append(exc)

    launch_thread = threading.Thread(target=_launch)
    launch_thread.start()
    try:
        assert launch_entered.wait(timeout=3)
        implementation._dbos_global_instance = None
        implementation._dbos_global_registry = None
        foreign = real_dbos_class(config=_raw_config(tmp_path, stem="flaunch"))
        foreign_registry = getattr(implementation, "_dbos_global_registry", None)
        release_launch.set()
        launch_thread.join(timeout=3)

        assert launch_thread.is_alive() is False
        assert len(failures) == 1
        assert isinstance(failures[0], _DbosOwnershipConflict)
        assert foreign._launched is False
        assert foreign._listening_queues is None
        assert getattr(implementation, "_dbos_global_instance", None) is foreign
        assert getattr(implementation, "_dbos_global_registry", None) is foreign_registry
    finally:
        release_launch.set()
        launch_thread.join(timeout=3)
        owned._runtime._destroy(workflow_completion_timeout_sec=0)


def test_owned_adapter_does_not_destroy_a_replacement_created_during_destroy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    owned = _construct_owned_runtime_226(dbos, _config(tmp_path))
    destroy_entered = threading.Event()
    release_destroy = threading.Event()
    failures: list[BaseException] = []
    original_destroy = owned._runtime._destroy

    def _blocked_destroy(*, workflow_completion_timeout_sec: int) -> None:
        del workflow_completion_timeout_sec
        destroy_entered.set()
        release_destroy.wait(timeout=3)

    monkeypatch.setattr(owned._runtime, "_destroy", _blocked_destroy)

    def _destroy() -> None:
        try:
            owned.destroy(workflow_completion_timeout_sec=1, deadline=time.monotonic() + 3)
        except BaseException as exc:
            failures.append(exc)

    destroy_thread = threading.Thread(target=_destroy)
    destroy_thread.start()
    try:
        assert destroy_entered.wait(timeout=3)
        implementation._dbos_global_instance = None
        implementation._dbos_global_registry = None
        foreign = real_dbos_class(config=_raw_config(tmp_path, stem="fdestroy"))
        foreign_registry = getattr(implementation, "_dbos_global_registry", None)
        release_destroy.set()
        destroy_thread.join(timeout=3)

        assert destroy_thread.is_alive() is False
        assert len(failures) == 1
        assert isinstance(failures[0], _DbosCleanupUncertain)
        assert foreign._initialized is True
        assert getattr(implementation, "_dbos_global_instance", None) is foreign
        assert getattr(implementation, "_dbos_global_registry", None) is foreign_registry
    finally:
        release_destroy.set()
        destroy_thread.join(timeout=3)
        monkeypatch.setattr(owned._runtime, "_destroy", original_destroy)
        original_destroy(workflow_completion_timeout_sec=0)


def test_owned_adapter_fences_a_nested_dbos_thread_that_survives_destroy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    owned = _construct_owned_runtime_226(dbos, _config(tmp_path))
    real_enumerate = threading.enumerate
    survivor = SimpleNamespace(
        name="queue-worker-host-test",
        _target=None,
        is_alive=lambda: True,
    )
    monkeypatch.setattr(threading, "enumerate", lambda: [*real_enumerate(), survivor])

    with pytest.raises(_DbosCleanupUncertain, match="survived the shutdown grace"):
        owned.destroy(workflow_completion_timeout_sec=0, deadline=time.monotonic() + 0.1)

    assert getattr(implementation, "_dbos_global_instance", None) is owned._runtime
    assert getattr(implementation, "_dbos_global_registry", None) is owned._registry


def test_owned_adapter_allows_an_unrelated_preexisting_queue_worker_name(
    tmp_path: Path,
) -> None:
    thread_entered = threading.Event()
    release_thread = threading.Event()

    def _preexisting_work() -> None:
        thread_entered.set()
        release_thread.wait(timeout=5)

    preexisting = threading.Thread(
        target=_preexisting_work,
        name="queue-worker-preexisting-runtime",
        daemon=True,
    )
    preexisting.start()
    assert thread_entered.wait(timeout=3)
    try:
        owned = _construct_owned_runtime_226(dbos, _config(tmp_path, stem="preexisting"))
        owned.listen_queues(())
        owned.launch()
        owned.destroy(workflow_completion_timeout_sec=0, deadline=time.monotonic() + 3)

        assert getattr(implementation, "_dbos_global_instance", None) is None
        assert getattr(implementation, "_dbos_global_registry", None) is None
        assert preexisting.is_alive()
    finally:
        release_thread.set()
        preexisting.join(timeout=3)


def test_owned_adapter_rejects_preexisting_dbos_thread_provenance(tmp_path: Path) -> None:
    thread_entered = threading.Event()
    release_thread = threading.Event()

    def _preexisting_dbos_work() -> None:
        thread_entered.set()
        release_thread.wait(timeout=5)

    _preexisting_dbos_work.__module__ = "dbos.synthetic"
    preexisting = threading.Thread(target=_preexisting_dbos_work, daemon=True)
    preexisting.start()
    assert thread_entered.wait(timeout=3)
    try:
        with pytest.raises(_DbosOwnershipConflict, match="worker threads are active"):
            _construct_owned_runtime_226(dbos, _config(tmp_path, stem="dbos-thread"))

        assert getattr(implementation, "_dbos_global_instance", None) is None
        assert getattr(implementation, "_dbos_global_registry", None) is None
        assert preexisting.is_alive()
    finally:
        release_thread.set()
        preexisting.join(timeout=3)


def test_owned_adapter_fences_an_asyncio_worker_that_survives_destroy(tmp_path: Path) -> None:
    owned = _construct_owned_runtime_226(dbos, _config(tmp_path, stem="asyncworker"))
    worker_entered = threading.Event()
    release_worker = threading.Event()

    def _blocking_work() -> str:
        worker_entered.set()
        release_worker.wait(timeout=5)
        return "done"

    @owned.workflow(name="monoid.reference.async-worker-workflow.v1")
    async def async_workflow() -> str:
        return await asyncio.to_thread(_blocking_work)

    queue_name = "monoid-reference-async-worker-v1"
    owned.listen_queues((queue_name,))
    owned.launch()
    owned.register_queue(
        queue_name,
        worker_concurrency=1,
        concurrency=1,
        polling_interval_sec=0.01,
        on_conflict="always_update",
    )
    owned.enqueue_workflow(queue_name, async_workflow)
    assert worker_entered.wait(timeout=3)

    try:
        with pytest.raises(_DbosCleanupUncertain, match="survived the shutdown grace"):
            owned.destroy(workflow_completion_timeout_sec=0, deadline=time.monotonic() + 2)

        assert getattr(implementation, "_dbos_global_instance", None) is owned._runtime
        assert getattr(implementation, "_dbos_global_registry", None) is owned._registry
    finally:
        release_worker.set()
        deadline = time.monotonic() + 3
        while (
            any(
                thread.name.startswith("monoid-dbos-asyncio") and thread.is_alive()
                for thread in threading.enumerate()
            )
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        if owned.owns_globals():
            real_dbos_class.destroy(
                destroy_registry=True,
                workflow_completion_timeout_sec=0,
            )


def test_owned_adapter_tracks_an_async_worker_created_during_destroy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    owned = _construct_owned_runtime_226(dbos, _config(tmp_path, stem="asyncrace"))
    owned.listen_queues(())
    owned.launch()
    executor = owned._async_executor
    assert executor is not None
    assert not executor._threads
    worker_entered = threading.Event()
    release_worker = threading.Event()
    original_destroy = owned._runtime._destroy

    def _blocking_work() -> None:
        worker_entered.set()
        release_worker.wait(timeout=5)

    def _spawn_during_destroy(*, workflow_completion_timeout_sec: int) -> None:
        executor.submit(_blocking_work)
        assert worker_entered.wait(timeout=3)
        original_destroy(workflow_completion_timeout_sec=workflow_completion_timeout_sec)

    monkeypatch.setattr(owned._runtime, "_destroy", _spawn_during_destroy)
    try:
        with pytest.raises(_DbosCleanupUncertain, match="survived the shutdown grace"):
            owned.destroy(workflow_completion_timeout_sec=0, deadline=time.monotonic() + 2)

        assert getattr(implementation, "_dbos_global_instance", None) is owned._runtime
        assert getattr(implementation, "_dbos_global_registry", None) is owned._registry
    finally:
        monkeypatch.setattr(owned._runtime, "_destroy", original_destroy)
        release_worker.set()
        deadline = time.monotonic() + 3
        while (
            any(thread.is_alive() for thread in executor._threads) and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        if owned.owns_globals():
            real_dbos_class.destroy(
                destroy_registry=True,
                workflow_completion_timeout_sec=0,
            )
