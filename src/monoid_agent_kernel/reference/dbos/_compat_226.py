"""Identity-safe instance adapter for the pinned DBOS 2.26 Reference runtime."""

from __future__ import annotations

import asyncio
import importlib
import math
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

_DBOS_ADAPTER_LOCK = threading.RLock()
_TERMINAL_WORKFLOW_STATUSES = frozenset(
    {
        "SUCCESS",
        "ERROR",
        "CANCELLED",
        "MAX_RECOVERY_ATTEMPTS_EXCEEDED",
    }
)


class _DbosOwnershipConflict(RuntimeError):
    """Raised before the adapter mutates a runtime owned by another accessor."""


class _DbosConstructionUncertain(RuntimeError):
    """Raised after DBOS construction may have populated process-global state."""


class _DbosCleanupUncertain(RuntimeError):
    """Raised when stopped state cannot be proven before ownership release."""


class _OwnedDbosWorkflowHandle226:
    """Workflow handle whose status and result access stay bound to one owned runtime."""

    def __init__(
        self,
        adapter: _OwnedDbosRuntime226,
        raw_handle: Any,
        workflow_id: str,
    ) -> None:
        self._adapter = adapter
        self._raw_handle = raw_handle
        self.workflow_id = workflow_id

    def get_workflow_id(self) -> str:
        return self.workflow_id

    def get_result(self, *, polling_interval_sec: float = 1.0) -> Any:
        if not math.isfinite(polling_interval_sec) or polling_interval_sec <= 0:
            raise ValueError("DBOS workflow result polling interval must be positive")
        while True:
            status = self.get_status()
            status_value = getattr(status.status, "value", status.status)
            if str(status_value) in _TERMINAL_WORKFLOW_STATUSES:
                break
            time.sleep(polling_interval_sec)
        with _DBOS_ADAPTER_LOCK:
            self._adapter.require_owned()
            try:
                result = self._raw_handle.get_result(
                    polling_interval_sec=polling_interval_sec
                )
            except BaseException:
                self._adapter._require_owned_after_failure_unlocked()
                raise
            self._adapter.require_owned()
            return result

    def get_status(self) -> Any:
        with _DBOS_ADAPTER_LOCK:
            self._adapter.require_owned()
            try:
                status = self._adapter._workflow_status_unlocked(self.workflow_id)
            except BaseException:
                self._adapter._require_owned_after_failure_unlocked()
                raise
            self._adapter.require_owned()
            if status is None:
                raise self._adapter._implementation.DBOSNonExistentWorkflowError(
                    "target", self.workflow_id
                )
            return status


class _OwnedDbosRuntime226:
    """Narrow DBOS 2.26 facade bound to one captured singleton and registry.

    DBOS exposes lifecycle and queue operations as classmethods routed through mutable globals.
    This adapter invokes the captured instance and registry directly, then verifies global identity
    at every boundary. Every Reference DBOS accessor must use this adapter for the process lifetime.
    A detected direct-access violation fences the caller without acting on the replacement runtime.
    """

    def __init__(
        self,
        dbos_module: Any,
        runtime: Any,
        registry: Any,
        config: Any,
        *,
        preexisting_name_collisions: tuple[Any, ...],
    ) -> None:
        self._dbos_module = dbos_module
        self._implementation = importlib.import_module("dbos._dbos")
        self._runtime = runtime
        self._registry = registry
        self._application_version = config.application_version
        self._executor_id = config.executor_id
        self._preexisting_name_collisions = preexisting_name_collisions
        self._async_executor: ThreadPoolExecutor | None = None

    def require_owned(self) -> None:
        with _DBOS_ADAPTER_LOCK:
            if not self._owns_globals_unlocked():
                raise _DbosOwnershipConflict(
                    "DBOS process-global runtime ownership changed; terminate the process"
                )

    def _require_owned_after_failure_unlocked(self) -> None:
        if not self._owns_globals_unlocked():
            raise _DbosOwnershipConflict(
                "DBOS process-global runtime ownership changed; terminate the process"
            ) from None

    def _require_cleanup_owned(self) -> None:
        if not self._owns_globals_unlocked():
            raise _DbosCleanupUncertain(
                "DBOS global identity changed during shutdown; terminate the process"
            )

    def owns_globals(self) -> bool:
        with _DBOS_ADAPTER_LOCK:
            return self._owns_globals_unlocked()

    def require_released(self) -> None:
        """Verify no DBOS singleton or registry was populated after owned shutdown."""

        with _DBOS_ADAPTER_LOCK:
            _require_globals_cleared(self._implementation)

    def _owns_globals_unlocked(self) -> bool:
        implementation = self._implementation
        return (
            getattr(self._runtime, "_registry", None) is self._registry
            and getattr(self._registry, "dbos", None) is self._runtime
            and getattr(implementation, "_dbos_global_instance", None) is self._runtime
            and getattr(implementation, "_dbos_global_registry", None) is self._registry
            and getattr(implementation.GlobalParams, "app_version", None)
            == self._application_version
            and getattr(implementation.GlobalParams, "executor_id", None) == self._executor_id
            and not bool(getattr(implementation.GlobalParams, "dbos_cloud", False))
            and getattr(self._runtime, "conductor_key", None) is None
        )

    def step(
        self,
        *,
        name: str | None = None,
        retries_allowed: bool = False,
        interval_seconds: float = 1.0,
        max_attempts: int = 3,
        backoff_rate: float = 2.0,
        should_retry: Callable[[BaseException], Any] | None = None,
        preemptible: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        with _DBOS_ADAPTER_LOCK:
            self.require_owned()
            decorator = self._implementation.decorate_step(
                self._registry,
                name=name,
                retries_allowed=retries_allowed,
                interval_seconds=interval_seconds,
                max_attempts=max_attempts,
                backoff_rate=backoff_rate,
                should_retry=should_retry,
                preemptible=preemptible,
            )

        def _guarded(func: Callable[..., Any]) -> Callable[..., Any]:
            with _DBOS_ADAPTER_LOCK:
                self.require_owned()
                registered = decorator(func)
                self.require_owned()
                return registered

        return _guarded

    def workflow(
        self,
        *,
        name: str | None = None,
        max_recovery_attempts: int | None = 100,
        serialization_type: Any = None,
        validate_args: Callable[..., Any] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        with _DBOS_ADAPTER_LOCK:
            self.require_owned()
            decorator = self._implementation.decorate_workflow(
                self._registry,
                name,
                max_recovery_attempts,
                serialization_type=serialization_type,
                validate_args=validate_args,
            )

        def _guarded(func: Callable[..., Any]) -> Callable[..., Any]:
            with _DBOS_ADAPTER_LOCK:
                self.require_owned()
                registered = decorator(func)
                self.require_owned()
                return registered

        return _guarded

    def listen_queues(self, queues: tuple[str, ...]) -> None:
        with _DBOS_ADAPTER_LOCK:
            self.require_owned()
            if self._runtime._launched:
                raise RuntimeError("listen_queues called after DBOS is launched")
            if self._runtime._listening_queues is not None:
                raise RuntimeError("listen_queues called more than once")
            self._runtime._listening_queues = list(queues)
            self.require_owned()

    def preflight_launch(self) -> None:
        """Reject an asynchronous lifecycle caller before launch mutates DBOS state."""

        with _DBOS_ADAPTER_LOCK:
            self.require_owned()
            self._require_synchronous_launch()

    def launch(self) -> None:
        with _DBOS_ADAPTER_LOCK:
            self.require_owned()
            self._require_synchronous_launch()
            if not self._runtime._launched and self._async_executor is None:
                background_event_loop = self._runtime._background_event_loop
                background_event_loop.start()
                background_loop = getattr(background_event_loop, "_loop", None)
                if background_loop is None:
                    raise _DbosOwnershipConflict(
                        "DBOS background event loop is unavailable; terminate the process"
                    )
                self._async_executor = ThreadPoolExecutor(thread_name_prefix="monoid-dbos-asyncio")
                background_loop.set_default_executor(self._async_executor)
            self._runtime._launch()
            self.require_owned()

    @staticmethod
    def _require_synchronous_launch() -> None:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is not None:
            raise _DbosOwnershipConflict(
                "DBOS Reference launch requires a dedicated synchronous lifecycle thread"
            )

    def register_queue(
        self,
        name: str,
        *,
        worker_concurrency: int | None = None,
        concurrency: int | None = None,
        limiter: Any = None,
        priority_enabled: bool = False,
        partition_queue: bool = False,
        polling_interval_sec: float = 1.0,
        on_conflict: str = "update_if_latest_version",
    ) -> Any:
        with _DBOS_ADAPTER_LOCK:
            self.require_owned()
            self._implementation.check_async("register_queue")
            queue_type = self._implementation.Queue
            queue_type._validate_queue(
                concurrency=concurrency,
                worker_concurrency=worker_concurrency,
                polling_interval_sec=polling_interval_sec,
                limiter=limiter,
            )
            if on_conflict == "always_update":
                update_existing = True
            elif on_conflict == "never_update":
                update_existing = False
            else:
                latest = self._runtime._sys_db.get_latest_application_version()
                update_existing = latest["version_name"] == self._application_version
            self._runtime._sys_db.upsert_queue(
                name=name,
                concurrency=concurrency,
                worker_concurrency=worker_concurrency,
                rate_limit_max=limiter["limit"] if limiter else None,
                rate_limit_period_sec=limiter["period"] if limiter else None,
                priority_enabled=priority_enabled,
                partition_queue=partition_queue,
                polling_interval_sec=polling_interval_sec,
                update_existing=update_existing,
            )
            queue = self._runtime._sys_db.get_queue(
                name,
                client_system_database=self._runtime._sys_db,
            )
            if queue is None:
                raise RuntimeError("DBOS queue is missing after registration")
            self.require_owned()
            return queue

    def retrieve_queue(self, name: str) -> Any:
        with _DBOS_ADAPTER_LOCK:
            self.require_owned()
            self._implementation.check_async("retrieve_queue")
            queue = self._runtime._sys_db.get_queue(
                name,
                client_system_database=self._runtime._sys_db,
            )
            self.require_owned()
            return queue

    def enqueue_workflow(
        self,
        queue_name: str,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with _DBOS_ADAPTER_LOCK:
            self.require_owned()
            try:
                raw_handle = self._enqueue_workflow_unlocked(queue_name, func, args, kwargs)
                handle = self._wrap_workflow_handle_unlocked(raw_handle)
            except BaseException:
                self._require_owned_after_failure_unlocked()
                raise
            self.require_owned()
            return handle

    def enqueue_workflow_with_identity(
        self,
        queue_name: str,
        func: Callable[..., Any],
        *args: Any,
        workflow_id: str,
        queue_partition_key: str,
        **kwargs: Any,
    ) -> Any:
        """Enqueue on the captured runtime with explicit durable and partition identities."""

        if type(workflow_id) is not str or not workflow_id:
            raise ValueError("DBOS workflow_id is required")
        if type(queue_partition_key) is not str or not queue_partition_key:
            raise ValueError("DBOS queue_partition_key is required")
        with _DBOS_ADAPTER_LOCK:
            self.require_owned()
            if self._implementation.get_local_dbos_context() is not None:
                raise RuntimeError(
                    "DBOS identity-scoped enqueue requires no ambient DBOS context"
                )
            try:
                with self._dbos_module.SetWorkflowID(
                    workflow_id
                ), self._dbos_module.SetEnqueueOptions(
                    queue_partition_key=queue_partition_key
                ):
                    raw_handle = self._enqueue_workflow_unlocked(queue_name, func, args, kwargs)
                handle = self._wrap_workflow_handle_unlocked(
                    raw_handle,
                    expected_workflow_id=workflow_id,
                )
            except BaseException:
                self._require_owned_after_failure_unlocked()
                raise
            self.require_owned()
            return handle

    def retrieve_workflow(self, workflow_id: str) -> Any:
        """Return a polling handle bound to the captured runtime, never the global singleton."""

        if type(workflow_id) is not str or not workflow_id:
            raise ValueError("DBOS workflow_id is required")
        with _DBOS_ADAPTER_LOCK:
            self.require_owned()
            try:
                self._implementation.check_async("retrieve_workflow")
                status = self._workflow_status_unlocked(workflow_id)
                if status is None:
                    raise self._implementation.DBOSNonExistentWorkflowError(
                        "target", workflow_id
                    )
                raw_handle = self._implementation.WorkflowHandlePolling(
                    workflow_id, self._runtime
                )
                handle = self._wrap_workflow_handle_unlocked(
                    raw_handle,
                    expected_workflow_id=workflow_id,
                )
            except BaseException:
                self._require_owned_after_failure_unlocked()
                raise
            self.require_owned()
            return handle

    def _enqueue_workflow_unlocked(
        self,
        queue_name: str,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        queue = self._implementation.Queue(queue_name, database_backed_queue=True)
        queue._validate_enqueue(self._implementation.get_local_dbos_context())
        return self._implementation.start_workflow(
            self._runtime,
            func,
            args,
            kwargs,
            queue_name=queue_name,
            execute_workflow=False,
        )

    def _workflow_status_unlocked(self, workflow_id: str) -> Any:
        self._implementation.check_async("get_workflow_status")
        system_database = self._runtime._sys_db

        def _load_status() -> Any:
            return self._implementation.get_workflow(system_database, workflow_id)

        return system_database.call_function_as_step(
            _load_status,
            "DBOS.getStatus",
            self._implementation.snapshot_step_context(reserve_sleep_id=False),
        )

    def _wrap_workflow_handle_unlocked(
        self,
        raw_handle: Any,
        *,
        expected_workflow_id: str | None = None,
    ) -> _OwnedDbosWorkflowHandle226:
        workflow_id = getattr(raw_handle, "workflow_id", None)
        if (
            type(workflow_id) is not str
            or not workflow_id
            or getattr(raw_handle, "dbos", None) is not self._runtime
            or (expected_workflow_id is not None and workflow_id != expected_workflow_id)
        ):
            raise _DbosOwnershipConflict(
                "DBOS returned an unexpected workflow handle; terminate the process"
            )
        return _OwnedDbosWorkflowHandle226(self, raw_handle, workflow_id)

    def destroy(self, *, workflow_completion_timeout_sec: int, deadline: float) -> None:
        """Stop the captured runtime; the host enforces an outer deadline around this call."""

        with _DBOS_ADAPTER_LOCK:
            self.require_owned()
            runtime_executors = tuple(
                executor for executor in (self._async_executor,) if executor is not None
            )
            runtime_threads = _capture_runtime_threads(
                self._runtime,
                runtime_executors=runtime_executors,
                preexisting_name_collisions=self._preexisting_name_collisions,
            )
            try:
                self._runtime._destroy(
                    workflow_completion_timeout_sec=workflow_completion_timeout_sec
                )
            except BaseException:
                raise _DbosCleanupUncertain(
                    "DBOS instance shutdown failed; terminate the process"
                ) from None
            for runtime_executor in runtime_executors:
                runtime_executor.shutdown(wait=False, cancel_futures=True)
            if time.monotonic() > deadline:
                raise _DbosCleanupUncertain(
                    "DBOS instance shutdown exceeded its deadline; terminate the process"
                )
            self._require_cleanup_owned()
            _wait_runtime_stopped(
                self._runtime,
                runtime_threads=runtime_threads,
                runtime_executors=runtime_executors,
                preexisting_name_collisions=self._preexisting_name_collisions,
                deadline=deadline,
            )
            self._require_cleanup_owned()
            _clear_owned_globals(self._implementation, self._runtime, self._registry)
            _require_globals_cleared(self._implementation)


def _construct_owned_runtime_226(dbos_module: Any, config: Any) -> _OwnedDbosRuntime226:
    """Construct a fresh self-hosted DBOS 2.26 runtime and capture its exact identity."""

    with _DBOS_ADAPTER_LOCK:
        _require_self_hosted(dbos_module)
        implementation = importlib.import_module("dbos._dbos")
        if (
            getattr(implementation, "_dbos_global_instance", None) is not None
            or getattr(implementation, "_dbos_global_registry", None) is not None
        ):
            raise _DbosOwnershipConflict(
                "an existing DBOS runtime or registry is active; shared host ownership is required"
            )
        preexisting_name_collisions = _classify_preexisting_threads()
        marker = object()
        owned_runtime_type = _owned_runtime_type(dbos_module.DBOS, marker)
        try:
            runtime = owned_runtime_type(
                config={
                    "name": config.name,
                    "system_database_url": config.system_database_url,
                    "application_version": config.application_version,
                    "executor_id": config.executor_id,
                    "run_admin_server": False,
                }
            )
        except BaseException:
            raise _DbosConstructionUncertain(
                "DBOS construction is uncertain; terminate the process"
            ) from None
        if getattr(runtime, "_monoid_owned_construction_marker", None) is not marker:
            raise _DbosConstructionUncertain(
                "DBOS construction ownership is uncertain; terminate the process"
            )
        try:
            registry = _capture_fresh_registry(implementation, runtime, config)
        except _DbosConstructionUncertain:
            raise
        except BaseException:
            raise _DbosConstructionUncertain(
                "DBOS construction validation failed; terminate the process"
            ) from None
        return _OwnedDbosRuntime226(
            dbos_module,
            runtime,
            registry,
            config,
            preexisting_name_collisions=preexisting_name_collisions,
        )


def _owned_runtime_type(dbos_class: type[Any], marker: object) -> type[Any]:
    class _OwnedDbosConstruction(dbos_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if not bool(getattr(self, "_initialized", False)):
                self._monoid_owned_construction_marker = marker
            super().__init__(*args, **kwargs)

    return _OwnedDbosConstruction


def _require_self_hosted(dbos_module: Any) -> None:
    conductor_env = (
        "DBOS__CONDUCTOR_KEY",
        "DBOS__CONDUCTOR_URL",
        "DBOS__CONDUCTOR_APP_NAME",
    )
    if os.environ.get("DBOS__CLOUD") == "true" or any(
        os.environ.get(name) for name in conductor_env
    ):
        raise _DbosOwnershipConflict(
            "DBOS Reference host requires self-hosted mode without Conductor"
        )
    implementation = importlib.import_module("dbos._dbos")
    if bool(getattr(getattr(implementation, "GlobalParams", None), "dbos_cloud", False)):
        raise _DbosOwnershipConflict(
            "DBOS Reference host requires self-hosted mode without Conductor"
        )
    del dbos_module


def _capture_fresh_registry(implementation: Any, runtime: Any, config: Any) -> Any:
    registry = getattr(runtime, "_registry", None)
    runtime_config = getattr(runtime, "_config", {})
    runtime_options = runtime_config.get("runtimeConfig", {})
    expected_internal_workflows = {
        getattr(implementation, "TEMP_SEND_WF_NAME", None),
        getattr(implementation, "DEBOUNCER_WORKFLOW_NAME", None),
    }
    if (
        registry is None
        or getattr(registry, "dbos", None) is not runtime
        or getattr(implementation, "_dbos_global_instance", None) is not runtime
        or getattr(implementation, "_dbos_global_registry", None) is not registry
        or set(getattr(registry, "workflow_info_map", ())) != expected_internal_workflows
        or set(getattr(registry, "function_type_map", ())) != expected_internal_workflows
        or bool(getattr(registry, "class_info_map", ()))
        or bool(getattr(registry, "instance_info_map", ()))
        or bool(getattr(registry, "queue_info_map", ()))
        or bool(getattr(registry, "pollers", ()))
        or runtime_config.get("name") != config.name
        or runtime_config.get("system_database_url") != config.system_database_url
        or runtime_options.get("run_admin_server") is not False
        or getattr(runtime, "conductor_key", None) is not None
        or getattr(runtime, "conductor_url", None) is not None
        or getattr(implementation.GlobalParams, "app_version", None) != config.application_version
        or getattr(implementation.GlobalParams, "executor_id", None) != config.executor_id
        or bool(getattr(implementation.GlobalParams, "dbos_cloud", False))
    ):
        raise _DbosConstructionUncertain(
            "DBOS registry or process identity changed during construction; terminate the process"
        )
    return registry


def _clear_owned_globals(implementation: Any, runtime: Any, registry: Any) -> None:
    if (
        getattr(implementation, "_dbos_global_instance", None) is not runtime
        or getattr(implementation, "_dbos_global_registry", None) is not registry
        or getattr(registry, "dbos", None) is not runtime
    ):
        raise _DbosCleanupUncertain(
            "DBOS global identity changed during shutdown; terminate the process"
        )
    implementation.GlobalParams.app_version = os.environ.get("DBOS__APPVERSION", "")
    implementation.GlobalParams.executor_id = os.environ.get("DBOS__VMID", "local")
    implementation._dbos_global_instance = None
    implementation._dbos_global_registry = None


def _require_globals_cleared(implementation: Any) -> None:
    if (
        getattr(implementation, "_dbos_global_instance", None) is not None
        or getattr(implementation, "_dbos_global_registry", None) is not None
    ):
        raise _DbosCleanupUncertain(
            "DBOS process-global state survived shutdown; terminate the process"
        )


def _capture_runtime_threads(
    runtime: Any,
    *,
    runtime_executors: tuple[Any, ...] = (),
    preexisting_name_collisions: tuple[Any, ...] = (),
) -> tuple[Any, ...]:
    threads: list[Any] = list(getattr(runtime, "_background_threads", ()))
    event_loop = getattr(runtime, "_background_event_loop", None)
    event_thread = getattr(event_loop, "_thread", None)
    if event_thread is not None:
        threads.append(event_thread)
    background_loop = getattr(event_loop, "_loop", None)
    default_executor = getattr(background_loop, "_default_executor", None)
    threads.extend(getattr(default_executor, "_threads", ()))
    executor = getattr(runtime, "_executor_field", None)
    threads.extend(getattr(executor, "_threads", ()))
    for runtime_executor in runtime_executors:
        threads.extend(getattr(runtime_executor, "_threads", ()))
    threads.extend(
        thread
        for thread in threading.enumerate()
        if _is_dbos_owned_thread(thread)
        and all(
            thread is not collision
            for collision in preexisting_name_collisions
        )
    )
    return _unique_threads(threads)


def _classify_preexisting_threads() -> tuple[Any, ...]:
    name_collisions: list[Any] = []
    for thread in threading.enumerate():
        name = str(getattr(thread, "name", ""))
        if _has_dbos_thread_provenance(thread) or name.startswith(
            ("dbos-executor", "monoid-dbos-asyncio")
        ):
            raise _DbosOwnershipConflict(
                "preexisting DBOS worker threads are active; process restart is required"
            )
        if name.startswith("queue-worker-"):
            name_collisions.append(thread)
    return tuple(name_collisions)


def _is_dbos_owned_thread(thread: Any) -> bool:
    name = str(getattr(thread, "name", ""))
    if name.startswith(("queue-worker-", "dbos-executor", "monoid-dbos-asyncio")):
        return True
    return _has_dbos_thread_provenance(thread)


def _has_dbos_thread_provenance(thread: Any) -> bool:
    target = getattr(thread, "_target", None)
    target_module = str(getattr(target, "__module__", ""))
    owner = getattr(target, "__self__", None)
    owner_module = str(getattr(getattr(owner, "__class__", None), "__module__", ""))
    return target_module.startswith("dbos.") or owner_module.startswith("dbos.")


def _unique_threads(threads: list[Any]) -> tuple[Any, ...]:
    unique: dict[int, Any] = {}
    for thread in threads:
        unique[id(thread)] = thread
    return tuple(unique.values())


def _wait_runtime_stopped(
    runtime: Any,
    *,
    runtime_threads: tuple[Any, ...],
    runtime_executors: tuple[Any, ...],
    preexisting_name_collisions: tuple[Any, ...],
    deadline: float,
) -> None:
    active_set = getattr(runtime, "_active_workflows_set", None)
    active_list = getattr(active_set, "activeList", None)
    if not callable(active_list):
        raise _DbosCleanupUncertain(
            "DBOS active-workflow state is unavailable; terminate the process"
        )
    observed_threads = runtime_threads
    while True:
        try:
            active_workflows = tuple(active_list())
            discovered = list(
                _capture_runtime_threads(
                    runtime,
                    runtime_executors=runtime_executors,
                    preexisting_name_collisions=preexisting_name_collisions,
                )
            )
            observed_threads = _unique_threads([*observed_threads, *discovered])
            live_threads = any(thread.is_alive() for thread in observed_threads)
        except BaseException:
            raise _DbosCleanupUncertain(
                "DBOS stopped-state probe failed; terminate the process"
            ) from None
        if not active_workflows and not live_threads:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _DbosCleanupUncertain(
                "DBOS work or threads survived the shutdown grace; terminate the process"
            )
        time.sleep(min(0.01, remaining))
