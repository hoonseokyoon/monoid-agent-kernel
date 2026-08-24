"""Managed Workflow and threaded Activity worker composition for Temporal."""

from __future__ import annotations

import math
from contextlib import AsyncExitStack
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any

from monoid_agent_kernel.core.safe_evidence import is_safe_opaque_id

from .activity import TemporalActivationActivity
from .dependency import TemporalDependencyMissing
from .workflow import TemporalRunWorkflow

try:
    from temporalio.worker import Worker
except ImportError as exc:  # pragma: no cover - exercised by isolated import tests
    raise TemporalDependencyMissing(
        "install monoid-agent-kernel[temporal] to compose Temporal workers"
    ) from exc


class TemporalWorkerGroup:
    """Own separate Workflow and Activity workers plus an optional thread pool."""

    def __init__(
        self,
        *,
        client: Any,
        workflow_task_queue: str,
        activity_task_queue: str,
        activation_activity: TemporalActivationActivity,
        max_concurrent_activities: int = 10,
        graceful_shutdown_timeout_s: float = 30.0,
        activity_executor: ThreadPoolExecutor | None = None,
    ) -> None:
        if not is_safe_opaque_id(workflow_task_queue) or len(workflow_task_queue) > 255:
            raise ValueError("Temporal Workflow task queue must be a bounded opaque id")
        if not is_safe_opaque_id(activity_task_queue) or len(activity_task_queue) > 255:
            raise ValueError("Temporal Activity task queue must be a bounded opaque id")
        if not isinstance(activation_activity, TemporalActivationActivity):
            raise TypeError("activation_activity must be TemporalActivationActivity")
        if type(max_concurrent_activities) is not int or max_concurrent_activities < 1:
            raise ValueError("max_concurrent_activities must be a positive integer")
        if (
            type(graceful_shutdown_timeout_s) not in {int, float}
            or isinstance(graceful_shutdown_timeout_s, bool)
            or not math.isfinite(float(graceful_shutdown_timeout_s))
            or float(graceful_shutdown_timeout_s) < 0
            or float(graceful_shutdown_timeout_s) > 7 * 24 * 60 * 60
        ):
            raise ValueError("graceful_shutdown_timeout_s must be non-negative")
        if activity_executor is not None and not isinstance(
            activity_executor, ThreadPoolExecutor
        ):
            raise TypeError("threaded Temporal Activities require ThreadPoolExecutor")
        if activity_executor is not None:
            executor_capacity = getattr(activity_executor, "_max_workers", None)
            if type(executor_capacity) is not int or executor_capacity < max_concurrent_activities:
                raise ValueError(
                    "activity_executor capacity must cover max_concurrent_activities"
                )
            if getattr(activity_executor, "_shutdown", True):
                raise ValueError("activity_executor must be active")
        minimum_graceful_timeout = max(
            float(activation_activity.policy.heartbeat_interval_s),
            float(activation_activity.policy.supervisor_join_timeout_s),
        )
        if float(graceful_shutdown_timeout_s) < minimum_graceful_timeout:
            raise ValueError(
                "graceful_shutdown_timeout_s must cover heartbeat and supervisor shutdown"
            )

        self._owns_executor = activity_executor is None
        self._activity_executor = activity_executor or ThreadPoolExecutor(
            max_workers=max_concurrent_activities,
            thread_name_prefix="monoid-temporal-activity",
        )
        heartbeat_interval = timedelta(
            seconds=float(activation_activity.policy.heartbeat_interval_s)
        )
        graceful_timeout = timedelta(seconds=float(graceful_shutdown_timeout_s))
        try:
            self.activity_worker = Worker(
                client,
                task_queue=activity_task_queue,
                activities=[activation_activity.run],
                activity_executor=self._activity_executor,
                max_concurrent_activities=max_concurrent_activities,
                max_heartbeat_throttle_interval=heartbeat_interval,
                default_heartbeat_throttle_interval=heartbeat_interval,
                graceful_shutdown_timeout=graceful_timeout,
            )
            self.workflow_worker = Worker(
                client,
                task_queue=workflow_task_queue,
                workflows=[TemporalRunWorkflow],
            )
        except BaseException:
            if self._owns_executor:
                self._activity_executor.shutdown(wait=False, cancel_futures=True)
            raise
        self._stack: AsyncExitStack | None = None
        self._closed = False

    async def __aenter__(self) -> TemporalWorkerGroup:
        if self._stack is not None:
            raise RuntimeError("TemporalWorkerGroup is already running")
        if self._closed:
            raise RuntimeError("TemporalWorkerGroup cannot be restarted")
        stack = AsyncExitStack()
        await stack.__aenter__()
        if self._owns_executor:
            stack.callback(
                self._activity_executor.shutdown,
                wait=False,
                cancel_futures=True,
            )
        try:
            await stack.enter_async_context(self.activity_worker)
            await stack.enter_async_context(self.workflow_worker)
        except BaseException:
            self._closed = True
            await stack.aclose()
            raise
        self._stack = stack
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        stack = self._stack
        if stack is None:
            raise RuntimeError("TemporalWorkerGroup was not started")
        self._stack = None
        self._closed = True
        return await stack.__aexit__(*exc_info)


__all__ = ["TemporalWorkerGroup"]
