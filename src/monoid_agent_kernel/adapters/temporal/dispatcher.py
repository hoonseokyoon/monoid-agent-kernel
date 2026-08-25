"""Signal-With-Start transport over a caller-owned Temporal client event loop."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import math
from dataclasses import dataclass, field
from datetime import timedelta

from monoid_agent_kernel.core.safe_evidence import (
    is_safe_opaque_address,
    is_safe_opaque_id,
)
from monoid_agent_kernel.hosting.admission import AdmittedCommand, DispatchResult

from .dependency import TemporalDependencyMissing
from .names import (
    DEFAULT_TEMPORAL_WORKFLOW_ID_PREFIX,
    TEMPORAL_COMMAND_SIGNAL,
    TEMPORAL_RUN_WORKFLOW_TYPE,
)
from .records import TemporalRunPolicy, TemporalRunState


MAX_TEMPORAL_RPC_TIMEOUT_S = 300.0

_RETRYABLE_RPC_STATUS_NAMES = frozenset(
    {
        "CANCELLED",
        "UNKNOWN",
        "DEADLINE_EXCEEDED",
        "RESOURCE_EXHAUSTED",
        "ABORTED",
        "INTERNAL",
        "UNAVAILABLE",
        "DATA_LOSS",
    }
)


def temporal_workflow_id(
    run_id: str,
    *,
    prefix: str = DEFAULT_TEMPORAL_WORKFLOW_ID_PREFIX,
) -> str:
    """Return the bounded deterministic Workflow ID for one opaque kernel run ID."""

    if not is_safe_opaque_id(run_id):
        raise ValueError("Temporal Workflow run_id must be a bounded opaque id")
    if not is_safe_opaque_id(prefix):
        raise ValueError("Temporal Workflow ID prefix must be a bounded opaque id")
    digest = hashlib.sha256(f"{TEMPORAL_RUN_WORKFLOW_TYPE}\0{run_id}".encode("utf-8")).hexdigest()
    workflow_id = f"{prefix}-{digest}"
    if len(workflow_id) > 255 or not is_safe_opaque_id(workflow_id):
        raise ValueError("Temporal Workflow ID prefix leaves no room for the run digest")
    return workflow_id


def temporal_dispatch_ref(workflow_id: str) -> str:
    """Return the transport-neutral acceptance reference for a Workflow ID."""

    ref = f"temporal:{workflow_id}"
    if not is_safe_opaque_address(ref):
        raise ValueError("Temporal Workflow ID cannot form a bounded dispatch reference")
    return ref


@dataclass
class TemporalSignalWithStartTransport:
    """Synchronous PR8 transport backed by an async Temporal client owner loop.

    The host connects the client on ``event_loop`` and calls ``dispatch`` from its finite
    PostgreSQL outbox polling thread. Async hosts may call ``dispatch_async`` on the owner loop.
    """

    client: object = field(repr=False)
    event_loop: asyncio.AbstractEventLoop = field(repr=False)
    workflow_task_queue: str
    run_policy: TemporalRunPolicy
    workflow_id_prefix: str = DEFAULT_TEMPORAL_WORKFLOW_ID_PREFIX
    rpc_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not callable(getattr(self.client, "start_workflow", None)):
            raise TypeError("Temporal transport client must expose start_workflow()")
        if any(
            not callable(getattr(self.event_loop, method_name, None))
            for method_name in ("call_soon_threadsafe", "is_closed", "is_running")
        ):
            raise TypeError("Temporal transport event_loop must support thread-safe submission")
        if not is_safe_opaque_id(self.workflow_task_queue) or len(self.workflow_task_queue) > 255:
            raise ValueError("Temporal Workflow task queue must be a bounded opaque id")
        if not isinstance(self.run_policy, TemporalRunPolicy):
            raise TypeError("Temporal transport run_policy must be TemporalRunPolicy")
        temporal_dispatch_ref(temporal_workflow_id("validation", prefix=self.workflow_id_prefix))
        if (
            type(self.rpc_timeout_s) not in {int, float}
            or isinstance(self.rpc_timeout_s, bool)
            or not math.isfinite(float(self.rpc_timeout_s))
            or not 0 < float(self.rpc_timeout_s) <= MAX_TEMPORAL_RPC_TIMEOUT_S
        ):
            raise ValueError("Temporal RPC timeout must be in the range (0, 300]")
        self.rpc_timeout_s = float(self.rpc_timeout_s)

    def dispatch(self, command: AdmittedCommand) -> DispatchResult:
        """Submit one Signal-With-Start request and wait only for server acceptance."""

        if not isinstance(command, AdmittedCommand):
            raise TypeError("Temporal transport dispatch requires AdmittedCommand")
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self.event_loop:
            raise RuntimeError(
                "Temporal synchronous dispatch must run outside the client owner event loop"
            )
        if self.event_loop.is_closed() or not self.event_loop.is_running():
            return DispatchResult(
                status="retry",
                error_code="temporal_client_loop_unavailable",
            )
        coroutine = self.dispatch_async(command)
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, self.event_loop)
        except RuntimeError:
            coroutine.close()
            return DispatchResult(
                status="retry",
                error_code="temporal_client_loop_unavailable",
            )
        try:
            return future.result(timeout=self.rpc_timeout_s + 1.0)
        except concurrent.futures.TimeoutError:
            future.cancel()
            return DispatchResult(
                status="retry",
                error_code="temporal_acceptance_unknown",
            )
        except concurrent.futures.CancelledError:
            return DispatchResult(
                status="retry",
                error_code="temporal_client_loop_unavailable",
            )

    async def dispatch_async(self, command: AdmittedCommand) -> DispatchResult:
        """Submit one Signal-With-Start request from the Temporal client owner loop."""

        if not isinstance(command, AdmittedCommand):
            raise TypeError("Temporal transport dispatch requires AdmittedCommand")
        if asyncio.get_running_loop() is not self.event_loop:
            raise RuntimeError("Temporal async dispatch must run on the client owner event loop")
        try:
            from temporalio.common import (
                WorkflowIDConflictPolicy,
                WorkflowIDReusePolicy,
            )
            from temporalio.exceptions import WorkflowAlreadyStartedError
            from temporalio.service import RPCError
        except ImportError as exc:  # pragma: no cover - isolated import test exercises this
            raise TemporalDependencyMissing(
                "install monoid-agent-kernel[temporal] to use the Temporal adapter"
            ) from exc

        workflow_id = temporal_workflow_id(
            command.run_id,
            prefix=self.workflow_id_prefix,
        )
        initial_state = TemporalRunState(
            run_id=command.run_id,
            policy=self.run_policy,
        )
        try:
            await self.client.start_workflow(  # type: ignore[attr-defined]
                TEMPORAL_RUN_WORKFLOW_TYPE,
                initial_state.to_json(),
                id=workflow_id,
                task_queue=self.workflow_task_queue,
                id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
                id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
                start_signal=TEMPORAL_COMMAND_SIGNAL,
                start_signal_args=[command.to_json()],
                rpc_timeout=timedelta(seconds=self.rpc_timeout_s),
            )
        except WorkflowAlreadyStartedError:
            return DispatchResult(
                status="rejected",
                error_code="temporal_workflow_closed",
            )
        except asyncio.TimeoutError:
            return DispatchResult(
                status="retry",
                error_code="temporal_acceptance_unknown",
            )
        except RPCError as exc:
            status_name = getattr(getattr(exc, "status", None), "name", "UNKNOWN")
            if status_name in _RETRYABLE_RPC_STATUS_NAMES:
                return DispatchResult(
                    status="retry",
                    error_code="temporal_rpc_retryable",
                )
            return DispatchResult(
                status="rejected",
                error_code="temporal_request_rejected",
            )
        return DispatchResult(
            status="accepted",
            dispatch_ref=temporal_dispatch_ref(workflow_id),
        )


__all__ = [
    "MAX_TEMPORAL_RPC_TIMEOUT_S",
    "temporal_workflow_id",
    "temporal_dispatch_ref",
    "TemporalSignalWithStartTransport",
]
