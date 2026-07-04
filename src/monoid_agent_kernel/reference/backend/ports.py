from __future__ import annotations

from collections.abc import Awaitable, Mapping
from pathlib import Path
from typing import Any, Protocol

from monoid_agent_kernel.core.agents import AgentRuntimeConfig
from monoid_agent_kernel.core.checkpoint import RunCheckpoint
from monoid_agent_kernel.core.lifecycle import SessionState
from monoid_agent_kernel.core.result import AgentRunResult, Suspension


class TokenClaimsPort(Protocol):
    tenant_id: str
    user_id: str
    run_id: str


class CancellationTokenPort(Protocol):
    requested: bool


class MessageQueuePort(Protocol):
    _queue: Any

    def qsize(self) -> int: ...

    def put_nowait(self, item: Any) -> None: ...

    def get(self) -> Awaitable[Any]: ...


class LoopPort(Protocol):
    def wait_for_pending_tasks(self, timeout_s: float) -> bool: ...

    def has_pending_tasks(self) -> bool: ...

    async def arun_until_suspended(self, user_input: Any | None = None) -> Suspension: ...

    def fail_recoverable(self, error: str, *, error_code: str) -> None: ...

    def await_user_input(self) -> None: ...

    async def aclose(self) -> AgentRunResult: ...

    def snapshot(self) -> RunCheckpoint | None: ...

    def collect_checkpoint_blobs(self) -> Mapping[str, bytes]: ...

    def interrupt_turn(self) -> None: ...

    def pause_turn(self) -> None: ...

    def revoke_capability(
        self,
        *,
        capability: str | None = None,
        lease_id: str | None = None,
        before: float | None = None,
        reason: str = "",
    ) -> dict[str, Any]: ...

    def report_task_result(
        self,
        task_id: str,
        result: dict[str, Any],
        *,
        status: str,
        persist_checkpoint: bool,
    ) -> dict[str, Any]: ...

    def create_task(self, kind: str, request: dict[str, Any]) -> str: ...

    def restore(self, checkpoint: RunCheckpoint, *, blobs: Mapping[str, bytes]) -> None: ...


class RunRecordPort(Protocol):
    run_id: str
    tenant_id: str
    user_id: str
    workspace_root: Path
    run_dir: Path
    state: SessionState
    terminal: bool
    created_at: float
    started_at: float
    finished_at: float
    last_event_seq: int
    last_event_type: str
    error: str
    error_code: str
    result: AgentRunResult | None
    last_final_output: Any
    runtime_config: AgentRuntimeConfig | None
    runtime_config_issuer: str
    runtime_config_reason: str
    runtime_config_committed_at: float


class MutableRunRecordPort(RunRecordPort, Protocol):
    message_queue: MessageQueuePort
    loop: LoopPort | None
    cancellation_token: CancellationTokenPort
    seen_inbox_ids: set[str]
    outbox_sender: Any


class RunRequestPort(Protocol):
    tenant_id: str
    user_id: str
    workspace_root: Path
    instruction: str
    input_parts: tuple[Any, ...]
    mode: str
    workspace_backend: str
    max_steps: int
    max_tool_calls: int
    max_bytes_read: int
    max_duration_s: int | None
    permission_policy: Any
    runtime_config: AgentRuntimeConfig | None
    multi_turn: bool
    metadata: dict[str, Any]


class LeaseStorePort(Protocol):
    def candidate_run_ids(self) -> list[str]: ...

    def is_stale(self, run_id: str) -> bool: ...

    def try_claim(self, run_id: str, worker_id: str, ttl_s: float) -> bool: ...

    def release(self, run_id: str) -> None: ...


class DriveOpenSessionPort(Protocol):
    def __call__(
        self,
        record: MutableRunRecordPort,
        request: RunRequestPort,
        loop: LoopPort,
        suspension: Suspension,
        *,
        started: float,
        turns: int,
    ) -> Awaitable[AgentRunResult]: ...
