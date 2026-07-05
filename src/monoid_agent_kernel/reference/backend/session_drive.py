from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from monoid_agent_kernel.core.checkpoint import CheckpointStore, RunCheckpoint
from monoid_agent_kernel.core.content import ContentPart, content_part_from_json
from monoid_agent_kernel.core.inbox import InboxMessage, is_inbox_envelope
from monoid_agent_kernel.core.lifecycle import state_from_suspension
from monoid_agent_kernel.core.result import AgentRunResult, Suspension
from monoid_agent_kernel.core.spec import ModelRetryConfig
from monoid_agent_kernel.reference.backend.ports import LoopPort, MutableRunRecordPort, RunRequestPort
from monoid_agent_kernel.reference.backend.run_state import set_record_state as _set_record_state


@dataclass(frozen=True)
class SessionDriveLimits:
    idle_timeout_s: float
    max_session_lifetime_s: float
    max_turns: int
    task_wait_poll_s: float
    max_consecutive_turn_failures: int
    turn_retry: ModelRetryConfig


@dataclass(frozen=True)
class SessionDriveContext:
    limits_provider: Callable[[], SessionDriveLimits]
    checkpoint_store_provider: Callable[[], CheckpointStore]
    drain_outbox: Callable[[MutableRunRecordPort, LoopPort], None]
    close_signal: object
    resume_signal: object


def _queued_message_to_loop_input(message: Any) -> str | tuple[ContentPart, ...]:
    """Convert a dequeued backend message into a loop ``submit`` input."""
    if is_inbox_envelope(message):
        message = InboxMessage.from_json(message).content
    if isinstance(message, list):
        return tuple(content_part_from_json(part) for part in message)
    return message


async def _async_sleep_before_retry(attempt: int, retry: ModelRetryConfig) -> None:
    """Awaitable, cancellable exponential backoff with jitter for turn-level retries."""
    delay = min(retry.max_delay_s, retry.initial_delay_s * (retry.backoff_multiplier ** max(0, attempt - 1)))
    if retry.jitter_s > 0:
        delay += random.uniform(0, retry.jitter_s)
    if delay > 0:
        await asyncio.sleep(delay)


class SessionDriveService:
    """Open-session driver for the RunnerBackend facade."""

    def __init__(self, context: SessionDriveContext) -> None:
        self._context = context

    async def drive_open_session(
        self,
        record: MutableRunRecordPort,
        request: RunRequestPort,
        loop: LoopPort,
        suspension: Suspension,
        *,
        started: float,
        turns: int,
    ) -> AgentRunResult:
        """Drive an already-open run until close, idle, cancellation, or terminal suspension."""
        consecutive_turn_failures = 0
        while True:
            limits = self._context.limits_provider()
            _set_record_state(
                record,
                state_from_suspension(suspension),
                terminal=suspension.reason in {"terminal", "limited"},
            )
            if suspension.turn is not None:
                record.last_final_output = suspension.turn.final_output
            if suspension.reason in {"terminal", "limited"}:
                break
            if suspension.reason == "awaiting_tasks":
                self.persist_run_checkpoint(record)
                ready = await asyncio.to_thread(loop.wait_for_pending_tasks, limits.task_wait_poll_s)
                if self.session_should_stop(record, started, turns):
                    break
                if ready or not loop.has_pending_tasks():
                    suspension = await loop.arun_until_suspended(None)
                continue
            if suspension.reason == "paused":
                if self.session_should_stop(record, started, turns):
                    break
                self.persist_run_checkpoint(record)
                try:
                    signal = await asyncio.wait_for(record.message_queue.get(), limits.idle_timeout_s)
                except asyncio.TimeoutError:
                    break
                if signal is self._context.close_signal:
                    break
                if signal is not self._context.resume_signal:
                    record.message_queue.put_nowait(signal)
                suspension = await loop.arun_until_suspended(None)
                continue
            if suspension.reason == "turn_failed":
                consecutive_turn_failures += 1
                if consecutive_turn_failures >= limits.max_consecutive_turn_failures or self.session_should_stop(
                    record, started, turns
                ):
                    loop.fail_recoverable(
                        suspension.error or "model turn failed repeatedly",
                        error_code=suspension.error_code or "model_error",
                    )
                    break
                if suspension.retryable:
                    self.persist_run_checkpoint(record)
                    await _async_sleep_before_retry(consecutive_turn_failures, limits.turn_retry)
                    if self.session_should_stop(record, started, turns):
                        break
                    suspension = await loop.arun_until_suspended(None)
                    continue
                if not request.multi_turn:
                    loop.fail_recoverable(
                        suspension.error or "model turn failed",
                        error_code=suspension.error_code or "model_error",
                    )
                    break
                loop.await_user_input()
                self.persist_run_checkpoint(record)
                try:
                    message = await self.await_session_message(record)
                except asyncio.TimeoutError:
                    break
                if message is self._context.close_signal:
                    break
                turns += 1
                suspension = await loop.arun_until_suspended(_queued_message_to_loop_input(message))
                continue
            if suspension.reason == "interrupted":
                if not request.multi_turn or self.session_should_stop(record, started, turns):
                    break
                loop.await_user_input()
                self.persist_run_checkpoint(record)
                try:
                    message = await self.await_session_message(record)
                except asyncio.TimeoutError:
                    break
                if message is self._context.close_signal:
                    break
                turns += 1
                suspension = await loop.arun_until_suspended(_queued_message_to_loop_input(message))
                continue
            consecutive_turn_failures = 0
            self._context.drain_outbox(record, loop)
            if not request.multi_turn:
                break
            if self.session_should_stop(record, started, turns):
                break
            loop.await_user_input()
            self.persist_run_checkpoint(record)
            try:
                message = await self.await_session_message(record)
            except asyncio.TimeoutError:
                break
            if message is self._context.close_signal:
                break
            turns += 1
            suspension = await loop.arun_until_suspended(_queued_message_to_loop_input(message))
        return await loop.aclose()

    async def await_session_message(self, record: MutableRunRecordPort) -> Any:
        """Await the next user message, ignoring stray resume signals and duplicate inbox ids."""
        deadline = time.monotonic() + self._context.limits_provider().idle_timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            message = await asyncio.wait_for(record.message_queue.get(), remaining)
            if message is self._context.resume_signal:
                continue
            if is_inbox_envelope(message):
                msg_id = str(message.get("id") or "")
                if msg_id and msg_id in record.seen_inbox_ids:
                    continue
                if msg_id:
                    record.seen_inbox_ids.add(msg_id)
            return message

    def persist_run_checkpoint(self, record: MutableRunRecordPort) -> None:
        """Augment the loop checkpoint with backend-owned queue and inbox state."""
        loop = record.loop
        if loop is None:
            return
        checkpoint = loop.snapshot()
        if checkpoint is None:
            return
        self.persist_run_checkpoint_payload(record, checkpoint, loop.collect_checkpoint_blobs())

    def persist_run_checkpoint_payload(
        self,
        record: MutableRunRecordPort,
        checkpoint: RunCheckpoint,
        blobs: Mapping[str, bytes],
    ) -> None:
        """Commit a loop checkpoint after adding backend-owned queue and inbox state."""
        loop = record.loop
        if loop is None:
            return
        checkpoint.queued_messages = [
            message for message in list(record.message_queue._queue) if isinstance(message, (str, list, dict))
        ]
        checkpoint.inbox_seen_ids = sorted(record.seen_inbox_ids)
        self._context.checkpoint_store_provider().put(checkpoint, blobs)
        self._context.drain_outbox(record, loop)

    async def persist_run_checkpoint_async(self, record: MutableRunRecordPort) -> None:
        self.persist_run_checkpoint(record)

    def session_should_stop(self, record: MutableRunRecordPort, started: float, turns: int) -> bool:
        limits = self._context.limits_provider()
        return (
            record.cancellation_token.requested
            or (time.time() - started) >= limits.max_session_lifetime_s
            or turns >= limits.max_turns
        )
