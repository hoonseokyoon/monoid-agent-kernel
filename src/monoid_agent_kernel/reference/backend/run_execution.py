from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core.content import ContentPart
from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.result import AgentRunResult, Suspension
from monoid_agent_kernel.errors import NativeAgentError
from monoid_agent_kernel.reference.backend.activation import (
    is_activation_lease_loss,
    raise_on_lease_loss,
)
from monoid_agent_kernel.reference.backend.content_hydration import (
    hydrate_settled_text,
    needs_settled_text,
)
from monoid_agent_kernel.reference.backend.ports import (
    DriveOpenSessionPort,
    LoopBuildPort,
    MutableRunRecordPort,
    PreparedRunPort,
    RunExecutionLoopPort,
    RunRequestPort,
)


def stream_item_frame(item: Any) -> dict[str, Any]:
    """Wrap one stream item as the Reference backend's neutral wire frame."""
    if isinstance(item, AgentEvent):
        return {"kind": "event", **item.to_json()}
    return {"kind": "delta", **item.to_json()}


def result_frame(result: AgentRunResult, suspension: Suspension | None) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "kind": "result",
        "status": result.status,
        "final_text": result.final_text,
        "error": result.error,
        "error_code": result.error_code,
        "interruption_cause": (
            None
            if result.interruption_cause is None
            else result.interruption_cause.value
        ),
    }
    if suspension is not None and suspension.has_external:
        frame["awaiting_task_ids"] = list(suspension.awaiting_task_ids)
        frame["note"] = "run closed; hosted task cancelled (HITL streaming deferred)"
    return frame


def failure_frame(exc: Exception) -> dict[str, Any]:
    return {
        "kind": "result",
        "status": "failed",
        "error": str(exc),
        "error_code": getattr(exc, "error_code", "internal_error"),
    }


@dataclass(frozen=True)
class RunExecutionContext:
    build_loop: Callable[[str, RunRequestPort, Path, str, str], LoopBuildPort]
    attach_loop: Callable[[MutableRunRecordPort, LoopBuildPort], None]
    unregister_record: Callable[[MutableRunRecordPort], None]
    record: Callable[[str], MutableRunRecordPort]
    drive_open_session: DriveOpenSessionPort
    record_run_result: Callable[[str, AgentRunResult], None]
    record_run_failure: Callable[[str, Exception], None]
    acquire_run_slot: Callable[[], Awaitable[None]]
    release_run_slot: Callable[[], None]
    submission_json: Callable[[PreparedRunPort], dict[str, Any]]


class RunExecutionService:
    """Autonomous and stream-driven run execution for the Reference backend facade."""

    def __init__(self, context: RunExecutionContext) -> None:
        self._context = context

    async def run_prepared(self, prepared: PreparedRunPort, request: RunRequestPort) -> None:
        await self._context.acquire_run_slot()
        loop: RunExecutionLoopPort | None = None
        released = False
        discarded = False
        try:
            try:
                loop_build = self._context.build_loop(
                    prepared.run_id,
                    request,
                    prepared.workspace_root,
                    prepared.llm_gateway_token,
                    prepared.web_gateway_token,
                )
                loop = loop_build.loop
                self._context.attach_loop(prepared.record, loop_build)
                result = await self.drive_session(prepared.run_id, request, loop)
                released = True
                self._context.record_run_result(prepared.run_id, result)
            except Exception as exc:
                lease_lost = is_activation_lease_loss(exc)
                if lease_lost:
                    # Retire host ownership before cleanup so status, commands, recovery, and the
                    # watchdog stop seeing this activation as live while discard is in progress.
                    self._context.unregister_record(prepared.record)
                if loop is not None and not released:
                    try:
                        await asyncio.to_thread(loop.discard_uncommitted)
                        discarded = True
                    except BaseException:
                        # The original execution failure remains the actionable cause. AgentLoop
                        # already attempts every owned resource before surfacing cleanup failure.
                        pass
                if lease_lost:
                    return
                self._context.record_run_failure(prepared.run_id, exc)
        finally:
            # Cancellation derives from BaseException, so it bypasses the failure-recording branch.
            # It still owns every resource materialized before the cancellation point.
            if loop is not None and not released and not discarded:
                try:
                    await asyncio.to_thread(loop.discard_uncommitted)
                except BaseException:
                    pass
            self._context.release_run_slot()

    async def drive_session(
        self,
        run_id: str,
        request: RunRequestPort,
        loop: RunExecutionLoopPort,
    ) -> AgentRunResult:
        await loop.aopen()
        first_input: str | tuple[ContentPart, ...] = request.input_parts or request.instruction
        try:
            suspension = await loop.arun_until_suspended(first_input)
        except NativeAgentError as exc:
            if is_activation_lease_loss(exc):
                raise
            return await loop.aclose()
        return await self._context.drive_open_session(
            self._context.record(run_id),
            request,
            loop,
            suspension,
            started=time.time(),
            turns=1,
        )

    async def stream_prepared(
        self,
        prepared: PreparedRunPort,
        request: RunRequestPort,
    ) -> AsyncIterator[dict[str, Any]]:
        await self._context.acquire_run_slot()
        loop: RunExecutionLoopPort | None = None
        closed = False
        try:
            yield {"kind": "meta", **self._context.submission_json(prepared)}
            loop_build = self._context.build_loop(
                prepared.run_id,
                request,
                prepared.workspace_root,
                prepared.llm_gateway_token,
                prepared.web_gateway_token,
            )
            loop = loop_build.loop
            self._context.attach_loop(prepared.record, loop_build)
            await loop.aopen()
            suspension: Suspension | None = None
            first_input: str | tuple[ContentPart, ...] = request.input_parts or request.instruction
            async with loop.astream(first_input) as stream:
                stream_run_dir = loop.spec.run_root / loop.spec.run_id
                async for item in stream:
                    frame = stream_item_frame(item)
                    # ``kind:event`` frames carry the settle events, and the settled-text record is
                    # written *before* its emit, so it is already on disk by the time the frame is
                    # built. Without this the live stream degrades asymmetrically — orchestration
                    # frames lose the text while ``kind:delta`` and ``kind:result`` keep it, which
                    # no consumer expects. Delta frames are deliberately untouched: they carry live
                    # token text that no turn-end record can supply.
                    if frame.get("kind") == "event":
                        # ``AgentEvent.to_json()`` hands back the live ``data`` dict *by
                        # reference*, so hydrating the frame in place would write the text into
                        # the event the bus still owns and every registered sink shares —
                        # including embedder-supplied ones. A sink that buffers events and
                        # serializes them later would then export exactly the content this change
                        # moves off that stream. Copy before filling.
                        data = frame.get("data")
                        if isinstance(data, dict):
                            frame["data"] = dict(data)
                        # Off-thread ONLY when there is a digest to resolve. Resolving one scans
                        # the transcript, which has no positional bound (any window drops text a
                        # reader legitimately asked for); inline that blocked the shared run loop
                        # for the whole read — ~0.15s on a 21MB transcript — and runs share that
                        # loop behind ``acquire_run_slot``.
                        #
                        # But the hop is not free either: the default executor is shared and
                        # bounded (32 workers), and parked runs hold workers for up to
                        # ``task_wait_poll_s`` each, so an unconditional hop would queue every
                        # frame's delivery behind them. Until the emit change lands, no event
                        # carries a digest and the resolver opens no file at all — so the
                        # emptiness check stays on the loop and only real work crosses the
                        # boundary.
                        if needs_settled_text([frame]):
                            await asyncio.to_thread(hydrate_settled_text, [frame], stream_run_dir)
                    yield frame
                suspension = stream.suspension
            raise_on_lease_loss(suspension)
            result = await loop.aclose()
            closed = True
            self._context.record_run_result(prepared.run_id, result)
            yield result_frame(result, suspension)
        except Exception as exc:
            lease_lost = is_activation_lease_loss(exc)
            if loop is not None and not closed and not lease_lost:
                try:
                    await loop.aclose()
                    closed = True
                except Exception:  # noqa: BLE001 - finalization best-effort; the failure is recorded below
                    pass
            if not lease_lost:
                self._context.record_run_failure(prepared.run_id, exc)
                yield failure_frame(exc)
            else:
                self._context.unregister_record(prepared.record)
        finally:
            if loop is not None and not closed:
                try:
                    await asyncio.to_thread(loop.discard_uncommitted)
                except BaseException:
                    pass
            self._context.release_run_slot()
