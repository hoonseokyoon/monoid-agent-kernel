from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core.content import ContentPart
from monoid_agent_kernel.core.events import AgentEvent
from monoid_agent_kernel.core.result import AgentRunResult, Suspension
from monoid_agent_kernel.errors import NativeAgentError
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
                self._context.record_run_result(prepared.run_id, result)
            except Exception as exc:
                self._context.record_run_failure(prepared.run_id, exc)
        finally:
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
        except NativeAgentError:
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
                async for item in stream:
                    yield stream_item_frame(item)
                suspension = stream.suspension
            result = await loop.aclose()
            closed = True
            self._context.record_run_result(prepared.run_id, result)
            yield result_frame(result, suspension)
        except Exception as exc:
            if loop is not None and not closed:
                try:
                    await loop.aclose()
                except Exception:  # noqa: BLE001 - finalization best-effort; the failure is recorded below
                    pass
            self._context.record_run_failure(prepared.run_id, exc)
            yield failure_frame(exc)
        finally:
            self._context.release_run_slot()
