from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.checkpoint import CheckpointRecord, LocalFsCheckpointStore, RunCheckpoint
from monoid_agent_kernel.core.inbox import InboxMessage
from monoid_agent_kernel.core.spec import ModelRetryConfig
from monoid_agent_kernel.reference.backend.session_drive import (
    SessionDriveContext,
    SessionDriveLimits,
    SessionDriveService,
)


def _limits(**overrides: Any) -> SessionDriveLimits:
    values = {
        "idle_timeout_s": 1.0,
        "max_session_lifetime_s": 60.0,
        "max_turns": 10,
        "task_wait_poll_s": 0.01,
        "max_consecutive_turn_failures": 3,
        "turn_retry": ModelRetryConfig(initial_delay_s=0.0, max_delay_s=0.0, jitter_s=0.0),
    }
    values.update(overrides)
    return SessionDriveLimits(**values)


def _service(
    tmp_path: Path,
    *,
    close_signal: object | None = None,
    resume_signal: object | None = None,
    limits_provider: Any | None = None,
    store: LocalFsCheckpointStore | None = None,
    drain_calls: list[tuple[Any, Any]] | None = None,
) -> SessionDriveService:
    checkpoint_store = store or LocalFsCheckpointStore(tmp_path / "runs")
    close = object() if close_signal is None else close_signal
    resume = object() if resume_signal is None else resume_signal

    def drain_outbox(record: Any, loop: Any) -> None:
        if drain_calls is not None:
            drain_calls.append((record, loop))

    return SessionDriveService(
        SessionDriveContext(
            limits_provider=limits_provider or (lambda: _limits()),
            checkpoint_store_provider=lambda: checkpoint_store,
            drain_outbox=drain_outbox,
            close_signal=close,
            resume_signal=resume,
        )
    )


class _Record:
    def __init__(self, run_id: str = "run_1") -> None:
        self.run_id = run_id
        self.message_queue: asyncio.Queue[Any] = asyncio.Queue()
        self.seen_inbox_ids: set[str] = set()
        self.loop: Any = None
        self.cancellation_token = CancellationToken()


def test_session_drive_wait_ignores_stray_resume_without_backend(tmp_path: Path) -> None:
    resume_signal = object()
    service = _service(tmp_path, resume_signal=resume_signal)
    record = _Record()
    record.message_queue.put_nowait(resume_signal)
    record.message_queue.put_nowait("next")

    assert asyncio.run(service.await_session_message(record)) == "next"


def test_session_drive_persist_uses_context_store_and_drain_callback(tmp_path: Path) -> None:
    drain_calls: list[tuple[Any, Any]] = []
    store = LocalFsCheckpointStore(tmp_path / "runs")
    service = _service(tmp_path, store=store, drain_calls=drain_calls)
    record = _Record("run_checkpoint")
    record.seen_inbox_ids.update({"msg_2", "msg_1"})
    envelope = InboxMessage(content="queued", id="msg_3").to_json()
    record.message_queue.put_nowait("plain")
    record.message_queue.put_nowait(object())
    record.message_queue.put_nowait(envelope)

    class _Loop:
        def snapshot(self) -> RunCheckpoint:
            return RunCheckpoint(run_id=record.run_id, seq=1)

        def collect_checkpoint_blobs(self) -> dict[str, bytes]:
            return {}

    loop = _Loop()
    record.loop = loop

    service.persist_run_checkpoint(record)

    stored: CheckpointRecord | None = store.latest(record.run_id)
    assert stored is not None
    assert stored.checkpoint.queued_messages == ["plain", envelope]
    assert stored.checkpoint.inbox_seen_ids == ["msg_1", "msg_2"]
    assert drain_calls == [(record, loop)]


def test_session_drive_limits_provider_is_live(tmp_path: Path) -> None:
    current_limits = _limits(max_turns=10)
    service = _service(tmp_path, limits_provider=lambda: current_limits)
    record = _Record()
    started = time.time()

    assert service.session_should_stop(record, started=started, turns=2) is False

    current_limits = _limits(max_turns=2)

    assert service.session_should_stop(record, started=started, turns=2) is True
