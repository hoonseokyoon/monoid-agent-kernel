from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.checkpoint import CheckpointRecord, LocalFsCheckpointStore, RunCheckpoint
from monoid_agent_kernel.core.inbox import InboxMessage
from monoid_agent_kernel.core.lifecycle import SessionState
from monoid_agent_kernel.core.result import AgentRunResult, Suspension
from monoid_agent_kernel.core.spec import ModelRetryConfig
from monoid_agent_kernel.reference.backend.projection import (
    RunProjectionContext,
    RunProjectionService,
)
from monoid_agent_kernel.reference.backend import session_drive
from monoid_agent_kernel.reference.backend.run_types import BackendRunRecord
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


def test_turn_retry_backoff_saturates_at_the_cap_instead_of_overflowing(
    monkeypatch: Any,
) -> None:
    """``max_delay_s`` bounds the exponent, not only the product it multiplies out to.

    ``ModelRetryConfig`` validates ``backoff_multiplier`` as any positive finite number and
    ``max_attempts`` as an integer above zero, neither with an upper bound, so both arms below
    are policies the spec ACCEPTS. Capping only the result lets ``float ** int`` leave the float
    range before the cap is consulted, and that raises ``OverflowError`` rather than saturating
    -- here inside the turn-failure handler, where it would REPLACE the failure being retried.
    """
    slept: list[float] = []

    async def _capture(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(session_drive.asyncio, "sleep", _capture)
    asyncio.run(
        session_drive._async_sleep_before_retry(
            3,
            ModelRetryConfig(
                initial_delay_s=0.5, max_delay_s=4.0, backoff_multiplier=1e308, jitter_s=0.0
            ),
        )
    )
    # The shipped default multiplier needs no exotic policy at all -- only an attempt count.
    asyncio.run(
        session_drive._async_sleep_before_retry(
            1100, ModelRetryConfig(initial_delay_s=0.5, max_delay_s=4.0, jitter_s=0.0)
        )
    )
    assert slept == [4.0, 4.0]


def test_manual_retry_emits_durable_identity_before_replacement_turn(tmp_path: Path) -> None:
    service = _service(tmp_path)
    record = _Record()
    record.state = "turn_failed"
    record.terminal = False
    record.last_final_output = None
    retry = InboxMessage(
        content="",
        id="studio_retry_7",
        source="studio-retry",
        run_id=record.run_id,
        metadata={
            "retry_of_event_seq": 7,
            "retry_of_turn_id": "turn_0001",
        },
    ).to_json()
    record.message_queue.put_nowait(retry)

    class _Request:
        multi_turn = True

    class _Loop:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, Any], str | None]] = []

        def snapshot(self) -> None:
            return None

        def await_user_input(self) -> None:
            return None

        def emit_external_event(
            self,
            event_type: str,
            *,
            data: dict[str, Any] | None = None,
            level: str = "info",
            turn_id: str | None = None,
        ) -> bool:
            assert level == "info"
            self.events.append((event_type, dict(data or {}), turn_id))
            return True

        async def arun_until_suspended(self, user_input: Any = None) -> Suspension:
            assert user_input == ""
            assert self.events == [
                (
                    "run.resumed",
                    {"reason": "studio-retry"},
                    "turn_0001",
                )
            ]
            return Suspension(reason="terminal", status="completed")

        async def aclose(self) -> str:
            return "closed"

    loop = _Loop()
    record.loop = loop

    result = asyncio.run(
        service.drive_open_session(
            record,
            _Request(),
            loop,
            Suspension(
                reason="turn_failed",
                status="failed",
                error="bad config",
                error_code="model_error",
                retryable=False,
            ),
            started=time.time(),
            turns=1,
        )
    )

    assert result == "closed"
    assert loop.events[0] == (
        "run.resumed",
        {"reason": "studio-retry"},
        "turn_0001",
    )


def test_a_config_recoverable_park_reaches_the_surfaces_an_operator_reads(
    tmp_path: Path,
) -> None:
    """The classification is promoted onto the record, and it does NOT change control flow.

    `config_recoverable` existed on the Suspension, on the event and on the wire, and the driver
    the classification was added for never named it: a config-fixable turn failure was driven
    exactly like any other non-retryable one, and the surfaces an operator reads (`GET /status`,
    `GET /result`) had no slot for it at all.

    The decision to make here was park-vs-expose. Parking would have changed a single-shot run
    from terminal to hung; exposing costs nothing and is what the classification is *for* — the
    fix is the caller's, not the driver's. So the terminal behaviour below is asserted in the
    same test as the surfaced flag.
    """
    service = _service(tmp_path)
    record = BackendRunRecord(
        run_id="run_config",
        tenant_id="tenant-1",
        user_id="user-1",
        workspace_root=tmp_path,
        run_dir=tmp_path,
        state=SessionState.RUNNING,
        terminal=False,
        created_at=0.0,
        run_token_sha256="",
        llm_gateway_token_sha256="",
    )
    assert record.config_recoverable is False

    class _Request:
        multi_turn = False

    class _Loop:
        def __init__(self) -> None:
            self.failed: list[tuple[str, str]] = []

        def snapshot(self) -> None:
            return None

        def fail_recoverable(self, error: str, *, error_code: str) -> None:
            self.failed.append((error, error_code))

        async def aclose(self) -> str:
            return "closed"

    loop = _Loop()
    record.loop = loop
    park = Suspension(
        reason="turn_failed",
        status="failed",
        error="the configured model is not available to this account",
        error_code="model_error",
        retryable=False,
        http_status=422,
        config_recoverable=True,
        provider_error_code="model_not_found",
        provider_retried=True,
    )

    result = asyncio.run(
        service.drive_open_session(
            record, _Request(), loop, park, started=time.time(), turns=1
        )
    )

    assert result == "closed"
    # Control flow unchanged: a single-shot run whose turn failed is still promoted, terminally.
    assert loop.failed == [
        ("the configured model is not available to this account", "model_error")
    ]
    # ...and the WHOLE classification the park carried is now on the record the projections
    # read — config_recoverable alone cannot separate an insufficient_quota (fix config) from
    # a rate_limit (wait). One rule, all five, plus the error text the park named.
    assert record.config_recoverable is True
    assert record.retryable is False
    assert record.http_status == 422
    assert record.provider_error_code == "model_not_found"
    assert record.provider_retried is True
    assert record.error == "the configured model is not available to this account"
    assert record.error_code == "model_error"

    projection = RunProjectionService(
        RunProjectionContext(
            authorized_run_dir=lambda run_id, token: tmp_path,
            authorize_run=lambda run_id, token: None,
            record=lambda run_id: record,
            active_record=lambda run_id: record,
            read_recover_attempts=lambda run_dir: 0,
            run_root_provider=lambda: tmp_path,
            checkpoint_store_provider=lambda: None,
            max_recover_attempts_provider=lambda: 0,
            issue_read_token=lambda *args: "",
            read_event_page=lambda events_path, *, from_seq, limit: {"events": []},
        )
    )

    status_payload = projection.status("run_config", "token")
    assert status_payload["config_recoverable"] is True
    assert status_payload["retryable"] is False
    assert status_payload["http_status"] == 422
    assert status_payload["provider_error_code"] == "model_not_found"
    assert status_payload["provider_retried"] is True
    # Both branches of result(): the run has no AgentRunResult yet, and once it does.
    for _ in range(2):
        result_payload = projection.result("run_config", "token")
        assert result_payload["config_recoverable"] is True
        assert result_payload["retryable"] is False
        assert result_payload["http_status"] == 422
        assert result_payload["provider_error_code"] == "model_not_found"
        assert result_payload["provider_retried"] is True
        record.result = AgentRunResult(
            run_id="run_config",
            status="failed",
            final_text="",
            run_dir=tmp_path,
            diff_path=tmp_path / "diff.patch",
            proposal_path=tmp_path / "proposal.json",
        )

    # A later clean park clears the classification AND the error text, rather than leaving a
    # stale answer behind. Assigned on every park, never or-ed.
    record.config_recoverable = True
    record.retryable = True
    record.http_status = 422
    record.provider_error_code = "model_not_found"
    record.provider_retried = True
    record.error = "the configured model is not available to this account"
    record.error_code = "model_error"
    asyncio.run(
        service.drive_open_session(
            record,
            _Request(),
            loop,
            Suspension(reason="terminal", status="completed"),
            started=time.time(),
            turns=1,
        )
    )
    assert record.config_recoverable is False
    assert record.retryable is False
    assert record.http_status is None
    assert record.provider_error_code == ""
    assert record.provider_retried is False
    assert record.error == ""
    assert record.error_code == ""


def test_a_restarted_backend_answers_status_like_the_live_one_did(tmp_path: Path) -> None:
    """The record-is-None branch of status() serves the classification status.json carries.

    After a restart the active record is gone and status.json is what remains; the operator
    polling GET /status must get the same answer the live record gave, not a payload with no
    error slot at all.
    """
    run_dir = tmp_path / "run_restarted"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "run_restarted",
                "state": "awaiting_input",
                "terminal": False,
                "error": "model rejected the key",
                "error_code": "model_error",
                "provider_error_code": "insufficient_quota",
                "http_status": 422,
                "retryable": False,
                "config_recoverable": True,
                "provider_retried": True,
                "last_event_seq": 3,
                "last_event_type": "run.awaiting_input",
                "updated_at": "2026-08-03T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    projection = RunProjectionService(
        RunProjectionContext(
            authorized_run_dir=lambda run_id, token: run_dir,
            authorize_run=lambda run_id, token: None,
            record=lambda run_id: (_ for _ in ()).throw(KeyError(run_id)),
            active_record=lambda run_id: None,
            read_recover_attempts=lambda run_dir: 0,
            run_root_provider=lambda: tmp_path,
            checkpoint_store_provider=lambda: None,
            max_recover_attempts_provider=lambda: 0,
            issue_read_token=lambda *args: "",
            read_event_page=lambda events_path, *, from_seq, limit: {"events": []},
        )
    )

    payload = projection.status("run_restarted", "token")

    assert payload["state"] == "awaiting_input"
    assert payload["error"] == "model rejected the key"
    assert payload["error_code"] == "model_error"
    assert payload["provider_error_code"] == "insufficient_quota"
    assert payload["http_status"] == 422
    assert payload["retryable"] is False
    assert payload["config_recoverable"] is True
    assert payload["provider_retried"] is True


def test_a_resumed_pause_marks_the_record_running_for_the_whole_turn(tmp_path: Path) -> None:
    """After resume, the backend record stayed "paused" until the NEXT park.

    The paused branch re-pumps on the resume signal without touching the record, so a
    resumed multi-minute turn served state="paused" over HTTP the whole way through. The
    driver marks the record RUNNING when it re-pumps, exactly as `record_event` does when a
    park's `model.turn.started` arrives.
    """
    resume_signal = object()
    service = _service(tmp_path, resume_signal=resume_signal)
    record = _Record("run_paused")
    record.state = SessionState.RUNNING
    record.terminal = False
    record.last_final_output = None
    record.message_queue.put_nowait(resume_signal)
    observed_states: list[SessionState] = []

    class _Request:
        multi_turn = True

    class _Loop:
        def snapshot(self) -> None:
            return None

        async def arun_until_suspended(self, user_input: Any = None) -> Suspension:
            assert user_input is None
            observed_states.append(record.state)
            return Suspension(reason="terminal", status="completed")

        async def aclose(self) -> str:
            return "closed"

    loop = _Loop()
    record.loop = loop

    result = asyncio.run(
        service.drive_open_session(
            record,
            _Request(),
            loop,
            Suspension(reason="paused", status="completed"),
            started=time.time(),
            turns=1,
        )
    )

    assert result == "closed"
    # The park itself was observable...
    # (drive_open_session set PAUSED at the top of the loop before waiting on the queue)
    # ...and the resumed pump ran as RUNNING, not as a phantom pause.
    assert observed_states == [SessionState.RUNNING]
