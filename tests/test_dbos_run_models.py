from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from monoid_agent_kernel.core.checkpoint import RunCheckpoint
from monoid_agent_kernel.core.result import Suspension, suspension_checkpoint_payload
from monoid_agent_kernel.errors import NativeAgentError
from monoid_agent_kernel.reference.dbos import (
    DbosResumeCommand,
    DbosRunConfig,
    DbosRunDriver,
    DbosRunReceipt,
)


def test_dbos_run_models_import_without_dbos_or_legacy_backend_services() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    code = """
import sys
from monoid_agent_kernel.reference.dbos import DbosResumeCommand, DbosRunDriver
assert DbosResumeCommand and DbosRunDriver
assert 'dbos' not in sys.modules
for module in (
    'monoid_agent_kernel.reference.backend.service',
    'monoid_agent_kernel.reference.backend.recovery',
    'monoid_agent_kernel.reference.command_inbox',
    'monoid_agent_kernel.reference.stores.lease',
):
    assert module not in sys.modules
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_dbos_resume_command_identity_is_retry_stable_and_round_trips() -> None:
    first = DbosResumeCommand("run_1", "resume_1", 7, created_at=1.0)
    retry = DbosResumeCommand("run_1", "resume_1", 7, created_at=99.0)

    assert first.identity_sha256 == retry.identity_sha256
    assert DbosResumeCommand.from_json(first.to_json()) == first
    assert first.checkpoint_marker == retry.checkpoint_marker


@pytest.mark.parametrize("invalid", [True, "1", float("nan"), float("inf"), 10**400])
def test_dbos_resume_command_rejects_invalid_created_at(invalid: object) -> None:
    with pytest.raises(ValueError, match="created_at must be a finite number"):
        DbosResumeCommand("run_1", "resume_1", 7, created_at=invalid)  # type: ignore[arg-type]

    payload = DbosResumeCommand("run_1", "resume_1", 7, created_at=1.0).to_json()
    payload["created_at"] = invalid
    with pytest.raises(ValueError, match="created_at must be a finite number"):
        DbosResumeCommand.from_json(payload)


def test_dbos_run_receipt_reconstructs_exact_durable_boundary() -> None:
    command = DbosResumeCommand("run_1", "resume_1", 4)
    suspension = Suspension(
        reason="turn_failed",
        status="failed",
        error="retry",
        error_code="model_error",
        retryable=True,
        http_status=429,
    )
    checkpoint = RunCheckpoint(
        run_id=command.run_id,
        seq=5,
        last_suspension=suspension_checkpoint_payload(suspension),
        applied_input_ids=[command.checkpoint_marker],
        applied_input_receipts={
            command.checkpoint_marker: {
                "checkpoint_seq": 5,
                "checkpoint_sha256": "a" * 64,
                "state": "turn_failed",
                "terminal": False,
                "suspension": suspension_checkpoint_payload(suspension),
            }
        },
    )

    receipt = DbosRunReceipt.from_checkpoint(command, checkpoint)

    assert receipt.status == "completed"
    assert receipt.checkpoint_seq == 5
    assert receipt.checkpoint_sha256 == "a" * 64
    assert receipt.state == "turn_failed"
    assert receipt.suspension == suspension_checkpoint_payload(suspension)
    assert DbosRunReceipt.from_json(receipt.to_json()) == receipt


def _limited_family_checkpoint(
    command: DbosResumeCommand,
    suspension: Suspension,
    *,
    recorded_state: str,
    terminal: bool,
) -> RunCheckpoint:
    return RunCheckpoint(
        run_id=command.run_id,
        seq=5,
        last_suspension=suspension_checkpoint_payload(suspension),
        applied_input_ids=[command.checkpoint_marker],
        applied_input_receipts={
            command.checkpoint_marker: {
                "checkpoint_seq": 5,
                "checkpoint_sha256": "a" * 64,
                "state": recorded_state,
                "terminal": terminal,
                "suspension": suspension_checkpoint_payload(suspension),
            }
        },
    )


def test_pre_v021_failed_receipt_is_accepted_at_a_terminal_limited_boundary() -> None:
    """The one boundary v0.21's mapping actually changed: a terminal ``status="limited"``
    projected to FAILED in every release before it, so a pre-v0.21 receipt there says
    "failed" for what this process names "limited". Both spellings pass, there only."""

    command = DbosResumeCommand("run_1", "resume_1", 4)
    terminal_limited = Suspension(
        reason="terminal",
        status="limited",
        error="budget spent",
        error_code="max_steps_exceeded",
    )
    receipt = DbosRunReceipt.from_checkpoint(
        command,
        _limited_family_checkpoint(
            command, terminal_limited, recorded_state="failed", terminal=True
        ),
    )
    assert receipt.state == "limited"


def test_failed_receipt_is_rejected_at_a_cancelled_boundary() -> None:
    """A cancel boundary also arrives ``status="limited"`` (with ``error_code="cancelled"``),
    but pre-v0.21 processes already recorded "cancelled" there — so a receipt claiming
    "failed" at a cancelled boundary is corrupt, not compatible."""

    command = DbosResumeCommand("run_1", "resume_1", 4)
    cancelled = Suspension(
        reason="terminal",
        status="limited",
        error="run cancelled",
        error_code="cancelled",
    )
    own_spelling = DbosRunReceipt.from_checkpoint(
        command,
        _limited_family_checkpoint(command, cancelled, recorded_state="cancelled", terminal=True),
    )
    assert own_spelling.state == "cancelled"

    with pytest.raises(NativeAgentError) as raised:
        DbosRunReceipt.from_checkpoint(
            command,
            _limited_family_checkpoint(command, cancelled, recorded_state="failed", terminal=True),
        )
    assert raised.value.error_code == "invalid_input_receipt"


def test_failed_receipt_is_rejected_at_a_live_limited_park() -> None:
    """A live-limited park (``reason="limited"``) carries ``status="limited"`` too, and its
    LIMITED projection predates v0.21 — a "failed" receipt there was never a valid spelling."""

    command = DbosResumeCommand("run_1", "resume_1", 4)
    live_limited = Suspension(
        reason="limited",
        status="limited",
        error="budget spent",
        error_code="max_steps_exceeded",
    )
    own_spelling = DbosRunReceipt.from_checkpoint(
        command,
        _limited_family_checkpoint(command, live_limited, recorded_state="limited", terminal=False),
    )
    assert own_spelling.state == "limited"

    with pytest.raises(NativeAgentError) as raised:
        DbosRunReceipt.from_checkpoint(
            command,
            _limited_family_checkpoint(
                command, live_limited, recorded_state="failed", terminal=False
            ),
        )
    assert raised.value.error_code == "invalid_input_receipt"


def test_old_dbos_resume_reconstructs_its_own_boundary_after_newer_input() -> None:
    old_command = DbosResumeCommand("run_1", "resume_old", 4)
    new_command = DbosResumeCommand("run_1", "resume_new", 5)
    old_suspension = Suspension(reason="settled", status="completed", final_text="old")
    new_suspension = Suspension(reason="settled", status="completed", final_text="new")
    checkpoint = RunCheckpoint(
        run_id="run_1",
        seq=6,
        last_suspension=suspension_checkpoint_payload(new_suspension),
        applied_input_ids=[old_command.checkpoint_marker, new_command.checkpoint_marker],
        applied_input_receipts={
            old_command.checkpoint_marker: {
                "checkpoint_seq": 5,
                "checkpoint_sha256": "a" * 64,
                "state": "awaiting_input",
                "terminal": False,
                "suspension": suspension_checkpoint_payload(old_suspension),
            },
            new_command.checkpoint_marker: {
                "checkpoint_seq": 6,
                "checkpoint_sha256": "b" * 64,
                "state": "awaiting_input",
                "terminal": False,
                "suspension": suspension_checkpoint_payload(new_suspension),
            },
        },
    )

    receipt = DbosRunReceipt.from_checkpoint(old_command, checkpoint)

    assert receipt.checkpoint_seq == 5
    assert receipt.checkpoint_sha256 == "a" * 64
    assert receipt.suspension == suspension_checkpoint_payload(old_suspension)


class _DecoratorRuntime:
    def step(self, **_kwargs):  # noqa: ANN201
        return lambda function: function

    def workflow(self, **_kwargs):  # noqa: ANN201
        return lambda function: function


def _workflow_with_drive_failure(monkeypatch: pytest.MonkeyPatch, failure: Exception):
    driver = object.__new__(DbosRunDriver)
    driver._runtime = _DecoratorRuntime()  # type: ignore[attr-defined]
    driver._drive_condition = __import__("threading").Condition()  # type: ignore[attr-defined]
    driver._active_drives = 0  # type: ignore[attr-defined]

    def fail(_command: DbosResumeCommand) -> DbosRunReceipt:
        raise failure

    monkeypatch.setattr(driver, "_drive_one", fail)
    return driver._register_workflow()  # type: ignore[attr-defined]


def test_expected_dispatch_failure_returns_safe_structured_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = DbosResumeCommand("run_1", "resume_1", 4)
    workflow = _workflow_with_drive_failure(
        monkeypatch,
        NativeAgentError("secret database detail", error_code="stale_resume_checkpoint"),
    )

    result = workflow(command.to_json())

    assert result["status"] == "failed"
    assert result["error_code"] == "stale_resume_checkpoint"
    assert result["error"] == "DBOS run resume was safely rejected"
    assert "secret database detail" not in repr(result)


def test_unexpected_dispatch_failure_drops_original_exception_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = DbosResumeCommand("run_1", "resume_1", 4)
    workflow = _workflow_with_drive_failure(
        monkeypatch,
        RuntimeError("secret database detail"),
    )

    with pytest.raises(RuntimeError, match="^DBOS run dispatch failed$") as raised:
        workflow(command.to_json())

    assert raised.value.__context__ is None
    assert raised.value.__cause__ is None


def test_dbos_run_workflow_and_queue_ids_escape_components() -> None:
    first = DbosRunDriver.workflow_id("run/resume/cmd", "tail")
    second = DbosRunDriver.workflow_id("run", "cmd/resume/tail")

    assert first != second
    assert "%2F" in first
    assert DbosRunDriver.versioned_queue_name("run", "v/one") != (
        DbosRunDriver.versioned_queue_name("run/v", "one")
    )


@pytest.mark.parametrize("polling_interval_s", (0.0, float("nan"), float("inf")))
def test_dbos_run_config_rejects_invalid_polling_interval(polling_interval_s: float) -> None:
    with pytest.raises(ValueError, match="queue settings"):
        DbosRunConfig(
            system_database_url="sqlite:///dbos.sqlite",
            polling_interval_s=polling_interval_s,
        )


@pytest.mark.parametrize("local_task_wait_s", (0.0, -1.0, float("nan"), float("inf")))
def test_dbos_run_config_rejects_invalid_local_task_wait(local_task_wait_s: float) -> None:
    with pytest.raises(ValueError, match="local_task_wait_s"):
        DbosRunConfig(
            system_database_url="sqlite:///dbos.sqlite",
            local_task_wait_s=local_task_wait_s,
        )


@pytest.mark.parametrize(
    "checkpoint_retry_interval_s",
    (0.0, -1.0, float("nan"), float("inf")),
)
def test_dbos_run_config_rejects_invalid_checkpoint_retry_interval(
    checkpoint_retry_interval_s: float,
) -> None:
    with pytest.raises(ValueError, match="checkpoint_retry_interval_s"):
        DbosRunConfig(
            system_database_url="sqlite:///dbos.sqlite",
            checkpoint_retry_interval_s=checkpoint_retry_interval_s,
        )
