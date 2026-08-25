"""Deterministic per-run Temporal Workflow.

Import this module only in environments with the ``temporal`` optional dependency installed.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from monoid_agent_kernel.hosting.admission import AdmittedCommand

from .dependency import TemporalDependencyMissing
from .names import (
    TEMPORAL_COMMAND_SIGNAL,
    TEMPORAL_DRIVE_ACTIVATION_ACTIVITY,
    TEMPORAL_RUN_WORKFLOW_TYPE,
    TEMPORAL_STATUS_QUERY,
)
from .records import (
    TemporalActivationResult,
    TemporalRunState,
    TemporalRunStatus,
)

try:
    from temporalio import workflow
    from temporalio.common import RetryPolicy
    from temporalio.exceptions import ActivityError, ApplicationError, RetryState
except ImportError as exc:  # pragma: no cover - exercised by isolated import tests
    raise TemporalDependencyMissing(
        "install monoid-agent-kernel[temporal] to use the Temporal Workflow"
    ) from exc


_NON_RETRYABLE_ACTIVITY_ERROR_TYPES = (
    "monoid.activation_corrupt",
    "monoid.activation_unsupported",
    "monoid.activation_config_conflict",
)
_ACTIVITY_REDRIVE_DELAY = timedelta(seconds=5)
_MAX_ACTIVITY_REDRIVES_PER_EXECUTION = 100
_ACTIVITY_RETRY_EXHAUSTED_CODE = "temporal_activity_retry_exhausted"


@workflow.defn(name=TEMPORAL_RUN_WORKFLOW_TYPE)
class TemporalRunWorkflow:
    """Order content-free command refs and drive one finite Activity at a time."""

    @workflow.init
    def __init__(self, initial_payload: dict[str, Any]) -> None:
        state = TemporalRunState.from_json(initial_payload)
        self._run_id = state.run_id
        self._policy = state.policy
        self._next_sequence = state.next_command_sequence
        self._pending = {command.command_sequence: command for command in state.pending_commands}
        self._latest_receipt_ref = state.latest_receipt_ref
        self._rollover_count = state.rollover_count
        self._duplicate_signal_count = state.duplicate_signal_count
        self._last_error_code = state.last_error_code
        self._in_flight: AdmittedCommand | None = None
        self._terminal = False
        self._commands_since_rollover = 0

    def _record_duplicate(self) -> None:
        self._duplicate_signal_count += 1

    @workflow.signal(name=TEMPORAL_COMMAND_SIGNAL)
    async def submit_command(self, payload: dict[str, Any]) -> None:
        try:
            command = AdmittedCommand.from_json(payload)
        except (TypeError, ValueError, OverflowError, RecursionError):
            self._last_error_code = "temporal_invalid_command"
            return
        if command.run_id != self._run_id:
            self._last_error_code = "temporal_command_run_mismatch"
            return
        if command.command_sequence < self._next_sequence:
            self._record_duplicate()
            return
        if (
            self._in_flight is not None
            and command.command_sequence == self._in_flight.command_sequence
        ):
            if command.identity_sha256 == self._in_flight.identity_sha256:
                self._record_duplicate()
            else:
                self._last_error_code = "temporal_command_sequence_conflict"
            return
        pending = self._pending.get(command.command_sequence)
        if pending is not None:
            if pending.identity_sha256 == command.identity_sha256:
                self._record_duplicate()
            else:
                self._last_error_code = "temporal_command_sequence_conflict"
            return
        self._pending[command.command_sequence] = command

    def _status(self, *, terminal: bool | None = None) -> TemporalRunStatus:
        pending_sequences = sorted(self._pending)
        is_terminal = self._terminal if terminal is None else terminal
        phase = (
            "terminal" if is_terminal else ("running" if self._in_flight is not None else "waiting")
        )
        return TemporalRunStatus(
            run_id=self._run_id,
            phase=phase,
            next_command_sequence=self._next_sequence,
            in_flight_sequence=(0 if self._in_flight is None else self._in_flight.command_sequence),
            pending_count=len(pending_sequences),
            pending_head_sequence=(pending_sequences[0] if pending_sequences else 0),
            latest_receipt_ref=self._latest_receipt_ref,
            rollover_count=self._rollover_count,
            duplicate_signal_count=self._duplicate_signal_count,
            last_error_code=self._last_error_code,
        )

    @workflow.query(name=TEMPORAL_STATUS_QUERY)
    def status(self) -> dict[str, Any]:
        return self._status().to_json()

    def _should_continue_as_new(self) -> bool:
        limit = self._policy.history_rollover_command_limit
        return workflow.info().is_continue_as_new_suggested() or (
            limit > 0 and self._commands_since_rollover >= limit
        )

    def _continue_state(self) -> TemporalRunState:
        return TemporalRunState(
            run_id=self._run_id,
            policy=self._policy,
            next_command_sequence=self._next_sequence,
            pending_commands=tuple(self._pending[sequence] for sequence in sorted(self._pending)),
            latest_receipt_ref=self._latest_receipt_ref,
            rollover_count=self._rollover_count + 1,
            duplicate_signal_count=self._duplicate_signal_count,
            last_error_code=self._last_error_code,
        )

    async def _drive(
        self,
        command: AdmittedCommand,
        *,
        redrive_count: int = 0,
    ) -> TemporalActivationResult:
        activity_id = f"activation-{command.command_sequence}-{command.identity_sha256[:16]}"
        if redrive_count:
            activity_id = f"{activity_id}-redrive-{redrive_count}"
        raw_result = await workflow.execute_activity(
            TEMPORAL_DRIVE_ACTIVATION_ACTIVITY,
            command.to_json(),
            result_type=dict,
            task_queue=self._policy.activity_task_queue,
            start_to_close_timeout=timedelta(
                seconds=self._policy.activity_start_to_close_timeout_s
            ),
            heartbeat_timeout=timedelta(seconds=self._policy.activity_heartbeat_timeout_s),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                backoff_coefficient=2.0,
                maximum_interval=timedelta(seconds=30),
                maximum_attempts=self._policy.activity_max_attempts,
                non_retryable_error_types=_NON_RETRYABLE_ACTIVITY_ERROR_TYPES,
            ),
            cancellation_type=workflow.ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
            activity_id=activity_id,
        )
        try:
            result = TemporalActivationResult.from_json(raw_result)
        except (TypeError, ValueError, OverflowError, RecursionError) as exc:
            raise ApplicationError(
                "Temporal activation Activity returned an invalid public result",
                type="monoid.activation_result_invalid",
                non_retryable=True,
            ) from exc
        if not result.matches(command):
            raise ApplicationError(
                "Temporal activation Activity result belongs to another command",
                type="monoid.activation_result_mismatch",
                non_retryable=True,
            )
        return result

    @workflow.run
    async def run(self, initial_payload: dict[str, Any]) -> dict[str, Any]:
        del initial_payload  # @workflow.init validated and installed the exact same payload.
        while True:
            await workflow.wait_condition(lambda: self._next_sequence in self._pending)
            command = self._pending.pop(self._next_sequence)
            self._in_flight = command
            redrive_count = 0
            while True:
                try:
                    result = await self._drive(command, redrive_count=redrive_count)
                    break
                except ActivityError as exc:
                    if exc.retry_state != RetryState.MAXIMUM_ATTEMPTS_REACHED:
                        raise
                    redrive_count += 1
                    self._last_error_code = _ACTIVITY_RETRY_EXHAUSTED_CODE
                    if (
                        redrive_count >= _MAX_ACTIVITY_REDRIVES_PER_EXECUTION
                        or workflow.info().is_continue_as_new_suggested()
                    ):
                        self._pending[command.command_sequence] = command
                        self._in_flight = None
                        await workflow.wait_condition(workflow.all_handlers_finished)
                        workflow.continue_as_new(self._continue_state().to_json())
                    await workflow.sleep(_ACTIVITY_REDRIVE_DELAY)
            self._latest_receipt_ref = result.receipt_ref
            self._next_sequence += 1
            self._commands_since_rollover += 1
            self._in_flight = None
            if result.terminal:
                self._terminal = True
                await workflow.wait_condition(workflow.all_handlers_finished)
                return self._status(terminal=True).to_json()
            if self._should_continue_as_new():
                await workflow.wait_condition(workflow.all_handlers_finished)
                workflow.continue_as_new(self._continue_state().to_json())


__all__ = ["TemporalRunWorkflow"]
