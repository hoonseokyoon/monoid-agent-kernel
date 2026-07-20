"""Finite DBOS workflows that drive one restored run to one durable suspension boundary.

DBOS owns operational admission, serialization, retry, and workflow recovery. A Core
``CheckpointStore`` owns the portable semantic snapshot and committed boundary receipt. The
legacy Reference lease, command inbox, recovery service, watchdog, and process-local run registry
are outside this import and construction path.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import quote

from monoid_agent_kernel.core._util import canonical_sha256
from monoid_agent_kernel.core.checkpoint import (
    CheckpointRecord,
    CheckpointStore,
    RunCheckpoint,
    load_latest_checked,
)
from monoid_agent_kernel.core.lifecycle import state_from_suspension
from monoid_agent_kernel.core.result import (
    Suspension,
    suspension_checkpoint_payload,
    suspension_from_checkpoint_payload,
)
from monoid_agent_kernel.errors import NativeAgentError
from monoid_agent_kernel.identifiers import namespaced_id
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.reference._shared.control_transport import CommandConflict
from monoid_agent_kernel.reference.dbos._compat_226 import _DbosOwnershipConflict
from monoid_agent_kernel.reference.dbos.runtime import (
    DbosHostConfig as _DbosHostConfig,
    DbosProcessOwnershipError as _DbosProcessOwnershipError,
    DbosRuntimeHost as _DbosRuntimeHost,
    DbosShutdownTimeout,
    _DbosHostParticipant,
    _require_shared_host_config,
    claim_process_owner,
    create_owned_runtime,
    load_dbos,
    release_process_owner,
)

DBOS_RESUME_COMMAND_VERSION = namespaced_id("dbos-resume-command.v1")
DBOS_RUN_RECEIPT_VERSION = namespaced_id("dbos-run-receipt.v1")
DBOS_RUN_WORKFLOW_NAME = namespaced_id("reference.dbos-run-workflow.v1")
DBOS_RUN_STEP_NAME = namespaced_id("reference.dbos-run-step.v1")

DbosRunReceiptStatus = Literal["pending", "completed", "failed"]
DbosRunLoopFactory = Callable[["DbosResumeCommand"], AgentLoop]
DbosRunFaultHook = Callable[[str, "DbosResumeCommand"], None]

_SAFE_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}").fullmatch
_SHA256_HEX = re.compile(r"[0-9a-f]{64}").fullmatch


def _checkpoint_receipt_sha256(checkpoint: RunCheckpoint) -> str:
    """Hash a checkpoint while blanking the receipt hashes stored inside that checkpoint."""

    payload = checkpoint.to_json()
    raw_receipts = payload.get("applied_input_receipts")
    if isinstance(raw_receipts, dict):
        receipts: dict[str, Any] = {}
        for input_id, raw_receipt in raw_receipts.items():
            receipt = dict(raw_receipt) if isinstance(raw_receipt, dict) else raw_receipt
            if isinstance(receipt, dict):
                receipt["checkpoint_sha256"] = ""
            receipts[str(input_id)] = receipt
        payload["applied_input_receipts"] = receipts
    return canonical_sha256(payload)


def _durable_error_code(exc: NativeAgentError) -> str:
    """Return a bounded machine code without persisting arbitrary exception content."""

    code = str(getattr(exc, "error_code", "") or "")
    return code if _SAFE_ERROR_CODE(code) is not None else "internal_error"


@dataclass(frozen=True)
class DbosRunConfig:
    """Stable executor-slot configuration for the optional DBOS run driver."""

    system_database_url: str
    name: str = "monoid-reference-dbos"
    application_version: str = "monoid-reference-dbos-v1"
    executor_id: str = "stable-local-slot"
    queue_name: str = "monoid-reference-run"
    polling_interval_s: float = 0.05
    checkpoint_retry_interval_s: float = 0.05
    shutdown_grace_s: int = 30
    local_task_wait_s: float = 300.0

    def __post_init__(self) -> None:
        if not self.system_database_url:
            raise ValueError("DBOS system_database_url is required")
        if not self.name or not self.application_version or not self.executor_id:
            raise ValueError("DBOS name, application_version, and executor_id are required")
        if (
            not self.queue_name
            or not math.isfinite(self.polling_interval_s)
            or self.polling_interval_s <= 0
        ):
            raise ValueError("DBOS queue settings must be positive and non-empty")
        if (
            not math.isfinite(self.checkpoint_retry_interval_s)
            or self.checkpoint_retry_interval_s <= 0
        ):
            raise ValueError("DBOS checkpoint_retry_interval_s must be positive")
        if (
            isinstance(self.shutdown_grace_s, bool)
            or not isinstance(self.shutdown_grace_s, int)
            or self.shutdown_grace_s < 1
        ):
            raise ValueError("DBOS shutdown_grace_s must be a positive whole number of seconds")
        if not math.isfinite(self.local_task_wait_s) or self.local_task_wait_s <= 0:
            raise ValueError("DBOS local_task_wait_s must be positive")

    def _host_config(self) -> _DbosHostConfig:
        """Project the process-wide fields requested by this run participant."""

        return _DbosHostConfig(
            system_database_url=self.system_database_url,
            name=self.name,
            application_version=self.application_version,
            executor_id=self.executor_id,
            shutdown_grace_s=self.shutdown_grace_s,
        )


@dataclass(frozen=True)
class DbosResumeCommand:
    """One retry-stable request to resume a specific committed checkpoint."""

    run_id: str
    command_id: str
    checkpoint_seq: int
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.run_id or not self.command_id:
            raise ValueError("run_id and command_id are required")
        if (
            isinstance(self.checkpoint_seq, bool)
            or not isinstance(self.checkpoint_seq, int)
            or self.checkpoint_seq < 1
        ):
            raise ValueError("checkpoint_seq must be a positive integer")

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(
            {
                "run_id": self.run_id,
                "command_id": self.command_id,
                "checkpoint_seq": self.checkpoint_seq,
            }
        )

    @property
    def checkpoint_marker(self) -> str:
        return f"monoid.dbos-resume/{self.identity_sha256}"

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": DBOS_RESUME_COMMAND_VERSION,
            "run_id": self.run_id,
            "command_id": self.command_id,
            "checkpoint_seq": self.checkpoint_seq,
            "created_at": self.created_at,
            "identity_sha256": self.identity_sha256,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> DbosResumeCommand:
        if payload.get("schema_version") != DBOS_RESUME_COMMAND_VERSION:
            raise ValueError("unsupported DBOS resume command version")
        checkpoint_seq = payload.get("checkpoint_seq")
        if isinstance(checkpoint_seq, bool) or not isinstance(checkpoint_seq, int):
            raise ValueError("DBOS resume checkpoint_seq must be an integer")
        command = cls(
            run_id=str(payload.get("run_id") or ""),
            command_id=str(payload.get("command_id") or ""),
            checkpoint_seq=checkpoint_seq,
            created_at=float(payload.get("created_at") or 0.0),
        )
        recorded_identity = str(payload.get("identity_sha256") or "")
        if recorded_identity and recorded_identity != command.identity_sha256:
            raise ValueError("DBOS resume command identity mismatch")
        return command


@dataclass(frozen=True)
class DbosRunReceipt:
    """The single durable workflow result for one resume command."""

    run_id: str
    command_id: str
    status: DbosRunReceiptStatus
    checkpoint_seq: int = 0
    checkpoint_sha256: str = ""
    state: str = ""
    terminal: bool = False
    suspension: dict[str, Any] | None = None
    error: str = ""
    error_code: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": DBOS_RUN_RECEIPT_VERSION,
            "run_id": self.run_id,
            "command_id": self.command_id,
            "status": self.status,
            "checkpoint_seq": self.checkpoint_seq,
            "checkpoint_sha256": self.checkpoint_sha256,
            "state": self.state,
            "terminal": self.terminal,
            "suspension": dict(self.suspension) if self.suspension is not None else None,
            "error": self.error,
            "error_code": self.error_code,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> DbosRunReceipt:
        if payload.get("schema_version") != DBOS_RUN_RECEIPT_VERSION:
            raise ValueError("unsupported DBOS run receipt version")
        status = str(payload.get("status") or "")
        if status not in {"pending", "completed", "failed"}:
            raise ValueError("unsupported DBOS run receipt status")
        checkpoint_seq = payload.get("checkpoint_seq", 0)
        if isinstance(checkpoint_seq, bool) or not isinstance(checkpoint_seq, int):
            raise ValueError("DBOS run receipt checkpoint_seq must be an integer")
        raw_suspension = payload.get("suspension")
        if raw_suspension is not None and not isinstance(raw_suspension, Mapping):
            raise ValueError("DBOS run receipt suspension must be an object or null")
        terminal = payload.get("terminal", False)
        if not isinstance(terminal, bool):
            raise ValueError("DBOS run receipt terminal must be a boolean")
        return cls(
            run_id=str(payload.get("run_id") or ""),
            command_id=str(payload.get("command_id") or ""),
            status=status,  # type: ignore[arg-type]
            checkpoint_seq=checkpoint_seq,
            checkpoint_sha256=str(payload.get("checkpoint_sha256") or ""),
            state=str(payload.get("state") or ""),
            terminal=terminal,
            suspension=dict(raw_suspension) if isinstance(raw_suspension, Mapping) else None,
            error=str(payload.get("error") or ""),
            error_code=str(payload.get("error_code") or ""),
        )

    @classmethod
    def from_checkpoint(
        cls,
        command: DbosResumeCommand,
        checkpoint: RunCheckpoint,
    ) -> DbosRunReceipt:
        raw_receipt = checkpoint.applied_input_receipts.get(command.checkpoint_marker)
        if not isinstance(raw_receipt, Mapping):
            raise NativeAgentError(
                "completed resume checkpoint has no identity-bound receipt",
                error_code="missing_input_receipt",
            )
        raw_suspension = raw_receipt.get("suspension")
        if not isinstance(raw_suspension, Mapping):
            raise NativeAgentError(
                "completed resume receipt has no durable suspension observation",
                error_code="missing_suspension_observation",
            )
        suspension = suspension_from_checkpoint_payload(raw_suspension)
        checkpoint_seq = raw_receipt.get("checkpoint_seq")
        terminal = raw_receipt.get("terminal")
        checkpoint_sha256 = str(raw_receipt.get("checkpoint_sha256") or "")
        state = state_from_suspension(suspension).value
        if (
            isinstance(checkpoint_seq, bool)
            or not isinstance(checkpoint_seq, int)
            or checkpoint_seq <= command.checkpoint_seq
            or not isinstance(terminal, bool)
            or _SHA256_HEX(checkpoint_sha256) is None
            or raw_receipt.get("state") != state
        ):
            raise NativeAgentError(
                "completed resume receipt has invalid boundary metadata",
                error_code="invalid_input_receipt",
            )
        return cls(
            run_id=command.run_id,
            command_id=command.command_id,
            status="completed",
            checkpoint_seq=checkpoint_seq,
            checkpoint_sha256=checkpoint_sha256,
            state=state,
            terminal=terminal,
            suspension=suspension_checkpoint_payload(suspension),
        )


@dataclass
class _CapturedCheckpoint:
    checkpoint: RunCheckpoint | None = None
    verified: RunCheckpoint | None = None
    blobs: dict[str, bytes] = field(default_factory=dict)


class DbosRunDriver:
    """Own finite resume workflows for one stable DBOS executor slot."""

    def __init__(
        self,
        config: DbosRunConfig,
        checkpoint_store: CheckpointStore,
        loop_factory: DbosRunLoopFactory,
        *,
        fault_hook: DbosRunFaultHook | None = None,
    ) -> None:
        self.config = config
        self._checkpoint_store = checkpoint_store
        self._loop_factory = loop_factory
        self._fault_hook = fault_hook
        self._state_lock = threading.Lock()
        self._drive_condition = threading.Condition()
        self._active_drives = 0
        self._launched = False
        self._accepting = False
        self._closing = False
        self._closed = False
        self._owner_token = claim_process_owner()
        self._queue_name = self.versioned_queue_name(
            config.queue_name,
            config.application_version,
        )
        try:
            self._dbos_module = load_dbos()
            self._runtime = create_owned_runtime(self._dbos_module, config)
            self._workflow = self._register_workflow()
        except Exception:
            runtime = getattr(self, "_runtime", None)
            if runtime is not None:
                runtime.destroy(destroy_registry=True, workflow_completion_timeout_sec=0)
            self._closed = True
            release_process_owner(self._owner_token)
            raise

    @property
    def registered_queue_name(self) -> str:
        return self._queue_name

    def _register_workflow(
        self,
        *,
        runtime: Any | None = None,
        step_name: str = DBOS_RUN_STEP_NAME,
        workflow_name: str = DBOS_RUN_WORKFLOW_NAME,
    ) -> Any:
        runtime = self._runtime if runtime is None else runtime

        @runtime.step(name=step_name, retries_allowed=False)
        def drive_step(payload: dict[str, Any]) -> dict[str, Any]:
            invalid_command = False
            try:
                command = DbosResumeCommand.from_json(payload)
            except Exception:
                invalid_command = True
            if invalid_command:
                # Raise after leaving the handler so the original exception is neither chained
                # nor available for DBOS to serialize into durable workflow state.
                raise RuntimeError("DBOS run dispatch rejected an invalid command")
            with self._drive_condition:
                self._active_drives += 1
            try:
                unexpected_failure = False
                try:
                    return self._drive_one(command).to_json()
                except NativeAgentError as exc:
                    return DbosRunReceipt(
                        run_id=command.run_id,
                        command_id=command.command_id,
                        status="failed",
                        error="DBOS run resume was safely rejected",
                        error_code=_durable_error_code(exc),
                    ).to_json()
                except Exception:
                    unexpected_failure = True
                if unexpected_failure:
                    # DBOS persists step failures. Raise outside the exception handler so neither
                    # the original text nor the original exception object enters durable state.
                    raise RuntimeError("DBOS run dispatch failed")
                raise AssertionError("unreachable DBOS run dispatch state")
            finally:
                with self._drive_condition:
                    self._active_drives -= 1
                    self._drive_condition.notify_all()

        @runtime.workflow(name=workflow_name)
        def run_workflow(payload: dict[str, Any]) -> dict[str, Any]:
            return drive_step(payload)

        return run_workflow

    def launch(self) -> None:
        with self._state_lock:
            if self._launched:
                return
            if self._closed:
                raise RuntimeError("a closed DBOS run driver cannot be relaunched")
            try:
                self._preflight_queue_configuration()
                self._runtime.listen_queues([self._queue_name])
                self._runtime.launch()
                queue = self._register_run_queue(self._runtime)
                self._assert_queue_configuration(queue)
            except Exception:
                self._runtime.destroy(
                    destroy_registry=True,
                    workflow_completion_timeout_sec=0,
                )
                self._closed = True
                release_process_owner(self._owner_token)
                raise
            self._launched = True
            self._accepting = True

    def _preflight_queue_configuration(self) -> None:
        from sqlalchemy.exc import DBAPIError

        client = self._dbos_module.DBOSClient(
            system_database_url=self.config.system_database_url
        )
        try:
            try:
                queue = self._register_run_queue(client)
            except DBAPIError as exc:
                if _is_uninitialized_queue_table(exc):
                    return
                raise
            self._assert_queue_configuration(queue)
        finally:
            client.destroy()

    def _register_run_queue(self, registrar: Any) -> Any:
        return registrar.register_queue(
            self._queue_name,
            worker_concurrency=1,
            concurrency=1,
            priority_enabled=False,
            partition_queue=True,
            polling_interval_sec=self.config.polling_interval_s,
            on_conflict="always_update",
        )

    def enqueue_resume(self, command: DbosResumeCommand) -> DbosRunReceipt:
        if not isinstance(command, DbosResumeCommand):
            raise TypeError("enqueue_resume requires a DbosResumeCommand")
        with self._state_lock:
            if not self._accepting:
                raise RuntimeError("DBOS run driver is not accepting commands")
            workflow_id = self.workflow_id(command.run_id, command.command_id)
            with self._dbos_module.SetWorkflowID(
                workflow_id
            ), self._dbos_module.SetEnqueueOptions(queue_partition_key=command.run_id):
                handle = self._runtime.enqueue_workflow(
                    self._queue_name,
                    self._workflow,
                    command.to_json(),
                )
        status = handle.get_status()
        self._assert_same_identity(command, status.input)
        return self._receipt_from_status(command, status)

    def run_receipt(self, command: DbosResumeCommand) -> DbosRunReceipt:
        self._require_launched()
        handle = self._runtime.retrieve_workflow(
            self.workflow_id(command.run_id, command.command_id)
        )
        status = handle.get_status()
        self._assert_same_identity(command, status.input)
        return self._receipt_from_status(command, status)

    def wait_for_receipt(
        self,
        command: DbosResumeCommand,
        *,
        timeout_s: float = 30.0,
        poll_interval_s: float = 0.01,
    ) -> DbosRunReceipt:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            receipt = self.run_receipt(command)
            if receipt.status in {"completed", "failed"}:
                return receipt
            time.sleep(poll_interval_s)
        raise TimeoutError(
            f"DBOS resume command {command.command_id!r} did not finish within {timeout_s}s"
        )

    def _drive_one(self, command: DbosResumeCommand) -> DbosRunReceipt:
        stored = self._load_checkpoint(command.run_id)
        checkpoint = stored.checkpoint
        if checkpoint.run_id != command.run_id:
            raise NativeAgentError(
                "checkpoint run_id does not match resume command",
                error_code="checkpoint_run_mismatch",
            )
        if command.checkpoint_marker in checkpoint.applied_input_ids:
            if checkpoint.seq <= command.checkpoint_seq:
                raise NativeAgentError(
                    "committed resume marker did not advance the checkpoint sequence",
                    error_code="invalid_resume_marker",
                )
            return DbosRunReceipt.from_checkpoint(command, checkpoint)
        active_input = checkpoint.active_input
        continuing_active_input = False
        if active_input is not None:
            if not isinstance(active_input, Mapping):
                raise NativeAgentError(
                    "checkpoint active input metadata is invalid",
                    error_code="invalid_active_input",
                )
            active_id = str(active_input.get("input_id") or "")
            active_phase = str(active_input.get("phase") or "")
            active_source_seq = active_input.get("source_seq")
            if isinstance(active_source_seq, bool) or not isinstance(active_source_seq, int):
                raise NativeAgentError(
                    "checkpoint active input source sequence is invalid",
                    error_code="invalid_active_input",
                )
            if active_phase == "running":
                if active_id != command.checkpoint_marker:
                    raise NativeAgentError(
                        "another input has an incomplete durable activation",
                        error_code="prior_activation_incomplete",
                    )
                if active_source_seq != command.checkpoint_seq:
                    raise NativeAgentError(
                        "resume identity has a different source checkpoint",
                        error_code="resume_identity_mismatch",
                    )
                continuing_active_input = True
            elif active_phase != "completed":
                raise NativeAgentError(
                    "checkpoint active input phase is invalid",
                    error_code="invalid_active_input",
                )
        if not continuing_active_input and checkpoint.seq != command.checkpoint_seq:
            raise NativeAgentError(
                f"resume expected checkpoint {command.checkpoint_seq}, found {checkpoint.seq}",
                error_code="stale_resume_checkpoint",
            )
        if checkpoint.terminal:
            raise NativeAgentError(
                "terminal checkpoint cannot be resumed",
                error_code="run_terminal",
            )

        self._run_fault_hook("before_restore", command)
        loop = self._loop_factory(command)
        if not isinstance(loop, AgentLoop):
            raise TypeError("DBOS run loop_factory must return AgentLoop")
        if loop.spec.run_id != command.run_id:
            raise NativeAgentError(
                "loop factory returned a different run_id",
                error_code="loop_run_mismatch",
            )
        if loop.checkpoint_store is not None or loop.checkpoint_persist_callback is not None:
            raise RuntimeError(
                "DBOS run loops must delegate checkpoint persistence exclusively to DbosRunDriver"
            )

        capture = _CapturedCheckpoint()
        durable_committed = False

        def persist_checkpoint(
            outgoing: RunCheckpoint,
            blobs: Mapping[str, bytes],
        ) -> bool:
            nonlocal durable_committed
            capture.checkpoint = outgoing
            capture.blobs = dict(blobs)
            completed = outgoing.last_suspension is not None
            outgoing.active_input = {
                "input_id": command.checkpoint_marker,
                "source_seq": command.checkpoint_seq,
                "phase": "completed" if completed else "running",
            }
            if outgoing.seq <= checkpoint.seq:
                raise NativeAgentError(
                    "resume boundary did not advance the checkpoint sequence",
                    error_code="checkpoint_not_advanced",
                )
            if completed:
                suspension = suspension_from_checkpoint_payload(outgoing.last_suspension or {})
                outgoing.applied_input_ids = sorted(
                    {*outgoing.applied_input_ids, command.checkpoint_marker}
                )
                receipts = {
                    input_id: dict(receipt)
                    for input_id, receipt in outgoing.applied_input_receipts.items()
                }
                receipt_record: dict[str, Any] = {
                    "checkpoint_seq": outgoing.seq,
                    "checkpoint_sha256": "",
                    "state": state_from_suspension(suspension).value,
                    "terminal": outgoing.terminal or suspension.reason == "terminal",
                    "suspension": suspension_checkpoint_payload(suspension),
                }
                receipts[command.checkpoint_marker] = receipt_record
                outgoing.applied_input_receipts = receipts
                receipt_record["checkpoint_sha256"] = _checkpoint_receipt_sha256(outgoing)
            verified = self._commit_checkpoint(outgoing, capture.blobs)
            capture.verified = verified
            if completed:
                durable_committed = True
            return True

        loop.checkpoint_persist_callback = persist_checkpoint
        try:
            loop.restore(checkpoint, blobs=stored.blob)
            suspension = self._drive_to_durable_boundary(loop)
            if durable_committed:
                # The callback has returned, so Core has synchronized its committed fingerprint
                # and hosted-task baseline before this crash/fault boundary is exposed.
                self._run_fault_hook("boundary_committed", command)
            committed = capture.checkpoint
            verified = capture.verified
            if committed is None or verified is None or not durable_committed:
                raise NativeAgentError(
                    "run reached no durable suspension checkpoint",
                    error_code="missing_boundary_checkpoint",
                )
            expected_observation = suspension_checkpoint_payload(suspension)
            if committed.last_suspension != expected_observation:
                raise NativeAgentError(
                    "checkpoint does not match the returned suspension boundary",
                    error_code="boundary_checkpoint_mismatch",
                )
            receipt = DbosRunReceipt.from_checkpoint(command, verified)
            self._run_fault_hook("before_step_return", command)
            return receipt
        finally:
            if durable_committed:
                try:
                    loop.release_parked()
                except Exception:
                    self._mark_cleanup_failure()
                    try:
                        loop.discard_uncommitted()
                    except Exception:
                        pass
            else:
                try:
                    loop.discard_uncommitted()
                except Exception:
                    self._mark_cleanup_failure()

    def _drive_to_durable_boundary(self, loop: AgentLoop) -> Suspension:
        suspension = loop.run_until_suspended(None)
        deadline = time.monotonic() + self.config.local_task_wait_s
        while suspension.reason == "awaiting_tasks" and not suspension.has_external:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not loop.wait_for_pending_tasks(min(remaining, 0.25)):
                if remaining <= 0:
                    raise TimeoutError("local task did not reach a durable suspension boundary")
                continue
            suspension = loop.run_until_suspended(None)
        return suspension

    def _load_checkpoint(self, run_id: str) -> CheckpointRecord:
        loaded = load_latest_checked(self._checkpoint_store, run_id)
        if loaded.status in {"loaded", "migrated"} and loaded.value is not None:
            return loaded.value
        if loaded.status == "missing":
            raise NativeAgentError(
                f"no committed checkpoint for run {run_id!r}",
                error_code="checkpoint_missing",
            )
        raise NativeAgentError(
            f"checkpoint for run {run_id!r} is {loaded.status}",
            error_code=f"checkpoint_{loaded.status}",
        )

    def _commit_checkpoint(
        self,
        outgoing: RunCheckpoint,
        blobs: Mapping[str, bytes],
    ) -> RunCheckpoint:
        """Commit and reconcile an ambiguous store result without terminalizing the workflow.

        A remote store may durably publish the checkpoint and then raise while returning its
        response. The DBOS step stays in progress until readback proves either the exact commit or
        a conflicting writer. Killing the process during this loop leaves the DBOS step pending,
        so same-slot startup recovery invokes it again from durable state.
        """

        expected_sha256 = canonical_sha256(outgoing.to_json())
        while True:
            try:
                self._checkpoint_store.put(outgoing, blobs)
            except Exception:
                # Reconcile below. Exception strings can contain credentials and never cross the
                # DBOS durable boundary.
                pass
            verified = self._reconcile_checkpoint(outgoing, expected_sha256)
            if verified is not None:
                return verified
            time.sleep(self.config.checkpoint_retry_interval_s)

    def _reconcile_checkpoint(
        self,
        outgoing: RunCheckpoint,
        expected_sha256: str,
    ) -> RunCheckpoint | None:
        try:
            loaded = load_latest_checked(self._checkpoint_store, outgoing.run_id)
        except Exception:
            return None
        if loaded.status not in {"loaded", "migrated"} or loaded.value is None:
            return None
        observed = loaded.value.checkpoint
        if observed.seq < outgoing.seq:
            return None
        if (
            observed.seq == outgoing.seq
            and canonical_sha256(observed.to_json()) == expected_sha256
        ):
            return observed
        raise NativeAgentError(
            "resume checkpoint lost ownership before commit verification",
            error_code="stale_resume_executor",
        )

    def _run_fault_hook(self, phase: str, command: DbosResumeCommand) -> None:
        if self._fault_hook is not None:
            self._fault_hook(phase, command)

    def _mark_cleanup_failure(self) -> None:
        with self._state_lock:
            self._accepting = False

    def close(self, *, timeout_s: int | None = None) -> None:
        with self._state_lock:
            if self._closed:
                return
            if self._closing:
                raise RuntimeError("DBOS run driver close is already in progress")
            grace_s = self.config.shutdown_grace_s if timeout_s is None else timeout_s
            if isinstance(grace_s, bool) or not isinstance(grace_s, int) or grace_s < 1:
                raise ValueError("DBOS shutdown timeout must be a positive whole second")
            self._accepting = False
            self._closing = True
        try:
            self._runtime.destroy(
                destroy_registry=True,
                workflow_completion_timeout_sec=grace_s,
            )
        except Exception:
            with self._state_lock:
                self._closing = False
            raise
        active_workflow_ids = self._active_workflow_ids()
        with self._drive_condition:
            active_drives = self._active_drives
        if active_drives or active_workflow_ids:
            raise DbosShutdownTimeout(
                "DBOS shutdown grace expired before run workflows drained; terminate the process"
            )
        with self._state_lock:
            self._launched = False
            release_process_owner(self._owner_token)
            self._closing = False
            self._closed = True

    def _active_workflow_ids(self) -> tuple[str, ...]:
        active_set = getattr(self._runtime, "_active_workflows_set", None)
        active_list = getattr(active_set, "activeList", None)
        if not callable(active_list):
            raise DbosShutdownTimeout(
                "DBOS active-workflow state is unavailable; terminate the process"
            )
        return tuple(str(workflow_id) for workflow_id in active_list())

    @staticmethod
    def workflow_id(run_id: str, command_id: str) -> str:
        if not run_id or not command_id:
            raise ValueError("run_id and command_id are required")
        return f"monoid/run/{quote(run_id, safe='')}/resume/{quote(command_id, safe='')}"

    @staticmethod
    def versioned_queue_name(queue_name: str, application_version: str) -> str:
        if not queue_name or not application_version:
            raise ValueError("queue_name and application_version are required")
        return (
            f"monoid/run-queue/{quote(queue_name, safe='')}"
            f"/version/{quote(application_version, safe='')}"
        )

    def _require_launched(self) -> None:
        if not self._launched:
            raise RuntimeError("DBOS run driver must be launched before use")

    @staticmethod
    def _assert_queue_configuration(queue: Any) -> None:
        if (
            queue.concurrency != 1
            or queue.worker_concurrency != 1
            or not queue.partition_queue
            or queue.priority_enabled
        ):
            raise RuntimeError("DBOS run queue does not preserve per-run serialization")

    @staticmethod
    def _assert_same_identity(command: DbosResumeCommand, workflow_input: Any) -> None:
        persisted = DbosRunDriver._command_from_workflow_input(workflow_input)
        if persisted.run_id != command.run_id or persisted.command_id != command.command_id:
            raise RuntimeError("DBOS workflow input does not match the resume command")
        if persisted.identity_sha256 != command.identity_sha256:
            raise CommandConflict(
                f"command_id {command.command_id!r} already belongs to a different resume"
            )

    @staticmethod
    def _command_from_workflow_input(workflow_input: Any) -> DbosResumeCommand:
        if not isinstance(workflow_input, Mapping):
            raise RuntimeError("DBOS run workflow did not expose its persisted input")
        args = workflow_input.get("args")
        if not isinstance(args, (list, tuple)) or not args or not isinstance(args[0], Mapping):
            raise RuntimeError("DBOS run workflow input has an unexpected shape")
        return DbosResumeCommand.from_json(args[0])

    @staticmethod
    def _receipt_from_status(command: DbosResumeCommand, status: Any) -> DbosRunReceipt:
        if status.status == "SUCCESS":
            if not isinstance(status.output, Mapping):
                raise RuntimeError("completed DBOS run workflow has an invalid receipt")
            receipt = DbosRunReceipt.from_json(status.output)
            if receipt.run_id != command.run_id or receipt.command_id != command.command_id:
                raise RuntimeError("completed DBOS run workflow receipt target mismatch")
            return receipt
        if status.status in {"ERROR", "MAX_RECOVERY_ATTEMPTS_EXCEEDED", "CANCELLED"}:
            return DbosRunReceipt(
                run_id=command.run_id,
                command_id=command.command_id,
                status="failed",
                error="DBOS run workflow failed",
                error_code="dbos_workflow_error",
            )
        return DbosRunReceipt(
            run_id=command.run_id,
            command_id=command.command_id,
            status="pending",
        )


class _HostedDbosRunDriver(DbosRunDriver):
    """Run participant whose runtime and lifecycle belong to one DBOS host."""

    def __init__(
        self,
        host: _DbosRuntimeHost,
        config: DbosRunConfig,
        checkpoint_store: CheckpointStore,
        loop_factory: DbosRunLoopFactory,
        *,
        fault_hook: DbosRunFaultHook | None = None,
    ) -> None:
        self.config = config
        self._host = host
        self._checkpoint_store = checkpoint_store
        self._loop_factory = loop_factory
        self._fault_hook = fault_hook
        self._state_lock = threading.Lock()
        self._drive_condition = threading.Condition()
        self._active_drives = 0
        self._active_facade_operations = 0
        self._launched = False
        self._accepting = False
        self._cleanup_failed = False
        self._closing = False
        self._closed = False
        self._queue_name = self.versioned_queue_name(
            config.queue_name,
            config.application_version,
        )
        self._dbos_module = load_dbos()
        self._runtime: Any | None = None
        self._workflow: Any | None = None

    def _host_participant(self) -> _DbosHostParticipant:
        return _DbosHostParticipant(
            participant_id="run",
            queue_name=self._queue_name,
            host_config=self._host.config,
            register_workflows=self._register_hosted_workflows,
            preflight=self._preflight_queue_configuration,
            register_queue=self._register_hosted_queue,
            stop_admission=self._stop_hosted_admission,
            admission_count=self._hosted_admission_count,
            active_count=self._hosted_active_count,
            mark_closed=self._mark_hosted_closed,
        )

    def _register_hosted_workflows(self, runtime: Any) -> None:
        workflow = super()._register_workflow(
            runtime=runtime,
            step_name=self._host.workflow_name("run", "step"),
            workflow_name=self._host.workflow_name("run", "workflow"),
        )
        with self._state_lock:
            self._runtime = runtime
            self._workflow = workflow

    def _register_hosted_queue(self, runtime: Any) -> None:
        with self._state_lock:
            if runtime is not self._runtime or self._workflow is None:
                raise _DbosProcessOwnershipError(
                    "DBOS run participant runtime registration is inconsistent"
                )
        queue = self._register_run_queue(runtime)
        self._assert_queue_configuration(queue)
        with self._state_lock:
            self._launched = True
            self._accepting = not self._cleanup_failed

    def _stop_hosted_admission(self) -> None:
        with self._state_lock:
            self._accepting = False
            self._closing = True

    def _hosted_admission_count(self) -> int:
        with self._drive_condition:
            return self._active_facade_operations

    def _hosted_active_count(self) -> int:
        with self._drive_condition:
            return self._active_drives

    def _mark_hosted_closed(self) -> None:
        with self._state_lock:
            self._launched = False
            self._accepting = False
            self._closing = False
            self._closed = True

    def launch(self) -> None:
        raise RuntimeError("DBOS runtime host owns hosted run-driver lifecycle")

    def close(self, *, timeout_s: int | None = None) -> None:
        del timeout_s
        raise RuntimeError("DBOS runtime host owns hosted run-driver lifecycle")

    def enqueue_resume(self, command: DbosResumeCommand) -> DbosRunReceipt:
        if not isinstance(command, DbosResumeCommand):
            raise TypeError("enqueue_resume requires a DbosResumeCommand")
        self._begin_hosted_operation(require_submission_admission=True)
        try:
            runtime = self._runtime
            workflow = self._workflow
            if runtime is None or workflow is None:
                raise RuntimeError("DBOS run participant runtime is unavailable")
            workflow_id = self.workflow_id(command.run_id, command.command_id)
            try:
                handle = runtime.enqueue_workflow_with_identity(
                    self._queue_name,
                    workflow,
                    command.to_json(),
                    workflow_id=workflow_id,
                    queue_partition_key=command.run_id,
                )
                status = handle.get_status()
            except _DbosOwnershipConflict:
                self._raise_hosted_ownership_error()
            self._assert_same_identity(command, status.input)
            return self._receipt_from_status(command, status)
        finally:
            self._end_hosted_operation()

    def run_receipt(self, command: DbosResumeCommand) -> DbosRunReceipt:
        self._begin_hosted_operation(require_submission_admission=False)
        try:
            runtime = self._runtime
            if runtime is None:
                raise RuntimeError("DBOS run participant runtime is unavailable")
            try:
                handle = runtime.retrieve_workflow(
                    self.workflow_id(command.run_id, command.command_id)
                )
                status = handle.get_status()
            except _DbosOwnershipConflict:
                self._raise_hosted_ownership_error()
            self._assert_same_identity(command, status.input)
            return self._receipt_from_status(command, status)
        finally:
            self._end_hosted_operation()

    def _require_launched(self) -> None:
        if self._closed or self._closing or not self._launched or self._host.state != "running":
            raise RuntimeError("DBOS run driver must be launched before use")

    def _begin_hosted_operation(self, *, require_submission_admission: bool) -> None:
        with self._state_lock:
            if self._closing or self._closed or not self._host.accepting:
                raise RuntimeError("DBOS run driver is not accepting commands")
            if require_submission_admission and not self._accepting:
                raise RuntimeError("DBOS run driver is not accepting commands")
            self._require_launched()
            with self._drive_condition:
                self._active_facade_operations += 1

    def _end_hosted_operation(self) -> None:
        with self._drive_condition:
            self._active_facade_operations -= 1
            self._drive_condition.notify_all()

    def _raise_hosted_ownership_error(self) -> None:
        with self._state_lock:
            self._accepting = False
        self._host._fence("DBOS run runtime ownership changed; terminate the process")
        raise _DbosProcessOwnershipError(
            "DBOS process-global runtime ownership changed; terminate the process"
        ) from None

    def _mark_cleanup_failure(self) -> None:
        with self._state_lock:
            self._cleanup_failed = True
            self._accepting = False


def _register_hosted_run_driver(
    host: _DbosRuntimeHost,
    config: DbosRunConfig,
    checkpoint_store: CheckpointStore,
    loop_factory: DbosRunLoopFactory,
    *,
    fault_hook: DbosRunFaultHook | None = None,
) -> DbosRunDriver:
    """Register one private run participant without launching the shared host."""

    if not isinstance(host, _DbosRuntimeHost):
        raise TypeError("hosted DBOS run driver requires DbosRuntimeHost")
    if not isinstance(config, DbosRunConfig):
        raise TypeError("hosted DBOS run driver requires DbosRunConfig")
    if not callable(loop_factory):
        raise TypeError("hosted DBOS run driver requires a loop factory")
    if fault_hook is not None and not callable(fault_hook):
        raise TypeError("hosted DBOS run driver fault hook must be callable")
    _require_shared_host_config(host.config, config._host_config())
    driver = _HostedDbosRunDriver(
        host,
        config,
        checkpoint_store,
        loop_factory,
        fault_hook=fault_hook,
    )
    host._register_participant(driver._host_participant())
    return driver


def _is_uninitialized_queue_table(exc: Exception) -> bool:
    message = str(exc).lower()
    missing_relation = "no such table" in message or "does not exist" in message
    missing_queue_storage = "queues" in message or 'schema "dbos"' in message
    return missing_relation and missing_queue_storage
